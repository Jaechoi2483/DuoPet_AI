"""
CustomScaleLayer 문제를 해결한 체크포인트 변환 스크립트
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
from pathlib import Path
import shutil

# CustomScaleLayer 정의 (InceptionResNetV2에서 사용)
class CustomScaleLayer(Layer):
    def __init__(self, scale, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config

# 현재 디렉토리 기준 상대 경로
SCRIPT_DIR = Path.cwd()
BASE_PATH = SCRIPT_DIR / "models" / "health_diagnosis" / "skin_disease"

print(f"Running on: {os.name}")
print(f"Working directory: {SCRIPT_DIR}")
print(f"Models base path: {BASE_PATH}")

# config
config = {
    'models': {
        'classification': {
            'cat_binary': {
                'checkpoint_prefix': 'model-007-0.511353-0.772705-0.776322-0.768861',
                'output_classes': 2,
                'model_type': 'checkpoint'
            },
            'dog_binary': {
                'checkpoint_prefix': 'model-004-0.437360-0.806570-0.806528-0.806891',
                'output_classes': 2,
                'model_type': 'checkpoint'
            }
        }
    }
}

def create_classification_model(output_classes):
    """분류 모델 생성 - weights=None으로 설정하여 초기화 없이 생성"""
    # CustomScaleLayer를 등록
    with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
        network = InceptionResNetV2(
            include_top=False, 
            weights=None,  # 가중치 초기화 없이 생성
            input_shape=(224, 224, 3),
            pooling='avg'
        )
    
    model = Sequential()
    model.add(network)
    model.add(Dense(2048, activation='relu'))
    
    if output_classes == 2:
        model.add(Dense(output_classes, activation='sigmoid'))
    else:
        model.add(Dense(output_classes, activation='softmax'))
    
    return model

def test_model_predictions(model, model_name, output_classes):
    """모델 예측 테스트"""
    print(f"\n=== Testing {model_name} ===")
    predictions = []
    
    # 5개의 다른 입력으로 테스트
    for i in range(5):
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32) * 255
        pred = model.predict(test_input, verbose=0)
        predictions.append(pred[0])
        
        if output_classes == 2:
            print(f"  Test {i+1}: Normal={pred[0][0]:.4f}, Disease={pred[0][1]:.4f}")
    
    # 표준편차 계산
    predictions = np.array(predictions)
    std_dev = np.std(predictions, axis=0)
    
    print(f"  Std deviation: {std_dev}")
    
    if np.all(std_dev < 0.01):
        print("  ⚠️  WARNING: Model outputs are nearly constant!")
        return False
    else:
        print("  ✓ Model outputs show healthy variation")
        return True

def convert_checkpoint_to_h5(model_key, model_info):
    """체크포인트를 H5로 변환"""
    checkpoint_prefix = model_info['checkpoint_prefix']
    output_classes = model_info['output_classes']
    
    model_dir = BASE_PATH / "classification" / model_key
    checkpoint_path = model_dir / checkpoint_prefix
    h5_path = model_dir / f"{model_key}_model_from_checkpoint_v2.h5"
    
    print(f"\n{'='*70}")
    print(f"Converting: {model_key}")
    print(f"Directory: {model_dir}")
    
    # 필수 파일 확인
    index_file = Path(f"{checkpoint_path}.index")
    data_file = Path(f"{checkpoint_path}.data-00000-of-00001")
    
    if not index_file.exists() or not data_file.exists():
        print(f"✗ Error: Required checkpoint files missing")
        return False
    
    try:
        # 파일 크기 표시
        total_size = (index_file.stat().st_size + data_file.stat().st_size) / (1024*1024)
        print(f"Checkpoint size: {total_size:.2f} MB")
        
        # 1. 먼저 imagenet 가중치로 모델 생성
        print("Creating model with ImageNet weights first...")
        with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
            temp_network = InceptionResNetV2(
                include_top=False, 
                weights='imagenet',
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            # 가중치 복사를 위해 저장
            imagenet_weights = temp_network.get_weights()
        
        # 2. 가중치 없이 모델 생성
        print(f"Creating model structure with {output_classes} outputs...")
        model = create_classification_model(output_classes)
        
        # 3. InceptionResNetV2 부분에 imagenet 가중치 설정
        model.layers[0].set_weights(imagenet_weights)
        
        print(f"Model created - Input: {model.input_shape}, Output: {model.output_shape}")
        
        # 4. 체크포인트에서 전체 가중치 로드
        print(f"Loading weights from checkpoint...")
        model.load_weights(str(checkpoint_path))
        print("✓ Weights loaded successfully!")
        
        # 5. 모델 저장 (CustomScaleLayer 포함)
        print(f"Saving to: {h5_path.name}")
        with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
            model.save(str(h5_path))
        
        # 저장된 파일 크기 확인
        if h5_path.exists():
            h5_size = h5_path.stat().st_size / (1024*1024)
            print(f"✓ Saved! H5 file size: {h5_size:.2f} MB")
            
            # 크기 비교
            size_ratio = h5_size / total_size * 100
            print(f"Size retention: {size_ratio:.1f}% of original")
        
        # 변동성 테스트
        test_model_predictions(model, model_key, output_classes)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("TensorFlow Checkpoint to H5 Converter - CustomScaleLayer Fixed")
    print("="*70)
    
    # 변환 시작
    success_count = 0
    total_count = len(config['models']['classification'])
    
    for model_key, model_info in config['models']['classification'].items():
        if convert_checkpoint_to_h5(model_key, model_info):
            success_count += 1
    
    # 최종 요약
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully converted: {success_count}/{total_count} models")
    
    # 변환된 파일 목록
    print("\n=== Converted Models ===")
    for model_key in ['cat_binary', 'dog_binary']:
        model_dir = BASE_PATH / "classification" / model_key
        converted_file = model_dir / f"{model_key}_model_from_checkpoint_v2.h5"
        
        if converted_file.exists():
            size = converted_file.stat().st_size / (1024*1024)
            print(f"\n{model_key}:")
            print(f"  ✓ {converted_file.name} ({size:.2f} MB)")
        else:
            print(f"\n{model_key}:")
            print(f"  ✗ Conversion failed")
    
    print("\n✅ Next steps:")
    print("1. Update model_registry.json to use _v2.h5 files")
    print("2. Restart the backend service")
    print("3. Test with skin disease images")

if __name__ == "__main__":
    main()
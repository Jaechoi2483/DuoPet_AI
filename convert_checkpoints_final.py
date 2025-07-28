"""
최종 체크포인트 변환 스크립트 - Windows 호환 + 누락 파일 처리
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
from pathlib import Path
import platform
import shutil

# 현재 디렉토리 기준 상대 경로
SCRIPT_DIR = Path.cwd()
BASE_PATH = SCRIPT_DIR / "models" / "health_diagnosis" / "skin_disease"

print(f"Running on: {platform.system()}")
print(f"Working directory: {SCRIPT_DIR}")
print(f"Models base path: {BASE_PATH}")

# config - dog_multi_456은 index 파일이 없어서 제외
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
            },
            'dog_multi_136': {
                'checkpoint_prefix': 'model-009-0.851382-0.821520',
                'output_classes': 3,
                'model_type': 'checkpoint'
            }
        }
    }
}

def create_classification_model(output_classes):
    """분류 모델 생성 - 원본과 동일한 구조"""
    network = InceptionResNetV2(
        include_top=False, 
        weights='imagenet', 
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

def test_model_variation(model, model_name, output_classes):
    """모델이 다양한 출력을 생성하는지 테스트"""
    print(f"\n=== Testing {model_name} Output Variation ===")
    predictions = []
    
    # 10개의 다른 입력으로 테스트
    for i in range(10):
        # 다양한 패턴의 입력 생성
        if i < 3:
            # 매우 밝은 이미지
            test_input = np.ones((1, 224, 224, 3), dtype=np.float32) * 255
        elif i < 6:
            # 매우 어두운 이미지
            test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        else:
            # 랜덤 이미지
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32) * 255
        
        pred = model.predict(test_input, verbose=0)
        predictions.append(pred[0])
        
        if output_classes == 2:
            print(f"  Input {i+1}: Normal={pred[0][0]:.4f}, Disease={pred[0][1]:.4f}")
    
    # 표준편차 계산
    predictions = np.array(predictions)
    std_dev = np.std(predictions, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    
    print(f"\nStatistics:")
    print(f"  Mean prediction: {mean_pred}")
    print(f"  Std deviation: {std_dev}")
    
    # 변동성 확인
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
    h5_path = model_dir / f"{model_key}_model_from_checkpoint.h5"
    
    print(f"\n{'='*70}")
    print(f"Converting: {model_key}")
    print(f"Directory: {model_dir}")
    
    # 필수 파일 확인
    index_file = Path(f"{checkpoint_path}.index")
    data_file = Path(f"{checkpoint_path}.data-00000-of-00001")
    
    if not index_file.exists() or not data_file.exists():
        print(f"✗ Error: Required checkpoint files missing")
        print(f"  Index file: {index_file.exists()}")
        print(f"  Data file: {data_file.exists()}")
        return False
    
    try:
        # 파일 크기 표시
        total_size = (index_file.stat().st_size + data_file.stat().st_size) / (1024*1024)
        print(f"Checkpoint size: {total_size:.2f} MB")
        
        # 모델 생성
        print(f"Creating model with {output_classes} outputs...")
        model = create_classification_model(output_classes)
        print(f"Model created - Input: {model.input_shape}, Output: {model.output_shape}")
        
        # 가중치 로드
        print(f"Loading weights from checkpoint...")
        model.load_weights(str(checkpoint_path))
        print("✓ Weights loaded successfully!")
        
        # 모델 저장
        print(f"Saving to: {h5_path.name}")
        model.save(str(h5_path))
        
        # 저장된 파일 크기 확인
        if h5_path.exists():
            h5_size = h5_path.stat().st_size / (1024*1024)
            print(f"✓ Saved! H5 file size: {h5_size:.2f} MB")
            
            # 크기 비교
            size_ratio = h5_size / total_size * 100
            print(f"Size retention: {size_ratio:.1f}% of original")
        
        # 변동성 테스트
        test_model_variation(model, model_key, output_classes)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def handle_dog_multi_456():
    """dog_multi_456은 index 파일이 없으므로 기존 H5 파일 중 최선을 선택"""
    print(f"\n{'='*70}")
    print("Handling dog_multi_456 (no checkpoint index file)")
    
    model_dir = BASE_PATH / "classification" / "dog_multi_456"
    h5_files = list(model_dir.glob("*.h5"))
    
    if h5_files:
        # 파일 크기순으로 정렬
        h5_files_with_size = [(f, f.stat().st_size / (1024*1024)) for f in h5_files]
        h5_files_with_size.sort(key=lambda x: x[1], reverse=True)
        
        print("\nAvailable H5 files:")
        for f, size in h5_files_with_size:
            print(f"  - {f.name}: {size:.2f} MB")
        
        # 가장 큰 파일 선택 (preserved 버전일 가능성 높음)
        largest_file = h5_files_with_size[0][0]
        target_path = model_dir / "dog_multi_456_model_from_checkpoint.h5"
        
        if largest_file.name != target_path.name:
            print(f"\nCopying {largest_file.name} as checkpoint version...")
            shutil.copy2(largest_file, target_path)
            print(f"✓ Created: {target_path.name}")
    else:
        print("✗ No H5 files found for dog_multi_456")

def main():
    print("="*70)
    print("TensorFlow Checkpoint to H5 Converter - Final Version")
    print("="*70)
    
    # 변환 시작
    success_count = 0
    total_count = len(config['models']['classification'])
    
    for model_key, model_info in config['models']['classification'].items():
        if convert_checkpoint_to_h5(model_key, model_info):
            success_count += 1
    
    # dog_multi_456 처리
    handle_dog_multi_456()
    
    # 최종 요약
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully converted: {success_count}/{total_count} models")
    print(f"dog_multi_456: Used existing H5 file (no checkpoint index)")
    
    # 변환된 파일 목록
    print("\n=== Converted Models ===")
    for model_key in ['cat_binary', 'dog_binary', 'dog_multi_136', 'dog_multi_456']:
        model_dir = BASE_PATH / "classification" / model_key
        converted_file = model_dir / f"{model_key}_model_from_checkpoint.h5"
        
        if converted_file.exists():
            size = converted_file.stat().st_size / (1024*1024)
            print(f"\n{model_key}:")
            print(f"  ✓ {converted_file.name} ({size:.2f} MB)")
        else:
            print(f"\n{model_key}:")
            print(f"  ✗ Conversion failed")
    
    print("\n✅ Next steps:")
    print("1. Run: python update_model_registry.py")
    print("2. Restart the backend service")
    print("3. Test with skin disease images")

if __name__ == "__main__":
    main()
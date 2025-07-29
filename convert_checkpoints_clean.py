"""
체크포인트에서 깨끗한 모델로 변환하는 스크립트
CustomScaleLayer 없이 표준 InceptionResNetV2 구조 사용
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
from pathlib import Path
import json

def create_standard_model(output_classes):
    """표준 모델 생성 (CustomScaleLayer 없음)"""
    # InceptionResNetV2 백본
    base_model = InceptionResNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Sequential 모델 구성
    model = Sequential([
        base_model,
        Dense(2048, activation='relu', name='dense_features'),
        Dense(output_classes, 
              activation='sigmoid' if output_classes == 2 else 'softmax',
              name='predictions')
    ])
    
    return model

def test_model_predictions(model, model_name):
    """모델 예측 테스트"""
    print(f"\nTesting {model_name} predictions...")
    
    # 테스트 데이터
    test_images = {
        'black': np.zeros((1, 224, 224, 3), dtype=np.float32),
        'white': np.ones((1, 224, 224, 3), dtype=np.float32),
        'gray': np.ones((1, 224, 224, 3), dtype=np.float32) * 0.5,
        'random': np.random.random((1, 224, 224, 3)).astype(np.float32)
    }
    
    predictions = []
    for name, img in test_images.items():
        pred = model.predict(img, verbose=0)
        predictions.append(pred[0])
        if len(pred[0]) == 2:
            print(f"  {name:8s}: normal={pred[0][0]:.4f}, disease={pred[0][1]:.4f}")
        else:
            print(f"  {name:8s}: {pred[0][:5]}...")  # 처음 5개만 표시
    
    # 변동성 체크
    predictions = np.array(predictions)
    std_dev = np.std(predictions, axis=0)
    mean_std = np.mean(std_dev)
    
    if mean_std < 0.01:
        print(f"  ⚠️  Low variation detected (mean std: {mean_std:.4f})")
        return False
    else:
        print(f"  ✓ Good variation (mean std: {mean_std:.4f})")
        return True

def convert_checkpoint(model_key, checkpoint_info, base_path):
    """체크포인트를 깨끗한 H5 모델로 변환"""
    print(f"\n{'='*70}")
    print(f"Converting {model_key}")
    print(f"{'='*70}")
    
    model_dir = base_path / "classification" / model_key
    checkpoint_path = model_dir / checkpoint_info['checkpoint_prefix']
    
    # 체크포인트 파일 확인
    if not Path(f"{checkpoint_path}.index").exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # 1. 표준 모델 생성
        output_classes = checkpoint_info['output_classes']
        print(f"Creating model with {output_classes} output classes...")
        model = create_standard_model(output_classes)
        
        # 2. 체크포인트에서 가중치 로드
        print(f"Loading weights from checkpoint...")
        model.load_weights(str(checkpoint_path))
        print("✓ Weights loaded successfully")
        
        # 3. 모델 컴파일 (옵션)
        if output_classes == 2:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # 4. 예측 테스트
        if not test_model_predictions(model, model_key):
            print("⚠️  Model may not be working correctly")
        
        # 5. 모델 저장
        save_path = model_dir / f"{model_key}_model_clean.h5"
        print(f"\nSaving to: {save_path.name}")
        model.save(str(save_path))
        
        # 파일 크기 확인
        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved successfully ({size_mb:.2f} MB)")
        
        # 6. 클래스 맵 복사 (있다면)
        class_map_src = model_dir / f"{model_key}_class_map.json"
        if class_map_src.exists():
            class_map_dst = model_dir / f"{model_key}_clean_class_map.json"
            import shutil
            shutil.copy2(class_map_src, class_map_dst)
            print(f"✓ Class map copied")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 설정
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease"
    
    checkpoints = {
        'cat_binary': {
            'checkpoint_prefix': 'model-007-0.511353-0.772705-0.776322-0.768861',
            'output_classes': 2
        },
        'dog_binary': {
            'checkpoint_prefix': 'model-004-0.437360-0.806570-0.806528-0.806891',
            'output_classes': 2
        },
        'dog_multi_136': {
            'checkpoint_prefix': 'model-009-0.851382-0.821520',
            'output_classes': 3
        }
    }
    
    print("="*70)
    print("CLEAN CHECKPOINT CONVERTER")
    print("="*70)
    print("This script creates clean models without CustomScaleLayer")
    
    success_count = 0
    for model_key, checkpoint_info in checkpoints.items():
        if convert_checkpoint(model_key, checkpoint_info, base_path):
            success_count += 1
    
    # 요약
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully converted: {success_count}/{len(checkpoints)} models")
    
    if success_count > 0:
        print("\n✅ Next steps:")
        print("1. Test the clean models: python test_clean_models.py")
        print("2. Update skin_disease_service.py to use _clean.h5 models")
        print("3. Remove or backup the problematic v2 models")
    else:
        print("\n❌ No models were successfully converted")
        print("Check the error messages above for details")

if __name__ == "__main__":
    main()
"""
Windows 환경에서 동작하는 체크포인트 변환 스크립트
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import json
from pathlib import Path
import platform

# Windows에서 실행 중인지 확인
IS_WINDOWS = platform.system() == 'Windows'

# 현재 스크립트 위치 기준으로 상대 경로 사용
SCRIPT_DIR = Path(__file__).parent.absolute()
BASE_PATH = SCRIPT_DIR / "models" / "health_diagnosis" / "skin_disease"

print(f"Running on: {platform.system()}")
print(f"Script directory: {SCRIPT_DIR}")
print(f"Base path: {BASE_PATH}")

# config 직접 정의
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
            },
            'dog_multi_456': {
                'checkpoint_prefix': 'model-005-0.881675-0.851780',
                'output_classes': 3,
                'model_type': 'checkpoint'
            }
        }
    }
}

def create_classification_model(output_classes):
    """원본 학습 코드와 동일한 구조로 분류 모델 생성"""
    network = InceptionResNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    model = Sequential()
    model.add(network)
    model.add(Dense(2048, activation='relu'))
    
    # Binary는 2개 출력 + sigmoid, Multi는 N개 출력 + softmax
    if output_classes == 2:
        model.add(Dense(output_classes, activation='sigmoid'))
    else:
        model.add(Dense(output_classes, activation='softmax'))
    
    return model

def verify_checkpoint_files(checkpoint_path):
    """체크포인트 파일 존재 여부 확인"""
    # TensorFlow 체크포인트는 여러 파일로 구성됨
    required_files = [
        f"{checkpoint_path}.index",
        f"{checkpoint_path}.data-00000-of-00001"
    ]
    
    print(f"Checking for checkpoint files:")
    for file in required_files:
        file_path = Path(file)
        exists = file_path.exists()
        print(f"  - {file_path.name}: {'✓ Found' if exists else '✗ Not found'}")
        if not exists:
            print(f"    Full path: {file_path.absolute()}")
    
    return all(Path(f).exists() for f in required_files)

def convert_checkpoint_to_h5(model_key, model_info):
    """체크포인트를 H5로 변환"""
    model_type = model_info.get('model_type')
    if model_type != 'checkpoint':
        print(f"Skipping {model_key} - not a checkpoint model")
        return
        
    checkpoint_prefix = model_info['checkpoint_prefix']
    output_classes = model_info['output_classes']
    
    # 모델 디렉토리 설정
    model_dir = BASE_PATH / "classification" / model_key
    checkpoint_path = model_dir / checkpoint_prefix
    
    # H5 저장 경로
    h5_path = model_dir / f"{model_key}_model_from_checkpoint.h5"
    
    print(f"\n{'='*60}")
    print(f"Converting: {model_key}")
    print(f"Model directory: {model_dir}")
    print(f"Checkpoint prefix: {checkpoint_prefix}")
    print(f"Output classes: {output_classes}")
    
    # 디렉토리 존재 확인
    if not model_dir.exists():
        print(f"✗ Error: Directory not found - {model_dir}")
        return
    
    # 체크포인트 파일 확인
    if not verify_checkpoint_files(str(checkpoint_path)):
        print(f"✗ Error: Checkpoint files not found")
        
        # 디렉토리 내용 표시
        print(f"\nDirectory contents:")
        try:
            for file in model_dir.iterdir():
                print(f"  - {file.name} ({file.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"  Error listing directory: {e}")
        return
    
    try:
        # 모델 생성
        print(f"\nCreating model...")
        model = create_classification_model(output_classes)
        print(f"Model structure - Input: {model.input_shape}, Output: {model.output_shape}")
        
        # 체크포인트 로드 - str()로 명시적 변환
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_weights(str(checkpoint_path))
        print("✓ Checkpoint loaded successfully!")
        
        # 모델 저장
        print(f"Saving to: {h5_path}")
        model.save(str(h5_path))
        print(f"✓ Saved successfully!")
        
        # 파일 크기 확인
        if h5_path.exists():
            size_mb = h5_path.stat().st_size / (1024 * 1024)
            print(f"✓ Output file size: {size_mb:.2f} MB")
        
        # 간단한 검증
        import numpy as np
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = model.predict(test_input, verbose=0)
        print(f"\nTest prediction shape: {pred.shape}")
        print(f"Test prediction: {pred[0]}")
        
        # 예측값 범위 확인
        if output_classes == 2:
            print(f"Binary output - Normal: {pred[0][0]:.4f}, Disease: {pred[0][1]:.4f}")
        
        # 다양한 입력으로 추가 테스트
        print("\nTesting with multiple inputs:")
        for i in range(3):
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32) * (i + 1) * 85
            pred = model.predict(test_input, verbose=0)
            if output_classes == 2:
                print(f"  Test {i+1}: Normal={pred[0][0]:.4f}, Disease={pred[0][1]:.4f}")
        
    except Exception as e:
        print(f"✗ Error converting {model_key}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """모든 classification 모델 변환"""
    print("="*60)
    print("Checkpoint to H5 Conversion Script (Windows Compatible)")
    print("="*60)
    
    # BASE_PATH 확인
    if not BASE_PATH.exists():
        print(f"✗ Error: Base path does not exist - {BASE_PATH}")
        return
    
    print(f"\nStarting checkpoint conversion...")
    
    # Classification 모델들만 변환
    classification_models = {
        k: v for k, v in config['models']['classification'].items()
    }
    
    success_count = 0
    for model_key, model_info in classification_models.items():
        convert_checkpoint_to_h5(model_key, model_info)
        
        # 성공 여부 확인
        h5_path = BASE_PATH / "classification" / model_key / f"{model_key}_model_from_checkpoint.h5"
        if h5_path.exists():
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete! {success_count}/{len(classification_models)} models converted successfully")
    
    # 변환 결과 요약
    print("\n=== Conversion Summary ===")
    for model_key in classification_models:
        model_dir = BASE_PATH / "classification" / model_key
        h5_files = list(model_dir.glob("*.h5"))
        print(f"\n{model_key}:")
        
        if h5_files:
            for h5_file in h5_files:
                size_mb = h5_file.stat().st_size / (1024 * 1024)
                print(f"  - {h5_file.name}: {size_mb:.2f} MB")
                if "from_checkpoint" in h5_file.name:
                    print(f"    ✓ NEW: Converted from checkpoint")
        else:
            print(f"  - No H5 files found")

if __name__ == "__main__":
    main()
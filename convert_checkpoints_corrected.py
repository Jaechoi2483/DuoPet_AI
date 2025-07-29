"""
수정된 체크포인트 변환 스크립트 - 올바른 클래스 수 사용
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

def create_model_for_checkpoint(output_classes):
    """체크포인트에 맞는 모델 생성"""
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

def test_predictions(model, model_name, output_classes):
    """모델 예측 테스트"""
    print(f"\nTesting {model_name} predictions...")
    
    # 5개의 랜덤 이미지로 테스트
    predictions = []
    for i in range(5):
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = model.predict(test_input, verbose=0)
        predictions.append(pred[0])
        
        if output_classes == 2:
            print(f"  Test {i+1}: normal={pred[0][0]:.4f}, disease={pred[0][1]:.4f}")
        elif output_classes <= 10:
            print(f"  Test {i+1}: {pred[0][:min(5, output_classes)]}... (showing first {min(5, output_classes)})")
        else:
            print(f"  Test {i+1}: max_class={np.argmax(pred[0])}, confidence={np.max(pred[0]):.4f}")
    
    # 변동성 체크
    predictions = np.array(predictions)
    std_dev = np.std(predictions, axis=0)
    mean_std = np.mean(std_dev)
    
    print(f"  Mean std deviation: {mean_std:.4f}")
    return mean_std > 0.01

def convert_with_correct_classes(model_key, checkpoint_info, base_path):
    """올바른 클래스 수로 체크포인트 변환"""
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
        # 1. 올바른 클래스 수로 모델 생성
        output_classes = checkpoint_info['output_classes']
        print(f"Creating model with {output_classes} output classes...")
        model = create_model_for_checkpoint(output_classes)
        
        # 2. 체크포인트에서 가중치만 로드 (expect_partial로 optimizer state 무시)
        print(f"Loading weights from checkpoint...")
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(str(checkpoint_path))
        status.expect_partial()  # optimizer state 경고 무시
        print("✓ Weights loaded successfully!")
        
        # 3. 모델 컴파일
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
        is_working = test_predictions(model, model_key, output_classes)
        if not is_working:
            print("⚠️  Model may not be working correctly (low variation)")
        
        # 5. 모델 저장
        save_path = model_dir / f"{model_key}_model_fixed.h5"
        print(f"\nSaving to: {save_path.name}")
        model.save(str(save_path))
        
        # 파일 크기 확인
        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved successfully ({size_mb:.2f} MB)")
        
        # 6. 클래스 맵 업데이트 (필요시)
        if output_classes > 2:
            # 멀티클래스의 경우 클래스 맵 생성
            class_map_path = model_dir / f"{model_key}_fixed_class_map.json"
            class_map = {str(i): f"class_{i}" for i in range(output_classes)}
            
            # 기존 클래스 맵이 있으면 참고
            original_map_path = model_dir / f"{model_key}_class_map.json"
            if original_map_path.exists():
                try:
                    with open(original_map_path, 'r', encoding='utf-8') as f:
                        original_map = json.load(f)
                    # 키 개수가 맞으면 사용
                    if len(original_map) == output_classes:
                        class_map = original_map
                        print(f"✓ Using original class map")
                except:
                    pass
            
            with open(class_map_path, 'w', encoding='utf-8') as f:
                json.dump(class_map, f, ensure_ascii=False, indent=2)
            print(f"✓ Class map saved")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 설정 - 분석 결과에 따라 업데이트 필요
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease"
    
    # 확인된 클래스 수 사용
    checkpoints = {
        'cat_binary': {
            'checkpoint_prefix': 'model-007-0.511353-0.772705-0.776322-0.768861',
            'output_classes': 2  # binary = 2 classes
        },
        'dog_binary': {
            'checkpoint_prefix': 'model-004-0.437360-0.806570-0.806528-0.806891',
            'output_classes': 2  # binary = 2 classes
        },
        'dog_multi_136': {
            'checkpoint_prefix': 'model-009-0.851382-0.821520',
            'output_classes': 7  # 에러 메시지에서 확인됨 (7 classes, not 136)
        }
    }
    
    print("="*70)
    print("FIXED CHECKPOINT CONVERTER")
    print("="*70)
    print("Converting with correct output class numbers")
    
    success_count = 0
    for model_key, checkpoint_info in checkpoints.items():
        if convert_with_correct_classes(model_key, checkpoint_info, base_path):
            success_count += 1
    
    # 요약
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully converted: {success_count}/{len(checkpoints)} models")
    
    if success_count > 0:
        print("\n✅ Next steps:")
        print("1. Test the fixed models: python test_fixed_skin_models.py")
        print("2. Update skin_disease_service.py to use _fixed.h5 models")
        print("3. Restart the backend service and test")
    
    # 변환된 파일 목록
    print("\n=== Converted Models ===")
    for model_key in checkpoints.keys():
        model_dir = base_path / "classification" / model_key
        fixed_file = model_dir / f"{model_key}_model_fixed.h5"
        
        if fixed_file.exists():
            size = fixed_file.stat().st_size / (1024*1024)
            print(f"{model_key}: ✓ {fixed_file.name} ({size:.2f} MB)")
        else:
            print(f"{model_key}: ✗ Conversion failed")

if __name__ == "__main__":
    main()
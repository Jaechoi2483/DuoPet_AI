"""
체크포인트에서 변환된 모델 테스트
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
import json

def test_model(model_path, model_name):
    """모델 테스트"""
    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    
    if not model_path.exists():
        print(f"✗ Model file not found!")
        return False
    
    try:
        # 모델 로드
        model = tf.keras.models.load_model(str(model_path))
        print(f"✓ Model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # 5번의 다른 랜덤 입력으로 테스트
        print("\nTesting with 5 different random inputs:")
        predictions = []
        
        for i in range(5):
            # 랜덤 테스트 입력 (다양한 값 범위)
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            if i % 2 == 0:
                test_input = test_input * 255  # 일부는 0-255 범위로
            
            pred = model.predict(test_input, verbose=0)
            predictions.append(pred[0])
            
            if len(pred[0]) == 2:  # Binary
                print(f"  Test {i+1}: Normal={pred[0][0]:.4f}, Disease={pred[0][1]:.4f}")
            elif len(pred[0]) == 1:  # Single output
                print(f"  Test {i+1}: Disease probability={pred[0][0]:.4f}")
            else:  # Multi-class
                print(f"  Test {i+1}: {pred[0]}")
        
        # 예측값 변동성 확인
        predictions = np.array(predictions)
        std_dev = np.std(predictions, axis=0)
        print(f"\nPrediction std deviation: {std_dev}")
        
        if np.all(std_dev < 0.01):
            print("⚠️  WARNING: Model outputs are nearly constant!")
            return False
        else:
            print("✓ Model outputs show proper variation")
            return True
            
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """모든 모델 테스트"""
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease"
    
    # 테스트할 모델들
    models_to_test = [
        # 기존 모델들
        ("classification/dog_binary/dog_binary_model_tf2.h5", "dog_binary_original"),
        ("classification/cat_binary/cat_binary_model_tf2.h5", "cat_binary_original"),
        
        # 체크포인트에서 변환된 모델들 (예상 경로)
        ("classification/dog_binary/dog_binary_model_from_checkpoint.h5", "dog_binary_from_checkpoint"),
        ("classification/cat_binary/cat_binary_model_from_checkpoint.h5", "cat_binary_from_checkpoint"),
        ("classification/dog_multi_136/dog_multi_136_model_from_checkpoint.h5", "dog_multi_136_from_checkpoint"),
        ("classification/dog_multi_456/dog_multi_456_model_from_checkpoint.h5", "dog_multi_456_from_checkpoint"),
    ]
    
    results = {}
    
    for model_path, model_name in models_to_test:
        full_path = base_path / model_path
        success = test_model(full_path, model_name)
        results[model_name] = success
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("TEST SUMMARY:")
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {model_name}: {status}")
    
    # 권장사항
    print(f"\n{'='*50}")
    print("RECOMMENDATIONS:")
    if not any("from_checkpoint" in name and success for name, success in results.items()):
        print("⚠️  No checkpoint-converted models found or all failed!")
        print("    Please run: python /mnt/d/final_project/DuoPet_AI/convert_all_checkpoints.py")
    else:
        print("✓ Checkpoint-converted models are ready!")
        print("    Next step: Run update_model_registry.py to update the registry")

if __name__ == "__main__":
    main()
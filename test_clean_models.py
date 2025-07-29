"""
깨끗한 모델(clean models) 테스트 스크립트
CustomScaleLayer 없이 변환된 모델들을 테스트
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

def load_class_map(model_dir, model_key):
    """클래스 맵 로드"""
    class_map_files = [
        f"{model_key}_clean_class_map.json",
        f"{model_key}_class_map.json"
    ]
    
    for filename in class_map_files:
        class_map_path = model_dir / filename
        if class_map_path.exists():
            with open(class_map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None

def test_clean_model(model_path, model_key):
    """Clean 모델 테스트"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_path.name}")
    print(f"Model key: {model_key}")
    
    try:
        # 모델 로드 (CustomScaleLayer 없이)
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print(f"✓ Model loaded successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Total parameters: {model.count_params():,}")
        
        # 클래스 맵 로드
        class_map = load_class_map(model_path.parent, model_key)
        if class_map:
            print(f"  Classes: {list(class_map.keys())}")
        
        # 모델 구조 요약 (간단히)
        print("\nModel structure:")
        for i, layer in enumerate(model.layers[:5]):  # 처음 5개 레이어만
            print(f"  Layer {i}: {layer.__class__.__name__} - {layer.name}")
        if len(model.layers) > 5:
            print(f"  ... and {len(model.layers) - 5} more layers")
        
        # 테스트 예측
        print("\nTest predictions:")
        test_cases = [
            ("Black (0)", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("White (255)", np.ones((1, 224, 224, 3), dtype=np.float32) * 255),
            ("Gray (128)", np.ones((1, 224, 224, 3), dtype=np.float32) * 128),
            ("Random 1", np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32)),
            ("Random 2", np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32))
        ]
        
        predictions = []
        for name, test_input in test_cases:
            # 정규화 (0-1 범위)
            test_input_norm = test_input / 255.0
            pred = model.predict(test_input_norm, verbose=0)
            predictions.append(pred[0])
            
            # 결과 출력
            if pred.shape[-1] == 2:  # Binary classification
                print(f"  {name:12s}: normal={pred[0][0]:.4f}, disease={pred[0][1]:.4f}")
            elif pred.shape[-1] == 3:  # 3-class
                print(f"  {name:12s}: class0={pred[0][0]:.4f}, class1={pred[0][1]:.4f}, class2={pred[0][2]:.4f}")
            else:
                print(f"  {name:12s}: {pred[0][:5]}... (showing first 5)")
        
        # 변동성 분석
        predictions = np.array(predictions)
        std_dev = np.std(predictions, axis=0)
        mean_std = np.mean(std_dev)
        
        print(f"\nVariation analysis:")
        print(f"  Std deviation per class: {std_dev}")
        print(f"  Mean std deviation: {mean_std:.4f}")
        
        if mean_std < 0.01:
            print("  ⚠️  WARNING: Low variation - model might not be working correctly")
            return False
        else:
            print("  ✓ Good variation - model appears to be working correctly")
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "classification"
    
    # 테스트할 clean 모델들
    clean_models = [
        ('cat_binary', 'cat_binary_model_clean.h5'),
        ('dog_binary', 'dog_binary_model_clean.h5'),
        ('dog_multi_136', 'dog_multi_136_model_clean.h5')
    ]
    
    print("="*80)
    print("CLEAN MODEL TESTING")
    print("="*80)
    print("Testing models without CustomScaleLayer")
    
    results = {}
    
    for model_key, model_file in clean_models:
        model_path = base_path / model_key / model_file
        if model_path.exists():
            success = test_clean_model(model_path, model_key)
            results[f"{model_key}/{model_file}"] = success
        else:
            print(f"\n✗ Model not found: {model_path}")
            results[f"{model_key}/{model_file}"] = None
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    all_success = True
    for model_name, result in results.items():
        if result is None:
            status = "NOT FOUND"
            all_success = False
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
            all_success = False
        print(f"{model_name:50s} {status}")
    
    # 권장사항
    print("\n" + "="*80)
    if all_success and len(results) > 0:
        print("✅ All clean models are working correctly!")
        print("\nNext steps:")
        print("1. Update skin_disease_service.py to use these clean models")
        print("2. Example configuration:")
        print("   'model_file': 'cat_binary_model_clean.h5'")
        print("   'model_file': 'dog_binary_model_clean.h5'")
        print("3. Restart the backend service")
    else:
        print("⚠️  Some models are missing or not working correctly")
        print("\nTroubleshooting:")
        print("1. Run: python convert_checkpoints_clean.py")
        print("2. Check for error messages above")
        print("3. Verify checkpoint files exist")

if __name__ == "__main__":
    main()
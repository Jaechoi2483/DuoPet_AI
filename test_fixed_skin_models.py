"""
수정된 피부질환 모델 테스트 스크립트
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

def test_model(model_path, model_key):
    """모델 테스트"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_path.name}")
    print(f"Model key: {model_key}")
    
    try:
        # CustomScaleLayer 정의
        class CustomScaleLayer(tf.keras.layers.Layer):
            def __init__(self, scale, **kwargs):
                super(CustomScaleLayer, self).__init__(**kwargs)
                self.scale = scale

            def call(self, inputs):
                if isinstance(inputs, list):
                    return inputs[0] * self.scale
                else:
                    return inputs * self.scale

            def get_config(self):
                config = super(CustomScaleLayer, self).get_config()
                config.update({'scale': self.scale})
                return config
        
        # 모델 로드 (CustomScaleLayer 포함)
        with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
            model = tf.keras.models.load_model(str(model_path), compile=False)
        print(f"✓ Model loaded successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Total parameters: {model.count_params():,}")
        
        # 클래스 맵 확인
        class_map_paths = [
            model_path.parent / f"{model_key}_fixed_class_map.json",
            model_path.parent / f"{model_key}_class_map.json"
        ]
        
        class_map = None
        for map_path in class_map_paths:
            if map_path.exists():
                with open(map_path, 'r', encoding='utf-8') as f:
                    class_map = json.load(f)
                    print(f"  Class map loaded from: {map_path.name}")
                    print(f"  Classes: {list(class_map.values())[:10]}...")
                    break
        
        # 다양한 테스트
        print("\nTest predictions:")
        test_cases = [
            ("Black", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("White", np.ones((1, 224, 224, 3), dtype=np.float32)),
            ("Gray", np.ones((1, 224, 224, 3), dtype=np.float32) * 0.5),
            ("Random 1", np.random.random((1, 224, 224, 3)).astype(np.float32)),
            ("Random 2", np.random.random((1, 224, 224, 3)).astype(np.float32)),
            ("Random 3", np.random.random((1, 224, 224, 3)).astype(np.float32))
        ]
        
        predictions = []
        for name, test_input in test_cases:
            pred = model.predict(test_input, verbose=0)
            predictions.append(pred[0])
            
            if pred.shape[-1] == 2:  # Binary
                print(f"  {name:10s}: normal={pred[0][0]:.4f}, disease={pred[0][1]:.4f}")
            elif pred.shape[-1] <= 7:  # Multi-class (small)
                top_3 = np.argsort(pred[0])[-3:][::-1]
                print(f"  {name:10s}: top_3_classes={top_3}, probs={pred[0][top_3]}")
            else:  # Multi-class (large)
                top_class = np.argmax(pred[0])
                top_prob = np.max(pred[0])
                print(f"  {name:10s}: top_class={top_class}, prob={top_prob:.4f}")
        
        # 변동성 분석
        predictions = np.array(predictions)
        std_dev = np.std(predictions, axis=0)
        mean_std = np.mean(std_dev)
        
        print(f"\nVariation analysis:")
        print(f"  Mean std deviation: {mean_std:.4f}")
        if pred.shape[-1] <= 10:
            print(f"  Std per class: {std_dev}")
        
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
    
    # 테스트할 모델들
    models_to_test = [
        # Fixed 모델
        ('cat_binary', 'cat_binary_model_fixed.h5'),
        ('dog_binary', 'dog_binary_model_fixed.h5'),
        ('dog_multi_136', 'dog_multi_136_model_fixed.h5'),
        # Clean 모델 (있다면)
        ('cat_binary', 'cat_binary_model_clean.h5'),
        ('dog_binary', 'dog_binary_model_clean.h5'),
        ('dog_multi_136', 'dog_multi_136_model_clean.h5'),
        # 원본 모델
        ('cat_binary', 'cat_binary_model.h5'),
        ('dog_binary', 'dog_binary_model.h5'),
    ]
    
    print("="*80)
    print("FIXED SKIN DISEASE MODEL TESTING")
    print("="*80)
    
    results = {}
    
    for model_key, model_file in models_to_test:
        model_path = base_path / model_key / model_file
        if model_path.exists():
            success = test_model(model_path, model_key)
            results[f"{model_key}/{model_file}"] = success
        else:
            results[f"{model_key}/{model_file}"] = None
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    working_models = {}
    for model_name, result in results.items():
        if result is None:
            status = "NOT FOUND"
        elif result:
            status = "✓ WORKING"
            model_key = model_name.split('/')[0]
            if model_key not in working_models:
                working_models[model_key] = []
            working_models[model_key].append(model_name.split('/')[1])
        else:
            status = "✗ FAILED"
        print(f"{model_name:50s} {status}")
    
    # 권장사항
    print("\n" + "="*80)
    if working_models:
        print("✅ Recommended models for each type:")
        for model_type, files in working_models.items():
            # Fixed 모델 우선
            if any('fixed' in f for f in files):
                recommended = next(f for f in files if 'fixed' in f)
            elif any('clean' in f for f in files):
                recommended = next(f for f in files if 'clean' in f)
            else:
                recommended = files[0]
            print(f"  {model_type}: {recommended}")
        
        print("\n✅ Next steps:")
        print("1. Update skin_disease_service.py with the recommended models")
        print("2. Restart the backend service")
        print("3. Test with actual skin disease images")
    else:
        print("❌ No working models found!")
        print("\nTroubleshooting:")
        print("1. Run: python analyze_checkpoint_structure.py")
        print("2. Update convert_checkpoints_corrected.py with correct class numbers")
        print("3. Run: python convert_checkpoints_corrected.py")

if __name__ == "__main__":
    main()
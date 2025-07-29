"""
수정된 모델 테스트 스크립트
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path

# 수정된 CustomScaleLayer
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

def test_model(model_path):
    """모델 테스트"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_path.name}")
    
    try:
        # 모델 로드
        with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
            model = tf.keras.models.load_model(str(model_path), compile=False)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # 다양한 테스트 케이스
        test_cases = [
            ("Black image", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("White image", np.ones((1, 224, 224, 3), dtype=np.float32) * 255),
            ("Gray image", np.ones((1, 224, 224, 3), dtype=np.float32) * 128),
            ("Random image", np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32))
        ]
        
        print("\nPredictions (0-255 range -> normalized):")
        predictions = []
        
        for name, test_input in test_cases:
            # 정규화
            test_input_norm = test_input / 255.0
            pred = model.predict(test_input_norm, verbose=0)
            predictions.append(pred[0])
            
            if pred.shape[-1] == 2:
                print(f"  {name:12s}: normal={pred[0][0]:.4f}, disease={pred[0][1]:.4f}")
            else:
                print(f"  {name:12s}: {pred[0]}")
        
        # 변동성 확인
        predictions = np.array(predictions)
        std_dev = np.std(predictions, axis=0)
        
        print(f"\nOutput variation (std dev): {std_dev}")
        if np.all(std_dev < 0.01):
            print("  ⚠️  WARNING: Model outputs are nearly constant!")
            return False
        else:
            print("  ✓ Model outputs show healthy variation")
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
        # 원본 모델
        ("dog_binary", "dog_binary_model.h5"),
        ("cat_binary", "cat_binary_model.h5"),
        # v2 모델
        ("dog_binary", "dog_binary_model_from_checkpoint_v2.h5"),
        ("cat_binary", "cat_binary_model_from_checkpoint_v2.h5"),
        # 수정된 모델 (있다면)
        ("dog_binary", "dog_binary_model_from_checkpoint_v2_fixed.h5"),
        ("cat_binary", "cat_binary_model_from_checkpoint_v2_fixed.h5"),
    ]
    
    print("="*80)
    print("MODEL TESTING SCRIPT")
    print("="*80)
    
    results = {}
    
    for model_dir, model_file in models_to_test:
        model_path = base_path / model_dir / model_file
        if model_path.exists():
            success = test_model(model_path)
            results[model_file] = success
        else:
            print(f"\n✗ Model not found: {model_path}")
            results[model_file] = None
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for model_name, result in results.items():
        if result is None:
            status = "NOT FOUND"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"{model_name:50s} {status}")
    
    # 권장사항
    print("\n✅ Recommendations:")
    working_models = [name for name, result in results.items() if result is True]
    if working_models:
        print(f"Use one of these working models: {', '.join(working_models)}")
    else:
        print("No working models found. Need to investigate further.")

if __name__ == "__main__":
    main()
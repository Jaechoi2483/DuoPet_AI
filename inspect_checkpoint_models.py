"""
체크포인트 변환 모델 검사 - CustomScaleLayer 포함
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path

# CustomScaleLayer 정의
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config

def inspect_checkpoint_model(model_path):
    """체크포인트 변환 모델 검사"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {model_path.name}")
    print(f"File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # CustomScaleLayer를 포함하여 모델 로드
        with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
            model = tf.keras.models.load_model(str(model_path), compile=False)
        
        # 기본 정보
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Total parameters: {model.count_params():,}")
        
        # 마지막 레이어 정보
        last_layer = model.layers[-1]
        print(f"\nLast layer:")
        print(f"  Type: {last_layer.__class__.__name__}")
        if hasattr(last_layer, 'units'):
            print(f"  Units: {last_layer.units}")
        if hasattr(last_layer, 'activation'):
            activation = last_layer.activation
            if hasattr(activation, '__name__'):
                print(f"  Activation: {activation.__name__}")
            else:
                print(f"  Activation: {activation}")
        
        # 가중치 분석
        print("\nWeight analysis:")
        total_weights = 0
        non_zero_weights = 0
        
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                for w in weights:
                    total_weights += w.size
                    non_zero_weights += np.count_nonzero(w)
        
        if total_weights > 0:
            zero_ratio = 1 - (non_zero_weights / total_weights)
            print(f"  Total weight parameters: {total_weights:,}")
            print(f"  Non-zero weights: {non_zero_weights:,}")
            print(f"  Zero ratio: {zero_ratio:.2%}")
        
        # 테스트 예측
        print("\nTest predictions (normalized inputs):")
        test_cases = [
            ("black (0.0)", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("white (1.0)", np.ones((1, 224, 224, 3), dtype=np.float32)),
            ("gray (0.5)", np.ones((1, 224, 224, 3), dtype=np.float32) * 0.5),
            ("random", np.random.random((1, 224, 224, 3)).astype(np.float32))
        ]
        
        for name, test_input in test_cases:
            pred = model.predict(test_input, verbose=0)
            if pred.shape[-1] == 2:
                print(f"  {name}: normal={pred[0][0]:.4f}, disease={pred[0][1]:.4f}")
            else:
                print(f"  {name}: {pred[0]}")
        
        # 극단적 테스트 (0-255 범위)
        print("\nTest predictions (0-255 range inputs):")
        test_cases_255 = [
            ("black", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("white", np.ones((1, 224, 224, 3), dtype=np.float32) * 255),
            ("random", np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32))
        ]
        
        for name, test_input in test_cases_255:
            # 정규화
            test_input_norm = test_input / 255.0
            pred = model.predict(test_input_norm, verbose=0)
            if pred.shape[-1] == 2:
                print(f"  {name}: normal={pred[0][0]:.4f}, disease={pred[0][1]:.4f}")
            else:
                print(f"  {name}: {pred[0]}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "classification"
    
    # 체크포인트 변환 모델들 검사
    models = [
        base_path / "dog_binary" / "dog_binary_model_from_checkpoint_v2.h5",
        base_path / "cat_binary" / "cat_binary_model_from_checkpoint_v2.h5"
    ]
    
    print("="*80)
    print("CHECKPOINT MODEL INSPECTION WITH CUSTOMSCALELAYER")
    print("="*80)
    
    success_count = 0
    for model_path in models:
        if model_path.exists():
            if inspect_checkpoint_model(model_path):
                success_count += 1
        else:
            print(f"\nModel not found: {model_path}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully inspected: {success_count}/{len(models)} models")
    print("\nKey observations:")
    print("1. Output units: Check if it's 2 (normal, disease) vs 1")
    print("2. Predictions: Should vary significantly between test inputs")
    print("3. File size: 231MB indicates full weights are present")
    print("4. Architecture: InceptionResNetV2 vs MobileNetV2")

if __name__ == "__main__":
    main()
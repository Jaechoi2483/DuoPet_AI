"""
CustomScaleLayer 문제 해결 및 모델 재변환 스크립트
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path

# 수정된 CustomScaleLayer - 리스트 입력 처리
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        # inputs가 리스트인 경우 처리
        if isinstance(inputs, list):
            # 리스트의 첫 번째 요소만 사용 (또는 평균을 낼 수도 있음)
            return inputs[0] * self.scale
        else:
            # 단일 텐서인 경우 기존 방식대로 처리
            return inputs * self.scale

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config

def load_and_fix_model(model_path):
    """모델을 로드하고 CustomScaleLayer 문제를 해결"""
    print(f"\n{'='*80}")
    print(f"Processing: {model_path.name}")
    print(f"File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # CustomScaleLayer를 포함하여 모델 로드
        with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
            model = tf.keras.models.load_model(str(model_path), compile=False)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # 모델 테스트
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = model.predict(test_input, verbose=0)
        print(f"  Test prediction shape: {pred.shape}")
        print(f"  Test prediction: {pred[0]}")
        
        return model, True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, False

def save_fixed_model(model, original_path, suffix="_fixed"):
    """수정된 모델 저장"""
    # 새 파일명 생성
    stem = original_path.stem
    new_name = f"{stem}{suffix}.h5"
    new_path = original_path.parent / new_name
    
    try:
        model.save(str(new_path))
        print(f"✓ Saved fixed model: {new_name}")
        return True
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False

def main():
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "classification"
    
    # 문제가 있는 v2 모델들 처리
    models_to_fix = [
        base_path / "dog_binary" / "dog_binary_model_from_checkpoint_v2.h5",
        base_path / "cat_binary" / "cat_binary_model_from_checkpoint_v2.h5"
    ]
    
    print("="*80)
    print("CUSTOMSCALELAYER FIX SCRIPT")
    print("="*80)
    
    success_count = 0
    for model_path in models_to_fix:
        if model_path.exists():
            model, loaded = load_and_fix_model(model_path)
            if loaded and model:
                if save_fixed_model(model, model_path):
                    success_count += 1
        else:
            print(f"\nModel not found: {model_path}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully fixed: {success_count}/{len(models_to_fix)} models")
    
    # 권장사항
    print("\n✅ Next steps:")
    print("1. Test the fixed models with: python test_fixed_models.py")
    print("2. Update the service to use fixed models")
    print("3. Or use the original models without CustomScaleLayer")

if __name__ == "__main__":
    main()
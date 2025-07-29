"""
모든 h5 모델 파일 테스트
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

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

def test_model(model_path):
    """모델 간단 테스트"""
    try:
        # 모델 로드 시도
        with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
            model = tf.keras.models.load_model(str(model_path), compile=False)
        
        # 테스트 입력
        test_inputs = [
            np.zeros((1, 224, 224, 3), dtype=np.float32),
            np.ones((1, 224, 224, 3), dtype=np.float32),
            np.random.random((1, 224, 224, 3)).astype(np.float32)
        ]
        
        predictions = []
        for test_input in test_inputs:
            pred = model.predict(test_input, verbose=0)
            predictions.append(pred[0])
        
        # 변동성 계산
        predictions = np.array(predictions)
        std_dev = np.std(predictions, axis=0)
        mean_std = np.mean(std_dev)
        
        return {
            'success': True,
            'output_shape': model.output_shape,
            'params': model.count_params(),
            'variation': mean_std,
            'working': mean_std > 0.001
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)[:100]
        }

def main():
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "classification"
    
    # 각 모델 타입별로 테스트
    model_types = {
        'cat_binary': [],
        'dog_binary': [],
        'dog_multi_136': [],
        'dog_multi_456': []
    }
    
    print("="*80)
    print("TESTING ALL H5 MODELS")
    print("="*80)
    
    # 모든 h5 파일 찾기
    for model_type in model_types.keys():
        model_dir = base_path / model_type
        if model_dir.exists():
            h5_files = list(model_dir.glob("*.h5"))
            
            print(f"\n{model_type.upper()} MODELS ({len(h5_files)} files)")
            print("-"*60)
            
            for h5_file in sorted(h5_files):
                result = test_model(h5_file)
                
                if result['success']:
                    status = "✓ WORKING" if result['working'] else "✗ NO VARIATION"
                    print(f"{h5_file.name:45s} {status}")
                    print(f"  Output: {result['output_shape']}, Params: {result['params']:,}, Var: {result['variation']:.4f}")
                    
                    if result['working']:
                        model_types[model_type].append(h5_file.name)
                else:
                    print(f"{h5_file.name:45s} ✗ LOAD FAILED")
                    print(f"  Error: {result['error']}")
    
    # 권장사항
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    for model_type, working_models in model_types.items():
        if working_models:
            print(f"\n{model_type}:")
            print(f"  Use: {working_models[0]}")
            if len(working_models) > 1:
                print(f"  Alternatives: {', '.join(working_models[1:])}")
        else:
            print(f"\n{model_type}:")
            print(f"  ⚠️  No working models found!")
    
    # 서비스 업데이트 제안
    print("\n" + "="*80)
    print("Update skin_disease_service.py with:")
    print("="*80)
    
    for model_type, working_models in model_types.items():
        if working_models:
            print(f"'{model_type}': {{'model_file': '{working_models[0]}', ...}}")

if __name__ == "__main__":
    main()
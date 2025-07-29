"""
모델 파일들을 검사하여 구조와 가중치 상태 확인
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path

def inspect_model(model_path):
    """모델 검사"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {model_path.name}")
    print(f"File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # 모델 로드
        model = tf.keras.models.load_model(str(model_path), compile=False)
        
        # 기본 정보
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Total parameters: {model.count_params():,}")
        
        # 레이어 정보
        print("\nLayer summary:")
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'units'):
                print(f"  {layer.name}: {layer.__class__.__name__} (units={layer.units})")
            else:
                print(f"  {layer.name}: {layer.__class__.__name__}")
        
        # 마지막 레이어 상세 정보
        last_layer = model.layers[-1]
        print(f"\nLast layer details:")
        print(f"  Type: {last_layer.__class__.__name__}")
        if hasattr(last_layer, 'units'):
            print(f"  Units: {last_layer.units}")
        if hasattr(last_layer, 'activation'):
            activation = last_layer.activation
            if hasattr(activation, '__name__'):
                print(f"  Activation: {activation.__name__}")
            elif hasattr(activation, 'name'):
                print(f"  Activation: {activation.name}")
            else:
                print(f"  Activation: {activation}")
        
        # 가중치 검사
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
            
            # 첫 번째 Conv 레이어의 가중치 샘플
            for layer in model.layers:
                if 'conv' in layer.name.lower() and layer.get_weights():
                    w = layer.get_weights()[0]
                    print(f"\n  Sample from {layer.name}:")
                    print(f"    Shape: {w.shape}")
                    print(f"    Mean: {np.mean(w):.6f}")
                    print(f"    Std: {np.std(w):.6f}")
                    print(f"    Min: {np.min(w):.6f}")
                    print(f"    Max: {np.max(w):.6f}")
                    break
        
        # 테스트 예측
        print("\nTest predictions:")
        test_inputs = [
            ("zeros", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("ones", np.ones((1, 224, 224, 3), dtype=np.float32)),
            ("random", np.random.random((1, 224, 224, 3)).astype(np.float32))
        ]
        
        for name, test_input in test_inputs:
            pred = model.predict(test_input, verbose=0)
            print(f"  {name}: {pred[0]}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def main():
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "classification"
    
    # 검사할 모델들
    models_to_check = [
        # Dog binary models
        base_path / "dog_binary" / "dog_binary_model.h5",
        base_path / "dog_binary" / "dog_binary_model_from_checkpoint_v2.h5",
        # Cat binary models  
        base_path / "cat_binary" / "cat_binary_model.h5",
        base_path / "cat_binary" / "cat_binary_model_from_checkpoint_v2.h5",
    ]
    
    print("="*80)
    print("MODEL INSPECTION REPORT")
    print("="*80)
    
    for model_path in models_to_check:
        if model_path.exists():
            inspect_model(model_path)
        else:
            print(f"\nModel not found: {model_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("1. Compare file sizes - larger models should have more parameters")
    print("2. Check if weights are mostly zeros (indicating corruption)")
    print("3. Verify output units match expected format")
    print("4. Test predictions should vary between inputs")

if __name__ == "__main__":
    main()
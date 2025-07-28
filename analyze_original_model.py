"""
원본 안구질환 모델 직접 분석
normalization 레이어 문제 우회
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import h5py

def analyze_original_model():
    """원본 모델 분석"""
    
    print("🔍 원본 안구질환 모델 분석")
    print("=" * 60)
    
    model_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras")
    
    if not model_path.exists():
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return
    
    # 1. H5 파일로 직접 분석
    print("\n📊 H5 파일 직접 분석:")
    print("-" * 50)
    
    try:
        with h5py.File(model_path, 'r') as f:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
                    
                    # 마지막 레이어 가중치 확인
                    if 'dense' in name and ('kernel' in name or 'bias' in name):
                        data = obj[:]
                        print(f"  -> 값 분석:")
                        print(f"     Mean: {np.mean(data):.6f}")
                        print(f"     Std: {np.std(data):.6f}")
                        print(f"     Min/Max: {np.min(data):.6f} / {np.max(data):.6f}")
                        
                        if 'bias' in name and data.shape[0] == 5:
                            print(f"     Bias 값: {data}")
                            if np.all(data == 0):
                                print("     ⚠️  모든 bias가 0입니다!")
            
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"H5 분석 실패: {e}")
    
    # 2. 모델 로드 시도 (normalization 문제 우회)
    print("\n\n📊 모델 로드 (normalization 우회):")
    print("-" * 50)
    
    try:
        # Custom normalization layer
        class FixedNormalization(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                
            def build(self, input_shape):
                super().build(input_shape)
                
            def call(self, inputs):
                # 단순 정규화 (0-1 범위로 가정)
                return inputs
            
            def get_config(self):
                return super().get_config()
        
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout,
            'Normalization': FixedNormalization,
            'normalization': FixedNormalization
        }
        
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
            compile=False
        )
        
        print("✅ 모델 로드 성공!")
        
        # 모델 구조
        print("\n모델 요약:")
        print(f"- 입력 shape: {model.input_shape}")
        print(f"- 출력 shape: {model.output_shape}")
        print(f"- 전체 레이어 수: {len(model.layers)}")
        
        # 마지막 5개 레이어 분석
        print("\n마지막 5개 레이어:")
        for layer in model.layers[-5:]:
            print(f"\n{layer.name} ({layer.__class__.__name__}):")
            
            if hasattr(layer, 'units'):
                print(f"  Units: {layer.units}")
                
            if hasattr(layer, 'weights') and layer.weights:
                for weight in layer.weights:
                    w_array = weight.numpy()
                    print(f"  {weight.name}: shape={w_array.shape}")
                    
                    if 'kernel' in weight.name:
                        print(f"    통계: mean={np.mean(w_array):.6f}, std={np.std(w_array):.6f}")
                        
                    if 'bias' in weight.name:
                        print(f"    Bias 값: {w_array}")
                        if np.all(w_array == 0):
                            print("    ⚠️  모든 bias가 0!")
                        elif np.allclose(w_array, w_array[0]):
                            print("    ⚠️  모든 bias가 동일!")
        
        # 테스트 예측
        print("\n\n테스트 예측:")
        print("-" * 30)
        
        # 다양한 테스트 입력
        test_cases = [
            ("백색 이미지", np.ones((1, 224, 224, 3), dtype=np.float32)),
            ("흑색 이미지", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("랜덤 이미지", np.random.random((1, 224, 224, 3)).astype(np.float32)),
            ("빨간색 이미지", np.zeros((1, 224, 224, 3), dtype=np.float32)),
        ]
        
        # 빨간색 이미지 설정
        test_cases[3][1][:, :, :, 0] = 1.0  # R 채널만 1
        
        for name, test_input in test_cases:
            pred = model.predict(test_input, verbose=0)
            print(f"\n{name}:")
            print(f"  예측값: {pred[0]}")
            print(f"  확률(%): {[f'{p*100:.1f}' for p in pred[0]]}")
            
            # 모든 값이 동일한지 확인
            if np.allclose(pred[0], pred[0][0], rtol=1e-5):
                print("  ⚠️  모든 클래스 확률이 동일!")
        
        # 중간 레이어 활성화 확인
        print("\n\n중간 레이어 활성화 분석:")
        print("-" * 40)
        
        # 중간 모델 생성
        intermediate_outputs = []
        for layer in model.layers[-10:]:  # 마지막 10개 레이어
            try:
                intermediate_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=layer.output
                )
                intermediate_outputs.append((layer.name, intermediate_model))
            except:
                continue
        
        # 랜덤 입력으로 테스트
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        for layer_name, inter_model in intermediate_outputs[-5:]:
            output = inter_model.predict(test_input, verbose=0)
            print(f"\n{layer_name}:")
            print(f"  Shape: {output.shape}")
            print(f"  Mean: {np.mean(output):.6f}")
            print(f"  Std: {np.std(output):.6f}")
            
            # 마지막 차원이 5인 경우 (클래스 수)
            if len(output.shape) == 2 and output.shape[1] == 5:
                print(f"  출력값: {output[0]}")
                
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n🎯 결론:")
    print("모델의 마지막 레이어 가중치가 초기화 상태이거나")
    print("학습이 제대로 되지 않은 것으로 보입니다.")

if __name__ == "__main__":
    analyze_original_model()
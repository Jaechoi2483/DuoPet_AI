"""
안구질환 모델 가중치 심층 분석
모든 클래스가 20%로 나오는 문제 원인 파악
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

def analyze_model_weights():
    """모델 가중치 분석"""
    
    print("🔍 안구질환 모델 가중치 분석")
    print("=" * 60)
    
    # 모델 경로들
    model_paths = [
        Path("models/health_diagnosis/eye_disease/best_grouped_model.keras"),
        Path("/mnt/c/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras")
    ]
    
    for model_path in model_paths:
        if not model_path.exists():
            continue
            
        print(f"\n📊 모델 분석: {model_path}")
        print("-" * 50)
        
        try:
            # 모델 로드
            custom_objects = {
                'swish': tf.nn.swish,
                'Swish': tf.keras.layers.Activation(tf.nn.swish),
                'FixedDropout': tf.keras.layers.Dropout
            }
            
            model = tf.keras.models.load_model(
                str(model_path),
                custom_objects=custom_objects,
                compile=False
            )
            
            # 모델 구조 출력
            print("\n모델 구조:")
            model.summary()
            
            # 마지막 레이어 분석
            print("\n마지막 레이어 분석:")
            for layer in model.layers[-5:]:
                print(f"  - {layer.name}: {layer.__class__.__name__}")
                if hasattr(layer, 'weights'):
                    for weight in layer.weights:
                        weight_array = weight.numpy()
                        print(f"    {weight.name}: shape={weight_array.shape}")
                        
                        # Dense 레이어의 가중치 분석
                        if 'dense' in layer.name.lower() and 'kernel' in weight.name:
                            print(f"      Mean: {np.mean(weight_array):.6f}")
                            print(f"      Std: {np.std(weight_array):.6f}")
                            print(f"      Min: {np.min(weight_array):.6f}")
                            print(f"      Max: {np.max(weight_array):.6f}")
                            
                        # Bias 분석
                        if 'bias' in weight.name:
                            print(f"      Bias values: {weight_array}")
                            if np.all(weight_array == 0):
                                print("      ⚠️  모든 bias가 0입니다!")
            
            # 테스트 입력으로 예측
            print("\n테스트 예측:")
            test_inputs = [
                np.ones((1, 224, 224, 3), dtype=np.float32),
                np.zeros((1, 224, 224, 3), dtype=np.float32),
                np.random.random((1, 224, 224, 3)).astype(np.float32)
            ]
            
            for i, test_input in enumerate(test_inputs):
                prediction = model.predict(test_input, verbose=0)
                print(f"  테스트 {i+1}: {prediction[0]}")
                print(f"    확률: {[f'{p:.1%}' for p in prediction[0]]}")
            
            # 중간 레이어 출력 확인
            print("\n중간 레이어 출력 분석:")
            intermediate_model = tf.keras.Model(
                inputs=model.input,
                outputs=[layer.output for layer in model.layers[-5:]]
            )
            
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            intermediate_outputs = intermediate_model.predict(test_input, verbose=0)
            
            for i, (layer, output) in enumerate(zip(model.layers[-5:], intermediate_outputs)):
                print(f"\n  {layer.name}:")
                print(f"    Shape: {output.shape}")
                print(f"    Mean: {np.mean(output):.6f}")
                print(f"    Std: {np.std(output):.6f}")
                print(f"    Min: {np.min(output):.6f}")
                print(f"    Max: {np.max(output):.6f}")
                
                # 모든 값이 동일한지 확인
                if len(output.shape) == 2 and output.shape[1] == 5:
                    print(f"    출력값: {output[0]}")
                    if np.allclose(output[0], output[0][0]):
                        print("    ⚠️  모든 출력이 동일합니다!")
            
        except Exception as e:
            print(f"❌ 모델 분석 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n💡 분석 완료")
    print("\n가능한 원인:")
    print("1. 모델이 제대로 학습되지 않음")
    print("2. 마지막 레이어의 가중치가 초기화 상태")
    print("3. 전이학습 시 마지막 레이어만 재학습이 필요")

if __name__ == "__main__":
    analyze_model_weights()
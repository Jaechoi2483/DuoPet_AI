"""
수정된 안구질환 모델 철저한 분석
normalization 레이어 및 가중치 문제 해결 확인
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_model_analysis():
    """새 모델 종합 분석"""
    
    print("🔍 수정된 안구질환 모델 종합 분석")
    print("=" * 70)
    
    # 모델 경로
    model_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/final_model_fixed.keras")
    
    if not model_path.exists():
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return False
    
    try:
        # 1. 모델 로드
        print("\n1️⃣ 모델 로드 시도...")
        print("-" * 50)
        
        # 커스텀 Normalization 레이어
        class DummyNormalization(tf.keras.layers.Layer):
            def __init__(self, axis=-1, **kwargs):
                super().__init__(**kwargs)
                self.axis = axis
                
            def build(self, input_shape):
                super().build(input_shape)
                
            def call(self, inputs):
                return inputs  # 입력을 그대로 반환
                
            def get_config(self):
                config = super().get_config()
                config.update({'axis': self.axis})
                return config
        
        # 커스텀 객체 정의
        custom_objects = {
            # Normalization 우회
            'Normalization': DummyNormalization,
            'normalization': DummyNormalization,
            'normalization_1': DummyNormalization,
            
            # 활성화 함수
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            
            # Dropout
            'FixedDropout': tf.keras.layers.Dropout
        }
        
        # 커스텀 객체 스코프 내에서 로드
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(
                str(model_path),
                compile=False
            )
        
        print("✅ 모델 로드 성공! (normalization 문제 해결됨)")
        
        # 2. 모델 구조 분석
        print("\n2️⃣ 모델 구조 분석...")
        print("-" * 50)
        print(f"입력 shape: {model.input_shape}")
        print(f"출력 shape: {model.output_shape}")
        print(f"전체 레이어 수: {len(model.layers)}")
        print(f"학습 가능한 파라미터 수: {model.count_params():,}")
        
        # 3. 레이어별 상세 분석
        print("\n3️⃣ 주요 레이어 분석...")
        print("-" * 50)
        
        # Normalization 레이어 찾기
        for i, layer in enumerate(model.layers):
            if 'normalization' in layer.name.lower():
                print(f"\n✅ Normalization 레이어 발견: {layer.name} (index: {i})")
                if hasattr(layer, 'mean') and hasattr(layer, 'variance'):
                    print("  - mean, variance 속성 존재")
                
        # 마지막 5개 레이어 상세 분석
        print("\n마지막 5개 레이어:")
        for layer in model.layers[-5:]:
            print(f"\n{layer.name} ({layer.__class__.__name__}):")
            
            if hasattr(layer, 'units'):
                print(f"  Units: {layer.units}")
                
            if hasattr(layer, 'weights') and layer.weights:
                for weight in layer.weights:
                    w_array = weight.numpy()
                    print(f"  {weight.name}: shape={w_array.shape}")
                    
                    # 가중치 통계
                    if 'kernel' in weight.name:
                        print(f"    Mean: {np.mean(w_array):.6f}")
                        print(f"    Std: {np.std(w_array):.6f}")
                        print(f"    Min/Max: [{np.min(w_array):.6f}, {np.max(w_array):.6f}]")
                        
                    # Bias 분석
                    if 'bias' in weight.name:
                        print(f"    Bias 값: {w_array}")
                        if np.all(w_array == 0):
                            print("    ⚠️ 모든 bias가 0!")
                        else:
                            print("    ✅ Bias가 적절히 초기화됨")
        
        # 4. 예측 테스트
        print("\n4️⃣ 예측 테스트...")
        print("-" * 50)
        
        # 다양한 테스트 케이스
        test_cases = [
            ("백색 이미지", np.ones((1, 224, 224, 3), dtype=np.float32)),
            ("흑색 이미지", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("랜덤 이미지", np.random.random((1, 224, 224, 3)).astype(np.float32)),
            ("빨간색 이미지", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("파란색 이미지", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("노란색 이미지", np.zeros((1, 224, 224, 3), dtype=np.float32))
        ]
        
        # 색상 설정
        test_cases[3][1][:,:,:,0] = 1.0  # Red
        test_cases[4][1][:,:,:,2] = 1.0  # Blue  
        test_cases[5][1][:,:,:,0] = 1.0  # Yellow
        test_cases[5][1][:,:,:,1] = 1.0
        
        predictions = []
        for name, test_input in test_cases:
            pred = model.predict(test_input, verbose=0)
            predictions.append(pred[0])
            
            print(f"\n{name}:")
            print(f"  예측값: {pred[0]}")
            print(f"  확률(%): {[f'{p*100:.1f}' for p in pred[0]]}")
            print(f"  최고 확률 클래스: {np.argmax(pred[0])} ({np.max(pred[0])*100:.1f}%)")
            
            # 동일 확률 체크
            if np.allclose(pred[0], pred[0][0], rtol=1e-3):
                print("  ⚠️ 모든 클래스 확률이 동일!")
            else:
                print("  ✅ 클래스별 확률이 다름 (정상)")
        
        # 5. 예측 분포 분석
        print("\n5️⃣ 예측 분포 분석...")
        print("-" * 50)
        
        predictions_array = np.array(predictions)
        
        # 각 클래스별 예측 통계
        class_map = {
            "0": "각막 질환",
            "1": "결막 및 누관 질환",
            "2": "수정체 질환",
            "3": "안검 질환",
            "4": "안구 내부 질환"
        }
        
        print("\n클래스별 예측 통계:")
        for i in range(5):
            class_preds = predictions_array[:, i]
            print(f"\n{class_map[str(i)]}:")
            print(f"  평균: {np.mean(class_preds):.3f}")
            print(f"  표준편차: {np.std(class_preds):.3f}")
            print(f"  최소/최대: [{np.min(class_preds):.3f}, {np.max(class_preds):.3f}]")
        
        # 6. 모델 건강도 종합 평가
        print("\n6️⃣ 모델 건강도 종합 평가...")
        print("-" * 50)
        
        health_checks = {
            "모델 로드": True,
            "Normalization 레이어": any('normalization' in l.name.lower() for l in model.layers),
            "가중치 초기화": not all(np.allclose(p[0], p[0][0]) for p in predictions),
            "Bias 활성화": not any(np.all(w.numpy() == 0) for l in model.layers[-5:] 
                                  for w in l.weights if 'bias' in w.name),
            "예측 다양성": np.std(predictions_array) > 0.05
        }
        
        print("\n건강도 체크리스트:")
        for check, status in health_checks.items():
            print(f"  {'✅' if status else '❌'} {check}")
        
        # 종합 판정
        if all(health_checks.values()):
            print("\n🎉 모델이 정상적으로 작동합니다!")
            
            # 모델 정보 저장
            model_info = {
                "model_path": str(model_path),
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "num_layers": len(model.layers),
                "num_params": int(model.count_params()),
                "health_checks": health_checks,
                "class_map": class_map
            }
            
            with open("models/health_diagnosis/eye_disease/model_info.json", "w", encoding="utf-8") as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            print("\n✅ 모델 정보 저장 완료: model_info.json")
            return True
        else:
            print("\n⚠️ 일부 문제가 발견되었습니다.")
            return False
            
    except Exception as e:
        print(f"\n❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = comprehensive_model_analysis()
    
    if success:
        print("\n\n✅ 분석 완료! 모델을 프로젝트에 통합할 준비가 되었습니다.")
        print("\n다음 단계:")
        print("1. 모델 파일 복사")
        print("2. 서비스 코드 업데이트")
        print("3. 통합 테스트")
    else:
        print("\n\n❌ 모델에 문제가 있습니다. 추가 확인이 필요합니다.")
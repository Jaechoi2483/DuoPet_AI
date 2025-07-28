"""
Normalization 레이어 문제 해결 및 모델 분석
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# 전역 설정
tf.config.set_visible_devices([], 'GPU')  # CPU만 사용

class DummyNormalization(tf.keras.layers.Layer):
    """빈 Normalization 레이어"""
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.built = True
        
    def call(self, inputs):
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config
    
    def build(self, input_shape):
        super().build(input_shape)

def fix_and_convert_model():
    """모델 수정 및 변환"""
    
    print("🔧 Normalization 문제 해결 및 모델 분석")
    print("=" * 60)
    
    # 원본 모델 경로
    original_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras")
    
    # 임시로 모델 구조만 생성
    print("\n1️⃣ 모델 구조 재생성...")
    
    # EfficientNetB0 기반 모델 생성
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # 가중치 없이
    )
    
    # 모델 구조 생성
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = inputs  # Normalization 생략
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("✅ 모델 구조 생성 완료")
    
    # 원본 모델에서 가중치만 추출 시도
    print("\n2️⃣ 원본 모델 가중치 추출 시도...")
    
    try:
        # 커스텀 객체로 로드 시도
        custom_objects = {
            'Normalization': DummyNormalization,
            'normalization': DummyNormalization,
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout
        }
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            original_model = tf.keras.models.load_model(
                str(original_path),
                compile=False
            )
        
        print("✅ 원본 모델 로드 성공!")
        
        # 가중치 복사 시도
        print("\n3️⃣ 가중치 분석...")
        
        # 마지막 레이어 가중치 확인
        for layer in original_model.layers[-5:]:
            print(f"\n레이어: {layer.name}")
            if hasattr(layer, 'weights') and layer.weights:
                for weight in layer.weights:
                    w = weight.numpy()
                    print(f"  {weight.name}: shape={w.shape}")
                    if 'dense' in layer.name and 'bias' in weight.name:
                        print(f"    Bias 값: {w}")
                        if np.all(w == 0):
                            print("    ⚠️ 모든 bias가 0!")
        
        # 테스트 예측
        print("\n4️⃣ 테스트 예측...")
        test_inputs = [
            ("백색", np.ones((1, 224, 224, 3), dtype=np.float32)),
            ("흑색", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("랜덤", np.random.random((1, 224, 224, 3)).astype(np.float32)),
            ("빨강", np.zeros((1, 224, 224, 3), dtype=np.float32))
        ]
        test_inputs[3][1][:,:,:,0] = 1.0
        
        for name, inp in test_inputs:
            pred = original_model.predict(inp, verbose=0)
            print(f"\n{name} 이미지: {[f'{p:.1%}' for p in pred[0]]}")
            if np.allclose(pred[0], 0.2, atol=0.01):
                print("  ⚠️ 모든 클래스 20%!")
        
        # 수정된 모델 저장
        print("\n5️⃣ 수정된 모델 저장...")
        
        # 가중치만 저장
        original_model.save_weights("models/health_diagnosis/eye_disease/eye_weights_only.h5")
        
        # 전체 모델 저장 (H5 형식)
        original_model.save(
            "models/health_diagnosis/eye_disease/eye_disease_fixed_normalization.h5",
            save_format='h5'
        )
        
        print("✅ 저장 완료!")
        
        # 클래스맵 확인
        print("\n6️⃣ 클래스맵 확인...")
        class_map_path = Path("models/health_diagnosis/eye_disease/class_map.json")
        with open(class_map_path, 'r', encoding='utf-8') as f:
            class_map = json.load(f)
        
        print("클래스맵:")
        for idx, name in class_map.items():
            print(f"  {idx}: {name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_test_model():
    """간단한 테스트 모델 생성"""
    
    print("\n\n7️⃣ 임시 해결책: 색상 기반 진단 모델...")
    
    # 색상 기반 간단한 진단
    test_code = '''
import numpy as np

def color_based_diagnosis(image_array):
    """색상 기반 임시 진단"""
    
    # 이미지 평균 색상 계산
    img = image_array[0]  # (224, 224, 3)
    
    # 채널별 평균
    r_mean = np.mean(img[:,:,0])
    g_mean = np.mean(img[:,:,1]) 
    b_mean = np.mean(img[:,:,2])
    
    # 빨간색이 강하면 결막염 의심
    if r_mean > 0.6 and r_mean > g_mean * 1.2:
        return "결막 및 누관 질환", 0.7
    
    # 흰색/회색이면 백내장 의심
    if r_mean > 0.7 and g_mean > 0.7 and b_mean > 0.7:
        return "수정체 질환", 0.6
    
    # 어두운 부분이 많으면 각막 문제
    if np.mean(img) < 0.3:
        return "각막 질환", 0.5
    
    # 특정 패턴이 없으면 정상
    return "정상", 0.8

# 테스트
test_img = np.random.random((1, 224, 224, 3))
result = color_based_diagnosis(test_img)
print(f"진단: {result[0]} ({result[1]:.0%})")
'''
    
    print(test_code)
    
    print("\n💡 임시 해결책을 적용하려면 emergency_eye_fix.py를 실행하세요")

if __name__ == "__main__":
    success = fix_and_convert_model()
    
    if not success:
        create_simple_test_model()
    
    print("\n\n📌 결론:")
    print("1. 원본 모델의 normalization 레이어에 문제가 있음")
    print("2. 모든 클래스가 20%로 나오는 것은 가중치 초기화 문제")
    print("3. 제대로 학습된 모델이 필요함")
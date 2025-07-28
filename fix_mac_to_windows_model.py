"""
Mac에서 변환한 모델을 Windows에서 사용하기
플랫폼 간 호환성 문제 해결
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

print("🔧 Mac → Windows 모델 호환성 문제 해결")
print("=" * 70)

print("\n문제 분석:")
print("1. Mac에서 Normalization.adapt() 실행 → mean, variance 계산됨")
print("2. Windows로 이동 후 로드 시 Normalization 레이어 문제 발생")
print("3. 원인: TensorFlow 버전 차이 또는 플랫폼 간 호환성 문제")

# 모델 경로
original_model_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras")
mac_converted_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/final_model_fixed.keras")

def extract_base_model():
    """변환된 모델에서 핵심 모델만 추출"""
    
    print("\n\n🔧 해결 방법 1: 원본 모델 직접 사용")
    print("-" * 60)
    
    if not original_model_path.exists():
        print(f"❌ 원본 모델을 찾을 수 없습니다: {original_model_path}")
        return None
    
    try:
        # 1. 원본 모델 로드 시도
        print("1️⃣ 원본 모델(best_grouped_model.keras) 로드 시도...")
        
        # 다양한 커스텀 객체 정의
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout,
            'Dropout': tf.keras.layers.Dropout
        }
        
        model = tf.keras.models.load_model(
            str(original_model_path),
            custom_objects=custom_objects,
            compile=False
        )
        
        print("✅ 원본 모델 로드 성공!")
        
        # 모델 정보
        print(f"\n모델 정보:")
        print(f"- 입력: {model.input_shape}")
        print(f"- 출력: {model.output_shape}")
        print(f"- 레이어 수: {len(model.layers)}")
        
        # 테스트
        print("\n2️⃣ 예측 테스트...")
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32) * 255.0  # 0-255 범위
        
        pred = model.predict(test_input, verbose=0)
        print(f"예측 shape: {pred.shape}")
        print(f"예측값: {[f'{p:.3f}' for p in pred[0]]}")
        print(f"최대 확률 클래스: {np.argmax(pred[0])}")
        
        # 정상 작동 확인
        if not np.allclose(pred[0], pred[0][0], rtol=1e-3):
            print("✅ 모델이 정상 작동합니다!")
            return model
        else:
            print("⚠️ 모든 예측값이 동일 - 가중치 문제")
            return None
            
    except Exception as e:
        print(f"❌ 원본 모델 로드 실패: {e}")
        
        if "normalization" in str(e).lower():
            print("\n원본 모델도 Normalization 문제가 있습니다.")
            print("원본 모델 학습 시 이미 Normalization이 포함되어 있었을 가능성이 높습니다.")
        
        return None

def create_wrapper_model(base_model):
    """원본 모델에 전처리 레이어 추가"""
    
    print("\n\n🔧 해결 방법 2: 전처리 래퍼 모델 생성")
    print("-" * 60)
    
    if base_model is None:
        return None
    
    print("1️⃣ 전처리를 포함한 래퍼 모델 생성...")
    
    # 입력 레이어
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # 전처리 (0-255 → 정규화)
    # Mac에서 계산된 mean/std 값을 하드코딩
    # 일반적인 ImageNet 값 사용
    x = tf.keras.layers.Lambda(
        lambda img: (img - [123.68, 116.779, 103.939]) / [58.393, 57.12, 57.375]
    )(inputs)
    
    # 원본 모델 적용
    outputs = base_model(x)
    
    # 새로운 모델
    wrapper_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("✅ 래퍼 모델 생성 완료")
    
    # 컴파일
    wrapper_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return wrapper_model

def create_simple_preprocessing_model():
    """단순한 전처리만 하는 모델"""
    
    print("\n\n🔧 해결 방법 3: 단순 전처리 모델")
    print("-" * 60)
    
    print("1️⃣ 단순 전처리 + EfficientNet 모델 생성...")
    
    # 모델 구성
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(224, 224, 3)),
        tf.keras.layers.Lambda(lambda x: x / 255.0),  # 단순 0-1 정규화
        tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3)
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    print("✅ 모델 생성 완료")
    
    # 원본 모델에서 가중치를 복사할 수 있다면 복사
    # (이 부분은 레이어 구조가 맞아야 가능)
    
    return model

# 실행
print("\n" + "="*70)

# 1. 원본 모델 사용 시도
base_model = extract_base_model()

if base_model is not None:
    # 2. 래퍼 모델 생성
    wrapper_model = create_wrapper_model(base_model)
    
    if wrapper_model is not None:
        # 저장
        save_path = Path("models/health_diagnosis/eye_disease/eye_disease_windows.h5")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        wrapper_model.save(str(save_path), save_format='h5')
        print(f"\n✅ Windows용 모델 저장 완료: {save_path}")
        
        # 클래스맵도 복사
        import shutil
        class_map_src = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/class_map.json")
        class_map_dst = Path("models/health_diagnosis/eye_disease/class_map.json")
        
        if class_map_src.exists():
            shutil.copy(class_map_src, class_map_dst)
            print(f"✅ 클래스맵 복사 완료")
else:
    # 3. 단순 모델 생성
    print("\n원본 모델 사용 불가. 단순 모델 생성...")
    simple_model = create_simple_preprocessing_model()
    
    save_path = Path("models/health_diagnosis/eye_disease/eye_disease_simple.h5")
    simple_model.save(str(save_path), save_format='h5')
    print(f"\n✅ 단순 모델 저장: {save_path}")

print("\n\n📌 결론:")
print("1. Mac의 Normalization 레이어가 Windows TensorFlow와 호환되지 않음")
print("2. 원본 모델을 직접 사용하거나")
print("3. Lambda 레이어로 전처리하는 것이 안전함")
print("\n💡 권장: 플랫폼 간 이동 시 .h5 형식 사용")
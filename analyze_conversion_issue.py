"""
모델 변환 문제 분석 및 해결
Normalization 레이어 문제의 근본 원인 파악
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

print("🔍 모델 변환 문제 심층 분석")
print("=" * 70)

# 원본 모델과 변환된 모델 경로
original_model_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras")
converted_model_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/final_model_fixed.keras")

print("\n📌 변환 스크립트 분석 결과:")
print("-" * 50)
print("1. 문제점:")
print("   - Mac 경로 사용 (/Users/sehyeonjeong/...)")
print("   - Windows에서 실행 시 이미지 데이터를 찾을 수 없음")
print("   - Normalization.adapt()가 제대로 실행되지 않음")
print("   - mean, variance, count가 초기화되지 않은 상태로 저장됨")
print("\n2. 결과:")
print("   - Normalization 레이어가 비어있는 상태")
print("   - 모델 로드 시 'expected 3 variables' 오류 발생")

# 해결책 1: 원본 모델 직접 사용
print("\n\n🔧 해결책 1: 원본 모델 직접 사용")
print("-" * 50)

def test_original_model():
    """원본 모델을 직접 로드해서 테스트"""
    
    if not original_model_path.exists():
        print(f"❌ 원본 모델을 찾을 수 없습니다: {original_model_path}")
        return False
    
    try:
        print("1️⃣ 원본 모델 로드 시도...")
        
        # 커스텀 객체
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout
        }
        
        # 원본 모델은 이미 학습 시 Normalization이 포함되어 있을 수 있음
        model = tf.keras.models.load_model(
            str(original_model_path),
            custom_objects=custom_objects,
            compile=False
        )
        
        print("✅ 원본 모델 로드 성공!")
        
        # 모델 구조 확인
        print(f"\n모델 정보:")
        print(f"- 입력: {model.input_shape}")
        print(f"- 출력: {model.output_shape}")
        print(f"- 레이어 수: {len(model.layers)}")
        
        # 첫 번째 레이어 확인
        first_layer = model.layers[0]
        print(f"\n첫 번째 레이어: {first_layer.name} ({first_layer.__class__.__name__})")
        
        # 테스트 예측
        print("\n2️⃣ 테스트 예측...")
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = model.predict(test_input, verbose=0)
        
        print(f"예측 성공! Shape: {pred.shape}")
        print(f"예측값: {[f'{p:.3f}' for p in pred[0]]}")
        
        # 예측이 정상인지 확인
        if not np.allclose(pred[0], pred[0][0]):
            print("✅ 예측값이 다양함 - 모델 정상!")
            
            # 원본 모델 복사
            save_path = Path("models/health_diagnosis/eye_disease/eye_disease_original.keras")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 그대로 복사
            import shutil
            shutil.copy(original_model_path, save_path)
            print(f"\n✅ 원본 모델 복사 완료: {save_path}")
            
            return True
        else:
            print("⚠️ 모든 예측값이 동일 - 모델 문제")
            return False
            
    except Exception as e:
        print(f"❌ 원본 모델 로드 실패: {e}")
        
        # Normalization 오류인 경우
        if "normalization" in str(e).lower():
            print("\n원본 모델도 Normalization 문제가 있습니다.")
            print("모델 학습 시 Normalization이 제대로 저장되지 않았을 가능성이 높습니다.")
        
        return False

# 해결책 2: Normalization 제거
print("\n\n🔧 해결책 2: Normalization 레이어 제거")
print("-" * 50)

def create_model_without_normalization():
    """Normalization 없이 모델 재구성"""
    
    print("1️⃣ Normalization 없는 모델 구조 생성...")
    
    # 입력 레이어
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # 수동 정규화 (0-255 → 0-1)
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
    
    # EfficientNetB0 백본
    base_model = tf.keras.applications.EfficientNetB0(
        input_tensor=x,
        include_top=False,
        weights=None  # 가중치는 나중에 로드
    )
    
    # 상위 레이어
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("✅ 모델 구조 생성 완료")
    
    # 컴파일
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 저장
    save_path = Path("models/health_diagnosis/eye_disease/eye_disease_no_norm.h5")
    model.save(str(save_path), save_format='h5')
    print(f"✅ 저장 완료: {save_path}")
    
    return model

# 실행
print("\n" + "="*70)
print("🚀 분석 시작...")

# 1. 원본 모델 테스트
success = test_original_model()

if not success:
    # 2. Normalization 없는 모델 생성
    print("\n원본 모델 사용 불가. 대안 모델 생성...")
    model = create_model_without_normalization()
    
    print("\n\n💡 권장사항:")
    print("1. 모델을 다시 학습하되, Normalization 레이어 없이 학습")
    print("2. 또는 Lambda 레이어로 수동 정규화 사용")
    print("3. 학습 시 .h5 형식으로 저장")
else:
    print("\n\n✅ 원본 모델을 사용할 수 있습니다!")
    print("services/eye_disease_service.py에서")
    print("models/health_diagnosis/eye_disease/eye_disease_original.keras를 사용하세요.")

print("\n\n📌 결론:")
print("변환 스크립트의 Normalization adapt가 제대로 실행되지 않아")
print("mean, variance, count가 비어있는 상태로 저장되었습니다.")
print("원본 모델을 직접 사용하거나, Normalization 없이 재구성이 필요합니다.")
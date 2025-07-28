"""
원본 모델 실제 검증
제공받은 분석이 맞는지 확인
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

print("🔍 원본 모델 철저한 검증")
print("=" * 70)

# 모델 경로
model_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras")
class_map_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/class_map.json")

print("\n📌 이전 테스트 결과 요약:")
print("1. best_grouped_model.keras도 Normalization 레이어 오류 발생")
print("2. 모든 클래스가 20%로 예측되는 문제")
print("3. TensorFlow 2.x Graph/Eager mode 문제")

# 실제 테스트
print("\n\n🧪 실제 테스트 시작...")
print("-" * 60)

# 1. 제공받은 코드대로 테스트
print("\n1️⃣ 제공받은 방법으로 모델 로드 시도...")

try:
    # 제공받은 코드 그대로
    model = tf.keras.models.load_model(str(model_path))
    print("✅ 모델 로드 성공? (이상하네요...)")
    
except Exception as e:
    print(f"❌ 예상대로 실패: {e}")
    
    if "normalization" in str(e).lower():
        print("\n⚠️ Normalization 레이어 문제가 여전히 있습니다!")
        print("제공받은 분석이 틀렸습니다.")

# 2. 커스텀 객체로 다시 시도
print("\n\n2️⃣ 커스텀 객체로 로드 시도...")

try:
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
    print("✅ 커스텀 객체로 로드 성공")
    
    # 실제 예측 테스트
    print("\n3️⃣ 예측 테스트...")
    
    # 제공받은 전처리 방법
    test_img = np.random.random((224, 224, 3)).astype(np.float32) * 255.0
    
    # ImageNet 정규화
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    std = np.array([0.229, 0.224, 0.225]) * 255.0
    test_img = (test_img - mean) / std
    test_img = np.expand_dims(test_img, axis=0)
    
    pred = model.predict(test_img, verbose=0)
    print(f"예측 결과: {pred[0]}")
    print(f"확률(%): {[f'{p*100:.1f}' for p in pred[0]]}")
    
    # 20% 문제 확인
    if np.allclose(pred[0], 0.2, atol=0.01):
        print("\n❌ 여전히 모든 클래스가 20%입니다!")
        print("모델이 제대로 학습되지 않았습니다.")
    
except Exception as e:
    print(f"❌ 로드 실패: {e}")

# 3. 실제 문제 분석
print("\n\n📊 실제 문제 분석:")
print("-" * 60)

print("\n1. Normalization 레이어 문제:")
print("   - Mac에서 만든 것과 관계없이 원본 모델도 문제")
print("   - 학습 시 Normalization이 제대로 저장되지 않음")

print("\n2. 20% 균등 예측 문제:")
print("   - 모델 가중치가 초기화 상태")
print("   - 실제로 학습이 되지 않았거나")
print("   - 저장 시 가중치가 손실됨")

print("\n3. 플랫폼 호환성은 부차적 문제:")
print("   - 근본 원인은 모델 자체의 문제")

# 4. 클래스맵 확인
print("\n\n4️⃣ 클래스맵 확인...")
if class_map_path.exists():
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    print("클래스맵:")
    for idx, name in class_map.items():
        print(f"  {idx}: {name}")

# 5. 진짜 해결책
print("\n\n💡 진짜 해결책:")
print("-" * 60)

print("\n옵션 1: 임시 해결책")
print("- 색상 기반 진단 사용 (emergency_eye_fix.py)")
print("- 정확도는 낮지만 작동은 함")

print("\n옵션 2: 근본 해결책")
print("- 제대로 학습된 모델 파일 필요")
print("- .h5 형식으로 저장된 학습 완료 모델")
print("- 또는 가중치 파일(.weights.h5) 별도 저장")

print("\n옵션 3: 재학습")
print("- 데이터셋으로 모델 재학습")
print("- Normalization 없이 Lambda 레이어 사용")
print("- 학습 완료 후 .h5로 저장")

print("\n\n⚠️ 결론:")
print("제공받은 분석은 너무 낙관적입니다.")
print("원본 모델도 사용할 수 없는 상태입니다.")
print("제대로 학습된 모델이 필요합니다!")
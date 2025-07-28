"""
안구질환 모델 심각한 문제 진단
결막염 데이터로도 정상 판정하는 문제 분석
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
from pathlib import Path

def critical_diagnosis():
    """모델의 근본적인 문제 진단"""
    
    print("🚨 안구질환 모델 긴급 진단")
    print("="*60)
    print("문제: 결막염 학습 데이터를 입력해도 '정상'으로 분류")
    print("="*60)
    
    # 1. 모델 구조 확인
    model_path = "models/health_diagnosis/eye_disease/eye_disease_fixed.h5"
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'swish': tf.nn.swish}
    )
    
    print("\n1️⃣ 모델 아키텍처 분석:")
    model.summary(show_trainable=True, print_fn=lambda x: print(x) if 'dense' in x.lower() or 'conv' in x.lower()[:20] else None)
    
    # 2. 출력층 상세 분석
    print("\n2️⃣ 출력층 분석:")
    output_layer = model.layers[-1]
    print(f"- 레이어 타입: {type(output_layer).__name__}")
    print(f"- 유닛 수: {output_layer.units if hasattr(output_layer, 'units') else 'N/A'}")
    print(f"- 활성화: {output_layer.activation if hasattr(output_layer, 'activation') else 'N/A'}")
    
    if output_layer.weights:
        weights = output_layer.weights[0].numpy()
        bias = output_layer.weights[1].numpy() if len(output_layer.weights) > 1 else None
        
        print(f"\n가중치 통계:")
        print(f"- 형태: {weights.shape}")
        print(f"- 평균: {np.mean(weights):.6f}")
        print(f"- 표준편차: {np.std(weights):.6f}")
        
        if bias is not None:
            print(f"\n편향(Bias) 값:")
            for i, b in enumerate(bias):
                print(f"  클래스 {i}: {b:.6f}")
    
    # 3. 극단적인 입력 테스트
    print("\n3️⃣ 극단적 입력 테스트:")
    
    extreme_tests = {
        "순수 빨강 (RGB: 1,0,0)": np.ones((1, 224, 224, 3)) * [1, 0, 0],
        "순수 흰색 (RGB: 1,1,1)": np.ones((1, 224, 224, 3)),
        "순수 검정 (RGB: 0,0,0)": np.zeros((1, 224, 224, 3)),
        "체크보드 패턴": create_checkerboard(),
        "그라디언트": create_gradient()
    }
    
    with open('models/health_diagnosis/eye_disease/class_map.json', 'r') as f:
        class_map = json.load(f)
    
    for test_name, test_input in extreme_tests.items():
        pred = model.predict(test_input, verbose=0)
        print(f"\n{test_name}:")
        
        # 상위 3개 출력
        top_indices = np.argsort(pred[0])[-3:][::-1]
        for idx in top_indices:
            print(f"  {class_map.get(str(idx), f'Unknown_{idx}')}: {pred[0][idx]:.4f}")
        
        # 엔트로피 계산 (불확실성 측정)
        entropy = -np.sum(pred[0] * np.log(pred[0] + 1e-10))
        print(f"  엔트로피: {entropy:.4f} (높을수록 불확실)")
    
    # 4. 학습 설정 추정
    print("\n4️⃣ 가능한 원인 분석:")
    
    # 모든 테스트에서 비슷한 확률이 나오는지 확인
    all_predictions = []
    for _ in range(10):
        random_input = np.random.random((1, 224, 224, 3))
        pred = model.predict(random_input, verbose=0)
        all_predictions.append(pred[0])
    
    all_predictions = np.array(all_predictions)
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)
    
    print("\n무작위 입력 10개의 평균 예측:")
    for i, (mean, std) in enumerate(zip(mean_pred, std_pred)):
        print(f"  클래스 {i} ({class_map.get(str(i), 'Unknown')}): {mean:.4f} ± {std:.4f}")
    
    # 진단 결과
    print("\n\n🔍 진단 결과:")
    
    uniform_distribution = np.all(std_pred < 0.05)
    low_confidence = np.max(mean_pred) < 0.4
    
    if uniform_distribution:
        print("❌ 심각: 모델이 모든 입력에 대해 거의 동일한 확률 출력")
        print("   → 모델이 전혀 학습되지 않았거나 손상됨")
    
    if low_confidence:
        print("❌ 심각: 최대 신뢰도가 40% 미만")
        print("   → 모델이 어떤 패턴도 학습하지 못함")
    
    print("\n📋 권장 조치:")
    print("1. 즉시: 원본 학습 코드 및 데이터셋 확인")
    print("2. 단기: 다른 사전학습 모델 사용 (EfficientNet 등)")
    print("3. 중기: 새로운 데이터셋으로 재학습")
    print("4. 장기: 전문가 검증된 데이터셋 구축")

def create_checkerboard():
    """체크보드 패턴 생성"""
    img = np.zeros((224, 224, 3))
    for i in range(0, 224, 20):
        for j in range(0, 224, 20):
            if (i//20 + j//20) % 2 == 0:
                img[i:i+20, j:j+20] = 1
    return np.expand_dims(img, axis=0).astype(np.float32)

def create_gradient():
    """그라디언트 패턴 생성"""
    img = np.zeros((224, 224, 3))
    for i in range(224):
        img[i, :] = i / 224
    return np.expand_dims(img, axis=0).astype(np.float32)

if __name__ == "__main__":
    critical_diagnosis()
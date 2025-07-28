"""
극단적 테스트로 피부질환 모델 검증
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# 모델 로드
model_path = "models/health_diagnosis/skin_disease/classification/dog_binary/dog_binary_model.h5"
model = tf.keras.models.load_model(model_path, compile=False)

print("=== 모델 정보 ===")
print(f"입력 형태: {model.input_shape}")
print(f"출력 형태: {model.output_shape}")
print(f"총 파라미터: {model.count_params()}")

# 모델 구조 출력
print("\n=== 모델 구조 ===")
for i, layer in enumerate(model.layers[-5:]):  # 마지막 5개 층만
    print(f"Layer {i}: {layer.name} - {layer.__class__.__name__}")
    if hasattr(layer, 'units'):
        print(f"  Units: {layer.units}")
    if hasattr(layer, 'activation'):
        print(f"  Activation: {layer.activation}")

# 극단적 테스트 케이스
test_cases = [
    ("완전 검은색", np.zeros((1, 224, 224, 3), dtype=np.float32)),
    ("완전 흰색", np.ones((1, 224, 224, 3), dtype=np.float32)),
    ("빨간색", np.zeros((1, 224, 224, 3), dtype=np.float32)),
    ("초록색", np.zeros((1, 224, 224, 3), dtype=np.float32)),
    ("파란색", np.zeros((1, 224, 224, 3), dtype=np.float32)),
    ("랜덤 노이즈", np.random.random((1, 224, 224, 3)).astype(np.float32)),
    ("체커보드", np.zeros((1, 224, 224, 3), dtype=np.float32)),
    ("그라디언트", np.zeros((1, 224, 224, 3), dtype=np.float32))
]

# 색상 설정
test_cases[2][1][0, :, :, 0] = 1.0  # 빨간색
test_cases[3][1][0, :, :, 1] = 1.0  # 초록색
test_cases[4][1][0, :, :, 2] = 1.0  # 파란색

# 체커보드 패턴
for i in range(0, 224, 16):
    for j in range(0, 224, 16):
        if (i//16 + j//16) % 2 == 0:
            test_cases[6][1][0, i:i+16, j:j+16, :] = 1.0

# 그라디언트
for i in range(224):
    test_cases[7][1][0, i, :, :] = i / 224.0

print("\n=== 극단적 입력 테스트 ===")
predictions = []

for name, img in test_cases:
    pred = model.predict(img, verbose=0)
    predictions.append(pred[0][0])
    print(f"{name:15s}: {pred[0][0]:.6f}")

# 예측값 분포 분석
print(f"\n=== 예측값 분석 ===")
print(f"최소값: {min(predictions):.6f}")
print(f"최대값: {max(predictions):.6f}")
print(f"평균값: {np.mean(predictions):.6f}")
print(f"표준편차: {np.std(predictions):.6f}")
print(f"변동 범위: {max(predictions) - min(predictions):.6f}")

# 실제 이미지 테스트
print("\n=== 실제 이미지 패턴 테스트 ===")
# 피부질환처럼 보이는 패턴 생성
disease_like = np.ones((1, 224, 224, 3), dtype=np.float32) * 0.8
# 중앙에 빨간 반점들 추가
for _ in range(20):
    x, y = np.random.randint(50, 174, 2)
    radius = np.random.randint(5, 15)
    for i in range(max(0, x-radius), min(224, x+radius)):
        for j in range(max(0, y-radius), min(224, y+radius)):
            if (i-x)**2 + (j-y)**2 < radius**2:
                disease_like[0, i, j, 0] = 1.0  # 빨간색
                disease_like[0, i, j, 1] = 0.3
                disease_like[0, i, j, 2] = 0.3

pred_disease = model.predict(disease_like, verbose=0)
print(f"질환 패턴 시뮬레이션: {pred_disease[0][0]:.6f}")

# 정상 피부처럼 보이는 패턴
normal_like = np.ones((1, 224, 224, 3), dtype=np.float32) * 0.9
# 약간의 노이즈 추가
normal_like += np.random.normal(0, 0.05, (1, 224, 224, 3))
normal_like = np.clip(normal_like, 0, 1)

pred_normal = model.predict(normal_like, verbose=0)
print(f"정상 패턴 시뮬레이션: {pred_normal[0][0]:.6f}")

# 가중치 분석
print("\n=== 마지막 층 가중치 분석 ===")
last_layer = model.layers[-1]
if hasattr(last_layer, 'weights') and len(last_layer.weights) > 0:
    weights = last_layer.weights[0].numpy()
    bias = last_layer.weights[1].numpy() if len(last_layer.weights) > 1 else None
    
    print(f"가중치 형태: {weights.shape}")
    print(f"가중치 평균: {np.mean(weights):.6f}")
    print(f"가중치 표준편차: {np.std(weights):.6f}")
    print(f"가중치 최소/최대: {np.min(weights):.6f} / {np.max(weights):.6f}")
    if bias is not None:
        print(f"편향값: {bias}")
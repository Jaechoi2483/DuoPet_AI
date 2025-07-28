"""
체크포인트에서 모델을 복원하여 H5로 저장
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

# 원본 학습 코드와 동일한 구조로 모델 생성
def create_model():
    network = InceptionResNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    model = Sequential()
    model.add(network)
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))  # Binary: 2 outputs
    
    return model

# 체크포인트 경로
checkpoint_path = "/mnt/e/DATA/152.반려동물 피부질환 데이터/AI 모델 소스코드/classification/workspace/data/dog/result_cla/binary_A2/model/model-004-0.437360-0.806570-0.806528-0.806891"

print("모델 생성 중...")
model = create_model()
print(f"모델 구조 - 입력: {model.input_shape}, 출력: {model.output_shape}")

# 체크포인트에서 가중치 로드
print(f"\n체크포인트 로드 중: {checkpoint_path}")
try:
    model.load_weights(checkpoint_path)
    print("체크포인트 로드 성공!")
except Exception as e:
    print(f"체크포인트 로드 실패: {e}")
    
# 모델 저장
save_path = "models/health_diagnosis/skin_disease/classification/dog_binary/dog_binary_from_checkpoint.h5"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

print(f"\n모델 저장 중: {save_path}")
model.save(save_path)
print("저장 완료!")

# 테스트
print("\n=== 간단한 테스트 ===")
import numpy as np

test_inputs = [
    ("검은색", np.zeros((1, 224, 224, 3))),
    ("흰색", np.ones((1, 224, 224, 3))),
    ("랜덤", np.random.random((1, 224, 224, 3)))
]

for name, img in test_inputs:
    pred = model.predict(img, verbose=0)
    print(f"{name}: {pred[0]} -> 정상: {pred[0][0]:.4f}, 질환: {pred[0][1]:.4f}")
    
print("\n완료!")
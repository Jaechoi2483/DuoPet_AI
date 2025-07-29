"""
체크포인트를 직접 사용하는 예제
"""
import tensorflow as tf
import numpy as np
from pathlib import Path

def load_model_from_checkpoint(checkpoint_path, model_structure):
    """
    체크포인트에서 모델을 직접 로드
    
    Args:
        checkpoint_path: 체크포인트 경로 (예: "model-007-0.511353")
        model_structure: 모델 구조를 반환하는 함수
    """
    # 1. 모델 구조 생성
    model = model_structure()
    
    # 2. 체크포인트 로드
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()
    
    return model

# 사용 예시
def create_skin_disease_model():
    """피부질환 모델 구조 생성"""
    from tensorflow.keras.applications import InceptionResNetV2
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    
    # CustomScaleLayer 등록
    with tf.keras.utils.custom_object_scope({'CustomScaleLayer': tf.keras.layers.Layer}):
        # InceptionResNetV2 기반 모델
        base_model = InceptionResNetV2(
            include_top=False,
            weights=None,  # 체크포인트에서 로드할 것이므로
            input_shape=(224, 224, 3),
            pooling='avg'
        )
    
    model = Sequential([
        base_model,
        Dense(2048, activation='relu'),
        Dense(2, activation='softmax')  # 2개 클래스: [정상, 질환]
    ])
    
    return model

# 체크포인트 직접 사용
if __name__ == "__main__":
    # 체크포인트 경로
    checkpoint_dir = Path("E:/DATA/152.반려동물 피부질환 데이터/AI 모델 소스코드/classification/workspace/dog_binary")
    checkpoint_path = checkpoint_dir / "model-004-0.437360-0.806570-0.806528-0.806891"
    
    # 모델 로드
    model = load_model_from_checkpoint(str(checkpoint_path), create_skin_disease_model)
    
    # 예측
    test_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
    prediction = model(test_image, training=False)
    print(f"Prediction: {prediction.numpy()}")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import numpy as np
from pathlib import Path

print(f"Test - TF {tf.__version__} - Eager: {tf.executing_eagerly()}")

# 모델 로드
model_path = Path("models/health_diagnosis/eye_disease/eye_disease_tf2_complete.h5")

if model_path.exists():
    print(f"\n모델 로드: {model_path}")
    
    custom_objects = {'swish': tf.nn.swish}
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
    
    # 테스트 입력
    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    
    # 1. 직접 호출
    print("\n1. 직접 호출 테스트...")
    try:
        output1 = model(test_input, training=False)
        print(f"  ✓ 성공: {output1.shape}")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    # 2. predict 메서드
    print("\n2. predict 메서드 테스트...")
    try:
        output2 = model.predict(test_input, verbose=0)
        print(f"  ✓ 성공: {output2.shape}")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    # 3. tf.function 래핑
    print("\n3. tf.function 테스트...")
    @tf.function
    def predict_fn(x):
        return model(x, training=False)
    
    try:
        output3 = predict_fn(test_input)
        print(f"  ✓ 성공: {output3.shape}")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
else:
    print(f"❌ 모델 파일 없음: {model_path}")

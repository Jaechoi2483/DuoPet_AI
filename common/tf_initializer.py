"""
TensorFlow 2.x 전역 초기화 모듈
모든 서비스보다 먼저 import되어야 함
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow import 및 설정
import tensorflow as tf

# Eager execution 강제 활성화
tf.config.run_functions_eagerly(True)

# GPU 메모리 증가 허용 (있는 경우)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# 설정 확인
print(f"[TF Initializer] TensorFlow {tf.__version__}")
print(f"[TF Initializer] Eager execution: {tf.executing_eagerly()}")
print(f"[TF Initializer] GPU devices: {len(gpus)}")

# 전역 변수로 설정 상태 저장
TF_INITIALIZED = True
TF_VERSION = tf.__version__
EAGER_MODE = tf.executing_eagerly()

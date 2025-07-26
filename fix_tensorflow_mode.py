"""
TensorFlow Eager/Graph 모드 충돌 해결
"""

import os
import tensorflow as tf

# TensorFlow 2.x에서 기본적으로 eager execution이 활성화되어 있음
# 모델 예측 시 graph 모드와 충돌 해결

print("현재 TensorFlow 버전:", tf.__version__)
print("Eager execution 상태:", tf.executing_eagerly())

# 환경 변수 설정
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 전역 설정
tf.config.run_functions_eagerly(True)

print("\n설정 완료!")
print("다음 내용을 api/main.py 상단에 추가하세요:")
print("""
import tensorflow as tf
tf.config.run_functions_eagerly(True)
""")
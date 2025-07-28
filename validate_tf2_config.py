"""
TensorFlow 2.x 설정 검증
"""
import tensorflow as tf
import os

print("🔍 TensorFlow 설정 확인")
print("=" * 50)

print(f"TensorFlow 버전: {tf.__version__}")
print(f"Eager execution 활성화: {tf.executing_eagerly()}")
print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"TF 로그 레벨: {os.environ.get('TF_CPP_MIN_LOG_LEVEL', 'Not set')}")

# 간단한 연산 테스트
try:
    x = tf.constant([1, 2, 3])
    y = tf.constant([4, 5, 6])
    z = tf.add(x, y)
    print(f"\n테스트 연산: {x.numpy()} + {y.numpy()} = {z.numpy()}")
    print("✅ Eager execution 정상 작동!")
except Exception as e:
    print(f"❌ Eager execution 오류: {e}")

print("\n💡 팁: 모든 서비스 파일이 올바르게 설정되었는지 확인하세요.")

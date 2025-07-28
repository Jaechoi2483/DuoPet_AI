
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.run_functions_eagerly(True)

print(f"Test - TensorFlow {tf.__version__} - Eager: {tf.executing_eagerly()}")

import numpy as np
from PIL import Image
from pathlib import Path

# 서비스 import 전에 eager 확인
print(f"Before import - Eager: {tf.executing_eagerly()}")

from services.eye_disease_service import EyeDiseaseService

# 서비스 import 후 eager 확인
print(f"After import - Eager: {tf.executing_eagerly()}")

# 테스트
model_path = Path("models/health_diagnosis/eye_disease/best_grouped_model.keras")
class_map_path = Path("models/health_diagnosis/eye_disease/class_map.json")

if model_path.exists() and class_map_path.exists():
    try:
        service = EyeDiseaseService(str(model_path), str(class_map_path))
        print("✅ 서비스 초기화 성공!")
        
        # 더미 이미지로 테스트
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = service.diagnose(dummy_image)
        print(f"진단 결과: {result}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ 모델 파일 없음")

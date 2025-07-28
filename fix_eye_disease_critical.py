"""
안구질환 서비스 Eager Execution 긴급 수정
- Import 순서 및 초기화 시점 개선
"""
import os
import shutil
from pathlib import Path

def create_fixed_eye_disease_service():
    """완전히 수정된 안구질환 서비스 생성"""
    
    fixed_content = '''"""
안구질환 진단 서비스
TensorFlow 2.x 호환 버전
"""
# 가장 먼저 TensorFlow 설정
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# 즉시 eager execution 활성화
tf.config.run_functions_eagerly(True)
print(f"[EyeDiseaseService] TensorFlow {tf.__version__} - Eager execution: {tf.executing_eagerly()}")

import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging
import sys

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'numpy'):  # Tensor 객체
        return obj.numpy().tolist() if hasattr(obj.numpy(), 'tolist') else float(obj.numpy())
    return obj

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """
        안구 질환 진단 서비스 초기화

        Args:
            model_path (str): Keras 모델 파일 경로
            class_map_path (str): 클래스 맵 JSON 파일 경로
        """
        try:
            # Custom objects 정의
            custom_objects = {
                'swish': tf.nn.swish,
                'Swish': tf.keras.layers.Activation(tf.nn.swish)
            }
            
            # 모델 파일 찾기
            model_loaded = False
            
            # 1. eye_disease_fixed.h5 시도
            fixed_model_path = model_path.replace('.keras', '_fixed.h5')
            if os.path.exists(fixed_model_path):
                try:
                    logger.info(f"Loading fixed model from {fixed_model_path}")
                    self.model = tf.keras.models.load_model(
                        fixed_model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    model_loaded = True
                    logger.info(f"Successfully loaded fixed eye disease model")
                except Exception as e:
                    logger.warning(f"Failed to load fixed model: {e}")
            
            # 2. eye_disease_tf2.h5 시도
            if not model_loaded:
                tf2_model_path = model_path.replace('.keras', '_tf2.h5')
                if os.path.exists(tf2_model_path):
                    try:
                        logger.info(f"Loading TF2 model from {tf2_model_path}")
                        self.model = tf.keras.models.load_model(
                            tf2_model_path,
                            custom_objects=custom_objects,
                            compile=False
                        )
                        model_loaded = True
                        logger.info(f"Successfully loaded TF2 eye disease model")
                    except Exception as e:
                        logger.warning(f"Failed to load TF2 model: {e}")
            
            # 3. 원본 .keras 파일 시도
            if not model_loaded and os.path.exists(model_path):
                try:
                    logger.info(f"Loading original model from {model_path}")
                    # .keras 파일은 직접 로드 시도
                    self.model = tf.keras.models.load_model(
                        model_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    model_loaded = True
                    logger.info(f"Successfully loaded original eye disease model")
                except Exception as e:
                    logger.warning(f"Failed to load original model: {e}")
            
            if not model_loaded:
                raise ValueError(f"Could not load any model from {model_path}")
            
            # 모델 컴파일
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        except Exception as e:
            logger.error(f"Failed to load eye disease model: {e}")
            raise
            
        # 클래스 맵 로드
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
            
        # 모델 입력 shape 확인
        try:
            if hasattr(self.model, 'input_shape') and self.model.input_shape:
                self.input_shape = self.model.input_shape[1:3]
            else:
                self.input_shape = (224, 224)
                logger.warning("Could not determine model input shape, using default (224, 224)")
        except:
            self.input_shape = (224, 224)
        
        logger.info(f"EyeDiseaseService initialized with input shape: {self.input_shape}")

    def preprocess_image(self, image_file) -> np.ndarray:
        """
        이미지를 모델 입력에 맞게 전처리합니다.

        Args:
            image_file: 업로드된 이미지 파일

        Returns:
            np.ndarray: 전처리된 이미지 배열
        """
        # 파일 포인터 리셋
        if hasattr(image_file, 'file'):
            image_file.file.seek(0)
            img = Image.open(image_file.file).convert('RGB')
        elif hasattr(image_file, 'seek'):
            image_file.seek(0)
            img = Image.open(image_file).convert('RGB')
        else:
            img = Image.open(image_file).convert('RGB')
            
        img = img.resize(self.input_shape)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 정규화
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array

    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """
        전처리된 이미지로 질병을 예측합니다.

        Args:
            image_array (np.ndarray): 전처리된 이미지 배열

        Returns:
            Tuple[str, float]: (예측된 질병 이름, 신뢰도 점수)
        """
        # NumPy 배열로 확실히 변환
        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)
            
        # 배치 차원 확인
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
            
        # 이미 정규화되어 있는지 확인
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
            
        # 예측 수행
        predictions = self.model.predict(image_array, verbose=0)
        
        # numpy 배열로 변환 (Tensor일 경우)
        if hasattr(predictions, 'numpy'):
            predictions_np = predictions.numpy()
        else:
            predictions_np = predictions
            
        predicted_class_index = int(np.argmax(predictions_np[0]))
        confidence = float(np.max(predictions_np[0]))
        
        # 클래스 인덱스를 질병 이름으로 변환
        predicted_class_name = self.class_map.get(str(predicted_class_index), "Unknown")
        
        return predicted_class_name, confidence

    def diagnose(self, image_file) -> Dict[str, any]:
        """
        이미지 파일을 받아 안구 질환을 진단합니다.

        Args:
            image_file: 업로드된 이미지 파일

        Returns:
            Dict[str, any]: 진단 결과 (질병 이름, 신뢰도)
        """
        preprocessed_image = self.preprocess_image(image_file)
        disease, confidence = self.predict(preprocessed_image)
        
        result = {
            "disease": disease,
            "confidence": confidence
        }
        
        # numpy 타입 변환
        return convert_numpy_types(result)
'''
    
    # 백업 생성
    service_path = Path("services/eye_disease_service.py")
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_critical')
        shutil.copy(service_path, backup_path)
        print(f"✓ 백업 생성: {backup_path}")
    
    # 파일 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ 안구질환 서비스 긴급 수정 완료!")
    print("  - TensorFlow import 및 설정을 최상단으로 이동")
    print("  - 모델 로드 시 compile=False 옵션 추가")
    print("  - custom_objects에 swish 함수 추가")
    print("  - 더 강력한 numpy 타입 변환")

def test_fixed_service():
    """수정된 서비스 테스트"""
    test_code = '''
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
'''
    
    with open("test_eye_critical.py", 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("\n💡 테스트 스크립트 생성: test_eye_critical.py")

if __name__ == "__main__":
    print("🚨 안구질환 서비스 긴급 수정")
    print("=" * 60)
    
    create_fixed_eye_disease_service()
    test_fixed_service()
    
    print("\n📋 다음 단계:")
    print("  1. 서버 중지 (Ctrl+C)")
    print("  2. 테스트: python test_eye_critical.py")
    print("  3. 서버 재시작: python api/main.py")
    print("  4. 프론트엔드에서 다시 테스트")
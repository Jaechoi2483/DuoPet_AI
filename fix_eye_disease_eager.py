"""
안구질환 서비스 Eager Execution 문제 해결
"""
import os
import shutil
from pathlib import Path

def fix_eye_disease_service():
    """안구질환 서비스 파일을 TF2 호환으로 수정"""
    
    service_path = Path("services/eye_disease_service.py")
    
    if not service_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {service_path}")
        return False
    
    print(f"🔧 {service_path} 수정 중...")
    
    # 백업 생성
    backup_path = service_path.with_suffix('.py.backup_eager')
    shutil.copy(service_path, backup_path)
    print(f"  ✓ 백업 생성: {backup_path}")
    
    # 수정된 내용 작성
    fixed_content = '''
import os
# TensorFlow 설정을 가장 먼저
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.run_functions_eagerly(True)

import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging
import sys

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import load_model_with_custom_objects, safe_model_predict

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
            # TF2 모델 우선 시도
            tf2_model_path = model_path.replace('.keras', '_fixed.h5')
            if os.path.exists(tf2_model_path):
                custom_objects = {'swish': tf.nn.swish}
                self.model = tf.keras.models.load_model(tf2_model_path, custom_objects=custom_objects)
                logger.info(f"Successfully loaded TF2 eye disease model from {tf2_model_path}")
            else:
                # 원본 모델 로드 시도
                tf2_model_path = model_path.replace('.keras', '_tf2.h5')
                if os.path.exists(tf2_model_path):
                    self.model = tf.keras.models.load_model(tf2_model_path)
                    logger.info(f"Successfully loaded TF2 eye disease model from {tf2_model_path}")
                else:
                    self.model = load_model_with_custom_objects(model_path)
                    logger.info(f"Successfully loaded eye disease model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load eye disease model: {e}")
            raise
            
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
            
        # 모델 입력 shape 확인
        try:
            if hasattr(self.model, 'input_shape') and self.model.input_shape:
                self.input_shape = self.model.input_shape[1:3]
            else:
                self.input_shape = (224, 224)  # 기본값
                logger.warning("Could not determine model input shape, using default (224, 224)")
        except:
            self.input_shape = (224, 224)

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
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch
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
            
        # 정규화 (0-255 범위를 0-1로)
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
            
        # 일반 predict 사용 (TF2 모델은 직접 사용 가능)
        predictions_np = self.model.predict(image_array, verbose=0)
            
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
    
    # 파일 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ 안구질환 서비스 수정 완료!")
    print("  - TensorFlow eager execution 설정 적용")
    print("  - numpy 타입 변환 함수 추가")
    print("  - import 순서 최적화")
    
    return True

def test_eye_disease_service():
    """수정된 서비스 테스트"""
    print("\n🧪 안구질환 서비스 테스트...")
    
    test_code = '''
import numpy as np
from PIL import Image
from services.eye_disease_service import EyeDiseaseService
from pathlib import Path

# 테스트 설정
model_path = Path("models/health_diagnosis/eye_disease/eye_disease_fixed.h5")
class_map_path = Path("models/health_diagnosis/eye_disease/class_map.json")

if model_path.exists() and class_map_path.exists():
    # 서비스 초기화
    service = EyeDiseaseService(str(model_path), str(class_map_path))
    
    # 더미 이미지 생성
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(dummy_image)
    
    # 진단 테스트
    result = service.diagnose(dummy_pil)
    print(f"✅ 테스트 성공!")
    print(f"  - 진단 결과: {result}")
    print(f"  - 타입 확인: disease={type(result['disease'])}, confidence={type(result['confidence'])}")
else:
    print("❌ 모델 파일을 찾을 수 없습니다")
'''
    
    # 테스트 파일 생성
    test_path = Path("test_eye_service_fixed.py")
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print(f"  ✓ 테스트 스크립트 생성: {test_path}")
    print("  💡 테스트 실행: python test_eye_service_fixed.py")

if __name__ == "__main__":
    print("🔧 안구질환 서비스 Eager Execution 수정")
    print("=" * 60)
    
    # 서비스 수정
    if fix_eye_disease_service():
        # 테스트 코드 생성
        test_eye_disease_service()
        
        print("\n📋 다음 단계:")
        print("  1. 테스트 실행: python test_eye_service_fixed.py")
        print("  2. 서버 재시작: 서비스 재구동")
        print("  3. 프론트엔드에서 다시 테스트")
    
    print("\n✅ 작업 완료!")
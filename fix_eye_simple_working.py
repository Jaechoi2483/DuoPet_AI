"""
안구질환 서비스 - 가장 간단한 해결책
"""
import shutil
from pathlib import Path

def create_simple_working_service():
    """즉시 작동하는 가장 간단한 서비스"""
    
    service_content = '''"""
안구질환 진단 서비스 - 간단한 작동 버전
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        # 모델 로드
        self.model = None
        
        # 가능한 모든 경로 시도
        paths_to_try = [
            model_path.replace('.keras', '_fixed.h5'),
            model_path.replace('.keras', '_tf2.h5'),
            model_path
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    logger.info(f"Trying to load model from {path}")
                    
                    # H5 파일인 경우
                    if path.endswith('.h5'):
                        # 간단하게 로드
                        import h5py
                        with h5py.File(path, 'r') as f:
                            # H5 파일 확인
                            pass
                        
                        # Keras 로드
                        from tensorflow.keras.models import load_model
                        custom_objects = {'swish': tf.nn.swish}
                        
                        # 컴파일 없이 로드
                        self.model = load_model(path, custom_objects=custom_objects, compile=False)
                        
                        # 수동으로 레이어 빌드
                        dummy_input = tf.zeros((1, 224, 224, 3))
                        _ = self.model(dummy_input)
                        
                        logger.info(f"Successfully loaded model from {path}")
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to load from {path}: {e}")
                    continue
        
        if self.model is None:
            raise ValueError("Could not load model from any path")
        
        # 클래스 맵 로드
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        
        logger.info("EyeDiseaseService initialized successfully")
    
    def preprocess_image(self, image_file) -> np.ndarray:
        """이미지 전처리"""
        if hasattr(image_file, 'file'):
            image_file.file.seek(0)
            img = Image.open(image_file.file).convert('RGB')
        elif hasattr(image_file, 'seek'):
            image_file.seek(0)
            img = Image.open(image_file).convert('RGB')
        else:
            img = Image.open(image_file).convert('RGB')
        
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """예측 수행"""
        # 모델을 함수처럼 직접 호출
        predictions = self.model(image_array)
        
        # numpy로 변환
        if hasattr(predictions, 'numpy'):
            predictions_np = predictions.numpy()
        else:
            predictions_np = predictions
        
        predicted_class_index = int(np.argmax(predictions_np[0]))
        confidence = float(predictions_np[0][predicted_class_index])
        
        # 클래스 이름 가져오기
        predicted_class_name = self.class_map.get(str(predicted_class_index), "Unknown")
        
        return predicted_class_name, confidence
    
    def diagnose(self, image_file) -> Dict[str, any]:
        """진단 수행"""
        try:
            preprocessed_image = self.preprocess_image(image_file)
            disease, confidence = self.predict(preprocessed_image)
            
            return {
                "disease": disease,
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            # 기본값 반환
            return {
                "disease": "진단 오류",
                "confidence": 0.0
            }
'''
    
    # 백업
    service_path = Path("services/eye_disease_service.py")
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_simple')
        shutil.copy(service_path, backup_path)
        print(f"✓ 백업 생성: {backup_path}")
    
    # 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    print("✅ 가장 간단한 서비스로 교체 완료!")
    print("  - model.predict() 대신 model() 직접 호출")
    print("  - 컴파일 없이 로드")
    print("  - 에러 핸들링 추가")

if __name__ == "__main__":
    print("🔧 안구질환 서비스 - 간단한 해결책")
    print("=" * 60)
    
    create_simple_working_service()
    
    print("\n✅ 완료!")
    print("\n📋 서버를 재시작하세요:")
    print("  python api/main.py")
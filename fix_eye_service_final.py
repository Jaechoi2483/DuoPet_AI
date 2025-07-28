"""
안구질환 서비스 최종 수정
@tf.function 제거 및 eager mode 보장
"""
import shutil
from pathlib import Path

def fix_eye_disease_service_final():
    """최종 수정된 서비스"""
    
    service_content = '''"""
안구질환 진단 서비스 - 최종 수정 버전
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# 디버그 로그
logger.info(f"[EyeDiseaseService] TensorFlow {tf.__version__} - Eager: {tf.executing_eagerly()}")

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        # Custom objects
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish)
        }
        
        # 모델 로드 우선순위
        model_loaded = False
        
        # 1. TF2 완전 변환 모델 우선
        tf2_complete_path = model_path.replace('.keras', '_tf2_complete.h5')
        if os.path.exists(tf2_complete_path):
            try:
                logger.info(f"Loading TF2 complete model from {tf2_complete_path}")
                self.model = tf.keras.models.load_model(
                    tf2_complete_path,
                    custom_objects=custom_objects,
                    compile=True  # 이미 컴파일된 상태로 로드
                )
                model_loaded = True
                logger.info("Successfully loaded TF2 complete model")
            except Exception as e:
                logger.warning(f"Failed to load TF2 complete model: {e}")
        
        # 2. 기존 fixed 모델
        if not model_loaded:
            fixed_path = model_path.replace('.keras', '_fixed.h5')
            if os.path.exists(fixed_path):
                try:
                    logger.info(f"Loading fixed model from {fixed_path}")
                    self.model = tf.keras.models.load_model(
                        fixed_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    # 재컴파일 - eager mode 강제
                    self.model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'],
                        run_eagerly=True  # 중요: eager mode 강제
                    )
                    model_loaded = True
                    logger.info("Successfully loaded fixed model with eager mode")
                except Exception as e:
                    logger.warning(f"Failed to load fixed model: {e}")
        
        if not model_loaded:
            raise ValueError("Could not load any eye disease model")
        
        # 클래스 맵 로드
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        
        self.input_shape = (224, 224)
        
        # 모델 워밍업 (첫 예측을 빠르게)
        try:
            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = self.model(dummy_input, training=False)
            logger.info("Model warmup completed")
        except:
            pass
        
        logger.info(f"EyeDiseaseService initialized successfully")
    
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
        
        img = img.resize(self.input_shape)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """예측 수행 - @tf.function 제거"""
        
        # 직접 호출 방식 (eager mode)
        try:
            # 모델을 함수처럼 직접 호출
            predictions = self.model(image_array, training=False)
            
            # numpy로 변환
            if tf.is_tensor(predictions):
                predictions_np = predictions.numpy()
            else:
                predictions_np = predictions
            
            # 인덱스와 신뢰도 추출
            predicted_class_index = int(np.argmax(predictions_np[0]))
            confidence = float(predictions_np[0][predicted_class_index])
            
        except Exception as e:
            logger.warning(f"Direct call failed, using predict: {e}")
            # 대체 방법: predict 메서드
            predictions_np = self.model.predict(image_array, verbose=0)
            predicted_class_index = int(np.argmax(predictions_np[0]))
            confidence = float(predictions_np[0][predicted_class_index])
        
        # 클래스 이름 매핑
        predicted_class_name = self.class_map.get(str(predicted_class_index), "Unknown")
        
        return predicted_class_name, confidence
    
    def diagnose(self, image_file) -> Dict[str, any]:
        """진단 수행"""
        try:
            preprocessed_image = self.preprocess_image(image_file)
            disease, confidence = self.predict(preprocessed_image)
            
            return {
                "disease": disease,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            raise
'''
    
    # 백업
    service_path = Path("services/eye_disease_service.py")
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_final')
        shutil.copy(service_path, backup_path)
        print(f"✓ 백업 생성: {backup_path}")
    
    # 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    print("✅ 안구질환 서비스 최종 수정 완료!")
    print("  - @tf.function 제거 (eager mode 보장)")
    print("  - run_eagerly=True 컴파일 옵션")
    print("  - tf.is_tensor() 검사 추가")
    print("  - 모델 워밍업 추가")

if __name__ == "__main__":
    print("🔧 안구질환 서비스 최종 수정")
    print("=" * 60)
    
    fix_eye_disease_service_final()
    
    print("\n✅ 완료!")
    print("\n📋 서버를 재시작하세요:")
    print("  python api/main.py")
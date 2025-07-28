"""
안구질환 서비스 즉시 수정
TF1/TF2 호환성 문제 우회
"""
import shutil
from pathlib import Path

def fix_eye_service_now():
    """즉시 작동하는 서비스로 수정"""
    
    fixed_service = '''"""
안구질환 진단 서비스 - 긴급 수정 버전
TF1/TF2 호환성 문제 우회
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# Graph execution 모드로 전환 (TF1 스타일 모델용)
tf.compat.v1.disable_eager_execution()

import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)
print(f"[EyeDiseaseService] TensorFlow {tf.__version__} - Eager: {tf.executing_eagerly()}")

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        # 세션 생성
        self.session = tf.compat.v1.Session()
        
        with self.session.as_default():
            # Custom objects
            custom_objects = {
                'swish': tf.nn.swish,
                'Swish': tf.keras.layers.Activation(tf.nn.swish)
            }
            
            # 모델 로드
            model_loaded = False
            
            # 모델 파일 찾기
            model_candidates = [
                model_path.replace('.keras', '_fixed.h5'),
                model_path.replace('.keras', '_tf2.h5'),
                model_path
            ]
            
            for candidate in model_candidates:
                if os.path.exists(candidate):
                    try:
                        logger.info(f"Loading model from {candidate}")
                        self.model = tf.keras.models.load_model(
                            candidate,
                            custom_objects=custom_objects,
                            compile=True
                        )
                        model_loaded = True
                        logger.info(f"Successfully loaded model")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {candidate}: {e}")
            
            if not model_loaded:
                raise ValueError("Could not load any eye disease model")
        
        # 클래스 맵 로드
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        
        self.input_shape = (224, 224)
        
        # Graph 초기화
        with self.session.as_default():
            # 더미 예측으로 graph 초기화
            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = self.model.predict(dummy_input)
        
        logger.info("EyeDiseaseService initialized with graph mode")
    
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
        """예측 수행"""
        with self.session.as_default():
            # Graph mode에서 예측
            predictions = self.model.predict(image_array)
        
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        predicted_class_name = self.class_map.get(str(predicted_class_index), "Unknown")
        
        return predicted_class_name, confidence
    
    def diagnose(self, image_file) -> Dict[str, any]:
        """진단 수행"""
        preprocessed_image = self.preprocess_image(image_file)
        disease, confidence = self.predict(preprocessed_image)
        
        return {
            "disease": disease,
            "confidence": confidence
        }
    
    def __del__(self):
        """세션 정리"""
        if hasattr(self, 'session'):
            self.session.close()
'''
    
    # 백업
    service_path = Path("services/eye_disease_service.py")
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_immediate')
        shutil.copy(service_path, backup_path)
        print(f"✓ 백업 생성: {backup_path}")
    
    # 서비스 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(fixed_service)
    
    print("✅ 안구질환 서비스 긴급 수정 완료!")
    print("  - Graph execution 모드 사용 (TF1 호환)")
    print("  - Session 기반 예측")
    print("  - 즉시 작동 가능")

def create_alternative_service():
    """대안 서비스 (model.predict_on_batch 사용)"""
    
    alt_service = '''"""
안구질환 진단 서비스 - 대안 버전
predict_on_batch 사용
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

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        # Custom objects
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish)
        }
        
        # 모델 로드
        model_loaded = False
        model_candidates = [
            model_path.replace('.keras', '_fixed.h5'),
            model_path.replace('.keras', '_tf2.h5'),
            model_path
        ]
        
        for candidate in model_candidates:
            if os.path.exists(candidate):
                try:
                    logger.info(f"Loading model from {candidate}")
                    self.model = tf.keras.models.load_model(
                        candidate,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    # 재컴파일
                    self.model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'],
                        run_eagerly=True  # Eager mode 강제
                    )
                    model_loaded = True
                    logger.info("Successfully loaded model")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {candidate}: {e}")
        
        if not model_loaded:
            raise ValueError("Could not load any eye disease model")
        
        # 클래스 맵 로드
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        
        self.input_shape = (224, 224)
        logger.info("EyeDiseaseService initialized")
    
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
        """예측 수행 - predict_on_batch 사용"""
        try:
            # predict_on_batch는 더 간단한 예측 메서드
            predictions = self.model.predict_on_batch(image_array)
            
            # numpy 변환
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
                
        except Exception as e:
            logger.warning(f"predict_on_batch failed, using __call__: {e}")
            # 대안: 모델을 함수처럼 호출
            predictions = self.model(image_array, training=False)
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
        
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        predicted_class_name = self.class_map.get(str(predicted_class_index), "Unknown")
        
        return predicted_class_name, confidence
    
    def diagnose(self, image_file) -> Dict[str, any]:
        """진단 수행"""
        preprocessed_image = self.preprocess_image(image_file)
        disease, confidence = self.predict(preprocessed_image)
        
        return {
            "disease": disease,
            "confidence": confidence
        }
'''
    
    # 대안 서비스 저장
    alt_path = Path("services/eye_disease_service_alt.py")
    with open(alt_path, 'w', encoding='utf-8') as f:
        f.write(alt_service)
    
    print(f"\n✅ 대안 서비스 생성: {alt_path}")
    print("  - predict_on_batch 메서드 사용")
    print("  - run_eagerly=True 컴파일 옵션")

if __name__ == "__main__":
    print("🚨 안구질환 서비스 즉시 수정")
    print("=" * 60)
    
    # 1. Graph mode 서비스 (즉시 작동)
    fix_eye_service_now()
    
    # 2. 대안 서비스
    create_alternative_service()
    
    print("\n📋 다음 단계:")
    print("  1. 서버 재시작: python api/main.py")
    print("  2. 프론트엔드에서 테스트")
    print("\n💡 대안:")
    print("  Graph mode가 문제가 있으면:")
    print("  cp services/eye_disease_service_alt.py services/eye_disease_service.py")
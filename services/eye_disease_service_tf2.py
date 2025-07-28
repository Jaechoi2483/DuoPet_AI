"""
안구질환 진단 서비스 - TensorFlow 2.x 완전 호환 버전
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
        
        # 모델 로드 우선순위
        model_loaded = False
        
        # 1. TF2 완전 변환 모델
        tf2_complete_path = model_path.replace('.keras', '_tf2_complete.h5')
        if os.path.exists(tf2_complete_path):
            try:
                logger.info(f"Loading TF2 complete model from {tf2_complete_path}")
                self.model = tf.keras.models.load_model(
                    tf2_complete_path,
                    custom_objects=custom_objects
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
                    self.model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    model_loaded = True
                    logger.info("Successfully loaded fixed model")
                except Exception as e:
                    logger.warning(f"Failed to load fixed model: {e}")
        
        if not model_loaded:
            raise ValueError("Could not load any eye disease model")
        
        # 클래스 맵 로드
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        
        self.input_shape = (224, 224)
        logger.info(f"EyeDiseaseService initialized")
    
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
    
    @tf.function
    def predict_tf2(self, image_array):
        """TF2 스타일 예측 (graph mode에서도 작동)"""
        return self.model(image_array, training=False)
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """예측 수행"""
        # TF2 스타일로 예측
        try:
            # 직접 호출 사용
            predictions = self.predict_tf2(image_array)
            
            # numpy로 변환
            if hasattr(predictions, 'numpy'):
                predictions_np = predictions.numpy()
            else:
                predictions_np = predictions
                
        except Exception as e:
            logger.warning(f"TF2 predict failed, using legacy predict: {e}")
            # 대체 방법
            predictions_np = self.model.predict(image_array, verbose=0)
        
        predicted_class_index = int(np.argmax(predictions_np[0]))
        confidence = float(np.max(predictions_np[0]))
        
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


import json
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Dict, Tuple
import logging
import sys
import os

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import load_model_with_custom_objects, safe_model_predict

logger = logging.getLogger(__name__)

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """
        안구 질환 진단 서비스 초기화

        Args:
            model_path (str): Keras 모델 파일 경로
            class_map_path (str): 클래스 맵 JSON 파일 경로
        """
        try:
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
        img = Image.open(image_file.file).convert('RGB')
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
        predictions = safe_model_predict(self.model, image_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
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
        
        return {
            "disease": disease,
            "confidence": confidence
        }

"""
Health Diagnosis Prediction Service

This module provides prediction functions for health diagnosis db_models.
It is fully integrated with the project's configuration and custom exceptions.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Union
from PIL import Image
import cv2

# 프로젝트의 공통 모듈들을 임포트합니다.
from common.logger import get_logger
from common.config import get_settings, get_model_path, get_model_config
# 'DataValidationError'를 'ValidationError'로 수정했습니다.
from common.exceptions import ModelNotLoadedError, ModelInferenceError, ValidationError

# 초기 설정
logger = get_logger(__name__)
settings = get_settings()
model_configs = get_model_config()


class HealthDiagnosisPredictor:
    """
    AI 모델을 로드하고 상태 진단 예측을 수행하는 핵심 클래스입니다.
    """

    def __init__(self):
        """
        모델을 저장할 딕셔너리를 초기화합니다.
        """
        self.loaded_models = {}

    def _load_model(self, model_key: str, model_filename: str) -> tf.keras.Model:
        """
        설정 파일에 정의된 경로에서 모델을 로드합니다.
        모델이 이미 로드된 경우, 기존 객체를 반환합니다.
        """
        if model_key not in self.loaded_models:
            model_path = get_model_path(model_filename)
            if not os.path.exists(model_path):
                raise ModelNotLoadedError(model_name=model_key, reason=f"Model file not found at {model_path}")

            try:
                logger.info(f"Loading {model_key} model from {model_path}")
                self.loaded_models[model_key] = tf.keras.models.load_model(model_path)
            except Exception as e:
                raise ModelNotLoadedError(model_name=model_key, reason=f"Failed to load model from {model_path}: {e}")

        return self.loaded_models[model_key]

    def _preprocess_image(self, image: Union[Image.Image, np.ndarray],
                          target_size: tuple,
                          normalize: bool = True) -> np.ndarray:
        """
        입력 이미지를 모델에 맞게 전처리합니다.
        (리사이즈, 정규화 등)
        """
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # 컬러 이미지인 경우
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32)

        if normalize:
            img_array = img_array / 255.0

        return img_array

    def predict_eye_disease(self, image: Union[Image.Image, np.ndarray]) -> Dict:
        """
        안구 질환을 예측합니다.
        """
        try:
            eye_model_config = model_configs.get_model_config("eye_disease")
            model_filename = eye_model_config.get("filename", "eye_disease_model.h5")
            model = self._load_model('eye_disease', model_filename)

            input_size = tuple(eye_model_config.get("input_size", [224, 224]))
            processed_image = self._preprocess_image(image, target_size=input_size)
            input_batch = np.expand_dims(processed_image, axis=0)

            predictions = model.predict(input_batch)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class])

            class_names = eye_model_config.get("class_names", [])
            if not class_names or predicted_class >= len(class_names):
                raise ModelInferenceError("eye_disease", "Class name configuration is invalid.")

            disease_type = class_names[predicted_class]
            disease_detected = disease_type != '정상'

            # 결과 포맷팅
            return {
                'disease_detected': disease_detected,
                'disease_type': disease_type,
                'confidence': round(confidence * 100, 2),
                'details': {class_names[i]: round(float(predictions[0][i]) * 100, 2) for i in range(len(class_names))}
            }
        except (ModelNotLoadedError, ValidationError) as e:
            raise e
        except Exception as e:
            logger.error(f"Unhandled error in eye disease prediction: {e}")
            raise ModelInferenceError(model_name="eye_disease", reason=str(e))

    def predict_skin_disease(self, image: Union[Image.Image, np.ndarray]) -> Dict:
        """
        피부 질환을 예측합니다.
        """
        try:
            skin_model_config = model_configs.get_model_config("skin_disease")
            model_filename = skin_model_config.get("filename", "skin_disease_model.h5")
            model = self._load_model('skin_disease', model_filename)

            input_size = tuple(skin_model_config.get("input_size", [224, 224]))
            processed_image = self._preprocess_image(image, target_size=input_size)
            input_batch = np.expand_dims(processed_image, axis=0)

            predictions = model.predict(input_batch)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class])

            class_names = skin_model_config.get("class_names", [])
            if not class_names or predicted_class >= len(class_names):
                raise ModelInferenceError("skin_disease", "Class name configuration is invalid.")

            disease_type = class_names[predicted_class]
            disease_detected = disease_type != '정상'

            return {
                'disease_detected': disease_detected,
                'disease_type': disease_type,
                'confidence': round(confidence * 100, 2),
                'details': {class_names[i]: round(float(predictions[0][i]) * 100, 2) for i in range(len(class_names))}
            }
        except (ModelNotLoadedError, ValidationError) as e:
            raise e
        except Exception as e:
            logger.error(f"Unhandled error in skin disease prediction: {e}")
            raise ModelInferenceError(model_name="skin_disease", reason=str(e))

    def predict_bcs(self, images: List[Union[Image.Image, np.ndarray]]) -> Dict:
        """
        BCS(신체 상태 점수)를 예측합니다.
        """
        # 'DataValidationError'를 'ValidationError'로 수정했습니다.
        if not isinstance(images, list) or len(images) != 13:
            raise ValidationError(message=f"BCS prediction requires a list of 13 images, but got {len(images)}.")

        try:
            bcs_model_config = model_configs.get_model_config("bcs")
            model_filename = bcs_model_config.get("filename", "bcs_model.h5")
            model = self._load_model('bcs', model_filename)

            input_size = tuple(bcs_model_config.get("input_size", [224, 224]))
            
            processed_images = [self._preprocess_image(img, target_size=input_size) for img in images]
            input_batch = np.array(processed_images)

            predictions = model.predict(input_batch)
            
            # 여러 이미지 예측 결과의 평균 사용
            avg_prediction = np.mean(predictions, axis=0)
            predicted_class = int(np.argmax(avg_prediction))
            confidence = float(avg_prediction[predicted_class])

            class_names = bcs_model_config.get("class_names", [])
            if not class_names or predicted_class >= len(class_names):
                raise ModelInferenceError("bcs", "Class name configuration is invalid.")

            bcs_category = class_names[predicted_class]

            # 클래스에 따른 BCS 점수 매핑
            score_map = bcs_model_config.get("score_map", {"저체중": 2, "정상": 5, "과체중": 8})
            bcs_score = score_map.get(bcs_category, 5)

            return {
                'bcs_score': bcs_score,
                'category': bcs_category,
                'confidence': round(confidence * 100, 2)
            }
        except (ModelNotLoadedError, ValidationError) as e:
            raise e
        except Exception as e:
            logger.error(f"Unhandled error in BCS prediction: {e}")
            raise ModelInferenceError(model_name="bcs", reason=str(e))


# --- Singleton 인스턴스 ---
# 애플리케이션 전체에서 HealthDiagnosisPredictor 인스턴스를 하나만 사용하도록 관리합니다.
try:
    predictor = HealthDiagnosisPredictor()
except Exception as e:
    logger.error(f"Failed to initialize HealthDiagnosisPredictor: {e}")
    predictor = None


# --- Convenience Functions (외부 호출용) ---
# API 라우터에서 쉽게 호출할 수 있도록 제공되는 함수들입니다.

def predict_eye_disease(image: Union[Image.Image, np.ndarray]) -> Dict:
    """안구 질환 예측을 위한 편의 함수"""
    if not predictor:
        raise ModelNotLoadedError(model_name="HealthDiagnosisPredictor", reason="Predictor is not initialized.")
    return predictor.predict_eye_disease(image)


def predict_skin_disease(image: Union[Image.Image, np.ndarray]) -> Dict:
    """피부 질환 예측을 위한 편의 함수"""
    if not predictor:
        raise ModelNotLoadedError(model_name="HealthDiagnosisPredictor", reason="Predictor is not initialized.")
    return predictor.predict_skin_disease(image)


def predict_bcs(images: List[Union[Image.Image, np.ndarray]]) -> Dict:
    """BCS 예측을 위한 편의 함수"""
    if not predictor:
        raise ModelNotLoadedError(model_name="HealthDiagnosisPredictor", reason="Predictor is not initialized.")
    return predictor.predict_bcs(images)
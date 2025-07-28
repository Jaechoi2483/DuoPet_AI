"""
피부질환 진단 서비스 (수정 버전)
TensorFlow 2.x eager execution 활성화
"""
import os
import tensorflow as tf

# TensorFlow 설정을 가장 먼저
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from common.logger import get_logger
from services.model_registry import ModelRegistry
from services.model_adapters.skin_disease_adapter import SkinDiseaseAdapter

logger = get_logger(__name__)

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class SkinDiseaseService:
    """피부질환 진단 서비스"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.models = {}
        self._load_models()
        
    def _load_models(self):
        """모델 로드"""
        try:
            # 모델 경로 설정
            base_path = Path(__file__).parent.parent
            models_path = base_path / "models" / "health_diagnosis" / "skin_disease" / "classification"
            
            # TF2 변환된 모델 우선 사용
            model_configs = {
                "cat_binary": [
                    models_path / "cat_binary" / "cat_binary_model_tf2_perfect.h5",
                    models_path / "cat_binary" / "cat_binary_model.h5"
                ],
                "dog_binary": [
                    models_path / "dog_binary" / "dog_binary_model_tf2_perfect.h5",
                    models_path / "dog_binary" / "dog_binary_model.h5"
                ],
                "dog_multi_136": [
                    models_path / "dog_multi_136" / "dog_multi_136_model_tf2_perfect.h5",
                    models_path / "dog_multi_136" / "dog_multi_136_model.h5"
                ],
                "dog_multi_456": [
                    models_path / "dog_multi_456" / "dog_multi_456_model_tf2_perfect.h5",
                    models_path / "dog_multi_456" / "dog_multi_456_model.h5"
                ]
            }
            
            # 각 모델 로드 시도
            for model_name, paths in model_configs.items():
                for path in paths:
                    if path.exists():
                        try:
                            logger.info(f"Loading {model_name} from {path}")
                            model = tf.keras.models.load_model(str(path), compile=False)
                            
                            # 컴파일
                            if "binary" in model_name:
                                model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy']
                                )
                            else:
                                model.compile(
                                    optimizer='adam',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                            
                            self.models[model_name] = model
                            logger.info(f"Successfully loaded {model_name}")
                            break
                            
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name} from {path}: {e}")
                            continue
            
            logger.info(f"Loaded {len(self.models)} skin disease models")
            
        except Exception as e:
            logger.error(f"Error loading skin disease models: {e}")
    
    def predict(self, image: np.ndarray, pet_type: str) -> Dict[str, Any]:
        """피부질환 예측"""
        
        # 이미지 전처리
        if image.shape != (224, 224, 3):
            image = tf.image.resize(image, (224, 224))
        
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # 배치 차원 추가
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # 펫 타입에 따른 모델 선택
        if pet_type.lower() == "cat":
            model_key = "cat_binary"
            multi_models = []
        else:
            model_key = "dog_binary"
            multi_models = ["dog_multi_136", "dog_multi_456"]
        
        result = {
            "status": "success",
            "pet_type": pet_type,
            "binary_classification": None,
            "multi_classification": {},
            "confidence": 0.0
        }
        
        # Binary classification
        if model_key in self.models:
            model = self.models[model_key]
            pred = model.predict(image, verbose=0)
            
            is_disease = float(pred[0][0]) > 0.5
            confidence = float(pred[0][0]) if is_disease else float(1 - pred[0][0])
            
            result["binary_classification"] = {
                "has_disease": is_disease,
                "confidence": confidence
            }
            result["confidence"] = confidence
        
        # Multi-class classification (개만)
        if pet_type.lower() == "dog" and is_disease:
            for multi_key in multi_models:
                if multi_key in self.models:
                    model = self.models[multi_key]
                    pred = model.predict(image, verbose=0)
                    
                    class_idx = int(np.argmax(pred[0]))
                    confidence = float(pred[0][class_idx])
                    
                    # 클래스 매핑
                    if "136" in multi_key:
                        classes = ["구진플라크", "무증상", "농포여드름"]
                    else:
                        classes = ["과다색소침착", "결절종괴", "미란궤양"]
                    
                    result["multi_classification"][multi_key] = {
                        "class": classes[class_idx],
                        "confidence": confidence
                    }
        
        # numpy 타입 변환
        return convert_numpy_types(result)

# 서비스 인스턴스
_service_instance = None

def get_skin_disease_service():
    """싱글톤 서비스 반환"""
    global _service_instance
    if _service_instance is None:
        _service_instance = SkinDiseaseService()
    return _service_instance

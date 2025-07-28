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
from PIL import Image
from tensorflow.keras import backend as K

from common.logger import get_logger
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
        return round(float(obj), 4)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class SkinDiseaseService:
    """피부질환 진단 서비스"""
    
    def __init__(self):
        # TF 모드 확인
        self.use_eager = tf.executing_eagerly()
        logger.info(f"[SkinDiseaseService] TensorFlow {tf.__version__} - Eager mode: {self.use_eager}")
        
        # Graph mode인 경우 Session 초기화
        if not self.use_eager:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.compat.v1.Session(config=config)
            logger.info("[SkinDiseaseService] Graph mode - Session created")
        else:
            self.session = None
            
        self.models = {}
        self._load_models()
    
    def preprocess_image(self, image_file) -> np.ndarray:
        """이미지 전처리 (UploadFile -> numpy array)"""
        # UploadFile 처리
        if hasattr(image_file, 'file'):
            image_file.file.seek(0)
            img = Image.open(image_file.file).convert('RGB')
        elif hasattr(image_file, 'seek'):
            image_file.seek(0)
            img = Image.open(image_file).convert('RGB')
        else:
            img = Image.open(image_file).convert('RGB')
        
        # 리사이즈 및 정규화
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        
        # 정규화는 predict 메소드에서 처리하므로 여기서는 하지 않음
        return img_array
        
    def _load_models(self):
        """모델 로드"""
        try:
            # 모델 경로 설정
            base_path = Path(__file__).parent.parent
            models_path = base_path / "models" / "health_diagnosis" / "skin_disease" / "classification"
            
            # 원본 모델 우선 사용으로 변경 (TF2 변환 모델에 문제 있음)
            model_configs = {
                "cat_binary": [
                    models_path / "cat_binary" / "cat_binary_model.h5",
                    models_path / "cat_binary" / "cat_binary_model_tf2_perfect.h5"
                ],
                "dog_binary": [
                    models_path / "dog_binary" / "dog_binary_model.h5",
                    models_path / "dog_binary" / "dog_binary_model_tf2_perfect.h5"
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
                            
                            # Graph mode인 경우 session context 사용
                            if not self.use_eager and self.session:
                                with self.session.as_default():
                                    with self.session.graph.as_default():
                                        model = tf.keras.models.load_model(str(path), compile=False)
                            else:
                                model = tf.keras.models.load_model(str(path), compile=False)
                            
                            # 모델 정보 로깅
                            logger.info(f"Model info - Input shape: {model.input_shape}, Output shape: {model.output_shape}")
                            if hasattr(model, 'layers') and len(model.layers) > 0:
                                # 마지막 몇 개 층 정보 출력
                                logger.info("=== Model layers (last 5) ===")
                                for i, layer in enumerate(model.layers[-5:]):
                                    logger.info(f"  Layer {i}: {layer.name} ({layer.__class__.__name__})")
                                    if hasattr(layer, 'units'):
                                        logger.info(f"    Units: {layer.units}")
                                    if hasattr(layer, 'activation'):
                                        activation = layer.activation
                                        if hasattr(activation, '__name__'):
                                            activation = activation.__name__
                                        elif hasattr(activation, 'name'):
                                            activation = activation.name
                                        logger.info(f"    Activation: {activation}")
                                
                                # 모델 파라미터 수
                                logger.info(f"Total parameters: {model.count_params():,}")
                            
                            # 컴파일하지 않고 모델만 저장 (predict 시 직접 호출)
                            self.models[model_name] = model
                            logger.info(f"Successfully loaded {model_name}")
                            break
                            
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name} from {path}: {e}")
                            continue
            
            logger.info(f"Loaded {len(self.models)} skin disease models")
            
        except Exception as e:
            logger.error(f"Error loading skin disease models: {e}")
    
    async def diagnose_skin_condition(
        self,
        image,  # UploadFile 객체
        pet_type: str = "dog",
        include_segmentation: bool = False
    ) -> Dict[str, Any]:
        """
        피부질환 진단 (라우터 호환성을 위한 async 메소드)
        
        Args:
            image: UploadFile 객체
            pet_type: 펫 타입 ("dog" 또는 "cat")
            include_segmentation: 세그멘테이션 포함 여부 (현재 미지원)
            
        Returns:
            진단 결과 딕셔너리
        """
        # 이미지 전처리
        preprocessed_image = self.preprocess_image(image)
        
        # 동기 predict 메소드 호출
        result = self.predict(preprocessed_image, pet_type)
        
        # 라우터에서 기대하는 형식으로 변환
        has_disease = result.get("binary_classification", {}).get("disease", 0) > 0.5
        
        return {
            "has_skin_disease": has_disease,
            "disease_type": result.get("disease_type", "정상"),
            "confidence": result.get("confidence", 0.0),
            "predictions": result.get("multi_classification", {}),
            "binary_classification": result.get("binary_classification", {}),
            "status": result.get("status", "success")
        }
    
    def predict(self, image: np.ndarray, pet_type: str) -> Dict[str, Any]:
        """피부질환 예측 (내부 메소드)"""
        
        # 디버깅: 원본 이미지 정보
        logger.info(f"[DEBUG] Original image - shape: {image.shape}, dtype: {image.dtype}, min: {image.min():.4f}, max: {image.max():.4f}")
        
        # 이미지 전처리
        if image.shape != (224, 224, 3):
            image = tf.image.resize(image, (224, 224))
            logger.info(f"[DEBUG] After resize - shape: {image.shape}")
        
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # 정규화 (0-255 -> 0-1)
        if image.max() > 1.0:
            image = image / 255.0
            logger.info(f"[DEBUG] After normalization - min: {image.min():.4f}, max: {image.max():.4f}")
        
        # 배치 차원 추가
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        logger.info(f"[DEBUG] Final input - shape: {image.shape}, dtype: {image.dtype}")
        
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
            logger.info(f"[DEBUG] Using {model_key} model for {pet_type}")
            
            # 모델 예측 (Graph/Eager mode 안전 처리)
            if not self.use_eager and self.session:
                with self.session.as_default():
                    with self.session.graph.as_default():
                        # Graph mode에서는 model()을 호출하고 K.get_value()로 평가
                        pred_tensor = model(image, training=False)
                        pred = K.get_value(pred_tensor)
            else:
                pred = model.predict(image, verbose=0)
            
            # 디버깅: raw prediction 값
            logger.info(f"[DEBUG] Raw prediction type: {type(pred)}, shape: {pred.shape if hasattr(pred, 'shape') else 'no shape'}")
            logger.info(f"[DEBUG] Raw prediction value: {pred}")
            
            # 극단적 테스트 (디버깅용)
            if model_key == "dog_binary":
                test_black = np.zeros((1, 224, 224, 3), dtype=np.float32)
                test_white = np.ones((1, 224, 224, 3), dtype=np.float32)
                test_random = np.random.random((1, 224, 224, 3)).astype(np.float32)
                
                pred_black = K.get_value(model(test_black, training=False)) if not self.use_eager else model.predict(test_black, verbose=0)
                pred_white = K.get_value(model(test_white, training=False)) if not self.use_eager else model.predict(test_white, verbose=0)
                pred_random = K.get_value(model(test_random, training=False)) if not self.use_eager else model.predict(test_random, verbose=0)
                
                logger.info(f"[DEBUG] Extreme tests - Black: {pred_black[0]}, White: {pred_white[0]}, Random: {pred_random[0]}")
            
            # numpy array로 확실히 변환
            if hasattr(pred, 'numpy'):
                pred_np = pred.numpy()
            else:
                pred_np = pred
            
            logger.info(f"[DEBUG] Numpy prediction: {pred_np}")
            
            # 예측값 추출 - 모델은 [정상 확률, 질환 확률] 형태로 출력
            if pred_np.shape[-1] == 2:
                # 2개 출력인 경우 (원본 모델 형식)
                normal_prob = float(pred_np[0][0])
                disease_prob = float(pred_np[0][1])
                logger.info(f"[DEBUG] Two outputs detected - normal: {normal_prob:.4f}, disease: {disease_prob:.4f}")
            else:
                # 1개 출력인 경우 (변환된 모델?)
                disease_prob = float(pred_np[0][0])
                normal_prob = 1.0 - disease_prob
                logger.info(f"[DEBUG] Single output detected - disease: {disease_prob:.4f}")
            
            is_disease = disease_prob > 0.5
            confidence = disease_prob if is_disease else normal_prob
            
            logger.info(f"[DEBUG] Binary classification - disease_prob: {disease_prob:.4f}, is_disease: {is_disease}, confidence: {confidence:.4f}")
            
            result["binary_classification"] = {
                "normal": normal_prob,
                "disease": disease_prob
            }
            result["confidence"] = confidence
        
        # Multi-class classification (개만)
        if pet_type.lower() == "dog" and is_disease:
            for multi_key in multi_models:
                if multi_key in self.models:
                    model = self.models[multi_key]
                    
                    # 모델 예측 (Graph/Eager mode 안전 처리)
                    if not self.use_eager and self.session:
                        with self.session.as_default():
                            with self.session.graph.as_default():
                                # Graph mode에서는 model()을 호출하고 K.get_value()로 평가
                                pred_tensor = model(image, training=False)
                                pred = K.get_value(pred_tensor)
                    else:
                        pred = model.predict(image, verbose=0)
                    
                    # numpy array로 확실히 변환
                    if hasattr(pred, 'numpy'):
                        pred_np = pred.numpy()
                    else:
                        pred_np = pred
                    
                    class_idx = int(np.argmax(pred_np[0]))
                    confidence = float(pred_np[0][class_idx])
                    
                    # 클래스 매핑
                    if "136" in multi_key:
                        classes = ["구진플라크", "무증상", "농포여드름"]
                    else:
                        classes = ["과다색소침착", "결절종괴", "미란궤양"]
                    
                    result["multi_classification"][multi_key] = {
                        "class": classes[class_idx],
                        "confidence": confidence
                    }
        
        # 가장 확률이 높은 질병을 disease_type으로 설정
        if result["binary_classification"] and result["binary_classification"]["disease"] > 0.5:
            # multi_classification에서 가장 확률이 높은 것 찾기
            max_confidence = 0
            disease_type = "미분류"
            
            for key, value in result["multi_classification"].items():
                if value["confidence"] > max_confidence:
                    max_confidence = value["confidence"]
                    disease_type = value["class"]
            
            result["disease_type"] = disease_type
            result["confidence"] = max_confidence
        else:
            result["disease_type"] = "정상"
            result["confidence"] = normal_prob if "normal_prob" in locals() else (float(result["binary_classification"]["normal"]) if result["binary_classification"] else 0.0)
        
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

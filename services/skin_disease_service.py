"""
피부질환 진단 서비스 - Graph/Eager mode 자동 처리
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from PIL import Image

# CustomScaleLayer 정의 (InceptionResNetV2 모델용)
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        if isinstance(inputs, list):
            return inputs[0] * self.scale
        else:
            return inputs * self.scale

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config

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
            self.placeholders = {}  # 각 모델별 placeholder 저장
            self.prediction_tensors = {}  # 각 모델별 예측 tensor 저장
            logger.info("[SkinDiseaseService] Graph mode - Session created")
        else:
            self.session = None
            self.placeholders = None
            self.prediction_tensors = None
            
        self.models = {}
        self.loaded_pet_types = set()  # 로드된 펫 타입 추적
        
        # 모델 경로 설정 (나중에 사용)
        base_path = Path(__file__).parent.parent
        self.models_path = base_path / "models" / "health_diagnosis" / "skin_disease" / "classification"
    
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
        
    def _load_models_for_pet_type(self, pet_type: str):
        """특정 펫 타입에 필요한 모델만 로드"""
        if pet_type in self.loaded_pet_types:
            logger.info(f"Models for {pet_type} already loaded")
            return
            
        try:
            logger.info(f"Loading models for pet type: {pet_type}")
            
            # 펫 타입별 필요한 모델 결정
            if pet_type.lower() == "cat":
                models_to_load = ["cat_binary"]
            else:  # dog or default
                models_to_load = ["dog_binary", "dog_multi_136", "dog_multi_456"]
            
            # 테스트 결과 기반 작동 확인된 모델만 사용
            model_configs = {
                "cat_binary": [
                    self.models_path / "cat_binary" / "cat_binary_model_simple.h5",  # CustomScaleLayer 없는 버전 우선
                    self.models_path / "cat_binary" / "cat_binary_model.h5",  # 원본 (작동 확인됨)
                    self.models_path / "cat_binary" / "cat_binary_model_tf2_perfect.h5",  # TF2 버전 (작동 확인됨)
                    self.models_path / "cat_binary" / "cat_binary_model_tf2.h5"  # TF2 버전 (작동 확인됨)
                ],
                "dog_binary": [
                    self.models_path / "dog_binary" / "dog_binary_model_simple.h5",  # CustomScaleLayer 없는 버전 우선
                    self.models_path / "dog_binary" / "dog_binary_model.h5",  # 원본 (작동 확인됨)
                    self.models_path / "dog_binary" / "dog_binary_model_tf2_perfect.h5",  # TF2 버전 (작동 확인됨)
                    self.models_path / "dog_binary" / "dog_binary_model_tf2.h5"  # TF2 버전 (작동 확인됨)
                ],
                "dog_multi_136": [
                    self.models_path / "dog_multi_136" / "dog_multi_136_model_simple.h5",  # CustomScaleLayer 없는 버전 우선
                    self.models_path / "dog_multi_136" / "dog_multi_136_model.h5",  # 원본 (작동 확인됨)
                    self.models_path / "dog_multi_136" / "dog_multi_136_model_tf2_perfect.h5",  # TF2 버전 (작동 확인됨)
                    self.models_path / "dog_multi_136" / "dog_multi_136_model_tf2.h5"  # TF2 버전 (작동 확인됨)
                ],
                "dog_multi_456": [
                    self.models_path / "dog_multi_456" / "dog_multi_456_model.h5",  # 원본 (작동 확인됨)
                    self.models_path / "dog_multi_456" / "dog_multi_456_model_tf2_perfect.h5",  # TF2 버전 (작동 확인됨)
                    self.models_path / "dog_multi_456" / "dog_multi_456_model_tf2.h5"  # TF2 버전 (작동 확인됨)
                ]
            }
            
            # 필요한 모델만 로드
            for model_name in models_to_load:
                if model_name in self.models:
                    logger.info(f"{model_name} already loaded, skipping")
                    continue
                    
                paths = model_configs.get(model_name, [])
                for path in paths:
                    if path.exists():
                        try:
                            logger.info(f"Loading {model_name} from {path}")
                            
                            # CustomScaleLayer를 포함한 custom objects
                            custom_objects = {'CustomScaleLayer': CustomScaleLayer}
                            
                            if not self.use_eager:
                                # Graph mode에서 로드
                                with self.session.as_default():
                                    with self.session.graph.as_default():
                                        with tf.keras.utils.custom_object_scope(custom_objects):
                                            model = tf.keras.models.load_model(str(path), compile=False)
                                        
                                        # 모델 컴파일
                                        model.compile(
                                            optimizer='adam',
                                            loss='binary_crossentropy' if 'binary' in model_name else 'sparse_categorical_crossentropy',
                                            metrics=['accuracy']
                                        )
                                        
                                        # placeholder와 prediction tensor 생성
                                        self.placeholders[model_name] = tf.compat.v1.placeholder(
                                            tf.float32,
                                            shape=[None, 224, 224, 3],
                                            name=f'{model_name}_input'
                                        )
                                        self.prediction_tensors[model_name] = model(self.placeholders[model_name])
                                        
                                        # Graph 초기화
                                        self.session.run(tf.compat.v1.global_variables_initializer())
                            else:
                                # Eager mode에서 로드
                                with tf.keras.utils.custom_object_scope(custom_objects):
                                    model = tf.keras.models.load_model(str(path), compile=False)
                                
                                # 모델 컴파일
                                model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy' if 'binary' in model_name else 'sparse_categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                            
                            # 모델 정보 로깅
                            logger.info(f"Model info - Input shape: {model.input_shape}, Output shape: {model.output_shape}")
                            logger.info(f"Total parameters: {model.count_params():,}")
                            
                            # 컴파일하지 않고 모델만 저장 (predict 시 직접 호출)
                            self.models[model_name] = model
                            logger.info(f"Successfully loaded {model_name}")
                            break
                            
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name} from {path}: {e}")
                            continue
            
            self.loaded_pet_types.add(pet_type)
            logger.info(f"Loaded {len([m for m in models_to_load if m in self.models])} models for {pet_type}")
            
        except Exception as e:
            logger.error(f"Error loading models for {pet_type}: {e}")
    
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
        # 펫 타입에 맞는 모델 로드
        self._load_models_for_pet_type(pet_type)
        
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
        
        # 펫 타입에 맞는 모델 로드 (이미 로드되어 있으면 스킵)
        self._load_models_for_pet_type(pet_type)
        
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
            
            # 모델 예측 - Graph/Eager mode 자동 처리
            try:
                if self.use_eager:
                    # Eager mode 예측
                    pred = model(image, training=False)
                    if hasattr(pred, 'numpy'):
                        pred_np = pred.numpy()
                    else:
                        pred_np = pred
                else:
                    # Graph mode 예측
                    pred_np = self.session.run(
                        self.prediction_tensors[model_key],
                        feed_dict={self.placeholders[model_key]: image}
                    )
                
                # 디버깅: raw prediction 값
                logger.info(f"[DEBUG] Raw prediction type: {type(pred_np)}, shape: {pred_np.shape if hasattr(pred_np, 'shape') else 'no shape'}")
                logger.info(f"[DEBUG] Raw prediction value: {pred_np}")
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                # Fallback: model.predict 사용
                try:
                    if self.use_eager:
                        pred_np = model.predict(image, verbose=0)
                    else:
                        with self.session.as_default():
                            pred_np = model.predict(image, verbose=0)
                except Exception as e2:
                    logger.error(f"Fallback prediction also failed: {e2}")
                    # 기본값 설정
                    pred_np = np.array([[0.5, 0.5]])
            
            # 극단적 테스트 (디버깅용) - 그래프 실행 에러 방지를 위해 주석 처리
            # if model_key == "dog_binary":
            #     test_black = np.zeros((1, 224, 224, 3), dtype=np.float32)
            #     test_white = np.ones((1, 224, 224, 3), dtype=np.float32)
            #     test_random = np.random.random((1, 224, 224, 3)).astype(np.float32)
            #     
            #     pred_black = K.get_value(model(test_black, training=False)) if not self.use_eager else model.predict(test_black, verbose=0)
            #     pred_white = K.get_value(model(test_white, training=False)) if not self.use_eager else model.predict(test_white, verbose=0)
            #     pred_random = K.get_value(model(test_random, training=False)) if not self.use_eager else model.predict(test_random, verbose=0)
            #     
            #     logger.info(f"[DEBUG] Extreme tests - Black: {pred_black[0]}, White: {pred_white[0]}, Random: {pred_random[0]}")
            
            # pred_np는 이미 numpy array
            logger.info(f"[DEBUG] Numpy prediction: {pred_np}")
            
            # 예측값 추출 - 체크포인트 변환 모델 확인
            if pred_np.shape[-1] == 2:
                # 2개 출력인 경우 (체크포인트 변환 모델 - 정상적인 형식)
                normal_prob = float(pred_np[0][0])
                disease_prob = float(pred_np[0][1])
                logger.info(f"[DEBUG] Two outputs detected - normal: {normal_prob:.4f}, disease: {disease_prob:.4f}")
            elif pred_np.shape[-1] == 1:
                # 1개 출력인 경우 (현재 10MB 모델)
                # InceptionResNetV2의 출력이 sigmoid이므로 이것은 disease 확률
                disease_prob = float(pred_np[0][0])
                normal_prob = 1.0 - disease_prob
                logger.info(f"[DEBUG] Single output detected - disease: {disease_prob:.4f}")
            else:
                # 예상치 못한 출력
                logger.error(f"[DEBUG] Unexpected output shape: {pred_np.shape}")
                disease_prob = 0.5
                normal_prob = 0.5
            
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
                    
                    # 모델 예측 - Graph/Eager mode 자동 처리
                    try:
                        if self.use_eager:
                            # Eager mode 예측
                            pred = model(image, training=False)
                            if hasattr(pred, 'numpy'):
                                pred_np = pred.numpy()
                            else:
                                pred_np = pred
                        else:
                            # Graph mode 예측
                            pred_np = self.session.run(
                                self.prediction_tensors[multi_key],
                                feed_dict={self.placeholders[multi_key]: image}
                            )
                    except Exception as e:
                        logger.error(f"Multi-class prediction error: {e}")
                        # Fallback
                        if self.use_eager:
                            pred_np = model.predict(image, verbose=0)
                        else:
                            with self.session.as_default():
                                pred_np = model.predict(image, verbose=0)
                    
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
    
    def get_model_status(self) -> Dict[str, Any]:
        """로드된 모델 상태 반환"""
        return {
            "loaded_models": list(self.models.keys()),
            "loaded_pet_types": list(self.loaded_pet_types),
            "total_models_loaded": len(self.models),
            "cat_models_available": "cat_binary" in self.models,
            "dog_models_available": all(m in self.models for m in ["dog_binary", "dog_multi_136", "dog_multi_456"]),
            "service_ready": True
        }
    
    def get_supported_diseases(self, pet_type: Optional[str] = None) -> Dict[str, Any]:
        """지원되는 피부질환 목록 반환"""
        diseases = {
            "dog": {
                "binary": ["정상", "피부질환"],
                "multi_136": ["구진플라크", "무증상", "농포여드름"],
                "multi_456": ["과다색소침착", "결절종괴", "미란궤양"]
            },
            "cat": {
                "binary": ["정상", "피부질환"]
            }
        }
        
        if pet_type:
            return diseases.get(pet_type.lower(), {})
        return diseases
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'session') and self.session is not None:
            self.session.close()

# 서비스 인스턴스
_service_instance = None

def get_skin_disease_service():
    """싱글톤 서비스 반환"""
    global _service_instance
    if _service_instance is None:
        _service_instance = SkinDiseaseService()
    return _service_instance

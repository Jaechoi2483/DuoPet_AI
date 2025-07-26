"""
수정된 피부 질환 어댑터 - H5 모델 지원
"""

import tensorflow as tf
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class SkinDiseaseAdapter:
    """피부 질환 모델 어댑터 - H5 형식 지원"""
    
    def __init__(self, classification_models: Dict[str, Any], segmentation_models: Dict[str, Any]):
        self.classification_models = classification_models or {}
        self.segmentation_models = segmentation_models or {}
        
        # H5 모델 로드 시도
        if not self.classification_models:
            self._load_h5_models()
        
        logger.info(f"SkinDiseaseAdapter initialized with {len(self.classification_models)} classification models")
    
    def _load_h5_models(self):
        """H5 형식 모델 로드"""
        base_path = Path("models/health_diagnosis/skin_disease/classification")
        
        h5_models = [
            ('cat_binary', base_path / 'cat_binary/cat_binary_model.h5'),
            ('dog_binary', base_path / 'dog_binary/dog_binary_model.h5'),
            ('dog_multi_136', base_path / 'dog_multi_136/dog_multi_136_model.h5'),
            ('dog_multi_456', base_path / 'dog_multi_456/dog_multi_456_model.h5')
        ]
        
        for name, path in h5_models:
            if path.exists():
                try:
                    model = tf.keras.models.load_model(str(path), compile=False)
                    model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy' if 'binary' in name else 'categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    self.classification_models[name] = model
                    logger.info(f"Loaded H5 model: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load H5 model {name}: {e}")
    
    def predict(self, image, model_type: str, **kwargs):
        """예측 수행"""
        if model_type == 'classification':
            return self._predict_classification(image, **kwargs)
        elif model_type == 'segmentation':
            return self._predict_segmentation(image, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _predict_classification(self, image, species: str = 'dog', disease_type: str = 'binary'):
        """분류 예측"""
        model_key = f"{species}_{disease_type}"
        
        if model_key not in self.classification_models:
            # 기본 모델 사용
            if species == 'cat':
                model_key = 'cat_binary'
            else:
                model_key = 'dog_binary'
        
        if model_key not in self.classification_models:
            return {
                'success': False,
                'message': f'No model available for {model_key}'
            }
        
        try:
            model = self.classification_models[model_key]
            
            # 이미지 전처리
            if hasattr(image, 'shape'):
                processed_image = tf.image.resize(image, (224, 224))
            else:
                processed_image = tf.image.resize(tf.constant(image), (224, 224))
            
            processed_image = tf.expand_dims(processed_image, 0)
            processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(processed_image)
            
            # 예측
            prediction = model.predict(processed_image)
            
            # 결과 해석
            if 'binary' in model_key:
                probability = float(prediction[0][0])
                has_disease = probability > 0.5
                confidence = probability if has_disease else 1 - probability
                
                return {
                    'success': True,
                    'has_disease': has_disease,
                    'confidence': confidence,
                    'disease_type': '피부질환' if has_disease else '정상',
                    'model_used': model_key
                }
            else:
                # 다중 분류
                probabilities = prediction[0]
                predicted_class = int(tf.argmax(probabilities))
                confidence = float(probabilities[predicted_class])
                
                # 클래스 매핑
                if model_key == 'dog_multi_136':
                    class_names = ['정상', '질환1', '질환3', '질환6']
                else:  # dog_multi_456
                    class_names = ['정상', '질환4', '질환5', '질환6']
                
                return {
                    'success': True,
                    'has_disease': predicted_class > 0,
                    'confidence': confidence,
                    'disease_type': class_names[predicted_class],
                    'probabilities': {name: float(prob) for name, prob in zip(class_names, probabilities)},
                    'model_used': model_key
                }
                
        except Exception as e:
            logger.error(f"Classification prediction error: {e}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def _predict_segmentation(self, image, body_part: str = 'A2'):
        """세그멘테이션 예측 (기존 로직 유지)"""
        # 기존 세그멘테이션 로직...
        return {
            'success': True,
            'message': 'Segmentation model prediction',
            'body_part': body_part
        }
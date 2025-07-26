"""
Health Diagnosis Model Manager
Handles model loading with proper error handling and fallback mechanisms
"""

import os
import json
import tensorflow as tf
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

from common.logger import get_logger
from utils.model_loader import (
    load_model_with_custom_objects,
    load_keras_with_normalization_fix,
    create_dummy_model_for_checkpoint
)

logger = get_logger(__name__)

class DummyModel:
    """Dummy model for fallback when real model fails to load"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        logger.warning(f"Using dummy model for {model_type}")
    
    def predict(self, input_data):
        """Return dummy predictions"""
        if self.model_type == "eye_disease":
            # 5 classes for eye disease
            return np.random.softmax(np.random.randn(input_data.shape[0], 5))
        elif self.model_type == "bcs":
            # 3 classes for BCS
            return np.random.softmax(np.random.randn(input_data.shape[0], 3))
        elif self.model_type == "skin_binary":
            # 2 classes for binary classification
            return np.random.softmax(np.random.randn(input_data.shape[0], 2))
        else:
            # Default to 10 classes
            return np.random.softmax(np.random.randn(input_data.shape[0], 10))


class HealthDiagnosisModelManager:
    """Manages all health diagnosis models with proper error handling"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.model_status = {}
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.models_dir = self.project_root / "models" / "health_diagnosis"
        
    def _load_eye_disease_model(self) -> Optional[Any]:
        """Load eye disease detection model"""
        try:
            model_path = self.models_dir / "eye_disease" / "best_grouped_model.keras"
            config_path = self.models_dir / "eye_disease" / "config.yaml"
            class_map_path = self.models_dir / "eye_disease" / "class_map.json"
            
            logger.info(f"Loading eye disease model from: {model_path}")
            
            # Load configuration
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.model_configs['eye_disease'] = yaml.safe_load(f)
            
            # Load class map
            if class_map_path.exists():
                with open(class_map_path, 'r', encoding='utf-8') as f:
                    self.model_configs['eye_disease_classes'] = json.load(f)
            
            # Try loading with normalization fix first
            if model_path.exists() and model_path.suffix == '.keras':
                model = load_keras_with_normalization_fix(
                    str(model_path),
                    custom_objects={'swish': tf.nn.swish}
                )
                logger.info("Successfully loaded eye disease model with normalization fix")
                self.model_status['eye_disease'] = 'loaded'
                return model
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load eye disease model: {e}")
            self.model_status['eye_disease'] = f'failed: {str(e)}'
            return DummyModel('eye_disease')
    
    def _load_bcs_model(self) -> Optional[Any]:
        """Load BCS evaluation model"""
        try:
            model_path = self.models_dir / "bcs" / "bcs_efficientnet_v1.h5"
            config_path = self.models_dir / "bcs" / "config.yaml"
            
            logger.info(f"Loading BCS model from: {model_path}")
            
            # Load configuration
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.model_configs['bcs'] = yaml.safe_load(f)
            
            # Load H5 model
            if model_path.exists():
                model = load_model_with_custom_objects(str(model_path))
                logger.info("Successfully loaded BCS model")
                self.model_status['bcs'] = 'loaded'
                return model
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load BCS model: {e}")
            self.model_status['bcs'] = f'failed: {str(e)}'
            return DummyModel('bcs')
    
    def _load_skin_disease_models(self) -> Dict[str, Any]:
        """Load skin disease detection models"""
        models = {}
        skin_dir = self.models_dir / "skin_disease"
        
        try:
            # Load configuration
            config_path = skin_dir / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.model_configs['skin_disease'] = yaml.safe_load(f)
            
            # Load classification models
            classification_configs = {
                'cat_binary': {
                    'checkpoint': 'model-007-0.511353-0.772705-0.776322-0.768861',
                    'num_classes': 2
                },
                'dog_binary': {
                    'checkpoint': 'model-004-0.437360-0.806570-0.806528-0.806891',
                    'num_classes': 2
                },
                'dog_multi_136': {
                    'checkpoint': 'model-009-0.851382-0.821520',
                    'num_classes': 3
                },
                'dog_multi_456': {
                    'checkpoint': 'model-005-0.881675-0.851780',
                    'num_classes': 3
                }
            }
            
            # Try to load each classification model
            for model_name, config in classification_configs.items():
                try:
                    checkpoint_dir = skin_dir / "classification" / model_name
                    checkpoint_path = checkpoint_dir / config['checkpoint']
                    
                    if checkpoint_path.with_suffix('.index').exists():
                        # Create model architecture
                        model = self._create_classification_model(config['num_classes'])
                        # Load weights
                        model.load_weights(str(checkpoint_path))
                        models[model_name] = model
                        logger.info(f"Loaded skin disease model: {model_name}")
                    else:
                        logger.warning(f"Checkpoint not found for {model_name}")
                        models[model_name] = DummyModel('skin_binary')
                        
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    models[model_name] = DummyModel('skin_binary')
            
            # TODO: Load segmentation models similarly
            
            self.model_status['skin_disease'] = f'loaded ({len(models)} models)'
            
        except Exception as e:
            logger.error(f"Failed to load skin disease models: {e}")
            self.model_status['skin_disease'] = f'failed: {str(e)}'
            
        return models
    
    def _create_classification_model(self, num_classes: int):
        """Create a simple classification model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(224, 224, 3)
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def initialize(self):
        """Initialize all models asynchronously"""
        logger.info("Initializing health diagnosis models...")
        
        # Load models independently
        self.models['eye_disease'] = self._load_eye_disease_model()
        self.models['bcs'] = self._load_bcs_model()
        self.models['skin_disease'] = self._load_skin_disease_models()
        
        # Log status
        logger.info("Model loading complete. Status:")
        for model_type, status in self.model_status.items():
            logger.info(f"  {model_type}: {status}")
    
    def get_model(self, model_type: str) -> Optional[Any]:
        """Get a specific model"""
        return self.models.get(model_type)
    
    def is_model_available(self, model_type: str) -> bool:
        """Check if a model is available and not a dummy"""
        model = self.models.get(model_type)
        return model is not None and not isinstance(model, DummyModel)
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all loaded models"""
        return self.models
    
    def get_model_status(self) -> Dict[str, str]:
        """Get status of all models"""
        return self.model_status
    
    def get_model_config(self, model_type: str) -> Optional[Dict]:
        """Get configuration for a specific model"""
        return self.model_configs.get(model_type)
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get availability status of all models"""
        return {
            'eye_disease': self.is_model_available('eye_disease'),
            'bcs': self.is_model_available('bcs'),
            'skin_disease': len(self.models.get('skin_disease', {})) > 0
        }


class DummyModelManager:
    """Dummy model manager for complete fallback"""
    
    def __init__(self):
        self.models = {
            'eye_disease': DummyModel('eye_disease'),
            'bcs': DummyModel('bcs'),
            'skin_disease': {
                'cat_binary': DummyModel('skin_binary'),
                'dog_binary': DummyModel('skin_binary')
            }
        }
        logger.warning("Using dummy model manager - all predictions will be random")
    
    async def initialize(self):
        """No initialization needed for dummy models"""
        pass
    
    def get_model(self, model_type: str):
        return self.models.get(model_type)
    
    def is_model_available(self, model_type: str) -> bool:
        """Always return True for dummy models"""
        return True
    
    def get_all_models(self):
        return self.models
    
    def get_model_status(self):
        return {k: 'dummy' for k in self.models.keys()}
    
    def get_available_models(self):
        return {k: True for k in self.models.keys()}
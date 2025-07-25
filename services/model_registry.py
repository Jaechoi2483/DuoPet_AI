"""
Model Registry for DuoPet AI Health Diagnosis System

This module provides centralized model management with caching and version control.
"""

import os
import json
from enum import Enum
from typing import Dict, Any, Optional, Union
from pathlib import Path
import tensorflow as tf
import torch
import numpy as np
from datetime import datetime

from common.logger import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """Supported model types in the system"""
    EYE_DISEASE = "eye_disease"
    BCS = "body_condition_score"
    SKIN_DISEASE_CLASSIFICATION = "skin_disease_classification"
    SKIN_DISEASE_SEGMENTATION = "skin_disease_segmentation"
    FACE_RECOGNITION = "face_recognition"
    BEHAVIOR_ANALYSIS = "behavior_analysis"


class ModelFramework(Enum):
    """Supported deep learning frameworks"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    KERAS = "keras"


class ModelRegistry:
    """
    Centralized model registry for managing all AI models in the system.
    
    Features:
    - Lazy loading with caching
    - Framework-agnostic interface
    - Version management
    - Performance monitoring
    """
    
    def __init__(self, models_dir: str, config_path: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Base directory containing all model files
            config_path: Path to models configuration file
        """
        self.models_dir = Path(models_dir)
        self.models_cache: Dict[str, Any] = {}
        self.load_times: Dict[str, float] = {}
        
        # Determine device for PyTorch models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelRegistry initialized with device: {self.device}")
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._build_default_config()
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return self._build_default_config()
    
    def _build_default_config(self) -> Dict[str, Any]:
        """Build default configuration based on project structure"""
        return {
            ModelType.EYE_DISEASE.value: {
                "path": "health_diagnosis/eye_disease/best_grouped_model.keras",
                "framework": ModelFramework.KERAS.value,
                "input_shape": [224, 224, 3],
                "class_map_path": "health_diagnosis/eye_disease/class_map.json"
            },
            ModelType.BCS.value: {
                "path": "health_diagnosis/bcs/bcs_efficientnet_v1.h5",
                "framework": ModelFramework.TENSORFLOW.value,
                "input_shape": [224, 224, 3],
                "num_inputs": 13,
                "config_path": "health_diagnosis/bcs/config.yaml"
            },
            ModelType.SKIN_DISEASE_CLASSIFICATION.value: {
                "cat_binary": {
                    "path": "health_diagnosis/skin_disease/classification/cat_binary",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "model-007-0.511353-0.772705-0.776322-0.768861",
                    "input_shape": [224, 224, 3]
                },
                "dog_binary": {
                    "path": "health_diagnosis/skin_disease/classification/dog_binary",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "model-004-0.437360-0.806570-0.806528-0.806891",
                    "input_shape": [224, 224, 3]
                },
                "dog_multi_136": {
                    "path": "health_diagnosis/skin_disease/classification/dog_multi_136",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "model-009-0.851382-0.821520",
                    "input_shape": [224, 224, 3]
                },
                "dog_multi_456": {
                    "path": "health_diagnosis/skin_disease/classification/dog_multi_456",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "model-005-0.881675-0.851780",
                    "input_shape": [224, 224, 3]
                }
            },
            ModelType.SKIN_DISEASE_SEGMENTATION.value: {
                "cat_A2": {
                    "path": "health_diagnosis/skin_disease/segmentation/cat_A2",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "A2",
                    "input_shape": [512, 512, 3]
                },
                "dog_A1": {
                    "path": "health_diagnosis/skin_disease/segmentation/dog_A1",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "A1",
                    "input_shape": [512, 512, 3]
                },
                "dog_A2": {
                    "path": "health_diagnosis/skin_disease/segmentation/dog_A2",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "A2",
                    "input_shape": [512, 512, 3]
                },
                "dog_A3": {
                    "path": "health_diagnosis/skin_disease/segmentation/dog_A3",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "A3",
                    "input_shape": [512, 512, 3]
                },
                "dog_A4": {
                    "path": "health_diagnosis/skin_disease/segmentation/dog_A4",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "A4",
                    "input_shape": [512, 512, 3]
                },
                "dog_A5": {
                    "path": "health_diagnosis/skin_disease/segmentation/dog_A5",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "A5",
                    "input_shape": [512, 512, 3]
                },
                "dog_A6": {
                    "path": "health_diagnosis/skin_disease/segmentation/dog_A6",
                    "framework": ModelFramework.TENSORFLOW.value,
                    "model_type": "checkpoint",
                    "checkpoint_prefix": "A6",
                    "input_shape": [512, 512, 3]
                }
            }
        }
    
    def load_model(self, model_type: ModelType, sub_type: Optional[str] = None) -> Any:
        """
        Load a model with caching support.
        
        Args:
            model_type: Type of model to load
            sub_type: Sub-type for models with multiple variants (e.g., dog/cat)
            
        Returns:
            Loaded model instance
        """
        cache_key = f"{model_type.value}_{sub_type}" if sub_type else model_type.value
        
        # Return cached model if available
        if cache_key in self.models_cache:
            logger.info(f"Returning cached model: {cache_key}")
            return self.models_cache[cache_key]
        
        # Load model based on type
        start_time = datetime.now()
        
        try:
            if model_type == ModelType.EYE_DISEASE:
                model = self._load_eye_disease_model()
            elif model_type == ModelType.BCS:
                model = self._load_bcs_model()
            elif model_type == ModelType.SKIN_DISEASE_CLASSIFICATION:
                model = self._load_skin_disease_classification_model(sub_type)
            elif model_type == ModelType.SKIN_DISEASE_SEGMENTATION:
                model = self._load_skin_disease_segmentation_model(sub_type)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Cache the loaded model
            self.models_cache[cache_key] = model
            
            # Record load time
            load_time = (datetime.now() - start_time).total_seconds()
            self.load_times[cache_key] = load_time
            logger.info(f"Model {cache_key} loaded in {load_time:.2f} seconds")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {cache_key}: {e}")
            raise
    
    def _load_eye_disease_model(self) -> tf.keras.Model:
        """Load eye disease detection model"""
        config = self.config[ModelType.EYE_DISEASE.value]
        model_path = self.models_dir / config["path"]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Eye disease model not found at {model_path}")
        
        return tf.keras.models.load_model(str(model_path))
    
    def _load_bcs_model(self) -> tf.keras.Model:
        """Load body condition score model"""
        config = self.config[ModelType.BCS.value]
        model_path = self.models_dir / config["path"]
        
        if not model_path.exists():
            raise FileNotFoundError(f"BCS model not found at {model_path}")
        
        return tf.keras.models.load_model(str(model_path))
    
    def _load_skin_disease_classification_model(self, sub_type: str) -> tf.keras.Model:
        """Load skin disease classification model"""
        config = self.config[ModelType.SKIN_DISEASE_CLASSIFICATION.value][sub_type]
        model_path = self.models_dir / config["path"]
        
        if config.get("model_type") == "checkpoint":
            # Load from checkpoint
            checkpoint_path = model_path / config["checkpoint_prefix"]
            
            # Build model architecture (this would need to be implemented based on the actual model)
            # For now, we'll create a placeholder
            # In real implementation, you would need the model architecture definition
            logger.warning(f"Loading checkpoint model for {sub_type} - architecture needs to be defined")
            
            # This is a placeholder - actual implementation would load the specific architecture
            model = self._build_skin_disease_model_architecture(config["input_shape"])
            
            # Load weights from checkpoint
            model.load_weights(str(checkpoint_path))
            
            return model
        else:
            # Load saved model
            return tf.keras.models.load_model(str(model_path))
    
    def _load_skin_disease_segmentation_model(self, sub_type: str) -> tf.keras.Model:
        """Load skin disease segmentation model"""
        config = self.config[ModelType.SKIN_DISEASE_SEGMENTATION.value][sub_type]
        model_path = self.models_dir / config["path"]
        
        if config.get("model_type") == "checkpoint":
            # Similar to classification model loading
            checkpoint_path = model_path / config["checkpoint_prefix"]
            
            # Build segmentation model architecture
            logger.warning(f"Loading segmentation model for {sub_type} - architecture needs to be defined")
            
            # Placeholder for segmentation model
            model = self._build_segmentation_model_architecture(config["input_shape"])
            
            # Load weights
            model.load_weights(str(checkpoint_path))
            
            return model
        else:
            return tf.keras.models.load_model(str(model_path))
    
    def _build_skin_disease_model_architecture(self, input_shape: list) -> tf.keras.Model:
        """
        Build skin disease classification model architecture.
        This is a placeholder - actual architecture should match the training code.
        """
        # Placeholder implementation
        inputs = tf.keras.Input(shape=input_shape)
        # Add actual model layers here based on the training architecture
        outputs = tf.keras.layers.Dense(2, activation='softmax')(inputs)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def _build_segmentation_model_architecture(self, input_shape: list) -> tf.keras.Model:
        """
        Build segmentation model architecture.
        This is a placeholder - actual architecture should match the training code.
        """
        # Placeholder implementation
        inputs = tf.keras.Input(shape=input_shape)
        # Add actual segmentation model layers here
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(inputs)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def get_model_config(self, model_type: ModelType, sub_type: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        if sub_type and model_type.value in self.config:
            config_data = self.config[model_type.value]
            if isinstance(config_data, dict) and sub_type in config_data:
                return config_data[sub_type]
        return self.config.get(model_type.value, {})
    
    def clear_cache(self):
        """Clear all cached models to free memory"""
        self.models_cache.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        return {
            "cached_models": list(self.models_cache.keys()),
            "cache_size": len(self.models_cache),
            "load_times": self.load_times
        }


# Singleton instance
_model_registry_instance: Optional[ModelRegistry] = None


def get_model_registry(models_dir: Optional[str] = None) -> ModelRegistry:
    """
    Get or create the singleton ModelRegistry instance.
    
    Args:
        models_dir: Base directory for models (required on first call)
        
    Returns:
        ModelRegistry instance
    """
    global _model_registry_instance
    
    if _model_registry_instance is None:
        if models_dir is None:
            # Try to use default path
            from pathlib import Path
            default_path = Path(__file__).parent.parent / "models"
            if default_path.exists():
                models_dir = str(default_path)
            else:
                raise ValueError("models_dir must be provided on first initialization")
        
        _model_registry_instance = ModelRegistry(models_dir)
    
    return _model_registry_instance
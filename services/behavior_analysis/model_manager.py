"""
Model Manager for Behavior Analysis

This module provides memory-efficient model loading and management
with retry logic and error handling.
"""

import os
import gc
import torch
import time
from typing import Dict, Optional, Any
from pathlib import Path
import threading
from common.logger import get_logger
from common.config import get_model_path
from common.exceptions import ModelNotLoadedError

logger = get_logger(__name__)


class ModelManager:
    """
    Manages AI model loading with memory optimization and retry logic.
    Implements lazy loading and automatic cleanup.
    """
    
    def __init__(self):
        self._models = {}
        self._load_lock = threading.Lock()
        self._retry_count = 3
        self._retry_delay = 1.0  # seconds
        
    def get_model(self, model_key: str, loader_func, *args, **kwargs) -> Any:
        """
        Get a model with lazy loading and automatic retry.
        
        Args:
            model_key: Unique identifier for the model
            loader_func: Function to load the model
            *args, **kwargs: Arguments for loader_func
            
        Returns:
            Loaded model instance
        """
        with self._load_lock:
            # Check if model is already loaded
            if model_key in self._models and self._models[model_key] is not None:
                return self._models[model_key]
            
            # Try to load the model with retry logic
            last_error = None
            for attempt in range(self._retry_count):
                try:
                    logger.info(f"Loading model {model_key} (attempt {attempt + 1}/{self._retry_count})")
                    
                    # Clean up memory before loading
                    self._cleanup_memory()
                    
                    # Load the model
                    model = loader_func(*args, **kwargs)
                    
                    if model is not None:
                        self._models[model_key] = model
                        logger.info(f"Successfully loaded model {model_key}")
                        return model
                        
                except Exception as e:
                    last_error = e
                    logger.error(f"Failed to load model {model_key} (attempt {attempt + 1}): {str(e)}")
                    
                    # Cleanup failed model
                    if model_key in self._models:
                        self._unload_model(model_key)
                    
                    # Wait before retry
                    if attempt < self._retry_count - 1:
                        time.sleep(self._retry_delay * (attempt + 1))
            
            # All attempts failed
            raise ModelNotLoadedError(f"Failed to load {model_key} after {self._retry_count} attempts: {str(last_error)}")
    
    def unload_model(self, model_key: str):
        """
        Unload a specific model from memory.
        
        Args:
            model_key: Model identifier
        """
        with self._load_lock:
            self._unload_model(model_key)
    
    def _unload_model(self, model_key: str):
        """Internal method to unload model without lock."""
        if model_key in self._models:
            model = self._models[model_key]
            
            # Handle different model types
            if hasattr(model, 'cpu'):
                # Move PyTorch models to CPU first
                try:
                    model.cpu()
                except:
                    pass
            
            # Delete the model
            del self._models[model_key]
            logger.info(f"Unloaded model {model_key}")
    
    def unload_all(self):
        """Unload all models from memory."""
        with self._load_lock:
            model_keys = list(self._models.keys())
            for key in model_keys:
                self._unload_model(key)
            
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Clean up GPU and system memory."""
        # PyTorch CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Python garbage collection
        gc.collect()
        
        # Log memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            logger.debug(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Get status of all models."""
        with self._load_lock:
            return {key: (model is not None) for key, model in self._models.items()}
    
    def reset(self):
        """Reset the model manager, unloading all models."""
        logger.info("Resetting model manager")
        self.unload_all()
        self._models.clear()


# Global model manager instance
model_manager = ModelManager()
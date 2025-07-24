"""
Error Handler for Behavior Analysis

This module provides advanced error handling and recovery mechanisms
for model loading and inference errors.
"""

import time
import functools
from typing import Callable, Any, Optional, Dict
from common.logger import get_logger
from common.exceptions import ModelNotLoadedError, ModelInferenceError

logger = get_logger(__name__)


class ErrorHandler:
    """Handles errors with intelligent retry and recovery strategies."""
    
    def __init__(self):
        self.error_counts = {}
        self.last_error_time = {}
        self.recovery_strategies = {
            "cuda out of memory": self._handle_cuda_oom,
            "model not loaded": self._handle_model_not_loaded,
            "connection error": self._handle_connection_error,
            "file not found": self._handle_file_not_found
        }
    
    def with_retry(self, max_retries: int = 3, delay: float = 1.0, 
                   backoff: float = 2.0, exceptions=(Exception,)):
        """
        Decorator for adding retry logic to functions.
        
        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff: Multiplier for delay after each retry
            exceptions: Tuple of exceptions to catch
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries + 1):
                    try:
                        # Clear any previous errors after successful retry
                        if attempt > 0:
                            logger.info(f"Retry {attempt} succeeded for {func.__name__}")
                        return func(*args, **kwargs)
                        
                    except exceptions as e:
                        last_exception = e
                        error_key = f"{func.__name__}_{type(e).__name__}"
                        
                        # Track error occurrences
                        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
                        self.last_error_time[error_key] = time.time()
                        
                        if attempt < max_retries:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: "
                                f"{type(e).__name__}: {str(e)}. Retrying in {current_delay:.1f}s..."
                            )
                            
                            # Apply recovery strategy if available
                            self._apply_recovery_strategy(str(e).lower())
                            
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(
                                f"All {max_retries + 1} attempts failed for {func.__name__}: "
                                f"{type(e).__name__}: {str(e)}"
                            )
                
                # All retries failed
                raise last_exception
            
            return wrapper
        return decorator
    
    def _apply_recovery_strategy(self, error_msg: str):
        """Apply recovery strategy based on error message."""
        for pattern, strategy in self.recovery_strategies.items():
            if pattern in error_msg:
                logger.info(f"Applying recovery strategy for: {pattern}")
                strategy()
                break
    
    def _handle_cuda_oom(self):
        """Handle CUDA out of memory errors."""
        try:
            import torch
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared CUDA cache")
                
                # Force garbage collection
                import gc
                gc.collect()
                logger.info("Forced garbage collection")
                
                # Wait a bit for memory to be released
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"Failed to handle CUDA OOM: {str(e)}")
    
    def _handle_model_not_loaded(self):
        """Handle model not loaded errors."""
        logger.info("Model not loaded - will attempt to load on next retry")
        # The retry mechanism will handle the actual loading
    
    def _handle_connection_error(self):
        """Handle connection errors."""
        logger.info("Connection error - waiting for network recovery")
        time.sleep(2.0)  # Wait longer for network issues
    
    def _handle_file_not_found(self):
        """Handle file not found errors."""
        logger.info("File not found - checking model paths")
        # Log available model files for debugging
        try:
            from common.config import get_model_path
            import os
            base_path = get_model_path("behavior_analysis")
            if os.path.exists(base_path):
                logger.info(f"Available models in {base_path}:")
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith(('.pt', '.pth', '.onnx')):
                            logger.info(f"  - {os.path.join(root, file)}")
        except Exception as e:
            logger.error(f"Failed to list model files: {str(e)}")
    
    def get_error_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get error statistics."""
        stats = {}
        current_time = time.time()
        
        for error_key, count in self.error_counts.items():
            last_time = self.last_error_time.get(error_key, 0)
            time_since_last = current_time - last_time if last_time > 0 else -1
            
            stats[error_key] = {
                "count": count,
                "last_occurrence": last_time,
                "time_since_last": time_since_last
            }
        
        return stats
    
    def reset_error_stats(self):
        """Reset error statistics."""
        self.error_counts.clear()
        self.last_error_time.clear()
        logger.info("Error statistics reset")


# Global error handler instance
error_handler = ErrorHandler()
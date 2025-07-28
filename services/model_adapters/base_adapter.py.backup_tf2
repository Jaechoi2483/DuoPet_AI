"""
Base Model Adapter for standardized model interface

This module provides an abstract base class for all model adapters
to ensure consistent interface across different model types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from datetime import datetime

from common.logger import get_logger

logger = get_logger(__name__)


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    All model adapters must implement:
    - preprocess: Prepare input data for the model
    - predict: Run model inference
    - postprocess: Convert raw predictions to standardized format
    """
    
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model adapter.
        
        Args:
            model: The loaded model instance
            config: Optional configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self._last_prediction_time: Optional[float] = None
        
    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input data for the model.
        
        Args:
            input_data: Raw input data (image, text, etc.)
            
        Returns:
            Preprocessed data ready for model input
        """
        pass
    
    @abstractmethod
    def predict(self, processed_data: Any) -> Any:
        """
        Run prediction with the model.
        
        Args:
            processed_data: Preprocessed input data
            
        Returns:
            Raw model predictions
        """
        pass
    
    @abstractmethod
    def postprocess(self, prediction: Any) -> Dict[str, Any]:
        """
        Convert raw predictions to standardized output format.
        
        Args:
            prediction: Raw model output
            
        Returns:
            Standardized dictionary with prediction results
        """
        pass
    
    def __call__(self, input_data: Any) -> Dict[str, Any]:
        """
        End-to-end prediction pipeline.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Standardized prediction results with timing information
        """
        start_time = datetime.now()
        
        try:
            # Preprocess
            processed = self.preprocess(input_data)
            
            # Predict
            prediction = self.predict(processed)
            
            # Postprocess
            result = self.postprocess(prediction)
            
            # Add timing information
            self._last_prediction_time = (datetime.now() - start_time).total_seconds()
            result['prediction_time'] = self._last_prediction_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error in model prediction pipeline: {e}")
            raise
    
    def get_last_prediction_time(self) -> Optional[float]:
        """Get the execution time of the last prediction in seconds"""
        return self._last_prediction_time
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data before processing.
        Override this method in subclasses for specific validation.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        return input_data is not None
    
    @staticmethod
    def normalize_confidence_scores(scores: Union[List[float], np.ndarray], 
                                   threshold: float = 0.0) -> Dict[str, float]:
        """
        Normalize confidence scores to ensure they sum to 1.
        
        Args:
            scores: Raw confidence scores
            threshold: Minimum threshold for including a score
            
        Returns:
            Dictionary of normalized scores
        """
        scores_array = np.array(scores)
        
        # Filter by threshold
        mask = scores_array >= threshold
        filtered_scores = scores_array[mask]
        
        # Normalize
        if filtered_scores.sum() > 0:
            normalized = filtered_scores / filtered_scores.sum()
        else:
            normalized = filtered_scores
            
        return {
            f"class_{i}": float(score) 
            for i, score in enumerate(normalized) 
            if mask[i]
        }
    
    @staticmethod
    def format_detection_result(
        class_name: str,
        confidence: float,
        bbox: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format a detection result in standardized format.
        
        Args:
            class_name: Detected class name
            confidence: Confidence score (0-1)
            bbox: Optional bounding box [x, y, width, height]
            metadata: Optional additional metadata
            
        Returns:
            Standardized detection result
        """
        result = {
            "class": class_name,
            "confidence": float(confidence)
        }
        
        if bbox:
            result["bbox"] = {
                "x": float(bbox[0]),
                "y": float(bbox[1]),
                "width": float(bbox[2]),
                "height": float(bbox[3])
            }
            
        if metadata:
            result["metadata"] = metadata
            
        return result
    
    @staticmethod
    def calculate_severity_score(confidence: float, 
                                thresholds: Dict[str, float] = None) -> str:
        """
        Calculate severity level based on confidence score.
        
        Args:
            confidence: Confidence score (0-1)
            thresholds: Custom thresholds for severity levels
            
        Returns:
            Severity level string
        """
        if thresholds is None:
            thresholds = {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.5,
                "low": 0.3,
                "minimal": 0.0
            }
        
        for severity, threshold in sorted(thresholds.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True):
            if confidence >= threshold:
                return severity
                
        return "minimal"
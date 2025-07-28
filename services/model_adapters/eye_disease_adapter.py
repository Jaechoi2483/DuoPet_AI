"""
Eye Disease Model Adapter

Adapter for eye disease detection model following the standardized interface.
"""

import json
import numpy as np
from PIL import Image
import tensorflow as tf

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

tf.config.run_functions_eagerly(True)
from typing import Dict, Any, Union
from pathlib import Path

from .base_adapter import ModelAdapter
from common.logger import get_logger
import io

logger = get_logger(__name__)


class EyeDiseaseAdapter(ModelAdapter):
    """
    Adapter for eye disease detection model.
    
    Handles preprocessing, prediction, and postprocessing for eye disease classification.
    """
    
    def __init__(self, model: tf.keras.Model, config: Dict[str, Any]):
        """
        Initialize the eye disease adapter.
        
        Args:
            model: Loaded Keras model
            config: Configuration including class_map_path
        """
        super().__init__(model, config)
        
        # Load class mapping
        class_map_path = config.get('class_map_path')
        if class_map_path and Path(class_map_path).exists():
            with open(class_map_path, 'r', encoding='utf-8') as f:
                self.class_map = json.load(f)
        else:
            logger.warning("Class map not found, using numeric indices")
            self.class_map = {}
            
        # Get input shape from model
        self.input_shape = self.model.input_shape[1:3]  # (height, width)
        
        logger.info(f"EyeDiseaseAdapter initialized with input shape: {self.input_shape}")
    
    def preprocess(self, input_data: Any) -> np.ndarray:
        """
        Preprocess image for eye disease model.
        
        Args:
            input_data: PIL Image, numpy array, or file-like object
            
        Returns:
            Preprocessed image array ready for model input
        """
        # 이미지 열기
        if hasattr(input_data, 'file'):
            # FastAPI UploadFile
            input_data.file.seek(0)
            image = Image.open(input_data.file)
        elif hasattr(input_data, 'read'):
            # File-like object
            content = input_data.read()
            if hasattr(input_data, 'seek'):
                input_data.seek(0)
            image = Image.open(io.BytesIO(content))
        elif isinstance(input_data, np.ndarray):
            image = Image.fromarray(input_data)
        elif isinstance(input_data, Image.Image):
            image = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # RGB로 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 리사이즈
        image = image.resize(self.input_shape)
        
        # NumPy 배열로 변환
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, processed_data: np.ndarray) -> np.ndarray:
        """
        Run prediction with the eye disease model.
        
        Args:
            processed_data: Preprocessed image array
            
        Returns:
            Model predictions (probability array)
        """
        predictions = self.model.predict(processed_data, verbose=0)
        return predictions
    
    def postprocess(self, prediction: np.ndarray) -> Dict[str, Any]:
        """
        Convert model predictions to standardized format.
        
        Args:
            prediction: Raw model output (probability array)
            
        Returns:
            Standardized prediction results
        """
        # Get probabilities for the first (and only) image
        probabilities = prediction[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        # Map class index to disease name
        if self.class_map:
            disease_name = self.class_map.get(str(predicted_class_idx), f"Unknown_{predicted_class_idx}")
        else:
            disease_name = f"Class_{predicted_class_idx}"
        
        # Create probability distribution for all classes
        all_probabilities = {}
        for idx, prob in enumerate(probabilities):
            class_name = self.class_map.get(str(idx), f"Class_{idx}") if self.class_map else f"Class_{idx}"
            all_probabilities[class_name] = float(prob)
        
        # Calculate severity based on disease type and confidence
        severity = self._calculate_eye_disease_severity(disease_name, confidence)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(disease_name, confidence, severity)
        
        result = {
            "disease": disease_name,
            "confidence": confidence,
            "severity": severity,
            "probabilities": all_probabilities,
            "top_3_predictions": self._get_top_predictions(all_probabilities, 3),
            "recommendations": recommendations,
            "requires_vet_visit": self._requires_vet_visit(disease_name, confidence)
        }
        
        return result
    
    def _calculate_eye_disease_severity(self, disease: str, confidence: float) -> str:
        """
        Calculate severity level based on disease type and confidence.
        
        Args:
            disease: Detected disease name
            confidence: Confidence score
            
        Returns:
            Severity level
        """
        # Disease-specific severity thresholds
        critical_diseases = ["glaucoma", "각막궤양", "corneal_ulcer"]
        high_severity_diseases = ["cataract", "백내장", "uveitis", "포도막염"]
        
        disease_lower = disease.lower()
        
        if any(critical in disease_lower for critical in critical_diseases):
            if confidence > 0.7:
                return "critical"
            elif confidence > 0.5:
                return "high"
            else:
                return "medium"
        elif any(high_sev in disease_lower for high_sev in high_severity_diseases):
            if confidence > 0.8:
                return "high"
            elif confidence > 0.6:
                return "medium"
            else:
                return "low"
        else:
            # For other diseases or healthy status
            if "healthy" in disease_lower or "정상" in disease_lower:
                return "none"
            else:
                return self.calculate_severity_score(confidence)
    
    def _generate_recommendations(self, disease: str, confidence: float, severity: str) -> list:
        """
        Generate recommendations based on diagnosis.
        
        Args:
            disease: Detected disease
            confidence: Confidence score
            severity: Severity level
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if severity == "critical":
            recommendations.append("즉시 동물병원 방문이 필요합니다")
            recommendations.append("안과 전문의 진료를 권장합니다")
        elif severity == "high":
            recommendations.append("가능한 빨리 동물병원 방문을 권장합니다")
            recommendations.append("증상이 악화되지 않도록 주의 깊게 관찰해주세요")
        elif severity == "medium":
            recommendations.append("정기적인 관찰이 필요합니다")
            recommendations.append("증상이 지속되면 동물병원 방문을 고려해주세요")
        elif severity == "low":
            recommendations.append("현재는 경미한 상태입니다")
            recommendations.append("일주일 후 재검사를 권장합니다")
        else:
            recommendations.append("정기적인 건강 검진을 유지해주세요")
            
        # Add disease-specific recommendations
        disease_lower = disease.lower()
        if "cataract" in disease_lower or "백내장" in disease_lower:
            recommendations.append("밝은 빛에 노출을 줄여주세요")
            recommendations.append("항산화제가 풍부한 식단을 고려해보세요")
        elif "conjunctivitis" in disease_lower or "결막염" in disease_lower:
            recommendations.append("눈 주변을 깨끗하게 유지해주세요")
            recommendations.append("다른 반려동물과의 접촉을 제한해주세요")
            
        return recommendations
    
    def _get_top_predictions(self, probabilities: Dict[str, float], top_k: int = 3) -> list:
        """
        Get top K predictions sorted by probability.
        
        Args:
            probabilities: Dictionary of class probabilities
            top_k: Number of top predictions to return
            
        Returns:
            List of top predictions
        """
        sorted_predictions = sorted(
            probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [
            {"disease": disease, "confidence": conf}
            for disease, conf in sorted_predictions
        ]
    
    def _requires_vet_visit(self, disease: str, confidence: float) -> bool:
        """
        Determine if vet visit is required based on diagnosis.
        
        Args:
            disease: Detected disease
            confidence: Confidence score
            
        Returns:
            True if vet visit is recommended
        """
        # Diseases that always require vet attention
        urgent_diseases = [
            "glaucoma", "각막궤양", "corneal_ulcer", 
            "uveitis", "포도막염", "retinal_detachment", "망막박리"
        ]
        
        disease_lower = disease.lower()
        
        # Check urgent diseases
        if any(urgent in disease_lower for urgent in urgent_diseases):
            return confidence > 0.5
        
        # Check other conditions
        if ("healthy" not in disease_lower and "정상" not in disease_lower 
            and confidence > 0.7):
            return True
            
        return False
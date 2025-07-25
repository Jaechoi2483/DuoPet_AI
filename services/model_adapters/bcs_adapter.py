"""
Body Condition Score (BCS) Model Adapter

Adapter for BCS assessment model that processes multiple images to evaluate pet body condition.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Dict, Any, List, Union
import yaml
from pathlib import Path

from .base_adapter import ModelAdapter
from common.logger import get_logger

logger = get_logger(__name__)


class BCSAdapter(ModelAdapter):
    """
    Adapter for Body Condition Score (BCS) model.
    
    This model requires 13 images from different angles to assess body condition.
    It classifies pets into three categories: underweight, normal, overweight.
    """
    
    # BCS score mapping
    BCS_CLASSES = {
        0: {"label": "저체중", "range": "BCS 1-3", "english": "Underweight"},
        1: {"label": "정상", "range": "BCS 4-6", "english": "Normal"},
        2: {"label": "과체중", "range": "BCS 7-9", "english": "Overweight"}
    }
    
    # Required image views for comprehensive BCS assessment
    REQUIRED_VIEWS = [
        "front", "back", "left_side", "right_side", "top",
        "front_left", "front_right", "back_left", "back_right",
        "abdomen", "chest", "spine", "overall"
    ]
    
    def __init__(self, model: tf.keras.Model, config: Dict[str, Any]):
        """
        Initialize the BCS adapter.
        
        Args:
            model: Loaded Keras model
            config: Configuration including model specifications
        """
        super().__init__(model, config)
        
        # Load additional config if yaml path is provided
        config_path = config.get('config_path')
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                self.config.update(yaml_config)
        
        # Model specifications
        self.num_required_images = self.config.get('num_inputs', 13)
        self.input_shape = tuple(self.config.get('input_shape', [224, 224, 3])[:2])
        
        # Preprocessing parameters
        self.normalize_mean = self.config.get('preprocessing', {}).get('mean', [0.485, 0.456, 0.406])
        self.normalize_std = self.config.get('preprocessing', {}).get('std', [0.229, 0.224, 0.225])
        
        logger.info(f"BCSAdapter initialized - requires {self.num_required_images} images")
    
    def preprocess(self, input_data: Union[List[Any], Any]) -> np.ndarray:
        """
        Preprocess multiple images for BCS model.
        
        Args:
            input_data: List of images (PIL, numpy, or file-like objects)
                       or single image that will be duplicated
            
        Returns:
            Preprocessed image array ready for model input
        """
        # Handle single image by duplicating it
        if not isinstance(input_data, list):
            logger.warning(f"Single image provided, duplicating to fill {self.num_required_images} slots")
            input_data = [input_data] * self.num_required_images
        
        # Validate number of images
        if len(input_data) < self.num_required_images:
            logger.warning(f"Only {len(input_data)} images provided, padding with duplicates")
            # Pad with copies of existing images
            while len(input_data) < self.num_required_images:
                input_data.append(input_data[len(input_data) % len(input_data)])
        elif len(input_data) > self.num_required_images:
            logger.warning(f"Too many images ({len(input_data)}), using first {self.num_required_images}")
            input_data = input_data[:self.num_required_images]
        
        # Process each image
        processed_images = []
        for idx, img_data in enumerate(input_data):
            processed_img = self._preprocess_single_image(img_data, idx)
            processed_images.append(processed_img)
        
        # Stack all images
        return np.array(processed_images)
    
    def _preprocess_single_image(self, img_data: Any, index: int) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            img_data: Image data
            index: Image index for logging
            
        Returns:
            Preprocessed image array
        """
        try:
            # Convert to PIL Image
            if hasattr(img_data, 'file'):
                image = Image.open(img_data.file).convert('RGB')
            elif isinstance(img_data, np.ndarray):
                image = Image.fromarray(img_data).convert('RGB')
            elif isinstance(img_data, Image.Image):
                image = img_data.convert('RGB')
            else:
                raise ValueError(f"Unsupported input type: {type(img_data)}")
            
            # Resize
            image = image.resize(self.input_shape)
            
            # Convert to array
            img_array = np.array(image, dtype=np.float32)
            
            # Normalize using ImageNet statistics
            img_array = img_array / 255.0
            for i in range(3):
                img_array[:, :, i] = (img_array[:, :, i] - self.normalize_mean[i]) / self.normalize_std[i]
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image {index}: {e}")
            # Return a blank image on error
            return np.zeros((*self.input_shape, 3), dtype=np.float32)
    
    def predict(self, processed_data: np.ndarray) -> np.ndarray:
        """
        Run prediction with the BCS model.
        
        Args:
            processed_data: Preprocessed images array
            
        Returns:
            Model predictions
        """
        # The model expects a batch dimension
        if len(processed_data.shape) == 4:
            # Already has batch dimension
            predictions = self.model.predict(processed_data, verbose=0)
        else:
            # Add batch dimension
            batch_data = np.expand_dims(processed_data, axis=0)
            predictions = self.model.predict(batch_data, verbose=0)
            
        return predictions
    
    def postprocess(self, prediction: np.ndarray) -> Dict[str, Any]:
        """
        Convert model predictions to standardized BCS assessment.
        
        Args:
            prediction: Raw model output
            
        Returns:
            Standardized BCS assessment results
        """
        # Get probabilities
        if len(prediction.shape) > 1:
            probabilities = prediction[0]  # First batch item
        else:
            probabilities = prediction
        
        # Get predicted class
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        
        # Get BCS information
        bcs_info = self.BCS_CLASSES[predicted_class]
        
        # Calculate detailed BCS score (1-9 scale)
        detailed_score = self._calculate_detailed_bcs_score(predicted_class, probabilities)
        
        # Generate health insights
        health_insights = self._generate_health_insights(detailed_score, confidence)
        
        # Generate recommendations
        recommendations = self._generate_bcs_recommendations(detailed_score, bcs_info['label'])
        
        result = {
            "bcs_category": bcs_info['label'],
            "bcs_category_english": bcs_info['english'],
            "bcs_range": bcs_info['range'],
            "bcs_score": detailed_score,
            "confidence": confidence,
            "probabilities": {
                "저체중": float(probabilities[0]),
                "정상": float(probabilities[1]),
                "과체중": float(probabilities[2])
            },
            "health_insights": health_insights,
            "recommendations": recommendations,
            "ideal_weight_guidance": self._get_ideal_weight_guidance(detailed_score),
            "requires_vet_consultation": self._requires_vet_consultation(detailed_score)
        }
        
        return result
    
    def _calculate_detailed_bcs_score(self, predicted_class: int, 
                                     probabilities: np.ndarray) -> int:
        """
        Calculate detailed BCS score (1-9) from class probabilities.
        
        Args:
            predicted_class: Predicted class (0, 1, or 2)
            probabilities: Class probabilities
            
        Returns:
            BCS score (1-9)
        """
        # Base scores for each class
        base_scores = {0: 2, 1: 5, 2: 8}  # Underweight: 2, Normal: 5, Overweight: 8
        
        base_score = base_scores[predicted_class]
        
        # Adjust based on confidence and neighboring class probabilities
        if predicted_class == 0:  # Underweight
            # Can be 1, 2, or 3
            if probabilities[1] > 0.3:  # Some normal characteristics
                return 3
            elif probabilities[0] > 0.8:  # Strong underweight
                return 1
            else:
                return 2
        elif predicted_class == 1:  # Normal
            # Can be 4, 5, or 6
            if probabilities[0] > 0.3:  # Lean normal
                return 4
            elif probabilities[2] > 0.3:  # Heavier normal
                return 6
            else:
                return 5
        else:  # Overweight
            # Can be 7, 8, or 9
            if probabilities[1] > 0.3:  # Mild overweight
                return 7
            elif probabilities[2] > 0.8:  # Severe overweight
                return 9
            else:
                return 8
    
    def _generate_health_insights(self, bcs_score: int, confidence: float) -> List[str]:
        """
        Generate health insights based on BCS score.
        
        Args:
            bcs_score: BCS score (1-9)
            confidence: Model confidence
            
        Returns:
            List of health insights
        """
        insights = []
        
        if bcs_score <= 3:
            insights.append("반려동물이 저체중 상태입니다")
            insights.append("갈비뼈, 척추, 골반뼈가 쉽게 만져집니다")
            insights.append("근육량이 부족할 수 있습니다")
        elif bcs_score <= 6:
            insights.append("반려동물이 이상적인 체중 범위에 있습니다")
            insights.append("갈비뼈를 적당한 압력으로 만질 수 있습니다")
            insights.append("허리선이 명확하게 보입니다")
        else:
            insights.append("반려동물이 과체중 상태입니다")
            insights.append("갈비뼈를 만지기 어렵습니다")
            insights.append("복부가 늘어져 있을 수 있습니다")
        
        if confidence < 0.7:
            insights.append("더 정확한 평가를 위해 다양한 각도의 사진을 제공해주세요")
        
        return insights
    
    def _generate_bcs_recommendations(self, bcs_score: int, category: str) -> List[str]:
        """
        Generate recommendations based on BCS score.
        
        Args:
            bcs_score: BCS score (1-9)
            category: BCS category (저체중/정상/과체중)
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if bcs_score <= 3:
            recommendations.extend([
                "수의사와 상담하여 저체중의 원인을 파악하세요",
                "고품질 사료로 변경하거나 급여량을 증가시키세요",
                "기생충 검사를 받아보세요",
                "하루 급여 횟수를 늘려 소화를 돕도록 하세요"
            ])
        elif bcs_score <= 6:
            recommendations.extend([
                "현재 식단과 운동량을 유지하세요",
                "정기적인 체중 모니터링을 계속하세요",
                "간식은 일일 칼로리의 10% 이내로 제한하세요"
            ])
        else:
            recommendations.extend([
                "수의사와 체중 감량 계획을 수립하세요",
                "저칼로리 사료로 변경을 고려하세요",
                "일일 급여량을 10-20% 감소시키세요",
                "규칙적인 운동 시간을 늘리세요",
                "간식을 제한하거나 저칼로리 간식으로 대체하세요"
            ])
        
        return recommendations
    
    def _get_ideal_weight_guidance(self, bcs_score: int) -> Dict[str, Any]:
        """
        Get ideal weight guidance based on BCS score.
        
        Args:
            bcs_score: BCS score (1-9)
            
        Returns:
            Weight adjustment guidance
        """
        if bcs_score <= 3:
            weight_change = "10-20% 증량 필요"
            timeframe = "2-3개월"
        elif bcs_score <= 6:
            weight_change = "현재 체중 유지"
            timeframe = "지속적 관리"
        else:
            weight_change = "10-20% 감량 필요"
            timeframe = "3-6개월"
        
        return {
            "weight_change_needed": weight_change,
            "recommended_timeframe": timeframe,
            "monitoring_frequency": "2주마다 체중 측정"
        }
    
    def _requires_vet_consultation(self, bcs_score: int) -> bool:
        """
        Determine if veterinary consultation is needed.
        
        Args:
            bcs_score: BCS score (1-9)
            
        Returns:
            True if vet consultation is recommended
        """
        # Extreme scores require vet consultation
        return bcs_score <= 2 or bcs_score >= 8
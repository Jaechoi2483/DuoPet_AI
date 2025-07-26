"""
Body Condition Score (BCS) Service

Service for assessing pet body condition using multi-view image analysis.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import UploadFile

from services.model_registry import ModelRegistry, ModelType, get_model_registry
from services.model_adapters.bcs_adapter import BCSAdapter
from common.logger import get_logger

logger = get_logger(__name__)


class BCSService:
    """
    Service for Body Condition Score assessment.
    
    This service uses a multi-head EfficientNet model to evaluate pet body condition
    from multiple angles and provide comprehensive health assessment.
    """
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize BCS service.
        
        Args:
            model_path: Optional custom model path
            config_path: Optional custom config path
        """
        try:
            # Get model registry
            if model_path:
                # Use custom path
                self.model = self._load_custom_model(model_path)
                config = {"config_path": config_path} if config_path else {}
            else:
                # Use registry
                registry = get_model_registry()
                self.model = registry.load_model(ModelType.BCS)
                config = registry.get_model_config(ModelType.BCS)
                
            # Initialize adapter
            self.adapter = BCSAdapter(self.model, config)
            
            logger.info("BCSService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BCSService: {e}")
            raise
    
    def _load_custom_model(self, model_path: str):
        """Load model from custom path"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.model_loader import load_model_with_custom_objects
        return load_model_with_custom_objects(model_path)
    
    async def assess_body_condition(
        self, 
        images: List[UploadFile],
        pet_type: str = "dog",
        pet_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess body condition from multiple images.
        
        Args:
            images: List of uploaded images (ideally 13 from different angles)
            pet_type: Type of pet (dog/cat)
            pet_info: Optional pet information (breed, age, etc.)
            
        Returns:
            BCS assessment results
        """
        try:
            # Validate inputs
            if not images:
                raise ValueError("At least one image is required for BCS assessment")
            
            # Log assessment request
            logger.info(f"BCS assessment request for {pet_type} with {len(images)} images")
            
            # Validate file formats
            valid_images = []
            for img in images:
                if self._validate_image_format(img):
                    valid_images.append(img)
                else:
                    logger.warning(f"Skipping invalid image: {img.filename}")
            
            if not valid_images:
                raise ValueError("No valid images provided")
            
            # Run assessment
            result = self.adapter(valid_images)
            
            # Enhance results with pet-specific information
            if pet_info:
                result = self._enhance_with_pet_info(result, pet_info, pet_type)
            
            # Add metadata
            result['metadata'] = {
                'images_analyzed': len(valid_images),
                'images_required': self.adapter.num_required_images,
                'pet_type': pet_type,
                'assessment_quality': self._calculate_assessment_quality(len(valid_images))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in BCS assessment: {e}")
            raise
    
    def _validate_image_format(self, image: UploadFile) -> bool:
        """Validate image file format"""
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        if not image.filename:
            return False
            
        ext = os.path.splitext(image.filename)[1].lower()
        return ext in allowed_extensions
    
    def _enhance_with_pet_info(
        self, 
        result: Dict[str, Any], 
        pet_info: Dict[str, Any],
        pet_type: str
    ) -> Dict[str, Any]:
        """
        Enhance BCS results with pet-specific information.
        
        Args:
            result: Base BCS assessment result
            pet_info: Pet information (breed, age, weight, etc.)
            pet_type: Type of pet
            
        Returns:
            Enhanced result
        """
        enhanced_result = result.copy()
        
        # Add breed-specific considerations
        breed = pet_info.get('breed', '').lower()
        if breed:
            breed_considerations = self._get_breed_considerations(breed, pet_type)
            if breed_considerations:
                enhanced_result['breed_specific_notes'] = breed_considerations
        
        # Add age-specific considerations
        age = pet_info.get('age')
        if age is not None:
            age_considerations = self._get_age_considerations(age, result['bcs_score'])
            enhanced_result['age_considerations'] = age_considerations
        
        # Calculate ideal weight if current weight is provided
        current_weight = pet_info.get('weight')
        if current_weight is not None:
            ideal_weight_range = self._calculate_ideal_weight(
                current_weight, 
                result['bcs_score'],
                breed,
                pet_type
            )
            enhanced_result['weight_analysis'] = {
                'current_weight': current_weight,
                'ideal_weight_range': ideal_weight_range,
                'unit': pet_info.get('weight_unit', 'kg')
            }
        
        return enhanced_result
    
    def _get_breed_considerations(self, breed: str, pet_type: str) -> List[str]:
        """Get breed-specific BCS considerations"""
        considerations = []
        
        # Breeds prone to obesity
        obesity_prone_breeds = {
            'dog': ['래브라도', 'labrador', '비글', 'beagle', '코커스패니얼', 'cocker spaniel', 
                   '닥스훈트', 'dachshund', '바셋하운드', 'basset hound'],
            'cat': ['브리티시숏헤어', 'british shorthair', '페르시안', 'persian', 
                   '랙돌', 'ragdoll']
        }
        
        # Naturally lean breeds
        lean_breeds = {
            'dog': ['그레이하운드', 'greyhound', '휘펫', 'whippet', '이탈리안그레이하운드',
                   'italian greyhound', '살루키', 'saluki'],
            'cat': ['시암', 'siamese', '오리엔탈', 'oriental', '아비시니안', 'abyssinian']
        }
        
        if pet_type in obesity_prone_breeds:
            for prone_breed in obesity_prone_breeds[pet_type]:
                if prone_breed in breed:
                    considerations.append(f"{breed}은(는) 비만에 취약한 품종입니다")
                    considerations.append("정기적인 체중 모니터링이 특히 중요합니다")
                    break
        
        if pet_type in lean_breeds:
            for lean_breed in lean_breeds[pet_type]:
                if lean_breed in breed:
                    considerations.append(f"{breed}은(는) 자연적으로 마른 체형의 품종입니다")
                    considerations.append("품종 특성을 고려한 평가가 필요합니다")
                    break
        
        return considerations
    
    def _get_age_considerations(self, age: float, bcs_score: int) -> List[str]:
        """Get age-specific BCS considerations"""
        considerations = []
        
        if age < 1:  # Puppy/Kitten
            considerations.append("성장기 동물은 적절한 영양 공급이 중요합니다")
            if bcs_score <= 3:
                considerations.append("성장 부진의 위험이 있으니 수의사 상담을 권장합니다")
        elif age >= 7:  # Senior
            considerations.append("노령 동물은 대사율이 감소하여 체중 관리가 중요합니다")
            if bcs_score >= 7:
                considerations.append("관절 건강을 위해 체중 감량이 특히 중요합니다")
        
        return considerations
    
    def _calculate_ideal_weight(
        self, 
        current_weight: float, 
        bcs_score: int,
        breed: str,
        pet_type: str
    ) -> Dict[str, float]:
        """
        Calculate ideal weight range based on BCS.
        
        Args:
            current_weight: Current weight
            bcs_score: BCS score (1-9)
            breed: Pet breed
            pet_type: Type of pet
            
        Returns:
            Ideal weight range
        """
        # Approximate weight adjustment per BCS point
        # Each BCS point roughly equals 10-15% body weight
        weight_adjustment_percent = 0.125  # 12.5% per BCS point
        
        # Calculate difference from ideal (BCS 5)
        bcs_difference = bcs_score - 5
        
        # Calculate ideal weight
        if bcs_difference == 0:
            ideal_weight = current_weight
        else:
            # Adjust weight based on BCS difference
            adjustment_factor = 1 - (bcs_difference * weight_adjustment_percent)
            ideal_weight = current_weight * adjustment_factor
        
        # Create range (±5% of ideal)
        return {
            'min': round(ideal_weight * 0.95, 1),
            'max': round(ideal_weight * 1.05, 1),
            'target': round(ideal_weight, 1)
        }
    
    def _calculate_assessment_quality(self, num_images: int) -> str:
        """
        Calculate assessment quality based on number of images.
        
        Args:
            num_images: Number of images provided
            
        Returns:
            Quality rating
        """
        required = self.adapter.num_required_images
        
        if num_images >= required:
            return "excellent"
        elif num_images >= required * 0.7:
            return "good"
        elif num_images >= required * 0.4:
            return "fair"
        else:
            return "limited"
    
    def get_image_guide(self) -> Dict[str, Any]:
        """
        Get guide for taking BCS assessment images.
        
        Returns:
            Image capture guide
        """
        return {
            'required_views': BCSAdapter.REQUIRED_VIEWS,
            'total_images_needed': self.adapter.num_required_images,
            'instructions': [
                "깨끗하고 밝은 배경에서 촬영하세요",
                "반려동물이 자연스럽게 서 있는 자세를 유지하도록 하세요",
                "각 각도에서 전신이 보이도록 촬영하세요",
                "털이 긴 동물의 경우, 체형이 잘 보이도록 촬영하세요"
            ],
            'view_descriptions': {
                'front': '정면에서 촬영',
                'back': '뒤에서 촬영',
                'left_side': '왼쪽 옆면에서 촬영',
                'right_side': '오른쪽 옆면에서 촬영',
                'top': '위에서 아래로 촬영',
                'abdomen': '복부가 잘 보이는 각도',
                'spine': '척추선이 잘 보이는 각도'
            }
        }
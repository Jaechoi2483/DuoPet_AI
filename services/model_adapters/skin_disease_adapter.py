"""
Skin Disease Model Adapter

Adapter for skin disease detection models including classification and segmentation.
"""

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
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
from pathlib import Path

from .base_adapter import ModelAdapter
from common.logger import get_logger

logger = get_logger(__name__)


class SkinDiseaseAdapter(ModelAdapter):
    """
    Adapter for skin disease detection models.
    
    Handles both classification (binary and multi-class) and segmentation models
    for comprehensive skin disease diagnosis.
    """
    
    # Disease class mappings
    DISEASE_CLASSES = {
        'dog_multi_136': {
            0: {"name": "A1_구진/플라크", "english": "Papule/Plaque", "code": "A1"},
            1: {"name": "A3_비듬/각질/상피성잔고리", "english": "Dandruff/Keratin", "code": "A3"},
            2: {"name": "A6_태선화/과다색소침착", "english": "Lichenification/Hyperpigmentation", "code": "A6"}
        },
        'dog_multi_456': {
            0: {"name": "A4_농포/여드름", "english": "Pustule/Acne", "code": "A4"},
            1: {"name": "A5_미란/궤양", "english": "Erosion/Ulcer", "code": "A5"},
            2: {"name": "A6_태선화/과다색소침착", "english": "Lichenification/Hyperpigmentation", "code": "A6"}
        }
    }
    
    def __init__(self, 
                 classification_models: Dict[str, Any],
                 segmentation_models: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize skin disease adapter with multiple models.
        
        Args:
            classification_models: Dictionary of classification models
            segmentation_models: Optional dictionary of segmentation models
            config: Configuration dictionary
        """
        # Store models dict as the main model
        models = {
            'classification': classification_models,
            'segmentation': segmentation_models or {}
        }
        super().__init__(models, config or {})
        
        self.classification_models = classification_models
        self.segmentation_models = segmentation_models or {}
        
        # Input shapes for different model types
        self.classification_input_shape = (224, 224)
        self.segmentation_input_shape = (512, 512)
        
        logger.info(f"SkinDiseaseAdapter initialized with {len(classification_models)} classification models")
    
    def preprocess(self, input_data: Any, model_type: str = "classification") -> np.ndarray:
        """
        Preprocess image for skin disease models.
        
        Args:
            input_data: Image data
            model_type: Type of model ("classification" or "segmentation")
            
        Returns:
            Preprocessed image array
        """
        # Convert to PIL Image
        if hasattr(input_data, 'file'):
            image = Image.open(input_data.file).convert('RGB')
        elif isinstance(input_data, np.ndarray):
            image = Image.fromarray(input_data).convert('RGB')
        elif isinstance(input_data, Image.Image):
            image = input_data.convert('RGB')
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Resize based on model type
        if model_type == "classification":
            image = image.resize(self.classification_input_shape)
        else:  # segmentation
            image = image.resize(self.segmentation_input_shape)
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, processed_data: Union[np.ndarray, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Run prediction with skin disease models.
        
        This method is overridden to handle the complex multi-model prediction flow.
        
        Args:
            processed_data: Preprocessed image or dict of preprocessed images
            
        Returns:
            Dictionary containing all model predictions
        """
        # This will be called from the diagnose method with proper context
        # The actual prediction logic is in _run_classification and _run_segmentation
        return {}
    
    def postprocess(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert model predictions to standardized format.
        
        Args:
            prediction: Dictionary containing various model outputs
            
        Returns:
            Standardized skin disease diagnosis results
        """
        # Extract results
        binary_result = prediction.get('binary_classification', {})
        multi_result = prediction.get('multi_classification', {})
        segmentation_result = prediction.get('segmentation', {})
        
        # Determine overall diagnosis
        has_disease = binary_result.get('has_disease', False)
        binary_confidence = binary_result.get('confidence', 0.0)
        
        result = {
            'has_skin_disease': has_disease,
            'disease_confidence': binary_confidence,
            'overall_status': 'abnormal' if has_disease else 'normal'
        }
        
        if has_disease and multi_result:
            # Add detailed disease information
            result.update({
                'disease_type': multi_result.get('disease_name', 'Unknown'),
                'disease_code': multi_result.get('disease_code', ''),
                'disease_type_confidence': multi_result.get('confidence', 0.0),
                'all_disease_probabilities': multi_result.get('probabilities', {})
            })
            
            # Calculate severity
            severity = self._calculate_severity(
                binary_confidence,
                multi_result.get('confidence', 0.0),
                segmentation_result.get('affected_area_percentage', 0.0)
            )
            result['severity'] = severity
        
        # Add segmentation results if available
        if segmentation_result:
            result['affected_area_percentage'] = segmentation_result.get('affected_area_percentage', 0.0)
            result['has_segmentation'] = True
            result['lesion_count'] = segmentation_result.get('lesion_count', 0)
        
        # Generate recommendations
        result['recommendations'] = self._generate_recommendations(result)
        result['requires_vet_visit'] = self._requires_vet_visit(result)
        
        return result
    
    def diagnose(self, 
                image: Any, 
                pet_type: str = "dog",
                include_segmentation: bool = True) -> Dict[str, Any]:
        """
        Complete skin disease diagnosis pipeline.
        
        Args:
            image: Input image
            pet_type: Type of pet ("dog" or "cat")
            include_segmentation: Whether to include segmentation analysis
            
        Returns:
            Comprehensive diagnosis results
        """
        try:
            results = {}
            
            # Step 1: Binary classification (normal vs disease)
            binary_result = self._run_binary_classification(image, pet_type)
            results['binary_classification'] = binary_result
            
            # Step 2: If disease detected, run multi-class classification
            if binary_result['has_disease'] and binary_result['confidence'] > 0.5:
                if pet_type == "dog":
                    # For dogs, we might need to try both multi-class models
                    multi_result = self._run_multi_classification(image, pet_type)
                    results['multi_classification'] = multi_result
                    
                    # Step 3: Run segmentation if requested and available
                    if include_segmentation and multi_result.get('disease_code'):
                        seg_result = self._run_segmentation(
                            image, 
                            pet_type, 
                            multi_result['disease_code']
                        )
                        if seg_result:
                            results['segmentation'] = seg_result
                else:
                    # For cats, we have limited models
                    logger.info("Limited multi-class models for cats, using binary result only")
            
            # Postprocess all results
            return self.postprocess(results)
            
        except Exception as e:
            logger.error(f"Error in skin disease diagnosis: {e}")
            raise
    
    def _run_binary_classification(self, image: Any, pet_type: str) -> Dict[str, Any]:
        """Run binary classification (normal vs disease)"""
        model_key = f"{pet_type}_binary"
        
        if model_key not in self.classification_models:
            logger.warning(f"Binary model not found for {pet_type}")
            return {'has_disease': False, 'confidence': 0.0}
        
        try:
            # Preprocess
            processed = self.preprocess(image, "classification")
            
            # Predict
            model = self.classification_models[model_key]
            predictions = model.predict(processed, verbose=0)
            
            # Interpret results (assuming index 1 is disease)
            disease_prob = float(predictions[0][1]) if predictions.shape[1] > 1 else float(predictions[0][0])
            
            return {
                'has_disease': disease_prob > 0.5,
                'confidence': disease_prob,
                'probabilities': {
                    'normal': float(1 - disease_prob),
                    'disease': disease_prob
                }
            }
            
        except Exception as e:
            logger.error(f"Error in binary classification: {e}")
            return {'has_disease': False, 'confidence': 0.0}
    
    def _run_multi_classification(self, image: Any, pet_type: str) -> Dict[str, Any]:
        """Run multi-class disease classification"""
        # For dogs, try different multi-class models
        if pet_type == "dog":
            # Try both multi-class models and use the one with higher confidence
            results = []
            
            for model_key in ['dog_multi_136', 'dog_multi_456']:
                if model_key in self.classification_models:
                    result = self._classify_with_model(image, model_key)
                    if result:
                        results.append(result)
            
            # Return the result with highest confidence
            if results:
                return max(results, key=lambda x: x['confidence'])
        
        return {}
    
    def _classify_with_model(self, image: Any, model_key: str) -> Optional[Dict[str, Any]]:
        """Classify with a specific multi-class model"""
        try:
            # Preprocess
            processed = self.preprocess(image, "classification")
            
            # Predict
            model = self.classification_models[model_key]
            predictions = model.predict(processed, verbose=0)
            
            # Get predicted class
            predicted_idx = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            # Get disease info
            disease_info = self.DISEASE_CLASSES[model_key][predicted_idx]
            
            # Create probability dict
            probabilities = {}
            for idx, prob in enumerate(predictions[0]):
                class_info = self.DISEASE_CLASSES[model_key].get(idx, {})
                class_name = class_info.get('name', f'Class_{idx}')
                probabilities[class_name] = float(prob)
            
            return {
                'disease_name': disease_info['name'],
                'disease_code': disease_info['code'],
                'disease_english': disease_info['english'],
                'confidence': confidence,
                'probabilities': probabilities,
                'model_used': model_key
            }
            
        except Exception as e:
            logger.error(f"Error in multi-class classification with {model_key}: {e}")
            return None
    
    def _run_segmentation(self, image: Any, pet_type: str, disease_code: str) -> Optional[Dict[str, Any]]:
        """Run segmentation for lesion detection"""
        seg_key = f"{pet_type}_{disease_code}"
        
        if seg_key not in self.segmentation_models:
            logger.info(f"No segmentation model for {seg_key}")
            return None
        
        try:
            # Preprocess for segmentation
            processed = self.preprocess(image, "segmentation")
            
            # Predict
            model = self.segmentation_models[seg_key]
            seg_mask = model.predict(processed, verbose=0)[0]
            
            # Process segmentation mask
            binary_mask = (seg_mask[:, :, 0] > 0.5).astype(np.uint8)
            
            # Calculate affected area
            total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
            affected_pixels = np.sum(binary_mask)
            affected_percentage = (affected_pixels / total_pixels) * 100
            
            # Count lesions (connected components)
            num_lesions, _ = cv2.connectedComponents(binary_mask)
            
            return {
                'affected_area_percentage': float(affected_percentage),
                'lesion_count': int(num_lesions - 1),  # Subtract background
                'mask_shape': binary_mask.shape,
                'has_lesions': affected_percentage > 0.1
            }
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            return None
    
    def _calculate_severity(self, 
                           binary_conf: float,
                           multi_conf: float,
                           affected_area: float) -> str:
        """Calculate disease severity based on multiple factors"""
        # Weighted severity score
        severity_score = (
            binary_conf * 0.3 +  # Binary classification confidence
            multi_conf * 0.3 +   # Multi-class confidence
            min(affected_area / 100, 1.0) * 0.4  # Affected area percentage
        )
        
        if severity_score >= 0.8:
            return "severe"
        elif severity_score >= 0.6:
            return "moderate"
        elif severity_score >= 0.3:
            return "mild"
        else:
            return "minimal"
    
    def _generate_recommendations(self, diagnosis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnosis"""
        recommendations = []
        
        if not diagnosis.get('has_skin_disease'):
            recommendations.append("피부 상태가 정상으로 보입니다")
            recommendations.append("정기적인 피부 관찰을 지속해주세요")
            return recommendations
        
        # Disease-specific recommendations
        disease_code = diagnosis.get('disease_code', '')
        severity = diagnosis.get('severity', 'unknown')
        
        # General recommendations
        if severity in ['severe', 'moderate']:
            recommendations.append("가능한 빨리 동물병원 방문을 권장합니다")
        
        # Disease-specific recommendations
        if disease_code == 'A1':  # 구진/플라크
            recommendations.extend([
                "알레르기 반응일 수 있으니 음식이나 환경 변화를 확인하세요",
                "긁지 않도록 주의하고 필요시 넥카라 착용을 고려하세요"
            ])
        elif disease_code == 'A3':  # 비듬/각질
            recommendations.extend([
                "정기적인 목욕과 보습이 도움이 될 수 있습니다",
                "사료를 피부 건강 전용 제품으로 변경을 고려하세요"
            ])
        elif disease_code == 'A4':  # 농포/여드름
            recommendations.extend([
                "세균 감염이 의심되므로 항생제 치료가 필요할 수 있습니다",
                "청결한 환경을 유지하고 상처 부위를 만지지 마세요"
            ])
        elif disease_code == 'A5':  # 미란/궤양
            recommendations.extend([
                "즉각적인 수의학적 치료가 필요합니다",
                "상처가 악화되지 않도록 보호 조치가 필요합니다"
            ])
        elif disease_code == 'A6':  # 태선화/과다색소침착
            recommendations.extend([
                "만성 피부염의 징후일 수 있으니 원인 파악이 중요합니다",
                "스트레스 요인을 줄이고 환경 개선을 고려하세요"
            ])
        
        # Affected area based recommendations
        affected_area = diagnosis.get('affected_area_percentage', 0)
        if affected_area > 20:
            recommendations.append("병변 범위가 넓으므로 전신적인 치료가 필요할 수 있습니다")
        
        return recommendations
    
    def _requires_vet_visit(self, diagnosis: Dict[str, Any]) -> bool:
        """Determine if vet visit is required"""
        if not diagnosis.get('has_skin_disease'):
            return False
        
        # Check severity
        severity = diagnosis.get('severity', 'unknown')
        if severity in ['severe', 'moderate']:
            return True
        
        # Check specific disease codes
        urgent_codes = ['A4', 'A5']  # 농포/여드름, 미란/궤양
        if diagnosis.get('disease_code') in urgent_codes:
            return True
        
        # Check affected area
        if diagnosis.get('affected_area_percentage', 0) > 15:
            return True
        
        return False
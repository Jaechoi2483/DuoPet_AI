"""
Health Diagnosis Orchestrator

Central orchestrator for coordinating multiple health diagnosis models
to provide comprehensive pet health assessment.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import UploadFile
import numpy as np

from services.eye_disease_service import EyeDiseaseService
from services.bcs_service import BCSService
from services.skin_disease_service import SkinDiseaseService
from services.model_registry import ModelRegistry, get_model_registry
from common.logger import get_logger


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

logger = get_logger(__name__)


class HealthDiagnosisOrchestrator:
    """
    Orchestrates multiple health diagnosis services to provide
    comprehensive health assessment.
    
    Features:
    - Parallel execution of independent diagnoses
    - Result combination and prioritization
    - Overall health scoring
    - Integrated recommendations
    """
    
    def __init__(self, 
                 eye_service: Optional[EyeDiseaseService] = None,
                 bcs_service: Optional[BCSService] = None,
                 skin_service: Optional[SkinDiseaseService] = None,
                 executor: Optional[ThreadPoolExecutor] = None):
        """
        Initialize the orchestrator with available services.
        
        Args:
            eye_service: Eye disease diagnosis service
            bcs_service: Body condition score service
            skin_service: Skin disease diagnosis service
            executor: Thread pool executor for parallel processing
        """
        self.eye_service = eye_service
        self.bcs_service = bcs_service
        self.skin_service = skin_service
        
        # Count available services
        self.available_services = sum([
            eye_service is not None,
            bcs_service is not None,
            skin_service is not None
        ])
        
        if self.available_services == 0:
            logger.warning("No health diagnosis services available!")
        else:
            logger.info(f"HealthDiagnosisOrchestrator initialized with {self.available_services} services")
        
        # Thread pool for CPU-bound model operations
        self.executor = executor or ThreadPoolExecutor(max_workers=3)
    
    async def comprehensive_diagnosis(
        self,
        images: List[UploadFile],
        pet_type: str = "dog",
        pet_info: Optional[Dict[str, Any]] = None,
        diagnosis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive health diagnosis using all available services.
        
        Args:
            images: List of pet images
            pet_type: Type of pet ("dog" or "cat")
            pet_info: Optional pet information (age, weight, breed, etc.)
            diagnosis_types: Optional list of specific diagnoses to perform
            
        Returns:
            Comprehensive health assessment results
        """
        start_time = datetime.now()
        
        # Validate inputs
        if not images:
            raise ValueError("At least one image is required for diagnosis")
        
        if pet_type not in ["dog", "cat"]:
            raise ValueError(f"Invalid pet type: {pet_type}")
        
        # Determine which diagnoses to perform
        if diagnosis_types is None:
            # 기본값: 이미지가 3장 미만이면 BCS를 제외
            if len(images) < 3:
                diagnosis_types = ["eye", "skin"]  # BCS 제외
                logger.info(f"Only {len(images)} images provided, excluding BCS from diagnosis")
            else:
                diagnosis_types = ["eye", "bcs", "skin"]  # All available
        
        # Prepare tasks for parallel execution
        tasks = []
        
        # Eye disease diagnosis (single image)
        if "eye" in diagnosis_types and self.eye_service:
            # Use the first image or find the most suitable one
            eye_image = self._select_eye_image(images)
            if eye_image:
                tasks.append(self._run_eye_diagnosis(eye_image))
        
        # BCS assessment (multiple images)
        if "bcs" in diagnosis_types and self.bcs_service:
            tasks.append(self._run_bcs_assessment(images, pet_type, pet_info))
        
        # Skin disease diagnosis (single image)
        if "skin" in diagnosis_types and self.skin_service:
            # Use the first image or find the most suitable one
            skin_image = self._select_skin_image(images)
            if skin_image:
                tasks.append(self._run_skin_diagnosis(skin_image, pet_type))
        
        # Execute all diagnoses in parallel
        if not tasks:
            logger.warning("No diagnosis tasks to execute")
            return self._create_empty_result()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        diagnosis_results = {}
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(str(result))
                logger.error(f"Diagnosis task {i} failed: {result}")
            else:
                diagnosis_results.update(result)
        
        # Combine results into comprehensive assessment
        comprehensive_result = self._combine_results(
            diagnosis_results,
            pet_type,
            pet_info
        )
        
        # Add metadata
        comprehensive_result['metadata'] = {
            'diagnosis_time': (datetime.now() - start_time).total_seconds(),
            'services_used': len(tasks),
            'images_analyzed': len(images),
            'pet_type': pet_type,
            'errors': errors if errors else None
        }
        
        return comprehensive_result
    
    async def _run_eye_diagnosis(self, image: UploadFile) -> Dict[str, Any]:
        """Run eye disease diagnosis"""
        try:
            logger.info("Running eye disease diagnosis")
            # Run synchronously in the main thread to avoid TensorFlow context issues
            result = self.eye_service.diagnose(image)
            return {'eye_health': result}
        except Exception as e:
            logger.error(f"Eye diagnosis failed: {e}")
            raise
    
    async def _run_bcs_assessment(
        self, 
        images: List[UploadFile],
        pet_type: str,
        pet_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run body condition score assessment"""
        try:
            logger.info(f"Running BCS assessment with {len(images)} images")
            result = await self.bcs_service.assess_body_condition(
                images=images,
                pet_type=pet_type,
                pet_info=pet_info
            )
            return {'body_condition': result}
        except Exception as e:
            logger.error(f"BCS assessment failed: {e}")
            raise
    
    async def _run_skin_diagnosis(
        self,
        image: UploadFile,
        pet_type: str
    ) -> Dict[str, Any]:
        """Run skin disease diagnosis"""
        try:
            logger.info("Running skin disease diagnosis")
            result = await self.skin_service.diagnose_skin_condition(
                image=image,
                pet_type=pet_type,
                include_segmentation=True
            )
            return {'skin_health': result}
        except Exception as e:
            logger.error(f"Skin diagnosis failed: {e}")
            raise
    
    def _select_eye_image(self, images: List[UploadFile]) -> Optional[UploadFile]:
        """
        Select the most suitable image for eye diagnosis.
        
        For now, returns the first image. Could be enhanced with
        image analysis to find close-ups of eyes.
        """
        return images[0] if images else None
    
    def _select_skin_image(self, images: List[UploadFile]) -> Optional[UploadFile]:
        """
        Select the most suitable image for skin diagnosis.
        
        For now, returns the first image. Could be enhanced with
        image analysis to find skin areas.
        """
        return images[0] if images else None
    
    def _combine_results(
        self,
        results: Dict[str, Any],
        pet_type: str,
        pet_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combine individual diagnosis results into comprehensive assessment.
        
        Args:
            results: Dictionary containing individual diagnosis results
            pet_type: Type of pet
            pet_info: Optional pet information
            
        Returns:
            Combined health assessment
        """
        # Extract individual results
        eye_result = results.get('eye_health', {})
        bcs_result = results.get('body_condition', {})
        skin_result = results.get('skin_health', {})
        
        # Calculate overall health score
        overall_score = self._calculate_overall_health_score(
            eye_result, bcs_result, skin_result
        )
        
        # Identify critical findings
        critical_findings = self._identify_critical_findings(
            eye_result, bcs_result, skin_result
        )
        
        # Generate comprehensive recommendations
        all_recommendations = self._generate_comprehensive_recommendations(
            eye_result, bcs_result, skin_result, critical_findings
        )
        
        # Determine if vet visit is required
        requires_vet = self._requires_vet_visit(
            eye_result, bcs_result, skin_result
        )
        
        # Build comprehensive result
        comprehensive_result = {
            'overall_health_score': overall_score,
            'health_status': self._determine_health_status(overall_score),
            'critical_findings': critical_findings,
            'requires_vet_visit': requires_vet,
            'priority_level': self._determine_priority_level(critical_findings, requires_vet),
            'comprehensive_recommendations': all_recommendations,
            'individual_assessments': {
                'eye_health': self._summarize_eye_health(eye_result),
                'body_condition': self._summarize_body_condition(bcs_result),
                'skin_health': self._summarize_skin_health(skin_result)
            }
        }
        
        # Add pet-specific insights if available
        if pet_info:
            comprehensive_result['personalized_insights'] = \
                self._generate_personalized_insights(comprehensive_result, pet_info, pet_type)
        
        return comprehensive_result
    
    def _calculate_overall_health_score(
        self,
        eye_result: Dict[str, Any],
        bcs_result: Dict[str, Any],
        skin_result: Dict[str, Any]
    ) -> float:
        """
        Calculate overall health score (0-100).
        
        Weighted scoring based on:
        - Eye health: 30%
        - Body condition: 40%
        - Skin health: 30%
        """
        scores = []
        weights = []
        
        # Eye health score
        if eye_result:
            eye_score = 100.0
            if eye_result.get('disease', '').lower() not in ['healthy', '정상', 'normal']:
                # Reduce score based on severity
                severity = eye_result.get('severity', 'medium')
                severity_penalties = {
                    'critical': 80, 'high': 60, 'medium': 40, 'low': 20, 'minimal': 10
                }
                eye_score -= severity_penalties.get(severity, 30)
            scores.append(eye_score)
            weights.append(0.3)
        
        # Body condition score
        if bcs_result:
            bcs_value = bcs_result.get('bcs_score', 5)
            # Ideal BCS is 4-6, calculate deviation
            if 4 <= bcs_value <= 6:
                bcs_score = 100.0
            else:
                deviation = min(abs(bcs_value - 5), 4)
                bcs_score = 100.0 - (deviation * 20)
            scores.append(bcs_score)
            weights.append(0.4)
        
        # Skin health score
        if skin_result:
            skin_score = 100.0
            if skin_result.get('has_skin_disease', False):
                # Reduce score based on severity and affected area
                severity = skin_result.get('severity', 'medium')
                severity_penalties = {
                    'severe': 70, 'moderate': 50, 'mild': 30, 'minimal': 15
                }
                skin_score -= severity_penalties.get(severity, 40)
                
                # Additional penalty for large affected area
                affected_area = skin_result.get('affected_area_percentage', 0)
                if affected_area > 20:
                    skin_score -= 10
            scores.append(skin_score)
            weights.append(0.3)
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return round(weighted_score, 1)
        
        return 85.0  # Default if no assessments available
    
    def _identify_critical_findings(
        self,
        eye_result: Dict[str, Any],
        bcs_result: Dict[str, Any],
        skin_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify critical health findings that need immediate attention"""
        critical_findings = []
        
        # Check eye health
        if eye_result:
            if eye_result.get('requires_vet_visit', False):
                critical_findings.append({
                    'type': 'eye_disease',
                    'severity': eye_result.get('severity', 'unknown'),
                    'condition': eye_result.get('disease', 'Unknown'),
                    'confidence': eye_result.get('confidence', 0.0),
                    'message': f"안구 질환 감지: {eye_result.get('disease', 'Unknown')}"
                })
        
        # Check body condition
        if bcs_result:
            bcs_score = bcs_result.get('bcs_score', 5)
            if bcs_score <= 2 or bcs_score >= 8:
                critical_findings.append({
                    'type': 'body_condition',
                    'severity': 'high' if bcs_score <= 2 or bcs_score >= 8 else 'medium',
                    'condition': bcs_result.get('bcs_category', 'Unknown'),
                    'score': bcs_score,
                    'message': f"비정상적인 체형 상태: BCS {bcs_score}/9"
                })
        
        # Check skin health
        if skin_result:
            if skin_result.get('has_skin_disease', False) and \
               skin_result.get('severity') in ['severe', 'moderate']:
                critical_findings.append({
                    'type': 'skin_disease',
                    'severity': skin_result.get('severity', 'unknown'),
                    'condition': skin_result.get('disease_type', 'Unknown'),
                    'affected_area': skin_result.get('affected_area_percentage', 0),
                    'message': f"피부 질환 감지: {skin_result.get('disease_type', 'Unknown')}"
                })
        
        return critical_findings
    
    def _generate_comprehensive_recommendations(
        self,
        eye_result: Dict[str, Any],
        bcs_result: Dict[str, Any],
        skin_result: Dict[str, Any],
        critical_findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate comprehensive health recommendations"""
        recommendations = []
        
        # Priority recommendations based on critical findings
        if critical_findings:
            recommendations.append("🚨 즉시 수의사 진료가 필요한 상태가 발견되었습니다")
            
            for finding in critical_findings:
                if finding['type'] == 'eye_disease':
                    recommendations.append(f"• 안과 전문 진료 필요: {finding['condition']}")
                elif finding['type'] == 'body_condition':
                    if finding['score'] <= 2:
                        recommendations.append("• 긴급한 영양 상태 개선이 필요합니다")
                    elif finding['score'] >= 8:
                        recommendations.append("• 체중 감량 프로그램이 시급합니다")
                elif finding['type'] == 'skin_disease':
                    recommendations.append(f"• 피부과 진료 필요: {finding['condition']}")
        
        # General health maintenance recommendations
        else:
            recommendations.append("✅ 전반적인 건강 상태가 양호합니다")
            recommendations.append("• 정기적인 건강 검진을 유지하세요")
            recommendations.append("• 균형 잡힌 식단과 적절한 운동을 지속하세요")
        
        # Add specific recommendations from each assessment
        if eye_result and 'recommendations' in eye_result:
            recommendations.extend([f"[눈 건강] {r}" for r in eye_result['recommendations'][:2]])
        
        if bcs_result and 'recommendations' in bcs_result:
            recommendations.extend([f"[체형 관리] {r}" for r in bcs_result['recommendations'][:2]])
        
        if skin_result and 'recommendations' in skin_result:
            recommendations.extend([f"[피부 건강] {r}" for r in skin_result['recommendations'][:2]])
        
        return recommendations
    
    def _requires_vet_visit(
        self,
        eye_result: Dict[str, Any],
        bcs_result: Dict[str, Any],
        skin_result: Dict[str, Any]
    ) -> bool:
        """Determine if veterinary visit is required"""
        # Check each assessment
        if eye_result and eye_result.get('requires_vet_visit', False):
            return True
        
        if bcs_result and bcs_result.get('requires_vet_consultation', False):
            return True
        
        if skin_result and skin_result.get('requires_vet_visit', False):
            return True
        
        return False
    
    def _determine_health_status(self, overall_score: float) -> str:
        """Determine health status based on overall score"""
        if overall_score >= 90:
            return "excellent"
        elif overall_score >= 75:
            return "good"
        elif overall_score >= 60:
            return "fair"
        elif overall_score >= 40:
            return "poor"
        else:
            return "critical"
    
    def _determine_priority_level(
        self,
        critical_findings: List[Dict[str, Any]],
        requires_vet: bool
    ) -> str:
        """Determine priority level for veterinary care"""
        if not critical_findings and not requires_vet:
            return "routine"
        
        # Check severity of findings
        severities = [f.get('severity', 'unknown') for f in critical_findings]
        
        if 'critical' in severities:
            return "emergency"
        elif 'high' in severities or len(critical_findings) >= 2:
            return "urgent"
        elif requires_vet:
            return "soon"
        else:
            return "monitor"
    
    def _summarize_eye_health(self, eye_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of eye health assessment"""
        if not eye_result:
            return {'status': 'not_assessed'}
        
        return {
            'status': 'normal' if eye_result.get('disease', '').lower() in ['healthy', '정상'] else 'abnormal',
            'condition': eye_result.get('disease', 'Unknown'),
            'confidence': eye_result.get('confidence', 0.0),
            'requires_attention': eye_result.get('requires_vet_visit', False)
        }
    
    def _summarize_body_condition(self, bcs_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of body condition assessment"""
        if not bcs_result:
            return {'status': 'not_assessed'}
        
        bcs_score = bcs_result.get('bcs_score', 5)
        return {
            'status': 'normal' if 4 <= bcs_score <= 6 else 'abnormal',
            'score': bcs_score,
            'category': bcs_result.get('bcs_category', 'Unknown'),
            'confidence': bcs_result.get('confidence', 0.0),
            'requires_attention': bcs_result.get('requires_vet_consultation', False)
        }
    
    def _summarize_skin_health(self, skin_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of skin health assessment"""
        if not skin_result:
            return {'status': 'not_assessed'}
        
        return {
            'status': 'normal' if not skin_result.get('has_skin_disease', False) else 'abnormal',
            'condition': skin_result.get('disease_type', 'None'),
            'severity': skin_result.get('severity', 'unknown'),
            'affected_area': skin_result.get('affected_area_percentage', 0.0),
            'requires_attention': skin_result.get('requires_vet_visit', False)
        }
    
    def _generate_personalized_insights(
        self,
        comprehensive_result: Dict[str, Any],
        pet_info: Dict[str, Any],
        pet_type: str
    ) -> List[str]:
        """Generate personalized insights based on pet information"""
        insights = []
        
        age = pet_info.get('age')
        breed = pet_info.get('breed', '').lower()
        weight = pet_info.get('weight')
        
        # Age-based insights
        if age:
            if age < 1:
                insights.append("성장기 반려동물은 특별한 영양 관리가 필요합니다")
            elif age >= 7:
                insights.append("노령 반려동물은 정기적인 건강 검진이 더욱 중요합니다")
        
        # Breed-specific insights
        if breed:
            # Add breed-specific health concerns
            # This would be expanded with actual breed data
            pass
        
        # Weight-based insights with BCS
        if weight and 'body_condition' in comprehensive_result['individual_assessments']:
            bcs_status = comprehensive_result['individual_assessments']['body_condition']
            if bcs_status['status'] == 'abnormal':
                if bcs_status['score'] < 4:
                    insights.append("체중 증량을 위한 영양 상담이 권장됩니다")
                elif bcs_status['score'] > 6:
                    insights.append("체중 감량을 위한 운동 프로그램이 필요합니다")
        
        return insights
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when no services are available"""
        return {
            'overall_health_score': 0.0,
            'health_status': 'not_assessed',
            'critical_findings': [],
            'requires_vet_visit': False,
            'priority_level': 'unknown',
            'comprehensive_recommendations': [
                "건강 진단 서비스를 사용할 수 없습니다",
                "시스템 관리자에게 문의하세요"
            ],
            'individual_assessments': {
                'eye_health': {'status': 'not_available'},
                'body_condition': {'status': 'not_available'},
                'skin_health': {'status': 'not_available'}
            }
        }
    
    def get_available_services(self) -> Dict[str, bool]:
        """Get information about available services"""
        return {
            'eye_disease': self.eye_service is not None,
            'body_condition': self.bcs_service is not None,
            'skin_disease': self.skin_service is not None,
            'total_available': self.available_services
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=False)
"""
Health Diagnosis Router

This module provides endpoints for pet health diagnosis from images.
"""

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from common.response import StandardResponse, create_success_response
from common.logger import get_logger
from services.health_diagnosis.predict import predict_eye_disease, predict_bcs, predict_skin_disease

logger = get_logger(__name__)

router = APIRouter()


class DiseaseDetection(BaseModel):
    """Disease detection result model"""
    disease_type: str = Field(..., description="Type of disease detected")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    severity: str = Field(..., description="Severity level (low/medium/high)")
    location: Optional[Dict[str, float]] = Field(None, description="Bounding box coordinates")


class HealthDiagnosisResult(BaseModel):
    """Health diagnosis result model"""
    detections: List[DiseaseDetection] = Field(..., description="List of detected diseases")
    overall_health_score: float = Field(..., ge=0, le=100, description="Overall health score")
    recommendations: List[str] = Field(..., description="Health recommendations")
    requires_vet_visit: bool = Field(..., description="Whether vet visit is recommended")


class EyeDiseaseResult(BaseModel):
    """Eye disease diagnosis result model"""
    disease_detected: bool = Field(..., description="Whether disease is detected")
    disease_type: str = Field(..., description="Type of eye disease detected")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    severity: str = Field(..., description="Severity level (low/medium/high)")
    recommendations: List[str] = Field(..., description="Treatment recommendations")
    all_predictions: Optional[Dict[str, float]] = Field(None, description="All class probabilities")


@router.post("/analyze", response_model=StandardResponse)
async def analyze_health(
    image: UploadFile = File(..., description="Pet image for health analysis"),
    pet_type: str = "dog",
    pet_age: Optional[int] = None
):
    """
    Analyze pet health from image
    
    Uses YOLOv12 for disease detection and EfficientNet for classification.
    
    - **image**: Image of the pet (JPEG, PNG)
    - **pet_type**: Type of pet (dog/cat)
    - **pet_age**: Age of pet in years (optional)
    """
    logger.info(f"Health analysis request for {pet_type}")
    
    # TODO: Implement health diagnosis logic
    # 1. Validate image format and size
    # 2. Preprocess image
    # 3. Run YOLOv12 for disease area detection
    # 4. Run EfficientNet for disease classification
    # 5. Generate health recommendations

    result = HealthDiagnosisResult(
        detections=[],
        overall_health_score=85.0,
        recommendations=["정기적인 건강 검진을 받아보는 것을 권장합니다."],
        requires_vet_visit=False
    )
    
    return create_success_response(data=result.model_dump())


@router.post("/analyze/eye", response_model=StandardResponse)
async def analyze_eye_disease(
    image: UploadFile = File(..., description="Pet eye image for disease detection")
):
    """
    Analyze pet eye disease from image
    
    Uses EfficientNetB0 model for eye disease classification.
    Detects: Normal, Cataract, Conjunctivitis, Corneal Ulcer, Other diseases
    
    - **image**: Close-up image of the pet's eye (JPEG, PNG)
    """
    logger.info("Eye disease analysis request")
    
    try:
        # Read and validate image
        contents = await image.read()
        
        # Convert bytes to PIL Image
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents))
        
        # Ensure RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Perform prediction
        result = predict_eye_disease(img)
        
        # Create response
        eye_result = EyeDiseaseResult(
            disease_detected=result['disease_detected'],
            disease_type=result['disease_type'],
            confidence=result['confidence'],
            severity=result['severity'],
            recommendations=result['recommendations'],
            all_predictions=result.get('all_predictions')
        )
        
        return create_success_response(
            data=eye_result.model_dump(),
            message="안구질환 분석이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"Error in eye disease analysis: {str(e)}")
        return create_success_response(
            data=None,
            message=f"안구질환 분석 중 오류가 발생했습니다: {str(e)}",
            code="500"
        )


@router.get("/diseases", response_model=StandardResponse)
async def get_supported_diseases():
    """
    Get list of diseases that can be detected
    """
    diseases = [
        {
            "id": "skin_disease",
            "name": "피부 질환",
            "description": "발진, 가려움, 염증 등 피부 관련 질환",
            "detectable": True
        },
        {
            "id": "eye_infection",
            "name": "눈 감염",
            "description": "눈물, 충혈 등 안구 관련 질환",
            "detectable": True
        },
        {
            "id": "ear_infection",
            "name": "귀 감염",
            "description": "귀지, 냄새 등 귀 관련 질환",
            "detectable": True
        },
        {
            "id": "dental_disease",
            "name": "치과 질환",
            "description": "치석, 치주염 등 구강 관련 질환",
            "detectable": True
        },
        {
            "id": "obesity",
            "name": "비만",
            "description": "체중 관리가 필요한 과체중 상태",
            "detectable": True
        },
        {
            "id": "cataract",
            "name": "백내장",
            "description": "수정체가 혼탁해져 시력이 저하되는 질환",
            "detectable": True
        },
        {
            "id": "corneal_ulcer",
            "name": "각막궤양",
            "description": "각막 표면에 상처나 궤양이 생긴 상태",
            "detectable": True
        }
    ]
    
    return create_success_response(data={"diseases": diseases})
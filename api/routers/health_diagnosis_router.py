"""
Health Diagnosis Router

This module provides endpoints for pet health diagnosis from images.
"""

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from common.response import StandardResponse, create_success_response
from common.logger import get_logger

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
        }
    ]
    
    return create_success_response(data={"diseases": diseases})
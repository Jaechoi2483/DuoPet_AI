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
        recommendations=["0x t ��D <8�"],
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
            "name": "<� �X",
            "description": "��, <��, Ȩ �",
            "detectable": True
        },
        {
            "id": "eye_infection",
            "name": "Hl �",
            "description": "���, �� �",
            "detectable": True
        },
        {
            "id": "ear_infection",
            "name": "� �",
            "description": "xt�, t� �",
            "detectable": True
        },
        {
            "id": "dental_disease",
            "name": "X� �X",
            "description": "X, X@� �",
            "detectable": True
        },
        {
            "id": "obesity",
            "name": "D�",
            "description": "�� ��",
            "detectable": True
        }
    ]
    
    return create_success_response(data={"diseases": diseases})
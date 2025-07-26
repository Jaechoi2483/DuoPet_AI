"""
Health Diagnosis Router

This module provides endpoints for pet health diagnosis from images.
"""

from fastapi import APIRouter, UploadFile, File, Depends, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os
from pathlib import Path

from common.response import StandardResponse, create_success_response, create_error_response
from common.logger import get_logger
from services.eye_disease_service import EyeDiseaseService
from services.bcs_service import BCSService
from services.skin_disease_service import SkinDiseaseService
from services.health_diagnosis_orchestrator import HealthDiagnosisOrchestrator

logger = get_logger(__name__)

router = APIRouter()

# --- Service Initialization (Robust Path Handling) ---
# 현재 라우터 파일의 위치를 기준으로 동적으로 경로를 설정합니다.
# 이렇게 하면 프로젝트가 어떤 경로에 있더라도 모델을 찾을 수 있습니다.
try:
    # 1. 현재 파일의 절대 경로를 찾습니다.
    current_file_path = Path(__file__).resolve()
    # 2. /api/routers/ 디렉토리를 벗어나 프로젝트의 루트 디렉토리로 이동합니다.
    # (DuoPet_AI/api/routers/health_diagnosis_router.py -> DuoPet_AI/)
    project_root = current_file_path.parent.parent.parent
    # 3. 모델 파일의 절대 경로를 조합합니다.
    MODEL_PATH = project_root / "models" / "health_diagnosis" / "eye_disease" / "best_grouped_model.keras"
    CLASS_MAP_PATH = project_root / "models" / "health_diagnosis" / "eye_disease" / "class_map.json"

    # 4. 서비스 인스턴스를 생성합니다.
    if MODEL_PATH.exists() and CLASS_MAP_PATH.exists():
        eye_disease_service = EyeDiseaseService(model_path=str(MODEL_PATH), class_map_path=str(CLASS_MAP_PATH))
        logger.info("EyeDiseaseService initialized successfully.")
    else:
        eye_disease_service = None
        logger.error(f"Model or class map file not found. Searched paths:\n- {MODEL_PATH}\n- {CLASS_MAP_PATH}")

except Exception as e:
    logger.error(f"Failed to initialize EyeDiseaseService: {e}", exc_info=True)
    eye_disease_service = None

# Initialize BCS Service
try:
    # Use same project root path resolution
    BCS_MODEL_PATH = project_root / "models" / "health_diagnosis" / "bcs" / "bcs_efficientnet_v1.h5"
    BCS_CONFIG_PATH = project_root / "models" / "health_diagnosis" / "bcs" / "config.yaml"
    
    if BCS_MODEL_PATH.exists():
        bcs_service = BCSService(
            model_path=str(BCS_MODEL_PATH),
            config_path=str(BCS_CONFIG_PATH) if BCS_CONFIG_PATH.exists() else None
        )
        logger.info("BCSService initialized successfully.")
    else:
        bcs_service = None
        logger.error(f"BCS model not found at: {BCS_MODEL_PATH}")
        
except Exception as e:
    logger.error(f"Failed to initialize BCSService: {e}", exc_info=True)
    bcs_service = None

# Initialize Skin Disease Service
try:
    skin_disease_service = SkinDiseaseService()
    logger.info("SkinDiseaseService initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize SkinDiseaseService: {e}", exc_info=True)
    skin_disease_service = None

def get_eye_disease_service():
    if not eye_disease_service:
        # 서비스가 초기화되지 않았을 때 더 명확한 에러를 발생시킵니다.
        raise RuntimeError("EyeDiseaseService is not available. Check model paths and initialization logs.")
    return eye_disease_service

def get_bcs_service():
    if not bcs_service:
        raise RuntimeError("BCSService is not available. Check model paths and initialization logs.")
    return bcs_service

def get_skin_disease_service():
    if not skin_disease_service:
        raise RuntimeError("SkinDiseaseService is not available. Check model paths and initialization logs.")
    return skin_disease_service

# Initialize Health Diagnosis Orchestrator
try:
    orchestrator = HealthDiagnosisOrchestrator(
        eye_service=eye_disease_service,
        bcs_service=bcs_service,
        skin_service=skin_disease_service
    )
    logger.info("HealthDiagnosisOrchestrator initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize HealthDiagnosisOrchestrator: {e}", exc_info=True)
    orchestrator = None

def get_orchestrator():
    if not orchestrator:
        raise RuntimeError("HealthDiagnosisOrchestrator is not available.")
    return orchestrator


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
    disease: str = Field(..., description="Type of eye disease detected")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class BCSResult(BaseModel):
    """Body Condition Score result model"""
    bcs_category: str = Field(..., description="BCS category (저체중/정상/과체중)")
    bcs_score: int = Field(..., ge=1, le=9, description="Detailed BCS score (1-9)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    health_insights: List[str] = Field(..., description="Health insights based on BCS")
    recommendations: List[str] = Field(..., description="Recommendations for pet health")
    requires_vet_consultation: bool = Field(..., description="Whether vet consultation is needed")


class SkinDiseaseResult(BaseModel):
    """Skin disease diagnosis result model"""
    has_skin_disease: bool = Field(..., description="Whether skin disease is detected")
    disease_confidence: float = Field(..., ge=0, le=1, description="Disease detection confidence")
    disease_type: Optional[str] = Field(None, description="Type of skin disease detected")
    disease_code: Optional[str] = Field(None, description="Disease code (A1-A6)")
    severity: Optional[str] = Field(None, description="Disease severity level")
    affected_area_percentage: Optional[float] = Field(None, description="Percentage of affected skin area")
    recommendations: List[str] = Field(..., description="Treatment recommendations")
    requires_vet_visit: bool = Field(..., description="Whether vet visit is required")


@router.post("/analyze", response_model=StandardResponse)
async def analyze_health(
    images: List[UploadFile] = File(..., description="Pet images for comprehensive health analysis"),
    pet_type: str = Form("dog"),
    pet_age: Optional[float] = Form(None),
    pet_weight: Optional[float] = Form(None),
    pet_breed: Optional[str] = Form(None),
    diagnosis_types: Optional[str] = Form(None),
    orchestrator_service: HealthDiagnosisOrchestrator = Depends(get_orchestrator)
):
    """
    Comprehensive pet health analysis using multiple AI models
    
    This endpoint orchestrates multiple health diagnosis models to provide
    a comprehensive health assessment including:
    - Eye disease detection
    - Body Condition Score (BCS) assessment
    - Skin disease diagnosis
    
    - **images**: Multiple images of the pet (recommended: various angles for BCS, close-ups for eye/skin)
    - **pet_type**: Type of pet (dog/cat)
    - **pet_age**: Age of pet in years (optional)
    - **pet_weight**: Current weight in kg (optional)
    - **pet_breed**: Breed of the pet (optional)
    - **diagnosis_types**: Comma-separated list of diagnoses to perform (eye,bcs,skin). If not specified, all available diagnoses are performed.
    """
    logger.info(f"Comprehensive health analysis request for {pet_type} with {len(images)} images")
    
    try:
        # Parse diagnosis types if provided
        if diagnosis_types:
            diagnosis_list = [d.strip() for d in diagnosis_types.split(",")]
        else:
            diagnosis_list = None
        
        # Build pet info
        pet_info = {}
        if pet_age is not None:
            pet_info['age'] = pet_age
        if pet_weight is not None:
            pet_info['weight'] = pet_weight
            pet_info['weight_unit'] = 'kg'
        if pet_breed:
            pet_info['breed'] = pet_breed
        
        # Run comprehensive diagnosis
        result = await orchestrator_service.comprehensive_diagnosis(
            images=images,
            pet_type=pet_type,
            pet_info=pet_info if pet_info else None,
            diagnosis_types=diagnosis_list
        )
        
        # Create response message
        health_status = result.get('health_status', 'unknown')
        overall_score = result.get('overall_health_score', 0)
        
        status_messages = {
            'excellent': f"반려동물의 건강 상태가 매우 좋습니다! (점수: {overall_score}/100)",
            'good': f"반려동물의 건강 상태가 양호합니다. (점수: {overall_score}/100)",
            'fair': f"주의가 필요한 건강 상태입니다. (점수: {overall_score}/100)",
            'poor': f"건강 관리가 시급합니다. (점수: {overall_score}/100)",
            'critical': f"즉시 수의사 진료가 필요합니다! (점수: {overall_score}/100)"
        }
        
        message = status_messages.get(health_status, f"건강 평가 완료 (점수: {overall_score}/100)")
        
        return create_success_response(
            data=result,
            message=message
        )
        
    except ValueError as e:
        logger.error(f"Validation error in comprehensive analysis: {str(e)}")
        return create_error_response(
            message=f"입력 검증 오류: {str(e)}",
            code="400"
        )
    except Exception as e:
        logger.error(f"Error in comprehensive health analysis: {str(e)}")
        return create_error_response(
            message=f"종합 건강 분석 중 오류가 발생했습니다: {str(e)}",
            code="500"
        )


@router.post("/analyze/eye", response_model=StandardResponse[EyeDiseaseResult])
async def analyze_eye_disease(
    image: UploadFile = File(..., description="Pet eye image for disease detection"),
    service: EyeDiseaseService = Depends(get_eye_disease_service)
):
    """
    Analyze pet eye disease from image
    
    Uses a Keras model for eye disease classification.
    
    - **image**: Close-up image of the pet's eye (JPEG, PNG)
    """
    logger.info(f"Eye disease analysis request for file: {image.filename}")
    
    try:
        # 파일 확장자 검사
        allowed_extensions = {".jpg", ".jpeg", ".png"}
        file_extension = os.path.splitext(image.filename)[1].lower()
        if file_extension not in allowed_extensions:
            return create_error_response(message="Invalid image format. Only JPG, JPEG, PNG are allowed.", code="400")

        # 서비스 호출하여 진단 수행
        result = service.diagnose(image)
        
        return create_success_response(
            data=result,
            message="안구질환 분석이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"Error in eye disease analysis: {str(e)}")
        return create_error_response(
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


@router.post("/analyze/bcs", response_model=StandardResponse[BCSResult])
async def analyze_body_condition(
    images: List[UploadFile] = File(..., description="Multiple pet images for BCS analysis"),
    pet_type: str = Form("dog"),
    pet_age: Optional[float] = Form(None),
    pet_weight: Optional[float] = Form(None),
    pet_breed: Optional[str] = Form(None),
    service: BCSService = Depends(get_bcs_service)
):
    """
    Analyze pet body condition score from multiple images
    
    Uses multi-head EfficientNet model for comprehensive BCS assessment.
    Ideally requires 13 images from different angles for best accuracy.
    
    - **images**: Multiple images of the pet from different angles
    - **pet_type**: Type of pet (dog/cat)
    - **pet_age**: Age of pet in years (optional)
    - **pet_weight**: Current weight in kg (optional)
    - **pet_breed**: Breed of the pet (optional)
    """
    logger.info(f"BCS analysis request for {pet_type} with {len(images)} images")
    
    try:
        # Build pet info if provided
        pet_info = {}
        if pet_age is not None:
            pet_info['age'] = pet_age
        if pet_weight is not None:
            pet_info['weight'] = pet_weight
            pet_info['weight_unit'] = 'kg'
        if pet_breed:
            pet_info['breed'] = pet_breed
            
        # Run BCS assessment
        result = await service.assess_body_condition(
            images=images,
            pet_type=pet_type,
            pet_info=pet_info if pet_info else None
        )
        
        # Extract key fields for response model
        bcs_response = BCSResult(
            bcs_category=result['bcs_category'],
            bcs_score=result['bcs_score'],
            confidence=result['confidence'],
            health_insights=result['health_insights'],
            recommendations=result['recommendations'],
            requires_vet_consultation=result['requires_vet_consultation']
        )
        
        return create_success_response(
            data=bcs_response.model_dump(),
            message=f"체형 평가가 완료되었습니다. BCS 점수: {result['bcs_score']}/9"
        )
        
    except Exception as e:
        logger.error(f"Error in BCS analysis: {str(e)}")
        return create_error_response(
            message=f"체형 평가 중 오류가 발생했습니다: {str(e)}",
            code="500"
        )


@router.get("/bcs/guide", response_model=StandardResponse)
async def get_bcs_image_guide(
    service: BCSService = Depends(get_bcs_service)
):
    """
    Get guide for taking BCS assessment images
    
    Returns instructions and required views for optimal BCS assessment.
    """
    try:
        guide = service.get_image_guide()
        return create_success_response(
            data=guide,
            message="BCS 촬영 가이드를 확인하세요"
        )
    except Exception as e:
        logger.error(f"Error getting BCS guide: {str(e)}")
        return create_error_response(
            message=f"가이드 조회 중 오류가 발생했습니다: {str(e)}",
            code="500"
        )


@router.post("/analyze/skin", response_model=StandardResponse[SkinDiseaseResult])
async def analyze_skin_disease(
    image: UploadFile = File(..., description="Pet skin image for disease detection"),
    pet_type: str = Form("dog"),
    include_segmentation: bool = Form(True),
    service: SkinDiseaseService = Depends(get_skin_disease_service)
):
    """
    Analyze pet skin disease from image
    
    Uses multiple models for comprehensive skin disease diagnosis:
    - Binary classification for disease detection
    - Multi-class classification for disease type
    - Segmentation for lesion area detection
    
    - **image**: Close-up image of the affected skin area
    - **pet_type**: Type of pet (dog/cat)
    - **include_segmentation**: Whether to include lesion area analysis
    """
    logger.info(f"Skin disease analysis request for {pet_type}")
    
    try:
        # Run skin disease diagnosis
        result = await service.diagnose_skin_condition(
            image=image,
            pet_type=pet_type,
            include_segmentation=include_segmentation
        )
        
        # Extract key fields for response model
        skin_response = SkinDiseaseResult(
            has_skin_disease=result['has_skin_disease'],
            disease_confidence=result['disease_confidence'],
            disease_type=result.get('disease_type'),
            disease_code=result.get('disease_code'),
            severity=result.get('severity'),
            affected_area_percentage=result.get('affected_area_percentage'),
            recommendations=result['recommendations'],
            requires_vet_visit=result['requires_vet_visit']
        )
        
        # Create message based on diagnosis
        if result['has_skin_disease']:
            message = f"피부질환이 감지되었습니다: {result.get('disease_type', '미분류')}"
        else:
            message = "정상적인 피부 상태입니다"
        
        return create_success_response(
            data=skin_response.model_dump(),
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error in skin disease analysis: {str(e)}")
        return create_error_response(
            message=f"피부질환 분석 중 오류가 발생했습니다: {str(e)}",
            code="500"
        )


@router.get("/skin/diseases", response_model=StandardResponse)
async def get_skin_diseases(
    pet_type: Optional[str] = None,
    service: SkinDiseaseService = Depends(get_skin_disease_service)
):
    """
    Get list of supported skin diseases
    
    Returns information about skin diseases that can be detected.
    
    - **pet_type**: Optional filter by pet type (dog/cat)
    """
    try:
        diseases = service.get_supported_diseases(pet_type)
        return create_success_response(
            data=diseases,
            message="지원되는 피부질환 목록입니다"
        )
    except Exception as e:
        logger.error(f"Error getting skin diseases: {str(e)}")
        return create_error_response(
            message=f"피부질환 목록 조회 중 오류가 발생했습니다: {str(e)}",
            code="500"
        )


@router.get("/skin/model-status", response_model=StandardResponse)
async def get_skin_model_status(
    service: SkinDiseaseService = Depends(get_skin_disease_service)
):
    """
    Get status of skin disease detection models
    
    Returns information about loaded models and their availability.
    """
    try:
        status = service.get_model_status()
        return create_success_response(
            data=status,
            message="피부질환 모델 상태입니다"
        )
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return create_error_response(
            message=f"모델 상태 조회 중 오류가 발생했습니다: {str(e)}",
            code="500"
        )


@router.get("/status", response_model=StandardResponse)
async def get_health_service_status(
    orchestrator_service: HealthDiagnosisOrchestrator = Depends(get_orchestrator)
):
    """
    Get status of all health diagnosis services
    
    Returns information about which services are available and operational.
    """
    try:
        available_services = orchestrator_service.get_available_services()
        
        # Get individual service status if needed
        service_details = {
            'eye_disease': {
                'available': available_services['eye_disease'],
                'model_loaded': eye_disease_service is not None
            },
            'body_condition': {
                'available': available_services['body_condition'],
                'model_loaded': bcs_service is not None
            },
            'skin_disease': {
                'available': available_services['skin_disease'],
                'model_loaded': skin_disease_service is not None
            }
        }
        
        status = {
            'orchestrator_ready': orchestrator is not None,
            'available_services': available_services,
            'service_details': service_details,
            'total_services': available_services['total_available']
        }
        
        return create_success_response(
            data=status,
            message=f"{available_services['total_available']}개의 건강 진단 서비스가 활성화되어 있습니다"
        )
    except Exception as e:
        logger.error(f"Error getting service status: {str(e)}")
        return create_error_response(
            message=f"서비스 상태 조회 중 오류가 발생했습니다: {str(e)}",
            code="500"
        )


@router.get("/guide", response_model=StandardResponse)
async def get_diagnosis_guide():
    """
    Get guide for taking photos for health diagnosis
    
    Returns comprehensive guide on how to take appropriate photos for each type of diagnosis.
    """
    guide = {
        "overview": "반려동물 건강 진단을 위한 사진 촬영 가이드입니다. 정확한 진단을 위해 아래 지침을 따라주세요.",
        "general_tips": [
            "밝은 자연광이나 충분한 조명 아래에서 촬영하세요",
            "흔들리지 않도록 카메라를 안정적으로 잡아주세요",
            "반려동물이 편안한 상태에서 촬영하세요",
            "여러 장을 촬영하여 가장 선명한 사진을 선택하세요"
        ],
        "diagnosis_specific_guides": {
            "eye_disease": {
                "title": "안구 질환 진단용 사진",
                "required_photos": 1,
                "instructions": [
                    "눈 주변을 클로즈업으로 촬영하세요",
                    "양쪽 눈을 각각 촬영하는 것이 좋습니다",
                    "눈꺼풀을 살짝 들어 눈 전체가 보이도록 하세요",
                    "플래시는 사용하지 마세요"
                ]
            },
            "body_condition": {
                "title": "체형 평가용 사진",
                "required_photos": "최소 3장, 이상적으로 13장",
                "instructions": [
                    "정면, 옆면(좌/우), 위에서 촬영하세요",
                    "전신이 모두 보이도록 적당한 거리에서 촬영하세요",
                    "털이 긴 경우 체형이 드러나도록 정리해주세요",
                    "서 있는 자세에서 촬영하는 것이 가장 좋습니다"
                ],
                "recommended_angles": [
                    "정면", "뒷면", "좌측면", "우측면", "위에서",
                    "좌측 대각선", "우측 대각선", "복부가 보이는 각도"
                ]
            },
            "skin_disease": {
                "title": "피부 질환 진단용 사진",
                "required_photos": 1,
                "instructions": [
                    "문제가 있는 피부 부위를 클로즈업으로 촬영하세요",
                    "털을 헤치고 피부가 잘 보이도록 하세요",
                    "병변의 크기를 가늠할 수 있도록 동전 등을 함께 놓고 촬영하면 좋습니다",
                    "여러 부위에 문제가 있다면 각각 촬영하세요"
                ]
            }
        },
        "comprehensive_diagnosis": {
            "title": "종합 건강 진단",
            "description": "가장 정확한 진단을 위해 다양한 각도의 사진을 제공해주세요",
            "recommended_photos": [
                "전신 사진 (여러 각도)",
                "눈 클로즈업",
                "피부에 이상이 있다면 해당 부위 클로즈업"
            ]
        }
    }
    
    return create_success_response(
        data=guide,
        message="건강 진단을 위한 사진 촬영 가이드입니다"
    )
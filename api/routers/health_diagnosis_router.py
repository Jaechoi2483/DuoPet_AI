"""
Health Diagnosis Router

This module provides endpoints for pet health diagnosis from images.
"""

from fastapi import APIRouter, UploadFile, File, Depends, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os
from pathlib import Path

from common.response import StandardResponse, create_success_response, create_error_response, ErrorCode
from common.logger import get_logger
from services.eye_disease_service import EyeDiseaseService
from services.bcs_service import BCSService
from services.skin_disease_service import SkinDiseaseService
from services.health_diagnosis_orchestrator import HealthDiagnosisOrchestrator

logger = get_logger(__name__)

router = APIRouter()

# --- Lazy Service Initialization ---
# 서비스들을 전역 변수로 선언하지만 초기화는 나중에 합니다
eye_disease_service = None
bcs_service = None
skin_disease_service = None
orchestrator = None

# 프로젝트 루트 경로만 미리 계산
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent

def initialize_service(service_type: str):
    """특정 서비스만 초기화합니다. 각 서비스는 독립적으로 로드됩니다."""
    global eye_disease_service, bcs_service, skin_disease_service
    
    if service_type == "eye":
        if eye_disease_service is not None:
            return eye_disease_service
            
        try:
            # 실제 존재하는 모델 파일 사용
            MODEL_PATH = project_root / "models" / "health_diagnosis" / "eye_disease" / "eye_disease_fixed.h5"
            # MODEL_PATH가 없으면 .keras 파일 시도
            if not MODEL_PATH.exists():
                MODEL_PATH = project_root / "models" / "health_diagnosis" / "eye_disease" / "best_grouped_model.keras"
            CLASS_MAP_PATH = project_root / "models" / "health_diagnosis" / "eye_disease" / "class_map.json"
            
            if MODEL_PATH.exists() and CLASS_MAP_PATH.exists():
                eye_disease_service = EyeDiseaseService(model_path=str(MODEL_PATH), class_map_path=str(CLASS_MAP_PATH))
                logger.info("EyeDiseaseService initialized successfully.")
                return eye_disease_service
            else:
                logger.error(f"Model or class map file not found. Searched paths:\n- {MODEL_PATH}\n- {CLASS_MAP_PATH}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize EyeDiseaseService: {e}", exc_info=True)
            return None
    
    elif service_type == "bcs":
        if bcs_service is not None:
            return bcs_service
            
        try:
            if os.getenv('SKIP_BCS_MODEL', 'false').lower() != 'true':
                BCS_MODEL_PATH = project_root / "models" / "health_diagnosis" / "bcs" / "bcs_efficientnet_v1.h5"
                BCS_CONFIG_PATH = project_root / "models" / "health_diagnosis" / "bcs" / "config.yaml"
                
                if BCS_MODEL_PATH.exists():
                    bcs_service = BCSService(
                        model_path=str(BCS_MODEL_PATH),
                        config_path=str(BCS_CONFIG_PATH) if BCS_CONFIG_PATH.exists() else None
                    )
                    logger.info("BCSService initialized successfully.")
                    return bcs_service
                else:
                    logger.error(f"BCS model not found at: {BCS_MODEL_PATH}")
                    return None
            else:
                logger.info("Skipping BCS model initialization (SKIP_BCS_MODEL=true)")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize BCSService: {e}", exc_info=True)
            return None
    
    elif service_type == "skin":
        if skin_disease_service is not None:
            return skin_disease_service
            
        try:
            skin_disease_service = SkinDiseaseService()
            logger.info("SkinDiseaseService initialized successfully.")
            return skin_disease_service
        except Exception as e:
            logger.error(f"Failed to initialize SkinDiseaseService: {e}", exc_info=True)
            return None
    
    else:
        logger.error(f"Unknown service type: {service_type}")
        return None

def get_eye_disease_service():
    service = initialize_service("eye")
    if not service:
        raise RuntimeError("EyeDiseaseService is not available. Check model paths and initialization logs.")
    return service

def get_bcs_service():
    service = initialize_service("bcs")
    if not service:
        raise RuntimeError("BCSService is not available. Check model paths and initialization logs.")
    return service

def get_skin_disease_service():
    service = initialize_service("skin")
    if not service:
        raise RuntimeError("SkinDiseaseService is not available. Check model paths and initialization logs.")
    return service


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


@router.post("/analyze-single", response_model=StandardResponse)
async def analyze_single_health(
    images: List[UploadFile] = File(..., description="Pet images for health analysis"),
    pet_type: str = Form("dog"),
    diagnosis_type: str = Form(..., description="Type of diagnosis: eye, bcs, or skin"),
    pet_age: Optional[float] = Form(None),
    pet_weight: Optional[float] = Form(None),
    pet_breed: Optional[str] = Form(None)
):
    """
    Single type pet health analysis to avoid TensorFlow graph conflicts
    
    - **images**: Images of the pet (number depends on diagnosis type)
    - **pet_type**: Type of pet (dog/cat)
    - **diagnosis_type**: Type of diagnosis to perform (eye/bcs/skin)
    - **pet_age**: Age of pet in years (optional)
    - **pet_weight**: Current weight in kg (optional)
    - **pet_breed**: Breed of the pet (optional)
    """
    logger.info(f"Single health analysis request for {diagnosis_type} with {len(images)} images")
    
    try:
        # Validate diagnosis type
        if diagnosis_type not in ["eye", "bcs", "skin"]:
            return create_error_response(
                error_code=ErrorCode.VALIDATION_ERROR,
                message=f"유효하지 않은 진단 유형입니다: {diagnosis_type}. eye, bcs, skin 중 하나를 선택하세요."
            )
        
        # Build pet info
        pet_info = {}
        if pet_age is not None:
            pet_info['age'] = pet_age
        if pet_weight is not None:
            pet_info['weight'] = pet_weight
            pet_info['weight_unit'] = 'kg'
        if pet_breed:
            pet_info['breed'] = pet_breed
        
        # Process based on diagnosis type
        if diagnosis_type == "eye":
            if len(images) < 1:
                return create_error_response(
                    error_code=ErrorCode.VALIDATION_ERROR,
                    message="안구 질환 진단을 위해서는 최소 1장의 사진이 필요합니다."
                )
            
            service = get_eye_disease_service()
            result = service.diagnose(images[0])  # Use first image
            
            return create_success_response(
                data={
                    "diagnosis_type": "eye",
                    "results": result,
                    "message": "안구질환 분석이 완료되었습니다."
                }
            )
            
        elif diagnosis_type == "bcs":
            if len(images) < 3:
                return create_error_response(
                    error_code=ErrorCode.VALIDATION_ERROR,
                    message="체형 평가를 위해서는 최소 3장의 사진이 필요합니다. 정확한 진단을 위해서는 13장을 권장합니다."
                )
            
            service = get_bcs_service()
            result = await service.assess_body_condition(
                images=images,
                pet_type=pet_type,
                pet_info=pet_info if pet_info else None
            )
            
            return create_success_response(
                data={
                    "diagnosis_type": "bcs",
                    "results": result,
                    "message": f"체형 평가가 완료되었습니다. BCS 점수: {result['bcs_score']}/9"
                }
            )
            
        elif diagnosis_type == "skin":
            if len(images) < 1:
                return create_error_response(
                    error_code=ErrorCode.VALIDATION_ERROR,
                    message="피부 질환 진단을 위해서는 최소 1장의 사진이 필요합니다."
                )
            
            service = get_skin_disease_service()
            result = await service.diagnose_skin_condition(
                image=images[0],  # Use first image
                pet_type=pet_type,
                include_segmentation=False  # 성능을 위해 기본값 False
            )
            
            # Create message based on diagnosis
            if result['has_skin_disease']:
                message = f"피부질환이 감지되었습니다: {result.get('disease_type', '미분류')}"
            else:
                message = "정상적인 피부 상태입니다"
            
            # 필요한 필드만 정리하여 반환
            cleaned_result = {
                "has_skin_disease": result.get("has_skin_disease", False),
                "disease_type": result.get("disease_type", "정상"),
                "confidence": round(result.get("confidence", 0.0), 4),
                "disease_confidence": round(result.get("confidence", 0.0), 4),  # 프론트엔드 호환성
                "status": result.get("status", "success"),
                "severity": "mild" if not result.get("has_skin_disease") else "moderate",
                "recommendations": ["정기적인 피부 관리를 유지하세요.", "청결한 환경을 유지하세요."] if not result.get("has_skin_disease") else ["수의사 상담을 권장합니다.", "피부 상태를 주의깊게 관찰하세요."],
                "requires_vet_visit": result.get("has_skin_disease", False)
            }
            
            # binary classification 확률 정보 추가 (정상/질환 확률)
            if "binary_classification" in result and result["binary_classification"]:
                binary_probs = result["binary_classification"]
                cleaned_result["binary_probabilities"] = {
                    "normal": round(binary_probs.get("normal", 0.0), 4),
                    "disease": round(binary_probs.get("disease", 0.0), 4)
                }
                
                # 경계선상의 경우 경고 추가
                disease_prob = binary_probs.get("disease", 0.0)
                if 0.3 < disease_prob < 0.5:
                    cleaned_result["warning"] = f"질환 가능성이 {round(disease_prob * 100, 1)}% 감지되었습니다. 주의깊게 관찰하시고 증상이 지속되면 수의사 상담을 권장합니다."
            
            # predictions가 있으면 포함 (간단히 정리)
            if "predictions" in result and result["predictions"]:
                cleaned_result["predictions"] = result["predictions"]
            
            return create_success_response(
                data={
                    "diagnosis_type": "skin",
                    "results": cleaned_result,
                    "message": message
                }
            )
            
    except Exception as e:
        logger.error(f"Error in single health analysis: {str(e)}", exc_info=True)
        return create_error_response(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message=f"{diagnosis_type} 분석 중 오류가 발생했습니다: {str(e)}"
        )


# Keep the old endpoint for backward compatibility but mark as deprecated
@router.post("/analyze", response_model=StandardResponse, deprecated=True)
async def analyze_health(
    images: List[UploadFile] = File(..., description="Pet images for comprehensive health analysis"),
    pet_type: str = Form("dog"),
    pet_age: Optional[float] = Form(None),
    pet_weight: Optional[float] = Form(None),
    pet_breed: Optional[str] = Form(None),
    diagnosis_types: Optional[str] = Form(None)
):
    """
    [DEPRECATED] This endpoint is deprecated due to TensorFlow graph conflicts.
    Please use /analyze-single endpoint instead.
    """
    return create_error_response(
        error_code=ErrorCode.METHOD_NOT_ALLOWED,
        message="이 엔드포인트는 더 이상 사용되지 않습니다. /analyze-single 엔드포인트를 사용해주세요. 진단 유형(eye, bcs, skin)을 명시적으로 선택해야 합니다."
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
            return create_error_response(error_code=ErrorCode.BAD_REQUEST, message="Invalid image format. Only JPG, JPEG, PNG are allowed.")

        # 서비스 호출하여 진단 수행
        result = service.diagnose(image)
        
        return create_success_response(
            data=result,
            metadata={"message": "안구질환 분석이 완료되었습니다."}
        )
        
    except Exception as e:
        logger.error(f"Error in eye disease analysis: {str(e)}")
        return create_error_response(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message=f"안구질환 분석 중 오류가 발생했습니다: {str(e)}"
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
            metadata={"message": f"체형 평가가 완료되었습니다. BCS 점수: {result['bcs_score']}/9"}
        )
        
    except Exception as e:
        logger.error(f"Error in BCS analysis: {str(e)}")
        return create_error_response(
            message=f"체형 평가 중 오류가 발생했습니다: {str(e)}",
            error_code=ErrorCode.UNKNOWN_ERROR
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
            metadata={"message": "BCS 촬영 가이드를 확인하세요"}
        )
    except Exception as e:
        logger.error(f"Error getting BCS guide: {str(e)}")
        return create_error_response(
            message=f"가이드 조회 중 오류가 발생했습니다: {str(e)}",
            error_code=ErrorCode.UNKNOWN_ERROR
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
            metadata={"message": message}
        )
        
    except Exception as e:
        logger.error(f"Error in skin disease analysis: {str(e)}")
        return create_error_response(
            message=f"피부질환 분석 중 오류가 발생했습니다: {str(e)}",
            error_code=ErrorCode.UNKNOWN_ERROR
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
            metadata={"message": "지원되는 피부질환 목록입니다"}
        )
    except Exception as e:
        logger.error(f"Error getting skin diseases: {str(e)}")
        return create_error_response(
            message=f"피부질환 목록 조회 중 오류가 발생했습니다: {str(e)}",
            error_code=ErrorCode.UNKNOWN_ERROR
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
            metadata={"message": "피부질환 모델 상태입니다"}
        )
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return create_error_response(
            message=f"모델 상태 조회 중 오류가 발생했습니다: {str(e)}",
            error_code=ErrorCode.UNKNOWN_ERROR
        )


@router.get("/diagnosis-types", response_model=StandardResponse)
async def get_diagnosis_types():
    """
    Get available diagnosis types and their requirements
    
    Returns information about each diagnosis type and image requirements.
    """
    diagnosis_types = {
        "types": [
            {
                "id": "eye",
                "name": "안구 질환 진단",
                "description": "백내장, 각막궤양 등 안구 질환을 진단합니다",
                "min_images": 1,
                "recommended_images": 2,
                "image_guide": "양쪽 눈을 각각 클로즈업으로 촬영하세요"
            },
            {
                "id": "bcs",
                "name": "체형 평가 (BCS)",
                "description": "반려동물의 체형 상태를 9단계로 평가합니다",
                "min_images": 3,
                "recommended_images": 13,
                "image_guide": "정면, 옆면, 위에서 등 다양한 각도로 전신을 촬영하세요"
            },
            {
                "id": "skin",
                "name": "피부 질환 진단",
                "description": "피부 질환의 유무와 종류를 진단합니다",
                "min_images": 1,
                "recommended_images": 1,
                "image_guide": "문제가 있는 피부 부위를 클로즈업으로 촬영하세요"
            }
        ],
        "usage_guide": {
            "endpoint": "/api/v1/health-diagnose/analyze-single",
            "method": "POST",
            "required_fields": {
                "images": "진단 유형에 따른 이미지 파일들",
                "diagnosis_type": "진단 유형 (eye, bcs, skin 중 하나)",
                "pet_type": "반려동물 종류 (dog 또는 cat)"
            },
            "optional_fields": {
                "pet_age": "나이 (년)",
                "pet_weight": "체중 (kg)",
                "pet_breed": "품종"
            }
        }
    }
    
    return create_success_response(
        data=diagnosis_types,
        metadata={"message": "사용 가능한 진단 유형 목록입니다"}
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
        metadata={"message": "건강 진단을 위한 사진 촬영 가이드입니다"}
    )
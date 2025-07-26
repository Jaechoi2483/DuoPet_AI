"""
Health Diagnosis Router - Simplified Version for Testing
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional, Dict, Any
import json
import random
from datetime import datetime

from common.response import create_success_response, create_error_response, ErrorCode
from common.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/analyze")
async def analyze_comprehensive(
    images: List[UploadFile] = File(...),
    pet_type: str = Form("dog"),
    pet_info: Optional[str] = Form(None)
):
    """
    종합 건강 진단 (더미 응답)
    """
    try:
        # 파일 검증
        for image in images:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid file type")
        
        # 더미 진단 결과 생성
        diagnosis_result = {
            "diagnosis_id": f"diag_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "pet_type": pet_type,
            "diagnosis": {
                "health_score": random.randint(70, 95),
                "health_status": random.choice(["excellent", "good", "fair"]),
                "summary": "전반적으로 건강한 상태입니다. 정기적인 관리를 계속해주세요.",
                "details": "AI 분석 결과, 반려동물의 전반적인 건강 상태가 양호합니다.",
                "confidence": random.uniform(0.8, 0.95),
                "recommendations": [
                    "규칙적인 운동과 적절한 식단 관리를 유지하세요",
                    "정기적인 건강 검진을 받으세요",
                    "충분한 수분 섭취를 확인하세요"
                ],
                "important_findings": [
                    "전반적으로 건강한 상태",
                    "체중 관리 지속 필요"
                ],
                "eye_diagnosis": {
                    "status": "정상",
                    "disease": None,
                    "confidence": 0.92,
                    "details": "안구 상태가 깨끗하고 건강합니다."
                },
                "bcs_evaluation": {
                    "bcs_score": 5,
                    "condition": "이상적",
                    "weight_status": "정상 체중",
                    "recommendations": ["현재 체중을 유지하세요"]
                },
                "skin_diagnosis": {
                    "status": "정상",
                    "diagnosis": None,
                    "severity": None,
                    "affected_area": 0,
                    "details": "피부 상태가 건강합니다."
                }
            }
        }
        
        return create_success_response(data=diagnosis_result)
        
    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=str(e)
        )


@router.post("/analyze/eye")
async def analyze_eye_disease(
    image: UploadFile = File(...),
    pet_type: str = Form("dog")
):
    """안구 질환 진단 (더미 응답)"""
    try:
        result = {
            "diagnosis_id": f"eye_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "disease": random.choice(["정상", "경미한 충혈", "결막염 의심"]),
            "confidence": random.uniform(0.8, 0.95),
            "severity": random.choice(["none", "mild", "moderate"]),
            "recommendations": [
                "정기적인 안구 검사를 받으세요",
                "눈 주변을 깨끗하게 유지하세요"
            ]
        }
        return create_success_response(data=result)
    except Exception as e:
        logger.error(f"Eye diagnosis error: {e}")
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))


@router.post("/analyze/bcs")
async def analyze_bcs(
    images: List[UploadFile] = File(...),
    pet_type: str = Form("dog"),
    pet_info: Optional[str] = Form(None)
):
    """체형 평가 (더미 응답)"""
    try:
        result = {
            "diagnosis_id": f"bcs_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "bcs_score": random.randint(4, 6),
            "condition": random.choice(["이상적", "약간 과체중", "정상"]),
            "confidence": random.uniform(0.8, 0.95),
            "weight_status": "정상 범위",
            "recommendations": [
                "현재 식단을 유지하세요",
                "규칙적인 운동을 계속하세요"
            ]
        }
        return create_success_response(data=result)
    except Exception as e:
        logger.error(f"BCS analysis error: {e}")
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))


@router.post("/analyze/skin")
async def analyze_skin_disease(
    image: UploadFile = File(...),
    pet_type: str = Form("dog")
):
    """피부 질환 진단 (더미 응답)"""
    try:
        result = {
            "diagnosis_id": f"skin_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "diagnosis": random.choice(["정상", "경미한 건조", "알레르기 의심"]),
            "confidence": random.uniform(0.8, 0.95),
            "severity": random.choice(["none", "mild", "moderate"]),
            "affected_area": random.uniform(0, 10),
            "recommendations": [
                "피부를 청결하게 유지하세요",
                "보습에 신경써주세요"
            ]
        }
        return create_success_response(data=result)
    except Exception as e:
        logger.error(f"Skin diagnosis error: {e}")
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))


@router.get("/status")
async def get_service_status():
    """서비스 상태 확인"""
    return create_success_response(data={
        "status": "operational",
        "version": "1.0.0",
        "models": {
            "eye_disease": {"status": "ready", "version": "test"},
            "bcs": {"status": "ready", "version": "test"},
            "skin_disease": {"status": "ready", "version": "test"}
        },
        "message": "테스트 모드로 실행 중입니다."
    })


@router.get("/guide")
async def get_photo_guide():
    """촬영 가이드"""
    return create_success_response(data={
        "eye": {
            "title": "안구 촬영 가이드",
            "instructions": [
                "밝은 곳에서 촬영하세요",
                "눈에 초점을 맞춰 선명하게 촬영하세요",
                "정면에서 촬영하는 것이 좋습니다"
            ]
        },
        "bcs": {
            "title": "체형 촬영 가이드",
            "instructions": [
                "전신이 보이도록 촬영하세요",
                "측면, 위, 정면 등 다양한 각도로 촬영하세요",
                "배경이 깔끔한 곳에서 촬영하세요"
            ]
        },
        "skin": {
            "title": "피부 촬영 가이드",
            "instructions": [
                "문제가 있는 부위를 가까이서 촬영하세요",
                "털을 헤치고 피부가 잘 보이도록 촬영하세요",
                "선명한 사진이 진단에 도움이 됩니다"
            ]
        }
    })


@router.get("/diseases")
async def get_supported_diseases():
    """지원하는 질병 목록"""
    return create_success_response(data={
        "eye_diseases": [
            {"code": "normal", "name": "정상", "description": "건강한 눈"},
            {"code": "conjunctivitis", "name": "결막염", "description": "눈의 염증"},
            {"code": "cataract", "name": "백내장", "description": "수정체 혼탁"}
        ],
        "skin_diseases": [
            {"code": "normal", "name": "정상", "description": "건강한 피부"},
            {"code": "dermatitis", "name": "피부염", "description": "피부 염증"},
            {"code": "allergy", "name": "알레르기", "description": "알레르기성 피부 반응"}
        ]
    })
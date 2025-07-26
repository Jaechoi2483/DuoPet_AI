"""
건강 진단 라우터 - 지연 로딩 버전
서비스를 처음 사용할 때 로드하여 서버 시작 속도 개선
"""

import os
import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
import json
import asyncio
from threading import Lock

from common.database_sqlalchemy import get_db
from api.models.health_record import HealthRecord
from api.schemas.health_schemas import HealthRecordCreate, HealthRecordResponse
from services.health_diagnosis_orchestrator import HealthDiagnosisOrchestrator
from services.eye_disease_service import EyeDiseaseService
from services.bcs_service import BCSService
from services.skin_disease_service import SkinDiseaseService
from api.models.user import User

# 설정
logger = logging.getLogger(__name__)
router = APIRouter()

# 환경 변수로 지연 로딩 제어
LAZY_LOAD_MODELS = os.getenv('LAZY_LOAD_MODELS', 'false').lower() == 'true'
SKIP_UNUSED_MODELS = os.getenv('SKIP_UNUSED_MODELS', 'false').lower() == 'true'

# 서비스 인스턴스 (지연 로딩)
_eye_disease_service: Optional[EyeDiseaseService] = None
_bcs_service: Optional[BCSService] = None
_skin_disease_service: Optional[SkinDiseaseService] = None
_orchestrator: Optional[HealthDiagnosisOrchestrator] = None

# 로딩 상태 추적
_loading_status = {
    'eye_disease': 'not_loaded',
    'bcs': 'not_loaded', 
    'skin_disease': 'not_loaded',
    'orchestrator': 'not_loaded'
}
_loading_lock = Lock()

def get_eye_disease_service() -> EyeDiseaseService:
    """안구 질환 서비스를 지연 로딩"""
    global _eye_disease_service
    
    if _eye_disease_service is None:
        with _loading_lock:
            if _eye_disease_service is None:  # Double-check
                _loading_status['eye_disease'] = 'loading'
                logger.info("Loading EyeDiseaseService...")
                
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    "..", "..", "models", "health_diagnosis", "eye_disease", "best_grouped_model.keras"
                )
                class_map_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    "..", "..", "models", "health_diagnosis", "eye_disease", "grouped_class_map.json"
                )
                
                _eye_disease_service = EyeDiseaseService(
                    model_path=model_path,
                    class_map_path=class_map_path
                )
                _loading_status['eye_disease'] = 'loaded'
                logger.info("EyeDiseaseService loaded successfully.")
    
    return _eye_disease_service

def get_bcs_service() -> BCSService:
    """BCS 서비스를 지연 로딩"""
    global _bcs_service
    
    if _bcs_service is None:
        with _loading_lock:
            if _bcs_service is None:  # Double-check
                _loading_status['bcs'] = 'loading'
                logger.info("Loading BCSService...")
                _bcs_service = BCSService()
                _loading_status['bcs'] = 'loaded'
                logger.info("BCSService loaded successfully.")
    
    return _bcs_service

def get_skin_disease_service() -> Optional[SkinDiseaseService]:
    """피부 질환 서비스를 지연 로딩 (선택적)"""
    global _skin_disease_service
    
    if SKIP_UNUSED_MODELS:
        logger.info("Skipping SkinDiseaseService loading (SKIP_UNUSED_MODELS=true)")
        return None
    
    if _skin_disease_service is None:
        with _loading_lock:
            if _skin_disease_service is None:  # Double-check
                _loading_status['skin_disease'] = 'loading'
                logger.info("Loading SkinDiseaseService...")
                _skin_disease_service = SkinDiseaseService()
                _loading_status['skin_disease'] = 'loaded'
                logger.info("SkinDiseaseService loaded successfully.")
    
    return _skin_disease_service

def get_orchestrator() -> HealthDiagnosisOrchestrator:
    """오케스트레이터를 지연 로딩"""
    global _orchestrator
    
    if _orchestrator is None:
        with _loading_lock:
            if _orchestrator is None:  # Double-check
                _loading_status['orchestrator'] = 'loading'
                logger.info("Loading HealthDiagnosisOrchestrator...")
                
                services = {}
                
                # 필수 서비스만 로드
                eye_disease_service = get_eye_disease_service()
                if eye_disease_service:
                    services['eye_disease'] = eye_disease_service
                
                bcs_service = get_bcs_service()
                if bcs_service:
                    services['bcs'] = bcs_service
                
                # 선택적 서비스
                if not SKIP_UNUSED_MODELS:
                    skin_disease_service = get_skin_disease_service()
                    if skin_disease_service:
                        services['skin_disease'] = skin_disease_service
                
                _orchestrator = HealthDiagnosisOrchestrator(services)
                _loading_status['orchestrator'] = 'loaded'
                logger.info("HealthDiagnosisOrchestrator loaded successfully.")
    
    return _orchestrator

# 즉시 로딩 옵션 (기존 동작 유지)
if not LAZY_LOAD_MODELS:
    logger.info("Eager loading all services (LAZY_LOAD_MODELS=false)")
    try:
        get_eye_disease_service()
        get_bcs_service()
        if not SKIP_UNUSED_MODELS:
            get_skin_disease_service()
        get_orchestrator()
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")

@router.get("/status")
async def get_loading_status():
    """서비스 로딩 상태 확인"""
    return {
        "lazy_load_enabled": LAZY_LOAD_MODELS,
        "skip_unused_models": SKIP_UNUSED_MODELS,
        "services": _loading_status
    }

@router.post("/analyze")
async def analyze_health(
    images: List[UploadFile] = File(...),
    category: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: User(MEMBER_ID="test_user"))  # 임시 사용자
):
    """
    반려동물 건강 상태 분석
    
    - **images**: 분석할 이미지 파일들 (최대 13개)
    - **category**: 진단 카테고리 (eye_disease, bcs, skin_disease, general)
    """
    try:
        # 오케스트레이터 가져오기 (필요시 로드)
        orchestrator = get_orchestrator()
        
        # 이미지 처리
        image_data_list = []
        for image in images:
            content = await image.read()
            image_data_list.append({
                'data': content,
                'filename': image.filename
            })
        
        # 분석 수행
        diagnosis_result = await orchestrator.analyze(
            images=image_data_list,
            category=category
        )
        
        # 결과 저장
        if diagnosis_result.get('success'):
            record = HealthRecordCreate(
                MEMBER_ID=current_user.MEMBER_ID,
                ANALYSIS_TYPE=category,
                RESULT_DATA=json.dumps(diagnosis_result, ensure_ascii=False),
                CONFIDENCE_SCORE=diagnosis_result.get('confidence', 0.0)
            )
            
            db_record = HealthRecord(**record.dict())
            db.add(db_record)
            db.commit()
            db.refresh(db_record)
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "record_id": db_record.RECORD_ID,
                    "results": diagnosis_result
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": diagnosis_result.get('message', 'Analysis failed')
                }
            )
    
    except Exception as e:
        logger.error(f"Health analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/records/{member_id}")
async def get_health_records(
    member_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: User(MEMBER_ID="test_user"))
):
    """사용자의 건강 진단 기록 조회"""
    try:
        records = db.query(HealthRecord).filter(
            HealthRecord.MEMBER_ID == member_id
        ).order_by(HealthRecord.CREATED_AT.desc()).all()
        
        return {
            "success": True,
            "count": len(records),
            "records": [
                HealthRecordResponse.from_orm(record) 
                for record in records
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching health records: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
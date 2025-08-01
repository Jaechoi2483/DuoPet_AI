"""
Behavior Analysis Router

This module provides endpoints for pet behavior analysis from videos.
"""

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import tempfile
import os
import uuid
import asyncio
import torch

from common.response import StandardResponse, create_success_response, create_error_response, ErrorCode
from common.logger import get_logger
from common.exceptions import ValidationError, ModelInferenceError
from services.behavior_analysis.model_manager import model_manager
from services.behavior_analysis.error_handler import error_handler

logger = get_logger(__name__)

# Use enhanced predictor if available
try:
    # Try to use enhanced predictor with pose estimation
    from services.behavior_analysis.enhanced_predict import enhanced_predictor, EnhancedBehaviorAnalysisPredictorSingleton
    predictor = enhanced_predictor
    BehaviorAnalysisPredictorSingleton = EnhancedBehaviorAnalysisPredictorSingleton
    logger.info("Using enhanced behavior analysis with pose estimation")
    logger.info(f"Enhanced predictor type: {type(enhanced_predictor).__name__}")
    logger.info(f"Enhanced predictor object: {enhanced_predictor}")
    logger.info(f"Enhanced predictor class: {enhanced_predictor.__class__.__name__}")
except Exception as e:
    import traceback
    logger.error(f"Failed to load enhanced predictor: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    from services.behavior_analysis.predict import predictor, BehaviorAnalysisPredictorSingleton
    logger.info(f"Standard predictor type: {type(predictor).__name__}")

router = APIRouter()

# 분석 상태를 저장할 임시 딕셔너리 (실제로는 Redis/DB 사용 권장)
analysis_status = {}


class BehaviorDetection(BaseModel):
    """Behavior detection result model"""
    behavior_type: str = Field(..., description="Type of behavior detected")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    is_abnormal: bool = Field(..., description="Whether behavior is abnormal")


class PoseMetrics(BaseModel):
    """Pose-based metrics for behavior quality assessment"""
    balance_index: float = Field(0.0, ge=0, le=1, description="Left-right symmetry score")
    stability_score: float = Field(0.0, ge=0, le=1, description="Posture stability score")
    movement_smoothness: float = Field(0.0, ge=0, le=1, description="Movement smoothness score")
    activity_level: float = Field(0.0, ge=0, le=1, description="Overall activity intensity")
    center_of_mass_stability: float = Field(0.0, ge=0, le=1, description="Center of mass stability")


class TemporalAnalysis(BaseModel):
    """Temporal analysis of behaviors"""
    behavior_durations: Dict[str, float] = Field(default_factory=dict, description="Duration of each behavior type")
    behavior_transitions: Dict[str, int] = Field(default_factory=dict, description="Behavior transition patterns")
    activity_periods: List[Dict[str, float]] = Field(default_factory=list, description="Active vs rest periods")


class BehaviorAnalysisResult(BaseModel):
    """Behavior analysis result model"""
    video_duration: float = Field(..., description="Total video duration in seconds")
    behaviors: List[BehaviorDetection] = Field(..., description="Detected behaviors")
    behavior_summary: Dict[str, int] = Field(..., description="Summary of behavior counts")
    abnormal_behaviors: List[BehaviorDetection] = Field(..., description="List of abnormal behaviors")
    analysis_id: str = Field(..., description="Unique analysis ID")
    pose_estimation_used: bool = Field(False, description="Whether pose estimation was used")
    pose_usage_percentage: float = Field(0.0, description="Percentage of frames with pose estimation")
    pose_metrics: Optional[PoseMetrics] = Field(None, description="Pose-based quality metrics")
    temporal_analysis: Optional[TemporalAnalysis] = Field(None, description="Temporal behavior analysis")

# API 엔드포인트 정의
@router.post("/analyze", response_model=StandardResponse)
async def analyze_behavior(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Pet video for behavior analysis"),
    pet_type: str = "dog",
    real_time: bool = False
):

    try:
        logger.info(f"Behavior analysis request for {pet_type}")
        
        # 비디오 파일 검증
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise ValidationError("video", "Unsupported video format. Please upload MP4, AVI, MOV, or MKV file.")
            
        # 파일 크기 검증 (최대 100MB)
        video.file.seek(0, 2)  # 파일 끝으로 이동
        file_size = video.file.tell()
        video.file.seek(0)  # 다시 처음으로
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise ValidationError("video", "Video file too large. Maximum size is 100MB.")
            
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
        analysis_id = f"analysis_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        # 긴 비디오는 백그라운드에서 처리
        if not real_time or file_size > 10 * 1024 * 1024:  # 10MB 이상
            # 초기 상태 저장
            analysis_status[analysis_id] = {
                "status": "processing",
                "progress": 0,
                "result": None,
                "error": None,
                "created_at": datetime.now(),
                "pet_type": pet_type
            }
            
            background_tasks.add_task(
                process_video_background, 
                tmp_file_path, 
                analysis_id, 
                pet_type
            )
            
            return create_success_response(
                data={
                    "analysis_id": analysis_id,
                    "status": "processing",
                    "message": "Video analysis started. Check status with analysis ID."
                }
            )
        
        # 실시간 처리 (작은 비디오)
        try:
            result = await asyncio.to_thread(
                predictor.analyze_video,
                tmp_file_path,
                pet_type
            )
            
            # 디버깅 로그 추가
            logger.info(f"Predictor type: {type(predictor).__name__}")
            logger.info(f"Result keys: {list(result.keys())}")
            logger.info(f"Has pose_metrics: {'pose_metrics' in result}")
            logger.info(f"Has temporal_analysis: {'temporal_analysis' in result}")
            
            # 결과 포맷팅
            behaviors = []
            for seq in result.get('behavior_sequences', []):
                behaviors.append(BehaviorDetection(
                    behavior_type=seq['behavior']['behavior'],
                    confidence=seq['behavior']['confidence'],
                    start_time=seq['time'],
                    end_time=seq['time'] + 1.0,  # 1초 단위로 가정
                    is_abnormal=seq['behavior']['is_abnormal']
                ))
                
            # Extract pose metrics if available
            pose_metrics = None
            if result.get('pose_metrics'):
                pose_metrics = PoseMetrics(**result['pose_metrics'])
            
            # Extract temporal analysis if available
            temporal_analysis = None
            if result.get('temporal_analysis'):
                temporal_analysis = TemporalAnalysis(**result['temporal_analysis'])
            
            analysis_result = BehaviorAnalysisResult(
                video_duration=result['video_duration'],
                behaviors=behaviors,
                behavior_summary=result['behavior_summary'],
                abnormal_behaviors=[
                    BehaviorDetection(
                        behavior_type=ab['behavior'],
                        confidence=ab['confidence'],
                        start_time=ab['time'],
                        end_time=ab['time'] + 1.0,
                        is_abnormal=True
                    ) for ab in result['abnormal_behaviors']
                ],
                analysis_id=analysis_id,
                pose_estimation_used=result.get('pose_estimation_used', False),
                pose_usage_percentage=result.get('pose_usage_percentage', 0.0),
                pose_metrics=pose_metrics,
                temporal_analysis=temporal_analysis
            )
            
            return create_success_response(data=analysis_result.model_dump())
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except ValidationError as e:
        return create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=str(e)
        )
    except Exception as e:
        logger.error(f"Behavior analysis failed: {str(e)}")
        return create_error_response(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Failed to analyze video. Please try again."
        )


@router.get("/analysis/{analysis_id}", response_model=StandardResponse)
async def get_analysis_status(analysis_id: str):
    """
    Get status of behavior analysis
    
    - **analysis_id**: Analysis ID from analyze endpoint
    """
    logger.info(f"Getting analysis status for: {analysis_id}")
    
    # 분석 상태 확인
    if analysis_id not in analysis_status:
        return create_error_response(
            error_code=ErrorCode.NOT_FOUND,
            message="Analysis ID not found"
        )
        
    status_info = analysis_status[analysis_id]
    
    # 에러가 있는 경우
    if status_info["error"]:
        return create_error_response(
            error_code=ErrorCode.MODEL_INFERENCE_ERROR,
            message=status_info["error"]
        )
        
    # 처리 중인 경우
    if status_info["status"] == "processing":
        return create_success_response(
            data={
                "analysis_id": analysis_id,
                "status": "processing",
                "progress": status_info["progress"],
                "message": status_info.get("message", "분석 진행 중...")
            }
        )
        
    # 완료된 경우
    if status_info["status"] == "completed" and status_info["result"]:
        result = status_info["result"]
        
        # 결과 포맷팅
        behaviors = []
        for seq in result.get('behavior_sequences', []):
            behaviors.append(BehaviorDetection(
                behavior_type=seq['behavior']['behavior'],
                confidence=seq['behavior']['confidence'],
                start_time=seq['time'],
                end_time=seq['time'] + 1.0,
                is_abnormal=seq['behavior']['is_abnormal']
            ).model_dump())
            
        # Extract pose metrics if available
        pose_metrics = None
        if result.get('pose_metrics'):
            pose_metrics = result['pose_metrics']
        
        # Extract temporal analysis if available
        temporal_analysis = None
        if result.get('temporal_analysis'):
            temporal_analysis = result['temporal_analysis']
        
        analysis_result = {
            "analysis_id": analysis_id,
            "status": "completed",
            "video_duration": result['video_duration'],
            "behaviors": behaviors[:50],  # 최대 50개만 반환
            "behavior_summary": result['behavior_summary'],
            "abnormal_behaviors": [
                {
                    "behavior_type": ab['behavior'],
                    "confidence": ab['confidence'],
                    "start_time": ab['time'],
                    "end_time": ab['time'] + 1.0,
                    "is_abnormal": True
                } for ab in result['abnormal_behaviors']
            ],
            "pose_estimation_used": result.get('pose_estimation_used', False),
            "pose_usage_percentage": result.get('pose_usage_percentage', 0.0),
            "pose_metrics": pose_metrics,
            "temporal_analysis": temporal_analysis
        }
        
        return create_success_response(data=analysis_result)
        
    return create_error_response(
        error_code=ErrorCode.INTERNAL_SERVER_ERROR,
        message="Unknown analysis status"
    )


@router.post("/test-pose", response_model=StandardResponse)
async def test_pose_estimation(
    image: UploadFile = File(..., description="Pet image for pose estimation test")
):
    """
    Test pose estimation on a single image
    
    - **image**: Image file (JPEG, PNG)
    """
    try:
        logger.info("Pose estimation test request")
        
        # Import enhanced predictor
        from services.behavior_analysis.enhanced_predict import enhanced_predictor
        
        # Validate image file
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise ValidationError("image", "Unsupported image format. Please upload JPEG or PNG file.")
        
        # Import numpy and cv2 if not already imported
        import numpy as np
        import cv2
        
        # Read image
        content = await image.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValidationError("image", "Failed to decode image")
        
        # First, detect pets to get bounding box
        detections = await asyncio.to_thread(
            enhanced_predictor.detect_pets,
            img,
            "catdog"
        )
        
        # Use first detection or full image
        if detections:
            bbox = detections[0]['bbox']
            pet_class = detections[0]['class']
        else:
            bbox = [0, 0, img.shape[1], img.shape[0]]
            pet_class = 'dog'  # Default
        
        # Extract pose using fallback
        if enhanced_predictor.use_fallback:
            pose_result = await asyncio.to_thread(
                enhanced_predictor.pose_fallback.estimate_keypoints_from_bbox,
                img,
                bbox
            )
        else:
            pose_result = await asyncio.to_thread(
                enhanced_predictor.extract_pose_only,
                img,
                bbox
            )
        
        # Check for errors
        if "error" in pose_result:
            return create_error_response(
                error_code=ErrorCode.MODEL_INFERENCE_ERROR,
                message=pose_result["error"]
            )
        
        # Format result
        response_data = {
            "num_keypoints": len(pose_result.get("keypoints", [])),
            "valid_keypoints": len(pose_result.get("valid_keypoints", [])),
            "keypoint_names": pose_result.get("keypoint_names", [])[:5],  # First 5 names
            "sample_keypoints": [
                {
                    "name": pose_result["keypoint_names"][i],
                    "position": pose_result["keypoints"][i],
                    "confidence": pose_result["confidence_scores"][i]
                }
                for i in pose_result.get("valid_keypoints", [])[:5]  # First 5 valid keypoints
            ],
            "method": pose_result.get("method", "fallback" if enhanced_predictor.use_fallback else "model"),
            "pet_detected": len(detections) > 0,
            "pet_class": pet_class if detections else None,
            "bbox": bbox
        }
        
        return create_success_response(data=response_data)
        
    except ValidationError as e:
        return create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=str(e)
        )
    except Exception as e:
        logger.error(f"Pose estimation test failed: {str(e)}")
        return create_error_response(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Failed to extract pose. Please try again."
        )


@router.get("/behaviors", response_model=StandardResponse)
async def get_supported_behaviors():
    """
    Get list of behaviors that can be detected
    """
    behaviors = [
        {
            "id": "walking",
            "name": "걷기",
            "description": "일반적인 걷는 행동",
            "is_normal": True
        },
        {
            "id": "running",
            "name": "달리기",
            "description": "빠르게 뛰는 행동",
            "is_normal": True
        },
        {
            "id": "eating",
            "name": "식사",
            "description": "사료나 간식을 먹는 행동",
            "is_normal": True
        },
        {
            "id": "sleeping",
            "name": "수면",
            "description": "잠을 자거나 쉬는 행동",
            "is_normal": True
        },
        {
            "id": "playing",
            "name": "놀이",
            "description": "장난감을 가지고 놀거나 활동적인 행동",
            "is_normal": True
        },
        {
            "id": "aggressive",
            "name": "공격성",
            "description": "으르렁거리거나 위협하는 등 공격적인 행동",
            "is_normal": False
        },
        {
            "id": "anxious",
            "name": "불안",
            "description": "꼬리를 내리거나 낑낑거리는 등 불안을 나타내는 행동",
            "is_normal": False
        }
    ]
    
    return create_success_response(data={"behaviors": behaviors})


@router.get("/model-status", response_model=StandardResponse)
async def get_model_status():
    """
    Get current model loading status and memory usage
    """
    try:
        # 모델 로드 상태
        loaded_models = model_manager.get_loaded_models()
        
        # GPU 메모리 상태 (PyTorch)
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        else:
            gpu_info = {"available": False}
        
        # Check pose estimation availability
        pose_info = {
            "available": hasattr(predictor, 'pose_adapter') and predictor.pose_adapter is not None,
            "model_type": "SuperAnimal-Quadruped" if hasattr(predictor, 'pose_adapter') else None,
            "keypoints": 39 if hasattr(predictor, 'pose_adapter') else 0
        }
        
        return create_success_response(data={
            "loaded_models": loaded_models,
            "gpu_info": gpu_info,
            "pose_estimation": pose_info
        })
    except Exception as e:
        logger.error(f"Failed to get model status: {str(e)}")
        return create_error_response(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Failed to get model status"
        )


@router.post("/reset-models", response_model=StandardResponse)
async def reset_models():
    """
    Reset all loaded models and clear memory
    """
    try:
        logger.info("Resetting all models via API")
        
        # 모든 모델 언로드
        model_manager.reset()
        
        # Predictor 싱글톤 리셋
        BehaviorAnalysisPredictorSingleton.reset()
        
        # 분석 상태 초기화
        analysis_status.clear()
        
        return create_success_response(data={
            "message": "All models have been reset successfully",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Failed to reset models: {str(e)}")
        return create_error_response(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message=f"Failed to reset models: {str(e)}"
        )


@router.get("/error-stats", response_model=StandardResponse)
async def get_error_statistics():
    """
    Get error statistics for debugging
    """
    try:
        stats = error_handler.get_error_stats()
        
        # Format statistics for response
        formatted_stats = []
        for error_key, data in stats.items():
            formatted_stats.append({
                "error_type": error_key,
                "count": data["count"],
                "last_occurrence": datetime.fromtimestamp(data["last_occurrence"]).isoformat() if data["last_occurrence"] > 0 else None,
                "time_since_last_seconds": data["time_since_last"] if data["time_since_last"] >= 0 else None
            })
        
        return create_success_response(data={
            "error_statistics": formatted_stats,
            "total_errors": sum(s["count"] for s in formatted_stats)
        })
    except Exception as e:
        logger.error(f"Failed to get error statistics: {str(e)}")
        return create_error_response(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Failed to get error statistics"
        )


@router.post("/reset-error-stats", response_model=StandardResponse)
async def reset_error_statistics():
    """
    Reset error statistics
    """
    try:
        error_handler.reset_error_stats()
        return create_success_response(data={
            "message": "Error statistics have been reset",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Failed to reset error statistics: {str(e)}")
        return create_error_response(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Failed to reset error statistics"
        )


async def process_video_background(video_path: str, analysis_id: str, pet_type: str):
    """Background task for video processing"""
    logger.info(f"Processing video in background: {analysis_id}, path: {video_path}")
    
    try:
        # 파일 존재 확인
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # 파일 크기 확인
        file_size = os.path.getsize(video_path)
        logger.info(f"Video file size: {file_size / (1024*1024):.2f} MB")
            
        # 진행 상황 업데이트 콜백
        def update_progress(progress: float):
            if analysis_id in analysis_status:
                analysis_status[analysis_id]["progress"] = int(progress)
                # 진행 상태에 따른 메시지 설정
                if progress < 10:
                    message = "비디오 로딩 중..."
                elif progress < 30:
                    message = "반려동물 탐지 중..."
                elif progress < 70:
                    message = "행동 분석 중..."
                elif progress < 90:
                    message = "결과 생성 중..."
                else:
                    message = "분석 마무리 중..."
                analysis_status[analysis_id]["message"] = message
                
        # 비디오 분석 수행
        logger.info(f"Starting video analysis for {analysis_id}")
        logger.info(f"Pet type: {pet_type}")
        
        try:
            logger.info(f"Using predictor: {type(predictor).__name__}")
            result = await asyncio.to_thread(
                predictor.analyze_video,
                video_path,
                pet_type,
                update_progress
            )
            logger.info(f"Analysis result received: {result is not None}")
            logger.info(f"Result keys: {list(result.keys()) if result else 'None'}")
            
            # 결과가 None이거나 비어있는 경우
            if not result:
                raise ValueError("Analysis returned empty result")
            
            # 결과 저장
            analysis_status[analysis_id].update({
                "status": "completed",
                "progress": 100,
                "result": result
            })
            
            logger.info(f"Video analysis completed: {analysis_id}")
            logger.info(f"Result summary: {result.get('behavior_summary', {})}")
            logger.info(f"Pose metrics included: {result.get('pose_metrics') is not None}")
            logger.info(f"Temporal analysis included: {result.get('temporal_analysis') is not None}")
            
        except Exception as analysis_error:
            logger.error(f"Error during analysis: {type(analysis_error).__name__}: {str(analysis_error)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # 모델 로드 관련 에러인 경우 모델 리셋 시도
            error_msg = str(analysis_error).lower()
            if any(keyword in error_msg for keyword in ['model', 'load', 'memory', 'cuda', 'gpu']):
                logger.warning("Model-related error detected, attempting to reset models")
                try:
                    # 모델 매니저 리셋
                    model_manager.reset()
                    # Predictor 싱글톤 리셋
                    BehaviorAnalysisPredictorSingleton.reset()
                    logger.info("Models reset successfully")
                except Exception as reset_error:
                    logger.error(f"Failed to reset models: {str(reset_error)}")
            
            raise
        
    except Exception as e:
        logger.error(f"Video analysis failed for {analysis_id}: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        analysis_status[analysis_id].update({
            "status": "failed",
            "error": f"{type(e).__name__}: {str(e)}"
        })
        
    finally:
        # 임시 파일 삭제
        try:
            if os.path.exists(video_path):
                logger.info(f"Deleting temporary file: {video_path}")
                os.unlink(video_path)
                logger.info(f"Temporary file deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete temp file: {str(e)}")
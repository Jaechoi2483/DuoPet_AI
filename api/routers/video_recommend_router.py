"""
Video Recommendation Router

This module provides endpoints for YouTube video recommendations.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from common.response import StandardResponse, create_success_response
from common.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class VideoRecommendRequest(BaseModel):
    """Video recommendation request model"""
    keywords: List[str] = Field(..., description="Keywords for video search")
    user_id: str = Field(..., description="User ID for personalization")
    category: Optional[str] = Field(None, description="Video category filter")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")


class VideoInfo(BaseModel):
    """Video information model"""
    video_id: str
    title: str
    description: str
    thumbnail_url: str
    channel_name: str
    duration: str
    view_count: int
    published_at: str


@router.post("/recommend", response_model=StandardResponse)
async def recommend_videos(request: VideoRecommendRequest):
    """
    Get YouTube video recommendations based on keywords
    
    Uses KeyBERT for keyword optimization and YouTube API for search.
    """
    logger.info(f"Video recommendation request from user: {request.user_id}")
    
    # TODO: Implement video recommendation logic
    # 1. Optimize keywords using KeyBERT
    # 2. Search YouTube API with optimized keywords
    # 3. Filter results based on category
    # 4. Rank videos by relevance
    
    return create_success_response(
        data={
            "videos": [],
            "total": 0,
            "keywords_used": request.keywords
        }
    )


@router.get("/categories", response_model=StandardResponse)
async def get_video_categories():
    """
    Get available video categories for filtering
    """
    categories = [
        {"id": "training", "name": "훈련", "description": "반려견 훈련 및 교육"},
        {"id": "health", "name": "건강", "description": "건강 관리, 질병 정보"},
        {"id": "nutrition", "name": "영양", "description": "사료, 간식, 영양 정보"},
        {"id": "grooming", "name": "미용", "description": "셀프 미용, 털 관리, 목욕 팁"},
        {"id": "entertainment", "name": "놀이", "description": "반려견과의 즐거운 놀이 및 활동"}
    ]
    
    return create_success_response(data={"categories": categories})
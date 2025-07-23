# api/routers/video_recommend_router.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional
from common.response import create_success_response, create_error_response, StandardResponse
from common.database_sqlalchemy import get_db
from services.video_recommend.recommender import recommend_youtube_videos_from_db_tags

router = APIRouter()

class VideoRecommendRequest(BaseModel):
    contentId: int = Field(..., description="추천 대상 게시글 ID")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="최대 추천 수")

@router.post("/recommend", response_model=StandardResponse)
def recommend_videos(request: VideoRecommendRequest, db: Session = Depends(get_db)):
    """
    게시글 ID를 기반으로 유튜브 영상을 추천합니다.
    - tags 기반 KeyBERT 키워드 추출
    - YouTube API 검색
    """

    try:
        videos = recommend_youtube_videos_from_db_tags(
            content_id=request.contentId,
            db=db,
            max_results=request.max_results
        )

        return create_success_response(data={
            "videos": videos,
            "total": len(videos)
        })

    except Exception as e:
        return create_error_response(message="추천 처리 중 오류 발생", detail=str(e))

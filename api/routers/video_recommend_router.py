# api/routers/video_recommend_router.py

from fastapi import APIRouter
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional

from common.response import create_success_response, create_error_response, StandardResponse, ErrorCode
from services.video_recommend.recommender import recommend_youtube_videos_from_db_tags
from services.video_recommend.db_models.board_entity import BoardEntity

# 추가된 부분: settings 불러오기 및 직접 Oracle 연결
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from common.config import get_settings

settings = get_settings()

# service_name 방식으로 직접 URL 생성
oracle_url = (
    f"oracle+oracledb://{settings.ORACLE_USER}:{settings.ORACLE_PASSWORD}"
    f"@{settings.ORACLE_HOST}:{settings.ORACLE_PORT}/?service_name={settings.ORACLE_SERVICE}"
)

print("🔍 최종 oracle_url:", oracle_url)  # 출력 확인용

# 엔진 및 세션 팩토리 생성
engine = create_engine(oracle_url, echo=True)
SessionLocal = sessionmaker(bind=engine)

router = APIRouter()

class VideoRecommendRequest(BaseModel):
    contentId: int = Field(..., description="추천 대상 게시글 ID")
    max_results: Optional[int] = Field(3, ge=1, le=20, description="최대 추천 수")


@router.post("/recommend", response_model=StandardResponse)
def recommend_videos(request: VideoRecommendRequest):
    """
    게시글 ID를 기반으로 유튜브 영상을 추천합니다.
    - 게시글의 tags 기반으로 유튜브 검색 키워드 생성 후 KeyBERT 키워드 추출
    - 추천 알고리즘 실행 후 YouTube API로 영상 리스트 반환
    """
    db: Session = SessionLocal()  # get_db() 대신 직접 생성한 세션 사용
    try:
        # 1. 게시글 조회
        content = db.query(BoardEntity).filter(BoardEntity.content_id == request.contentId).first()
        print("content 조회 완료")

        # 2. 태그 접근
        try:
            print("🔍 content.tags 접근 시도")
            tags = content.tags
            print("📌 tags 값:", tags)
        except Exception as e:
            import traceback
            print("❌ tags 접근 중 오류:", traceback.format_exc())
            raise e

        # 3. 추천 로직 실행 (내부적으로 KeyBERT + YouTube API 호출)
        videos = recommend_youtube_videos_from_db_tags(
            content_id=request.contentId,
            db=db,
            max_results=request.max_results,
        )

        # 4. 성공 응답 반환
        return create_success_response(data={
            "videos": videos,
            "total": len(videos)
        })

        # 예외 발생 시 표준 오류 응답 반환
    except Exception as e:
        import traceback
        print("🔥 추천 오류 발생:", traceback.format_exc())
        return create_error_response(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message="추천 처리 중 오류 발생",
            detail={"exception": str(e)}
        )
    finally:
        db.close()

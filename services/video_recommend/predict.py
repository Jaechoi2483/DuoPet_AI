# services/video_recommend/predict.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from services.video_recommend.db_models.content_entity import ContentEntity
from common.config import get_settings

"""
게시글 ID를 기반으로 tags 컬럼 조회
"""
# ✅ settings 초기화
settings = get_settings()

# ✅ 내 코드 안에서만 직접 service_name 방식으로 연결
oracle_url = f"oracle+oracledb://{settings.ORACLE_USER}:{settings.ORACLE_PASSWORD}@{settings.ORACLE_HOST}:{settings.ORACLE_PORT}/?service_name={settings.ORACLE_SERVICE}"

# ✅ 엔진 및 세션 팩토리 생성
engine = create_engine(oracle_url, echo=True)
SessionLocal = sessionmaker(bind=engine)

def fetch_tags_from_db(content_id: int) -> str:
    print("🔍 최종 oracle_url:", oracle_url)
    db: Session = SessionLocal()  # ← 핵심
    try:
        board = db.query(ContentEntity).filter(ContentEntity.content_id == content_id).first()
        return board.tags if board and board.tags else ""
    finally:
        db.close()
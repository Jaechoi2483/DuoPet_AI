# services/video_recommend/predict.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from services.video_recommend.db_models.board_entity import BoardEntity
from common.config import get_settings

"""
게시글 ID를 기반으로 tags (또는 category 포함) 조회
"""

# ✅ 환경변수 기반 Oracle 연결
settings = get_settings()
oracle_url = f"oracle+oracledb://{settings.ORACLE_USER}:{settings.ORACLE_PASSWORD}@{settings.ORACLE_HOST}:{settings.ORACLE_PORT}/?service_name={settings.ORACLE_SERVICE}"

# ✅ SQLAlchemy 세션 팩토리
engine = create_engine(oracle_url, echo=True)
SessionLocal = sessionmaker(bind=engine)

# ✅ 게시글 전체 정보 조회 (tags + category 등)
def fetch_board_from_db(content_id: int) -> BoardEntity | None:
    print("🔍 Oracle 연결 URL:", oracle_url)
    db: Session = SessionLocal()
    try:
        board = db.query(BoardEntity).filter(BoardEntity.content_id == content_id).first()
        return board
    finally:
        db.close()

# ✅ tags만 추출하고 싶을 경우
def fetch_tags_from_db(content_id: int) -> str:
    board = fetch_board_from_db(content_id)
    return board.tags if board and board.tags else ""
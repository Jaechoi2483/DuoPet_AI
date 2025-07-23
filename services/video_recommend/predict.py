from sqlalchemy.orm import Session
from services.video_recommend.models.content_entity import ContentEntity
from common.database_sqlalchemy import get_db_session  # 세션 유틸

def fetch_tags_from_db(content_id: int) -> str:
    """
    게시글 ID를 기반으로 tags 컬럼 조회
    """
    db: Session = get_db_session()
    try:
        board = db.query(ContentEntity).filter(ContentEntity.content_id == content_id).first()
        return board.tags if board and board.tags else ""
    finally:
        db.close()

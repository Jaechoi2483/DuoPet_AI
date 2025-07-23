# services/video_recommend/db_models/content_entity.py

from sqlalchemy import Column, Integer, String
from common.database_sqlalchemy import Base  # 기존 Base import 위치 맞춰줘

class ContentEntity(Base):
    __tablename__ = "CONTENT"  # Oracle DB의 실제 테이블명

    content_id = Column("CONTENT_ID", Integer, primary_key=True, index=True)
    user_id = Column("USER_ID", Integer)
    title = Column("TITLE", String(1000))
    content_body = Column("CONTENT_BODY", String)
    tags = Column("TAGS", String(100))

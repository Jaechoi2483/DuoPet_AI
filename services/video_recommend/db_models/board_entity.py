# services/video_recommend/db_models/board_entity.py

from sqlalchemy import Column, String, Integer, Date, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BoardEntity(Base):
    __tablename__ = "CONTENT"

    content_id = Column("CONTENT_ID", Integer, primary_key=True)
    user_id = Column("USER_ID", Integer, nullable=False)
    title = Column("TITLE", String(1000), nullable=False)
    content_body = Column("CONTENT_BODY", Text, nullable=False)
    content_type = Column("CONTENT_TYPE", String(50), nullable=False)
    category = Column("CATEGORY", String(200))
    tags = Column("TAGS", String(100))
    view_count = Column("VIEW_COUNT", Integer, default=0)
    like_count = Column("LIKE_COUNT", Integer, default=0)
    bookmark_count = Column("BOOKMARK_COUNT", Integer, default=0)
    rename_filename = Column("RENAME_FILENAME", String)
    original_filename = Column("ORIGINAL_FILENAME", String)
    created_at = Column("CREATED_AT", Date)
    update_at = Column("UPDATE_AT", Date)
    status = Column("STATUS", String, default="ACTIVE")


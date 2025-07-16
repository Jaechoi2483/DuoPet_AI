"""
SQLAlchemy models for API Key authentication
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, String, DateTime, Boolean, Integer, JSON, Index
from sqlalchemy.sql import func

from common.database_sqlalchemy import Base


class APIKey(Base):
    """API Key 테이블 모델"""
    __tablename__ = "api_keys"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # API Key 정보
    key_hash = Column(String(128), unique=True, nullable=False, index=True)
    key_prefix = Column(String(16), nullable=False, index=True)  # 키의 처음 8자 (검색용)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    
    # 권한 및 제한
    scopes = Column(JSON, default=list)  # ["read", "write", "admin"] 등
    rate_limit = Column(Integer, default=1000)  # 시간당 요청 제한
    ip_whitelist = Column(JSON, default=list)  # 허용된 IP 목록
    
    # 상태 및 만료
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # 메타데이터
    metadata_ = Column("metadata", JSON, default=dict)  # 추가 정보
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # 인덱스
    __table_args__ = (
        Index('idx_api_keys_active_expires', 'is_active', 'expires_at'),
        Index('idx_api_keys_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', prefix='{self.key_prefix}')>"
    
    def is_expired(self) -> bool:
        """API 키가 만료되었는지 확인"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def has_scope(self, scope: str) -> bool:
        """특정 권한을 가지고 있는지 확인"""
        return scope in (self.scopes or [])
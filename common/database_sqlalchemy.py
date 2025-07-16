"""
SQLAlchemy database configuration for Oracle DB
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import oracledb

from common.config import get_settings
from common.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Oracle thick mode 활성화 (필요한 경우)
# 주석 처리 - thick mode 충돌 방지
# try:
#     oracledb.init_oracle_client()
#     logger.info("Oracle thick mode initialized")
# except Exception as e:
#     logger.warning(f"Oracle thick mode not available, using thin mode: {e}")
#     # Thin mode will be used automatically

# Oracle 연결 문자열 생성 - 동기 버전 사용
# Service Name 방식 (SID가 아님)
DATABASE_URL = f"oracle+oracledb://{settings.ORACLE_USER}:{settings.ORACLE_PASSWORD}@{settings.ORACLE_HOST}:{settings.ORACLE_PORT}/?service_name={settings.ORACLE_SERVICE}"

# 동기 엔진 생성
engine = create_engine(
    DATABASE_URL,
    echo=False,  # SQL 로깅
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # 연결 상태 체크
)

# 세션 팩토리
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base 모델
Base = declarative_base()

# 의존성 주입용 함수
def get_db():
    """FastAPI 의존성: 데이터베이스 세션을 가져옵니다."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 데이터베이스 초기화
async def init_db():
    """데이터베이스 테이블 생성"""
    try:
        # 동기 함수를 비동기 컨텍스트에서 실행
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

# 데이터베이스 연결 종료
async def close_db():
    """데이터베이스 연결 종료"""
    engine.dispose()
    logger.info("Database connection closed")
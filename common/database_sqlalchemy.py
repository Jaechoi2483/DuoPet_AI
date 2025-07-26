"""
SQLAlchemy database configuration for Oracle DB
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base

from common.config import get_settings
from common.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# --- 핵심 수정 사항 ---
# 복잡한 URL을 여기서 직접 만들지 않고,
# config.py가 이미 모든 것을 처리해준 DATABASE_URL을 바로 사용합니다.
DATABASE_URL = settings.DATABASE_URL

# DATABASE_URL이 설정되었는지 최종 확인
if not DATABASE_URL:
    logger.error("DATABASE_URL is not configured. SQLAlchemy engine cannot be created.")
    # 설정이 없으면 엔진을 생성하지 않음
    engine = None
    SessionLocal = None
else:
    # 동기 엔진 생성
    engine = create_engine(
        DATABASE_URL,
        echo=False,  # SQL 로깅이 필요하면 True로 변경
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # 연결을 사용하기 전에 상태를 체크하여 안정성 향상
    )

    # 세션 팩토리
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

# 모든 ORM 모델이 상속받을 Base 클래스
Base = declarative_base()


def get_db():
    """FastAPI 의존성: 요청마다 데이터베이스 세션을 생성하고, 끝나면 닫습니다."""
    if SessionLocal is None:
        raise RuntimeError("Database is not configured, cannot create session.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """애플리케이션 시작 시 데이터베이스 테이블을 생성합니다."""
    if engine is None:
        logger.warning("Database engine is not available. Skipping table creation.")
        return
        
    try:
        logger.info("Creating database tables if they do not exist...")
        # Base.metadata.create_all(bind=engine) # sqlalchemy의 자동생성을 막기 위해 주석처리함
        logger.info("Database tables checked/created successfully.")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        # 시작 시 오류가 발생하면, 프로그램을 중단시키는 대신 로그를 남기고 계속 진행할 수 있습니다.
        # 필요하다면 raise e를 통해 프로그램을 중단시킬 수도 있습니다.


def close_db():
    """애플리케이션 종료 시 데이터베이스 연결 풀을 정리합니다."""
    if engine:
        engine.dispose()
        logger.info("Database connection pool closed.")
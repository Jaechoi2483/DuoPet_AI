"""
Database connection management for DuoPet AI Service

This module provides database connection utilities and connection pooling
for all supported databases (MongoDB, Oracle, Redis).
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
    import redis.asyncio as redis
import oracledb
import sys
print("="*50)
print(f"현재 사용 중인 파이썬: {sys.executable}")
print(f"oracledb 라이브러리 위치: {oracledb.__file__}")
print(f"oracledb 라이브러리 버전: {oracledb.__version__}")
print("="*50)
# from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
# import redis.asyncio as redis

from common.config import get_settings
from common.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# --- Global connection instances ---
_mongodb_client = None  # Optional[AsyncIOMotorClient]
_oracle_pool = None  # Optional[oracledb.Pool]
_redis_client = None  # Optional[redis.Redis]




# --- 통합 연결/종료 함수 ---

async def connect_to_databases():
    """
    애플리케이션 시작 시 모든 데이터베이스에 연결합니다.
    main.py의 startup 이벤트에서 이 함수를 호출하십시오.
    """
    global _mongodb_client, _oracle_pool, _redis_client

    # MongoDB는 사용하지 않음 - Oracle DB 사용

    # 2. Oracle 연결
    try:
        if _oracle_pool is None:
            logger.info("Connecting to Oracle Database...")
            _oracle_pool = oracledb.create_pool(
                user=settings.ORACLE_USER,
                password=settings.ORACLE_PASSWORD,
                dsn=f"{settings.ORACLE_HOST}:{settings.ORACLE_PORT}/{settings.ORACLE_SERVICE}",
                min=2, max=10 # 필요에 맞게 풀 사이즈 조절
            )
            logger.info("Successfully connected to Oracle Database")
    except Exception as e:
        logger.error(f"Failed to connect to Oracle: {e}")

    # Redis는 선택사항 - 현재 사용하지 않음


async def close_database_connections():
    """
    애플리케이션 종료 시 모든 데이터베이스 연결을 닫습니다.
    main.py의 shutdown 이벤트에서 이 함수를 호출하십시오.
    """
    global _mongodb_client, _oracle_pool, _redis_client

    if _mongodb_client:
        logger.info("Closing MongoDB connection...")
        _mongodb_client.close()

    if _oracle_pool:
        logger.info("Closing Oracle connection pool...")
        _oracle_pool.close()

    if _redis_client:
        logger.info("Closing Redis connection...")
        await _redis_client.close()


# --- 의존성 주입용 Getters ---

async def get_mongo_db():  # -> AsyncIOMotorDatabase
    """FastAPI 의존성: MongoDB는 사용하지 않습니다. Oracle DB를 사용하세요."""
    raise NotImplementedError("MongoDB is not used. Use Oracle DB instead.")

def get_oracle_connection() -> oracledb.Connection:
    """FastAPI 의존성: Oracle 커넥션 풀에서 커넥션을 가져옵니다."""
    if not _oracle_pool:
        raise RuntimeError("Oracle is not connected. Check startup logic.")
    connection = _oracle_pool.acquire()
    try:
        yield connection
    finally:
        _oracle_pool.release(connection)

async def get_redis_client():  # -> Optional[redis.Redis]
    """FastAPI 의존성: Redis 클라이언트 인스턴스를 가져옵니다."""
    return _redis_client


# --- 상태 확인(Health Check) 함수들 ---

async def check_mongodb_health() -> bool:
    """MongoDB는 사용하지 않습니다."""
    return False  # MongoDB 미사용

async def check_oracle_health() -> bool:
    """Oracle 상태를 확인합니다."""
    if not _oracle_pool:
        return False
    try:
        with _oracle_pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUAL")
        return True
    except Exception as e:
        logger.error(f"Oracle health check failed: {e}")
        return False

async def check_redis_health() -> bool:
    """Redis는 선택사항입니다."""
    return False  # Redis 미사용
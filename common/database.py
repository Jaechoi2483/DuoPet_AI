"""
Database connection management for DuoPet AI Service

This module provides database connection utilities and connection pooling
for all supported databases (MongoDB, Oracle, Redis).
"""

from typing import Optional
import oracledb
import sys
print("="*50)
print(f"현재 사용 중인 파이썬: {sys.executable}")
print(f"oracledb 라이브러리 위치: {oracledb.__file__}")
print(f"oracledb 라이브러리 버전: {oracledb.__version__}")
print("="*50)
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import redis.asyncio as redis

from common.config import get_settings
from common.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# --- Global connection instances ---
_mongodb_client: Optional[AsyncIOMotorClient] = None
_oracle_pool: Optional['oracledb.AsyncPool'] = None
_redis_client: Optional[redis.Redis] = None


# --- 통합 연결/종료 함수 ---

async def connect_to_databases():
    """
    애플리케이션 시작 시 모든 데이터베이스에 연결합니다.
    main.py의 startup 이벤트에서 이 함수를 호출하십시오.
    """
    global _mongodb_client, _oracle_pool, _redis_client

    # # 1. MongoDB 연결
    # try:
    #     if _mongodb_client is None:
    #         logger.info("Connecting to MongoDB...")
    #         _mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)
    #         await _mongodb_client.admin.command('ping')
    #         logger.info("Successfully connected to MongoDB")
    # except Exception as e:
    #     logger.error(f"Failed to connect to MongoDB: {e}")
    #     # 필요에 따라 raise e 를 사용하여 서버 시작을 중단할 수 있습니다.

    # 2. Oracle 연결
    try:
        if _oracle_pool is None:
            logger.info("Connecting to Oracle Database...")
            _oracle_pool = oracledb.create_pool_async(
                user=settings.ORACLE_USER,
                password=settings.ORACLE_PASSWORD,
                dsn=settings.ORACLE_DSN,
                min=2, max=10 # 필요에 맞게 풀 사이즈 조절
            )
            logger.info("Successfully connected to Oracle Database")
    except Exception as e:
        logger.error(f"Failed to connect to Oracle: {e}")

    # # 3. Redis 연결
    # try:
    #     if _redis_client is None:
    #         logger.info("Connecting to Redis...")
    #         _redis_client = await redis.from_url(settings.REDIS_URL, decode_responses=True)
    #         await _redis_client.ping()
    #         logger.info("Successfully connected to Redis")
    # except Exception as e:
    #     logger.error(f"Failed to connect to Redis: {e}")


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
        await _oracle_pool.close()

    if _redis_client:
        logger.info("Closing Redis connection...")
        await _redis_client.close()


# --- 의존성 주입용 Getters ---

async def get_mongo_db() -> AsyncIOMotorDatabase:
    """FastAPI 의존성: MongoDB 데이터베이스 인스턴스를 가져옵니다."""
    if not _mongodb_client:
        raise RuntimeError("MongoDB is not connected. Check startup logic.")
    return _mongodb_client[settings.MONGODB_DATABASE]

async def get_oracle_connection() -> oracledb.AsyncConnection:
    """FastAPI 의존성: Oracle 커넥션 풀에서 커넥션을 가져옵니다."""
    if not _oracle_pool:
        raise RuntimeError("Oracle is not connected. Check startup logic.")
    async with _oracle_pool.acquire() as connection:
        yield connection # with 구문으로 사용 후 자동 반환 보장

async def get_redis_client() -> Optional[redis.Redis]:
    """FastAPI 의존성: Redis 클라이언트 인스턴스를 가져옵니다."""
    return _redis_client


# --- 상태 확인(Health Check) 함수들 ---

async def check_mongodb_health() -> bool:
    """MongoDB 상태를 확인합니다."""
    if not _mongodb_client:
        return False
    try:
        await _mongodb_client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        return False

async def check_oracle_health() -> bool:
    """Oracle 상태를 확인합니다."""
    if not _oracle_pool:
        return False
    try:
        async with _oracle_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT 1 FROM DUAL")
        return True
    except Exception as e:
        logger.error(f"Oracle health check failed: {e}")
        return False

async def check_redis_health() -> bool:
    """Redis 상태를 확인합니다."""
    if not _redis_client:
        return False
    try:
        await _redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False
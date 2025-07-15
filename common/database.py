"""
Database connection management for DuoPet AI Service

This module provides database connection utilities and connection pooling.
"""

from typing import Optional
from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import redis.asyncio as redis

from common.config import get_settings
from common.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Global connection instances
_mongodb_client: Optional[AsyncIOMotorClient] = None
_redis_client: Optional[redis.Redis] = None


async def connect_mongodb() -> AsyncIOMotorClient:
    """
    Create and return MongoDB connection
    """
    global _mongodb_client
    
    if _mongodb_client is None:
        logger.info("Connecting to MongoDB...")
        _mongodb_client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            maxPoolSize=10,
            minPoolSize=1,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        try:
            await _mongodb_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    return _mongodb_client


async def connect_redis() -> redis.Redis:
    """
    Create and return Redis connection
    """
    global _redis_client
    
    if _redis_client is None:
        logger.info("Connecting to Redis...")
        _redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=10
        )
        
        # Test connection
        try:
            await _redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            # Redis is optional, so we don't raise here
            _redis_client = None
    
    return _redis_client


async def get_database() -> AsyncIOMotorDatabase:
    """
    Get MongoDB database instance
    """
    client = await connect_mongodb()
    return client[settings.MONGODB_DATABASE]


async def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client instance
    """
    return await connect_redis()


async def close_connections():
    """
    Close all database connections
    """
    global _mongodb_client, _redis_client
    
    if _mongodb_client:
        logger.info("Closing MongoDB connection...")
        _mongodb_client.close()
        _mongodb_client = None
    
    if _redis_client:
        logger.info("Closing Redis connection...")
        await _redis_client.close()
        _redis_client = None


# Health check functions
async def check_mongodb_health() -> bool:
    """
    Check MongoDB health
    """
    try:
        client = await connect_mongodb()
        await client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        return False


async def check_redis_health() -> bool:
    """
    Check Redis health
    """
    try:
        client = await connect_redis()
        if client:
            await client.ping()
            return True
        return False
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        return False
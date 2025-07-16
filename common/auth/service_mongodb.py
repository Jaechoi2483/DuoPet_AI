"""
API Key management service for DuoPet AI Service

This module provides functionality for creating, validating, and managing API keys.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import re

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError
import redis.asyncio as redis

from common.config import get_settings
from common.logger import get_logger
from common.monitoring import track_cache_operation, record_cache_hit, record_cache_miss
from common.exceptions import (
    AuthenticationError, 
    AuthorizationError, 
    NotFoundError,
    ValidationError
)
from .models import (
    APIKeyModel,
    APIKeyCreate,
    APIKeyUpdate,
    APIKeyResponse,
    APIKeyWithSecret,
    APIKeyValidation,
    APIKeyStatus,
    APIKeyScope
)

settings = get_settings()
logger = get_logger(__name__)


class APIKeyService:
    """Service for managing API keys"""
    
    # API key format: sk_live_<32 random characters>
    KEY_PREFIX = "sk_live_"
    KEY_LENGTH = 32
    KEY_PATTERN = re.compile(r"^sk_live_[a-zA-Z0-9]{32}$")
    
    # Cache settings
    CACHE_PREFIX = "api_key:"
    CACHE_TTL = 300  # 5 minutes
    
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        redis_client: Optional[redis.Redis] = None
    ):
        self.db = db
        self.collection = db.api_keys
        self.redis_client = redis_client
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for API keys"""
        # Unique index on key_id
        self.collection.create_index("key_id", unique=True)
        # Index on user_id for quick lookups
        self.collection.create_index("user_id")
        # Index on status for filtering
        self.collection.create_index("status")
        # Compound index for active keys by user
        self.collection.create_index([("user_id", 1), ("status", 1)])
    
    @staticmethod
    def _generate_api_key() -> tuple[str, str]:
        """
        Generate a new API key
        
        Returns:
            Tuple of (api_key, key_id)
        """
        # Generate random key
        random_part = secrets.token_urlsafe(KEY_LENGTH)[:KEY_LENGTH]
        api_key = f"{APIKeyService.KEY_PREFIX}{random_part}"
        
        # Generate key ID (first 8 chars of hash)
        key_id = f"key_{hashlib.sha256(api_key.encode()).hexdigest()[:8]}"
        
        return api_key, key_id
    
    @staticmethod
    def _hash_api_key(api_key: str) -> str:
        """
        Hash an API key for storage
        
        Args:
            api_key: The API key to hash
            
        Returns:
            Hashed API key
        """
        # Use SHA256 with salt from settings
        salt = settings.API_KEY_SALT.encode()
        return hashlib.sha256(salt + api_key.encode()).hexdigest()
    
    async def create_api_key(
        self,
        user_id: str,
        key_data: APIKeyCreate,
        organization_id: Optional[str] = None
    ) -> APIKeyWithSecret:
        """
        Create a new API key
        
        Args:
            user_id: ID of the user creating the key
            key_data: API key creation data
            organization_id: Optional organization ID
            
        Returns:
            Created API key with secret
        """
        # Generate API key
        api_key, key_id = self._generate_api_key()
        key_hash = self._hash_api_key(api_key)
        
        # Calculate expiration
        expires_at = None
        if key_data.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)
        
        # Create model
        api_key_model = APIKeyModel(
            key_id=key_id,
            key_hash=key_hash,
            name=key_data.name,
            description=key_data.description,
            user_id=user_id,
            organization_id=organization_id,
            scopes=key_data.scopes,
            allowed_ips=key_data.allowed_ips,
            allowed_origins=key_data.allowed_origins,
            expires_at=expires_at,
            rate_limit=key_data.rate_limit,
            metadata=key_data.metadata
        )
        
        try:
            # Insert into database
            result = await self.collection.insert_one(
                api_key_model.model_dump(by_alias=True, exclude={"id"})
            )
            api_key_model.id = str(result.inserted_id)
            
            logger.info(
                f"Created API key {key_id} for user {user_id}",
                extra={"key_id": key_id, "user_id": user_id}
            )
            
            # Return with secret (only time it's shown)
            return APIKeyWithSecret(
                api_key=api_key,
                key_id=key_id,
                name=api_key_model.name,
                description=api_key_model.description,
                scopes=api_key_model.scopes,
                status=api_key_model.status,
                created_at=api_key_model.created_at,
                expires_at=api_key_model.expires_at,
                last_used_at=api_key_model.last_used_at,
                usage_count=api_key_model.usage_count,
                rate_limit=api_key_model.rate_limit
            )
            
        except DuplicateKeyError:
            # Extremely unlikely but handle collision
            logger.error(f"API key ID collision: {key_id}")
            raise ValidationError("API key generation failed, please try again")
    
    async def validate_api_key(
        self,
        api_key: str,
        required_scopes: Optional[List[APIKeyScope]] = None,
        client_ip: Optional[str] = None,
        origin: Optional[str] = None
    ) -> APIKeyValidation:
        """
        Validate an API key
        
        Args:
            api_key: The API key to validate
            required_scopes: Optional required scopes
            client_ip: Optional client IP for validation
            origin: Optional origin for validation
            
        Returns:
            Validation result
        """
        # Check format
        if not self.KEY_PATTERN.match(api_key):
            return APIKeyValidation(
                valid=False,
                reason="Invalid API key format"
            )
        
        # Hash the key
        key_hash = self._hash_api_key(api_key)
        
        # Check cache first
        if self.redis_client:
            cached = await self._get_cached_key(key_hash)
            if cached:
                record_cache_hit("api_key")
                api_key_model = APIKeyModel(**cached)
            else:
                record_cache_miss("api_key")
                api_key_model = await self._get_key_from_db(key_hash)
                if api_key_model:
                    await self._cache_key(key_hash, api_key_model)
        else:
            api_key_model = await self._get_key_from_db(key_hash)
        
        # Key not found
        if not api_key_model:
            return APIKeyValidation(
                valid=False,
                reason="Invalid API key"
            )
        
        # Check if key is valid
        if not api_key_model.is_valid():
            reason = "API key is inactive"
            if api_key_model.status == APIKeyStatus.EXPIRED:
                reason = "API key has expired"
            elif api_key_model.status == APIKeyStatus.REVOKED:
                reason = "API key has been revoked"
            
            return APIKeyValidation(
                valid=False,
                key_id=api_key_model.key_id,
                reason=reason
            )
        
        # Check IP whitelist
        if client_ip and not api_key_model.is_ip_allowed(client_ip):
            return APIKeyValidation(
                valid=False,
                key_id=api_key_model.key_id,
                reason=f"IP address {client_ip} not allowed"
            )
        
        # Check origin
        if origin and not api_key_model.is_origin_allowed(origin):
            return APIKeyValidation(
                valid=False,
                key_id=api_key_model.key_id,
                reason=f"Origin {origin} not allowed"
            )
        
        # Check required scopes
        if required_scopes and not api_key_model.has_any_scope(required_scopes):
            return APIKeyValidation(
                valid=False,
                key_id=api_key_model.key_id,
                reason=f"Missing required scopes: {required_scopes}"
            )
        
        # Update usage
        await self._update_usage(api_key_model.key_id)
        
        return APIKeyValidation(
            valid=True,
            key_id=api_key_model.key_id,
            user_id=api_key_model.user_id,
            scopes=api_key_model.scopes,
            rate_limit=api_key_model.rate_limit,
            metadata=api_key_model.metadata
        )
    
    async def get_api_key(self, key_id: str) -> Optional[APIKeyResponse]:
        """
        Get API key by ID (without secret)
        
        Args:
            key_id: The key ID
            
        Returns:
            API key information or None
        """
        result = await self.collection.find_one({"key_id": key_id})
        
        if not result:
            return None
        
        api_key_model = APIKeyModel(**result)
        
        return APIKeyResponse(
            key_id=api_key_model.key_id,
            name=api_key_model.name,
            description=api_key_model.description,
            scopes=api_key_model.scopes,
            status=api_key_model.status,
            created_at=api_key_model.created_at,
            expires_at=api_key_model.expires_at,
            last_used_at=api_key_model.last_used_at,
            usage_count=api_key_model.usage_count,
            rate_limit=api_key_model.rate_limit
        )
    
    async def list_user_keys(
        self,
        user_id: str,
        include_inactive: bool = False
    ) -> List[APIKeyResponse]:
        """
        List all API keys for a user
        
        Args:
            user_id: The user ID
            include_inactive: Whether to include inactive keys
            
        Returns:
            List of API keys
        """
        query = {"user_id": user_id}
        if not include_inactive:
            query["status"] = APIKeyStatus.ACTIVE
        
        cursor = self.collection.find(query).sort("created_at", -1)
        keys = []
        
        async for doc in cursor:
            api_key_model = APIKeyModel(**doc)
            keys.append(APIKeyResponse(
                key_id=api_key_model.key_id,
                name=api_key_model.name,
                description=api_key_model.description,
                scopes=api_key_model.scopes,
                status=api_key_model.status,
                created_at=api_key_model.created_at,
                expires_at=api_key_model.expires_at,
                last_used_at=api_key_model.last_used_at,
                usage_count=api_key_model.usage_count,
                rate_limit=api_key_model.rate_limit
            ))
        
        return keys
    
    async def update_api_key(
        self,
        key_id: str,
        user_id: str,
        update_data: APIKeyUpdate
    ) -> Optional[APIKeyResponse]:
        """
        Update an API key
        
        Args:
            key_id: The key ID to update
            user_id: The user ID (for ownership verification)
            update_data: Update data
            
        Returns:
            Updated API key or None
        """
        # Build update dict
        update_dict = {
            "updated_at": datetime.utcnow()
        }
        
        if update_data.name is not None:
            update_dict["name"] = update_data.name
        if update_data.description is not None:
            update_dict["description"] = update_data.description
        if update_data.scopes is not None:
            update_dict["scopes"] = update_data.scopes
        if update_data.allowed_ips is not None:
            update_dict["allowed_ips"] = update_data.allowed_ips
        if update_data.allowed_origins is not None:
            update_dict["allowed_origins"] = update_data.allowed_origins
        if update_data.status is not None:
            update_dict["status"] = update_data.status
        if update_data.rate_limit is not None:
            update_dict["rate_limit"] = update_data.rate_limit
        if update_data.metadata is not None:
            update_dict["metadata"] = update_data.metadata
        
        # Update in database
        result = await self.collection.find_one_and_update(
            {"key_id": key_id, "user_id": user_id},
            {"$set": update_dict},
            return_document=True
        )
        
        if not result:
            return None
        
        # Clear cache
        if self.redis_client:
            api_key_model = APIKeyModel(**result)
            await self._clear_cache(api_key_model.key_hash)
        
        api_key_model = APIKeyModel(**result)
        
        logger.info(
            f"Updated API key {key_id}",
            extra={"key_id": key_id, "user_id": user_id}
        )
        
        return APIKeyResponse(
            key_id=api_key_model.key_id,
            name=api_key_model.name,
            description=api_key_model.description,
            scopes=api_key_model.scopes,
            status=api_key_model.status,
            created_at=api_key_model.created_at,
            expires_at=api_key_model.expires_at,
            last_used_at=api_key_model.last_used_at,
            usage_count=api_key_model.usage_count,
            rate_limit=api_key_model.rate_limit
        )
    
    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """
        Revoke an API key
        
        Args:
            key_id: The key ID to revoke
            user_id: The user ID (for ownership verification)
            
        Returns:
            True if revoked, False if not found
        """
        result = await self.collection.find_one_and_update(
            {"key_id": key_id, "user_id": user_id},
            {
                "$set": {
                    "status": APIKeyStatus.REVOKED,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result:
            # Clear cache
            if self.redis_client:
                api_key_model = APIKeyModel(**result)
                await self._clear_cache(api_key_model.key_hash)
            
            logger.info(
                f"Revoked API key {key_id}",
                extra={"key_id": key_id, "user_id": user_id}
            )
            return True
        
        return False
    
    async def _get_key_from_db(self, key_hash: str) -> Optional[APIKeyModel]:
        """Get API key from database by hash"""
        result = await self.collection.find_one({"key_hash": key_hash})
        return APIKeyModel(**result) if result else None
    
    async def _get_cached_key(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Get API key from cache"""
        if not self.redis_client:
            return None
        
        try:
            data = await self.redis_client.get(f"{self.CACHE_PREFIX}{key_hash}")
            if data:
                import json
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
        
        return None
    
    async def _cache_key(self, key_hash: str, api_key: APIKeyModel):
        """Cache API key"""
        if not self.redis_client:
            return
        
        try:
            import json
            data = json.dumps(api_key.model_dump(mode="json"))
            await self.redis_client.setex(
                f"{self.CACHE_PREFIX}{key_hash}",
                self.CACHE_TTL,
                data
            )
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
    
    async def _clear_cache(self, key_hash: str):
        """Clear API key from cache"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.delete(f"{self.CACHE_PREFIX}{key_hash}")
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
    
    async def _update_usage(self, key_id: str):
        """Update API key usage statistics"""
        await self.collection.update_one(
            {"key_id": key_id},
            {
                "$inc": {"usage_count": 1},
                "$set": {"last_used_at": datetime.utcnow()}
            }
        )
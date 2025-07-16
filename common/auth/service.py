"""
API Key management service for DuoPet AI Service - Oracle DB version

This module provides functionality for creating, validating, and managing API keys
using Oracle DB instead of MongoDB.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import re

from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from common.config import get_settings
from common.logger import get_logger
from common.exceptions import (
    AuthenticationError, 
    AuthorizationError, 
    NotFoundError,
    ValidationError
)
from .models import APIKeyResponse, APIKeyCreate, APIKeyUpdate
from .models_sqlalchemy import APIKey

settings = get_settings()
logger = get_logger(__name__)


class APIKeyService:
    """API Key management service using Oracle DB"""
    
    API_KEY_PREFIX = "duopet_"
    API_KEY_LENGTH = 32
    HASH_ALGORITHM = "sha256"
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    @staticmethod
    def _generate_api_key() -> str:
        """Generate a secure random API key"""
        return f"{APIKeyService.API_KEY_PREFIX}{secrets.token_urlsafe(APIKeyService.API_KEY_LENGTH)}"
    
    @staticmethod
    def _hash_api_key(api_key: str) -> str:
        """Hash an API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def _get_key_prefix(api_key: str) -> str:
        """Extract prefix from API key for identification"""
        return api_key[:16] if len(api_key) >= 16 else api_key
    
    @staticmethod
    def _validate_api_key_format(api_key: str) -> bool:
        """Validate API key format"""
        pattern = rf"^{APIKeyService.API_KEY_PREFIX}[A-Za-z0-9_-]{{{APIKeyService.API_KEY_LENGTH}}}$"
        return bool(re.match(pattern, api_key))
    
    async def create_api_key(self, key_data: APIKeyCreate) -> tuple[APIKeyResponse, str]:
        """
        Create a new API key
        
        Returns:
            Tuple of (APIKeyResponse, raw_api_key)
        """
        # Generate new API key
        raw_api_key = self._generate_api_key()
        key_hash = self._hash_api_key(raw_api_key)
        key_prefix = self._get_key_prefix(raw_api_key)
        
        # Calculate expiration
        expires_at = None
        if key_data.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)
        
        # Create database record
        db_key = APIKey(
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=key_data.name,
            description=key_data.description,
            scopes=key_data.scopes or ["read"],
            rate_limit=key_data.rate_limit or 1000,
            ip_whitelist=key_data.ip_whitelist or [],
            expires_at=expires_at,
            metadata_=key_data.metadata or {}
        )
        
        try:
            self.db.add(db_key)
            await self.db.commit()
            await self.db.refresh(db_key)
            
            logger.info(f"Created new API key: {db_key.name} (ID: {db_key.id})")
            
            response = APIKeyResponse(
                id=str(db_key.id),
                name=db_key.name,
                description=db_key.description,
                key_prefix=db_key.key_prefix,
                scopes=db_key.scopes,
                rate_limit=db_key.rate_limit,
                ip_whitelist=db_key.ip_whitelist,
                is_active=db_key.is_active,
                expires_at=db_key.expires_at,
                created_at=db_key.created_at,
                updated_at=db_key.updated_at,
                last_used_at=db_key.last_used_at
            )
            
            return response, raw_api_key
            
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Duplicate API key: {e}")
            raise ValidationError("API key generation failed. Please try again.")
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create API key: {e}")
            raise
    
    async def validate_api_key(
        self, 
        api_key: str, 
        required_scopes: Optional[List[str]] = None,
        ip_address: Optional[str] = None
    ) -> APIKey:
        """
        Validate an API key and check permissions
        
        Args:
            api_key: The raw API key to validate
            required_scopes: Optional list of required scopes
            ip_address: Optional IP address to check against whitelist
            
        Returns:
            APIKey model if valid
            
        Raises:
            AuthenticationError: If key is invalid
            AuthorizationError: If key lacks required permissions
        """
        # Validate format
        if not self._validate_api_key_format(api_key):
            raise AuthenticationError("Invalid API key format")
        
        # Hash the key for lookup
        key_hash = self._hash_api_key(api_key)
        
        # Find the key in database
        stmt = select(APIKey).where(APIKey.key_hash == key_hash)
        result = await self.db.execute(stmt)
        db_key = result.scalar_one_or_none()
        
        if not db_key:
            raise AuthenticationError("Invalid API key")
        
        # Check if active
        if not db_key.is_active:
            raise AuthenticationError("API key is inactive")
        
        # Check expiration
        if db_key.is_expired():
            raise AuthenticationError("API key has expired")
        
        # Check IP whitelist
        if db_key.ip_whitelist and ip_address:
            if ip_address not in db_key.ip_whitelist:
                raise AuthorizationError(f"IP address {ip_address} not authorized")
        
        # Check required scopes
        if required_scopes:
            missing_scopes = [s for s in required_scopes if not db_key.has_scope(s)]
            if missing_scopes:
                raise AuthorizationError(f"Missing required scopes: {', '.join(missing_scopes)}")
        
        # Update last used timestamp
        stmt = update(APIKey).where(APIKey.id == db_key.id).values(
            last_used_at=datetime.utcnow()
        )
        await self.db.execute(stmt)
        await self.db.commit()
        
        return db_key
    
    async def get_api_key(self, key_id: str) -> APIKeyResponse:
        """Get API key by ID"""
        stmt = select(APIKey).where(APIKey.id == int(key_id))
        result = await self.db.execute(stmt)
        db_key = result.scalar_one_or_none()
        
        if not db_key:
            raise NotFoundError(f"API key not found: {key_id}")
        
        return APIKeyResponse(
            id=str(db_key.id),
            name=db_key.name,
            description=db_key.description,
            key_prefix=db_key.key_prefix,
            scopes=db_key.scopes,
            rate_limit=db_key.rate_limit,
            ip_whitelist=db_key.ip_whitelist,
            is_active=db_key.is_active,
            expires_at=db_key.expires_at,
            created_at=db_key.created_at,
            updated_at=db_key.updated_at,
            last_used_at=db_key.last_used_at
        )
    
    async def list_api_keys(
        self, 
        skip: int = 0, 
        limit: int = 100,
        is_active: Optional[bool] = None
    ) -> List[APIKeyResponse]:
        """List all API keys with optional filtering"""
        stmt = select(APIKey).offset(skip).limit(limit)
        
        if is_active is not None:
            stmt = stmt.where(APIKey.is_active == is_active)
        
        result = await self.db.execute(stmt)
        db_keys = result.scalars().all()
        
        return [
            APIKeyResponse(
                id=str(key.id),
                name=key.name,
                description=key.description,
                key_prefix=key.key_prefix,
                scopes=key.scopes,
                rate_limit=key.rate_limit,
                ip_whitelist=key.ip_whitelist,
                is_active=key.is_active,
                expires_at=key.expires_at,
                created_at=key.created_at,
                updated_at=key.updated_at,
                last_used_at=key.last_used_at
            )
            for key in db_keys
        ]
    
    async def update_api_key(self, key_id: str, update_data: APIKeyUpdate) -> APIKeyResponse:
        """Update an existing API key"""
        # Get existing key
        stmt = select(APIKey).where(APIKey.id == int(key_id))
        result = await self.db.execute(stmt)
        db_key = result.scalar_one_or_none()
        
        if not db_key:
            raise NotFoundError(f"API key not found: {key_id}")
        
        # Update fields
        update_dict = update_data.dict(exclude_unset=True)
        
        if update_dict:
            stmt = update(APIKey).where(APIKey.id == int(key_id)).values(**update_dict)
            await self.db.execute(stmt)
            await self.db.commit()
            await self.db.refresh(db_key)
        
        logger.info(f"Updated API key: {db_key.name} (ID: {key_id})")
        
        return await self.get_api_key(key_id)
    
    async def delete_api_key(self, key_id: str) -> bool:
        """Delete an API key"""
        stmt = delete(APIKey).where(APIKey.id == int(key_id))
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        if result.rowcount == 0:
            raise NotFoundError(f"API key not found: {key_id}")
        
        logger.info(f"Deleted API key: {key_id}")
        return True
    
    async def cleanup_expired_keys(self) -> int:
        """Remove expired API keys"""
        stmt = delete(APIKey).where(
            and_(
                APIKey.expires_at.isnot(None),
                APIKey.expires_at < datetime.utcnow()
            )
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        count = result.rowcount
        if count > 0:
            logger.info(f"Cleaned up {count} expired API keys")
        
        return count
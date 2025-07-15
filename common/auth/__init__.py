"""
Authentication module for DuoPet AI Service
"""

from .models import (
    APIKeyStatus,
    APIKeyScope,
    APIKeyModel,
    APIKeyCreate,
    APIKeyUpdate,
    APIKeyResponse,
    APIKeyWithSecret,
    APIKeyValidation
)
from .service import APIKeyService
from .dependencies import (
    get_api_key,
    require_api_key,
    require_scopes,
    get_current_api_key
)

__all__ = [
    # Models
    "APIKeyStatus",
    "APIKeyScope",
    "APIKeyModel",
    "APIKeyCreate",
    "APIKeyUpdate",
    "APIKeyResponse",
    "APIKeyWithSecret",
    "APIKeyValidation",
    
    # Service
    "APIKeyService",
    
    # Dependencies
    "get_api_key",
    "require_api_key",
    "require_scopes",
    "get_current_api_key"
]
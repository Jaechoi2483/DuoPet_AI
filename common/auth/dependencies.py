"""
FastAPI authentication dependencies for DuoPet AI Service

This module provides reusable dependencies for API key authentication.
"""

from typing import Optional, List, Annotated
from functools import lru_cache

from fastapi import Depends, HTTPException, Security, Request, status
from fastapi.security import APIKeyHeader, APIKeyQuery, APIKeyCookie

from common.config import get_settings
from common.logger import get_logger
from common.exceptions import AuthenticationError, AuthorizationError
from common.database import get_database, get_redis_client
from .service import APIKeyService
from .models import APIKeyScope, APIKeyValidation

settings = get_settings()
logger = get_logger(__name__)

# API Key extractors
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)
api_key_cookie = APIKeyCookie(name="api_key", auto_error=False)


@lru_cache()
def get_api_key_service() -> APIKeyService:
    """Get cached API key service instance"""
    # This will be properly initialized in the actual implementation
    # For now, return a placeholder
    return None


async def get_api_key_from_request(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
    api_key_cookie: Optional[str] = Security(api_key_cookie),
) -> Optional[str]:
    """
    Extract API key from request (header, query, or cookie)
    
    Priority: Header > Query > Cookie
    """
    if api_key_header:
        return api_key_header
    elif api_key_query:
        return api_key_query
    elif api_key_cookie:
        return api_key_cookie
    return None


async def get_api_key(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key_from_request),
) -> APIKeyValidation:
    """
    Validate API key and return validation result
    
    This dependency validates the API key but doesn't enforce it.
    Use require_api_key for endpoints that must have authentication.
    """
    if not api_key:
        return APIKeyValidation(
            valid=False,
            reason="No API key provided"
        )
    
    # Get service
    db = await get_database()
    redis_client = await get_redis_client()
    service = APIKeyService(db, redis_client)
    
    # Get client info
    client_ip = request.client.host if request.client else None
    origin = request.headers.get("origin")
    
    # Validate key
    validation = await service.validate_api_key(
        api_key=api_key,
        client_ip=client_ip,
        origin=origin
    )
    
    # Store validation result in request state
    request.state.api_key_validation = validation
    
    return validation


async def require_api_key(
    validation: APIKeyValidation = Depends(get_api_key)
) -> APIKeyValidation:
    """
    Require valid API key for endpoint access
    
    Raises HTTPException if API key is invalid.
    """
    if not validation.valid:
        logger.warning(
            f"Invalid API key attempt: {validation.reason}",
            extra={"reason": validation.reason, "key_id": validation.key_id}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=validation.reason or "Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return validation


def require_scopes(*required_scopes: APIKeyScope):
    """
    Create a dependency that requires specific scopes
    
    Usage:
        @app.get("/admin", dependencies=[Depends(require_scopes(APIKeyScope.ADMIN))])
        async def admin_endpoint():
            return {"message": "Admin access granted"}
    """
    async def scope_checker(
        validation: APIKeyValidation = Depends(require_api_key)
    ) -> APIKeyValidation:
        """Check if API key has required scopes"""
        # Admin scope has access to everything
        if APIKeyScope.ADMIN in validation.scopes:
            return validation
        
        # Check if key has any required scope
        missing_scopes = [
            scope for scope in required_scopes 
            if scope not in validation.scopes
        ]
        
        if missing_scopes:
            logger.warning(
                f"Insufficient scopes for key {validation.key_id}",
                extra={
                    "key_id": validation.key_id,
                    "required_scopes": list(required_scopes),
                    "key_scopes": validation.scopes
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scopes: {missing_scopes}"
            )
        
        return validation
    
    return scope_checker


async def get_current_api_key(
    validation: APIKeyValidation = Depends(require_api_key)
) -> dict:
    """
    Get current API key information
    
    Returns a dictionary with key information for use in endpoints.
    """
    return {
        "key_id": validation.key_id,
        "user_id": validation.user_id,
        "scopes": validation.scopes,
        "rate_limit": validation.rate_limit,
        "metadata": validation.metadata
    }


# Convenience dependencies for specific scopes
RequireReadScope = Depends(require_scopes(APIKeyScope.READ))
RequireWriteScope = Depends(require_scopes(APIKeyScope.WRITE))
RequireDeleteScope = Depends(require_scopes(APIKeyScope.DELETE))
RequireAdminScope = Depends(require_scopes(APIKeyScope.ADMIN))

# Service-specific scope dependencies
RequireFaceLoginScope = Depends(require_scopes(APIKeyScope.FACE_LOGIN))
RequireChatbotScope = Depends(require_scopes(APIKeyScope.CHATBOT))
RequireVideoRecommendScope = Depends(require_scopes(APIKeyScope.VIDEO_RECOMMEND))
RequireHealthDiagnosisScope = Depends(require_scopes(APIKeyScope.HEALTH_DIAGNOSIS))
RequireBehaviorAnalysisScope = Depends(require_scopes(APIKeyScope.BEHAVIOR_ANALYSIS))


class OptionalAPIKey:
    """
    Dependency for endpoints that work with or without authentication
    
    Usage:
        @app.get("/public")
        async def public_endpoint(auth: OptionalAPIKey = Depends()):
            if auth.is_authenticated:
                return {"message": f"Hello {auth.user_id}"}
            return {"message": "Hello anonymous"}
    """
    
    def __init__(self, validation: APIKeyValidation = Depends(get_api_key)):
        self.validation = validation
        self.is_authenticated = validation.valid
        self.key_id = validation.key_id if validation.valid else None
        self.user_id = validation.user_id if validation.valid else None
        self.scopes = validation.scopes if validation.valid else []
        self.metadata = validation.metadata if validation.valid else {}
    
    def has_scope(self, scope: APIKeyScope) -> bool:
        """Check if authenticated user has specific scope"""
        if not self.is_authenticated:
            return False
        return scope in self.scopes or APIKeyScope.ADMIN in self.scopes


# Type annotations for cleaner function signatures
CurrentAPIKey = Annotated[dict, Depends(get_current_api_key)]
ValidatedAPIKey = Annotated[APIKeyValidation, Depends(require_api_key)]
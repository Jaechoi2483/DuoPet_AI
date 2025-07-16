"""
Authentication and API Key management endpoints for DuoPet AI Service

This module provides endpoints for creating, managing, and revoking API keys.
"""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status, Request
from fastapi.responses import JSONResponse

from common.auth.dependencies import CurrentAPIKey
from common.response import create_success_response, create_error_response, ErrorCode
from common.logger import get_logger
from common.database import get_mongo_db, get_redis_client
from common.exceptions import NotFoundError, ValidationError, AuthorizationError
from common.auth import (
    APIKeyService,
    APIKeyCreate,
    APIKeyUpdate,
    APIKeyResponse,
    APIKeyWithSecret,
    APIKeyStatus,
    APIKeyScope,
    get_current_api_key
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"}
    }
)

logger = get_logger(__name__)


@router.post(
    "/keys",
    response_model=APIKeyWithSecret,
    status_code=status.HTTP_201_CREATED,
    summary="Create new API key",
    description="Create a new API key with specified permissions and settings"
)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: CurrentAPIKey = Depends(get_current_api_key),
    request: Request = None
) -> JSONResponse:
    """
    Create a new API key.

    Requires admin scope or appropriate permissions.
    Returns the full API key only once - it cannot be retrieved again.
    """
    try:
        # Check permissions
        if APIKeyScope.ADMIN not in current_user["scopes"]:
            # Non-admin users can only create keys with their own scopes or less
            for scope in key_data.scopes:
                if scope not in current_user["scopes"]:
                    raise AuthorizationError(
                        f"Cannot grant scope '{scope}' - you don't have this permission"
                    )

        # Get database and service
        db = await get_mongo_db()
        redis_client = await get_redis_client()
        service = APIKeyService(db, redis_client)

        # Create API key
        api_key = await service.create_api_key(
            user_id=current_user["user_id"],
            key_data=key_data,
            organization_id=current_user.get("metadata", {}).get("organization_id")
        )

        logger.info(
            f"Created API key {api_key.key_id} for user {current_user['user_id']}",
            extra={
                "key_id": api_key.key_id,
                "user_id": current_user["user_id"],
                "scopes": key_data.scopes,
                "request_id": getattr(request.state, "request_id", None)
            }
        )

        return create_success_response(
            data=api_key.model_dump(),
            message="API key created successfully. Save the key securely - it won't be shown again."
        )

    except AuthorizationError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Failed to create API key: {str(e)}",
            extra={"user_id": current_user["user_id"]},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )


@router.get(
    "/keys",
    response_model=List[APIKeyResponse],
    summary="List API keys",
    description="List all API keys for the current user"
)
async def list_api_keys(
    include_inactive: bool = Query(False, description="Include revoked/expired keys"),
    current_user: CurrentAPIKey = Depends(get_current_api_key),
    request: Request = None
) -> JSONResponse:
    """
    List all API keys for the authenticated user.

    By default, only active keys are returned.
    """
    try:
        # Get database and service
        db = await get_mongo_db()
        redis_client = await get_redis_client()
        service = APIKeyService(db, redis_client)

        # Get user's keys
        keys = await service.list_user_keys(
            user_id=current_user["user_id"],
            include_inactive=include_inactive
        )

        logger.info(
            f"Listed {len(keys)} API keys for user {current_user['user_id']}",
            extra={
                "user_id": current_user["user_id"],
                "count": len(keys),
                "include_inactive": include_inactive,
                "request_id": getattr(request.state, "request_id", None)
            }
        )

        return create_success_response(
            data=[key.model_dump() for key in keys],
            metadata={"total": len(keys), "include_inactive": include_inactive}
        )

    except Exception as e:
        logger.error(
            f"Failed to list API keys: {str(e)}",
            extra={"user_id": current_user["user_id"]},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys"
        )


@router.get(
    "/keys/{key_id}",
    response_model=APIKeyResponse,
    summary="Get API key details",
    description="Get details of a specific API key"
)
async def get_api_key(
    key_id: str,
    current_user: CurrentAPIKey = Depends(get_current_api_key),
    request: Request = None
) -> JSONResponse:
    """
    Get details of a specific API key.

    Users can only access their own keys unless they have admin scope.
    """
    try:
        # Get database and service
        db = await get_mongo_db()
        redis_client = await get_redis_client()
        service = APIKeyService(db, redis_client)

        # Get key details
        key = await service.get_api_key(key_id)

        if not key:
            raise NotFoundError(f"API key {key_id} not found")

        # Check permissions
        if APIKeyScope.ADMIN not in current_user["scopes"] and key.user_id != current_user["user_id"]:
             raise NotFoundError(f"API key {key_id} not found or access denied")

        logger.info(
            f"Retrieved API key details for {key_id}",
            extra={
                "key_id": key_id,
                "user_id": current_user["user_id"],
                "request_id": getattr(request.state, "request_id", None)
            }
        )

        return create_success_response(data=key.model_dump())

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Failed to get API key: {str(e)}",
            extra={"key_id": key_id, "user_id": current_user["user_id"]},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API key"
        )


@router.patch(
    "/keys/{key_id}",
    response_model=APIKeyResponse,
    summary="Update API key",
    description="Update API key settings (name, description, permissions, etc.)"
)
async def update_api_key(
    key_id: str,
    update_data: APIKeyUpdate,
    current_user: CurrentAPIKey = Depends(get_current_api_key),
    request: Request = None
) -> JSONResponse:
    """
    Update an API key's settings.

    Users can only update their own keys.
    Changing scopes requires appropriate permissions.
    """
    try:
        # Check scope permissions if updating scopes
        if update_data.scopes is not None and APIKeyScope.ADMIN not in current_user["scopes"]:
            for scope in update_data.scopes:
                if scope not in current_user["scopes"]:
                    raise AuthorizationError(
                        f"Cannot grant scope '{scope}' - you don't have this permission"
                    )

        # Get database and service
        db = await get_mongo_db()
        redis_client = await get_redis_client()
        service = APIKeyService(db, redis_client)

        # Update key
        updated_key = await service.update_api_key(
            key_id=key_id,
            user_id=current_user["user_id"],
            update_data=update_data
        )

        if not updated_key:
            raise NotFoundError(f"API key {key_id} not found or access denied")

        logger.info(
            f"Updated API key {key_id}",
            extra={
                "key_id": key_id,
                "user_id": current_user["user_id"],
                "updates": update_data.model_dump(exclude_unset=True),
                "request_id": getattr(request.state, "request_id", None)
            }
        )

        return create_success_response(
            data=updated_key.model_dump(),
            message="API key updated successfully"
        )

    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if isinstance(e, NotFoundError) else status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Failed to update API key: {str(e)}",
            extra={"key_id": key_id, "user_id": current_user["user_id"]},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update API key"
        )


@router.delete(
    "/keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
    description="Revoke an API key, making it permanently unusable"
)
async def revoke_api_key(
    key_id: str,
    current_user: CurrentAPIKey = Depends(get_current_api_key),
    request: Request = None
):
    """
    Revoke an API key.

    This action is permanent and cannot be undone.
    Users can only revoke their own keys.
    """
    try:
        # Get database and service
        db = await get_mongo_db()
        redis_client = await get_redis_client()
        service = APIKeyService(db, redis_client)

        # Revoke key
        success = await service.revoke_api_key(
            key_id=key_id,
            user_id=current_user["user_id"]
        )

        if not success:
            raise NotFoundError(f"API key {key_id} not found or access denied")

        logger.info(
            f"Revoked API key {key_id}",
            extra={
                "key_id": key_id,
                "user_id": current_user["user_id"],
                "request_id": getattr(request.state, "request_id", None)
            }
        )

        return None  # 204 No Content

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Failed to revoke API key: {str(e)}",
            extra={"key_id": key_id, "user_id": current_user["user_id"]},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key"
        )


@router.get(
    "/me",
    summary="Get current API key info",
    description="Get information about the currently authenticated API key"
)
async def get_current_key_info(
    current_user: CurrentAPIKey = Depends(get_current_api_key),
    request: Request = None
) -> JSONResponse:
    """
    Get information about the currently authenticated API key.

    This is useful for debugging and verifying permissions.
    """
    return create_success_response(
        data={
            "key_id": current_user["key_id"],
            "user_id": current_user["user_id"],
            "scopes": current_user["scopes"],
            "rate_limit": current_user["rate_limit"],
            "metadata": current_user.get("metadata", {})
        }
    )


@router.get(
    "/scopes",
    summary="List available scopes",
    description="Get a list of all available API scopes and their descriptions"
)
async def list_scopes(request: Request = None) -> JSONResponse:
    """
    List all available API scopes.
    
    This endpoint is publicly accessible to help with API integration.
    """
    scopes = [
        {
            "scope": scope.value,
            "name": scope.name,
            "description": _get_scope_description(scope)
        }
        for scope in APIKeyScope
    ]
    
    return create_success_response(
        data=scopes,
        metadata={"total": len(scopes)}
    )


def _get_scope_description(scope: APIKeyScope) -> str:
    """Get human-readable description for each scope"""
    descriptions = {
        APIKeyScope.READ: "Read access to resources",
        APIKeyScope.WRITE: "Write access to resources",
        APIKeyScope.DELETE: "Delete access to resources",
        APIKeyScope.ADMIN: "Full administrative access",
        APIKeyScope.FACE_LOGIN: "Access to face login service",
        APIKeyScope.CHATBOT: "Access to chatbot service",
        APIKeyScope.VIDEO_RECOMMEND: "Access to video recommendation service",
        APIKeyScope.HEALTH_DIAGNOSIS: "Access to health diagnosis service",
        APIKeyScope.BEHAVIOR_ANALYSIS: "Access to behavior analysis service"
    }
    return descriptions.get(scope, "No description available")
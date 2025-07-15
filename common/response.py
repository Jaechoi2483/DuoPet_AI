"""
Standard API Response Models and Utilities

This module provides standardized response formats for all API endpoints
following the DuoPet API specification.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List, Union, TypeVar, Generic
from datetime import datetime
from enum import Enum


class ErrorCode(str, Enum):
    """Standard error codes for API responses"""
    # General errors
    UNKNOWN_ERROR = "E00_UNKNOWN_ERROR"
    VALIDATION_ERROR = "E01_VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "E02_AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "E03_AUTHORIZATION_ERROR"
    NOT_FOUND = "E04_NOT_FOUND"
    METHOD_NOT_ALLOWED = "E05_METHOD_NOT_ALLOWED"
    TIMEOUT_ERROR = "E06_TIMEOUT_ERROR"
    RATE_LIMIT_ERROR = "E07_RATE_LIMIT_ERROR"
    
    # File/Media errors
    FILE_NOT_FOUND = "E10_FILE_NOT_FOUND"
    FILE_TOO_LARGE = "E11_FILE_TOO_LARGE"
    INVALID_FILE_FORMAT = "E12_INVALID_FILE_FORMAT"
    IMAGE_PROCESSING_ERROR = "E13_IMAGE_PROCESSING_ERROR"
    VIDEO_PROCESSING_ERROR = "E14_VIDEO_PROCESSING_ERROR"
    
    # AI Model errors
    MODEL_NOT_LOADED = "E20_MODEL_NOT_LOADED"
    MODEL_INFERENCE_ERROR = "E21_MODEL_INFERENCE_ERROR"
    INVALID_MODEL_INPUT = "E22_INVALID_MODEL_INPUT"
    MODEL_TIMEOUT = "E23_MODEL_TIMEOUT"
    
    # External service errors
    EXTERNAL_API_ERROR = "E30_EXTERNAL_API_ERROR"
    DATABASE_ERROR = "E31_DATABASE_ERROR"
    CACHE_ERROR = "E32_CACHE_ERROR"
    
    # Business logic errors
    FACE_NOT_DETECTED = "E40_FACE_NOT_DETECTED"
    MULTIPLE_FACES_DETECTED = "E41_MULTIPLE_FACES_DETECTED"
    LOW_CONFIDENCE_RESULT = "E42_LOW_CONFIDENCE_RESULT"
    NO_DISEASE_DETECTED = "E43_NO_DISEASE_DETECTED"
    INVALID_BEHAVIOR_SEQUENCE = "E44_INVALID_BEHAVIOR_SEQUENCE"


class ErrorDetail(BaseModel):
    """Error detail structure"""
    code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

T = TypeVar('T')


class StandardResponse(BaseModel, Generic[T]):
    """
    Standard API response format for all endpoints

    All API responses follow this structure to ensure consistency
    across the entire DuoPet AI service.
    """
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[T] = Field(None, description="Response data if successful")
    error: Optional[ErrorDetail] = Field(None, description="Error details if failed")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata (processing time, version, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "result": "example data"
                },
                "error": None,
                "metadata": {
                    "processing_time_ms": 123,
                    "api_version": "1.0.0"
                }
            }
        }

class PaginatedData(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool

class PaginatedResponse(StandardResponse[PaginatedData[T]]):
    """Response format for paginated data"""
    pass

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "items": [],
                    "total": 100,
                    "page": 1,
                    "page_size": 20,
                    "total_pages": 5
                },
                "error": None
            }
        }


def create_success_response(
    data: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    processing_time_ms: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create a standardized success response dictionary.
    """
    if metadata is None:
        metadata = {}

    if processing_time_ms is not None:
        metadata["processing_time_ms"] = round(processing_time_ms, 2)

    metadata["api_version"] = "1.0.0"

    response_model = StandardResponse(
        success=True,
        data=data,
        error=None,
        metadata=metadata
    )
    return response_model.model_dump(mode='json')


def create_error_response(
    error_code: ErrorCode,
    message: str,
    detail: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.
    """
    if metadata is None:
        metadata = {}

    metadata["api_version"] = "1.0.0"

    error_detail = ErrorDetail(
        code=error_code,
        message=message,
        detail=detail
    )

    response_model = StandardResponse(
        success=False,
        data=None,
        error=error_detail,
        metadata=metadata
    )
    return response_model.model_dump(mode='json')


def create_paginated_response(
    items: List[Any],
    total: int,
    page: int = 1,
    page_size: int = 20,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a paginated response dictionary.
    """
    total_pages = (total + page_size - 1) // page_size

    paginated_data = PaginatedData(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )

    if metadata is None:
        metadata = {}

    metadata["api_version"] = "1.0.0"

    response_model = PaginatedResponse(
        success=True,
        data=paginated_data,
        error=None,
        metadata=metadata
    )
    return response_model.model_dump(mode='json')
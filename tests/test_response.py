"""
Test cases for standard API response models
"""

import pytest
from datetime import datetime

from common.response import (
    StandardResponse,
    ErrorDetail,
    ErrorCode,
    PaginatedResponse,
    create_success_response,
    create_error_response,
    create_paginated_response
)


class TestStandardResponse:
    """Test StandardResponse model"""
    
    def test_success_response_creation(self):
        """Test creating a successful response"""
        response = StandardResponse(
            success=True,
            data={"result": "test"},
            error=None,
            metadata={"version": "1.0"}
        )
        
        assert response.success is True
        assert response.data == {"result": "test"}
        assert response.error is None
        assert response.metadata == {"version": "1.0"}
    
    def test_error_response_creation(self):
        """Test creating an error response"""
        error = ErrorDetail(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid input",
            detail={"field": "email"}
        )
        
        response = StandardResponse(
            success=False,
            data=None,
            error=error,
            metadata=None
        )
        
        assert response.success is False
        assert response.data is None
        assert response.error.code == ErrorCode.VALIDATION_ERROR
        assert response.error.message == "Invalid input"
    
    def test_response_json_serialization(self):
        """Test response JSON serialization"""
        response = create_success_response(
            data={"test": "data"},
            metadata={"custom": "value"}
        )
        
        json_data = response.model_dump()
        
        assert json_data["success"] is True
        assert json_data["data"] == {"test": "data"}
        assert json_data["error"] is None
        assert "timestamp" in json_data["metadata"]
        assert json_data["metadata"]["custom"] == "value"


class TestErrorDetail:
    """Test ErrorDetail model"""
    
    def test_error_detail_creation(self):
        """Test creating error detail"""
        error = ErrorDetail(
            code=ErrorCode.FILE_NOT_FOUND,
            message="File not found",
            detail={"filename": "test.jpg"}
        )
        
        assert error.code == ErrorCode.FILE_NOT_FOUND
        assert error.message == "File not found"
        assert error.detail == {"filename": "test.jpg"}
        assert isinstance(error.timestamp, datetime)
    
    def test_error_detail_without_detail(self):
        """Test error detail without additional details"""
        error = ErrorDetail(
            code=ErrorCode.AUTHENTICATION_ERROR,
            message="Authentication failed"
        )
        
        assert error.code == ErrorCode.AUTHENTICATION_ERROR
        assert error.message == "Authentication failed"
        assert error.detail is None


class TestResponseHelpers:
    """Test response helper functions"""
    
    def test_create_success_response(self):
        """Test create_success_response helper"""
        response = create_success_response(
            data={"id": 123, "name": "test"},
            processing_time_ms=150.5
        )
        
        assert response.success is True
        assert response.data == {"id": 123, "name": "test"}
        assert response.error is None
        assert response.metadata["processing_time_ms"] == 150.5
        assert response.metadata["api_version"] == "1.0.0"
        assert "timestamp" in response.metadata
    
    def test_create_success_response_minimal(self):
        """Test create_success_response with minimal parameters"""
        response = create_success_response()
        
        assert response.success is True
        assert response.data is None
        assert response.error is None
        assert response.metadata["api_version"] == "1.0.0"
    
    def test_create_error_response(self):
        """Test create_error_response helper"""
        response = create_error_response(
            error_code=ErrorCode.MODEL_INFERENCE_ERROR,
            message="Model failed to process image",
            detail={"model": "yolo", "reason": "timeout"}
        )
        
        assert response.success is False
        assert response.data is None
        assert response.error.code == ErrorCode.MODEL_INFERENCE_ERROR
        assert response.error.message == "Model failed to process image"
        assert response.error.detail == {"model": "yolo", "reason": "timeout"}
        assert response.metadata["api_version"] == "1.0.0"
    
    def test_create_paginated_response(self):
        """Test create_paginated_response helper"""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        response = create_paginated_response(
            items=items,
            total=10,
            page=2,
            page_size=3
        )
        
        assert response.success is True
        assert response.data["items"] == items
        assert response.data["total"] == 10
        assert response.data["page"] == 2
        assert response.data["page_size"] == 3
        assert response.data["total_pages"] == 4
        assert response.data["has_next"] is True
        assert response.data["has_prev"] is True
    
    def test_create_paginated_response_first_page(self):
        """Test paginated response for first page"""
        response = create_paginated_response(
            items=[1, 2, 3],
            total=10,
            page=1,
            page_size=3
        )
        
        assert response.data["has_prev"] is False
        assert response.data["has_next"] is True
    
    def test_create_paginated_response_last_page(self):
        """Test paginated response for last page"""
        response = create_paginated_response(
            items=[10],
            total=10,
            page=4,
            page_size=3
        )
        
        assert response.data["has_prev"] is True
        assert response.data["has_next"] is False
        assert response.data["total_pages"] == 4


class TestErrorCodes:
    """Test ErrorCode enum"""
    
    def test_error_code_values(self):
        """Test error code string values"""
        assert ErrorCode.UNKNOWN_ERROR == "E00_UNKNOWN_ERROR"
        assert ErrorCode.VALIDATION_ERROR == "E01_VALIDATION_ERROR"
        assert ErrorCode.FILE_NOT_FOUND == "E10_FILE_NOT_FOUND"
        assert ErrorCode.MODEL_NOT_LOADED == "E20_MODEL_NOT_LOADED"
        assert ErrorCode.EXTERNAL_API_ERROR == "E30_EXTERNAL_API_ERROR"
        assert ErrorCode.FACE_NOT_DETECTED == "E40_FACE_NOT_DETECTED"
    
    def test_error_code_categories(self):
        """Test error code categories are properly grouped"""
        # General errors (E00-E09)
        general_errors = [
            ErrorCode.UNKNOWN_ERROR,
            ErrorCode.VALIDATION_ERROR,
            ErrorCode.AUTHENTICATION_ERROR,
            ErrorCode.AUTHORIZATION_ERROR,
            ErrorCode.NOT_FOUND,
            ErrorCode.METHOD_NOT_ALLOWED,
            ErrorCode.TIMEOUT_ERROR,
            ErrorCode.RATE_LIMIT_ERROR
        ]
        
        for error in general_errors:
            assert error.value.startswith("E0")
        
        # File/Media errors (E10-E19)
        file_errors = [
            ErrorCode.FILE_NOT_FOUND,
            ErrorCode.FILE_TOO_LARGE,
            ErrorCode.INVALID_FILE_FORMAT,
            ErrorCode.IMAGE_PROCESSING_ERROR,
            ErrorCode.VIDEO_PROCESSING_ERROR
        ]
        
        for error in file_errors:
            assert error.value.startswith("E1")
        
        # Model errors (E20-E29)
        model_errors = [
            ErrorCode.MODEL_NOT_LOADED,
            ErrorCode.MODEL_INFERENCE_ERROR,
            ErrorCode.INVALID_MODEL_INPUT,
            ErrorCode.MODEL_TIMEOUT
        ]
        
        for error in model_errors:
            assert error.value.startswith("E2")
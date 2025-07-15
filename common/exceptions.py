"""
Custom exceptions for DuoPet AI Service

This module defines custom exceptions that map to standardized error responses.
"""

from typing import Optional, Dict, Any
from common.response import ErrorCode


class DuoPetException(Exception):
    """Base exception for all DuoPet AI service errors"""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.error_code = error_code
        self.message = message
        self.detail = detail or {}
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(DuoPetException):
    """Raised when request validation fails"""
    
    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            detail=detail,
            status_code=400
        )


class AuthenticationError(DuoPetException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            error_code=ErrorCode.AUTHENTICATION_ERROR,
            message=message,
            status_code=401
        )


class AuthorizationError(DuoPetException):
    """Raised when user lacks required permissions"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            error_code=ErrorCode.AUTHORIZATION_ERROR,
            message=message,
            status_code=403
        )


class NotFoundError(DuoPetException):
    """Raised when requested resource is not found"""
    
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            error_code=ErrorCode.NOT_FOUND,
            message=f"{resource} not found: {identifier}",
            detail={"resource": resource, "identifier": identifier},
            status_code=404
        )


class RateLimitError(DuoPetException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, retry_after: int):
        super().__init__(
            error_code=ErrorCode.RATE_LIMIT_ERROR,
            message="Rate limit exceeded",
            detail={"retry_after_seconds": retry_after},
            status_code=429
        )


class FileProcessingError(DuoPetException):
    """Base class for file processing errors"""
    pass


class FileTooLargeError(FileProcessingError):
    """Raised when uploaded file exceeds size limit"""
    
    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            error_code=ErrorCode.FILE_TOO_LARGE,
            message=f"File size {file_size} bytes exceeds maximum {max_size} bytes",
            detail={"file_size": file_size, "max_size": max_size},
            status_code=413
        )


class InvalidFileFormatError(FileProcessingError):
    """Raised when file format is not supported"""
    
    def __init__(self, file_format: str, allowed_formats: list):
        super().__init__(
            error_code=ErrorCode.INVALID_FILE_FORMAT,
            message=f"File format '{file_format}' not supported",
            detail={"format": file_format, "allowed_formats": allowed_formats},
            status_code=415
        )


class ModelError(DuoPetException):
    """Base class for AI model errors"""
    pass


class ModelNotLoadedError(ModelError):
    """Raised when AI model is not loaded"""
    
    def __init__(self, model_name: str):
        super().__init__(
            error_code=ErrorCode.MODEL_NOT_LOADED,
            message=f"Model '{model_name}' is not loaded",
            detail={"model_name": model_name},
            status_code=503
        )


class ModelInferenceError(ModelError):
    """Raised when model inference fails"""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            error_code=ErrorCode.MODEL_INFERENCE_ERROR,
            message=f"Model inference failed: {reason}",
            detail={"model_name": model_name, "reason": reason},
            status_code=500
        )


class ExternalAPIError(DuoPetException):
    """Raised when external API call fails"""
    
    def __init__(self, service: str, reason: str):
        super().__init__(
            error_code=ErrorCode.EXTERNAL_API_ERROR,
            message=f"External API error: {service}",
            detail={"service": service, "reason": reason},
            status_code=502
        )


class FaceDetectionError(DuoPetException):
    """Base class for face detection errors"""
    pass


class FaceNotDetectedError(FaceDetectionError):
    """Raised when no face is detected in image"""
    
    def __init__(self):
        super().__init__(
            error_code=ErrorCode.FACE_NOT_DETECTED,
            message="No face detected in the image",
            status_code=400
        )


class MultipleFacesDetectedError(FaceDetectionError):
    """Raised when multiple faces are detected"""
    
    def __init__(self, face_count: int):
        super().__init__(
            error_code=ErrorCode.MULTIPLE_FACES_DETECTED,
            message=f"Multiple faces detected: {face_count}",
            detail={"face_count": face_count},
            status_code=400
        )


class LowConfidenceError(DuoPetException):
    """Raised when AI model confidence is below threshold"""
    
    def __init__(self, confidence: float, threshold: float):
        super().__init__(
            error_code=ErrorCode.LOW_CONFIDENCE_RESULT,
            message=f"Confidence {confidence:.2f} is below threshold {threshold:.2f}",
            detail={"confidence": confidence, "threshold": threshold},
            status_code=422
        )
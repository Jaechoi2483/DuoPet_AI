"""
Custom middleware for DuoPet AI Service

This module provides additional middleware for request/response processing,
authentication, rate limiting, and other cross-cutting concerns.
"""

import time
import uuid
from typing import Callable, Optional
from datetime import datetime

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from common.config import get_settings
from common.logger import get_logger, log_api_request, log_api_response
from common.response import create_error_response, ErrorCode
from common.exceptions import AuthenticationError, RateLimitError

settings = get_settings()
logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracing"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timing
        start_time = time.time()
        
        # Get request details
        method = request.method
        path = request.url.path
        client_host = request.client.host if request.client else "unknown"
        
        # Log request
        log_api_request(
            logger,
            method=method,
            path=path,
            client_host=client_host,
            request_id=getattr(request.state, "request_id", None)
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        log_api_response(
            logger,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=getattr(request.state, "request_id", None)
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validate API key for protected endpoints"""
    
    # Endpoints that don't require authentication
    EXEMPT_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/metrics"}
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        # API key validation will be handled by dependencies
        # This middleware now just extracts and stores the API key
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract API key from various sources
        api_key = None
        
        # Check header first
        api_key = request.headers.get("X-API-Key")
        
        # Check query parameter if not in header
        if not api_key and "api_key" in request.query_params:
            api_key = request.query_params["api_key"]
        
        # Check cookie if not in header or query
        if not api_key and "api_key" in request.cookies:
            api_key = request.cookies["api_key"]
        
        # Store API key in request state for other middleware/dependencies
        if api_key:
            request.state.api_key = api_key
        
        # Continue processing
        # Actual authentication is handled by FastAPI dependencies
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_counts = {}  # {client_id: [(timestamp, count)]}
        self.default_rate_limit = 60  # Default requests per minute
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Use API key validation result if available
        if hasattr(request.state, "api_key_validation"):
            validation = request.state.api_key_validation
            if validation and validation.valid:
                return f"key:{validation.key_id}"
        
        # Fall back to API key if available
        if hasattr(request.state, "api_key"):
            return f"key:{request.state.api_key}"
        
        # Otherwise use IP
        if request.client:
            return f"ip:{request.client.host}"
        
        return "unknown"
    
    def _get_rate_limit(self, request: Request) -> int:
        """Get rate limit for the request"""
        # Check if API key has custom rate limit
        if hasattr(request.state, "api_key_validation"):
            validation = request.state.api_key_validation
            if validation and validation.valid and validation.rate_limit:
                return validation.rate_limit
        
        return self.default_rate_limit
    
    def _clean_old_requests(self, client_id: str, current_time: float):
        """Remove requests older than 1 minute"""
        if client_id in self.request_counts:
            cutoff_time = current_time - 60  # 1 minute ago
            self.request_counts[client_id] = [
                (ts, count) for ts, count in self.request_counts[client_id]
                if ts > cutoff_time
            ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in {"/", "/health"}:
            return await call_next(request)
        
        current_time = time.time()
        client_id = self._get_client_id(request)
        
        # Clean old requests
        self._clean_old_requests(client_id, current_time)
        
        # Count requests in the last minute
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []
        
        request_count = sum(count for _, count in self.request_counts[client_id])
        
        # Get rate limit for this client
        rate_limit = self._get_rate_limit(request)
        
        # Check rate limit
        if request_count >= rate_limit:
            # Calculate retry after
            oldest_request = min(self.request_counts[client_id], key=lambda x: x[0])
            retry_after = int(60 - (current_time - oldest_request[0]))
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=create_error_response(
                    error_code=ErrorCode.RATE_LIMIT_ERROR,
                    message="Rate limit exceeded",
                    detail={"retry_after_seconds": retry_after}
                ).model_dump(),
                headers={"Retry-After": str(retry_after)}
            )
        
        # Record this request
        self.request_counts[client_id].append((current_time, 1))
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, rate_limit - request_count - 1)
        )
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add CSP header for production
        if settings.is_production:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "img-src 'self' data: https:; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline';"
            )
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Add gzip compression for responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" in accept_encoding and len(response.body) > 1024:  # Only compress if > 1KB
            # Note: In production, use a proper compression middleware
            # This is just a placeholder
            response.headers["Content-Encoding"] = "gzip"
        
        return response
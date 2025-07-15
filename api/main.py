"""
DuoPet AI Service - Main FastAPI Application

This is the main entry point for the DuoPet AI microservice.
It provides AI-powered features for pet health and behavior analysis.
"""

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from common.config import get_settings
from common.logger import get_logger
from common.response import (
    create_success_response, 
    create_error_response,
    ErrorCode,
    StandardResponse
)
from common.exceptions import DuoPetException
from common.monitoring import (
    MetricsMiddleware,
    collect_system_metrics,
    get_metrics,
    CONTENT_TYPE_LATEST
)
from api.middleware import (
    RequestIDMiddleware,
    LoggingMiddleware,
    APIKeyAuthMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware
)

# Import routers
from api.routers import (
    auth,
    face_login_router,
    chatbot_router,
    video_recommend_router,
    health_diagnosis_router,
    behavior_analysis_router
)

# Initialize
settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events
    """
    # Startup
    logger.info("=� Starting DuoPet AI Service...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize models, connections, etc.
    from common.database import connect_mongodb, connect_redis, close_connections
    
    try:
        # Initialize database connections
        await connect_mongodb()
        await connect_redis()
        logger.info("Database connections established")
    except Exception as e:
        logger.error(f"Failed to initialize database connections: {str(e)}")
        # Continue anyway - some endpoints might still work
    
    # Start background tasks
    asyncio.create_task(collect_system_metrics())
    # asyncio.create_task(periodic_model_update())
    
    yield
    
    # Shutdown
    logger.info("=K Shutting down DuoPet AI Service...")
    # Cleanup resources
    await close_connections()
    logger.info("Database connections closed")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Add custom middleware in order
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyAuthMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MetricsMiddleware)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request processing time to response headers and metadata"""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    return response


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    """Global exception handler middleware"""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                error_code=ErrorCode.UNKNOWN_ERROR,
                message="An unexpected error occurred"
            ).model_dump()
        )


# Exception handlers
@app.exception_handler(DuoPetException)
async def duopet_exception_handler(request: Request, exc: DuoPetException):
    """Handle custom DuoPet exceptions"""
    logger.error(f"DuoPet exception: {exc.message}", extra={"detail": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            error_code=exc.error_code,
            message=exc.message,
            detail=exc.detail
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    errors = exc.errors()
    logger.warning(f"Validation error: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Request validation failed",
            detail={"errors": errors}
        ).model_dump()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    # Map HTTP status codes to error codes
    error_code_map = {
        401: ErrorCode.AUTHENTICATION_ERROR,
        403: ErrorCode.AUTHORIZATION_ERROR,
        404: ErrorCode.NOT_FOUND,
        405: ErrorCode.METHOD_NOT_ALLOWED,
        429: ErrorCode.RATE_LIMIT_ERROR,
    }
    
    error_code = error_code_map.get(exc.status_code, ErrorCode.UNKNOWN_ERROR)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            error_code=error_code,
            message=str(exc.detail)
        ).model_dump()
    )


# Root endpoint
@app.get("/", response_model=StandardResponse)
async def root():
    """
    Health check and service information endpoint
    """
    data = {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "active",
        "environment": settings.ENVIRONMENT,
        "features": {
            "face_login": settings.FEATURES_FACE_LOGIN_ENABLED,
            "chatbot": settings.FEATURES_CHATBOT_ENABLED,
            "health_diagnosis": settings.FEATURES_HEALTH_DIAGNOSIS_ENABLED,
            "behavior_analysis": settings.FEATURES_BEHAVIOR_ANALYSIS_ENABLED,
            "video_recommend": settings.FEATURES_VIDEO_RECOMMEND_ENABLED,
        }
    }
    return create_success_response(data=data)


@app.get("/health", response_model=StandardResponse)
async def health_check():
    """
    Detailed health check endpoint
    """
    from common.database import check_mongodb_health, check_redis_health
    
    # Check database health
    mongodb_ok = await check_mongodb_health()
    redis_ok = await check_redis_health()
    
    # Overall health status
    all_ok = mongodb_ok and redis_ok
    
    health_status = {
        "status": "healthy" if all_ok else "degraded",
        "checks": {
            "api": "ok",
            "models": "ok",  # TODO: Implement actual model health check
            "database": "ok" if mongodb_ok else "error",
            "cache": "ok" if redis_ok else "warning",  # Redis is optional
        }
    }
    return create_success_response(data=health_status)


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    
    Returns metrics in Prometheus text format
    """
    metrics_data = get_metrics()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


# Include routers
app.include_router(
    auth.router,
    prefix="/api/v1",
    tags=["Authentication"]
)

app.include_router(
    face_login_router.router,
    prefix="/api/v1/face-login",
    tags=["Face Login"]
)

app.include_router(
    chatbot_router.router,
    prefix="/api/v1/chatbot",
    tags=["Chatbot"]
)

app.include_router(
    video_recommend_router.router,
    prefix="/api/v1/video-recommend",
    tags=["Video Recommendation"]
)

app.include_router(
    health_diagnosis_router.router,
    prefix="/api/v1/health-diagnose",
    tags=["Health Diagnosis"]
)

app.include_router(
    behavior_analysis_router.router,
    prefix="/api/v1/behavior-analysis",
    tags=["Behavior Analysis"]
)


# Development/Debug endpoints
if settings.DEBUG:
    @app.get("/debug/config", response_model=StandardResponse)
    async def debug_config():
        """Show current configuration (DEBUG only)"""
        config_data = {
            "app_name": settings.APP_NAME,
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "api_host": settings.API_HOST,
            "api_port": settings.API_PORT,
            "features": {
                "face_login": settings.FEATURES_FACE_LOGIN_ENABLED,
                "chatbot": settings.FEATURES_CHATBOT_ENABLED,
                "health_diagnosis": settings.FEATURES_HEALTH_DIAGNOSIS_ENABLED,
                "behavior_analysis": settings.FEATURES_BEHAVIOR_ANALYSIS_ENABLED,
                "video_recommend": settings.FEATURES_VIDEO_RECOMMEND_ENABLED,
            }
        }
        return create_success_response(data=config_data)
    
    @app.post("/debug/error-test/{error_code}", response_model=StandardResponse)
    async def debug_error_test(error_code: str):
        """Test error responses (DEBUG only)"""
        from common.exceptions import (
            ValidationError, AuthenticationError, NotFoundError,
            ModelNotLoadedError, FaceNotDetectedError
        )
        
        error_map = {
            "validation": ValidationError("Test validation error", {"field": "test"}),
            "auth": AuthenticationError("Test authentication error"),
            "not_found": NotFoundError("User", "test-id"),
            "model": ModelNotLoadedError("test-model"),
            "face": FaceNotDetectedError(),
        }
        
        if error_code in error_map:
            raise error_map[error_code]
        
        return create_success_response(data={"message": "No error raised"})
    
    @app.get("/debug/metrics-summary", response_model=StandardResponse)
    async def debug_metrics_summary():
        """Get metrics summary (DEBUG only)"""
        from common.monitoring import get_metrics_summary
        summary = get_metrics_summary()
        return create_success_response(data=summary)


# Background tasks
async def periodic_model_update():
    """Periodically update AI models (example background task)"""
    while True:
        try:
            await asyncio.sleep(86400)  # 24 hours
            logger.info("Running periodic model update...")
            # TODO: Implement model update logic
        except Exception as e:
            logger.error(f"Error in periodic model update: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1 if settings.DEBUG else settings.API_WORKERS
    )
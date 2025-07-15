"""
Monitoring and metrics collection for DuoPet AI Service

This module provides Prometheus metrics collection and monitoring utilities.
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime
import time
import psutil
import asyncio
from functools import wraps

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess, start_http_server
)
from fastapi import Request, Response

from common.config import get_settings
from common.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Create a registry for metrics
if settings.API_WORKERS > 1:
    # Use multiprocess mode for multiple workers
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
else:
    registry = CollectorRegistry()

# Define metrics
# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

http_request_size_bytes = Summary(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

http_response_size_bytes = Summary(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

# Model inference metrics
model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_name', 'model_version'],
    registry=registry
)

model_inference_total = Counter(
    'model_inference_total',
    'Total model inference requests',
    ['model_name', 'model_version', 'status'],
    registry=registry
)

# System metrics
active_requests = Gauge(
    'active_requests',
    'Number of active requests',
    registry=registry
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    registry=registry
)

cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)

# Cache metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=registry
)

# External API metrics
external_api_calls_total = Counter(
    'external_api_calls_total',
    'Total external API calls',
    ['service', 'endpoint', 'status'],
    registry=registry
)

external_api_duration_seconds = Histogram(
    'external_api_duration_seconds',
    'External API call duration in seconds',
    ['service', 'endpoint'],
    registry=registry
)

# Error metrics
error_total = Counter(
    'error_total',
    'Total errors',
    ['error_type', 'error_code'],
    registry=registry
)


class MetricsMiddleware:
    """Middleware for collecting request metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Skip metrics endpoint itself
        if scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return
        
        # Start timing
        start_time = time.time()
        
        # Increment active requests
        active_requests.inc()
        
        # Get request details
        method = scope["method"]
        path = scope["path"]
        
        # Measure request size
        content_length = 0
        for header_name, header_value in scope["headers"]:
            if header_name == b"content-length":
                content_length = int(header_value)
                break
        
        if content_length > 0:
            http_request_size_bytes.labels(method=method, endpoint=path).observe(content_length)
        
        # Process request
        status_code = 500
        response_size = 0
        
        async def send_wrapper(message):
            nonlocal status_code, response_size
            
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Get response size from headers
                for header_name, header_value in message.get("headers", []):
                    if header_name == b"content-length":
                        response_size = int(header_value)
                        break
            
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            # Request count and duration
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            # Response size
            if response_size > 0:
                http_response_size_bytes.labels(
                    method=method,
                    endpoint=path
                ).observe(response_size)
            
            # Decrement active requests
            active_requests.dec()
            
            # Log slow requests
            if duration > 1.0:  # Log requests taking more than 1 second
                logger.warning(
                    f"Slow request: {method} {path} took {duration:.2f}s",
                    extra={
                        "method": method,
                        "path": path,
                        "duration": duration,
                        "status_code": status_code
                    }
                )


def track_model_inference(model_name: str, model_version: str = "1.0"):
    """
    Decorator to track model inference metrics
    
    Usage:
        @track_model_inference("face_recognition", "1.0")
        async def predict_face(image):
            # Model inference code
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_total.labels(
                    error_type="model_inference",
                    error_code=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                
                model_inference_duration_seconds.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(duration)
                
                model_inference_total.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status=status
                ).inc()
                
                logger.info(
                    f"Model inference: {model_name} v{model_version} - {duration:.3f}s",
                    extra={
                        "model_name": model_name,
                        "model_version": model_version,
                        "duration": duration,
                        "status": status
                    }
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_total.labels(
                    error_type="model_inference",
                    error_code=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                
                model_inference_duration_seconds.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(duration)
                
                model_inference_total.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status=status
                ).inc()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_external_api_call(service: str, endpoint: str):
    """
    Decorator to track external API calls
    
    Usage:
        @track_external_api_call("openai", "completions")
        async def call_gpt(prompt):
            # API call code
            return response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_total.labels(
                    error_type="external_api",
                    error_code=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                
                external_api_duration_seconds.labels(
                    service=service,
                    endpoint=endpoint
                ).observe(duration)
                
                external_api_calls_total.labels(
                    service=service,
                    endpoint=endpoint,
                    status=status
                ).inc()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_total.labels(
                    error_type="external_api",
                    error_code=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                
                external_api_duration_seconds.labels(
                    service=service,
                    endpoint=endpoint
                ).observe(duration)
                
                external_api_calls_total.labels(
                    service=service,
                    endpoint=endpoint,
                    status=status
                ).inc()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_cache_operation(cache_type: str, operation: str):
    """
    Track cache hits and misses
    
    Usage:
        result = track_cache_operation("redis", "hit")
        result = track_cache_operation("redis", "miss")
    """
    if operation == "hit":
        cache_hits_total.labels(cache_type=cache_type).inc()
    elif operation == "miss":
        cache_misses_total.labels(cache_type=cache_type).inc()


async def collect_system_metrics():
    """Collect system metrics periodically"""
    while True:
        try:
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_bytes.set(memory_info.rss)
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=1)
            cpu_usage_percent.set(cpu_percent)
            
            await asyncio.sleep(10)  # Collect every 10 seconds
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            await asyncio.sleep(60)  # Wait longer on error


def get_metrics() -> bytes:
    """Generate metrics in Prometheus format"""
    return generate_latest(registry)


# Convenience functions for manual metric updates
def increment_error(error_type: str, error_code: str):
    """Increment error counter"""
    error_total.labels(error_type=error_type, error_code=error_code).inc()


def record_cache_hit(cache_type: str):
    """Record a cache hit"""
    cache_hits_total.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str):
    """Record a cache miss"""
    cache_misses_total.labels(cache_type=cache_type).inc()


def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of current metrics"""
    # This is a simplified version - in production, you'd query Prometheus
    return {
        "active_requests": active_requests._value.get(),
        "total_requests": sum(http_requests_total._metrics.values()),
        "memory_usage_mb": memory_usage_bytes._value.get() / 1024 / 1024,
        "cpu_usage_percent": cpu_usage_percent._value.get(),
    }
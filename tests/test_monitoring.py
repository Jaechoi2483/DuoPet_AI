"""
Test cases for monitoring and metrics collection
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock

from common.monitoring import (
    track_model_inference,
    track_external_api_call,
    track_cache_operation,
    increment_error,
    record_cache_hit,
    record_cache_miss,
    get_metrics_summary,
    MetricsMiddleware
)


class TestModelInferenceTracking:
    """Test model inference tracking decorator"""
    
    @pytest.mark.asyncio
    async def test_async_model_inference_success(self):
        """Test successful async model inference tracking"""
        @track_model_inference("test_model", "1.0")
        async def mock_inference():
            await asyncio.sleep(0.1)
            return {"result": "success"}
        
        result = await mock_inference()
        assert result == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_async_model_inference_error(self):
        """Test async model inference error tracking"""
        @track_model_inference("test_model", "1.0")
        async def mock_inference():
            await asyncio.sleep(0.1)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await mock_inference()
    
    def test_sync_model_inference_success(self):
        """Test successful sync model inference tracking"""
        @track_model_inference("test_model", "1.0")
        def mock_inference():
            time.sleep(0.1)
            return {"result": "success"}
        
        result = mock_inference()
        assert result == {"result": "success"}
    
    def test_sync_model_inference_error(self):
        """Test sync model inference error tracking"""
        @track_model_inference("test_model", "1.0")
        def mock_inference():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            mock_inference()


class TestExternalAPITracking:
    """Test external API call tracking decorator"""
    
    @pytest.mark.asyncio
    async def test_async_api_call_success(self):
        """Test successful async API call tracking"""
        @track_external_api_call("test_service", "test_endpoint")
        async def mock_api_call():
            await asyncio.sleep(0.1)
            return {"status": "ok"}
        
        result = await mock_api_call()
        assert result == {"status": "ok"}
    
    @pytest.mark.asyncio
    async def test_async_api_call_error(self):
        """Test async API call error tracking"""
        @track_external_api_call("test_service", "test_endpoint")
        async def mock_api_call():
            await asyncio.sleep(0.1)
            raise RuntimeError("API error")
        
        with pytest.raises(RuntimeError):
            await mock_api_call()
    
    def test_sync_api_call_success(self):
        """Test successful sync API call tracking"""
        @track_external_api_call("test_service", "test_endpoint")
        def mock_api_call():
            time.sleep(0.1)
            return {"status": "ok"}
        
        result = mock_api_call()
        assert result == {"status": "ok"}


class TestCacheOperations:
    """Test cache operation tracking"""
    
    def test_cache_hit_tracking(self):
        """Test cache hit tracking"""
        track_cache_operation("redis", "hit")
        record_cache_hit("redis")
        # Metrics should be incremented (verified through integration tests)
    
    def test_cache_miss_tracking(self):
        """Test cache miss tracking"""
        track_cache_operation("redis", "miss")
        record_cache_miss("redis")
        # Metrics should be incremented (verified through integration tests)
    
    def test_invalid_cache_operation(self):
        """Test invalid cache operation is ignored"""
        # Should not raise error
        track_cache_operation("redis", "invalid")


class TestErrorTracking:
    """Test error tracking functionality"""
    
    def test_increment_error(self):
        """Test error counter increment"""
        increment_error("api_error", "E01_VALIDATION_ERROR")
        increment_error("model_error", "E20_MODEL_NOT_LOADED")
        # Metrics should be incremented (verified through integration tests)


class TestMetricsMiddleware:
    """Test metrics middleware"""
    
    @pytest.mark.asyncio
    async def test_middleware_http_request(self):
        """Test middleware processes HTTP requests"""
        app = MagicMock()
        middleware = MetricsMiddleware(app)
        
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [(b"content-length", b"100")]
        }
        
        receive = MagicMock()
        
        messages = []
        async def send(message):
            messages.append(message)
        
        # Mock the app to send response
        async def mock_app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"50")]
            })
            await send({
                "type": "http.response.body",
                "body": b"test response"
            })
        
        app.side_effect = mock_app
        
        await middleware(scope, receive, send)
        
        assert len(messages) == 2
        assert messages[0]["status"] == 200
    
    @pytest.mark.asyncio
    async def test_middleware_skip_metrics_endpoint(self):
        """Test middleware skips metrics endpoint"""
        app = MagicMock()
        middleware = MetricsMiddleware(app)
        
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/metrics",
            "headers": []
        }
        
        receive = MagicMock()
        send = MagicMock()
        
        await middleware(scope, receive, send)
        
        # App should be called directly without processing
        app.assert_called_once_with(scope, receive, send)
    
    @pytest.mark.asyncio
    async def test_middleware_non_http(self):
        """Test middleware passes through non-HTTP requests"""
        app = MagicMock()
        middleware = MetricsMiddleware(app)
        
        scope = {
            "type": "websocket",
            "path": "/ws"
        }
        
        receive = MagicMock()
        send = MagicMock()
        
        await middleware(scope, receive, send)
        
        # App should be called directly
        app.assert_called_once_with(scope, receive, send)


class TestMetricsSummary:
    """Test metrics summary functionality"""
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary"""
        summary = get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert "active_requests" in summary
        assert "total_requests" in summary
        assert "memory_usage_mb" in summary
        assert "cpu_usage_percent" in summary


class TestSystemMetrics:
    """Test system metrics collection"""
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self):
        """Test system metrics collection task"""
        from common.monitoring import collect_system_metrics
        
        # Run for a short time then cancel
        task = asyncio.create_task(collect_system_metrics())
        await asyncio.sleep(0.1)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
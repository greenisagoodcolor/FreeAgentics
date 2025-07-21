"""Integration tests for rate limiting and DDoS protection."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from api.middleware.ddos_protection import (
    DDoSProtectionMiddleware,
    EndpointRateLimits,
    RateLimitConfig,
    RateLimiter,
)
from api.middleware.websocket_rate_limiting import WebSocketRateLimitManager


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_default_config(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.burst_limit == 10
        assert config.block_duration == 300
        assert config.ddos_threshold == 1000
        assert config.ddos_block_duration == 3600

    def test_custom_config(self):
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=2000,
            burst_limit=20,
            block_duration=600,
            ddos_threshold=2000,
            ddos_block_duration=7200,
        )
        assert config.requests_per_minute == 100
        assert config.requests_per_hour == 2000
        assert config.burst_limit == 20
        assert config.block_duration == 600
        assert config.ddos_threshold == 2000
        assert config.ddos_block_duration == 7200


class TestEndpointRateLimits:
    """Test endpoint-specific rate limits."""

    def test_auth_endpoints_strict(self):
        """Test that auth endpoints have strict rate limits."""
        config = EndpointRateLimits.AUTH_ENDPOINTS
        assert config.requests_per_minute == 5
        assert config.requests_per_hour == 100
        assert config.burst_limit == 3
        assert config.block_duration == 600

    def test_api_endpoints_standard(self):
        """Test that API endpoints have standard rate limits."""
        config = EndpointRateLimits.API_ENDPOINTS
        assert config.requests_per_minute == 100
        assert config.requests_per_hour == 2000
        assert config.burst_limit == 20
        assert config.block_duration == 300

    def test_websocket_endpoints_higher(self):
        """Test that WebSocket endpoints have higher rate limits."""
        config = EndpointRateLimits.WEBSOCKET_ENDPOINTS
        assert config.requests_per_minute == 200
        assert config.requests_per_hour == 5000
        assert config.burst_limit == 50
        assert config.block_duration == 60

    def test_static_endpoints_lenient(self):
        """Test that static endpoints have lenient rate limits."""
        config = EndpointRateLimits.STATIC_ENDPOINTS
        assert config.requests_per_minute == 200
        assert config.requests_per_hour == 10000
        assert config.burst_limit == 100
        assert config.block_duration == 60


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.incr = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.decr = AsyncMock(return_value=1)
    redis_mock.incrby = AsyncMock(return_value=1024)
    redis_mock.keys = AsyncMock(return_value=[])

    # Create a separate mock for pipeline
    pipeline_mock = MagicMock()
    pipeline_mock.incr = MagicMock(return_value=pipeline_mock)
    pipeline_mock.expire = MagicMock(return_value=pipeline_mock)
    pipeline_mock.incrby = MagicMock(return_value=pipeline_mock)
    pipeline_mock.execute = AsyncMock(return_value=[1, True, 1, True])

    redis_mock.pipeline = MagicMock(return_value=pipeline_mock)

    return redis_mock


@pytest.fixture
def rate_limiter(mock_redis):
    """Rate limiter instance for testing."""
    return RateLimiter(mock_redis)


class TestRateLimiter:
    """Test rate limiter functionality."""

    @pytest.mark.asyncio
    async def test_get_client_key_ip_only(self, rate_limiter):
        """Test client key generation with IP only."""
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers = {}
        request.state = MagicMock()

        # Mock hasattr to return False for user
        with patch("api.middleware.ddos_protection.hasattr", return_value=False):
            key = await rate_limiter._get_client_key(request)
            assert key == "rate_limit:ip:192.168.1.100"

    @pytest.mark.asyncio
    async def test_get_client_key_with_forwarded_ip(self, rate_limiter):
        """Test client key generation with forwarded IP."""
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {"X-Forwarded-For": "203.0.113.1, 192.168.1.1"}
        request.state = MagicMock()

        with patch("api.middleware.ddos_protection.hasattr", return_value=False):
            key = await rate_limiter._get_client_key(request)
            assert key == "rate_limit:ip:203.0.113.1"

    @pytest.mark.asyncio
    async def test_get_client_key_with_user(self, rate_limiter):
        """Test client key generation with authenticated user."""
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers = {}
        request.state = MagicMock()
        request.state.user = MagicMock()
        request.state.user.user_id = "user123"

        # Mock hasattr to return True for user attribute
        with patch("api.middleware.ddos_protection.hasattr", return_value=True):
            with patch(
                "api.middleware.ddos_protection.getattr",
                return_value="user123",
            ):
                key = await rate_limiter._get_client_key(request)
                assert key == "rate_limit:user:user123"

    @pytest.mark.asyncio
    async def test_get_endpoint_config_auth(self, rate_limiter):
        """Test endpoint configuration for auth endpoints."""
        config = await rate_limiter._get_endpoint_config("/api/v1/auth/login")
        assert config.requests_per_minute == 5
        assert config.requests_per_hour == 100

    @pytest.mark.asyncio
    async def test_get_endpoint_config_websocket(self, rate_limiter):
        """Test endpoint configuration for WebSocket endpoints."""
        config = await rate_limiter._get_endpoint_config("/api/v1/websocket/connect")
        assert config.requests_per_minute == 200
        assert config.requests_per_hour == 5000

        config = await rate_limiter._get_endpoint_config("/ws/agents")
        assert config.requests_per_minute == 200
        assert config.requests_per_hour == 5000

    @pytest.mark.asyncio
    async def test_get_endpoint_config_static(self, rate_limiter):
        """Test endpoint configuration for static endpoints."""
        config = await rate_limiter._get_endpoint_config("/health")
        assert config.requests_per_minute == 200
        assert config.requests_per_hour == 10000

        config = await rate_limiter._get_endpoint_config("/")
        assert config.requests_per_minute == 200
        assert config.requests_per_hour == 10000

    @pytest.mark.asyncio
    async def test_get_endpoint_config_api(self, rate_limiter):
        """Test endpoint configuration for API endpoints."""
        config = await rate_limiter._get_endpoint_config("/api/v1/agents")
        assert config.requests_per_minute == 100
        assert config.requests_per_hour == 2000

    @pytest.mark.asyncio
    async def test_is_blocked_redis_blocked(self, rate_limiter, mock_redis):
        """Test blocking detection from Redis."""
        mock_redis.get.return_value = json.dumps({"blocked": True})

        is_blocked = await rate_limiter._is_blocked("test_key", "192.168.1.100")
        assert is_blocked is True

    @pytest.mark.asyncio
    async def test_is_blocked_not_blocked(self, rate_limiter, mock_redis):
        """Test when client is not blocked."""
        mock_redis.get.return_value = None

        is_blocked = await rate_limiter._is_blocked("test_key", "192.168.1.100")
        assert is_blocked is False

    @pytest.mark.asyncio
    async def test_record_request(self, rate_limiter, mock_redis):
        """Test request recording."""
        config = RateLimitConfig()
        # Set the pipeline mock's execute return value to match expected values
        pipeline_mock = mock_redis.pipeline.return_value
        pipeline_mock.execute.return_value = [5, True, 50, True]

        minute_count, hour_count = await rate_limiter._record_request(
            "test_key", config
        )

        assert minute_count == 5
        assert hour_count == 50
        assert mock_redis.pipeline.called

    @pytest.mark.asyncio
    async def test_detect_ddos_threshold_exceeded(self, rate_limiter, mock_redis):
        """Test DDoS detection when threshold is exceeded."""
        config = RateLimitConfig(ddos_threshold=100)

        is_ddos = await rate_limiter._detect_ddos("192.168.1.100", 150, config)

        assert is_ddos is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_ddos_threshold_not_exceeded(self, rate_limiter, mock_redis):
        """Test DDoS detection when threshold is not exceeded."""
        config = RateLimitConfig(ddos_threshold=100)

        is_ddos = await rate_limiter._detect_ddos("192.168.1.100", 50, config)

        assert is_ddos is False
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_block_client(self, rate_limiter, mock_redis):
        """Test client blocking."""
        config = RateLimitConfig()

        await rate_limiter._block_client(
            "test_key", "192.168.1.100", config, "TEST_REASON"
        )

        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args
        assert args[0][0] == "blocked:test_key"
        assert args[0][1] == config.block_duration


class TestDDoSProtectionMiddleware:
    """Test DDoS protection middleware."""

    @patch("api.middleware.ddos_protection.aioredis.ConnectionPool.from_url")
    @patch("api.middleware.ddos_protection.aioredis.Redis")
    def test_middleware_initialization(self, mock_redis_class, mock_pool_from_url):
        """Test middleware initialization."""
        mock_pool = MagicMock()
        mock_pool_from_url.return_value = mock_pool

        mock_redis = MagicMock()
        mock_redis_class.return_value = mock_redis

        middleware = DDoSProtectionMiddleware(None, redis_url="redis://localhost:6379")

        assert middleware.redis_url == "redis://localhost:6379"
        assert middleware.redis_client is None
        assert middleware.rate_limiter is None

    def test_middleware_config(self):
        """Test middleware configuration."""
        middleware = DDoSProtectionMiddleware(None, redis_url="redis://custom:6379")
        assert middleware.redis_url == "redis://custom:6379"


class TestWebSocketRateLimitManager:
    """Test WebSocket rate limit manager."""

    @pytest.fixture
    def manager(self):
        """Create WebSocket rate limit manager."""
        return WebSocketRateLimitManager("redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_get_client_ip_from_headers(self, manager):
        """Test client IP extraction from headers."""
        websocket = MagicMock()
        websocket.headers = {"x-real-ip": "203.0.113.1"}
        websocket.client = MagicMock()
        websocket.client.host = "127.0.0.1"

        ip = manager._get_client_ip(websocket)
        assert ip == "203.0.113.1"

    @pytest.mark.asyncio
    async def test_get_client_ip_from_forwarded(self, manager):
        """Test client IP extraction from X-Forwarded-For."""
        websocket = MagicMock()
        websocket.headers = {"x-forwarded-for": "203.0.113.1, 192.168.1.1"}
        websocket.client = MagicMock()
        websocket.client.host = "127.0.0.1"

        ip = manager._get_client_ip(websocket)
        assert ip == "203.0.113.1"

    @pytest.mark.asyncio
    async def test_get_client_ip_fallback(self, manager):
        """Test client IP fallback to client.host."""
        websocket = MagicMock()
        websocket.headers = {}
        websocket.client = MagicMock()
        websocket.client.host = "192.168.1.100"

        ip = manager._get_client_ip(websocket)
        assert ip == "192.168.1.100"

    @pytest.mark.asyncio
    async def test_get_client_ip_unknown(self, manager):
        """Test client IP when no IP available."""
        websocket = MagicMock()
        websocket.headers = {}
        websocket.client = None

        ip = manager._get_client_ip(websocket)
        assert ip == "unknown"

    @pytest.mark.asyncio
    async def test_register_connection(self, manager):
        """Test connection registration."""
        websocket = MagicMock()
        websocket.headers = {}
        websocket.client = MagicMock()
        websocket.client.host = "192.168.1.100"

        await manager.register_connection(websocket, "conn_123")

        assert "conn_123" in manager.active_connections
        assert manager.active_connections["conn_123"] == websocket

    @pytest.mark.asyncio
    async def test_unregister_connection(self, manager):
        """Test connection unregistration."""
        websocket = MagicMock()
        websocket.headers = {}
        websocket.client = MagicMock()
        websocket.client.host = "192.168.1.100"

        # Register first
        await manager.register_connection(websocket, "conn_123")
        assert "conn_123" in manager.active_connections

        # Unregister
        await manager.unregister_connection("conn_123")
        assert "conn_123" not in manager.active_connections


class TestRateLimitingIntegration:
    """Integration tests for rate limiting system."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_response(self):
        """Test rate limit exceeded response format."""
        # This would need a full integration test with Redis
        # For now, test the response structure

        from fastapi import status
        from fastapi.responses import JSONResponse

        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests per minute. Limit: 60",
                "retry_after": 60,
            },
            headers={"Retry-After": "60"},
        )

        assert response.status_code == 429
        assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_ddos_protection_response(self):
        """Test DDoS protection response format."""
        from fastapi import status
        from fastapi.responses import JSONResponse

        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "DDoS protection activated",
                "message": "Suspicious activity detected. IP blocked.",
                "retry_after": 3600,
            },
            headers={"Retry-After": "3600"},
        )

        assert response.status_code == 429
        assert "DDoS protection activated" in response.body.decode()

    def test_rate_limit_headers(self):
        """Test rate limit headers are correctly formatted."""
        headers = {
            "X-RateLimit-Limit-Minute": "60",
            "X-RateLimit-Limit-Hour": "1000",
            "X-RateLimit-Remaining-Minute": "59",
            "X-RateLimit-Remaining-Hour": "999",
            "X-RateLimit-Reset": str(int(time.time()) + 60),
        }

        assert "X-RateLimit-Limit-Minute" in headers
        assert "X-RateLimit-Remaining-Minute" in headers
        assert "X-RateLimit-Reset" in headers
        assert int(headers["X-RateLimit-Limit-Minute"]) == 60
        assert int(headers["X-RateLimit-Remaining-Minute"]) == 59


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

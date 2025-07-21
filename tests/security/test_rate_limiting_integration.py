"""
Comprehensive integration tests for rate limiting and DDoS protection.

Tests all aspects of rate limiting including:
- Per-endpoint rate limits
- User vs IP-based limiting
- DDoS detection
- Redis integration
- Metrics collection
"""

import asyncio
import json
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.rate_limiter import (
    BlockReason,
    EndpointConfig,
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimiter,
    RateLimitMiddleware,
    create_rate_limiter,
)


class TestRateLimiterIntegration:
    """Test rate limiter with full integration."""

    @pytest.fixture
    async def redis_client(self):
        """Create test Redis client."""
        client = await redis.from_url("redis://localhost:6379", decode_responses=True)

        # Clean up test keys
        async for key in client.scan_iter("rate_limit:*"):
            await client.delete(key)
        async for key in client.scan_iter("blocked:*"):
            await client.delete(key)
        async for key in client.scan_iter("pattern:*"):
            await client.delete(key)

        yield client

        # Cleanup after tests
        async for key in client.scan_iter("rate_limit:*"):
            await client.delete(key)
        async for key in client.scan_iter("blocked:*"):
            await client.delete(key)

        await client.close()

    @pytest.fixture
    def rate_limiter(self, redis_client):
        """Create rate limiter with test configuration."""
        limiter = RateLimiter(
            redis_url="redis://localhost:6379",
            default_anonymous_limit=RateLimitConfig(max_requests=10, window_seconds=60),
            default_authenticated_limit=RateLimitConfig(max_requests=50, window_seconds=60),
        )

        # Add test endpoint configurations
        limiter.endpoint_configs = [
            EndpointConfig(
                path_pattern="/api/v1/auth/login",
                anonymous_limit=RateLimitConfig(max_requests=3, window_seconds=60, burst_size=5),
                authenticated_limit=RateLimitConfig(
                    max_requests=10, window_seconds=60, burst_size=20
                ),
                priority=100,
            ),
            EndpointConfig(
                path_pattern="/api/v1/agents",
                anonymous_limit=RateLimitConfig(
                    max_requests=20,
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_size=40,
                ),
                authenticated_limit=RateLimitConfig(
                    max_requests=100,
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_size=200,
                ),
                priority=50,
            ),
        ]

        return limiter

    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self, rate_limiter, redis_client):
        """Test basic rate limiting functionality."""
        await rate_limiter.connect()

        # Create mock request
        request = MagicMock()
        request.url.path = "/api/v1/auth/login"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Test anonymous rate limiting
        for i in range(3):
            allowed, response = await rate_limiter.process_request(request)
            assert allowed, f"Request {i+1} should be allowed"
            assert response is None

        # 4th request should be rate limited
        allowed, response = await rate_limiter.process_request(request)
        assert not allowed, "4th request should be rate limited"
        assert response is not None
        assert response.status_code == 429

        # Check response content
        content = json.loads(response.body)
        assert "Rate limit exceeded" in content["detail"]

    @pytest.mark.asyncio
    async def test_user_based_rate_limiting(self, rate_limiter, redis_client):
        """Test user-based rate limiting."""
        await rate_limiter.connect()

        # Create mock request
        request = MagicMock()
        request.url.path = "/api/v1/auth/login"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Test with authenticated user
        user_id = "user123"

        # Authenticated users should have higher limits
        for i in range(10):
            allowed, response = await rate_limiter.process_request(request, user_id=user_id)
            assert allowed, f"Authenticated request {i+1} should be allowed"

        # 11th request should be rate limited
        allowed, response = await rate_limiter.process_request(request, user_id=user_id)
        assert not allowed, "11th authenticated request should be rate limited"

    @pytest.mark.asyncio
    async def test_token_bucket_algorithm(self, rate_limiter, redis_client):
        """Test token bucket rate limiting algorithm."""
        await rate_limiter.connect()

        # Create mock request for endpoint with token bucket
        request = MagicMock()
        request.url.path = "/api/v1/agents"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Burst requests should be allowed up to burst size
        burst_size = 40  # Configured burst size
        requests_made = 0

        # Make burst requests
        for i in range(burst_size):
            allowed, response = await rate_limiter.process_request(request)
            if allowed:
                requests_made += 1
            else:
                break

        # Should allow close to burst size
        assert requests_made >= burst_size - 5, f"Should allow burst requests, got {requests_made}"

    @pytest.mark.asyncio
    async def test_ddos_detection(self, rate_limiter, redis_client):
        """Test DDoS attack detection."""
        await rate_limiter.connect()

        # Configure aggressive DDoS threshold for testing
        rate_limiter.pattern_detector.patterns["rapid_errors"]["threshold"] = 5
        rate_limiter.pattern_detector.patterns["rapid_errors"]["window"] = 10

        # Create mock request
        request = MagicMock()
        request.url.path = "/api/v1/test"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Simulate rapid error responses
        ip = rate_limiter.get_client_ip(request)

        # Generate rapid errors to trigger DDoS detection
        for i in range(6):
            block_reason = await rate_limiter.check_ddos_patterns(request, ip, response_status=500)

            if i < 5:
                assert block_reason is None, f"Should not detect DDoS on request {i+1}"
            else:
                assert (
                    block_reason == BlockReason.SUSPICIOUS_PATTERN
                ), "Should detect suspicious pattern"

    @pytest.mark.asyncio
    async def test_path_scanning_detection(self, rate_limiter, redis_client):
        """Test path scanning detection."""
        await rate_limiter.connect()

        ip = "192.168.1.100"

        # Simulate path scanning
        paths = [f"/api/v1/test{i}" for i in range(20)]

        for i, path in enumerate(paths):
            request = MagicMock()
            request.url.path = path
            request.headers = {}

            block_reason = await rate_limiter.check_ddos_patterns(request, ip)

            if i < 15:
                assert block_reason is None, f"Should not detect path scanning at path {i+1}"
            else:
                assert block_reason == BlockReason.SUSPICIOUS_PATTERN, "Should detect path scanning"
                break

    @pytest.mark.asyncio
    async def test_ip_blocking(self, rate_limiter, redis_client):
        """Test IP blocking functionality."""
        await rate_limiter.connect()

        ip = "192.168.1.100"
        identifier = f"ip:{ip}"

        # Block the IP
        await rate_limiter.block_identifier(
            identifier, BlockReason.RATE_LIMIT, duration=timedelta(minutes=5)
        )

        # Check if blocked
        is_blocked, reason = await rate_limiter.is_blocked(identifier)
        assert is_blocked, "IP should be blocked"
        assert reason == BlockReason.RATE_LIMIT

        # Try to make request with blocked IP
        request = MagicMock()
        request.url.path = "/api/v1/test"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = ip

        allowed, response = await rate_limiter.process_request(request)
        assert not allowed, "Blocked IP should not be allowed"
        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_whitelist_bypass(self, rate_limiter, redis_client):
        """Test that whitelisted IPs bypass rate limiting."""
        await rate_limiter.connect()

        # Add IP to whitelist
        whitelisted_ip = "10.0.0.1"
        rate_limiter.whitelist.add(whitelisted_ip)

        request = MagicMock()
        request.url.path = "/api/v1/auth/login"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = whitelisted_ip

        # Make many requests - all should be allowed
        for i in range(100):
            allowed, response = await rate_limiter.process_request(request)
            assert allowed, f"Whitelisted IP request {i+1} should be allowed"
            assert response is None

    @pytest.mark.asyncio
    async def test_blacklist_immediate_block(self, rate_limiter, redis_client):
        """Test that blacklisted IPs are immediately blocked."""
        await rate_limiter.connect()

        # Add IP to blacklist
        blacklisted_ip = "192.168.1.200"
        rate_limiter.blacklist.add(blacklisted_ip)

        request = MagicMock()
        request.url.path = "/api/v1/test"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = blacklisted_ip

        # Should be immediately blocked
        allowed, response = await rate_limiter.process_request(request)
        assert not allowed, "Blacklisted IP should be blocked"
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self, rate_limiter, redis_client):
        """Test that rate limit headers are properly set."""
        await rate_limiter.connect()

        request = MagicMock()
        request.url.path = "/api/v1/agents"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.state = MagicMock()

        # Make a successful request
        allowed, response = await rate_limiter.process_request(request)
        assert allowed, "Request should be allowed"

        # Check that headers were set
        assert hasattr(request.state, "rate_limit_headers")
        headers = request.state.rate_limit_headers

        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers

        # Verify header values
        assert int(headers["X-RateLimit-Limit"]) == 20  # Anonymous limit for /api/v1/agents
        assert int(headers["X-RateLimit-Remaining"]) == 19  # One request made

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, rate_limiter, redis_client):
        """Test rate limiting under concurrent load."""
        await rate_limiter.connect()

        ip = "192.168.1.100"
        endpoint = "/api/v1/auth/login"
        max_allowed = 3  # Rate limit for this endpoint

        async def make_request():
            request = MagicMock()
            request.url.path = endpoint
            request.headers = {}
            request.client = MagicMock()
            request.client.host = ip

            allowed, response = await rate_limiter.process_request(request)
            return allowed

        # Make concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Count allowed requests
        allowed_count = sum(1 for r in results if r)

        # Should allow exactly the rate limit amount
        assert (
            allowed_count <= max_allowed
        ), f"Should allow at most {max_allowed} requests, got {allowed_count}"

    @pytest.mark.asyncio
    async def test_rate_limit_reset(self, rate_limiter, redis_client):
        """Test that rate limits reset after window expires."""
        await rate_limiter.connect()

        # Use a very short window for testing
        rate_limiter.endpoint_configs[0].anonymous_limit.window_seconds = 2
        rate_limiter.endpoint_configs[0].anonymous_limit.max_requests = 2

        request = MagicMock()
        request.url.path = "/api/v1/auth/login"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Use up rate limit
        for i in range(2):
            allowed, _ = await rate_limiter.process_request(request)
            assert allowed, f"Request {i+1} should be allowed"

        # Next request should be limited
        allowed, _ = await rate_limiter.process_request(request)
        assert not allowed, "Should be rate limited"

        # Wait for window to expire
        await asyncio.sleep(2.5)

        # Should be allowed again
        allowed, _ = await rate_limiter.process_request(request)
        assert allowed, "Should be allowed after window reset"

    @pytest.mark.asyncio
    async def test_large_request_blocking(self, rate_limiter, redis_client):
        """Test blocking of large requests."""
        await rate_limiter.connect()

        # Set max request size
        rate_limiter.max_request_size = 1024 * 1024  # 1MB

        request = MagicMock()
        request.url.path = "/api/v1/upload"
        request.headers = {"content-length": str(2 * 1024 * 1024)}  # 2MB
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Should block large request
        allowed, response = await rate_limiter.process_request(request)
        assert not allowed, "Large request should be blocked"
        assert response.status_code == 413

    @pytest.mark.asyncio
    async def test_header_size_limit(self, rate_limiter, redis_client):
        """Test header size limiting."""
        await rate_limiter.connect()

        # Create request with large headers
        large_headers = {f"X-Custom-Header-{i}": "a" * 100 for i in range(100)}

        request = MagicMock()
        request.url.path = "/api/v1/test"
        request.headers = large_headers
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Should block due to large headers
        allowed, response = await rate_limiter.process_request(request)
        assert not allowed, "Request with large headers should be blocked"
        assert response.status_code == 431


class TestRateLimitMiddleware:
    """Test rate limiting middleware integration."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        @app.get("/api/v1/auth/login")
        async def login_endpoint():
            return {"token": "test-token"}

        return app

    @pytest.fixture
    async def app_with_middleware(self, app):
        """Add rate limiting middleware to app."""
        rate_limiter = create_rate_limiter(redis_url="redis://localhost:6379", config_file=None)

        async def get_user_id(request):
            # Mock user ID extraction
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                return "user123"
            return None

        app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=rate_limiter,
            get_user_id=get_user_id,
        )

        return app

    @pytest.mark.asyncio
    async def test_middleware_integration(self, app_with_middleware):
        """Test middleware integration with FastAPI."""
        client = TestClient(app_with_middleware)

        # Make requests
        responses = []
        for i in range(15):
            response = client.get("/test")
            responses.append(response)

        # Check that some requests were rate limited
        rate_limited = [r for r in responses if r.status_code == 429]
        assert len(rate_limited) > 0, "Some requests should be rate limited"

        # Check rate limit headers
        for response in responses[:10]:  # Check early responses
            if response.status_code == 200:
                assert "X-RateLimit-Limit" in response.headers
                assert "X-RateLimit-Remaining" in response.headers
                assert "X-RateLimit-Reset" in response.headers

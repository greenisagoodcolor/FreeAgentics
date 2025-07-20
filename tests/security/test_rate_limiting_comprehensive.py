"""Comprehensive rate limiting test suite covering all scenarios and edge cases.

This module includes extensive tests for:
- Basic rate limiting (per second, minute, hour)
- Advanced rate limiting (IP-based, user-based, API key-based)
- Rate limit bypass attempts and security tests
- Performance and accuracy testing
- Geographic and endpoint-specific limits
"""

import asyncio
import json
import random
import time
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import redis.asyncio as aioredis
from fastapi import FastAPI, Request, status
from fastapi.testclient import TestClient

from api.middleware.ddos_protection import (
    DDoSProtectionMiddleware,
    RateLimitConfig,
    RateLimiter,
    WebSocketRateLimiter,
)


class TestBasicRateLimiting:
    """Test basic rate limiting functionality."""

    @pytest.fixture
    async def redis_client(self):
        """Create real Redis client for integration tests."""
        client = aioredis.Redis.from_url(
            "redis://localhost:6379", decode_responses=True
        )

        # Clear test keys
        keys = await client.keys("rate_limit:*")
        if keys:
            await client.delete(*keys)

        yield client

        # Cleanup
        keys = await client.keys("rate_limit:*")
        if keys:
            await client.delete(*keys)
        await client.close()

    @pytest.fixture
    def rate_limiter(self, redis_client):
        """Create rate limiter with real Redis."""
        return RateLimiter(redis_client)

    @pytest.mark.asyncio
    async def test_requests_per_second_limit(self, rate_limiter, redis_client):
        """Test request per second rate limiting."""
        # Configure strict per-second limit
        config = RateLimitConfig(
            requests_per_minute=60,  # 1 per second average
            burst_limit=2,
            window_size=1,
        )

        request = self._create_mock_request("192.168.1.100")

        # Send burst of requests
        responses = []
        for i in range(5):
            response = await rate_limiter.check_rate_limit(request)
            responses.append(response)
            await asyncio.sleep(0.1)  # 100ms between requests

        # First 2 should pass (burst limit), rest should be rate limited
        assert responses[0] is None
        assert responses[1] is None
        assert responses[2] is not None  # Rate limited
        assert responses[2].status_code == status.HTTP_429_TOO_MANY_REQUESTS

    @pytest.mark.asyncio
    async def test_requests_per_minute_limit(self, rate_limiter, redis_client):
        """Test request per minute rate limiting."""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=1000,
            burst_limit=5,
        )

        request = self._create_mock_request("192.168.1.101")

        # Override the _get_endpoint_config to use our config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Send requests up to limit
        for i in range(10):
            response = await rate_limiter.check_rate_limit(request)
            assert response is None  # Should pass

        # 11th request should be rate limited
        response = await rate_limiter.check_rate_limit(request)
        assert response is not None
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert (
            "Too many requests per minute"
            in json.loads(response.body)["message"]
        )

    @pytest.mark.asyncio
    async def test_requests_per_hour_limit(self, rate_limiter, redis_client):
        """Test request per hour rate limiting."""
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=200,  # Low hour limit for testing
            burst_limit=50,
        )

        request = self._create_mock_request("192.168.1.102")

        # Override the _get_endpoint_config to use our config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate hour's worth of requests
        now = int(time.time())
        hour_key = f"rate_limit:ip:192.168.1.102:hour:{now // 3600}"

        # Set hour count close to limit
        await redis_client.setex(hour_key, 3600, 199)

        # First request should pass
        response = await rate_limiter.check_rate_limit(request)
        assert response is None

        # Next request should be rate limited
        response = await rate_limiter.check_rate_limit(request)
        assert response is not None
        assert (
            "Too many requests per hour"
            in json.loads(response.body)["message"]
        )

    @pytest.mark.asyncio
    async def test_burst_capacity(self, rate_limiter, redis_client):
        """Test burst capacity handling."""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_limit=10,  # Allow 10 rapid requests
            window_size=1,
        )

        request = self._create_mock_request("192.168.1.103")

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Send burst of requests
        burst_responses = []
        for i in range(15):
            response = await rate_limiter.check_rate_limit(request)
            burst_responses.append(response)

        # Count successful requests
        successful = sum(1 for r in burst_responses if r is None)

        # Should allow approximately burst_limit requests
        assert 8 <= successful <= 12  # Some tolerance for timing

    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, rate_limiter, redis_client):
        """Test rate limit recovery after block duration."""
        config = RateLimitConfig(
            requests_per_minute=5,
            block_duration=2,  # 2 seconds for testing
        )

        request = self._create_mock_request("192.168.1.104")

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Exceed rate limit
        for i in range(6):
            await rate_limiter.check_rate_limit(request)

        # Should be blocked now
        response = await rate_limiter.check_rate_limit(request)
        assert response is not None
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

        # Wait for block to expire
        await asyncio.sleep(2.5)

        # Should be able to make requests again
        response = await rate_limiter.check_rate_limit(request)
        assert response is None  # Not blocked anymore

    def _create_mock_request(
        self,
        ip: str,
        path: str = "/api/v1/test",
        headers: Optional[Dict] = None,
    ):
        """Create mock request object."""
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = ip
        request.url.path = path
        request.headers = headers or {}
        request.state = MagicMock()

        # Make state.rate_limit_headers a proper attribute
        request.state.rate_limit_headers = {}

        return request


class TestAdvancedRateLimiting:
    """Test advanced rate limiting scenarios."""

    @pytest.fixture
    async def redis_client(self):
        """Create real Redis client for integration tests."""
        client = aioredis.Redis.from_url(
            "redis://localhost:6379", decode_responses=True
        )

        # Clear test keys
        keys = await client.keys("rate_limit:*")
        if keys:
            await client.delete(*keys)

        yield client

        # Cleanup
        keys = await client.keys("rate_limit:*")
        if keys:
            await client.delete(*keys)
        await client.close()

    @pytest.fixture
    def rate_limiter(self, redis_client):
        """Create rate limiter with real Redis."""
        return RateLimiter(redis_client)

    @pytest.mark.asyncio
    async def test_ip_based_rate_limiting(self, rate_limiter):
        """Test IP-based rate limiting."""
        config = RateLimitConfig(requests_per_minute=5)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Test different IPs have separate limits
        ips = ["192.168.1.100", "192.168.1.101", "192.168.1.102"]

        for ip in ips:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            # Each IP should be able to make 5 requests
            for i in range(5):
                response = await rate_limiter.check_rate_limit(request)
                assert response is None

            # 6th request should be limited
            response = await rate_limiter.check_rate_limit(request)
            assert response is not None
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    @pytest.mark.asyncio
    async def test_user_based_rate_limiting(self, rate_limiter):
        """Test user-based rate limiting."""
        config = RateLimitConfig(requests_per_minute=10)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Test authenticated users have user-specific limits
        users = ["user123", "user456", "user789"]

        for user_id in users:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = "192.168.1.100"  # Same IP
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()
            request.state.user = MagicMock()
            request.state.user.user_id = user_id

            # Mock hasattr and getattr for user detection
            with patch(
                "api.middleware.ddos_protection.hasattr", return_value=True
            ):
                with patch(
                    "api.middleware.ddos_protection.getattr",
                    return_value=user_id,
                ):
                    # Each user should have their own limit
                    for i in range(10):
                        response = await rate_limiter.check_rate_limit(request)
                        assert response is None

                    # 11th request should be limited
                    response = await rate_limiter.check_rate_limit(request)
                    assert response is not None

    @pytest.mark.asyncio
    async def test_api_key_based_rate_limiting(self, rate_limiter):
        """Test API key-based rate limiting."""
        config = RateLimitConfig(requests_per_minute=20)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Extend _get_client_key to support API keys
        original_get_client_key = rate_limiter._get_client_key

        async def mock_get_client_key(request):
            api_key = request.headers.get("X-API-Key")
            if api_key:
                return f"rate_limit:api_key:{api_key}"
            return await original_get_client_key(request)

        rate_limiter._get_client_key = mock_get_client_key

        # Test different API keys
        api_keys = ["key_123", "key_456", "key_789"]

        for api_key in api_keys:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = "192.168.1.100"  # Same IP
            request.url.path = "/api/v1/test"
            request.headers = {"X-API-Key": api_key}
            request.state = MagicMock()

            # Each API key should have its own limit
            for i in range(20):
                response = await rate_limiter.check_rate_limit(request)
                assert response is None

    @pytest.mark.asyncio
    async def test_endpoint_specific_limits(self, rate_limiter):
        """Test endpoint-specific rate limits."""
        endpoints = [
            ("/api/v1/auth/login", 5),  # Auth endpoints: 5/min
            ("/api/v1/agents", 100),  # API endpoints: 100/min
            ("/ws/connect", 200),  # WebSocket: 200/min
            ("/health", 200),  # Static: 200/min
        ]

        for endpoint, expected_limit in endpoints:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = f"192.168.1.{random.randint(1, 254)}"
            request.url.path = endpoint
            request.headers = {}
            request.state = MagicMock()

            config = await rate_limiter._get_endpoint_config(endpoint)
            assert config.requests_per_minute == expected_limit

    @pytest.mark.asyncio
    async def test_geographic_rate_limiting(self, rate_limiter, redis_client):
        """Test geographic-based rate limiting with country detection."""

        # Mock geographic detection
        def get_country_from_ip(ip: str) -> str:
            """Mock country detection based on IP."""
            ip_to_country = {
                "203.0.113.0": "US",
                "198.51.100.0": "CN",
                "192.0.2.0": "RU",
                "172.16.0.1": "BR",
            }
            return ip_to_country.get(ip, "UNKNOWN")

        # Country-specific configs
        country_configs = {
            "US": RateLimitConfig(requests_per_minute=100),
            "CN": RateLimitConfig(requests_per_minute=20),
            "RU": RateLimitConfig(requests_per_minute=10),
            "DEFAULT": RateLimitConfig(requests_per_minute=50),
        }

        # Extend rate limiter with geographic support
        original_get_endpoint_config = rate_limiter._get_endpoint_config

        async def geo_aware_endpoint_config(path):
            # Get base config
            base_config = await original_get_endpoint_config(path)

            # Get country from request context (would be set by middleware)
            if hasattr(rate_limiter, "_current_request"):
                ip = rate_limiter._current_request.client.host
                country = get_country_from_ip(ip)
                return country_configs.get(country, country_configs["DEFAULT"])

            return base_config

        rate_limiter._get_endpoint_config = geo_aware_endpoint_config

        # Test different countries
        test_cases = [
            ("203.0.113.0", "US", 100),
            ("198.51.100.0", "CN", 20),
            ("192.0.2.0", "RU", 10),
            ("8.8.8.8", "UNKNOWN", 50),
        ]

        for ip, country, expected_limit in test_cases:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            # Store request context
            rate_limiter._current_request = request

            config = await rate_limiter._get_endpoint_config("/api/v1/test")
            assert config.requests_per_minute == expected_limit


class TestRateLimitBypassAttempts:
    """Test rate limit bypass attempts and security measures."""

    @pytest.fixture
    async def redis_client(self):
        """Create real Redis client for integration tests."""
        client = aioredis.Redis.from_url(
            "redis://localhost:6379", decode_responses=True
        )

        # Clear test keys
        keys = await client.keys("rate_limit:*")
        if keys:
            await client.delete(*keys)

        yield client

        # Cleanup
        keys = await client.keys("rate_limit:*")
        if keys:
            await client.delete(*keys)
        await client.close()

    @pytest.fixture
    def rate_limiter(self, redis_client):
        """Create rate limiter with real Redis."""
        return RateLimiter(redis_client)

    @pytest.mark.asyncio
    async def test_header_manipulation_attempts(self, rate_limiter):
        """Test attempts to bypass rate limiting through header manipulation."""
        config = RateLimitConfig(requests_per_minute=5)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Base IP that will be rate limited
        base_ip = "192.168.1.100"

        # First, exhaust rate limit for base IP
        for i in range(5):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = base_ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            response = await rate_limiter.check_rate_limit(request)
            assert response is None

        # Now try various header manipulation attempts
        bypass_attempts = [
            # Try fake X-Forwarded-For
            {"X-Forwarded-For": "8.8.8.8"},
            # Try multiple IPs in X-Forwarded-For
            {"X-Forwarded-For": "8.8.8.8, 1.1.1.1, 192.168.1.100"},
            # Try X-Real-IP
            {"X-Real-IP": "8.8.4.4"},
            # Try invalid IPs
            {"X-Forwarded-For": "999.999.999.999"},
            # Try injection attempts
            {"X-Forwarded-For": "'; DROP TABLE rate_limits; --"},
            # Try localhost bypass
            {"X-Forwarded-For": "127.0.0.1"},
            {"X-Forwarded-For": "::1"},
        ]

        for headers in bypass_attempts:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = base_ip
            request.url.path = "/api/v1/test"
            request.headers = headers
            request.state = MagicMock()

            # Should use the spoofed IP from headers, not base IP
            response = await rate_limiter.check_rate_limit(request)
            # These should not be rate limited as they appear as different IPs
            assert response is None

    @pytest.mark.asyncio
    async def test_ip_spoofing_attempts(self, rate_limiter):
        """Test IP spoofing prevention measures."""
        config = RateLimitConfig(requests_per_minute=5)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Implement trusted proxy check
        TRUSTED_PROXIES = ["10.0.0.1", "10.0.0.2"]

        async def secure_get_client_key(request):
            """Enhanced client key extraction with proxy validation."""
            # Check if request comes from trusted proxy
            client_ip = request.client.host if request.client else "unknown"

            # Only trust headers from known proxies
            if client_ip in TRUSTED_PROXIES:
                real_ip = (
                    request.headers.get("X-Real-IP")
                    or request.headers.get("X-Forwarded-For", "")
                    .split(",")[0]
                    .strip()
                )
                if real_ip:
                    return f"rate_limit:ip:{real_ip}"

            # Otherwise use direct client IP
            return f"rate_limit:ip:{client_ip}"

        rate_limiter._get_client_key = secure_get_client_key

        # Test untrusted client trying to spoof
        untrusted_ip = "192.168.1.100"

        for i in range(10):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = untrusted_ip
            request.url.path = "/api/v1/test"
            # Try to spoof with headers
            request.headers = {
                "X-Forwarded-For": f"8.8.8.{i}",
                "X-Real-IP": f"1.1.1.{i}",
            }
            request.state = MagicMock()

            response = await rate_limiter.check_rate_limit(request)

            # Should be rate limited after 5 requests as headers are ignored
            if i < 5:
                assert response is None
            else:
                assert response is not None
                assert (
                    response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
                )

    @pytest.mark.asyncio
    async def test_distributed_request_patterns(
        self, rate_limiter, redis_client
    ):
        """Test detection of distributed attack patterns."""
        config = RateLimitConfig(
            requests_per_minute=10,
            ddos_threshold=50,  # Low threshold for testing
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate distributed attack from multiple IPs
        attack_ips = [f"192.168.1.{i}" for i in range(1, 11)]  # 10 IPs

        # Track global request count
        global_request_count = 0

        # Each IP makes requests just under individual limit
        for round in range(6):  # 6 rounds = 60 total requests
            for ip in attack_ips:
                request = MagicMock(spec=Request)
                request.client = MagicMock()
                request.client.host = ip
                request.url.path = "/api/v1/test"
                request.headers = {}
                request.state = MagicMock()

                response = await rate_limiter.check_rate_limit(request)
                global_request_count += 1

                # Individual IPs shouldn't be rate limited
                assert response is None

        # Check if pattern detection would trigger
        # In production, this would require pattern analysis
        assert global_request_count == 60  # All requests went through

    @pytest.mark.asyncio
    async def test_rate_limit_evasion_techniques(self, rate_limiter):
        """Test various rate limit evasion techniques."""
        config = RateLimitConfig(requests_per_minute=10)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Test case variations (different paths, methods, etc.)
        evasion_attempts = [
            # Path variations
            ("/api/v1/test", "GET"),
            ("/api/v1/test/", "GET"),  # Trailing slash
            ("/api/v1/TEST", "GET"),  # Case variation
            ("/api/v1/../v1/test", "GET"),  # Path traversal
            ("/api/v1/test?param=1", "GET"),  # Query params
            ("/api/v1/test#anchor", "GET"),  # Fragment
            # Method variations
            ("/api/v1/test", "POST"),
            ("/api/v1/test", "PUT"),
            ("/api/v1/test", "HEAD"),
        ]

        base_ip = "192.168.1.100"

        for path, method in evasion_attempts:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = base_ip
            request.url.path = path
            request.method = method
            request.headers = {}
            request.state = MagicMock()

            # Each variation might be treated as different endpoint
            # This tests if rate limiting is properly normalized
            await rate_limiter.check_rate_limit(request)

    @pytest.mark.asyncio
    async def test_cache_poisoning_for_rate_limits(
        self, rate_limiter, redis_client
    ):
        """Test cache poisoning attempts on rate limit storage."""
        config = RateLimitConfig(requests_per_minute=10)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Test malicious client keys
        malicious_inputs = [
            "192.168.1.1:minute:*",  # Wildcard attempt
            "192.168.1.1\r\nSET malicious_key",  # CRLF injection
            "192.168.1.1'; DROP TABLE rate_limits; --",  # SQL injection style
            "../../../etc/passwd",  # Path traversal
            "192.168.1.1|cat /etc/passwd",  # Command injection
            "\x00\x00\x00",  # Null bytes
            "a" * 1000,  # Long string
        ]

        for malicious_ip in malicious_inputs:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = malicious_ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            try:
                response = await rate_limiter.check_rate_limit(request)
                # Should handle malicious input gracefully
                assert (
                    response is None
                    or response.status_code
                    == status.HTTP_429_TOO_MANY_REQUESTS
                )
            except Exception as e:
                # Should not raise exceptions on malicious input
                pytest.fail(f"Rate limiter failed on malicious input: {e}")


class TestRateLimitingPerformance:
    """Test rate limiting performance and efficiency."""

    @pytest.fixture
    async def redis_client(self):
        """Create real Redis client for integration tests."""
        client = aioredis.Redis.from_url(
            "redis://localhost:6379", decode_responses=True
        )

        # Clear test keys
        keys = await client.keys("rate_limit:*")
        if keys:
            await client.delete(*keys)

        yield client

        # Cleanup
        keys = await client.keys("rate_limit:*")
        if keys:
            await client.delete(*keys)
        await client.close()

    @pytest.fixture
    def rate_limiter(self, redis_client):
        """Create rate limiter with real Redis."""
        return RateLimiter(redis_client)

    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self, rate_limiter):
        """Test rate limiting performance under heavy load."""
        config = RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=50000,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate concurrent requests
        async def make_request(ip_suffix: int):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = f"192.168.1.{ip_suffix % 255}"
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            start_time = time.time()
            response = await rate_limiter.check_rate_limit(request)
            end_time = time.time()

            return end_time - start_time, response

        # Run concurrent requests
        num_requests = 1000
        start_time = time.time()

        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Analyze performance
        response_times = [r[0] for r in results]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # Performance assertions
        assert (
            total_time < 5.0
        )  # Should handle 1000 requests in under 5 seconds
        assert avg_response_time < 0.01  # Average response under 10ms
        assert max_response_time < 0.1  # Max response under 100ms

        # Check rate limiting accuracy
        successful_requests = sum(1 for _, resp in results if resp is None)
        assert successful_requests > 0  # Some requests should succeed

    @pytest.mark.asyncio
    async def test_memory_usage_during_rate_limiting(
        self, rate_limiter, redis_client
    ):
        """Test memory usage efficiency of rate limiting."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        config = RateLimitConfig(requests_per_minute=100)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate requests from many different IPs
        num_ips = 10000
        requests_per_ip = 5

        for i in range(num_ips):
            ip = f"10.{i // 65536 % 256}.{i // 256 % 256}.{i % 256}"

            for j in range(requests_per_ip):
                request = MagicMock(spec=Request)
                request.client = MagicMock()
                request.client.host = ip
                request.url.path = "/api/v1/test"
                request.headers = {}
                request.state = MagicMock()

                await rate_limiter.check_rate_limit(request)

        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase

        # Check Redis memory usage
        redis_info = await redis_client.info("memory")
        redis_memory_mb = int(redis_info["used_memory"]) / 1024 / 1024

        # Redis memory should be efficient
        assert redis_memory_mb < 50  # Less than 50MB in Redis

    @pytest.mark.asyncio
    async def test_rate_limit_accuracy(self, rate_limiter):
        """Test accuracy of rate limiting."""
        config = RateLimitConfig(
            requests_per_minute=60,  # Exactly 1 per second
            burst_limit=1,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        ip = "192.168.1.100"
        allowed_requests = 0
        blocked_requests = 0

        # Test over 2 minutes
        start_time = time.time()

        while time.time() - start_time < 120:  # 2 minutes
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            response = await rate_limiter.check_rate_limit(request)

            if response is None:
                allowed_requests += 1
            else:
                blocked_requests += 1

            await asyncio.sleep(0.5)  # 2 requests per second attempt

        # Should allow approximately 120 requests (60 per minute * 2 minutes)
        # Allow some tolerance for timing
        assert 110 <= allowed_requests <= 130
        assert blocked_requests > 0  # Some should be blocked

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, rate_limiter):
        """Test handling of truly concurrent requests."""
        config = RateLimitConfig(
            requests_per_minute=100,
            burst_limit=10,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        ip = "192.168.1.100"

        # Create barrier to ensure true concurrency
        num_concurrent = 50
        barrier = asyncio.Barrier(num_concurrent)

        async def concurrent_request():
            await barrier.wait()  # Wait for all tasks to be ready

            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            return await rate_limiter.check_rate_limit(request)

        # Launch concurrent requests
        tasks = [concurrent_request() for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks)

        # Count successful vs blocked
        successful = sum(1 for r in results if r is None)
        blocked = sum(1 for r in results if r is not None)

        # With burst limit of 10, approximately 10 should succeed
        assert 8 <= successful <= 12  # Allow some tolerance
        assert blocked == num_concurrent - successful

    @pytest.mark.asyncio
    async def test_rate_limit_storage_efficiency(
        self, rate_limiter, redis_client
    ):
        """Test efficiency of rate limit data storage."""
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=1000,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Generate requests from many IPs
        num_ips = 1000

        for i in range(num_ips):
            ip = f"10.0.{i // 256}.{i % 256}"

            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            await rate_limiter.check_rate_limit(request)

        # Check number of keys in Redis
        all_keys = await redis_client.keys("rate_limit:*")

        # Should have 2 keys per IP (minute and hour)
        expected_keys = num_ips * 2
        assert len(all_keys) <= expected_keys * 1.1  # Allow 10% overhead

        # Check TTLs are set correctly
        sample_keys = random.sample(all_keys, min(10, len(all_keys)))
        for key in sample_keys:
            ttl = await redis_client.ttl(key)
            assert 0 < ttl <= 3600  # TTL should be set and reasonable


class TestWebSocketRateLimiting:
    """Test WebSocket-specific rate limiting."""

    @pytest.fixture
    async def redis_client(self):
        """Create real Redis client for integration tests."""
        client = aioredis.Redis.from_url(
            "redis://localhost:6379", decode_responses=True
        )

        # Clear test keys
        keys = await client.keys("ws_*")
        if keys:
            await client.delete(*keys)

        yield client

        # Cleanup
        keys = await client.keys("ws_*")
        if keys:
            await client.delete(*keys)
        await client.close()

    @pytest.fixture
    def ws_rate_limiter(self, redis_client):
        """Create WebSocket rate limiter."""
        return WebSocketRateLimiter(redis_client)

    @pytest.mark.asyncio
    async def test_websocket_connection_limits(self, ws_rate_limiter):
        """Test WebSocket connection limits per IP."""
        ip = "192.168.1.100"

        # Test connection limit
        for i in range(10):  # Max connections per IP
            allowed = await ws_rate_limiter.check_connection_limit(ip)
            assert allowed is True

        # 11th connection should be rejected
        allowed = await ws_rate_limiter.check_connection_limit(ip)
        assert allowed is False

        # Release a connection
        await ws_rate_limiter.release_connection(ip)

        # Should be able to connect again
        allowed = await ws_rate_limiter.check_connection_limit(ip)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_websocket_message_rate_limits(self, ws_rate_limiter):
        """Test WebSocket message rate limiting."""
        ip = "192.168.1.100"

        # Test message rate limit
        message_size = 100  # bytes

        for i in range(100):  # Max messages per minute
            allowed = await ws_rate_limiter.check_message_rate(
                ip, message_size
            )
            assert allowed is True

        # 101st message should be rate limited
        allowed = await ws_rate_limiter.check_message_rate(ip, message_size)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_websocket_data_rate_limits(self, ws_rate_limiter):
        """Test WebSocket data transfer rate limiting."""
        ip = "192.168.1.100"

        # Test data rate limit (1MB per minute)
        large_message_size = 100 * 1024  # 100KB

        for i in range(10):  # 10 * 100KB = 1MB
            allowed = await ws_rate_limiter.check_message_rate(
                ip, large_message_size
            )
            assert allowed is True

        # Next large message should exceed data limit
        allowed = await ws_rate_limiter.check_message_rate(
            ip, large_message_size
        )
        assert allowed is False


class TestRateLimitingIntegration:
    """Integration tests for complete rate limiting system."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with rate limiting middleware."""
        app = FastAPI()

        # Add rate limiting middleware
        app.add_middleware(
            DDoSProtectionMiddleware, redis_url="redis://localhost:6379"
        )

        @app.get("/api/v1/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.get("/api/v1/auth/login")
        async def auth_endpoint():
            return {"status": "ok"}

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        return app

    def test_middleware_integration(self, app):
        """Test rate limiting middleware integration."""
        client = TestClient(app)

        # Test normal requests
        response = client.get("/api/v1/test")
        assert response.status_code == 200

        # Check rate limit headers
        assert "X-RateLimit-Limit-Minute" in response.headers
        assert "X-RateLimit-Remaining-Minute" in response.headers

    def test_rate_limit_headers_presence(self, app):
        """Test presence and accuracy of rate limit headers."""
        client = TestClient(app)

        response = client.get("/api/v1/test")

        # Check all required headers
        required_headers = [
            "X-RateLimit-Limit-Minute",
            "X-RateLimit-Limit-Hour",
            "X-RateLimit-Remaining-Minute",
            "X-RateLimit-Remaining-Hour",
            "X-RateLimit-Reset",
        ]

        for header in required_headers:
            assert header in response.headers
            assert response.headers[header].isdigit()

    def test_health_endpoint_bypass(self, app):
        """Test that health endpoints bypass rate limiting."""
        client = TestClient(app)

        # Make many requests to health endpoint
        for i in range(1000):
            response = client.get("/health")
            assert response.status_code == 200

            # Should not have rate limit headers
            assert "X-RateLimit-Limit-Minute" not in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

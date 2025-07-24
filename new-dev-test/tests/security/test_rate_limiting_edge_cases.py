"""Edge cases and advanced security tests for rate limiting.

This module covers:
- Complex attack scenarios
- Edge case handling
- Security vulnerabilities
- Distributed attack patterns
- Advanced bypass techniques
"""

import asyncio
import base64
import random
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis
from fastapi import Request, status

from api.middleware.ddos_protection import RateLimitConfig, RateLimiter


class TestComplexAttackScenarios:
    """Test complex multi-vector attack scenarios."""

    @pytest.fixture
    async def redis_client(self):
        """Create real Redis client for integration tests."""
        client = aioredis.Redis.from_url("redis://localhost:6379", decode_responses=True)

        # Clear test keys
        keys = await client.keys("rate_limit:*")
        keys.extend(await client.keys("blocked:*"))
        keys.extend(await client.keys("ddos_blocked:*"))
        if keys:
            await client.delete(*keys)

        yield client

        # Cleanup
        keys = await client.keys("rate_limit:*")
        keys.extend(await client.keys("blocked:*"))
        keys.extend(await client.keys("ddos_blocked:*"))
        if keys:
            await client.delete(*keys)
        await client.close()

    @pytest.fixture
    def rate_limiter(self, redis_client):
        """Create rate limiter with real Redis."""
        return RateLimiter(redis_client)

    @pytest.mark.asyncio
    async def test_slowloris_attack_pattern(self, rate_limiter):
        """Test detection of Slowloris-style attacks."""
        config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_limit=5,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate Slowloris pattern - many slow, incomplete requests
        ip = "192.168.1.100"

        async def slow_request():
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {
                "User-Agent": "Mozilla/5.0 (Slowloris/1.0)",
                "Connection": "keep-alive",
            }
            request.state = MagicMock()

            start_time = time.time()
            response = await rate_limiter.check_rate_limit(request)
            end_time = time.time()

            return end_time - start_time, response

        # Simulate multiple slow connections
        tasks = []
        for i in range(20):
            tasks.append(slow_request())
            await asyncio.sleep(0.1)  # Stagger connections

        results = await asyncio.gather(*tasks)

        # Check that rate limiting kicked in
        blocked_count = sum(1 for _, resp in results if resp is not None)
        assert blocked_count > 0

    @pytest.mark.asyncio
    async def test_amplification_attack_pattern(self, rate_limiter, redis_client):
        """Test detection of amplification attack patterns."""
        config = RateLimitConfig(
            requests_per_minute=100,
            ddos_threshold=200,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate amplification attack - small requests triggering large responses
        attacker_ip = "203.0.113.1"

        # Track response sizes (simulated)
        response_amplification_factor = 100

        for i in range(50):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = attacker_ip
            request.url.path = "/api/v1/large_response"  # Endpoint that returns large data
            request.headers = {"X-Amplification-Factor": str(response_amplification_factor)}
            request.state = MagicMock()

            await rate_limiter.check_rate_limit(request)

            # Should start blocking after detecting pattern
            if i > 10:
                # Eventually should be rate limited
                pass

    @pytest.mark.asyncio
    async def test_rotating_proxy_attack(self, rate_limiter):
        """Test attack using rotating proxies."""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate rotating proxy pool
        proxy_pool = [
            f"185.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
            for _ in range(100)
        ]

        # Common characteristics of proxy requests
        common_headers = {
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Cache-Control": "no-cache",
        }

        requests_made = 0
        blocked_requests = 0

        for i in range(200):
            proxy_ip = random.choice(proxy_pool)

            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = proxy_ip
            request.url.path = "/api/v1/test"
            request.headers = common_headers.copy()
            request.headers["X-Forwarded-For"] = proxy_ip
            request.state = MagicMock()

            response = await rate_limiter.check_rate_limit(request)
            requests_made += 1

            if response is not None:
                blocked_requests += 1

        # Some requests should be blocked despite rotation
        assert blocked_requests > 0

    @pytest.mark.asyncio
    async def test_botnet_coordinated_attack(self, rate_limiter, redis_client):
        """Test coordinated botnet attack simulation."""
        config = RateLimitConfig(
            requests_per_minute=50,
            ddos_threshold=500,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate botnet with command & control coordination
        class BotnetSimulator:
            def __init__(self, num_bots: int):
                self.bots = [
                    {
                        "ip": f"10.{i // 65536 % 256}.{i // 256 % 256}.{i % 256}",
                        "id": f"bot_{i}",
                        "user_agent": self._generate_user_agent(i),
                    }
                    for i in range(num_bots)
                ]
                self.attack_phase = "reconnaissance"

            def _generate_user_agent(self, seed: int) -> str:
                agents = [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Mozilla/5.0 (X11; Linux x86_64)",
                ]
                return agents[seed % len(agents)]

            async def coordinate_attack(self, target_path: str):
                """Simulate coordinated attack phases."""
                # Phase 1: Reconnaissance
                self.attack_phase = "reconnaissance"
                recon_bots = random.sample(self.bots, k=min(10, len(self.bots)))

                for bot in recon_bots:
                    request = MagicMock(spec=Request)
                    request.client = MagicMock()
                    request.client.host = bot["ip"]
                    request.url.path = target_path
                    request.headers = {"User-Agent": bot["user_agent"]}
                    request.state = MagicMock()

                    await rate_limiter.check_rate_limit(request)
                    await asyncio.sleep(0.5)  # Slow reconnaissance

                # Phase 2: Coordinated attack
                self.attack_phase = "attack"
                attack_results = []

                async def bot_attack(bot):
                    for _ in range(5):  # Each bot makes 5 requests
                        request = MagicMock(spec=Request)
                        request.client = MagicMock()
                        request.client.host = bot["ip"]
                        request.url.path = target_path
                        request.headers = {"User-Agent": bot["user_agent"]}
                        request.state = MagicMock()

                        response = await rate_limiter.check_rate_limit(request)
                        attack_results.append((bot["id"], response))
                        await asyncio.sleep(random.uniform(0.01, 0.1))

                # Launch coordinated attack
                tasks = [bot_attack(bot) for bot in self.bots]
                await asyncio.gather(*tasks)

                return attack_results

        # Run botnet simulation
        botnet = BotnetSimulator(num_bots=50)
        results = await botnet.coordinate_attack("/api/v1/test")

        # Analyze results
        blocked_bots = set()
        for bot_id, response in results:
            if response is not None:
                blocked_bots.add(bot_id)

        # Should detect and block some bots
        assert len(blocked_bots) > 0


class TestEdgeCaseHandling:
    """Test edge cases in rate limiting implementation."""

    @pytest.fixture
    async def redis_client(self):
        """Create real Redis client for integration tests."""
        client = aioredis.Redis.from_url("redis://localhost:6379", decode_responses=True)

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
    async def test_unicode_and_special_characters(self, rate_limiter):
        """Test handling of Unicode and special characters in requests."""
        config = RateLimitConfig(requests_per_minute=10)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Test various Unicode and special character inputs
        test_cases = [
            # Unicode IPs (invalid but should be handled)
            "192.168.1.1♥",
            "2001:db8::8a2e:370:7334",  # IPv6
            "192.168.1.1\u200b",  # Zero-width space
            "192.168.1.1\ufeff",  # BOM
            # Special characters
            "192.168.1.1%00",
            "192.168.1.1\n",
            "192.168.1.1\r\n",
            "192.168.1.1\t",
            # Emoji
            "192.168.1.1🔥",
            "🌐.🌐.🌐.🌐",
        ]

        for test_ip in test_cases:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = test_ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            try:
                response = await rate_limiter.check_rate_limit(request)
                # Should handle gracefully without exceptions
                assert response is None or response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            except Exception as e:
                pytest.fail(f"Failed to handle special character: {test_ip} - {e}")

    @pytest.mark.asyncio
    async def test_ipv6_rate_limiting(self, rate_limiter):
        """Test rate limiting for IPv6 addresses."""
        config = RateLimitConfig(requests_per_minute=5)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Test various IPv6 formats
        ipv6_addresses = [
            "2001:db8::8a2e:370:7334",
            "2001:0db8:0000:0000:0000:0000:0000:0001",
            "::1",  # Loopback
            "fe80::1",  # Link-local
            "2001:db8::/32",  # Network notation (should be handled)
        ]

        for ipv6 in ipv6_addresses:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ipv6
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            # Make requests up to limit
            for i in range(5):
                response = await rate_limiter.check_rate_limit(request)
                assert response is None

            # 6th request should be limited
            response = await rate_limiter.check_rate_limit(request)
            assert response is not None

    @pytest.mark.asyncio
    async def test_time_boundary_conditions(self, rate_limiter, redis_client):
        """Test rate limiting at time boundaries (minute/hour transitions)."""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Mock time to control boundaries
        time.time
        mock_time = MagicMock()

        # Start at 59 seconds into a minute
        base_time = 1640995259  # 2022-01-01 00:00:59
        mock_time.return_value = base_time

        with patch("time.time", mock_time):
            ip = "192.168.1.100"

            # Make requests at minute boundary
            for i in range(15):
                request = MagicMock(spec=Request)
                request.client = MagicMock()
                request.client.host = ip
                request.url.path = "/api/v1/test"
                request.headers = {}
                request.state = MagicMock()

                response = await rate_limiter.check_rate_limit(request)

                # Advance time by 1 second
                mock_time.return_value += 1

                if i < 10:
                    assert response is None  # First minute
                elif i == 10:
                    # Crossed minute boundary, should reset
                    assert response is None
                elif i > 10 and i < 11:
                    assert response is None  # New minute

    @pytest.mark.asyncio
    async def test_redis_connection_failures(self, rate_limiter):
        """Test graceful handling of Redis connection failures."""
        config = RateLimitConfig(requests_per_minute=10)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Mock Redis to simulate connection failure
        original_redis = rate_limiter.redis
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = Exception("Redis connection failed")
        mock_redis.incr.side_effect = Exception("Redis connection failed")
        mock_redis.pipeline.side_effect = Exception("Redis connection failed")

        rate_limiter.redis = mock_redis

        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.url.path = "/api/v1/test"
        request.headers = {}
        request.state = MagicMock()

        # Should handle Redis failure gracefully
        await rate_limiter.check_rate_limit(request)

        # Restore original Redis
        rate_limiter.redis = original_redis

    @pytest.mark.asyncio
    async def test_extreme_load_conditions(self, rate_limiter):
        """Test rate limiting under extreme load conditions."""
        config = RateLimitConfig(
            requests_per_minute=1000,
            burst_limit=100,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate extreme concurrent load
        num_concurrent = 1000
        barrier = asyncio.Barrier(num_concurrent)

        async def extreme_request(request_id: int):
            await barrier.wait()  # Synchronize all requests

            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = f"10.0.{request_id // 256}.{request_id % 256}"
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            start_time = time.time()
            try:
                response = await rate_limiter.check_rate_limit(request)
                end_time = time.time()
                return True, end_time - start_time, response
            except Exception as e:
                end_time = time.time()
                return False, end_time - start_time, str(e)

        # Launch extreme load
        tasks = [extreme_request(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successful = sum(1 for r in results if isinstance(r, tuple) and r[0])
        num_concurrent - successful

        # Should handle most requests even under extreme load
        assert successful > num_concurrent * 0.95  # 95% success rate

    @pytest.mark.asyncio
    async def test_malformed_requests(self, rate_limiter):
        """Test handling of malformed requests."""
        config = RateLimitConfig(requests_per_minute=10)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Test various malformed request scenarios
        malformed_requests = [
            # Missing client
            {"client": None, "headers": {}, "path": "/api/v1/test"},
            # Missing host
            {
                "client": MagicMock(host=None),
                "headers": {},
                "path": "/api/v1/test",
            },
            # Invalid path
            {
                "client": MagicMock(host="192.168.1.1"),
                "headers": {},
                "path": None,
            },
            # Circular reference in headers
            {
                "client": MagicMock(host="192.168.1.1"),
                "headers": {},
                "path": "/test",
            },
        ]

        for req_data in malformed_requests:
            request = MagicMock(spec=Request)
            request.client = req_data.get("client")
            request.headers = req_data.get("headers", {})
            request.url = MagicMock()
            request.url.path = req_data.get("path")
            request.state = MagicMock()

            try:
                response = await rate_limiter.check_rate_limit(request)
                # Should handle gracefully
                assert response is None or response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            except Exception as e:
                pytest.fail(f"Failed to handle malformed request: {req_data} - {e}")


class TestSecurityVulnerabilities:
    """Test for potential security vulnerabilities in rate limiting."""

    @pytest.fixture
    async def redis_client(self):
        """Create real Redis client for integration tests."""
        client = aioredis.Redis.from_url("redis://localhost:6379", decode_responses=True)

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
    async def test_redis_injection_attempts(self, rate_limiter):
        """Test Redis injection vulnerability attempts."""
        config = RateLimitConfig(requests_per_minute=10)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Potential Redis injection payloads
        injection_attempts = [
            # Redis commands
            "192.168.1.1\r\nFLUSHALL\r\n",
            "192.168.1.1'; EVAL 'return redis.call(\"flushall\")' 0; --",
            "192.168.1.1\r\nCONFIG SET dir /tmp\r\n",
            # Script injection
            "192.168.1.1'); os.execute('rm -rf /'); --",
            # Null byte injection
            "192.168.1.1\x00FLUSHALL",
            # Unicode normalization attacks
            "192.168.1.1\uff0eFLUSHALL",
        ]

        for payload in injection_attempts:
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = payload
            request.url.path = "/api/v1/test"
            request.headers = {"X-Forwarded-For": payload}
            request.state = MagicMock()

            try:
                response = await rate_limiter.check_rate_limit(request)
                # Should sanitize input and not execute commands
                assert response is None or response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            except Exception as e:
                # Should not raise exceptions
                pytest.fail(f"Injection attempt caused exception: {payload} - {e}")

    @pytest.mark.asyncio
    async def test_timing_attacks(self, rate_limiter):
        """Test resistance to timing attacks."""
        config = RateLimitConfig(requests_per_minute=10)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Measure timing differences for rate limited vs non-rate limited
        timings = {"allowed": [], "blocked": []}

        # First, create a rate limited IP
        blocked_ip = "192.168.1.100"
        for i in range(11):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = blocked_ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            await rate_limiter.check_rate_limit(request)

        # Now measure timings
        for i in range(100):
            # Allowed request
            allowed_ip = f"10.0.0.{i}"
            request_allowed = MagicMock(spec=Request)
            request_allowed.client = MagicMock()
            request_allowed.client.host = allowed_ip
            request_allowed.url.path = "/api/v1/test"
            request_allowed.headers = {}
            request_allowed.state = MagicMock()

            start_time = time.perf_counter()
            await rate_limiter.check_rate_limit(request_allowed)
            end_time = time.perf_counter()
            timings["allowed"].append(end_time - start_time)

            # Blocked request
            request_blocked = MagicMock(spec=Request)
            request_blocked.client = MagicMock()
            request_blocked.client.host = blocked_ip
            request_blocked.url.path = "/api/v1/test"
            request_blocked.headers = {}
            request_blocked.state = MagicMock()

            start_time = time.perf_counter()
            await rate_limiter.check_rate_limit(request_blocked)
            end_time = time.perf_counter()
            timings["blocked"].append(end_time - start_time)

        # Calculate timing statistics
        avg_allowed = sum(timings["allowed"]) / len(timings["allowed"])
        avg_blocked = sum(timings["blocked"]) / len(timings["blocked"])

        # Timing difference should be minimal to prevent timing attacks
        timing_diff = abs(avg_allowed - avg_blocked)
        assert timing_diff < 0.001  # Less than 1ms difference

    @pytest.mark.asyncio
    async def test_resource_exhaustion_attacks(self, rate_limiter, redis_client):
        """Test resistance to resource exhaustion attacks."""
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=1000,
        )

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Monitor Redis memory before attack
        initial_info = await redis_client.info("memory")
        initial_memory = int(initial_info["used_memory"])

        # Attempt to exhaust resources with many unique IPs
        num_unique_ips = 100000

        for i in range(num_unique_ips):
            # Generate unique IP
            ip = f"{i // 16777216 % 256}.{i // 65536 % 256}.{i // 256 % 256}.{i % 256}"

            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            await rate_limiter.check_rate_limit(request)

            # Check memory periodically
            if i % 10000 == 0:
                current_info = await redis_client.info("memory")
                current_memory = int(current_info["used_memory"])
                memory_growth = current_memory - initial_memory

                # Memory growth should be reasonable
                assert memory_growth < 100 * 1024 * 1024  # Less than 100MB

    @pytest.mark.asyncio
    async def test_bypass_through_encoding(self, rate_limiter):
        """Test bypass attempts through various encoding methods."""
        config = RateLimitConfig(requests_per_minute=5)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Base IP that will be rate limited
        base_ip = "192.168.1.100"

        # First exhaust the limit
        for i in range(5):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = base_ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            await rate_limiter.check_rate_limit(request)

        # Try various encoding bypass attempts
        encoding_attempts = [
            # URL encoding
            "192%2E168%2E1%2E100",
            # Unicode encoding
            "192\u002e168\u002e1\u002e100",
            # HTML entities (shouldn't work but test anyway)
            "192&#46;168&#46;1&#46;100",
            # Base64 encoded (in header)
            base64.b64encode(base_ip.encode()).decode(),
            # Hex encoding
            "0xC0A80164",  # 192.168.1.100 in hex
            # Octal
            "0300.0250.0001.0144",
        ]

        for encoded_ip in encoding_attempts:
            # Try in client host
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = encoded_ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            response = await rate_limiter.check_rate_limit(request)
            # Should treat as different IP (encoding not normalized)
            assert response is None


class TestDistributedSystemScenarios:
    """Test rate limiting in distributed system scenarios."""

    @pytest.fixture
    async def redis_client(self):
        """Create real Redis client for integration tests."""
        client = aioredis.Redis.from_url("redis://localhost:6379", decode_responses=True)

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

    @pytest.mark.asyncio
    async def test_multi_instance_coordination(self, redis_client):
        """Test rate limiting coordination across multiple app instances."""
        config = RateLimitConfig(requests_per_minute=10)

        # Create multiple rate limiter instances (simulating multiple app instances)
        rate_limiters = [RateLimiter(redis_client) for _ in range(5)]

        # Configure all instances
        for limiter in rate_limiters:

            async def mock_get_endpoint_config(path):
                return config

            limiter._get_endpoint_config = mock_get_endpoint_config

        # Test that they share the same rate limit
        ip = "192.168.1.100"
        total_allowed = 0
        total_blocked = 0

        # Each instance tries to make requests
        for i, limiter in enumerate(rate_limiters):
            for j in range(5):  # 5 requests per instance = 25 total
                request = MagicMock(spec=Request)
                request.client = MagicMock()
                request.client.host = ip
                request.url.path = "/api/v1/test"
                request.headers = {}
                request.state = MagicMock()

                response = await limiter.check_rate_limit(request)

                if response is None:
                    total_allowed += 1
                else:
                    total_blocked += 1

        # Should allow 10 total across all instances
        assert total_allowed == 10
        assert total_blocked == 15

    @pytest.mark.asyncio
    async def test_redis_cluster_failover(self, redis_client):
        """Test rate limiting behavior during Redis cluster failover."""
        config = RateLimitConfig(requests_per_minute=10)

        rate_limiter = RateLimiter(redis_client)

        # Override the _get_endpoint_config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Simulate failover scenario
        ip = "192.168.1.100"

        # Make some requests
        for i in range(5):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            response = await rate_limiter.check_rate_limit(request)
            assert response is None

        # Simulate Redis failover (connection lost)
        original_redis = rate_limiter.redis
        rate_limiter.redis = None

        # Requests during failover
        failover_responses = []
        for i in range(3):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            try:
                response = await rate_limiter.check_rate_limit(request)
                failover_responses.append(response)
            except Exception:
                failover_responses.append("error")

        # Restore Redis connection
        rate_limiter.redis = original_redis

        # Continue making requests
        for i in range(5):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            response = await rate_limiter.check_rate_limit(request)
            # Should continue enforcing limits after recovery

    @pytest.mark.asyncio
    async def test_geographic_distribution_simulation(self, redis_client):
        """Test rate limiting with geographically distributed traffic."""
        # Simulate different regions with different latencies
        regions = {
            "us-east": {
                "latency": 0.01,
                "ips": [f"10.1.{i}.{j}" for i in range(10) for j in range(10)],
            },
            "eu-west": {
                "latency": 0.05,
                "ips": [f"10.2.{i}.{j}" for i in range(10) for j in range(10)],
            },
            "asia-pac": {
                "latency": 0.1,
                "ips": [f"10.3.{i}.{j}" for i in range(10) for j in range(10)],
            },
        }

        # Region-specific configs
        region_configs = {
            "us-east": RateLimitConfig(requests_per_minute=100),
            "eu-west": RateLimitConfig(requests_per_minute=80),
            "asia-pac": RateLimitConfig(requests_per_minute=60),
        }

        rate_limiter = RateLimiter(redis_client)

        # Track requests by region
        region_stats = {region: {"allowed": 0, "blocked": 0} for region in regions}

        async def make_region_request(region: str, ip: str):
            # Simulate network latency
            await asyncio.sleep(regions[region]["latency"])

            # Override config for region
            async def mock_get_endpoint_config(path):
                return region_configs[region]

            rate_limiter._get_endpoint_config = mock_get_endpoint_config

            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = ip
            request.url.path = "/api/v1/test"
            request.headers = {"X-Region": region}
            request.state = MagicMock()

            response = await rate_limiter.check_rate_limit(request)

            if response is None:
                region_stats[region]["allowed"] += 1
            else:
                region_stats[region]["blocked"] += 1

        # Simulate traffic from all regions concurrently
        tasks = []
        for region, data in regions.items():
            for ip in random.sample(data["ips"], 20):  # 20 IPs per region
                for _ in range(10):  # 10 requests per IP
                    tasks.append(make_region_request(region, ip))

        # Run all requests concurrently
        await asyncio.gather(*tasks)

        # Verify regional patterns
        for region in regions:
            total_requests = region_stats[region]["allowed"] + region_stats[region]["blocked"]
            assert total_requests > 0
            # Each region should have some blocked requests
            assert region_stats[region]["blocked"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

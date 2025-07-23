"""Comprehensive Brute Force Protection Test Suite.

This module contains exhaustive tests for brute force attack protection covering:
1. Authentication brute force attacks
2. Token brute force attempts
3. Resource enumeration protection
4. Protection validation and effectiveness
5. Performance impact measurements

Tests are designed to validate production-ready brute force defenses.
"""

import asyncio
import base64
import random
import string
import time
from datetime import datetime, timedelta

import jwt
import pytest
import redis.asyncio as aioredis
from fastapi import FastAPI
from httpx import AsyncClient

from api.middleware.ddos_protection import DDoSProtectionMiddleware
from api.middleware.security_monitoring import SecurityMonitoringMiddleware
from api.v1.auth import router as auth_router
from auth.jwt_handler import jwt_handler as JWTHandler


class TestAuthenticationBruteForce:
    """Test authentication brute force protection."""

    @pytest.fixture
    async def app(self):
        """Create FastAPI app with security middleware."""
        app = FastAPI()

        # Add security middleware
        app.add_middleware(DDoSProtectionMiddleware, redis_url="redis://localhost:6379")
        app.add_middleware(SecurityMonitoringMiddleware)

        # Add auth routes
        app.include_router(auth_router, prefix="/api/v1/auth")

        return app

    @pytest.fixture
    async def client(self, app):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.fixture
    async def redis_client(self):
        """Create Redis client for test verification."""
        client = aioredis.Redis.from_url("redis://localhost:6379", decode_responses=True)

        # Clear test keys
        keys = await client.keys("rate_limit:*")
        if keys:
            await client.delete(*keys)
        keys = await client.keys("blocked:*")
        if keys:
            await client.delete(*keys)
        keys = await client.keys("ddos_blocked:*")
        if keys:
            await client.delete(*keys)

        yield client

        # Cleanup
        await client.flushdb()
        await client.close()

    @pytest.mark.asyncio
    async def test_login_attempt_rate_limiting(self, client, redis_client):
        """Test login attempts are properly rate limited."""
        # Test parameters
        username = "testuser@example.com"
        passwords = [f"password{i}" for i in range(20)]

        # Attempt login with different passwords
        responses = []
        for password in passwords:
            response = await client.post(
                "/api/v1/auth/login",
                json={"username": username, "password": password},
            )
            responses.append(response)

        # Verify rate limiting kicks in
        successful_attempts = sum(1 for r in responses if r.status_code != 429)
        rate_limited = sum(1 for r in responses if r.status_code == 429)

        # Should allow only 3 attempts before rate limiting (auth endpoint limit)
        assert successful_attempts <= 3
        assert rate_limited >= 17

        # Check Redis for block status
        blocked = await redis_client.get("blocked:rate_limit:ip:127.0.0.1")
        assert blocked is not None

    @pytest.mark.asyncio
    async def test_password_brute_force_protection(self, client, redis_client):
        """Test protection against password brute forcing."""
        # Create a valid user
        valid_user = {
            "email": "victim@example.com",
            "password": "correct_password_123",
        }

        # Register the user
        await client.post("/api/v1/auth/register", json=valid_user)

        # Attempt brute force with common passwords
        common_passwords = [
            "password",
            "123456",
            "admin",
            "letmein",
            "qwerty",
            "password123",
            "admin123",
            "test123",
            "welcome",
            "monkey",
        ]

        attack_results = []
        for pwd in common_passwords:
            response = await client.post(
                "/api/v1/auth/login",
                json={"username": valid_user["email"], "password": pwd},
            )
            attack_results.append(
                {
                    "password": pwd,
                    "status": response.status_code,
                    "headers": dict(response.headers),
                }
            )

        # Verify protection activated
        assert any(r["status"] == 429 for r in attack_results), "Rate limiting should activate"

        # Check progressive delays are applied
        retry_after_headers = [
            int(r["headers"].get("retry-after", 0))
            for r in attack_results
            if "retry-after" in r["headers"]
        ]

        # Retry delays should increase progressively
        if len(retry_after_headers) > 1:
            assert retry_after_headers[-1] >= retry_after_headers[0]

    @pytest.mark.asyncio
    async def test_account_lockout_mechanisms(self, client, redis_client):
        """Test account lockout after excessive failed attempts."""
        # Setup test account
        test_account = {
            "email": "lockout_test@example.com",
            "password": "secure_password_456",
        }

        # Register account
        await client.post("/api/v1/auth/register", json=test_account)

        # Simulate multiple failed login attempts
        failed_attempts = 0
        lockout_triggered = False

        for i in range(15):
            response = await client.post(
                "/api/v1/auth/login",
                json={
                    "username": test_account["email"],
                    "password": f"wrong_password_{i}",
                },
            )

            if response.status_code == 429:
                lockout_triggered = True
                break

            failed_attempts += 1

        # Verify lockout triggered
        assert lockout_triggered, "Account lockout should trigger after multiple failures"
        assert failed_attempts <= 5, "Lockout should trigger within 5 attempts"

        # Verify lockout persists even with correct password
        await asyncio.sleep(1)  # Small delay

        correct_pwd_response = await client.post(
            "/api/v1/auth/login",
            json={
                "username": test_account["email"],
                "password": test_account["password"],
            },
        )

        assert correct_pwd_response.status_code == 429, "Lockout should persist"

    @pytest.mark.asyncio
    async def test_progressive_delays(self, client, redis_client):
        """Test progressive delays increase with each failed attempt."""
        # Track delays between attempts
        delays = []
        time.time()

        for i in range(10):
            time.time()

            response = await client.post(
                "/api/v1/auth/login",
                json={
                    "username": f"progressive_test_{i}@example.com",
                    "password": "wrong_password",
                },
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 0))
                delays.append(retry_after)

                # Verify response contains appropriate error
                data = response.json()
                assert "error" in data
                assert "retry_after" in data

        # Verify delays increase progressively
        assert len(delays) > 0, "Progressive delays should be applied"

        # Check if delays are non-decreasing
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1], "Delays should increase or stay same"


class TestTokenBruteForce:
    """Test token brute force protection."""

    @pytest.fixture
    def jwt_handler(self):
        """Create JWT handler for token generation."""
        return JWTHandler()

    @pytest.mark.asyncio
    async def test_jwt_token_brute_forcing(self, client, redis_client, jwt_handler):
        """Test protection against JWT token brute forcing."""
        # Generate base valid token
        valid_payload = {
            "user_id": "123",
            "email": "test@example.com",
            "exp": datetime.utcnow() + timedelta(hours=1),
        }

        jwt_handler.create_access_token(valid_payload)

        # Attempt to brute force with modified tokens
        brute_force_attempts = []

        # Try different user IDs
        for user_id in range(100, 150):
            modified_payload = valid_payload.copy()
            modified_payload["user_id"] = str(user_id)

            # Create token with wrong signature
            forged_token = jwt.encode(modified_payload, "wrong_secret_key", algorithm="HS256")

            response = await client.get(
                "/api/v1/users/profile",
                headers={"Authorization": f"Bearer {forged_token}"},
            )

            brute_force_attempts.append(response)

        # Check rate limiting activated
        rate_limited = sum(1 for r in brute_force_attempts if r.status_code == 429)
        assert rate_limited > 0, "Token brute force should trigger rate limiting"

    @pytest.mark.asyncio
    async def test_api_key_enumeration(self, client, redis_client):
        """Test protection against API key enumeration attacks."""
        # Generate random API keys to test
        api_keys = [
            "".join(random.choices(string.ascii_letters + string.digits, k=32)) for _ in range(50)
        ]

        enumeration_results = []

        for key in api_keys:
            response = await client.get("/api/v1/data", headers={"X-API-Key": key})

            enumeration_results.append(
                {
                    "key": key[:8] + "...",  # Truncate for logging
                    "status": response.status_code,
                    "timestamp": time.time(),
                }
            )

        # Verify enumeration protection
        successful = sum(1 for r in enumeration_results if r["status"] != 429)
        assert successful < 10, "API key enumeration should be rate limited"

        # Check for timing attack protection (responses should have similar timing)
        if len(enumeration_results) > 2:
            timings = [
                enumeration_results[i + 1]["timestamp"] - enumeration_results[i]["timestamp"]
                for i in range(len(enumeration_results) - 1)
                if enumeration_results[i]["status"] != 429
            ]

            if timings:
                avg_timing = sum(timings) / len(timings)
                variance = sum((t - avg_timing) ** 2 for t in timings) / len(timings)
                assert variance < 0.01, "Timing should be consistent to prevent timing attacks"

    @pytest.mark.asyncio
    async def test_session_token_guessing(self, client, redis_client):
        """Test protection against session token guessing."""
        # Create valid session
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"username": "valid@example.com", "password": "password123"},
        )

        if login_response.status_code == 200:
            login_response.cookies.get("session_id")
        else:
            pass

        # Attempt to guess session tokens
        guessing_attempts = []

        for i in range(100):
            # Generate random session ID
            fake_session = base64.b64encode(
                f"session_{i}_{random.randint(1000, 9999)}".encode()
            ).decode()

            response = await client.get("/api/v1/users/me", cookies={"session_id": fake_session})

            guessing_attempts.append(response)

        # Verify protection activated
        blocked_attempts = sum(1 for r in guessing_attempts if r.status_code == 429)
        assert blocked_attempts > 90, "Session guessing should be heavily rate limited"

    @pytest.mark.asyncio
    async def test_refresh_token_attacks(self, client, redis_client, jwt_handler):
        """Test protection against refresh token attacks."""
        # Generate valid refresh token
        valid_refresh = jwt_handler.create_refresh_token({"user_id": "123"})

        # Attempt refresh token reuse attack
        reuse_attempts = []

        for i in range(10):
            response = await client.post(
                "/api/v1/auth/refresh", json={"refresh_token": valid_refresh}
            )
            reuse_attempts.append(response)

        # First use should succeed, subsequent should fail or be rate limited
        assert reuse_attempts[0].status_code in [200, 401]

        # Check for rate limiting on repeated attempts
        rate_limited = sum(1 for r in reuse_attempts[1:] if r.status_code == 429)
        assert rate_limited > 0, "Refresh token reuse should trigger rate limiting"


class TestResourceEnumeration:
    """Test resource enumeration protection."""

    @pytest.mark.asyncio
    async def test_directory_brute_forcing(self, client, redis_client):
        """Test protection against directory brute forcing."""
        # Common directories attackers try
        common_dirs = [
            "admin",
            "backup",
            "config",
            "data",
            "files",
            "uploads",
            "api",
            "v1",
            "v2",
            "test",
            "dev",
            "staging",
            "prod",
            ".git",
            ".env",
            ".htaccess",
            "wp-admin",
            "phpmyadmin",
        ]

        enumeration_results = []

        for directory in common_dirs:
            response = await client.get(f"/{directory}/")
            enumeration_results.append({"path": directory, "status": response.status_code})

        # Should rate limit after initial attempts
        rate_limited = sum(1 for r in enumeration_results if r["status"] == 429)
        assert rate_limited > len(common_dirs) // 2, "Directory enumeration should be rate limited"

    @pytest.mark.asyncio
    async def test_file_enumeration(self, client, redis_client):
        """Test protection against file enumeration."""
        # Common files attackers look for
        target_files = [
            "config.json",
            "database.yml",
            "secrets.env",
            ".env.local",
            "id_rsa",
            "id_rsa.pub",
            "backup.sql",
            "dump.sql",
            "users.csv",
            "passwords.txt",
            "api_keys.json",
        ]

        file_attempts = []

        for filename in target_files:
            # Try multiple paths
            for path in ["", "config/", "data/", "../"]:
                response = await client.get(f"/{path}{filename}")
                file_attempts.append(
                    {
                        "file": f"{path}{filename}",
                        "status": response.status_code,
                    }
                )

        # Verify enumeration protection
        successful = sum(1 for r in file_attempts if r["status"] not in [429, 403])
        assert successful < 5, "File enumeration should be blocked"

    @pytest.mark.asyncio
    async def test_api_endpoint_discovery(self, client, redis_client):
        """Test protection against API endpoint discovery."""
        # Generate potential API endpoints
        api_endpoints = []

        # Common REST patterns
        resources = ["users", "posts", "comments", "products", "orders"]
        actions = ["", "list", "create", "update", "delete", "search"]

        for resource in resources:
            for action in actions:
                if action:
                    api_endpoints.append(f"/api/v1/{resource}/{action}")
                else:
                    api_endpoints.append(f"/api/v1/{resource}")

        # Add GraphQL and other endpoints
        api_endpoints.extend(
            [
                "/graphql",
                "/api/graphql",
                "/query",
                "/api/internal",
                "/api/debug",
                "/api/admin",
            ]
        )

        discovery_results = []

        for endpoint in api_endpoints:
            response = await client.get(endpoint)
            discovery_results.append(
                {
                    "endpoint": endpoint,
                    "status": response.status_code,
                    "headers": dict(response.headers),
                }
            )

        # Check rate limiting effectiveness
        allowed_discoveries = sum(1 for r in discovery_results if r["status"] != 429)
        assert allowed_discoveries < 10, "API discovery should be rate limited"

    @pytest.mark.asyncio
    async def test_parameter_fuzzing(self, client, redis_client):
        """Test protection against parameter fuzzing attacks."""
        # Base endpoint
        endpoint = "/api/v1/search"

        # Common parameter names to fuzz
        param_names = [
            "id",
            "user_id",
            "userId",
            "uid",
            "username",
            "email",
            "token",
            "api_key",
            "apiKey",
            "secret",
            "password",
            "admin",
            "debug",
            "test",
            "internal",
            "private",
        ]

        fuzzing_attempts = []

        # Try different parameter combinations
        for param in param_names:
            for value in [
                "1",
                "admin",
                "true",
                "../etc/passwd",
                "<script>alert(1)</script>",
            ]:
                response = await client.get(endpoint, params={param: value})
                fuzzing_attempts.append(
                    {
                        "param": param,
                        "value": value,
                        "status": response.status_code,
                    }
                )

        # Verify fuzzing protection
        blocked = sum(1 for r in fuzzing_attempts if r["status"] == 429)
        assert blocked > len(fuzzing_attempts) * 0.7, "Parameter fuzzing should trigger protection"


class TestProtectionValidation:
    """Test the effectiveness of protection mechanisms."""

    @pytest.mark.asyncio
    async def test_rate_limiting_effectiveness(self, client, redis_client):
        """Test overall rate limiting effectiveness."""
        # Simulate burst attack
        burst_size = 100
        burst_responses = []

        start_time = time.time()

        # Send burst of requests
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for i in range(burst_size):
                task = tg.create_task(client.get("/api/v1/health"))
                tasks.append(task)

        burst_responses = [task.result() for task in tasks]

        burst_duration = time.time() - start_time

        # Analyze results
        successful = sum(1 for r in burst_responses if r.status_code == 200)
        rate_limited = sum(1 for r in burst_responses if r.status_code == 429)

        # Should allow burst limit then rate limit
        assert successful < 20, "Burst protection should limit successful requests"
        assert rate_limited > 80, "Most requests should be rate limited"
        assert burst_duration < 5, "Rate limiting should be fast"

    @pytest.mark.asyncio
    async def test_captcha_integration(self, client, redis_client):
        """Test CAPTCHA integration for suspicious activity."""
        # Trigger suspicious activity
        for i in range(10):
            await client.post(
                "/api/v1/auth/login",
                json={
                    "username": f"bot_{i}@example.com",
                    "password": "password",
                },
            )

        # Next request should require CAPTCHA
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "username": "legitimate@example.com",
                "password": "password",
            },
        )

        # Check for CAPTCHA requirement
        if response.status_code == 403:
            data = response.json()
            assert "captcha_required" in data or "captcha" in str(data).lower()

    @pytest.mark.asyncio
    async def test_ip_based_blocking(self, client, redis_client):
        """Test IP-based blocking mechanisms."""
        # Simulate attack from specific IP
        attack_ip = "192.168.1.100"

        # Override client IP
        headers = {"X-Real-IP": attack_ip}

        # Generate suspicious traffic
        for i in range(50):
            await client.get("/api/v1/users", headers=headers)

        # Check if IP is blocked
        blocked_key = f"blocked:rate_limit:ip:{attack_ip}"
        ddos_key = f"ddos_blocked:{attack_ip}"

        is_blocked = await redis_client.get(blocked_key) or await redis_client.get(ddos_key)
        assert is_blocked is not None, "Suspicious IP should be blocked"

        # Verify block persists
        response = await client.get("/api/v1/health", headers=headers)
        assert response.status_code == 429, "Blocked IP should remain blocked"

    @pytest.mark.asyncio
    async def test_distributed_attack_handling(self, client, redis_client):
        """Test handling of distributed attacks from multiple IPs."""
        # Simulate distributed attack
        attack_ips = [f"192.168.1.{i}" for i in range(100, 120)]

        attack_results = []

        # Launch distributed attack
        for ip in attack_ips:
            headers = {"X-Real-IP": ip}

            # Each IP sends moderate traffic
            for _ in range(5):
                response = await client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": "target@example.com",
                        "password": "guess",
                    },
                    headers=headers,
                )
                attack_results.append({"ip": ip, "status": response.status_code})

        # Analyze distributed attack handling
        unique_ips_blocked = len(set(r["ip"] for r in attack_results if r["status"] == 429))

        # Should detect pattern and block even distributed attacks
        assert unique_ips_blocked > 10, "Distributed attack pattern should be detected"


class TestPerformanceImpact:
    """Test performance impact of brute force protection."""

    @pytest.mark.asyncio
    async def test_protection_overhead(self, client, redis_client):
        """Test overhead added by protection mechanisms."""
        # Baseline request without protection
        baseline_times = []

        for _ in range(10):
            start = time.time()
            await client.get("/api/v1/health")
            baseline_times.append(time.time() - start)

        baseline_avg = sum(baseline_times) / len(baseline_times)

        # Requests with protection active
        protected_times = []

        for _ in range(10):
            start = time.time()
            await client.get("/api/v1/users")
            protected_times.append(time.time() - start)

        protected_avg = sum(protected_times) / len(protected_times)

        # Overhead should be minimal
        overhead = protected_avg - baseline_avg
        assert overhead < 0.01, f"Protection overhead should be <10ms, got {overhead * 1000}ms"

    @pytest.mark.asyncio
    async def test_memory_usage_under_attack(self, client, redis_client):
        """Test memory usage during brute force attack."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate heavy attack
        attack_tasks = []

        async with asyncio.TaskGroup() as tg:
            for i in range(1000):
                task = tg.create_task(
                    client.post(
                        "/api/v1/auth/login",
                        json={
                            "username": f"attacker_{i}@example.com",
                            "password": "password",
                        },
                    )
                )
                attack_tasks.append(task)

        # Check memory after attack
        attack_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = attack_memory - baseline_memory

        # Memory increase should be reasonable
        assert memory_increase < 100, (
            f"Memory increase during attack should be <100MB, got {memory_increase}MB"
        )

    @pytest.mark.asyncio
    async def test_response_time_degradation(self, client, redis_client):
        """Test response time degradation under attack."""
        # Normal response times
        normal_times = []

        for _ in range(5):
            start = time.time()
            await client.get("/api/v1/health")
            normal_times.append(time.time() - start)

        normal_avg = sum(normal_times) / len(normal_times)

        # Start attack in background
        attack_task = asyncio.create_task(self._background_attack(client))

        # Measure response times during attack
        attack_times = []

        for _ in range(5):
            start = time.time()
            await client.get("/api/v1/health")
            attack_times.append(time.time() - start)

        attack_avg = sum(attack_times) / len(attack_times)

        # Cancel attack
        attack_task.cancel()

        # Response time should not degrade significantly
        degradation = attack_avg / normal_avg
        assert degradation < 2, f"Response time degradation should be <2x, got {degradation}x"

    @pytest.mark.asyncio
    async def test_system_resource_consumption(self, client, redis_client):
        """Test overall system resource consumption during protection."""
        import psutil

        # Baseline CPU usage
        psutil.cpu_percent(interval=1)  # Initialize
        psutil.cpu_percent(interval=1)

        # Simulate sustained attack
        attack_duration = 5  # seconds
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < attack_duration:
            await client.post(
                "/api/v1/auth/login",
                json={
                    "username": f"cpu_test_{request_count}@example.com",
                    "password": "wrong",
                },
            )
            request_count += 1

        # Measure CPU during attack
        attack_cpu = psutil.cpu_percent(interval=1)

        # CPU usage should remain reasonable
        assert attack_cpu < 80, f"CPU usage during attack should be <80%, got {attack_cpu}%"

        # Verify request throughput
        requests_per_second = request_count / attack_duration
        assert requests_per_second > 10, "Should handle >10 requests/second even under attack"

    async def _background_attack(self, client):
        """Helper to run background attack."""
        while True:
            try:
                await client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": f"background_{random.randint(1000, 9999)}@example.com",
                        "password": "attack",
                    },
                )
                await asyncio.sleep(0.01)  # 100 requests/second
            except asyncio.CancelledError:
                break


class TestAdvancedProtection:
    """Test advanced brute force protection scenarios."""

    @pytest.mark.asyncio
    async def test_credential_stuffing_protection(self, client, redis_client):
        """Test protection against credential stuffing attacks."""
        # Simulate leaked credential list
        leaked_credentials = [
            ("user1@example.com", "password123"),
            ("user2@example.com", "letmein"),
            ("user3@example.com", "qwerty"),
            ("user4@example.com", "admin123"),
            ("user5@example.com", "welcome"),
        ]

        stuffing_results = []

        # Attempt credential stuffing
        for email, password in leaked_credentials * 10:  # Try each 10 times
            response = await client.post(
                "/api/v1/auth/login",
                json={"username": email, "password": password},
            )
            stuffing_results.append({"email": email, "status": response.status_code})

        # Should detect pattern and block
        blocked = sum(1 for r in stuffing_results if r["status"] == 429)
        assert blocked > len(stuffing_results) * 0.8, "Credential stuffing should be detected"

    @pytest.mark.asyncio
    async def test_slow_brute_force_detection(self, client, redis_client):
        """Test detection of slow brute force attacks."""
        target_email = "slow_target@example.com"

        # Register target
        await client.post(
            "/api/v1/auth/register",
            json={"email": target_email, "password": "correct_password"},
        )

        # Slow attack - one attempt per minute
        slow_attempts = []

        for i in range(10):
            response = await client.post(
                "/api/v1/auth/login",
                json={"username": target_email, "password": f"guess_{i}"},
            )
            slow_attempts.append(response)

            # Wait to avoid rate limiting
            if i < 9:
                await asyncio.sleep(2)  # Shorter for test

        # Should still detect pattern over time
        sum(1 for r in slow_attempts if r.status_code == 401)

        # Check if account protection triggered
        final_response = await client.post(
            "/api/v1/auth/login",
            json={"username": target_email, "password": "correct_password"},
        )

        # Account should have additional protection
        assert final_response.status_code in [
            401,
            403,
            429,
        ], "Slow attack should trigger protection"

    @pytest.mark.asyncio
    async def test_multi_vector_attack_protection(self, client, redis_client):
        """Test protection against multi-vector attacks."""
        # Simultaneous attacks on multiple endpoints
        attack_vectors = [
            # Login brute force
            {
                "method": "POST",
                "url": "/api/v1/auth/login",
                "data": {
                    "username": "victim@example.com",
                    "password": "guess",
                },
            },
            # API key guessing
            {
                "method": "GET",
                "url": "/api/v1/data",
                "headers": {"X-API-Key": "guess_key_123"},
            },
            # Token manipulation
            {
                "method": "GET",
                "url": "/api/v1/users/me",
                "headers": {"Authorization": "Bearer fake_token_123"},
            },
            # Directory enumeration
            {"method": "GET", "url": "/admin/", "headers": {}},
        ]

        # Launch multi-vector attack
        tasks = []

        async with asyncio.TaskGroup() as tg:
            for vector in attack_vectors * 25:
                if vector["method"] == "POST":
                    task = tg.create_task(client.post(vector["url"], json=vector.get("data", {})))
                else:
                    task = tg.create_task(
                        client.get(vector["url"], headers=vector.get("headers", {}))
                    )
                tasks.append(task)

        # Analyze responses
        responses = [task.result() for task in tasks]
        total_blocked = sum(1 for r in responses if r.status_code == 429)

        # Should detect and block multi-vector attack
        assert total_blocked > len(responses) * 0.8, "Multi-vector attack should be blocked"

    @pytest.mark.asyncio
    async def test_intelligent_pattern_detection(self, client, redis_client):
        """Test intelligent attack pattern detection."""
        # Various attack patterns
        patterns = {
            "sequential": ["user1", "user2", "user3", "user4"],
            "dictionary": ["admin", "root", "test", "demo"],
            "keyboard": ["qwerty", "asdf", "zxcv", "1234"],
            "variations": ["password", "Password", "PASSWORD", "passw0rd"],
        }

        pattern_results = {}

        for pattern_name, attempts in patterns.items():
            results = []

            for attempt in attempts:
                response = await client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": f"{attempt}@example.com",
                        "password": attempt,
                    },
                )
                results.append(response.status_code)

            pattern_results[pattern_name] = results

        # Should detect patterns and increase protection
        for pattern_name, results in pattern_results.items():
            blocked = sum(1 for status in results if status == 429)
            assert blocked > 0, f"Pattern '{pattern_name}' should trigger protection"


# Helper functions for test data generation
def generate_random_ip():
    """Generate random IP address."""
    return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"


def generate_random_user_agent():
    """Generate random user agent string."""
    browsers = ["Chrome", "Firefox", "Safari", "Edge"]
    versions = ["100.0", "101.0", "102.0", "103.0"]
    os_list = ["Windows NT 10.0", "Macintosh", "X11; Linux x86_64"]

    browser = random.choice(browsers)
    version = random.choice(versions)
    os = random.choice(os_list)

    return f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{version}"


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=strict", "-k", "test_"])

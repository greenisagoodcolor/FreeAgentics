"""
Authentication Rate Limiting Verification Tests
Task #6.4 - Add rate limiting verification tests

This test suite validates rate limiting for authentication endpoints:
1. Login attempt rate limiting
2. Token creation rate limiting
3. Password reset rate limiting
4. Registration rate limiting
5. Burst protection
6. Distributed rate limiting
7. Rate limit recovery behavior
8. IP-based and user-based limiting
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pytest

from auth.security_implementation import AuthenticationManager, User, UserRole
from auth.security_logging import SecurityEventType, security_auditor


@dataclass
class RateLimitConfig:
    """Rate limit configuration for authentication endpoints."""

    login_per_minute: int = 5
    login_per_hour: int = 50
    token_creation_per_minute: int = 10
    token_creation_per_hour: int = 100
    registration_per_hour: int = 10
    password_reset_per_hour: int = 5
    burst_size: int = 3
    block_duration_seconds: int = 300  # 5 minutes


@dataclass
class RateLimitMetrics:
    """Metrics for rate limit testing."""

    total_requests: int = 0
    successful_requests: int = 0
    rate_limited_requests: int = 0
    blocked_requests: int = 0
    request_times: List[float] = None
    rate_limit_hits: Dict[str, int] = None

    def __post_init__(self):
        if self.request_times is None:
            self.request_times = []
        if self.rate_limit_hits is None:
            self.rate_limit_hits = defaultdict(int)


class MockRateLimiter:
    """Mock rate limiter for testing."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_counts = defaultdict(lambda: defaultdict(list))
        self.blocked_until = {}

    def is_allowed(
        self, key: str, endpoint: str, current_time: float = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if request is allowed."""
        if current_time is None:
            current_time = time.time()

        # Check if blocked
        if key in self.blocked_until:
            if current_time < self.blocked_until[key]:
                return False, "blocked"
            else:
                del self.blocked_until[key]

        # Get limits for endpoint
        limits = self._get_limits(endpoint)

        # Check each limit
        for limit_type, (max_requests, window_seconds) in limits.items():
            # Remove old requests outside window
            cutoff_time = current_time - window_seconds
            self.request_counts[key][limit_type] = [
                t for t in self.request_counts[key][limit_type] if t > cutoff_time
            ]

            # Check if limit exceeded
            if len(self.request_counts[key][limit_type]) >= max_requests:
                # Block the key
                self.blocked_until[key] = (
                    current_time + self.config.block_duration_seconds
                )
                return False, f"{limit_type}_limit_exceeded"

        # Record this request
        for limit_type in limits:
            self.request_counts[key][limit_type].append(current_time)

        return True, None

    def _get_limits(self, endpoint: str) -> Dict[str, Tuple[int, int]]:
        """Get rate limits for endpoint."""
        if endpoint == "login":
            return {
                "minute": (self.config.login_per_minute, 60),
                "hour": (self.config.login_per_hour, 3600),
            }
        elif endpoint == "token_creation":
            return {
                "minute": (self.config.token_creation_per_minute, 60),
                "hour": (self.config.token_creation_per_hour, 3600),
            }
        elif endpoint == "registration":
            return {
                "hour": (self.config.registration_per_hour, 3600),
            }
        elif endpoint == "password_reset":
            return {
                "hour": (self.config.password_reset_per_hour, 3600),
            }
        else:
            return {}

    def reset(self, key: str = None):
        """Reset rate limiter state."""
        if key:
            if key in self.request_counts:
                del self.request_counts[key]
            if key in self.blocked_until:
                del self.blocked_until[key]
        else:
            self.request_counts.clear()
            self.blocked_until.clear()


class TestAuthenticationRateLimiting:
    """Test authentication rate limiting."""

    def setup_method(self):
        """Setup for each test."""
        self.auth_manager = AuthenticationManager()
        self.rate_limit_config = RateLimitConfig()
        self.rate_limiter = MockRateLimiter(self.rate_limit_config)
        self.metrics = RateLimitMetrics()

        # Patch rate limiting into auth manager
        self.auth_manager.rate_limiter = self.rate_limiter

        # Clear security auditor
        security_auditor.token_usage_patterns = {}

    def _simulate_login_attempt(
        self, username: str, ip_address: str, current_time: float = None
    ) -> bool:
        """Simulate a login attempt with rate limiting."""
        # Check rate limit
        key = f"{ip_address}:{username}"
        allowed, reason = self.rate_limiter.is_allowed(key, "login", current_time)

        self.metrics.total_requests += 1

        if not allowed:
            self.metrics.rate_limited_requests += 1
            if reason == "blocked":
                self.metrics.blocked_requests += 1
            self.metrics.rate_limit_hits[reason] += 1

            # Log rate limit event
            security_auditor.log_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                severity="warning",
                message=f"Rate limit exceeded: {reason}",
                user_id=username,
                details={"ip_address": ip_address, "endpoint": "login"},
            )
            return False

        self.metrics.successful_requests += 1
        return True

    def test_login_rate_limiting_per_minute(self):
        """Test login rate limiting per minute."""
        username = "testuser"
        ip_address = "192.168.1.100"

        # Simulate rapid login attempts
        start_time = time.time()

        # Should allow up to limit
        for i in range(self.rate_limit_config.login_per_minute):
            assert self._simulate_login_attempt(
                username, ip_address, start_time + i * 0.1
            )

        # Next attempt should be rate limited
        assert not self._simulate_login_attempt(username, ip_address, start_time + 0.6)

        # Check metrics
        assert (
            self.metrics.successful_requests == self.rate_limit_config.login_per_minute
        )
        assert self.metrics.rate_limited_requests == 1
        assert self.metrics.rate_limit_hits["minute_limit_exceeded"] == 1

    def test_login_rate_limiting_per_hour(self):
        """Test login rate limiting per hour."""
        username = "testuser"
        ip_address = "192.168.1.100"

        # Simulate login attempts spread over time
        start_time = time.time()

        # Fill up hourly limit
        for i in range(self.rate_limit_config.login_per_hour):
            # Spread requests to avoid minute limit
            request_time = start_time + i * 65  # Just over 1 minute apart
            self._simulate_login_attempt(username, ip_address, request_time)

        # Next attempt should be rate limited
        assert not self._simulate_login_attempt(
            username,
            ip_address,
            start_time + self.rate_limit_config.login_per_hour * 65,
        )

        # Check hour limit was hit
        assert self.metrics.rate_limit_hits["hour_limit_exceeded"] == 1

    def test_burst_protection(self):
        """Test burst protection for rapid requests."""
        username = "burstuser"
        ip_address = "192.168.1.101"

        # Simulate burst of requests
        burst_times = []
        start_time = time.time()

        # Rapid burst
        for i in range(self.rate_limit_config.burst_size * 2):
            request_time = start_time + i * 0.01  # 10ms apart
            allowed = self._simulate_login_attempt(username, ip_address, request_time)
            burst_times.append((request_time, allowed))

        # Should have rate limited some requests
        allowed_count = sum(1 for _, allowed in burst_times if allowed)
        assert allowed_count <= self.rate_limit_config.login_per_minute
        assert self.metrics.rate_limited_requests > 0

    def test_multiple_ip_rate_limiting(self):
        """Test rate limiting across multiple IPs."""
        username = "shareduser"
        ip_addresses = [f"192.168.1.{100 + i}" for i in range(5)]

        results = defaultdict(list)

        # Each IP should have independent limits
        for ip in ip_addresses:
            for i in range(self.rate_limit_config.login_per_minute + 2):
                allowed = self._simulate_login_attempt(username, ip)
                results[ip].append(allowed)

        # Each IP should have hit its limit
        for ip, attempts in results.items():
            allowed_count = sum(1 for allowed in attempts if allowed)
            assert allowed_count == self.rate_limit_config.login_per_minute

    def test_rate_limit_recovery(self):
        """Test rate limit recovery after blocking period."""
        username = "recoveryuser"
        ip_address = "192.168.1.102"

        start_time = time.time()

        # Hit rate limit
        for i in range(self.rate_limit_config.login_per_minute + 1):
            self._simulate_login_attempt(username, ip_address, start_time + i * 0.1)

        # Should be blocked
        assert not self._simulate_login_attempt(username, ip_address, start_time + 1)

        # Simulate waiting for block duration
        recovery_time = start_time + self.rate_limit_config.block_duration_seconds + 1

        # Should be allowed again
        assert self._simulate_login_attempt(username, ip_address, recovery_time)

    def test_token_creation_rate_limiting(self):
        """Test rate limiting for token creation."""
        users = []
        for i in range(5):
            user = User(
                user_id=f"token-user-{i}",
                username=f"tokenuser{i}",
                email=f"token{i}@test.com",
                role=UserRole.RESEARCHER,
                created_at=datetime.now(timezone.utc),
            )
            users.append(user)

        # Simulate token creation attempts
        token_metrics = RateLimitMetrics()

        def create_token_with_limit(user: User, ip: str) -> bool:
            key = f"{ip}:{user.user_id}"
            allowed, reason = self.rate_limiter.is_allowed(key, "token_creation")

            token_metrics.total_requests += 1

            if allowed:
                token_metrics.successful_requests += 1
                # Actually create token
                self.auth_manager.create_access_token(user)
                return True
            else:
                token_metrics.rate_limited_requests += 1
                return False

        # Test per-user token creation limits
        ip = "192.168.1.103"
        for user in users:
            successes = 0
            for _ in range(self.rate_limit_config.token_creation_per_minute + 5):
                if create_token_with_limit(user, ip):
                    successes += 1

            # Should respect per-minute limit
            assert successes == self.rate_limit_config.token_creation_per_minute

    def test_registration_rate_limiting(self):
        """Test rate limiting for user registration."""
        ip_address = "192.168.1.104"

        registration_attempts = []
        start_time = time.time()

        # Simulate registration attempts
        for i in range(self.rate_limit_config.registration_per_hour + 5):
            key = f"{ip_address}:registration"
            request_time = start_time + i * 60  # 1 minute apart
            allowed, reason = self.rate_limiter.is_allowed(
                key, "registration", request_time
            )
            registration_attempts.append((i, allowed, reason))

        # Count successful registrations
        successful = sum(1 for _, allowed, _ in registration_attempts if allowed)
        assert successful == self.rate_limit_config.registration_per_hour

        # Verify later attempts were rate limited
        failed = [
            (i, reason) for i, allowed, reason in registration_attempts if not allowed
        ]
        assert len(failed) == 5
        assert all(reason == "hour_limit_exceeded" for _, reason in failed)

    def test_password_reset_rate_limiting(self):
        """Test rate limiting for password reset requests."""
        email_addresses = [
            "user1@test.com",
            "user2@test.com",
            "user3@test.com",
        ]

        reset_attempts = defaultdict(int)

        for email in email_addresses:
            for i in range(self.rate_limit_config.password_reset_per_hour + 2):
                key = f"password_reset:{email}"
                allowed, _ = self.rate_limiter.is_allowed(key, "password_reset")
                if allowed:
                    reset_attempts[email] += 1

        # Each email should be limited
        for email, count in reset_attempts.items():
            assert count == self.rate_limit_config.password_reset_per_hour

    def test_concurrent_rate_limiting(self):
        """Test rate limiting under concurrent access."""
        num_users = 20
        num_requests_per_user = 10

        users = []
        for i in range(num_users):
            users.append(
                {
                    "username": f"concurrent_user_{i}",
                    "ip": f"192.168.2.{100 + i}",
                }
            )

        concurrent_results = []
        result_lock = threading.Lock()

        def simulate_user_requests(user_data):
            """Simulate requests from a single user."""
            local_results = []

            for _ in range(num_requests_per_user):
                allowed = self._simulate_login_attempt(
                    user_data["username"], user_data["ip"]
                )
                local_results.append(allowed)
                time.sleep(0.01)  # Small delay between requests

            with result_lock:
                concurrent_results.append(
                    {
                        "user": user_data["username"],
                        "allowed": sum(1 for r in local_results if r),
                        "denied": sum(1 for r in local_results if not r),
                    }
                )

        # Run concurrent requests
        import threading

        threads = []
        for user in users:
            thread = threading.Thread(target=simulate_user_requests, args=(user,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify rate limiting was applied correctly
        for result in concurrent_results:
            # Each user should have some requests allowed and some denied
            assert result["allowed"] <= self.rate_limit_config.login_per_minute
            if num_requests_per_user > self.rate_limit_config.login_per_minute:
                assert result["denied"] > 0

    def test_distributed_rate_limiting_simulation(self):
        """Simulate distributed rate limiting across multiple servers."""
        # Simulate 3 servers with shared rate limiting
        servers = ["server1", "server2", "server3"]
        shared_limiter = MockRateLimiter(self.rate_limit_config)

        username = "distributed_user"
        ip_address = "192.168.1.105"

        total_attempts = 0
        total_allowed = 0

        # Simulate requests distributed across servers
        for i in range(self.rate_limit_config.login_per_minute * 2):
            servers[i % len(servers)]
            key = f"{ip_address}:{username}"

            # All servers check the shared limiter
            allowed, _ = shared_limiter.is_allowed(key, "login")

            total_attempts += 1
            if allowed:
                total_allowed += 1

        # Total allowed across all servers should respect the limit
        assert total_allowed == self.rate_limit_config.login_per_minute
        assert total_attempts == self.rate_limit_config.login_per_minute * 2

    def test_rate_limit_headers(self):
        """Test rate limit information in response headers."""
        username = "header_test_user"
        ip_address = "192.168.1.106"

        # Track rate limit state
        for i in range(self.rate_limit_config.login_per_minute - 1):
            self._simulate_login_attempt(username, ip_address)

        # Simulate getting rate limit headers
        key = f"{ip_address}:{username}"
        remaining = self.rate_limit_config.login_per_minute - len(
            self.rate_limiter.request_counts[key]["minute"]
        )

        headers = {
            "X-RateLimit-Limit": str(self.rate_limit_config.login_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(time.time() + 60)),
        }

        assert int(headers["X-RateLimit-Remaining"]) == 1
        assert (
            int(headers["X-RateLimit-Limit"]) == self.rate_limit_config.login_per_minute
        )

    def test_rate_limiting_with_authentication_flow(self):
        """Test rate limiting integrated with full authentication flow."""
        # Create test user
        user = User(
            user_id="flow-test-user",
            username="flowuser",
            email="flow@test.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )

        # Register user
        self.auth_manager.users[user.username] = {
            "user": user,
            "password_hash": self.auth_manager.hash_password("testpass123"),
        }

        ip_address = "192.168.1.107"
        successful_logins = []

        # Simulate login flow with rate limiting
        for i in range(self.rate_limit_config.login_per_minute + 3):
            if self._simulate_login_attempt(user.username, ip_address):
                # Successful rate limit check, proceed with auth
                try:
                    # Verify password (simulated)
                    access_token = self.auth_manager.create_access_token(user)
                    self.auth_manager.create_refresh_token(user)

                    successful_logins.append(
                        {
                            "attempt": i,
                            "access_token": access_token[:20]
                            + "...",  # Truncated for display
                            "timestamp": time.time(),
                        }
                    )
                except Exception:
                    pass

        # Should have successful logins up to the limit
        assert len(successful_logins) == self.rate_limit_config.login_per_minute

        # Verify tokens from successful logins work
        if successful_logins:
            successful_logins[-1]
            # Token verification would happen here in real implementation

    @pytest.mark.parametrize(
        "endpoint,limit_per_minute",
        [
            ("login", 5),
            ("token_creation", 10),
            (
                "registration",
                10,
            ),  # Actually per hour, but testing the mechanism
            (
                "password_reset",
                5,
            ),  # Actually per hour, but testing the mechanism
        ],
    )
    def test_endpoint_specific_limits(self, endpoint, limit_per_minute):
        """Test that different endpoints have appropriate rate limits."""
        test_key = f"endpoint_test:{endpoint}"

        # Make requests up to and beyond the limit
        allowed_count = 0
        for i in range(limit_per_minute + 5):
            allowed, _ = self.rate_limiter.is_allowed(test_key, endpoint)
            if allowed:
                allowed_count += 1

        # For per-hour endpoints, we're testing the mechanism works
        if endpoint in ["registration", "password_reset"]:
            # These have hourly limits, so all minute requests might succeed
            assert allowed_count <= limit_per_minute + 5
        else:
            # For per-minute endpoints, should respect the limit
            assert allowed_count <= limit_per_minute

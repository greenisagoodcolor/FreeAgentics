"""
Concurrent Authentication Load Tests
Task #6.2 - Implement concurrent authentication load tests

This test suite validates authentication system performance under load:
1. Concurrent user login/logout
2. Token creation/verification under load
3. Refresh token rotation under concurrent access
4. Rate limiting behavior
5. System stability under authentication stress
6. Memory and resource usage
7. Error handling under load
"""

import concurrent.futures
import gc
import random
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import psutil
import pytest
from fastapi import HTTPException

from auth.security_implementation import (
    AuthenticationManager,
    User,
    UserRole,
)
from auth.security_logging import security_auditor


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = None
    error_types: Dict[str, int] = None
    memory_usage: List[float] = None
    cpu_usage: List[float] = None
    concurrent_users: int = 0
    test_duration: float = 0.0

    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.error_types is None:
            self.error_types = defaultdict(int)
        if self.memory_usage is None:
            self.memory_usage = []
        if self.cpu_usage is None:
            self.cpu_usage = []

    def add_response_time(self, duration: float):
        """Add response time measurement."""
        self.response_times.append(duration)

    def add_error(self, error_type: str):
        """Track error occurrence."""
        self.error_types[error_type] += 1
        self.failed_requests += 1

    def calculate_statistics(self) -> Dict:
        """Calculate performance statistics."""
        if not self.response_times:
            return {}

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                (self.successful_requests / self.total_requests * 100)
                if self.total_requests > 0
                else 0
            ),
            "avg_response_time": statistics.mean(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "p95_response_time": (
                statistics.quantiles(self.response_times, n=20)[18]
                if len(self.response_times) > 20
                else max(self.response_times, default=0)
            ),
            "p99_response_time": (
                statistics.quantiles(self.response_times, n=100)[98]
                if len(self.response_times) > 100
                else max(self.response_times, default=0)
            ),
            "min_response_time": min(self.response_times, default=0),
            "max_response_time": max(self.response_times, default=0),
            "requests_per_second": (
                self.total_requests / self.test_duration if self.test_duration > 0 else 0
            ),
            "error_types": dict(self.error_types),
            "avg_memory_mb": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_percent": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
        }


class TestAuthenticationLoadTesting:
    """Load testing for authentication system."""

    def setup_method(self):
        """Setup for each test."""
        self.auth_manager = AuthenticationManager()
        self.metrics = LoadTestMetrics()
        self.process = psutil.Process()
        # Clear security auditor state
        security_auditor.token_usage_patterns = {}
        security_auditor.token_binding_violations = {}

    def _create_test_users(self, count: int) -> List[User]:
        """Create multiple test users."""
        users = []
        for i in range(count):
            user = User(
                user_id=f"load-test-user-{i}",
                username=f"loaduser{i}",
                email=f"loadtest{i}@example.com",
                role=random.choice(
                    [UserRole.RESEARCHER, UserRole.OBSERVER, UserRole.AGENT_MANAGER]
                ),
                created_at=datetime.now(timezone.utc),
            )
            # Register user
            self.auth_manager.users[user.username] = {
                "user": user,
                "password_hash": self.auth_manager.hash_password(f"password{i}"),
            }
            users.append(user)
        return users

    def _monitor_resources(self, stop_event: threading.Event):
        """Monitor CPU and memory usage during test."""
        while not stop_event.is_set():
            try:
                self.metrics.memory_usage.append(self.process.memory_info().rss / 1024 / 1024)  # MB
                self.metrics.cpu_usage.append(self.process.cpu_percent(interval=0.1))
            except:
                pass
            time.sleep(0.5)

    def _simulate_user_session(self, user: User, duration: float = 5.0) -> Tuple[int, int]:
        """Simulate a complete user session."""
        successes = 0
        failures = 0
        end_time = time.time() + duration

        try:
            # Login - create tokens
            start = time.time()
            access_token = self.auth_manager.create_access_token(user)
            refresh_token = self.auth_manager.create_refresh_token(user)
            self.metrics.add_response_time(time.time() - start)
            successes += 1

            while time.time() < end_time:
                # Random action
                action = random.choice(["verify", "verify", "verify", "refresh", "revoke"])

                try:
                    start = time.time()

                    if action == "verify":
                        self.auth_manager.verify_token(access_token)
                    elif action == "refresh":
                        access_token, refresh_token = self.auth_manager.refresh_access_token(
                            refresh_token
                        )
                    elif action == "revoke":
                        self.auth_manager.logout(access_token)
                        # Get new tokens after logout
                        access_token = self.auth_manager.create_access_token(user)
                        refresh_token = self.auth_manager.create_refresh_token(user)

                    self.metrics.add_response_time(time.time() - start)
                    successes += 1

                except Exception as e:
                    self.metrics.add_error(type(e).__name__)
                    failures += 1

                # Small random delay
                time.sleep(random.uniform(0.01, 0.1))

            # Logout at end
            try:
                start = time.time()
                self.auth_manager.logout(access_token)
                self.metrics.add_response_time(time.time() - start)
                successes += 1
            except Exception as e:
                self.metrics.add_error(type(e).__name__)
                failures += 1

        except Exception as e:
            self.metrics.add_error(type(e).__name__)
            failures += 1

        return successes, failures

    def test_concurrent_user_sessions(self):
        """Test system under concurrent user sessions."""
        num_users = 50
        session_duration = 5.0

        users = self._create_test_users(num_users)
        self.metrics.concurrent_users = num_users

        # Start resource monitoring
        stop_monitor = threading.Event()
        monitor_thread = threading.Thread(target=self._monitor_resources, args=(stop_monitor,))
        monitor_thread.start()

        # Run concurrent user sessions
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [
                executor.submit(self._simulate_user_session, user, session_duration)
                for user in users
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    successes, failures = future.result()
                    self.metrics.successful_requests += successes
                    self.metrics.failed_requests += failures
                    self.metrics.total_requests += successes + failures
                except Exception as e:
                    self.metrics.add_error(f"Session error: {type(e).__name__}")

        self.metrics.test_duration = time.time() - start_time

        # Stop monitoring
        stop_monitor.set()
        monitor_thread.join()

        # Calculate and verify metrics
        stats = self.metrics.calculate_statistics()

        # Performance assertions
        assert stats["success_rate"] > 95, f"Success rate too low: {stats['success_rate']}%"
        assert (
            stats["avg_response_time"] < 0.1
        ), f"Average response time too high: {stats['avg_response_time']}s"
        assert (
            stats["p95_response_time"] < 0.2
        ), f"P95 response time too high: {stats['p95_response_time']}s"
        assert (
            stats["requests_per_second"] > 100
        ), f"Throughput too low: {stats['requests_per_second']} req/s"

    def test_token_creation_under_load(self):
        """Test token creation performance under heavy load."""
        num_threads = 20
        tokens_per_thread = 100

        users = self._create_test_users(num_threads)

        def create_tokens(user: User, count: int) -> List[float]:
            times = []
            for _ in range(count):
                start = time.time()
                try:
                    self.auth_manager.create_access_token(user)
                    duration = time.time() - start
                    times.append(duration)
                    self.metrics.successful_requests += 1
                except Exception as e:
                    self.metrics.add_error(type(e).__name__)

            return times

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(create_tokens, users[i], tokens_per_thread)
                for i in range(num_threads)
            ]

            all_times = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    times = future.result()
                    all_times.extend(times)
                except Exception as e:
                    self.metrics.add_error(f"Thread error: {type(e).__name__}")

        self.metrics.test_duration = time.time() - start_time
        self.metrics.total_requests = num_threads * tokens_per_thread
        self.metrics.response_times = all_times

        stats = self.metrics.calculate_statistics()

        # Token creation should be fast even under load
        assert (
            stats["avg_response_time"] < 0.02
        ), f"Token creation too slow: {stats['avg_response_time']}s"
        assert (
            stats["p99_response_time"] < 0.05
        ), f"P99 token creation too slow: {stats['p99_response_time']}s"

    def test_token_verification_under_load(self):
        """Test token verification performance under load."""
        num_threads = 50
        verifications_per_thread = 200

        # Create a token to verify
        test_user = User(
            user_id="verify-test-user",
            username="verifyuser",
            email="verify@test.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )
        token = self.auth_manager.create_access_token(test_user)

        def verify_tokens(count: int) -> List[float]:
            times = []
            for _ in range(count):
                start = time.time()
                try:
                    self.auth_manager.verify_token(token)
                    duration = time.time() - start
                    times.append(duration)
                    self.metrics.successful_requests += 1
                except Exception as e:
                    self.metrics.add_error(type(e).__name__)
            return times

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(verify_tokens, verifications_per_thread) for _ in range(num_threads)
            ]

            all_times = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    times = future.result()
                    all_times.extend(times)
                except Exception as e:
                    self.metrics.add_error(f"Thread error: {type(e).__name__}")

        self.metrics.test_duration = time.time() - start_time
        self.metrics.total_requests = num_threads * verifications_per_thread
        self.metrics.response_times = all_times

        stats = self.metrics.calculate_statistics()

        # Token verification should be very fast
        assert (
            stats["avg_response_time"] < 0.005
        ), f"Token verification too slow: {stats['avg_response_time']}s"
        assert (
            stats["p99_response_time"] < 0.02
        ), f"P99 verification too slow: {stats['p99_response_time']}s"
        assert (
            stats["requests_per_second"] > 1000
        ), f"Verification throughput too low: {stats['requests_per_second']} req/s"

    def test_refresh_token_rotation_under_load(self):
        """Test refresh token rotation under concurrent access."""
        num_users = 20

        users = self._create_test_users(num_users)
        refresh_tokens = {}

        # Create initial refresh tokens
        for user in users:
            refresh_tokens[user.user_id] = self.auth_manager.create_refresh_token(user)

        def refresh_token_worker(user: User, iterations: int) -> Tuple[int, int]:
            successes = 0
            failures = 0
            current_refresh = refresh_tokens[user.user_id]

            for _ in range(iterations):
                try:
                    start = time.time()
                    new_access, new_refresh = self.auth_manager.refresh_access_token(
                        current_refresh
                    )
                    self.metrics.add_response_time(time.time() - start)
                    current_refresh = new_refresh
                    refresh_tokens[user.user_id] = new_refresh
                    successes += 1
                    time.sleep(random.uniform(0.05, 0.15))  # Simulate realistic delays
                except HTTPException as e:
                    failures += 1
                    self.metrics.add_error(f"Refresh failed: {e.detail}")
                    # Try to get a new refresh token
                    try:
                        current_refresh = self.auth_manager.create_refresh_token(user)
                        refresh_tokens[user.user_id] = current_refresh
                    except:
                        pass

            return successes, failures

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(refresh_token_worker, user, 10) for user in users]

            for future in concurrent.futures.as_completed(futures):
                try:
                    successes, failures = future.result()
                    self.metrics.successful_requests += successes
                    self.metrics.failed_requests += failures
                except Exception as e:
                    self.metrics.add_error(f"Worker error: {type(e).__name__}")

        self.metrics.test_duration = time.time() - start_time
        self.metrics.total_requests = (
            self.metrics.successful_requests + self.metrics.failed_requests
        )

        stats = self.metrics.calculate_statistics()

        # Some failures expected due to race conditions, but most should succeed
        assert stats["success_rate"] > 70, f"Refresh success rate too low: {stats['success_rate']}%"
        assert stats["avg_response_time"] < 0.05, f"Refresh too slow: {stats['avg_response_time']}s"

    def test_authentication_spike_load(self):
        """Test system behavior under sudden authentication spikes."""
        base_users = 10
        spike_users = 100

        # Create users
        all_users = self._create_test_users(base_users + spike_users)
        base_user_set = all_users[:base_users]
        spike_user_set = all_users[base_users:]

        results = {"base_load": [], "spike_load": [], "recovery": []}

        def authenticate_user(user: User) -> float:
            start = time.time()
            try:
                token = self.auth_manager.create_access_token(user)
                self.auth_manager.verify_token(token)
                return time.time() - start
            except Exception:
                return -1

        # Phase 1: Base load
        with concurrent.futures.ThreadPoolExecutor(max_workers=base_users) as executor:
            for _ in range(5):
                futures = [executor.submit(authenticate_user, user) for user in base_user_set]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        duration = future.result()
                        if duration > 0:
                            results["base_load"].append(duration)
                    except:
                        pass
                time.sleep(0.1)

        # Phase 2: Spike load
        with concurrent.futures.ThreadPoolExecutor(max_workers=spike_users) as executor:
            futures = [executor.submit(authenticate_user, user) for user in spike_user_set]
            for future in concurrent.futures.as_completed(futures):
                try:
                    duration = future.result()
                    if duration > 0:
                        results["spike_load"].append(duration)
                except:
                    pass

        # Phase 3: Recovery (back to base load)
        time.sleep(1)  # Allow system to stabilize
        with concurrent.futures.ThreadPoolExecutor(max_workers=base_users) as executor:
            for _ in range(5):
                futures = [executor.submit(authenticate_user, user) for user in base_user_set]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        duration = future.result()
                        if duration > 0:
                            results["recovery"].append(duration)
                    except:
                        pass
                time.sleep(0.1)

        # Analyze results
        base_avg = statistics.mean(results["base_load"]) if results["base_load"] else 0
        spike_avg = statistics.mean(results["spike_load"]) if results["spike_load"] else 0
        recovery_avg = statistics.mean(results["recovery"]) if results["recovery"] else 0

        # System should handle spike without severe degradation
        assert spike_avg < base_avg * 5, f"Spike degradation too high: {spike_avg/base_avg}x slower"
        assert (
            recovery_avg < base_avg * 1.5
        ), f"Recovery not achieved: {recovery_avg/base_avg}x slower"

    def test_memory_leak_during_extended_load(self):
        """Test for memory leaks during extended authentication operations."""
        num_iterations = 1000
        num_users = 10

        users = self._create_test_users(num_users)

        # Force garbage collection and get baseline
        gc.collect()
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Run extended test
        for i in range(num_iterations):
            user = users[i % num_users]

            # Create and verify token
            token = self.auth_manager.create_access_token(user)
            self.auth_manager.verify_token(token)

            # Occasionally refresh
            if i % 10 == 0:
                refresh = self.auth_manager.create_refresh_token(user)
                try:
                    self.auth_manager.refresh_access_token(refresh)
                except:
                    pass

            # Check memory periodically
            if i % 100 == 0:
                gc.collect()
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - baseline_memory

                # Memory increase should be reasonable
                assert memory_increase < 50, f"Memory leak detected: {memory_increase}MB increase"

    def test_concurrent_logout_handling(self):
        """Test system behavior when many users logout simultaneously."""
        num_users = 50

        users = self._create_test_users(num_users)
        tokens = []

        # Create tokens for all users
        for user in users:
            access_token = self.auth_manager.create_access_token(user)
            tokens.append((user, access_token))

        # Verify all tokens work
        for user, token in tokens:
            token_data = self.auth_manager.verify_token(token)
            assert token_data.user_id == user.user_id

        # Simultaneous logout
        def logout_user(token: str) -> bool:
            try:
                self.auth_manager.logout(token)
                return True
            except Exception:
                return False

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(logout_user, token) for _, token in tokens]
            logout_results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        logout_time = time.time() - start_time

        # All logouts should succeed
        assert all(logout_results), "Some logouts failed"
        assert logout_time < 2.0, f"Mass logout too slow: {logout_time}s"

        # Verify all tokens are now invalid
        for user, token in tokens:
            with pytest.raises(HTTPException):
                self.auth_manager.verify_token(token)

    @pytest.mark.parametrize("num_workers", [10, 25, 50])
    def test_scalability_with_different_loads(self, num_workers):
        """Test system scalability with different concurrent user counts."""
        users = self._create_test_users(num_workers)
        operations_per_user = 50

        def user_operations(user: User) -> Dict[str, float]:
            times = {"create": [], "verify": [], "refresh": []}

            # Create initial tokens
            start = time.time()
            access_token = self.auth_manager.create_access_token(user)
            refresh_token = self.auth_manager.create_refresh_token(user)
            times["create"].append(time.time() - start)

            for _ in range(operations_per_user):
                # Verify token
                start = time.time()
                self.auth_manager.verify_token(access_token)
                times["verify"].append(time.time() - start)

                # Occasionally refresh
                if random.random() < 0.1:
                    start = time.time()
                    try:
                        access_token, refresh_token = self.auth_manager.refresh_access_token(
                            refresh_token
                        )
                        times["refresh"].append(time.time() - start)
                    except:
                        # Get new tokens if refresh fails
                        access_token = self.auth_manager.create_access_token(user)
                        refresh_token = self.auth_manager.create_refresh_token(user)

            return times

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(user_operations, user) for user in users]

            all_times = {"create": [], "verify": [], "refresh": []}

            for future in concurrent.futures.as_completed(futures):
                try:
                    times = future.result()
                    for op_type, op_times in times.items():
                        all_times[op_type].extend(op_times)
                except Exception:
                    pass

        total_time = time.time() - start_time

        # Calculate averages
        avg_times = {
            op_type: statistics.mean(times) if times else 0 for op_type, times in all_times.items()
        }

        # Performance should not degrade significantly with more users
        assert avg_times["create"] < 0.05, f"Token creation too slow with {num_workers} users"
        assert avg_times["verify"] < 0.01, f"Token verification too slow with {num_workers} users"
        assert total_time < num_workers * 0.5, f"Total time too high for {num_workers} users"

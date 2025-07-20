"""
Authentication Performance and Load Testing

This test suite provides comprehensive performance testing for authentication:
- High-volume authentication request handling
- Concurrent user authentication scenarios
- Resource exhaustion testing under load
- Authentication performance benchmarks
- Stress testing for rate limiting
- Token generation and validation performance
- Database connection pooling under load
- Memory usage optimization validation
"""

import gc
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List

import psutil
import pytest
from fastapi.testclient import TestClient

from api.main import app
from auth.security_implementation import (
    AuthenticationManager,
    User,
    UserRole,
    rate_limiter,
)
from tests.performance.performance_utils import replace_sleep, cpu_work


class PerformanceMetrics:
    """Track performance metrics during testing."""

    def __init__(self):
        self.metrics = {
            "response_times": [],
            "throughput": [],
            "error_rates": [],
            "memory_usage": [],
            "cpu_usage": [],
            "concurrent_users": [],
            "token_generation_times": [],
            "token_validation_times": [],
            "database_query_times": [],
            "rate_limit_hits": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "successful_requests": 0,
        }

    def record_response_time(self, response_time: float):
        """Record response time."""
        self.metrics["response_times"].append(response_time)

    def record_throughput(self, requests_per_second: float):
        """Record throughput metric."""
        self.metrics["throughput"].append(requests_per_second)

    def record_error_rate(self, error_rate: float):
        """Record error rate."""
        self.metrics["error_rates"].append(error_rate)

    def record_memory_usage(self, memory_mb: float):
        """Record memory usage."""
        self.metrics["memory_usage"].append(memory_mb)

    def record_cpu_usage(self, cpu_percent: float):
        """Record CPU usage."""
        self.metrics["cpu_usage"].append(cpu_percent)

    def record_token_generation_time(self, generation_time: float):
        """Record token generation time."""
        self.metrics["token_generation_times"].append(generation_time)

    def record_token_validation_time(self, validation_time: float):
        """Record token validation time."""
        self.metrics["token_validation_times"].append(validation_time)

    def increment_requests(self, success: bool = True):
        """Increment request counters."""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

    def increment_rate_limit_hits(self):
        """Increment rate limit hit counter."""
        self.metrics["rate_limit_hits"] += 1

    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report."""

        def calculate_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {
                    "min": 0,
                    "max": 0,
                    "avg": 0,
                    "median": 0,
                    "p95": 0,
                    "p99": 0,
                }

            return {
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "p95": statistics.quantiles(values, n=20)[18]
                if len(values) >= 20
                else max(values),
                "p99": (
                    statistics.quantiles(values, n=100)[98]
                    if len(values) >= 100
                    else max(values)
                ),
            }

        return {
            "summary": {
                "total_requests": self.metrics["total_requests"],
                "successful_requests": self.metrics["successful_requests"],
                "failed_requests": self.metrics["failed_requests"],
                "success_rate": (
                    (
                        self.metrics["successful_requests"]
                        / self.metrics["total_requests"]
                        * 100
                    )
                    if self.metrics["total_requests"] > 0
                    else 0
                ),
                "rate_limit_hits": self.metrics["rate_limit_hits"],
            },
            "response_times": calculate_stats(self.metrics["response_times"]),
            "token_generation": calculate_stats(
                self.metrics["token_generation_times"]
            ),
            "token_validation": calculate_stats(
                self.metrics["token_validation_times"]
            ),
            "throughput": calculate_stats(self.metrics["throughput"]),
            "memory_usage": calculate_stats(self.metrics["memory_usage"]),
            "cpu_usage": calculate_stats(self.metrics["cpu_usage"]),
        }


class SystemMonitor:
    """Monitor system resources during testing."""

    def __init__(self):
        self.monitoring = False
        self.metrics = PerformanceMetrics()

    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring."""
        self.monitoring = True

        def monitor():
            while self.monitoring:
                # Get memory usage
                memory_info = psutil.virtual_memory()
                self.metrics.record_memory_usage(
                    memory_info.used / 1024 / 1024
                )  # MB

                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.metrics.record_cpu_usage(cpu_percent)

                replace_sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=2.0)


@contextmanager
def performance_timer():
    """Context manager for timing operations."""
    start_time = time.time()
    yield
    end_time = time.time()
    return end_time - start_time


class TestAuthenticationPerformance:
    """Authentication performance and load testing."""

    def setup_method(self):
        """Setup for each test method."""
        self.client = TestClient(app)
        self.auth_manager = AuthenticationManager()
        self.metrics = PerformanceMetrics()
        self.monitor = SystemMonitor()

        # Clear any existing data
        self.auth_manager.users.clear()
        self.auth_manager.refresh_tokens.clear()
        self.auth_manager.blacklist.clear()
        rate_limiter.requests.clear()

        # Pre-create test users for performance testing
        self.test_users = self._create_test_users(100)

    def _create_test_users(self, count: int) -> List[Dict]:
        """Create test users for performance testing."""
        users = []
        for i in range(count):
            user_data = {
                "username": f"perfuser{i}",
                "email": f"perf{i}@example.com",
                "password": f"PerfPass{i}123!",
                "role": random.choice(
                    ["admin", "researcher", "agent_manager", "observer"]
                ),
            }
            users.append(user_data)

            # Register with auth manager
            user = User(
                user_id=f"perf-user-{i}",
                username=user_data["username"],
                email=user_data["email"],
                role=UserRole(user_data["role"]),
                created_at=datetime.now(timezone.utc),
            )

            self.auth_manager.users[user_data["username"]] = {
                "user": user,
                "password_hash": self.auth_manager.hash_password(
                    user_data["password"]
                ),
            }

        return users

    def test_login_performance_baseline(self):
        """Test baseline login performance."""
        user = self.test_users[0]
        response_times = []

        # Warm up
        for _ in range(10):
            response = self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": user["username"],
                    "password": user["password"],
                },
            )
            assert response.status_code == 200

        # Measure performance
        for _ in range(100):
            start_time = time.time()
            response = self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": user["username"],
                    "password": user["password"],
                },
            )
            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)

            self.metrics.record_response_time(response_time)
            self.metrics.increment_requests(response.status_code == 200)

        # Analyze results
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]

        print(f"Login Performance Baseline:")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  95th percentile: {p95_response_time:.3f}s")
        print(f"  Requests per second: {100 / sum(response_times):.1f}")

        # Performance assertions
        assert (
            avg_response_time < 0.1
        ), f"Average login time too slow: {avg_response_time:.3f}s"
        assert (
            p95_response_time < 0.2
        ), f"95th percentile too slow: {p95_response_time:.3f}s"

    def test_token_generation_performance(self):
        """Test token generation performance."""
        user = self.auth_manager.users[self.test_users[0]["username"]]["user"]
        generation_times = []

        # Measure token generation performance
        for _ in range(1000):
            start_time = time.time()
            token = self.auth_manager.create_access_token(user)
            end_time = time.time()

            generation_time = end_time - start_time
            generation_times.append(generation_time)
            self.metrics.record_token_generation_time(generation_time)

            assert token is not None

        # Analyze results
        avg_generation_time = statistics.mean(generation_times)
        p95_generation_time = statistics.quantiles(generation_times, n=20)[18]

        print(f"Token Generation Performance:")
        print(f"  Average generation time: {avg_generation_time:.6f}s")
        print(f"  95th percentile: {p95_generation_time:.6f}s")
        print(f"  Tokens per second: {1000 / sum(generation_times):.1f}")

        # Performance assertions
        assert (
            avg_generation_time < 0.01
        ), f"Token generation too slow: {avg_generation_time:.6f}s"
        assert (
            p95_generation_time < 0.02
        ), f"95th percentile too slow: {p95_generation_time:.6f}s"

    def test_token_validation_performance(self):
        """Test token validation performance."""
        user = self.auth_manager.users[self.test_users[0]["username"]]["user"]
        token = self.auth_manager.create_access_token(user)

        validation_times = []

        # Measure token validation performance
        for _ in range(1000):
            start_time = time.time()
            token_data = self.auth_manager.verify_token(token)
            end_time = time.time()

            validation_time = end_time - start_time
            validation_times.append(validation_time)
            self.metrics.record_token_validation_time(validation_time)

            assert token_data is not None
            assert token_data.user_id == user.user_id

        # Analyze results
        avg_validation_time = statistics.mean(validation_times)
        p95_validation_time = statistics.quantiles(validation_times, n=20)[18]

        print(f"Token Validation Performance:")
        print(f"  Average validation time: {avg_validation_time:.6f}s")
        print(f"  95th percentile: {p95_validation_time:.6f}s")
        print(f"  Validations per second: {1000 / sum(validation_times):.1f}")

        # Performance assertions
        assert (
            avg_validation_time < 0.005
        ), f"Token validation too slow: {avg_validation_time:.6f}s"
        assert (
            p95_validation_time < 0.01
        ), f"95th percentile too slow: {p95_validation_time:.6f}s"

    def test_concurrent_login_performance(self):
        """Test concurrent login performance."""
        self.monitor.start_monitoring()

        def login_worker(user_data):
            """Worker function for concurrent login."""
            try:
                start_time = time.time()
                response = self.client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": user_data["username"],
                        "password": user_data["password"],
                    },
                )
                end_time = time.time()

                response_time = end_time - start_time
                success = response.status_code == 200

                return {
                    "response_time": response_time,
                    "success": success,
                    "status_code": response.status_code,
                }
            except Exception as e:
                return {"response_time": 0, "success": False, "error": str(e)}

        # Test with increasing concurrent users
        concurrent_levels = [10, 25, 50, 100]

        for concurrent_users in concurrent_levels:
            print(f"\\nTesting {concurrent_users} concurrent logins...")

            # Select users for this test
            selected_users = random.sample(
                self.test_users, min(concurrent_users, len(self.test_users))
            )

            start_time = time.time()

            # Execute concurrent logins
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [
                    executor.submit(login_worker, user)
                    for user in selected_users
                ]
                results = [future.result() for future in as_completed(futures)]

            end_time = time.time()
            total_time = end_time - start_time

            # Analyze results
            successful_logins = sum(1 for r in results if r["success"])
            failed_logins = len(results) - successful_logins
            avg_response_time = statistics.mean(
                [r["response_time"] for r in results if r["response_time"] > 0]
            )
            throughput = successful_logins / total_time

            print(f"  Successful logins: {successful_logins}/{len(results)}")
            print(f"  Failed logins: {failed_logins}")
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} logins/second")

            # Record metrics
            self.metrics.record_throughput(throughput)
            for result in results:
                if result["response_time"] > 0:
                    self.metrics.record_response_time(result["response_time"])
                self.metrics.increment_requests(result["success"])

            # Performance assertions
            success_rate = successful_logins / len(results)
            assert (
                success_rate >= 0.95
            ), f"Success rate too low: {success_rate:.2f}"
            assert (
                avg_response_time < 1.0
            ), f"Average response time too slow: {avg_response_time:.3f}s"

        self.monitor.stop_monitoring()

    def test_high_volume_token_refresh(self):
        """Test high volume token refresh performance."""
        # Create tokens for all test users
        tokens = []
        for user_data in self.test_users[:50]:  # Use first 50 users
            response = self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                },
            )
            assert response.status_code == 200
            tokens.append(response.json()["refresh_token"])

        def refresh_worker(refresh_token):
            """Worker function for token refresh."""
            try:
                start_time = time.time()
                response = self.client.post(
                    "/api/v1/auth/refresh",
                    json={"refresh_token": refresh_token},
                )
                end_time = time.time()

                return {
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                }
            except Exception as e:
                return {"response_time": 0, "success": False, "error": str(e)}

        # Test concurrent token refresh
        self.monitor.start_monitoring()

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = [
                executor.submit(refresh_worker, token) for token in tokens
            ]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.time()
        total_time = end_time - start_time

        # Analyze results
        successful_refreshes = sum(1 for r in results if r["success"])
        avg_response_time = statistics.mean(
            [r["response_time"] for r in results if r["response_time"] > 0]
        )
        throughput = successful_refreshes / total_time

        print(f"Token Refresh Performance:")
        print(f"  Successful refreshes: {successful_refreshes}/{len(results)}")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} refreshes/second")

        # Performance assertions
        success_rate = successful_refreshes / len(results)
        assert (
            success_rate >= 0.8
        ), f"Token refresh success rate too low: {success_rate:.2f}"
        assert (
            avg_response_time < 0.5
        ), f"Token refresh too slow: {avg_response_time:.3f}s"

        self.monitor.stop_monitoring()

    def test_rate_limiting_performance(self):
        """Test rate limiting performance under load."""
        user = self.test_users[0]

        # Clear rate limiter
        rate_limiter.requests.clear()

        def rate_limited_request():
            """Make a rate-limited request."""
            try:
                start_time = time.time()
                response = self.client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": user["username"],
                        "password": "wrong_password",  # Intentionally wrong to avoid successful logins
                    },
                )
                end_time = time.time()

                return {
                    "response_time": end_time - start_time,
                    "status_code": response.status_code,
                    "rate_limited": response.status_code == 429,
                }
            except Exception as e:
                return {
                    "response_time": 0,
                    "status_code": 500,
                    "rate_limited": False,
                    "error": str(e),
                }

        # Make rapid requests to trigger rate limiting
        results = []
        for i in range(20):  # Rate limit is 10 per 5 minutes
            result = rate_limited_request()
            results.append(result)
            self.metrics.increment_requests(
                result["status_code"] not in [429, 500]
            )

            if result["rate_limited"]:
                self.metrics.increment_rate_limit_hits()

        # Analyze results
        rate_limited_requests = sum(1 for r in results if r["rate_limited"])
        successful_requests = sum(
            1 for r in results if r["status_code"] == 401
        )  # 401 is expected for wrong password

        print(f"Rate Limiting Performance:")
        print(f"  Total requests: {len(results)}")
        print(f"  Rate limited requests: {rate_limited_requests}")
        print(f"  Successful requests: {successful_requests}")

        # Should have rate limiting active
        assert rate_limited_requests > 0, "Rate limiting should be active"
        assert (
            rate_limited_requests <= 10
        ), "Rate limiting should not block all requests immediately"

    def test_memory_usage_under_load(self):
        """Test memory usage under authentication load."""
        self.monitor.start_monitoring()

        # Record initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        def memory_intensive_operation():
            """Perform memory-intensive authentication operations."""
            user = random.choice(self.test_users)

            # Login
            response = self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": user["username"],
                    "password": user["password"],
                },
            )

            if response.status_code == 200:
                token = response.json()["access_token"]

                # Validate token multiple times
                for _ in range(10):
                    headers = {"Authorization": f"Bearer {token}"}
                    self.client.get("/api/v1/auth/me", headers=headers)

                # Refresh token
                refresh_token = response.json()["refresh_token"]
                self.client.post(
                    "/api/v1/auth/refresh",
                    json={"refresh_token": refresh_token},
                )

        # Perform many operations
        for i in range(500):
            memory_intensive_operation()

            # Check memory usage every 50 operations
            if i % 50 == 0:
                current_memory = (
                    psutil.Process().memory_info().rss / 1024 / 1024
                )  # MB
                self.metrics.record_memory_usage(current_memory)

                # Force garbage collection
                gc.collect()

        # Record final memory usage
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory Usage Under Load:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")

        self.monitor.stop_monitoring()

        # Memory usage should not increase dramatically
        assert (
            memory_increase < 100
        ), f"Memory usage increased too much: {memory_increase:.1f} MB"

    def test_database_connection_pool_performance(self):
        """Test database connection pool performance simulation."""
        # Simulate database connection pool behavior
        connection_times = []

        def simulate_db_operation():
            """Simulate database operation."""
            start_time = time.time()

            # Simulate connection acquisition delay
            replace_sleep(0.001)  # 1ms delay

            # Simulate query execution
            user = random.choice(self.test_users)
            stored_user = self.auth_manager.users.get(user["username"])

            if stored_user:
                # Simulate password verification
                self.auth_manager.verify_password(
                    user["password"], stored_user["password_hash"]
                )

            end_time = time.time()
            return end_time - start_time

        # Test with concurrent database operations
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(simulate_db_operation) for _ in range(200)
            ]
            connection_times = [
                future.result() for future in as_completed(futures)
            ]

        # Analyze results
        avg_connection_time = statistics.mean(connection_times)
        p95_connection_time = statistics.quantiles(connection_times, n=20)[18]

        print(f"Database Connection Pool Performance:")
        print(f"  Average operation time: {avg_connection_time:.6f}s")
        print(f"  95th percentile: {p95_connection_time:.6f}s")
        print(f"  Operations per second: {200 / sum(connection_times):.1f}")

        # Performance assertions
        assert (
            avg_connection_time < 0.01
        ), f"Database operations too slow: {avg_connection_time:.6f}s"

    def test_stress_test_authentication_system(self):
        """Comprehensive stress test of the authentication system."""
        self.monitor.start_monitoring()

        # Test configuration
        duration_seconds = 30
        max_concurrent_users = 100

        print(
            f"Starting {duration_seconds}s stress test with up to {max_concurrent_users} concurrent users..."
        )

        # Track results
        results = []
        stop_time = time.time() + duration_seconds

        def stress_worker():
            """Worker function for stress testing."""
            worker_results = []

            while time.time() < stop_time:
                user = random.choice(self.test_users)
                operation = random.choice(["login", "refresh", "validate"])

                try:
                    start_time = time.time()

                    if operation == "login":
                        response = self.client.post(
                            "/api/v1/auth/login",
                            json={
                                "username": user["username"],
                                "password": user["password"],
                            },
                        )
                        success = response.status_code == 200

                    elif operation == "refresh":
                        # First login to get refresh token
                        login_response = self.client.post(
                            "/api/v1/auth/login",
                            json={
                                "username": user["username"],
                                "password": user["password"],
                            },
                        )
                        if login_response.status_code == 200:
                            refresh_token = login_response.json()[
                                "refresh_token"
                            ]
                            response = self.client.post(
                                "/api/v1/auth/refresh",
                                json={"refresh_token": refresh_token},
                            )
                            success = response.status_code == 200
                        else:
                            success = False

                    elif operation == "validate":
                        # First login to get access token
                        login_response = self.client.post(
                            "/api/v1/auth/login",
                            json={
                                "username": user["username"],
                                "password": user["password"],
                            },
                        )
                        if login_response.status_code == 200:
                            access_token = login_response.json()[
                                "access_token"
                            ]
                            headers = {
                                "Authorization": f"Bearer {access_token}"
                            }
                            response = self.client.get(
                                "/api/v1/auth/me", headers=headers
                            )
                            success = response.status_code == 200
                        else:
                            success = False

                    end_time = time.time()
                    response_time = end_time - start_time

                    worker_results.append(
                        {
                            "operation": operation,
                            "response_time": response_time,
                            "success": success,
                        }
                    )

                except Exception as e:
                    worker_results.append(
                        {
                            "operation": operation,
                            "response_time": 0,
                            "success": False,
                            "error": str(e),
                        }
                    )

                # Small delay between operations
                replace_sleep(0.01)

            return worker_results

        # Start stress test
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_concurrent_users) as executor:
            futures = [
                executor.submit(stress_worker)
                for _ in range(max_concurrent_users)
            ]

            for future in as_completed(futures):
                results.extend(future.result())

        end_time = time.time()
        actual_duration = end_time - start_time

        self.monitor.stop_monitoring()

        # Analyze stress test results
        total_operations = len(results)
        successful_operations = sum(1 for r in results if r["success"])
        failed_operations = total_operations - successful_operations

        if total_operations > 0:
            success_rate = successful_operations / total_operations
            avg_response_time = statistics.mean(
                [r["response_time"] for r in results if r["response_time"] > 0]
            )
            throughput = successful_operations / actual_duration

            print(f"Stress Test Results:")
            print(f"  Duration: {actual_duration:.1f}s")
            print(f"  Total operations: {total_operations}")
            print(f"  Successful operations: {successful_operations}")
            print(f"  Failed operations: {failed_operations}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} operations/second")

            # Record final metrics
            self.metrics.record_throughput(throughput)
            for result in results:
                if result["response_time"] > 0:
                    self.metrics.record_response_time(result["response_time"])
                self.metrics.increment_requests(result["success"])

            # Performance assertions
            assert (
                success_rate >= 0.90
            ), f"Stress test success rate too low: {success_rate:.2%}"
            assert (
                throughput >= 50
            ), f"Stress test throughput too low: {throughput:.1f} ops/sec"

        else:
            pytest.fail("No operations completed during stress test")

    def test_generate_performance_report(self):
        """Generate comprehensive performance report."""
        # Run key performance tests
        print("Running authentication performance test suite...")

        self.test_login_performance_baseline()
        self.test_token_generation_performance()
        self.test_token_validation_performance()
        self.test_concurrent_login_performance()
        self.test_high_volume_token_refresh()
        self.test_memory_usage_under_load()
        self.test_stress_test_authentication_system()

        # Generate report
        report = self.metrics.generate_report()

        print("\\n" + "=" * 60)
        print("COMPREHENSIVE AUTHENTICATION PERFORMANCE REPORT")
        print("=" * 60)

        print(f"\\nSUMMARY:")
        print(f"  Total Requests: {report['summary']['total_requests']}")
        print(
            f"  Successful Requests: {report['summary']['successful_requests']}"
        )
        print(f"  Failed Requests: {report['summary']['failed_requests']}")
        print(f"  Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"  Rate Limit Hits: {report['summary']['rate_limit_hits']}")

        print(f"\\nRESPONSE TIMES:")
        print(f"  Average: {report['response_times']['avg']:.3f}s")
        print(f"  95th Percentile: {report['response_times']['p95']:.3f}s")
        print(f"  99th Percentile: {report['response_times']['p99']:.3f}s")

        print(f"\\nTOKEN GENERATION:")
        print(f"  Average: {report['token_generation']['avg']:.6f}s")
        print(f"  95th Percentile: {report['token_generation']['p95']:.6f}s")

        print(f"\\nTOKEN VALIDATION:")
        print(f"  Average: {report['token_validation']['avg']:.6f}s")
        print(f"  95th Percentile: {report['token_validation']['p95']:.6f}s")

        print(f"\\nTHROUGHPUT:")
        print(f"  Average: {report['throughput']['avg']:.1f} ops/sec")
        print(f"  Peak: {report['throughput']['max']:.1f} ops/sec")

        if report["memory_usage"]["avg"] > 0:
            print(f"\\nMEMORY USAGE:")
            print(f"  Average: {report['memory_usage']['avg']:.1f} MB")
            print(f"  Peak: {report['memory_usage']['max']:.1f} MB")

        # Performance assertions
        assert (
            report["summary"]["success_rate"] >= 90
        ), f"Overall success rate too low: {report['summary']['success_rate']:.1f}%"
        assert (
            report["response_times"]["avg"] < 1.0
        ), f"Average response time too slow: {report['response_times']['avg']:.3f}s"
        assert (
            report["token_generation"]["avg"] < 0.01
        ), f"Token generation too slow: {report['token_generation']['avg']:.6f}s"
        assert (
            report["token_validation"]["avg"] < 0.01
        ), f"Token validation too slow: {report['token_validation']['avg']:.6f}s"

        print("\\nPERFORMANCE TESTING COMPLETED SUCCESSFULLY!")
        return report

    def teardown_method(self):
        """Cleanup after each test."""
        self.monitor.stop_monitoring()
        self.auth_manager.users.clear()
        self.auth_manager.refresh_tokens.clear()
        self.auth_manager.blacklist.clear()
        rate_limiter.requests.clear()

        # Force garbage collection
        gc.collect()


if __name__ == "__main__":
    # Run performance tests
    test_suite = TestAuthenticationPerformance()
    test_suite.setup_method()

    try:
        report = test_suite.test_generate_performance_report()
        print(f"\\nPERFORMANCE TESTING COMPLETED")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Average Response Time: {report['response_times']['avg']:.3f}s")
        print(f"Peak Throughput: {report['throughput']['max']:.1f} ops/sec")
    except Exception as e:
        print(f"\\nPERFORMANCE TESTING FAILED: {e}")
        raise
    finally:
        test_suite.teardown_method()

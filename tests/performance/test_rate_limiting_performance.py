"""Performance benchmarks for rate limiting system.

This module tests:
- Throughput under various loads
- Latency measurements
- Memory efficiency
- Scalability testing
- Redis performance optimization
"""

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import psutil
import pytest
import redis.asyncio as aioredis
from fastapi import Request

from api.middleware.ddos_protection import RateLimitConfig, RateLimiter


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""

    total_requests: int
    successful_requests: int
    blocked_requests: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    redis_operations: int
    redis_memory_mb: float
    error_count: int
    test_duration_seconds: float


class RateLimitingPerformanceTester:
    """Performance testing framework for rate limiting."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.results: List[PerformanceMetrics] = []

    async def setup(self):
        """Setup test environment."""
        self.redis_client = aioredis.Redis.from_url(
            self.redis_url,
            decode_responses=True,
            max_connections=50,
        )

        # Clear all test data
        await self.redis_client.flushdb()

        self.rate_limiter = RateLimiter(self.redis_client)

    async def teardown(self):
        """Cleanup test environment."""
        await self.redis_client.flushdb()
        await self.redis_client.close()

    async def measure_throughput(
        self,
        num_requests: int,
        num_concurrent: int,
        config: RateLimitConfig,
        unique_ips: int = 100,
    ) -> PerformanceMetrics:
        """Measure rate limiting throughput."""

        # Override config
        async def mock_get_endpoint_config(path):
            return config

        self.rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Prepare IP pool
        ip_pool = [f"10.0.{i // 256}.{i % 256}" for i in range(unique_ips)]

        # Metrics collection
        latencies = []
        successful = 0
        blocked = 0
        errors = 0

        # CPU and memory monitoring
        process = psutil.Process()
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Redis monitoring
        initial_redis_info = await self.redis_client.info()
        initial_redis_ops = int(
            initial_redis_info.get("total_commands_processed", 0)
        )

        # Barrier for synchronized start
        barrier = asyncio.Barrier(num_concurrent)

        async def make_request(request_id: int):
            """Make a single request and measure performance."""
            nonlocal successful, blocked, errors

            await barrier.wait()  # Synchronize start

            ip = random.choice(ip_pool)
            request = self._create_request(ip)

            start_time = time.perf_counter()
            try:
                response = await self.rate_limiter.check_rate_limit(request)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                if response is None:
                    successful += 1
                else:
                    blocked += 1

            except Exception as e:
                errors += 1
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

        # Run test
        test_start = time.time()

        # Create batches of requests
        batches = num_requests // num_concurrent
        for batch in range(batches):
            tasks = [
                make_request(batch * num_concurrent + i)
                for i in range(num_concurrent)
            ]
            await asyncio.gather(*tasks)

        # Handle remaining requests
        remaining = num_requests % num_concurrent
        if remaining:
            tasks = [
                make_request(batches * num_concurrent + i)
                for i in range(remaining)
            ]
            await asyncio.gather(*tasks)

        test_end = time.time()
        test_duration = test_end - test_start

        # Final metrics
        final_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss / 1024 / 1024

        final_redis_info = await self.redis_client.info()
        final_redis_ops = int(
            final_redis_info.get("total_commands_processed", 0)
        )
        redis_memory = (
            int(final_redis_info.get("used_memory", 0)) / 1024 / 1024
        )

        # Calculate statistics
        if latencies:
            latencies.sort()
            metrics = PerformanceMetrics(
                total_requests=num_requests,
                successful_requests=successful,
                blocked_requests=blocked,
                average_latency_ms=statistics.mean(latencies),
                p50_latency_ms=latencies[len(latencies) // 2],
                p95_latency_ms=latencies[int(len(latencies) * 0.95)],
                p99_latency_ms=latencies[int(len(latencies) * 0.99)],
                max_latency_ms=max(latencies),
                throughput_rps=num_requests / test_duration,
                memory_usage_mb=final_memory - initial_memory,
                cpu_usage_percent=(final_cpu + initial_cpu) / 2,
                redis_operations=final_redis_ops - initial_redis_ops,
                redis_memory_mb=redis_memory,
                error_count=errors,
                test_duration_seconds=test_duration,
            )
        else:
            # All requests failed
            metrics = PerformanceMetrics(
                total_requests=num_requests,
                successful_requests=0,
                blocked_requests=0,
                average_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                max_latency_ms=0,
                throughput_rps=0,
                memory_usage_mb=final_memory - initial_memory,
                cpu_usage_percent=(final_cpu + initial_cpu) / 2,
                redis_operations=final_redis_ops - initial_redis_ops,
                redis_memory_mb=redis_memory,
                error_count=errors,
                test_duration_seconds=test_duration,
            )

        self.results.append(metrics)
        return metrics

    def _create_request(self, ip: str) -> Request:
        """Create mock request for testing."""
        from unittest.mock import MagicMock

        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = ip
        request.url.path = "/api/v1/test"
        request.headers = {}
        request.state = MagicMock()

        return request

    def generate_report(
        self, output_file: str = "rate_limiting_performance_report.json"
    ):
        """Generate performance report."""
        report = {"timestamp": datetime.now().isoformat(), "tests": []}

        for metrics in self.results:
            report["tests"].append(
                {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "blocked_requests": metrics.blocked_requests,
                    "throughput_rps": round(metrics.throughput_rps, 2),
                    "latencies": {
                        "average_ms": round(metrics.average_latency_ms, 3),
                        "p50_ms": round(metrics.p50_latency_ms, 3),
                        "p95_ms": round(metrics.p95_latency_ms, 3),
                        "p99_ms": round(metrics.p99_latency_ms, 3),
                        "max_ms": round(metrics.max_latency_ms, 3),
                    },
                    "resource_usage": {
                        "memory_mb": round(metrics.memory_usage_mb, 2),
                        "cpu_percent": round(metrics.cpu_usage_percent, 2),
                        "redis_memory_mb": round(metrics.redis_memory_mb, 2),
                        "redis_operations": metrics.redis_operations,
                    },
                    "errors": metrics.error_count,
                    "duration_seconds": round(
                        metrics.test_duration_seconds, 2
                    ),
                }
            )

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        return report


class TestRateLimitingPerformance:
    """Performance test suite for rate limiting."""

    @pytest.fixture
    async def tester(self):
        """Create performance tester instance."""
        tester = RateLimitingPerformanceTester()
        await tester.setup()
        yield tester
        await tester.teardown()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_baseline_performance(self, tester):
        """Test baseline performance with moderate load."""
        config = RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=50000,
            burst_limit=50,
        )

        metrics = await tester.measure_throughput(
            num_requests=10000,
            num_concurrent=100,
            config=config,
            unique_ips=100,
        )

        # Performance assertions
        assert metrics.throughput_rps > 1000  # At least 1000 RPS
        assert metrics.average_latency_ms < 10  # Average under 10ms
        assert metrics.p95_latency_ms < 50  # 95th percentile under 50ms
        assert metrics.p99_latency_ms < 100  # 99th percentile under 100ms
        assert metrics.error_count == 0  # No errors

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_high_throughput(self, tester):
        """Test performance under high throughput."""
        config = RateLimitConfig(
            requests_per_minute=10000,
            requests_per_hour=500000,
            burst_limit=500,
        )

        metrics = await tester.measure_throughput(
            num_requests=50000,
            num_concurrent=500,
            config=config,
            unique_ips=1000,
        )

        # High throughput assertions
        assert metrics.throughput_rps > 5000  # At least 5000 RPS
        assert metrics.average_latency_ms < 20  # Average under 20ms
        assert metrics.memory_usage_mb < 100  # Memory usage under 100MB

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_burst_handling(self, tester):
        """Test performance during burst traffic."""
        config = RateLimitConfig(
            requests_per_minute=1000,
            burst_limit=100,
            window_size=1,
        )

        # Send burst of requests
        metrics = await tester.measure_throughput(
            num_requests=1000,
            num_concurrent=1000,  # All at once
            config=config,
            unique_ips=10,
        )

        # Burst handling assertions
        assert metrics.successful_requests <= 110  # Burst limit + small buffer
        assert metrics.blocked_requests > 800  # Most should be blocked
        assert (
            metrics.max_latency_ms < 500
        )  # Max latency under 500ms even during burst

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency(self, tester):
        """Test memory efficiency with many unique IPs."""
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=1000,
        )

        # Test with many unique IPs
        metrics = await tester.measure_throughput(
            num_requests=10000,
            num_concurrent=100,
            config=config,
            unique_ips=10000,  # Many unique IPs
        )

        # Memory efficiency assertions
        assert metrics.memory_usage_mb < 50  # Application memory under 50MB
        assert metrics.redis_memory_mb < 100  # Redis memory under 100MB

        # Calculate memory per IP
        memory_per_ip = metrics.redis_memory_mb / 10000 * 1024  # KB per IP
        assert memory_per_ip < 10  # Less than 10KB per IP

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_latency_consistency(self, tester):
        """Test latency consistency over time."""
        config = RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=50000,
        )

        # Run multiple rounds and check consistency
        latencies = []

        for _ in range(5):
            metrics = await tester.measure_throughput(
                num_requests=1000,
                num_concurrent=50,
                config=config,
                unique_ips=100,
            )
            latencies.append(metrics.average_latency_ms)
            await asyncio.sleep(1)  # Brief pause between rounds

        # Check consistency
        latency_std = statistics.stdev(latencies)
        assert latency_std < 5  # Standard deviation under 5ms
        assert max(latencies) - min(latencies) < 10  # Range under 10ms

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_scalability(self, tester):
        """Test scalability with increasing load."""
        config = RateLimitConfig(
            requests_per_minute=10000,
            requests_per_hour=500000,
        )

        # Test with increasing concurrent connections
        concurrent_levels = [10, 50, 100, 200, 500, 1000]
        throughputs = []

        for concurrent in concurrent_levels:
            metrics = await tester.measure_throughput(
                num_requests=5000,
                num_concurrent=concurrent,
                config=config,
                unique_ips=1000,
            )
            throughputs.append(metrics.throughput_rps)

        # Scalability assertions
        # Throughput should increase with concurrency up to a point
        assert throughputs[1] > throughputs[0]  # 50 > 10
        assert throughputs[2] > throughputs[1]  # 100 > 50

        # But should plateau, not degrade significantly
        min_throughput = min(throughputs[3:])  # 200+ concurrent
        max_throughput = max(throughputs)
        degradation = (max_throughput - min_throughput) / max_throughput
        assert degradation < 0.2  # Less than 20% degradation

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_redis_pipeline_efficiency(self, tester):
        """Test Redis pipeline optimization."""
        config = RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=50000,
        )

        # Measure Redis operations
        metrics = await tester.measure_throughput(
            num_requests=10000,
            num_concurrent=100,
            config=config,
            unique_ips=100,
        )

        # Calculate Redis operation efficiency
        # Should use pipeline, so operations should be less than 2x requests
        redis_ops_per_request = (
            metrics.redis_operations / metrics.total_requests
        )
        assert redis_ops_per_request < 3  # Less than 3 Redis ops per request

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_sustained_load(self, tester):
        """Test performance under sustained load."""
        config = RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=50000,
        )

        # Run for extended period
        start_time = time.time()
        metrics_over_time = []

        while time.time() - start_time < 60:  # Run for 1 minute
            metrics = await tester.measure_throughput(
                num_requests=1000,
                num_concurrent=50,
                config=config,
                unique_ips=100,
            )
            metrics_over_time.append(metrics)

        # Analyze sustained performance
        avg_throughputs = [m.throughput_rps for m in metrics_over_time]
        avg_latencies = [m.average_latency_ms for m in metrics_over_time]

        # Performance should remain stable
        throughput_cv = statistics.stdev(avg_throughputs) / statistics.mean(
            avg_throughputs
        )
        latency_cv = statistics.stdev(avg_latencies) / statistics.mean(
            avg_latencies
        )

        assert throughput_cv < 0.1  # Coefficient of variation < 10%
        assert latency_cv < 0.2  # Latency CV < 20%

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_with_blocks(self, tester):
        """Test performance when many requests are blocked."""
        config = RateLimitConfig(
            requests_per_minute=100,  # Low limit
            requests_per_hour=1000,
            burst_limit=10,
        )

        # Most requests will be blocked
        metrics = await tester.measure_throughput(
            num_requests=10000,
            num_concurrent=100,
            config=config,
            unique_ips=10,  # Few IPs to ensure blocking
        )

        # Even with high block rate, performance should be good
        assert metrics.average_latency_ms < 5  # Blocking should be fast
        assert (
            metrics.throughput_rps > 5000
        )  # Can handle many requests even if blocking

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_limit_types(self, tester):
        """Test performance with multiple concurrent limit types."""
        # Test rate limiter that checks multiple limits
        config = RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=10000,
            burst_limit=50,
            window_size=1,
        )

        # Add custom limit checks
        async def enhanced_check_rate_limit(request):
            # Check regular limits
            response = await tester.rate_limiter.check_rate_limit(request)

            # Additional checks (user limits, API key limits, etc.)
            # These would normally be implemented in the rate limiter

            return response

        # Replace method temporarily
        original_method = tester.rate_limiter.check_rate_limit
        tester.rate_limiter.check_rate_limit = enhanced_check_rate_limit

        try:
            metrics = await tester.measure_throughput(
                num_requests=5000,
                num_concurrent=100,
                config=config,
                unique_ips=100,
            )

            # Multiple checks shouldn't significantly impact performance
            assert metrics.average_latency_ms < 15  # Still under 15ms
            assert metrics.throughput_rps > 1000  # Maintain good throughput

        finally:
            tester.rate_limiter.check_rate_limit = original_method


class TestRateLimitingOptimizations:
    """Test specific optimizations for rate limiting performance."""

    @pytest.mark.asyncio
    async def test_connection_pooling(self):
        """Test Redis connection pooling effectiveness."""
        # Create rate limiter with connection pool
        pool = aioredis.ConnectionPool.from_url(
            "redis://localhost:6379",
            max_connections=50,
            min_connections=10,
        )
        redis_client = aioredis.Redis(connection_pool=pool)
        rate_limiter = RateLimiter(redis_client)

        config = RateLimitConfig(requests_per_minute=1000)

        # Override config
        async def mock_get_endpoint_config(path):
            return config

        rate_limiter._get_endpoint_config = mock_get_endpoint_config

        # Measure connection pool performance
        start_time = time.time()
        tasks = []

        for i in range(1000):
            request = MagicMock(spec=Request)
            request.client = MagicMock()
            request.client.host = f"10.0.0.{i % 256}"
            request.url.path = "/api/v1/test"
            request.headers = {}
            request.state = MagicMock()

            tasks.append(rate_limiter.check_rate_limit(request))

        await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Cleanup
        await redis_client.close()
        await pool.disconnect()

        # With connection pooling, should handle 1000 requests quickly
        assert duration < 2.0  # Under 2 seconds

    @pytest.mark.asyncio
    async def test_lua_script_optimization(self):
        """Test performance with Lua script optimization."""
        # Lua script for atomic rate limit check and increment
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])

        local current = redis.call('GET', key)
        if current == false then
            current = 0
        else
            current = tonumber(current)
        end

        if current >= limit then
            return {1, current}  -- Rate limited
        end

        redis.call('INCR', key)
        redis.call('EXPIRE', key, window)

        return {0, current + 1}  -- Allowed
        """

        redis_client = aioredis.Redis.from_url("redis://localhost:6379")

        # Register script
        script = redis_client.register_script(lua_script)

        # Test performance
        start_time = time.time()

        for i in range(10000):
            key = f"rate_limit:test:{i % 100}"
            result = await script(keys=[key], args=[100, 60, int(time.time())])

        duration = time.time() - start_time

        await redis_client.close()

        # Lua script should be very fast
        assert duration < 3.0  # 10000 operations in under 3 seconds

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing optimization."""
        redis_client = aioredis.Redis.from_url("redis://localhost:6379")

        # Batch rate limit checks
        async def batch_check_rate_limits(requests: List[Tuple[str, str]]):
            """Check rate limits for multiple requests in batch."""
            pipe = redis_client.pipeline()

            for ip, path in requests:
                key = f"rate_limit:ip:{ip}"
                pipe.incr(key)
                pipe.expire(key, 60)

            results = await pipe.execute()

            # Process results
            rate_limited = []
            for i in range(0, len(results), 2):
                count = results[i]
                if count > 100:  # Simple limit check
                    rate_limited.append(i // 2)

            return rate_limited

        # Test batch performance
        batch_size = 100
        num_batches = 100

        start_time = time.time()

        for _ in range(num_batches):
            requests = [
                (f"10.0.0.{i}", "/api/v1/test") for i in range(batch_size)
            ]
            await batch_check_rate_limits(requests)

        duration = time.time() - start_time
        total_requests = batch_size * num_batches

        await redis_client.close()

        # Batch processing should be efficient
        throughput = total_requests / duration
        assert throughput > 5000  # Over 5000 requests per second


if __name__ == "__main__":
    import asyncio

    async def run_performance_tests():
        """Run performance tests and generate report."""
        tester = RateLimitingPerformanceTester()
        await tester.setup()

        print("Running performance tests...")

        # Test configurations
        test_configs = [
            (
                "Baseline",
                10000,
                100,
                RateLimitConfig(requests_per_minute=1000),
            ),
            (
                "High Load",
                50000,
                500,
                RateLimitConfig(requests_per_minute=10000),
            ),
            (
                "Burst",
                1000,
                1000,
                RateLimitConfig(requests_per_minute=100, burst_limit=50),
            ),
            ("Many IPs", 10000, 100, RateLimitConfig(requests_per_minute=100)),
        ]

        for name, requests, concurrent, config in test_configs:
            print(f"\nRunning {name} test...")
            metrics = await tester.measure_throughput(
                num_requests=requests,
                num_concurrent=concurrent,
                config=config,
                unique_ips=1000 if name != "Many IPs" else 10000,
            )

            print(f"  Throughput: {metrics.throughput_rps:.2f} RPS")
            print(f"  Avg Latency: {metrics.average_latency_ms:.2f} ms")
            print(f"  P99 Latency: {metrics.p99_latency_ms:.2f} ms")
            print(f"  Memory Usage: {metrics.memory_usage_mb:.2f} MB")

        # Generate report
        report = tester.generate_report()
        print(f"\nReport saved to: rate_limiting_performance_report.json")

        await tester.teardown()

    # Run if executed directly
    if __name__ == "__main__":
        asyncio.run(run_performance_tests())

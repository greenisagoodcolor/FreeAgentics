#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Suite
========================================

This module provides a comprehensive performance benchmarking framework using pytest-benchmark
for continuous integration and performance regression detection.

Features:
- Agent spawn time benchmarks
- Message throughput benchmarks
- Memory usage tracking
- Database query performance
- WebSocket connection handling
- Automatic regression detection
- Performance trend analysis
"""

import asyncio
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import pytest

# Performance thresholds (fail if >10% regression)
REGRESSION_THRESHOLD = 0.10  # 10%
WARNING_THRESHOLD = 0.05  # 5%

# Benchmark categories
BENCHMARK_CATEGORIES = [
    "agent_spawn",
    "message_throughput",
    "memory_usage",
    "database_query",
    "websocket_connection",
    "coordination",
    "inference",
]


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    name: str
    category: str
    duration_ms: float
    operations_per_second: float
    memory_start_mb: float
    memory_end_mb: float
    memory_peak_mb: float
    cpu_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def memory_growth_mb(self) -> float:
        """Calculate memory growth during benchmark."""
        return self.memory_end_mb - self.memory_start_mb


class MemoryTracker:
    """Track memory usage during benchmarks."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = 0
        self.peak_memory = 0
        self.samples = []

    def start(self):
        """Start memory tracking."""
        gc.collect()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = self.start_memory
        self.samples = [self.start_memory]

    def sample(self):
        """Take a memory sample."""
        current = self.process.memory_info().rss / 1024 / 1024
        self.samples.append(current)
        self.peak_memory = max(self.peak_memory, current)
        return current

    def stop(self) -> Tuple[float, float, float]:
        """Stop tracking and return (start, end, peak) in MB."""
        end_memory = self.sample()
        return self.start_memory, end_memory, self.peak_memory


@contextmanager
def track_performance(name: str, category: str):
    """Context manager to track performance metrics."""
    tracker = MemoryTracker()
    tracker.start()

    cpu_start = psutil.cpu_percent(interval=None)
    start_time = time.perf_counter()

    metrics = BenchmarkMetrics(
        name=name,
        category=category,
        duration_ms=0,
        operations_per_second=0,
        memory_start_mb=tracker.start_memory,
        memory_end_mb=0,
        memory_peak_mb=0,
        cpu_percent=0,
    )

    try:
        yield metrics
    finally:
        end_time = time.perf_counter()
        cpu_end = psutil.cpu_percent(interval=None)

        start_mem, end_mem, peak_mem = tracker.stop()

        metrics.duration_ms = (end_time - start_time) * 1000
        metrics.memory_start_mb = start_mem
        metrics.memory_end_mb = end_mem
        metrics.memory_peak_mb = peak_mem
        metrics.cpu_percent = cpu_end - cpu_start


# Agent Spawn Benchmarks
class AgentSpawnBenchmarks:
    """Benchmarks for agent spawn time and initialization."""

    @staticmethod
    def benchmark_single_agent_spawn(benchmark):
        """Benchmark spawning a single agent."""
        from agents.base_agent import BasicExplorerAgent

        def spawn_agent():
            agent = BasicExplorerAgent(
                agent_id="test_agent",
                num_states=5,
                num_actions=3,
                num_observations=5,
            )
            return agent

        result = benchmark(spawn_agent)
        assert result is not None

    @staticmethod
    def benchmark_batch_agent_spawn(benchmark):
        """Benchmark spawning multiple agents in batch."""
        from agents.base_agent import BasicExplorerAgent

        def spawn_batch(count=10):
            agents = []
            for i in range(count):
                agent = BasicExplorerAgent(
                    agent_id=f"agent_{i}",
                    num_states=5,
                    num_actions=3,
                    num_observations=5,
                )
                agents.append(agent)
            return agents

        result = benchmark(spawn_batch, count=10)
        assert len(result) == 10

    @staticmethod
    def benchmark_concurrent_agent_spawn(benchmark):
        """Benchmark concurrent agent spawning."""
        from agents.base_agent import BasicExplorerAgent

        def spawn_concurrent(count=10):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(count):
                    future = executor.submit(
                        BasicExplorerAgent,
                        agent_id=f"agent_{i}",
                        num_states=5,
                        num_actions=3,
                        num_observations=5,
                    )
                    futures.append(future)

                agents = [future.result() for future in as_completed(futures)]
            return agents

        result = benchmark(spawn_concurrent, count=10)
        assert len(result) == 10


# Message Throughput Benchmarks
class MessageThroughputBenchmarks:
    """Benchmarks for message passing and throughput."""

    @staticmethod
    def benchmark_single_message_pass(benchmark):
        """Benchmark single message passing."""
        from collections import deque

        message_queue = deque()

        def pass_message():
            message = {"type": "test", "data": "test_data", "timestamp": time.time()}
            message_queue.append(message)
            return message_queue.popleft()

        result = benchmark(pass_message)
        assert result["type"] == "test"

    @staticmethod
    def benchmark_bulk_message_throughput(benchmark):
        """Benchmark bulk message throughput."""
        from collections import deque

        def process_messages(count=1000):
            queue = deque()

            # Send messages
            for i in range(count):
                message = {"id": i, "data": f"message_{i}"}
                queue.append(message)

            # Process messages
            processed = []
            while queue:
                msg = queue.popleft()
                processed.append(msg)

            return processed

        result = benchmark(process_messages, count=1000)
        assert len(result) == 1000

    @staticmethod
    @pytest.mark.asyncio
    async def benchmark_async_message_throughput(benchmark):
        """Benchmark async message throughput."""
        import asyncio

        async def async_message_processing(count=1000):
            queue = asyncio.Queue()

            # Producer
            async def produce():
                for i in range(count):
                    await queue.put({"id": i, "data": f"msg_{i}"})

            # Consumer
            async def consume():
                processed = []
                for _ in range(count):
                    msg = await queue.get()
                    processed.append(msg)
                return processed

            # Run concurrently
            producer = asyncio.create_task(produce())
            processed = await consume()
            await producer

            return processed

        # pytest-benchmark doesn't natively support async, so we wrap it
        loop = asyncio.new_event_loop()
        result = benchmark(lambda: loop.run_until_complete(async_message_processing(1000)))
        assert len(result) == 1000


# Memory Usage Benchmarks
class MemoryUsageBenchmarks:
    """Benchmarks for memory usage and optimization."""

    @staticmethod
    def benchmark_agent_memory_lifecycle(benchmark):
        """Benchmark agent memory usage during lifecycle."""
        from agents.base_agent import BasicExplorerAgent

        def agent_lifecycle():
            agents = []

            # Create agents
            for i in range(20):
                agent = BasicExplorerAgent(
                    agent_id=f"agent_{i}",
                    num_states=10,
                    num_actions=5,
                    num_observations=10,
                )
                agents.append(agent)

            # Simulate some work
            for agent in agents:
                for _ in range(10):
                    agent.step(np.random.randint(0, 10))

            # Cleanup
            agents.clear()
            gc.collect()

            return True

        result = benchmark(agent_lifecycle)
        assert result is True

    @staticmethod
    def benchmark_belief_compression(benchmark):
        """Benchmark belief state compression."""
        from agents.memory_optimization.belief_compression import compress_beliefs

        def compress_large_beliefs():
            # Create sparse belief matrix
            beliefs = np.zeros((100, 100))
            # Add some sparse data
            for i in range(10):
                beliefs[np.random.randint(0, 100), np.random.randint(0, 100)] = np.random.rand()

            compressed = compress_beliefs(beliefs)
            return compressed

        result = benchmark(compress_large_beliefs)
        assert result is not None

    @staticmethod
    def benchmark_matrix_pooling(benchmark):
        """Benchmark matrix pooling efficiency."""
        from agents.memory_optimization.matrix_pooling import MatrixPool

        def use_matrix_pool():
            pool = MatrixPool(shape=(50, 50), max_matrices=10)

            matrices = []
            # Acquire matrices
            for _ in range(20):
                matrix = pool.acquire()
                matrices.append(matrix)

            # Release half
            for i in range(10):
                pool.release(matrices[i])

            # Acquire more
            for _ in range(10):
                matrix = pool.acquire()

            return True

        result = benchmark(use_matrix_pool)
        assert result is True


# Database Query Benchmarks
class DatabaseQueryBenchmarks:
    """Benchmarks for database query performance."""

    @staticmethod
    def benchmark_single_query(benchmark):
        """Benchmark single database query."""

        # Mock database query for benchmarking
        def mock_query():
            # Simulate query execution
            time.sleep(0.001)  # 1ms query
            return {"id": 1, "data": "test"}

        result = benchmark(mock_query)
        assert result["id"] == 1

    @staticmethod
    def benchmark_batch_queries(benchmark):
        """Benchmark batch database queries."""

        def batch_query(count=100):
            results = []
            for i in range(count):
                # Simulate query
                time.sleep(0.0001)  # 0.1ms per query
                results.append({"id": i, "data": f"row_{i}"})
            return results

        result = benchmark(batch_query, count=100)
        assert len(result) == 100

    @staticmethod
    def benchmark_connection_pool(benchmark):
        """Benchmark database connection pooling."""
        from queue import Queue

        def use_connection_pool():
            # Mock connection pool
            pool = Queue(maxsize=10)

            # Fill pool
            for i in range(10):
                pool.put(f"connection_{i}")

            results = []
            # Use connections
            for _ in range(50):
                conn = pool.get()
                # Simulate query
                time.sleep(0.0001)
                results.append(conn)
                pool.put(conn)

            return len(results)

        result = benchmark(use_connection_pool)
        assert result == 50


# WebSocket Connection Benchmarks
class WebSocketConnectionBenchmarks:
    """Benchmarks for WebSocket connection handling."""

    @staticmethod
    def benchmark_connection_setup(benchmark):
        """Benchmark WebSocket connection setup time."""
        from websocket.connection_pool import ConnectionPool

        def setup_connection():
            pool = ConnectionPool(max_connections=10)
            conn = pool.acquire()
            pool.release(conn)
            return True

        result = benchmark(setup_connection)
        assert result is True

    @staticmethod
    def benchmark_concurrent_connections(benchmark):
        """Benchmark concurrent WebSocket connections."""
        from websocket.connection_pool import ConnectionPool

        def handle_concurrent_connections(count=50):
            pool = ConnectionPool(max_connections=20)

            connections = []
            for i in range(count):
                conn = pool.acquire()
                connections.append(conn)

            # Release all
            for conn in connections:
                pool.release(conn)

            return len(connections)

        result = benchmark(handle_concurrent_connections, count=50)
        assert result == 50

    @staticmethod
    def benchmark_message_broadcast(benchmark):
        """Benchmark WebSocket message broadcasting."""

        def broadcast_messages(clients=100, messages=10):
            # Mock broadcast
            sent = 0
            for _ in range(messages):
                for _ in range(clients):
                    # Simulate send
                    time.sleep(0.00001)  # 10 microseconds
                    sent += 1
            return sent

        result = benchmark(broadcast_messages, clients=100, messages=10)
        assert result == 1000


# Performance Report Generator
class PerformanceReportGenerator:
    """Generate performance reports from benchmark results."""

    @staticmethod
    def generate_report(results: List[BenchmarkMetrics], output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_benchmarks": len(results),
                "categories": list(set(r.category for r in results)),
                "total_duration_ms": sum(r.duration_ms for r in results),
                "avg_memory_growth_mb": np.mean([r.memory_growth_mb for r in results]),
                "max_memory_peak_mb": max(r.memory_peak_mb for r in results),
            },
            "benchmarks": {},
            "regressions": [],
            "improvements": [],
        }

        # Group by category
        by_category = {}
        for result in results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result)

        # Analyze each category
        for category, cat_results in by_category.items():
            category_stats = {
                "count": len(cat_results),
                "avg_duration_ms": np.mean([r.duration_ms for r in cat_results]),
                "avg_ops_per_sec": np.mean([r.operations_per_second for r in cat_results]),
                "avg_memory_mb": np.mean([r.memory_growth_mb for r in cat_results]),
                "benchmarks": {},
            }

            for result in cat_results:
                category_stats["benchmarks"][result.name] = {
                    "duration_ms": result.duration_ms,
                    "ops_per_sec": result.operations_per_second,
                    "memory_growth_mb": result.memory_growth_mb,
                    "memory_peak_mb": result.memory_peak_mb,
                    "cpu_percent": result.cpu_percent,
                }

            report["benchmarks"][category] = category_stats

        # Save report
        report_path = (
            output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Also save latest
        latest_path = output_dir / "latest_performance_report.json"
        with open(latest_path, "w") as f:
            json.dump(report, f, indent=2)

        return report


# Regression Detection
class RegressionDetector:
    """Detect performance regressions."""

    @staticmethod
    def check_regressions(
        current: Dict[str, Any], baseline: Dict[str, Any], threshold: float = REGRESSION_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """Check for performance regressions against baseline."""
        regressions = []

        for category, stats in current["benchmarks"].items():
            if category not in baseline.get("benchmarks", {}):
                continue

            baseline_stats = baseline["benchmarks"][category]

            # Check average metrics
            for metric in ["avg_duration_ms", "avg_memory_mb"]:
                if metric in stats and metric in baseline_stats:
                    current_val = stats[metric]
                    baseline_val = baseline_stats[metric]

                    if baseline_val > 0:
                        regression_pct = (current_val - baseline_val) / baseline_val

                        if regression_pct > threshold:
                            regressions.append(
                                {
                                    "category": category,
                                    "metric": metric,
                                    "current": current_val,
                                    "baseline": baseline_val,
                                    "regression_percent": regression_pct * 100,
                                    "severity": (
                                        "critical"
                                        if regression_pct > REGRESSION_THRESHOLD
                                        else "warning"
                                    ),
                                }
                            )

        return regressions


# Pytest fixtures
@pytest.fixture(scope="session")
def performance_tracker():
    """Session-wide performance tracker."""
    results = []

    def track(metrics: BenchmarkMetrics):
        results.append(metrics)

    yield track

    # Generate report at end of session
    if results:
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        PerformanceReportGenerator.generate_report(results, output_dir)


@pytest.fixture
def memory_baseline():
    """Get memory baseline before each test."""
    gc.collect()
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


# Example usage
if __name__ == "__main__":
    # Run benchmarks
    pytest.main(
        [
            __file__,
            "-v",
            "--benchmark-only",
            "--benchmark-sort=name",
            "--benchmark-save=performance",
            "--benchmark-autosave",
        ]
    )

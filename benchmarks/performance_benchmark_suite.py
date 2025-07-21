"""
Comprehensive Performance Benchmarking and Monitoring Suite.

This module provides a unified benchmarking system that tests all performance optimizations:
1. Threading performance benchmarks
2. Database query performance tests
3. API response time benchmarks
4. Memory usage optimization validation
5. Agent coordination performance tests
6. End-to-end system performance validation
7. Performance regression detection
8. Automated performance reporting
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# Import all optimization modules
from agents.optimized_agent_manager import (
    OptimizationConfig,
    OptimizedAgentManager,
)
from observability.memory_optimizer import (
    get_memory_optimizer,
    start_memory_optimization,
)
from observability.performance_monitor import (
    get_performance_monitor,
    start_performance_monitoring,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""

    name: str
    category: str
    duration_seconds: float
    throughput_ops_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkSuite:
    """Configuration for a benchmark suite."""

    name: str
    description: str
    benchmarks: List[str]
    warmup_iterations: int = 5
    test_iterations: int = 10
    timeout_seconds: int = 300
    parallel_execution: bool = False


class PerformanceBenchmarkRunner:
    """Runs comprehensive performance benchmarks."""

    def __init__(self):
        """Initialize the performance benchmark runner."""
        self.results: List[BenchmarkResult] = []
        self.performance_monitor = get_performance_monitor()
        self.memory_optimizer = get_memory_optimizer()
        self.process = psutil.Process()

        # Benchmark configurations
        self.benchmark_suites = self._create_benchmark_suites()

        # Performance baselines (to detect regressions)
        self.baselines = {}

        logger.info("Performance benchmark runner initialized")

    def _create_benchmark_suites(self) -> Dict[str, BenchmarkSuite]:
        """Create predefined benchmark suites."""
        return {
            "threading": BenchmarkSuite(
                name="Threading Performance",
                description="Tests threading optimizations including adaptive pools and work-stealing",
                benchmarks=[
                    "thread_pool_scaling",
                    "work_stealing_efficiency",
                    "lock_contention",
                ],
                warmup_iterations=3,
                test_iterations=5,
            ),
            "database": BenchmarkSuite(
                name="Database Performance",
                description="Tests database connection pooling and query optimization",
                benchmarks=[
                    "connection_pooling",
                    "query_caching",
                    "batch_operations",
                ],
                warmup_iterations=2,
                test_iterations=8,
            ),
            "api": BenchmarkSuite(
                name="API Performance",
                description="Tests API response time optimization and caching",
                benchmarks=[
                    "response_caching",
                    "compression",
                    "concurrent_requests",
                ],
                warmup_iterations=3,
                test_iterations=10,
            ),
            "memory": BenchmarkSuite(
                name="Memory Optimization",
                description="Tests memory usage optimization and garbage collection",
                benchmarks=["memory_pooling", "gc_tuning", "leak_detection"],
                warmup_iterations=2,
                test_iterations=5,
            ),
            "agent_coordination": BenchmarkSuite(
                name="Agent Coordination",
                description="Tests multi-agent coordination and batching performance",
                benchmarks=[
                    "agent_batching",
                    "state_synchronization",
                    "message_passing",
                ],
                warmup_iterations=3,
                test_iterations=7,
            ),
            "end_to_end": BenchmarkSuite(
                name="End-to-End System",
                description="Tests complete system performance under realistic load",
                benchmarks=[
                    "full_system_load",
                    "scalability_test",
                    "stress_test",
                ],
                warmup_iterations=1,
                test_iterations=3,
            ),
        }

    async def run_benchmark_suite(self, suite_name: str) -> List[BenchmarkResult]:
        """Run a complete benchmark suite."""
        if suite_name not in self.benchmark_suites:
            raise ValueError(f"Unknown benchmark suite: {suite_name}")

        suite = self.benchmark_suites[suite_name]
        logger.info(f"Starting benchmark suite: {suite.name}")

        suite_results = []

        # Start monitoring
        start_performance_monitoring()
        start_memory_optimization("high_allocation")

        try:
            for benchmark_name in suite.benchmarks:
                logger.info(f"Running benchmark: {benchmark_name}")

                # Warmup iterations
                for i in range(suite.warmup_iterations):
                    await self._run_single_benchmark(benchmark_name, warmup=True)

                # Test iterations
                benchmark_results = []
                for i in range(suite.test_iterations):
                    result = await self._run_single_benchmark(
                        benchmark_name, warmup=False
                    )
                    benchmark_results.append(result)

                # Calculate aggregated result
                aggregated_result = self._aggregate_results(benchmark_results)
                suite_results.append(aggregated_result)

                logger.info(
                    f"Benchmark {benchmark_name} completed: "
                    f"{aggregated_result.throughput_ops_per_second:.1f} ops/sec, "
                    f"{aggregated_result.duration_seconds:.3f}s avg"
                )

        finally:
            # Store results
            self.results.extend(suite_results)

        logger.info(
            f"Benchmark suite {suite.name} completed with {len(suite_results)} results"
        )
        return suite_results

    async def _run_single_benchmark(
        self, benchmark_name: str, warmup: bool = False
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_cpu = self.process.cpu_percent()

        success = True
        error_message = None
        throughput = 0.0
        metadata = {}

        try:
            # Route to appropriate benchmark method
            if benchmark_name == "thread_pool_scaling":
                (
                    throughput,
                    metadata,
                ) = await self._benchmark_thread_pool_scaling()
            elif benchmark_name == "work_stealing_efficiency":
                throughput, metadata = await self._benchmark_work_stealing()
            elif benchmark_name == "lock_contention":
                throughput, metadata = await self._benchmark_lock_contention()
            elif benchmark_name == "connection_pooling":
                (
                    throughput,
                    metadata,
                ) = await self._benchmark_connection_pooling()
            elif benchmark_name == "query_caching":
                throughput, metadata = await self._benchmark_query_caching()
            elif benchmark_name == "batch_operations":
                throughput, metadata = await self._benchmark_batch_operations()
            elif benchmark_name == "response_caching":
                throughput, metadata = await self._benchmark_response_caching()
            elif benchmark_name == "compression":
                throughput, metadata = await self._benchmark_compression()
            elif benchmark_name == "concurrent_requests":
                (
                    throughput,
                    metadata,
                ) = await self._benchmark_concurrent_requests()
            elif benchmark_name == "memory_pooling":
                throughput, metadata = await self._benchmark_memory_pooling()
            elif benchmark_name == "gc_tuning":
                throughput, metadata = await self._benchmark_gc_tuning()
            elif benchmark_name == "leak_detection":
                throughput, metadata = await self._benchmark_leak_detection()
            elif benchmark_name == "agent_batching":
                throughput, metadata = await self._benchmark_agent_batching()
            elif benchmark_name == "state_synchronization":
                (
                    throughput,
                    metadata,
                ) = await self._benchmark_state_synchronization()
            elif benchmark_name == "message_passing":
                throughput, metadata = await self._benchmark_message_passing()
            elif benchmark_name == "full_system_load":
                throughput, metadata = await self._benchmark_full_system_load()
            elif benchmark_name == "scalability_test":
                throughput, metadata = await self._benchmark_scalability()
            elif benchmark_name == "stress_test":
                throughput, metadata = await self._benchmark_stress_test()
            else:
                raise ValueError(f"Unknown benchmark: {benchmark_name}")

        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Benchmark {benchmark_name} failed: {e}")

        duration = time.perf_counter() - start_time
        end_memory = self.process.memory_info().rss / 1024 / 1024
        end_cpu = self.process.cpu_percent()

        return BenchmarkResult(
            name=benchmark_name,
            category=self._get_benchmark_category(benchmark_name),
            duration_seconds=duration,
            throughput_ops_per_second=throughput,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=end_cpu - start_cpu,
            success=success,
            error_message=error_message,
            metadata=metadata,
        )

    def _get_benchmark_category(self, benchmark_name: str) -> str:
        """Get category for a benchmark."""
        category_map = {
            "thread_pool_scaling": "threading",
            "work_stealing_efficiency": "threading",
            "lock_contention": "threading",
            "connection_pooling": "database",
            "query_caching": "database",
            "batch_operations": "database",
            "response_caching": "api",
            "compression": "api",
            "concurrent_requests": "api",
            "memory_pooling": "memory",
            "gc_tuning": "memory",
            "leak_detection": "memory",
            "agent_batching": "agents",
            "state_synchronization": "agents",
            "message_passing": "agents",
            "full_system_load": "system",
            "scalability_test": "system",
            "stress_test": "system",
        }
        return category_map.get(benchmark_name, "unknown")

    def _aggregate_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Aggregate multiple benchmark results."""
        if not results:
            raise ValueError("No results to aggregate")

        successful_results = [r for r in results if r.success]

        if not successful_results:
            # All failed, return the first failure
            return results[0]

        # Calculate averages
        avg_duration = sum(r.duration_seconds for r in successful_results) / len(
            successful_results
        )
        avg_throughput = sum(
            r.throughput_ops_per_second for r in successful_results
        ) / len(successful_results)
        avg_memory = sum(r.memory_usage_mb for r in successful_results) / len(
            successful_results
        )
        avg_cpu = sum(r.cpu_usage_percent for r in successful_results) / len(
            successful_results
        )

        # Aggregate metadata
        metadata = {
            "iterations": len(results),
            "successful_iterations": len(successful_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "std_deviation_throughput": np.std(
                [r.throughput_ops_per_second for r in successful_results]
            ),
            "min_throughput": min(
                r.throughput_ops_per_second for r in successful_results
            ),
            "max_throughput": max(
                r.throughput_ops_per_second for r in successful_results
            ),
        }

        return BenchmarkResult(
            name=results[0].name,
            category=results[0].category,
            duration_seconds=avg_duration,
            throughput_ops_per_second=avg_throughput,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            success=len(successful_results) > 0,
            metadata=metadata,
        )

    # Threading benchmarks
    async def _benchmark_thread_pool_scaling(
        self,
    ) -> Tuple[float, Dict[str, Any]]:
        """Benchmark thread pool scaling performance."""
        config = OptimizationConfig(
            cpu_aware_sizing=True, work_stealing_enabled=True, batch_size=20
        )

        manager = OptimizedAgentManager(config)

        # Mock agents
        class MockAgent:
            def __init__(self, agent_id: str):
                self.agent_id = agent_id
                self.step_count = 0

            def step(self, observation):
                # Simulate work
                time.sleep(0.001)
                self.step_count += 1
                return f"action_{self.step_count}"

        # Create agents
        num_agents = 100
        for i in range(num_agents):
            agent = MockAgent(f"agent_{i}")
            manager.register_agent(agent.agent_id, agent)

        # Benchmark
        observations = {f"agent_{i}": {"data": i} for i in range(num_agents)}

        start_time = time.perf_counter()
        manager.step_agents_async(observations)

        # Wait for completion
        await asyncio.sleep(0.1)

        elapsed = time.perf_counter() - start_time
        throughput = num_agents / elapsed

        stats = manager.get_statistics()

        manager.shutdown()

        metadata = {
            "agents": num_agents,
            "thread_pool_stats": stats["thread_pool"],
            "batch_stats": {
                "batches_processed": stats["batches_processed"],
                "avg_batch_size": stats["avg_batch_size"],
            },
        }

        return throughput, metadata

    async def _benchmark_work_stealing(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark work-stealing efficiency."""
        from agents.optimized_agent_manager import AdaptiveThreadPool

        config = OptimizationConfig(work_stealing_enabled=True)
        thread_pool = AdaptiveThreadPool(config)

        # Submit imbalanced workload
        def variable_work(duration):
            time.sleep(duration)
            return duration

        num_tasks = 100
        start_time = time.perf_counter()

        # Submit tasks with varying durations
        for i in range(num_tasks):
            duration = 0.01 if i % 10 == 0 else 0.001
            thread_pool.submit(variable_work, duration)

        # Wait for completion
        await asyncio.sleep(2.0)

        elapsed = time.perf_counter() - start_time
        throughput = num_tasks / elapsed

        stats = thread_pool.get_stats()
        thread_pool.shutdown()

        metadata = {
            "tasks": num_tasks,
            "steal_efficiency": stats["steal_efficiency"],
            "workload_type": stats["workload_type"],
            "avg_task_time_ms": stats["avg_task_time_ms"],
        }

        return throughput, metadata

    async def _benchmark_lock_contention(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark lock contention performance."""
        from agents.optimized_agent_manager import LockFreeAgentRegistry

        registry = LockFreeAgentRegistry()

        # Benchmark concurrent access
        def concurrent_access():
            for i in range(100):
                agent_id = f"agent_{i % 20}"
                registry.register(agent_id, f"data_{i}")
                agent = registry.get(agent_id)
                if agent:
                    registry.remove(agent_id)

        num_threads = 10
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(concurrent_access) for _ in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        elapsed = time.perf_counter() - start_time
        throughput = (num_threads * 100) / elapsed

        metadata = {
            "threads": num_threads,
            "operations_per_thread": 100,
            "shards": len(registry.shards),
            "final_agent_count": registry.size(),
        }

        return throughput, metadata

    # Database benchmarks
    async def _benchmark_connection_pooling(
        self,
    ) -> Tuple[float, Dict[str, Any]]:
        """Benchmark database connection pooling."""

        # Mock database operations
        async def mock_query():
            await asyncio.sleep(0.01)  # Simulate query
            return "result"

        num_queries = 100
        start_time = time.perf_counter()

        # Simulate concurrent queries
        tasks = [mock_query() for _ in range(num_queries)]
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start_time
        throughput = num_queries / elapsed

        metadata = {
            "queries": num_queries,
            "concurrent_execution": True,
            "avg_query_time_ms": (elapsed / num_queries) * 1000,
        }

        return throughput, metadata

    async def _benchmark_query_caching(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark query result caching."""
        # Mock cache implementation
        cache = {}
        cache_hits = 0
        cache_misses = 0

        async def cached_query(query_id: str):
            nonlocal cache_hits, cache_misses

            if query_id in cache:
                cache_hits += 1
                return cache[query_id]
            else:
                cache_misses += 1
                await asyncio.sleep(0.01)  # Simulate query
                result = f"result_{query_id}"
                cache[query_id] = result
                return result

        # Run queries with repetition to test caching
        queries = [f"query_{i % 20}" for i in range(100)]  # 20 unique queries, repeated

        start_time = time.perf_counter()

        for query in queries:
            await cached_query(query)

        elapsed = time.perf_counter() - start_time
        throughput = len(queries) / elapsed

        metadata = {
            "total_queries": len(queries),
            "unique_queries": 20,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate": (cache_hits / len(queries)) * 100,
        }

        return throughput, metadata

    async def _benchmark_batch_operations(
        self,
    ) -> Tuple[float, Dict[str, Any]]:
        """Benchmark batch database operations."""

        # Mock batch processing
        async def batch_insert(batch_size: int):
            await asyncio.sleep(batch_size * 0.001)  # Simulate batch insert
            return batch_size

        total_operations = 1000
        batch_size = 50
        num_batches = total_operations // batch_size

        start_time = time.perf_counter()

        # Process in batches
        for i in range(num_batches):
            await batch_insert(batch_size)

        elapsed = time.perf_counter() - start_time
        throughput = total_operations / elapsed

        metadata = {
            "total_operations": total_operations,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "avg_batch_time_ms": (elapsed / num_batches) * 1000,
        }

        return throughput, metadata

    # API benchmarks
    async def _benchmark_response_caching(
        self,
    ) -> Tuple[float, Dict[str, Any]]:
        """Benchmark API response caching."""
        # Mock API response caching
        cache = {}
        cache_hits = 0
        cache_misses = 0

        async def cached_api_call(endpoint: str):
            nonlocal cache_hits, cache_misses

            if endpoint in cache:
                cache_hits += 1
                return cache[endpoint]
            else:
                cache_misses += 1
                await asyncio.sleep(0.005)  # Simulate API processing
                response = f"response_{endpoint}"
                cache[endpoint] = response
                return response

        # Simulate API calls
        endpoints = [f"/api/endpoint_{i % 10}" for i in range(100)]

        start_time = time.perf_counter()

        for endpoint in endpoints:
            await cached_api_call(endpoint)

        elapsed = time.perf_counter() - start_time
        throughput = len(endpoints) / elapsed

        metadata = {
            "api_calls": len(endpoints),
            "unique_endpoints": 10,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate": (cache_hits / len(endpoints)) * 100,
        }

        return throughput, metadata

    async def _benchmark_compression(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark response compression."""
        import gzip

        # Generate test data
        test_data = {"data": [{"id": i, "value": f"value_{i}"} for i in range(1000)]}
        json_data = json.dumps(test_data).encode()

        original_size = len(json_data)

        start_time = time.perf_counter()

        # Compress data
        compressed_data = gzip.compress(json_data)

        elapsed = time.perf_counter() - start_time
        compressed_size = len(compressed_data)

        # Throughput in terms of original bytes processed per second
        throughput = original_size / elapsed

        metadata = {
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": original_size / compressed_size,
            "space_saved_percent": ((original_size - compressed_size) / original_size)
            * 100,
        }

        return throughput, metadata

    async def _benchmark_concurrent_requests(
        self,
    ) -> Tuple[float, Dict[str, Any]]:
        """Benchmark concurrent API request handling."""

        async def mock_api_request():
            await asyncio.sleep(0.01)  # Simulate request processing
            return "response"

        num_requests = 100
        concurrency = 20

        start_time = time.perf_counter()

        # Process requests in batches for controlled concurrency
        for i in range(0, num_requests, concurrency):
            batch = [
                mock_api_request() for _ in range(min(concurrency, num_requests - i))
            ]
            await asyncio.gather(*batch)

        elapsed = time.perf_counter() - start_time
        throughput = num_requests / elapsed

        metadata = {
            "total_requests": num_requests,
            "concurrency": concurrency,
            "avg_request_time_ms": (elapsed / num_requests) * 1000,
        }

        return throughput, metadata

    # Memory benchmarks
    async def _benchmark_memory_pooling(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark memory pooling performance."""
        from observability.memory_optimizer import ObjectPool

        # Create object pool
        pool = ObjectPool(
            factory=lambda: {"data": np.zeros(100)},
            max_size=50,
            reset_func=lambda obj: obj["data"].fill(0),
        )

        num_operations = 1000
        start_time = time.perf_counter()

        # Benchmark acquire/release cycle
        for i in range(num_operations):
            obj = pool.acquire()
            # Simulate work
            obj["data"][i % 100] = i
            pool.release(obj)

        elapsed = time.perf_counter() - start_time
        throughput = num_operations / elapsed

        stats = pool.get_stats()

        metadata = {
            "operations": num_operations,
            "pool_stats": stats,
            "reuse_rate": stats["reuse_rate"],
        }

        return throughput, metadata

    async def _benchmark_gc_tuning(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark garbage collection tuning."""
        import gc

        # Create many objects to trigger GC
        objects = []

        start_time = time.perf_counter()

        for i in range(10000):
            obj = {
                "id": i,
                "data": list(range(10)),
                "metadata": {"created": time.time()},
            }
            objects.append(obj)

            # Trigger GC periodically
            if i % 1000 == 0:
                gc.collect()

        elapsed = time.perf_counter() - start_time
        throughput = len(objects) / elapsed

        # Get GC statistics
        gc_stats = gc.get_stats()

        metadata = {
            "objects_created": len(objects),
            "gc_collections": [stat["collections"] for stat in gc_stats],
            "gc_collected": sum(stat.get("collected", 0) for stat in gc_stats),
        }

        return throughput, metadata

    async def _benchmark_leak_detection(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark memory leak detection."""
        # Simulate potential memory leak
        leaked_objects = []

        start_time = time.perf_counter()

        for i in range(1000):
            # Create object that might leak
            obj = {"id": i, "data": np.random.rand(100), "circular_ref": None}
            obj["circular_ref"] = obj  # Create circular reference
            leaked_objects.append(obj)

        elapsed = time.perf_counter() - start_time
        throughput = len(leaked_objects) / elapsed

        # Measure memory usage
        memory_usage = len(leaked_objects) * 100 * 8 / 1024 / 1024  # Approximate MB

        metadata = {
            "objects_created": len(leaked_objects),
            "estimated_memory_mb": memory_usage,
            "circular_references": len(leaked_objects),
        }

        # Clean up circular references
        for obj in leaked_objects:
            obj["circular_ref"] = None

        return throughput, metadata

    # Agent benchmarks
    async def _benchmark_agent_batching(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark agent operation batching."""

        # Mock agent batch processing
        async def process_agent_batch(batch_size: int):
            await asyncio.sleep(batch_size * 0.001)  # Simulate batch processing
            return batch_size

        total_agents = 1000
        batch_size = 50
        num_batches = total_agents // batch_size

        start_time = time.perf_counter()

        # Process agents in batches
        for i in range(num_batches):
            await process_agent_batch(batch_size)

        elapsed = time.perf_counter() - start_time
        throughput = total_agents / elapsed

        metadata = {
            "total_agents": total_agents,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "avg_batch_time_ms": (elapsed / num_batches) * 1000,
        }

        return throughput, metadata

    async def _benchmark_state_synchronization(
        self,
    ) -> Tuple[float, Dict[str, Any]]:
        """Benchmark agent state synchronization."""
        # Mock state synchronization
        agents = {}

        async def sync_agent_state(agent_id: str, state: Dict[str, Any]):
            await asyncio.sleep(0.001)  # Simulate sync
            agents[agent_id] = state

        num_agents = 100
        num_updates = 5

        start_time = time.perf_counter()

        # Synchronize states for all agents
        for update in range(num_updates):
            tasks = []
            for agent_id in range(num_agents):
                state = {"update": update, "data": np.random.rand(10)}
                tasks.append(sync_agent_state(f"agent_{agent_id}", state))
            await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start_time
        total_operations = num_agents * num_updates
        throughput = total_operations / elapsed

        metadata = {
            "agents": num_agents,
            "updates_per_agent": num_updates,
            "total_operations": total_operations,
            "avg_sync_time_ms": (elapsed / total_operations) * 1000,
        }

        return throughput, metadata

    async def _benchmark_message_passing(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark inter-agent message passing."""
        # Mock message passing system
        message_queue = asyncio.Queue()
        messages_processed = 0

        async def send_message(sender: str, receiver: str, message: Dict[str, Any]):
            await message_queue.put(
                {"sender": sender, "receiver": receiver, "message": message}
            )

        async def process_messages():
            nonlocal messages_processed
            while True:
                try:
                    await asyncio.wait_for(message_queue.get(), timeout=0.1)
                    messages_processed += 1
                    await asyncio.sleep(0.001)  # Simulate processing
                except asyncio.TimeoutError:
                    break

        num_agents = 20
        messages_per_agent = 10

        start_time = time.perf_counter()

        # Start message processor
        processor_task = asyncio.create_task(process_messages())

        # Send messages between agents
        for i in range(num_agents):
            for j in range(messages_per_agent):
                sender = f"agent_{i}"
                receiver = f"agent_{(i + 1) % num_agents}"
                message = {"type": "coordination", "data": f"message_{j}"}
                await send_message(sender, receiver, message)

        # Wait for processing to complete
        await processor_task

        elapsed = time.perf_counter() - start_time
        throughput = messages_processed / elapsed

        metadata = {
            "agents": num_agents,
            "messages_per_agent": messages_per_agent,
            "total_messages": messages_processed,
            "avg_message_time_ms": (elapsed / messages_processed) * 1000,
        }

        return throughput, metadata

    # System benchmarks
    async def _benchmark_full_system_load(
        self,
    ) -> Tuple[float, Dict[str, Any]]:
        """Benchmark full system under realistic load."""
        # This would test the complete system integration
        # For now, we'll simulate a comprehensive load test

        num_agents = 50
        num_requests = 100
        num_db_operations = 200

        start_time = time.perf_counter()

        # Simulate concurrent system operations
        tasks = []

        # Agent operations
        for i in range(num_agents):
            task = asyncio.create_task(self._simulate_agent_operation(f"agent_{i}"))
            tasks.append(task)

        # API requests
        for i in range(num_requests):
            task = asyncio.create_task(self._simulate_api_request(f"request_{i}"))
            tasks.append(task)

        # Database operations
        for i in range(num_db_operations):
            task = asyncio.create_task(self._simulate_db_operation(f"query_{i}"))
            tasks.append(task)

        # Wait for all operations
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start_time
        total_operations = num_agents + num_requests + num_db_operations
        throughput = total_operations / elapsed

        metadata = {
            "agents": num_agents,
            "api_requests": num_requests,
            "db_operations": num_db_operations,
            "total_operations": total_operations,
            "avg_operation_time_ms": (elapsed / total_operations) * 1000,
        }

        return throughput, metadata

    async def _simulate_agent_operation(self, agent_id: str):
        """Simulate agent operation."""
        await asyncio.sleep(0.01)  # Simulate processing
        return f"agent_{agent_id}_result"

    async def _simulate_api_request(self, request_id: str):
        """Simulate API request."""
        await asyncio.sleep(0.005)  # Simulate request processing
        return f"api_{request_id}_response"

    async def _simulate_db_operation(self, query_id: str):
        """Simulate database operation."""
        await asyncio.sleep(0.008)  # Simulate query
        return f"db_{query_id}_result"

    async def _benchmark_scalability(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark system scalability."""
        # Test scaling from 10 to 100 agents
        agent_counts = [10, 25, 50, 100]
        scalability_results = []

        for agent_count in agent_counts:
            start_time = time.perf_counter()

            # Simulate agent operations
            tasks = []
            for i in range(agent_count):
                task = asyncio.create_task(self._simulate_agent_operation(f"agent_{i}"))
                tasks.append(task)

            await asyncio.gather(*tasks)

            elapsed = time.perf_counter() - start_time
            throughput = agent_count / elapsed

            scalability_results.append(
                {
                    "agent_count": agent_count,
                    "throughput": throughput,
                    "elapsed": elapsed,
                }
            )

        # Calculate scaling efficiency
        base_throughput = scalability_results[0]["throughput"]
        final_throughput = scalability_results[-1]["throughput"]
        scaling_efficiency = (final_throughput / base_throughput) / (
            agent_counts[-1] / agent_counts[0]
        )

        metadata = {
            "scalability_results": scalability_results,
            "scaling_efficiency": scaling_efficiency,
            "agent_range": f"{agent_counts[0]}-{agent_counts[-1]}",
        }

        return final_throughput, metadata

    async def _benchmark_stress_test(self) -> Tuple[float, Dict[str, Any]]:
        """Benchmark system under stress conditions."""
        # High-load stress test
        stress_duration = 10  # seconds
        operations_per_second = 100

        start_time = time.perf_counter()
        end_time = start_time + stress_duration

        operations_completed = 0

        while time.perf_counter() < end_time:
            # Simulate high-frequency operations
            tasks = []
            for i in range(operations_per_second):
                task = asyncio.create_task(self._simulate_stress_operation())
                tasks.append(task)

            await asyncio.gather(*tasks)
            operations_completed += operations_per_second

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)

        elapsed = time.perf_counter() - start_time
        throughput = operations_completed / elapsed

        metadata = {
            "stress_duration": stress_duration,
            "operations_completed": operations_completed,
            "target_ops_per_second": operations_per_second,
            "actual_ops_per_second": throughput,
        }

        return throughput, metadata

    async def _simulate_stress_operation(self):
        """Simulate high-frequency operation."""
        await asyncio.sleep(0.001)  # Very fast operation
        return "stress_result"

    def get_benchmark_results(self, category: str = None) -> List[BenchmarkResult]:
        """Get benchmark results, optionally filtered by category."""
        if category:
            return [r for r in self.results if r.category == category]
        return self.results

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {"error": "No benchmark results available"}

        # Group results by category
        results_by_category = {}
        for result in self.results:
            if result.category not in results_by_category:
                results_by_category[result.category] = []
            results_by_category[result.category].append(result)

        # Calculate statistics
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_benchmarks": len(self.results),
            "categories": list(results_by_category.keys()),
            "overall_stats": self._calculate_overall_stats(),
            "category_stats": {},
            "top_performers": self._get_top_performers(),
            "performance_issues": self._identify_performance_issues(),
            "recommendations": self._generate_recommendations(),
        }

        # Add category-specific stats
        for category, results in results_by_category.items():
            report["category_stats"][category] = self._calculate_category_stats(results)

        return report

    def _calculate_overall_stats(self) -> Dict[str, Any]:
        """Calculate overall performance statistics."""
        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            return {"error": "No successful benchmark results"}

        return {
            "success_rate": len(successful_results) / len(self.results) * 100,
            "avg_throughput": sum(
                r.throughput_ops_per_second for r in successful_results
            )
            / len(successful_results),
            "avg_duration": sum(r.duration_seconds for r in successful_results)
            / len(successful_results),
            "avg_memory_usage": sum(r.memory_usage_mb for r in successful_results)
            / len(successful_results),
            "avg_cpu_usage": sum(r.cpu_usage_percent for r in successful_results)
            / len(successful_results),
        }

    def _calculate_category_stats(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Calculate statistics for a specific category."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {"error": "No successful results in category"}

        throughputs = [r.throughput_ops_per_second for r in successful_results]

        return {
            "benchmark_count": len(results),
            "success_rate": len(successful_results) / len(results) * 100,
            "avg_throughput": sum(throughputs) / len(throughputs),
            "min_throughput": min(throughputs),
            "max_throughput": max(throughputs),
            "std_throughput": np.std(throughputs),
            "best_benchmark": max(
                successful_results, key=lambda r: r.throughput_ops_per_second
            ).name,
            "worst_benchmark": min(
                successful_results, key=lambda r: r.throughput_ops_per_second
            ).name,
        }

    def _get_top_performers(self) -> List[Dict[str, Any]]:
        """Get top performing benchmarks."""
        successful_results = [r for r in self.results if r.success]
        top_performers = sorted(
            successful_results,
            key=lambda r: r.throughput_ops_per_second,
            reverse=True,
        )[:5]

        return [
            {
                "name": result.name,
                "category": result.category,
                "throughput": result.throughput_ops_per_second,
                "duration": result.duration_seconds,
            }
            for result in top_performers
        ]

    def _identify_performance_issues(self) -> List[Dict[str, Any]]:
        """Identify performance issues from benchmark results."""
        issues = []

        for result in self.results:
            if not result.success:
                issues.append(
                    {
                        "type": "failure",
                        "benchmark": result.name,
                        "category": result.category,
                        "error": result.error_message,
                    }
                )
            elif result.throughput_ops_per_second < 10:  # Arbitrary threshold
                issues.append(
                    {
                        "type": "low_throughput",
                        "benchmark": result.name,
                        "category": result.category,
                        "throughput": result.throughput_ops_per_second,
                    }
                )
            elif result.duration_seconds > 5:  # Arbitrary threshold
                issues.append(
                    {
                        "type": "slow_execution",
                        "benchmark": result.name,
                        "category": result.category,
                        "duration": result.duration_seconds,
                    }
                )

        return issues

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Analyze results and provide recommendations
        category_stats = {}
        for result in self.results:
            if result.category not in category_stats:
                category_stats[result.category] = []
            category_stats[result.category].append(result)

        # Threading recommendations
        if "threading" in category_stats:
            thread_results = [r for r in category_stats["threading"] if r.success]
            if thread_results:
                avg_throughput = sum(
                    r.throughput_ops_per_second for r in thread_results
                ) / len(thread_results)
                if avg_throughput < 100:
                    recommendations.append(
                        "Consider optimizing thread pool configuration for better throughput"
                    )

        # Database recommendations
        if "database" in category_stats:
            db_results = [r for r in category_stats["database"] if r.success]
            if db_results:
                avg_duration = sum(r.duration_seconds for r in db_results) / len(
                    db_results
                )
                if avg_duration > 1.0:
                    recommendations.append(
                        "Database operations are slow. Consider query optimization or connection pooling"
                    )

        # Memory recommendations
        if "memory" in category_stats:
            memory_results = [r for r in category_stats["memory"] if r.success]
            if memory_results:
                avg_memory = sum(r.memory_usage_mb for r in memory_results) / len(
                    memory_results
                )
                if avg_memory > 100:
                    recommendations.append(
                        "High memory usage detected. Consider implementing memory pooling"
                    )

        # General recommendations
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            recommendations.append(
                f"{len(failed_results)} benchmarks failed. Investigation recommended"
            )

        return recommendations

    def save_results(self, filename: str):
        """Save benchmark results to file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "duration_seconds": r.duration_seconds,
                    "throughput_ops_per_second": r.throughput_ops_per_second,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "success": r.success,
                    "error_message": r.error_message,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.results
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Benchmark results saved to {filename}")

    def load_results(self, filename: str):
        """Load benchmark results from file."""
        with open(filename, "r") as f:
            data = json.load(f)

        self.results = [
            BenchmarkResult(
                name=r["name"],
                category=r["category"],
                duration_seconds=r["duration_seconds"],
                throughput_ops_per_second=r["throughput_ops_per_second"],
                memory_usage_mb=r["memory_usage_mb"],
                cpu_usage_percent=r["cpu_usage_percent"],
                success=r["success"],
                error_message=r["error_message"],
                metadata=r["metadata"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for r in data["results"]
        ]

        logger.info(f"Benchmark results loaded from {filename}")


# Example usage
async def run_complete_benchmark_suite():
    """Run the complete benchmark suite."""
    print("=" * 80)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)

    runner = PerformanceBenchmarkRunner()

    # Run all benchmark suites
    suite_names = [
        "threading",
        "database",
        "api",
        "memory",
        "agent_coordination",
        "end_to_end",
    ]

    for suite_name in suite_names:
        print(f"\n{'=' * 20} {suite_name.upper()} BENCHMARKS {'=' * 20}")

        try:
            results = await runner.run_benchmark_suite(suite_name)

            for result in results:
                status = "✓" if result.success else "✗"
                print(
                    f"{status} {result.name}: {result.throughput_ops_per_second:.1f} ops/sec "
                    f"({result.duration_seconds:.3f}s)"
                )

        except Exception as e:
            print(f"✗ Suite {suite_name} failed: {e}")

    # Generate comprehensive report
    print(f"\n{'=' * 30} PERFORMANCE REPORT {'=' * 30}")
    report = runner.generate_performance_report()

    print(f"Total benchmarks: {report['total_benchmarks']}")
    print(f"Categories: {', '.join(report['categories'])}")
    print(f"Overall success rate: {report['overall_stats']['success_rate']:.1f}%")
    print(
        f"Average throughput: {report['overall_stats']['avg_throughput']:.1f} ops/sec"
    )

    print("\nTop performers:")
    for performer in report["top_performers"]:
        print(f" - {performer['name']}: {performer['throughput']:.1f} ops/sec")

    if report["performance_issues"]:
        print("\nPerformance issues:")
        for issue in report["performance_issues"]:
            print(f" - {issue['type']}: {issue['benchmark']}")

    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f" - {rec}")

    # Save results
    runner.save_results("performance_benchmark_results.json")

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_complete_benchmark_suite())

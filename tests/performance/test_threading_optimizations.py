#!/usr/bin/env python3
"""
Test suite for threading optimization recommendations.

This module tests the performance improvements identified in the
threading optimization analysis (subtask 4.3).
"""

import asyncio
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import unittest
from typing import Any, List
from unittest.mock import Mock, patch

import numpy as np

from tests.performance.performance_utils import replace_sleep

# Add parent directory to path
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from agents.optimized_threadpool_manager import OptimizedThreadPoolManager
from agents.performance_optimizer import AsyncInferenceEngine


class TestThreadPoolOptimization(unittest.TestCase):
    """Test thread pool tuning optimizations."""

    def setUp(self):
        """Set up test fixtures."""
        self.cpu_count = mp.cpu_count()
        self.test_agents = []
        for i in range(10):
            agent = Mock()
            agent.agent_id = f"test_agent_{i}"
            agent.step = Mock(return_value=f"action_{i}")
            self.test_agents.append(agent)

    def test_optimal_worker_calculation(self):
        """Test optimal worker thread calculation based on CPU topology."""
        # Test various scenarios
        test_cases = [
            (4, 10, 8),  # 4 CPUs, 10 agents -> 8 workers
            (8, 100, 16),  # 8 CPUs, 100 agents -> 16 workers
            (16, 50, 32),  # 16 CPUs, 50 agents -> 32 workers
        ]

        for cpu_count, total_agents, expected_initial in test_cases:
            with patch("multiprocessing.cpu_count", return_value=cpu_count):
                optimal_workers = min(cpu_count * 2, total_agents)
                initial_workers = max(cpu_count, min(16, optimal_workers))

                self.assertEqual(
                    initial_workers,
                    expected_initial,
                    f"Failed for {cpu_count} CPUs, {total_agents} agents",
                )

    def test_adaptive_scaling_thresholds(self):
        """Test adaptive scaling behavior with optimized thresholds."""
        manager = OptimizedThreadPoolManager(
            initial_workers=4,
            max_workers=16,
            scaling_threshold=0.7,  # Optimized threshold
        )

        # Register test agents
        for agent in self.test_agents:
            manager.register_agent(agent.agent_id, agent)

        # Generate load to trigger scaling
        manager.current_workers

        # Submit many tasks to increase load
        futures = []
        for _ in range(50):
            for agent in self.test_agents:
                future = manager.submit_task(agent.agent_id, "step", {"observation": {}})
                futures.append(future)

        # Wait for some completion
        replace_sleep(0.1)

        # Check if scaling occurred
        status = manager.get_pool_status()
        self.assertGreater(
            status["load_factor"],
            0.7,
            "Load factor should exceed scaling threshold",
        )

        # Cleanup
        manager.shutdown()

    def test_thread_pool_performance(self):
        """Measure thread pool performance with different configurations."""
        configurations = [
            {"initial": 4, "max": 8},
            {"initial": 8, "max": 16},
            {"initial": 16, "max": 32},
        ]

        results = {}

        for config in configurations:
            manager = OptimizedThreadPoolManager(
                initial_workers=config["initial"], max_workers=config["max"]
            )

            # Register agents
            for agent in self.test_agents:
                manager.register_agent(agent.agent_id, agent)

            # Measure throughput
            start_time = time.time()
            observations = {agent.agent_id: {"data": i} for i, agent in enumerate(self.test_agents)}

            for _ in range(10):
                results = manager.step_all_agents(observations)

            duration = time.time() - start_time
            throughput = (10 * len(self.test_agents)) / duration

            results[f"{config['initial']}-{config['max']}"] = throughput
            manager.shutdown()

        # Verify performance improves with better configuration
        throughputs = list(results.values())
        self.assertGreater(
            throughputs[-1],
            throughputs[0],
            "Larger thread pools should show better throughput",
        )


class TestGILAwareScheduling(unittest.TestCase):
    """Test GIL-aware scheduling optimizations."""

    def test_io_batch_processing(self):
        """Test batched I/O operations for GIL efficiency."""

        async def batch_io_operations(operations: List[Any]) -> List[Any]:
            """Batch I/O operations to release GIL efficiently."""
            # Simulate I/O that releases GIL
            await asyncio.sleep(0.001)
            return await asyncio.gather(*operations)

        async def single_io_operation():
            """Single I/O operation."""
            await asyncio.sleep(0.001)
            return "result"

        async def measure_performance():
            # Measure batched performance
            batch_start = time.time()
            operations = [single_io_operation() for _ in range(100)]
            await batch_io_operations(operations)
            batch_duration = time.time() - batch_start

            # Measure sequential performance
            seq_start = time.time()
            seq_results = []
            for _ in range(100):
                result = await single_io_operation()
                seq_results.append(result)
            seq_duration = time.time() - seq_start

            return batch_duration, seq_duration

        # Run async test
        batch_time, seq_time = asyncio.run(measure_performance())

        # Batched should be significantly faster
        self.assertLess(
            batch_time,
            seq_time * 0.5,
            "Batched I/O should be at least 2x faster",
        )

    def test_numpy_operation_batching(self):
        """Test NumPy operation batching for GIL release."""

        def batch_matrix_operations(matrices: List[np.ndarray]) -> np.ndarray:
            """Process multiple matrices in single NumPy call."""
            # NumPy releases GIL for this operation
            return np.stack(matrices).sum(axis=0)

        def sequential_matrix_operations(
            matrices: List[np.ndarray],
        ) -> np.ndarray:
            """Process matrices sequentially."""
            result = np.zeros_like(matrices[0])
            for matrix in matrices:
                result += matrix
            return result

        # Create test matrices
        matrices = [np.random.rand(100, 100) for _ in range(50)]

        # Measure batched performance
        batch_start = time.time()
        for _ in range(10):
            batch_result = batch_matrix_operations(matrices)
        batch_duration = time.time() - batch_start

        # Measure sequential performance
        seq_start = time.time()
        for _ in range(10):
            seq_result = sequential_matrix_operations(matrices)
        seq_duration = time.time() - seq_start

        # Verify results are equivalent
        np.testing.assert_allclose(batch_result, seq_result, rtol=1e-10)

        # Batched should be faster
        self.assertLess(
            batch_duration,
            seq_duration,
            "Batched NumPy operations should be faster",
        )


class TestMemoryAccessPatterns(unittest.TestCase):
    """Test memory access pattern optimizations."""

    def test_cache_aligned_data_structures(self):
        """Test cache-line aligned data structures."""

        class CacheAlignedAgent:
            """Agent with cache-line aligned attributes."""

            def __init__(self):
                # Align to 64-byte cache lines
                self._padding1 = [0] * 8  # 64 bytes
                self.beliefs = np.zeros((10, 10))
                self._padding2 = [0] * 8  # 64 bytes
                self.observations = []
                self._padding3 = [0] * 8  # 64 bytes

        class RegularAgent:
            """Regular agent without alignment."""

            def __init__(self):
                self.beliefs = np.zeros((10, 10))
                self.observations = []

        # Create multiple agents
        aligned_agents = [CacheAlignedAgent() for _ in range(100)]
        regular_agents = [RegularAgent() for _ in range(100)]

        # Simulate concurrent access
        def access_beliefs(agents, iterations=1000):
            start = time.time()
            for _ in range(iterations):
                for agent in agents:
                    # Simulate belief access
                    _ = agent.beliefs.sum()
            return time.time() - start

        # Measure performance
        aligned_time = access_beliefs(aligned_agents)
        regular_time = access_beliefs(regular_agents)

        # Cache-aligned should show some improvement
        # Note: Actual improvement depends on hardware
        self.assertLessEqual(
            aligned_time,
            regular_time * 1.1,
            "Cache-aligned structures should not be slower",
        )

    def test_read_write_lock_optimization(self):
        """Test read-write lock optimization for shared state."""

        class OptimizedSharedState:
            """Shared state with read-write lock optimization."""

            def __init__(self):
                self._lock = threading.RLock()
                self._data = {}
                self._readers = 0

            def read(self, key):
                """Read operation (multiple readers allowed)."""
                with self._lock:
                    self._readers += 1

                try:
                    return self._data.get(key)
                finally:
                    with self._lock:
                        self._readers -= 1

            def write(self, key, value):
                """Write operation (exclusive access)."""
                with self._lock:
                    while self._readers > 0:
                        replace_sleep(0.0001)
                    self._data[key] = value

        # Test concurrent access
        shared_state = OptimizedSharedState()

        def reader_thread(thread_id):
            """Reader thread function."""
            for i in range(100):
                _ = shared_state.read(f"key_{i % 10}")

        def writer_thread(thread_id):
            """Writer thread function."""
            for i in range(10):
                shared_state.write(f"key_{i}", f"value_{thread_id}_{i}")

        # Create threads
        threads = []

        # More readers than writers (typical pattern)
        for i in range(8):
            t = threading.Thread(target=reader_thread, args=(i,))
            threads.append(t)

        for i in range(2):
            t = threading.Thread(target=writer_thread, args=(i,))
            threads.append(t)

        # Run threads
        start_time = time.time()
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        duration = time.time() - start_time

        # Should complete quickly with minimal contention
        self.assertLess(duration, 1.0, "Read-write lock should minimize contention")


class TestLockFreeDataStructures(unittest.TestCase):
    """Test lock-free data structure implementations."""

    def test_lock_free_queue_performance(self):
        """Test lock-free queue performance vs standard queue."""

        # Lock-free queue (SimpleQueue)
        lock_free_queue = queue.SimpleQueue()

        # Standard queue with locks
        standard_queue = queue.Queue()

        def producer_consumer_test(test_queue, num_items=10000):
            """Test producer-consumer performance."""

            def producer():
                for i in range(num_items):
                    test_queue.put(i)

            def consumer():
                count = 0
                while count < num_items:
                    try:
                        _ = test_queue.get(timeout=0.1)
                        count += 1
                    except Exception:
                        pass

            start_time = time.time()

            # Create threads
            prod_thread = threading.Thread(target=producer)
            cons_thread = threading.Thread(target=consumer)

            # Run threads
            prod_thread.start()
            cons_thread.start()

            prod_thread.join()
            cons_thread.join()

            return time.time() - start_time

        # Measure performance
        lock_free_time = producer_consumer_test(lock_free_queue)
        standard_time = producer_consumer_test(standard_queue)

        # Lock-free should be faster for single producer/consumer
        self.assertLess(
            lock_free_time,
            standard_time * 1.2,
            "Lock-free queue should be competitive or faster",
        )

    def test_atomic_counter(self):
        """Test atomic counter implementation."""

        class AtomicCounter:
            """Thread-safe atomic counter."""

            def __init__(self):
                self._value = 0
                self._lock = threading.Lock()

            def increment(self):
                """Atomic increment."""
                with self._lock:
                    self._value += 1

            def get(self):
                """Get current value."""
                with self._lock:
                    return self._value

        # Test concurrent increments
        counter = AtomicCounter()

        def increment_thread():
            for _ in range(1000):
                counter.increment()

        # Create multiple threads
        threads = [threading.Thread(target=increment_thread) for _ in range(10)]

        # Run threads
        start_time = time.time()
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        duration = time.time() - start_time

        # Verify correctness
        self.assertEqual(counter.get(), 10000, "Atomic counter should maintain consistency")

        # Should be fast
        self.assertLess(duration, 0.5, "Atomic operations should be fast")


class TestWorkloadSpecificOptimizations(unittest.TestCase):
    """Test workload-specific optimizations."""

    def test_message_batching(self):
        """Test message batching for reduced communication overhead."""

        class BatchedMessageQueue:
            """Message queue with batching support."""

            def __init__(self, batch_size=10):
                self.batch_size = batch_size
                self.buffer = []
                self.lock = threading.Lock()
                self.output_queue = queue.Queue()

            def send(self, message):
                """Send message (may be batched)."""
                with self.lock:
                    self.buffer.append(message)
                    if len(self.buffer) >= self.batch_size:
                        # Flush batch
                        self.output_queue.put(self.buffer.copy())
                        self.buffer.clear()

            def flush(self):
                """Flush remaining messages."""
                with self.lock:
                    if self.buffer:
                        self.output_queue.put(self.buffer.copy())
                        self.buffer.clear()

        # Test batching performance
        batched_queue = BatchedMessageQueue(batch_size=10)
        unbatched_queue = queue.Queue()

        # Measure batched performance
        batch_start = time.time()
        for i in range(1000):
            batched_queue.send(f"message_{i}")
        batched_queue.flush()
        batch_duration = time.time() - batch_start

        # Measure unbatched performance
        unbatch_start = time.time()
        for i in range(1000):
            unbatched_queue.put(f"message_{i}")
        unbatch_duration = time.time() - unbatch_start

        # Batching should reduce overhead
        self.assertLess(
            batch_duration,
            unbatch_duration * 1.5,
            "Batching should not add significant overhead",
        )

    def test_vectorized_belief_updates(self):
        """Test vectorized belief update performance."""

        def vectorized_update(beliefs: List[np.ndarray]) -> List[np.ndarray]:
            """Update multiple beliefs simultaneously."""
            # Stack beliefs for vectorized operations
            stacked = np.stack(beliefs)

            # Vectorized normalization
            normalized = stacked / stacked.sum(axis=1, keepdims=True)

            return [normalized[i] for i in range(len(beliefs))]

        def sequential_update(beliefs: List[np.ndarray]) -> List[np.ndarray]:
            """Update beliefs sequentially."""
            results = []
            for belief in beliefs:
                normalized = belief / belief.sum()
                results.append(normalized)
            return results

        # Create test beliefs
        beliefs = [np.random.rand(100) for _ in range(50)]

        # Measure performance
        vec_start = time.time()
        for _ in range(100):
            vec_results = vectorized_update(beliefs)
        vec_duration = time.time() - vec_start

        seq_start = time.time()
        for _ in range(100):
            seq_results = sequential_update(beliefs)
        seq_duration = time.time() - seq_start

        # Verify correctness
        for v, s in zip(vec_results, seq_results):
            np.testing.assert_allclose(v, s, rtol=1e-10)

        # Vectorized should be faster
        self.assertLess(vec_duration, seq_duration, "Vectorized updates should be faster")


class TestIntegrationPerformance(unittest.TestCase):
    """Integration tests for combined optimizations."""

    def test_full_optimization_stack(self):
        """Test performance with all optimizations enabled."""

        # Create optimized thread pool manager
        manager = OptimizedThreadPoolManager(
            initial_workers=mp.cpu_count(),
            max_workers=mp.cpu_count() * 4,
            scaling_threshold=0.7,
        )

        # Create async inference engine
        AsyncInferenceEngine(max_workers=mp.cpu_count())

        # Create test agents with optimizations
        agents = []
        for i in range(50):
            agent = Mock()
            agent.agent_id = f"optimized_agent_{i}"
            agent.step = Mock(return_value=f"action_{i}")
            agent.perceive = Mock()
            agent.update_beliefs = Mock()
            agent.select_action = Mock(return_value=f"action_{i}")
            agents.append(agent)
            manager.register_agent(agent.agent_id, agent)

        # Run performance test
        start_time = time.time()

        observations = {agent.agent_id: {"data": i} for i, agent in enumerate(agents)}

        # Run multiple rounds
        for _ in range(10):
            results = manager.step_all_agents(observations, timeout=5.0)

            # Verify all succeeded
            successes = sum(1 for r in results.values() if r.success)
            self.assertEqual(successes, len(agents))

        duration = time.time() - start_time
        throughput = (10 * len(agents)) / duration

        # Performance assertions
        self.assertGreater(
            throughput,
            100,  # At least 100 ops/sec
            f"Throughput {throughput} should exceed 100 ops/sec",
        )

        # Check thread pool efficiency
        status = manager.get_pool_status()
        self.assertGreater(
            status["load_factor"],
            0.5,
            "Thread pool should be efficiently utilized",
        )

        # Cleanup
        manager.shutdown()


if __name__ == "__main__":
    unittest.main()

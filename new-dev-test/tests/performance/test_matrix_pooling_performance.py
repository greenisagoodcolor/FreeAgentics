#!/usr/bin/env python3
"""Performance tests for matrix pooling implementation.

Based on Task 5.4: Create matrix operation memory pooling
"""

import gc
import time
import unittest

import numpy as np
import psutil

from agents.memory_optimization.matrix_pooling import get_global_pool, pooled_dot, pooled_einsum
from tests.performance.performance_utils import replace_sleep


class MatrixPoolingPerformanceTest(unittest.TestCase):
    """Performance benchmarks for matrix pooling."""

    def setUp(self):
        """Set up test environment."""
        self.pool = get_global_pool()
        self.pool.clear_all()
        gc.collect()

        # Track initial memory
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def tearDown(self):
        """Clean up after tests."""
        self.pool.clear_all()
        gc.collect()

    def measure_operation_time(self, operation, iterations: int = 100) -> float:
        """Measure average time for an operation."""
        times = []

        # Warmup
        for _ in range(10):
            operation()

        # Measure
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return np.mean(times)

    def test_allocation_performance(self):
        """Test matrix allocation performance."""
        print("\n=== Matrix Allocation Performance ===")

        sizes = [(10, 10), (100, 100), (500, 500), (1000, 1000)]
        results = {}

        for shape in sizes:
            # Numpy allocation
            def numpy_alloc():
                return np.zeros(shape, dtype=np.float32)

            numpy_time = self.measure_operation_time(numpy_alloc)

            # Pooled allocation
            def pooled_alloc():
                with self.pool.allocate_matrix(shape, np.float32):
                    pass

            pooled_time = self.measure_operation_time(pooled_alloc)

            speedup = numpy_time / pooled_time if pooled_time > 0 else 0

            results[shape] = {
                "numpy_ms": numpy_time * 1000,
                "pooled_ms": pooled_time * 1000,
                "speedup": speedup,
            }

            print(f"\nShape {shape}:")
            print(f"  Numpy: {numpy_time * 1000:.3f} ms")
            print(f"  Pooled: {pooled_time * 1000:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")

        # After warmup, pooled should be significantly faster
        for shape in sizes:
            if shape[0] >= 100:  # For non-tiny matrices
                self.assertGreater(
                    results[shape]["speedup"],
                    1.5,
                    f"Pooling not efficient for shape {shape}",
                )

    def test_matrix_operation_performance(self):
        """Test performance of pooled matrix operations."""
        print("\n=== Matrix Operation Performance ===")

        sizes = [100, 500, 1000]

        for size in sizes:
            print(f"\nSize: {size}x{size}")

            # Create test matrices
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)

            # Test dot product
            numpy_dot_time = self.measure_operation_time(lambda: np.dot(a, b), iterations=20)

            pooled_dot_time = self.measure_operation_time(lambda: pooled_dot(a, b), iterations=20)

            print("  Dot product:")
            print(f"    Numpy: {numpy_dot_time * 1000:.2f} ms")
            print(f"    Pooled: {pooled_dot_time * 1000:.2f} ms")
            print(f"    Overhead: {(pooled_dot_time / numpy_dot_time - 1) * 100:.1f}%")

            # Test einsum
            numpy_einsum_time = self.measure_operation_time(
                lambda: np.einsum("ij,jk->ik", a, b), iterations=20
            )

            pooled_einsum_time = self.measure_operation_time(
                lambda: pooled_einsum("ij,jk->ik", a, b), iterations=20
            )

            print("  Einsum:")
            print(f"    Numpy: {numpy_einsum_time * 1000:.2f} ms")
            print(f"    Pooled: {pooled_einsum_time * 1000:.2f} ms")
            print(f"    Overhead: {(pooled_einsum_time / numpy_einsum_time - 1) * 100:.1f}%")

            # Overhead should be reasonable
            self.assertLess(
                pooled_dot_time / numpy_dot_time,
                1.3,  # Max 30% overhead
                f"Too much overhead for size {size}",
            )

    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        print("\n=== Memory Efficiency Test ===")

        # Parameters
        matrix_size = 1000
        iterations = 50

        # Measure memory without pooling
        gc.collect()
        start_mem = self.process.memory_info().rss / 1024 / 1024

        no_pool_results = []
        for _ in range(iterations):
            a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            c = np.dot(a, b)
            no_pool_results.append(c[0, 0])  # Keep reference

        gc.collect()
        no_pool_mem = self.process.memory_info().rss / 1024 / 1024 - start_mem

        # Clear
        no_pool_results.clear()
        gc.collect()
        replace_sleep(0.1)

        # Measure memory with pooling
        gc.collect()
        start_mem = self.process.memory_info().rss / 1024 / 1024

        pool_results = []
        for _ in range(iterations):
            a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            c = pooled_dot(a, b)
            pool_results.append(c[0, 0])  # Keep reference

        gc.collect()
        pool_mem = self.process.memory_info().rss / 1024 / 1024 - start_mem

        print(
            f"\nMemory usage for {iterations} iterations of {matrix_size}x{matrix_size} dot products:"
        )
        print(f"  Without pooling: {no_pool_mem:.1f} MB")
        print(f"  With pooling: {pool_mem:.1f} MB")
        print(
            f"  Savings: {no_pool_mem - pool_mem:.1f} MB ({(1 - pool_mem / no_pool_mem) * 100:.1f}%)"
        )

        # Get pool statistics
        stats = self.pool.get_statistics()
        total_pool_mem = stats["global"]["total_memory_mb"]

        print("\nPool statistics:")
        print(f"  Total pool memory: {total_pool_mem:.1f} MB")
        print(f"  Number of pools: {stats['global']['total_pools']}")

        # Pooling should use less memory
        self.assertLess(pool_mem, no_pool_mem * 1.1)  # Allow 10% variance

    def test_concurrent_access_performance(self):
        """Test performance with concurrent access patterns."""
        import threading

        print("\n=== Concurrent Access Performance ===")

        num_threads = 4
        operations_per_thread = 25
        matrix_size = 500

        results = {"times": [], "errors": []}

        def worker(thread_id):
            """Worker thread that performs matrix operations."""
            try:
                thread_times = []

                for _ in range(operations_per_thread):
                    a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
                    b = np.random.rand(matrix_size, matrix_size).astype(np.float32)

                    start = time.perf_counter()
                    pooled_dot(a, b)
                    elapsed = time.perf_counter() - start

                    thread_times.append(elapsed)

                results["times"].extend(thread_times)

            except Exception as e:
                results["errors"].append(e)

        # Run concurrent operations
        start_time = time.time()
        threads = []

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        total_time = time.time() - start_time

        # Check results
        self.assertEqual(len(results["errors"]), 0, "Errors in concurrent access")

        total_ops = num_threads * operations_per_thread
        avg_time = np.mean(results["times"])

        print(f"\nConcurrent operations ({num_threads} threads):")
        print(f"  Total operations: {total_ops}")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Throughput: {total_ops / total_time:.1f} ops/s")
        print(f"  Average op time: {avg_time * 1000:.2f} ms")

        # Check pool efficiency
        stats = self.pool.get_statistics()
        print("\nPool performance:")

        for pool_key, pool_stats in stats["pools"].items():
            if pool_stats["stats"]["total_requests"] > 0:
                print(f"  {pool_key}:")
                print(f"    Hit rate: {pool_stats['hit_rate']:.1%}")
                print(f"    Total requests: {pool_stats['stats']['total_requests']}")

                # Should have good hit rate with concurrent access
                self.assertGreater(pool_stats["hit_rate"], 0.7)

    def test_pymdp_style_operations(self):
        """Test performance with PyMDP-style operations."""
        print("\n=== PyMDP-Style Operations Performance ===")

        # Typical PyMDP dimensions
        num_states = 25
        num_obs = 5
        num_actions = 4
        num_agents = 10
        timesteps = 100

        # Create model matrices
        A = np.random.rand(num_obs, num_states).astype(np.float32)
        A = A / A.sum(axis=0, keepdims=True)

        B = np.random.rand(num_states, num_states, num_actions).astype(np.float32)
        B = B / B.sum(axis=0, keepdims=True)

        # Simulate multiple agents
        print(f"\nSimulating {num_agents} agents for {timesteps} timesteps")

        # Without pooling
        start = time.time()
        beliefs_no_pool = [np.ones(num_states) / num_states for _ in range(num_agents)]

        for t in range(timesteps):
            for i in range(num_agents):
                # Observation update
                obs = np.random.randint(0, num_obs)
                likelihood = A[obs, :]
                beliefs_no_pool[i] = likelihood * beliefs_no_pool[i]
                beliefs_no_pool[i] /= beliefs_no_pool[i].sum()

                # Action and transition
                action = np.random.randint(0, num_actions)
                beliefs_no_pool[i] = np.dot(B[:, :, action], beliefs_no_pool[i])

        no_pool_time = time.time() - start

        # With pooling
        start = time.time()
        beliefs_pool = [np.ones(num_states) / num_states for _ in range(num_agents)]

        for t in range(timesteps):
            for i in range(num_agents):
                # Observation update with pooled array
                obs = np.random.randint(0, num_obs)
                with self.pool.allocate_matrix((num_states,), np.float32) as temp:
                    likelihood = A[obs, :]
                    np.multiply(likelihood, beliefs_pool[i], out=temp)
                    temp /= temp.sum()
                    beliefs_pool[i] = temp.copy()

                # Action and transition with pooled operation
                action = np.random.randint(0, num_actions)
                beliefs_pool[i] = pooled_dot(B[:, :, action], beliefs_pool[i])

        pool_time = time.time() - start

        print("\nResults:")
        print(f"  Without pooling: {no_pool_time:.2f} s")
        print(f"  With pooling: {pool_time:.2f} s")
        print(f"  Speedup: {no_pool_time / pool_time:.2f}x")

        # Check accuracy
        for i in range(num_agents):
            np.testing.assert_allclose(
                beliefs_no_pool[i],
                beliefs_pool[i],
                rtol=1e-5,
                err_msg=f"Belief mismatch for agent {i}",
            )

        # Get final statistics
        stats = self.pool.get_statistics()
        print("\nFinal pool statistics:")
        print(f"  Total operations: {stats['global']['operation_counts']}")

        # Pooling should provide some benefit
        self.assertLess(pool_time, no_pool_time * 1.2)  # Within 20%


def run_performance_suite():
    """Run the complete performance test suite."""
    print("Matrix Pooling Performance Test Suite")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(MatrixPoolingPerformanceTest)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_performance_suite()
    exit(0 if success else 1)

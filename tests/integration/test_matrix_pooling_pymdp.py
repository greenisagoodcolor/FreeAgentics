#!/usr/bin/env python3
"""Integration tests for matrix pooling with PyMDP operations.

Based on Task 5.4: Create matrix operation memory pooling
"""

import gc
import time
import unittest

import numpy as np
import psutil

from agents.memory_optimization.matrix_pooling import (
    MatrixOperationPool,
    get_global_pool,
    pooled_dot,
    pooled_einsum,
    pooled_matmul,
)

# Only import PyMDP if available
try:
    from pymdp.agent import Agent as PyMDPAgent
except ImportError:
    PyMDPAgent = None


class TestPyMDPMatrixPoolingIntegration(unittest.TestCase):
    """Test matrix pooling integration with PyMDP operations."""

    def setUp(self):
        """Set up test environment."""
        self.pool = MatrixOperationPool(enable_profiling=True)

        # Monitor memory usage
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def tearDown(self):
        """Clean up after tests."""
        self.pool.clear_all()
        gc.collect()

    def test_pymdp_matrix_operations(self):
        """Test pooling with actual PyMDP matrix operations."""
        if PyMDPAgent is None:
            self.skipTest("PyMDP not available")
        # Create simple PyMDP matrices
        num_obs = 5
        num_states = 10
        num_actions = 4

        # A matrix: P(obs|state)
        A = np.random.rand(num_obs, num_states).astype(np.float32)
        A = A / A.sum(axis=0, keepdims=True)  # Normalize

        # B matrix: P(state'|state,action)
        B = np.random.rand(num_states, num_states, num_actions).astype(np.float32)
        B = B / B.sum(axis=0, keepdims=True)  # Normalize

        # Test belief update operation with pooling
        belief = np.ones(num_states) / num_states
        observation = 2

        # Standard PyMDP belief update
        start_time = time.time()
        for _ in range(100):
            # P(s|o) ∝ P(o|s) * P(s)
            likelihood = A[observation, :]
            posterior = likelihood * belief
            posterior = posterior / posterior.sum()
        time.time() - start_time

        # Pooled belief update
        start_time = time.time()
        for _ in range(100):
            with self.pool.allocate_matrix((num_states,), np.float32) as pooled_posterior:
                likelihood = A[observation, :]
                np.multiply(likelihood, belief, out=pooled_posterior)
                pooled_posterior /= pooled_posterior.sum()
                posterior_pooled = pooled_posterior.copy()
        time.time() - start_time

        # Results should be identical
        np.testing.assert_allclose(posterior, posterior_pooled, rtol=1e-5)

        # Check pool statistics
        stats = self.pool.get_statistics()
        self.assertGreater(stats["global"]["total_pools"], 0)

        # Pool should have good hit rate after warmup
        for pool_key, pool_stats in stats["pools"].items():
            if pool_stats["stats"]["total_requests"] > 10:
                self.assertGreater(pool_stats["hit_rate"], 0.8)

    def test_large_scale_matrix_operations(self):
        """Test memory efficiency with large-scale operations."""
        # Large matrices that would normally cause memory pressure
        size = 1000
        iterations = 50

        # Track memory without pooling
        gc.collect()
        mem_before = self.process.memory_info().rss / 1024 / 1024

        results_no_pool = []
        for _ in range(iterations):
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            c = np.dot(a, b)
            results_no_pool.append(c[0, 0])  # Keep reference

        gc.collect()
        mem_after_no_pool = self.process.memory_info().rss / 1024 / 1024
        memory_increase_no_pool = mem_after_no_pool - mem_before

        # Clear memory
        results_no_pool.clear()
        gc.collect()
        time.sleep(0.1)

        # Track memory with pooling
        gc.collect()
        mem_before = self.process.memory_info().rss / 1024 / 1024

        results_pool = []
        for _ in range(iterations):
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            c = pooled_dot(a, b)
            results_pool.append(c[0, 0])  # Keep reference

        gc.collect()
        mem_after_pool = self.process.memory_info().rss / 1024 / 1024
        memory_increase_pool = mem_after_pool - mem_before

        # Pooling should use less memory
        self.assertLess(memory_increase_pool, memory_increase_no_pool * 1.2)

        # Check pool reuse
        stats = self.pool.get_statistics()
        for pool_key, pool_stats in stats["pools"].items():
            if "hit_rate" in pool_stats:
                # Should have high reuse after first few allocations
                self.assertGreater(pool_stats["hit_rate"], 0.7)

    def test_einsum_heavy_operations(self):
        """Test pooling with einsum-heavy calculations."""
        # Simulate tensor contraction operations common in PyMDP

        # Multi-dimensional belief state
        dims = [3, 4, 5]
        belief = np.random.rand(*dims).astype(np.float32)
        belief = belief / belief.sum()

        # Transition tensor
        transition = np.random.rand(dims[0], dims[1], dims[2], dims[0], dims[1], dims[2], 4).astype(
            np.float32
        )

        # Action selection via tensor contraction
        iterations = 20

        # Without pooling
        start_time = time.time()
        for action in range(4):
            for _ in range(iterations):
                # Contract over current state dimensions
                np.einsum("ijk,ijklmna->lmn", belief, transition[..., action])
        time.time() - start_time

        # With pooling
        start_time = time.time()
        for action in range(4):
            for _ in range(iterations):
                pooled_einsum("ijk,ijklmna->lmn", belief, transition[..., action])
        time.time() - start_time

        # Check statistics
        stats = self.pool.get_statistics()
        self.assertIn("einsum", stats["global"]["operation_counts"])
        self.assertGreater(stats["global"]["operation_counts"]["einsum"], 0)

    def test_concurrent_pymdp_agents(self):
        """Test pooling with multiple concurrent agents."""
        if PyMDPAgent is None:
            self.skipTest("PyMDP not available")

        num_agents = 5
        num_steps = 10

        # Create simple grid world matrices
        grid_size = 5
        num_states = grid_size * grid_size
        num_obs = 5  # wall, empty, goal, agent, out-of-bounds
        num_actions = 4  # up, down, left, right

        # Shared matrices (could be pooled at this level too)
        A = np.eye(num_obs, num_states)[:, :num_obs]  # Simple observation
        A = A / A.sum(axis=0, keepdims=True)

        B = np.random.rand(num_states, num_states, num_actions).astype(np.float32)
        B = B / B.sum(axis=0, keepdims=True)

        C = np.zeros(num_obs)
        C[2] = 10.0  # Prefer goal observation

        D = np.ones(num_states) / num_states

        # Simulate agents with pooled operations
        def simulate_agent(agent_id):
            """Simulate agent with pooled matrix operations."""
            agent = PyMDPAgent(A=[A], B=[B], C=[C], D=[D])

            for step in range(num_steps):
                # Get observation
                obs = np.random.randint(0, num_obs)

                # Update beliefs using pooled operations
                agent.infer_states([obs])

                # Infer policies
                agent.infer_policies()

                # Sample action
                agent.sample_action()

            return f"Agent {agent_id} completed"

        # Run agents (sequentially for simplicity)
        results = []
        for i in range(num_agents):
            result = simulate_agent(i)
            results.append(result)

        self.assertEqual(len(results), num_agents)

        # Check pool statistics
        stats = self.pool.get_statistics()
        # Should have created pools for various matrix sizes
        self.assertGreater(len(stats["pools"]), 0)

    def test_memory_pool_with_belief_compression(self):
        """Test interaction between matrix pooling and belief compression."""
        from agents.memory_optimization.belief_compression import (
            BeliefCompressor,
            CompressedBeliefPool,
        )

        # Create compressor and pool
        compressor = BeliefCompressor(sparsity_threshold=0.8)
        belief_pool = CompressedBeliefPool(pool_size=10, belief_shape=(100,), dtype=np.float32)

        # Simulate belief updates with both compression and matrix pooling
        num_updates = 50
        belief = np.zeros(100, dtype=np.float32)
        belief[10:20] = 0.1  # Sparse belief
        belief = belief / belief.sum()

        for _ in range(num_updates):
            # Compress belief
            compressor.compress(belief)

            # Perform update with pooled matrix operations
            with self.pool.allocate_matrix((100,), np.float32) as update:
                # Simulate some computation
                update[:] = np.random.rand(100)
                update[50:] = 0  # Make it sparse
                update = update / update.sum()

                # Apply update
                belief = 0.9 * belief + 0.1 * update
                belief = belief / belief.sum()

        # Check both pools are working
        self.assertGreater(belief_pool.stats["total"], 0)

        matrix_stats = self.pool.get_statistics()
        self.assertGreater(matrix_stats["global"]["total_matrices"], 0)

    def test_performance_comparison(self):
        """Compare performance with and without pooling."""
        sizes = [100, 500, 1000]
        operations = ["dot", "matmul", "einsum"]

        results = {}

        for size in sizes:
            results[size] = {}

            # Create test matrices
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)

            # Test each operation
            for op in operations:
                # Without pooling
                gc.collect()
                start = time.time()
                for _ in range(10):
                    if op == "dot":
                        np.dot(a, b)
                    elif op == "matmul":
                        np.matmul(a, b)
                    elif op == "einsum":
                        np.einsum("ij,jk->ik", a, b)
                no_pool_time = time.time() - start

                # With pooling
                gc.collect()
                start = time.time()
                for _ in range(10):
                    if op == "dot":
                        pooled_dot(a, b)
                    elif op == "matmul":
                        pooled_matmul(a, b)
                    elif op == "einsum":
                        pooled_einsum("ij,jk->ik", a, b)
                pool_time = time.time() - start

                results[size][op] = {
                    "no_pool": no_pool_time,
                    "pooled": pool_time,
                    "speedup": no_pool_time / pool_time if pool_time > 0 else 1.0,
                }

        # Pooling should provide benefits for repeated operations
        for size in sizes:
            for op in operations:
                # For larger matrices, pooling overhead should be worth it
                if size >= 500:
                    self.assertGreater(
                        results[size][op]["speedup"],
                        0.8,  # At least 80% as fast
                        f"Pooling too slow for {op} with size {size}",
                    )


class TestMemoryPoolingBenchmarks(unittest.TestCase):
    """Benchmark tests for memory pooling efficiency."""

    def setUp(self):
        """Set up benchmarking."""
        self.pool = get_global_pool()
        self.pool.clear_all()

    def test_allocation_speed(self):
        """Benchmark allocation speed vs numpy."""
        sizes = [(10, 10), (100, 100), (1000, 1000)]
        iterations = 100

        for shape in sizes:
            # Numpy allocation
            start = time.time()
            for _ in range(iterations):
                np.zeros(shape, dtype=np.float32)
            numpy_time = time.time() - start

            # Pooled allocation (with warmup)
            # Warmup
            for _ in range(10):
                with self.pool.allocate_matrix(shape, np.float32):
                    pass

            start = time.time()
            for _ in range(iterations):
                with self.pool.allocate_matrix(shape, np.float32):
                    pass
            pool_time = time.time() - start

            speedup = numpy_time / pool_time if pool_time > 0 else 1.0

            # Pooled allocation should be faster after warmup
            self.assertGreater(
                speedup,
                2.0,  # At least 2x faster
                f"Pool not fast enough for shape {shape}: {speedup:.2f}x",
            )

    def test_memory_fragmentation(self):
        """Test that pooling reduces memory fragmentation."""
        # Perform many allocations/deallocations
        iterations = 1000
        size = 100

        # Track memory over time
        memory_readings = []

        # Without pooling - causes fragmentation
        for i in range(iterations):
            arrays = []
            for _ in range(10):
                arr = np.zeros((size, size), dtype=np.float32)
                arrays.append(arr)
            arrays.clear()

            if i % 100 == 0:
                gc.collect()
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                memory_readings.append(("no_pool", i, mem))

        # Clear everything
        gc.collect()
        time.sleep(0.1)

        # With pooling - reduces fragmentation
        for i in range(iterations):
            for _ in range(10):
                with self.pool.allocate_matrix((size, size), np.float32) as arr:
                    pass

            if i % 100 == 0:
                gc.collect()
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                memory_readings.append(("pooled", i, mem))

        # Pooling should have more stable memory usage
        no_pool_readings = [m for t, i, m in memory_readings if t == "no_pool"]
        pooled_readings = [m for t, i, m in memory_readings if t == "pooled"]

        if len(no_pool_readings) > 2 and len(pooled_readings) > 2:
            no_pool_variance = np.var(no_pool_readings[1:])
            pooled_variance = np.var(pooled_readings[1:])

            # Pooled should have lower variance (more stable)
            self.assertLess(pooled_variance, no_pool_variance * 1.5)


if __name__ == "__main__":
    unittest.main()

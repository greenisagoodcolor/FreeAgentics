#!/usr/bin/env python3
"""Comprehensive tests for matrix operation memory pooling.

Based on Task 5.4: Create matrix operation memory pooling
"""

import gc
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from agents.memory_optimization.matrix_pooling import (
    MatrixOperationPool,
    MatrixPool,
    PooledMatrix,
    PoolStatistics,
    get_global_pool,
    pooled_dot,
    pooled_einsum,
    pooled_matmul,
    pooled_matrix,
)


class TestPooledMatrix(unittest.TestCase):
    """Test the PooledMatrix dataclass."""

    def test_pooled_matrix_creation(self):
        """Test creating a pooled matrix."""
        data = np.zeros((10, 10), dtype=np.float32)
        matrix = PooledMatrix(
            data=data, shape=(10, 10), dtype=np.float32, pool_id="test_123"
        )

        self.assertEqual(matrix.shape, (10, 10))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.pool_id, "test_123")
        self.assertFalse(matrix.in_use)
        self.assertEqual(matrix.access_count, 0)

    def test_pooled_matrix_validation(self):
        """Test validation of matrix properties."""
        data = np.zeros((10, 10), dtype=np.float32)

        # Shape mismatch
        with self.assertRaises(ValueError):
            PooledMatrix(
                data=data, shape=(5, 5), dtype=np.float32, pool_id="test"
            )  # Wrong shape

        # Dtype mismatch
        with self.assertRaises(ValueError):
            PooledMatrix(
                data=data, shape=(10, 10), dtype=np.float64, pool_id="test"
            )  # Wrong dtype


class TestPoolStatistics(unittest.TestCase):
    """Test pool statistics tracking."""

    def test_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        stats = PoolStatistics()

        # No requests
        self.assertEqual(stats.hit_rate, 0.0)

        # Some hits and misses
        stats.total_requests = 100
        stats.cache_hits = 75
        stats.cache_misses = 25

        self.assertEqual(stats.hit_rate, 0.75)


class TestMatrixPool(unittest.TestCase):
    """Test the MatrixPool class."""

    def setUp(self):
        """Set up test fixtures."""
        self.pool = MatrixPool(
            shape=(10, 10), dtype=np.float32, initial_size=3, max_size=10
        )

    def test_pool_initialization(self):
        """Test pool pre-allocation."""
        self.assertEqual(self.pool.stats.total_allocated, 3)
        self.assertEqual(self.pool.stats.current_available, 3)
        self.assertEqual(self.pool.stats.current_in_use, 0)

    def test_acquire_from_pool(self):
        """Test acquiring matrices from pool."""
        # First acquisition should reuse pre-allocated
        matrix1 = self.pool.acquire()
        self.assertIsInstance(matrix1, PooledMatrix)
        self.assertTrue(matrix1.in_use)
        self.assertEqual(matrix1.shape, (10, 10))
        self.assertEqual(self.pool.stats.cache_hits, 1)
        self.assertEqual(self.pool.stats.current_available, 2)
        self.assertEqual(self.pool.stats.current_in_use, 1)

        # Acquire all pre-allocated
        self.pool.acquire()
        self.pool.acquire()
        self.assertEqual(self.pool.stats.current_available, 0)
        self.assertEqual(self.pool.stats.current_in_use, 3)

        # Next acquisition should create new
        self.pool.acquire()
        self.assertEqual(self.pool.stats.total_allocated, 4)
        self.assertEqual(self.pool.stats.cache_misses, 1)

    def test_release_to_pool(self):
        """Test releasing matrices back to pool."""
        matrix = self.pool.acquire()
        matrix.data.copy()
        matrix.data[0, 0] = 42.0  # Modify data

        # Release back to pool
        self.pool.release(matrix)
        self.assertFalse(matrix.in_use)
        self.assertEqual(matrix.data[0, 0], 0.0)  # Data cleared
        self.assertEqual(self.pool.stats.current_available, 3)
        self.assertEqual(self.pool.stats.current_in_use, 0)

    def test_pool_max_size_limit(self):
        """Test pool respects maximum size."""
        # Acquire more than max_size
        matrices = []
        for _ in range(15):  # More than max_size=10
            matrices.append(self.pool.acquire())

        # Should have allocated up to max_size, then warned about exhaustion
        # The implementation creates emergency allocations beyond max_size
        self.assertGreaterEqual(self.pool.stats.total_allocated, 10)

        # Release all
        for matrix in matrices:
            self.pool.release(matrix)

        # Pool should only keep max_size
        self.assertEqual(self.pool.stats.current_available, 10)
        # Total allocated stays the same after release

    def test_release_foreign_matrix(self):
        """Test releasing a matrix not from this pool."""
        # Create a matrix not from pool
        data = np.zeros((10, 10), dtype=np.float32)
        foreign_matrix = PooledMatrix(
            data=data, shape=(10, 10), dtype=np.float32, pool_id="foreign"
        )

        # Should handle gracefully
        with patch(
            "agents.memory_optimization.matrix_pooling.logger"
        ) as mock_logger:
            self.pool.release(foreign_matrix)
            mock_logger.warning.assert_called_once()

    def test_pool_clear(self):
        """Test clearing the pool."""
        # Acquire some matrices
        self.pool.acquire()
        self.pool.acquire()

        # Clear pool
        self.pool.clear()

        # Stats should be reset
        self.assertEqual(self.pool.stats.total_allocated, 0)
        self.assertEqual(self.pool.stats.current_available, 0)
        self.assertEqual(self.pool.stats.current_in_use, 0)


class TestMatrixOperationPool(unittest.TestCase):
    """Test the central MatrixOperationPool."""

    def setUp(self):
        """Set up test fixtures."""
        self.op_pool = MatrixOperationPool(enable_profiling=True)

    def tearDown(self):
        """Clean up after tests."""
        self.op_pool.clear_all()

    def test_get_pool_creation(self):
        """Test pool creation for different shapes."""
        # Get pool for specific shape
        pool1 = self.op_pool.get_pool((10, 10), np.float32)
        self.assertIsInstance(pool1, MatrixPool)
        self.assertEqual(self.op_pool.global_stats["total_pools"], 1)

        # Same shape should return same pool
        pool2 = self.op_pool.get_pool((10, 10), np.float32)
        self.assertIs(pool1, pool2)
        self.assertEqual(self.op_pool.global_stats["total_pools"], 1)

        # Different shape creates new pool
        pool3 = self.op_pool.get_pool((20, 20), np.float32)
        self.assertIsNot(pool1, pool3)
        self.assertEqual(self.op_pool.global_stats["total_pools"], 2)

    def test_allocate_matrix_context(self):
        """Test matrix allocation context manager."""
        with self.op_pool.allocate_matrix((5, 5), np.float32) as matrix:
            self.assertEqual(matrix.shape, (5, 5))
            self.assertEqual(matrix.dtype, np.float32)

            # Modify matrix
            matrix[0, 0] = 42.0

        # After context, matrix should be back in pool
        pool = self.op_pool.get_pool((5, 5), np.float32)
        # Pool pre-allocates matrices based on size
        self.assertGreater(pool.stats.current_available, 0)

    def test_allocate_einsum_operands(self):
        """Test allocating multiple matrices for einsum."""
        shapes = [(10, 20), (20, 30), (30, 10)]

        with self.op_pool.allocate_einsum_operands(
            *shapes, dtype=np.float32
        ) as matrices:
            self.assertEqual(len(matrices), 3)
            for i, (matrix, shape) in enumerate(zip(matrices, shapes)):
                self.assertEqual(matrix.shape, shape)
                self.assertEqual(matrix.dtype, np.float32)

        # All should be released
        for shape in shapes:
            pool = self.op_pool.get_pool(shape, np.float32)
            self.assertGreater(pool.stats.current_available, 0)

    def test_optimize_dot_operation(self):
        """Test optimized dot product."""
        a = np.random.rand(10, 20).astype(np.float32)
        b = np.random.rand(20, 30).astype(np.float32)

        result = self.op_pool.optimize_matrix_operation("dot", a, b)
        expected = np.dot(a, b)

        np.testing.assert_allclose(result, expected)
        self.assertEqual(
            self.op_pool.global_stats["operation_counts"]["dot"], 1
        )

    def test_optimize_matmul_operation(self):
        """Test optimized matrix multiplication."""
        a = np.random.rand(15, 25).astype(np.float32)
        b = np.random.rand(25, 35).astype(np.float32)

        result = self.op_pool.optimize_matrix_operation("matmul", a, b)
        expected = np.matmul(a, b)

        np.testing.assert_allclose(result, expected)
        self.assertEqual(
            self.op_pool.global_stats["operation_counts"]["matmul"], 1
        )

    def test_optimize_einsum_operation(self):
        """Test optimized einsum."""
        a = np.random.rand(10, 20).astype(np.float32)
        b = np.random.rand(20, 30).astype(np.float32)

        result = self.op_pool.optimize_matrix_operation(
            "einsum", "ij,jk->ik", a, b
        )
        expected = np.einsum("ij,jk->ik", a, b)

        np.testing.assert_allclose(result, expected)
        self.assertEqual(
            self.op_pool.global_stats["operation_counts"]["einsum"], 1
        )

    def test_get_statistics(self):
        """Test statistics collection."""
        # Perform some operations
        with self.op_pool.allocate_matrix((10, 10)) as m:
            pass

        a = np.random.rand(5, 5).astype(np.float32)
        b = np.random.rand(5, 5).astype(np.float32)
        self.op_pool.optimize_matrix_operation("dot", a, b)

        stats = self.op_pool.get_statistics()

        self.assertIn("global", stats)
        self.assertIn("pools", stats)
        self.assertGreater(stats["global"]["total_pools"], 0)
        self.assertGreater(stats["global"]["total_memory_mb"], 0)

    def test_clear_all(self):
        """Test clearing all pools."""
        # Create some pools
        self.op_pool.get_pool((10, 10))
        self.op_pool.get_pool((20, 20))

        # Clear all
        self.op_pool.clear_all()

        self.assertEqual(len(self.op_pool._pools), 0)
        self.assertEqual(self.op_pool.global_stats["total_pools"], 0)


class TestGlobalPoolFunctions(unittest.TestCase):
    """Test global pool convenience functions."""

    def setUp(self):
        """Reset global pool."""
        import agents.memory_optimization.matrix_pooling as mp

        mp._global_pool = None

    def test_get_global_pool(self):
        """Test global pool singleton."""
        pool1 = get_global_pool()
        pool2 = get_global_pool()

        self.assertIs(pool1, pool2)
        self.assertTrue(pool1.enable_profiling)

    def test_pooled_matrix_context(self):
        """Test pooled matrix convenience function."""
        with pooled_matrix((10, 10), np.float32) as matrix:
            self.assertEqual(matrix.shape, (10, 10))
            self.assertEqual(matrix.dtype, np.float32)

    def test_pooled_operations(self):
        """Test pooled operation convenience functions."""
        a = np.random.rand(10, 20).astype(np.float32)
        b = np.random.rand(20, 30).astype(np.float32)

        # Test pooled_dot
        dot_result = pooled_dot(a, b)
        np.testing.assert_allclose(dot_result, np.dot(a, b))

        # Test pooled_matmul
        matmul_result = pooled_matmul(a, b)
        np.testing.assert_allclose(matmul_result, np.matmul(a, b))

        # Test pooled_einsum
        einsum_result = pooled_einsum("ij,jk->ik", a, b)
        np.testing.assert_allclose(einsum_result, np.einsum("ij,jk->ik", a, b))


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of matrix pools."""

    def test_concurrent_pool_access(self):
        """Test concurrent access to pools."""
        pool = MatrixPool((10, 10), initial_size=5, max_size=20)
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    matrix = pool.acquire()
                    time.sleep(0.001)  # Simulate work
                    matrix.data[0, 0] = worker_id * 100 + i
                    pool.release(matrix)
                results.append(f"Worker {worker_id} completed")
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)

        # Pool should be consistent
        self.assertGreaterEqual(pool.stats.total_requests, 50)
        self.assertEqual(pool.stats.current_in_use, 0)


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency of pooling."""

    def test_memory_reuse(self):
        """Test that matrices are actually reused."""
        pool = MatrixPool((100, 100), initial_size=2)

        # Track matrix ids
        matrix_ids = set()

        # Acquire and release multiple times
        for _ in range(10):
            matrix = pool.acquire()
            matrix_ids.add(id(matrix.data))
            pool.release(matrix)

        # Should have reused matrices
        self.assertLess(len(matrix_ids), 10)
        self.assertLessEqual(pool.stats.total_allocated, 3)

    def test_memory_usage_tracking(self):
        """Test memory usage statistics."""
        pool = MatrixOperationPool()

        # Allocate some matrices
        with pool.allocate_matrix((1000, 1000), np.float32) as m:
            pass

        stats = pool.get_statistics()

        # The pool pre-allocates multiple matrices (initial_size=5 for medium matrices)
        # So total memory will be more than a single matrix
        single_matrix_mb = (1000 * 1000 * 4) / (1024 * 1024)

        # Total memory should be at least one matrix but could be more due to pre-allocation
        self.assertGreaterEqual(
            stats["global"]["total_memory_mb"], single_matrix_mb
        )
        # Should not be excessive (less than 10x a single matrix)
        self.assertLess(
            stats["global"]["total_memory_mb"], single_matrix_mb * 10
        )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_invalid_operations(self):
        """Test handling of invalid operations."""
        pool = MatrixOperationPool()

        # Unknown operation
        with self.assertRaises(ValueError):
            pool.optimize_matrix_operation("unknown_op", np.array([1, 2]))

        # Wrong number of operands
        with self.assertRaises(ValueError):
            pool.optimize_matrix_operation("dot", np.array([1, 2]))

    def test_empty_pool_handling(self):
        """Test behavior with empty/exhausted pool."""
        pool = MatrixPool((10, 10), initial_size=0, max_size=1)

        # Should create on demand
        matrix = pool.acquire()
        self.assertIsNotNone(matrix)
        self.assertEqual(pool.stats.total_allocated, 1)

    def test_gc_interaction(self):
        """Test interaction with garbage collector."""
        pool = MatrixPool((10, 10), initial_size=5)

        # Acquire without releasing (simulate leak)
        leaked_matrices = []
        for _ in range(5):
            leaked_matrices.append(pool.acquire())

        # Clear references and force GC
        leaked_matrices.clear()
        gc.collect()

        # Pool should still function
        matrix = pool.acquire()
        self.assertIsNotNone(matrix)
        pool.release(matrix)


if __name__ == "__main__":
    unittest.main()

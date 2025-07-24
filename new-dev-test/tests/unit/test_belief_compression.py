#!/usr/bin/env python3
"""Unit tests for belief state compression strategies."""

import numpy as np
import pytest

from agents.memory_optimization.belief_compression import (
    BeliefCompressor,
    CompressedBeliefPool,
    SparseBeliefState,
)


class TestBeliefCompressor:
    """Test cases for belief state compression."""

    def test_sparse_belief_creation(self):
        """Test creation of sparse belief states."""
        # Create a mostly zero belief state
        belief = np.zeros((10, 10))
        belief[2, 3] = 0.5
        belief[5, 7] = 0.3
        belief[8, 1] = 0.2

        compressor = BeliefCompressor()
        sparse_belief = compressor.compress(belief)

        assert isinstance(sparse_belief, SparseBeliefState)
        assert sparse_belief.nnz == 3
        assert sparse_belief.shape == (10, 10)

    def test_sparse_belief_reconstruction(self):
        """Test reconstruction from sparse format."""
        # Create test belief
        belief = np.zeros((10, 10))
        belief[1, 1] = 0.4
        belief[3, 5] = 0.6

        compressor = BeliefCompressor()
        sparse_belief = compressor.compress(belief)
        reconstructed = compressor.decompress(sparse_belief)

        np.testing.assert_array_almost_equal(belief, reconstructed)

    def test_compression_ratio(self):
        """Test memory savings from compression."""
        # Create sparse belief (95% zeros)
        size = 100
        belief = np.zeros((size, size))
        # Add 5% non-zero values
        n_nonzero = int(size * size * 0.05)
        indices = np.random.choice(size * size, n_nonzero, replace=False)
        for idx in indices:
            i, j = idx // size, idx % size
            belief[i, j] = np.random.rand()

        # Normalize
        belief = belief / belief.sum()

        compressor = BeliefCompressor()
        sparse_belief = compressor.compress(belief)

        # Calculate memory usage
        dense_memory = belief.nbytes
        sparse_memory = sparse_belief.memory_usage()

        compression_ratio = dense_memory / sparse_memory
        assert compression_ratio > 10  # Should achieve >10x compression for 95% sparse

    def test_incremental_update(self):
        """Test incremental belief updates."""
        # Initial belief
        belief = np.zeros((10, 10))
        belief[5, 5] = 1.0  # Start with certainty at center

        compressor = BeliefCompressor()
        sparse_belief = compressor.compress(belief)

        # Incremental update: spread uncertainty
        update = np.zeros((10, 10))
        update[4:7, 4:7] = 0.1  # Small update around center

        updated_sparse = compressor.incremental_update(sparse_belief, update)

        # Verify update was applied
        updated_dense = compressor.decompress(updated_sparse)
        assert updated_dense[5, 5] < 1.0  # Center belief should decrease
        assert updated_dense[4, 4] > 0.0  # Neighbors should increase

    def test_belief_pool(self):
        """Test belief state pooling."""
        pool = CompressedBeliefPool(pool_size=10, belief_shape=(10, 10))

        # Get beliefs from pool
        beliefs = []
        for _ in range(5):
            belief = pool.acquire()
            beliefs.append(belief)

        assert len(pool.available) == 5
        assert len(pool.in_use) == 5

        # Return beliefs to pool
        for belief in beliefs:
            pool.release(belief)

        assert len(pool.available) == 10
        assert len(pool.in_use) == 0

    def test_shared_components(self):
        """Test sharing common belief components."""
        # Create similar beliefs
        belief1 = np.zeros((10, 10))
        belief1[5, 5] = 0.8
        belief1[5, 6] = 0.2

        belief2 = np.zeros((10, 10))
        belief2[5, 5] = 0.7
        belief2[5, 6] = 0.3

        compressor = BeliefCompressor()

        # Compress with component sharing
        sparse1, components1 = compressor.compress_with_sharing(belief1)
        sparse2, components2 = compressor.compress_with_sharing(
            belief2, base_components=components1
        )

        # Should share structure but have different values
        assert components1["structure"] is components2["structure"]
        assert not np.array_equal(sparse1.data, sparse2.data)

    def test_adaptive_precision(self):
        """Test adaptive precision for belief values."""
        belief = np.random.rand(10, 10)
        belief = belief / belief.sum()

        compressor = BeliefCompressor()

        # Compress with different precisions
        sparse_f64 = compressor.compress(belief, dtype=np.float64)
        sparse_f32 = compressor.compress(belief, dtype=np.float32)
        sparse_f16 = compressor.compress(belief, dtype=np.float16)

        # Check memory usage decreases
        assert sparse_f32.memory_usage() < sparse_f64.memory_usage()
        assert sparse_f16.memory_usage() < sparse_f32.memory_usage()

        # Check reconstruction accuracy
        recon_f32 = compressor.decompress(sparse_f32)
        error_f32 = np.abs(belief - recon_f32).max()
        assert error_f32 < 1e-6  # float32 should be very accurate

    def test_batch_compression(self):
        """Test batch compression of multiple beliefs."""
        n_agents = 10
        beliefs = []
        for _ in range(n_agents):
            belief = np.zeros((10, 10))
            # Random sparse belief
            for _ in range(3):
                i, j = np.random.randint(0, 10, 2)
                belief[i, j] = np.random.rand()
            belief = belief / belief.sum() if belief.sum() > 0 else belief
            beliefs.append(belief)

        compressor = BeliefCompressor()
        compressed = compressor.compress_batch(beliefs)

        assert len(compressed) == n_agents

        # Verify reconstruction
        for i, sparse_belief in enumerate(compressed):
            reconstructed = compressor.decompress(sparse_belief)
            np.testing.assert_array_almost_equal(beliefs[i], reconstructed)


class TestSparseBeliefState:
    """Test the sparse belief state data structure."""

    def test_initialization(self):
        """Test sparse belief state initialization."""
        data = np.array([0.5, 0.3, 0.2])
        indices = np.array([10, 25, 87])
        shape = (10, 10)

        sparse_belief = SparseBeliefState(data, indices, shape)

        assert sparse_belief.nnz == 3
        assert sparse_belief.shape == shape
        assert np.array_equal(sparse_belief.data, data)

    def test_memory_calculation(self):
        """Test memory usage calculation."""
        # Create sparse belief with 10 non-zero elements
        data = np.random.rand(10).astype(np.float32)
        indices = np.arange(10, dtype=np.int32)
        shape = (100, 100)

        sparse_belief = SparseBeliefState(data, indices, shape)

        # Expected memory: 10 * 4 bytes (float32) + 10 * 4 bytes (int32) + overhead
        expected_memory = 10 * 4 + 10 * 4
        assert sparse_belief.memory_usage() >= expected_memory

    def test_to_dense(self):
        """Test conversion to dense format."""
        data = np.array([0.5, 0.5])
        indices = np.array([0, 99])  # First and last elements
        shape = (10, 10)

        sparse_belief = SparseBeliefState(data, indices, shape)
        dense = sparse_belief.to_dense()

        assert dense.shape == shape
        assert dense[0, 0] == 0.5
        assert dense[9, 9] == 0.5
        assert dense.sum() == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

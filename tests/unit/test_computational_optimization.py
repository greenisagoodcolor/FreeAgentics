"""
Module for FreeAgentics Active Inference implementation.
"""

import time
from unittest.mock import Mock

import pytest
import torch

from inference.engine.computational_optimization import (
    BatchProcessor,
    CachedInference,
    ComputationalOptimizer,
    GPUOptimizer,
    OptimizationConfig,
    ParallelInference,
    SparseOperations,
)


class TestOptimizationConfig:
    def test_default_config(self) -> None:
        ."""Test default optimization configuration."""
        config = OptimizationConfig()
        assert config.use_sparse_operations is True
        assert config.use_parallel_processing is True
        assert config.use_gpu is True
        assert config.batch_size == 32
        assert config.cache_size == 1000

    def test_custom_config(self) -> None:
        ."""Test custom optimization configuration."""
        config = OptimizationConfig(
            use_sparse_operations=False, num_threads=8, batch_size=64, use_gpu=False
        )
        assert config.use_sparse_operations is False
        assert config.num_threads == 8
        assert config.batch_size == 64
        assert config.use_gpu is False


class TestSparseOperations:
    def setup_method(self) -> None:
        .."""Setup for tests.."""
        self.config = OptimizationConfig(use_gpu=False)
        self.sparse_ops = SparseOperations(self.config)

    def test_sparsify_tensor(self) -> None:
        """Test tensor sparsification"""
        tensor = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.001, 0.0, 3.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        sparse = self.sparse_ops.sparsify_tensor(tensor)
        assert sparse.is_sparse
        assert sparse.shape == tensor.shape
        dense_result = sparse.to_dense()
        assert dense_result[0, 0] == 1.0
        assert dense_result[1, 2] == 2.0
        assert dense_result[2, 3] == 3.0
        assert dense_result[2, 1] == 0.0

    def test_sparse_matmul(self) -> None:
        ."""Test sparse-dense matrix multiplication."""
        sparse_a = torch.sparse_coo_tensor(indices=[[0, 1], [0, 2]], values=[1.0, 2.0], size=(3, 3))
        dense_b = torch.tensor([[1.0], [2.0], [3.0]])
        result = self.sparse_ops.sparse_matmul(sparse_a, dense_b)
        expected = torch.tensor([[1.0], [6.0], [0.0]])
        assert torch.allclose(result, expected)

    def test_optimize_belief_update(self) -> None:
        ."""Test optimized belief update with sparse operations."""
        A = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.9, 0.0]])
        sparse_A = self.sparse_ops.sparsify_tensor(A)
        observation = torch.tensor([0.0, 1.0, 0.0])
        prior = torch.tensor([0.25, 0.25, 0.25, 0.25])
        posterior = self.sparse_ops.optimize_belief_update(sparse_A, observation, prior)
        assert posterior.shape == prior.shape
        assert torch.allclose(posterior.sum(), torch.tensor(1.0))


class TestParallelInference:
    def setup_method(self) -> None:
        .."""Setup for tests.."""
        self.config = OptimizationConfig(use_parallel_processing=True, num_threads=2, use_gpu=False)
        self.parallel = ParallelInference(self.config)

    def teardown_method(self):
        ."""Cleanup after tests."""
        self.parallel.cleanup()

    def test_parallel_belief_updates(self) -> None:
        """Test parallel belief update processing"""
        beliefs = [
            torch.tensor([0.25, 0.25, 0.25, 0.25]),
            torch.tensor([0.4, 0.3, 0.2, 0.1]),
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
        ]
        observations = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]
        A_matrices = [torch.rand(3, 4) for _ in range(3)]
        for A in A_matrices:
            A /= A.sum(dim=0, keepdim=True)
        updated_beliefs = self.parallel.parallel_belief_updates(beliefs, observations, A_matrices)
        assert len(updated_beliefs) == 3
        for belief in updated_beliefs:
            assert belief.shape == (4,)
            assert torch.allclose(belief.sum(), torch.tensor(1.0))

    def test_parallel_expected_free_energy(self) -> None:
        ."""Test parallel EFE computation."""
        qs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        A = torch.rand(3, 4)
        A /= A.sum(dim=0, keepdim=True)
        B = torch.rand(4, 4, 2)
        B /= B.sum(dim=0, keepdim=True)
        C = torch.tensor([0.8, 0.1, 0.1])
        C /= C.sum()
        actions = [0, 1]
        G_values = self.parallel.parallel_expected_free_energy(qs, A, B, C, actions)
        assert G_values.shape == (2,)
        assert all(torch.isfinite(G_values))

    def test_non_parallel_fallback(self) -> None:
        ."""Test fallback to sequential processing."""
        self.parallel.config.use_parallel_processing = False
        beliefs = [torch.rand(4) for _ in range(2)]
        observations = [torch.rand(3) for _ in range(2)]
        A_matrices = [torch.rand(3, 4) for _ in range(2)]
        updated_beliefs = self.parallel.parallel_belief_updates(beliefs, observations, A_matrices)
        assert len(updated_beliefs) == 2


class TestCachedInference:
    def setup_method(self) -> None:
        ."""Setup for tests."""
        self.config = OptimizationConfig(
            use_caching=True, cache_size=10, cache_ttl=60, use_gpu=False
        )
        self.cache = CachedInference(self.config)

    def test_cache_key_generation(self) -> None:
        ."""Test cache key generation."""
        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scalar = 42
        key1 = self.cache._cache_key(tensor1, scalar)
        key2 = self.cache._cache_key(tensor1, scalar)
        key3 = self.cache._cache_key(tensor2, scalar)
        assert key1 == key2
        assert key1 != key3

    def test_cached_belief_update(self) -> None:
        ."""Test belief update caching."""
        belief = torch.tensor([0.25, 0.25, 0.25, 0.25])
        observation = torch.tensor([1.0, 0.0, 0.0])
        A = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.9, 0.0]])
        result1 = self.cache.cached_belief_update(belief, observation, A)
        assert self.cache.cache_misses == 1
        assert self.cache.cache_hits == 0
        result2 = self.cache.cached_belief_update(belief, observation, A)
        assert self.cache.cache_misses == 1
        assert self.cache.cache_hits == 1
        assert torch.allclose(result1, result2)

    def test_cache_eviction(self) -> None:
        ."""Test cache eviction when full."""
        for i in range(15):
            belief = torch.rand(4)
            observation = torch.rand(3)
            A = torch.rand(3, 4)
            self.cache.cached_belief_update(belief, observation, A)
        assert len(self.cache.cache) <= self.config.cache_size

    def test_cache_ttl(self) -> None:
        ."""Test cache TTL expiration."""
        self.cache.config.cache_ttl = 0.1
        belief = torch.rand(4)
        observation = torch.rand(3)
        A = torch.rand(3, 4)
        self.cache.cached_belief_update(belief, observation, A)
        time.sleep(0.2)
        key = self.cache._cache_key(belief, observation, A)
        assert not self.cache._is_cache_valid(key)

    def test_get_cache_stats(self) -> None:
        ."""Test cache statistics."""
        for i in range(5):
            belief = torch.rand(4)
            observation = torch.rand(3)
            A = torch.rand(3, 4)
            self.cache.cached_belief_update(belief, observation, A)
            self.cache.cached_belief_update(belief, observation, A)
        stats = self.cache.get_cache_stats()
        assert stats["cache_hits"] == 5
        assert stats["cache_misses"] == 5
        assert stats["hit_rate"] == 0.5
        assert stats["cache_size"] == 5


class TestGPUOptimizer:
    def setup_method(self) -> None:
        .."""Setup for tests.."""
        self.config = OptimizationConfig(use_gpu=False, use_mixed_precision=True)
        self.gpu_opt = GPUOptimizer(self.config)

    def test_optimize_tensor_operations(self) -> None:
        """Test tensor optimization"""
        tensors = [torch.rand(10, 10), torch.rand(5, 5), torch.rand(3, 3, 3)]
        optimized = self.gpu_opt.optimize_tensor_operations(tensors)
        assert len(optimized) == 3
        for opt, orig in zip(optimized, tensors):
            assert opt.shape == orig.shape
            assert opt.is_contiguous()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_inference(self) -> None:
        ."""Test mixed precision inference."""
        model = torch.nn.Linear(10, 5)
        inputs = torch.rand(32, 10)
        outputs = self.gpu_opt.mixed_precision_inference(model, inputs)
        assert outputs.shape == (32, 5)

    def test_cuda_graph_creation(self) -> None:
        ."""Test CUDA graph creation (CPU fallback)."""

        def simple_func(x, y):
            return x + y

        sample_inputs = (torch.rand(5), torch.rand(5))
        graphed_func = self.gpu_opt.create_cuda_graph(simple_func, sample_inputs)
        result = graphed_func(torch.rand(5), torch.rand(5))
        assert result.shape == (5,)


class TestBatchProcessor:
    def setup_method(self) -> None:
        .."""Setup for tests.."""
        self.config = OptimizationConfig(batch_size=3, use_gpu=False)
        self.batch_proc = BatchProcessor(self.config)

    def test_add_request(self) -> None:
        """Test adding requests to batch"""

        def dummy_computation(x):
            return x * 2

        for i in range(2):
            self.batch_proc.add_request(f"req_{i}", dummy_computation, torch.tensor(float(i)))
        assert len(self.batch_proc.pending_requests) == 2
        assert len(self.batch_proc.results) == 0

    def test_batch_processing(self) -> None:
        ."""Test batch processing when full."""

        def dummy_computation(x):
            return x * 2

        for i in range(3):
            self.batch_proc.add_request(f"req_{i}", dummy_computation, torch.tensor(float(i)))
        assert len(self.batch_proc.pending_requests) == 0
        for i in range(3):
            result = self.batch_proc.get_result(f"req_{i}", timeout=0.1)
            assert result == i * 2

    def test_batch_belief_updates(self) -> None:
        ."""Test batched belief update processing."""
        requests = []
        for i in range(3):
            belief = torch.rand(4)
            observation = torch.rand(3)
            A = torch.rand(3, 4)
            request = {
                "id": f"req_{i}",
                "computation": Mock(__name__="belief_update"),
                "args": (belief, observation, A),
                "kwargs": {},
            }
            requests.append(request)
        self.batch_proc._batch_belief_updates(requests)
        assert len(self.batch_proc.results) == 3
        for i in range(3):
            result = self.batch_proc.results[f"req_{i}"]
            assert result.shape == (4,)

    def test_get_result_timeout(self) -> None:
        ."""Test result retrieval with timeout."""
        result = self.batch_proc.get_result("nonexistent", timeout=0.1)
        assert result is None


class TestComputationalOptimizer:
    def setup_method(self) -> None:
        ."""Setup for tests."""
        self.config = OptimizationConfig(
            use_sparse_operations=True,
            use_parallel_processing=True,
            use_caching=True,
            use_gpu=False,
        )
        self.optimizer = ComputationalOptimizer(self.config)

    def teardown_method(self):
        .."""Cleanup after tests.."""
        self.optimizer.cleanup()

    def test_optimized_belief_update(self) -> None:
        """Test optimized belief update"""
        belief = torch.tensor([0.25, 0.25, 0.25, 0.25])
        observation = torch.tensor([0.0, 1.0, 0.0])
        A = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.9, 0.0]])
        result = self.optimizer.optimized_belief_update(belief, observation, A)
        assert result.shape == belief.shape
        assert torch.allclose(result.sum(), torch.tensor(1.0))
        assert "belief_update" in self.optimizer.timing_stats

    def test_optimized_belief_update_sparse(self) -> None:
        ."""Test belief update with sparse matrix."""
        belief = torch.rand(10)
        belief /= belief.sum()
        observation = torch.zeros(8)
        observation[2] = 1.0
        A = torch.zeros(8, 10)
        A[2, 3] = 0.9
        A[2, 4] = 0.1
        for i in range(8):
            if A[i].sum() == 0:
                A[i, 0] = 1.0
        result = self.optimizer.optimized_belief_update(belief, observation, A, use_sparse=True)
        assert result.shape == belief.shape
        assert torch.allclose(result.sum(), torch.tensor(1.0), atol=1e-06)

    def test_optimized_action_selection(self) -> None:
        ."""Test optimized action selection."""
        qs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        A = torch.rand(3, 4)
        A /= A.sum(dim=0, keepdim=True)
        B = torch.rand(4, 4, 2)
        B /= B.sum(dim=0, keepdim=True)
        C = torch.tensor([0.8, 0.1, 0.1])
        action_probs, G_values = self.optimizer.optimized_action_selection(
            qs, A, B, C, num_actions=2
        )
        assert action_probs.shape == (2,)
        assert G_values.shape == (2,)
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0))
        assert "action_selection" in self.optimizer.timing_stats

    def test_performance_report(self) -> None:
        ."""Test performance report generation."""
        for _ in range(5):
            belief = torch.rand(4)
            belief /= belief.sum()
            observation = torch.rand(3)
            A = torch.rand(3, 4)
            self.optimizer.optimized_belief_update(belief, observation, A)
        report = self.optimizer.get_performance_report()
        assert "timing_stats" in report
        assert "cache_stats" in report
        assert "config" in report
        belief_stats = report["timing_stats"]["belief_update"]
        assert belief_stats["count"] == 5
        assert belief_stats["avg_time_ms"] > 0

    def test_automatic_sparse_detection(self) -> None:
        ."""Test automatic sparse matrix detection."""
        belief = torch.rand(20)
        belief /= belief.sum()
        observation = torch.zeros(15)
        observation[5] = 1.0
        A = torch.zeros(15, 20)
        for i in range(15):
            A[i, i % 20] = 0.9
            A[i, (i + 1) % 20] = 0.1
        result = self.optimizer.optimized_belief_update(belief, observation, A)
        assert result.shape == belief.shape
        assert torch.isfinite(result).all()


class TestIntegration:
    .."""Integration tests for computational optimization.."""

    def test_full_inference_cycle(self) -> None:
        """Test complete inference cycle with optimizations"""
        config = OptimizationConfig(
            use_sparse_operations=True,
            use_parallel_processing=True,
            use_caching=True,
            use_gpu=False,
        )
        optimizer = ComputationalOptimizer(config)
        belief = torch.tensor([0.25, 0.25, 0.25, 0.25])
        A = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.9, 0.0]])
        B = torch.rand(4, 4, 2)
        B /= B.sum(dim=0, keepdim=True)
        C = torch.tensor([0.7, 0.2, 0.1])
        for t in range(10):
            obs_idx = t % 3
            observation = torch.zeros(3)
            observation[obs_idx] = 1.0
            belief = optimizer.optimized_belief_update(belief, observation, A)
            action_probs, _ = optimizer.optimized_action_selection(belief, A, B, C, num_actions=2)
            action = torch.multinomial(action_probs, 1).item()
            belief = torch.matmul(B[:, :, action], belief)
        report = optimizer.get_performance_report()
        assert report["timing_stats"]["belief_update"]["count"] == 10
        assert report["timing_stats"]["action_selection"]["count"] == 10
        assert report["cache_stats"]["hit_rate"] >= 0
        optimizer.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])

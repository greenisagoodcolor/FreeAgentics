#!/usr/bin/env python3
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from typing import Optional

from inference.gnn.layers import GNNStack
from inference.gnn.performance_optimizer import (
    GraphCache,
    HardwareAccelerator,
    MemoryOptimizer,
    OptimizationConfig,
    ParallelProcessor,
    PerformanceOptimizer,
    PerformanceProfiler,
    cache_result,
    optimize_for_inference,
)


def extract_features_standalone(graph):
    """Standalone feature extraction function for parallel processing."""
    return {"id": graph["id"], "features": graph["data"].mean().item()}


class TestOptimizationConfig(unittest.TestCase):
    """Test optimization configuration"""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = OptimizationConfig()

        assert config.enable_mixed_precision is True
        assert config.enable_gradient_checkpointing is False
        assert config.enable_graph_caching is True
        assert config.num_workers == 4
        assert config.max_cache_size_mb == 1024

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = OptimizationConfig(
            enable_mixed_precision=False, num_workers=8, max_cache_size_mb=2048
        )

        assert config.enable_mixed_precision is False
        assert config.num_workers == 8
        assert config.max_cache_size_mb == 2048


class TestMemoryOptimizer:
    """Test memory optimization functionality"""

    def test_memory_optimizer_init(self) -> None:
        """Test memory optimizer initialization"""
        config = OptimizationConfig()
        optimizer = MemoryOptimizer(config)

        assert optimizer.config == config
        assert optimizer._memory_threshold_mb == 1024

    def test_optimize_model(self) -> None:
        """Test model optimization"""
        config = OptimizationConfig(
            enable_gradient_checkpointing=False,
            enable_mixed_precision=False,
            optimize_for_inference=True,
        )
        optimizer = MemoryOptimizer(config)

        # Create simple model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

        # Optimize model
        optimized_model = optimizer.optimize_model(model)

        # Check that model is in eval mode
        assert not optimized_model.training

        # Check that gradients are disabled
        for param in optimized_model.parameters():
            assert not param.requires_grad

    def test_optimize_batch_processing(self) -> None:
        """Test batch size optimization"""
        # Test with limited memory
        optimized_batch = MemoryOptimizer.optimize_batch_processing(
            batch_size=32,
            available_memory_mb=100,
            node_features_dim=64,
            avg_nodes_per_graph=100,
        )

        # Should return smaller batch size due to memory constraints
        assert optimized_batch < 32
        assert optimized_batch >= 1

        # Test with plenty of memory
        optimized_batch = MemoryOptimizer.optimize_batch_processing(
            batch_size=32,
            available_memory_mb=10000,
            node_features_dim=64,
            avg_nodes_per_graph=100,
        )

        # Should return requested batch size
        assert optimized_batch == 32

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.synchronize")
    def test_clear_cache(self, mock_sync, mock_empty, mock_cuda) -> None:
        """Test cache clearing"""
        mock_cuda.return_value = True

        config = OptimizationConfig()
        optimizer = MemoryOptimizer(config)

        optimizer.clear_cache()

        mock_empty.assert_called_once()
        mock_sync.assert_called_once()


class TestHardwareAccelerator:
    """Test hardware acceleration functionality"""

    @patch("torch.cuda.is_available")
    def test_device_detection_cpu(self, mock_cuda) -> None:
        """Test CPU device detection"""
        mock_cuda.return_value = False

        config = OptimizationConfig()
        accelerator = HardwareAccelerator(config)

        assert accelerator.device.type == "cpu"
        assert accelerator.scaler is None

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    def test_device_detection_single_gpu(self, mock_count, mock_cuda) -> None:
        """Test single GPU detection"""
        mock_cuda.return_value = True
        mock_count.return_value = 1

        config = OptimizationConfig()
        accelerator = HardwareAccelerator(config)

        assert accelerator.device.type == "cuda"
        assert accelerator.device.index == 0

    def test_accelerate_forward_cpu(self) -> None:
        """Test forward acceleration on CPU"""
        config = OptimizationConfig()
        accelerator = HardwareAccelerator(config)
        accelerator.device = torch.device("cpu")

        def forward_fn(x):
            return x * 2

        x = torch.randn(10)
        result = accelerator.accelerate_forward(None, forward_fn, x)

        assert torch.allclose(result, x * 2)

    @patch("torch.cuda.is_available")
    def test_accelerate_backward(self, mock_cuda) -> None:
        """Test backward acceleration"""
        mock_cuda.return_value = False

        config = OptimizationConfig()
        accelerator = HardwareAccelerator(config)

        # Create simple model and optimizer
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create loss
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.mean()

        # Test backward pass
        optimizer.zero_grad()
        accelerator.accelerate_backward(loss, optimizer)

        # Check that gradients were computed
        for param in model.parameters():
            assert param.grad is not None


class TestGraphCache:
    """Test graph caching functionality"""

    def setUp(self) -> None:
        """Set up test cache directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "test_cache"

    def tearDown(self) -> None:
        """Clean up test cache directory"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_cache_operations(self) -> None:
        """Test basic cache operations"""
        self.setUp()

        config = OptimizationConfig(max_cache_size_mb=10)
        cache = GraphCache(config)
        cache.cache_dir = self.cache_dir

        # Test cache miss
        result = cache.get("test_key", "operation")
        assert result is None
        assert cache.misses == 1

        # Test cache set
        data = torch.randn(10, 20)
        cache.set("test_key", "operation", data)

        # Test cache hit
        cached_data = cache.get("test_key", "operation")
        assert cached_data is not None
        assert torch.allclose(cached_data, data)
        assert cache.hits == 1

        self.tearDown()

    def test_cache_eviction(self) -> None:
        """Test cache eviction"""
        self.setUp()

        # Small cache size to trigger eviction
        config = OptimizationConfig(max_cache_size_mb=1)
        cache = GraphCache(config)
        cache.cache_dir = self.cache_dir

        # Fill cache
        for i in range(10):
            data = torch.randn(100, 100)  # Large tensors
            cache.set(f"graph_{i}", "operation", data)

        # Check that cache size is limited
        assert cache._cache_size_mb <= config.max_cache_size_mb

        self.tearDown()

    def test_cache_stats(self) -> None:
        """Test cache statistics"""
        self.setUp()

        config = OptimizationConfig()
        cache = GraphCache(config)
        cache.cache_dir = self.cache_dir

        # Generate some cache activity
        cache.get("miss_1", "op")
        cache.set("hit_1", "op", torch.randn(10))
        cache.get("hit_1", "op")
        cache.get("miss_2", "op")

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 1 / 3
        assert stats["memory_items"] == 1

        self.tearDown()


class TestParallelProcessor(unittest.TestCase):
    """Test parallel processing functionality"""

    def test_parallel_processor_init(self) -> None:
        """Test parallel processor initialization"""
        config = OptimizationConfig(num_workers=8)
        processor = ParallelProcessor(config)

        assert processor.num_workers <= 8
        assert processor.num_workers > 0

    def test_create_data_loader(self) -> None:
        """Test data loader creation"""
        config = OptimizationConfig(num_workers=2)
        processor = ParallelProcessor(config)

        # Create dummy dataset
        dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))

        loader = processor.create_data_loader(dataset, batch_size=16)

        assert loader.batch_size == 16
        assert loader.num_workers == 2
        assert loader.pin_memory == config.pin_memory

    def test_parallel_feature_extraction(self) -> None:
        """Test parallel feature extraction"""
        config = OptimizationConfig(num_workers=2)
        processor = ParallelProcessor(config)

        # Create test graphs
        graphs = [{"id": i, "data": torch.randn(10)} for i in range(10)]

        # Extract features
        results = processor.parallel_feature_extraction(graphs, extract_features_standalone)

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["id"] == i


class TestPerformanceProfiler(unittest.TestCase):
    """Test performance profiling functionality"""

    def setUp(self) -> None:
        """Set up test environment"""
        self.config = OptimizationConfig(enable_profiling=True)
        self.profiler = PerformanceProfiler(self.config)

    def test_profiler_init(self) -> None:
        """Test profiler initialization"""
        assert self.profiler.config.enable_profiling
        assert len(self.profiler.profiles) == 0

    def test_profile_operation(self) -> None:
        """Test operation profiling"""

        # Profile a simple operation

        def test_operation(x) -> None:
            time.sleep(0.01)  # Simulate work
            return x * 2

        x = torch.randn(10)
        result = self.profiler.profile_operation("test_op", test_operation, x)

        assert torch.allclose(result, x * 2)
        assert "test_op" in self.profiler.profiles
        assert len(self.profiler.profiles["test_op"]) == 1

        profile = self.profiler.profiles["test_op"][0]
        assert profile["duration"] >= 0.01
        assert "memory_mb" in profile

    def test_get_bottlenecks(self) -> None:
        """Test bottleneck identification"""
        # Simulate some operations
        for i in range(5):
            self.profiler.profiles[f"op_{i}"] = [
                {"duration": 0.1 * (i + 1), "memory_mb": 10 * i} for _ in range(i + 1)
            ]

        bottlenecks = self.profiler.get_bottlenecks(top_k=3)

        assert len(bottlenecks) == 3
        # op_4 should be the biggest bottleneck (5 calls * 0.5s each)
        assert bottlenecks[0]["operation"] == "op_4"


class TestPerformanceOptimizer:
    """Test main performance optimizer"""

    def test_optimizer_init(self) -> None:
        """Test optimizer initialization"""
        config = OptimizationConfig()
        optimizer = PerformanceOptimizer(config)

        assert optimizer.config == config
        assert optimizer.memory_optimizer is not None
        assert optimizer.hardware_accelerator is not None
        assert optimizer.cache is not None
        assert optimizer.parallel_processor is not None
        assert optimizer.profiler is not None

    @patch("torch.cuda.is_available")
    def test_optimize_model(self, mock_cuda) -> None:
        """Test model optimization"""
        mock_cuda.return_value = False

        config = OptimizationConfig()
        optimizer = PerformanceOptimizer(config)

        # Create model
        from inference.gnn.layers import LayerConfig

        layer_configs = [
            LayerConfig(in_channels=16, out_channels=32),
            LayerConfig(in_channels=32, out_channels=32),
            LayerConfig(in_channels=32, out_channels=8),
        ]
        model = GNNStack(layer_configs)

        # Optimize model
        optimized_model = optimizer.optimize_model(model)

        assert next(optimized_model.parameters()).device.type == "cpu"

    def test_optimize_batch_size(self) -> None:
        """Test batch size optimization"""
        config = OptimizationConfig()
        optimizer = PerformanceOptimizer(config)

        # Mock device as CPU
        optimizer.hardware_accelerator.device = torch.device("cpu")

        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024  # 1GB

            batch_size = optimizer.optimize_batch_size(
                requested_batch_size=64,
                node_features_dim=32,
                avg_nodes_per_graph=50,
            )

            assert batch_size > 0
            assert batch_size <= 64

    def test_get_optimization_stats(self) -> None:
        """Test optimization statistics"""
        config = OptimizationConfig()
        optimizer = PerformanceOptimizer(config)

        stats = optimizer.get_optimization_stats()

        assert "device" in stats
        assert "mixed_precision" in stats
        assert "cache_stats" in stats
        assert "parallel_workers" in stats


class TestOptimizationDecorators:
    """Test optimization decorators"""

    def test_optimize_for_inference_decorator(self) -> None:
        """Test inference optimization decorator"""

        @optimize_for_inference
        def test_function(x) -> None:
            return x * 2

        x = torch.randn(10, requires_grad=True)
        result = test_function(x)

        assert torch.allclose(result, x * 2)
        # Result should not require gradients
        assert not result.requires_grad

    def test_cache_result_decorator(self) -> None:
        """Test result caching decorator"""
        # Create mock cache
        mock_cache = Mock()
        mock_cache.get.return_value = None

        def cache_key_fn(*args, **kwargs):
            return "test_key"

        @cache_result(cache_key_fn)
        def test_function(self, x) -> None:
            return x * 2

        # Create object with cache
        obj = Mock()
        obj.cache = mock_cache

        # Call function
        x = 5
        result = test_function(obj, x)

        assert result == 10
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()


if __name__ == "__main__":
    unittest.main()

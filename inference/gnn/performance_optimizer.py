import gc
import multiprocessing as mp
import pickle
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .layers import GNNStack
from .monitoring import get_logger, monitor_performance

"""
Performance Optimization Module for GNN Processing
This module provides various optimization techniques to improve the performance
of GNN processing operations including memory optimization, hardware acceleration,
caching, and parallel processing.
"""
logger = get_logger().logger


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations"""

    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    enable_graph_caching: bool = True
    enable_cuda_graphs: bool = False
    enable_memory_efficient_attention: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    max_cache_size_mb: int = 1024
    enable_profiling: bool = False
    optimize_for_inference: bool = False


class MemoryOptimizer:
    """
    Optimizes memory usage for large graph processing.
    Features:
    - Dynamic memory allocation
    - Gradient checkpointing
    - Memory-efficient operations
    - Automatic garbage collection
    """

    def __init__(self, config: OptimizationConfig) -> None:
        """
        Initialize memory optimizer.
        Args:
            config: Optimization configuration
        """
        self.config = config
        self._memory_threshold_mb = 1024

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply memory optimizations to model.
        Args:
            model: PyTorch model to optimize
        Returns:
            Optimized model
        """
        if self.config.enable_gradient_checkpointing:
            self._enable_gradient_checkpointing(model)
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            model = model.half()
        if self.config.enable_memory_efficient_attention:
            self._optimize_attention_layers(model)
        if self.config.optimize_for_inference:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        return model

    def _enable_gradient_checkpointing(self, model: nn.Module) -> None:
        """Enable gradient checkpointing for specific layers"""

        def checkpoint_wrapper(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
            if module.training:
                return checkpoint(module._forward_impl, *args, **kwargs)
            else:
                return module._forward_impl(*args, **kwargs)

        for name, module in model.named_modules():
            if hasattr(module, "_forward_impl"):
                module.forward = lambda *args, m=module, **kwargs: checkpoint_wrapper(
                    m, *args, **kwargs
                )

    def _optimize_attention_layers(self, model: nn.Module) -> None:
        """Optimize attention layers for memory efficiency"""
        for module in model.modules():
            if hasattr(module, "attention_dropout"):
                module.use_flash_attention = True  # type: ignore[attr-defined]

    @staticmethod
    def optimize_batch_processing(
        batch_size: int,
        available_memory_mb: float,
        node_features_dim: int,
        avg_nodes_per_graph: int,
    ) -> int:
        """
        Calculate optimal batch size based on available memory.
        Args:
            batch_size: Requested batch size
            available_memory_mb: Available memory in MB
            node_features_dim: Dimension of node features
            avg_nodes_per_graph: Average nodes per graph
        Returns:
            Optimized batch size
        """
        bytes_per_float = 4
        memory_per_graph_mb = (
            avg_nodes_per_graph * node_features_dim * bytes_per_float / 1024 / 1024
        )
        memory_per_graph_mb *= 3
        max_batch_size = int(available_memory_mb / memory_per_graph_mb)
        return min(batch_size, max(max_batch_size, 1))

    def clear_cache(self) -> None:
        """Clear GPU cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


class HardwareAccelerator:
    """
    Manages hardware acceleration for GNN processing.
    Features:
    - GPU/TPU detection and setup
    - Mixed precision training
    - CUDA graphs for inference
    - Multi-GPU support
    """

    def __init__(self, config: OptimizationConfig) -> None:
        """
        Initialize hardware accelerator.
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.device = self._detect_device()
        self.scaler = None
        if self.config.enable_mixed_precision and self.device.type == "cuda":
            self.scaler = amp.GradScaler()

    def _detect_device(self) -> torch.device:
        """Detect and return best available device"""
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                free_memory = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    free_memory.append(
                        torch.cuda.get_device_properties(i).total_memory
                        - torch.cuda.memory_allocated(i)
                    )
                best_gpu = np.argmax(free_memory)
                return torch.device(f"cuda:{best_gpu}")
            else:
                return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def accelerate_forward(
        self, model: nn.Module, forward_fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Accelerate forward pass with mixed precision.
        Args:
            model: Model to run
            forward_fn: Forward function
            *args: Forward function arguments
            **kwargs: Forward function keyword arguments
        Returns:
            Forward pass output
        """
        if self.config.enable_mixed_precision and self.device.type == "cuda":
            with amp.autocast():
                return forward_fn(*args, **kwargs)
        else:
            return forward_fn(*args, **kwargs)

    def accelerate_backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """
        Accelerate backward pass with mixed precision.
        Args:
            loss: Loss tensor
            optimizer: Optimizer
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

    def create_cuda_graph(self, model: nn.Module, sample_input: torch.Tensor) -> Callable:
        """
        Create CUDA graph for faster inference.
        Args:
            model: Model to create graph for
            sample_input: Sample input tensor
        Returns:
            CUDA graph callable
        """
        if not (self.config.enable_cuda_graphs and self.device.type == "cuda"):
            return lambda x: model(x)
        static_input = sample_input.clone()
        static_output = model(static_input)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = model(static_input)

        def cuda_graph_forward(input_tensor: torch.Tensor) -> torch.Tensor:
            static_input.copy_(input_tensor)
            graph.replay()
            return static_output.clone()

        return cuda_graph_forward

    def setup_distributed(self, rank: int, world_size: int) -> None:
        """Setup distributed training"""
        if self.device.type == "cuda":
            torch.cuda.set_device(rank)
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)


class GraphCache:
    """
    Caching mechanism for graph processing results.
    Features:
    - LRU cache for processed features
    - Persistent cache storage
    - Memory-aware caching
    - Thread-safe operations
    """

    def __init__(self, config: OptimizationConfig) -> None:
        """
        Initialize graph cache.
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.cache_dir = Path(".cache/gnn")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self._cache_size_mb = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def cache_key(self, graph_id: str, operation: str) -> str:
        """Generate cache key"""
        return f"{graph_id}_{operation}"

    def get(self, graph_id: str, operation: str) -> Optional[Any]:
        """
        Get cached result.
        Args:
            graph_id: Graph identifier
            operation: Operation identifier
        Returns:
            Cached result or None
        """
        if not self.config.enable_graph_caching:
            return None
        key = self.cache_key(graph_id, operation)
        with self._lock:
            if key in self._memory_cache:
                self.hits += 1
                return self._memory_cache[key]
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        data = pickle.load(f)
                    self._add_to_memory_cache(key, data)
                    self.hits += 1
                    return data
                except Exception as e:
                    logger.error(f"Failed to load cache: {e}")
            self.misses += 1
            return None

    def set(self, graph_id: str, operation: str, data: Any) -> None:
        """
        Cache result.
        Args:
            graph_id: Graph identifier
            operation: Operation identifier
            data: Data to cache
        """
        if not self.config.enable_graph_caching:
            return
        key = self.cache_key(graph_id, operation)
        with self._lock:
            self._add_to_memory_cache(key, data)
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

    def _add_to_memory_cache(self, key: str, data: Any):
        """Add data to memory cache with size management"""
        size_mb = self._estimate_size_mb(data)
        while (
            self._cache_size_mb + size_mb > self.config.max_cache_size_mb
            and len(self._memory_cache) > 0
        ):
            evict_key = next(iter(self._memory_cache))
            evicted_data = self._memory_cache.pop(evict_key)
            self._cache_size_mb -= self._estimate_size_mb(evicted_data)
        self._memory_cache[key] = data
        self._cache_size_mb += size_mb

    def _estimate_size_mb(self, data: Any) -> float:
        """Estimate size of data in MB"""
        if isinstance(data, torch.Tensor):
            return data.element_size() * data.nelement() / 1024 / 1024
        elif isinstance(data, np.ndarray):
            return data.nbytes / 1024 / 1024
        else:
            return 1.0

    def clear(self) -> None:
        """Clear all caches"""
        with self._lock:
            self._memory_cache.clear()
            self._cache_size_mb = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "memory_size_mb": self._cache_size_mb,
            "memory_items": len(self._memory_cache),
        }


class ParallelProcessor:
    """
    Implements parallel processing for graph operations.
    Features:
    - Multi-threaded data loading
    - Parallel feature extraction
    - Distributed graph processing
    - Asynchronous operations
    """

    def __init__(self, config: OptimizationConfig) -> None:
        """
        Initialize parallel processor.
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.num_workers = min(config.num_workers, mp.cpu_count())

    def parallel_feature_extraction(
        self, graphs: List[Dict[str, Any]], extractor_fn: Callable
    ) -> List[Any]:
        """
        Extract features from multiple graphs in parallel.
        Args:
            graphs: List of graphs
            extractor_fn: Feature extraction function
        Returns:
            List of extracted features
        """
        if len(graphs) < self.num_workers * 2:
            return [extractor_fn(g) for g in graphs]
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(extractor_fn, graphs)
        return results

    def create_data_loader(
        self, dataset: Any, batch_size: int, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """
        Create optimized data loader.
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
        Returns:
            Optimized DataLoader
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )

    def parallel_graph_processing(
        self, graphs: List[Any], process_fn: Callable, chunk_size: Optional[int] = None
    ) -> List[Any]:
        """
        Process multiple graphs in parallel chunks.
        Args:
            graphs: List of graphs to process
            process_fn: Processing function
            chunk_size: Size of chunks for processing
        Returns:
            List of processed results
        """
        if chunk_size is None:
            chunk_size = max(1, len(graphs) // (self.num_workers * 4))
        chunks = [graphs[i : i + chunk_size] for i in range(0, len(graphs), chunk_size)]
        with mp.Pool(processes=self.num_workers) as pool:
            chunk_results = pool.map(lambda chunk: [process_fn(g) for g in chunk], chunks)
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        return results


class PerformanceProfiler:
    """
    Profile GNN operations to identify bottlenecks.
    Features:
    - Operation timing
    - Memory profiling
    - GPU utilization tracking
    - Bottleneck identification
    """

    def __init__(self, config: OptimizationConfig) -> None:
        """
        Initialize profiler.
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.profiles = defaultdict(list)
        self._profiler = None

    def start_profiling(self):
        """Start profiling session"""
        if not self.config.enable_profiling:
            return
        if torch.cuda.is_available():
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            self._profiler.__enter__()

    def stop_profiling(self) -> Optional[str]:
        """
        Stop profiling and return report.
        Returns:
            Profiling report or None
        """
        if self._profiler is None:
            return None
        self._profiler.__exit__(None, None, None)
        report = self._profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        trace_file = f"gnn_profile_{int(time.time())}.json"
        self._profiler.export_chrome_trace(trace_file)
        self._profiler = None
        return report

    @monitor_performance("profiled_operation")
    def profile_operation(
        self, operation_name: str, operation_fn: Callable, *args, **kwargs
    ) -> Any:
        """
        Profile a specific operation.
        Args:
            operation_name: Name of operation
            operation_fn: Operation function
            *args: Operation arguments
            **kwargs: Operation keyword arguments
        Returns:
            Operation result
        """
        start_time = time.time()
        start_memory = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        result = operation_fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - start_memory) / 1024 / 1024
        else:
            memory_used = 0
        end_time = time.time()
        duration = end_time - start_time
        self.profiles[operation_name].append(
            {"duration": duration, "memory_mb": memory_used, "timestamp": start_time}
        )
        return result

    def get_bottlenecks(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Identify top bottlenecks.
        Args:
            top_k: Number of top bottlenecks to return
        Returns:
            List of bottleneck operations
        """
        bottlenecks = []
        for operation, profiles in self.profiles.items():
            if not profiles:
                continue
            avg_duration = np.mean([p["duration"] for p in profiles])
            avg_memory = np.mean([p["memory_mb"] for p in profiles])
            total_time = sum((p["duration"] for p in profiles))
            bottlenecks.append(
                {
                    "operation": operation,
                    "avg_duration": avg_duration,
                    "total_time": total_time,
                    "avg_memory_mb": avg_memory,
                    "call_count": len(profiles),
                }
            )
        bottlenecks.sort(key=lambda x: x["total_time"], reverse=True)
        return bottlenecks[:top_k]


class PerformanceOptimizer:
    """
    Main performance optimization orchestrator.
    Combines all optimization techniques for maximum performance.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None) -> None:
        """
        Initialize performance optimizer.
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.hardware_accelerator = HardwareAccelerator(self.config)
        self.cache = GraphCache(self.config)
        self.parallel_processor = ParallelProcessor(self.config)
        self.profiler = PerformanceProfiler(self.config)
        logger.info("Performance optimizer initialized")

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply all optimizations to model.
        Args:
            model: Model to optimize
        Returns:
            Optimized model
        """
        model = self.memory_optimizer.optimize_model(model)
        model = model.to(self.hardware_accelerator.device)
        return model

    def optimize_batch_size(
        self,
        requested_batch_size: int,
        node_features_dim: int,
        avg_nodes_per_graph: int,
    ) -> int:
        """
        Calculate optimal batch size.
        Args:
            requested_batch_size: Requested batch size
            node_features_dim: Node feature dimension
            avg_nodes_per_graph: Average nodes per graph
        Returns:
            Optimized batch size
        """
        if self.hardware_accelerator.device.type == "cuda":
            available_memory = (
                (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())
                / 1024
                / 1024
            )
        else:
            available_memory = psutil.virtual_memory().available / 1024 / 1024
        return self.memory_optimizer.optimize_batch_processing(
            requested_batch_size,
            available_memory * 0.8,
            node_features_dim,
            avg_nodes_per_graph,
        )

    def cached_forward(self, graph_id: str, forward_fn: Callable, *args, **kwargs) -> Any:
        """
        Forward pass with caching.
        Args:
            graph_id: Graph identifier
            forward_fn: Forward function
            *args: Forward arguments
            **kwargs: Forward keyword arguments
        Returns:
            Forward result
        """
        cached_result = self.cache.get(graph_id, "forward")
        if cached_result is not None:
            return cached_result
        result = forward_fn(*args, **kwargs)
        self.cache.set(graph_id, "forward", result)
        return result

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = {
            "device": str(self.hardware_accelerator.device),
            "mixed_precision": self.config.enable_mixed_precision,
            "cache_stats": self.cache.get_stats(),
            "parallel_workers": self.parallel_processor.num_workers,
        }
        if self.config.enable_profiling:
            stats["bottlenecks"] = self.profiler.get_bottlenecks()
        return stats


def optimize_for_inference(func):
    """Decorator to optimize function for inference"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                return func(*args, **kwargs)

    return wrapper


def cache_result(cache_key_fn: Callable):
    """Decorator to cache function results"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "cache"):
                key = cache_key_fn(*args, **kwargs)
                cached = self.cache.get(key, func.__name__)
                if cached is not None:
                    return cached
                result = func(self, *args, **kwargs)
                self.cache.set(key, func.__name__, result)
                return result
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    config = OptimizationConfig(
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True,
        enable_graph_caching=True,
        num_workers=4,
    )
    optimizer = PerformanceOptimizer(config)
    model = GNNStack(input_dim=32, hidden_dims=[64, 64, 32], output_dim=10, architecture="gcn")
    model = optimizer.optimize_model(model)
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Optimization stats: {optimizer.get_optimization_stats()}")
    optimizer.profiler.start_profiling()
    x = torch.randn(100, 32).to(optimizer.hardware_accelerator.device)
    edge_index = torch.randint(0, 100, (2, 300)).to(optimizer.hardware_accelerator.device)
    output = optimizer.profiler.profile_operation("forward_pass", model, x, edge_index)
    report = optimizer.profiler.stop_profiling()
    if report:
        print("\nProfiling Report:")
        print(report)

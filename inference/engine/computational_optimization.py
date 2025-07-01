"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

"""
Computational Optimization for Active Inference
This module implements various optimization techniques to improve computational
performance of Active Inference algorithms, including sparse computations,
parallel processing, and caching strategies.
"""


logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for computational optimizations"""

    # Sparse computation
    use_sparse_operations: bool = True
    sparsity_threshold: float = 0.01  # Values below this are treated as zero
    # Parallel processing
    use_parallel_processing: bool = True
    num_threads: int = 4
    num_processes: int = 2
    # GPU optimization
    use_gpu: bool = True
    use_mixed_precision: bool = True
    use_cuda_graphs: bool = True
    # Caching
    use_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    # Batch processing
    batch_size: int = 32
    max_batch_size: int = 128
    dynamic_batching: bool = True
    # Memory optimization
    gradient_checkpointing: bool = True
    memory_efficient_attention: bool = True
    # Computational
    eps: float = 1e-16
    dtype: torch.dtype = torch.float32


class SparseOperations:
    """
    Implements sparse matrix operations for Active Inference computations.
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    def sparsify_tensor(self, tensor: torch.Tensor) -> torch.sparse.Tensor:
        """Convert dense tensor to sparse format"""
        # Apply threshold
        mask = torch.abs(tensor) > self.config.sparsity_threshold
        indices = torch.nonzero(mask).t()
        values = tensor[mask]
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, tensor.shape, dtype=tensor.dtype, device=tensor.device
        )
        return sparse_tensor

    def sparse_matmul(self, sparse_a: torch.sparse.Tensor, dense_b: torch.Tensor) -> torch.Tensor:
        """Efficient sparse-dense matrix multiplication"""
        return torch.sparse.mm(sparse_a, dense_b)  # type: ignore[no-any-return]

    def sparse_log_sum_exp(self, sparse_tensor: torch.sparse.Tensor, dim: int) -> torch.Tensor:
        """Compute log-sum-exp for sparse tensors"""
        # Convert to dense for stability (can be optimized further)
        dense = sparse_tensor.to_dense()
        return torch.logsumexp(dense, dim=dim)

    def optimize_belief_update(
        self,
        sparse_A: torch.sparse.Tensor,
        observation: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Optimized belief update using sparse operations"""
        # Sparse likelihood computation
        if observation.dim() == 0:
            # Scalar observation (single index)
            obs_idx = int(observation.item())
            likelihood = sparse_A[obs_idx]
        elif observation.dim() == 1 and observation.numel() == 1:
            # 1D tensor with single element
            obs_idx = int(observation.item())
            likelihood = sparse_A[obs_idx]
        elif observation.dim() == 1:
            # 1D observation vector (one-hot or soft)
            obs_idx = int(torch.argmax(observation).item())
            likelihood = sparse_A[obs_idx]
        else:
            # Multi-dimensional observation - weighted sum for soft observations
            likelihood = torch.sparse.mm(sparse_A.t(), observation.view(-1, 1)).squeeze()
        # Convert sparse likelihood to dense for computation
        if likelihood.is_sparse:
            likelihood = likelihood.to_dense()
        # Posterior computation
        log_posterior = torch.log(likelihood + self.config.eps) + torch.log(prior + self.config.eps)
        posterior = F.softmax(log_posterior, dim=0)
        return posterior


class ParallelInference:
    """
    Implements parallel processing for Active Inference computations.
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.num_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=config.num_processes)

    def parallel_belief_updates(
        self,
        beliefs: List[torch.Tensor],
        observations: List[torch.Tensor],
        A_matrices: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Parallel belief updates for multiple agents/timesteps"""
        if not self.config.use_parallel_processing:
            return [
                self._single_belief_update(b, o, A)
                for b, o, A in zip(beliefs, observations, A_matrices)
            ]
        # Use thread pool for GPU operations
        futures = []
        for belief, obs, A in zip(beliefs, observations, A_matrices):
            future = self.thread_pool.submit(self._single_belief_update, belief, obs, A)
            futures.append(future)
        # Collect results
        updated_beliefs = [future.result() for future in futures]
        return updated_beliefs

    def _single_belief_update(
        self, belief: torch.Tensor, observation: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """Single belief update computation"""
        # Compute likelihood
        if observation.dim() == 1:
            obs_idx = torch.argmax(observation)
            likelihood = A[obs_idx]
        else:
            likelihood = torch.matmul(A.t(), observation)
        # Update belief
        posterior = likelihood * belief
        posterior = posterior / (posterior.sum() + self.config.eps)
        return posterior

    def parallel_expected_free_energy(
        self,
        qs: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        actions: List[int],
    ) -> torch.Tensor:
        """Parallel computation of expected free energy for multiple actions"""
        if not self.config.use_parallel_processing:
            return torch.stack([self._single_efe(qs, A, B, C, action) for action in actions])
        # Parallel EFE computation
        futures = []
        for action in actions:
            future = self.thread_pool.submit(self._single_efe, qs, A, B, C, action)
            futures.append(future)
        # Collect results
        efe_values = torch.stack([future.result() for future in futures])
        return efe_values

    def _single_efe(
        self,
        qs: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        action: int,
    ) -> torch.Tensor:
        """Compute EFE for single action"""
        # Predict future state
        qs_future = torch.matmul(B[:, :, action], qs)
        # Expected observations
        qo_future = torch.matmul(A, qs_future)
        # Epistemic value (information gain)
        H_A_given_s = -torch.sum(A * torch.log(A + self.config.eps), dim=0)
        epistemic = torch.sum(qs_future * H_A_given_s)
        # Pragmatic value
        pragmatic = torch.sum(qo_future * torch.log(qo_future / C + self.config.eps))
        return epistemic + pragmatic

    def cleanup(self) -> None:
        """Cleanup thread/process pools"""
        self.thread_pool.shutdown()
        self.process_pool.shutdown()


class CachedInference:
    """
    Implements caching strategies for Active Inference computations.
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_times: Dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(self, *args: Any) -> str:
        """Generate cache key from arguments"""
        # Convert tensors to tuples for hashing
        key_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                key_parts.append(tuple(arg.flatten().tolist()))
            else:
                key_parts.append(arg)
        return str(hash(tuple(key_parts)))

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self.cache_times:
            return False
        elapsed = time.time() - self.cache_times[key]
        return elapsed < self.config.cache_ttl

    @lru_cache(maxsize=1000)
    def cached_matrix_product(self, A_hash: int, B_hash: int) -> torch.Tensor:
        """Cached matrix multiplication"""
        # This is a placeholder - in practice, we'd store the actual matrices
        # in a separate structure indexed by hash
        return torch.empty(0)  # Placeholder return

    def cached_belief_update(
        self, belief: torch.Tensor, observation: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """Cached belief update computation"""
        if not self.config.use_caching:
            return self._compute_belief_update(belief, observation, A)
        # Generate cache key
        key = self._cache_key(belief, observation, A)
        # Check cache
        if key in self.cache and self._is_cache_valid(key):
            self.cache_hits += 1
            return self.cache[key].clone()
        # Compute and cache
        self.cache_misses += 1
        result = self._compute_belief_update(belief, observation, A)
        # Update cache
        self.cache[key] = result.clone()
        self.cache_times[key] = time.time()
        # Evict old entries if cache is full
        if len(self.cache) > self.config.cache_size:
            self._evict_oldest()
        return result

    def _compute_belief_update(
        self, belief: torch.Tensor, observation: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """Actual belief update computation"""
        if observation.dim() == 0:
            # Scalar observation (single index)
            obs_idx = int(observation.item())
            likelihood = A[obs_idx]
        elif observation.dim() == 1 and observation.numel() == 1:
            # 1D tensor with single element
            obs_idx = int(observation.item())
            likelihood = A[obs_idx]
        elif observation.dim() == 1:
            # 1D observation vector (one-hot or soft)
            obs_idx = int(torch.argmax(observation).item())
            likelihood = A[obs_idx]
        else:
            # Multi-dimensional observation
            likelihood = torch.matmul(A.t(), observation.view(-1))
        posterior = likelihood * belief
        posterior = posterior / (posterior.sum() + self.config.eps)
        return posterior

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries"""
        # Find oldest entry
        oldest_key = min(self.cache_times, key=self.cache_times.get)
        # Remove from cache
        del self.cache[oldest_key]
        del self.cache_times[oldest_key]

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }


class GPUOptimizer:
    """
    GPU-specific optimizations for Active Inference.
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        # CUDA graphs cache
        self.graph_cache = {}

    def optimize_tensor_operations(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Move tensors to GPU and optimize memory layout"""
        optimized = []
        for tensor in tensors:
            # Move to GPU
            gpu_tensor = tensor.to(self.device)
            # Ensure contiguous memory layout
            if not gpu_tensor.is_contiguous():
                gpu_tensor = gpu_tensor.contiguous()
            optimized.append(gpu_tensor)
        return optimized

    @torch.amp.autocast('cuda')
    def mixed_precision_inference(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference with mixed precision"""
        if not self.config.use_mixed_precision:
            return model(inputs)
        with autocast():
            outputs = model(inputs)
        return outputs

    def create_cuda_graph(
        self, func: Callable, sample_inputs: tuple[torch.Tensor, ...]
    ) -> Callable:
        """Create CUDA graph for repeated operations"""
        if not self.config.use_cuda_graphs or self.device.type != "cuda":
            return func
        # Warm up
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                func(*sample_inputs)
        torch.cuda.current_stream().wait_stream(s)
        # Create graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outputs = func(*sample_inputs)

        def graphed_func(*inputs):
            # Ensure inputs have same shape
            for i, (inp, sample) in enumerate(zip(inputs, sample_inputs)):
                if inp.shape != sample.shape:
                    return func(*inputs)  # Fallback to regular execution
            # Copy inputs
            for inp, sample in zip(inputs, sample_inputs):
                sample.copy_(inp)
            # Replay graph
            g.replay()
            return outputs

        return graphed_func


class BatchProcessor:
    """
    Implements dynamic batching for Active Inference computations.
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.pending_requests = []
        self.results = {}

    def add_request(
        self, request_id: str, computation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """Add computation request to batch"""
        self.pending_requests.append(
            {
                "id": request_id,
                "computation": computation,
                "args": args,
                "kwargs": kwargs,
            }
        )
        # Process batch if full
        if len(self.pending_requests) >= self.config.batch_size:
            self.process_batch()

    def process_batch(self) -> None:
        """Process pending requests as a batch"""
        if not self.pending_requests:
            return
        # Group by computation type
        computation_groups = {}
        for request in self.pending_requests:
            comp_name = request["computation"].__name__
            if comp_name not in computation_groups:
                computation_groups[comp_name] = []
            computation_groups[comp_name].append(request)
        # Process each group
        for comp_name, requests in computation_groups.items():
            self._process_computation_group(requests)
        # Clear pending requests
        self.pending_requests = []

    def _process_computation_group(self, requests: List[Dict[str, Any]]) -> None:
        """Process a group of similar computations"""
        # Stack inputs
        computation = requests[0]["computation"]
        # Batch process
        if computation.__name__ == "belief_update":
            self._batch_belief_updates(requests)
        elif computation.__name__ == "expected_free_energy":
            self._batch_efe_computations(requests)
        else:
            # Fallback to individual processing
            for request in requests:
                result = request["computation"](*request["args"], **request["kwargs"])
                self.results[request["id"]] = result

    def _batch_belief_updates(self, requests: List[Dict[str, Any]]) -> None:
        """Batch process belief updates"""
        # Stack inputs
        beliefs = torch.stack([r["args"][0] for r in requests])
        observations = torch.stack([r["args"][1] for r in requests])
        A_matrices = torch.stack([r["args"][2] for r in requests])
        # Batch computation
        likelihoods = torch.matmul(A_matrices.transpose(1, 2), observations.unsqueeze(2)).squeeze()
        posteriors = likelihoods * beliefs
        posteriors = posteriors / (posteriors.sum(dim=1, keepdim=True) + self.config.eps)
        # Store results
        for i, request in enumerate(requests):
            self.results[request["id"]] = posteriors[i]

    def _batch_efe_computations(self, requests: List[Dict[str, Any]]) -> None:
        """Batch process EFE computations"""
        # Similar batching logic for EFE
        pass

    def get_result(self, request_id: str, timeout: float = 1.0) -> Optional[torch.Tensor]:
        """Get computation result"""
        start_time = time.time()
        while request_id not in self.results:
            if time.time() - start_time > timeout:
                return None
            # Process pending batch if waiting too long
            if self.pending_requests:
                self.process_batch()
            time.sleep(0.001)
        return self.results.pop(request_id)


class ComputationalOptimizer:
    """
    Main optimizer class that combines all optimization techniques.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None) -> None:
        self.config = config or OptimizationConfig()
        # Initialize components
        self.sparse_ops = SparseOperations(self.config)
        self.parallel = ParallelInference(self.config)
        self.cache = CachedInference(self.config)
        self.gpu_optimizer = GPUOptimizer(self.config)
        self.batch_processor = BatchProcessor(self.config)
        # Performance metrics
        self.timing_stats = {}

    def optimized_belief_update(
        self,
        belief: torch.Tensor,
        observation: torch.Tensor,
        A: torch.Tensor,
        use_sparse: Optional[bool] = None,
    ) -> torch.Tensor:
        """Optimized belief update using best available method"""
        start_time = time.time()
        # Determine if sparse operations should be used
        if use_sparse is None:
            sparsity = (A == 0).sum().float() / A.numel()
            use_sparse = sparsity > 0.5 and self.config.use_sparse_operations
        # Move to GPU if available
        if self.config.use_gpu:
            belief, observation, A = self.gpu_optimizer.optimize_tensor_operations(
                [belief, observation, A]
            )
        # Use appropriate method
        if use_sparse:
            sparse_A = self.sparse_ops.sparsify_tensor(A)
            result = self.sparse_ops.optimize_belief_update(sparse_A, observation, belief)
        else:
            result = self.cache.cached_belief_update(belief, observation, A)
        # Track timing
        elapsed = time.time() - start_time
        self._update_timing_stats("belief_update", elapsed)
        return result

    def optimized_action_selection(
        self,
        qs: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        num_actions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimized action selection with parallel EFE computation"""
        start_time = time.time()
        # Move to GPU
        if self.config.use_gpu:
            qs, A, B, C = self.gpu_optimizer.optimize_tensor_operations([qs, A, B, C])
        # Parallel EFE computation
        actions = list(range(num_actions))
        G_values = self.parallel.parallel_expected_free_energy(qs, A, B, C, actions)
        # Select action
        action_probs = F.softmax(-G_values, dim=0)
        # Track timing
        elapsed = time.time() - start_time
        self._update_timing_stats("action_selection", elapsed)
        return action_probs, G_values

    def _update_timing_stats(self, operation: str, elapsed: float) -> None:
        """Update timing statistics"""
        if operation not in self.timing_stats:
            self.timing_stats[operation] = {
                "count": 0,
                "total_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
            }
        stats = self.timing_stats[operation]
        stats["count"] += 1
        stats["total_time"] += elapsed
        stats["min_time"] = min(stats["min_time"], elapsed)
        stats["max_time"] = max(stats["max_time"], elapsed)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            "timing_stats": {},
            "cache_stats": self.cache.get_cache_stats(),
            "device": str(self.gpu_optimizer.device),
            "config": {
                "use_sparse": self.config.use_sparse_operations,
                "use_parallel": self.config.use_parallel_processing,
                "use_caching": self.config.use_caching,
                "use_gpu": self.config.use_gpu,
            },
        }
        # Process timing stats
        for op, stats in self.timing_stats.items():
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            report["timing_stats"][op] = {
                "count": stats["count"],
                "avg_time_ms": avg_time * 1000,
                "min_time_ms": stats["min_time"] * 1000,
                "max_time_ms": stats["max_time"] * 1000,
                "total_time_s": stats["total_time"],
            }
        return report

    def cleanup(self) -> None:
        """Cleanup resources"""
        self.parallel.cleanup()


# Example usage
if __name__ == "__main__":
    # Configuration
    config = OptimizationConfig(
        use_sparse_operations=True,
        use_parallel_processing=True,
        use_gpu=torch.cuda.is_available(),
        use_mixed_precision=True,
    )
    # Create optimizer
    optimizer = ComputationalOptimizer(config)
    # Example belief update
    belief = torch.tensor([0.25, 0.25, 0.25, 0.25])
    observation = torch.tensor([0.0, 1.0, 0.0])
    A = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.9, 0.0]])
    # Optimized computation
    updated_belief = optimizer.optimized_belief_update(belief, observation, A)
    print(f"Updated belief: {updated_belief}")
    # Performance report
    report = optimizer.get_performance_report()
    print(f"\nPerformance report: {report}")
    # Cleanup
    optimizer.cleanup()

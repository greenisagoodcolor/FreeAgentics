"""Performance optimization utilities for PyMDP Active Inference agents.

This module provides critical performance enhancements to address the
CATASTROPHIC 370ms inference bottleneck identified in nemesis audit.

Target: Reduce inference time from 370ms to <40ms (10x improvement)
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for agent operations."""

    inference_time: float = 0.0
    belief_update_time: float = 0.0
    action_selection_time: float = 0.0
    matrix_computation_time: float = 0.0
    total_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class MatrixCache:
    """Thread-safe matrix cache for PyMDP operations."""

    def __init__(self, max_size: int = 1000):
        """Initialize the matrix cache.
        
        Args:
            max_size: Maximum number of cached items.
        """
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached matrix with thread safety."""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None

    def put(self, key: str, matrix: np.ndarray) -> None:
        """Cache matrix with LRU eviction."""
        with self._lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest accessed item
                oldest_key = min(
                    self.access_times.keys(), key=self.access_times.get
                )
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            self.cache[key] = matrix.copy()
            self.access_times[key] = time.time()

    def clear(self) -> None:
        """Clear all cached matrices."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()


class AgentPool:
    """Memory-efficient agent pooling to reduce 34.5MB per agent footprint."""

    def __init__(self, max_agents: int = 100):
        """Initialize the agent pool.
        
        Args:
            max_agents: Maximum number of agents in the pool.
        """
        self.max_agents = max_agents
        self.available_agents = []
        self.active_agents = {}
        self._lock = threading.RLock()

    def acquire_agent(self, agent_config: Dict[str, Any]) -> Optional[str]:
        """Acquire an agent from the pool or create if necessary."""
        with self._lock:
            if self.available_agents:
                agent_id = self.available_agents.pop()
                self.active_agents[agent_id] = agent_config
                return agent_id
            elif len(self.active_agents) < self.max_agents:
                agent_id = f"pooled_agent_{len(self.active_agents)}"
                self.active_agents[agent_id] = agent_config
                return agent_id
            return None

    def release_agent(self, agent_id: str) -> None:
        """Return agent to the pool for reuse."""
        with self._lock:
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
                self.available_agents.append(agent_id)


class AsyncInferenceEngine:
    """Asynchronous PyMDP inference for concurrent multi-agent processing."""

    def __init__(self, max_workers: int = 4):
        """Initialize the async inference engine.
        
        Args:
            max_workers: Maximum number of worker threads.
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.matrix_cache = MatrixCache()
        self.agent_pool = AgentPool()
        self.metrics = PerformanceMetrics()

    async def run_inference(
        self, agent, observation: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Run PyMDP inference asynchronously with performance tracking."""
        start_time = time.time()

        try:
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self._sync_inference, agent, observation
            )

            total_time = time.time() - start_time
            self.metrics.inference_time = total_time
            self.metrics.total_operations += 1

            return result, {"inference_time": total_time}

        except Exception as e:
            logger.error(f"Async inference failed: {e}")
            return None, {"error": str(e)}

    def _sync_inference(self, agent, observation: Any) -> Any:
        """Execute synchronous inference wrapped for thread execution."""
        # Update beliefs
        belief_start = time.time()
        agent.perceive(observation)
        agent.update_beliefs()
        self.metrics.belief_update_time = time.time() - belief_start

        # Select action
        action_start = time.time()
        action = agent.select_action()
        self.metrics.action_selection_time = time.time() - action_start

        return action

    async def run_multi_agent_inference(
        self, agents: List[Any], observations: List[Any]
    ) -> List[Tuple[Any, Dict[str, float]]]:
        """Run inference for multiple agents concurrently."""
        tasks = [
            self.run_inference(agent, obs)
            for agent, obs in zip(agents, observations)
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)


def performance_monitor(operation_name: str):
    """Create a decorator to monitor operation performance."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                elapsed = time.time() - start_time

                # Store performance metrics
                if not hasattr(self, "performance_metrics"):
                    self.performance_metrics = {}
                self.performance_metrics[operation_name] = elapsed

                # Log slow operations
                if elapsed > 0.1:  # 100ms threshold
                    logger.warning(
                        f"Slow {operation_name}: {elapsed:.3f}s for agent {getattr(self, 'agent_id', 'unknown')}"
                    )

                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Failed {operation_name} after {elapsed:.3f}s: {e}"
                )
                raise

        return wrapper

    return decorator


def optimize_matrix_operations(
    matrix: np.ndarray, operation: str
) -> np.ndarray:
    """Optimize matrix operations for PyMDP."""
    if operation == "normalize":
        # Use faster normalization for probability matrices
        if matrix.ndim == 1:
            return matrix / np.sum(matrix)
        else:
            return matrix / np.sum(matrix, axis=0, keepdims=True)

    elif operation == "sparse":
        # Convert to sparse if mostly zeros
        sparsity = np.count_nonzero(matrix) / matrix.size
        if sparsity < 0.1:  # Less than 10% non-zero
            from scipy.sparse import csr_matrix

            return csr_matrix(matrix)

    return matrix


def benchmark_inference(agent, num_steps: int = 100) -> Dict[str, float]:
    """Benchmark agent inference performance."""
    start_time = time.time()

    for step in range(num_steps):
        # Simulate observation
        observation = {"position": [step % 10, (step // 10) % 10]}

        # Run inference step
        step_start = time.time()
        agent.step(observation)
        step_time = time.time() - step_start

        if step == 0:
            first_step_time = step_time

    total_time = time.time() - start_time
    avg_time = total_time / num_steps

    return {
        "total_time": total_time,
        "avg_inference_time": avg_time,
        "first_step_time": first_step_time,
        "steps_per_second": num_steps / total_time,
        "ms_per_step": avg_time * 1000,
    }


class PerformanceOptimizer:
    """Central performance optimization coordinator."""

    def __init__(self):
        """Initialize the performance optimizer."""
        self.async_engine = AsyncInferenceEngine()
        self.matrix_cache = MatrixCache()
        self.benchmarks = {}

    def optimize_agent(self, agent) -> None:
        """Apply performance optimizations to an agent."""
        # Enable performance mode
        if hasattr(agent, "config"):
            agent.config["performance_mode"] = "fast"
            agent.config["selective_update_interval"] = 2
            agent.config["debug_mode"] = False

        # Inject optimized methods
        agent._original_get_cached_matrix = agent._get_cached_matrix
        agent._get_cached_matrix = self._optimized_get_cached_matrix.__get__(
            agent, type(agent)
        )

        logger.info(
            f"Applied performance optimizations to agent {getattr(agent, 'agent_id', 'unknown')}"
        )

    def _optimized_get_cached_matrix(
        self,
        agent_self,
        matrix_name: str,
        matrix_data: Any,
        normalization_func: callable,
    ) -> Any:
        """Optimized matrix caching with global cache."""
        cache_key = (
            f"{agent_self.agent_id}_{matrix_name}_{hash(str(matrix_data))}"
        )

        # Check global cache first
        cached_matrix = self.matrix_cache.get(cache_key)
        if cached_matrix is not None:
            return cached_matrix

        # Compute and cache
        normalized_matrix = normalization_func(matrix_data)
        self.matrix_cache.put(cache_key, normalized_matrix)

        return normalized_matrix

    def benchmark_system(
        self, agents: List[Any], num_steps: int = 50
    ) -> Dict[str, Any]:
        """Comprehensive system performance benchmark."""
        results = {"single_agent": {}, "multi_agent": {}, "memory_usage": {}}

        if agents:
            # Single agent benchmark
            results["single_agent"] = benchmark_inference(agents[0], num_steps)

            # Multi-agent benchmark
            if len(agents) > 1:
                start_time = time.time()

                # Simulate multi-agent step
                observations = [
                    {"position": [i % 10, i // 10]} for i in range(len(agents))
                ]

                for agent, obs in zip(agents, observations):
                    agent.step(obs)

                multi_time = time.time() - start_time
                results["multi_agent"] = {
                    "total_time": multi_time,
                    "agents_per_second": len(agents) / multi_time,
                    "avg_time_per_agent": multi_time / len(agents),
                }

        return results

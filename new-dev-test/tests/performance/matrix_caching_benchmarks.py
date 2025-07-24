"""Comprehensive matrix caching performance benchmarks for PyMDP.

Evaluates the effectiveness of matrix caching strategies including transition matrices,
observation likelihoods, and intermediate results. Measures cache hit rates, memory
overhead, and computation speedup across different model sizes and update frequencies.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tests.performance.pymdp_benchmarks import BenchmarkSuite, PyMDPBenchmark

# Try to import PyMDP
try:
    import pymdp
    from pymdp import utils
    from pymdp.agent import Agent as PyMDPAgent

    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False


@dataclass
class CacheMetrics:
    """Container for cache performance metrics."""

    hit_rate: float
    miss_rate: float
    total_hits: int
    total_misses: int
    memory_overhead_mb: float
    speedup_factor: float


class MatrixCache:
    """Simple matrix cache implementation for benchmarking."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_counts = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get matrix from cache."""
        if key in self.cache:
            self.hits += 1
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, key: str, matrix: np.ndarray):
        """Store matrix in cache."""
        if len(self.cache) >= self.max_size:
            # Simple LRU eviction
            least_used = min(self.access_counts.items(), key=lambda x: x[1])
            del self.cache[least_used[0]]
            del self.access_counts[least_used[0]]

        self.cache[key] = matrix.copy()
        self.access_counts[key] = 0

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_counts.clear()
        self.hits = 0
        self.misses = 0

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        total_bytes = 0
        for matrix in self.cache.values():
            total_bytes += matrix.nbytes
        return total_bytes / (1024 * 1024)


class TransitionMatrixCachingBenchmark(PyMDPBenchmark):
    """Benchmark caching of transition matrices (B matrices)."""

    def __init__(
        self,
        state_size: int = 25,
        num_actions: int = 4,
        cache_enabled: bool = True,
    ):
        super().__init__(f"transition_matrix_caching_{'enabled' if cache_enabled else 'disabled'}")
        self.state_size = state_size
        self.num_actions = num_actions
        self.cache_enabled = cache_enabled
        self.cache = MatrixCache() if cache_enabled else None
        self.B_matrices = None
        self.computation_times = []

    def setup(self):
        """Initialize transition matrices."""
        if not PYMDP_AVAILABLE:
            return

        # Create transition matrices for all actions
        num_states = [self.state_size] * 2
        self.B_matrices = utils.random_B_matrix(num_states, self.num_actions)

        if self.cache:
            self.cache.clear()

    def run_iteration(self) -> Dict[str, Any]:
        """Run transition matrix computations with/without caching."""
        if not PYMDP_AVAILABLE:
            raise ImportError("PyMDP is required for this benchmark")

        total_computation_time = 0
        matrix_operations = 0

        # Simulate multiple state transitions
        for _ in range(50):  # Multiple operations per iteration
            action = np.random.randint(0, self.num_actions)
            state_idx = np.random.randint(0, len(self.B_matrices))

            # Create cache key
            cache_key = f"B_{state_idx}_{action}"

            start_time = time.perf_counter()

            if self.cache_enabled and self.cache:
                # Try to get from cache
                cached_matrix = self.cache.get(cache_key)
                if cached_matrix is None:
                    # Compute and cache
                    result_matrix = self._compute_transition_result(state_idx, action)
                    self.cache.put(cache_key, result_matrix)
                else:
                    result_matrix = cached_matrix
            else:
                # Always compute
                result_matrix = self._compute_transition_result(state_idx, action)

            computation_time = (time.perf_counter() - start_time) * 1000
            total_computation_time += computation_time
            matrix_operations += 1

        # Calculate metrics
        if self.cache:
            cache_metrics = {
                "cache_hit_rate": self.cache.get_hit_rate(),
                "cache_hits": self.cache.hits,
                "cache_misses": self.cache.misses,
                "cache_memory_mb": self.cache.get_memory_usage_mb(),
            }
        else:
            cache_metrics = {
                "cache_hit_rate": 0.0,
                "cache_hits": 0,
                "cache_misses": matrix_operations,
                "cache_memory_mb": 0.0,
            }

        return {
            "avg_computation_time_ms": total_computation_time / matrix_operations,
            "total_matrix_operations": matrix_operations,
            **cache_metrics,
        }

    def _compute_transition_result(self, state_idx: int, action: int) -> np.ndarray:
        """Simulate expensive transition matrix computation."""
        B_matrix = self.B_matrices[state_idx][:, :, action]

        # Simulate some computational work (matrix operations)
        result = B_matrix.copy()
        for _ in range(10):  # Simulate multiple operations
            result = np.dot(result.T, result)
            result = result / np.sum(result, axis=1, keepdims=True)

        return result

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "state_size": self.state_size,
            "num_actions": self.num_actions,
            "cache_enabled": self.cache_enabled,
        }


class ObservationLikelihoodCachingBenchmark(PyMDPBenchmark):
    """Benchmark caching of observation likelihood computations."""

    def __init__(
        self,
        state_size: int = 30,
        num_modalities: int = 3,
        cache_enabled: bool = True,
    ):
        super().__init__(
            f"observation_likelihood_caching_{'enabled' if cache_enabled else 'disabled'}"
        )
        self.state_size = state_size
        self.num_modalities = num_modalities
        self.cache_enabled = cache_enabled
        self.cache = MatrixCache() if cache_enabled else None
        self.A_matrices = None

    def setup(self):
        """Initialize observation likelihood matrices."""
        if not PYMDP_AVAILABLE:
            return

        # Create observation matrices
        num_states = [self.state_size] * self.num_modalities
        num_obs = [self.state_size] * self.num_modalities
        self.A_matrices = utils.random_A_matrix(num_obs, num_states)

        if self.cache:
            self.cache.clear()

    def run_iteration(self) -> Dict[str, Any]:
        """Run likelihood computations with/without caching."""
        if not PYMDP_AVAILABLE:
            raise ImportError("PyMDP is required for this benchmark")

        total_computation_time = 0
        likelihood_operations = 0

        # Simulate multiple likelihood computations
        for _ in range(30):  # Multiple operations per iteration
            modality = np.random.randint(0, self.num_modalities)
            observation = np.random.randint(0, self.state_size)
            state_config = tuple(np.random.randint(0, self.state_size, self.num_modalities))

            # Create cache key
            cache_key = f"likelihood_{modality}_{observation}_{state_config}"

            start_time = time.perf_counter()

            if self.cache_enabled and self.cache:
                # Try to get from cache
                cached_likelihood = self.cache.get(cache_key)
                if cached_likelihood is None:
                    # Compute and cache
                    likelihood = self._compute_likelihood(modality, observation, state_config)
                    self.cache.put(cache_key, likelihood)
                else:
                    likelihood = cached_likelihood
            else:
                # Always compute
                likelihood = self._compute_likelihood(modality, observation, state_config)

            computation_time = (time.perf_counter() - start_time) * 1000
            total_computation_time += computation_time
            likelihood_operations += 1

        # Calculate metrics
        if self.cache:
            cache_metrics = {
                "cache_hit_rate": self.cache.get_hit_rate(),
                "cache_hits": self.cache.hits,
                "cache_misses": self.cache.misses,
                "cache_memory_mb": self.cache.get_memory_usage_mb(),
            }
        else:
            cache_metrics = {
                "cache_hit_rate": 0.0,
                "cache_hits": 0,
                "cache_misses": likelihood_operations,
                "cache_memory_mb": 0.0,
            }

        return {
            "avg_computation_time_ms": total_computation_time / likelihood_operations,
            "total_likelihood_operations": likelihood_operations,
            **cache_metrics,
        }

    def _compute_likelihood(
        self, modality: int, observation: int, state_config: Tuple[int, ...]
    ) -> np.ndarray:
        """Simulate likelihood computation."""
        A_matrix = self.A_matrices[modality]

        # Simulate expensive likelihood computation
        # Extract relevant slice and perform computations
        likelihood_vector = A_matrix[observation]

        # Simulate multiple operations
        for _ in range(5):
            likelihood_vector = np.exp(likelihood_vector)
            likelihood_vector = likelihood_vector / np.sum(likelihood_vector)
            likelihood_vector = np.log(likelihood_vector + 1e-16)

        return likelihood_vector

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "state_size": self.state_size,
            "num_modalities": self.num_modalities,
            "cache_enabled": self.cache_enabled,
        }


class IntermediateResultCachingBenchmark(PyMDPBenchmark):
    """Benchmark caching of intermediate computation results."""

    def __init__(self, complexity_level: int = 3, cache_enabled: bool = True):
        super().__init__(
            f"intermediate_result_caching_{'enabled' if cache_enabled else 'disabled'}"
        )
        self.complexity_level = complexity_level
        self.cache_enabled = cache_enabled
        self.cache = MatrixCache() if cache_enabled else None

    def setup(self):
        """Initialize benchmark."""
        if not PYMDP_AVAILABLE:
            return

        if self.cache:
            self.cache.clear()

    def run_iteration(self) -> Dict[str, Any]:
        """Run computations with intermediate result caching."""
        if not PYMDP_AVAILABLE:
            raise ImportError("PyMDP is required for this benchmark")

        total_computation_time = 0
        operations = 0

        # Simulate complex computational pipeline with reusable intermediate results
        for step in range(20):
            # Generate some input parameters
            input_size = 10 + self.complexity_level * 5
            param_set = np.random.randint(0, 5, 3)  # Limited parameter space for cache hits

            cache_key = f"intermediate_{param_set[0]}_{param_set[1]}_{param_set[2]}_{input_size}"

            start_time = time.perf_counter()

            if self.cache_enabled and self.cache:
                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is None:
                    # Compute and cache
                    result = self._compute_intermediate_result(input_size, param_set)
                    self.cache.put(cache_key, result)
                else:
                    result = cached_result
            else:
                # Always compute
                result = self._compute_intermediate_result(input_size, param_set)

            computation_time = (time.perf_counter() - start_time) * 1000
            total_computation_time += computation_time
            operations += 1

        # Calculate metrics
        if self.cache:
            cache_metrics = {
                "cache_hit_rate": self.cache.get_hit_rate(),
                "cache_hits": self.cache.hits,
                "cache_misses": self.cache.misses,
                "cache_memory_mb": self.cache.get_memory_usage_mb(),
            }
        else:
            cache_metrics = {
                "cache_hit_rate": 0.0,
                "cache_hits": 0,
                "cache_misses": operations,
                "cache_memory_mb": 0.0,
            }

        return {
            "avg_computation_time_ms": total_computation_time / operations,
            "total_operations": operations,
            **cache_metrics,
        }

    def _compute_intermediate_result(self, size: int, params: np.ndarray) -> np.ndarray:
        """Simulate expensive intermediate computation."""
        # Create a matrix computation that takes some time
        matrix = np.random.rand(size, size)

        # Apply transformations based on parameters
        for param in params:
            for _ in range(param + 1):
                matrix = np.dot(matrix, matrix.T)
                matrix = matrix / np.max(matrix)  # Normalize to prevent overflow

        # Final processing
        result = np.linalg.svd(matrix)[1]  # Singular values
        return result

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "complexity_level": self.complexity_level,
            "cache_enabled": self.cache_enabled,
        }


class CacheComparisonBenchmark(PyMDPBenchmark):
    """Compare performance with and without caching across different scenarios."""

    def __init__(self, scenario: str = "mixed_workload"):
        super().__init__(f"cache_comparison_{scenario}")
        self.scenario = scenario
        self.cached_benchmark = None
        self.uncached_benchmark = None

    def setup(self):
        """Initialize comparison benchmarks."""
        if not PYMDP_AVAILABLE:
            return

        if self.scenario == "transition_heavy":
            self.cached_benchmark = TransitionMatrixCachingBenchmark(
                state_size=20, cache_enabled=True
            )
            self.uncached_benchmark = TransitionMatrixCachingBenchmark(
                state_size=20, cache_enabled=False
            )
        elif self.scenario == "observation_heavy":
            self.cached_benchmark = ObservationLikelihoodCachingBenchmark(
                state_size=25, cache_enabled=True
            )
            self.uncached_benchmark = ObservationLikelihoodCachingBenchmark(
                state_size=25, cache_enabled=False
            )
        else:  # mixed_workload
            self.cached_benchmark = IntermediateResultCachingBenchmark(
                complexity_level=3, cache_enabled=True
            )
            self.uncached_benchmark = IntermediateResultCachingBenchmark(
                complexity_level=3, cache_enabled=False
            )

        self.cached_benchmark.setup()
        self.uncached_benchmark.setup()

    def run_iteration(self) -> Dict[str, Any]:
        """Run comparison between cached and uncached implementations."""
        if not PYMDP_AVAILABLE:
            raise ImportError("PyMDP is required for this benchmark")

        # Run cached version
        cached_result = self.cached_benchmark.run_iteration()

        # Run uncached version
        uncached_result = self.uncached_benchmark.run_iteration()

        # Calculate speedup
        cached_time = cached_result.get("avg_computation_time_ms", 1)
        uncached_time = uncached_result.get("avg_computation_time_ms", 1)
        speedup = uncached_time / cached_time if cached_time > 0 else 1.0

        return {
            "cached_avg_time_ms": cached_time,
            "uncached_avg_time_ms": uncached_time,
            "speedup_factor": speedup,
            "cache_hit_rate": cached_result.get("cache_hit_rate", 0.0),
            "cache_memory_mb": cached_result.get("cache_memory_mb", 0.0),
            "efficiency_gain": max(0, (uncached_time - cached_time) / uncached_time) * 100,
        }

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "cached_config": (
                self.cached_benchmark.get_configuration() if self.cached_benchmark else {}
            ),
            "uncached_config": (
                self.uncached_benchmark.get_configuration() if self.uncached_benchmark else {}
            ),
        }


def run_matrix_caching_benchmarks():
    """Run comprehensive matrix caching benchmark suite."""
    suite = BenchmarkSuite()

    print(f"\n{'=' * 70}")
    print("MATRIX CACHING PERFORMANCE BENCHMARK SUITE")
    print(f"{'=' * 70}")
    print("Testing cache effectiveness across different PyMDP operations")

    # Transition matrix caching
    suite.add_benchmark(TransitionMatrixCachingBenchmark(state_size=20, cache_enabled=True))
    suite.add_benchmark(TransitionMatrixCachingBenchmark(state_size=20, cache_enabled=False))
    suite.add_benchmark(TransitionMatrixCachingBenchmark(state_size=40, cache_enabled=True))
    suite.add_benchmark(TransitionMatrixCachingBenchmark(state_size=40, cache_enabled=False))

    # Observation likelihood caching
    suite.add_benchmark(
        ObservationLikelihoodCachingBenchmark(state_size=25, num_modalities=2, cache_enabled=True)
    )
    suite.add_benchmark(
        ObservationLikelihoodCachingBenchmark(state_size=25, num_modalities=2, cache_enabled=False)
    )
    suite.add_benchmark(
        ObservationLikelihoodCachingBenchmark(state_size=35, num_modalities=3, cache_enabled=True)
    )
    suite.add_benchmark(
        ObservationLikelihoodCachingBenchmark(state_size=35, num_modalities=3, cache_enabled=False)
    )

    # Intermediate result caching
    suite.add_benchmark(IntermediateResultCachingBenchmark(complexity_level=2, cache_enabled=True))
    suite.add_benchmark(IntermediateResultCachingBenchmark(complexity_level=2, cache_enabled=False))
    suite.add_benchmark(IntermediateResultCachingBenchmark(complexity_level=4, cache_enabled=True))
    suite.add_benchmark(IntermediateResultCachingBenchmark(complexity_level=4, cache_enabled=False))

    # Cache comparison benchmarks
    suite.add_benchmark(CacheComparisonBenchmark("transition_heavy"))
    suite.add_benchmark(CacheComparisonBenchmark("observation_heavy"))
    suite.add_benchmark(CacheComparisonBenchmark("mixed_workload"))

    # Run benchmarks with fewer iterations for comprehensive testing
    results = suite.run_all(iterations=40)

    # Analyze and save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite.save_results(f"matrix_caching_benchmark_results_{timestamp}.json")

    # Generate analysis report
    print(f"\n{'=' * 70}")
    print("MATRIX CACHING PERFORMANCE ANALYSIS")
    print(f"{'=' * 70}")

    cache_enabled_results = [r for r in results if "enabled" in r.name]
    cache_disabled_results = [r for r in results if "disabled" in r.name]
    comparison_results = [r for r in results if "comparison" in r.name]

    # Calculate overall cache effectiveness
    if cache_enabled_results and cache_disabled_results:
        avg_cached_time = np.mean([r.mean_time_ms for r in cache_enabled_results])
        avg_uncached_time = np.mean([r.mean_time_ms for r in cache_disabled_results])
        overall_speedup = avg_uncached_time / avg_cached_time if avg_cached_time > 0 else 1.0

        print("\nOverall Cache Performance:")
        print(f"  Average cached time: {avg_cached_time:.2f} ms")
        print(f"  Average uncached time: {avg_uncached_time:.2f} ms")
        print(f"  Overall speedup: {overall_speedup:.2f}x")

        if overall_speedup >= 2.0:
            print("  ✅ Excellent cache performance (>2x speedup)")
        elif overall_speedup >= 1.5:
            print("  ✅ Good cache performance (>1.5x speedup)")
        elif overall_speedup >= 1.2:
            print("  ⚠️  Moderate cache performance (>1.2x speedup)")
        else:
            print("  ❌ Poor cache performance (<1.2x speedup)")

    # Analyze cache hit rates
    hit_rates = []
    memory_usage = []

    for result in cache_enabled_results:
        if result.additional_metrics:
            hit_rate = result.additional_metrics.get("cache_hit_rate", 0)
            memory_mb = result.additional_metrics.get("cache_memory_mb", 0)
            if hit_rate > 0:
                hit_rates.append(hit_rate)
            if memory_mb > 0:
                memory_usage.append(memory_mb)

    if hit_rates:
        avg_hit_rate = np.mean(hit_rates) * 100
        print("\nCache Hit Rate Analysis:")
        print(f"  Average hit rate: {avg_hit_rate:.1f}%")
        print(f"  Hit rate range: {np.min(hit_rates) * 100:.1f}% - {np.max(hit_rates) * 100:.1f}%")

        if avg_hit_rate >= 80:
            print("  ✅ Excellent cache utilization (>80%)")
        elif avg_hit_rate >= 60:
            print("  ✅ Good cache utilization (>60%)")
        elif avg_hit_rate >= 40:
            print("  ⚠️  Moderate cache utilization (>40%)")
        else:
            print("  ❌ Poor cache utilization (<40%)")

    if memory_usage:
        avg_memory = np.mean(memory_usage)
        print("\nCache Memory Usage:")
        print(f"  Average memory overhead: {avg_memory:.1f} MB")
        print(f"  Memory range: {np.min(memory_usage):.1f} - {np.max(memory_usage):.1f} MB")

    # Analyze comparison results
    if comparison_results:
        print("\nScenario-Specific Analysis:")
        for result in comparison_results:
            if result.additional_metrics:
                speedup = result.additional_metrics.get("speedup_factor", 1.0)
                efficiency = result.additional_metrics.get("efficiency_gain", 0.0)
                scenario = result.additional_metrics.get("scenario", "unknown")

                print(f"  {scenario.replace('_', ' ').title()}:")
                print(f"    Speedup: {speedup:.2f}x")
                print(f"    Efficiency gain: {efficiency:.1f}%")

    return results


if __name__ == "__main__":
    run_matrix_caching_benchmarks()

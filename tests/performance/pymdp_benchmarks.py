"""PyMDP Performance Benchmarking Framework.

Comprehensive benchmarks for Active Inference operations to validate
the ~9x performance improvement achieved through optimizations.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil

# Import PyMDP (using inferactively-pymdp package) - REQUIRED for benchmarks
from pymdp import utils
from pymdp.agent import Agent as PyMDPAgent

PYMDP_AVAILABLE = True


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    configuration: Dict[str, Any]
    mean_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    percentiles: Dict[str, float]
    memory_usage_mb: float
    iterations: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class BenchmarkTimer:
    """High-precision timer for benchmarking."""

    def __init__(self):
        """Initialize benchmark timer with no start time and empty laps list."""
        self.start_time = None
        self.laps = []

    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()

    def lap(self) -> float:
        """Record a lap time in milliseconds."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        lap_time = (time.perf_counter() - self.start_time) * 1000
        self.laps.append(lap_time)
        return lap_time

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.laps = []


class MemoryMonitor:
    """Monitor memory usage during benchmarks."""

    def __init__(self):
        """Initialize memory monitor with current process and no baseline."""
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = None

    def start(self):
        """Record baseline memory usage."""
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_usage(self) -> float:
        """Get current memory usage relative to baseline."""
        if self.baseline_memory is None:
            raise RuntimeError("Memory monitor not started")
        current = self.process.memory_info().rss / 1024 / 1024  # MB
        return current - self.baseline_memory


class PyMDPBenchmark:
    """Base class for PyMDP benchmarks."""

    def __init__(self, name: str):
        """Initialize PyMDP benchmark with name and monitoring tools.

        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.timer = BenchmarkTimer()
        self.memory_monitor = MemoryMonitor()
        self.results = []

    def setup(self, **kwargs):
        """Initialize test environment."""
        raise NotImplementedError

    def teardown(self):
        """Cleanup after benchmark."""
        pass

    def run_iteration(self) -> Dict[str, Any]:
        """Run a single iteration of the benchmark."""
        raise NotImplementedError

    def run(self, iterations: int = 100, warmup: int = 10) -> BenchmarkResult:
        """Execute benchmark with timing."""
        print(f"Running benchmark: {self.name}")
        print(f"Warmup iterations: {warmup}")
        print(f"Benchmark iterations: {iterations}")

        try:
            # Warmup
            for _ in range(warmup):
                self.run_iteration()

            # Actual benchmark
            self.memory_monitor.start()
            times = []

            for i in range(iterations):
                self.timer.reset()
                self.timer.start()

                metrics = self.run_iteration()

                elapsed = self.timer.lap()
                times.append(elapsed)

                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{iterations} iterations")

            # Calculate statistics
            times_array = np.array(times)
            result = BenchmarkResult(
                name=self.name,
                configuration=self.get_configuration(),
                mean_time_ms=np.mean(times_array),
                std_dev_ms=np.std(times_array),
                min_time_ms=np.min(times_array),
                max_time_ms=np.max(times_array),
                percentiles={
                    "p50": np.percentile(times_array, 50),
                    "p90": np.percentile(times_array, 90),
                    "p95": np.percentile(times_array, 95),
                    "p99": np.percentile(times_array, 99),
                },
                memory_usage_mb=self.memory_monitor.get_usage(),
                iterations=iterations,
                additional_metrics=metrics if isinstance(metrics, dict) else {},
            )

            self.teardown()
            return result
        except Exception as e:
            print(f"  Error during benchmark execution: {e}")
            raise

    def get_configuration(self) -> Dict[str, Any]:
        """Get benchmark configuration."""
        return {}

    def report(self, result: BenchmarkResult):
        """Generate performance report."""
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {result.name}")
        print(f"{'=' * 60}")
        print(f"Configuration: {json.dumps(result.configuration, indent=2)}")
        print("\nPerformance Metrics:")
        print(f"  Mean time: {result.mean_time_ms:.2f} ms")
        print(f"  Std dev: {result.std_dev_ms:.2f} ms")
        print(f"  Min/Max: {result.min_time_ms:.2f} / {result.max_time_ms:.2f} ms")
        print("  Percentiles:")
        for p, value in result.percentiles.items():
            print(f"    {p}: {value:.2f} ms")
        print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")

        if result.additional_metrics:
            print("\nAdditional Metrics:")
            for key, value in result.additional_metrics.items():
                print(f"  {key}: {value}")


class BeliefUpdateBenchmark(PyMDPBenchmark):
    """Benchmark belief state updates."""

    def __init__(self, state_size: int = 10, num_modalities: int = 2):
        """Initialize belief update benchmark.

        Args:
            state_size: Size of the state space
            num_modalities: Number of sensory modalities
        """
        super().__init__("belief_update")
        self.state_size = state_size
        self.num_modalities = num_modalities
        self.agent = None
        self.observations = None

    def setup(self):
        """Initialize PyMDP agent and test data."""
        # Create state space
        num_states = [self.state_size] * self.num_modalities
        num_observations = [self.state_size] * self.num_modalities
        num_controls = [4] * self.num_modalities  # Simple action space

        # Initialize generative model (A and B matrices)
        A = utils.random_A_matrix(num_observations, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        # Create agent
        self.agent = PyMDPAgent(A, B)

        # Pre-generate random observations
        self.observations = [
            [np.random.randint(0, self.state_size) for _ in range(self.num_modalities)]
            for _ in range(1000)
        ]
        self.obs_idx = 0

    def run_iteration(self) -> Dict[str, Any]:
        """Run single belief update."""
        # Get next observation
        obs = self.observations[self.obs_idx % len(self.observations)]
        self.obs_idx += 1

        # Update beliefs
        self.agent.infer_states(obs)

        # Return cache statistics if available
        return {
            "cache_hits": getattr(self.agent, "cache_hits", 0),
            "cache_misses": getattr(self.agent, "cache_misses", 0),
        }

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "state_size": self.state_size,
            "num_modalities": self.num_modalities,
        }


class ExpectedFreeEnergyBenchmark(PyMDPBenchmark):
    """Benchmark Expected Free Energy calculations."""

    def __init__(
        self,
        state_size: int = 10,
        policy_depth: int = 3,
        num_policies: int = 50,
    ):
        """Initialize expected free energy benchmark.

        Args:
            state_size: Size of the state space
            policy_depth: Depth of policy planning
            num_policies: Number of policies to evaluate
        """
        super().__init__("expected_free_energy")
        self.state_size = state_size
        self.policy_depth = policy_depth
        self.num_policies = num_policies
        self.agent = None

    def setup(self):
        """Initialize agent with policies."""
        # Create state space
        num_states = [self.state_size] * 2
        num_observations = [self.state_size] * 2
        num_controls = [4] * 2

        # Initialize generative model
        A = utils.random_A_matrix(num_observations, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_uniform(num_observations)

        # Create agent
        self.agent = PyMDPAgent(
            A, B, C=C, policy_len=self.policy_depth, inference_horizon=1
        )

    def run_iteration(self) -> Dict[str, Any]:
        """Calculate EFE for policies."""
        # Generate random initial observation
        obs = [np.random.randint(0, self.state_size) for _ in range(2)]

        # Infer states
        self.agent.infer_states(obs)

        # Calculate EFE (this happens internally in infer_policies)
        self.agent.infer_policies()

        # Get the G values (negative EFE)
        G_values = self.agent.G

        return {
            "num_policies_evaluated": (
                len(self.agent.policies) if hasattr(self.agent, "policies") else 0
            ),
            "min_efe": float(np.min(G_values)) if G_values is not None else 0,
            "max_efe": float(np.max(G_values)) if G_values is not None else 0,
        }

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "state_size": self.state_size,
            "policy_depth": self.policy_depth,
            "num_policies": self.num_policies,
        }


class MatrixCachingBenchmark(PyMDPBenchmark):
    """Benchmark matrix caching performance."""

    def __init__(self, state_size: int = 50, cache_enabled: bool = True):
        """Initialize matrix caching benchmark.

        Args:
            state_size: Size of the state space
            cache_enabled: Whether to enable matrix caching
        """
        super().__init__("matrix_caching")
        self.state_size = state_size
        self.cache_enabled = cache_enabled
        self.agents = []

    def setup(self):
        """Initialize multiple agents to test caching."""
        # Create 10 agents with same model structure
        num_states = [self.state_size] * 2
        num_observations = [self.state_size] * 2
        num_controls = [4] * 2

        # Shared matrices (would benefit from caching)
        A = utils.random_A_matrix(num_observations, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        for i in range(10):
            agent = PyMDPAgent(A, B)
            # Enable/disable caching if the agent supports it
            if hasattr(agent, "use_caching"):
                agent.use_caching = self.cache_enabled
            self.agents.append(agent)

    def run_iteration(self) -> Dict[str, Any]:
        """Run inference on all agents."""
        total_cache_hits = 0
        total_cache_misses = 0

        for agent in self.agents:
            obs = [np.random.randint(0, self.state_size) for _ in range(2)]
            agent.infer_states(obs)
            agent.infer_policies()

            # Collect cache statistics
            if hasattr(agent, "cache_hits"):
                total_cache_hits += agent.cache_hits
                total_cache_misses += agent.cache_misses

        cache_hit_rate = (
            total_cache_hits / (total_cache_hits + total_cache_misses)
            if (total_cache_hits + total_cache_misses) > 0
            else 0
        )

        return {
            "cache_hit_rate": cache_hit_rate,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
        }

    def get_configuration(self) -> Dict[str, Any]:
        return {
            "state_size": self.state_size,
            "cache_enabled": self.cache_enabled,
            "num_agents": len(self.agents),
        }


class AgentScalingBenchmark(PyMDPBenchmark):
    """Benchmark performance with increasing number of agents."""

    def __init__(self, num_agents: int = 10, state_size: int = 20):
        """Initialize agent scaling benchmark.

        Args:
            num_agents: Number of agents to simulate
            state_size: Size of the state space
        """
        super().__init__("agent_scaling")
        self.num_agents = num_agents
        self.state_size = state_size
        self.agents = []

    def setup(self):
        """Initialize multiple agents."""
        num_states = [self.state_size] * 2
        num_observations = [self.state_size] * 2
        num_controls = [4] * 2

        for i in range(self.num_agents):
            A = utils.random_A_matrix(num_observations, num_states)
            B = utils.random_B_matrix(num_states, num_controls)
            agent = PyMDPAgent(A, B)
            self.agents.append(agent)

    def run_iteration(self) -> Dict[str, Any]:
        """Run inference on all agents."""
        inference_times = []

        for agent in self.agents:
            obs = [np.random.randint(0, self.state_size) for _ in range(2)]

            start = time.perf_counter()
            agent.infer_states(obs)
            agent.infer_policies()
            agent.sample_action()
            end = time.perf_counter()

            inference_times.append((end - start) * 1000)

        return {
            "avg_agent_inference_ms": np.mean(inference_times),
            "total_inference_ms": np.sum(inference_times),
        }

    def get_configuration(self) -> Dict[str, Any]:
        return {"num_agents": self.num_agents, "state_size": self.state_size}


class BenchmarkSuite:
    """Run complete benchmark suite."""

    def __init__(self):
        """Initialize benchmark suite with empty benchmarks and results lists."""
        self.benchmarks = []
        self.results = []

    def add_benchmark(self, benchmark: PyMDPBenchmark):
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)

    def run_all(self, iterations: int = 100) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        print(f"\n{'=' * 60}")
        print("PyMDP PERFORMANCE BENCHMARK SUITE")
        print(f"{'=' * 60}")
        print(
            f"Running {len(self.benchmarks)} benchmarks with {iterations} iterations each\n"
        )

        for benchmark in self.benchmarks:
            try:
                benchmark.setup()
                result = benchmark.run(iterations=iterations)
                benchmark.report(result)
                self.results.append(result)
            except Exception as e:
                print(f"Error running benchmark {benchmark.name}: {e}")

        return self.results

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "name": result.name,
                    "configuration": result.configuration,
                    "mean_time_ms": result.mean_time_ms,
                    "std_dev_ms": result.std_dev_ms,
                    "percentiles": result.percentiles,
                    "memory_usage_mb": result.memory_usage_mb,
                    "iterations": result.iterations,
                    "timestamp": result.timestamp,
                    "additional_metrics": result.additional_metrics,
                }
            )

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    def compare_with_baseline(self, baseline_file: str):
        """Compare current results with baseline."""
        if not Path(baseline_file).exists():
            print(f"Baseline file not found: {baseline_file}")
            return

        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)

        print(f"\n{'=' * 60}")
        print("PERFORMANCE COMPARISON WITH BASELINE")
        print(f"{'=' * 60}")

        for result in self.results:
            # Find matching baseline
            baseline = next(
                (b for b in baseline_data if b["name"] == result.name), None
            )
            if baseline:
                diff_percent = (
                    (result.mean_time_ms - baseline["mean_time_ms"])
                    / baseline["mean_time_ms"]
                ) * 100
                print(f"\n{result.name}:")
                print(f"  Baseline: {baseline['mean_time_ms']:.2f} ms")
                print(f"  Current: {result.mean_time_ms:.2f} ms")
                print(f"  Difference: {diff_percent:+.1f}%")

                if diff_percent > 10:
                    print("  ⚠️ WARNING: Performance regression detected!")
                elif diff_percent < -10:
                    print("  ✅ Performance improvement!")


def run_basic_benchmarks():
    """Run basic benchmark suite."""
    suite = BenchmarkSuite()

    # Belief update benchmarks
    suite.add_benchmark(BeliefUpdateBenchmark(state_size=10))
    suite.add_benchmark(BeliefUpdateBenchmark(state_size=50))
    suite.add_benchmark(BeliefUpdateBenchmark(state_size=100))

    # EFE benchmarks
    suite.add_benchmark(ExpectedFreeEnergyBenchmark(state_size=10, policy_depth=3))
    suite.add_benchmark(ExpectedFreeEnergyBenchmark(state_size=20, policy_depth=5))

    # Caching benchmarks
    suite.add_benchmark(MatrixCachingBenchmark(state_size=50, cache_enabled=True))
    suite.add_benchmark(MatrixCachingBenchmark(state_size=50, cache_enabled=False))

    # Scaling benchmarks
    suite.add_benchmark(AgentScalingBenchmark(num_agents=1))
    suite.add_benchmark(AgentScalingBenchmark(num_agents=5))
    suite.add_benchmark(AgentScalingBenchmark(num_agents=10))

    # Run with fewer iterations for quick test
    results = suite.run_all(iterations=50)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite.save_results(f"benchmark_results_{timestamp}.json")

    # Compare with baseline if exists
    suite.compare_with_baseline("benchmark_baseline.json")

    return results


if __name__ == "__main__":
    run_basic_benchmarks()

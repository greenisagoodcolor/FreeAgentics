"""
Performance benchmark tests comparing FreeAgentics PyMDP integration to baseline PyMDP.

These tests ensure our integration layer doesn't introduce significant overhead
compared to using PyMDP directly. Critical for VC demo to show we maintain
PyMDP's performance while adding our agent framework.
"""

import gc
import statistics
import time
from typing import Dict

import numpy as np
import pytest

# PyMDP imports
try:
    from pymdp.agent import Agent as PyMDPAgent

    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False
    pytest.skip("PyMDP not available for baseline comparison tests")

# FreeAgentics imports
from agents.base_agent import BasicExplorerAgent
from agents.pymdp_adapter import PyMDPCompatibilityAdapter


class TestPyMDPBaselineComparison:
    """
    Performance benchmarks comparing our implementation against baseline PyMDP.

    These tests validate that our integration maintains acceptable performance
    characteristics for real-time active inference applications.
    """

    @pytest.fixture
    def benchmark_models(self):
        """Standard models for benchmarking."""
        return {
            "small": {
                "num_states": 5,
                "num_obs": 5,
                "num_actions": 5,
                "A": lambda: np.eye(5)[np.newaxis, :, :],
                "B": lambda: np.eye(5)[np.newaxis, :, :].repeat(5, axis=0),
                "C": lambda: np.array([[3.0, 2.0, 1.0, 0.5, 0.0]]),
                "D": lambda: np.ones(5) / 5,
            },
            "medium": {
                "num_states": 25,
                "num_obs": 25,
                "num_actions": 10,
                "A": lambda: np.eye(25)[np.newaxis, :, :],
                "B": lambda: np.eye(25)[np.newaxis, :, :].repeat(10, axis=0),
                "C": lambda: np.random.rand(1, 25),
                "D": lambda: np.ones(25) / 25,
            },
            "large": {
                "num_states": 100,
                "num_obs": 100,
                "num_actions": 20,
                "A": lambda: np.eye(100)[np.newaxis, :, :],
                "B": lambda: np.eye(100)[np.newaxis, :, :].repeat(20, axis=0),
                "C": lambda: np.random.rand(1, 100),
                "D": lambda: np.ones(100) / 100,
            },
        }

    def _measure_operation_time(self, operation, num_iterations: int = 100) -> Dict[str, float]:
        """Measure operation timing statistics."""
        times = []

        # Warmup
        for _ in range(5):
            operation()

        # Actual measurement
        for _ in range(num_iterations):
            gc.collect()  # Minimize GC impact
            start = time.perf_counter()
            operation()
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "total": sum(times),
            "count": len(times),
        }

    def test_belief_update_performance_comparison(self, benchmark_models):
        """Compare belief update performance: baseline PyMDP vs our integration."""
        results = {}

        for model_name, model_config in benchmark_models.items():
            # Setup models
            A = model_config["A"]()
            B = model_config["B"]()
            C = model_config["C"]()
            D = model_config["D"]()

            # Baseline PyMDP
            def baseline_belief_update():
                agent = PyMDPAgent(A, B, C, D)
                obs = np.random.randint(0, model_config["num_obs"])
                agent.infer_states([obs])

            baseline_stats = self._measure_operation_time(baseline_belief_update, 50)

            # Our integration
            def our_belief_update():
                agent = BasicExplorerAgent(f"perf_test_{model_name}", (0, 0))
                # Manually set PyMDP agent for fair comparison
                agent.pymdp_agent = PyMDPAgent(A, B, C, D)
                obs = {
                    "position": (0, 0),
                    "surroundings": np.random.randint(0, 2, (3, 3)),
                }
                agent.perceive(obs)
                agent.update_beliefs()

            our_stats = self._measure_operation_time(our_belief_update, 50)

            # Calculate overhead
            overhead_ratio = our_stats["mean"] / baseline_stats["mean"]

            results[model_name] = {
                "baseline_mean": baseline_stats["mean"],
                "our_mean": our_stats["mean"],
                "overhead_ratio": overhead_ratio,
                "acceptable": overhead_ratio < 1.5,  # Max 50% overhead
            }

            # Assert performance is acceptable
            assert (
                overhead_ratio < 1.5
            ), f"{model_name} model: Our implementation is {overhead_ratio:.2f}x slower than baseline"

        # Print results
        print("\nBelief Update Performance Comparison:")
        print("-" * 60)
        for model, stats in results.items():
            print(
                f"{model:10} | Baseline: {stats['baseline_mean'] * 1000:.3f}ms | "
                f"Ours: {stats['our_mean'] * 1000:.3f}ms | "
                f"Overhead: {stats['overhead_ratio']:.2f}x"
            )

        return results

    def test_action_selection_performance_comparison(self, benchmark_models):
        """Compare action selection performance."""
        results = {}

        for model_name, model_config in benchmark_models.items():
            # Skip large model for action selection (too many policies)
            if model_name == "large":
                continue

            # Setup models
            A = model_config["A"]()
            B = model_config["B"]()
            C = model_config["C"]()
            D = model_config["D"]()

            # Baseline PyMDP
            def baseline_action_selection():
                agent = PyMDPAgent(A, B, C, D, planning_horizon=2)
                agent.infer_policies()
                action = agent.sample_action()
                return action

            baseline_stats = self._measure_operation_time(baseline_action_selection, 20)

            # Our integration with adapter
            def our_action_selection():
                agent = BasicExplorerAgent(f"action_test_{model_name}", (0, 0))
                agent.pymdp_agent = PyMDPAgent(A, B, C, D, planning_horizon=2)
                agent.pymdp_agent.infer_policies()
                action = agent.select_action()
                return action

            our_stats = self._measure_operation_time(our_action_selection, 20)

            # Calculate overhead
            overhead_ratio = our_stats["mean"] / baseline_stats["mean"]

            results[model_name] = {
                "baseline_mean": baseline_stats["mean"],
                "our_mean": our_stats["mean"],
                "overhead_ratio": overhead_ratio,
                "acceptable": overhead_ratio < 1.3,  # Max 30% overhead for action selection
            }

            # Assert performance is acceptable
            assert (
                overhead_ratio < 1.3
            ), f"{model_name} model: Action selection is {overhead_ratio:.2f}x slower than baseline"

        # Print results
        print("\nAction Selection Performance Comparison:")
        print("-" * 60)
        for model, stats in results.items():
            print(
                f"{model:10} | Baseline: {stats['baseline_mean'] * 1000:.3f}ms | "
                f"Ours: {stats['our_mean'] * 1000:.3f}ms | "
                f"Overhead: {stats['overhead_ratio']:.2f}x"
            )

        return results

    def test_full_inference_cycle_performance(self, benchmark_models):
        """Compare full active inference cycle performance."""
        results = {}

        model_config = benchmark_models["small"]  # Use small model for full cycle
        A = model_config["A"]()
        B = model_config["B"]()
        C = model_config["C"]()
        D = model_config["D"]()

        # Baseline PyMDP full cycle
        def baseline_full_cycle():
            agent = PyMDPAgent(A, B, C, D, planning_horizon=3)
            for _ in range(5):  # 5 steps
                obs = np.random.randint(0, model_config["num_obs"])
                agent.infer_states([obs])
                agent.infer_policies()
                agent.sample_action()
                # In real scenario, environment would update here

        baseline_stats = self._measure_operation_time(baseline_full_cycle, 10)

        # Our integration full cycle
        def our_full_cycle():
            agent = BasicExplorerAgent("full_cycle_test", (0, 0))
            for _ in range(5):  # 5 steps
                obs = {
                    "position": (0, 0),
                    "surroundings": np.random.randint(0, 2, (3, 3)),
                }
                agent.perceive(obs)
                agent.update_beliefs()
                agent.select_action()
                # Agent position would update in real scenario

        our_stats = self._measure_operation_time(our_full_cycle, 10)

        # Calculate overhead
        overhead_ratio = our_stats["mean"] / baseline_stats["mean"]

        results["full_cycle"] = {
            "baseline_mean": baseline_stats["mean"],
            "our_mean": our_stats["mean"],
            "overhead_ratio": overhead_ratio,
            "acceptable": overhead_ratio < 1.5,
        }

        # Print results
        print("\nFull Inference Cycle Performance:")
        print("-" * 60)
        print(
            f"Baseline: {baseline_stats['mean'] * 1000:.3f}ms | "
            f"Ours: {our_stats['mean'] * 1000:.3f}ms | "
            f"Overhead: {overhead_ratio:.2f}x"
        )

        # Assert acceptable performance
        assert (
            overhead_ratio < 1.5
        ), f"Full cycle overhead too high: {overhead_ratio:.2f}x slower than baseline"

        return results

    def test_memory_efficiency_comparison(self, benchmark_models):
        """Compare memory usage between baseline and our implementation."""
        import tracemalloc

        model_config = benchmark_models["medium"]
        A = model_config["A"]()
        B = model_config["B"]()
        C = model_config["C"]()
        D = model_config["D"]()

        # Baseline PyMDP memory usage
        gc.collect()
        tracemalloc.start()

        baseline_agents = []
        for i in range(10):
            agent = PyMDPAgent(A, B, C, D)
            agent.infer_states([0])
            baseline_agents.append(agent)

        baseline_current, baseline_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Our implementation memory usage
        gc.collect()
        tracemalloc.start()

        our_agents = []
        for i in range(10):
            agent = BasicExplorerAgent(f"mem_test_{i}", (0, 0))
            agent.perceive({"position": (0, 0), "surroundings": np.zeros((3, 3))})
            agent.update_beliefs()
            our_agents.append(agent)

        our_current, our_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate overhead
        memory_overhead_ratio = our_peak / baseline_peak

        print("\nMemory Usage Comparison (10 agents):")
        print("-" * 60)
        print(f"Baseline Peak: {baseline_peak / 1024 / 1024:.2f} MB")
        print(f"Our Peak: {our_peak / 1024 / 1024:.2f} MB")
        print(f"Memory Overhead: {memory_overhead_ratio:.2f}x")

        # Memory overhead should be reasonable (less than 2x)
        assert (
            memory_overhead_ratio < 2.0
        ), f"Memory overhead too high: {memory_overhead_ratio:.2f}x more than baseline"

        return {
            "baseline_peak_mb": baseline_peak / 1024 / 1024,
            "our_peak_mb": our_peak / 1024 / 1024,
            "overhead_ratio": memory_overhead_ratio,
        }

    def test_adapter_overhead_isolated(self):
        """Test overhead of just the adapter layer."""
        # Create simple PyMDP agent
        A = np.eye(3)[np.newaxis, :, :]
        B = np.eye(3)[np.newaxis, :, :].repeat(3, axis=0)
        C = np.array([[1.0, 0.0, 0.0]])
        D = np.array([0.33, 0.33, 0.34])

        pymdp_agent = PyMDPAgent(A, B, C, D)
        pymdp_agent.infer_policies()

        adapter = PyMDPCompatibilityAdapter()

        # Direct PyMDP action sampling
        def direct_sampling():
            return pymdp_agent.sample_action()

        direct_stats = self._measure_operation_time(direct_sampling, 1000)

        # Through adapter
        def adapter_sampling():
            return adapter.sample_action(pymdp_agent)

        adapter_stats = self._measure_operation_time(adapter_sampling, 1000)

        # Calculate overhead
        overhead_ratio = adapter_stats["mean"] / direct_stats["mean"]

        print("\nAdapter Layer Overhead:")
        print("-" * 60)
        print(f"Direct PyMDP: {direct_stats['mean'] * 1000000:.2f} μs")
        print(f"With Adapter: {adapter_stats['mean'] * 1000000:.2f} μs")
        print(f"Overhead: {overhead_ratio:.2f}x")

        # Adapter should add minimal overhead (less than 20%)
        assert overhead_ratio < 1.2, f"Adapter overhead too high: {overhead_ratio:.2f}x slower"

        return {
            "direct_mean_us": direct_stats["mean"] * 1000000,
            "adapter_mean_us": adapter_stats["mean"] * 1000000,
            "overhead_ratio": overhead_ratio,
        }

    def test_scalability_with_state_space_size(self):
        """Test how performance scales with state space size."""
        state_sizes = [5, 10, 25, 50, 100]
        results = []

        for num_states in state_sizes:
            # Create models
            A = np.eye(num_states)[np.newaxis, :, :]
            B = np.eye(num_states)[np.newaxis, :, :].repeat(min(num_states, 10), axis=0)
            C = np.random.rand(1, num_states)
            D = np.ones(num_states) / num_states

            # Time belief update
            def belief_update():
                agent = PyMDPAgent(A, B, C, D)
                obs = np.random.randint(0, num_states)
                agent.infer_states([obs])

            stats = self._measure_operation_time(belief_update, 20)

            results.append(
                {
                    "num_states": num_states,
                    "mean_time": stats["mean"],
                    "ops_per_second": 1.0 / stats["mean"],
                }
            )

        print("\nScalability Analysis:")
        print("-" * 60)
        print("States | Time (ms) | Ops/sec")
        print("-" * 60)
        for r in results:
            print(
                f"{r['num_states']:6} | {r['mean_time'] * 1000:9.3f} | {r['ops_per_second']:7.1f}"
            )

        # Check that scaling is reasonable (not exponential)
        time_ratio = results[-1]["mean_time"] / results[0]["mean_time"]
        size_ratio = results[-1]["num_states"] / results[0]["num_states"]

        # Time should scale sub-quadratically with state space size
        assert (
            time_ratio < size_ratio**2
        ), f"Performance scaling too poor: {time_ratio:.1f}x time for {size_ratio:.1f}x states"

        return results


class TestRealTimePerformanceRequirements:
    """Test that performance meets real-time requirements for VC demo."""

    def test_agent_response_time_under_load(self):
        """Test agent can maintain response times under realistic load."""
        # Create multiple agents
        num_agents = 10
        agents = []

        for i in range(num_agents):
            agent = BasicExplorerAgent(f"load_test_{i}", (i, i))
            agents.append(agent)

        # Simulate concurrent updates
        total_updates = 0
        start_time = time.perf_counter()

        for cycle in range(10):
            for agent in agents:
                obs = {
                    "position": agent.position,
                    "surroundings": np.random.randint(0, 2, (3, 3)),
                }
                agent.perceive(obs)
                agent.update_beliefs()
                agent.select_action()
                total_updates += 1

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_update = total_time / total_updates
        updates_per_second = total_updates / total_time

        print(f"\nReal-time Performance ({num_agents} agents, {total_updates} updates):")
        print(f"Average time per update: {avg_time_per_update * 1000:.3f}ms")
        print(f"Updates per second: {updates_per_second:.1f}")

        # For real-time operation, need at least 10 updates/second per agent
        required_updates_per_second = num_agents * 10
        assert (
            updates_per_second > required_updates_per_second
        ), f"Performance too low: {updates_per_second:.1f} updates/sec, need {required_updates_per_second}"

    def test_latency_percentiles(self):
        """Test latency percentiles for consistent performance."""
        agent = BasicExplorerAgent("latency_test", (0, 0))

        # Collect latency samples
        latencies = []

        for _ in range(1000):
            start = time.perf_counter()

            obs = {
                "position": (0, 0),
                "surroundings": np.random.randint(0, 2, (3, 3)),
            }
            agent.perceive(obs)
            agent.update_beliefs()
            agent.select_action()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        # Calculate percentiles
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        print("\nLatency Percentiles (ms):")
        print(f"P50: {p50:.3f}")
        print(f"P95: {p95:.3f}")
        print(f"P99: {p99:.3f}")

        # For good user experience
        assert p50 < 10, f"Median latency too high: {p50:.3f}ms"
        assert p95 < 50, f"P95 latency too high: {p95:.3f}ms"
        assert p99 < 100, f"P99 latency too high: {p99:.3f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

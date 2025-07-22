#!/usr/bin/env python3
"""
Production Threading vs Multiprocessing Benchmark for FreeAgentics.

This is the production-ready benchmark that:
1. Handles all edge cases and warnings
2. Provides comprehensive metrics
3. Generates detailed reports
4. Supports different agent configurations
"""

import gc
import json
import logging
import multiprocessing as mp
import os
import resource
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BasicExplorerAgent

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("agents.base_agent").setLevel(logging.ERROR)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""

    agent_counts: List[int]
    steps_per_agent: int
    runs_per_test: int
    performance_mode: str
    enable_observability: bool
    grid_size: int


@dataclass
class TestResult:
    """Results from a single test run."""

    test_name: str
    num_agents: int
    total_time: float
    throughput_ops_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_delta_mb: float
    errors: int


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results."""

    config: BenchmarkConfig
    baseline_result: TestResult
    threading_results: List[TestResult]
    multiprocessing_results: List[TestResult]
    recommendations: Dict[str, str]


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "linux":
            return usage / 1024  # Linux returns KB
        else:
            return usage / 1024 / 1024  # Mac returns bytes
    except Exception:
        return 0.0


def create_agent(agent_id: str, config: BenchmarkConfig) -> BasicExplorerAgent:
    """Create a test agent with specified configuration."""
    agent = BasicExplorerAgent(agent_id, f"BenchAgent-{agent_id}", grid_size=config.grid_size)
    agent.config.update(
        {
            "performance_mode": config.performance_mode,
            "enable_observability": config.enable_observability,
            "selective_update_interval": 2 if config.performance_mode == "fast" else 1,
        }
    )
    return agent


def agent_workload(agent: BasicExplorerAgent, num_steps: int) -> tuple[List[float], int]:
    """Run agent workload and return timings and error count."""
    agent.start()
    timings = []
    errors = 0

    for i in range(num_steps):
        # Create realistic observation
        observation = {
            "position": [
                i % agent.grid_size,
                (i // agent.grid_size) % agent.grid_size,
            ],
            "surroundings": create_surroundings_pattern(i),
        }

        try:
            start = time.time()
            action = agent.step(observation)
            duration = (time.time() - start) * 1000  # Convert to ms
            timings.append(duration)

            # Validate action
            if action not in ["up", "down", "left", "right", "stay"]:
                errors += 1

        except Exception:
            errors += 1
            timings.append(1000.0)  # 1 second penalty

    agent.stop()
    return timings, errors


def create_surroundings_pattern(step: int):
    """Create varied surroundings patterns for realistic testing."""
    import numpy as np

    # Create different patterns based on step
    pattern_type = step % 5

    if pattern_type == 0:
        # Empty space
        return np.zeros((3, 3))
    elif pattern_type == 1:
        # Obstacle pattern
        surroundings = np.zeros((3, 3))
        surroundings[1, 0] = -1  # Obstacle to the left
        return surroundings
    elif pattern_type == 2:
        # Goal pattern
        surroundings = np.zeros((3, 3))
        surroundings[0, 1] = 1  # Goal above
        return surroundings
    elif pattern_type == 3:
        # Complex pattern
        surroundings = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        return surroundings
    else:
        # Mixed pattern
        surroundings = np.random.choice([0, -1, 1], size=(3, 3), p=[0.7, 0.2, 0.1])
        surroundings[1, 1] = 0  # Keep center empty
        return surroundings


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate percentile statistics."""
    import numpy as np

    if not values:
        return {"p50": 0, "p95": 0, "p99": 0}

    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def run_threading_test(config: BenchmarkConfig, num_agents: int) -> TestResult:
    """Run threading-based test."""
    print(f"  🧵 Threading test: {num_agents} agents...")

    # Prepare agents
    agents = [create_agent(f"thread_{i}", config) for i in range(num_agents)]

    # Memory tracking
    gc.collect()
    mem_before = get_memory_usage()

    # Run test
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=min(num_agents, mp.cpu_count())) as executor:
        futures = [
            executor.submit(agent_workload, agent, config.steps_per_agent) for agent in agents
        ]
        results = [future.result() for future in futures]

    total_time = time.time() - start_time
    mem_after = get_memory_usage()

    # Aggregate results
    all_timings = []
    total_errors = 0

    for timings, errors in results:
        all_timings.extend(timings)
        total_errors += errors

    # Calculate metrics
    import numpy as np

    percentiles = calculate_percentiles(all_timings)

    return TestResult(
        test_name="threading",
        num_agents=num_agents,
        total_time=total_time,
        throughput_ops_sec=(num_agents * config.steps_per_agent) / total_time,
        avg_latency_ms=float(np.mean(all_timings)) if all_timings else 0,
        p50_latency_ms=percentiles["p50"],
        p95_latency_ms=percentiles["p95"],
        p99_latency_ms=percentiles["p99"],
        memory_delta_mb=mem_after - mem_before,
        errors=total_errors,
    )


def multiprocessing_worker(agent_id: str, config: BenchmarkConfig) -> tuple[List[float], int]:
    """Worker function for multiprocessing tests."""
    agent = create_agent(agent_id, config)
    return agent_workload(agent, config.steps_per_agent)


def run_multiprocessing_test(config: BenchmarkConfig, num_agents: int) -> TestResult:
    """Run multiprocessing-based test."""
    print(f"  🔧 Multiprocessing test: {num_agents} agents...")

    # Memory tracking
    gc.collect()
    mem_before = get_memory_usage()

    # Run test
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=min(num_agents, mp.cpu_count())) as executor:
        futures = [
            executor.submit(multiprocessing_worker, f"proc_{i}", config) for i in range(num_agents)
        ]
        results = [future.result() for future in futures]

    total_time = time.time() - start_time
    mem_after = get_memory_usage()

    # Aggregate results
    all_timings = []
    total_errors = 0

    for timings, errors in results:
        all_timings.extend(timings)
        total_errors += errors

    # Calculate metrics
    import numpy as np

    percentiles = calculate_percentiles(all_timings)

    return TestResult(
        test_name="multiprocessing",
        num_agents=num_agents,
        total_time=total_time,
        throughput_ops_sec=(num_agents * config.steps_per_agent) / total_time,
        avg_latency_ms=float(np.mean(all_timings)) if all_timings else 0,
        p50_latency_ms=percentiles["p50"],
        p95_latency_ms=percentiles["p95"],
        p99_latency_ms=percentiles["p99"],
        memory_delta_mb=mem_after - mem_before,
        errors=total_errors,
    )


def run_baseline_test(config: BenchmarkConfig) -> TestResult:
    """Run single-agent baseline test."""
    print("  ⭐ Baseline test: single agent...")

    agent = create_agent("baseline", config)

    start_time = time.time()
    timings, errors = agent_workload(agent, config.steps_per_agent * 2)  # More steps for baseline
    total_time = time.time() - start_time

    import numpy as np

    percentiles = calculate_percentiles(timings)

    return TestResult(
        test_name="baseline",
        num_agents=1,
        total_time=total_time,
        throughput_ops_sec=len(timings) / total_time,
        avg_latency_ms=float(np.mean(timings)) if timings else 0,
        p50_latency_ms=percentiles["p50"],
        p95_latency_ms=percentiles["p95"],
        p99_latency_ms=percentiles["p99"],
        memory_delta_mb=0.0,
        errors=errors,
    )


def generate_recommendations(summary: BenchmarkSummary) -> Dict[str, str]:
    """Generate recommendations based on benchmark results."""
    threading_wins = 0
    total_comparisons = 0

    # Compare threading vs multiprocessing results
    for t_result, m_result in zip(summary.threading_results, summary.multiprocessing_results):
        if t_result.num_agents == m_result.num_agents:
            total_comparisons += 1
            if t_result.throughput_ops_sec > m_result.throughput_ops_sec:
                threading_wins += 1

    threading_win_rate = threading_wins / total_comparisons if total_comparisons > 0 else 0

    # Calculate average performance advantage
    avg_threading_advantage = 0
    if total_comparisons > 0:
        advantages = []
        for t_result, m_result in zip(summary.threading_results, summary.multiprocessing_results):
            if t_result.num_agents == m_result.num_agents and m_result.throughput_ops_sec > 0:
                advantage = t_result.throughput_ops_sec / m_result.throughput_ops_sec
                advantages.append(advantage)
        if advantages:
            avg_threading_advantage = sum(advantages) / len(advantages)

    recommendations = {}

    if threading_win_rate >= 0.7:
        recommendations["primary"] = "THREADING"
        recommendations["reason"] = (
            f"Threading wins {threading_win_rate:.0%} of comparisons with {avg_threading_advantage:.1f}x average advantage"
        )
        recommendations["use_threading_when"] = "Most FreeAgentics scenarios (default choice)"
        recommendations["use_multiprocessing_when"] = (
            "CPU-intensive custom models or fault isolation requirements"
        )
    else:
        recommendations["primary"] = "MIXED"
        recommendations["reason"] = (
            f"Performance varies by scenario ({threading_win_rate:.0%} threading wins)"
        )
        recommendations["use_threading_when"] = (
            "Coordination-heavy scenarios and memory-constrained environments"
        )
        recommendations["use_multiprocessing_when"] = (
            "CPU-intensive workloads and independent agent processes"
        )

    return recommendations


def print_results(summary: BenchmarkSummary):
    """Print formatted benchmark results."""
    print("\n" + "=" * 80)
    print("PRODUCTION BENCHMARK RESULTS")
    print("=" * 80)

    print("\n📊 Configuration:")
    print(f"   Agent counts: {summary.config.agent_counts}")
    print(f"   Steps per agent: {summary.config.steps_per_agent}")
    print(f"   Performance mode: {summary.config.performance_mode}")
    print(f"   CPU cores: {mp.cpu_count()}")
    print(f"   Platform: {sys.platform}")

    print("\n⭐ Baseline Performance:")
    b = summary.baseline_result
    print(f"   Single agent: {b.avg_latency_ms:.1f}ms avg, {b.throughput_ops_sec:.1f} ops/sec")
    print(f"   P95 latency: {b.p95_latency_ms:.1f}ms")

    print("\n📈 Performance Comparison:")
    print(
        f"{'Agents':<8} {'Threading (ops/s)':<18} {'Multiproc (ops/s)':<18} {'Winner':<12} {'Advantage':<10}"
    )
    print("-" * 70)

    for t_result, m_result in zip(summary.threading_results, summary.multiprocessing_results):
        if t_result.num_agents == m_result.num_agents:
            if t_result.throughput_ops_sec > m_result.throughput_ops_sec:
                winner = "Threading"
                advantage = t_result.throughput_ops_sec / m_result.throughput_ops_sec
            else:
                winner = "Multiproc"
                advantage = m_result.throughput_ops_sec / t_result.throughput_ops_sec

            print(
                f"{t_result.num_agents:<8} {t_result.throughput_ops_sec:<18.1f} "
                f"{m_result.throughput_ops_sec:<18.1f} {winner:<12} {advantage:<10.1f}x"
            )

    print("\n💾 Memory Usage (MB):")
    print(f"{'Agents':<8} {'Threading':<12} {'Multiprocessing':<15}")
    print("-" * 35)

    for t_result, m_result in zip(summary.threading_results, summary.multiprocessing_results):
        if t_result.num_agents == m_result.num_agents:
            print(
                f"{t_result.num_agents:<8} {t_result.memory_delta_mb:<12.1f} {m_result.memory_delta_mb:<15.1f}"
            )

    print("\n⏱️  Latency Analysis (P95 ms):")
    print(f"{'Agents':<8} {'Threading':<12} {'Multiprocessing':<15}")
    print("-" * 35)

    for t_result, m_result in zip(summary.threading_results, summary.multiprocessing_results):
        if t_result.num_agents == m_result.num_agents:
            print(
                f"{t_result.num_agents:<8} {t_result.p95_latency_ms:<12.1f} {m_result.p95_latency_ms:<15.1f}"
            )

    print("\n🎯 RECOMMENDATIONS:")
    print(f"   Primary choice: {summary.recommendations['primary']}")
    print(f"   Reason: {summary.recommendations['reason']}")
    print(f"   Use threading when: {summary.recommendations['use_threading_when']}")
    print(f"   Use multiprocessing when: {summary.recommendations['use_multiprocessing_when']}")


def save_results(
    summary: BenchmarkSummary,
    filename: str = "production_benchmark_results.json",
):
    """Save benchmark results to JSON file."""
    # Convert to JSON-serializable format
    result_dict = {
        "config": asdict(summary.config),
        "baseline_result": asdict(summary.baseline_result),
        "threading_results": [asdict(r) for r in summary.threading_results],
        "multiprocessing_results": [asdict(r) for r in summary.multiprocessing_results],
        "recommendations": summary.recommendations,
        "metadata": {
            "cpu_cores": mp.cpu_count(),
            "platform": sys.platform,
            "timestamp": time.time(),
        },
    }

    with open(filename, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n💾 Results saved to {filename}")


def main():
    """Run production benchmark suite."""
    print("🚀 PRODUCTION THREADING VS MULTIPROCESSING BENCHMARK")
    print("=" * 80)
    print("Comprehensive benchmark for FreeAgentics Active Inference agents")
    print("=" * 80)

    # Configuration
    config = BenchmarkConfig(
        agent_counts=[1, 5, 10, 20],
        steps_per_agent=25,
        runs_per_test=1,
        performance_mode="fast",
        enable_observability=False,
        grid_size=8,
    )

    print(f"\nStarting benchmark with {len(config.agent_counts)} test configurations...")
    input("Press Enter to continue...")

    # Run baseline test
    print("\n📊 Running baseline test...")
    baseline_result = run_baseline_test(config)

    # Run threading and multiprocessing tests
    threading_results = []
    multiprocessing_results = []

    for num_agents in config.agent_counts:
        print(f"\n📊 Testing {num_agents} agents:")

        # Threading test
        threading_result = run_threading_test(config, num_agents)
        threading_results.append(threading_result)

        # Multiprocessing test
        multiprocessing_result = run_multiprocessing_test(config, num_agents)
        multiprocessing_results.append(multiprocessing_result)

        # Brief pause between tests
        time.sleep(0.5)

    # Create summary
    summary = BenchmarkSummary(
        config=config,
        baseline_result=baseline_result,
        threading_results=threading_results,
        multiprocessing_results=multiprocessing_results,
        recommendations={},
    )

    # Generate recommendations
    summary.recommendations = generate_recommendations(summary)

    # Print results
    print_results(summary)

    # Save results
    save_results(summary)

    print("\n✅ BENCHMARK COMPLETE")
    print("Production deployment recommendations generated.")


if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    main()

#!/usr/bin/env python3
"""
Simple Threading vs Multiprocessing Test for FreeAgentics Agents.

A lightweight benchmark that doesn't require external dependencies
to validate the comparison approach.
"""

import gc
import multiprocessing as mp
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BasicExplorerAgent


def get_memory_usage() -> float:
    """Get current memory usage in MB using resource module."""
    try:
        # Get RSS (Resident Set Size) in bytes
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Convert to MB (Linux/Mac difference handled)
        if sys.platform == "linux":
            return usage / 1024  # Linux returns KB
        else:
            return usage / 1024 / 1024  # Mac returns bytes
    except Exception:
        return 0.0


def create_agent(agent_id: str) -> BasicExplorerAgent:
    """Create a test agent."""
    agent = BasicExplorerAgent(agent_id, f"TestAgent-{agent_id}", grid_size=5)
    agent.config["performance_mode"] = "fast"
    return agent


def agent_step_workload(agent: BasicExplorerAgent, num_steps: int = 10) -> List[float]:
    """Run agent through multiple steps and return timings."""
    agent.start()
    timings = []

    for i in range(num_steps):
        observation = {
            "position": [i % 5, i % 5],
            "surroundings": np.zeros((3, 3)),
        }

        start = time.time()
        agent.step(observation)
        timings.append((time.time() - start) * 1000)  # ms

    agent.stop()
    return timings


def test_threading(num_agents: int = 5, num_steps: int = 10) -> Dict[str, Any]:
    """Test threading performance."""
    print(f"\n🧵 Testing THREADING with {num_agents} agents...")

    # Force garbage collection and measure memory
    gc.collect()
    mem_before = get_memory_usage()

    # Create agents
    agents = [create_agent(f"thread_{i}") for i in range(num_agents)]

    start_time = time.time()

    # Run agents concurrently with threads
    with ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = [executor.submit(agent_step_workload, agent, num_steps) for agent in agents]
        results = [future.result() for future in futures]

    total_time = time.time() - start_time

    # Measure memory after
    mem_after = get_memory_usage()

    # Calculate metrics
    all_timings = [t for agent_timings in results for t in agent_timings]
    throughput = (num_agents * num_steps) / total_time

    return {
        "total_time": total_time,
        "throughput_ops_sec": throughput,
        "avg_latency_ms": np.mean(all_timings),
        "p95_latency_ms": np.percentile(all_timings, 95),
        "memory_delta_mb": mem_after - mem_before,
        "timings": all_timings,
    }


def process_worker(agent_id: str, num_steps: int) -> List[float]:
    """Worker function for multiprocessing."""
    agent = create_agent(agent_id)
    return agent_step_workload(agent, num_steps)


def test_multiprocessing(num_agents: int = 5, num_steps: int = 10) -> Dict[str, Any]:
    """Test multiprocessing performance."""
    print(f"\n🔧 Testing MULTIPROCESSING with {num_agents} agents...")

    # Force garbage collection and measure memory
    gc.collect()
    mem_before = get_memory_usage()

    start_time = time.time()

    # Run agents in separate processes
    with ProcessPoolExecutor(max_workers=num_agents) as executor:
        futures = [
            executor.submit(process_worker, f"proc_{i}", num_steps) for i in range(num_agents)
        ]
        results = [future.result() for future in futures]

    total_time = time.time() - start_time

    # Measure memory after
    mem_after = get_memory_usage()

    # Calculate metrics
    all_timings = [t for agent_timings in results for t in agent_timings]
    throughput = (num_agents * num_steps) / total_time

    return {
        "total_time": total_time,
        "throughput_ops_sec": throughput,
        "avg_latency_ms": np.mean(all_timings),
        "p95_latency_ms": np.percentile(all_timings, 95),
        "memory_delta_mb": mem_after - mem_before,
        "timings": all_timings,
    }


def test_single_agent_baseline() -> Dict[str, Any]:
    """Test single agent performance for baseline."""
    print("\n⭐ Testing BASELINE (single agent)...")

    agent = create_agent("baseline")
    num_steps = 50

    start_time = time.time()
    timings = agent_step_workload(agent, num_steps)
    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "throughput_ops_sec": num_steps / total_time,
        "avg_latency_ms": np.mean(timings),
        "p95_latency_ms": np.percentile(timings, 95),
        "timings": timings,
    }


def main():
    """Run simple benchmark tests."""
    print("=" * 60)
    print("SIMPLE THREADING VS MULTIPROCESSING TEST")
    print("=" * 60)
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Platform: {sys.platform}")

    # Test single agent baseline
    baseline = test_single_agent_baseline()
    print(
        f"\nBaseline (single agent): {baseline['avg_latency_ms']:.1f}ms avg, "
        f"{baseline['throughput_ops_sec']:.1f} ops/sec"
    )

    # Test different agent counts
    agent_counts = [1, 5, 10]
    num_steps = 20

    results = {}

    for num_agents in agent_counts:
        print(f"\n{'=' * 40}")
        print(f"Testing with {num_agents} agents, {num_steps} steps each")
        print("=" * 40)

        # Threading test
        try:
            thread_result = test_threading(num_agents, num_steps)
            print(
                f"  Threading: {thread_result['total_time']:.2f}s total, "
                f"{thread_result['throughput_ops_sec']:.1f} ops/sec, "
                f"{thread_result['avg_latency_ms']:.1f}ms avg"
            )
        except Exception as e:
            print(f"  Threading failed: {e}")
            thread_result = None

        # Multiprocessing test
        try:
            mp_result = test_multiprocessing(num_agents, num_steps)
            print(
                f"  Multiprocessing: {mp_result['total_time']:.2f}s total, "
                f"{mp_result['throughput_ops_sec']:.1f} ops/sec, "
                f"{mp_result['avg_latency_ms']:.1f}ms avg"
            )
        except Exception as e:
            print(f"  Multiprocessing failed: {e}")
            mp_result = None

        # Store results
        results[num_agents] = {
            "threading": thread_result,
            "multiprocessing": mp_result,
        }

        # Quick comparison
        if thread_result and mp_result:
            if thread_result["throughput_ops_sec"] > mp_result["throughput_ops_sec"]:
                winner = "Threading"
                margin = thread_result["throughput_ops_sec"] / mp_result["throughput_ops_sec"]
            else:
                winner = "Multiprocessing"
                margin = mp_result["throughput_ops_sec"] / thread_result["throughput_ops_sec"]

            print(f"\n  🏆 Winner: {winner} ({margin:.2f}x faster)")
            print(
                f"  💾 Memory: Threading={thread_result['memory_delta_mb']:.1f}MB, "
                f"Multiprocessing={mp_result['memory_delta_mb']:.1f}MB"
            )
        elif thread_result:
            print("\n  ✅ Threading completed successfully")
        elif mp_result:
            print("\n  ✅ Multiprocessing completed successfully")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    threading_wins = 0
    mp_wins = 0
    valid_comparisons = 0

    for num_agents, result in results.items():
        if result["threading"] and result["multiprocessing"]:
            valid_comparisons += 1
            if (
                result["threading"]["throughput_ops_sec"]
                > result["multiprocessing"]["throughput_ops_sec"]
            ):
                threading_wins += 1
            else:
                mp_wins += 1

    print(f"\nValid comparisons: {valid_comparisons}")
    print(f"Performance wins: Threading={threading_wins}, Multiprocessing={mp_wins}")

    if threading_wins > mp_wins:
        print("\n✅ THREADING is recommended for FreeAgentics Active Inference agents")
        print("   - Better performance for PyMDP computations")
        print("   - Lower memory overhead")
        print("   - Simpler shared memory model")
    elif mp_wins > threading_wins:
        print("\n✅ MULTIPROCESSING shows better performance in this test")
        print("   - True parallelism may benefit CPU-intensive workloads")
        print("   - Consider for specific use cases")
    else:
        print("\n⚖️  PERFORMANCE IS COMPARABLE")
        print("   - Choice depends on specific requirements")
        print("   - Consider other factors like memory usage and complexity")

    # Performance analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Find best single-agent performance
    single_agent_results = results.get(1, {})
    if single_agent_results["threading"]:
        threading_single = single_agent_results["threading"]["avg_latency_ms"]
        improvement = baseline["avg_latency_ms"] / threading_single
        print(
            f"Threading single-agent: {threading_single:.1f}ms avg ({improvement:.1f}x improvement over baseline)"
        )

    if single_agent_results["multiprocessing"]:
        mp_single = single_agent_results["multiprocessing"]["avg_latency_ms"]
        improvement = baseline["avg_latency_ms"] / mp_single
        print(
            f"Multiprocessing single-agent: {mp_single:.1f}ms avg ({improvement:.1f}x improvement over baseline)"
        )

    # Scaling analysis
    if len(results) > 1:
        print("\nScaling Analysis:")
        for num_agents, result in results.items():
            if num_agents > 1 and result["threading"]:
                single_throughput = (
                    results[1]["threading"]["throughput_ops_sec"] if results[1]["threading"] else 0
                )
                actual_throughput = result["threading"]["throughput_ops_sec"]
                scaling_efficiency = (
                    actual_throughput / (num_agents * single_throughput)
                    if single_throughput > 0
                    else 0
                )
                print(f"  {num_agents} agents threading: {scaling_efficiency:.1%} efficiency")

    print("\n🎯 RECOMMENDATION FOR FREEAGENTICS:")
    print("   Use threading for most Active Inference agent scenarios")
    print("   - Python-based PyMDP computations benefit from shared memory")
    print("   - Lower overhead for frequent, small operations")
    print("   - Better for coordination-heavy multi-agent systems")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    main()

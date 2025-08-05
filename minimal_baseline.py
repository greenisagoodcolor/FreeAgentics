#!/usr/bin/env python3
"""
Minimal Performance Baseline for Developer Release

Captures only the most essential performance metrics without complex infrastructure.
Focus: Agent spawn time and memory usage (the two critical requirements from CLAUDE.md).
"""

import json
import time
import gc
from datetime import datetime
from pathlib import Path

# Import core components
from agents.base_agent import BasicExplorerAgent
from tests.benchmarks.test_pymdp_benchmark import PyMDPBenchmarkSuite, BenchmarkConfig


def run_minimal_baseline():
    """Run minimal baseline measurement focusing on critical metrics."""
    print("=" * 60)
    print("MINIMAL DEVELOPER BASELINE")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0-dev-minimal",
        "metrics": {},
        "status": "unknown",
    }

    # Test 1: Agent Spawn Performance (Critical Requirement: <50ms)
    print("\n1. Testing Agent Spawn Performance...")
    spawn_times = []

    for i in range(10):  # Quick test with 10 spawns
        gc.collect()  # Clean memory before test

        start_time = time.perf_counter()
        agent = BasicExplorerAgent(f"test_agent_{i}", (0, 0))
        end_time = time.perf_counter()

        spawn_time_ms = (end_time - start_time) * 1000
        spawn_times.append(spawn_time_ms)
        print(f"   Agent {i}: {spawn_time_ms:.1f}ms")

    # Calculate P95 (90th percentile for small sample)
    spawn_times.sort()
    p95_spawn = spawn_times[int(len(spawn_times) * 0.9)]
    avg_spawn = sum(spawn_times) / len(spawn_times)

    results["metrics"]["agent_spawn_avg_ms"] = avg_spawn
    results["metrics"]["agent_spawn_p95_ms"] = p95_spawn
    results["metrics"]["agent_spawn_max_ms"] = max(spawn_times)

    print(f"   Average: {avg_spawn:.1f}ms")
    print(f"   P95: {p95_spawn:.1f}ms")
    print(f"   Max: {max(spawn_times):.1f}ms")
    print(f"   Target: <50ms - {'✅ PASS' if p95_spawn < 50.0 else '❌ FAIL'}")

    # Test 2: Memory Usage (Critical Requirement: <34.5MB per agent)
    print("\n2. Testing Memory Usage...")

    try:
        import psutil

        process = psutil.Process()

        # Get baseline memory
        gc.collect()
        baseline_mb = process.memory_info().rss / 1024 / 1024

        # Create 5 agents and measure memory
        agents = []
        for i in range(5):
            agent = BasicExplorerAgent(f"memory_test_{i}", (i, i))
            agents.append(agent)

        # Force some agent activity
        for agent in agents:
            obs = {"position": (1, 1), "surroundings": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]}
            agent.perceive(obs)
            try:
                agent.update_beliefs()
            except Exception:
                pass  # Ignore errors, just testing memory

        # Measure memory after agent creation
        peak_mb = process.memory_info().rss / 1024 / 1024
        memory_used_mb = peak_mb - baseline_mb
        memory_per_agent_mb = memory_used_mb / 5

        results["metrics"]["baseline_memory_mb"] = baseline_mb
        results["metrics"]["peak_memory_mb"] = peak_mb
        results["metrics"]["memory_used_mb"] = memory_used_mb
        results["metrics"]["memory_per_agent_mb"] = memory_per_agent_mb

        print(f"   Baseline Memory: {baseline_mb:.1f}MB")
        print(f"   Peak Memory: {peak_mb:.1f}MB")
        print(f"   Memory Used: {memory_used_mb:.1f}MB")
        print(f"   Per Agent: {memory_per_agent_mb:.1f}MB")
        print(f"   Budget: <34.5MB - {'✅ PASS' if memory_per_agent_mb < 34.5 else '❌ FAIL'}")

    except Exception as e:
        print(f"   Memory test failed: {e}")
        results["metrics"]["memory_test_error"] = str(e)

    # Test 3: Quick PyMDP Benchmark (Optional)
    print("\n3. Testing PyMDP Performance...")
    try:
        suite = PyMDPBenchmarkSuite()
        config = BenchmarkConfig("quick_test", state_size=10, iterations=5)  # Very small test

        start_time = time.time()
        result = suite.benchmark_freeagentics_agent_spawn(config)
        test_duration = time.time() - start_time

        if result.success:
            results["metrics"]["pymdp_spawn_p95_ms"] = result.timing.p95_ms
            results["metrics"]["pymdp_spawn_avg_ms"] = result.timing.mean_ms
            print(f"   PyMDP Agent Spawn P95: {result.timing.p95_ms:.1f}ms")
            print(f"   PyMDP Agent Spawn Avg: {result.timing.mean_ms:.1f}ms")
        else:
            print(f"   PyMDP test failed: {result.error_message}")
            results["metrics"]["pymdp_test_error"] = result.error_message

        print(f"   Test Duration: {test_duration:.1f}s")

    except Exception as e:
        print(f"   PyMDP test failed: {e}")
        results["metrics"]["pymdp_test_error"] = str(e)

    # Overall Assessment
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)

    # Critical requirements check
    agent_spawn_pass = results["metrics"].get("agent_spawn_p95_ms", 1000) < 50.0
    memory_pass = results["metrics"].get("memory_per_agent_mb", 1000) < 34.5

    critical_pass = agent_spawn_pass and memory_pass

    results["status"] = "PASS" if critical_pass else "FAIL"

    print(f"\nCritical Requirements:")
    print(f"  Agent Spawn <50ms:    {'✅ PASS' if agent_spawn_pass else '❌ FAIL'}")
    print(f"  Memory <34.5MB:       {'✅ PASS' if memory_pass else '❌ FAIL'}")
    print(
        f"\nOverall Status:         {'✅ DEVELOPER READY' if critical_pass else '❌ NEEDS OPTIMIZATION'}"
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"minimal_baseline_{timestamp}.json")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline saved to: {results_file}")

    if critical_pass:
        print("\n🎉 Developer baseline established successfully!")
        print("System meets critical performance requirements for development use.")
    else:
        print("\n⚠️  Developer baseline shows performance issues.")
        print("Consider optimization before proceeding with development.")

    return results


if __name__ == "__main__":
    run_minimal_baseline()

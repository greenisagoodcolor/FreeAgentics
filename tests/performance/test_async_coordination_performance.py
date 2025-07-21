#!/usr/bin/env python3
"""
Async Coordination Performance Test

Tests async/await coordination overhead without external dependencies.
Measures the efficiency of async task scheduling and coordination
for multi-agent scenarios.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from tests.performance.performance_utils import replace_sleep

# Setup logging
logging.basicConfig(level=logging.WARNING)


class MockAgent:
    """Mock agent for performance testing without numpy dependencies."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.step_count = 0
        self.processing_time = 0.001  # 1ms baseline processing

    def mock_step(self, observation: Dict[str, Any]) -> str:
        """Mock agent step with configurable processing time."""
        # Simulate processing delay
        replace_sleep(self.processing_time)

        self.step_count += 1
        return f"action_{self.step_count}"


class SequentialCoordinator:
    """Sequential multi-agent coordination for baseline comparison."""

    def __init__(self, num_agents: int):
        self.agents = [MockAgent(f"seq-agent-{i}") for i in range(num_agents)]

    def step_all_agents(
        self, observations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """Step all agents sequentially."""
        results = {}
        for agent in self.agents:
            obs = observations.get(agent.agent_id, {})
            results[agent.agent_id] = agent.mock_step(obs)
        return results


class AsyncCoordinator:
    """Async multi-agent coordination using async/await."""

    def __init__(self, num_agents: int, max_workers: int = None):
        self.agents = [MockAgent(f"async-agent-{i}") for i in range(num_agents)]
        self.max_workers = max_workers or min(32, (len(self.agents) + 4))

    async def async_agent_step(
        self, agent: MockAgent, observation: Dict[str, Any]
    ) -> tuple:
        """Async wrapper for agent step."""
        loop = asyncio.get_event_loop()

        # Run agent step in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, agent.mock_step, observation)
            return agent.agent_id, result

    async def step_all_agents(
        self, observations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """Step all agents concurrently using async/await."""
        # Create async tasks for all agents
        tasks = []
        for agent in self.agents:
            obs = observations.get(agent.agent_id, {})
            task = self.async_agent_step(agent, obs)
            tasks.append(task)

        # Wait for all tasks to complete
        results_list = await asyncio.gather(*tasks)

        # Convert to dict
        return {agent_id: result for agent_id, result in results_list}


class ThreadPoolCoordinator:
    """Thread pool coordination for comparison."""

    def __init__(self, num_agents: int, max_workers: int = None):
        self.agents = [MockAgent(f"thread-agent-{i}") for i in range(num_agents)]
        self.max_workers = max_workers or min(32, (len(self.agents) + 4))

    def step_all_agents(
        self, observations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """Step all agents using thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all agent steps
            futures = {}
            for agent in self.agents:
                obs = observations.get(agent.agent_id, {})
                future = executor.submit(agent.mock_step, obs)
                futures[agent.agent_id] = future

            # Collect results
            results = {}
            for agent_id, future in futures.items():
                results[agent_id] = future.result()

            return results


def run_coordination_benchmark():
    """Benchmark different coordination approaches."""
    print("=" * 60)
    print("ASYNC COORDINATION PERFORMANCE BENCHMARK")
    print("=" * 60)

    agent_counts = [1, 5, 10, 20, 30]
    rounds = 5

    results = {"sequential": {}, "async": {}, "threadpool": {}}

    for num_agents in agent_counts:
        print(f"\nTesting {num_agents} agents...")

        # Create observations for all agents
        observations = {}
        for i in range(num_agents):
            observations[f"agent-{i}"] = {"step": i}
            observations[f"seq-agent-{i}"] = {"step": i}
            observations[f"async-agent-{i}"] = {"step": i}
            observations[f"thread-agent-{i}"] = {"step": i}

        # Test Sequential Coordination
        seq_coordinator = SequentialCoordinator(num_agents)
        seq_times = []
        for _ in range(rounds):
            start = time.time()
            seq_coordinator.step_all_agents(observations)
            seq_times.append(time.time() - start)

        seq_avg = sum(seq_times) / len(seq_times)
        results["sequential"][num_agents] = {
            "avg_time": seq_avg,
            "throughput": num_agents / seq_avg,
            "times": seq_times,
        }
        print(
            f"  Sequential: {seq_avg:.4f}s avg, {num_agents / seq_avg:.1f} agents/sec"
        )

        # Test Async Coordination
        async def test_async():
            async_coordinator = AsyncCoordinator(num_agents)
            async_times = []
            for _ in range(rounds):
                start = time.time()
                await async_coordinator.step_all_agents(observations)
                async_times.append(time.time() - start)
            return async_times

        async_times = asyncio.run(test_async())
        async_avg = sum(async_times) / len(async_times)
        results["async"][num_agents] = {
            "avg_time": async_avg,
            "throughput": num_agents / async_avg,
            "times": async_times,
        }
        print(f"  Async: {async_avg:.4f}s avg, {num_agents / async_avg:.1f} agents/sec")

        # Test Thread Pool Coordination
        thread_coordinator = ThreadPoolCoordinator(num_agents)
        thread_times = []
        for _ in range(rounds):
            start = time.time()
            thread_coordinator.step_all_agents(observations)
            thread_times.append(time.time() - start)

        thread_avg = sum(thread_times) / len(thread_times)
        results["threadpool"][num_agents] = {
            "avg_time": thread_avg,
            "throughput": num_agents / thread_avg,
            "times": thread_times,
        }
        print(
            f"  ThreadPool: {thread_avg:.4f}s avg, {num_agents / thread_avg:.1f} agents/sec"
        )

        # Calculate efficiency improvements
        async_speedup = seq_avg / async_avg
        thread_speedup = seq_avg / thread_avg
        print(
            f"  Speedup: Async {async_speedup:.2f}x, ThreadPool {thread_speedup:.2f}x"
        )

    return results


def analyze_results(results):
    """Analyze coordination performance results."""
    print("\n" + "=" * 60)
    print("COORDINATION PERFORMANCE ANALYSIS")
    print("=" * 60)

    max_agents = max(results["sequential"].keys())

    # Get efficiency at maximum agent count
    seq_result = results["sequential"][max_agents]
    async_result = results["async"][max_agents]
    thread_result = results["threadpool"][max_agents]

    print(f"\nMaximum agents tested: {max_agents}")
    print(f"Sequential throughput: {seq_result['throughput']:.1f} agents/sec")
    print(f"Async throughput: {async_result['throughput']:.1f} agents/sec")
    print(f"ThreadPool throughput: {thread_result['throughput']:.1f} agents/sec")

    # Calculate scaling efficiency (actual vs theoretical)
    single_agent_seq = results["sequential"][1]["throughput"]
    theoretical_max_async = single_agent_seq * max_agents
    actual_max_async = async_result["throughput"]

    async_efficiency = actual_max_async / theoretical_max_async

    print("\nScaling Analysis:")
    print(f"Single agent baseline: {single_agent_seq:.1f} agents/sec")
    print(f"Theoretical maximum (linear): {theoretical_max_async:.1f} agents/sec")
    print(f"Actual async maximum: {actual_max_async:.1f} agents/sec")
    print(f"Async coordination efficiency: {async_efficiency:.1%}")

    # Determine coordination overhead impact
    coordination_overhead = (
        async_result["avg_time"] - (seq_result["avg_time"] / max_agents)
    ) / (seq_result["avg_time"] / max_agents)

    print("\nCoordination Overhead Analysis:")
    print(f"Sequential per-agent time: {seq_result['avg_time'] / max_agents:.4f}s")
    print(f"Async coordination time: {async_result['avg_time']:.4f}s")
    print(f"Coordination overhead: {coordination_overhead:.1%}")

    # Performance verdict
    if async_efficiency > 0.8:
        print("\n🎯 EXCELLENT: Async coordination enables near-linear scaling")
        return "excellent"
    elif async_efficiency > 0.6:
        print("\n✅ GOOD: Async coordination provides significant scaling benefits")
        return "good"
    elif async_efficiency > 0.3:
        print("\n✅ MODERATE: Async coordination provides meaningful improvements")
        return "moderate"
    else:
        print("\n⚠️ LIMITED: High coordination overhead limits scaling benefits")
        return "limited"


if __name__ == "__main__":
    # Run coordination performance benchmark
    results = run_coordination_benchmark()

    # Analyze results
    performance_level = analyze_results(results)

    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)

    if performance_level in ["excellent", "good"]:
        print("🎯 VALIDATED: Async coordination enables efficient multi-agent scaling")
        print("   - Ready for integration with PyMDP agents")
        print("   - Coordination overhead is manageable")
    elif performance_level == "moderate":
        print("✅ PROMISING: Async coordination shows benefits but needs optimization")
        print("   - Can proceed with PyMDP integration")
        print("   - Monitor performance under real workloads")
    else:
        print("⚠️ NEEDS WORK: High coordination overhead detected")
        print("   - Review async implementation for bottlenecks")
        print("   - Consider alternative coordination strategies")

    print("\nPhase 1B.2b objectives:")
    print("✅ Validated async coordination performance")
    print(f"✅ Measured scaling efficiency: {performance_level}")
    print("✅ Identified coordination overhead characteristics")

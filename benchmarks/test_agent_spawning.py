#!/usr/bin/env python3
"""
Agent Spawning Performance Benchmarks
PERF-ENGINEER: Bryan Cantrill + Brendan Gregg Methodology
"""

import asyncio
import gc
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import psutil
import pytest

# Import agent classes
from agents.base_agent import BaseAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.planning_agent import PlanningAgent


class PerformanceMetrics:
    """Collect and analyze performance metrics."""

    def __init__(self):
        self.spawn_times: List[float] = []
        self.memory_usage: List[int] = []
        self.cpu_usage: List[float] = []

    def record_spawn(self, duration: float, memory: int, cpu: float):
        """Record spawn metrics."""
        self.spawn_times.append(duration)
        self.memory_usage.append(memory)
        self.cpu_usage.append(cpu)

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.spawn_times:
            return {}

        return {
            "spawn_time": {
                "mean": statistics.mean(self.spawn_times),
                "median": statistics.median(self.spawn_times),
                "p95": np.percentile(self.spawn_times, 95),
                "p99": np.percentile(self.spawn_times, 99),
                "min": min(self.spawn_times),
                "max": max(self.spawn_times),
                "stdev": statistics.stdev(self.spawn_times) if len(self.spawn_times) > 1 else 0,
            },
            "memory": {
                "mean": statistics.mean(self.memory_usage),
                "median": statistics.median(self.memory_usage),
                "p95": np.percentile(self.memory_usage, 95),
                "max": max(self.memory_usage),
            },
            "cpu": {"mean": statistics.mean(self.cpu_usage), "max": max(self.cpu_usage)},
        }


class AgentSpawnBenchmarks:
    """Comprehensive agent spawning benchmarks."""

    @pytest.fixture
    def metrics(self):
        """Provide metrics collector."""
        return PerformanceMetrics()

    @pytest.mark.benchmark(group="agent-spawn", min_rounds=5)
    def test_single_agent_spawn(self, benchmark, metrics):
        """Benchmark single agent spawning."""

        def spawn_agent():
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss
            start_cpu = psutil.cpu_percent(interval=0)

            # Create agent
            agent = BaseAgent(agent_id="test-agent", model="gpt-4", temperature=0.7)

            # Measure metrics
            duration = time.perf_counter() - start_time
            memory = psutil.Process().memory_info().rss - start_memory
            cpu = psutil.cpu_percent(interval=0) - start_cpu

            metrics.record_spawn(duration * 1000, memory, cpu)

            return agent

        # Run benchmark
        agent = benchmark(spawn_agent)
        assert agent is not None

        # Print summary
        summary = metrics.get_summary()
        print("\nSingle Agent Spawn Performance:")
        print(f"  Mean: {summary['spawn_time']['mean']:.2f}ms")
        print(f"  P95: {summary['spawn_time']['p95']:.2f}ms")
        print(f"  P99: {summary['spawn_time']['p99']:.2f}ms")

    @pytest.mark.benchmark(group="agent-spawn", min_rounds=5)
    def test_parallel_agent_spawn(self, benchmark, metrics):
        """Benchmark parallel agent spawning."""

        def spawn_agents_parallel(count: int = 10):
            agents = []

            with ThreadPoolExecutor(max_workers=4) as executor:
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss

                # Submit spawn tasks
                futures = []
                for i in range(count):
                    future = executor.submit(
                        BaseAgent, agent_id=f"agent-{i}", model="gpt-4", temperature=0.7
                    )
                    futures.append(future)

                # Collect results
                for future in futures:
                    agents.append(future.result())

                # Measure metrics
                total_duration = time.perf_counter() - start_time
                total_memory = psutil.Process().memory_info().rss - start_memory

                metrics.record_spawn(
                    total_duration * 1000 / count,  # Per-agent time
                    total_memory // count,  # Per-agent memory
                    psutil.cpu_percent(interval=0),
                )

            return agents

        # Run benchmark
        agents = benchmark(spawn_agents_parallel)
        assert len(agents) == 10

        # Print summary
        summary = metrics.get_summary()
        print("\nParallel Agent Spawn Performance (10 agents):")
        print(f"  Mean per agent: {summary['spawn_time']['mean']:.2f}ms")
        print(f"  Total memory: {summary['memory']['mean'] * 10 / 1024 / 1024:.1f}MB")

    @pytest.mark.benchmark(group="agent-spawn", min_rounds=3)
    def test_agent_spawn_scaling(self, benchmark):
        """Test how agent spawning scales."""

        scaling_results = []

        for count in [1, 10, 50, 100]:
            start_time = time.perf_counter()

            # Spawn agents
            agents = []
            for i in range(count):
                agent = BaseAgent(agent_id=f"scaling-agent-{i}", model="gpt-4", temperature=0.7)
                agents.append(agent)

            duration = time.perf_counter() - start_time
            per_agent_time = duration * 1000 / count

            scaling_results.append(
                {"count": count, "total_time": duration * 1000, "per_agent_time": per_agent_time}
            )

            # Cleanup
            del agents
            gc.collect()

        # Print scaling analysis
        print("\nAgent Spawn Scaling Analysis:")
        for result in scaling_results:
            print(
                f"  {result['count']} agents: "
                f"{result['total_time']:.1f}ms total, "
                f"{result['per_agent_time']:.2f}ms per agent"
            )

        # Check scaling efficiency
        base_time = scaling_results[0]["per_agent_time"]
        for result in scaling_results[1:]:
            efficiency = base_time / result["per_agent_time"] * 100
            print(f"  Scaling efficiency at {result['count']} agents: {efficiency:.1f}%")

    @pytest.mark.benchmark(group="agent-spawn", min_rounds=5)
    @pytest.mark.asyncio
    async def test_async_agent_spawn(self, benchmark):
        """Benchmark async agent spawning."""

        async def spawn_agent_async():
            start_time = time.perf_counter()

            # Simulate async agent creation
            agent = BaseAgent(agent_id="async-agent", model="gpt-4", temperature=0.7)

            # Simulate async initialization
            await asyncio.sleep(0.001)  # Minimal async operation

            duration = time.perf_counter() - start_time
            return agent, duration * 1000

        # Run benchmark
        result = await benchmark(spawn_agent_async)
        agent, duration = result

        assert agent is not None
        print(f"\nAsync Agent Spawn: {duration:.2f}ms")

    @pytest.mark.benchmark(group="agent-spawn")
    def test_agent_type_comparison(self, benchmark):
        """Compare spawn times for different agent types."""

        agent_types = [
            ("BaseAgent", BaseAgent),
            ("PlanningAgent", PlanningAgent),
            ("KnowledgeAgent", KnowledgeAgent),
        ]

        results = {}

        for agent_name, agent_class in agent_types:
            spawn_times = []

            for i in range(10):
                start_time = time.perf_counter()

                agent = agent_class(
                    agent_id=f"{agent_name.lower()}-{i}", model="gpt-4", temperature=0.7
                )

                duration = (time.perf_counter() - start_time) * 1000
                spawn_times.append(duration)

                del agent

            results[agent_name] = {
                "mean": statistics.mean(spawn_times),
                "median": statistics.median(spawn_times),
                "min": min(spawn_times),
                "max": max(spawn_times),
            }

        # Print comparison
        print("\nAgent Type Spawn Comparison:")
        for agent_name, stats in results.items():
            print(f"  {agent_name}:")
            print(f"    Mean: {stats['mean']:.2f}ms")
            print(f"    Median: {stats['median']:.2f}ms")
            print(f"    Range: {stats['min']:.2f}ms - {stats['max']:.2f}ms")

    @pytest.mark.benchmark(group="agent-spawn")
    def test_memory_efficient_spawn(self, benchmark):
        """Test memory-efficient agent spawning patterns."""

        # Pattern 1: Regular spawning
        regular_memory = []
        for i in range(20):
            BaseAgent(agent_id=f"regular-{i}", model="gpt-4", temperature=0.7)
            regular_memory.append(psutil.Process().memory_info().rss)

        # Pattern 2: With object pooling simulation
        class AgentPool:
            def __init__(self, size=5):
                self.agents = []
                for i in range(size):
                    self.agents.append(
                        BaseAgent(agent_id=f"pooled-{i}", model="gpt-4", temperature=0.7)
                    )
                self.index = 0

            def get_agent(self):
                agent = self.agents[self.index % len(self.agents)]
                self.index += 1
                return agent

        pool = AgentPool()
        pooled_memory = []
        for i in range(20):
            pool.get_agent()
            pooled_memory.append(psutil.Process().memory_info().rss)

        # Compare memory usage
        regular_growth = regular_memory[-1] - regular_memory[0]
        pooled_growth = pooled_memory[-1] - pooled_memory[0]

        print("\nMemory Usage Comparison:")
        print(f"  Regular spawning growth: {regular_growth / 1024 / 1024:.2f}MB")
        print(f"  Pooled spawning growth: {pooled_growth / 1024 / 1024:.2f}MB")
        print(
            f"  Memory saved: {(regular_growth - pooled_growth) / 1024 / 1024:.2f}MB "
            f"({(1 - pooled_growth/regular_growth) * 100:.1f}%)"
        )


def run_spawn_benchmarks():
    """Run all agent spawning benchmarks."""
    pytest.main(
        [
            __file__,
            "-v",
            "--benchmark-only",
            "--benchmark-columns=min,max,mean,stddev,median,iqr,outliers,rounds",
            "--benchmark-sort=mean",
            "--benchmark-group-by=group",
            "--benchmark-warmup=on",
            "--benchmark-disable-gc",
        ]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PERF-ENGINEER: Agent Spawning Performance Benchmarks")
    print("Bryan Cantrill + Brendan Gregg Methodology")
    print("=" * 60)

    run_spawn_benchmarks()

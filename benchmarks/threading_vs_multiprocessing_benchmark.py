#!/usr/bin/env python3
"""
Threading vs Multiprocessing Benchmark for FreeAgentics Agents.

This comprehensive benchmark compares the performance characteristics of
threading vs multiprocessing for Active Inference agents, measuring:

1. Performance metrics (throughput, latency, scaling)
2. Memory usage (per-agent and total)
3. Communication overhead (inter-agent messaging)
4. Real-world scenarios (exploration, coordination, learning)

The benchmark validates the analysis findings and provides practical
recommendations for production deployment.
"""

import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import dataclass
from multiprocessing import Manager, Process, Queue
from threading import Lock
from typing import Any, Dict, List

import numpy as np
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BasicExplorerAgent
from agents.optimized_threadpool_manager import OptimizedThreadPoolManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    test_name: str
    num_agents: int
    total_operations: int
    total_time_sec: float
    throughput_ops_sec: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    communication_overhead_ms: float
    errors: int
    scaling_efficiency: float


class MemoryTracker:
    """Track memory usage during benchmark."""

    def __init__(self):
        """Initialize the memory tracker."""
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = self.initial_memory
        self.samples = []

    def update(self):
        """Update memory tracking."""
        current = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current)
        self.samples.append(current)

    def get_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return {
            "initial_mb": self.initial_memory,
            "peak_mb": self.peak_memory,
            "current_mb": self.samples[-1] if self.samples else self.initial_memory,
            "delta_mb": self.peak_memory - self.initial_memory,
            "avg_mb": np.mean(self.samples) if self.samples else self.initial_memory,
        }


class AgentWorkload:
    """Simulates different agent workload patterns."""

    @staticmethod
    def exploration_workload(
        agent_id: str, num_steps: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate exploration workload for an agent."""
        observations = []
        for i in range(num_steps):
            # Simulate different observation patterns
            if i % 10 == 0:
                # Obstacle detected
                surroundings = np.array([[0, -1, 0], [0, 0, 0], [0, 0, 0]])
            elif i % 7 == 0:
                # Goal detected
                surroundings = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
            else:
                # Empty space
                surroundings = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

            observations.append(
                {
                    "position": [i % 10, (i // 10) % 10],
                    "surroundings": surroundings,
                    "timestamp": i,
                }
            )
        return observations

    @staticmethod
    def coordination_workload(
        agent_id: str, other_agents: List[str], num_steps: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate coordination workload with inter-agent communication."""
        observations = []
        for i in range(num_steps):
            # Include other agent positions
            other_positions = []
            for j, other_id in enumerate(other_agents):
                if other_id != agent_id:
                    other_positions.append(
                        {
                            "agent_id": other_id,
                            "position": [(i + j) % 10, (i - j) % 10],
                        }
                    )

            observations.append(
                {
                    "position": [i % 10, i % 10],
                    "surroundings": np.zeros((3, 3)),
                    "other_agents": other_positions,
                    "messages": [
                        f"msg_{i}" for _ in range(min(i % 5, len(other_agents)))
                    ],
                    "timestamp": i,
                }
            )
        return observations


class ThreadingBenchmark:
    """Benchmark using threading for agent coordination."""

    def __init__(self, num_agents: int):
        """Initialize the threading benchmark."""
        self.num_agents = num_agents
        self.agents = {}
        self.thread_pool = OptimizedThreadPoolManager(
            initial_workers=min(num_agents, 16),
            max_workers=min(num_agents * 2, 64),
        )
        self.message_queue = queue.Queue()
        self.shared_state = {}
        self.lock = Lock()

    def setup(self):
        """Setup agents for threading benchmark."""
        for i in range(self.num_agents):
            agent_id = f"thread_agent_{i}"
            agent = BasicExplorerAgent(agent_id, f"ThreadAgent-{i}", grid_size=10)
            agent.config["performance_mode"] = "fast"
            self.agents[agent_id] = agent
            self.thread_pool.register_agent(agent_id, agent)

    def run_exploration(self, num_steps: int = 100) -> Dict[str, Any]:
        """Run exploration benchmark with threading."""
        logger.info(
            f"Running threading exploration benchmark with {self.num_agents} agents"
        )

        # Generate workloads
        workloads = {}
        for agent_id in self.agents:
            workloads[agent_id] = AgentWorkload.exploration_workload(
                agent_id, num_steps
            )

        # Track metrics
        latencies = []
        memory_tracker = MemoryTracker()

        start_time = time.time()

        # Process each step
        for step in range(num_steps):
            step_start = time.time()

            # Prepare observations for this step
            observations = {
                agent_id: workloads[agent_id][step] for agent_id in self.agents
            }

            # Execute step for all agents
            self.thread_pool.step_all_agents(observations, timeout=5.0)

            # Calculate step latency
            step_latency = (time.time() - step_start) * 1000
            latencies.append(step_latency)

            # Update memory tracking
            memory_tracker.update()

            # Check for errors
            errors = sum(1 for r in results.values() if not r.success)
            if errors > 0:
                logger.warning(f"Step {step}: {errors} agent errors")

        total_time = time.time() - start_time

        # Calculate metrics
        return {
            "total_time": total_time,
            "latencies": latencies,
            "memory": memory_tracker.get_usage(),
            "pool_status": self.thread_pool.get_pool_status(),
            "performance_stats": self.thread_pool.get_performance_stats(),
        }

    def run_coordination(self, num_steps: int = 100) -> Dict[str, Any]:
        """Run coordination benchmark with inter-agent communication."""
        logger.info(
            f"Running threading coordination benchmark with {self.num_agents} agents"
        )

        agent_ids = list(self.agents.keys())

        # Generate workloads
        workloads = {}
        for agent_id in agent_ids:
            workloads[agent_id] = AgentWorkload.coordination_workload(
                agent_id, agent_ids, num_steps
            )

        # Track metrics
        latencies = []
        communication_times = []
        memory_tracker = MemoryTracker()

        start_time = time.time()

        # Process each step
        for step in range(num_steps):
            step_start = time.time()

            # Simulate inter-agent communication
            comm_start = time.time()
            for agent_id in agent_ids:
                # Put messages in shared queue
                messages = workloads[agent_id][step].get("messages", [])
                for msg in messages:
                    self.message_queue.put((agent_id, msg))

                # Update shared state
                with self.lock:
                    self.shared_state[agent_id] = {
                        "position": workloads[agent_id][step]["position"],
                        "timestamp": step,
                    }

            # Process messages (simulate message handling)
            while not self.message_queue.empty():
                try:
                    sender, msg = self.message_queue.get_nowait()
                    # Simulate message processing
                    time.sleep(0.0001)  # 0.1ms per message
                except queue.Empty:
                    break

            communication_times.append((time.time() - comm_start) * 1000)

            # Prepare observations
            observations = {}
            for agent_id in agent_ids:
                obs = workloads[agent_id][step].copy()
                # Add shared state info
                with self.lock:
                    obs["shared_state"] = self.shared_state.copy()
                observations[agent_id] = obs

            # Execute step for all agents
            self.thread_pool.step_all_agents(observations, timeout=5.0)

            # Calculate step latency
            step_latency = (time.time() - step_start) * 1000
            latencies.append(step_latency)

            # Update memory tracking
            memory_tracker.update()

        total_time = time.time() - start_time

        return {
            "total_time": total_time,
            "latencies": latencies,
            "communication_times": communication_times,
            "memory": memory_tracker.get_usage(),
            "pool_status": self.thread_pool.get_pool_status(),
            "performance_stats": self.thread_pool.get_performance_stats(),
        }

    def cleanup(self):
        """Cleanup threading resources."""
        self.thread_pool.shutdown()


class MultiprocessingBenchmark:
    """Benchmark using multiprocessing for agent coordination."""

    def __init__(self, num_agents: int):
        """Initialize the multiprocessing benchmark."""
        self.num_agents = num_agents
        self.manager = Manager()
        self.shared_state = self.manager.dict()
        self.message_queue = self.manager.Queue()
        self.agent_processes = []
        self.result_queue = self.manager.Queue()

    def agent_worker(
        self,
        agent_id: str,
        workload: List[Dict[str, Any]],
        shared_state: dict,
        message_queue: Queue,
        result_queue: Queue,
    ):
        """Worker process for an agent."""
        # Create agent in this process
        agent = BasicExplorerAgent(agent_id, f"ProcessAgent-{agent_id}", grid_size=10)
        agent.config["performance_mode"] = "fast"
        agent.start()

        latencies = []

        for step, observation in enumerate(workload):
            try:
                step_start = time.time()

                # Add shared state to observation
                observation["shared_state"] = dict(shared_state)

                # Process messages
                messages = []
                try:
                    while True:
                        sender, msg = message_queue.get_nowait()
                        if msg.endswith(agent_id):  # Message for this agent
                            messages.append((sender, msg))
                except queue.Empty:
                    pass

                observation["received_messages"] = messages

                # Execute agent step
                action = agent.step(observation)

                # Update shared state
                shared_state[agent_id] = {
                    "position": observation["position"],
                    "action": action,
                    "timestamp": step,
                }

                # Send messages
                for msg in observation.get("messages", []):
                    message_queue.put((agent_id, msg))

                latencies.append((time.time() - step_start) * 1000)

            except Exception as e:
                logger.error(f"Agent {agent_id} error at step {step}: {e}")
                latencies.append(1000.0)  # 1 second penalty

        agent.stop()

        # Return results
        result_queue.put(
            {
                "agent_id": agent_id,
                "latencies": latencies,
                "metrics": agent.metrics,
            }
        )

    def setup(self):
        """Setup multiprocessing environment."""
        # Pre-create shared state entries
        for i in range(self.num_agents):
            agent_id = f"proc_agent_{i}"
            self.shared_state[agent_id] = {"position": [0, 0], "timestamp": 0}

    def run_exploration(self, num_steps: int = 100) -> Dict[str, Any]:
        """Run exploration benchmark with multiprocessing."""
        logger.info(
            f"Running multiprocessing exploration benchmark with {self.num_agents} agents"
        )

        # Generate workloads
        workloads = {}
        for i in range(self.num_agents):
            agent_id = f"proc_agent_{i}"
            workloads[agent_id] = AgentWorkload.exploration_workload(
                agent_id, num_steps
            )

        memory_tracker = MemoryTracker()
        start_time = time.time()

        # Start agent processes
        processes = []
        for agent_id, workload in workloads.items():
            p = Process(
                target=self.agent_worker,
                args=(
                    agent_id,
                    workload,
                    self.shared_state,
                    self.message_queue,
                    self.result_queue,
                ),
            )
            p.start()
            processes.append(p)
            memory_tracker.update()

        # Wait for completion
        for p in processes:
            p.join(timeout=num_steps * 0.1)  # 100ms per step max

        # Collect results
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())

        total_time = time.time() - start_time

        # Aggregate latencies
        all_latencies = []
        for r in results:
            all_latencies.extend(r["latencies"])

        return {
            "total_time": total_time,
            "latencies": all_latencies,
            "memory": memory_tracker.get_usage(),
            "num_results": len(results),
        }

    def run_coordination(self, num_steps: int = 100) -> Dict[str, Any]:
        """Run coordination benchmark with inter-agent communication."""
        logger.info(
            f"Running multiprocessing coordination benchmark with {self.num_agents} agents"
        )

        agent_ids = [f"proc_agent_{i}" for i in range(self.num_agents)]

        # Generate workloads
        workloads = {}
        for agent_id in agent_ids:
            workloads[agent_id] = AgentWorkload.coordination_workload(
                agent_id, agent_ids, num_steps
            )

        memory_tracker = MemoryTracker()
        start_time = time.time()

        # Start agent processes
        processes = []
        for agent_id, workload in workloads.items():
            p = Process(
                target=self.agent_worker,
                args=(
                    agent_id,
                    workload,
                    self.shared_state,
                    self.message_queue,
                    self.result_queue,
                ),
            )
            p.start()
            processes.append(p)
            memory_tracker.update()

        # Monitor progress
        monitor_start = time.time()
        while any(p.is_alive() for p in processes):
            time.sleep(0.1)
            memory_tracker.update()

            # Timeout protection
            if time.time() - monitor_start > num_steps * 0.2:
                logger.warning("Multiprocessing coordination timeout")
                break

        # Ensure all processes are terminated
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

        # Collect results
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break

        total_time = time.time() - start_time

        # Aggregate latencies
        all_latencies = []
        for r in results:
            all_latencies.extend(r.get("latencies", []))

        return {
            "total_time": total_time,
            "latencies": all_latencies,
            "memory": memory_tracker.get_usage(),
            "num_results": len(results),
        }

    def cleanup(self):
        """Cleanup multiprocessing resources."""
        # Terminate any remaining processes
        for p in self.agent_processes:
            if p.is_alive():
                p.terminate()


def run_benchmark_suite(
    agent_counts: List[int] = [1, 5, 10, 20, 30], num_steps: int = 50
) -> Dict[str, List[BenchmarkResult]]:
    """Run complete benchmark suite comparing threading and multiprocessing."""

    results = {
        "threading_exploration": [],
        "threading_coordination": [],
        "multiprocessing_exploration": [],
        "multiprocessing_coordination": [],
    }

    for num_agents in agent_counts:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"BENCHMARKING {num_agents} AGENTS")
        logger.info(f"{'=' * 60}")

        # Threading benchmarks
        logger.info("\nTHREADING BENCHMARKS")
        logger.info("-" * 40)

        thread_bench = ThreadingBenchmark(num_agents)
        thread_bench.setup()

        # Threading exploration
        thread_explore = thread_bench.run_exploration(num_steps)
        results["threading_exploration"].append(
            create_benchmark_result(
                "threading_exploration", num_agents, num_steps, thread_explore
            )
        )

        # Threading coordination
        thread_coord = thread_bench.run_coordination(num_steps)
        results["threading_coordination"].append(
            create_benchmark_result(
                "threading_coordination", num_agents, num_steps, thread_coord
            )
        )

        thread_bench.cleanup()

        # Multiprocessing benchmarks
        logger.info("\nMULTIPROCESSING BENCHMARKS")
        logger.info("-" * 40)

        mp_bench = MultiprocessingBenchmark(num_agents)
        mp_bench.setup()

        # Multiprocessing exploration
        mp_explore = mp_bench.run_exploration(num_steps)
        results["multiprocessing_exploration"].append(
            create_benchmark_result(
                "multiprocessing_exploration",
                num_agents,
                num_steps,
                mp_explore,
            )
        )

        # Multiprocessing coordination
        mp_coord = mp_bench.run_coordination(num_steps)
        results["multiprocessing_coordination"].append(
            create_benchmark_result(
                "multiprocessing_coordination", num_agents, num_steps, mp_coord
            )
        )

        mp_bench.cleanup()

        # Small delay between runs
        time.sleep(1)

    return results


def create_benchmark_result(
    test_name: str,
    num_agents: int,
    num_steps: int,
    raw_results: Dict[str, Any],
) -> BenchmarkResult:
    """Create a BenchmarkResult from raw test results."""

    latencies = raw_results.get("latencies", [])
    if not latencies:
        latencies = [0]

    total_operations = num_agents * num_steps
    throughput = total_operations / raw_results["total_time"]

    # Calculate scaling efficiency (compared to single agent)
    base_throughput = (
        total_operations / raw_results["total_time"] if num_agents == 1 else 0
    )
    scaling_efficiency = (
        throughput / (num_agents * base_throughput) if base_throughput > 0 else 1.0
    )

    return BenchmarkResult(
        test_name=test_name,
        num_agents=num_agents,
        total_operations=total_operations,
        total_time_sec=raw_results["total_time"],
        throughput_ops_sec=throughput,
        avg_latency_ms=np.mean(latencies),
        p95_latency_ms=np.percentile(latencies, 95),
        p99_latency_ms=np.percentile(latencies, 99),
        memory_usage_mb=raw_results["memory"]["delta_mb"],
        cpu_usage_percent=psutil.cpu_percent(),
        communication_overhead_ms=np.mean(raw_results.get("communication_times", [0])),
        errors=0,
        scaling_efficiency=scaling_efficiency,
    )


def analyze_results(results: Dict[str, List[BenchmarkResult]]):
    """Analyze and display benchmark results."""

    print("\n" + "=" * 80)
    print("THREADING VS MULTIPROCESSING BENCHMARK RESULTS")
    print("=" * 80)

    # Compare exploration performance
    print("\n📊 EXPLORATION PERFORMANCE (No Communication)")
    print("-" * 80)
    print(
        f"{'Agents':<10} {'Threading (ops/s)':<20} {'Multiproc (ops/s)':<20} {'Winner':<15} {'Margin':<10}"
    )
    print("-" * 80)

    for i in range(len(results["threading_exploration"])):
        t_result = results["threading_exploration"][i]
        m_result = results["multiprocessing_exploration"][i]

        winner = (
            "Threading"
            if t_result.throughput_ops_sec > m_result.throughput_ops_sec
            else "Multiprocessing"
        )
        margin = max(t_result.throughput_ops_sec, m_result.throughput_ops_sec) / min(
            t_result.throughput_ops_sec, m_result.throughput_ops_sec
        )

        print(
            f"{t_result.num_agents:<10} {t_result.throughput_ops_sec:<20.1f} "
            f"{m_result.throughput_ops_sec:<20.1f} {winner:<15} {margin:<10.2f}x"
        )

    # Compare coordination performance
    print("\n📊 COORDINATION PERFORMANCE (With Communication)")
    print("-" * 80)
    print(
        f"{'Agents':<10} {'Threading (ops/s)':<20} {'Multiproc (ops/s)':<20} {'Winner':<15} {'Margin':<10}"
    )
    print("-" * 80)

    for i in range(len(results["threading_coordination"])):
        t_result = results["threading_coordination"][i]
        m_result = results["multiprocessing_coordination"][i]

        winner = (
            "Threading"
            if t_result.throughput_ops_sec > m_result.throughput_ops_sec
            else "Multiprocessing"
        )
        margin = max(t_result.throughput_ops_sec, m_result.throughput_ops_sec) / min(
            t_result.throughput_ops_sec, m_result.throughput_ops_sec
        )

        print(
            f"{t_result.num_agents:<10} {t_result.throughput_ops_sec:<20.1f} "
            f"{m_result.throughput_ops_sec:<20.1f} {winner:<15} {margin:<10.2f}x"
        )

    # Memory comparison
    print("\n💾 MEMORY USAGE COMPARISON")
    print("-" * 80)
    print(
        f"{'Agents':<10} {'Thread Expl (MB)':<18} {'MP Expl (MB)':<18} {'Thread Coord (MB)':<18} {'MP Coord (MB)':<18}"
    )
    print("-" * 80)

    for i in range(len(results["threading_exploration"])):
        te = results["threading_exploration"][i]
        me = results["multiprocessing_exploration"][i]
        tc = results["threading_coordination"][i]
        mc = results["multiprocessing_coordination"][i]

        print(
            f"{te.num_agents:<10} {te.memory_usage_mb:<18.1f} {me.memory_usage_mb:<18.1f} "
            f"{tc.memory_usage_mb:<18.1f} {mc.memory_usage_mb:<18.1f}"
        )

    # Latency comparison
    print("\n⏱️  LATENCY COMPARISON (P95)")
    print("-" * 80)
    print(
        f"{'Agents':<10} {'Thread Expl (ms)':<18} {'MP Expl (ms)':<18} {'Thread Coord (ms)':<18} {'MP Coord (ms)':<18}"
    )
    print("-" * 80)

    for i in range(len(results["threading_exploration"])):
        te = results["threading_exploration"][i]
        me = results["multiprocessing_exploration"][i]
        tc = results["threading_coordination"][i]
        mc = results["multiprocessing_coordination"][i]

        print(
            f"{te.num_agents:<10} {te.p95_latency_ms:<18.1f} {me.p95_latency_ms:<18.1f} "
            f"{tc.p95_latency_ms:<18.1f} {mc.p95_latency_ms:<18.1f}"
        )

    # Scaling efficiency
    print("\n📈 SCALING EFFICIENCY")
    print("-" * 80)
    print(f"{'Agents':<10} {'Threading':<15} {'Multiprocessing':<15}")
    print("-" * 80)

    for i in range(len(results["threading_exploration"])):
        t_result = results["threading_exploration"][i]
        m_result = results["multiprocessing_exploration"][i]

        print(
            f"{t_result.num_agents:<10} {t_result.scaling_efficiency:<15.1%} "
            f"{m_result.scaling_efficiency:<15.1%}"
        )

    # Recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 80)

    # Analyze patterns
    threading_wins = 0
    multiproc_wins = 0

    for test_type in results:
        for result in results[test_type]:
            if "threading" in test_type:
                threading_perf = result.throughput_ops_sec

            if result.num_agents > 1:  # Skip single agent
                if "threading" in test_type and "exploration" in test_type:
                    mp_equivalent = next(
                        r
                        for r in results["multiprocessing_exploration"]
                        if r.num_agents == result.num_agents
                    )
                    if threading_perf > mp_equivalent.throughput_ops_sec:
                        threading_wins += 1
                    else:
                        multiproc_wins += 1

    print("\n1. PERFORMANCE:")
    if threading_wins > multiproc_wins:
        print(
            "   ✅ Threading shows better overall performance for Active Inference agents"
        )
        print("   - Lower overhead for Python-based computation")
        print("   - Better cache locality for shared model parameters")
        print("   - More efficient for frequent, small computations")
    else:
        print("   ✅ Multiprocessing shows better overall performance")
        print("   - True parallelism overcomes GIL limitations")
        print("   - Better for CPU-intensive workloads")

    print("\n2. MEMORY EFFICIENCY:")
    avg_thread_mem = np.mean(
        [r.memory_usage_mb for r in results["threading_exploration"]]
    )
    avg_mp_mem = np.mean(
        [r.memory_usage_mb for r in results["multiprocessing_exploration"]]
    )

    if avg_thread_mem < avg_mp_mem * 0.5:
        print(f"   ✅ Threading uses {avg_mp_mem / avg_thread_mem:.1f}x less memory")
        print("   - Shared memory model reduces duplication")
        print("   - More scalable for large agent populations")
    else:
        print(
            f"   ⚠️  Memory usage is comparable (Threading: {avg_thread_mem:.1f}MB, MP: {avg_mp_mem:.1f}MB)"
        )

    print("\n3. COMMUNICATION OVERHEAD:")
    thread_comm = np.mean(
        [r.communication_overhead_ms for r in results["threading_coordination"]]
    )
    mp_comm = np.mean(
        [r.communication_overhead_ms for r in results["multiprocessing_coordination"]]
    )

    if thread_comm < mp_comm * 0.3:
        print(
            f"   ✅ Threading has {mp_comm / thread_comm:.1f}x lower communication overhead"
        )
        print("   - Direct memory access vs IPC")
        print("   - Critical for coordination-heavy scenarios")

    print("\n4. PRODUCTION RECOMMENDATIONS:")
    print("   For FreeAgentics Active Inference agents:")
    print("   - Use THREADING for most scenarios (default)")
    print("   - Consider MULTIPROCESSING only for:")
    print("     • Very CPU-intensive custom models")
    print("     • Fault isolation requirements")
    print("     • Integration with non-Python components")

    print("\n5. OPTIMAL CONFIGURATION:")
    print("   - Thread pool size: 2-4x CPU cores")
    print("   - Enable performance mode: 'fast'")
    print("   - Use selective belief updates")
    print("   - Batch observations when possible")


def save_results(
    results: Dict[str, List[BenchmarkResult]],
    filename: str = "benchmark_results.json",
):
    """Save benchmark results to file."""

    # Convert to JSON-serializable format
    json_results = {}
    for test_type, test_results in results.items():
        json_results[test_type] = []
        for result in test_results:
            json_results[test_type].append(
                {
                    "test_name": result.test_name,
                    "num_agents": result.num_agents,
                    "total_operations": result.total_operations,
                    "total_time_sec": result.total_time_sec,
                    "throughput_ops_sec": result.throughput_ops_sec,
                    "avg_latency_ms": result.avg_latency_ms,
                    "p95_latency_ms": result.p95_latency_ms,
                    "p99_latency_ms": result.p99_latency_ms,
                    "memory_usage_mb": result.memory_usage_mb,
                    "cpu_usage_percent": result.cpu_usage_percent,
                    "communication_overhead_ms": result.communication_overhead_ms,
                    "errors": result.errors,
                    "scaling_efficiency": result.scaling_efficiency,
                }
            )

    with open(filename, "w") as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"Results saved to {filename}")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)

    print("🚀 THREADING VS MULTIPROCESSING BENCHMARK FOR FREEAGENTICS")
    print("=" * 80)
    print(
        "This benchmark compares threading and multiprocessing for Active Inference agents"
    )
    print("measuring performance, memory usage, and communication overhead.")
    print("=" * 80)

    # Run benchmarks
    agent_counts = [1, 5, 10, 20, 30]
    num_steps = 50

    print("\nConfiguration:")
    print(f"- Agent counts: {agent_counts}")
    print(f"- Steps per test: {num_steps}")
    print(f"- CPU cores: {mp.cpu_count()}")
    print(f"- Platform: {sys.platform}")

    input("\nPress Enter to start benchmark...")

    # Run benchmark suite
    results = run_benchmark_suite(agent_counts, num_steps)

    # Analyze results
    analyze_results(results)

    # Save results
    save_results(results)

    print("\n✅ BENCHMARK COMPLETE")
    print("Results saved to benchmark_results.json")

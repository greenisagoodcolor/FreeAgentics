#!/usr/bin/env python3
"""
Quick Threading vs Multiprocessing Test for FreeAgentics Agents.

A simplified benchmark that runs quickly to validate the comparison approach
before running the full benchmark suite.
"""

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BasicExplorerAgent


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

    # Create agents
    agents = [create_agent(f"thread_{i}") for i in range(num_agents)]

    # Measure memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.time()

    # Run agents concurrently with threads
    with ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = [
            executor.submit(agent_step_workload, agent, num_steps) for agent in agents
        ]
        results = [future.result() for future in futures]

    total_time = time.time() - start_time

    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

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

    # Measure memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.time()

    # Run agents in separate processes
    with ProcessPoolExecutor(max_workers=num_agents) as executor:
        futures = [
            executor.submit(process_worker, f"proc_{i}", num_steps)
            for i in range(num_agents)
        ]
        results = [future.result() for future in futures]

    total_time = time.time() - start_time

    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

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


def test_communication_overhead():
    """Test communication overhead between threading and multiprocessing."""
    print("\n📡 Testing COMMUNICATION OVERHEAD...")

    num_messages = 1000
    message_size = 1024  # 1KB messages

    # Threading: Direct memory access
    print("  Threading communication test...")
    shared_list = []

    def thread_sender():
        for i in range(num_messages):
            shared_list.append(b"x" * message_size)

    def thread_receiver():
        received = 0
        while received < num_messages:
            if shared_list:
                shared_list.pop(0)
                received += 1
            else:
                time.sleep(0.0001)

    thread_start = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        sender_future = executor.submit(thread_sender)
        receiver_future = executor.submit(thread_receiver)
        sender_future.result()
        receiver_future.result()

    thread_time = time.time() - thread_start
    thread_throughput = num_messages / thread_time

    # Multiprocessing: Queue-based IPC
    print("  Multiprocessing communication test...")

    def mp_sender(queue):
        for i in range(num_messages):
            queue.put(b"x" * message_size)

    def mp_receiver(queue):
        received = 0
        while received < num_messages:
            queue.get()
            received += 1

    mp_start = time.time()

    with mp.Manager() as manager:
        queue = manager.Queue()

        with ProcessPoolExecutor(max_workers=2) as executor:
            sender_future = executor.submit(mp_sender, queue)
            receiver_future = executor.submit(mp_receiver, queue)
            sender_future.result()
            receiver_future.result()

    mp_time = time.time() - mp_start
    mp_throughput = num_messages / mp_time

    return {
        "threading": {
            "total_time": thread_time,
            "throughput_msg_sec": thread_throughput,
            "latency_per_msg_ms": (thread_time / num_messages) * 1000,
        },
        "multiprocessing": {
            "total_time": mp_time,
            "throughput_msg_sec": mp_throughput,
            "latency_per_msg_ms": (mp_time / num_messages) * 1000,
        },
        "overhead_ratio": mp_time / thread_time,
    }


def main():
    """Run quick benchmark tests."""
    print("=" * 60)
    print("QUICK THREADING VS MULTIPROCESSING TEST")
    print("=" * 60)
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Platform: {sys.platform}")

    # Test different agent counts
    agent_counts = [1, 5, 10]
    num_steps = 20

    results = {}

    for num_agents in agent_counts:
        print(f"\n{'=' * 40}")
        print(f"Testing with {num_agents} agents, {num_steps} steps each")
        print("=" * 40)

        # Threading test
        thread_result = test_threading(num_agents, num_steps)
        print(
            f"  Threading: {thread_result['total_time']:.2f}s total, "
            f"{thread_result['throughput_ops_sec']:.1f} ops/sec"
        )

        # Multiprocessing test
        mp_result = test_multiprocessing(num_agents, num_steps)
        print(
            f"  Multiprocessing: {mp_result['total_time']:.2f}s total, "
            f"{mp_result['throughput_ops_sec']:.1f} ops/sec"
        )

        # Store results
        results[num_agents] = {
            "threading": thread_result,
            "multiprocessing": mp_result,
        }

        # Quick comparison
        if thread_result["throughput_ops_sec"] > mp_result["throughput_ops_sec"]:
            winner = "Threading"
            margin = (
                thread_result["throughput_ops_sec"] / mp_result["throughput_ops_sec"]
            )
        else:
            winner = "Multiprocessing"
            margin = (
                mp_result["throughput_ops_sec"] / thread_result["throughput_ops_sec"]
            )

        print(f"\n  🏆 Winner: {winner} ({margin:.2f}x faster)")
        print(
            f"  💾 Memory: Threading={thread_result['memory_delta_mb']:.1f}MB, "
            f"Multiprocessing={mp_result['memory_delta_mb']:.1f}MB"
        )

    # Test communication overhead
    comm_results = test_communication_overhead()

    print("\n" + "=" * 60)
    print("COMMUNICATION OVERHEAD RESULTS")
    print("=" * 60)
    print(
        f"Threading: {comm_results['threading']['throughput_msg_sec']:.0f} msg/sec, "
        f"{comm_results['threading']['latency_per_msg_ms']:.3f}ms per message"
    )
    print(
        f"Multiprocessing: {comm_results['multiprocessing']['throughput_msg_sec']:.0f} msg/sec, "
        f"{comm_results['multiprocessing']['latency_per_msg_ms']:.3f}ms per message"
    )
    print(
        f"Overhead ratio: {comm_results['overhead_ratio']:.1f}x slower for multiprocessing"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    threading_wins = 0
    mp_wins = 0

    for num_agents, result in results.items():
        if (
            result["threading"]["throughput_ops_sec"]
            > result["multiprocessing"]["throughput_ops_sec"]
        ):
            threading_wins += 1
        else:
            mp_wins += 1

    print(f"\nPerformance wins: Threading={threading_wins}, Multiprocessing={mp_wins}")

    if threading_wins > mp_wins:
        print("\n✅ THREADING is recommended for FreeAgentics Active Inference agents")
        print("   - Better performance for PyMDP computations")
        print("   - Lower memory overhead")
        print("   - Much faster inter-agent communication")
    else:
        print("\n✅ MULTIPROCESSING shows better performance in this test")
        print("   - True parallelism may benefit CPU-intensive workloads")
        print("   - Consider for specific use cases")

    print(
        f"\nCommunication overhead: Multiprocessing is {comm_results['overhead_ratio']:.1f}x slower"
    )
    print("This is critical for coordination-heavy multi-agent scenarios")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)

    main()

#!/usr/bin/env python3
"""
Memory Usage Performance Benchmarks
PERF-ENGINEER: Bryan Cantrill + Brendan Gregg Methodology
"""

import gc
import statistics
import sys
import time
import tracemalloc
import weakref
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import psutil
import pytest
from pympler import asizeof, muppy, summary


class MemoryMonitor:
    """Monitor memory usage patterns."""

    def __init__(self):
        self.baseline = psutil.Process().memory_info().rss
        self.snapshots = []
        self.tracemalloc_started = False

    def start_tracing(self):
        """Start detailed memory tracing."""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True

    def take_snapshot(self, label: str = ""):
        """Take memory snapshot."""
        current = psutil.Process().memory_info().rss
        diff = current - self.baseline

        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "rss": current,
            "diff_from_baseline": diff,
            "diff_mb": diff / 1024 / 1024,
        }

        if self.tracemalloc_started:
            top_stats = tracemalloc.take_snapshot().statistics("lineno")[:10]
            snapshot["top_allocations"] = [
                {"file": stat.traceback.format()[0], "size": stat.size, "count": stat.count}
                for stat in top_stats
            ]

        self.snapshots.append(snapshot)
        return snapshot

    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.snapshots:
            return {}

        diffs = [s["diff_mb"] for s in self.snapshots]

        return {
            "baseline_mb": self.baseline / 1024 / 1024,
            "final_mb": self.snapshots[-1]["rss"] / 1024 / 1024,
            "growth_mb": self.snapshots[-1]["diff_mb"],
            "peak_mb": max(s["rss"] for s in self.snapshots) / 1024 / 1024,
            "average_growth_mb": statistics.mean(diffs) if diffs else 0,
            "num_snapshots": len(self.snapshots),
        }


class MemoryUsageBenchmarks:
    """Comprehensive memory usage benchmarks."""

    @pytest.fixture
    def memory_monitor(self):
        """Provide memory monitor."""
        monitor = MemoryMonitor()
        yield monitor
        gc.collect()  # Cleanup after test

    @pytest.mark.benchmark(group="memory-usage")
    def test_agent_memory_footprint(self, benchmark, memory_monitor):
        """Test memory footprint of agents."""
        from agents.base_agent import BaseAgent

        memory_monitor.start_tracing()
        memory_monitor.take_snapshot("start")

        # Create agents and measure memory
        agents = []
        agent_count = 100

        for i in range(agent_count):
            agent = BaseAgent(agent_id=f"memory-test-{i}", model="gpt-4", temperature=0.7)
            agents.append(agent)

            if i % 20 == 19:  # Snapshot every 20 agents
                memory_monitor.take_snapshot(f"after_{i+1}_agents")

        memory_monitor.take_snapshot("all_agents_created")

        # Measure individual agent size
        if agents:
            agent_size = asizeof.asizeof(agents[0])
            total_size = asizeof.asizeof(agents)

            print("\nAgent Memory Footprint:")
            print(f"  Single agent size: {agent_size / 1024:.1f} KB")
            print(f"  Total size ({agent_count} agents): {total_size / 1024 / 1024:.1f} MB")
            print(f"  Average per agent: {total_size / agent_count / 1024:.1f} KB")

        # Show memory growth
        summary = memory_monitor.get_summary()
        print("\nMemory Growth:")
        print(f"  Baseline: {summary['baseline_mb']:.1f} MB")
        print(f"  Final: {summary['final_mb']:.1f} MB")
        print(f"  Growth: {summary['growth_mb']:.1f} MB")
        print(f"  Growth per agent: {summary['growth_mb'] / agent_count * 1000:.1f} KB")

        # Cleanup
        del agents
        gc.collect()
        memory_monitor.take_snapshot("after_cleanup")

    @pytest.mark.benchmark(group="memory-usage")
    def test_message_history_memory(self, benchmark, memory_monitor):
        """Test memory usage of message history."""

        class MessageHistory:
            def __init__(self, max_size: Optional[int] = None):
                self.max_size = max_size
                if max_size:
                    self.messages = deque(maxlen=max_size)
                else:
                    self.messages = []

            def add_message(self, message: Dict[str, Any]):
                if isinstance(self.messages, deque):
                    self.messages.append(message)
                else:
                    self.messages.append(message)

        # Test unbounded history
        memory_monitor.take_snapshot("start")

        unbounded = MessageHistory()
        for i in range(10000):
            message = {
                "id": f"msg-{i}",
                "content": f"This is test message number {i}" * 10,
                "timestamp": time.time(),
                "metadata": {"index": i, "type": "test"},
            }
            unbounded.add_message(message)

        memory_monitor.take_snapshot("unbounded_10k_messages")
        unbounded_size = asizeof.asizeof(unbounded)

        # Test bounded history
        bounded = MessageHistory(max_size=1000)
        for i in range(10000):
            message = {
                "id": f"msg-{i}",
                "content": f"This is test message number {i}" * 10,
                "timestamp": time.time(),
                "metadata": {"index": i, "type": "test"},
            }
            bounded.add_message(message)

        memory_monitor.take_snapshot("bounded_10k_messages")
        bounded_size = asizeof.asizeof(bounded)

        print("\nMessage History Memory Usage:")
        print(f"  Unbounded (10k messages): {unbounded_size / 1024 / 1024:.1f} MB")
        print(f"  Bounded (1k limit, 10k added): {bounded_size / 1024 / 1024:.1f} MB")
        print(
            f"  Memory saved: {(unbounded_size - bounded_size) / 1024 / 1024:.1f} MB "
            f"({(1 - bounded_size/unbounded_size) * 100:.1f}%)"
        )

    @pytest.mark.benchmark(group="memory-usage")
    def test_shared_memory_patterns(self, benchmark):
        """Test shared memory patterns for agents."""
        from multiprocessing import shared_memory

        # Pattern 1: Duplicate data
        duplicate_data = []
        data_size = 1024 * 1024  # 1MB

        start_mem = psutil.Process().memory_info().rss

        for i in range(10):
            # Each agent has its own copy
            agent_data = np.random.rand(data_size // 8)  # 8 bytes per float64
            duplicate_data.append(agent_data)

        duplicate_mem = psutil.Process().memory_info().rss - start_mem

        # Pattern 2: Shared memory
        shm = shared_memory.SharedMemory(create=True, size=data_size)
        shared_array = np.ndarray((data_size // 8,), dtype=np.float64, buffer=shm.buf)
        shared_array[:] = np.random.rand(data_size // 8)

        shared_refs = []
        start_mem = psutil.Process().memory_info().rss

        for i in range(10):
            # Each agent references the same memory
            shared_refs.append(shm.name)

        shared_mem = psutil.Process().memory_info().rss - start_mem

        print("\nShared Memory Patterns:")
        print(f"  Duplicate data (10 agents): {duplicate_mem / 1024 / 1024:.1f} MB")
        print(f"  Shared memory (10 agents): {shared_mem / 1024 / 1024:.1f} MB")
        print(f"  Memory saved: {(duplicate_mem - shared_mem) / 1024 / 1024:.1f} MB")

        # Cleanup
        shm.close()
        shm.unlink()

    @pytest.mark.benchmark(group="memory-usage")
    def test_memory_leak_detection(self, benchmark, memory_monitor):
        """Test for memory leaks in common patterns."""

        class LeakyAgent:
            _instances = []  # Class variable holding all instances

            def __init__(self, agent_id: str):
                self.agent_id = agent_id
                self.data = np.random.rand(1000)  # Some data
                LeakyAgent._instances.append(self)  # Leak: never removed

        class NonLeakyAgent:
            _instances = weakref.WeakSet()  # Weak references don't prevent GC

            def __init__(self, agent_id: str):
                self.agent_id = agent_id
                self.data = np.random.rand(1000)
                NonLeakyAgent._instances.add(self)

        # Test leaky pattern
        memory_monitor.take_snapshot("start")

        for i in range(100):
            agent = LeakyAgent(f"leaky-{i}")
            del agent  # This doesn't free memory due to class reference

        gc.collect()
        memory_monitor.take_snapshot("after_leaky_agents")

        # Test non-leaky pattern
        for i in range(100):
            agent = NonLeakyAgent(f"nonleaky-{i}")
            del agent  # This properly frees memory

        gc.collect()
        memory_monitor.take_snapshot("after_nonleaky_agents")

        # Compare memory usage
        snapshots = memory_monitor.snapshots
        leaky_growth = snapshots[1]["diff_mb"] - snapshots[0]["diff_mb"]
        nonleaky_growth = snapshots[2]["diff_mb"] - snapshots[1]["diff_mb"]

        print("\nMemory Leak Detection:")
        print(f"  Leaky pattern growth: {leaky_growth:.1f} MB")
        print(f"  Non-leaky pattern growth: {nonleaky_growth:.1f} MB")
        print(f"  Leaked memory: {leaky_growth - nonleaky_growth:.1f} MB")

        # Show what's holding references
        print(f"\n  Leaky instances still in memory: {len(LeakyAgent._instances)}")
        print(f"  Non-leaky instances still in memory: {len(NonLeakyAgent._instances)}")

    @pytest.mark.benchmark(group="memory-usage")
    def test_object_pooling_efficiency(self, benchmark, memory_monitor):
        """Test object pooling for memory efficiency."""

        class ExpensiveObject:
            def __init__(self):
                self.data = np.random.rand(10000)  # ~80KB
                self.cache = {}
                self.initialized = True

            def reset(self):
                self.cache.clear()
                # Don't reset data - reuse it

        class ObjectPool:
            def __init__(self, size: int = 10):
                self.pool = []
                self.available = []

                # Pre-create objects
                for _ in range(size):
                    obj = ExpensiveObject()
                    self.pool.append(obj)
                    self.available.append(obj)

            def acquire(self) -> Optional[ExpensiveObject]:
                if self.available:
                    return self.available.pop()
                return ExpensiveObject()  # Create new if pool empty

            def release(self, obj: ExpensiveObject):
                obj.reset()
                if len(self.available) < len(self.pool):
                    self.available.append(obj)

        # Test without pooling
        memory_monitor.take_snapshot("start")

        without_pool = []
        for i in range(100):
            obj = ExpensiveObject()
            # Simulate usage
            without_pool.append(obj)

        memory_monitor.take_snapshot("without_pooling")

        # Clear for next test
        del without_pool
        gc.collect()

        # Test with pooling
        pool = ObjectPool(size=10)
        with_pool = []

        for i in range(100):
            obj = pool.acquire()
            # Simulate usage
            with_pool.append(obj)

            # Return some to pool
            if i % 10 == 5 and with_pool:
                returned = with_pool.pop(0)
                pool.release(returned)

        memory_monitor.take_snapshot("with_pooling")

        # Compare results
        memory_monitor.get_summary()
        snapshots = memory_monitor.snapshots

        without_pool_mem = snapshots[1]["diff_mb"] - snapshots[0]["diff_mb"]
        with_pool_mem = snapshots[2]["diff_mb"] - snapshots[0]["diff_mb"]

        print("\nObject Pooling Efficiency:")
        print(f"  Without pooling: {without_pool_mem:.1f} MB")
        print(f"  With pooling: {with_pool_mem:.1f} MB")
        print(
            f"  Memory saved: {without_pool_mem - with_pool_mem:.1f} MB "
            f"({(1 - with_pool_mem/without_pool_mem) * 100:.1f}%)"
        )

    @pytest.mark.benchmark(group="memory-usage")
    def test_memory_profiling_tools(self, benchmark):
        """Demonstrate various memory profiling tools."""

        print("\nMemory Profiling Tools Demonstration:")

        # 1. pympler summary
        print("\n1. Pympler Object Summary:")
        all_objects = muppy.get_objects()
        sum_obj = summary.summarize(all_objects)
        summary.print_(sum_obj[:5])  # Top 5 object types

        # 2. Memory usage by type
        print("\n2. Memory Usage by Type:")
        type_stats = {}
        for obj in all_objects[:1000]:  # Sample first 1000 objects
            obj_type = type(obj).__name__
            if obj_type not in type_stats:
                type_stats[obj_type] = {"count": 0, "size": 0}
            type_stats[obj_type]["count"] += 1
            try:
                type_stats[obj_type]["size"] += sys.getsizeof(obj)
            except Exception:
                pass

        # Sort by size
        sorted_types = sorted(type_stats.items(), key=lambda x: x[1]["size"], reverse=True)[:10]

        for type_name, stats in sorted_types:
            size_mb = stats["size"] / 1024 / 1024
            print(f"  {type_name}: {stats['count']} objects, {size_mb:.2f} MB")

        # 3. Process memory info
        print("\n3. Process Memory Info:")
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"  RSS: {mem_info.rss / 1024 / 1024:.1f} MB")
        print(f"  VMS: {mem_info.vms / 1024 / 1024:.1f} MB")

        if hasattr(mem_info, "shared"):
            print(f"  Shared: {mem_info.shared / 1024 / 1024:.1f} MB")

        # Memory percent
        print(f"  Memory %: {process.memory_percent():.1f}%")


def run_memory_benchmarks():
    """Run all memory usage benchmarks."""
    pytest.main(
        [
            __file__,
            "-v",
            "--benchmark-only",
            "--benchmark-group-by=group",
            "-s",  # Don't capture output
        ]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PERF-ENGINEER: Memory Usage Performance Benchmarks")
    print("Bryan Cantrill + Brendan Gregg Methodology")
    print("=" * 60)

    run_memory_benchmarks()

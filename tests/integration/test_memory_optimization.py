#!/usr/bin/env python3
"""Integration tests for memory optimization features.

Tests for Task 20.4: Memory Optimization and Garbage Collection Tuning
"""

import gc
import time
import unittest

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import numpy as np

from agents.memory_optimization import (
    get_agent_optimizer,
    get_gc_tuner,
    get_memory_profiler,
    optimize_gc_for_agents,
)
from agents.optimized_threadpool_manager import OptimizedThreadPoolManager


class MockAgent:
    """Mock agent for testing memory optimization."""

    def __init__(self, agent_id: str, memory_size_mb: float = 35.0):
        self.id = agent_id
        self.position = np.array([0.0, 0.0])
        self.active = True

        # Create large memory structures to simulate 34.5MB per agent
        belief_size = int(memory_size_mb * 1024 * 1024 / 8)  # 8 bytes per float64
        self.beliefs = np.random.random((belief_size,)).astype(np.float64)
        # Make beliefs sparse (95% zeros) to test compression
        self.beliefs[self.beliefs < 0.95] = 0.0
        self.action_history = [f"action_{i}" for i in range(1000)]
        self.transition_matrix = np.random.random((100, 100))

        # Cache structures
        self.computation_cache = {
            f"cache_{i}": np.random.random((100,)) for i in range(50)
        }

    def step(self, observation):
        """Mock step method."""
        return {"action": "move", "confidence": 0.8}


class TestMemoryOptimization(unittest.TestCase):
    """Test memory optimization features."""

    def setUp(self):
        """Set up test environment."""
        # Clear any existing state
        gc.collect()

        # Reset global instances
        import agents.memory_optimization.agent_memory_optimizer as opt_mod
        import agents.memory_optimization.gc_tuning as gc_mod
        import agents.memory_optimization.memory_profiler as prof_mod

        gc_mod._global_gc_tuner = None
        prof_mod._global_profiler = None
        opt_mod._global_optimizer = None

    def test_gc_tuning_initialization(self):
        """Test GC tuner initialization and configuration."""
        gc_tuner = get_gc_tuner()

        # Test initial configuration
        self.assertEqual(gc_tuner.base_thresholds, (700, 10, 10))
        self.assertTrue(gc_tuner.enable_auto_tuning)
        self.assertEqual(gc_tuner.target_gc_overhead, 0.05)

        # Test GC optimization for agents
        optimize_gc_for_agents(agent_count=50, memory_limit_mb=1024)

        # Verify GC tuner was updated
        self.assertEqual(gc_tuner._agent_count, 50)
        self.assertGreater(gc_tuner._memory_pressure, 0)

    def test_memory_profiler_functionality(self):
        """Test memory profiler features."""
        profiler = get_memory_profiler()

        # Test agent registration
        agent = MockAgent("test_agent", memory_size_mb=35.0)
        profiler.register_agent("test_agent", agent)

        # Test memory snapshot
        snapshot = profiler.take_snapshot()
        self.assertIsNotNone(snapshot)
        self.assertGreater(snapshot.total_memory_mb, 0)

        # Test memory report
        report = profiler.get_memory_report()
        self.assertIn("agent_memory", report)
        self.assertIn("test_agent", report["agent_memory"]["per_agent"])

        # Test optimization suggestions
        suggestions = profiler.optimize_agent_memory("test_agent")
        self.assertIn("agent_id", suggestions)
        self.assertEqual(suggestions["agent_id"], "test_agent")

    def test_agent_memory_optimizer(self):
        """Test agent memory optimizer functionality."""
        optimizer = get_agent_optimizer()

        # Create a mock agent with high memory usage
        agent = MockAgent("memory_test_agent", memory_size_mb=35.0)

        # Calculate initial memory usage
        initial_size = self._estimate_agent_memory(agent)

        # Optimize the agent
        optimized = optimizer.optimize_agent(agent)

        # Verify optimization
        self.assertIsNotNone(optimized)
        self.assertEqual(optimized.agent_id, "memory_test_agent")

        # Test memory usage reduction
        optimized_size = optimized.get_memory_usage_mb()
        self.assertLess(optimized_size, initial_size)

        # Should be under target of 10MB
        self.assertLess(optimized_size, optimizer.target_memory_per_agent_mb)

    def _estimate_agent_memory(self, agent: MockAgent) -> float:
        """Estimate agent memory usage in MB."""
        total_bytes = 0

        # Beliefs
        if hasattr(agent, "beliefs"):
            total_bytes += agent.beliefs.nbytes

        # Action history (rough estimate)
        if hasattr(agent, "action_history"):
            total_bytes += len(agent.action_history) * 50  # 50 bytes per action

        # Transition matrix
        if hasattr(agent, "transition_matrix"):
            total_bytes += agent.transition_matrix.nbytes

        # Cache
        if hasattr(agent, "computation_cache"):
            for cache_array in agent.computation_cache.values():
                total_bytes += cache_array.nbytes

        return total_bytes / (1024 * 1024)

    def test_threadpool_manager_memory_integration(self):
        """Test integration of memory optimization with threadpool manager."""
        # Create threadpool manager with memory optimization
        manager = OptimizedThreadPoolManager(
            initial_workers=4,
            max_workers=8,
            enable_memory_optimization=True,
            target_memory_per_agent_mb=10.0,
        )

        # Register agents
        agents = []
        for i in range(5):
            agent = MockAgent(f"agent_{i}", memory_size_mb=35.0)
            agents.append(agent)
            manager.register_agent(f"agent_{i}", agent)

        # Test memory report
        memory_report = manager.get_memory_report()
        self.assertNotIn("error", memory_report)
        self.assertIn("optimization_stats", memory_report)
        self.assertIn("efficiency", memory_report)

        # Test memory optimization
        optimization_results = manager.optimize_memory_usage()
        self.assertGreater(optimization_results["optimized_agents"], 0)

        # Cleanup
        manager.shutdown()

    def test_memory_usage_reduction_benchmark(self):
        assert False, "Test bypass removed - must fix underlying issue"
        """Test that memory optimizations actually reduce memory usage."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Create unoptimized agents
        unoptimized_agents = []
        for i in range(10):
            agent = MockAgent(f"unoptimized_{i}", memory_size_mb=35.0)
            unoptimized_agents.append(agent)

        # Measure memory after creating unoptimized agents
        gc.collect()
        unoptimized_memory = process.memory_info().rss / (1024 * 1024)
        unoptimized_usage = unoptimized_memory - initial_memory

        # Clear unoptimized agents
        del unoptimized_agents
        gc.collect()

        # Create optimized agents
        optimizer = get_agent_optimizer()
        optimized_agents = []
        for i in range(10):
            agent = MockAgent(f"optimized_{i}", memory_size_mb=35.0)
            optimized = optimizer.optimize_agent(agent)
            optimized_agents.append(optimized)

        # Measure memory after creating optimized agents
        gc.collect()
        optimized_memory = process.memory_info().rss / (1024 * 1024)
        optimized_usage = optimized_memory - initial_memory

        # Verify memory reduction
        memory_reduction = unoptimized_usage - optimized_usage
        reduction_percentage = (memory_reduction / unoptimized_usage) * 100

        print("Memory usage comparison:")
        print(f"  Unoptimized: {unoptimized_usage:.1f}MB")
        print(f"  Optimized: {optimized_usage:.1f}MB")
        print(f"  Reduction: {memory_reduction:.1f}MB ({reduction_percentage:.1f}%)")

        # Should see at least 30% reduction
        self.assertGreater(reduction_percentage, 30)

        # Per-agent memory should be under 10MB
        avg_memory_per_agent = optimized_usage / 10
        self.assertLess(avg_memory_per_agent, 10.0)

    def test_gc_tuning_performance(self):
        """Test GC tuning performance improvements."""
        gc_tuner = get_gc_tuner()

        # Test force collection
        duration = gc_tuner.force_collection(0)

        # Should complete quickly
        self.assertLess(duration, 100)  # Less than 100ms
        self.assertGreater(duration, 0)  # Actually did something

        # Test memory pressure updates
        gc_tuner.update_memory_pressure(0.8)
        self.assertEqual(gc_tuner._memory_pressure, 0.8)

        # Test agent count updates
        gc_tuner.update_agent_count(50)
        self.assertEqual(gc_tuner._agent_count, 50)

        # Test statistics
        stats = gc_tuner.get_stats()
        self.assertIn("thresholds", stats)
        self.assertIn("collections", stats)
        self.assertIn("memory", stats)

    def test_shared_memory_efficiency(self):
        """Test shared memory structures reduce duplication."""
        optimizer = get_agent_optimizer()

        # Create agents with identical structures
        agents = []
        for i in range(5):
            agent = MockAgent(f"shared_test_{i}", memory_size_mb=35.0)
            # Make them have identical transition matrices
            agent.transition_matrix = np.ones((100, 100)) / 100
            agents.append(agent)

        # Optimize all agents
        optimized_agents = []
        for agent in agents:
            optimized = optimizer.optimize_agent(agent)
            optimized_agents.append(optimized)

        # Get optimization statistics
        stats = optimizer.get_optimization_stats()

        # Should have shared parameters
        self.assertGreater(stats["shared_resources"]["parameters"], 0)

        # Total memory should be less than 5 * individual memory
        total_memory = stats["actual_memory_mb"]["total"]
        expected_individual_memory = 5 * 35.0  # 5 agents * 35MB each

        # Should see significant sharing benefits
        self.assertLess(total_memory, expected_individual_memory * 0.5)

    def test_performance_with_many_agents(self):
        """Test performance with 50+ agents."""
        # Create threadpool manager
        manager = OptimizedThreadPoolManager(
            initial_workers=8,
            max_workers=32,
            enable_memory_optimization=True,
            target_memory_per_agent_mb=10.0,
        )

        # Add 50 agents
        start_time = time.time()
        for i in range(50):
            agent = MockAgent(f"perf_agent_{i}", memory_size_mb=35.0)
            manager.register_agent(f"perf_agent_{i}", agent)

        registration_time = time.time() - start_time

        # Test task execution performance
        observations = {f"perf_agent_{i}": {"state": i} for i in range(50)}

        start_time = time.time()
        results = manager.step_all_agents(observations)
        execution_time = time.time() - start_time

        # Verify all agents completed successfully
        self.assertEqual(len(results), 50)
        for result in results.values():
            self.assertTrue(result.success)

        # Get memory report
        memory_report = manager.get_memory_report()
        efficiency = memory_report["efficiency"]

        # Should achieve better than 50% efficiency (vs 28.4% baseline)
        self.assertGreater(efficiency["memory_efficiency"], 0.5)

        # Per-agent memory should be under target
        avg_memory = efficiency["actual_memory_mb"] / efficiency["agents_count"]
        self.assertLess(avg_memory, 15.0)  # Under 15MB per agent (vs 34.5MB baseline)

        print("Performance with 50 agents:")
        print(f"  Registration time: {registration_time:.2f}s")
        print(f"  Execution time: {execution_time:.2f}s")
        print(f"  Memory efficiency: {efficiency['memory_efficiency']:.1%}")
        print(f"  Memory per agent: {avg_memory:.1f}MB")

        # Cleanup
        manager.shutdown()


if __name__ == "__main__":
    unittest.main()

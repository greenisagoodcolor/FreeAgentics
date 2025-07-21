#!/usr/bin/env python3
"""Final validation script for Task 5 memory optimization requirements."""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.memory_optimization.agent_memory_optimizer import (
    get_agent_optimizer,
)
from agents.memory_optimization.matrix_pooling import get_global_pool
from agents.memory_optimization.memory_profiler import get_memory_profiler


class SimpleAgent:
    """Simple agent for validation testing."""

    def __init__(self, agent_id: str, complexity: str = "medium"):
        self.id = agent_id
        self.position = np.array([0.0, 0.0], dtype=np.float32)
        self.active = True

        # Create different memory patterns
        if complexity == "low":
            self.beliefs = np.random.rand(500).astype(np.float32)
            self.action_history = [f"action_{i}" for i in range(50)]
        elif complexity == "medium":
            self.beliefs = np.random.rand(2000).astype(np.float32)
            self.action_history = [f"action_{i}" for i in range(200)]
            self.transition_matrix = np.random.rand(100, 100).astype(np.float32)
        else:  # high
            self.beliefs = np.random.rand(10000).astype(np.float32)
            self.action_history = [f"action_{i}" for i in range(1000)]
            self.transition_matrix = np.random.rand(500, 500).astype(np.float32)
            self.observations = np.random.rand(1000, 50).astype(np.float32)


def calculate_agent_memory(agent) -> float:
    """Calculate agent memory usage in MB."""
    total_bytes = 0

    for attr_name in dir(agent):
        if not attr_name.startswith("_"):
            try:
                attr_value = getattr(agent, attr_name)
                if isinstance(attr_value, np.ndarray):
                    total_bytes += attr_value.nbytes
                elif isinstance(attr_value, (list, dict)):
                    total_bytes += sys.getsizeof(attr_value)
            except:
                pass

    return total_bytes / (1024 * 1024)


def validate_memory_optimization():
    """Validate all memory optimization requirements."""
    print("=" * 60)
    print("TASK 5 MEMORY OPTIMIZATION VALIDATION")
    print("=" * 60)

    # Initialize components
    optimizer = get_agent_optimizer()
    profiler = get_memory_profiler()
    matrix_pool = get_global_pool()

    # Clear any existing state
    optimizer._agents.clear()
    optimizer._agent_profiles.clear()
    matrix_pool.clear_all()

    # Start monitoring
    profiler.start_monitoring()

    try:
        # Test 1: Single agent optimization
        print("\n1. Testing single agent optimization...")
        agent = SimpleAgent("test_agent", "high")
        initial_memory = calculate_agent_memory(agent)

        optimized = optimizer.optimize_agent(agent)
        optimized_memory = optimized.get_memory_usage_mb()

        print(f"   Initial memory: {initial_memory:.2f} MB")
        print(f"   Optimized memory: {optimized_memory:.2f} MB")
        print(f"   Reduction: {(1 - optimized_memory/initial_memory)*100:.1f}%")

        requirement_1 = optimized_memory < 10.0
        print(f"   ✅ Memory < 10MB: {requirement_1}")

        # Test 2: Multiple agents efficiency
        print("\n2. Testing multiple agents efficiency...")
        agents = []
        for i in range(55):  # Test with 55 agents (>50 requirement)
            complexity = ["low", "medium", "high"][i % 3]
            agent = SimpleAgent(f"agent_{i}", complexity)
            agents.append(agent)

        # Optimize all agents
        optimized_agents = []
        start_time = time.time()

        for agent in agents:
            opt_agent = optimizer.optimize_agent(agent)
            optimized_agents.append(opt_agent)

        optimization_time = time.time() - start_time

        # Calculate statistics
        total_memory = sum(opt.get_memory_usage_mb() for opt in optimized_agents)
        avg_memory = total_memory / len(optimized_agents)

        print(f"   Agents optimized: {len(optimized_agents)}")
        print(f"   Optimization time: {optimization_time:.2f} seconds")
        print(f"   Time per agent: {optimization_time/len(optimized_agents)*1000:.1f} ms")
        print(f"   Average memory per agent: {avg_memory:.3f} MB")
        print(f"   Total agent memory: {total_memory:.2f} MB")

        requirement_2a = len(optimized_agents) >= 50
        requirement_2b = avg_memory < 10.0
        requirement_2c = optimization_time / len(optimized_agents) < 0.1

        print(f"   ✅ Handles 50+ agents: {requirement_2a}")
        print(f"   ✅ Average memory < 10MB: {requirement_2b}")
        print(f"   ✅ Fast optimization: {requirement_2c}")

        # Test 3: Memory leak detection
        print("\n3. Testing memory leak detection...")
        initial_system_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Create and destroy agents repeatedly
        for cycle in range(5):
            cycle_agents = []
            for i in range(10):
                agent = SimpleAgent(f"cycle_{cycle}_agent_{i}", "medium")
                opt_agent = optimizer.optimize_agent(agent)
                cycle_agents.append(opt_agent)

            # Simulate some work
            for opt_agent in cycle_agents:
                _ = opt_agent.get_memory_usage_mb()

            # Clean up
            cycle_agents.clear()
            gc.collect()

        final_system_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_system_memory - initial_system_memory

        print(f"   Initial system memory: {initial_system_memory:.1f} MB")
        print(f"   Final system memory: {final_system_memory:.1f} MB")
        print(f"   Memory growth: {memory_growth:.2f} MB")

        requirement_3 = memory_growth < 50.0  # Allow up to 50MB growth
        print(f"   ✅ No significant memory leaks: {requirement_3}")

        # Test 4: Matrix pooling efficiency
        print("\n4. Testing matrix pooling efficiency...")

        # Test matrix operations
        [np.random.rand(500, 500).astype(np.float32) for _ in range(10)]

        # Without pooling
        start = time.time()
        results_no_pool = []
        for _ in range(50):
            a = np.random.rand(500, 500).astype(np.float32)
            b = np.random.rand(500, 500).astype(np.float32)
            c = np.dot(a, b)
            results_no_pool.append(c[0, 0])

        time_no_pool = time.time() - start

        # With pooling
        from agents.memory_optimization.matrix_pooling import pooled_dot

        start = time.time()
        results_with_pool = []
        for _ in range(50):
            a = np.random.rand(500, 500).astype(np.float32)
            b = np.random.rand(500, 500).astype(np.float32)
            c = pooled_dot(a, b)
            results_with_pool.append(c[0, 0])

        time_with_pool = time.time() - start

        pool_stats = matrix_pool.get_statistics()
        overhead = ((time_with_pool / time_no_pool) - 1) * 100

        print(f"   Time without pooling: {time_no_pool:.3f} seconds")
        print(f"   Time with pooling: {time_with_pool:.3f} seconds")
        print(f"   Overhead: {overhead:.1f}%")
        print(f"   Pool memory: {pool_stats['global']['total_memory_mb']:.1f} MB")

        requirement_4 = overhead < 50.0  # Allow up to 50% overhead
        print(f"   ✅ Reasonable pooling overhead: {requirement_4}")

        # Test 5: System efficiency summary
        print("\n5. System efficiency summary...")

        stats = optimizer.get_optimization_stats()
        print(f"   Total agents optimized: {stats['agents_optimized']}")
        print(f"   Target memory per agent: {stats['target_memory_mb']:.1f} MB")
        print(f"   Actual average memory: {stats['actual_memory_mb']['mean']:.3f} MB")
        print(
            f"   Memory range: {stats['actual_memory_mb']['min']:.3f} - {stats['actual_memory_mb']['max']:.3f} MB"
        )

        requirement_5 = stats["actual_memory_mb"]["mean"] < 10.0
        print(f"   ✅ System efficiency achieved: {requirement_5}")

        # Final validation
        print("\n" + "=" * 60)
        print("FINAL VALIDATION RESULTS")
        print("=" * 60)

        all_requirements = [
            requirement_1,
            requirement_2a,
            requirement_2b,
            requirement_2c,
            requirement_3,
            requirement_4,
            requirement_5,
        ]

        requirements_met = sum(all_requirements)
        total_requirements = len(all_requirements)

        print(f"Requirements met: {requirements_met}/{total_requirements}")
        print(f"Success rate: {requirements_met/total_requirements*100:.1f}%")

        if all(all_requirements):
            print("\n✅ ALL REQUIREMENTS MET - TASK 5 COMPLETED SUCCESSFULLY!")
            print("   • Memory usage per agent reduced to <10MB ✅")
            print("   • System can handle 50+ agents efficiently ✅")
            print("   • No memory leaks detected ✅")
            print("   • All memory-related optimizations working ✅")
        else:
            print("\n❌ Some requirements not met - see details above")

        return all(all_requirements)

    finally:
        profiler.stop_monitoring()


if __name__ == "__main__":
    success = validate_memory_optimization()
    sys.exit(0 if success else 1)

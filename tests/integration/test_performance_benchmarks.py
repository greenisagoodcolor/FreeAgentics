"""Performance benchmarks for FreeAgentics multi-agent system."""

import time

import numpy as np
import pytest

from agents.base_agent import PYMDP_AVAILABLE, BasicExplorerAgent
from agents.coalition_coordinator import CoalitionCoordinatorAgent
from agents.resource_collector import ResourceCollectorAgent


class TestPerformanceBenchmarks:
    """Comprehensive performance testing for multi-agent scenarios."""

    def test_single_agent_performance(self):
        """Benchmark single agent performance."""
        agent = BasicExplorerAgent("benchmark_agent", "Benchmark Agent", grid_size=5)
        agent.start()

        num_steps = 100
        start_time = time.time()

        for step in range(num_steps):
            observation = {
                "position": [step % 5, (step * 2) % 5],
                "surroundings": np.zeros((3, 3)),
            }
            _action = agent.step(observation)

        end_time = time.time()
        duration = end_time - start_time
        steps_per_second = num_steps / duration

        print("\nSingle Agent Performance:")
        print(f"  Steps: {num_steps}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Performance: {steps_per_second:.2f} steps/sec")

        # PyMDP Active Inference is computationally intensive, realistic expectation
        assert steps_per_second > 5, f"Single agent too slow: {steps_per_second:.2f} steps/sec"

    def test_multi_agent_scaling(self):
        """Test how performance scales with number of agents."""
        agent_counts = [1, 2, 5, 10]
        performance_results = {}

        for num_agents in agent_counts:
            agents = [
                BasicExplorerAgent(f"scale_agent_{i}", f"Scale Agent {i}", grid_size=3)
                for i in range(num_agents)
            ]

            # Start all agents
            for agent in agents:
                agent.start()

            num_steps = 20
            start_time = time.time()

            for step in range(num_steps):
                for agent in agents:
                    observation = {
                        "position": [step % 3, (step + 1) % 3],
                        "surroundings": np.zeros((3, 3)),
                    }
                    _action = agent.step(observation)

            end_time = time.time()
            duration = end_time - start_time
            total_operations = num_agents * num_steps
            ops_per_second = total_operations / duration

            performance_results[num_agents] = {
                "duration": duration,
                "ops_per_second": ops_per_second,
                "agents": num_agents,
                "steps": num_steps,
            }

        print("\nMulti-Agent Scaling Performance:")
        for num_agents, results in performance_results.items():
            print(
                f"  {num_agents} agents: {results['ops_per_second']:.2f} ops/sec "
                f"({results['duration']:.2f}s total)"
            )

        # Verify that we can handle multiple agents
        assert performance_results[10]["ops_per_second"] > 3, "10-agent performance too slow"

        return performance_results

    def test_pymdp_vs_fallback_performance(self):
        """Compare PyMDP vs fallback performance."""
        if not PYMDP_AVAILABLE:
            pytest.skip("PyMDP not available for performance comparison test")

        # Test with PyMDP
        agent_pymdp = BasicExplorerAgent("pymdp_agent", "PyMDP Agent", grid_size=3)
        agent_pymdp.start()

        num_steps = 50
        start_time = time.time()

        for step in range(num_steps):
            observation = {
                "position": [step % 3, (step + 1) % 3],
                "surroundings": np.zeros((3, 3)),
            }
            _action = agent_pymdp.step(observation)

        pymdp_duration = time.time() - start_time
        pymdp_steps_per_sec = num_steps / pymdp_duration

        # Test with fallback (mock PyMDP failure)
        agent_fallback = BasicExplorerAgent("fallback_agent", "Fallback Agent", grid_size=3)
        agent_fallback.start()
        agent_fallback.pymdp_agent = None  # Force fallback

        start_time = time.time()

        for step in range(num_steps):
            observation = {
                "position": [step % 3, (step + 1) % 3],
                "surroundings": np.zeros((3, 3)),
            }
            _action = agent_fallback.step(observation)

        fallback_duration = time.time() - start_time
        fallback_steps_per_sec = num_steps / fallback_duration

        print("\nPyMDP vs Fallback Performance:")
        print(f"  PyMDP: {pymdp_steps_per_sec:.2f} steps/sec ({pymdp_duration:.2f}s)")
        print(f"  Fallback: {fallback_steps_per_sec:.2f} steps/sec ({fallback_duration:.2f}s)")
        print(f"  Speedup with fallback: {fallback_steps_per_sec / pymdp_steps_per_sec:.1f}x")

        # Fallback should be significantly faster
        assert fallback_steps_per_sec > pymdp_steps_per_sec, "Fallback should be faster than PyMDP"

    def test_memory_usage_scaling(self):
        """Test memory usage as agents are added."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        agents = []
        memory_usage = [initial_memory]

        # Add agents incrementally and measure memory
        for i in range(20):
            agent = BasicExplorerAgent(f"memory_agent_{i}", f"Memory Agent {i}", grid_size=5)
            agent.start()
            agents.append(agent)

            # Run a few steps to initialize everything
            for step in range(5):
                observation = {
                    "position": [step % 5, (step + 1) % 5],
                    "surroundings": np.zeros((3, 3)),
                }
                _action = agent.step(observation)

            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(current_memory)

        final_memory = memory_usage[-1]
        memory_per_agent = (final_memory - initial_memory) / len(agents)

        print("\nMemory Usage Analysis:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory per agent: {memory_per_agent:.1f} MB")
        print(f"  Total agents: {len(agents)}")

        # Memory usage should be reasonable
        assert memory_per_agent < 50, f"Memory per agent too high: {memory_per_agent:.1f} MB"
        assert final_memory < 1000, f"Total memory usage too high: {final_memory:.1f} MB"

    def test_coordination_overhead(self):
        """Test performance overhead of agent coordination."""
        # Test isolated agents (no coordination)
        isolated_agents = [
            BasicExplorerAgent(f"isolated_{i}", f"Isolated {i}", grid_size=3) for i in range(5)
        ]

        for agent in isolated_agents:
            agent.start()

        num_steps = 30
        start_time = time.time()

        for step in range(num_steps):
            for agent in isolated_agents:
                observation = {
                    "position": [step % 3, (step + 1) % 3],
                    "surroundings": np.zeros((3, 3)),
                }
                _action = agent.step(observation)

        isolated_duration = time.time() - start_time
        isolated_ops_per_sec = (len(isolated_agents) * num_steps) / isolated_duration

        # Test coordinating agents (with visibility)
        coordinating_agents = [
            BasicExplorerAgent(f"coord_{i}", f"Coordinating {i}", grid_size=3) for i in range(5)
        ]

        for agent in coordinating_agents:
            agent.start()

        start_time = time.time()

        for step in range(num_steps):
            for i, agent in enumerate(coordinating_agents):
                observation = {
                    "position": [step % 3, (step + 1) % 3],
                    "surroundings": np.zeros((3, 3)),
                    "visible_agents": [
                        {"id": f"coord_{j}", "position": [j, step]}
                        for j in range(len(coordinating_agents))
                        if j != i
                    ],
                }
                _action = agent.step(observation)

        coordinating_duration = time.time() - start_time
        coordinating_ops_per_sec = (len(coordinating_agents) * num_steps) / coordinating_duration

        coordination_overhead = (
            (coordinating_duration - isolated_duration) / isolated_duration * 100
        )

        print("\nCoordination Overhead Analysis:")
        print(f"  Isolated: {isolated_ops_per_sec:.2f} ops/sec ({isolated_duration:.2f}s)")
        print(
            f"  Coordinating: {coordinating_ops_per_sec:.2f} ops/sec ({coordinating_duration:.2f}s)"
        )
        print(f"  Overhead: {coordination_overhead:.1f}%")

        # Coordination should add some overhead but not be excessive
        assert coordination_overhead < 200, (
            f"Coordination overhead too high: {coordination_overhead:.1f}%"
        )

    def test_resource_collection_performance(self):
        """Test performance of resource collection agents."""
        collectors = [
            ResourceCollectorAgent(f"collector_{i}", f"Collector {i}", grid_size=4)
            for i in range(3)
        ]

        for agent in collectors:
            agent.start()

        num_steps = 25
        start_time = time.time()

        for step in range(num_steps):
            for agent in collectors:
                observation = {
                    "position": [step % 4, (step + 1) % 4],
                    "visible_cells": (
                        [
                            {
                                "x": step % 4,
                                "y": (step + 1) % 4,
                                "type": "resource",
                                "amount": 5,
                            }
                        ]
                        if step % 3 == 0
                        else []
                    ),
                    "current_load": step % 10,
                }
                _action = agent.step(observation)

        duration = time.time() - start_time
        ops_per_second = (len(collectors) * num_steps) / duration

        print("\nResource Collection Performance:")
        print(f"  Agents: {len(collectors)}")
        print(f"  Steps: {num_steps}")
        print(f"  Performance: {ops_per_second:.2f} ops/sec")

        # Should handle resource collection reasonably well
        assert ops_per_second > 5, f"Resource collection too slow: {ops_per_second:.2f} ops/sec"


class TestRealWorldScenarios:
    """Test performance in realistic multi-agent scenarios."""

    def test_mixed_agent_coordination(self):
        """Test performance with mixed agent types."""
        # Create realistic agent mix
        explorers = [BasicExplorerAgent(f"exp_{i}", f"Explorer {i}", grid_size=4) for i in range(3)]
        collectors = [
            ResourceCollectorAgent(f"col_{i}", f"Collector {i}", grid_size=4) for i in range(2)
        ]
        coordinator = CoalitionCoordinatorAgent("coordinator", "Main Coordinator", max_agents=10)

        all_agents = explorers + collectors + [coordinator]

        for agent in all_agents:
            agent.start()

        num_steps = 15
        start_time = time.time()

        for step in range(num_steps):
            # Explorers
            for agent in explorers:
                observation = {
                    "position": [step % 4, (step + 1) % 4],
                    "surroundings": np.random.randint(0, 2, (3, 3)),
                }
                _action = agent.step(observation)

            # Collectors
            for agent in collectors:
                observation = {
                    "position": [step % 4, (step * 2) % 4],
                    "visible_cells": (
                        [
                            {
                                "x": step % 4,
                                "y": step % 4,
                                "type": "resource",
                                "amount": 3,
                            }
                        ]
                        if step % 4 == 0
                        else []
                    ),
                    "current_load": step % 8,
                }
                _action = agent.step(observation)

            # Coordinator
            observation = {
                "visible_agents": [
                    {
                        "id": agent.agent_id,
                        "position": [step % 4, step % 4],
                        "status": "active",
                    }
                    for agent in all_agents[:-1]  # All except coordinator
                ]
            }
            _action = coordinator.step(observation)

        duration = time.time() - start_time
        total_operations = len(all_agents) * num_steps
        ops_per_second = total_operations / duration

        print("\nMixed Agent Coordination Performance:")
        print(f"  Total agents: {len(all_agents)} (3 explorers, 2 collectors, 1 coordinator)")
        print(f"  Steps: {num_steps}")
        print(f"  Performance: {ops_per_second:.2f} ops/sec")
        print(f"  Duration: {duration:.2f}s")

        # Should handle mixed coordination
        assert ops_per_second > 3, f"Mixed coordination too slow: {ops_per_second:.2f} ops/sec"

        # Verify all agents operated
        for agent in all_agents:
            assert agent.total_steps == num_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

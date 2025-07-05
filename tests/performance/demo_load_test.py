#!/usr/bin/env python3
"""
Demo: Multi-Agent Coordination Load Testing

This script demonstrates the load testing framework with a simple example
that validates the documented 72% efficiency loss at scale.
"""

import time

import numpy as np
from agent_simulation_framework import AgentSpawnConfig, AgentType, SimulationEnvironment
from test_coordination_load import CoordinationAgent, CoordinationLoadTester


def demo_coordination_overhead():
    """Demonstrate coordination overhead measurement."""
    print("=" * 60)
    print("DEMO: Measuring Coordination Overhead")
    print("=" * 60)

    tester = CoordinationLoadTester()

    # Test with increasing agent counts
    for agent_count in [1, 10, 30, 50]:
        print(f"\n🤖 Testing with {agent_count} agents...")

        metrics = tester.measure_coordination_overhead(agent_count)

        print("  📊 Results:")
        print(f"     - Actual efficiency: {metrics.actual_efficiency:.1%}")
        print(f"     - Coordination overhead: {metrics.coordination_overhead:.1%}")
        print(f"     - Efficiency loss: {metrics.efficiency_loss():.1f}%")
        print(f"     - Message latency: {metrics.coordination_latency_ms:.1f}ms")
        print(f"     - Total messages: {metrics.total_messages}")

        # Show if it matches documentation
        if agent_count == 50:
            if 70 <= metrics.efficiency_loss() <= 75:
                print("  ✅ Efficiency loss matches documented ~72%!")
            else:
                print("  ❌ Efficiency loss doesn't match documentation")


def demo_agent_simulation():
    """Demonstrate agent simulation framework."""
    print("\n" + "=" * 60)
    print("DEMO: Agent Simulation Framework")
    print("=" * 60)

    # Create simulation environment
    env = SimulationEnvironment(world_size=10)

    # Spawn different types of agents
    print("\n🏭 Spawning agents...")

    # Explorers
    explorer_config = AgentSpawnConfig(
        agent_type=AgentType.EXPLORER, count=5, grid_size=10, performance_mode="fast"
    )
    explorers = env.lifecycle_manager.spawn_batch(explorer_config)
    print(f"  ✅ Spawned {len(explorers)} explorers")

    # Collectors
    collector_config = AgentSpawnConfig(
        agent_type=AgentType.COLLECTOR, count=3, grid_size=10, performance_mode="fast"
    )
    collectors = env.lifecycle_manager.spawn_batch(collector_config)
    print(f"  ✅ Spawned {len(collectors)} collectors")

    # Coordinator
    coordinator_config = AgentSpawnConfig(
        agent_type=AgentType.COORDINATOR, count=1, config_overrides={"max_agents": 10}
    )
    coordinators = env.lifecycle_manager.spawn_batch(coordinator_config)
    print(f"  ✅ Spawned {len(coordinators)} coordinator")

    # Run simulation
    print("\n🏃 Running simulation for 5 seconds...")
    start_time = time.time()
    tick_count = 0

    while time.time() - start_time < 5.0:
        tick_result = env.run_tick()
        tick_count += 1

        if tick_count % 10 == 0:
            print(f"  Tick {tick_count}: {tick_result['agents_processed']} agents processed")

    # Get final metrics
    all_metrics = env.lifecycle_manager.get_all_metrics()

    print("\n📈 Simulation Summary:")
    print(f"  - Total ticks: {tick_count}")
    print(f"  - Ticks/second: {tick_count / 5.0:.1f}")

    total_steps = sum(m["total_steps"] for m in all_metrics.values())
    print(f"  - Total agent steps: {total_steps}")
    print(f"  - Steps/second: {total_steps / 5.0:.1f}")

    # Cleanup
    env.lifecycle_manager.terminate_all()


def demo_coordination_scenarios():
    """Demonstrate specific coordination scenarios."""
    print("\n" + "=" * 60)
    print("DEMO: Coordination Scenarios")
    print("=" * 60)

    tester = CoordinationLoadTester()

    # Spawn agents for testing
    print("\n🤖 Setting up 20 agents for coordination tests...")
    tester.spawn_agents(20)

    # Test 1: Task Handoffs
    print("\n📦 Testing task handoffs...")
    handoff_results = tester.simulate_task_handoffs(duration_seconds=3.0)
    print(f"  - Total handoffs: {handoff_results['total_handoffs']}")
    print(f"  - Success rate: {handoff_results['success_rate']:.1%}")
    print(f"  - Handoffs/second: {handoff_results['handoffs_per_second']:.1f}")

    # Test 2: Resource Contention
    print("\n🔒 Testing resource contention...")
    contention_results = tester.simulate_resource_contention(duration_seconds=3.0)
    print(f"  - Contention events: {contention_results['contentions']}")
    print(f"  - Contention rate: {contention_results['contention_rate']:.1%}")
    print(f"  - Resolution rate: {contention_results['resolution_rate']:.1%}")

    # Test 3: Consensus Building
    print("\n🗳️ Testing consensus building...")
    consensus_results = tester.simulate_consensus_building(consensus_rounds=5)
    print(
        f"  - Successful consensus: {consensus_results['successful_consensus']}/{consensus_results['rounds']}"
    )
    print(f"  - Average time: {consensus_results['avg_consensus_time_ms']:.1f}ms")
    print(f"  - Success rate: {consensus_results['success_rate']:.1%}")


def main():
    """Run all demonstrations."""
    print("🚀 FreeAgentics Load Testing Framework Demo")
    print("=" * 80)
    print("This demo shows how the load tests validate architectural limitations.")
    print("Expected: ~72% efficiency loss at 50 agents due to Python GIL\n")

    # Run demonstrations
    demo_coordination_overhead()
    demo_agent_simulation()
    demo_coordination_scenarios()

    # Summary
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\n📚 Key Takeaways:")
    print("1. Coordination overhead increases non-linearly with agent count")
    print("2. Python's GIL limits practical scaling to ~50 agents")
    print("3. At 50 agents, expect ~28.4% efficiency (72% loss)")
    print("4. Message-based coordination adds measurable latency")
    print("5. System remains stable under failures and contention")

    print("\n🔍 To run full load tests:")
    print("   python tests/performance/run_coordination_load_tests.py")

    print("\n📖 See README_LOAD_TESTS.md for detailed documentation")


if __name__ == "__main__":
    main()

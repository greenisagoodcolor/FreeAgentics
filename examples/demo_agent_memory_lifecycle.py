#!/usr/bin/env python3
"""Demonstration of Agent Memory Lifecycle Management.

This demo shows the functionality implemented for Task 5.5: Design agent memory lifecycle management.

The demo showcases:
1. Agent registration and memory tracking
2. Memory usage updates and limit checking
3. Agent hibernation and awakening
4. Agent recycling for reuse
5. Automatic cleanup of idle agents
6. Memory pressure monitoring
7. Lifecycle statistics collection
"""

import logging
import random
import time

from agents.memory_optimization.lifecycle_manager import (
    AgentLifecycleState,
    cleanup_agent_memory,
    get_global_lifecycle_manager,
    get_memory_statistics,
    managed_agent_memory,
    register_agent_memory,
    update_agent_memory_usage,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def demo_basic_lifecycle():
    """Demonstrate basic agent lifecycle operations."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Agent Lifecycle Operations")
    print("=" * 60)

    # Get the global manager
    manager = get_global_lifecycle_manager()

    # Register some agents
    agent_ids = ["explorer_001", "collector_002", "analyzer_003"]

    print("\n1. Registering agents...")
    for agent_id in agent_ids:
        profile = register_agent_memory(agent_id, memory_limit_mb=30.0)
        print(f"   Registered {agent_id} with {profile.memory_limit_mb}MB limit")

    # Update memory usage for agents
    print("\n2. Updating memory usage...")
    update_agent_memory_usage(
        "explorer_001", belief_mb=8.0, matrix_mb=4.0, other_mb=2.0
    )
    update_agent_memory_usage(
        "collector_002", belief_mb=12.0, matrix_mb=6.0, other_mb=3.0
    )
    update_agent_memory_usage(
        "analyzer_003", belief_mb=15.0, matrix_mb=8.0, other_mb=4.0
    )

    # Show memory statistics
    stats = get_memory_statistics()
    print(f"   Total memory usage: {stats['global']['total_memory_mb']:.1f}MB")
    print(f"   Memory pressure: {stats['global']['memory_pressure']:.1%}")

    # Hibernate an agent
    print("\n3. Hibernating agent...")
    success = manager.hibernate_agent("analyzer_003")
    print(f"   Hibernated analyzer_003: {success}")

    # Wake the agent
    print("\n4. Waking agent...")
    success = manager.wake_agent("analyzer_003")
    print(f"   Awakened analyzer_003: {success}")

    # Recycle an agent
    print("\n5. Recycling agent...")
    success = manager.recycle_agent("collector_002")
    print(f"   Recycled collector_002: {success}")

    # Show final statistics
    final_stats = get_memory_statistics()
    print("\n6. Final statistics:")
    print(f"   Total agents: {final_stats['global']['total_agents']}")
    print(f"   Agent states: {final_stats['agent_states']}")
    print(f"   Total memory: {final_stats['global']['total_memory_mb']:.1f}MB")


def demo_memory_context_manager():
    """Demonstrate the memory context manager."""
    print("\n" + "=" * 60)
    print("DEMO 2: Memory Context Manager")
    print("=" * 60)

    agent_id = "context_agent_001"

    print(f"\n1. Using managed agent memory context for {agent_id}...")

    with managed_agent_memory(agent_id, memory_limit_mb=25.0) as profile:
        print(
            f"   Agent {profile.agent_id} created with {profile.memory_limit_mb}MB limit"
        )
        print(f"   Initial state: {profile.state}")

        # Simulate some memory operations
        profile.update_memory_usage(10.0, 5.0, 3.0)
        print(f"   Updated memory usage: {profile.current_memory_mb}MB")
        print(f"   Within limit: {profile.check_memory_limit()}")
        print(f"   Memory efficiency: {profile.get_memory_efficiency():.1%}")

    # Check agent state after context
    manager = get_global_lifecycle_manager()
    final_profile = manager.get_agent_profile(agent_id)
    print(f"\n2. Agent state after context: {final_profile.state}")
    print(f"   Lifecycle events: {len(final_profile.lifecycle_events)}")


def demo_memory_pressure_cleanup():
    """Demonstrate memory pressure and automatic cleanup."""
    print("\n" + "=" * 60)
    print("DEMO 3: Memory Pressure and Cleanup")
    print("=" * 60)

    get_global_lifecycle_manager()

    # Create multiple agents with high memory usage
    print("\n1. Creating agents with high memory usage...")
    agent_ids = [f"heavy_agent_{i:03d}" for i in range(5)]

    for agent_id in agent_ids:
        profile = register_agent_memory(agent_id, memory_limit_mb=50.0)
        # Simulate high memory usage
        update_agent_memory_usage(
            agent_id,
            belief_mb=random.uniform(15, 25),  # nosec B311 - Demo simulation only
            matrix_mb=random.uniform(10, 15),  # nosec B311 - Demo simulation only
            other_mb=random.uniform(5, 10),  # nosec B311 - Demo simulation only
        )

        # Set agent to active state and old access time for some
        if random.random() > 0.5:  # nosec B311 - Demo simulation only
            profile.state = AgentLifecycleState.ACTIVE
            profile.last_accessed = time.time() - random.uniform(70, 150)  # nosec B311 - Demo simulation only  # Old access

    # Check memory pressure
    stats = get_memory_statistics()
    print(f"   Total memory: {stats['global']['total_memory_mb']:.1f}MB")
    print(f"   Memory pressure: {stats['global']['memory_pressure']:.1%}")

    # Force cleanup
    print("\n2. Forcing cleanup...")
    cleanup_stats = cleanup_agent_memory()
    print(f"   Cleanup results: {cleanup_stats}")

    # Check memory after cleanup
    final_stats = get_memory_statistics()
    print("\n3. Memory after cleanup:")
    print(f"   Total memory: {final_stats['global']['total_memory_mb']:.1f}MB")
    print(f"   Memory pressure: {final_stats['global']['memory_pressure']:.1%}")
    print(f"   Agent states: {final_stats['agent_states']}")


def demo_lifecycle_events_tracking():
    """Demonstrate lifecycle events tracking."""
    print("\n" + "=" * 60)
    print("DEMO 4: Lifecycle Events Tracking")
    print("=" * 60)

    manager = get_global_lifecycle_manager()
    agent_id = "tracked_agent_001"

    # Register agent
    print(f"\n1. Registering {agent_id}...")
    profile = register_agent_memory(agent_id, memory_limit_mb=40.0)

    # Perform various operations
    print("\n2. Performing operations and tracking events...")

    # Memory updates
    update_agent_memory_usage(agent_id, 10.0, 5.0, 2.0)
    profile.record_lifecycle_event(
        "custom_operation", {"operation": "inference", "duration_ms": 150}
    )

    # State transitions
    manager.hibernate_agent(agent_id)
    manager.wake_agent(agent_id)
    manager.recycle_agent(agent_id)

    # Show lifecycle events
    print(f"\n3. Lifecycle events for {agent_id}:")
    for timestamp, event, metadata in profile.lifecycle_events[
        -10:
    ]:  # Show last 10 events
        event_time = time.strftime("%H:%M:%S", time.localtime(timestamp))
        print(f"   {event_time}: {event} - {metadata}")

    # Show memory snapshots
    print("\n4. Memory snapshots (last 3):")
    for snapshot in list(profile.memory_snapshots)[-3:]:
        snapshot_time = time.strftime("%H:%M:%S", time.localtime(snapshot.timestamp))
        print(
            f"   {snapshot_time}: {snapshot.total_memory_mb:.1f}MB (state: {snapshot.state.value})"
        )


def demo_agent_pool_simulation():
    """Demonstrate agent pool with realistic simulation."""
    print("\n" + "=" * 60)
    print("DEMO 5: Agent Pool Simulation")
    print("=" * 60)

    manager = get_global_lifecycle_manager()

    # Simulate a pool of agents with different roles
    agent_roles = [
        ("explorer", 20.0),
        ("collector", 25.0),
        ("analyzer", 35.0),
        ("coordinator", 30.0),
        ("monitor", 15.0),
    ]

    print("\n1. Creating agent pool...")
    active_agents = []

    for i in range(8):  # Create 8 agents
        role, memory_limit = random.choice(agent_roles)  # nosec B311 - Demo simulation only
        agent_id = f"{role}_{i:03d}"

        profile = register_agent_memory(agent_id, memory_limit_mb=memory_limit)
        profile.state = AgentLifecycleState.ACTIVE
        active_agents.append(agent_id)

        # Simulate initial memory usage
        belief_mb = random.uniform(5, memory_limit * 0.4)  # nosec B311 - Demo simulation only
        matrix_mb = random.uniform(2, memory_limit * 0.3)  # nosec B311 - Demo simulation only
        other_mb = random.uniform(1, memory_limit * 0.2)  # nosec B311 - Demo simulation only

        update_agent_memory_usage(agent_id, belief_mb, matrix_mb, other_mb)
        print(
            f"   Created {agent_id}: {belief_mb + matrix_mb + other_mb:.1f}MB/{memory_limit}MB"
        )

    # Simulate agent activity over time
    print("\n2. Simulating agent activity...")
    for round_num in range(3):
        print(f"\n   Round {round_num + 1}:")

        # Some agents do work (update memory)
        working_agents = random.sample(active_agents, random.randint(3, 6))  # nosec B311 - Demo simulation only
        for agent_id in working_agents:
            profile = manager.get_agent_profile(agent_id)
            if profile and profile.state == AgentLifecycleState.ACTIVE:
                # Simulate memory usage change
                delta_mb = random.uniform(-2, 5)  # nosec B311 - Demo simulation only
                new_total = max(1.0, profile.current_memory_mb + delta_mb)

                # Distribute the change across categories
                belief_mb = new_total * 0.5
                matrix_mb = new_total * 0.3
                other_mb = new_total * 0.2

                update_agent_memory_usage(agent_id, belief_mb, matrix_mb, other_mb)
                print(
                    f"     {agent_id}: {new_total:.1f}MB (efficiency: {profile.get_memory_efficiency():.1%})"
                )

        # Some agents become idle (simulate old access time)
        idle_agents = random.sample(active_agents, random.randint(1, 3))  # nosec B311 - Demo simulation only
        for agent_id in idle_agents:
            profile = manager.get_agent_profile(agent_id)
            if profile:
                profile.last_accessed = time.time() - random.uniform(70, 200)  # nosec B311 - Demo simulation only

        # Run cleanup
        cleanup_stats = manager.cleanup_idle_agents()
        if cleanup_stats["hibernated"] > 0 or cleanup_stats["recycled"] > 0:
            print(
                f"     Cleanup: hibernated {cleanup_stats['hibernated']}, recycled {cleanup_stats['recycled']}"
            )

        # Show current statistics
        stats = get_memory_statistics()
        print(f"     Total memory: {stats['global']['total_memory_mb']:.1f}MB")
        print(f"     Memory pressure: {stats['global']['memory_pressure']:.1%}")
        print(f"     States: {stats['agent_states']}")

        # Pause between rounds
        time.sleep(0.5)

    # Final statistics
    print("\n3. Final pool statistics:")
    final_stats = get_memory_statistics()
    print(f"   Lifecycle stats: {final_stats['lifecycle_stats']}")


def main():
    """Run all demonstrations."""
    print("Agent Memory Lifecycle Management Demo")
    print("Implementing Task 5.5: Design agent memory lifecycle management")

    try:
        # Run demonstrations
        demo_basic_lifecycle()
        demo_memory_context_manager()
        demo_memory_pressure_cleanup()
        demo_lifecycle_events_tracking()
        demo_agent_pool_simulation()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey features demonstrated:")
        print("✓ Agent registration and memory tracking")
        print("✓ Memory usage updates and limit checking")
        print("✓ Agent hibernation and awakening")
        print("✓ Agent recycling for reuse")
        print("✓ Automatic cleanup of idle agents")
        print("✓ Memory pressure monitoring")
        print("✓ Lifecycle statistics collection")
        print("✓ Context manager for resource management")
        print("✓ Lifecycle events tracking")
        print("✓ Realistic agent pool simulation")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        # Cleanup
        manager = get_global_lifecycle_manager()
        manager.shutdown()
        print("\nCleanup completed.")


if __name__ == "__main__":
    main()

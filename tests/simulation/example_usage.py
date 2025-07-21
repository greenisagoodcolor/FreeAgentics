"""Example usage of the concurrent simulation framework.

This script demonstrates how to use the simulation framework
programmatically for custom testing scenarios.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from tests.simulation.concurrent_simulator import (
    ConcurrentSimulator,
    SimulationConfig,
)
from tests.simulation.scenarios import ScenarioScheduler, SimulationScenarios
from tests.simulation.user_personas import (
    PersonaType,
    ResearcherBehavior,
)


async def example_basic_simulation():
    """Run a basic simulation with mixed workload."""
    print("Example 1: Basic Mixed Workload Simulation")
    print("-" * 40)

    # Get a predefined scenario
    config = SimulationScenarios.mixed_workload()

    # Customize some parameters
    config.duration_seconds = 600  # 10 minutes instead of 1 hour
    config.user_spawn_rate = 5.0  # Faster spawning

    # Create and run simulator
    simulator = ConcurrentSimulator(config)
    await simulator.run()

    # Get results
    summary = simulator.get_summary()
    print("\nSimulation completed!")
    print(f"Total messages sent: {summary['metrics']['messages']['sent']}")
    print(f"Average DB latency: {summary['metrics']['database']['avg_latency_ms']:.1f}ms")


async def example_custom_scenario():
    """Create and run a custom scenario."""
    print("\nExample 2: Custom Scenario")
    print("-" * 40)

    # Create custom configuration
    config = SimulationConfig(
        name="custom_test",
        description="Custom test scenario with specific user distribution",
        duration_seconds=300,  # 5 minutes
        user_distribution={
            PersonaType.RESEARCHER: 10,
            PersonaType.COORDINATOR: 5,
            PersonaType.OBSERVER: 15,
        },
        user_spawn_rate=2.0,
        warmup_period=30,
        cooldown_period=30,
        enable_errors=True,
        error_injection_rate=0.02,
    )

    # Run simulation
    simulator = ConcurrentSimulator(config)
    await simulator.run()

    # Analyze persona-specific metrics
    summary = simulator.get_summary()
    print("\nPer-persona metrics:")
    for persona, metrics in summary["metrics"]["personas"].items():
        print(f"  {persona}: {metrics['messages']} messages")


async def example_stress_test_with_monitoring():
    """Run a stress test with detailed monitoring."""
    print("\nExample 3: Stress Test with Monitoring")
    print("-" * 40)

    # Use stress test scenario
    config = SimulationScenarios.stress_test()
    config.duration_seconds = 300  # 5 minutes for example
    config.metrics_interval = 10.0  # More frequent monitoring

    # Create simulator
    simulator = ConcurrentSimulator(config)

    # Custom monitoring during simulation
    async def monitor_simulation():
        while simulator.is_running:
            active_users = sum(1 for u in simulator.users.values() if u.connected)
            print(f"Active users: {active_users}, Messages: {simulator.metrics.messages_sent}")
            await asyncio.sleep(10)

    # Run simulation with monitoring
    monitor_task = asyncio.create_task(monitor_simulation())
    await simulator.run()
    monitor_task.cancel()

    print("\nStress test completed!")


async def example_scheduled_scenarios():
    """Run multiple scenarios in sequence."""
    print("\nExample 4: Scheduled Scenarios")
    print("-" * 40)

    # Create scheduler
    scheduler = ScenarioScheduler(results_base_path=Path("example_results"))

    # Add custom sequence
    scheduler.add_scenario(SimulationScenarios.mixed_workload(), delay_minutes=0)
    scheduler.add_scenario(
        SimulationScenarios.burst_activity(),
        delay_minutes=2,  # 2 minutes after first scenario
    )
    scheduler.add_scenario(
        SimulationScenarios.database_intensive(),
        delay_minutes=2,  # 2 minutes after second scenario
    )

    # Run schedule
    await scheduler.run_schedule()

    print("\nAll scheduled scenarios completed!")


async def example_custom_user_behavior():
    """Demonstrate custom user behavior implementation."""
    print("\nExample 5: Custom User Behavior")
    print("-" * 40)

    # Create a custom behavior for a specific test
    class TestingBehavior(ResearcherBehavior):
        """Custom behavior for specific testing patterns."""

        async def decide_next_action(self):
            # Always query specific agents in sequence
            if not self.state.get("test_agents"):
                self.state["test_agents"] = [f"test_agent_{i}" for i in range(10)]
                self.state["current_index"] = 0

            # Query each agent in sequence
            idx = self.state["current_index"]
            if idx < len(self.state["test_agents"]):
                agent_id = self.state["test_agents"][idx]
                self.state["current_index"] += 1

                return {
                    "type": "query",
                    "query_type": "agent_status",
                    "agent_ids": [agent_id],
                    "include_history": True,
                }
            else:
                # Reset and start over
                self.state["current_index"] = 0
                return {"type": "ping"}

    # Use in simulation
    print("Custom behavior would query agents in specific sequence")


async def example_results_analysis():
    """Analyze simulation results."""
    print("\nExample 6: Results Analysis")
    print("-" * 40)

    # Run a short simulation
    config = SimulationScenarios.mixed_workload()
    config.duration_seconds = 120  # 2 minutes
    config.export_results = True
    config.results_path = Path("example_analysis")

    simulator = ConcurrentSimulator(config)
    await simulator.run()

    # Load and analyze results
    (config.results_path / f"metrics_{config.name}_{datetime.now().strftime('%Y%m%d')}")

    # Get summary for analysis
    summary = simulator.get_summary()

    # Calculate some statistics
    total_users = summary["metrics"]["users"]["created"]
    total_messages = summary["metrics"]["messages"]["sent"]
    messages_per_user = total_messages / max(total_users, 1)

    print("\nAnalysis Results:")
    print(f"Average messages per user: {messages_per_user:.1f}")
    print(f"System efficiency: {summary['metrics']['messages']['success_rate']:.1%}")

    # Analyze by persona
    print("\nPersona Performance:")
    for persona, data in summary["metrics"]["personas"].items():
        if "messages" in data and data.get("users", 0) > 0:
            avg_messages = data["messages"] / data.get("users", 1)
            print(f"  {persona}: {avg_messages:.1f} messages/user")


async def example_database_integration():
    """Example showing database integration testing."""
    print("\nExample 7: Database Integration Testing")
    print("-" * 40)

    # Configure for database testing
    config = SimulationScenarios.database_intensive()
    config.duration_seconds = 180  # 3 minutes
    config.db_pool_size = 50  # Large pool for testing

    # Add custom database monitoring
    from tests.db_infrastructure.performance_monitor import PerformanceMonitor

    db_monitor = PerformanceMonitor()

    # Run simulation
    simulator = ConcurrentSimulator(config)

    # Monitor database performance
    db_monitor.start_monitoring()
    await simulator.run()
    db_stats = db_monitor.stop_monitoring()

    print("\nDatabase Performance:")
    print(f"Total queries: {db_stats.get('total_queries', 0)}")
    print(f"Average query time: {db_stats.get('avg_query_time', 0):.1f}ms")


async def example_websocket_patterns():
    """Example testing different WebSocket patterns."""
    print("\nExample 8: WebSocket Pattern Testing")
    print("-" * 40)

    # Create config for WebSocket pattern testing
    config = SimulationConfig(
        name="websocket_patterns",
        description="Test various WebSocket interaction patterns",
        duration_seconds=300,
        user_distribution={
            PersonaType.RESEARCHER: 5,  # Steady connections
            PersonaType.COORDINATOR: 5,  # High message rate
            PersonaType.OBSERVER: 10,  # Long-lived connections
            PersonaType.DEVELOPER: 5,  # Connect/disconnect cycles
        },
        ws_reconnect_attempts=10,
        ws_reconnect_delay=1.0,
    )

    simulator = ConcurrentSimulator(config)
    await simulator.run()

    print("\nWebSocket Statistics:")
    print(f"Total connections: {simulator.metrics.ws_connections}")
    print(f"Total disconnections: {simulator.metrics.ws_disconnections}")
    print(f"Connection errors: {simulator.metrics.ws_errors}")


async def main():
    """Run all examples."""
    examples = [
        example_basic_simulation,
        example_custom_scenario,
        example_stress_test_with_monitoring,
        # example_scheduled_scenarios,  # Commented out as it takes longer
        example_custom_user_behavior,
        example_results_analysis,
        # example_database_integration,  # Requires database setup
        example_websocket_patterns,
    ]

    for example in examples:
        try:
            await example()
            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            print(f"Example failed: {e}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("Concurrent Simulation Framework Examples")
    print("=" * 60)
    print("These examples demonstrate various features of the simulation framework.")
    print("Some examples are shortened for demonstration purposes.\n")

    asyncio.run(main())

    print("\nAll examples completed!")
    print("\nFor production use, run simulations with appropriate durations and monitoring.")
    print("Use the command-line interface for easier execution:")
    print("  python run_simulation.py run mixed_workload --duration 3600")
    print("  python run_simulation.py schedule daily")
    print("  python run_simulation.py list")

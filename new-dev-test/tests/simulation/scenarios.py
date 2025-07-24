"""Predefined simulation scenarios for testing different system behaviors.

This module provides various simulation scenarios that test different
aspects of the system under realistic concurrent load.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tests.simulation.concurrent_simulator import SimulationConfig
from tests.simulation.user_personas import PersonaType


class SimulationScenarios:
    """Collection of predefined simulation scenarios."""

    @staticmethod
    def research_conference() -> SimulationConfig:
        """Simulate a research conference with many researchers and observers.

        This scenario tests:
        - High query load from researchers
        - Many passive observers subscribing to events
        - Burst patterns during presentation times
        - Complex analytics queries
        """
        return SimulationConfig(
            name="research_conference",
            description="Virtual research conference with presentations and discussions",
            duration_seconds=3600,  # 1 hour conference
            user_distribution={
                PersonaType.RESEARCHER: 50,
                PersonaType.OBSERVER: 150,
                PersonaType.ANALYST: 20,
                PersonaType.COORDINATOR: 10,
                PersonaType.ADMIN: 2,
            },
            user_spawn_rate=5.0,  # Fast spawning for conference start
            warmup_period=300,  # 5 min warmup
            cooldown_period=180,  # 3 min cooldown
            ws_reconnect_attempts=5,  # Higher for stability
            enable_monitoring=True,
            metrics_interval=10.0,
        )

    @staticmethod
    def coalition_operations() -> SimulationConfig:
        """Simulate intensive coalition formation and management.

        This scenario tests:
        - Coalition formation under load
        - Coordinator-heavy operations
        - Agent assignment and management
        - Real-time coordination messages
        """
        return SimulationConfig(
            name="coalition_operations",
            description="Military-style coalition operations with multiple coordinators",
            duration_seconds=1800,  # 30 min operation
            user_distribution={
                PersonaType.COORDINATOR: 30,
                PersonaType.RESEARCHER: 5,
                PersonaType.OBSERVER: 20,
                PersonaType.ADMIN: 3,
            },
            user_spawn_rate=2.0,
            warmup_period=120,
            cooldown_period=120,
            ws_path="/ws",
            enable_errors=True,
            error_injection_rate=0.02,  # 2% error rate for stress testing
        )

    @staticmethod
    def system_monitoring() -> SimulationConfig:
        """Simulate heavy monitoring and admin operations.

        This scenario tests:
        - Monitoring system load
        - Admin commands and queries
        - System metrics collection
        - Alert handling
        """
        return SimulationConfig(
            name="system_monitoring",
            description="24/7 system monitoring simulation",
            duration_seconds=7200,  # 2 hours
            user_distribution={
                PersonaType.ADMIN: 10,
                PersonaType.ANALYST: 15,
                PersonaType.OBSERVER: 30,
                PersonaType.DEVELOPER: 5,
            },
            user_spawn_rate=1.0,
            warmup_period=300,
            cooldown_period=300,
            enable_monitoring=True,
            metrics_interval=5.0,
            db_pool_size=30,  # Larger pool for monitoring queries
        )

    @staticmethod
    def mixed_workload() -> SimulationConfig:
        """Simulate realistic mixed workload with all persona types.

        This scenario tests:
        - Balanced load across all operations
        - Diverse query and command patterns
        - Natural user behavior mix
        - System stability under varied load
        """
        return SimulationConfig(
            name="mixed_workload",
            description="Realistic mixed workload simulation",
            duration_seconds=3600,  # 1 hour
            user_distribution={
                PersonaType.RESEARCHER: 40,
                PersonaType.COORDINATOR: 25,
                PersonaType.OBSERVER: 60,
                PersonaType.ADMIN: 5,
                PersonaType.DEVELOPER: 15,
                PersonaType.ANALYST: 20,
            },
            user_spawn_rate=3.0,
            warmup_period=300,
            cooldown_period=300,
            network_latency_range=(0.01, 0.05),
            enable_errors=True,
            error_injection_rate=0.01,
        )

    @staticmethod
    def stress_test() -> SimulationConfig:
        """High-stress scenario to find system limits.

        This scenario tests:
        - Maximum concurrent connections
        - High message throughput
        - Database connection pooling
        - Error recovery
        """
        return SimulationConfig(
            name="stress_test",
            description="High-stress test to find system breaking points",
            duration_seconds=1800,  # 30 minutes
            user_distribution={
                PersonaType.RESEARCHER: 100,
                PersonaType.COORDINATOR: 100,
                PersonaType.OBSERVER: 100,
                PersonaType.ADMIN: 10,
                PersonaType.DEVELOPER: 50,
                PersonaType.ANALYST: 50,
            },
            user_spawn_rate=10.0,  # Aggressive spawning
            warmup_period=60,  # Quick warmup
            cooldown_period=60,  # Quick cooldown
            ws_reconnect_attempts=10,
            enable_errors=True,
            error_injection_rate=0.05,  # 5% error rate
            db_pool_size=50,
            db_max_overflow=20,
        )

    @staticmethod
    def burst_activity() -> SimulationConfig:
        """Simulate burst activity patterns.

        This scenario tests:
        - Sudden load spikes
        - Connection/disconnection cycles
        - Queue handling
        - Recovery from bursts
        """
        return SimulationConfig(
            name="burst_activity",
            description="Burst activity patterns with idle periods",
            duration_seconds=2400,  # 40 minutes
            user_distribution={
                PersonaType.RESEARCHER: 30,
                PersonaType.COORDINATOR: 20,
                PersonaType.OBSERVER: 40,
                PersonaType.DEVELOPER: 10,
            },
            user_spawn_rate=20.0,  # Very fast during bursts
            warmup_period=30,  # Quick warmup
            cooldown_period=30,  # Quick cooldown
            network_latency_range=(
                0.001,
                0.01,
            ),  # Low latency for burst response
        )

    @staticmethod
    def development_testing() -> SimulationConfig:
        """Simulate developer testing patterns.

        This scenario tests:
        - Frequent create/delete operations
        - Debug queries
        - Error injection
        - Rapid iteration
        """
        return SimulationConfig(
            name="development_testing",
            description="Developer testing and debugging simulation",
            duration_seconds=1200,  # 20 minutes
            user_distribution={
                PersonaType.DEVELOPER: 20,
                PersonaType.RESEARCHER: 10,
                PersonaType.ADMIN: 5,
                PersonaType.ANALYST: 5,
            },
            user_spawn_rate=2.0,
            warmup_period=60,
            cooldown_period=60,
            enable_errors=True,
            error_injection_rate=0.1,  # 10% error rate for testing
            ws_reconnect_attempts=3,
        )

    @staticmethod
    def long_running() -> SimulationConfig:
        """Simulate long-running stable operations.

        This scenario tests:
        - Memory leaks
        - Connection stability
        - Performance degradation
        - Resource exhaustion
        """
        return SimulationConfig(
            name="long_running",
            description="Long-running stability test",
            duration_seconds=14400,  # 4 hours
            user_distribution={
                PersonaType.RESEARCHER: 20,
                PersonaType.COORDINATOR: 15,
                PersonaType.OBSERVER: 50,
                PersonaType.ADMIN: 5,
                PersonaType.ANALYST: 10,
            },
            user_spawn_rate=0.5,  # Slow, steady spawning
            warmup_period=600,  # 10 min warmup
            cooldown_period=600,  # 10 min cooldown
            ws_reconnect_attempts=10,
            ws_reconnect_delay=5.0,
            enable_monitoring=True,
            metrics_interval=30.0,  # Less frequent for long run
        )

    @staticmethod
    def database_intensive() -> SimulationConfig:
        """Simulate database-intensive operations.

        This scenario tests:
        - Complex queries
        - High write load
        - Transaction handling
        - Connection pool exhaustion
        """
        return SimulationConfig(
            name="database_intensive",
            description="Database-intensive operations simulation",
            duration_seconds=1800,  # 30 minutes
            user_distribution={
                PersonaType.RESEARCHER: 60,  # Heavy queries
                PersonaType.COORDINATOR: 40,  # Many writes
                PersonaType.ANALYST: 30,  # Complex analytics
                PersonaType.ADMIN: 10,
            },
            user_spawn_rate=3.0,
            warmup_period=180,
            cooldown_period=180,
            db_pool_size=40,
            db_max_overflow=20,
            enable_monitoring=True,
        )

    @staticmethod
    def failover_test() -> SimulationConfig:
        """Simulate failover and recovery scenarios.

        This scenario tests:
        - Connection recovery
        - State persistence
        - Error handling
        - Graceful degradation
        """
        return SimulationConfig(
            name="failover_test",
            description="Failover and recovery simulation",
            duration_seconds=1800,  # 30 minutes
            user_distribution={
                PersonaType.RESEARCHER: 25,
                PersonaType.COORDINATOR: 25,
                PersonaType.OBSERVER: 25,
                PersonaType.ADMIN: 10,
                PersonaType.DEVELOPER: 15,
            },
            user_spawn_rate=2.0,
            warmup_period=120,
            cooldown_period=120,
            ws_reconnect_attempts=20,
            ws_reconnect_delay=2.0,
            enable_errors=True,
            error_injection_rate=0.1,  # High error rate
            network_latency_range=(0.01, 0.5),  # Variable latency
        )

    @staticmethod
    def get_scenario(name: str) -> Optional[SimulationConfig]:
        """Get a scenario by name."""
        scenarios = {
            "research_conference": SimulationScenarios.research_conference,
            "coalition_operations": SimulationScenarios.coalition_operations,
            "system_monitoring": SimulationScenarios.system_monitoring,
            "mixed_workload": SimulationScenarios.mixed_workload,
            "stress_test": SimulationScenarios.stress_test,
            "burst_activity": SimulationScenarios.burst_activity,
            "development_testing": SimulationScenarios.development_testing,
            "long_running": SimulationScenarios.long_running,
            "database_intensive": SimulationScenarios.database_intensive,
            "failover_test": SimulationScenarios.failover_test,
        }

        scenario_func = scenarios.get(name)
        return scenario_func() if scenario_func else None

    @staticmethod
    def list_scenarios() -> List[str]:
        """List all available scenarios."""
        return [
            "research_conference",
            "coalition_operations",
            "system_monitoring",
            "mixed_workload",
            "stress_test",
            "burst_activity",
            "development_testing",
            "long_running",
            "database_intensive",
            "failover_test",
        ]

    @staticmethod
    def create_custom(
        name: str,
        duration_seconds: float,
        user_counts: Dict[str, int],
        **kwargs,
    ) -> SimulationConfig:
        """Create a custom simulation scenario.

        Args:
            name: Scenario name
            duration_seconds: Duration in seconds
            user_counts: Dictionary mapping persona names to counts
            **kwargs: Additional configuration parameters

        Returns:
            SimulationConfig instance
        """
        # Convert string keys to PersonaType enum
        user_distribution = {}
        for persona_name, count in user_counts.items():
            try:
                persona_type = PersonaType(persona_name.lower())
                user_distribution[persona_type] = count
            except ValueError:
                raise ValueError(f"Unknown persona type: {persona_name}")

        # Create config with defaults
        config = SimulationConfig(
            name=name,
            description=kwargs.get("description", f"Custom scenario: {name}"),
            duration_seconds=duration_seconds,
            user_distribution=user_distribution,
        )

        # Apply custom parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


class ScenarioScheduler:
    """Schedule and run multiple scenarios in sequence."""

    def __init__(self, results_base_path: Path = Path("simulation_results")):
        """Initialize scheduler."""
        self.results_base_path = results_base_path
        self.scheduled_scenarios: List[SimulationConfig] = []
        self.completed_scenarios: List[Dict] = []

    def add_scenario(self, scenario: SimulationConfig, delay_minutes: int = 0):
        """Add a scenario to the schedule.

        Args:
            scenario: Simulation scenario to run
            delay_minutes: Minutes to wait before starting this scenario
        """
        scenario._delay_minutes = delay_minutes
        self.scheduled_scenarios.append(scenario)

    def add_daily_scenarios(self):
        """Add a typical daily scenario sequence."""
        # Morning: System monitoring
        self.add_scenario(SimulationScenarios.system_monitoring(), delay_minutes=0)

        # Mid-morning: Mixed workload ramp-up
        self.add_scenario(SimulationScenarios.mixed_workload(), delay_minutes=5)

        # Noon: Burst activity (lunch break patterns)
        self.add_scenario(SimulationScenarios.burst_activity(), delay_minutes=5)

        # Afternoon: Research conference
        self.add_scenario(SimulationScenarios.research_conference(), delay_minutes=10)

        # Evening: Coalition operations
        self.add_scenario(SimulationScenarios.coalition_operations(), delay_minutes=5)

        # Night: Long-running stability
        self.add_scenario(SimulationScenarios.long_running(), delay_minutes=10)

    def add_stress_test_sequence(self):
        """Add a stress testing sequence."""
        # Warmup with mixed load
        self.add_scenario(SimulationScenarios.mixed_workload(), delay_minutes=0)

        # Database stress
        self.add_scenario(SimulationScenarios.database_intensive(), delay_minutes=5)

        # Connection stress
        self.add_scenario(SimulationScenarios.stress_test(), delay_minutes=5)

        # Failover testing
        self.add_scenario(SimulationScenarios.failover_test(), delay_minutes=10)

        # Recovery verification
        self.add_scenario(SimulationScenarios.mixed_workload(), delay_minutes=10)

    async def run_schedule(self):
        """Run all scheduled scenarios."""
        import asyncio

        from tests.simulation.concurrent_simulator import ConcurrentSimulator

        start_time = datetime.now()

        for i, scenario in enumerate(self.scheduled_scenarios):
            # Wait for delay
            if hasattr(scenario, "_delay_minutes") and scenario._delay_minutes > 0:
                wait_time = scenario._delay_minutes * 60
                print(f"Waiting {scenario._delay_minutes} minutes before next scenario...")
                await asyncio.sleep(wait_time)

            # Update results path for this scenario
            scenario.results_path = (
                self.results_base_path / scenario.name / start_time.strftime("%Y%m%d_%H%M%S")
            )

            # Run scenario
            print(f"\n{'=' * 60}")
            print(f"Running scenario {i + 1}/{len(self.scheduled_scenarios)}: {scenario.name}")
            print(f"Description: {scenario.description}")
            print(f"Duration: {scenario.duration_seconds}s")
            print(f"Total users: {scenario.total_users}")
            print(f"{'=' * 60}\n")

            simulator = ConcurrentSimulator(scenario)

            try:
                await simulator.run()

                # Record completion
                self.completed_scenarios.append(
                    {
                        "name": scenario.name,
                        "start_time": datetime.now().isoformat(),
                        "duration": scenario.duration_seconds,
                        "status": "completed",
                        "summary": simulator.get_summary(),
                    }
                )

            except Exception as e:
                print(f"Scenario {scenario.name} failed: {e}")
                self.completed_scenarios.append(
                    {
                        "name": scenario.name,
                        "start_time": datetime.now().isoformat(),
                        "duration": scenario.duration_seconds,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # Generate summary report
        self._generate_summary_report()

    def _generate_summary_report(self):
        """Generate a summary report of all scenarios."""
        report_path = self.results_base_path / "schedule_summary.json"

        import json

        with open(report_path, "w") as f:
            json.dump(
                {
                    "schedule_start": datetime.now().isoformat(),
                    "total_scenarios": len(self.scheduled_scenarios),
                    "completed": len(
                        [s for s in self.completed_scenarios if s["status"] == "completed"]
                    ),
                    "failed": len([s for s in self.completed_scenarios if s["status"] == "failed"]),
                    "scenarios": self.completed_scenarios,
                },
                f,
                indent=2,
            )

        print(f"\nSchedule summary saved to: {report_path}")

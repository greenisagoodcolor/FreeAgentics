"""Load testing scenarios for WebSocket connections."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .client_manager import WebSocketClientManager
from .connection_lifecycle import ConnectionLifecycleManager
from .message_generators import (
    CommandMessageGenerator,
    EventMessageGenerator,
    MessageGenerator,
    MixedMessageGenerator,
    MonitoringMessageGenerator,
    QueryMessageGenerator,
    RealisticScenarioGenerator,
)
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for a load testing scenario."""

    name: str
    description: str
    total_clients: int
    duration_seconds: float
    base_url: str = "ws://localhost:8000"

    # Client behavior
    connection_pattern: str = "persistent"  # persistent, intermittent, bursty, failover
    message_generator_type: str = "mixed"  # event, command, query, monitoring, mixed, realistic
    message_interval: float = 1.0  # seconds between messages

    # Connection parameters
    concurrent_connections: int = 50
    connection_ramp_up_seconds: float = 0.0
    reconnect_on_failure: bool = True
    max_reconnect_attempts: int = 3

    # Lifecycle parameters
    lifecycle_kwargs: Dict[str, Any] = None

    # Metrics
    enable_prometheus: bool = False
    metrics_export_path: Optional[Path] = None

    def __post_init__(self):
        if self.lifecycle_kwargs is None:
            self.lifecycle_kwargs = {}


class LoadScenario(ABC):
    """Abstract base class for load testing scenarios."""

    def __init__(self, config: ScenarioConfig):
        """Initialize load scenario."""
        self.config = config
        self.client_manager = WebSocketClientManager(
            base_url=config.base_url, client_prefix=f"{config.name}_client"
        )
        self.metrics = MetricsCollector(enable_prometheus=config.enable_prometheus)
        self.lifecycle_manager = ConnectionLifecycleManager(self.client_manager, self.metrics)

        # Message callbacks
        self.client_manager.global_on_message = self._on_message
        self.client_manager.global_on_error = self._on_error
        self.client_manager.global_on_close = self._on_close

        # Scenario state
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.is_running = False
        self.scenario_task: Optional[asyncio.Task] = None

    @abstractmethod
    async def run(self):
        """Run the load testing scenario."""
        pass

    async def setup(self):
        """Setup the scenario before running."""
        logger.info(f"Setting up scenario: {self.config.name}")
        logger.info(f"Description: {self.config.description}")
        logger.info(f"Total clients: {self.config.total_clients}")
        logger.info(f"Duration: {self.config.duration_seconds} seconds")

        # Start metrics collection
        await self.metrics.start_real_time_stats()

    async def teardown(self):
        """Cleanup after scenario completion."""
        logger.info(f"Tearing down scenario: {self.config.name}")

        # Stop metrics collection
        await self.metrics.stop_real_time_stats()

        # Disconnect all clients
        await self.client_manager.disconnect_all()

        # Stop lifecycle management
        await self.lifecycle_manager.stop_all_lifecycles()

        # Export metrics if configured
        if self.config.metrics_export_path:
            self.metrics.save_metrics(self.config.metrics_export_path)

            # Also save summary report
            report_path = self.config.metrics_export_path.with_suffix(".txt")
            with open(report_path, "w") as f:
                f.write(self.metrics.generate_summary_report())

    async def execute(self):
        """Execute the complete scenario with setup and teardown."""
        try:
            self.start_time = time.time()
            self.is_running = True

            await self.setup()
            await self.run()

        except Exception as e:
            logger.error(f"Scenario error: {e}", exc_info=True)
            raise
        finally:
            self.end_time = time.time()
            self.is_running = False
            await self.teardown()

    def create_message_generator(self) -> MessageGenerator:
        """Create message generator based on configuration."""
        generator_type = self.config.message_generator_type

        if generator_type == "event":
            return EventMessageGenerator()
        elif generator_type == "command":
            return CommandMessageGenerator()
        elif generator_type == "query":
            return QueryMessageGenerator()
        elif generator_type == "monitoring":
            return MonitoringMessageGenerator()
        elif generator_type == "mixed":
            return MixedMessageGenerator()
        elif generator_type == "realistic":
            return RealisticScenarioGenerator()
        else:
            raise ValueError(f"Unknown message generator type: {generator_type}")

    async def _on_message(self, client, message):
        """Handle incoming messages."""
        self.metrics.record_message_received(message.get("type", "unknown"), len(str(message)))

        # Track latency if message has ID
        if "id" in message and message["id"] in client.pending_messages:
            latency = time.time() - client.pending_messages[message["id"]]
            self.metrics.record_latency(latency)

    async def _on_error(self, client, error):
        """Handle client errors."""
        logger.error(f"Client {client.client_id} error: {error}")
        self.metrics.record_error("receive")

    async def _on_close(self, client):
        """Handle client disconnection."""
        logger.debug(f"Client {client.client_id} closed")
        duration = client.stats.connection_duration
        self.metrics.record_connection_closed(duration)

    async def monitor_progress(self, interval: float = 5.0):
        """Monitor and log scenario progress."""
        while self.is_running:
            stats = self.metrics.get_real_time_stats()
            state_dist = self.lifecycle_manager.get_state_distribution()

            logger.info(
                f"Progress - Active: {state_dist.get('connected', 0)}, "
                f"Messages/sec: {stats['messages_per_second']:.1f}, "
                f"Latency: {stats['current_latency_ms']:.1f}ms, "
                f"Errors: {stats['error_rate']:.1%}"
            )

            await asyncio.sleep(interval)


class SteadyLoadScenario(LoadScenario):
    """Steady load with consistent connection and message rates."""

    async def run(self):
        """Run steady load scenario."""
        # Create all clients
        clients = await self.client_manager.create_clients(
            self.config.total_clients,
            stagger_delay=(
                self.config.connection_ramp_up_seconds / self.config.total_clients
                if self.config.connection_ramp_up_seconds > 0
                else 0.0
            ),
        )

        # Connect clients with concurrency limit
        await self.client_manager.connect_clients(
            clients,
            concurrent_limit=self.config.concurrent_connections,
            retry_count=self.config.max_reconnect_attempts,
        )

        # Create message generator
        message_generator = self.create_message_generator()

        # Configure lifecycle parameters
        lifecycle_kwargs = {
            "activity_generator": message_generator,
            "activity_interval": self.config.message_interval,
            **self.config.lifecycle_kwargs,
        }

        # Start lifecycle management
        await self.lifecycle_manager.start_lifecycle_management(
            clients,
            pattern=self.config.connection_pattern,
            **lifecycle_kwargs,
        )

        # Start progress monitoring
        monitor_task = asyncio.create_task(self.monitor_progress())

        # Run for specified duration
        await asyncio.sleep(self.config.duration_seconds)

        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


class BurstLoadScenario(LoadScenario):
    """Burst load with periods of high activity."""

    def __init__(
        self,
        config: ScenarioConfig,
        burst_size: int = 100,
        burst_duration: float = 30.0,
        idle_duration: float = 60.0,
    ):
        """Initialize burst load scenario."""
        super().__init__(config)
        self.burst_size = burst_size
        self.burst_duration = burst_duration
        self.idle_duration = idle_duration

    async def run(self):
        """Run burst load scenario."""
        # Create initial pool of clients
        base_clients = await self.client_manager.create_clients(self.config.total_clients // 2)

        # Connect base clients
        await self.client_manager.connect_clients(base_clients)

        # Start base load
        message_generator = self.create_message_generator()
        await self.lifecycle_manager.start_lifecycle_management(
            base_clients,
            pattern="persistent",
            activity_generator=message_generator,
            activity_interval=self.config.message_interval * 2,  # Slower base rate
        )

        # Burst cycle management
        burst_clients = []
        burst_tasks = []
        total_duration = 0

        while total_duration < self.config.duration_seconds:
            # Burst phase
            logger.info(f"Starting burst phase with {self.burst_size} additional clients")

            # Create burst clients
            new_burst_clients = await self.client_manager.create_clients(self.burst_size)
            burst_clients.extend(new_burst_clients)

            # Connect burst clients
            await self.client_manager.connect_clients(new_burst_clients)

            # Start burst activity
            new_tasks = await self.lifecycle_manager.start_lifecycle_management(
                new_burst_clients,
                pattern="bursty",
                burst_size=(20, 50),
                burst_interval=0.1,
                message_generator=message_generator,
                cycles=int(self.burst_duration / 10),  # Approximate cycles
            )
            burst_tasks.extend(new_tasks)

            # Monitor during burst
            await asyncio.sleep(
                min(
                    self.burst_duration,
                    self.config.duration_seconds - total_duration,
                )
            )
            total_duration += self.burst_duration

            # Idle phase
            if total_duration < self.config.duration_seconds:
                logger.info("Entering idle phase")

                # Disconnect burst clients
                for client in new_burst_clients:
                    if client.is_connected:
                        await client.disconnect()

                # Wait for idle duration
                await asyncio.sleep(
                    min(
                        self.idle_duration,
                        self.config.duration_seconds - total_duration,
                    )
                )
                total_duration += self.idle_duration


class RampUpScenario(LoadScenario):
    """Gradually ramp up load to test system capacity."""

    def __init__(
        self,
        config: ScenarioConfig,
        initial_clients: int = 10,
        ramp_steps: int = 10,
        step_duration: float = 30.0,
    ):
        """Initialize ramp-up scenario."""
        super().__init__(config)
        self.initial_clients = initial_clients
        self.ramp_steps = ramp_steps
        self.step_duration = step_duration
        self.clients_per_step = (config.total_clients - initial_clients) // ramp_steps

    async def run(self):
        """Run ramp-up load scenario."""
        all_clients = []
        all_tasks = []
        message_generator = self.create_message_generator()

        # Calculate actual steps based on duration
        max_steps = int(self.config.duration_seconds / self.step_duration)
        actual_steps = min(self.ramp_steps, max_steps)

        for step in range(actual_steps + 1):
            # Calculate clients for this step
            if step == 0:
                step_clients = self.initial_clients
            else:
                step_clients = self.clients_per_step

            logger.info(f"Ramp step {step}: Adding {step_clients} clients")

            # Create and connect new clients
            new_clients = await self.client_manager.create_clients(step_clients)
            all_clients.extend(new_clients)

            await self.client_manager.connect_clients(new_clients)

            # Start lifecycle management
            new_tasks = await self.lifecycle_manager.start_lifecycle_management(
                new_clients,
                pattern=self.config.connection_pattern,
                activity_generator=message_generator,
                activity_interval=self.config.message_interval,
            )
            all_tasks.extend(new_tasks)

            # Log current state
            stats = self.client_manager.get_statistics()
            logger.info(
                f"Current load - Total: {len(all_clients)}, "
                f"Active: {stats['active_clients']}, "
                f"Messages/sec: {self.metrics.real_time_stats['messages_per_second']:.1f}"
            )

            # Wait for step duration (except last step)
            if step < actual_steps:
                await asyncio.sleep(self.step_duration)

        # Continue running at peak load for remaining duration
        remaining_time = self.config.duration_seconds - (actual_steps * self.step_duration)
        if remaining_time > 0:
            logger.info(
                f"Running at peak load ({len(all_clients)} clients) for {remaining_time:.1f}s"
            )
            await asyncio.sleep(remaining_time)


class StressTestScenario(LoadScenario):
    """Stress test to find system limits."""

    def __init__(
        self,
        config: ScenarioConfig,
        target_latency_ms: float = 100.0,
        error_rate_threshold: float = 0.05,
        clients_increment: int = 50,
    ):
        """Initialize stress test scenario."""
        super().__init__(config)
        self.target_latency_ms = target_latency_ms
        self.error_rate_threshold = error_rate_threshold
        self.clients_increment = clients_increment
        self.peak_clients = 0
        self.breaking_point_found = False

    async def run(self):
        """Run stress test to find system limits."""
        all_clients = []
        all_tasks = []
        message_generator = self.create_message_generator()
        current_clients = 0

        while (
            current_clients < self.config.total_clients
            and not self.breaking_point_found
            and self.is_running
        ):
            # Add more clients
            new_client_count = min(
                self.clients_increment,
                self.config.total_clients - current_clients,
            )

            logger.info(
                f"Adding {new_client_count} clients (total: {current_clients + new_client_count})"
            )

            # Create and connect new clients
            new_clients = await self.client_manager.create_clients(new_client_count)
            all_clients.extend(new_clients)

            await self.client_manager.connect_clients(new_clients)

            # Start lifecycle management
            new_tasks = await self.lifecycle_manager.start_lifecycle_management(
                new_clients,
                pattern="persistent",
                activity_generator=message_generator,
                activity_interval=self.config.message_interval,
            )
            all_tasks.extend(new_tasks)

            current_clients += new_client_count

            # Let system stabilize
            await asyncio.sleep(30.0)

            # Check system metrics
            self.metrics.get_real_time_stats()
            metrics_dict = self.metrics.current_metrics.to_dict()

            avg_latency = metrics_dict["latency_metrics"]["avg_ms"]
            p95_latency = metrics_dict["latency_metrics"]["p95_ms"]
            error_rate = metrics_dict["error_metrics"]["error_rate"]

            logger.info(
                f"Metrics - Clients: {current_clients}, "
                f"Avg Latency: {avg_latency:.1f}ms, "
                f"P95 Latency: {p95_latency:.1f}ms, "
                f"Error Rate: {error_rate:.1%}"
            )

            # Check if we've hit limits
            if (
                avg_latency > self.target_latency_ms
                or error_rate > self.error_rate_threshold
                or p95_latency > self.target_latency_ms * 2
            ):
                self.breaking_point_found = True
                self.peak_clients = current_clients - self.clients_increment

                logger.warning(
                    f"System limit reached at {self.peak_clients} clients! "
                    f"Latency: {avg_latency:.1f}ms (target: {self.target_latency_ms}ms), "
                    f"Error rate: {error_rate:.1%} (threshold: {self.error_rate_threshold:.1%})"
                )

                # Optionally reduce load to confirm breaking point
                logger.info("Reducing load to confirm breaking point...")

                # Disconnect recent clients
                for client in new_clients:
                    if client.is_connected:
                        await client.disconnect()

                await asyncio.sleep(30.0)

                # Check metrics again
                metrics_dict = self.metrics.current_metrics.to_dict()
                recovered_latency = metrics_dict["latency_metrics"]["avg_ms"]
                recovered_error_rate = metrics_dict["error_metrics"]["error_rate"]

                logger.info(
                    f"After reduction - Latency: {recovered_latency:.1f}ms, "
                    f"Error rate: {recovered_error_rate:.1%}"
                )

        # Continue running at current load for remaining duration
        if not self.breaking_point_found:
            self.peak_clients = current_clients
            logger.info(
                f"Completed ramp without finding breaking point. Peak: {self.peak_clients} clients"
            )

        # Generate final report
        logger.info("\nStress Test Summary:")
        logger.info(f"Peak sustainable load: {self.peak_clients} concurrent clients")
        logger.info(f"Target latency: {self.target_latency_ms}ms")
        logger.info(f"Error threshold: {self.error_rate_threshold:.1%}")


class RealisticUsageScenario(LoadScenario):
    """Simulate realistic usage patterns."""

    def __init__(
        self,
        config: ScenarioConfig,
        user_profiles: Optional[Dict[str, float]] = None,
    ):
        """Initialize realistic usage scenario."""
        super().__init__(config)

        # Default user behavior profiles
        self.user_profiles = user_profiles or {
            "active": 0.2,  # 20% very active users
            "regular": 0.5,  # 50% regular users
            "passive": 0.2,  # 20% passive users
            "idle": 0.1,  # 10% mostly idle
        }

    async def run(self):
        """Run realistic usage patterns."""
        # Distribute clients across profiles
        profile_counts = {
            profile: int(self.config.total_clients * ratio)
            for profile, ratio in self.user_profiles.items()
        }

        # Adjust for rounding
        total_assigned = sum(profile_counts.values())
        if total_assigned < self.config.total_clients:
            profile_counts["regular"] += self.config.total_clients - total_assigned

        all_tasks = []

        for profile, count in profile_counts.items():
            if count == 0:
                continue

            logger.info(f"Creating {count} {profile} users")

            # Create clients for this profile
            clients = await self.client_manager.create_clients(
                count,
                stagger_delay=0.1,  # Stagger creation
            )

            # Connect clients
            await self.client_manager.connect_clients(clients)

            # Configure behavior based on profile
            if profile == "active":
                # Very active users - frequent messages, persistent connection
                generator = MixedMessageGenerator(
                    weights={
                        "command": 0.4,
                        "query": 0.3,
                        "event": 0.2,
                        "monitoring": 0.05,
                        "ping": 0.05,
                    }
                )
                pattern = "persistent"
                interval = 0.5  # Fast message rate

            elif profile == "regular":
                # Regular users - moderate activity
                generator = MixedMessageGenerator()  # Default weights
                pattern = "persistent"
                interval = 2.0

            elif profile == "passive":
                # Passive users - mostly listening
                generator = EventMessageGenerator()  # Mostly subscriptions
                pattern = "intermittent"
                interval = 10.0

            else:  # idle
                # Idle users - minimal activity
                generator = EventMessageGenerator()
                pattern = "intermittent"
                interval = 30.0

            # Start lifecycle management for this profile
            tasks = await self.lifecycle_manager.start_lifecycle_management(
                clients,
                pattern=pattern,
                activity_generator=generator,
                activity_interval=interval,
                connect_duration=(300, 600) if pattern == "intermittent" else None,
                disconnect_duration=(60, 300) if pattern == "intermittent" else None,
            )
            all_tasks.extend(tasks)

        # Simulate day/night patterns if duration is long enough
        if self.config.duration_seconds > 3600:  # More than 1 hour
            asyncio.create_task(self._simulate_daily_patterns())

        # Monitor progress
        monitor_task = asyncio.create_task(self.monitor_progress())

        # Run for specified duration
        await asyncio.sleep(self.config.duration_seconds)

        # Cleanup
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    async def _simulate_daily_patterns(self):
        """Simulate daily usage patterns with peaks and valleys."""
        cycle_duration = 3600  # 1 hour represents a "day"

        while self.is_running:
            # Morning ramp-up
            logger.info("Simulating morning activity increase")
            # Could adjust message rates or reconnect idle clients

            await asyncio.sleep(cycle_duration / 4)

            # Midday peak
            logger.info("Simulating midday peak activity")
            # Could create temporary burst clients

            await asyncio.sleep(cycle_duration / 4)

            # Afternoon decline
            logger.info("Simulating afternoon activity decline")
            # Could disconnect some clients

            await asyncio.sleep(cycle_duration / 4)

            # Night time low activity
            logger.info("Simulating night-time low activity")
            # Could reduce message rates

            await asyncio.sleep(cycle_duration / 4)

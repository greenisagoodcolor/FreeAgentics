"""Example usage of the WebSocket load testing framework."""

import asyncio
import logging
from pathlib import Path

from .client_manager import WebSocketClientManager
from .connection_lifecycle import ConnectionLifecycleManager, ConnectionPool
from .load_scenarios import ScenarioConfig, SteadyLoadScenario
from .message_generators import (
    CommandMessageGenerator,
    EventMessageGenerator,
    MixedMessageGenerator,
    RealisticScenarioGenerator,
)
from .metrics_collector import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_client_test():
    """Example: Basic client connection and messaging."""
    logger.info("=== Basic Client Test ===")

    # Create client manager
    manager = WebSocketClientManager(base_url="ws://localhost:8000")

    # Create metrics collector
    metrics = MetricsCollector()

    # Create a few clients
    clients = await manager.create_clients(5)

    # Connect clients
    results = await manager.connect_clients(clients)
    logger.info(f"Connection results: {results}")

    # Send some messages
    message_gen = EventMessageGenerator()

    for _ in range(10):
        message = message_gen.generate()
        await manager.broadcast_message(message)
        await asyncio.sleep(1)

    # Get statistics
    stats = manager.get_statistics()
    logger.info(f"Statistics: {stats}")

    # Disconnect
    await manager.disconnect_all()

    # Get final metrics
    final_metrics = metrics.finalize()
    logger.info(f"Final metrics: {final_metrics.to_dict()}")


async def example_message_generators():
    """Example: Different message generators."""
    logger.info("=== Message Generator Examples ===")

    # Event messages
    event_gen = EventMessageGenerator()
    logger.info("Event messages:")
    for _ in range(3):
        logger.info(f"  {event_gen.generate()}")

    # Command messages
    command_gen = CommandMessageGenerator()
    logger.info("\nCommand messages:")
    for _ in range(3):
        logger.info(f"  {command_gen.generate()}")

    # Mixed messages
    mixed_gen = MixedMessageGenerator(
        weights={
            "event": 0.4,
            "command": 0.3,
            "query": 0.2,
            "monitoring": 0.1,
        }
    )
    logger.info("\nMixed messages:")
    for _ in range(5):
        logger.info(f"  {mixed_gen.generate()}")

    # Realistic scenario
    realistic_gen = RealisticScenarioGenerator(scenario="startup")
    logger.info("\nRealistic startup sequence:")
    for _ in range(5):
        logger.info(f"  {realistic_gen.generate()}")


async def example_connection_patterns():
    """Example: Different connection lifecycle patterns."""
    logger.info("=== Connection Pattern Examples ===")

    manager = WebSocketClientManager()
    metrics = MetricsCollector()
    lifecycle = ConnectionLifecycleManager(manager, metrics)

    # Create clients
    clients = await manager.create_clients(3)

    # Example 1: Persistent connection
    logger.info("\n1. Persistent connection pattern:")
    task1 = asyncio.create_task(
        lifecycle.manage_connection_lifecycle(
            clients[0],
            pattern="persistent",
            activity_generator=EventMessageGenerator(),
            activity_interval=2.0,
        )
    )

    # Example 2: Intermittent connection
    logger.info("\n2. Intermittent connection pattern:")
    task2 = asyncio.create_task(
        lifecycle.manage_connection_lifecycle(
            clients[1],
            pattern="intermittent",
            connect_duration=(5.0, 10.0),
            disconnect_duration=(2.0, 5.0),
            cycles=3,
        )
    )

    # Example 3: Bursty connection
    logger.info("\n3. Bursty connection pattern:")
    task3 = asyncio.create_task(
        lifecycle.manage_connection_lifecycle(
            clients[2],
            pattern="bursty",
            burst_size=(5, 10),
            burst_interval=0.5,
            idle_duration=(3.0, 5.0),
            message_generator=CommandMessageGenerator(),
            cycles=3,
        )
    )

    # Let them run for a bit
    await asyncio.sleep(30)

    # Check states
    states = lifecycle.get_connection_states()
    logger.info(f"\nConnection states: {states}")

    # Stop all
    await lifecycle.stop_all_lifecycles()
    await manager.disconnect_all()


async def example_connection_pool():
    """Example: Using connection pool."""
    logger.info("=== Connection Pool Example ===")

    manager = WebSocketClientManager()

    # Create connection pool
    pool = ConnectionPool(
        manager, min_size=5, max_size=20, acquire_timeout=5.0
    )

    # Initialize pool
    await pool.initialize()
    logger.info(f"Pool initialized with {pool.available.qsize()} connections")

    # Use connections from pool
    tasks = []

    async def use_connection(task_id: int):
        # Acquire connection
        client = await pool.acquire()
        if client:
            logger.info(
                f"Task {task_id} acquired connection {client.client_id}"
            )

            # Use connection
            message_gen = EventMessageGenerator()
            for _ in range(5):
                await client.send_message(message_gen.generate())
                await asyncio.sleep(1)

            # Release back to pool
            await pool.release(client)
            logger.info(
                f"Task {task_id} released connection {client.client_id}"
            )
        else:
            logger.error(f"Task {task_id} failed to acquire connection")

    # Create multiple tasks using the pool
    for i in range(10):
        task = asyncio.create_task(use_connection(i))
        tasks.append(task)
        await asyncio.sleep(0.5)  # Stagger task creation

    # Wait for all tasks
    await asyncio.gather(*tasks)

    # Shutdown pool
    await pool.shutdown()


async def example_metrics_collection():
    """Example: Detailed metrics collection and analysis."""
    logger.info("=== Metrics Collection Example ===")

    # Create metrics collector with Prometheus support
    metrics = MetricsCollector(enable_prometheus=False)

    # Start real-time stats
    await metrics.start_real_time_stats(update_interval=1.0)

    # Simulate some activity
    for i in range(100):
        # Connection attempts
        success = i % 10 != 0  # 90% success rate
        metrics.record_connection_attempt(success)

        if success:
            # Messages
            metrics.record_message_sent("command", 256)
            metrics.record_message_received("response", 512)

            # Latency
            latency = 0.010 + (i % 5) * 0.005  # 10-30ms
            metrics.record_latency(latency)

            # Occasional errors
            if i % 20 == 0:
                metrics.record_error("timeout")

        await asyncio.sleep(0.1)

    # Get real-time stats
    real_time = metrics.get_real_time_stats()
    logger.info(f"\nReal-time stats: {real_time}")

    # Get time series data
    latency_series = metrics.get_time_series_data(
        "latency_ms", duration_seconds=10
    )
    logger.info(
        f"\nLatency time series (last 10s): {len(latency_series)} points"
    )

    # Stop real-time stats
    await metrics.stop_real_time_stats()

    # Get final metrics
    final_metrics = metrics.finalize()
    logger.info("\nFinal metrics summary:")
    logger.info(metrics.generate_summary_report())

    # Save metrics
    metrics.save_metrics(Path("example_metrics.json"), format="json")
    metrics.save_metrics(Path("example_metrics.csv"), format="csv")


async def example_simple_load_scenario():
    """Example: Running a simple load scenario."""
    logger.info("=== Simple Load Scenario Example ===")

    # Configure scenario
    config = ScenarioConfig(
        name="example_steady",
        description="Example steady load test",
        total_clients=20,
        duration_seconds=60,
        base_url="ws://localhost:8000",
        connection_pattern="persistent",
        message_generator_type="mixed",
        message_interval=2.0,
        concurrent_connections=10,
        metrics_export_path=Path("example_scenario_metrics.json"),
    )

    # Create and run scenario
    scenario = SteadyLoadScenario(config)

    # Add custom message handler
    async def handle_message(client, message):
        if message.get("type") == "special_event":
            logger.info(f"Received special event: {message}")

    scenario.client_manager.global_on_message = handle_message

    # Execute scenario
    await scenario.execute()

    logger.info("Scenario completed!")


async def example_custom_scenario():
    """Example: Creating a custom load scenario."""
    logger.info("=== Custom Scenario Example ===")

    # Custom scenario that simulates a game server load
    class GameServerLoadScenario:
        def __init__(self):
            self.manager = WebSocketClientManager()
            self.metrics = MetricsCollector()
            self.lifecycle = ConnectionLifecycleManager(
                self.manager, self.metrics
            )

        async def run(self):
            # Phase 1: Login rush (many connections at once)
            logger.info("Phase 1: Login rush")
            players = await self.manager.create_clients(
                50, client_prefix="player"
            )
            await self.manager.connect_clients(players, concurrent_limit=50)

            # Everyone subscribes to game events
            game_events = ["game:started", "player:joined", "player:action"]
            subscribe_msg = {"type": "subscribe", "event_types": game_events}
            await self.manager.broadcast_message(subscribe_msg)

            # Phase 2: Lobby (light activity)
            logger.info("Phase 2: Lobby phase")
            await asyncio.sleep(10)

            # Phase 3: Game start (heavy activity)
            logger.info("Phase 3: Game started - heavy activity")

            # Simulate game actions
            for round_num in range(5):
                logger.info(f"Game round {round_num + 1}")

                # Each player performs actions
                for player in players[:30]:  # Active players
                    action_msg = {
                        "type": "agent_command",
                        "data": {
                            "agent_id": player.client_id,
                            "command": "act",
                            "params": {
                                "action": "move",
                                "target": f"pos_{round_num}",
                            },
                        },
                    }
                    await player.send_message(action_msg)

                await asyncio.sleep(2)

            # Phase 4: Game end (mass disconnect)
            logger.info("Phase 4: Game ended - disconnecting")
            await self.manager.disconnect_all()

            # Report
            logger.info("\nGame server load test complete!")
            logger.info(self.metrics.generate_summary_report())

    # Run custom scenario
    scenario = GameServerLoadScenario()
    await scenario.run()


async def main():
    """Run all examples."""
    examples = [
        ("Basic Client Test", example_basic_client_test),
        ("Message Generators", example_message_generators),
        ("Connection Patterns", example_connection_patterns),
        ("Connection Pool", example_connection_pool),
        ("Metrics Collection", example_metrics_collection),
        ("Simple Load Scenario", example_simple_load_scenario),
        ("Custom Scenario", example_custom_scenario),
    ]

    for name, example_func in examples:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {name}")
        logger.info(f"{'=' * 60}\n")

        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example failed: {e}", exc_info=True)

        # Pause between examples
        await asyncio.sleep(2)

    logger.info("\n✅ All examples completed!")


if __name__ == "__main__":
    # Note: This assumes the WebSocket server is running at ws://localhost:8000
    # Some examples may fail if the server is not available
    asyncio.run(main())

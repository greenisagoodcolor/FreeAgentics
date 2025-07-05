"""Main script to run WebSocket load tests."""

import argparse
import asyncio
import logging
from pathlib import Path

from .load_scenarios import (
    BurstLoadScenario,
    RampUpScenario,
    RealisticUsageScenario,
    ScenarioConfig,
    SteadyLoadScenario,
    StressTestScenario,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_steady_load_test(args):
    """Run a steady load test."""
    config = ScenarioConfig(
        name="steady_load",
        description="Steady load with constant connection and message rate",
        total_clients=args.clients,
        duration_seconds=args.duration,
        base_url=args.url,
        connection_pattern="persistent",
        message_generator_type=args.message_type,
        message_interval=args.message_interval,
        concurrent_connections=args.concurrent,
        enable_prometheus=args.prometheus,
        metrics_export_path=Path(f"metrics/steady_load_{args.clients}c_{args.duration}s.json"),
    )

    scenario = SteadyLoadScenario(config)
    await scenario.execute()


async def run_burst_load_test(args):
    """Run a burst load test."""
    config = ScenarioConfig(
        name="burst_load",
        description="Burst load with periods of high activity",
        total_clients=args.clients,
        duration_seconds=args.duration,
        base_url=args.url,
        connection_pattern="bursty",
        message_generator_type=args.message_type,
        message_interval=args.message_interval,
        concurrent_connections=args.concurrent,
        enable_prometheus=args.prometheus,
        metrics_export_path=Path(f"metrics/burst_load_{args.clients}c_{args.duration}s.json"),
    )

    scenario = BurstLoadScenario(
        config,
        burst_size=args.burst_size,
        burst_duration=args.burst_duration,
        idle_duration=args.idle_duration,
    )
    await scenario.execute()


async def run_ramp_up_test(args):
    """Run a ramp-up load test."""
    config = ScenarioConfig(
        name="ramp_up",
        description="Gradually increase load to test system capacity",
        total_clients=args.clients,
        duration_seconds=args.duration,
        base_url=args.url,
        connection_pattern="persistent",
        message_generator_type=args.message_type,
        message_interval=args.message_interval,
        concurrent_connections=args.concurrent,
        enable_prometheus=args.prometheus,
        metrics_export_path=Path(f"metrics/ramp_up_{args.clients}c_{args.duration}s.json"),
    )

    scenario = RampUpScenario(
        config,
        initial_clients=args.initial_clients,
        ramp_steps=args.ramp_steps,
        step_duration=args.step_duration,
    )
    await scenario.execute()


async def run_stress_test(args):
    """Run a stress test to find system limits."""
    config = ScenarioConfig(
        name="stress_test",
        description="Stress test to find system breaking point",
        total_clients=args.max_clients,
        duration_seconds=args.duration,
        base_url=args.url,
        connection_pattern="persistent",
        message_generator_type=args.message_type,
        message_interval=args.message_interval,
        concurrent_connections=args.concurrent,
        enable_prometheus=args.prometheus,
        metrics_export_path=Path(f"metrics/stress_test_{args.max_clients}c.json"),
    )

    scenario = StressTestScenario(
        config,
        target_latency_ms=args.target_latency,
        error_rate_threshold=args.error_threshold,
        clients_increment=args.increment,
    )
    await scenario.execute()


async def run_realistic_test(args):
    """Run a realistic usage pattern test."""
    config = ScenarioConfig(
        name="realistic_usage",
        description="Simulate realistic user behavior patterns",
        total_clients=args.clients,
        duration_seconds=args.duration,
        base_url=args.url,
        message_generator_type="realistic",
        message_interval=1.0,  # Will be overridden by profile
        concurrent_connections=args.concurrent,
        enable_prometheus=args.prometheus,
        metrics_export_path=Path(f"metrics/realistic_{args.clients}c_{args.duration}s.json"),
    )

    # Parse user profiles if provided
    user_profiles = None
    if args.user_profiles:
        profiles = {}
        for profile_str in args.user_profiles:
            name, ratio = profile_str.split(":")
            profiles[name] = float(ratio)
        user_profiles = profiles

    scenario = RealisticUsageScenario(config, user_profiles=user_profiles)
    await scenario.execute()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WebSocket Load Testing Tool")

    # Common arguments
    parser.add_argument(
        "--url",
        default="ws://localhost:8000",
        help="WebSocket server URL (default: ws://localhost:8000)",
    )
    parser.add_argument(
        "--clients", type=int, default=100, help="Total number of clients (default: 100)"
    )
    parser.add_argument(
        "--duration", type=int, default=300, help="Test duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--concurrent", type=int, default=50, help="Maximum concurrent connections (default: 50)"
    )
    parser.add_argument(
        "--message-type",
        choices=["event", "command", "query", "monitoring", "mixed", "realistic"],
        default="mixed",
        help="Type of messages to generate (default: mixed)",
    )
    parser.add_argument(
        "--message-interval",
        type=float,
        default=1.0,
        help="Seconds between messages (default: 1.0)",
    )
    parser.add_argument("--prometheus", action="store_true", help="Enable Prometheus metrics")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Subcommands for different scenarios
    subparsers = parser.add_subparsers(dest="scenario", help="Load test scenario")

    # Steady load
    steady_parser = subparsers.add_parser("steady", help="Run steady load test")

    # Burst load
    burst_parser = subparsers.add_parser("burst", help="Run burst load test")
    burst_parser.add_argument(
        "--burst-size", type=int, default=100, help="Number of clients in each burst (default: 100)"
    )
    burst_parser.add_argument(
        "--burst-duration",
        type=float,
        default=30.0,
        help="Duration of each burst in seconds (default: 30)",
    )
    burst_parser.add_argument(
        "--idle-duration",
        type=float,
        default=60.0,
        help="Duration of idle period between bursts (default: 60)",
    )

    # Ramp up
    ramp_parser = subparsers.add_parser("ramp", help="Run ramp-up load test")
    ramp_parser.add_argument(
        "--initial-clients", type=int, default=10, help="Initial number of clients (default: 10)"
    )
    ramp_parser.add_argument(
        "--ramp-steps", type=int, default=10, help="Number of ramp-up steps (default: 10)"
    )
    ramp_parser.add_argument(
        "--step-duration",
        type=float,
        default=30.0,
        help="Duration of each step in seconds (default: 30)",
    )

    # Stress test
    stress_parser = subparsers.add_parser("stress", help="Run stress test")
    stress_parser.add_argument(
        "--max-clients", type=int, default=1000, help="Maximum clients to test (default: 1000)"
    )
    stress_parser.add_argument(
        "--target-latency",
        type=float,
        default=100.0,
        help="Target latency threshold in ms (default: 100)",
    )
    stress_parser.add_argument(
        "--error-threshold", type=float, default=0.05, help="Error rate threshold (default: 0.05)"
    )
    stress_parser.add_argument(
        "--increment", type=int, default=50, help="Clients to add per step (default: 50)"
    )

    # Realistic usage
    realistic_parser = subparsers.add_parser("realistic", help="Run realistic usage test")
    realistic_parser.add_argument(
        "--user-profiles",
        nargs="+",
        help="User profiles as name:ratio pairs (e.g., active:0.2 regular:0.5)",
    )

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure metrics directory exists
    Path("metrics").mkdir(exist_ok=True)

    # Run the appropriate scenario
    if args.scenario == "steady":
        asyncio.run(run_steady_load_test(args))
    elif args.scenario == "burst":
        asyncio.run(run_burst_load_test(args))
    elif args.scenario == "ramp":
        asyncio.run(run_ramp_up_test(args))
    elif args.scenario == "stress":
        asyncio.run(run_stress_test(args))
    elif args.scenario == "realistic":
        asyncio.run(run_realistic_test(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

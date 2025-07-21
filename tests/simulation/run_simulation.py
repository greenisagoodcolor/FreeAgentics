#!/usr/bin/env python3
"""Run concurrent user simulations.

This script provides a command-line interface for running various
simulation scenarios to test the FreeAgentics system under realistic
concurrent load.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict

from tests.simulation.concurrent_simulator import (
    ConcurrentSimulator,
)
from tests.simulation.scenarios import ScenarioScheduler, SimulationScenarios
from tests.simulation.user_personas import PersonaType


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("simulation.log"),
        ],
    )


def parse_user_distribution(distribution_str: str) -> Dict[PersonaType, int]:
    """Parse user distribution from string format.

    Format: "researcher:20,coordinator:10,observer:30"
    """
    distribution = {}

    for part in distribution_str.split(","):
        persona_name, count = part.strip().split(":")
        try:
            persona_type = PersonaType(persona_name.lower())
            distribution[persona_type] = int(count)
        except (ValueError, KeyError):
            raise ValueError(f"Invalid persona type or count: {part}")

    return distribution


async def run_single_scenario(args):
    """Run a single simulation scenario."""
    # Get scenario
    if args.scenario == "custom":
        if not args.custom_users:
            raise ValueError("Custom scenario requires --custom-users")

        distribution = parse_user_distribution(args.custom_users)
        config = SimulationScenarios.create_custom(
            name=args.custom_name or "custom_scenario",
            duration_seconds=args.duration,
            user_counts={p.value: c for p, c in distribution.items()},
            description=args.custom_description or "Custom simulation scenario",
            user_spawn_rate=args.spawn_rate,
            warmup_period=args.warmup,
            cooldown_period=args.cooldown,
            ws_base_url=args.ws_url,
            db_url=args.db_url,
            enable_monitoring=not args.no_monitoring,
            export_results=not args.no_export,
        )
    else:
        config = SimulationScenarios.get_scenario(args.scenario)
        if not config:
            raise ValueError(f"Unknown scenario: {args.scenario}")

        # Override config with command-line arguments
        if args.duration:
            config.duration_seconds = args.duration
        if args.spawn_rate:
            config.user_spawn_rate = args.spawn_rate
        if args.warmup:
            config.warmup_period = args.warmup
        if args.cooldown:
            config.cooldown_period = args.cooldown
        if args.ws_url:
            config.ws_base_url = args.ws_url
        if args.db_url:
            config.db_url = args.db_url
        if args.no_monitoring:
            config.enable_monitoring = False
        if args.no_export:
            config.export_results = False

    # Set results path
    if args.output:
        config.results_path = Path(args.output)

    # Create and run simulator
    simulator = ConcurrentSimulator(config)

    print(f"\nStarting simulation: {config.name}")
    print(f"Description: {config.description}")
    print(f"Duration: {config.duration_seconds}s")
    print(f"Total users: {config.total_users}")
    print("User distribution:")
    for persona, count in config.user_distribution.items():
        print(f"  - {persona.value}: {count}")
    print()

    try:
        await simulator.run()

        # Print summary
        summary = simulator.get_summary()
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Duration: {summary['metrics']['duration_seconds']:.1f}s")
        print(f"Users created: {summary['metrics']['users']['created']}")
        print(f"Messages sent: {summary['metrics']['messages']['sent']}")
        print(
            f"Message success rate: {summary['metrics']['messages']['success_rate']:.1%}"
        )
        print(
            f"Avg DB latency: {summary['metrics']['database']['avg_latency_ms']:.1f}ms"
        )
        print(
            f"Avg WS latency: {summary['metrics']['websocket']['avg_latency_ms']:.1f}ms"
        )

        if config.export_results:
            print(f"\nResults exported to: {config.results_path}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nSimulation failed: {e}")
        raise


async def run_schedule(args):
    """Run a scheduled sequence of scenarios."""
    scheduler = ScenarioScheduler(
        results_base_path=Path(args.output)
        if args.output
        else Path("simulation_results")
    )

    if args.schedule == "daily":
        scheduler.add_daily_scenarios()
    elif args.schedule == "stress":
        scheduler.add_stress_test_sequence()
    else:
        # Custom schedule from file
        import json

        with open(args.schedule, "r") as f:
            schedule_data = json.load(f)

        for scenario_def in schedule_data["scenarios"]:
            scenario_name = scenario_def["name"]
            delay = scenario_def.get("delay_minutes", 0)

            if scenario_name == "custom":
                config = SimulationScenarios.create_custom(**scenario_def["config"])
            else:
                config = SimulationScenarios.get_scenario(scenario_name)
                if not config:
                    print(f"Warning: Unknown scenario {scenario_name}, skipping")
                    continue

            scheduler.add_scenario(config, delay_minutes=delay)

    print(
        f"\nRunning scenario schedule with {len(scheduler.scheduled_scenarios)} scenarios"
    )
    await scheduler.run_schedule()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run concurrent user simulations for FreeAgentics"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single scenario command
    single_parser = subparsers.add_parser("run", help="Run a single scenario")
    single_parser.add_argument(
        "scenario",
        choices=SimulationScenarios.list_scenarios() + ["custom"],
        help="Scenario to run",
    )
    single_parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="Override scenario duration (seconds)",
    )
    single_parser.add_argument(
        "--spawn-rate",
        "-r",
        type=float,
        help="User spawn rate (users per second)",
    )
    single_parser.add_argument(
        "--warmup", "-w", type=float, help="Warmup period (seconds)"
    )
    single_parser.add_argument(
        "--cooldown", "-c", type=float, help="Cooldown period (seconds)"
    )
    single_parser.add_argument(
        "--ws-url", help="WebSocket server URL (default: ws://localhost:8000)"
    )
    single_parser.add_argument(
        "--db-url",
        help="Database URL (default: postgresql://localhost/freeagentics_test)",
    )
    single_parser.add_argument("--output", "-o", help="Output directory for results")
    single_parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable performance monitoring",
    )
    single_parser.add_argument(
        "--no-export", action="store_true", help="Disable result export"
    )

    # Custom scenario options
    single_parser.add_argument("--custom-name", help="Name for custom scenario")
    single_parser.add_argument(
        "--custom-description", help="Description for custom scenario"
    )
    single_parser.add_argument(
        "--custom-users",
        help="User distribution (e.g., 'researcher:20,coordinator:10,observer:30')",
    )

    # Schedule command
    schedule_parser = subparsers.add_parser("schedule", help="Run a scenario schedule")
    schedule_parser.add_argument(
        "schedule",
        help="Schedule to run ('daily', 'stress', or path to JSON file)",
    )
    schedule_parser.add_argument(
        "--output", "-o", help="Base output directory for results"
    )

    # List command
    subparsers.add_parser("list", help="List available scenarios")

    # Common options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if args.command == "run":
        asyncio.run(run_single_scenario(args))
    elif args.command == "schedule":
        asyncio.run(run_schedule(args))
    elif args.command == "list":
        print("\nAvailable scenarios:")
        for scenario in SimulationScenarios.list_scenarios():
            config = SimulationScenarios.get_scenario(scenario)
            print(f"\n{scenario}:")
            print(f"  Description: {config.description}")
            print(f"  Duration: {config.duration_seconds}s")
            print(f"  Total users: {config.total_users}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main Runner for Multi-Agent Coordination Load Tests

This script orchestrates comprehensive load testing that validates
the architectural limitations documented in the system.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from agent_simulation_framework import (
    MixedWorkloadScenario,
    ScalingTestScenario,
    SimulationEnvironment,
)

# Import our test modules
from test_coordination_load import (
    CoordinationLoadTester,
)


class LoadTestReport:
    """Generates comprehensive load test reports."""

    def __init__(self, results: dict):
        self.results = results
        self.timestamp = datetime.now()

    def generate_summary(self) -> str:
        """Generate text summary of results."""
        summary = []
        summary.append("=" * 80)
        summary.append("MULTI-AGENT COORDINATION LOAD TEST REPORT")
        summary.append(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("=" * 80)

        # Key findings
        summary.append("\n🔍 KEY FINDINGS:")

        if "coordination_metrics" in self.results:
            metrics = self.results["coordination_metrics"]
            max_agents = max(m.agent_count for m in metrics)
            max_metric = next(m for m in metrics if m.agent_count == max_agents)

            summary.append(f"- Maximum agents tested: {max_agents}")
            summary.append(f"- Efficiency at scale: {max_metric.actual_efficiency:.1%}")
            summary.append(f"- Coordination overhead: {max_metric.coordination_overhead:.1%}")
            summary.append(f"- Message latency: {max_metric.coordination_latency_ms:.1f}ms")

            # Validate against documentation
            efficiency_loss = max_metric.efficiency_loss()
            if 70 <= efficiency_loss <= 75:
                summary.append(
                    f"✅ Efficiency loss ({efficiency_loss:.1f}%) matches documented ~72%"
                )
            else:
                summary.append(
                    f"❌ Efficiency loss ({efficiency_loss:.1f}%) differs from documented 72%"
                )

        # Performance characteristics
        summary.append("\n📊 PERFORMANCE CHARACTERISTICS:")

        if "scaling_results" in self.results:
            phases = self.results["scaling_results"]["phases"]

            summary.append("\nAgent Count | Steps/sec | Efficiency | Memory/agent")
            summary.append("------------|-----------|------------|-------------")

            baseline_sps = phases[0]["steps_per_second"] if phases else 1

            for phase in phases:
                agents = phase["agent_count"]
                sps = phase["steps_per_second"]
                efficiency = (sps / baseline_sps) / agents if agents > 0 else 0
                memory_mb = phase.get("memory_per_agent", 0)

                summary.append(
                    f"{agents:11} | {sps:9.1f} | {efficiency:10.1%} | {memory_mb:10.1f}MB"
                )

        # Coordination scenarios
        summary.append("\n🤝 COORDINATION PERFORMANCE:")

        if "handoff_results" in self.results:
            handoff = self.results["handoff_results"]
            summary.append(f"- Task handoffs/sec: {handoff.get('handoffs_per_second', 0):.1f}")
            summary.append(f"- Handoff success rate: {handoff.get('success_rate', 0):.1%}")

        if "consensus_results" in self.results:
            consensus = self.results["consensus_results"]
            summary.append(f"- Consensus success rate: {consensus.get('success_rate', 0):.1%}")
            summary.append(
                f"- Average consensus time: {consensus.get('avg_consensus_time_ms', 0):.1f}ms"
            )

        # Failure resilience
        summary.append("\n🔥 FAILURE RESILIENCE:")

        if "failure_results" in self.results:
            failure = self.results["failure_results"]
            summary.append(f"- Failures simulated: {failure.get('failure_count', 0)}")
            summary.append(
                f"- Average recovery time: {failure.get('avg_recovery_time_ms', 0):.1f}ms"
            )
            summary.append(f"- Max recovery time: {failure.get('max_recovery_time_ms', 0):.1f}ms")

        return "\n".join(summary)

    def generate_plots(self, output_dir: Path) -> None:
        """Generate visualization plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Efficiency degradation plot
        if "coordination_metrics" in self.results:
            self._plot_efficiency_degradation(output_dir)

        # Message queue performance
        if "message_queue_metrics" in self.results:
            self._plot_message_queue_performance(output_dir)

        # Scaling characteristics
        if "scaling_results" in self.results:
            self._plot_scaling_characteristics(output_dir)

    def _plot_efficiency_degradation(self, output_dir: Path) -> None:
        """Plot efficiency vs agent count."""
        metrics = self.results["coordination_metrics"]

        agent_counts = [m.agent_count for m in metrics]
        efficiencies = [m.actual_efficiency * 100 for m in metrics]
        overheads = [m.coordination_overhead * 100 for m in metrics]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Efficiency plot
        ax1.plot(agent_counts, efficiencies, "b-o", label="Actual Efficiency")
        ax1.axhline(y=28.4, color="r", linestyle="--", label="Documented Limit (28.4%)")
        ax1.set_xlabel("Number of Agents")
        ax1.set_ylabel("Efficiency (%)")
        ax1.set_title("Multi-Agent Coordination Efficiency")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Overhead plot
        ax2.plot(agent_counts, overheads, "r-o", label="Coordination Overhead")
        ax2.axhline(y=72, color="g", linestyle="--", label="Expected Overhead (72%)")
        ax2.set_xlabel("Number of Agents")
        ax2.set_ylabel("Overhead (%)")
        ax2.set_title("Coordination Overhead")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "efficiency_degradation.png", dpi=150)
        plt.close()

    def _plot_message_queue_performance(self, output_dir: Path) -> None:
        """Plot message queue performance metrics."""
        # Implementation for message queue plots
        pass

    def _plot_scaling_characteristics(self, output_dir: Path) -> None:
        """Plot scaling characteristics."""
        phases = self.results["scaling_results"]["phases"]

        agent_counts = [p["agent_count"] for p in phases]
        steps_per_sec = [p["steps_per_second"] for p in phases]
        avg_tick_ms = [p["average_tick_ms"] for p in phases]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Throughput plot
        ax1.plot(agent_counts, steps_per_sec, "g-o")
        ax1.set_xlabel("Number of Agents")
        ax1.set_ylabel("Steps per Second")
        ax1.set_title("Agent Throughput Scaling")
        ax1.grid(True, alpha=0.3)

        # Tick time plot
        ax2.plot(agent_counts, avg_tick_ms, "r-o")
        ax2.set_xlabel("Number of Agents")
        ax2.set_ylabel("Average Tick Time (ms)")
        ax2.set_title("Simulation Tick Performance")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "scaling_characteristics.png", dpi=150)
        plt.close()

    def save_json_results(self, filepath: Path) -> None:
        """Save detailed results as JSON."""
        # Convert dataclasses to dicts for JSON serialization
        serializable_results = {}

        for key, value in self.results.items():
            if key == "coordination_metrics":
                serializable_results[key] = [{k: v for k, v in m.__dict__.items()} for m in value]
            else:
                serializable_results[key] = value

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)


def run_full_load_test_suite(args):
    """Run the complete load test suite."""
    print("🚀 Starting comprehensive multi-agent coordination load tests...")
    print(f"Test configuration: max_agents={args.max_agents}, duration={args.duration}s")

    results = {}

    # Phase 1: Coordination overhead analysis
    print("\n" + "=" * 60)
    print("PHASE 1: Coordination Overhead Analysis")
    print("=" * 60)

    tester = CoordinationLoadTester()
    coordination_metrics = []

    # Test increasing agent counts
    test_counts = [1, 5, 10, 20, 30, 40, min(50, args.max_agents)]

    for count in test_counts:
        if count > args.max_agents:
            break

        print(f"\nTesting {count} agents...")
        metrics = tester.measure_coordination_overhead(count)
        coordination_metrics.append(metrics)

        print(f"  Efficiency: {metrics.actual_efficiency:.1%}")
        print(f"  Overhead: {metrics.coordination_overhead:.1%}")
        print(f"  Message latency: {metrics.coordination_latency_ms:.1f}ms")

    results["coordination_metrics"] = coordination_metrics

    # Phase 2: Simulation framework tests
    print("\n" + "=" * 60)
    print("PHASE 2: Agent Simulation Framework")
    print("=" * 60)

    environment = SimulationEnvironment(world_size=20)

    # Scaling test
    print("\nRunning scaling scenario...")
    scaling_scenario = ScalingTestScenario(environment, max_agents=args.max_agents)
    scaling_results = scaling_scenario.run(duration_seconds=min(args.duration, 30))
    results["scaling_results"] = scaling_results

    # Mixed workload test
    print("\nRunning mixed workload scenario...")
    mixed_scenario = MixedWorkloadScenario("Mixed Workload", environment)
    mixed_results = mixed_scenario.run(duration_seconds=min(args.duration, 20))
    results["mixed_workload_results"] = mixed_results

    # Phase 3: Specific coordination scenarios
    print("\n" + "=" * 60)
    print("PHASE 3: Coordination Scenarios")
    print("=" * 60)

    # Reset tester with optimal agent count
    tester = CoordinationLoadTester()
    tester.spawn_agents(min(30, args.max_agents))

    # Task handoffs
    print("\nTesting task handoffs...")
    handoff_results = tester.simulate_task_handoffs(duration_seconds=5.0)
    results["handoff_results"] = handoff_results
    print(f"  Handoffs/sec: {handoff_results['handoffs_per_second']:.1f}")
    print(f"  Success rate: {handoff_results['success_rate']:.1%}")

    # Resource contention
    print("\nTesting resource contention...")
    contention_results = tester.simulate_resource_contention(duration_seconds=5.0)
    results["contention_results"] = contention_results
    print(f"  Contention rate: {contention_results['contention_rate']:.1%}")

    # Consensus building
    print("\nTesting consensus building...")
    consensus_results = tester.simulate_consensus_building(consensus_rounds=10)
    results["consensus_results"] = consensus_results
    print(f"  Success rate: {consensus_results['success_rate']:.1%}")
    print(f"  Avg time: {consensus_results['avg_consensus_time_ms']:.1f}ms")

    # Agent failures
    print("\nTesting failure resilience...")
    failure_results = tester.simulate_agent_failures(failure_rate=0.2)
    results["failure_results"] = failure_results
    print(f"  Recovery time: {failure_results['avg_recovery_time_ms']:.1f}ms")

    # Phase 4: Generate report
    print("\n" + "=" * 60)
    print("PHASE 4: Generating Report")
    print("=" * 60)

    report = LoadTestReport(results)

    # Print summary
    print(report.generate_summary())

    # Save results
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = (
            output_path / f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report.save_json_results(json_path)
        print(f"\n📄 Detailed results saved to: {json_path}")

        # Generate plots
        if args.generate_plots:
            print("📊 Generating visualization plots...")
            report.generate_plots(output_path)
            print(f"📊 Plots saved to: {output_path}")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Check if results match documented limitations
    max_metric = coordination_metrics[-1]
    efficiency_loss = max_metric.efficiency_loss()

    all_checks_pass = True

    # Check 1: Efficiency loss
    if 70 <= efficiency_loss <= 75:
        print("✅ Efficiency loss matches documented ~72%")
    else:
        print(f"❌ Efficiency loss ({efficiency_loss:.1f}%) differs from documented 72%")
        all_checks_pass = False

    # Check 2: Practical agent limit
    if max_metric.agent_count >= 50 and max_metric.actual_efficiency >= 0.25:
        print("✅ System handles ~50 agents at documented efficiency")
    else:
        print("❌ Cannot handle 50 agents at expected efficiency")
        all_checks_pass = False

    # Check 3: Coordination latency
    if max_metric.coordination_latency_ms < 100:
        print("✅ Coordination latency within acceptable bounds")
    else:
        print("❌ Coordination latency too high")
        all_checks_pass = False

    if all_checks_pass:
        print("\n🎉 All architectural limitations validated!")
        print("The system performs within documented constraints.")
    else:
        print("\n⚠️ Some checks failed - review architectural assumptions")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive multi-agent coordination load tests"
    )

    parser.add_argument(
        "--max-agents",
        type=int,
        default=50,
        help="Maximum number of agents to test (default: 50)",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./load_test_results",
        help="Directory for output files",
    )

    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests with reduced parameters",
    )

    args = parser.parse_args()

    # Adjust for quick mode
    if args.quick:
        args.max_agents = min(args.max_agents, 20)
        args.duration = min(args.duration, 10)
        print("🏃 Running in quick mode (reduced parameters)")

    # Run the test suite
    try:
        run_full_load_test_suite(args)
        return 0
    except Exception as e:
        print(f"\n❌ Load test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

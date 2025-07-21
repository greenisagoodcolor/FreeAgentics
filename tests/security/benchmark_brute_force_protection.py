"""Performance Benchmark for Brute Force Protection.

This script benchmarks the performance impact of brute force protection
on legitimate users while under various attack scenarios.
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from httpx import AsyncClient
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    scenario: str
    legitimate_requests: int
    attack_requests: int
    legitimate_success_rate: float
    legitimate_avg_latency: float
    legitimate_p95_latency: float
    legitimate_p99_latency: float
    attack_block_rate: float
    total_duration: float
    memory_usage_mb: float
    cpu_usage_percent: float


class BruteForceBenchmark:
    """Benchmark suite for brute force protection."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []

    async def run_all_benchmarks(self):
        """Run all benchmark scenarios."""
        console.print(
            "[bold blue]Starting Brute Force Protection Benchmarks[/bold blue]"
        )

        scenarios = [
            self.benchmark_no_attack,
            self.benchmark_light_attack,
            self.benchmark_moderate_attack,
            self.benchmark_heavy_attack,
            self.benchmark_distributed_attack,
            self.benchmark_sophisticated_attack,
        ]

        for scenario in scenarios:
            console.print(f"\n[yellow]Running: {scenario.__name__}[/yellow]")
            result = await scenario()
            self.results.append(result)
            self._print_result(result)

        self._generate_report()

    async def benchmark_no_attack(self) -> BenchmarkResult:
        """Benchmark with no attack (baseline)."""
        return await self._run_scenario(
            scenario_name="No Attack (Baseline)",
            legitimate_users=50,
            legitimate_rps=10,
            attackers=0,
            attack_rps=0,
            duration=30,
        )

    async def benchmark_light_attack(self) -> BenchmarkResult:
        """Benchmark with light brute force attack."""
        return await self._run_scenario(
            scenario_name="Light Attack",
            legitimate_users=50,
            legitimate_rps=10,
            attackers=5,
            attack_rps=5,
            duration=30,
        )

    async def benchmark_moderate_attack(self) -> BenchmarkResult:
        """Benchmark with moderate brute force attack."""
        return await self._run_scenario(
            scenario_name="Moderate Attack",
            legitimate_users=50,
            legitimate_rps=10,
            attackers=20,
            attack_rps=20,
            duration=30,
        )

    async def benchmark_heavy_attack(self) -> BenchmarkResult:
        """Benchmark with heavy brute force attack."""
        return await self._run_scenario(
            scenario_name="Heavy Attack",
            legitimate_users=50,
            legitimate_rps=10,
            attackers=50,
            attack_rps=100,
            duration=30,
        )

    async def benchmark_distributed_attack(self) -> BenchmarkResult:
        """Benchmark with distributed brute force attack."""
        return await self._run_scenario(
            scenario_name="Distributed Attack",
            legitimate_users=50,
            legitimate_rps=10,
            attackers=100,
            attack_rps=50,
            duration=30,
            distributed=True,
        )

    async def benchmark_sophisticated_attack(self) -> BenchmarkResult:
        """Benchmark with sophisticated attack patterns."""
        return await self._run_scenario(
            scenario_name="Sophisticated Attack",
            legitimate_users=50,
            legitimate_rps=10,
            attackers=30,
            attack_rps=30,
            duration=30,
            sophisticated=True,
        )

    async def _run_scenario(
        self,
        scenario_name: str,
        legitimate_users: int,
        legitimate_rps: float,
        attackers: int,
        attack_rps: float,
        duration: int,
        distributed: bool = False,
        sophisticated: bool = False,
    ) -> BenchmarkResult:
        """Run a specific benchmark scenario."""

        # Initialize metrics
        legitimate_metrics = {"requests": 0, "successes": 0, "latencies": []}

        attack_metrics = {"requests": 0, "blocked": 0}

        # Resource monitoring
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        cpu_samples = []

        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {scenario_name}", total=duration)

            # Start time
            start_time = time.time()

            # Create tasks
            tasks = []

            # Legitimate user tasks
            for i in range(legitimate_users):
                task_coro = self._legitimate_user_task(
                    user_id=i,
                    rps=legitimate_rps / legitimate_users,
                    duration=duration,
                    metrics=legitimate_metrics,
                )
                tasks.append(asyncio.create_task(task_coro))

            # Attacker tasks
            if attackers > 0:
                for i in range(attackers):
                    if distributed:
                        ip = f"192.168.1.{i % 250}"
                    else:
                        ip = "attacker.example.com"

                    task_coro = self._attacker_task(
                        attacker_id=i,
                        rps=attack_rps / attackers,
                        duration=duration,
                        metrics=attack_metrics,
                        ip=ip,
                        sophisticated=sophisticated,
                    )
                    tasks.append(asyncio.create_task(task_coro))

            # Monitor progress
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                progress.update(task, completed=elapsed)

                # Sample CPU
                cpu_samples.append(process.cpu_percent())

                await asyncio.sleep(0.5)

            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate results
        total_duration = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024

        # Legitimate user metrics
        if legitimate_metrics["requests"] > 0:
            legitimate_success_rate = (
                legitimate_metrics["successes"] / legitimate_metrics["requests"]
            )
            latencies = legitimate_metrics["latencies"]
            legitimate_avg_latency = statistics.mean(latencies) if latencies else 0
            legitimate_p95_latency = np.percentile(latencies, 95) if latencies else 0
            legitimate_p99_latency = np.percentile(latencies, 99) if latencies else 0
        else:
            legitimate_success_rate = 0
            legitimate_avg_latency = 0
            legitimate_p95_latency = 0
            legitimate_p99_latency = 0

        # Attack metrics
        if attack_metrics["requests"] > 0:
            attack_block_rate = attack_metrics["blocked"] / attack_metrics["requests"]
        else:
            attack_block_rate = 0

        return BenchmarkResult(
            scenario=scenario_name,
            legitimate_requests=legitimate_metrics["requests"],
            attack_requests=attack_metrics["requests"],
            legitimate_success_rate=legitimate_success_rate,
            legitimate_avg_latency=legitimate_avg_latency * 1000,  # Convert to ms
            legitimate_p95_latency=legitimate_p95_latency * 1000,
            legitimate_p99_latency=legitimate_p99_latency * 1000,
            attack_block_rate=attack_block_rate,
            total_duration=total_duration,
            memory_usage_mb=final_memory - initial_memory,
            cpu_usage_percent=statistics.mean(cpu_samples) if cpu_samples else 0,
        )

    async def _legitimate_user_task(
        self, user_id: int, rps: float, duration: int, metrics: Dict
    ):
        """Simulate legitimate user behavior."""
        async with AsyncClient(base_url=self.base_url) as client:
            start_time = time.time()

            # Login once
            login_response = await client.post(
                "/api/v1/auth/login",
                json={
                    "username": f"user_{user_id}@example.com",
                    "password": "correct_password",
                },
            )

            if login_response.status_code == 200:
                token = login_response.json().get("access_token")
                headers = {"Authorization": f"Bearer {token}"}
            else:
                headers = {}

            # Normal API usage
            while time.time() - start_time < duration:
                request_start = time.perf_counter()

                # Mix of different endpoints
                endpoint = np.random.choice(
                    ["/api/v1/users/me", "/api/v1/data", "/api/v1/health"]
                )

                try:
                    response = await client.get(endpoint, headers=headers)

                    latency = time.perf_counter() - request_start

                    metrics["requests"] += 1
                    if response.status_code == 200:
                        metrics["successes"] += 1
                    metrics["latencies"].append(latency)

                except Exception:
                    metrics["requests"] += 1

                # Maintain request rate
                await asyncio.sleep(1.0 / rps)

    async def _attacker_task(
        self,
        attacker_id: int,
        rps: float,
        duration: int,
        metrics: Dict,
        ip: str,
        sophisticated: bool = False,
    ):
        """Simulate attacker behavior."""
        async with AsyncClient(base_url=self.base_url) as client:
            start_time = time.time()

            headers = {"X-Real-IP": ip}

            while time.time() - start_time < duration:
                # Attack patterns
                if sophisticated:
                    # Sophisticated attack with evasion
                    username = f"target_{attacker_id % 10}@example.com"
                    password = f"attempt_{int(time.time()) % 1000}"

                    # Add random delays to avoid detection
                    if np.random.random() < 0.3:
                        await asyncio.sleep(np.random.uniform(1, 3))
                else:
                    # Simple brute force
                    username = f"victim_{attacker_id}@example.com"
                    password = f"password_{int(time.time())}"

                try:
                    response = await client.post(
                        "/api/v1/auth/login",
                        json={"username": username, "password": password},
                        headers=headers,
                    )

                    metrics["requests"] += 1
                    if response.status_code == 429:
                        metrics["blocked"] += 1

                except Exception:
                    metrics["requests"] += 1

                # Maintain attack rate
                await asyncio.sleep(1.0 / rps)

    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        table = Table(title=f"Results: {result.scenario}")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Legitimate Requests", f"{result.legitimate_requests:,}")
        table.add_row(
            "Legitimate Success Rate", f"{result.legitimate_success_rate:.2%}"
        )
        table.add_row(
            "Legitimate Avg Latency", f"{result.legitimate_avg_latency:.2f} ms"
        )
        table.add_row(
            "Legitimate P95 Latency", f"{result.legitimate_p95_latency:.2f} ms"
        )
        table.add_row(
            "Legitimate P99 Latency", f"{result.legitimate_p99_latency:.2f} ms"
        )
        table.add_row("Attack Requests", f"{result.attack_requests:,}")
        table.add_row("Attack Block Rate", f"{result.attack_block_rate:.2%}")
        table.add_row("Memory Usage", f"{result.memory_usage_mb:.2f} MB")
        table.add_row("CPU Usage", f"{result.cpu_usage_percent:.1f}%")

        console.print(table)

    def _generate_report(self):
        """Generate comprehensive benchmark report."""
        console.print("\n[bold green]Benchmark Report[/bold green]")

        # Create summary table
        summary_table = Table(title="Summary Comparison")
        summary_table.add_column("Scenario", style="cyan")
        summary_table.add_column("Legit Success", style="green")
        summary_table.add_column("Avg Latency", style="yellow")
        summary_table.add_column("P99 Latency", style="yellow")
        summary_table.add_column("Attack Block", style="red")
        summary_table.add_column("Memory", style="blue")
        summary_table.add_column("CPU", style="blue")

        for result in self.results:
            summary_table.add_row(
                result.scenario,
                f"{result.legitimate_success_rate:.2%}",
                f"{result.legitimate_avg_latency:.1f}ms",
                f"{result.legitimate_p99_latency:.1f}ms",
                f"{result.attack_block_rate:.2%}",
                f"{result.memory_usage_mb:.1f}MB",
                f"{result.cpu_usage_percent:.1f}%",
            )

        console.print(summary_table)

        # Generate visualizations
        self._generate_charts()

        # Save detailed report
        self._save_detailed_report()

    def _generate_charts(self):
        """Generate performance charts."""
        scenarios = [r.scenario for r in self.results]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Brute Force Protection Performance Impact", fontsize=16)

        # Success Rate Chart
        ax1 = axes[0, 0]
        success_rates = [r.legitimate_success_rate for r in self.results]
        ax1.bar(scenarios, success_rates, color="green", alpha=0.7)
        ax1.set_title("Legitimate User Success Rate")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_ylim(0, 1.1)
        ax1.tick_params(axis="x", rotation=45)

        # Latency Chart
        ax2 = axes[0, 1]
        avg_latencies = [r.legitimate_avg_latency for r in self.results]
        p99_latencies = [r.legitimate_p99_latency for r in self.results]

        x = np.arange(len(scenarios))
        width = 0.35

        ax2.bar(x - width / 2, avg_latencies, width, label="Average", alpha=0.7)
        ax2.bar(x + width / 2, p99_latencies, width, label="P99", alpha=0.7)
        ax2.set_title("Response Latency")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()

        # Attack Block Rate Chart
        ax3 = axes[1, 0]
        block_rates = [r.attack_block_rate for r in self.results[1:]]  # Skip baseline
        attack_scenarios = scenarios[1:]
        ax3.bar(attack_scenarios, block_rates, color="red", alpha=0.7)
        ax3.set_title("Attack Block Rate")
        ax3.set_ylabel("Block Rate (%)")
        ax3.set_ylim(0, 1.1)
        ax3.tick_params(axis="x", rotation=45)

        # Resource Usage Chart
        ax4 = axes[1, 1]
        memory_usage = [r.memory_usage_mb for r in self.results]
        cpu_usage = [r.cpu_usage_percent for r in self.results]

        ax4_cpu = ax4.twinx()

        ax4.bar(
            x - width / 2,
            memory_usage,
            width,
            label="Memory (MB)",
            color="blue",
            alpha=0.7,
        )
        ax4_cpu.bar(
            x + width / 2,
            cpu_usage,
            width,
            label="CPU (%)",
            color="orange",
            alpha=0.7,
        )

        ax4.set_title("Resource Usage")
        ax4.set_ylabel("Memory (MB)", color="blue")
        ax4_cpu.set_ylabel("CPU (%)", color="orange")
        ax4.set_xticks(x)
        ax4.set_xticklabels(scenarios, rotation=45)
        ax4.tick_params(axis="y", labelcolor="blue")
        ax4_cpu.tick_params(axis="y", labelcolor="orange")

        # Add legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_cpu.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        plt.savefig(
            "brute_force_protection_benchmark.png",
            dpi=300,
            bbox_inches="tight",
        )
        console.print(
            "\n[green]Charts saved to brute_force_protection_benchmark.png[/green]"
        )

    def _save_detailed_report(self):
        """Save detailed benchmark report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "baseline_latency": self.results[0].legitimate_avg_latency,
                "max_latency_increase": max(
                    r.legitimate_avg_latency / self.results[0].legitimate_avg_latency
                    for r in self.results[1:]
                ),
                "min_success_rate": min(
                    r.legitimate_success_rate for r in self.results
                ),
                "avg_attack_block_rate": statistics.mean(
                    r.attack_block_rate for r in self.results if r.attack_requests > 0
                ),
            },
            "scenarios": [
                {
                    "scenario": r.scenario,
                    "legitimate_requests": r.legitimate_requests,
                    "attack_requests": r.attack_requests,
                    "legitimate_success_rate": r.legitimate_success_rate,
                    "legitimate_avg_latency_ms": r.legitimate_avg_latency,
                    "legitimate_p95_latency_ms": r.legitimate_p95_latency,
                    "legitimate_p99_latency_ms": r.legitimate_p99_latency,
                    "attack_block_rate": r.attack_block_rate,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                }
                for r in self.results
            ],
        }

        with open("brute_force_protection_benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2)

        console.print(
            "[green]Detailed report saved to brute_force_protection_benchmark_report.json[/green]"
        )

        # Print recommendations
        self._print_recommendations()

    def _print_recommendations(self):
        """Print performance recommendations based on results."""
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")

        baseline = self.results[0]
        worst_case = max(self.results[1:], key=lambda r: r.legitimate_avg_latency)

        # Success rate recommendation
        min_success = min(r.legitimate_success_rate for r in self.results)
        if min_success < 0.95:
            console.print(
                f"[red]⚠️  Legitimate success rate drops to {min_success:.2%} under attack. "
                "Consider tuning rate limits to be less aggressive for authenticated users.[/red]"
            )
        else:
            console.print(
                f"[green]✓ Legitimate success rate maintained above 95% ({min_success:.2%})[/green]"
            )

        # Latency recommendation
        max_latency_increase = (
            worst_case.legitimate_avg_latency / baseline.legitimate_avg_latency
        )
        if max_latency_increase > 2:
            console.print(
                f"[red]⚠️  Latency increases {max_latency_increase:.1f}x under heavy attack. "
                "Consider optimizing protection algorithms or adding caching.[/red]"
            )
        else:
            console.print(
                f"[green]✓ Latency increase acceptable ({max_latency_increase:.1f}x)[/green]"
            )

        # Attack effectiveness
        avg_block_rate = statistics.mean(
            r.attack_block_rate for r in self.results if r.attack_requests > 0
        )
        if avg_block_rate < 0.9:
            console.print(
                f"[yellow]⚠️  Average attack block rate is {avg_block_rate:.2%}. "
                "Consider strengthening protection thresholds.[/yellow]"
            )
        else:
            console.print(
                f"[green]✓ Excellent attack block rate ({avg_block_rate:.2%})[/green]"
            )

        # Resource usage
        max_memory = max(r.memory_usage_mb for r in self.results)
        max_cpu = max(r.cpu_usage_percent for r in self.results)

        if max_memory > 100:
            console.print(
                f"[yellow]⚠️  Memory usage peaks at {max_memory:.1f}MB. "
                "Monitor for memory leaks during sustained attacks.[/yellow]"
            )

        if max_cpu > 80:
            console.print(
                f"[yellow]⚠️  CPU usage peaks at {max_cpu:.1f}%. "
                "Consider rate limiting at load balancer level.[/yellow]"
            )


async def main():
    """Run the benchmark suite."""
    benchmark = BruteForceBenchmark()
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())

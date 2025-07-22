#!/usr/bin/env python3
"""Run all brute force protection tests and generate a comprehensive report."""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class BruteForceTestRunner:
    """Orchestrates running all brute force protection tests."""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_all_tests(self):
        """Run all brute force protection tests."""
        self.start_time = time.time()

        console.print(
            Panel.fit(
                "[bold blue]Brute Force Protection Test Suite[/bold blue]\n"
                "Running comprehensive security validation...",
                border_style="blue",
            )
        )

        # Define test suites
        test_suites = [
            {
                "name": "Basic Protection Tests",
                "module": "tests/security/test_brute_force_protection.py",
                "tests": [
                    "TestAuthenticationBruteForce",
                    "TestTokenBruteForce",
                    "TestResourceEnumeration",
                    "TestProtectionValidation",
                    "TestPerformanceImpact",
                ],
            },
            {
                "name": "Advanced Scenario Tests",
                "module": "tests/security/test_brute_force_advanced_scenarios.py",
                "tests": [
                    "TestTimingAttackPrevention",
                    "TestDistributedCoordinatedAttacks",
                    "TestAccountTakeoverProtection",
                    "TestZeroDayPatternDetection",
                    "TestAdaptiveProtectionMechanisms",
                ],
            },
        ]

        # Run each test suite
        for suite in test_suites:
            self._run_test_suite(suite)

        # Run performance benchmark
        self._run_benchmark()

        self.end_time = time.time()

        # Generate final report
        self._generate_report()

    def _run_test_suite(self, suite: Dict):
        """Run a specific test suite."""
        console.print(f"\n[yellow]Running {suite['name']}[/yellow]")

        results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for test_class in suite["tests"]:
                task = progress.add_task(f"Testing {test_class}...", total=None)

                # Run pytest for specific test class
                cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"{suite['module']}::{test_class}",
                    "-v",
                    "--tb=short",
                    "--json-report",
                    f"--json-report-file=test_report_{test_class}.json",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                # Parse results
                test_passed = result.returncode == 0

                try:
                    with open(f"test_report_{test_class}.json", "r") as f:
                        report = json.load(f)

                    results[test_class] = {
                        "passed": test_passed,
                        "total": report["summary"]["total"],
                        "passed_count": report["summary"]["passed"],
                        "failed_count": report["summary"]["failed"],
                        "duration": report["duration"],
                    }

                    # Clean up report file
                    Path(f"test_report_{test_class}.json").unlink()

                except Exception:
                    results[test_class] = {
                        "passed": test_passed,
                        "total": 0,
                        "passed_count": 0,
                        "failed_count": 0,
                        "duration": 0,
                    }

                progress.update(task, completed=True)

        self.results[suite["name"]] = results
        self._print_suite_results(suite["name"], results)

    def _run_benchmark(self):
        """Run performance benchmark."""
        console.print("\n[yellow]Running Performance Benchmark[/yellow]")

        cmd = [
            sys.executable,
            "tests/security/benchmark_brute_force_protection.py",
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmark scenarios...", total=None)

            result = subprocess.run(cmd, capture_output=True, text=True)

            progress.update(task, completed=True)

        self.results["benchmark"] = {
            "completed": result.returncode == 0,
            "output": result.stdout if result.returncode == 0 else result.stderr,
        }

    def _print_suite_results(self, suite_name: str, results: Dict):
        """Print results for a test suite."""
        table = Table(title=f"{suite_name} Results")

        table.add_column("Test Class", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Tests", style="blue")
        table.add_column("Duration", style="yellow")

        for test_class, result in results.items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            tests = f"{result['passed_count']}/{result['total']}"
            duration = f"{result['duration']:.2f}s"

            table.add_row(test_class, status, tests, duration)

        console.print(table)

    def _generate_report(self):
        """Generate comprehensive test report."""
        total_duration = self.end_time - self.start_time

        console.print("\n" + "=" * 80)
        console.print(
            Panel.fit(
                "[bold green]Brute Force Protection Test Report[/bold green]",
                border_style="green",
            )
        )

        # Calculate totals
        total_tests = 0
        total_passed = 0
        total_failed = 0

        for suite_name, suite_results in self.results.items():
            if suite_name != "benchmark":
                for test_class, result in suite_results.items():
                    total_tests += result["total"]
                    total_passed += result["passed_count"]
                    total_failed += result["failed_count"]

        # Summary table
        summary_table = Table(title="Test Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Tests", str(total_tests))
        summary_table.add_row("Passed", f"{total_passed} ({total_passed / total_tests * 100:.1f}%)")
        summary_table.add_row("Failed", f"{total_failed} ({total_failed / total_tests * 100:.1f}%)")
        summary_table.add_row("Duration", f"{total_duration:.2f} seconds")
        summary_table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        console.print(summary_table)

        # Security validation checklist
        console.print("\n[bold yellow]Security Validation Checklist:[/bold yellow]")

        checklist = [
            ("Authentication Rate Limiting", self._check_auth_protection()),
            ("Token Security", self._check_token_security()),
            (
                "Resource Enumeration Protection",
                self._check_enumeration_protection(),
            ),
            ("Timing Attack Prevention", self._check_timing_protection()),
            ("Distributed Attack Defense", self._check_distributed_defense()),
            (
                "Performance Impact Acceptable",
                self._check_performance_impact(),
            ),
        ]

        for item, passed in checklist:
            status = "✅" if passed else "❌"
            console.print(f"  {status} {item}")

        # Save detailed report
        self._save_json_report()

        # Print recommendations
        self._print_recommendations()

    def _check_auth_protection(self) -> bool:
        """Check if authentication protection tests passed."""
        suite = self.results.get("Basic Protection Tests", {})
        test = suite.get("TestAuthenticationBruteForce", {})
        return test.get("passed", False)

    def _check_token_security(self) -> bool:
        """Check if token security tests passed."""
        suite = self.results.get("Basic Protection Tests", {})
        test = suite.get("TestTokenBruteForce", {})
        return test.get("passed", False)

    def _check_enumeration_protection(self) -> bool:
        """Check if enumeration protection tests passed."""
        suite = self.results.get("Basic Protection Tests", {})
        test = suite.get("TestResourceEnumeration", {})
        return test.get("passed", False)

    def _check_timing_protection(self) -> bool:
        """Check if timing attack prevention tests passed."""
        suite = self.results.get("Advanced Scenario Tests", {})
        test = suite.get("TestTimingAttackPrevention", {})
        return test.get("passed", False)

    def _check_distributed_defense(self) -> bool:
        """Check if distributed attack defense tests passed."""
        suite = self.results.get("Advanced Scenario Tests", {})
        test = suite.get("TestDistributedCoordinatedAttacks", {})
        return test.get("passed", False)

    def _check_performance_impact(self) -> bool:
        """Check if performance impact is acceptable."""
        suite = self.results.get("Basic Protection Tests", {})
        test = suite.get("TestPerformanceImpact", {})
        return test.get("passed", False)

    def _save_json_report(self):
        """Save detailed JSON report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": self.end_time - self.start_time,
            "results": self.results,
            "summary": {
                "auth_protection": self._check_auth_protection(),
                "token_security": self._check_token_security(),
                "enumeration_protection": self._check_enumeration_protection(),
                "timing_protection": self._check_timing_protection(),
                "distributed_defense": self._check_distributed_defense(),
                "performance_acceptable": self._check_performance_impact(),
            },
        }

        filename = f"brute_force_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        console.print(f"\n[green]Detailed report saved to {filename}[/green]")

    def _print_recommendations(self):
        """Print security recommendations based on test results."""
        console.print("\n[bold cyan]Security Recommendations:[/bold cyan]")

        if not self._check_auth_protection():
            console.print(
                "[red]⚠️  Authentication protection failed. Review rate limiting "
                "thresholds and lockout mechanisms.[/red]"
            )

        if not self._check_timing_protection():
            console.print(
                "[red]⚠️  Timing attack vulnerabilities detected. Implement "
                "constant-time comparison functions.[/red]"
            )

        if not self._check_performance_impact():
            console.print(
                "[yellow]⚠️  Performance impact exceeds thresholds. Consider "
                "optimizing protection algorithms.[/yellow]"
            )

        all_passed = all(
            [
                self._check_auth_protection(),
                self._check_token_security(),
                self._check_enumeration_protection(),
                self._check_timing_protection(),
                self._check_distributed_defense(),
                self._check_performance_impact(),
            ]
        )

        if all_passed:
            console.print(
                "[green]✅ All brute force protection tests passed! "
                "The system is well-protected against brute force attacks.[/green]"
            )
        else:
            console.print(
                "[red]❌ Some protection mechanisms need improvement. "
                "Address the issues above before deployment.[/red]"
            )


def check_prerequisites():
    """Check if prerequisites are met."""
    console.print("[cyan]Checking prerequisites...[/cyan]")

    # Check Redis
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379)
        r.ping()
        console.print("✅ Redis is running")
    except Exception:
        console.print("[red]❌ Redis is not running. Please start Redis first.[/red]")
        console.print("   Run: docker run -d -p 6379:6379 redis:latest")
        return False

    # Check pytest
    try:
        pass

        console.print("✅ pytest is installed")
    except ImportError:
        console.print("[red]❌ pytest is not installed.[/red]")
        console.print("   Run: pip install pytest pytest-asyncio pytest-json-report")
        return False

    return True


def main():
    """Main entry point."""
    if not check_prerequisites():
        sys.exit(1)

    runner = BruteForceTestRunner()

    try:
        runner.run_all_tests()
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error running tests: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

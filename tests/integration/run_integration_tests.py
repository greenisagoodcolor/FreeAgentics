#!/usr/bin/env python3
"""
Integration Test Runner

Orchestrates and runs all integration tests with proper reporting.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Runs integration tests with reporting."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialize test runner."""
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, Any] = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "summary": {},
        }

    def run_test_suite(
        self, suite_name: str, test_file: str, markers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run a test suite.

        Args:
            suite_name: Name of the test suite
            test_file: Path to test file
            markers: Optional pytest markers

        Returns:
            Test results
        """
        logger.info(f"Running {suite_name}...")

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_file,
            "-v",
            "--asyncio-mode=auto",
            "--tb=short",
            f"--junit-xml={self.output_dir}/{suite_name}_results.xml",
            f"--html={self.output_dir}/{suite_name}_report.html",
            "--self-contained-html",
        ]

        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        # Run tests
        start_time = time.time()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

            duration = time.time() - start_time

            # Parse output
            passed = failed = skipped = 0

            for line in result.stdout.split("\n"):
                if " passed" in line and " failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            passed = int(parts[i - 1])
                        elif part == "failed":
                            failed = int(parts[i - 1])
                        elif part == "skipped":
                            skipped = int(parts[i - 1])

            test_result = {
                "suite": suite_name,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "total": passed + failed + skipped,
                "duration": duration,
                "success": result.returncode == 0,
                "output": result.stdout[-1000:] if not result.returncode == 0 else "",
            }

            self.results["tests"][suite_name] = test_result

            # Print summary
            status = "✓ PASSED" if test_result["success"] else "✗ FAILED"
            logger.info(
                f"{status} {suite_name}: "
                f"{passed} passed, {failed} failed, {skipped} skipped "
                f"({duration:.2f}s)"
            )

            return test_result

        except Exception as e:
            logger.error(f"Error running {suite_name}: {e}")
            return {"suite": suite_name, "error": str(e), "success": False}

    def run_all_tests(self, include_performance: bool = False):
        """Run all integration tests."""
        logger.info("Starting integration test suite...")

        test_suites: List[tuple[str, str, Optional[list[str]]]] = [
            ("Agent Integration", "test_agent_integration.py", None),
            ("System Integration", "test_system_integration.py", None),
        ]

        if include_performance:
            test_suites.append(("Performance Tests", "test_performance.py", ["performance"]))

        # Run each test suite
        for suite_name, test_file, markers in test_suites:
            self.run_test_suite(suite_name, test_file, markers)

        # Generate summary
        self._generate_summary()

        # Save results
        self._save_results()

        # Generate report
        self._generate_report()

    def _generate_summary(self):
        """Generate test summary."""
        tests_dict = self.results["tests"]
        if isinstance(tests_dict, dict):
            total_passed = sum(t.get("passed", 0) for t in tests_dict.values())
            total_failed = sum(t.get("failed", 0) for t in tests_dict.values())
            total_skipped = sum(t.get("skipped", 0) for t in tests_dict.values())
            total_duration = sum(t.get("duration", 0) for t in tests_dict.values())
        else:
            total_passed = total_failed = total_skipped = total_duration = 0

        self.results["summary"] = {
            "total_tests": total_passed + total_failed + total_skipped,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "duration": total_duration,
            "success_rate": (
                (total_passed / (total_passed + total_failed) * 100)
                if (total_passed + total_failed) > 0
                else 0
            ),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _save_results(self):
        """Save test results to JSON."""
        results_file = self.output_dir / "integration_test_results.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to: {results_file}")

    def _generate_report(self):
        """Generate HTML report."""
        report_file = self.output_dir / "integration_test_report.html"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .skipped {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>FreeAgentics Integration Test Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Start Time:</strong> {self.results['start_time']}</p>
        <p><strong>End Time:</strong> {self.results['summary']['end_time']}</p>
        <p><strong>Total Duration:</strong> {self.results['summary']['duration']:.2f} seconds</p>
        <p><strong>Total Tests:</strong> {self.results['summary']['total_tests']}</p>
        <p class="passed"><strong>Passed:</strong> {self.results['summary']['passed']}</p>
        <p class="failed"><strong>Failed:</strong> {self.results['summary']['failed']}</p>
        <p class="skipped"><strong>Skipped:</strong> {self.results['summary']['skipped']}</p>
        <p><strong>Success Rate:</strong> {self.results['summary']['success_rate']:.1f}%</p>
    </div>

    <h2>Test Suites</h2>
    <table>
        <tr>
            <th>Suite</th>
            <th>Status</th>
            <th>Passed</th>
            <th>Failed</th>
            <th>Skipped</th>
            <th>Duration</th>
        </tr>
"""

        for suite_name, result in self.results["tests"].items():
            status = "✓" if result.get("success", False) else "✗"
            status_class = "passed" if result.get("success", False) else "failed"

            html_content += f"""
        <tr>
            <td>{suite_name}</td>
            <td class="{status_class}">{status}</td>
            <td>{result.get('passed', 0)}</td>
            <td>{result.get('failed', 0)}</td>
            <td>{result.get('skipped', 0)}</td>
            <td>{result.get('duration', 0):.2f}s</td>
        </tr>
"""

        html_content += """
    </table>

    <h2>Detailed Reports</h2>
    <ul>
"""

        # Add links to individual reports
        for suite_name in self.results["tests"]:
            report_name = f"{suite_name.replace(' ', '_')}_report.html"
            if (self.output_dir / report_name).exists():
                html_content += (
                    f'        <li><a href="{report_name}">{suite_name} Detailed Report</a></li>\n'
                )

        html_content += """
    </ul>
</body>
</html>
"""

        with open(report_file, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {report_file}")


def setup_test_environment() -> None:
    """Set up test environment."""
    # Install required test packages
    requirements = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-html>=3.1.0",
        "pytest-cov>=4.0.0",
        "pytest-timeout>=2.1.0",
    ]

    logger.info("Installing test requirements...")

    for req in requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", req], capture_output=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run FreeAgentics integration tests")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results",
        help="Output directory for test results",
    )
    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="Include performance tests (takes longer)",
    )
    parser.add_argument("--setup", action="store_true", help="Set up test environment first")

    args = parser.parse_args()

    if args.setup:
        setup_test_environment()

    # Create test runner
    runner = IntegrationTestRunner(Path(args.output_dir))

    # Run tests
    try:
        runner.run_all_tests(include_performance=args.include_performance)

        # Print final summary
        summary = runner.results["summary"]

        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ({summary['passed']/summary['total_tests']*100:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Duration: {summary['duration']:.2f} seconds")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print("=" * 60)

        # Exit with appropriate code
        sys.exit(0 if summary["failed"] == 0 else 1)

    except KeyboardInterrupt:
        logger.warning("Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

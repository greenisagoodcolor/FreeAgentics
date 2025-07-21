#!/usr/bin/env python3
"""
Integration Test Dashboard for FreeAgentics

This dashboard provides a comprehensive view of all integration tests,
their status, and allows easy execution with proper categorization.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

# Test categories with their descriptions
TEST_CATEGORIES = {
    "simple": {
        "description": "Tests that don't require external services",
        "tests": [
            "test_coordination_interface_simple.py",
            "test_pymdp_validation.py",
            "test_action_sampling_issue.py",
            "test_nemesis_pymdp_validation.py",
            "test_pymdp_hard_failure_integration.py",
        ],
    },
    "database": {
        "description": "Tests requiring PostgreSQL database",
        "tests": [
            "test_authentication_flow.py",
            "test_session_management.py",
            "test_knowledge_graph_auto_updates.py",
            "test_gmn_versioned_storage_integration.py",
        ],
    },
    "messaging": {
        "description": "Tests requiring RabbitMQ/Redis",
        "tests": [
            "test_rate_limiting.py",
            "test_websocket_auth_integration.py",
            "test_alerting_system.py",
        ],
    },
    "monitoring": {
        "description": "Tests for observability and monitoring",
        "tests": [
            "test_agent_inference_metrics.py",
            "test_belief_monitoring.py",
            "test_coordination_metrics.py",
            "test_monitoring_dashboard.py",
        ],
    },
    "performance": {
        "description": "Performance and load tests",
        "tests": [
            "test_auth_load.py",
            "test_rbac_scale.py",
            "test_monitoring_load.py",
        ],
    },
    "security": {
        "description": "Security validation tests",
        "tests": [
            "test_auth_security_headers.py",
            "test_security_headers_simple.py",
            "test_auth_rate_limiting.py",
        ],
    },
    "comprehensive": {
        "description": "Full end-to-end integration tests",
        "tests": [
            "test_comprehensive_gnn_llm_coalition_integration.py",
            "test_gnn_llm_interface_integration.py",
            "test_llm_coalition_interface_integration.py",
            "test_coalition_agents_interface_integration.py",
            "test_matrix_pooling_pymdp.py",
            "test_comprehensive_pymdp_integration_nemesis.py",
        ],
    },
}


class IntegrationTestDashboard:
    """Dashboard for managing and running integration tests."""

    def __init__(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(os.path.dirname(self.test_dir))
        self.venv_python = os.path.join(self.project_root, "venv", "bin", "python")
        self.results = {}

    def check_environment(self) -> Dict[str, bool]:
        """Check if test environment is properly set up."""
        checks = {
            "venv_exists": os.path.exists(self.venv_python),
            "env_test_exists": os.path.exists(
                os.path.join(self.project_root, ".env.test")
            ),
            "docker_running": self._check_docker(),
            "containers_available": self._check_containers_available(),
        }
        return checks

    def _check_docker(self) -> bool:
        """Check if Docker is running."""
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def _check_containers_available(self) -> bool:
        """Check if docker-compose.test.yml exists."""
        compose_file = os.path.join(self.project_root, "docker-compose.test.yml")
        return os.path.exists(compose_file)

    def list_tests(self, category: str = None) -> List[str]:
        """List all available tests or tests in a specific category."""
        if category:
            return TEST_CATEGORIES.get(category, {}).get("tests", [])

        all_tests = []
        for cat_data in TEST_CATEGORIES.values():
            all_tests.extend(cat_data["tests"])
        return list(set(all_tests))  # Remove duplicates

    def run_test(self, test_file: str, verbose: bool = False) -> Tuple[bool, str]:
        """Run a single test file and return success status and output."""
        test_path = os.path.join(self.test_dir, test_file)

        if not os.path.exists(test_path):
            return False, f"Test file not found: {test_file}"

        # Load test environment
        env = os.environ.copy()
        env_file = os.path.join(self.project_root, ".env.test")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        env[key] = value

        # Run test
        cmd = [
            self.venv_python,
            "-m",
            "pytest",
            test_path,
            "-v" if verbose else "-q",
            "--tb=short",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=60
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Test timed out after 60 seconds"
        except Exception as e:
            return False, f"Error running test: {str(e)}"

    def run_category(
        self, category: str, verbose: bool = False
    ) -> Dict[str, Tuple[bool, str]]:
        """Run all tests in a category."""
        tests = self.list_tests(category)
        results = {}

        for test in tests:
            print(f"Running {test}...", end=" ", flush=True)
            success, output = self.run_test(test, verbose)
            results[test] = (success, output)
            print("✓" if success else "✗")

        return results

    def generate_report(self, results: Dict[str, Tuple[bool, str]]) -> str:
        """Generate a summary report of test results."""
        total = len(results)
        passed = sum(1 for success, _ in results.values() if success)
        failed = total - passed

        report = f"""
Integration Test Report
======================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Summary:
--------
Total Tests: {total}
Passed: {passed} ({passed / total * 100:.1f}%)
Failed: {failed} ({failed / total * 100:.1f}%)

Failed Tests:
-------------
"""

        for test, (success, output) in results.items():
            if not success:
                # Extract failure summary from output
                lines = output.split("\n")
                error_lines = [l for l in lines if "FAILED" in l or "ERROR" in l][:3]
                error_summary = (
                    "\n  ".join(error_lines)
                    if error_lines
                    else "Check output for details"
                )
                report += f"\n{test}:\n  {error_summary}\n"

        return report

    def save_results(self, results: Dict[str, Tuple[bool, str]], filename: str = None):
        """Save test results to a JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integration_test_results_{timestamp}.json"

        filepath = os.path.join(self.project_root, filename)

        data = {
            "timestamp": datetime.now().isoformat(),
            "results": {
                test: {"success": success, "output": output}
                for test, (success, output) in results.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath


def main():
    """Main entry point for the dashboard."""
    parser = argparse.ArgumentParser(
        description="FreeAgentics Integration Test Dashboard"
    )
    parser.add_argument(
        "command",
        choices=["check", "list", "run", "report"],
        help="Command to execute",
    )
    parser.add_argument("-c", "--category", help="Test category to run")
    parser.add_argument("-t", "--test", help="Specific test file to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-s", "--save", help="Save results to file")

    args = parser.parse_args()

    dashboard = IntegrationTestDashboard()

    if args.command == "check":
        # Check environment
        checks = dashboard.check_environment()
        print("Environment Check:")
        print("-" * 40)
        for check, status in checks.items():
            status_str = "✓" if status else "✗"
            print(f"{check:<25} {status_str}")

        if not all(checks.values()):
            print("\nSome checks failed. Please ensure:")
            print("1. Virtual environment is activated")
            print("2. .env.test file exists")
            print("3. Docker is running (for full tests)")

    elif args.command == "list":
        # List tests
        if args.category:
            tests = dashboard.list_tests(args.category)
            print(f"\nTests in category '{args.category}':")
            print(
                f"Description: {TEST_CATEGORIES.get(args.category, {}).get('description', 'Unknown')}"
            )
            print("-" * 40)
            for test in sorted(tests):
                print(f"  {test}")
        else:
            print("\nTest Categories:")
            print("-" * 40)
            for cat, data in TEST_CATEGORIES.items():
                print(f"\n{cat}: {data['description']}")
                print(f"  Tests: {len(data['tests'])}")

    elif args.command == "run":
        # Run tests
        if args.test:
            # Run single test
            print(f"Running test: {args.test}")
            success, output = dashboard.run_test(args.test, args.verbose)
            print("\nResult:", "✓ PASSED" if success else "✗ FAILED")
            if args.verbose or not success:
                print("\nOutput:")
                print(output)
            results = {args.test: (success, output)}

        elif args.category:
            # Run category
            print(f"Running category: {args.category}")
            results = dashboard.run_category(args.category, args.verbose)

        else:
            # Run all simple tests by default
            print("Running simple tests (no external dependencies)...")
            results = dashboard.run_category("simple", args.verbose)

        # Generate and display report
        report = dashboard.generate_report(results)
        print(report)

        # Save results if requested
        if args.save:
            filepath = dashboard.save_results(results, args.save)
            print(f"\nResults saved to: {filepath}")

    elif args.command == "report":
        # Generate report from saved results
        if not args.save:
            print("Please specify a results file with -s")
            return

        with open(args.save) as f:
            data = json.load(f)

        results = {
            test: (res["success"], res["output"])
            for test, res in data["results"].items()
        }

        report = dashboard.generate_report(results)
        print(report)


if __name__ == "__main__":
    main()

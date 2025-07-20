#!/usr/bin/env python3
"""
Comprehensive Authentication Test Suite Runner

This script runs the complete authentication test suite including:
- Security tests (injection attacks, brute force, etc.)
- Integration tests (complete authentication flows)
- Unit tests (edge cases and error handling)
- Performance tests (load testing and benchmarks)

Usage:
    python tests/run_comprehensive_auth_tests.py [options]

Options:
    --security-only     Run only security tests
    --integration-only  Run only integration tests
    --unit-only         Run only unit tests
    --performance-only  Run only performance tests
    --quick             Run quick tests only (skip intensive performance tests)
    --verbose           Enable verbose output
    --report            Generate detailed HTML report
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
try:
    from tests.integration.test_comprehensive_auth_flows import (
        TestCompleteAuthenticationFlows,
    )
    from tests.performance.test_authentication_performance import (
        TestAuthenticationPerformance,
    )
    from tests.security.test_comprehensive_auth_security import (
        TestAuthenticationSecuritySuite,
    )
    from tests.unit.test_authentication_edge_cases import (
        TestAuthenticationEdgeCases,
        TestAuthenticationErrorHandling,
    )
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Please run this script from the project root directory")
    sys.exit(1)


class ComprehensiveAuthTestRunner:
    """Comprehensive authentication test runner."""

    def __init__(self):
        self.results = {
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "start_time": None,
                "end_time": None,
                "duration": 0,
            },
            "test_suites": {},
            "errors": [],
            "performance_metrics": {},
            "security_report": {},
            "recommendations": [],
        }

    def run_security_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run comprehensive security tests."""
        print("Running Authentication Security Tests...")

        suite_results = {
            "name": "Security Tests",
            "status": "running",
            "tests": [],
            "errors": [],
        }

        try:
            # Run security test suite
            security_suite = TestAuthenticationSecuritySuite()
            security_suite.setup_method()

            # Run comprehensive security tests
            security_report = (
                security_suite.test_generate_comprehensive_security_report()
            )

            suite_results["status"] = "passed"
            suite_results["security_report"] = security_report

            # Check for critical vulnerabilities
            if security_report["summary"]["vulnerabilities_found"] > 0:
                suite_results["status"] = "failed"
                suite_results["errors"].append(
                    "Critical security vulnerabilities found"
                )

            security_suite.teardown_method()

            print(
                f"✓ Security tests completed - Score: {security_report['summary']['security_score']:.1f}%"
            )

        except Exception as e:
            suite_results["status"] = "failed"
            suite_results["errors"].append(f"Security test failure: {str(e)}")
            print(f"✗ Security tests failed: {e}")
            if verbose:
                traceback.print_exc()

        return suite_results

    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run comprehensive integration tests."""
        print("Running Authentication Integration Tests...")

        suite_results = {
            "name": "Integration Tests",
            "status": "running",
            "tests": [],
            "errors": [],
        }

        try:
            # Run integration test suite
            integration_suite = TestCompleteAuthenticationFlows()
            integration_suite.setup_method()

            # Run comprehensive flow tests
            flow_report = (
                integration_suite.test_generate_comprehensive_flow_report()
            )

            suite_results["status"] = "passed"
            suite_results["flow_report"] = flow_report

            # Check success rate
            if flow_report["summary"]["success_rate"] < 90:
                suite_results["status"] = "failed"
                suite_results["errors"].append(
                    "Integration test success rate too low"
                )

            integration_suite.teardown_method()

            print(
                f"✓ Integration tests completed - Success rate: {flow_report['summary']['success_rate']:.1f}%"
            )

        except Exception as e:
            suite_results["status"] = "failed"
            suite_results["errors"].append(
                f"Integration test failure: {str(e)}"
            )
            print(f"✗ Integration tests failed: {e}")
            if verbose:
                traceback.print_exc()

        return suite_results

    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run comprehensive unit tests."""
        print("Running Authentication Unit Tests...")

        suite_results = {
            "name": "Unit Tests",
            "status": "running",
            "tests": [],
            "errors": [],
        }

        try:
            # Run edge case tests
            edge_case_suite = TestAuthenticationEdgeCases()
            edge_case_suite.setup_method()

            # Run key edge case tests
            edge_case_tests = [
                "test_malformed_request_handling",
                "test_invalid_token_formats",
                "test_unicode_and_encoding_edge_cases",
                "test_input_validation_edge_cases",
                "test_concurrent_user_creation",
                "test_rate_limiter_edge_cases",
                "test_password_hashing_edge_cases",
                "test_memory_pressure_handling",
                "test_extreme_input_values",
            ]

            for test_name in edge_case_tests:
                try:
                    test_method = getattr(edge_case_suite, test_name)
                    test_method()
                    suite_results["tests"].append(
                        {"name": test_name, "status": "passed"}
                    )
                except Exception as e:
                    suite_results["tests"].append(
                        {
                            "name": test_name,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    suite_results["errors"].append(f"{test_name}: {str(e)}")

            edge_case_suite.teardown_method()

            # Run error handling tests
            error_suite = TestAuthenticationErrorHandling()
            error_suite.setup_method()

            error_handling_tests = [
                "test_exception_propagation",
                "test_error_message_safety",
                "test_graceful_degradation",
                "test_resource_cleanup_on_error",
            ]

            for test_name in error_handling_tests:
                try:
                    test_method = getattr(error_suite, test_name)
                    test_method()
                    suite_results["tests"].append(
                        {"name": test_name, "status": "passed"}
                    )
                except Exception as e:
                    suite_results["tests"].append(
                        {
                            "name": test_name,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    suite_results["errors"].append(f"{test_name}: {str(e)}")

            error_suite.teardown_method()

            # Determine overall status
            failed_tests = [
                t for t in suite_results["tests"] if t["status"] == "failed"
            ]
            if failed_tests:
                suite_results["status"] = "failed"
            else:
                suite_results["status"] = "passed"

            print(
                f"✓ Unit tests completed - {len(suite_results['tests']) - len(failed_tests)}/{len(suite_results['tests'])} passed"
            )

        except Exception as e:
            suite_results["status"] = "failed"
            suite_results["errors"].append(f"Unit test failure: {str(e)}")
            print(f"✗ Unit tests failed: {e}")
            if verbose:
                traceback.print_exc()

        return suite_results

    def run_performance_tests(
        self, quick: bool = False, verbose: bool = False
    ) -> Dict[str, Any]:
        """Run comprehensive performance tests."""
        print("Running Authentication Performance Tests...")

        suite_results = {
            "name": "Performance Tests",
            "status": "running",
            "tests": [],
            "errors": [],
        }

        try:
            # Run performance test suite
            performance_suite = TestAuthenticationPerformance()
            performance_suite.setup_method()

            if quick:
                # Run quick performance tests
                performance_tests = [
                    "test_login_performance_baseline",
                    "test_token_generation_performance",
                    "test_token_validation_performance",
                ]
            else:
                # Run comprehensive performance tests
                performance_tests = [
                    "test_login_performance_baseline",
                    "test_token_generation_performance",
                    "test_token_validation_performance",
                    "test_concurrent_login_performance",
                    "test_high_volume_token_refresh",
                    "test_rate_limiting_performance",
                    "test_memory_usage_under_load",
                    "test_stress_test_authentication_system",
                ]

            for test_name in performance_tests:
                try:
                    test_method = getattr(performance_suite, test_name)
                    test_method()
                    suite_results["tests"].append(
                        {"name": test_name, "status": "passed"}
                    )
                except Exception as e:
                    suite_results["tests"].append(
                        {
                            "name": test_name,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    suite_results["errors"].append(f"{test_name}: {str(e)}")

            # Generate performance report
            performance_report = performance_suite.metrics.generate_report()
            suite_results["performance_report"] = performance_report

            performance_suite.teardown_method()

            # Determine overall status
            failed_tests = [
                t for t in suite_results["tests"] if t["status"] == "failed"
            ]
            if failed_tests:
                suite_results["status"] = "failed"
            else:
                suite_results["status"] = "passed"

            print(
                f"✓ Performance tests completed - {len(suite_results['tests']) - len(failed_tests)}/{len(suite_results['tests'])} passed"
            )

        except Exception as e:
            suite_results["status"] = "failed"
            suite_results["errors"].append(
                f"Performance test failure: {str(e)}"
            )
            print(f"✗ Performance tests failed: {e}")
            if verbose:
                traceback.print_exc()

        return suite_results

    def run_comprehensive_tests(
        self,
        security_only: bool = False,
        integration_only: bool = False,
        unit_only: bool = False,
        performance_only: bool = False,
        quick: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run comprehensive authentication test suite."""
        print("=" * 60)
        print("COMPREHENSIVE AUTHENTICATION TEST SUITE")
        print("=" * 60)

        self.results["summary"]["start_time"] = datetime.now(
            timezone.utc
        ).isoformat()
        start_time = time.time()

        # Run selected test suites
        if security_only:
            self.results["test_suites"]["security"] = self.run_security_tests(
                verbose
            )
        elif integration_only:
            self.results["test_suites"][
                "integration"
            ] = self.run_integration_tests(verbose)
        elif unit_only:
            self.results["test_suites"]["unit"] = self.run_unit_tests(verbose)
        elif performance_only:
            self.results["test_suites"][
                "performance"
            ] = self.run_performance_tests(quick, verbose)
        else:
            # Run all test suites
            self.results["test_suites"]["security"] = self.run_security_tests(
                verbose
            )
            self.results["test_suites"][
                "integration"
            ] = self.run_integration_tests(verbose)
            self.results["test_suites"]["unit"] = self.run_unit_tests(verbose)
            self.results["test_suites"][
                "performance"
            ] = self.run_performance_tests(quick, verbose)

        end_time = time.time()
        self.results["summary"]["end_time"] = datetime.now(
            timezone.utc
        ).isoformat()
        self.results["summary"]["duration"] = end_time - start_time

        # Calculate summary statistics
        for suite_name, suite_results in self.results["test_suites"].items():
            if suite_results["status"] == "passed":
                self.results["summary"]["passed_tests"] += 1
            else:
                self.results["summary"]["failed_tests"] += 1

            self.results["summary"]["total_tests"] += 1

            # Collect errors
            if suite_results["errors"]:
                self.results["errors"].extend(
                    [
                        f"{suite_name}: {error}"
                        for error in suite_results["errors"]
                    ]
                )

        # Generate recommendations
        self._generate_recommendations()

        return self.results

    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []

        # Security recommendations
        if "security" in self.results["test_suites"]:
            security_results = self.results["test_suites"]["security"]
            if security_results["status"] == "failed":
                recommendations.append(
                    "CRITICAL: Address security vulnerabilities immediately"
                )
            elif "security_report" in security_results:
                score = security_results["security_report"]["summary"][
                    "security_score"
                ]
                if score < 95:
                    recommendations.append(
                        "Improve security posture - current score below 95%"
                    )

        # Integration recommendations
        if "integration" in self.results["test_suites"]:
            integration_results = self.results["test_suites"]["integration"]
            if integration_results["status"] == "failed":
                recommendations.append("Fix authentication flow issues")
            elif "flow_report" in integration_results:
                success_rate = integration_results["flow_report"]["summary"][
                    "success_rate"
                ]
                if success_rate < 95:
                    recommendations.append(
                        "Improve authentication flow reliability"
                    )

        # Performance recommendations
        if "performance" in self.results["test_suites"]:
            performance_results = self.results["test_suites"]["performance"]
            if performance_results["status"] == "failed":
                recommendations.append("Address performance bottlenecks")
            elif "performance_report" in performance_results:
                perf_report = performance_results["performance_report"]
                if perf_report["response_times"]["avg"] > 0.5:
                    recommendations.append(
                        "Optimize response times - average too high"
                    )
                if perf_report["summary"]["success_rate"] < 95:
                    recommendations.append(
                        "Improve system stability under load"
                    )

        # Unit test recommendations
        if "unit" in self.results["test_suites"]:
            unit_results = self.results["test_suites"]["unit"]
            if unit_results["status"] == "failed":
                recommendations.append("Fix edge case handling issues")

        # General recommendations
        if self.results["summary"]["failed_tests"] > 0:
            recommendations.append(
                "Review and fix all failing tests before production deployment"
            )

        if not recommendations:
            recommendations.append(
                "All tests passed - system ready for production"
            )

        self.results["recommendations"] = recommendations

    def print_summary(self):
        """Print test summary."""
        print("\\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        summary = self.results["summary"]
        print(f"Total Test Suites: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Duration: {summary['duration']:.1f} seconds")

        if summary["failed_tests"] == 0:
            print("\\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\\n⚠️  {summary['failed_tests']} TEST SUITE(S) FAILED")

        if self.results["errors"]:
            print("\\nERRORS:")
            for error in self.results["errors"]:
                print(f"  - {error}")

        if self.results["recommendations"]:
            print("\\nRECOMMENDATIONS:")
            for rec in self.results["recommendations"]:
                print(f"  - {rec}")

    def save_report(self, filename: str = "auth_test_report.json"):
        """Save test report to file."""
        report_path = Path(filename)
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\\nTest report saved to: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Authentication Test Suite Runner"
    )
    parser.add_argument(
        "--security-only", action="store_true", help="Run only security tests"
    )
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests",
    )
    parser.add_argument(
        "--unit-only", action="store_true", help="Run only unit tests"
    )
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Run only performance tests",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick tests only"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument("--report", type=str, help="Save report to file")

    args = parser.parse_args()

    # Create test runner
    runner = ComprehensiveAuthTestRunner()

    # Run tests
    try:
        results = runner.run_comprehensive_tests(
            security_only=args.security_only,
            integration_only=args.integration_only,
            unit_only=args.unit_only,
            performance_only=args.performance_only,
            quick=args.quick,
            verbose=args.verbose,
        )

        # Print summary
        runner.print_summary()

        # Save report if requested
        if args.report:
            runner.save_report(args.report)

        # Exit with appropriate code
        if results["summary"]["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\\nTest execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\nTest execution failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

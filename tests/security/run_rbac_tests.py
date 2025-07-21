#!/usr/bin/env python3
"""
RBAC Authorization Matrix Test Runner.

This script runs comprehensive RBAC authorization tests and generates
a detailed report of the test results, including security findings
and recommendations.

Usage:
    python run_rbac_tests.py [--verbose] [--report-file=PATH]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional


class RBACTestRunner:
    """Test runner for RBAC authorization matrix tests."""

    def __init__(self, verbose: bool = False, report_file: Optional[str] = None):
        self.verbose = verbose
        self.report_file = report_file or "rbac_test_report.json"
        self.test_results = {}
        self.security_findings = []
        self.performance_metrics = {}

    def run_rbac_tests(self) -> Dict:
        """Run all RBAC authorization tests."""
        print("🔒 Starting RBAC Authorization Matrix Tests...")
        print("=" * 60)

        # Test files to run
        test_files = [
            "tests/integration/test_rbac_permissions_scale.py",
            "tests/security/test_rbac_authorization_matrix.py",
            "tests/integration/test_authentication_flow.py",
        ]

        # Run each test file
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"📋 Running {test_file}...")
                result = self._run_test_file(test_file)
                self.test_results[test_file] = result
            else:
                print(f"⚠️  Test file not found: {test_file}")
                self.test_results[test_file] = {"status": "file_not_found"}

        # Generate report
        report = self._generate_report()
        self._save_report(report)

        return report

    def _run_test_file(self, test_file: str) -> Dict:
        """Run a specific test file."""
        start_time = time.time()

        try:
            # Run pytest with specific options
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--disable-warnings",
            ]

            if self.verbose:
                print(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            end_time = time.time()
            execution_time = end_time - start_time

            # Extract test results
            test_result = {
                "status": "passed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            if result.returncode == 0:
                print(f"✅ {test_file} passed ({execution_time:.2f}s)")
            else:
                print(f"❌ {test_file} failed ({execution_time:.2f}s)")
                if self.verbose:
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")

            return test_result

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} timed out after 5 minutes")
            return {
                "status": "timeout",
                "execution_time": 300,
                "error": "Test execution timed out",
            }
        except Exception as e:
            print(f"💥 Error running {test_file}: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_report(self) -> Dict:
        """Generate comprehensive test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._generate_summary(),
            "test_results": self.test_results,
            "security_findings": self._analyze_security_findings(),
            "performance_metrics": self._calculate_performance_metrics(),
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_summary(self) -> Dict:
        """Generate test summary."""
        total_tests = len(self.test_results)
        passed_tests = len(
            [r for r in self.test_results.values() if r.get("status") == "passed"]
        )
        failed_tests = len(
            [r for r in self.test_results.values() if r.get("status") == "failed"]
        )

        summary = {
            "total_test_files": total_tests,
            "passed_files": passed_tests,
            "failed_files": failed_tests,
            "success_rate": (passed_tests / total_tests * 100)
            if total_tests > 0
            else 0,
            "total_execution_time": sum(
                r.get("execution_time", 0) for r in self.test_results.values()
            ),
        }

        return summary

    def _analyze_security_findings(self) -> List[Dict]:
        """Analyze security findings from test results."""
        findings = []

        for test_file, result in self.test_results.items():
            if result.get("status") == "failed":
                # Analyze failure patterns for security issues
                stderr = result.get("stderr", "")
                stdout = result.get("stdout", "")

                # Look for specific security-related failures
                if "403" in stderr or "403" in stdout:
                    findings.append(
                        {
                            "type": "authorization_failure",
                            "severity": "high",
                            "test_file": test_file,
                            "description": "Authorization tests failed - possible security vulnerability",
                            "details": stderr,
                        }
                    )

                if "401" in stderr or "401" in stdout:
                    findings.append(
                        {
                            "type": "authentication_failure",
                            "severity": "high",
                            "test_file": test_file,
                            "description": "Authentication tests failed - possible security vulnerability",
                            "details": stderr,
                        }
                    )

                if (
                    "sql injection" in stderr.lower()
                    or "sql injection" in stdout.lower()
                ):
                    findings.append(
                        {
                            "type": "sql_injection_vulnerability",
                            "severity": "critical",
                            "test_file": test_file,
                            "description": "SQL injection vulnerability detected",
                            "details": stderr,
                        }
                    )

                if (
                    "privilege escalation" in stderr.lower()
                    or "privilege escalation" in stdout.lower()
                ):
                    findings.append(
                        {
                            "type": "privilege_escalation",
                            "severity": "critical",
                            "test_file": test_file,
                            "description": "Privilege escalation vulnerability detected",
                            "details": stderr,
                        }
                    )

        return findings

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        execution_times = [
            r.get("execution_time", 0) for r in self.test_results.values()
        ]

        if not execution_times:
            return {}

        metrics = {
            "average_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "total_execution_time": sum(execution_times),
        }

        return metrics

    def _generate_recommendations(self) -> List[Dict]:
        """Generate security recommendations based on test results."""
        recommendations = []

        # Check for failed tests
        failed_tests = [
            f for f, r in self.test_results.items() if r.get("status") == "failed"
        ]
        if failed_tests:
            recommendations.append(
                {
                    "type": "test_failures",
                    "priority": "high",
                    "title": "Fix failing RBAC tests",
                    "description": f"There are {len(failed_tests)} failing test files that need attention",
                    "action": "Review and fix failing tests to ensure proper authorization",
                }
            )

        # Check for security findings
        critical_findings = [
            f for f in self.security_findings if f.get("severity") == "critical"
        ]
        if critical_findings:
            recommendations.append(
                {
                    "type": "critical_security",
                    "priority": "critical",
                    "title": "Address critical security vulnerabilities",
                    "description": f"Found {len(critical_findings)} critical security issues",
                    "action": "Immediately address critical security vulnerabilities",
                }
            )

        # Performance recommendations
        avg_time = self.performance_metrics.get("average_execution_time", 0)
        if avg_time > 30:  # 30 seconds
            recommendations.append(
                {
                    "type": "performance",
                    "priority": "medium",
                    "title": "Optimize test performance",
                    "description": f"Tests are taking {avg_time:.2f}s on average",
                    "action": "Optimize test execution time for better CI/CD performance",
                }
            )

        # General recommendations
        recommendations.extend(
            [
                {
                    "type": "security_best_practices",
                    "priority": "medium",
                    "title": "Implement resource-level authorization",
                    "description": "Current implementation allows cross-user access to resources",
                    "action": "Implement resource ownership and tenant isolation",
                },
                {
                    "type": "monitoring",
                    "priority": "medium",
                    "title": "Add authorization monitoring",
                    "description": "Monitor authorization patterns for anomalies",
                    "action": "Implement real-time authorization monitoring and alerting",
                },
                {
                    "type": "testing",
                    "priority": "low",
                    "title": "Expand test coverage",
                    "description": "Add more edge cases and attack scenario tests",
                    "action": "Continuously expand RBAC test coverage",
                },
            ]
        )

        return recommendations

    def _save_report(self, report: Dict):
        """Save test report to file."""
        try:
            with open(self.report_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"📊 Test report saved to: {self.report_file}")
        except Exception as e:
            print(f"⚠️  Could not save report: {e}")

    def print_summary(self):
        """Print test summary to console."""
        if not self.test_results:
            print("❌ No test results to summarize")
            return

        print("\n🔒 RBAC Authorization Matrix Test Summary")
        print("=" * 60)

        summary = self._generate_summary()

        print(f"📋 Total test files: {summary['total_test_files']}")
        print(f"✅ Passed: {summary['passed_files']}")
        print(f"❌ Failed: {summary['failed_files']}")
        print(f"📊 Success rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Total execution time: {summary['total_execution_time']:.2f}s")

        # Security findings
        if self.security_findings:
            print(f"\n🚨 Security Findings: {len(self.security_findings)}")
            for finding in self.security_findings:
                severity_icon = "🔴" if finding["severity"] == "critical" else "🟡"
                print(f"  {severity_icon} {finding['type']}: {finding['description']}")

        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            print(f"\n💡 Recommendations: {len(recommendations)}")
            for rec in recommendations[:5]:  # Show top 5
                priority_icon = (
                    "🔴"
                    if rec["priority"] == "critical"
                    else "🟡"
                    if rec["priority"] == "high"
                    else "🟢"
                )
                print(f"  {priority_icon} {rec['title']}")

        print("\n" + "=" * 60)
        print("🔒 RBAC Authorization Matrix Testing Complete")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run RBAC Authorization Matrix Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--report-file",
        "-r",
        help="Report file path",
        default="rbac_test_report.json",
    )

    args = parser.parse_args()

    # Create test runner
    runner = RBACTestRunner(verbose=args.verbose, report_file=args.report_file)

    # Run tests
    runner.run_rbac_tests()

    # Print summary
    runner.print_summary()

    # Exit with appropriate code
    failed_tests = len(
        [r for r in runner.test_results.values() if r.get("status") == "failed"]
    )
    sys.exit(1 if failed_tests > 0 else 0)


if __name__ == "__main__":
    main()

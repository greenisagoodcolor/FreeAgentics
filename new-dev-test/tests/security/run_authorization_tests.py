#!/usr/bin/env python3
"""
Authorization Security Test Runner.

This script runs all authorization boundary tests and generates a comprehensive
security report for production validation.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class AuthorizationTestRunner:
    """Manages execution of authorization security tests."""

    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent.parent
        self.results = {
            "metadata": {
                "test_run_id": f"auth_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "start_time": datetime.now().isoformat(),
                "environment": os.getenv("ENVIRONMENT", "test"),
                "python_version": sys.version,
            },
            "test_suites": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "warnings": [],
                "critical_issues": [],
            },
        }

    def run_test_suite(self, test_file: str, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite and capture results."""
        print(f"\n{'=' * 60}")
        print(f"Running {suite_name}")
        print(f"{'=' * 60}")

        start_time = time.time()

        # Run pytest with JSON report
        json_report = f"/tmp/{suite_name}_report.json"
        cmd = [
            "pytest",
            str(self.test_dir / test_file),
            "-v",
            "--json-report",
            f"--json-report-file={json_report}",
            "--tb=short",
            "-x",  # Stop on first failure for security tests
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))

            # Parse results
            suite_results = {
                "name": suite_name,
                "file": test_file,
                "duration": time.time() - start_time,
                "exit_code": result.exit_code,
                "tests": [],
            }

            # Load JSON report if available
            if Path(json_report).exists():
                with open(json_report, "r") as f:
                    pytest_report = json.load(f)

                    suite_results["summary"] = pytest_report.get("summary", {})

                    # Extract test details
                    for test in pytest_report.get("tests", []):
                        test_info = {
                            "name": test["nodeid"],
                            "outcome": test["outcome"],
                            "duration": test.get("duration", 0),
                            "error": (
                                test.get("call", {}).get("longrepr")
                                if test["outcome"] == "failed"
                                else None
                            ),
                        }
                        suite_results["tests"].append(test_info)

                        # Update summary
                        self.results["summary"]["total_tests"] += 1
                        if test["outcome"] == "passed":
                            self.results["summary"]["passed"] += 1
                        elif test["outcome"] == "failed":
                            self.results["summary"]["failed"] += 1
                            self._analyze_failure(test_info)
                        elif test["outcome"] == "skipped":
                            self.results["summary"]["skipped"] += 1

                # Clean up
                os.remove(json_report)
            else:
                # Fallback parsing from stdout
                suite_results["stdout"] = result.stdout
                suite_results["stderr"] = result.stderr

                # Basic parsing
                if "passed" in result.stdout:
                    import re

                    match = re.search(r"(\d+) passed", result.stdout)
                    if match:
                        passed_count = int(match.group(1))
                        self.results["summary"]["passed"] += passed_count
                        self.results["summary"]["total_tests"] += passed_count

            return suite_results

        except Exception as e:
            return {
                "name": suite_name,
                "file": test_file,
                "error": str(e),
                "duration": time.time() - start_time,
            }

    def _analyze_failure(self, test_info: Dict[str, Any]):
        """Analyze test failure for security implications."""
        test_name = test_info["name"].lower()
        error = str(test_info.get("error", "")).lower()

        # Check for critical security test failures
        critical_patterns = [
            (
                "privilege_escalation",
                "CRITICAL: Privilege escalation vulnerability detected",
            ),
            ("idor", "CRITICAL: IDOR vulnerability detected"),
            (
                "authorization_bypass",
                "CRITICAL: Authorization bypass detected",
            ),
            (
                "token_manipulation",
                "CRITICAL: Token manipulation vulnerability",
            ),
            ("injection", "CRITICAL: Injection vulnerability detected"),
            ("race_condition", "HIGH: Race condition in authorization"),
        ]

        for pattern, message in critical_patterns:
            if pattern in test_name or pattern in error:
                self.results["summary"]["critical_issues"].append(
                    {
                        "test": test_info["name"],
                        "issue": message,
                        "severity": "CRITICAL" if "CRITICAL" in message else "HIGH",
                    }
                )
                break

    def run_all_tests(self):
        """Run all authorization test suites."""
        test_suites = [
            ("test_authorization_boundaries.py", "Authorization Boundaries"),
            ("test_authorization_attacks.py", "Authorization Attack Vectors"),
            ("test_authorization_integration.py", "Authorization Integration"),
            ("test_rbac_authorization_matrix.py", "RBAC Authorization Matrix"),
        ]

        print("\n" + "=" * 60)
        print("FreeAgentics Authorization Security Test Suite")
        print("=" * 60)
        print(f"Test Run ID: {self.results['metadata']['test_run_id']}")
        print(f"Start Time: {self.results['metadata']['start_time']}")

        for test_file, suite_name in test_suites:
            if Path(self.test_dir / test_file).exists():
                suite_results = self.run_test_suite(test_file, suite_name)
                self.results["test_suites"][suite_name] = suite_results
            else:
                print(f"\nWARNING: Test file {test_file} not found")
                self.results["summary"]["warnings"].append(f"Test file {test_file} not found")

        self.results["metadata"]["end_time"] = datetime.now().isoformat()

    def generate_report(self):
        """Generate comprehensive security test report."""
        report_path = self.project_root / "authorization_test_report.json"

        # Add analysis
        self._add_security_analysis()

        # Save JSON report
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Generate markdown report
        self._generate_markdown_report()

        # Print summary
        self._print_summary()

        return report_path

    def _add_security_analysis(self):
        """Add security analysis to results."""
        analysis = {
            "authorization_coverage": self._calculate_coverage(),
            "vulnerability_summary": self._summarize_vulnerabilities(),
            "recommendations": self._generate_recommendations(),
        }

        self.results["security_analysis"] = analysis

    def _calculate_coverage(self) -> Dict[str, Any]:
        """Calculate test coverage for authorization scenarios."""
        total_scenarios = {
            "rbac_boundaries": [
                "role_hierarchy",
                "permission_inheritance",
                "cross_role_access",
            ],
            "resource_authorization": [
                "ownership",
                "cross_tenant",
                "hierarchy",
            ],
            "api_authorization": [
                "endpoint_permissions",
                "http_methods",
                "parameter_validation",
            ],
            "advanced_scenarios": [
                "abac",
                "context_aware",
                "time_based",
                "location_based",
            ],
            "attack_vectors": [
                "idor",
                "privilege_escalation",
                "bypass_techniques",
                "token_attacks",
            ],
        }

        covered = 0
        total = 0

        for category, scenarios in total_scenarios.items():
            total += len(scenarios)
            # Check which scenarios were tested
            for suite_results in self.results["test_suites"].values():
                for test in suite_results.get("tests", []):
                    test_name = test.get("name", "").lower()
                    for scenario in scenarios:
                        if scenario.replace("_", " ") in test_name or scenario in test_name:
                            covered += 1
                            break

        return {
            "total_scenarios": total,
            "covered_scenarios": covered,
            "coverage_percentage": (covered / total * 100) if total > 0 else 0,
        }

    def _summarize_vulnerabilities(self) -> Dict[str, List[str]]:
        """Summarize detected vulnerabilities by category."""
        vulnerabilities = {"critical": [], "high": [], "medium": [], "low": []}

        for issue in self.results["summary"]["critical_issues"]:
            severity = issue["severity"].lower()
            if severity in vulnerabilities:
                vulnerabilities[severity].append(issue["issue"])

        return vulnerabilities

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []

        # Check for failed tests
        if self.results["summary"]["failed"] > 0:
            recommendations.append(
                "CRITICAL: Fix all failed authorization tests before production deployment"
            )

        # Check for specific vulnerabilities
        critical_issues = self.results["summary"]["critical_issues"]
        if any("privilege_escalation" in issue["issue"].lower() for issue in critical_issues):
            recommendations.append("Implement additional privilege escalation prevention measures")

        if any("idor" in issue["issue"].lower() for issue in critical_issues):
            recommendations.append("Review and strengthen object-level authorization checks")

        # Check coverage
        coverage = self._calculate_coverage()
        if coverage["coverage_percentage"] < 80:
            recommendations.append(
                f"Increase authorization test coverage (currently {coverage['coverage_percentage']:.1f}%)"
            )

        # General recommendations
        recommendations.extend(
            [
                "Enable comprehensive audit logging for all authorization decisions",
                "Implement rate limiting on authorization failures",
                "Regular security reviews of RBAC and ABAC policies",
                "Monitor for unusual authorization patterns in production",
            ]
        )

        return recommendations

    def _generate_markdown_report(self):
        """Generate human-readable markdown report."""
        report_path = self.project_root / "AUTHORIZATION_TEST_REPORT.md"

        with open(report_path, "w") as f:
            f.write("# FreeAgentics Authorization Security Test Report\n\n")
            f.write(f"**Test Run ID:** {self.results['metadata']['test_run_id']}\n")
            f.write(f"**Date:** {self.results['metadata']['start_time']}\n")
            f.write(f"**Environment:** {self.results['metadata']['environment']}\n\n")

            # Summary
            f.write("## Summary\n\n")
            summary = self.results["summary"]
            f.write(f"- **Total Tests:** {summary['total_tests']}\n")
            f.write(f"- **Passed:** {summary['passed']} ✅\n")
            f.write(f"- **Failed:** {summary['failed']} ❌\n")
            f.write(f"- **Skipped:** {summary['skipped']} ⏭️\n\n")

            # Critical Issues
            if summary["critical_issues"]:
                f.write("## 🚨 Critical Security Issues\n\n")
                for issue in summary["critical_issues"]:
                    f.write(f"- **{issue['severity']}**: {issue['issue']}\n")
                    f.write(f"  - Test: `{issue['test']}`\n")
                f.write("\n")

            # Test Suite Results
            f.write("## Test Suite Results\n\n")
            for suite_name, suite_results in self.results["test_suites"].items():
                f.write(f"### {suite_name}\n\n")

                if "summary" in suite_results:
                    suite_summary = suite_results["summary"]
                    f.write(f"- Duration: {suite_results['duration']:.2f}s\n")
                    f.write(f"- Total: {suite_summary.get('total', 0)}\n")
                    f.write(f"- Passed: {suite_summary.get('passed', 0)}\n")
                    f.write(f"- Failed: {suite_summary.get('failed', 0)}\n\n")

                # Failed tests details
                failed_tests = [
                    t for t in suite_results.get("tests", []) if t["outcome"] == "failed"
                ]
                if failed_tests:
                    f.write("**Failed Tests:**\n")
                    for test in failed_tests:
                        f.write(f"- `{test['name']}`\n")
                    f.write("\n")

            # Security Analysis
            if "security_analysis" in self.results:
                f.write("## Security Analysis\n\n")

                analysis = self.results["security_analysis"]

                # Coverage
                coverage = analysis["authorization_coverage"]
                f.write("### Test Coverage\n\n")
                f.write(
                    f"- Scenarios Covered: {coverage['covered_scenarios']}/{coverage['total_scenarios']}\n"
                )
                f.write(f"- Coverage Percentage: {coverage['coverage_percentage']:.1f}%\n\n")

                # Vulnerabilities
                f.write("### Vulnerability Summary\n\n")
                vulns = analysis["vulnerability_summary"]
                for severity, issues in vulns.items():
                    if issues:
                        f.write(f"**{severity.upper()}:**\n")
                        for issue in issues:
                            f.write(f"- {issue}\n")
                        f.write("\n")

                # Recommendations
                f.write("### Recommendations\n\n")
                for rec in analysis["recommendations"]:
                    f.write(f"- {rec}\n")

            f.write("\n---\n")
            f.write(f"*Report generated at {datetime.now().isoformat()}*\n")

    def _print_summary(self):
        """Print test summary to console."""
        print("\n" + "=" * 60)
        print("AUTHORIZATION TEST SUMMARY")
        print("=" * 60)

        summary = self.results["summary"]
        total = summary["total_tests"]
        passed = summary["passed"]
        failed = summary["failed"]

        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed / total * 100:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed: {failed}")
        print(f"Skipped: {summary['skipped']}")

        if summary["critical_issues"]:
            print("\n⚠️  CRITICAL SECURITY ISSUES DETECTED:")
            for issue in summary["critical_issues"]:
                print(f"  - {issue['severity']}: {issue['issue']}")

        if failed > 0:
            print("\n❌ AUTHORIZATION TESTS FAILED - DO NOT DEPLOY TO PRODUCTION")
            sys.exit(1)
        else:
            print("\n✅ All authorization tests passed")

        print("\nFull report saved to: authorization_test_report.json")
        print("Markdown report saved to: AUTHORIZATION_TEST_REPORT.md")


def main():
    """Main entry point."""
    runner = AuthorizationTestRunner()

    try:
        # Run all tests
        runner.run_all_tests()

        # Generate report
        report_path = runner.generate_report()

        print("\n✅ Authorization security testing completed")
        print(f"📊 Report generated: {report_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

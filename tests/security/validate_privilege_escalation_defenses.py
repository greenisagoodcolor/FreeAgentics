#!/usr/bin/env python
"""
Comprehensive validation script for privilege escalation defenses.

This script runs all privilege escalation tests and generates a detailed
security assessment report.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"{title.center(80)}")
    print(f"{'=' * 80}\n")


def run_test_suite(test_file, test_name):
    """Run a specific test suite and return results."""
    print(f"Running {test_name}...")

    cmd = [
        "python",
        "-m",
        "pytest",
        test_file,
        "-v",
        "--tb=short",
        "--json-report",
        f"--json-report-file={test_name}_report.json",
        "-x",  # Stop on first failure for security tests
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse results
    report_file = f"{test_name}_report.json"
    if Path(report_file).exists():
        with open(report_file, "r") as f:
            report = json.load(f)

        # Clean up
        Path(report_file).unlink()

        return {
            "success": result.returncode == 0,
            "summary": report.get("summary", {}),
            "duration": report.get("duration", 0),
            "tests": report.get("tests", []),
        }

    return {
        "success": False,
        "error": "No report generated",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def analyze_security_posture(results):
    """Analyze overall security posture based on test results."""
    total_tests = sum(
        r.get("summary", {}).get("total", 0) for r in results.values()
    )
    total_passed = sum(
        r.get("summary", {}).get("passed", 0) for r in results.values()
    )
    total_failed = sum(
        r.get("summary", {}).get("failed", 0) for r in results.values()
    )

    if total_tests == 0:
        return "UNKNOWN", "No tests were run"

    success_rate = (total_passed / total_tests) * 100

    # Categorize security posture
    if success_rate == 100:
        return (
            "EXCELLENT",
            f"All {total_tests} privilege escalation tests passed",
        )
    elif success_rate >= 95:
        return (
            "GOOD",
            f"{total_failed} potential vulnerabilities found in {total_tests} tests",
        )
    elif success_rate >= 80:
        return (
            "MODERATE",
            f"{total_failed} vulnerabilities found in {total_tests} tests",
        )
    elif success_rate >= 60:
        return (
            "POOR",
            f"{total_failed} significant vulnerabilities found in {total_tests} tests",
        )
    else:
        return (
            "CRITICAL",
            f"{total_failed} critical vulnerabilities found in {total_tests} tests",
        )


def generate_recommendations(results):
    """Generate security recommendations based on test results."""
    recommendations = []

    for suite_name, suite_results in results.items():
        if not suite_results.get("success", True):
            # Analyze failures
            failed_tests = [
                test
                for test in suite_results.get("tests", [])
                if test.get("outcome") == "failed"
            ]

            for test in failed_tests:
                test_name = test.get("nodeid", "").split("::")[-1]

                # Map test failures to recommendations
                if "role_elevation" in test_name:
                    recommendations.append(
                        "- Strengthen role validation in user registration and profile updates"
                    )
                elif "jwt" in test_name or "token" in test_name:
                    recommendations.append(
                        "- Implement stronger JWT validation and consider token binding"
                    )
                elif "cross_user" in test_name or "horizontal" in test_name:
                    recommendations.append(
                        "- Enhance resource ownership checks and user isolation"
                    )
                elif "sql_injection" in test_name:
                    recommendations.append(
                        "- Implement parameterized queries and input sanitization"
                    )
                elif "api" in test_name or "endpoint" in test_name:
                    recommendations.append(
                        "- Add stricter API endpoint validation and method restrictions"
                    )

    # Remove duplicates
    recommendations = list(set(recommendations))

    # Add general recommendations
    if len(recommendations) == 0:
        recommendations.append(
            "- Continue regular security testing and monitoring"
        )
        recommendations.append(
            "- Implement security headers and rate limiting"
        )
        recommendations.append(
            "- Regular security training for development team"
        )

    return recommendations


def main():
    """Run comprehensive privilege escalation defense validation."""
    print_section("PRIVILEGE ESCALATION DEFENSE VALIDATION")
    print(f"Started at: {datetime.now().isoformat()}")

    # Test suites to run
    test_suites = [
        (
            "tests/security/test_privilege_escalation_comprehensive.py",
            "comprehensive_escalation",
        ),
        (
            "tests/security/test_privilege_escalation_integration.py",
            "integration_escalation",
        ),
        (
            "tests/security/test_jwt_manipulation_vulnerabilities.py",
            "jwt_manipulation",
        ),
        (
            "tests/security/test_authorization_attacks.py",
            "authorization_attacks",
        ),
    ]

    results = {}

    # Run each test suite
    for test_file, suite_name in test_suites:
        if Path(test_file).exists():
            results[suite_name] = run_test_suite(test_file, suite_name)
        else:
            print(f"Warning: {test_file} not found, skipping...")
            results[suite_name] = {"success": True, "skipped": True}

    # Analyze results
    print_section("SECURITY ASSESSMENT RESULTS")

    posture, description = analyze_security_posture(results)

    print(f"Security Posture: {posture}")
    print(f"Assessment: {description}")

    # Detailed results
    print("\nDetailed Results by Category:")
    for suite_name, suite_results in results.items():
        if suite_results.get("skipped"):
            print(f"\n  {suite_name}: SKIPPED")
            continue

        summary = suite_results.get("summary", {})
        print(f"\n  {suite_name}:")
        print(f"    Total Tests: {summary.get('total', 0)}")
        print(f"    Passed: {summary.get('passed', 0)}")
        print(f"    Failed: {summary.get('failed', 0)}")
        print(f"    Duration: {suite_results.get('duration', 0):.2f}s")

    # Generate recommendations
    print_section("SECURITY RECOMMENDATIONS")

    recommendations = generate_recommendations(results)
    if recommendations:
        print("Based on the test results, we recommend:")
        for rec in recommendations:
            print(rec)

    # Critical failures
    critical_failures = []
    for suite_name, suite_results in results.items():
        for test in suite_results.get("tests", []):
            if test.get("outcome") == "failed":
                test_name = test.get("nodeid", "").split("::")[-1]
                if any(
                    critical in test_name
                    for critical in [
                        "admin",
                        "elevation",
                        "bypass",
                        "injection",
                    ]
                ):
                    critical_failures.append(f"{suite_name}::{test_name}")

    if critical_failures:
        print_section("CRITICAL SECURITY FAILURES")
        print("The following critical security tests failed:")
        for failure in critical_failures[:10]:  # Show top 10
            print(f"  - {failure}")
        if len(critical_failures) > 10:
            print(f"  ... and {len(critical_failures) - 10} more")

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "security_posture": posture,
        "assessment": description,
        "test_results": results,
        "recommendations": recommendations,
        "critical_failures": critical_failures,
    }

    report_path = f"privilege_escalation_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_path}")

    # Exit code based on security posture
    if posture in ["EXCELLENT", "GOOD"]:
        print("\n✅ Security validation PASSED")
        return 0
    else:
        print("\n❌ Security validation FAILED - vulnerabilities detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())

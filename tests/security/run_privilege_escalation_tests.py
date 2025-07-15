#!/usr/bin/env python
"""
Runner script for comprehensive privilege escalation tests.

This script executes all privilege escalation tests with detailed reporting
and validates that the system properly defends against all escalation attempts.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_privilege_escalation_tests():
    """Run all privilege escalation tests and generate report."""
    print("=" * 80)
    print("PRIVILEGE ESCALATION SECURITY TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    test_categories = [
        {
            "name": "Vertical Privilege Escalation",
            "class": "TestVerticalPrivilegeEscalation",
            "description": "Tests role elevation and administrative access attempts",
        },
        {
            "name": "Horizontal Privilege Escalation",
            "class": "TestHorizontalPrivilegeEscalation",
            "description": "Tests cross-user data access and session hijacking",
        },
        {
            "name": "Token-Based Escalation",
            "class": "TestTokenBasedEscalation",
            "description": "Tests JWT manipulation and token abuse",
        },
        {
            "name": "API Endpoint Escalation",
            "class": "TestAPIEndpointEscalation",
            "description": "Tests parameter manipulation and path traversal",
        },
        {
            "name": "Database-Level Escalation",
            "class": "TestDatabaseLevelEscalation",
            "description": "Tests SQL injection and database manipulation",
        },
        {
            "name": "Advanced Escalation Scenarios",
            "class": "TestAdvancedEscalationScenarios",
            "description": "Tests chained attacks and complex scenarios",
        },
    ]

    results = {
        "timestamp": datetime.now().isoformat(),
        "categories": {},
        "summary": {"total_tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0},
    }

    for category in test_categories:
        print(f"\n{'=' * 60}")
        print(f"Testing: {category['name']}")
        print(f"Description: {category['description']}")
        print(f"{'=' * 60}")

        # Run tests for this category
        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/security/test_privilege_escalation_comprehensive.py",
            f"-k",
            category["class"],
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file=privilege_escalation_{category['class']}_report.json",
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        # Parse results
        report_file = f"privilege_escalation_{category['class']}_report.json"
        if Path(report_file).exists():
            with open(report_file, "r") as f:
                test_report = json.load(f)

            category_results = {
                "duration": duration,
                "tests_run": test_report["summary"].get("total", 0),
                "passed": test_report["summary"].get("passed", 0),
                "failed": test_report["summary"].get("failed", 0),
                "errors": test_report["summary"].get("error", 0),
                "skipped": test_report["summary"].get("skipped", 0),
                "exit_code": result.returncode,
            }

            # Update totals
            results["summary"]["total_tests"] += category_results["tests_run"]
            results["summary"]["passed"] += category_results["passed"]
            results["summary"]["failed"] += category_results["failed"]
            results["summary"]["errors"] += category_results["errors"]
            results["summary"]["skipped"] += category_results["skipped"]

            # Extract failed test details
            if category_results["failed"] > 0 or category_results["errors"] > 0:
                category_results["failures"] = []
                for test in test_report.get("tests", []):
                    if test["outcome"] in ["failed", "error"]:
                        category_results["failures"].append(
                            {
                                "test": test["nodeid"],
                                "outcome": test["outcome"],
                                "message": test.get("call", {}).get("longrepr", ""),
                            }
                        )

            results["categories"][category["name"]] = category_results

            # Clean up report file
            Path(report_file).unlink()
        else:
            print(f"Warning: No report file generated for {category['name']}")
            results["categories"][category["name"]] = {
                "error": "No report file generated",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        # Print summary for this category
        print(f"\nResults for {category['name']}:")
        if "error" not in results["categories"][category["name"]]:
            cat_res = results["categories"][category["name"]]
            print(f"  Total Tests: {cat_res['tests_run']}")
            print(f"  Passed: {cat_res['passed']}")
            print(f"  Failed: {cat_res['failed']}")
            print(f"  Errors: {cat_res['errors']}")
            print(f"  Duration: {cat_res['duration']:.2f}s")
        else:
            print(f"  Error: {results['categories'][category['name']]['error']}")

    # Generate final report
    print(f"\n{'=' * 80}")
    print("FINAL PRIVILEGE ESCALATION TEST REPORT")
    print(f"{'=' * 80}")
    print(f"Total Tests Run: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Errors: {results['summary']['errors']}")
    print(f"Skipped: {results['summary']['skipped']}")

    # Calculate security score
    if results["summary"]["total_tests"] > 0:
        success_rate = (results["summary"]["passed"] / results["summary"]["total_tests"]) * 100
        print(f"\nSecurity Score: {success_rate:.1f}%")

        if success_rate == 100:
            print("\n✅ EXCELLENT: All privilege escalation attempts were properly blocked!")
        elif success_rate >= 95:
            print(
                "\n⚠️  GOOD: Most privilege escalation attempts blocked, but some vulnerabilities may exist."
            )
        elif success_rate >= 80:
            print("\n⚠️  CONCERNING: Several privilege escalation vulnerabilities detected.")
        else:
            print("\n❌ CRITICAL: Significant privilege escalation vulnerabilities found!")

    # Save detailed report
    report_path = (
        f"privilege_escalation_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed report saved to: {report_path}")

    # List any critical failures
    if results["summary"]["failed"] > 0 or results["summary"]["errors"] > 0:
        print("\n" + "=" * 60)
        print("CRITICAL SECURITY FAILURES DETECTED:")
        print("=" * 60)

        for category_name, category_data in results["categories"].items():
            if "failures" in category_data:
                print(f"\n{category_name}:")
                for failure in category_data["failures"]:
                    print(f"  - {failure['test']}")
                    if failure["message"]:
                        print(f"    {failure['message'][:200]}...")

    return results["summary"]["failed"] == 0 and results["summary"]["errors"] == 0


def run_quick_security_check():
    """Run a quick subset of critical privilege escalation tests."""
    print("\nRunning quick security check...")

    critical_tests = [
        "test_role_elevation_via_registration",
        "test_jwt_role_manipulation",
        "test_cross_user_data_access",
        "test_sql_injection_privilege_escalation",
        "test_administrative_function_access",
    ]

    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/security/test_privilege_escalation_comprehensive.py",
        "-k",
        " or ".join(critical_tests),
        "-v",
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


if __name__ == "__main__":
    # Check for quick mode
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_security_check()
    else:
        success = run_privilege_escalation_tests()

    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Generate comprehensive quality gate status report."""

import json
import subprocess
from datetime import datetime
from pathlib import Path


def run_command(cmd: list) -> tuple:
    """Run command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def get_flake8_status():
    """Get flake8 linting status."""
    # First get the raw output
    success, stdout, stderr = run_command(
        ["flake8", "--config=.flake8.minimal", "."]
    )
    error_count = len(
        [line for line in stdout.split("\n") if line.strip() and ":" in line]
    )

    # Then get statistics
    success2, stdout2, stderr2 = run_command(
        ["flake8", "--config=.flake8.minimal", "--statistics", "--quiet", "."]
    )

    # Count errors by type
    error_types = {}
    total_errors = error_count

    for line in stdout2.split("\n"):
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                count = int(parts[0])
                error_code = parts[1]
                error_types[error_code] = count

    return {
        "passed": total_errors == 0,
        "total_errors": total_errors,
        "error_types": error_types,
        "top_errors": sorted(
            error_types.items(), key=lambda x: x[1], reverse=True
        )[:10],
    }


def get_mypy_status():
    """Get mypy type checking status."""
    success, stdout, stderr = run_command(
        ["mypy", ".", "--ignore-missing-imports", "--no-error-summary"]
    )

    error_count = len(
        [line for line in stdout.split("\n") if "error:" in line]
    )

    # Get error categories
    error_categories = {
        "no-any-return": 0,
        "var-annotated": 0,
        "arg-type": 0,
        "assignment": 0,
        "other": 0,
    }

    for line in stdout.split("\n"):
        if "error:" in line:
            if "[no-any-return]" in line:
                error_categories["no-any-return"] += 1
            elif "[var-annotated]" in line:
                error_categories["var-annotated"] += 1
            elif "[arg-type]" in line:
                error_categories["arg-type"] += 1
            elif "[assignment]" in line:
                error_categories["assignment"] += 1
            else:
                error_categories["other"] += 1

    return {
        "passed": error_count == 0,
        "total_errors": error_count,
        "error_categories": error_categories,
    }


def get_test_status():
    """Get unit test status."""
    success, stdout, stderr = run_command(
        ["pytest", "tests/unit", "--tb=short", "-q"]
    )

    # Parse pytest output
    passed = failed = errors = 0
    for line in stdout.split("\n"):
        if "passed" in line:
            try:
                passed = int(line.split()[0])
            except:
                pass
        elif "failed" in line:
            try:
                failed = int(line.split()[0])
            except:
                pass
        elif "error" in line.lower():
            errors += 1

    # Check for collection errors
    collection_errors = stderr.count("ERROR collecting")

    return {
        "passed": failed == 0 and errors == 0 and collection_errors == 0,
        "tests_passed": passed,
        "tests_failed": failed,
        "errors": errors,
        "collection_errors": collection_errors,
    }


def get_frontend_build_status():
    """Get frontend build status."""
    # Change to web directory
    web_path = Path("web")
    if not web_path.exists():
        return {"passed": False, "error": "Web directory not found"}

    success, stdout, stderr = run_command(["npm", "run", "build"])

    return {
        "passed": success,
        "warnings": stdout.count("Warning:"),
        "errors": stderr.count("error") if stderr else 0,
    }


def generate_report():
    """Generate comprehensive quality gate report."""
    print("=" * 80)
    print("QUALITY GATE STATUS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Flake8 Linting
    print("1. LINTING STATUS (flake8)")
    print("-" * 40)
    flake8_status = get_flake8_status()
    status = "✅ PASSED" if flake8_status["passed"] else "❌ FAILED"
    print(f"Status: {status}")
    print(f"Total Errors: {flake8_status['total_errors']}")
    if flake8_status["top_errors"]:
        print("\nTop Error Types:")
        for error_code, count in flake8_status["top_errors"]:
            print(f"  - {error_code}: {count} occurrences")
    print()

    # Mypy Type Checking
    print("2. TYPE CHECKING STATUS (mypy)")
    print("-" * 40)
    mypy_status = get_mypy_status()
    status = "✅ PASSED" if mypy_status["passed"] else "❌ FAILED"
    print(f"Status: {status}")
    print(f"Total Errors: {mypy_status['total_errors']}")
    if mypy_status["error_categories"]:
        print("\nError Categories:")
        for category, count in mypy_status["error_categories"].items():
            if count > 0:
                print(f"  - {category}: {count}")
    print()

    # Unit Tests
    print("3. UNIT TEST STATUS")
    print("-" * 40)
    test_status = get_test_status()
    status = "✅ PASSED" if test_status["passed"] else "❌ FAILED"
    print(f"Status: {status}")
    print(f"Tests Passed: {test_status['tests_passed']}")
    print(f"Tests Failed: {test_status['tests_failed']}")
    print(f"Errors: {test_status['errors']}")
    print(f"Collection Errors: {test_status['collection_errors']}")
    print()

    # Frontend Build
    print("4. FRONTEND BUILD STATUS")
    print("-" * 40)
    frontend_status = get_frontend_build_status()
    status = "✅ PASSED" if frontend_status["passed"] else "❌ FAILED"
    print(f"Status: {status}")
    if "error" in frontend_status:
        print(f"Error: {frontend_status['error']}")
    else:
        print(f"Warnings: {frontend_status['warnings']}")
        print(f"Errors: {frontend_status['errors']}")
    print()

    # Overall Summary
    print("=" * 80)
    print("OVERALL QUALITY GATE STATUS")
    print("=" * 80)

    all_passed = (
        flake8_status["passed"]
        and mypy_status["passed"]
        and test_status["passed"]
        and frontend_status["passed"]
    )

    if all_passed:
        print("✅ ALL QUALITY GATES PASSED - READY FOR RELEASE")
    else:
        print("❌ QUALITY GATES FAILED - FIX REQUIRED")
        print("\nRequired Actions:")
        if not flake8_status["passed"]:
            print(f"  - Fix {flake8_status['total_errors']} linting errors")
        if not mypy_status["passed"]:
            print(
                f"  - Fix {mypy_status['total_errors']} type checking errors"
            )
        if not test_status["passed"]:
            print(
                f"  - Fix {test_status['tests_failed']} failing tests and {test_status['collection_errors']} collection errors"
            )
        if not frontend_status["passed"]:
            print("  - Fix frontend build issues")

    print("\n" + "=" * 80)

    # Save report to JSON
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "flake8": flake8_status,
        "mypy": mypy_status,
        "tests": test_status,
        "frontend": frontend_status,
        "overall_passed": all_passed,
    }

    with open("quality_gate_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nDetailed report saved to: quality_gate_report.json")


if __name__ == "__main__":
    generate_report()

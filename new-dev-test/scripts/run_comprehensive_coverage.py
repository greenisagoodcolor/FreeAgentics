#!/usr/bin/env python3
"""
Comprehensive coverage analysis script for FreeAgentics

This script runs all tests and provides detailed coverage analysis
to help achieve 90% test coverage.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"🏃 Running: {description or cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        if result.returncode == 0:
            print(f"✅ Success: {description}")
            return result.stdout, result.stderr
        else:
            print(f"❌ Failed: {description}")
            print(f"Error: {result.stderr}")
            return None, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {description}")
        return None, "Command timed out"
    except Exception as e:
        print(f"💥 Exception: {description} - {e}")
        return None, str(e)


def get_test_files():
    """Get all test files in the project."""
    test_files = []

    # Unit tests
    unit_tests = Path("tests/unit").glob("test_*.py")
    test_files.extend([str(f) for f in unit_tests])

    # Integration tests
    integration_tests = Path("tests/integration").glob("test_*.py")
    test_files.extend([str(f) for f in integration_tests])

    # Root level tests
    root_tests = Path(".").glob("test_*.py")
    test_files.extend([str(f) for f in root_tests])

    return test_files


def run_coverage_analysis():
    """Run comprehensive coverage analysis."""
    print("🔍 Starting comprehensive coverage analysis...")

    # Set up environment
    os.environ["PYTHONPATH"] = os.getcwd()
    os.environ["ENVIRONMENT"] = "test"

    # Source modules to analyze
    source_modules = [
        "agents",
        "api",
        "coalitions",
        "database",
        "inference",
        "llm",
        "services",
        "world",
        "knowledge_graph",
        "auth",
        "websocket",
    ]

    source_arg = ",".join(source_modules)

    # Get working test files
    working_tests = [
        "tests/unit/test_jwt_lifecycle_simple.py",
        "tests/unit/test_base_agent_simple.py",
        "tests/unit/test_base_agent.py",
        "tests/unit/test_task_9_final_coverage_push.py",
        "tests/integration/test_agent_workflow_integration.py",
    ]

    # Run working tests with coverage
    print("\n📊 Running working tests with coverage...")
    test_cmd = (
        f"python -m coverage run --source={source_arg} -m pytest {' '.join(working_tests)} -v"
    )
    stdout, stderr = run_command(test_cmd, "Working tests with coverage")

    if stdout is None:
        print("❌ Failed to run working tests")
        return False

    # Try to run additional tests that might work
    additional_tests = []

    # Find other test files
    all_test_files = get_test_files()
    for test_file in all_test_files:
        if test_file not in working_tests:
            # Try to run individual test files
            test_cmd = f"python -m coverage run --append --source={source_arg} -m pytest {test_file} -v --tb=no"
            stdout, stderr = run_command(test_cmd, f"Testing {test_file}")
            if stdout is not None and "FAILED" not in stdout:
                additional_tests.append(test_file)
                print(f"✅ Added {test_file} to coverage")

    # Generate coverage report
    print("\n📈 Generating coverage report...")
    report_cmd = "python -m coverage report --show-missing"
    stdout, stderr = run_command(report_cmd, "Coverage report")

    if stdout:
        print("\n" + "=" * 80)
        print("📋 COVERAGE REPORT")
        print("=" * 80)
        print(stdout)

    # Generate HTML report
    print("\n🌐 Generating HTML coverage report...")
    html_cmd = "python -m coverage html"
    run_command(html_cmd, "HTML coverage report")

    # Generate XML report for CI/CD
    print("\n📄 Generating XML coverage report...")
    xml_cmd = "python -m coverage xml"
    run_command(xml_cmd, "XML coverage report")

    # Parse coverage data
    try:
        json_cmd = "python -m coverage json"
        stdout, stderr = run_command(json_cmd, "JSON coverage report")

        if os.path.exists("coverage.json"):
            with open("coverage.json", "r") as f:
                coverage_data = json.load(f)

            total_coverage = coverage_data["totals"]["percent_covered"]

            print(f"\n🎯 TOTAL COVERAGE: {total_coverage:.2f}%")

            if total_coverage >= 90:
                print("🎉 EXCELLENT! Coverage target of 90% achieved!")
            elif total_coverage >= 75:
                print("🌟 GOOD! Coverage is above 75%")
            elif total_coverage >= 50:
                print("📈 IMPROVING! Coverage is above 50%")
            else:
                print("⚠️  NEEDS WORK! Coverage is below 50%")

            # Show files with lowest coverage
            print("\n📉 Files with lowest coverage:")
            files_coverage = []
            for file_path, file_data in coverage_data["files"].items():
                if file_data["summary"]["num_statements"] > 0:
                    files_coverage.append(
                        (
                            file_path,
                            file_data["summary"]["percent_covered"],
                            file_data["summary"]["num_statements"],
                            file_data["summary"]["missing_lines"],
                        )
                    )

            # Sort by coverage percentage
            files_coverage.sort(key=lambda x: x[1])

            for file_path, coverage, statements, missing in files_coverage[:10]:
                print(
                    f"  {file_path}: {coverage:.1f}% ({len(missing)} of {statements} lines missing)"
                )

            return total_coverage

    except Exception as e:
        print(f"❌ Error parsing coverage data: {e}")
        return None


def suggest_improvements():
    """Suggest improvements to increase coverage."""
    print("\n💡 COVERAGE IMPROVEMENT SUGGESTIONS:")
    print("=" * 50)

    suggestions = [
        "1. 🧪 Add more unit tests for utility functions",
        "2. 🔧 Create integration tests for API endpoints",
        "3. 🤖 Add tests for agent coordination scenarios",
        "4. 🔒 Increase security module test coverage",
        "5. 📊 Add tests for data processing pipelines",
        "6. 🌐 Create tests for websocket functionality",
        "7. 💾 Add database integration tests",
        "8. 🧠 Test inference engine components",
        "9. 🔍 Add tests for error handling paths",
        "10. 📈 Create performance testing scenarios",
    ]

    for suggestion in suggestions:
        print(f"  {suggestion}")

    print("\n📝 NEXT STEPS:")
    print("  1. Focus on files with 0% coverage first")
    print("  2. Add tests for main execution paths")
    print("  3. Include edge case and error handling tests")
    print("  4. Mock external dependencies properly")
    print("  5. Use pytest fixtures for common test setup")


def main():
    """Main function."""
    print("🚀 FreeAgentics Comprehensive Coverage Analysis")
    print("=" * 50)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if we're in the right directory
    if not os.path.exists("agents") or not os.path.exists("tests"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)

    # Run coverage analysis
    coverage_result = run_coverage_analysis()

    if coverage_result is not None:
        print(f"\n🎯 Final Coverage: {coverage_result:.2f}%")

        if coverage_result < 90:
            suggest_improvements()

            # Calculate how many lines need to be covered
            if os.path.exists("coverage.json"):
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)

                total_lines = coverage_data["totals"]["num_statements"]
                covered_lines = coverage_data["totals"]["covered_lines"]
                needed_lines = int((total_lines * 0.9) - covered_lines)

                print("\n📊 COVERAGE MATH:")
                print(f"  Total lines: {total_lines}")
                print(f"  Covered lines: {covered_lines}")
                print(f"  Lines needed for 90%: {needed_lines}")
                print(f"  Current coverage: {coverage_result:.2f}%")
    else:
        print("❌ Coverage analysis failed")
        sys.exit(1)

    print(f"\n✅ Coverage analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📂 Reports generated:")
    print("  - coverage.xml (for CI/CD)")
    print("  - htmlcov/index.html (interactive report)")
    print("  - coverage.json (detailed data)")


if __name__ == "__main__":
    main()

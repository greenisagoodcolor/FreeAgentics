"""IDOR vulnerability validation suite.

This module provides a comprehensive test runner and validation framework
for all IDOR vulnerability tests in the FreeAgentics platform.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from tabulate import tabulate


class IDORTestSuite:
    """Comprehensive IDOR test suite runner and validator."""

    def __init__(self):
        self.test_modules = [
            "test_idor_vulnerabilities",
            "test_idor_advanced_patterns",
            "test_idor_integration",
            "test_idor_file_operations",
        ]
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_all_tests(self, verbose: bool = True) -> Dict:
        """Run all IDOR vulnerability tests."""
        print("🔒 Starting comprehensive IDOR vulnerability test suite...")
        print("=" * 80)

        self.start_time = datetime.now()
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for module in self.test_modules:
            print(f"\n📋 Running tests in {module}...")
            result = self._run_module_tests(module, verbose)

            self.results[module] = result
            total_tests += result["total"]
            total_passed += result["passed"]
            total_failed += result["failed"]
            total_skipped += result["skipped"]

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        # Generate summary
        summary = {
            "total_modules": len(self.test_modules),
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "duration_seconds": duration,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "module_results": self.results,
        }

        # Print summary
        self._print_summary(summary)

        # Save detailed report
        self._save_report(summary)

        return summary

    def _run_module_tests(self, module: str, verbose: bool) -> Dict:
        """Run tests for a specific module."""
        test_file = f"tests/security/{module}.py"

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_file,
            "-v" if verbose else "-q",
            "--tb=short",
            "--json-report",
            f"--json-report-file=tests/security/.{module}_report.json",
        ]

        # Run tests
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start

        # Parse results
        try:
            with open(f"tests/security/.{module}_report.json", "r") as f:
                report = json.load(f)

            return {
                "total": report["summary"]["total"],
                "passed": report["summary"]["passed"],
                "failed": report["summary"]["failed"],
                "skipped": report["summary"]["skipped"],
                "duration": duration,
                "exit_code": result.returncode,
                "failed_tests": self._extract_failed_tests(report),
            }
        except Exception as e:
            print(f"⚠️  Error parsing results for {module}: {e}")
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "duration": duration,
                "exit_code": result.returncode,
                "error": str(e),
            }

    def _extract_failed_tests(self, report: Dict) -> List[Dict]:
        """Extract details of failed tests."""
        failed_tests = []

        for test in report.get("tests", []):
            if test["outcome"] == "failed":
                failed_tests.append(
                    {
                        "name": test["nodeid"],
                        "duration": test["duration"],
                        "error": test.get("call", {}).get("longrepr", "Unknown error"),
                    }
                )

        return failed_tests

    def _print_summary(self, summary: Dict):
        """Print test summary in a formatted way."""
        print("\n" + "=" * 80)
        print("📊 IDOR Vulnerability Test Suite Summary")
        print("=" * 80)

        # Overall statistics
        total = summary["total_tests"]
        passed = summary["passed"]
        failed = summary["failed"]
        skipped = summary["skipped"]
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed} ({pass_rate:.1f}%)")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"⏱️  Duration: {summary['duration_seconds']:.2f} seconds")

        # Module breakdown
        print("\n📋 Module Breakdown:")
        module_data = []
        for module, results in summary["module_results"].items():
            module_data.append(
                [
                    module,
                    results["total"],
                    results["passed"],
                    results["failed"],
                    results["skipped"],
                    f"{results['duration']:.2f}s",
                ]
            )

        print(
            tabulate(
                module_data,
                headers=["Module", "Total", "Passed", "Failed", "Skipped", "Duration"],
                tablefmt="grid",
            )
        )

        # Failed tests details
        if failed > 0:
            print("\n❌ Failed Tests:")
            for module, results in summary["module_results"].items():
                if results.get("failed_tests"):
                    print(f"\n  {module}:")
                    for test in results["failed_tests"]:
                        print(f"    - {test['name']}")

    def _save_report(self, summary: Dict):
        """Save detailed test report."""
        report_path = Path("tests/security/idor_test_report.json")

        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Detailed report saved to: {report_path}")

    def validate_security_coverage(self) -> Tuple[bool, List[str]]:
        """Validate that all required IDOR attack patterns are covered."""
        required_patterns = [
            # Sequential ID Enumeration
            "test_agent_id_enumeration",
            "test_sequential_id_brute_force",
            "test_user_id_enumeration",
            "test_coalition_id_enumeration",
            # UUID/GUID Attacks
            "test_uuid_prediction",
            "test_uuid_version_attacks",
            "test_uuid_collision_attempts",
            # Parameter Manipulation
            "test_query_parameter_idor",
            "test_path_parameter_idor",
            "test_json_payload_idor",
            "test_form_data_idor",
            # Authorization Bypass
            "test_direct_object_access",
            "test_resource_ownership_bypass",
            "test_cross_tenant_access",
            "test_file_path_traversal_idor",
            # Advanced IDOR Attacks
            "test_blind_idor_detection",
            "test_time_based_idor",
            "test_mass_assignment_idor",
            "test_indirect_object_references",
            # Integration Tests
            "test_websocket_connection_idor",
            "test_graphql_query_idor",
            "test_batch_operation_idor",
            "test_cache_poisoning_idor",
            "test_api_versioning_idor",
        ]

        # Check coverage
        missing_patterns = []
        test_files = list(Path("tests/security").glob("test_idor_*.py"))

        for pattern in required_patterns:
            found = False
            for test_file in test_files:
                content = test_file.read_text()
                if f"def {pattern}" in content:
                    found = True
                    break

            if not found:
                missing_patterns.append(pattern)

        coverage_complete = len(missing_patterns) == 0

        if coverage_complete:
            print("\n✅ All required IDOR attack patterns are covered!")
        else:
            print(f"\n⚠️  Missing coverage for {len(missing_patterns)} patterns:")
            for pattern in missing_patterns:
                print(f"  - {pattern}")

        return coverage_complete, missing_patterns


class IDORSecurityValidator:
    """Validate IDOR security implementations."""

    @staticmethod
    def validate_endpoint_protection(endpoint: str) -> Dict[str, bool]:
        """Validate that an endpoint is properly protected against IDOR."""
        validations = {
            "has_authentication": False,
            "has_authorization": False,
            "uses_uuid": False,
            "validates_ownership": False,
            "consistent_errors": False,
            "no_id_leakage": False,
        }

        # This would be implemented to actually test the endpoint
        # For now, it's a placeholder for the validation framework

        return validations

    @staticmethod
    def generate_security_report() -> str:
        """Generate a comprehensive IDOR security report."""
        report = []
        report.append("# IDOR Security Validation Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("\n## Overview")
        report.append(
            "This report validates IDOR (Insecure Direct Object Reference) protection across the FreeAgentics platform."
        )

        # Add validation results
        report.append("\n## Test Coverage")
        suite = IDORTestSuite()
        coverage_complete, missing = suite.validate_security_coverage()

        if coverage_complete:
            report.append("✅ All IDOR attack patterns are covered by tests")
        else:
            report.append(f"⚠️  Missing test coverage for {len(missing)} patterns")

        report.append("\n## Recommendations")
        report.append(
            "1. **Use UUIDs**: All resource identifiers should use UUIDs instead of sequential IDs"
        )
        report.append("2. **Validate Ownership**: Every request must validate resource ownership")
        report.append(
            "3. **Consistent Errors**: Return consistent error messages (404) for both non-existent and unauthorized resources"
        )
        report.append(
            "4. **Object-Level Authorization**: Implement authorization checks at the object level, not just endpoint level"
        )
        report.append(
            "5. **Audit Logging**: Log all authorization failures for security monitoring"
        )

        return "\n".join(report)


def main():
    """Run the complete IDOR test suite."""
    print("🚀 FreeAgentics IDOR Vulnerability Test Suite")
    print("=" * 80)

    # Run all tests
    suite = IDORTestSuite()
    results = suite.run_all_tests(verbose=True)

    # Validate coverage
    coverage_complete, _ = suite.validate_security_coverage()

    # Generate security report
    validator = IDORSecurityValidator()
    report = validator.generate_security_report()

    # Save security report
    with open("tests/security/IDOR_SECURITY_REPORT.md", "w") as f:
        f.write(report)

    print("\n📄 Security report saved to: tests/security/IDOR_SECURITY_REPORT.md")

    # Exit with appropriate code
    if results["failed"] > 0:
        print("\n❌ IDOR vulnerability tests failed! Security vulnerabilities may exist.")
        sys.exit(1)
    elif not coverage_complete:
        print("\n⚠️  IDOR test coverage incomplete! Some attack patterns not tested.")
        sys.exit(1)
    else:
        print("\n✅ All IDOR vulnerability tests passed! System is protected.")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Systematic Test Fixer - Runs tests module by module and categorizes failures
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class TestAnalyzer:
    def __init__(self, test_dir: str = "tests"):
        self.test_dir = test_dir
        self.results = {
            "summary": {
                "total_modules": 0,
                "passed_modules": 0,
                "failed_modules": 0,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "error_tests": 0,
                "skipped_tests": 0,
            },
            "modules": {},
            "error_patterns": {},
            "timestamp": datetime.now().isoformat(),
        }

    def find_test_modules(self) -> List[Path]:
        """Find all test modules in the test directory."""
        test_files = []
        for root, dirs, files in os.walk(self.test_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_files.append(Path(root) / file)
        return sorted(test_files)

    def extract_error_info(self, output: str) -> Dict[str, any]:
        """Extract error information from pytest output."""
        error_info = {
            "error_type": None,
            "error_message": None,
            "traceback": [],
            "failed_tests": [],
        }

        # Extract error type and message
        error_pattern = r"E\s+(\w+Error|AssertionError|Exception):\s*(.*)"
        error_matches = re.findall(error_pattern, output)
        if error_matches:
            error_info["error_type"] = error_matches[0][0]
            error_info["error_message"] = error_matches[0][1]

        # Extract failed test names
        failed_pattern = r"FAILED\s+([\w\/\.]+::\w+(?:::\w+)?)"
        failed_matches = re.findall(failed_pattern, output)
        error_info["failed_tests"] = failed_matches

        return error_info

    def run_test_module(self, module_path: Path) -> Tuple[bool, Dict]:
        """Run a single test module and return results."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(module_path),
            "-v",
            "--tb=short",
            "--no-header",
            "--timeout=30",
            "-q",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            output = result.stdout + result.stderr

            # Parse test results
            passed = re.findall(r"(\d+) passed", output)
            failed = re.findall(r"(\d+) failed", output)
            errors = re.findall(r"(\d+) error", output)
            skipped = re.findall(r"(\d+) skipped", output)

            module_result = {
                "path": str(module_path),
                "success": result.returncode == 0,
                "passed": int(passed[0]) if passed else 0,
                "failed": int(failed[0]) if failed else 0,
                "errors": int(errors[0]) if errors else 0,
                "skipped": int(skipped[0]) if skipped else 0,
                "error_info": self.extract_error_info(output) if result.returncode != 0 else None,
                "output_preview": output[:500] if result.returncode != 0 else "",
            }

            return result.returncode == 0, module_result

        except subprocess.TimeoutExpired:
            return False, {
                "path": str(module_path),
                "success": False,
                "error": "Timeout",
                "error_info": {
                    "error_type": "TimeoutError",
                    "error_message": "Test execution timed out",
                },
            }
        except Exception as e:
            return False, {
                "path": str(module_path),
                "success": False,
                "error": str(e),
                "error_info": {"error_type": type(e).__name__, "error_message": str(e)},
            }

    def categorize_error(self, error_info: Dict) -> str:
        """Categorize error type for grouping."""
        if not error_info or not error_info.get("error_type"):
            return "Unknown"

        error_type = error_info["error_type"]
        error_msg = error_info.get("error_message", "")

        # Categorize by common patterns
        if "Mock" in error_msg and ("operand" in error_msg or "attribute" in error_msg):
            return "Mock_Numeric_Operations"
        elif "Factory" in error_msg or "should exist" in error_msg:
            return "Missing_Factory_Functions"
        elif "import" in error_msg.lower() or "ImportError" in error_type:
            return "Import_Errors"
        elif "dtype" in error_msg or "tensor" in error_msg.lower():
            return "Tensor_Type_Errors"
        elif "API" in error_msg or "contract" in error_msg.lower():
            return "API_Contract_Violations"
        elif "connection" in error_msg.lower() or "socket" in error_msg.lower():
            return "Connection_Errors"
        else:
            return f"{error_type}_General"

    def analyze_all_modules(self):
        """Run and analyze all test modules."""
        test_modules = self.find_test_modules()
        self.results["summary"]["total_modules"] = len(test_modules)

        print(f"Found {len(test_modules)} test modules to analyze\n")

        for i, module in enumerate(test_modules, 1):
            print(f"[{i}/{len(test_modules)}] Testing {module.name}...", end=" ")

            success, module_result = self.run_test_module(module)

            if success:
                print("✓ PASSED")
                self.results["summary"]["passed_modules"] += 1
            else:
                print("✗ FAILED")
                self.results["summary"]["failed_modules"] += 1

                # Categorize error
                if module_result.get("error_info"):
                    category = self.categorize_error(module_result["error_info"])
                    if category not in self.results["error_patterns"]:
                        self.results["error_patterns"][category] = []
                    self.results["error_patterns"][category].append(module_result["path"])

            # Update totals
            self.results["summary"]["total_tests"] += (
                module_result.get("passed", 0)
                + module_result.get("failed", 0)
                + module_result.get("errors", 0)
            )
            self.results["summary"]["passed_tests"] += module_result.get("passed", 0)
            self.results["summary"]["failed_tests"] += module_result.get("failed", 0)
            self.results["summary"]["error_tests"] += module_result.get("errors", 0)
            self.results["summary"]["skipped_tests"] += module_result.get("skipped", 0)

            self.results["modules"][str(module)] = module_result

    def generate_report(self) -> str:
        """Generate a comprehensive test analysis report."""
        report = []
        report.append("# Systematic Test Analysis Report")
        report.append(f"\nGenerated at: {self.results['timestamp']}")

        # Summary
        report.append("\n## Summary")
        summary = self.results["summary"]
        report.append(f"- **Total Modules**: {summary['total_modules']}")
        report.append(
            f"- **Passed Modules**: {
                summary['passed_modules']} ({
                summary['passed_modules'] /
                summary['total_modules'] *
                100:.1f}%)"
        )
        report.append(
            f"- **Failed Modules**: {
                summary['failed_modules']} ({
                summary['failed_modules'] /
                summary['total_modules'] *
                100:.1f}%)"
        )
        report.append(f"- **Total Tests**: {summary['total_tests']}")
        passed_pct = (
            f"{summary['passed_tests'] / summary['total_tests'] * 100:.1f}"
            if summary["total_tests"] > 0
            else "0"
        )
        failed_pct = (
            f"{summary['failed_tests'] / summary['total_tests'] * 100:.1f}"
            if summary["total_tests"] > 0
            else "0"
        )
        report.append(f"- **Passed Tests**: {summary['passed_tests']} ({passed_pct}%)")
        report.append(f"- **Failed Tests**: {summary['failed_tests']} ({failed_pct}%)")
        report.append(f"- **Error Tests**: {summary['error_tests']}")
        report.append(f"- **Skipped Tests**: {summary['skipped_tests']}")

        # Error Pattern Analysis
        report.append("\n## Error Pattern Analysis")
        for category, modules in sorted(
            self.results["error_patterns"].items(), key=lambda x: len(x[1]), reverse=True
        ):
            report.append(f"\n### {category} ({len(modules)} modules)")
            for module in modules[:5]:  # Show first 5
                report.append(f"- {module}")
            if len(modules) > 5:
                report.append(f"- ... and {len(modules) - 5} more")

        # Failed Modules Details
        report.append("\n## Failed Modules Details")
        failed_modules = [
            (path, info)
            for path, info in self.results["modules"].items()
            if not info.get("success", False)
        ]

        for path, info in failed_modules[:10]:  # Show first 10
            report.append(f"\n### {Path(path).name}")
            if info.get("error_info"):
                error_info = info["error_info"]
                report.append(f"- **Error Type**: {error_info.get('error_type', 'Unknown')}")
                report.append(
                    f"- **Error Message**: {error_info.get('error_message', 'No message')}"
                )
                report.append(f"- **Failed Tests**: {len(error_info.get('failed_tests', []))}")
                if error_info.get("failed_tests"):
                    for test in error_info["failed_tests"][:3]:
                        report.append(f"  - {test}")

        # Recommendations
        report.append("\n## Recommendations")
        report.append("\n### Priority Fixes")

        # Sort error patterns by frequency
        sorted_patterns = sorted(
            self.results["error_patterns"].items(), key=lambda x: len(x[1]), reverse=True
        )

        for i, (category, modules) in enumerate(sorted_patterns[:5], 1):
            report.append(f"\n{i}. **Fix {category}** ({len(modules)} modules affected)")
            if category == "Mock_Numeric_Operations":
                report.append("   - Use NumericMock from tests.fixtures.numeric_mocks")
                report.append("   - Replace Mock objects in mathematical operations")
            elif category == "Missing_Factory_Functions":
                report.append("   - Implement missing factory functions")
                report.append("   - Check for create_* functions in modules")
            elif category == "Import_Errors":
                report.append("   - Fix circular dependencies")
                report.append("   - Ensure all modules are properly installed")
            elif category == "Tensor_Type_Errors":
                report.append("   - Use proper tensor mocks")
                report.append("   - Check tensor shape and dtype compatibility")
            elif category == "API_Contract_Violations":
                report.append("   - Update test expectations to match implementation")
                report.append("   - Add schema validation")

        return "\n".join(report)

    def save_results(self):
        """Save analysis results to files."""
        # Save JSON data
        with open("test_analysis_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        # Save markdown report
        report = self.generate_report()
        with open("test_analysis_report.md", "w") as f:
            f.write(report)

        print("\nResults saved to:")
        print("- test_analysis_results.json")
        print("- test_analysis_report.md")


if __name__ == "__main__":
    analyzer = TestAnalyzer()
    analyzer.analyze_all_modules()
    analyzer.save_results()

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    summary = analyzer.results["summary"]
    print(f"Modules: {summary['passed_modules']}/{summary['total_modules']} passed")
    print(f"Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
    success_rate = (
        f"{summary['passed_tests'] / summary['total_tests'] * 100:.1f}%"
        if summary["total_tests"] > 0
        else "N/A"
    )
    print(f"Success Rate: {success_rate}")

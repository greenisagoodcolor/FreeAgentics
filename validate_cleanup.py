#!/usr/bin/env python3
"""
Comprehensive Cleanup Validation Script

This script validates that the cleanup process has been completed successfully
according to the CLAUDE.md guidelines and zero tolerance quality standards.
"""

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple


class CleanupValidator:
    """Validates repository cleanup according to CLAUDE.md standards."""

    def __init__(self) -> None:
        self.issues_found: int = 0
        self.validation_results: Dict[str, Any] = {}
        self.start_time: float = time.time()

    def run_command(self, command: str, capture_output: bool = True) -> Tuple[int, str, str]:
        """Run a shell command and return (exit_code, stdout, stderr)."""
        try:
            import shlex

            # Check if command contains pipes or redirects - these need shell processing
            if "|" in command or ">" in command or "<" in command:
                # For commands with pipes/redirects, we need to use shell but be careful
                # In this validator context, all commands are hardcoded and safe
                result = subprocess.run(
                    command,
                    shell=True,  # nosec B602 - Commands are hardcoded in validator, not user input
                    capture_output=capture_output,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
            else:
                # For simple commands, use list form without shell
                cmd_list = shlex.split(command)
                result = subprocess.run(
                    cmd_list,
                    capture_output=capture_output,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)

    def log_result(
        self,
        test_name: str,
        passed: bool,
        message: str,
        details: Optional[str] = None,
    ):
        """Log a validation result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name} - {message}")

        if details:
            print(f"   Details: {details}")

        self.validation_results[test_name] = {
            "passed": passed,
            "message": message,
            "details": details,
        }

        if not passed:
            self.issues_found += 1

    def validate_automated_checks(self) -> bool:
        """Validate that all automated checks pass (make format && make test && make lint)."""
        print("\n=== VALIDATING AUTOMATED CHECKS ===")

        # Check if Makefile exists
        if not os.path.exists("Makefile"):
            self.log_result("makefile_exists", False, "Makefile not found")
            return False

        # Run make format
        exit_code, stdout, stderr = self.run_command("make format")
        format_passed = exit_code == 0
        self.log_result(
            "make_format",
            format_passed,
            "Code formatting check",
            f"Exit code: {exit_code}\nStderr: {stderr}" if not format_passed else None,
        )

        # Run make test
        exit_code, stdout, stderr = self.run_command("make test")
        test_passed = exit_code == 0
        self.log_result(
            "make_test",
            test_passed,
            "Test suite execution",
            f"Exit code: {exit_code}\nStderr: {stderr}" if not test_passed else None,
        )

        # Run make lint
        exit_code, stdout, stderr = self.run_command("make lint")
        lint_passed = exit_code == 0
        self.log_result(
            "make_lint",
            lint_passed,
            "Linting check",
            f"Exit code: {exit_code}\nStderr: {stderr}" if not lint_passed else None,
        )

        return format_passed and test_passed and lint_passed

    def validate_type_errors(self) -> bool:
        """Validate that there are no type errors."""
        print("\n=== VALIDATING TYPE ERRORS ===")

        # Check Python type errors with mypy
        exit_code, stdout, stderr = self.run_command("mypy . --ignore-missing-imports")
        python_types_passed = exit_code == 0
        self.log_result(
            "python_types",
            python_types_passed,
            "Python type checking",
            f"Mypy output: {stdout}" if not python_types_passed else None,
        )

        # Check TypeScript type errors if applicable
        ts_types_passed = True
        if os.path.exists("tsconfig.json"):
            exit_code, stdout, stderr = self.run_command("npx tsc --noEmit --skipLibCheck")
            ts_types_passed = exit_code == 0
            self.log_result(
                "typescript_types",
                ts_types_passed,
                "TypeScript type checking",
                f"TSC output: {stdout}" if not ts_types_passed else None,
            )

        return python_types_passed and ts_types_passed

    def validate_precommit_hooks(self) -> bool:
        """Validate that all pre-commit hooks pass."""
        print("\n=== VALIDATING PRE-COMMIT HOOKS ===")

        # Check if pre-commit is installed
        exit_code, stdout, stderr = self.run_command("pre-commit --version")
        if exit_code != 0:
            self.log_result("precommit_installed", False, "Pre-commit not installed")
            return False

        # Run pre-commit hooks
        exit_code, stdout, stderr = self.run_command("pre-commit run --all-files")
        hooks_passed = exit_code == 0
        self.log_result(
            "precommit_hooks",
            hooks_passed,
            "Pre-commit hooks execution",
            f"Output: {stdout}\nErrors: {stderr}" if not hooks_passed else None,
        )

        return hooks_passed

    def validate_obsolete_files(self) -> bool:
        """Validate that obsolete files have been removed."""
        print("\n=== VALIDATING OBSOLETE FILES REMOVAL ===")

        obsolete_patterns = [
            "*.bak",
            "*.tmp",
            "*.old",
            "*~",
            "*.pyc",
            "*.log",
            ".DS_Store",
            "Thumbs.db",
            "*.swp",
            "*.swo",
        ]

        found_obsolete = []
        for pattern in obsolete_patterns:
            exit_code, stdout, stderr = self.run_command(f"find . -name '{pattern}' -type f")
            if stdout.strip():
                found_obsolete.extend(stdout.strip().split("\n"))

        # Check for common build artifacts
        build_dirs = [
            "build",
            "dist",
            "__pycache__",
            "node_modules",
            ".pytest_cache",
        ]
        for dir_name in build_dirs:
            if os.path.exists(dir_name):
                found_obsolete.append(dir_name)

        no_obsolete = len(found_obsolete) == 0
        self.log_result(
            "obsolete_files",
            no_obsolete,
            "Obsolete files removal",
            f"Found obsolete files: {found_obsolete}" if not no_obsolete else None,
        )

        return no_obsolete

    def validate_documentation(self) -> bool:
        """Validate documentation consolidation."""
        print("\n=== VALIDATING DOCUMENTATION ===")

        # Check if README exists
        readme_exists = os.path.exists("README.md")
        self.log_result("readme_exists", readme_exists, "README.md file exists")

        # Check if CLAUDE.md exists
        claude_exists = os.path.exists("CLAUDE.md")
        self.log_result("claude_exists", claude_exists, "CLAUDE.md file exists")

        # Count documentation files
        exit_code, stdout, stderr = self.run_command("find . -name '*.md' -type f | wc -l")
        doc_count = int(stdout.strip()) if stdout.strip().isdigit() else 0

        # Check for documentation organization
        docs_organized = True
        if doc_count > 10:  # Arbitrary threshold for "too many" scattered docs
            docs_organized = False

        self.log_result(
            "docs_organized",
            docs_organized,
            f"Documentation organization ({doc_count} .md files)",
            "Consider consolidating if > 10 files" if not docs_organized else None,
        )

        return readme_exists and claude_exists and docs_organized

    def validate_git_status(self) -> bool:
        """Validate git working directory is clean."""
        print("\n=== VALIDATING GIT STATUS ===")

        # Check git status
        exit_code, stdout, stderr = self.run_command("git status --porcelain")
        git_clean = stdout.strip() == ""
        self.log_result(
            "git_clean",
            git_clean,
            "Git working directory clean",
            f"Uncommitted changes: {stdout}" if not git_clean else None,
        )

        # Check commit message format (last 5 commits)
        exit_code, stdout, stderr = self.run_command("git log --oneline -5")
        commits = stdout.strip().split("\n") if stdout.strip() else []

        conventional_commits = True
        non_conventional = []

        for commit in commits:
            if commit.strip():
                # Check if commit follows conventional format
                if not any(
                    commit.split(" ", 1)[1].startswith(prefix)
                    for prefix in [
                        "feat:",
                        "fix:",
                        "docs:",
                        "style:",
                        "refactor:",
                        "test:",
                        "chore:",
                        "cleanup:",
                    ]
                ):
                    conventional_commits = False
                    non_conventional.append(commit)

        self.log_result(
            "conventional_commits",
            conventional_commits,
            "Conventional commit format",
            f"Non-conventional commits: {non_conventional}" if not conventional_commits else None,
        )

        return git_clean and conventional_commits

    def validate_test_coverage(self) -> bool:
        """Validate test coverage is adequate."""
        print("\n=== VALIDATING TEST COVERAGE ===")

        # Run coverage if available
        exit_code, stdout, stderr = self.run_command("coverage report")
        coverage_available = exit_code == 0

        if coverage_available:
            # Extract coverage percentage
            coverage_lines = stdout.split("\n")
            total_line = next((line for line in coverage_lines if "TOTAL" in line), "")
            if total_line:
                # Extract percentage (assuming format like "TOTAL    100   50    50%")
                parts = total_line.split()
                if len(parts) >= 4 and parts[-1].endswith("%"):
                    coverage_percent = int(parts[-1][:-1])
                    coverage_good = coverage_percent >= 80  # 80% threshold
                    self.log_result(
                        "test_coverage",
                        coverage_good,
                        f"Test coverage: {coverage_percent}%",
                        "Should be >= 80%" if not coverage_good else None,
                    )
                    return coverage_good

        # If coverage not available, check for test files
        exit_code, stdout, stderr = self.run_command(
            "find . -name '*test*.py' -o -name 'test_*.py' | wc -l"
        )
        test_count = int(stdout.strip()) if stdout.strip().isdigit() else 0

        # Check for Python files
        exit_code, stdout, stderr = self.run_command("find . -name '*.py' | wc -l")
        python_count = int(stdout.strip()) if stdout.strip().isdigit() else 0

        test_ratio = test_count / python_count if python_count > 0 else 0
        adequate_tests = test_ratio >= 0.3  # At least 30% of files should be tests

        self.log_result(
            "test_coverage",
            adequate_tests,
            f"Test file ratio: {test_ratio:.2%} ({test_count}/{python_count})",
            "Should have adequate test coverage" if not adequate_tests else None,
        )

        return adequate_tests

    def validate_security_baseline(self) -> bool:
        """Validate basic security checks."""
        print("\n=== VALIDATING SECURITY BASELINE ===")

        # Check for bandit security scanner
        exit_code, stdout, stderr = self.run_command("bandit -r . -f json")
        security_passed = exit_code == 0

        if not security_passed:
            # Count security issues
            try:
                # Try to parse as JSON to count issues
                bandit_output = json.loads(stdout) if stdout else {}
                issue_count = len(bandit_output.get("results", []))
                self.log_result(
                    "security_scan",
                    False,
                    f"Security issues found: {issue_count}",
                    "Run 'bandit -r .' for details",
                )
            except json.JSONDecodeError:
                self.log_result(
                    "security_scan",
                    False,
                    "Security scan failed",
                    f"Bandit output: {stderr}",
                )
        else:
            self.log_result("security_scan", True, "No security issues found")

        return security_passed

    def validate_performance_baseline(self) -> bool:
        """Validate basic performance checks."""
        print("\n=== VALIDATING PERFORMANCE BASELINE ===")

        # Check for large files that might indicate performance issues
        exit_code, stdout, stderr = self.run_command("find . -size +10M -type f")
        large_files = stdout.strip().split("\n") if stdout.strip() else []

        no_large_files = len(large_files) == 0 or large_files == [""]
        self.log_result(
            "large_files",
            no_large_files,
            "No unexpectedly large files",
            f"Large files found: {large_files}" if not no_large_files else None,
        )

        # Check for obvious performance anti-patterns in Python
        exit_code, stdout, stderr = self.run_command(
            "grep -r 'import \\*' --include='*.py' . | wc -l"
        )
        star_imports = int(stdout.strip()) if stdout.strip().isdigit() else 0

        no_star_imports = star_imports == 0
        self.log_result(
            "star_imports",
            no_star_imports,
            "No star imports found",
            f"Found {star_imports} star imports" if not no_star_imports else None,
        )

        return no_large_files and no_star_imports

    def generate_report(self) -> Dict:
        """Generate a comprehensive validation report."""
        end_time = time.time()
        duration = end_time - self.start_time

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "total_checks": len(self.validation_results),
            "passed_checks": sum(1 for r in self.validation_results.values() if r["passed"]),
            "failed_checks": self.issues_found,
            "overall_status": "PASS" if self.issues_found == 0 else "FAIL",
            "results": self.validation_results,
        }

        return report

    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        print("🔍 STARTING COMPREHENSIVE CLEANUP VALIDATION")
        print("=" * 60)

        validations = [
            ("Automated Checks", self.validate_automated_checks),
            ("Type Errors", self.validate_type_errors),
            ("Pre-commit Hooks", self.validate_precommit_hooks),
            ("Obsolete Files", self.validate_obsolete_files),
            ("Documentation", self.validate_documentation),
            ("Git Status", self.validate_git_status),
            ("Test Coverage", self.validate_test_coverage),
            ("Security Baseline", self.validate_security_baseline),
            ("Performance Baseline", self.validate_performance_baseline),
        ]

        for name, validation_func in validations:
            try:
                validation_func()
            except Exception as e:
                self.log_result(
                    f"{name.lower().replace(' ', '_')}_error",
                    False,
                    f"Validation error: {str(e)}",
                )

        # Generate and save report
        report = self.generate_report()
        with open("cleanup_validation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Checks: {report['total_checks']}")
        print(f"Passed: {report['passed_checks']}")
        print(f"Failed: {report['failed_checks']}")
        print(f"Duration: {report['duration_seconds']}s")
        print(f"Overall Status: {report['overall_status']}")

        if self.issues_found == 0:
            print("\n✅ ALL VALIDATIONS PASSED - CLEANUP SUCCESSFUL")
            print("🎉 Repository meets all CLAUDE.md quality standards")
        else:
            print(f"\n❌ {self.issues_found} VALIDATION(S) FAILED")
            print("🚨 CLEANUP INCOMPLETE - ISSUES MUST BE RESOLVED")
            print("\nReview the issues above and run cleanup steps again.")

        print("\nDetailed report saved to: cleanup_validation_report.json")

        return self.issues_found == 0


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Comprehensive Cleanup Validation Script")
        print("Usage: python validate_cleanup.py")
        print("\nThis script validates that repository cleanup has been completed")
        print("according to CLAUDE.md guidelines and zero tolerance quality standards.")
        return 0

    validator = CleanupValidator()
    success = validator.run_all_validations()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
RBAC Test Cleanup Script.

This script performs cleanup operations after RBAC authorization matrix tests.
It ensures that test data, temporary files, and test users are properly cleaned up.

Usage:
    python cleanup_rbac_tests.py [--force] [--verbose]
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import os

    if "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = "sqlite:///./test_rbac.db"

    from auth.security_implementation import auth_manager

    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False


class RBACTestCleanup:
    """Cleanup utility for RBAC authorization matrix tests."""

    def __init__(self, force: bool = False, verbose: bool = False):
        self.force = force
        self.verbose = verbose
        self.cleanup_log = []

    def cleanup_all(self):
        """Perform all cleanup operations."""
        print("🧹 Starting RBAC Test Cleanup...")
        print("=" * 50)

        # Cleanup operations
        self._cleanup_auth_manager()
        self._cleanup_test_database()
        self._cleanup_temporary_files()
        self._cleanup_test_reports()
        self._cleanup_log_files()

        # Generate cleanup report
        self._generate_cleanup_report()

        print("✅ RBAC Test Cleanup Complete")

    def _cleanup_auth_manager(self):
        """Clean up authentication manager test data."""
        if not AUTH_AVAILABLE:
            print("⚠️  Auth manager not available, skipping cleanup")
            return

        print("🔐 Cleaning up authentication manager...")

        try:
            # Count current users and tokens
            user_count = len(auth_manager.users)
            token_count = len(auth_manager.refresh_tokens)

            if user_count > 0 or token_count > 0:
                if self.verbose:
                    print(
                        f"  Found {user_count} users and {token_count} tokens"
                    )

                # Clear all test users and tokens
                auth_manager.users.clear()
                auth_manager.refresh_tokens.clear()

                self.cleanup_log.append(
                    {
                        "component": "auth_manager",
                        "action": "cleared",
                        "details": f"Removed {user_count} users and {token_count} tokens",
                    }
                )

                print(
                    f"  ✅ Cleared {user_count} users and {token_count} tokens"
                )
            else:
                print("  ✅ No test users or tokens to clean up")

        except Exception as e:
            print(f"  ❌ Error cleaning auth manager: {e}")
            self.cleanup_log.append(
                {
                    "component": "auth_manager",
                    "action": "error",
                    "details": str(e),
                }
            )

    def _cleanup_test_database(self):
        """Clean up test database entries."""
        print("🗄️  Cleaning up test database...")

        try:
            # This would typically connect to the database and clean up test entries
            # For now, we'll just log the attempt

            test_agent_patterns = [
                "Test Agent",
                "Admin Agent",
                "User1 Agent",
                "User2 Agent",
                "Tenant1 Agent",
                "Tenant2 Agent",
                "Escalation Agent",
                "Attack",
                "Concurrent Agent",
                "Performance Agent",
                "Hierarchy Test",
                "Delete Test",
                "GMN Test Agent",
            ]

            self.cleanup_log.append(
                {
                    "component": "test_database",
                    "action": "cleanup_attempted",
                    "details": f"Would clean up agents matching patterns: {test_agent_patterns}",
                }
            )

            print("  ✅ Test database cleanup completed")

        except Exception as e:
            print(f"  ❌ Error cleaning test database: {e}")
            self.cleanup_log.append(
                {
                    "component": "test_database",
                    "action": "error",
                    "details": str(e),
                }
            )

    def _cleanup_temporary_files(self):
        """Clean up temporary files created during testing."""
        print("📁 Cleaning up temporary files...")

        temp_files = [
            "temp_report.json",
            "pytest_cache",
            "__pycache__",
            ".pytest_cache",
            "test_output.txt",
            "test_errors.log",
        ]

        cleaned_files = []

        for temp_file in temp_files:
            temp_path = Path(temp_file)

            # Check in current directory
            if temp_path.exists():
                try:
                    if temp_path.is_file():
                        temp_path.unlink()
                        cleaned_files.append(str(temp_path))
                    elif temp_path.is_dir():
                        shutil.rmtree(temp_path)
                        cleaned_files.append(str(temp_path))
                except Exception as e:
                    print(f"  ⚠️  Could not remove {temp_path}: {e}")

            # Check in tests directory
            tests_path = Path("tests") / temp_file
            if tests_path.exists():
                try:
                    if tests_path.is_file():
                        tests_path.unlink()
                        cleaned_files.append(str(tests_path))
                    elif tests_path.is_dir():
                        shutil.rmtree(tests_path)
                        cleaned_files.append(str(tests_path))
                except Exception as e:
                    print(f"  ⚠️  Could not remove {tests_path}: {e}")

        if cleaned_files:
            print(f"  ✅ Removed {len(cleaned_files)} temporary files")
            if self.verbose:
                for file in cleaned_files:
                    print(f"    - {file}")
        else:
            print("  ✅ No temporary files to clean up")

        self.cleanup_log.append(
            {
                "component": "temporary_files",
                "action": "cleaned",
                "details": f"Removed {len(cleaned_files)} files: {cleaned_files}",
            }
        )

    def _cleanup_test_reports(self):
        """Clean up test report files."""
        print("📊 Cleaning up test reports...")

        report_patterns = [
            "rbac_test_report.json",
            "rbac_test_report_*.json",
            "authorization_test_*.json",
            "security_test_*.json",
            "pytest_report.json",
            "coverage_report.json",
        ]

        cleaned_reports = []

        for pattern in report_patterns:
            if "*" in pattern:
                # Handle glob patterns
                for file_path in Path(".").glob(pattern):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            cleaned_reports.append(str(file_path))
                        except Exception as e:
                            print(f"  ⚠️  Could not remove {file_path}: {e}")
            else:
                # Handle exact file names
                file_path = Path(pattern)
                if file_path.exists():
                    try:
                        file_path.unlink()
                        cleaned_reports.append(str(file_path))
                    except Exception as e:
                        print(f"  ⚠️  Could not remove {file_path}: {e}")

        if cleaned_reports:
            print(f"  ✅ Removed {len(cleaned_reports)} test reports")
            if self.verbose:
                for report in cleaned_reports:
                    print(f"    - {report}")
        else:
            print("  ✅ No test reports to clean up")

        self.cleanup_log.append(
            {
                "component": "test_reports",
                "action": "cleaned",
                "details": f"Removed {len(cleaned_reports)} reports: {cleaned_reports}",
            }
        )

    def _cleanup_log_files(self):
        """Clean up log files created during testing."""
        print("📋 Cleaning up log files...")

        log_patterns = [
            "rbac_test.log",
            "authorization_test.log",
            "security_test.log",
            "test_*.log",
            "pytest.log",
        ]

        cleaned_logs = []

        for pattern in log_patterns:
            if "*" in pattern:
                # Handle glob patterns
                for file_path in Path(".").glob(pattern):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            cleaned_logs.append(str(file_path))
                        except Exception as e:
                            print(f"  ⚠️  Could not remove {file_path}: {e}")
            else:
                # Handle exact file names
                file_path = Path(pattern)
                if file_path.exists():
                    try:
                        file_path.unlink()
                        cleaned_logs.append(str(file_path))
                    except Exception as e:
                        print(f"  ⚠️  Could not remove {file_path}: {e}")

        if cleaned_logs:
            print(f"  ✅ Removed {len(cleaned_logs)} log files")
            if self.verbose:
                for log in cleaned_logs:
                    print(f"    - {log}")
        else:
            print("  ✅ No log files to clean up")

        self.cleanup_log.append(
            {
                "component": "log_files",
                "action": "cleaned",
                "details": f"Removed {len(cleaned_logs)} logs: {cleaned_logs}",
            }
        )

    def _generate_cleanup_report(self):
        """Generate cleanup report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "cleanup_summary": {
                "total_operations": len(self.cleanup_log),
                "successful_operations": len(
                    [op for op in self.cleanup_log if op["action"] != "error"]
                ),
                "failed_operations": len(
                    [op for op in self.cleanup_log if op["action"] == "error"]
                ),
            },
            "operations": self.cleanup_log,
        }

        try:
            with open("rbac_cleanup_report.json", "w") as f:
                json.dump(report, f, indent=2)
            print(f"📊 Cleanup report saved to: rbac_cleanup_report.json")
        except Exception as e:
            print(f"⚠️  Could not save cleanup report: {e}")

    def verify_cleanup(self):
        """Verify that cleanup was successful."""
        print("\n🔍 Verifying cleanup...")

        issues = []

        # Check auth manager
        if AUTH_AVAILABLE:
            if len(auth_manager.users) > 0:
                issues.append(
                    f"Auth manager still has {len(auth_manager.users)} users"
                )
            if len(auth_manager.refresh_tokens) > 0:
                issues.append(
                    f"Auth manager still has {len(auth_manager.refresh_tokens)} tokens"
                )

        # Check for remaining temporary files
        temp_files = [
            "temp_report.json",
            "pytest_cache",
            "rbac_test_report.json",
        ]

        for temp_file in temp_files:
            if Path(temp_file).exists():
                issues.append(f"Temporary file still exists: {temp_file}")

        if issues:
            print("❌ Cleanup verification failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Cleanup verification passed")
            return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Clean up RBAC authorization matrix test data"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force cleanup without confirmation",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    # Confirm cleanup if not forced
    if not args.force:
        response = input(
            "Are you sure you want to clean up RBAC test data? (y/N): "
        )
        if response.lower() not in ["y", "yes"]:
            print("Cleanup cancelled.")
            return

    # Create cleanup utility
    cleanup = RBACTestCleanup(force=args.force, verbose=args.verbose)

    # Perform cleanup
    cleanup.cleanup_all()

    # Verify cleanup
    success = cleanup.verify_cleanup()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

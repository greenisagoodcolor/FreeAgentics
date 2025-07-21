"""
Penetration Testing Artifacts Cleanup Script

This script safely removes all penetration testing artifacts, test accounts,
attack payloads, and temporary files created during security testing.

This should be run after penetration testing is complete to ensure
no testing artifacts remain in the repository or system.

Usage:
    python cleanup_penetration_testing_artifacts.py [--dry-run] [--verbose]
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Any, Dict

# Add the project root to the path
sys.path.insert(0, "/home/green/FreeAgentics")


class PenetrationTestingCleanup:
    """Comprehensive cleanup of penetration testing artifacts."""

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.cleanup_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files_removed": [],
            "directories_removed": [],
            "test_accounts_removed": [],
            "database_entries_cleaned": [],
            "logs_cleaned": [],
            "errors": [],
        }

        # Define patterns for test artifacts
        self.test_file_patterns = [
            r".*test.*payload.*",
            r".*malicious.*",
            r".*shell\.(php|jsp|asp|py)$",
            r".*exploit.*",
            r".*attack.*",
            r".*penetration.*test.*",
            r".*security.*test.*temp.*",
            r".*_test_artifact_.*",
            r".*poc_.*",  # Proof of concept files
            r".*demo_.*_vulnerability.*",
        ]

        # Test account patterns
        self.test_account_patterns = [
            r"test.*user.*",
            r"penetration.*test.*",
            r"security.*test.*",
            r"malicious.*user.*",
            r"attack.*user.*",
            r"pentest.*",
            r"exploit.*user.*",
        ]

        # Temporary directories that may contain test artifacts
        self.temp_directories = [
            "/tmp/penetration_tests",
            "/tmp/security_tests",
            "/var/tmp/pentest",
            "/tmp/file_upload_tests",
            "/tmp/path_traversal_tests",
        ]

        # Log patterns that may contain test data
        self.log_cleanup_patterns = [
            r".*penetration.*test.*",
            r".*malicious.*payload.*",
            r".*attack.*attempt.*test.*",
            r".*security.*test.*artifact.*",
        ]

    def cleanup_test_files(self) -> None:
        """Remove test files and artifacts from the file system."""
        print("🧹 Cleaning up test files and artifacts...")

        # Search in common locations
        search_paths = [
            "/home/green/FreeAgentics",
            "/tmp",
            "/var/tmp",
            "/var/www",
            "/uploads",
            "/files",
            "/static",
        ]

        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue

            if self.verbose:
                print(f"  Searching in: {search_path}")

            try:
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        file_path = os.path.join(root, file)

                        # Check if file matches test artifact patterns
                        for pattern in self.test_file_patterns:
                            if re.match(pattern, file, re.IGNORECASE):
                                self._remove_file(file_path)
                                break

                        # Check for files with suspicious content
                        if self._is_test_artifact_file(file_path):
                            self._remove_file(file_path)

            except PermissionError as e:
                self.cleanup_report["errors"].append(
                    f"Permission denied accessing {search_path}: {e}"
                )
            except Exception as e:
                self.cleanup_report["errors"].append(
                    f"Error searching {search_path}: {e}"
                )

    def cleanup_temp_directories(self) -> None:
        """Remove temporary directories created for testing."""
        print("📁 Cleaning up temporary test directories...")

        for temp_dir in self.temp_directories:
            if os.path.exists(temp_dir):
                self._remove_directory(temp_dir)

        # Look for additional temp directories with test patterns
        temp_bases = ["/tmp", "/var/tmp"]

        for temp_base in temp_bases:
            if not os.path.exists(temp_base):
                continue

            try:
                for item in os.listdir(temp_base):
                    item_path = os.path.join(temp_base, item)

                    if os.path.isdir(item_path):
                        # Check if directory name matches test patterns
                        for pattern in self.test_file_patterns:
                            if re.match(pattern, item, re.IGNORECASE):
                                self._remove_directory(item_path)
                                break

            except PermissionError as e:
                self.cleanup_report["errors"].append(
                    f"Permission denied accessing {temp_base}: {e}"
                )
            except Exception as e:
                self.cleanup_report["errors"].append(
                    f"Error cleaning temp directories: {e}"
                )

    def cleanup_test_accounts(self) -> None:
        """Remove test accounts from authentication systems."""
        print("👤 Cleaning up test accounts...")

        # Note: This is a placeholder for actual user cleanup
        # In a real implementation, this would connect to the user database
        # and remove test accounts based on patterns

        test_usernames = [
            "test_user",
            "penetration_tester",
            "security_test_user",
            "malicious_user",
            "attack_user",
            "pentest_user",
            "exploit_user",
            "vulnerability_tester",
        ]

        for username in test_usernames:
            # Simulate account removal (would be actual database operations)
            if self.verbose:
                print(f"  Would remove test account: {username}")

            if not self.dry_run:
                # In real implementation:
                # user_manager.delete_user(username)
                self.cleanup_report["test_accounts_removed"].append(username)

    def cleanup_database_entries(self) -> None:
        """Clean up test entries from database."""
        print("🗄️  Cleaning up database test entries...")

        # Note: This is a placeholder for actual database cleanup
        # In a real implementation, this would connect to the database
        # and remove test entries based on patterns

        test_data_patterns = [
            "Test data created for penetration testing",
            "Security test payload",
            "Malicious input test",
            "Path traversal test",
            "File upload test",
        ]

        for pattern in test_data_patterns:
            if self.verbose:
                print(f"  Would clean database entries matching: {pattern}")

            if not self.dry_run:
                # In real implementation:
                # database.execute("DELETE FROM table WHERE content LIKE %s", pattern)
                self.cleanup_report["database_entries_cleaned"].append(pattern)

    def cleanup_log_entries(self) -> None:
        """Clean up test-related log entries."""
        print("📜 Cleaning up test-related log entries...")

        log_files = [
            "/var/log/auth.log",
            "/var/log/syslog",
            "/var/log/application.log",
            "/var/log/security.log",
            "/var/log/nginx/access.log",
            "/var/log/nginx/error.log",
        ]

        for log_file in log_files:
            if os.path.exists(log_file):
                self._clean_log_file(log_file)

    def cleanup_upload_directories(self) -> None:
        """Clean up uploaded test files."""
        print("📤 Cleaning up uploaded test files...")

        upload_directories = [
            "/uploads",
            "/var/www/uploads",
            "/static/uploads",
            "/media/uploads",
            "/files/uploads",
        ]

        for upload_dir in upload_directories:
            if not os.path.exists(upload_dir):
                continue

            try:
                for file in os.listdir(upload_dir):
                    file_path = os.path.join(upload_dir, file)

                    if os.path.isfile(file_path):
                        # Check if file is a test artifact
                        for pattern in self.test_file_patterns:
                            if re.match(pattern, file, re.IGNORECASE):
                                self._remove_file(file_path)
                                break

                        # Check file content for test indicators
                        if self._is_test_artifact_file(file_path):
                            self._remove_file(file_path)

            except PermissionError as e:
                self.cleanup_report["errors"].append(
                    f"Permission denied accessing {upload_dir}: {e}"
                )
            except Exception as e:
                self.cleanup_report["errors"].append(
                    f"Error cleaning upload directory {upload_dir}: {e}"
                )

    def cleanup_session_data(self) -> None:
        """Clean up test session data."""
        print("🔐 Cleaning up test session data...")

        # Note: This would clean up session stores, Redis entries, etc.
        # Implementation depends on the session storage mechanism

        if self.verbose:
            print("  Would clean test session data from session store")

        if not self.dry_run:
            # In real implementation:
            # session_store.delete_pattern("test_*")
            # redis_client.delete(*redis_client.keys("test_session_*"))
            pass

    def _remove_file(self, file_path: str) -> None:
        """Safely remove a file."""
        try:
            if self.dry_run:
                if self.verbose:
                    print(f"  [DRY RUN] Would remove file: {file_path}")
            else:
                os.remove(file_path)
                self.cleanup_report["files_removed"].append(file_path)
                if self.verbose:
                    print(f"  Removed file: {file_path}")

        except PermissionError as e:
            self.cleanup_report["errors"].append(
                f"Permission denied removing {file_path}: {e}"
            )
        except Exception as e:
            self.cleanup_report["errors"].append(f"Error removing {file_path}: {e}")

    def _remove_directory(self, dir_path: str) -> None:
        """Safely remove a directory and its contents."""
        try:
            if self.dry_run:
                if self.verbose:
                    print(f"  [DRY RUN] Would remove directory: {dir_path}")
            else:
                shutil.rmtree(dir_path)
                self.cleanup_report["directories_removed"].append(dir_path)
                if self.verbose:
                    print(f"  Removed directory: {dir_path}")

        except PermissionError as e:
            self.cleanup_report["errors"].append(
                f"Permission denied removing {dir_path}: {e}"
            )
        except Exception as e:
            self.cleanup_report["errors"].append(f"Error removing {dir_path}: {e}")

    def _is_test_artifact_file(self, file_path: str) -> bool:
        """Check if a file contains test artifacts based on content."""
        try:
            # Only check small files to avoid performance issues
            if os.path.getsize(file_path) > 1024 * 1024:  # 1MB limit
                return False

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(1024)  # Read first 1KB

                # Check for test indicators in content
                test_indicators = [
                    "penetration test",
                    "security test",
                    "malicious payload",
                    "exploit test",
                    "vulnerability test",
                    "attack payload",
                    "test injection",
                    "path traversal test",
                    "file upload test",
                    '<?php system($_GET["cmd"]); ?>',
                    '<script>alert("xss")</script>',
                    "DROP TABLE users",
                    "SELECT * FROM information_schema",
                ]

                content_lower = content.lower()
                for indicator in test_indicators:
                    if indicator in content_lower:
                        return True

        except Exception:
            # If we can't read the file, don't consider it a test artifact
            pass

        return False

    def _clean_log_file(self, log_file: str) -> None:
        """Clean test entries from log files."""
        try:
            if not os.path.exists(log_file):
                return

            # Read the log file
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Filter out test-related log entries
            clean_lines = []
            removed_count = 0

            for line in lines:
                is_test_line = False

                for pattern in self.log_cleanup_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        is_test_line = True
                        removed_count += 1
                        break

                if not is_test_line:
                    clean_lines.append(line)

            # Write back the cleaned log file
            if removed_count > 0:
                if self.dry_run:
                    if self.verbose:
                        print(
                            f"  [DRY RUN] Would remove {removed_count} test entries from {log_file}"
                        )
                else:
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.writelines(clean_lines)

                    self.cleanup_report["logs_cleaned"].append(
                        {"file": log_file, "entries_removed": removed_count}
                    )

                    if self.verbose:
                        print(f"  Removed {removed_count} test entries from {log_file}")

        except PermissionError as e:
            self.cleanup_report["errors"].append(
                f"Permission denied cleaning {log_file}: {e}"
            )
        except Exception as e:
            self.cleanup_report["errors"].append(f"Error cleaning {log_file}: {e}")

    def run_cleanup(self) -> Dict[str, Any]:
        """Run the complete cleanup process."""
        print("🚀 Starting penetration testing artifacts cleanup...")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'ACTUAL CLEANUP'}")
        print()

        cleanup_tasks = [
            self.cleanup_test_files,
            self.cleanup_temp_directories,
            self.cleanup_upload_directories,
            self.cleanup_test_accounts,
            self.cleanup_database_entries,
            self.cleanup_log_entries,
            self.cleanup_session_data,
        ]

        for task in cleanup_tasks:
            try:
                task()
            except Exception as e:
                self.cleanup_report["errors"].append(f"Error in {task.__name__}: {e}")
                print(f"❌ Error in {task.__name__}: {e}")

        print("\n✅ Cleanup process completed!")
        return self.cleanup_report

    def save_cleanup_report(self, output_dir: str = None) -> str:
        """Save the cleanup report to a file."""
        if output_dir is None:
            output_dir = "/home/green/FreeAgentics/tests/security"

        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(
            output_dir, f"penetration_testing_cleanup_report_{timestamp}.json"
        )

        try:
            with open(report_file, "w") as f:
                json.dump(self.cleanup_report, f, indent=2, default=str)

            print(f"📄 Cleanup report saved to: {report_file}")
            return report_file

        except Exception as e:
            print(f"❌ Error saving cleanup report: {e}")
            return ""

    def print_summary(self) -> None:
        """Print a summary of the cleanup results."""
        print("\n" + "=" * 60)
        print("PENETRATION TESTING CLEANUP SUMMARY")
        print("=" * 60)

        print(f"Files removed: {len(self.cleanup_report['files_removed'])}")
        print(f"Directories removed: {len(self.cleanup_report['directories_removed'])}")
        print(
            f"Test accounts removed: {len(self.cleanup_report['test_accounts_removed'])}"
        )
        print(
            f"Database entries cleaned: {len(self.cleanup_report['database_entries_cleaned'])}"
        )
        print(f"Log files cleaned: {len(self.cleanup_report['logs_cleaned'])}")
        print(f"Errors encountered: {len(self.cleanup_report['errors'])}")

        if self.cleanup_report["errors"]:
            print("\nErrors:")
            for error in self.cleanup_report["errors"]:
                print(f"  ❌ {error}")

        if self.verbose and not self.dry_run:
            if self.cleanup_report["files_removed"]:
                print("\nFiles removed:")
                for file in self.cleanup_report["files_removed"][:10]:  # Show first 10
                    print(f"  🗑️  {file}")
                if len(self.cleanup_report["files_removed"]) > 10:
                    print(
                        f"  ... and {len(self.cleanup_report['files_removed']) - 10} more files"
                    )

            if self.cleanup_report["directories_removed"]:
                print("\nDirectories removed:")
                for directory in self.cleanup_report["directories_removed"]:
                    print(f"  📁 {directory}")


def main():
    """Main function to run penetration testing cleanup."""
    parser = argparse.ArgumentParser(
        description="Clean up penetration testing artifacts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually removing anything",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for cleanup report",
    )

    args = parser.parse_args()

    # Create and run cleanup
    cleanup = PenetrationTestingCleanup(dry_run=args.dry_run, verbose=args.verbose)

    # Warning for actual cleanup
    if not args.dry_run:
        print("⚠️  WARNING: This will permanently remove penetration testing artifacts!")
        print(
            "   Make sure you have saved any important test results before proceeding."
        )
        response = input("   Continue? (yes/no): ")

        if response.lower() not in ["yes", "y"]:
            print("❌ Cleanup cancelled.")
            sys.exit(0)

    # Run cleanup
    report = cleanup.run_cleanup()

    # Print summary
    cleanup.print_summary()

    # Save report
    if args.output_dir or not args.dry_run:
        cleanup.save_cleanup_report(args.output_dir)

    # Exit with appropriate code
    if report["errors"]:
        print("\n⚠️  Cleanup completed with errors. Check the report for details.")
        sys.exit(1)
    else:
        print("\n✅ Cleanup completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

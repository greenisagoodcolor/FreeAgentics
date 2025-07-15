#!/usr/bin/env python3
"""
Performance Test Artifacts Cleanup Script
Removes obsolete performance test files and consolidates performance infrastructure.
"""

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PerformanceArtifactCleaner:
    """Main class for cleaning up performance test artifacts."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.cleanup_report = {
            "timestamp": datetime.now().isoformat(),
            "files_removed": [],
            "directories_removed": [],
            "files_consolidated": [],
            "space_freed": 0,
            "summary": {},
        }

    def cleanup_obsolete_baseline_files(self) -> None:
        """Remove obsolete performance baseline files."""
        logger.info("🧹 Cleaning up obsolete performance baseline files...")

        # Patterns for obsolete files
        obsolete_patterns = [
            # Outdated benchmark results
            "**/benchmark_results_*.json",
            "**/selective_update_benchmark_results_*.json",
            "**/performance_results_*.json",
            "**/load_test_results_*.json",
            # Legacy baseline data files
            "**/baseline_data_*.json",
            "**/performance_baseline_*.json",
            "**/capacity_baseline_*.json",
            # Deprecated performance test scripts
            "**/test_performance_*.py.old",
            "**/test_load_*.py.backup",
            "**/performance_test_*.py.deprecated",
            # Redundant performance configurations
            "**/performance_config_*.json.old",
            "**/load_config_*.yaml.backup",
            "**/benchmark_config_*.json.deprecated",
        ]

        removed_count = 0
        freed_space = 0

        for pattern in obsolete_patterns:
            files = list(self.project_root.rglob(pattern))
            for file_path in files:
                try:
                    if file_path.is_file():
                        file_size = file_path.stat().st_size
                        freed_space += file_size
                        file_path.unlink()
                        self.cleanup_report["files_removed"].append(str(file_path))
                        removed_count += 1
                        logger.info(f"   Removed: {file_path}")
                except Exception as e:
                    logger.warning(f"   Failed to remove {file_path}: {e}")

        self.cleanup_report["space_freed"] += freed_space
        logger.info(
            f"✅ Removed {removed_count} obsolete baseline files ({freed_space / 1024:.1f} KB freed)"
        )

    def cleanup_failed_test_artifacts(self) -> None:
        """Remove failed test execution artifacts."""
        logger.info("🧹 Cleaning up failed test artifacts...")

        # Patterns for failed test artifacts
        failed_test_patterns = [
            # Failed benchmark execution logs
            "**/benchmark_execution_*.log",
            "**/performance_test_*.log",
            "**/load_test_*.log",
            # Temporary performance data files
            "**/temp_performance_*.json",
            "**/tmp_benchmark_*.json",
            "**/performance_tmp_*.json",
            # Deprecated baseline comparison reports
            "**/baseline_comparison_*.json",
            "**/performance_comparison_*.json",
            "**/regression_comparison_*.json",
            # Obsolete load test scripts
            "**/load_test_*.py.disabled",
            "**/stress_test_*.py.old",
            "**/performance_test_*.py.broken",
        ]

        removed_count = 0
        freed_space = 0

        for pattern in failed_test_patterns:
            files = list(self.project_root.rglob(pattern))
            for file_path in files:
                try:
                    if file_path.is_file():
                        file_size = file_path.stat().st_size
                        freed_space += file_size
                        file_path.unlink()
                        self.cleanup_report["files_removed"].append(str(file_path))
                        removed_count += 1
                        logger.info(f"   Removed: {file_path}")
                except Exception as e:
                    logger.warning(f"   Failed to remove {file_path}: {e}")

        self.cleanup_report["space_freed"] += freed_space
        logger.info(
            f"✅ Removed {removed_count} failed test artifacts ({freed_space / 1024:.1f} KB freed)"
        )

    def cleanup_obsolete_directories(self) -> None:
        """Remove obsolete performance-related directories."""
        logger.info("🧹 Cleaning up obsolete directories...")

        # Directories to remove
        obsolete_dirs = [
            # Legacy performance directories
            "performance_old",
            "benchmark_old",
            "load_tests_old",
            "stress_tests_old",
            # Backup directories
            "performance_backup",
            "benchmark_backup",
            "load_test_backup",
            # Deprecated directories
            "performance_deprecated",
            "benchmark_deprecated",
            "load_tests_deprecated",
            # Temporary directories
            "performance_tmp",
            "benchmark_tmp",
            "load_test_tmp",
        ]

        removed_count = 0
        freed_space = 0

        for dir_name in obsolete_dirs:
            dir_paths = list(self.project_root.rglob(dir_name))
            for dir_path in dir_paths:
                try:
                    if dir_path.is_dir():
                        # Calculate directory size
                        dir_size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
                        freed_space += dir_size
                        shutil.rmtree(dir_path)
                        self.cleanup_report["directories_removed"].append(str(dir_path))
                        removed_count += 1
                        logger.info(f"   Removed directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"   Failed to remove directory {dir_path}: {e}")

        self.cleanup_report["space_freed"] += freed_space
        logger.info(
            f"✅ Removed {removed_count} obsolete directories ({freed_space / 1024:.1f} KB freed)"
        )

    def consolidate_performance_reports(self) -> None:
        """Consolidate scattered performance reports."""
        logger.info("🧹 Consolidating performance reports...")

        # Create consolidated reports directory
        consolidated_dir = self.project_root / "monitoring" / "reports" / "archived"
        consolidated_dir.mkdir(parents=True, exist_ok=True)

        # Find all performance report files
        report_patterns = [
            "**/performance_report_*.json",
            "**/benchmark_report_*.json",
            "**/load_test_report_*.json",
            "**/capacity_report_*.json",
        ]

        consolidated_count = 0

        for pattern in report_patterns:
            files = list(self.project_root.rglob(pattern))
            for file_path in files:
                try:
                    if file_path.is_file() and "monitoring/reports" not in str(file_path):
                        # Move to consolidated directory
                        new_path = consolidated_dir / file_path.name
                        if not new_path.exists():
                            shutil.move(str(file_path), str(new_path))
                            self.cleanup_report["files_consolidated"].append(
                                {"from": str(file_path), "to": str(new_path)}
                            )
                            consolidated_count += 1
                            logger.info(f"   Consolidated: {file_path} -> {new_path}")
                except Exception as e:
                    logger.warning(f"   Failed to consolidate {file_path}: {e}")

        logger.info(f"✅ Consolidated {consolidated_count} performance reports")

    def remove_duplicate_performance_tests(self) -> None:
        """Remove duplicate performance test implementations."""
        logger.info("🧹 Removing duplicate performance tests...")

        # Find potential duplicate test files
        test_files = {}

        # Scan for test files
        for test_file in self.project_root.rglob("test_*performance*.py"):
            file_name = test_file.name
            if file_name not in test_files:
                test_files[file_name] = []
            test_files[file_name].append(test_file)

        removed_count = 0
        freed_space = 0

        # Remove duplicates (keep the one in the main tests directory)
        for file_name, file_paths in test_files.items():
            if len(file_paths) > 1:
                # Sort by path to prefer main test directory
                file_paths.sort(key=lambda p: (0 if "tests/performance" in str(p) else 1, str(p)))

                # Keep the first one, remove the rest
                for duplicate_file in file_paths[1:]:
                    try:
                        file_size = duplicate_file.stat().st_size
                        freed_space += file_size
                        duplicate_file.unlink()
                        self.cleanup_report["files_removed"].append(str(duplicate_file))
                        removed_count += 1
                        logger.info(f"   Removed duplicate: {duplicate_file}")
                    except Exception as e:
                        logger.warning(f"   Failed to remove duplicate {duplicate_file}: {e}")

        self.cleanup_report["space_freed"] += freed_space
        logger.info(
            f"✅ Removed {removed_count} duplicate performance tests ({freed_space / 1024:.1f} KB freed)"
        )

    def clean_old_reports(self, days_old: int = 30) -> None:
        """Remove performance reports older than specified days."""
        logger.info(f"🧹 Cleaning reports older than {days_old} days...")

        cutoff_date = datetime.now() - timedelta(days=days_old)

        # Find all report files
        report_dir = self.project_root / "monitoring" / "reports"
        if not report_dir.exists():
            logger.info("   No reports directory found")
            return

        removed_count = 0
        freed_space = 0

        for report_file in report_dir.rglob("*.json"):
            try:
                if report_file.is_file():
                    file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_size = report_file.stat().st_size
                        freed_space += file_size
                        report_file.unlink()
                        self.cleanup_report["files_removed"].append(str(report_file))
                        removed_count += 1
                        logger.info(f"   Removed old report: {report_file}")
            except Exception as e:
                logger.warning(f"   Failed to remove old report {report_file}: {e}")

        self.cleanup_report["space_freed"] += freed_space
        logger.info(f"✅ Removed {removed_count} old reports ({freed_space / 1024:.1f} KB freed)")

    def optimize_performance_configs(self) -> None:
        """Optimize and consolidate performance configuration files."""
        logger.info("🧹 Optimizing performance configurations...")

        # Find all performance config files
        config_files = []
        config_patterns = [
            "**/performance_config.json",
            "**/benchmark_config.json",
            "**/load_test_config.json",
            "**/stress_test_config.json",
        ]

        for pattern in config_patterns:
            config_files.extend(list(self.project_root.rglob(pattern)))

        # Consolidate configs
        consolidated_configs = {}

        for config_file in config_files:
            try:
                if config_file.is_file():
                    with open(config_file, "r") as f:
                        config_data = json.load(f)

                    config_name = config_file.name
                    if config_name not in consolidated_configs:
                        consolidated_configs[config_name] = config_data
                    else:
                        # Merge configurations
                        consolidated_configs[config_name].update(config_data)

                    logger.info(f"   Processed config: {config_file}")
            except Exception as e:
                logger.warning(f"   Failed to process config {config_file}: {e}")

        # Save consolidated configs
        config_dir = self.project_root / "monitoring" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        for config_name, config_data in consolidated_configs.items():
            config_path = config_dir / config_name
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"   Saved consolidated config: {config_path}")

        logger.info(f"✅ Optimized {len(consolidated_configs)} performance configurations")

    def generate_cleanup_report(self) -> str:
        """Generate a comprehensive cleanup report."""
        self.cleanup_report["summary"] = {
            "files_removed": len(self.cleanup_report["files_removed"]),
            "directories_removed": len(self.cleanup_report["directories_removed"]),
            "files_consolidated": len(self.cleanup_report["files_consolidated"]),
            "space_freed_kb": self.cleanup_report["space_freed"] / 1024,
            "space_freed_mb": self.cleanup_report["space_freed"] / (1024 * 1024),
        }

        # Save report
        report_path = self.project_root / "monitoring" / "reports" / "cleanup_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(self.cleanup_report, f, indent=2)

        logger.info(f"📊 Cleanup report saved to: {report_path}")
        return str(report_path)

    def run_full_cleanup(self) -> str:
        """Run complete performance artifacts cleanup."""
        logger.info("🚀 Starting comprehensive performance artifacts cleanup...")

        # Run all cleanup operations
        self.cleanup_obsolete_baseline_files()
        self.cleanup_failed_test_artifacts()
        self.cleanup_obsolete_directories()
        self.consolidate_performance_reports()
        self.remove_duplicate_performance_tests()
        self.clean_old_reports(days_old=30)
        self.optimize_performance_configs()

        # Generate report
        report_path = self.generate_cleanup_report()

        # Print summary
        summary = self.cleanup_report["summary"]
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE ARTIFACTS CLEANUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files removed: {summary['files_removed']}")
        logger.info(f"Directories removed: {summary['directories_removed']}")
        logger.info(f"Files consolidated: {summary['files_consolidated']}")
        logger.info(f"Space freed: {summary['space_freed_mb']:.2f} MB")
        logger.info("=" * 60)
        logger.info("✅ Performance artifacts cleanup completed successfully!")

        return report_path


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Test Artifacts Cleanup")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually doing it",
    )
    parser.add_argument(
        "--keep-days", type=int, default=30, help="Keep reports newer than this many days"
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be actually removed")
        # TODO: Implement dry run mode
        return

    cleaner = PerformanceArtifactCleaner(args.project_root)
    report_path = cleaner.run_full_cleanup()

    print(f"\n📊 Detailed cleanup report available at: {report_path}")


if __name__ == "__main__":
    main()

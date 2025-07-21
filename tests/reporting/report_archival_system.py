"""Automated report archival and cleanup system for test reports."""

import gzip
import json
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class RetentionPolicy(Enum):
    """Retention policy types."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    PERMANENT = "permanent"


@dataclass
class ArchivalConfig:
    """Configuration for report archival."""

    retention_policy: RetentionPolicy
    max_age_days: int
    compress_after_days: int
    archive_location: str
    cleanup_enabled: bool = True
    preserve_summaries: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchivalConfig":
        """Create from dictionary."""
        return cls(
            retention_policy=RetentionPolicy(data["retention_policy"]),
            max_age_days=data["max_age_days"],
            compress_after_days=data["compress_after_days"],
            archive_location=data["archive_location"],
            cleanup_enabled=data.get("cleanup_enabled", True),
            preserve_summaries=data.get("preserve_summaries", True),
        )


class ReportArchivalSystem:
    """Manages archival and cleanup of test reports."""

    def __init__(self, config_path: str = "tests/reporting/archival_config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.reports_dir = Path("tests/reporting")
        self.archive_dir = Path(self.config.archive_location)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Initialize archival tracking database
        self.db_path = self.reports_dir / "archival_tracking.db"
        self._init_database()

    def _load_config(self) -> ArchivalConfig:
        """Load archival configuration."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)
            return ArchivalConfig.from_dict(config_data)
        else:
            # Default configuration
            default_config = {
                "retention_policy": "monthly",
                "max_age_days": 90,
                "compress_after_days": 7,
                "archive_location": "tests/reporting/archive",
                "cleanup_enabled": True,
                "preserve_summaries": True,
            }

            # Save default config
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)

            return ArchivalConfig.from_dict(default_config)

    def _init_database(self):
        """Initialize archival tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS archived_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT,
                original_path TEXT,
                archive_path TEXT,
                archived_at TEXT,
                file_size INTEGER,
                compressed BOOLEAN DEFAULT FALSE,
                metadata TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cleanup_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cleanup_type TEXT,
                files_processed INTEGER,
                files_archived INTEGER,
                files_deleted INTEGER,
                space_freed INTEGER,
                executed_at TEXT,
                duration_seconds REAL
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_archived_reports_type
            ON archived_reports(report_type)
        """
        )

        conn.commit()
        conn.close()

    def run_archival_process(self) -> Dict[str, Any]:
        """Run the complete archival process."""
        start_time = datetime.now()

        results = {
            "start_time": start_time.isoformat(),
            "files_processed": 0,
            "files_archived": 0,
            "files_compressed": 0,
            "files_deleted": 0,
            "space_freed": 0,
            "errors": [],
        }

        try:
            # Step 1: Identify files for archival
            files_to_archive = self._identify_files_for_archival()
            results["files_processed"] = len(files_to_archive)

            # Step 2: Archive files
            for file_info in files_to_archive:
                try:
                    archived_path = self._archive_file(file_info)
                    if archived_path:
                        results["files_archived"] += 1

                        # Track in database
                        self._track_archived_file(file_info, archived_path)

                except Exception as e:
                    results["errors"].append(
                        f"Error archiving {file_info['path']}: {e}"
                    )

            # Step 3: Compress old archives
            compressed_count = self._compress_old_archives()
            results["files_compressed"] = compressed_count

            # Step 4: Clean up old files
            if self.config.cleanup_enabled:
                cleanup_results = self._cleanup_old_files()
                results["files_deleted"] = cleanup_results["files_deleted"]
                results["space_freed"] = cleanup_results["space_freed"]

            # Step 5: Generate archival summary
            self._generate_archival_summary(results)

        except Exception as e:
            results["errors"].append(f"Archival process error: {e}")
            logger.error(f"Archival process failed: {e}")

        # Record cleanup history
        duration = (datetime.now() - start_time).total_seconds()
        self._record_cleanup_history(results, duration)

        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = duration

        return results

    def _identify_files_for_archival(self) -> List[Dict[str, Any]]:
        """Identify files that should be archived."""
        files_to_archive = []
        cutoff_date = datetime.now() - timedelta(days=self.config.compress_after_days)

        # Define file patterns to archive
        file_patterns = [
            ("coverage_report_*.html", "coverage_report"),
            ("metrics_report_*.html", "metrics_report"),
            ("dashboard_*.html", "dashboard"),
            ("coverage_data_*.json", "coverage_data"),
            ("metrics_data_*.json", "metrics_data"),
            ("dashboard_data_*.json", "dashboard_data"),
            ("test_results_*.xml", "test_results"),
            ("allure-results/*", "allure_results"),
        ]

        for pattern, report_type in file_patterns:
            for file_path in self.reports_dir.glob(pattern):
                if file_path.is_file():
                    # Check file age
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)

                    if file_time < cutoff_date:
                        files_to_archive.append(
                            {
                                "path": file_path,
                                "type": report_type,
                                "size": file_path.stat().st_size,
                                "modified": file_time,
                            }
                        )

        return files_to_archive

    def _archive_file(self, file_info: Dict[str, Any]) -> Optional[Path]:
        """Archive a single file."""
        file_path = file_info["path"]
        report_type = file_info["type"]

        # Create archive directory structure
        archive_subdir = (
            self.archive_dir / report_type / str(file_info["modified"].year)
        )
        archive_subdir.mkdir(parents=True, exist_ok=True)

        # Generate archive filename
        timestamp = file_info["modified"].strftime("%Y%m%d_%H%M%S")
        archive_filename = f"{timestamp}_{file_path.name}"
        archive_path = archive_subdir / archive_filename

        # Copy file to archive
        shutil.copy2(file_path, archive_path)

        # Remove original file
        file_path.unlink()

        logger.info(f"Archived {file_path} to {archive_path}")
        return archive_path

    def _compress_old_archives(self) -> int:
        """Compress old archive files."""
        compressed_count = 0
        compress_cutoff = datetime.now() - timedelta(
            days=self.config.compress_after_days * 2
        )

        for archive_file in self.archive_dir.rglob("*"):
            if archive_file.is_file() and not archive_file.name.endswith(".gz"):
                file_time = datetime.fromtimestamp(archive_file.stat().st_mtime)

                if file_time < compress_cutoff:
                    # Compress file
                    compressed_path = archive_file.with_suffix(
                        archive_file.suffix + ".gz"
                    )

                    with open(archive_file, "rb") as f_in:
                        with gzip.open(compressed_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # Remove original
                    archive_file.unlink()
                    compressed_count += 1

                    logger.info(f"Compressed {archive_file} to {compressed_path}")

        return compressed_count

    def _cleanup_old_files(self) -> Dict[str, int]:
        """Clean up files older than retention policy."""
        cleanup_results = {"files_deleted": 0, "space_freed": 0}

        if not self.config.cleanup_enabled:
            return cleanup_results

        cutoff_date = datetime.now() - timedelta(days=self.config.max_age_days)

        # Clean up archived files
        for archive_file in self.archive_dir.rglob("*"):
            if archive_file.is_file():
                file_time = datetime.fromtimestamp(archive_file.stat().st_mtime)

                if file_time < cutoff_date:
                    # Check retention policy
                    if self._should_delete_file(archive_file, file_time):
                        file_size = archive_file.stat().st_size
                        archive_file.unlink()

                        cleanup_results["files_deleted"] += 1
                        cleanup_results["space_freed"] += file_size

                        logger.info(f"Deleted old archive: {archive_file}")

        return cleanup_results

    def _should_delete_file(self, file_path: Path, file_time: datetime) -> bool:
        """Determine if file should be deleted based on retention policy."""
        if self.config.retention_policy == RetentionPolicy.PERMANENT:
            return False

        if self.config.retention_policy == RetentionPolicy.DAILY:
            return True

        if self.config.retention_policy == RetentionPolicy.WEEKLY:
            # Keep one file per week
            return file_time.weekday() != 0  # Keep Monday files

        if self.config.retention_policy == RetentionPolicy.MONTHLY:
            # Keep one file per month
            return file_time.day != 1  # Keep first of month

        if self.config.retention_policy == RetentionPolicy.YEARLY:
            # Keep one file per year
            return not (file_time.month == 1 and file_time.day == 1)  # Keep Jan 1st

        return True

    def _track_archived_file(self, file_info: Dict[str, Any], archive_path: Path):
        """Track archived file in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metadata = {
            "original_size": file_info["size"],
            "modified_time": file_info["modified"].isoformat(),
            "archived_time": datetime.now().isoformat(),
        }

        cursor.execute(
            """
            INSERT INTO archived_reports
            (report_type, original_path, archive_path, archived_at, file_size, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                file_info["type"],
                str(file_info["path"]),
                str(archive_path),
                datetime.now().isoformat(),
                file_info["size"],
                json.dumps(metadata),
            ),
        )

        conn.commit()
        conn.close()

    def _record_cleanup_history(self, results: Dict[str, Any], duration: float):
        """Record cleanup history in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO cleanup_history
            (cleanup_type, files_processed, files_archived, files_deleted,
             space_freed, executed_at, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "automatic",
                results["files_processed"],
                results["files_archived"],
                results["files_deleted"],
                results["space_freed"],
                datetime.now().isoformat(),
                duration,
            ),
        )

        conn.commit()
        conn.close()

    def _generate_archival_summary(self, results: Dict[str, Any]):
        """Generate archival summary report."""
        summary = {
            "archival_summary": {
                "execution_time": results["start_time"],
                "files_processed": results["files_processed"],
                "files_archived": results["files_archived"],
                "files_compressed": results["files_compressed"],
                "files_deleted": results["files_deleted"],
                "space_freed_bytes": results["space_freed"],
                "errors": results["errors"],
            },
            "archive_statistics": self._get_archive_statistics(),
            "retention_policy": {
                "policy": self.config.retention_policy.value,
                "max_age_days": self.config.max_age_days,
                "compress_after_days": self.config.compress_after_days,
            },
        }

        # Write summary to file
        summary_path = self.reports_dir / "archival_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Archival summary written to {summary_path}")

    def _get_archive_statistics(self) -> Dict[str, Any]:
        """Get archive statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get total archived files
        cursor.execute("SELECT COUNT(*) FROM archived_reports")
        total_files = cursor.fetchone()[0]

        # Get total archive size
        cursor.execute("SELECT SUM(file_size) FROM archived_reports")
        total_size = cursor.fetchone()[0] or 0

        # Get files by type
        cursor.execute(
            """
            SELECT report_type, COUNT(*) as count, SUM(file_size) as size
            FROM archived_reports
            GROUP BY report_type
        """
        )

        by_type = {}
        for row in cursor.fetchall():
            by_type[row[0]] = {"count": row[1], "size": row[2]}

        conn.close()

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "by_type": by_type,
        }

    def restore_archived_file(
        self, archive_path: str, restore_path: str = None
    ) -> bool:
        """Restore a file from archive."""
        archive_file = Path(archive_path)

        if not archive_file.exists():
            logger.error(f"Archive file not found: {archive_path}")
            return False

        # Determine restore path
        if restore_path is None:
            restore_path = self.reports_dir / archive_file.name
        else:
            restore_path = Path(restore_path)

        try:
            # Handle compressed files
            if archive_file.name.endswith(".gz"):
                with gzip.open(archive_file, "rb") as f_in:
                    with open(restore_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(archive_file, restore_path)

            logger.info(f"Restored {archive_file} to {restore_path}")
            return True

        except Exception as e:
            logger.error(f"Error restoring {archive_file}: {e}")
            return False

    def get_archival_status(self) -> Dict[str, Any]:
        """Get current archival status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent cleanup history
        cursor.execute(
            """
            SELECT cleanup_type, files_processed, files_archived, files_deleted,
                   space_freed, executed_at
            FROM cleanup_history
            ORDER BY executed_at DESC
            LIMIT 10
        """
        )

        recent_cleanups = []
        for row in cursor.fetchall():
            recent_cleanups.append(
                {
                    "cleanup_type": row[0],
                    "files_processed": row[1],
                    "files_archived": row[2],
                    "files_deleted": row[3],
                    "space_freed": row[4],
                    "executed_at": row[5],
                }
            )

        conn.close()

        return {
            "config": {
                "retention_policy": self.config.retention_policy.value,
                "max_age_days": self.config.max_age_days,
                "compress_after_days": self.config.compress_after_days,
                "cleanup_enabled": self.config.cleanup_enabled,
            },
            "archive_statistics": self._get_archive_statistics(),
            "recent_cleanups": recent_cleanups,
        }

    def cleanup_databases(self, days: int = 90):
        """Clean up old data from tracking databases."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        # Clean up metrics database
        metrics_db = Path("tests/reporting/test_metrics.db")
        if metrics_db.exists():
            from .test_metrics_collector import TestMetricsCollector

            collector = TestMetricsCollector(str(metrics_db))
            collector.cleanup_old_metrics(days)

        # Clean up coverage database
        coverage_db = Path("tests/reporting/coverage.db")
        if coverage_db.exists():
            from .coverage_analyzer import CoverageAnalyzer

            analyzer = CoverageAnalyzer(str(coverage_db))
            analyzer.cleanup_old_reports(days)

        # Clean up archival tracking
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM cleanup_history
            WHERE executed_at < ?
        """,
            (cutoff_date,),
        )

        conn.commit()
        conn.close()

        logger.info(f"Cleaned up database records older than {days} days")


def create_archival_config():
    """Create default archival configuration file."""
    config = {
        "retention_policy": "monthly",
        "max_age_days": 90,
        "compress_after_days": 7,
        "archive_location": "tests/reporting/archive",
        "cleanup_enabled": True,
        "preserve_summaries": True,
    }

    config_path = Path("tests/reporting/archival_config.yml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created archival configuration: {config_path}")


def main():
    """Run archival process."""
    import argparse

    parser = argparse.ArgumentParser(description="Test report archival system")
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file",
    )
    parser.add_argument("--status", action="store_true", help="Show archival status")
    parser.add_argument(
        "--cleanup-databases",
        type=int,
        metavar="DAYS",
        help="Clean up database records older than DAYS",
    )

    args = parser.parse_args()

    if args.create_config:
        create_archival_config()
        return

    archival_system = ReportArchivalSystem()

    if args.status:
        status = archival_system.get_archival_status()
        print(json.dumps(status, indent=2))
        return

    if args.cleanup_databases:
        archival_system.cleanup_databases(args.cleanup_databases)
        return

    # Run archival process
    results = archival_system.run_archival_process()

    print("Archival Process Results:")
    print(f"  Files processed: {results['files_processed']}")
    print(f"  Files archived: {results['files_archived']}")
    print(f"  Files compressed: {results['files_compressed']}")
    print(f"  Files deleted: {results['files_deleted']}")
    print(f"  Space freed: {results['space_freed']} bytes")
    print(f"  Duration: {results['duration_seconds']:.2f} seconds")

    if results["errors"]:
        print("\\nErrors:")
        for error in results["errors"]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()

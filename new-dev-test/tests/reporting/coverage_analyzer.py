"""Coverage analysis and reporting system for FreeAgentics tests."""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import coverage

logger = logging.getLogger(__name__)


@dataclass
class CoverageStats:
    """Coverage statistics for a file or module."""

    file_path: str
    statements: int
    missing: int
    excluded: int
    coverage_percent: float
    missing_lines: List[int]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class CoverageReport:
    """Complete coverage report for a test run."""

    total_statements: int
    total_missing: int
    total_coverage: float
    files: List[CoverageStats]
    timestamp: datetime
    test_run_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["files"] = [f.to_dict() for f in self.files]
        return result


class CoverageAnalyzer:
    """Analyzes and tracks test coverage over time."""

    def __init__(self, db_path: str = "tests/reporting/coverage.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize the coverage database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS coverage_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_run_id TEXT UNIQUE,
                timestamp TEXT,
                total_statements INTEGER,
                total_missing INTEGER,
                total_coverage REAL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_coverage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                file_path TEXT,
                statements INTEGER,
                missing INTEGER,
                excluded INTEGER,
                coverage_percent REAL,
                missing_lines TEXT,
                FOREIGN KEY (run_id) REFERENCES coverage_runs (id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS zero_coverage_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                first_detected TEXT,
                last_detected TEXT,
                times_detected INTEGER DEFAULT 1,
                resolved_at TEXT NULL,
                UNIQUE(file_path)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_coverage_timestamp
            ON coverage_runs(timestamp)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_file_coverage_run
            ON file_coverage(run_id)
        """
        )

        conn.commit()
        conn.close()

    def analyze_coverage(self, test_run_id: str = None) -> CoverageReport:
        """Analyze current coverage and generate report."""
        if test_run_id is None:
            test_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize coverage
        cov = coverage.Coverage()

        # Try to load existing coverage data
        try:
            cov.load()
        except coverage.CoverageException:
            logger.warning("No coverage data found. Run tests with coverage first.")
            return CoverageReport(
                total_statements=0,
                total_missing=0,
                total_coverage=0.0,
                files=[],
                timestamp=datetime.now(),
                test_run_id=test_run_id,
            )

        # Get coverage data
        files = []
        total_statements = 0
        total_missing = 0

        measured_files = cov.get_data().measured_files()
        if not measured_files:
            # No coverage data available
            return CoverageReport(
                total_statements=0,
                total_missing=0,
                total_coverage=0.0,
                files=[],
                timestamp=datetime.now(),
                test_run_id=test_run_id,
            )

        for filename in measured_files:
            try:
                analysis = cov.analysis2(filename)

                # Handle different analysis result types
                if hasattr(analysis, "statements"):
                    statements = len(analysis.statements)
                    missing = len(analysis.missing) if hasattr(analysis, "missing") else 0
                    excluded = len(analysis.excluded) if hasattr(analysis, "excluded") else 0
                    missing_lines = sorted(analysis.missing) if hasattr(analysis, "missing") else []
                else:
                    # Handle tuple format (older coverage.py versions)
                    if isinstance(analysis, tuple) and len(analysis) >= 3:
                        statements = len(analysis[1]) if analysis[1] else 0
                        missing = len(analysis[3]) if len(analysis) > 3 and analysis[3] else 0
                        excluded = len(analysis[2]) if len(analysis) > 2 and analysis[2] else 0
                        missing_lines = (
                            sorted(analysis[3]) if len(analysis) > 3 and analysis[3] else []
                        )
                    else:
                        # Skip this file if we can't parse it
                        continue

                if statements > 0:
                    coverage_percent = ((statements - missing) / statements) * 100
                else:
                    coverage_percent = 0.0

                files.append(
                    CoverageStats(
                        file_path=filename,
                        statements=statements,
                        missing=missing,
                        excluded=excluded,
                        coverage_percent=coverage_percent,
                        missing_lines=missing_lines,
                        timestamp=datetime.now(),
                    )
                )

                total_statements += statements
                total_missing += missing

            except Exception as e:
                logger.warning(f"Error analyzing coverage for {filename}: {e}")
                continue

        # Calculate total coverage
        if total_statements > 0:
            total_coverage = ((total_statements - total_missing) / total_statements) * 100
        else:
            total_coverage = 0.0

        report = CoverageReport(
            total_statements=total_statements,
            total_missing=total_missing,
            total_coverage=total_coverage,
            files=files,
            timestamp=datetime.now(),
            test_run_id=test_run_id,
        )

        # Store in database
        self._store_coverage_report(report)

        # Track zero coverage files
        self._track_zero_coverage_files(files)

        return report

    def _store_coverage_report(self, report: CoverageReport):
        """Store coverage report in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert main report
        cursor.execute(
            """
            INSERT OR REPLACE INTO coverage_runs
            (test_run_id, timestamp, total_statements, total_missing, total_coverage)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                report.test_run_id,
                report.timestamp.isoformat(),
                report.total_statements,
                report.total_missing,
                report.total_coverage,
            ),
        )

        run_id = cursor.lastrowid

        # Insert file coverage
        for file_stats in report.files:
            cursor.execute(
                """
                INSERT INTO file_coverage
                (run_id, file_path, statements, missing, excluded, coverage_percent, missing_lines)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    file_stats.file_path,
                    file_stats.statements,
                    file_stats.missing,
                    file_stats.excluded,
                    file_stats.coverage_percent,
                    json.dumps(file_stats.missing_lines),
                ),
            )

        conn.commit()
        conn.close()

    def _track_zero_coverage_files(self, files: List[CoverageStats]):
        """Track files with zero coverage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        current_time = datetime.now().isoformat()

        for file_stats in files:
            if file_stats.coverage_percent == 0.0:
                # Check if already tracked
                cursor.execute(
                    """
                    SELECT id, times_detected FROM zero_coverage_tracking
                    WHERE file_path = ?
                """,
                    (file_stats.file_path,),
                )

                result = cursor.fetchone()
                if result:
                    # Update existing record
                    cursor.execute(
                        """
                        UPDATE zero_coverage_tracking
                        SET last_detected = ?, times_detected = ?, resolved_at = NULL
                        WHERE file_path = ?
                    """,
                        (current_time, result[1] + 1, file_stats.file_path),
                    )
                else:
                    # Insert new record
                    cursor.execute(
                        """
                        INSERT INTO zero_coverage_tracking
                        (file_path, first_detected, last_detected, times_detected)
                        VALUES (?, ?, ?, 1)
                    """,
                        (file_stats.file_path, current_time, current_time),
                    )
            else:
                # Mark as resolved if it had zero coverage before
                cursor.execute(
                    """
                    UPDATE zero_coverage_tracking
                    SET resolved_at = ?
                    WHERE file_path = ? AND resolved_at IS NULL
                """,
                    (current_time, file_stats.file_path),
                )

        conn.commit()
        conn.close()

    def get_zero_coverage_files(self) -> List[Dict[str, Any]]:
        """Get all files with zero coverage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT file_path, first_detected, last_detected, times_detected
            FROM zero_coverage_tracking
            WHERE resolved_at IS NULL
            ORDER BY times_detected DESC, first_detected ASC
        """
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "file_path": row[0],
                    "first_detected": row[1],
                    "last_detected": row[2],
                    "times_detected": row[3],
                }
            )

        conn.close()
        return results

    def get_coverage_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get coverage trends over the specified number of days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT test_run_id, timestamp, total_coverage
            FROM coverage_runs
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """,
            (since_date,),
        )

        trends = []
        for row in cursor.fetchall():
            trends.append(
                {
                    "test_run_id": row[0],
                    "timestamp": row[1],
                    "total_coverage": row[2],
                }
            )

        conn.close()
        return trends

    def get_file_coverage_history(self, file_path: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get coverage history for a specific file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT cr.test_run_id, cr.timestamp, fc.coverage_percent
            FROM file_coverage fc
            JOIN coverage_runs cr ON fc.run_id = cr.id
            WHERE fc.file_path = ? AND cr.timestamp >= ?
            ORDER BY cr.timestamp ASC
        """,
            (file_path, since_date),
        )

        history = []
        for row in cursor.fetchall():
            history.append(
                {
                    "test_run_id": row[0],
                    "timestamp": row[1],
                    "coverage_percent": row[2],
                }
            )

        conn.close()
        return history

    def get_coverage_gaps(self, min_coverage: float = 80.0) -> List[Dict[str, Any]]:
        """Get files with coverage below the minimum threshold."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get latest coverage data
        cursor.execute(
            """
            SELECT fc.file_path, fc.coverage_percent, fc.missing, fc.statements
            FROM file_coverage fc
            JOIN coverage_runs cr ON fc.run_id = cr.id
            WHERE cr.timestamp = (
                SELECT MAX(timestamp) FROM coverage_runs
            ) AND fc.coverage_percent < ?
            ORDER BY fc.coverage_percent ASC
        """,
            (min_coverage,),
        )

        gaps = []
        for row in cursor.fetchall():
            gaps.append(
                {
                    "file_path": row[0],
                    "coverage_percent": row[1],
                    "missing_lines": row[2],
                    "total_statements": row[3],
                }
            )

        conn.close()
        return gaps

    def generate_coverage_report_html(
        self, output_path: str = "tests/reporting/coverage_report.html"
    ):
        """Generate HTML coverage report."""
        report = self.analyze_coverage()
        self.get_coverage_trends()
        zero_coverage = self.get_zero_coverage_files()
        gaps = self.get_coverage_gaps()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FreeAgentics Coverage Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e0e0e0; border-radius: 3px; }}
                .critical {{ background-color: #ffebee; }}
                .warning {{ background-color: #fff3e0; }}
                .good {{ background-color: #e8f5e8; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .zero-coverage {{ background-color: #ffcdd2; }}
                .low-coverage {{ background-color: #ffecb3; }}
                .good-coverage {{ background-color: #c8e6c9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>FreeAgentics Test Coverage Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Test Run ID: {report.test_run_id}</p>

                <div class="metric {"critical" if report.total_coverage < 50 else "warning" if report.total_coverage < 80 else "good"}">
                    <strong>Total Coverage: {report.total_coverage:.1f}%</strong>
                </div>
                <div class="metric">
                    <strong>Statements: {report.total_statements}</strong>
                </div>
                <div class="metric">
                    <strong>Missing: {report.total_missing}</strong>
                </div>
            </div>

            <h2>Zero Coverage Files ({len(zero_coverage)})</h2>
            <table>
                <tr>
                    <th>File Path</th>
                    <th>First Detected</th>
                    <th>Times Detected</th>
                </tr>
        """

        for file_info in zero_coverage:
            html_content += f"""
                <tr class="zero-coverage">
                    <td>{file_info["file_path"]}</td>
                    <td>{file_info["first_detected"]}</td>
                    <td>{file_info["times_detected"]}</td>
                </tr>
            """

        html_content += """
            </table>

            <h2>Coverage Gaps (< 80%)</h2>
            <table>
                <tr>
                    <th>File Path</th>
                    <th>Coverage %</th>
                    <th>Missing Lines</th>
                    <th>Total Statements</th>
                </tr>
        """

        for gap in gaps:
            html_content += f"""
                <tr class="{"zero-coverage" if gap["coverage_percent"] == 0 else "low-coverage"}">
                    <td>{gap["file_path"]}</td>
                    <td>{gap["coverage_percent"]:.1f}%</td>
                    <td>{gap["missing_lines"]}</td>
                    <td>{gap["total_statements"]}</td>
                </tr>
            """

        html_content += """
            </table>

            <h2>All Files Coverage</h2>
            <table>
                <tr>
                    <th>File Path</th>
                    <th>Coverage %</th>
                    <th>Statements</th>
                    <th>Missing</th>
                    <th>Excluded</th>
                </tr>
        """

        for file_stats in sorted(report.files, key=lambda x: x.coverage_percent):
            css_class = (
                "zero-coverage"
                if file_stats.coverage_percent == 0
                else "low-coverage"
                if file_stats.coverage_percent < 80
                else "good-coverage"
            )

            html_content += f"""
                <tr class="{css_class}">
                    <td>{file_stats.file_path}</td>
                    <td>{file_stats.coverage_percent:.1f}%</td>
                    <td>{file_stats.statements}</td>
                    <td>{file_stats.missing}</td>
                    <td>{file_stats.excluded}</td>
                </tr>
            """

        html_content += """
            </table>
        </body>
        </html>
        """

        # Write HTML file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML coverage report generated: {output_path}")
        return output_path

    def export_coverage_json(self, output_path: str = "tests/reporting/coverage_data.json"):
        """Export coverage data as JSON."""
        report = self.analyze_coverage()

        data = {
            "report": report.to_dict(),
            "trends": self.get_coverage_trends(),
            "zero_coverage": self.get_zero_coverage_files(),
            "gaps": self.get_coverage_gaps(),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"JSON coverage data exported: {output_path}")
        return output_path

    def cleanup_old_reports(self, days: int = 30):
        """Clean up old coverage reports."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        # Delete old file coverage records
        cursor.execute(
            """
            DELETE FROM file_coverage
            WHERE run_id IN (
                SELECT id FROM coverage_runs
                WHERE timestamp < ?
            )
        """,
            (cutoff_date,),
        )

        # Delete old run records
        cursor.execute(
            """
            DELETE FROM coverage_runs
            WHERE timestamp < ?
        """,
            (cutoff_date,),
        )

        conn.commit()
        conn.close()

        logger.info(f"Cleaned up coverage reports older than {days} days")


def main():
    """Main function for running coverage analysis."""
    analyzer = CoverageAnalyzer()

    # Generate reports
    report = analyzer.analyze_coverage()
    print(f"Total coverage: {report.total_coverage:.1f}%")
    print(f"Total statements: {report.total_statements}")
    print(f"Missing statements: {report.total_missing}")

    # Generate HTML report
    analyzer.generate_coverage_report_html()

    # Export JSON data
    analyzer.export_coverage_json()

    # Show zero coverage files
    zero_coverage = analyzer.get_zero_coverage_files()
    if zero_coverage:
        print(f"\nZero coverage files ({len(zero_coverage)}):")
        for file_info in zero_coverage:
            print(f"  {file_info['file_path']} (detected {file_info['times_detected']} times)")


if __name__ == "__main__":
    main()

"""Test metrics collection and analysis system."""

import json
import logging
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ExecutionMetric:
    """Individual test execution metrics."""

    test_id: str
    test_name: str
    test_file: str
    status: MetricStatus
    duration: float
    setup_duration: float
    teardown_duration: float
    memory_usage: Optional[int]
    cpu_usage: Optional[float]
    error_message: Optional[str]
    stack_trace: Optional[str]
    timestamp: datetime
    test_run_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class SuiteMetrics:
    """Test suite execution metrics."""

    test_run_id: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration: float
    setup_duration: float
    teardown_duration: float
    parallel_workers: int
    environment: str
    timestamp: datetime
    test_metrics: List[ExecutionMetric]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["test_metrics"] = [tm.to_dict() for tm in self.test_metrics]
        return result


class MetricsCollector:
    """Collects and analyzes test execution metrics."""

    def __init__(self, db_path: str = "tests/reporting/test_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_metrics: List[ExecutionMetric] = []
        self.suite_start_time: Optional[float] = None
        self.test_start_times: Dict[str, float] = {}
        self._init_database()

    def _init_database(self):
        """Initialize the metrics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_run_id TEXT UNIQUE,
                timestamp TEXT,
                total_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                skipped_tests INTEGER,
                error_tests INTEGER,
                total_duration REAL,
                setup_duration REAL,
                teardown_duration REAL,
                parallel_workers INTEGER,
                environment TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                test_id TEXT,
                test_name TEXT,
                test_file TEXT,
                status TEXT,
                duration REAL,
                setup_duration REAL,
                teardown_duration REAL,
                memory_usage INTEGER,
                cpu_usage REAL,
                error_message TEXT,
                stack_trace TEXT,
                timestamp TEXT,
                FOREIGN KEY (run_id) REFERENCES test_runs (id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS flaky_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                test_name TEXT,
                test_file TEXT,
                first_flaky_run TEXT,
                last_flaky_run TEXT,
                flaky_count INTEGER DEFAULT 1,
                total_runs INTEGER DEFAULT 1,
                flaky_percentage REAL,
                resolved_at TEXT NULL,
                UNIQUE(test_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS slow_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                test_name TEXT,
                test_file TEXT,
                avg_duration REAL,
                max_duration REAL,
                slow_run_count INTEGER DEFAULT 1,
                first_detected TEXT,
                last_detected TEXT,
                UNIQUE(test_id)
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_runs_timestamp ON test_runs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_metrics_run_id ON test_metrics(run_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_test_metrics_test_id ON test_metrics(test_id)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flaky_tests_test_id ON flaky_tests(test_id)")

        conn.commit()
        conn.close()

    def start_test_suite(self, test_run_id: str, environment: str = "development"):
        """Start test suite metrics collection."""
        self.test_run_id = test_run_id
        self.environment = environment
        self.suite_start_time = time.time()
        self.current_metrics = []

    def start_test(self, test_id: str):
        """Start individual test metrics collection."""
        self.test_start_times[test_id] = time.time()

    def end_test(
        self,
        test_id: str,
        test_name: str,
        test_file: str,
        status: MetricStatus,
        error_message: str = None,
        stack_trace: str = None,
    ):
        """End individual test metrics collection."""
        end_time = time.time()
        start_time = self.test_start_times.get(test_id, end_time)
        duration = end_time - start_time

        # Get memory and CPU usage (simplified)
        memory_usage = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()

        metric = ExecutionMetric(
            test_id=test_id,
            test_name=test_name,
            test_file=test_file,
            status=status,
            duration=duration,
            setup_duration=0.0,  # Could be enhanced to track setup/teardown
            teardown_duration=0.0,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            stack_trace=stack_trace,
            timestamp=datetime.now(),
            test_run_id=self.test_run_id,
        )

        self.current_metrics.append(metric)

    def end_test_suite(self) -> SuiteMetrics:
        """End test suite metrics collection."""
        end_time = time.time()
        total_duration = end_time - (self.suite_start_time or end_time)

        # Count test statuses
        status_counts = Counter(metric.status for metric in self.current_metrics)

        suite_metrics = SuiteMetrics(
            test_run_id=self.test_run_id,
            total_tests=len(self.current_metrics),
            passed_tests=status_counts[MetricStatus.PASSED],
            failed_tests=status_counts[MetricStatus.FAILED],
            skipped_tests=status_counts[MetricStatus.SKIPPED],
            error_tests=status_counts[MetricStatus.ERROR],
            total_duration=total_duration,
            setup_duration=0.0,
            teardown_duration=0.0,
            parallel_workers=1,  # Could be enhanced to detect parallel workers
            environment=self.environment,
            timestamp=datetime.now(),
            test_metrics=self.current_metrics,
        )

        # Store in database
        self._store_suite_metrics(suite_metrics)

        # Analyze for flaky and slow tests
        self._analyze_test_patterns()

        return suite_metrics

    def _store_suite_metrics(self, suite_metrics: SuiteMetrics):
        """Store suite metrics in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert suite metrics
        cursor.execute(
            """
            INSERT OR REPLACE INTO test_runs
            (test_run_id, timestamp, total_tests, passed_tests, failed_tests,
             skipped_tests, error_tests, total_duration, setup_duration,
             teardown_duration, parallel_workers, environment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                suite_metrics.test_run_id,
                suite_metrics.timestamp.isoformat(),
                suite_metrics.total_tests,
                suite_metrics.passed_tests,
                suite_metrics.failed_tests,
                suite_metrics.skipped_tests,
                suite_metrics.error_tests,
                suite_metrics.total_duration,
                suite_metrics.setup_duration,
                suite_metrics.teardown_duration,
                suite_metrics.parallel_workers,
                suite_metrics.environment,
            ),
        )

        run_id = cursor.lastrowid

        # Insert individual test metrics
        for metric in suite_metrics.test_metrics:
            cursor.execute(
                """
                INSERT INTO test_metrics
                (run_id, test_id, test_name, test_file, status, duration,
                 setup_duration, teardown_duration, memory_usage, cpu_usage,
                 error_message, stack_trace, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    metric.test_id,
                    metric.test_name,
                    metric.test_file,
                    metric.status.value,
                    metric.duration,
                    metric.setup_duration,
                    metric.teardown_duration,
                    metric.memory_usage,
                    metric.cpu_usage,
                    metric.error_message,
                    metric.stack_trace,
                    metric.timestamp.isoformat(),
                ),
            )

        conn.commit()
        conn.close()

    def _analyze_test_patterns(self):
        """Analyze test patterns to identify flaky and slow tests."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Analyze flaky tests (tests that pass sometimes and fail sometimes)
        cursor.execute(
            """
            SELECT test_id, test_name, test_file,
                   COUNT(*) as total_runs,
                   COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_runs,
                   COUNT(CASE WHEN status = 'passed' THEN 1 END) as passed_runs,
                   MIN(timestamp) as first_run,
                   MAX(timestamp) as last_run
            FROM test_metrics
            WHERE test_id IN (
                SELECT test_id FROM test_metrics
                GROUP BY test_id
                HAVING COUNT(DISTINCT status) > 1
            )
            GROUP BY test_id, test_name, test_file
        """
        )

        for row in cursor.fetchall():
            (
                test_id,
                test_name,
                test_file,
                total_runs,
                failed_runs,
                passed_runs,
                first_run,
                last_run,
            ) = row

            if failed_runs > 0 and passed_runs > 0:
                flaky_percentage = (failed_runs / total_runs) * 100

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO flaky_tests
                    (test_id, test_name, test_file, first_flaky_run, last_flaky_run,
                     flaky_count, total_runs, flaky_percentage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        test_id,
                        test_name,
                        test_file,
                        first_run,
                        last_run,
                        failed_runs,
                        total_runs,
                        flaky_percentage,
                    ),
                )

        # Analyze slow tests (tests that take longer than average)
        cursor.execute(
            """
            SELECT test_id, test_name, test_file,
                   AVG(duration) as avg_duration,
                   MAX(duration) as max_duration,
                   COUNT(*) as run_count,
                   MIN(timestamp) as first_detected,
                   MAX(timestamp) as last_detected
            FROM test_metrics
            WHERE duration > (
                SELECT AVG(duration) * 2 FROM test_metrics
            )
            GROUP BY test_id, test_name, test_file
        """
        )

        for row in cursor.fetchall():
            (
                test_id,
                test_name,
                test_file,
                avg_duration,
                max_duration,
                run_count,
                first_detected,
                last_detected,
            ) = row

            cursor.execute(
                """
                INSERT OR REPLACE INTO slow_tests
                (test_id, test_name, test_file, avg_duration, max_duration,
                 slow_run_count, first_detected, last_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    test_id,
                    test_name,
                    test_file,
                    avg_duration,
                    max_duration,
                    run_count,
                    first_detected,
                    last_detected,
                ),
            )

        conn.commit()
        conn.close()

    def _get_memory_usage(self) -> Optional[int]:
        """Get current memory usage (simplified implementation)."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return None

    def _get_cpu_usage(self) -> Optional[float]:
        """Get current CPU usage (simplified implementation)."""
        try:
            import psutil

            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return None

    def get_flaky_tests(self, min_flaky_percentage: float = 10.0) -> List[Dict[str, Any]]:
        """Get flaky tests above the minimum percentage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT test_id, test_name, test_file, flaky_count, total_runs,
                   flaky_percentage, first_flaky_run, last_flaky_run
            FROM flaky_tests
            WHERE flaky_percentage >= ? AND resolved_at IS NULL
            ORDER BY flaky_percentage DESC
        """,
            (min_flaky_percentage,),
        )

        flaky_tests = []
        for row in cursor.fetchall():
            flaky_tests.append(
                {
                    "test_id": row[0],
                    "test_name": row[1],
                    "test_file": row[2],
                    "flaky_count": row[3],
                    "total_runs": row[4],
                    "flaky_percentage": row[5],
                    "first_flaky_run": row[6],
                    "last_flaky_run": row[7],
                }
            )

        conn.close()
        return flaky_tests

    def get_slow_tests(self, min_duration: float = 1.0) -> List[Dict[str, Any]]:
        """Get slow tests above the minimum duration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT test_id, test_name, test_file, avg_duration, max_duration,
                   slow_run_count, first_detected, last_detected
            FROM slow_tests
            WHERE avg_duration >= ?
            ORDER BY avg_duration DESC
        """,
            (min_duration,),
        )

        slow_tests = []
        for row in cursor.fetchall():
            slow_tests.append(
                {
                    "test_id": row[0],
                    "test_name": row[1],
                    "test_file": row[2],
                    "avg_duration": row[3],
                    "max_duration": row[4],
                    "slow_run_count": row[5],
                    "first_detected": row[6],
                    "last_detected": row[7],
                }
            )

        conn.close()
        return slow_tests

    def get_test_trends(self, days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Get test execution trends over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        # Get pass rate trends
        cursor.execute(
            """
            SELECT DATE(timestamp) as date,
                   COUNT(*) as total_tests,
                   SUM(CASE WHEN passed_tests > 0 THEN 1 ELSE 0 END) as passed_runs,
                   AVG(CAST(passed_tests AS REAL) / total_tests * 100) as pass_rate
            FROM test_runs
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """,
            (since_date,),
        )

        pass_rate_trends = []
        for row in cursor.fetchall():
            pass_rate_trends.append(
                {
                    "date": row[0],
                    "total_tests": row[1],
                    "passed_runs": row[2],
                    "pass_rate": row[3],
                }
            )

        # Get duration trends
        cursor.execute(
            """
            SELECT DATE(timestamp) as date,
                   AVG(total_duration) as avg_duration,
                   MAX(total_duration) as max_duration,
                   MIN(total_duration) as min_duration
            FROM test_runs
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """,
            (since_date,),
        )

        duration_trends = []
        for row in cursor.fetchall():
            duration_trends.append(
                {
                    "date": row[0],
                    "avg_duration": row[1],
                    "max_duration": row[2],
                    "min_duration": row[3],
                }
            )

        conn.close()
        return {
            "pass_rate_trends": pass_rate_trends,
            "duration_trends": duration_trends,
        }

    def generate_metrics_report(self, output_path: str = "tests/reporting/metrics_report.html"):
        """Generate HTML metrics report."""
        flaky_tests = self.get_flaky_tests()
        slow_tests = self.get_slow_tests()
        self.get_test_trends()

        # Get latest test run
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT test_run_id, timestamp, total_tests, passed_tests, failed_tests,
                   skipped_tests, error_tests, total_duration
            FROM test_runs
            ORDER BY timestamp DESC LIMIT 1
        """
        )
        latest_run = cursor.fetchone()
        conn.close()

        if not latest_run:
            logger.warning("No test run data found")
            return

        pass_rate = (latest_run[3] / latest_run[2]) * 100 if latest_run[2] > 0 else 0

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FreeAgentics Test Metrics Report</title>
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
                .flaky {{ background-color: #ffcdd2; }}
                .slow {{ background-color: #ffecb3; }}
                .chart {{ width: 100%; height: 300px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>FreeAgentics Test Metrics Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Latest Run: {latest_run[0]} ({latest_run[1]})</p>

                <div class="metric {'critical' if pass_rate < 50 else 'warning' if pass_rate < 80 else 'good'}">
                    <strong>Pass Rate: {pass_rate:.1f}%</strong>
                </div>
                <div class="metric">
                    <strong>Total Tests: {latest_run[2]}</strong>
                </div>
                <div class="metric">
                    <strong>Duration: {latest_run[7]:.1f}s</strong>
                </div>
                <div class="metric {'warning' if len(flaky_tests) > 0 else 'good'}">
                    <strong>Flaky Tests: {len(flaky_tests)}</strong>
                </div>
                <div class="metric {'warning' if len(slow_tests) > 0 else 'good'}">
                    <strong>Slow Tests: {len(slow_tests)}</strong>
                </div>
            </div>

            <h2>Flaky Tests ({len(flaky_tests)})</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>File</th>
                    <th>Flaky %</th>
                    <th>Failed/Total</th>
                    <th>First Detected</th>
                </tr>
        """

        for test in flaky_tests:
            html_content += f"""
                <tr class="flaky">
                    <td>{test['test_name']}</td>
                    <td>{test['test_file']}</td>
                    <td>{test['flaky_percentage']:.1f}%</td>
                    <td>{test['flaky_count']}/{test['total_runs']}</td>
                    <td>{test['first_flaky_run']}</td>
                </tr>
            """

        html_content += f"""
            </table>

            <h2>Slow Tests ({len(slow_tests)})</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>File</th>
                    <th>Avg Duration</th>
                    <th>Max Duration</th>
                    <th>Run Count</th>
                </tr>
        """

        for test in slow_tests:
            html_content += f"""
                <tr class="slow">
                    <td>{test['test_name']}</td>
                    <td>{test['test_file']}</td>
                    <td>{test['avg_duration']:.2f}s</td>
                    <td>{test['max_duration']:.2f}s</td>
                    <td>{test['slow_run_count']}</td>
                </tr>
            """

        html_content += """
            </table>

            <h2>Test Execution Status</h2>
            <table>
                <tr>
                    <th>Status</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """

        total_tests = latest_run[2]
        statuses = [
            ("Passed", latest_run[3], "good"),
            ("Failed", latest_run[4], "critical"),
            ("Skipped", latest_run[5], "warning"),
            ("Error", latest_run[6], "critical"),
        ]

        for status, count, css_class in statuses:
            percentage = (count / total_tests) * 100 if total_tests > 0 else 0
            html_content += f"""
                <tr class="{css_class}">
                    <td>{status}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
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

        logger.info(f"HTML metrics report generated: {output_path}")
        return output_path

    def export_metrics_json(self, output_path: str = "tests/reporting/metrics_data.json"):
        """Export metrics data as JSON."""
        data = {
            "flaky_tests": self.get_flaky_tests(),
            "slow_tests": self.get_slow_tests(),
            "trends": self.get_test_trends(),
        }

        # Get latest test run
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT test_run_id, timestamp, total_tests, passed_tests, failed_tests,
                   skipped_tests, error_tests, total_duration, environment
            FROM test_runs
            ORDER BY timestamp DESC LIMIT 1
        """
        )
        latest_run = cursor.fetchone()

        if latest_run:
            data["latest_run"] = {
                "test_run_id": latest_run[0],
                "timestamp": latest_run[1],
                "total_tests": latest_run[2],
                "passed_tests": latest_run[3],
                "failed_tests": latest_run[4],
                "skipped_tests": latest_run[5],
                "error_tests": latest_run[6],
                "total_duration": latest_run[7],
                "environment": latest_run[8],
            }

        conn.close()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"JSON metrics data exported: {output_path}")
        return output_path

    def cleanup_old_metrics(self, days: int = 30):
        """Clean up old metrics data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        # Delete old test metrics
        cursor.execute(
            """
            DELETE FROM test_metrics
            WHERE run_id IN (
                SELECT id FROM test_runs
                WHERE timestamp < ?
            )
        """,
            (cutoff_date,),
        )

        # Delete old test runs
        cursor.execute(
            """
            DELETE FROM test_runs
            WHERE timestamp < ?
        """,
            (cutoff_date,),
        )

        conn.commit()
        conn.close()

        logger.info(f"Cleaned up metrics data older than {days} days")

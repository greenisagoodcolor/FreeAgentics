"""Performance monitoring utilities for database operations."""

import json
import logging
import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import psutil
from sqlalchemy import text

from .pool_config import get_pool

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and record database performance metrics."""

    def __init__(self, test_name: str, pool_name: str = "default"):
        """Initialize performance monitor for a test run."""
        self.test_name = test_name
        self.pool_name = pool_name
        self.test_run_id = None
        self.start_time = None
        self.metrics = defaultdict(list)
        self.operation_timings = defaultdict(list)
        self.pool = None

        # System monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.system_metrics = []

    def start_test_run(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new test run and return the run ID."""
        self.pool = get_pool(self.pool_name)
        self.start_time = time.time()

        with self.pool.get_session() as session:
            result = session.execute(
                text(
                    """
                    INSERT INTO test_runs (test_name, start_time, metadata)
                    VALUES (:test_name, :start_time, :metadata)
                    RETURNING id
                """
                ),
                {
                    "test_name": self.test_name,
                    "start_time": datetime.now(),
                    "metadata": json.dumps(metadata or {}),
                },
            )

            self.test_run_id = result.scalar()

        # Start system monitoring
        self._start_system_monitoring()

        logger.info(f"Started test run: {self.test_run_id}")
        return self.test_run_id

    def end_test_run(self, status: str = "COMPLETED") -> Dict[str, Any]:
        """End the test run and return summary statistics."""
        if not self.test_run_id:
            raise RuntimeError("No active test run")

        # Stop system monitoring
        self._stop_system_monitoring()

        end_time = time.time()
        duration = end_time - self.start_time

        # Calculate summary statistics
        summary = self._calculate_summary()

        # Update test run record
        with self.pool.get_connection() as (conn, cursor):
            cursor.execute(
                """
                UPDATE test_runs
                SET end_time = %s,
                    status = %s,
                    total_operations = %s,
                    metadata = metadata || %s
                WHERE id = %s
            """,
                (
                    datetime.now(),
                    status,
                    summary.get("total_operations", 0),
                    json.dumps({"summary": summary}),
                    self.test_run_id,
                ),
            )

        # Save all collected metrics
        self._save_metrics()

        logger.info(
            f"Ended test run: {self.test_run_id} (duration: {duration:.2f}s)"
        )

        return summary

    @contextmanager
    def measure_operation(self, operation_type: str, operation_name: str):
        """Context manager to measure operation timing."""
        start_time = time.time()

        try:
            yield
            success = True
        except Exception as e:
            success = False
            logger.error(f"Operation failed: {operation_name} - {e}")
            raise
        finally:
            duration = time.time() - start_time

            self.operation_timings[operation_type].append(
                {
                    "name": operation_name,
                    "duration": duration,
                    "success": success,
                    "timestamp": datetime.now(),
                }
            )

            # Record metric
            self.record_metric(
                f"{operation_type}_duration",
                operation_name,
                duration,
                "seconds",
            )

    def record_metric(
        self,
        metric_type: str,
        metric_name: str,
        value: float,
        unit: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a performance metric."""
        if not self.test_run_id:
            raise RuntimeError("No active test run")

        metric = {
            "test_run_id": self.test_run_id,
            "metric_type": metric_type,
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now(),
            "metadata": metadata,
        }

        self.metrics[metric_type].append(metric)

    def measure_query_performance(
        self, query: str, params: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """Execute a query and measure its performance."""
        with self.pool.get_connection() as (conn, cursor):
            # Enable timing
            cursor.execute("SET log_duration = ON")

            # Get query plan
            cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS) {query}", params)
            plan = cursor.fetchall()

            # Execute actual query
            start_time = time.time()
            cursor.execute(query, params)

            if cursor.description:
                results = cursor.fetchall()
                row_count = len(results)
            else:
                results = None
                row_count = cursor.rowcount

            duration = time.time() - start_time

            # Record metrics
            self.record_metric(
                "query_performance",
                query.split()[0],  # SELECT, INSERT, etc.
                duration,
                "seconds",
                {
                    "row_count": row_count,
                    "query": query[:100],
                },  # First 100 chars
            )

            return {
                "duration": duration,
                "row_count": row_count,
                "plan": plan,
                "results": results,
            }

    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics."""
        stats = {}

        with self.pool.get_connection() as (conn, cursor):
            # Database size
            cursor.execute(
                """
                SELECT pg_database_size(current_database()) as size
            """
            )
            stats["database_size_mb"] = cursor.fetchone()["size"] / (
                1024 * 1024
            )

            # Table sizes
            cursor.execute(
                """
                SELECT
                    relname as table_name,
                    pg_size_pretty(pg_total_relation_size(relid)) as total_size,
                    pg_total_relation_size(relid) as size_bytes
                FROM pg_stat_user_tables
                ORDER BY pg_total_relation_size(relid) DESC
            """
            )
            stats["table_sizes"] = cursor.fetchall()

            # Connection stats
            cursor.execute(
                """
                SELECT
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity
                WHERE datname = current_database()
            """
            )
            stats["connections"] = cursor.fetchone()

            # Cache hit ratio
            cursor.execute(
                """
                SELECT
                    sum(blks_hit) / nullif(sum(blks_hit + blks_read), 0) as cache_hit_ratio
                FROM pg_stat_database
                WHERE datname = current_database()
            """
            )
            stats["cache_hit_ratio"] = cursor.fetchone()["cache_hit_ratio"]

            # Slow queries
            cursor.execute(
                """
                SELECT
                    query,
                    calls,
                    mean_exec_time,
                    total_exec_time
                FROM pg_stat_statements
                WHERE query NOT LIKE '%pg_stat%'
                ORDER BY mean_exec_time DESC
                LIMIT 10
            """
            )
            stats["slow_queries"] = (
                cursor.fetchall() if cursor.rowcount > 0 else []
            )

        # Pool statistics
        stats["connection_pool"] = self.pool.get_pool_stats()

        return stats

    def _start_system_monitoring(self):
        """Start monitoring system resources."""

        def monitor():
            while not self.stop_monitoring.is_set():
                metrics = {
                    "timestamp": datetime.now(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_mb": psutil.virtual_memory().used / (1024 * 1024),
                    "disk_io": (
                        psutil.disk_io_counters()._asdict()
                        if psutil.disk_io_counters()
                        else {}
                    ),
                    "net_io": (
                        psutil.net_io_counters()._asdict()
                        if psutil.net_io_counters()
                        else {}
                    ),
                }

                self.system_metrics.append(metrics)

                # Also record as performance metrics
                self.record_metric(
                    "system", "cpu_usage", metrics["cpu_percent"], "percent"
                )
                self.record_metric(
                    "system",
                    "memory_usage",
                    metrics["memory_percent"],
                    "percent",
                )

                time.sleep(5)  # Monitor every 5 seconds

        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()

    def _stop_system_monitoring(self):
        """Stop monitoring system resources."""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=10)

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for the test run."""
        summary = {
            "duration": time.time() - self.start_time,
            "total_operations": sum(
                len(ops) for ops in self.operation_timings.values()
            ),
        }

        # Operation timing statistics
        timing_stats = {}
        for op_type, timings in self.operation_timings.items():
            if timings:
                durations = [t["duration"] for t in timings]
                timing_stats[op_type] = {
                    "count": len(timings),
                    "total_time": sum(durations),
                    "min_time": min(durations),
                    "max_time": max(durations),
                    "avg_time": statistics.mean(durations),
                    "median_time": statistics.median(durations),
                    "success_rate": sum(1 for t in timings if t["success"])
                    / len(timings),
                }

        summary["operation_timings"] = timing_stats

        # System resource statistics
        if self.system_metrics:
            cpu_values = [m["cpu_percent"] for m in self.system_metrics]
            memory_values = [m["memory_percent"] for m in self.system_metrics]

            summary["system_resources"] = {
                "cpu": {
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "avg": statistics.mean(cpu_values),
                },
                "memory": {
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "avg": statistics.mean(memory_values),
                },
            }

        # Database statistics
        summary["database_stats"] = self.get_database_stats()

        return summary

    def _save_metrics(self):
        """Save all collected metrics to the database."""
        if not self.metrics:
            return

        all_metrics = []
        for metric_type, metrics_list in self.metrics.items():
            for metric in metrics_list:
                all_metrics.append(
                    (
                        metric["test_run_id"],
                        metric["metric_type"],
                        metric["metric_name"],
                        metric["value"],
                        metric.get("unit"),
                        metric["timestamp"],
                        json.dumps(metric.get("metadata", {})),
                    )
                )

        with self.pool.get_connection() as (conn, cursor):
            cursor.executemany(
                """
                INSERT INTO performance_metrics
                (test_run_id, metric_type, metric_name, value, unit, timestamp, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
                all_metrics,
            )

        logger.info(f"Saved {len(all_metrics)} performance metrics")

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a performance report."""
        if not self.test_run_id:
            raise RuntimeError("No test run to report on")

        summary = self._calculate_summary()

        report = """
# Performance Test Report

**Test Name:** {self.test_name}
**Test Run ID:** {self.test_run_id}
**Duration:** {summary['duration']:.2f} seconds
**Total Operations:** {summary['total_operations']}

## Operation Timings

"""

        for op_type, stats in summary["operation_timings"].items():
            report += """
### {op_type}
- Count: {stats['count']}
- Total Time: {stats['total_time']:.2f}s
- Average Time: {stats['avg_time']:.3f}s
- Min/Max Time: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s
- Success Rate: {stats['success_rate']*100:.1f}%
"""

        if "system_resources" in summary:
            report += """
## System Resources

### CPU Usage
- Average: {:.1f}%
- Min/Max: {:.1f}% / {:.1f}%

### Memory Usage
- Average: {:.1f}%
- Min/Max: {:.1f}% / {:.1f}%
""".format(
                summary["system_resources"]["cpu"]["avg"],
                summary["system_resources"]["cpu"]["min"],
                summary["system_resources"]["cpu"]["max"],
                summary["system_resources"]["memory"]["avg"],
                summary["system_resources"]["memory"]["min"],
                summary["system_resources"]["memory"]["max"],
            )

        report += """
## Database Statistics

- Database Size: {summary['database_stats']['database_size_mb']:.1f} MB
- Cache Hit Ratio: {summary['database_stats']['cache_hit_ratio']*100:.1f}%

### Connection Pool
- Current Connections: {summary['database_stats']['connection_pool']['current_size']}
- Available Connections: {summary['database_stats']['connection_pool']['available']}
"""

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"Report saved to: {output_file}")

        return report


class LoadTestRunner:
    """Run load tests with performance monitoring."""

    def __init__(self, test_name: str):
        """Initialize load test runner."""
        self.test_name = test_name
        self.monitor = PerformanceMonitor(test_name)

    def run_concurrent_test(
        self,
        test_function: Callable,
        num_threads: int = 10,
        duration_seconds: int = 60,
        ramp_up_seconds: int = 10,
    ) -> Dict[str, Any]:
        """Run a concurrent load test."""
        import concurrent.futures

        logger.info(
            f"Starting concurrent test: {num_threads} threads for {duration_seconds}s"
        )

        # Start monitoring
        self.monitor.start_test_run(
            {
                "test_type": "concurrent",
                "num_threads": num_threads,
                "duration_seconds": duration_seconds,
                "ramp_up_seconds": ramp_up_seconds,
            }
        )

        stop_event = threading.Event()
        results = defaultdict(int)

        def worker(thread_id: int):
            # Ramp up delay
            time.sleep(thread_id * (ramp_up_seconds / num_threads))

            while not stop_event.is_set():
                try:
                    with self.monitor.measure_operation(
                        "concurrent_operation", f"thread_{thread_id}"
                    ):
                        test_function()
                    results["success"] += 1
                except Exception as e:
                    results["failure"] += 1
                    logger.error(f"Thread {thread_id} error: {e}")

        # Start threads
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
        ) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]

            # Run for specified duration
            time.sleep(duration_seconds)
            stop_event.set()

            # Wait for completion
            concurrent.futures.wait(futures)

        # End monitoring and get summary
        summary = self.monitor.end_test_run()
        summary["results"] = dict(results)

        return summary

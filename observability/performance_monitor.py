"""
Comprehensive Performance Monitoring System for FreeAgentics.
Tracks system performance metrics, identifies bottlenecks, and provides optimization insights.
"""

import gc
import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_rss_mb: float = 0.0
    memory_vms_mb: float = 0.0

    # Thread metrics
    thread_count: int = 0
    gil_contention: float = 0.0
    context_switches: int = 0

    # Database metrics
    db_connections: int = 0
    db_query_time_ms: float = 0.0
    db_connection_pool_size: int = 0

    # API metrics
    api_requests_per_second: float = 0.0
    api_response_time_ms: float = 0.0
    api_error_rate: float = 0.0

    # Agent metrics
    agent_count: int = 0
    agent_step_time_ms: float = 0.0
    agent_memory_mb: float = 0.0

    # WebSocket metrics
    websocket_connections: int = 0
    websocket_messages_per_second: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceAlert:
    """Performance alert when thresholds are exceeded."""

    metric: str
    current_value: float
    threshold: float
    severity: str  # 'warning', 'critical'
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque[PerformanceMetrics] = deque(
            maxlen=1000
        )  # Keep last 1000 measurements
        self.alerts: deque[PerformanceAlert] = deque(maxlen=100)  # Keep last 100 alerts
        self.is_monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()

        # Performance thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 80.0, "critical": 95.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "api_response_time_ms": {"warning": 500.0, "critical": 2000.0},
            "db_query_time_ms": {"warning": 100.0, "critical": 500.0},
            "agent_step_time_ms": {"warning": 50.0, "critical": 200.0},
            "gil_contention": {"warning": 0.7, "critical": 0.9},
            "api_error_rate": {"warning": 5.0, "critical": 10.0},
        }

        # Metric accumulators
        self.request_times: deque[float] = deque(maxlen=100)
        self.db_query_times: deque[float] = deque(maxlen=100)
        self.agent_step_times: deque[float] = deque(maxlen=100)
        self.api_requests: deque[float] = deque(maxlen=100)
        self.websocket_messages: deque[float] = deque(maxlen=100)

        # Thread-local storage for request timing
        self.local = threading.local()

        # GIL measurement cache
        self._last_gil_measurement: float = 0.0
        self._cached_gil_contention: float = 0.0

        logger.info("Performance monitor initialized")

    def start_monitoring(self):
        """Start the performance monitoring thread."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop the performance monitoring thread."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Performance monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self._check_alerts(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        metrics = PerformanceMetrics()

        # System metrics
        metrics.cpu_usage = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        metrics.memory_rss_mb = memory_info.rss / 1024 / 1024
        metrics.memory_vms_mb = memory_info.vms / 1024 / 1024

        # Memory percentage
        system_memory = psutil.virtual_memory()
        metrics.memory_usage = system_memory.percent

        # Thread metrics
        metrics.thread_count = self.process.num_threads()
        metrics.context_switches = sum(self.process.num_ctx_switches())

        # Database metrics
        metrics.db_connections = getattr(self, "_db_connections", 0)
        metrics.db_query_time_ms = self._get_average_time(self.db_query_times)
        metrics.db_connection_pool_size = getattr(self, "_db_pool_size", 0)

        # API metrics
        current_time = time.time()
        recent_requests = [t for t in self.api_requests if current_time - t < 60]
        metrics.api_requests_per_second = len(recent_requests) / 60.0
        metrics.api_response_time_ms = self._get_average_time(self.request_times)
        metrics.api_error_rate = getattr(self, "_api_error_rate", 0.0)

        # Agent metrics
        metrics.agent_count = getattr(self, "_agent_count", 0)
        metrics.agent_step_time_ms = self._get_average_time(self.agent_step_times)
        metrics.agent_memory_mb = getattr(self, "_agent_memory_mb", 0.0)

        # WebSocket metrics
        metrics.websocket_connections = getattr(self, "_websocket_connections", 0)
        recent_messages = [t for t in self.websocket_messages if current_time - t < 60]
        metrics.websocket_messages_per_second = len(recent_messages) / 60.0

        # GIL contention (periodically measured)
        if (
            not hasattr(self, "_last_gil_measurement")
            or current_time - self._last_gil_measurement > 30
        ):
            metrics.gil_contention = self._measure_gil_contention()
            self._last_gil_measurement = current_time
        else:
            metrics.gil_contention = getattr(self, "_cached_gil_contention", 0.0)

        return metrics

    def _get_average_time(self, times: deque[float]) -> float:
        """Get average time from a deque of times."""
        if not times:
            return 0.0
        return float(sum(times) / len(times))

    def _measure_gil_contention(self) -> float:
        """Measure GIL contention using CPU-bound workload."""
        try:
            import math

            def cpu_work():
                result = 0
                for i in range(100000):
                    result += math.sqrt(i)
                return result

            # Single-threaded baseline
            start = time.perf_counter()
            for _ in range(4):
                cpu_work()
            single_time = time.perf_counter() - start

            # Multi-threaded test
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=4) as executor:
                start = time.perf_counter()
                futures = [executor.submit(cpu_work) for _ in range(4)]
                for future in futures:
                    future.result()
                multi_time = time.perf_counter() - start

            gil_contention = multi_time / single_time
            self._cached_gil_contention = gil_contention
            return gil_contention

        except Exception as e:
            logger.warning(f"Could not measure GIL contention: {e}")
            return 0.0

    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts."""
        for metric_name, thresholds in self.thresholds.items():
            current_value = getattr(metrics, metric_name, 0.0)

            if current_value >= thresholds["critical"]:
                self._add_alert(
                    metric_name,
                    current_value,
                    thresholds["critical"],
                    "critical",
                )
            elif current_value >= thresholds["warning"]:
                self._add_alert(
                    metric_name,
                    current_value,
                    thresholds["warning"],
                    "warning",
                )

    def _add_alert(
        self,
        metric: str,
        current_value: float,
        threshold: float,
        severity: str,
    ):
        """Add a performance alert."""
        message = (
            f"{metric} is {current_value:.2f}, exceeding {severity} threshold of {threshold:.2f}"
        )

        alert = PerformanceAlert(
            metric=metric,
            current_value=current_value,
            threshold=threshold,
            severity=severity,
            message=message,
        )

        self.alerts.append(alert)
        logger.warning(f"Performance alert: {message}")

    # Context managers for timing operations

    @contextmanager
    def time_api_request(self):
        """Context manager for timing API requests."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.request_times.append(elapsed)
            self.api_requests.append(time.time())

    @contextmanager
    def time_db_query(self):
        """Context manager for timing database queries."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.db_query_times.append(elapsed)

    @contextmanager
    def time_agent_step(self):
        """Context manager for timing agent steps."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.agent_step_times.append(elapsed)

    # Metric update methods

    def update_db_connections(self, count: int):
        """Update database connection count."""
        self._db_connections = count

    def update_db_pool_size(self, size: int):
        """Update database connection pool size."""
        self._db_pool_size = size

    def update_agent_count(self, count: int):
        """Update agent count."""
        self._agent_count = count

    def update_agent_memory(self, memory_mb: float):
        """Update agent memory usage."""
        self._agent_memory_mb = memory_mb

    def update_websocket_connections(self, count: int):
        """Update WebSocket connection count."""
        self._websocket_connections = count

    def log_websocket_message(self):
        """Log a WebSocket message."""
        self.websocket_messages.append(time.time())

    def update_api_error_rate(self, error_rate: float):
        """Update API error rate."""
        self._api_error_rate = error_rate

    # Reporting methods

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_metrics_history(self, minutes: int = 10) -> List[PerformanceMetrics]:
        """Get metrics history for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_alerts(self, severity: Optional[str] = None) -> List[PerformanceAlert]:
        """Get recent alerts, optionally filtered by severity."""
        alerts = list(self.alerts)
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        current_metrics = self.get_current_metrics()
        if not current_metrics:
            return {"error": "No metrics available"}

        # Calculate averages over the last 10 minutes
        recent_metrics = self.get_metrics_history(10)

        if not recent_metrics:
            return {"error": "No recent metrics available"}

        def avg_metric(metric_name: str) -> float:
            values = [getattr(m, metric_name) for m in recent_metrics]
            return float(sum(values) / len(values)) if values else 0.0

        report = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "memory_rss_mb": current_metrics.memory_rss_mb,
                "thread_count": current_metrics.thread_count,
                "agent_count": current_metrics.agent_count,
                "api_requests_per_second": current_metrics.api_requests_per_second,
                "api_response_time_ms": current_metrics.api_response_time_ms,
                "db_query_time_ms": current_metrics.db_query_time_ms,
                "websocket_connections": current_metrics.websocket_connections,
                "gil_contention": current_metrics.gil_contention,
            },
            "averages_10min": {
                "cpu_usage": avg_metric("cpu_usage"),
                "memory_usage": avg_metric("memory_usage"),
                "api_response_time_ms": avg_metric("api_response_time_ms"),
                "db_query_time_ms": avg_metric("db_query_time_ms"),
                "agent_step_time_ms": avg_metric("agent_step_time_ms"),
                "api_requests_per_second": avg_metric("api_requests_per_second"),
            },
            "alerts": {
                "total": len(self.alerts),
                "critical": len([a for a in self.alerts if a.severity == "critical"]),
                "warning": len([a for a in self.alerts if a.severity == "warning"]),
                "recent": [
                    {
                        "metric": a.metric,
                        "severity": a.severity,
                        "message": a.message,
                        "timestamp": a.timestamp.isoformat(),
                    }
                    for a in list(self.alerts)[-5:]  # Last 5 alerts
                ],
            },
            "performance_insights": self._generate_insights(recent_metrics),
        }

        return report

    def _generate_insights(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Generate performance insights based on metrics."""
        insights: List[str] = []

        if not metrics:
            return insights

        # CPU usage insights
        cpu_values = [m.cpu_usage for m in metrics]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        if avg_cpu > 70:
            insights.append(
                f"High CPU usage detected (avg: {avg_cpu:.1f}%). Consider optimizing CPU-intensive operations."
            )

        # Memory usage insights
        memory_values = [m.memory_usage for m in metrics]
        avg_memory = sum(memory_values) / len(memory_values)
        if avg_memory > 80:
            insights.append(
                f"High memory usage detected (avg: {avg_memory:.1f}%). Check for memory leaks."
            )

        # API response time insights
        api_times = [m.api_response_time_ms for m in metrics if m.api_response_time_ms > 0]
        if api_times:
            avg_api_time = sum(api_times) / len(api_times)
            if avg_api_time > 200:
                insights.append(
                    f"Slow API responses detected (avg: {avg_api_time:.1f}ms). Consider caching or optimization."
                )

        # Database query insights
        db_times = [m.db_query_time_ms for m in metrics if m.db_query_time_ms > 0]
        if db_times:
            avg_db_time = sum(db_times) / len(db_times)
            if avg_db_time > 50:
                insights.append(
                    f"Slow database queries detected (avg: {avg_db_time:.1f}ms). Consider query optimization."
                )

        # Threading insights
        thread_counts = [m.thread_count for m in metrics]
        avg_threads = sum(thread_counts) / len(thread_counts)
        if avg_threads > 50:
            insights.append(
                f"High thread count detected (avg: {avg_threads:.1f}). Consider thread pool optimization."
            )

        # GIL contention insights
        gil_values = [m.gil_contention for m in metrics if m.gil_contention > 0]
        if gil_values:
            avg_gil = sum(gil_values) / len(gil_values)
            if avg_gil > 0.8:
                insights.append(
                    f"High GIL contention detected ({avg_gil:.2f}x). Consider using asyncio or multiprocessing."
                )

        return insights

    def force_garbage_collection(self):
        """Force garbage collection and report statistics."""
        before_objects = len(gc.get_objects())
        before_memory = self.process.memory_info().rss / 1024 / 1024

        # Force collection
        collected = gc.collect()

        after_objects = len(gc.get_objects())
        after_memory = self.process.memory_info().rss / 1024 / 1024

        gc_stats = {
            "objects_before": before_objects,
            "objects_after": after_objects,
            "objects_collected": collected,
            "memory_before_mb": before_memory,
            "memory_after_mb": after_memory,
            "memory_freed_mb": before_memory - after_memory,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Garbage collection: {collected} objects collected, "
            f"{gc_stats['memory_freed_mb']:.2f}MB freed"
        )

        return gc_stats


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor


# Convenience functions
def start_performance_monitoring():
    """Start performance monitoring."""
    performance_monitor.start_monitoring()


def stop_performance_monitoring():
    """Stop performance monitoring."""
    performance_monitor.stop_monitoring()


def get_performance_report() -> Dict[str, Any]:
    """Get the current performance report."""
    return performance_monitor.get_performance_report()

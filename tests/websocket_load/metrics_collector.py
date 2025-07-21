"""Metrics collection and analysis for WebSocket load testing."""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


@dataclass
class WebSocketMetrics:
    """Container for WebSocket performance metrics."""

    # Connection metrics
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    active_connections: int = 0
    connection_duration_seconds: List[float] = field(default_factory=list)

    # Message metrics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    message_send_failures: int = 0

    # Latency metrics (in milliseconds)
    latencies_ms: List[float] = field(default_factory=list)
    latency_min_ms: float = float("inf")
    latency_max_ms: float = 0.0
    latency_avg_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Throughput metrics
    messages_per_second_sent: float = 0.0
    messages_per_second_received: float = 0.0
    bytes_per_second_sent: float = 0.0
    bytes_per_second_received: float = 0.0

    # Error metrics
    total_errors: int = 0
    connection_errors: int = 0
    send_errors: int = 0
    receive_errors: int = 0
    timeout_errors: int = 0

    # Time-based metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: float = 0.0

    def calculate_statistics(self):
        """Calculate derived statistics from raw metrics."""
        if self.latencies_ms:
            latencies_array = np.array(self.latencies_ms)
            self.latency_min_ms = float(np.min(latencies_array))
            self.latency_max_ms = float(np.max(latencies_array))
            self.latency_avg_ms = float(np.mean(latencies_array))
            self.latency_p50_ms = float(np.percentile(latencies_array, 50))
            self.latency_p95_ms = float(np.percentile(latencies_array, 95))
            self.latency_p99_ms = float(np.percentile(latencies_array, 99))

        # Calculate duration
        if self.end_time:
            self.duration_seconds = self.end_time - self.start_time
        else:
            self.duration_seconds = time.time() - self.start_time

        # Calculate throughput
        if self.duration_seconds > 0:
            self.messages_per_second_sent = self.messages_sent / self.duration_seconds
            self.messages_per_second_received = self.messages_received / self.duration_seconds
            self.bytes_per_second_sent = self.bytes_sent / self.duration_seconds
            self.bytes_per_second_received = self.bytes_received / self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        self.calculate_statistics()

        return {
            "connection_metrics": {
                "total_connections": self.total_connections,
                "successful_connections": self.successful_connections,
                "failed_connections": self.failed_connections,
                "active_connections": self.active_connections,
                "success_rate": (
                    self.successful_connections / self.total_connections
                    if self.total_connections > 0
                    else 0.0
                ),
            },
            "message_metrics": {
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
                "message_send_failures": self.message_send_failures,
            },
            "latency_metrics": {
                "min_ms": self.latency_min_ms,
                "max_ms": self.latency_max_ms,
                "avg_ms": self.latency_avg_ms,
                "p50_ms": self.latency_p50_ms,
                "p95_ms": self.latency_p95_ms,
                "p99_ms": self.latency_p99_ms,
                "sample_count": len(self.latencies_ms),
            },
            "throughput_metrics": {
                "messages_per_second_sent": self.messages_per_second_sent,
                "messages_per_second_received": self.messages_per_second_received,
                "bytes_per_second_sent": self.bytes_per_second_sent,
                "bytes_per_second_received": self.bytes_per_second_received,
            },
            "error_metrics": {
                "total_errors": self.total_errors,
                "connection_errors": self.connection_errors,
                "send_errors": self.send_errors,
                "receive_errors": self.receive_errors,
                "timeout_errors": self.timeout_errors,
                "error_rate": (
                    self.total_errors / self.messages_sent if self.messages_sent > 0 else 0.0
                ),
            },
            "test_info": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": (
                    datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
                ),
                "duration_seconds": self.duration_seconds,
            },
        }


class MetricsCollector:
    """Collects and aggregates metrics from WebSocket load tests."""

    def __init__(
        self,
        enable_prometheus: bool = False,
        time_window_seconds: int = 300,  # 5 minutes
    ):
        """Initialize metrics collector."""
        self.enable_prometheus = enable_prometheus
        self.time_window_seconds = time_window_seconds

        # Current test metrics
        self.current_metrics = WebSocketMetrics()

        # Time-series data for monitoring
        self.time_series: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Real-time statistics
        self.real_time_stats = {
            "connections_per_second": 0.0,
            "messages_per_second": 0.0,
            "current_latency_ms": 0.0,
            "error_rate": 0.0,
        }

        # Prometheus metrics (if enabled)
        if self.enable_prometheus:
            self._init_prometheus_metrics()

        # Background tasks
        self._running = False
        self._stats_task: Optional[asyncio.Task] = None

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Connection metrics
        self.prom_connections_total = Counter(
            "websocket_connections_total",
            "Total WebSocket connection attempts",
            ["status"],
        )
        self.prom_active_connections = Gauge(
            "websocket_active_connections",
            "Currently active WebSocket connections",
        )

        # Message metrics
        self.prom_messages_total = Counter(
            "websocket_messages_total",
            "Total WebSocket messages",
            ["direction", "type"],
        )
        self.prom_bytes_total = Counter(
            "websocket_bytes_total", "Total bytes transferred", ["direction"]
        )

        # Latency metrics
        self.prom_latency_seconds = Histogram(
            "websocket_latency_seconds",
            "WebSocket message round-trip latency",
            buckets=[
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
            ],
        )

        # Error metrics
        self.prom_errors_total = Counter(
            "websocket_errors_total", "Total WebSocket errors", ["error_type"]
        )

    def record_connection_attempt(self, success: bool):
        """Record a connection attempt."""
        self.current_metrics.total_connections += 1

        if success:
            self.current_metrics.successful_connections += 1
            self.current_metrics.active_connections += 1

            if self.enable_prometheus:
                self.prom_connections_total.labels(status="success").inc()
                self.prom_active_connections.inc()
        else:
            self.current_metrics.failed_connections += 1
            self.current_metrics.connection_errors += 1
            self.current_metrics.total_errors += 1

            if self.enable_prometheus:
                self.prom_connections_total.labels(status="failed").inc()
                self.prom_errors_total.labels(error_type="connection").inc()

        # Update time series
        self._record_time_series("connections_total", self.current_metrics.total_connections)
        self._record_time_series("active_connections", self.current_metrics.active_connections)

    def record_connection_closed(self, duration_seconds: float):
        """Record a closed connection."""
        self.current_metrics.active_connections = max(
            0, self.current_metrics.active_connections - 1
        )
        self.current_metrics.connection_duration_seconds.append(duration_seconds)

        if self.enable_prometheus:
            self.prom_active_connections.dec()

        self._record_time_series("active_connections", self.current_metrics.active_connections)

    def record_message_sent(
        self,
        message_type: str,
        size_bytes: int,
        success: bool = True,
    ):
        """Record a sent message."""
        if success:
            self.current_metrics.messages_sent += 1
            self.current_metrics.bytes_sent += size_bytes

            if self.enable_prometheus:
                self.prom_messages_total.labels(direction="sent", type=message_type).inc()
                self.prom_bytes_total.labels(direction="sent").inc(size_bytes)
        else:
            self.current_metrics.message_send_failures += 1
            self.current_metrics.send_errors += 1
            self.current_metrics.total_errors += 1

            if self.enable_prometheus:
                self.prom_errors_total.labels(error_type="send").inc()

        self._record_time_series("messages_sent", self.current_metrics.messages_sent)
        self._record_time_series("bytes_sent", self.current_metrics.bytes_sent)

    def record_message_received(self, message_type: str, size_bytes: int):
        """Record a received message."""
        self.current_metrics.messages_received += 1
        self.current_metrics.bytes_received += size_bytes

        if self.enable_prometheus:
            self.prom_messages_total.labels(direction="received", type=message_type).inc()
            self.prom_bytes_total.labels(direction="received").inc(size_bytes)

        self._record_time_series("messages_received", self.current_metrics.messages_received)
        self._record_time_series("bytes_received", self.current_metrics.bytes_received)

    def record_latency(self, latency_seconds: float):
        """Record message round-trip latency."""
        latency_ms = latency_seconds * 1000
        self.current_metrics.latencies_ms.append(latency_ms)

        if self.enable_prometheus:
            self.prom_latency_seconds.observe(latency_seconds)

        self._record_time_series("latency_ms", latency_ms)
        self.real_time_stats["current_latency_ms"] = latency_ms

    def record_error(self, error_type: str):
        """Record an error."""
        self.current_metrics.total_errors += 1

        if error_type == "connection":
            self.current_metrics.connection_errors += 1
        elif error_type == "send":
            self.current_metrics.send_errors += 1
        elif error_type == "receive":
            self.current_metrics.receive_errors += 1
        elif error_type == "timeout":
            self.current_metrics.timeout_errors += 1

        if self.enable_prometheus:
            self.prom_errors_total.labels(error_type=error_type).inc()

        self._record_time_series("errors_total", self.current_metrics.total_errors)

    def _record_time_series(self, metric_name: str, value: float):
        """Record a time-series data point."""
        timestamp = time.time()
        self.time_series[metric_name].append((timestamp, value))

        # Clean old data
        cutoff_time = timestamp - self.time_window_seconds
        while self.time_series[metric_name] and self.time_series[metric_name][0][0] < cutoff_time:
            self.time_series[metric_name].popleft()

    async def start_real_time_stats(self, update_interval: float = 1.0):
        """Start calculating real-time statistics."""
        self._running = True
        self._stats_task = asyncio.create_task(self._update_real_time_stats(update_interval))

    async def stop_real_time_stats(self):
        """Stop real-time statistics calculation."""
        self._running = False
        if self._stats_task:
            await self._stats_task

    async def _update_real_time_stats(self, update_interval: float):
        """Update real-time statistics periodically."""
        last_connections = self.current_metrics.total_connections
        last_messages = self.current_metrics.messages_sent + self.current_metrics.messages_received
        last_errors = self.current_metrics.total_errors

        while self._running:
            await asyncio.sleep(update_interval)

            # Calculate rates
            current_connections = self.current_metrics.total_connections
            current_messages = (
                self.current_metrics.messages_sent + self.current_metrics.messages_received
            )
            current_errors = self.current_metrics.total_errors

            self.real_time_stats["connections_per_second"] = (
                current_connections - last_connections
            ) / update_interval
            self.real_time_stats["messages_per_second"] = (
                current_messages - last_messages
            ) / update_interval

            if current_messages > last_messages:
                self.real_time_stats["error_rate"] = (current_errors - last_errors) / (
                    current_messages - last_messages
                )

            last_connections = current_connections
            last_messages = current_messages
            last_errors = current_errors

    def get_real_time_stats(self) -> Dict[str, float]:
        """Get current real-time statistics."""
        return self.real_time_stats.copy()

    def get_time_series_data(
        self,
        metric_name: str,
        duration_seconds: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Get time-series data for a specific metric."""
        if metric_name not in self.time_series:
            return []

        data = list(self.time_series[metric_name])

        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            data = [(t, v) for t, v in data if t >= cutoff_time]

        return data

    def finalize(self) -> WebSocketMetrics:
        """Finalize metrics collection and return results."""
        self.current_metrics.end_time = time.time()
        self.current_metrics.calculate_statistics()
        return self.current_metrics

    def save_metrics(self, filepath: Path, format: str = "json"):
        """Save metrics to file."""
        metrics = self.finalize()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(filepath, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
        elif format == "csv":
            # Flatten metrics for CSV export
            import csv

            flat_metrics = self._flatten_dict(metrics.to_dict())

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for key, value in flat_metrics.items():
                    writer.writerow([key, value])
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Metrics saved to {filepath}")

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []

        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        metrics = self.finalize()
        stats = metrics.to_dict()

        report = [
            "=" * 60,
            "WebSocket Load Test Summary Report",
            "=" * 60,
            "",
            f"Test Duration: {stats['test_info']['duration_seconds']:.2f} seconds",
            f"Start Time: {stats['test_info']['start_time']}",
            f"End Time: {stats['test_info']['end_time'] or 'Still running'}",
            "",
            "Connection Statistics:",
            f"  Total Connections: {stats['connection_metrics']['total_connections']}",
            f"  Successful: {stats['connection_metrics']['successful_connections']}",
            f"  Failed: {stats['connection_metrics']['failed_connections']}",
            f"  Success Rate: {stats['connection_metrics']['success_rate']:.2%}",
            f"  Active at End: {stats['connection_metrics']['active_connections']}",
            "",
            "Message Statistics:",
            f"  Messages Sent: {stats['message_metrics']['messages_sent']:,}",
            f"  Messages Received: {stats['message_metrics']['messages_received']:,}",
            f"  Bytes Sent: {stats['message_metrics']['bytes_sent']:,}",
            f"  Bytes Received: {stats['message_metrics']['bytes_received']:,}",
            f"  Send Failures: {stats['message_metrics']['message_send_failures']}",
            "",
            "Latency Statistics (ms):",
            f"  Min: {stats['latency_metrics']['min_ms']:.2f}",
            f"  Max: {stats['latency_metrics']['max_ms']:.2f}",
            f"  Average: {stats['latency_metrics']['avg_ms']:.2f}",
            f"  P50: {stats['latency_metrics']['p50_ms']:.2f}",
            f"  P95: {stats['latency_metrics']['p95_ms']:.2f}",
            f"  P99: {stats['latency_metrics']['p99_ms']:.2f}",
            f"  Samples: {stats['latency_metrics']['sample_count']}",
            "",
            "Throughput:",
            f"  Messages/sec Sent: {stats['throughput_metrics']['messages_per_second_sent']:.2f}",
            f"  Messages/sec Received: {stats['throughput_metrics']['messages_per_second_received']:.2f}",
            f"  Bytes/sec Sent: {stats['throughput_metrics']['bytes_per_second_sent']:.2f}",
            f"  Bytes/sec Received: {stats['throughput_metrics']['bytes_per_second_received']:.2f}",
            "",
            "Error Statistics:",
            f"  Total Errors: {stats['error_metrics']['total_errors']}",
            f"  Connection Errors: {stats['error_metrics']['connection_errors']}",
            f"  Send Errors: {stats['error_metrics']['send_errors']}",
            f"  Receive Errors: {stats['error_metrics']['receive_errors']}",
            f"  Timeout Errors: {stats['error_metrics']['timeout_errors']}",
            f"  Error Rate: {stats['error_metrics']['error_rate']:.2%}",
            "",
            "=" * 60,
        ]

        return "\n".join(report)

"""Unified Performance Metrics Collection System.

This module provides a comprehensive metrics collection framework that integrates
data from all load testing components (database, WebSocket, agents) into a unified
system for real-time monitoring, analysis, and reporting.
"""

import asyncio
import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from observability.performance_metrics import (
    RealTimePerformanceTracker,
)
from tests.db_infrastructure.performance_monitor import (
    DatabasePerformanceMonitor,
)

# Import metrics from different subsystems
from tests.websocket_load.metrics_collector import MetricsCollector as WSMetricsCollector

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""

    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summary
    RATE = "rate"  # Rate of change


class MetricSource(Enum):
    """Source systems for metrics."""

    DATABASE = "database"
    WEBSOCKET = "websocket"
    AGENT = "agent"
    SYSTEM = "system"
    INFERENCE = "inference"
    COALITION = "coalition"


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    source: MetricSource
    name: str
    value: float
    type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric statistics."""

    name: str
    source: MetricSource
    type: MetricType
    count: int
    sum: float
    min: float
    max: float
    avg: float
    std: float
    p50: float
    p95: float
    p99: float
    latest: float
    rate: float  # Per second
    window_seconds: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "source": self.source.value,
            "type": self.type.value,
            "stats": {
                "count": self.count,
                "sum": self.sum,
                "min": self.min,
                "max": self.max,
                "avg": self.avg,
                "std": self.std,
                "p50": self.p50,
                "p95": self.p95,
                "p99": self.p99,
                "latest": self.latest,
                "rate": self.rate,
            },
            "window_seconds": self.window_seconds,
        }


class UnifiedMetricsCollector:
    """Unified metrics collection system for all performance data."""

    def __init__(
        self,
        buffer_size: int = 10000,
        aggregation_windows: List[int] = None,
        persistence_enabled: bool = True,
        persistence_dir: str = "tests/performance/metrics_data",
    ):
        """Initialize unified metrics collector.

        Args:
            buffer_size: Maximum size of metric buffers
            aggregation_windows: Time windows for aggregation (seconds)
            persistence_enabled: Enable metric persistence to disk
            persistence_dir: Directory for metric persistence
        """
        self.buffer_size = buffer_size
        self.aggregation_windows = aggregation_windows or [
            60,
            300,
            900,
            3600,
        ]  # 1m, 5m, 15m, 1h
        self.persistence_enabled = persistence_enabled
        self.persistence_dir = Path(persistence_dir)

        if self.persistence_enabled:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe metric buffers
        self._lock = threading.RLock()
        self._metrics: Dict[str, Deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )

        # Metric metadata registry
        self._metric_registry: Dict[str, Dict[str, Any]] = {}

        # Subsystem collectors
        self.ws_collector = WSMetricsCollector(enable_prometheus=False)
        self.perf_tracker = RealTimePerformanceTracker(collection_interval=1.0)
        self.db_monitor = DatabasePerformanceMonitor()

        # Alert rules
        self._alert_rules: List[Dict[str, Any]] = []
        self._alert_history: Deque[Dict[str, Any]] = deque(maxlen=1000)

        # Background tasks
        self._running = False
        self._aggregation_task: Optional[asyncio.Task] = None
        self._persistence_task: Optional[asyncio.Task] = None

        logger.info("Unified metrics collector initialized")

    def register_metric(
        self,
        name: str,
        source: MetricSource,
        type: MetricType,
        description: str = "",
        unit: str = "",
        tags: Dict[str, str] = None,
    ):
        """Register a metric in the system."""
        with self._lock:
            self._metric_registry[f"{source.value}.{name}"] = {
                "name": name,
                "source": source,
                "type": type,
                "description": description,
                "unit": unit,
                "tags": tags or {},
                "created_at": datetime.now(),
            }

    def record_metric(
        self,
        name: str,
        value: float,
        source: MetricSource,
        type: MetricType = MetricType.GAUGE,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None,
    ):
        """Record a single metric value."""
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            source=source,
            name=name,
            value=value,
            type=type,
            tags=tags or {},
            metadata=metadata or {},
        )

        with self._lock:
            key = f"{source.value}.{name}"
            self._metrics[key].append(metric_point)

    async def start(self):
        """Start the unified metrics collection system."""
        if self._running:
            logger.warning("Metrics collector already running")
            return

        self._running = True

        # Start subsystem collectors
        await self.ws_collector.start_real_time_stats()
        await self.perf_tracker.start()

        # Start background tasks
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        if self.persistence_enabled:
            self._persistence_task = asyncio.create_task(self._persistence_loop())

        logger.info("Unified metrics collection started")

    async def stop(self):
        """Stop the metrics collection system."""
        self._running = False

        # Stop subsystem collectors
        await self.ws_collector.stop_real_time_stats()
        await self.perf_tracker.stop()

        # Cancel background tasks
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass

        if self._persistence_task:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass

        # Final persistence
        if self.persistence_enabled:
            await self._persist_metrics()

        logger.info("Unified metrics collection stopped")

    async def _aggregation_loop(self):
        """Background loop for metric aggregation."""
        while self._running:
            try:
                await self._aggregate_metrics()
                await asyncio.sleep(10)  # Aggregate every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation error: {e}")
                await asyncio.sleep(10)

    async def _persistence_loop(self):
        """Background loop for metric persistence."""
        while self._running:
            try:
                await self._persist_metrics()
                await asyncio.sleep(60)  # Persist every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Persistence error: {e}")
                await asyncio.sleep(60)

    async def _aggregate_metrics(self):
        """Aggregate metrics for different time windows."""
        with self._lock:
            for window_seconds in self.aggregation_windows:
                cutoff_time = datetime.now() - timedelta(seconds=window_seconds)

                # Process each metric
                for metric_key, points in self._metrics.items():
                    if not points:
                        continue

                    # Filter points within window
                    window_points = [p for p in points if p.timestamp >= cutoff_time]
                    if not window_points:
                        continue

                    # Calculate aggregates
                    values = [p.value for p in window_points]
                    source = window_points[0].source
                    name = window_points[0].name
                    type = window_points[0].type

                    aggregated = self._calculate_aggregates(
                        name, source, type, values, window_seconds
                    )

                    # Check alert rules
                    await self._check_alerts(aggregated)

    def _calculate_aggregates(
        self,
        name: str,
        source: MetricSource,
        type: MetricType,
        values: List[float],
        window_seconds: int,
    ) -> AggregatedMetric:
        """Calculate statistical aggregates for a metric."""
        if not values:
            return None

        np_values = np.array(values)

        return AggregatedMetric(
            name=name,
            source=source,
            type=type,
            count=len(values),
            sum=float(np.sum(np_values)),
            min=float(np.min(np_values)),
            max=float(np.max(np_values)),
            avg=float(np.mean(np_values)),
            std=float(np.std(np_values)),
            p50=float(np.percentile(np_values, 50)),
            p95=float(np.percentile(np_values, 95)),
            p99=float(np.percentile(np_values, 99)),
            latest=values[-1],
            rate=len(values) / window_seconds if window_seconds > 0 else 0,
            window_seconds=window_seconds,
        )

    async def _persist_metrics(self):
        """Persist current metrics to disk."""
        try:
            timestamp = datetime.now()
            filename = f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.persistence_dir / filename

            # Prepare data for persistence
            data = {
                "timestamp": timestamp.isoformat(),
                "metrics": {},
                "registry": self._metric_registry,
                "aggregations": {},
            }

            with self._lock:
                # Convert metric points to serializable format
                for key, points in self._metrics.items():
                    data["metrics"][key] = [
                        {
                            "timestamp": p.timestamp.isoformat(),
                            "value": p.value,
                            "tags": p.tags,
                            "metadata": p.metadata,
                        }
                        for p in points
                    ]

                # Add aggregations for each window
                for window in self.aggregation_windows:
                    cutoff_time = timestamp - timedelta(seconds=window)
                    window_aggregations = {}

                    for metric_key, points in self._metrics.items():
                        window_points = [p for p in points if p.timestamp >= cutoff_time]
                        if window_points:
                            values = [p.value for p in window_points]
                            agg = self._calculate_aggregates(
                                window_points[0].name,
                                window_points[0].source,
                                window_points[0].type,
                                values,
                                window,
                            )
                            if agg:
                                window_aggregations[metric_key] = agg.to_dict()

                    data["aggregations"][f"{window}s"] = window_aggregations

            # Write to file
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Persisted metrics to {filepath}")

            # Clean old files (keep last 24 hours)
            await self._cleanup_old_files()

        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")

    async def _cleanup_old_files(self):
        """Clean up old metric files."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)

            for filepath in self.persistence_dir.glob("metrics_*.json"):
                # Extract timestamp from filename
                try:
                    timestamp_str = filepath.stem.replace("metrics_", "")
                    file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                    if file_time < cutoff_time:
                        filepath.unlink()
                        logger.debug(f"Cleaned up old metric file: {filepath}")
                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")

    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        source: MetricSource,
        condition: str,  # e.g., "avg > 100", "p95 > 500"
        threshold: float,
        window_seconds: int = 300,
        severity: str = "warning",
        description: str = "",
    ):
        """Add an alert rule for automatic monitoring."""
        rule = {
            "name": name,
            "metric_name": metric_name,
            "source": source,
            "condition": condition,
            "threshold": threshold,
            "window_seconds": window_seconds,
            "severity": severity,
            "description": description,
            "created_at": datetime.now(),
            "enabled": True,
        }

        self._alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")

    async def _check_alerts(self, metric: AggregatedMetric):
        """Check if metric triggers any alert rules."""
        for rule in self._alert_rules:
            if not rule["enabled"]:
                continue

            if (
                rule["metric_name"] == metric.name
                and rule["source"] == metric.source
                and rule["window_seconds"] == metric.window_seconds
            ):
                # Evaluate condition
                triggered = False
                value = 0.0

                if "avg" in rule["condition"]:
                    value = metric.avg
                    triggered = self._evaluate_condition(
                        value, rule["condition"], rule["threshold"]
                    )
                elif "max" in rule["condition"]:
                    value = metric.max
                    triggered = self._evaluate_condition(
                        value, rule["condition"], rule["threshold"]
                    )
                elif "p95" in rule["condition"]:
                    value = metric.p95
                    triggered = self._evaluate_condition(
                        value, rule["condition"], rule["threshold"]
                    )
                elif "p99" in rule["condition"]:
                    value = metric.p99
                    triggered = self._evaluate_condition(
                        value, rule["condition"], rule["threshold"]
                    )
                elif "rate" in rule["condition"]:
                    value = metric.rate
                    triggered = self._evaluate_condition(
                        value, rule["condition"], rule["threshold"]
                    )

                if triggered:
                    await self._emit_alert(rule, metric, value)

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if ">" in condition:
            return value > threshold
        elif "<" in condition:
            return value < threshold
        elif ">=" in condition:
            return value >= threshold
        elif "<=" in condition:
            return value <= threshold
        elif "==" in condition:
            return abs(value - threshold) < 0.001
        return False

    async def _emit_alert(self, rule: Dict[str, Any], metric: AggregatedMetric, value: float):
        """Emit an alert when rule is triggered."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "rule_name": rule["name"],
            "metric_name": metric.name,
            "source": metric.source.value,
            "severity": rule["severity"],
            "condition": rule["condition"],
            "threshold": rule["threshold"],
            "actual_value": value,
            "window_seconds": metric.window_seconds,
            "description": rule["description"],
            "metric_stats": metric.to_dict(),
        }

        self._alert_history.append(alert)

        # Log alert based on severity
        if rule["severity"] == "critical":
            logger.error(
                f"🚨 CRITICAL ALERT: {rule['name']} - {metric.name} "
                f"{rule['condition']} (value: {value:.2f}, threshold: {rule['threshold']:.2f})"
            )
        elif rule["severity"] == "warning":
            logger.warning(
                f"⚠️ WARNING ALERT: {rule['name']} - {metric.name} "
                f"{rule['condition']} (value: {value:.2f}, threshold: {rule['threshold']:.2f})"
            )
        else:
            logger.info(
                f"ℹ️ INFO ALERT: {rule['name']} - {metric.name} "
                f"{rule['condition']} (value: {value:.2f}, threshold: {rule['threshold']:.2f})"
            )

    # Integration methods for subsystems

    async def collect_database_metrics(self):
        """Collect metrics from database operations."""
        db_stats = await self.db_monitor.get_current_stats()

        # Record database metrics
        self.record_metric(
            "query_latency_ms",
            db_stats.get("avg_query_time_ms", 0),
            MetricSource.DATABASE,
            MetricType.HISTOGRAM,
        )

        self.record_metric(
            "connection_pool_size",
            db_stats.get("pool_size", 0),
            MetricSource.DATABASE,
            MetricType.GAUGE,
        )

        self.record_metric(
            "transaction_rate",
            db_stats.get("transactions_per_second", 0),
            MetricSource.DATABASE,
            MetricType.RATE,
        )

    async def collect_websocket_metrics(self):
        """Collect metrics from WebSocket operations."""
        ws_stats = self.ws_collector.get_real_time_stats()

        # Record WebSocket metrics
        self.record_metric(
            "connections_per_second",
            ws_stats["connections_per_second"],
            MetricSource.WEBSOCKET,
            MetricType.RATE,
        )

        self.record_metric(
            "messages_per_second",
            ws_stats["messages_per_second"],
            MetricSource.WEBSOCKET,
            MetricType.RATE,
        )

        self.record_metric(
            "current_latency_ms",
            ws_stats["current_latency_ms"],
            MetricSource.WEBSOCKET,
            MetricType.HISTOGRAM,
        )

        self.record_metric(
            "error_rate",
            ws_stats["error_rate"],
            MetricSource.WEBSOCKET,
            MetricType.GAUGE,
        )

    async def collect_agent_metrics(self):
        """Collect metrics from agent operations."""
        perf_snapshot = await self.perf_tracker.get_current_performance_snapshot()

        # Record agent metrics
        self.record_metric(
            "inference_time_ms",
            perf_snapshot.inference_time_ms,
            MetricSource.AGENT,
            MetricType.HISTOGRAM,
        )

        self.record_metric(
            "active_agents",
            perf_snapshot.active_agents,
            MetricSource.AGENT,
            MetricType.GAUGE,
        )

        self.record_metric(
            "agent_throughput",
            perf_snapshot.agent_throughput,
            MetricSource.AGENT,
            MetricType.RATE,
        )

        self.record_metric(
            "belief_updates_per_sec",
            perf_snapshot.belief_updates_per_sec,
            MetricSource.AGENT,
            MetricType.RATE,
        )

        self.record_metric(
            "free_energy_avg",
            perf_snapshot.free_energy_avg,
            MetricSource.INFERENCE,
            MetricType.GAUGE,
        )

    async def collect_system_metrics(self):
        """Collect system-level metrics."""
        import psutil

        # CPU metrics
        self.record_metric(
            "cpu_usage_percent",
            psutil.cpu_percent(interval=None),
            MetricSource.SYSTEM,
            MetricType.GAUGE,
        )

        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_metric(
            "memory_usage_percent",
            memory.percent,
            MetricSource.SYSTEM,
            MetricType.GAUGE,
        )

        self.record_metric(
            "memory_available_mb",
            memory.available / (1024 * 1024),
            MetricSource.SYSTEM,
            MetricType.GAUGE,
        )

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.record_metric(
                "disk_read_mb_per_sec",
                disk_io.read_bytes / (1024 * 1024),
                MetricSource.SYSTEM,
                MetricType.RATE,
            )

            self.record_metric(
                "disk_write_mb_per_sec",
                disk_io.write_bytes / (1024 * 1024),
                MetricSource.SYSTEM,
                MetricType.RATE,
            )

    async def get_metrics_summary(
        self, source: Optional[MetricSource] = None, window_seconds: int = 300
    ) -> Dict[str, Any]:
        """Get summary of all metrics for a time window."""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        summary = {
            "window_seconds": window_seconds,
            "timestamp": datetime.now().isoformat(),
            "sources": {},
            "total_metrics": 0,
            "total_points": 0,
        }

        with self._lock:
            for metric_key, points in self._metrics.items():
                metric_source = points[0].source if points else None

                # Filter by source if specified
                if source and metric_source != source:
                    continue

                # Filter by time window
                window_points = [p for p in points if p.timestamp >= cutoff_time]
                if not window_points:
                    continue

                # Group by source
                source_key = metric_source.value if metric_source else "unknown"
                if source_key not in summary["sources"]:
                    summary["sources"][source_key] = {}

                # Calculate aggregates
                values = [p.value for p in window_points]
                agg = self._calculate_aggregates(
                    window_points[0].name,
                    window_points[0].source,
                    window_points[0].type,
                    values,
                    window_seconds,
                )

                if agg:
                    summary["sources"][source_key][metric_key] = agg.to_dict()
                    summary["total_metrics"] += 1
                    summary["total_points"] += len(window_points)

        # Add recent alerts
        recent_alerts = [
            alert
            for alert in self._alert_history
            if datetime.fromisoformat(alert["timestamp"]) >= cutoff_time
        ]
        summary["recent_alerts"] = recent_alerts
        summary["alert_count"] = len(recent_alerts)

        return summary

    def get_metric_history(
        self,
        metric_name: str,
        source: MetricSource,
        duration_seconds: int = 3600,
    ) -> List[Tuple[datetime, float]]:
        """Get historical values for a specific metric."""
        key = f"{source.value}.{metric_name}"
        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)

        with self._lock:
            if key not in self._metrics:
                return []

            return [
                (p.timestamp, p.value) for p in self._metrics[key] if p.timestamp >= cutoff_time
            ]

    async def export_metrics(
        self, format: str = "json", filepath: Optional[Path] = None
    ) -> Union[str, Dict[str, Any]]:
        """Export metrics in various formats."""
        data = await self.get_metrics_summary(window_seconds=3600)

        if format == "json":
            if filepath:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
                return str(filepath)
            else:
                return json.dumps(data, indent=2)

        elif format == "prometheus":
            # Export in Prometheus text format
            lines = []
            lines.append("# FreeAgentics Performance Metrics")
            lines.append(f"# Exported at {datetime.now().isoformat()}")

            for source_name, metrics in data["sources"].items():
                for metric_key, metric_data in metrics.items():
                    stats = metric_data["stats"]
                    metric_name = metric_key.replace(".", "_").replace("-", "_")

                    # Export different statistics
                    lines.append(f"{metric_name}_avg {stats['avg']:.6f}")
                    lines.append(f"{metric_name}_min {stats['min']:.6f}")
                    lines.append(f"{metric_name}_max {stats['max']:.6f}")
                    lines.append(f"{metric_name}_p95 {stats['p95']:.6f}")
                    lines.append(f"{metric_name}_p99 {stats['p99']:.6f}")
                    lines.append(f"{metric_name}_count {stats['count']}")

            output = "\n".join(lines)

            if filepath:
                with open(filepath, "w") as f:
                    f.write(output)
                return str(filepath)
            else:
                return output

        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global instance for easy access
unified_collector = UnifiedMetricsCollector()


async def start_unified_collection():
    """Start the unified metrics collection system."""
    await unified_collector.start()


async def stop_unified_collection():
    """Stop the unified metrics collection system."""
    await unified_collector.stop()


def record_metric(
    name: str,
    value: float,
    source: MetricSource,
    type: MetricType = MetricType.GAUGE,
    tags: Dict[str, str] = None,
    metadata: Dict[str, Any] = None,
):
    """Record a metric to the unified system."""
    unified_collector.record_metric(name, value, source, type, tags, metadata)


async def get_metrics_summary(
    source: Optional[MetricSource] = None, window_seconds: int = 300
) -> Dict[str, Any]:
    """Get a summary of metrics."""
    return await unified_collector.get_metrics_summary(source, window_seconds)

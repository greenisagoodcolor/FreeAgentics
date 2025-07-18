"""
Enhanced Metrics Exporter for FreeAgentics Production Monitoring

This module provides comprehensive metrics export capabilities for monitoring
agent performance, system health, and business metrics.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psutil
from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest

from observability.performance_metrics import performance_tracker
from observability.prometheus_metrics import (
    freeagentics_registry,
    prometheus_collector,
    record_agent_coordination_request,
    record_agent_error,
    record_agent_step,
    record_belief_state_update,
    record_business_inference_operation,
    record_user_interaction,
    update_belief_accuracy_ratio,
    update_belief_free_energy,
    update_response_quality_score,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricDataPoint:
    """Represents a single metric data point."""

    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """Summary of metrics over a time period."""

    metric_name: str
    time_window: float
    data_points: int
    min_value: float
    max_value: float
    avg_value: float
    latest_value: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    anomalies: List[Dict[str, Any]] = field(default_factory=list)


class EnhancedMetricsExporter:
    """Enhanced metrics exporter with comprehensive monitoring capabilities."""

    def __init__(self, export_interval: float = 10.0):
        """Initialize the metrics exporter."""
        self.export_interval = export_interval
        self.running = False
        self.export_task = None
        self.metrics_buffer = []
        self.buffer_size = 10000
        self.last_export_time = time.time()

        # Custom metrics for production monitoring
        self._setup_custom_metrics()

        logger.info("🔧 Enhanced metrics exporter initialized")

    def _setup_custom_metrics(self):
        """Set up custom metrics for production monitoring."""
        # Agent lifecycle metrics
        self.agent_lifecycle_events = Counter(
            "freeagentics_agent_lifecycle_events_total",
            "Total agent lifecycle events",
            ["event_type", "agent_id", "success"],
            registry=freeagentics_registry,
        )

        # Agent resource consumption
        self.agent_resource_usage = Gauge(
            "freeagentics_agent_resource_usage",
            "Agent resource usage metrics",
            ["agent_id", "resource_type"],
            registry=freeagentics_registry,
        )

        # Coalition formation metrics
        self.coalition_formation_metrics = Histogram(
            "freeagentics_coalition_formation_duration_seconds",
            "Coalition formation duration",
            ["coalition_type", "agent_count"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=freeagentics_registry,
        )

        # Knowledge graph metrics
        self.knowledge_graph_operations = Counter(
            "freeagentics_knowledge_graph_operations_total",
            "Knowledge graph operations",
            ["operation_type", "success"],
            registry=freeagentics_registry,
        )

        # Inference performance metrics
        self.inference_performance = Histogram(
            "freeagentics_inference_performance_seconds",
            "Inference operation performance",
            ["inference_type", "agent_id", "complexity"],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=freeagentics_registry,
        )

        # Business value metrics
        self.business_value_metrics = Gauge(
            "freeagentics_business_value_score",
            "Business value score",
            ["metric_type", "category"],
            registry=freeagentics_registry,
        )

        # System health indicators
        self.system_health_indicators = Gauge(
            "freeagentics_system_health_indicators",
            "System health indicators",
            ["indicator_type"],
            registry=freeagentics_registry,
        )

        # Custom application metrics
        self.custom_application_metrics = Counter(
            "freeagentics_custom_application_events_total",
            "Custom application events",
            ["event_type", "component", "status"],
            registry=freeagentics_registry,
        )

    async def start(self):
        """Start the metrics exporter."""
        if self.running:
            logger.warning("Metrics exporter already running")
            return

        self.running = True
        self.export_task = asyncio.create_task(self._export_loop())

        logger.info("📊 Enhanced metrics exporter started")

    async def stop(self):
        """Stop the metrics exporter."""
        if not self.running:
            logger.warning("Metrics exporter not running")
            return

        self.running = False

        if self.export_task:
            self.export_task.cancel()
            try:
                await self.export_task
            except asyncio.CancelledError:
                pass

        logger.info("📊 Enhanced metrics exporter stopped")

    async def _export_loop(self):
        """Main export loop."""
        while self.running:
            try:
                await self._export_metrics()
                await asyncio.sleep(self.export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics export loop: {e}")
                await asyncio.sleep(self.export_interval)

    async def _export_metrics(self):
        """Export metrics to various systems."""
        try:
            # Update system health indicators
            await self._update_system_health_indicators()

            # Update agent resource usage
            await self._update_agent_resource_usage()

            # Update business value metrics
            await self._update_business_value_metrics()

            # Export to time series database
            await self._export_to_timeseries()

            # Export to log aggregation system
            await self._export_to_logs()

            # Update export timestamp
            self.last_export_time = time.time()

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

    async def _update_system_health_indicators(self):
        """Update system health indicators."""
        try:
            # CPU health
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.system_health_indicators.labels(
                indicator_type="cpu_health"
            ).set(
                1.0 if cpu_percent < 80 else 0.5 if cpu_percent < 90 else 0.0
            )

            # Memory health
            memory = psutil.virtual_memory()
            memory_health = (
                1.0
                if memory.percent < 80
                else 0.5
                if memory.percent < 90
                else 0.0
            )
            self.system_health_indicators.labels(
                indicator_type="memory_health"
            ).set(memory_health)

            # Disk health
            disk = psutil.disk_usage("/")
            disk_health = (
                1.0 if disk.percent < 80 else 0.5 if disk.percent < 90 else 0.0
            )
            self.system_health_indicators.labels(
                indicator_type="disk_health"
            ).set(disk_health)

            # Performance tracker health
            perf_snapshot = (
                await performance_tracker.get_current_performance_snapshot()
            )
            perf_health = (
                1.0
                if perf_snapshot.active_agents < 40
                else 0.5
                if perf_snapshot.active_agents < 50
                else 0.0
            )
            self.system_health_indicators.labels(
                indicator_type="performance_health"
            ).set(perf_health)

        except Exception as e:
            logger.error(f"Error updating system health indicators: {e}")

    async def _update_agent_resource_usage(self):
        """Update agent resource usage metrics."""
        try:
            # Get performance snapshot
            snapshot = (
                await performance_tracker.get_current_performance_snapshot()
            )

            # Update agent memory usage (approximation)
            if snapshot.active_agents > 0:
                memory_per_agent = (
                    snapshot.memory_usage_mb / snapshot.active_agents
                )
                self.agent_resource_usage.labels(
                    agent_id="system_average", resource_type="memory_mb"
                ).set(memory_per_agent)

            # Update agent CPU usage (approximation)
            if snapshot.active_agents > 0:
                cpu_per_agent = (
                    snapshot.cpu_usage_percent / snapshot.active_agents
                )
                self.agent_resource_usage.labels(
                    agent_id="system_average", resource_type="cpu_percent"
                ).set(cpu_per_agent)

            # Update agent throughput
            self.agent_resource_usage.labels(
                agent_id="system_total", resource_type="throughput_ops_per_sec"
            ).set(snapshot.agent_throughput)

        except Exception as e:
            logger.error(f"Error updating agent resource usage: {e}")

    async def _update_business_value_metrics(self):
        """Update business value metrics."""
        try:
            # Get current performance snapshot
            snapshot = (
                await performance_tracker.get_current_performance_snapshot()
            )

            # System efficiency score
            efficiency_score = min(
                1.0,
                snapshot.agent_throughput / max(1.0, snapshot.active_agents),
            )
            self.business_value_metrics.labels(
                metric_type="efficiency", category="system"
            ).set(efficiency_score)

            # Resource utilization score
            memory_utilization = (
                snapshot.memory_usage_mb / 2048
            )  # Assuming 2GB limit
            utilization_score = 1.0 - min(1.0, memory_utilization)
            self.business_value_metrics.labels(
                metric_type="resource_utilization", category="system"
            ).set(utilization_score)

            # Performance consistency score
            if snapshot.belief_updates_per_sec > 0:
                consistency_score = min(
                    1.0,
                    snapshot.belief_updates_per_sec
                    / snapshot.agent_throughput,
                )
                self.business_value_metrics.labels(
                    metric_type="consistency", category="performance"
                ).set(consistency_score)

        except Exception as e:
            logger.error(f"Error updating business value metrics: {e}")

    async def _export_to_timeseries(self):
        """Export metrics to time series database."""
        try:
            # This would integrate with InfluxDB, TimescaleDB, etc.
            # For now, we'll log the metrics

            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent,
                },
                "agent_metrics": {
                    "active_agents": len(performance_tracker.agent_metrics),
                    "total_inferences": sum(
                        buffer["inference_times"].count
                        for buffer in performance_tracker.agent_metrics.values()
                    ),
                },
            }

            logger.debug(
                f"Exported metrics to time series: {json.dumps(metrics_data, indent=2)}"
            )

        except Exception as e:
            logger.error(f"Error exporting to time series: {e}")

    async def _export_to_logs(self):
        """Export metrics to log aggregation system."""
        try:
            # Create structured log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "metrics_export",
                "export_interval": self.export_interval,
                "metrics_summary": {
                    "system_health": {
                        "cpu_health": psutil.cpu_percent() < 80,
                        "memory_health": psutil.virtual_memory().percent < 80,
                        "disk_health": psutil.disk_usage("/").percent < 80,
                    },
                    "agent_performance": {
                        "active_agents": len(
                            performance_tracker.agent_metrics
                        ),
                        "average_inference_time": performance_tracker.get_average_inference_time(),
                    },
                },
            }

            logger.info(
                f"Metrics export summary: {json.dumps(log_entry, indent=2)}"
            )

        except Exception as e:
            logger.error(f"Error exporting to logs: {e}")

    def record_agent_lifecycle_event(
        self, event_type: str, agent_id: str, success: bool
    ):
        """Record agent lifecycle event."""
        self.agent_lifecycle_events.labels(
            event_type=event_type,
            agent_id=agent_id,
            success=str(success).lower(),
        ).inc()

    def record_coalition_formation(
        self, coalition_type: str, agent_count: int, duration: float
    ):
        """Record coalition formation metrics."""
        self.coalition_formation_metrics.labels(
            coalition_type=coalition_type, agent_count=str(agent_count)
        ).observe(duration)

    def record_knowledge_graph_operation(
        self, operation_type: str, success: bool
    ):
        """Record knowledge graph operation."""
        self.knowledge_graph_operations.labels(
            operation_type=operation_type, success=str(success).lower()
        ).inc()

    def record_inference_performance(
        self,
        inference_type: str,
        agent_id: str,
        complexity: str,
        duration: float,
    ):
        """Record inference performance metrics."""
        self.inference_performance.labels(
            inference_type=inference_type,
            agent_id=agent_id,
            complexity=complexity,
        ).observe(duration)

    def record_custom_application_event(
        self, event_type: str, component: str, status: str
    ):
        """Record custom application event."""
        self.custom_application_metrics.labels(
            event_type=event_type, component=component, status=status
        ).inc()

    def get_metrics_summary(
        self, metric_name: str, time_window: float = 300.0
    ) -> MetricsSummary:
        """Get metrics summary for a specific metric."""
        try:
            # This would query the metrics from the buffer or database
            # For now, return a placeholder
            return MetricsSummary(
                metric_name=metric_name,
                time_window=time_window,
                data_points=100,
                min_value=0.0,
                max_value=100.0,
                avg_value=50.0,
                latest_value=45.0,
                trend="stable",
                anomalies=[],
            )
        except Exception as e:
            logger.error(
                f"Error getting metrics summary for {metric_name}: {e}"
            )
            return MetricsSummary(
                metric_name=metric_name,
                time_window=time_window,
                data_points=0,
                min_value=0.0,
                max_value=0.0,
                avg_value=0.0,
                latest_value=0.0,
                trend="unknown",
                anomalies=[],
            )

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the metrics exporter."""
        return {
            "running": self.running,
            "last_export_time": self.last_export_time,
            "export_interval": self.export_interval,
            "metrics_buffer_size": len(self.metrics_buffer),
            "buffer_capacity": self.buffer_size,
        }

    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        current_time = time.time()
        time_since_last_export = current_time - self.last_export_time

        return {
            "total_exports": int(current_time - self.last_export_time)
            // int(self.export_interval),
            "time_since_last_export": time_since_last_export,
            "export_health": (
                "healthy"
                if time_since_last_export < self.export_interval * 2
                else "unhealthy"
            ),
            "buffer_utilization": len(self.metrics_buffer) / self.buffer_size,
            "export_interval": self.export_interval,
        }


# Global metrics exporter instance
_metrics_exporter = None


def get_metrics_exporter(
    export_interval: float = 10.0,
) -> EnhancedMetricsExporter:
    """Get the global metrics exporter instance."""
    global _metrics_exporter

    if _metrics_exporter is None:
        _metrics_exporter = EnhancedMetricsExporter(export_interval)

    return _metrics_exporter


async def start_metrics_export(export_interval: float = 10.0):
    """Start the global metrics export system."""
    exporter = get_metrics_exporter(export_interval)
    await exporter.start()
    return exporter


async def stop_metrics_export():
    """Stop the global metrics export system."""
    global _metrics_exporter

    if _metrics_exporter:
        await _metrics_exporter.stop()
        _metrics_exporter = None


# Integration with existing prometheus metrics
async def integrate_with_prometheus_metrics():
    """Integrate with existing Prometheus metrics system."""
    try:
        # Start prometheus metrics collection if not already started
        await prometheus_collector.start_collection()

        # Start enhanced metrics export
        await start_metrics_export()

        logger.info("✅ Successfully integrated with Prometheus metrics")

    except Exception as e:
        logger.error(f"Failed to integrate with Prometheus metrics: {e}")
        raise


# Convenience functions for common metrics operations
def record_agent_startup(agent_id: str, success: bool):
    """Record agent startup event."""
    exporter = get_metrics_exporter()
    exporter.record_agent_lifecycle_event("startup", agent_id, success)


def record_agent_shutdown(agent_id: str, success: bool):
    """Record agent shutdown event."""
    exporter = get_metrics_exporter()
    exporter.record_agent_lifecycle_event("shutdown", agent_id, success)


def record_coalition_formed(
    coalition_type: str, agent_count: int, duration: float
):
    """Record coalition formation."""
    exporter = get_metrics_exporter()
    exporter.record_coalition_formation(coalition_type, agent_count, duration)


def record_knowledge_graph_update(success: bool):
    """Record knowledge graph update."""
    exporter = get_metrics_exporter()
    exporter.record_knowledge_graph_operation("update", success)


def record_inference_operation(
    inference_type: str, agent_id: str, complexity: str, duration: float
):
    """Record inference operation."""
    exporter = get_metrics_exporter()
    exporter.record_inference_performance(
        inference_type, agent_id, complexity, duration
    )

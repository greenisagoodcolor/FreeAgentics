"""Real-time performance metrics tracking for FreeAgentics.

Implements comprehensive performance monitoring including inference speed,
memory usage, agent throughput, and system resource utilization for
production deployment and optimization.
"""

import asyncio
import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

# Try to import monitoring system
try:
    from api.v1.monitoring import record_agent_metric, record_system_metric

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

    # Mock monitoring functions
    async def record_system_metric(
        metric: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        logger.debug(f"MOCK System - {metric}: {value}")

    async def record_agent_metric(
        agent_id: str,
        metric: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        logger.debug(f"MOCK Agent {agent_id} - {metric}: {value}")


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""

    timestamp: datetime
    inference_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    agent_throughput: float
    active_agents: int
    belief_updates_per_sec: float
    free_energy_avg: float


class MetricsBuffer:
    """Thread-safe circular buffer for metrics with automatic aggregation."""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()

    def add(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a value to the buffer."""
        with self.lock:
            self.buffer.append((timestamp or datetime.now(), value))

    def get_stats(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get statistical summary for recent window."""
        with self.lock:
            if not self.buffer:
                return {
                    "count": 0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "latest": 0.0,
                }

            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            recent_values = [
                value for timestamp, value in self.buffer if timestamp >= cutoff
            ]

            if not recent_values:
                return {
                    "count": 0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "latest": 0.0,
                }

            return {
                "count": len(recent_values),
                "avg": sum(recent_values) / len(recent_values),
                "min": min(recent_values),
                "max": max(recent_values),
                "latest": recent_values[-1] if recent_values else 0.0,
                "p95": self._percentile(recent_values, 95),
                "p99": self._percentile(recent_values, 99),
            }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        return sorted_values[f]


class RealTimePerformanceTracker:
    """Real-time performance metrics tracking system."""

    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.running = False
        self.collection_task = None

        # Metrics buffers
        self.inference_times = MetricsBuffer(max_size=10000)
        self.memory_usage = MetricsBuffer(max_size=1000)
        self.cpu_usage = MetricsBuffer(max_size=1000)
        self.agent_throughput = MetricsBuffer(max_size=1000)
        self.belief_update_rates = MetricsBuffer(max_size=1000)
        self.free_energy_values = MetricsBuffer(max_size=1000)

        # Agent-specific metrics
        self.agent_metrics = defaultdict(
            lambda: {
                "inference_times": MetricsBuffer(max_size=1000),
                "belief_updates": MetricsBuffer(max_size=1000),
                "action_counts": MetricsBuffer(max_size=1000),
                "error_counts": MetricsBuffer(max_size=100),
            }
        )

        # Performance baselines for alerting
        self.baselines = {
            "inference_time_ms": 10.0,  # 10ms baseline
            "memory_usage_mb": 500.0,  # 500MB baseline
            "cpu_usage_percent": 70.0,  # 70% CPU baseline
            "agent_throughput": 100.0,  # 100 ops/sec baseline
            "belief_update_rate": 50.0,  # 50 updates/sec baseline
        }

        # Alert thresholds (multipliers of baseline)
        self.alert_thresholds = {
            "warning": 2.0,  # 2x baseline = warning
            "critical": 5.0,  # 5x baseline = critical
        }

        # System info
        self.process = psutil.Process()
        self.start_time = datetime.now()

        # Performance counters
        self.counters = {
            "total_inferences": 0,
            "total_belief_updates": 0,
            "total_agent_steps": 0,
            "total_errors": 0,
        }

        logger.info("🎯 Real-time performance tracker initialized")

    async def start(self) -> None:
        """Start real-time performance collection."""
        if self.running:
            logger.warning("Performance tracker already running")
            return

        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("📊 Real-time performance tracking started")

    async def stop(self) -> None:
        """Stop performance collection."""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("📊 Real-time performance tracking stopped")

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance collection error: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_system_metrics(self) -> None:
        """Collect system-wide performance metrics."""
        try:
            timestamp = datetime.now()

            # Memory metrics
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.memory_usage.add(memory_mb, timestamp)

            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            self.cpu_usage.add(cpu_percent, timestamp)

            # System-wide CPU and memory
            system_cpu = psutil.cpu_percent(interval=None)
            system_memory = psutil.virtual_memory().percent

            # Calculate throughput metrics
            uptime_seconds = (timestamp - self.start_time).total_seconds()
            if uptime_seconds > 0:
                belief_update_throughput = (
                    self.counters["total_belief_updates"] / uptime_seconds
                )
                agent_step_throughput = (
                    self.counters["total_agent_steps"] / uptime_seconds
                )

                self.agent_throughput.add(agent_step_throughput, timestamp)
                self.belief_update_rates.add(belief_update_throughput, timestamp)

            # Record to monitoring system
            if MONITORING_AVAILABLE:
                await record_system_metric("performance_memory_mb", memory_mb)
                await record_system_metric("performance_cpu_percent", cpu_percent)
                await record_system_metric("performance_system_cpu", system_cpu)
                await record_system_metric("performance_system_memory", system_memory)
                await record_system_metric(
                    "performance_agent_throughput",
                    self.agent_throughput.get_stats(60)["latest"],
                )
                await record_system_metric(
                    "performance_belief_update_rate",
                    self.belief_update_rates.get_stats(60)["latest"],
                )

            # Check for performance alerts
            await self._check_performance_alerts()

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def record_inference_performance(
        self,
        agent_id: str,
        inference_time_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record individual inference performance."""
        timestamp = datetime.now()

        # Update global metrics
        self.inference_times.add(inference_time_ms, timestamp)
        self.counters["total_inferences"] += 1

        # Update agent-specific metrics
        agent_buffers = self.agent_metrics[agent_id]
        agent_buffers["inference_times"].add(inference_time_ms, timestamp)

        if not success:
            self.counters["total_errors"] += 1
            agent_buffers["error_counts"].add(1.0, timestamp)

        # Record to monitoring system
        if MONITORING_AVAILABLE:
            await record_agent_metric(
                agent_id,
                "inference_time_ms",
                inference_time_ms,
                {"success": success, "error": error},
            )
            await record_system_metric(
                "performance_inference_time_avg",
                self.inference_times.get_stats(60)["avg"],
            )

    async def record_belief_update(
        self,
        agent_id: str,
        update_time_ms: Optional[float] = None,
        free_energy: Optional[float] = None,
    ) -> None:
        """Record belief update performance."""
        timestamp = datetime.now()

        # Update counters
        self.counters["total_belief_updates"] += 1

        # Update agent-specific metrics
        agent_buffers = self.agent_metrics[agent_id]
        agent_buffers["belief_updates"].add(1.0, timestamp)

        if update_time_ms:
            agent_buffers["belief_updates"].add(update_time_ms, timestamp)

        if free_energy is not None:
            self.free_energy_values.add(free_energy, timestamp)

        # Record to monitoring system
        if MONITORING_AVAILABLE:
            if update_time_ms:
                await record_agent_metric(
                    agent_id, "belief_update_time_ms", update_time_ms
                )
            if free_energy is not None:
                await record_agent_metric(agent_id, "free_energy", free_energy)
                await record_system_metric(
                    "performance_avg_free_energy",
                    self.free_energy_values.get_stats(60)["avg"],
                )

    async def record_agent_step(
        self, agent_id: str, step_time_ms: Optional[float] = None
    ) -> None:
        """Record agent step performance."""
        timestamp = datetime.now()

        # Update counters
        self.counters["total_agent_steps"] += 1

        # Update agent-specific metrics
        agent_buffers = self.agent_metrics[agent_id]
        agent_buffers["action_counts"].add(1.0, timestamp)

        if step_time_ms:
            agent_buffers["action_counts"].add(step_time_ms, timestamp)

        # Record to monitoring system
        if MONITORING_AVAILABLE:
            if step_time_ms:
                await record_agent_metric(agent_id, "step_time_ms", step_time_ms)

    async def _check_performance_alerts(self) -> None:
        """Check for performance threshold violations."""
        try:
            current_stats = await self.get_current_performance_snapshot()

            # Check inference time
            if (
                current_stats.inference_time_ms
                > self.baselines["inference_time_ms"]
                * self.alert_thresholds["critical"]
            ):
                await self._emit_alert(
                    "critical",
                    "inference_time",
                    current_stats.inference_time_ms,
                    self.baselines["inference_time_ms"],
                )
            elif (
                current_stats.inference_time_ms
                > self.baselines["inference_time_ms"] * self.alert_thresholds["warning"]
            ):
                await self._emit_alert(
                    "warning",
                    "inference_time",
                    current_stats.inference_time_ms,
                    self.baselines["inference_time_ms"],
                )

            # Check memory usage
            if (
                current_stats.memory_usage_mb
                > self.baselines["memory_usage_mb"] * self.alert_thresholds["critical"]
            ):
                await self._emit_alert(
                    "critical",
                    "memory_usage",
                    current_stats.memory_usage_mb,
                    self.baselines["memory_usage_mb"],
                )

            # Check CPU usage
            if (
                current_stats.cpu_usage_percent
                > self.baselines["cpu_usage_percent"] * self.alert_thresholds["warning"]
            ):
                await self._emit_alert(
                    "warning",
                    "cpu_usage",
                    current_stats.cpu_usage_percent,
                    self.baselines["cpu_usage_percent"],
                )

        except Exception as e:
            logger.error(f"Failed to check performance alerts: {e}")

    async def _emit_alert(
        self, level: str, metric: str, current_value: float, baseline: float
    ) -> None:
        """Emit performance alert."""
        multiplier = current_value / baseline if baseline > 0 else 0

        alert_data = {
            "level": level,
            "metric": metric,
            "current_value": current_value,
            "baseline": baseline,
            "multiplier": multiplier,
            "timestamp": datetime.now().isoformat(),
        }

        if level == "critical":
            logger.error(
                f"🚨 CRITICAL PERFORMANCE ALERT: {metric} = {current_value:.2f} "
                f"({multiplier:.1f}x baseline of {baseline:.2f})"
            )
        else:
            logger.warning(
                f"⚠️ PERFORMANCE WARNING: {metric} = {current_value:.2f} "
                f"({multiplier:.1f}x baseline of {baseline:.2f})"
            )

        # Record alert to monitoring system
        if MONITORING_AVAILABLE:
            await record_system_metric(f"performance_alert_{level}", 1.0, alert_data)

    async def get_current_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        inference_stats = self.inference_times.get_stats(60)
        memory_stats = self.memory_usage.get_stats(60)
        cpu_stats = self.cpu_usage.get_stats(60)
        throughput_stats = self.agent_throughput.get_stats(60)
        belief_stats = self.belief_update_rates.get_stats(60)
        free_energy_stats = self.free_energy_values.get_stats(60)

        return PerformanceSnapshot(
            timestamp=datetime.now(),
            inference_time_ms=inference_stats["avg"],
            memory_usage_mb=memory_stats["latest"],
            cpu_usage_percent=cpu_stats["latest"],
            agent_throughput=throughput_stats["latest"],
            active_agents=len(self.agent_metrics),
            belief_updates_per_sec=belief_stats["latest"],
            free_energy_avg=free_energy_stats["avg"],
        )

    async def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get performance summary for specific agent."""
        if agent_id not in self.agent_metrics:
            return {"error": f"No metrics found for agent {agent_id}"}

        agent_buffers = self.agent_metrics[agent_id]

        return {
            "agent_id": agent_id,
            "inference_performance": agent_buffers["inference_times"].get_stats(
                300
            ),  # 5 min window
            "belief_update_rate": agent_buffers["belief_updates"].get_stats(300),
            "action_rate": agent_buffers["action_counts"].get_stats(300),
            "error_rate": agent_buffers["error_counts"].get_stats(300),
            "total_steps": sum(1 for _, _ in agent_buffers["action_counts"].buffer),
            "total_errors": sum(1 for _, _ in agent_buffers["error_counts"].buffer),
        }

    async def get_system_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report."""
        snapshot = await self.get_current_performance_snapshot()
        uptime = datetime.now() - self.start_time

        return {
            "system_snapshot": {
                "timestamp": snapshot.timestamp.isoformat(),
                "inference_time_ms": snapshot.inference_time_ms,
                "memory_usage_mb": snapshot.memory_usage_mb,
                "cpu_usage_percent": snapshot.cpu_usage_percent,
                "agent_throughput": snapshot.agent_throughput,
                "active_agents": snapshot.active_agents,
                "belief_updates_per_sec": snapshot.belief_updates_per_sec,
                "free_energy_avg": snapshot.free_energy_avg,
            },
            "system_counters": self.counters.copy(),
            "system_uptime_seconds": uptime.total_seconds(),
            "performance_baselines": self.baselines.copy(),
            "alert_thresholds": self.alert_thresholds.copy(),
            "detailed_stats": {
                "inference_times": self.inference_times.get_stats(300),
                "memory_usage": self.memory_usage.get_stats(300),
                "cpu_usage": self.cpu_usage.get_stats(300),
                "agent_throughput": self.agent_throughput.get_stats(300),
                "belief_update_rates": self.belief_update_rates.get_stats(300),
                "free_energy": self.free_energy_values.get_stats(300),
            },
            "agent_count": len(self.agent_metrics),
        }

    def update_baselines(self, new_baselines: Dict[str, float]) -> None:
        """Update performance baselines."""
        self.baselines.update(new_baselines)
        logger.info(f"Updated performance baselines: {new_baselines}")

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics for external analysis."""
        try:
            if format == "json":
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "system_counters": self.counters,
                    "baselines": self.baselines,
                    "metrics_summary": {
                        "inference_times": self.inference_times.get_stats(
                            3600
                        ),  # 1 hour
                        "memory_usage": self.memory_usage.get_stats(3600),
                        "cpu_usage": self.cpu_usage.get_stats(3600),
                        "agent_throughput": self.agent_throughput.get_stats(3600),
                        "belief_update_rates": self.belief_update_rates.get_stats(3600),
                        "free_energy": self.free_energy_values.get_stats(3600),
                    },
                    "agent_count": len(self.agent_metrics),
                }
                return json.dumps(export_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return f"Export error: {e}"


# Global performance tracker instance
performance_tracker = RealTimePerformanceTracker()


async def start_performance_tracking() -> None:
    """Start global performance tracking."""
    await performance_tracker.start()


async def stop_performance_tracking() -> None:
    """Stop global performance tracking."""
    await performance_tracker.stop()


async def record_inference_metric(
    agent_id: str,
    inference_time_ms: float,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """Record inference performance metric."""
    await performance_tracker.record_inference_performance(
        agent_id, inference_time_ms, success, error
    )


async def record_belief_metric(
    agent_id: str,
    update_time_ms: Optional[float] = None,
    free_energy: Optional[float] = None,
) -> None:
    """Record belief update metric."""
    await performance_tracker.record_belief_update(
        agent_id, update_time_ms, free_energy
    )


async def record_step_metric(
    agent_id: str, step_time_ms: Optional[float] = None
) -> None:
    """Record agent step metric."""
    await performance_tracker.record_agent_step(agent_id, step_time_ms)


async def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report."""
    return await performance_tracker.get_system_performance_report()


async def get_agent_report(agent_id: str) -> Dict[str, Any]:
    """Get agent-specific performance report."""
    return await performance_tracker.get_agent_performance_summary(agent_id)

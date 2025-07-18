"""Real-time monitoring dashboard for agent operations.

This module provides APIs and data structures for real-time monitoring
dashboards that visualize agent performance, belief states, and coordination
metrics.
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from observability.belief_monitoring import belief_monitoring_hooks
from observability.coordination_metrics import coordination_metrics
from observability.performance_metrics import performance_tracker

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetric:
    """Single metric for dashboard display."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    trend: Optional[str] = None  # 'up', 'down', 'stable'
    status: Optional[str] = None  # 'healthy', 'warning', 'critical'


@dataclass
class TimeSeriesData:
    """Time series data for charts."""

    metric_name: str
    timestamps: List[datetime]
    values: List[float]
    unit: str


@dataclass
class AgentDashboard:
    """Dashboard data for a single agent."""

    agent_id: str
    name: str
    status: str  # 'active', 'inactive', 'error'
    current_metrics: Dict[str, DashboardMetric]
    time_series: Dict[str, TimeSeriesData]
    alerts: List[Dict[str, Any]]


@dataclass
class SystemDashboard:
    """System-wide dashboard data."""

    timestamp: datetime
    system_metrics: Dict[str, DashboardMetric]
    agent_dashboards: Dict[str, AgentDashboard]
    active_coalitions: List[Dict[str, Any]]
    recent_events: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]


class MonitoringDashboard:
    """Real-time monitoring dashboard manager."""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.running = False
        self.update_task = None

        # Data storage
        self.current_dashboard = None
        self.metric_history = defaultdict(
            lambda: deque(maxlen=300)
        )  # 5 min history
        self.event_history = deque(maxlen=100)
        self.alert_history = deque(maxlen=50)

        # Alert thresholds
        self.alert_thresholds = {
            "inference_time_ms": {"warning": 50, "critical": 100},
            "belief_entropy": {"warning": 3.0, "critical": 4.0},
            "coordination_time_ms": {"warning": 100, "critical": 200},
            "free_energy": {"warning": 10.0, "critical": 20.0},
            "cpu_usage_percent": {"warning": 70, "critical": 90},
            "memory_usage_mb": {"warning": 500, "critical": 800},
        }

        logger.info("Initialized monitoring dashboard")

    async def start(self):
        """Start dashboard updates."""
        if self.running:
            return

        self.running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("Started monitoring dashboard")

    async def stop(self):
        """Stop dashboard updates."""
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped monitoring dashboard")

    async def _update_loop(self):
        """Main update loop."""
        while self.running:
            try:
                await self._update_dashboard()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(self.update_interval)

    async def _update_dashboard(self):
        """Update all dashboard data."""
        timestamp = datetime.now()

        # Get system metrics
        system_metrics = await self._get_system_metrics()

        # Get agent dashboards
        agent_dashboards = await self._get_agent_dashboards()

        # Get coalition data
        active_coalitions = await self._get_coalition_data()

        # Get recent events
        recent_events = list(self.event_history)[-20:]

        # Get current alerts
        current_alerts = await self._check_alerts(
            system_metrics, agent_dashboards
        )

        # Create dashboard snapshot
        self.current_dashboard = SystemDashboard(
            timestamp=timestamp,
            system_metrics=system_metrics,
            agent_dashboards=agent_dashboards,
            active_coalitions=active_coalitions,
            recent_events=recent_events,
            alerts=current_alerts,
        )

    async def _get_system_metrics(self) -> Dict[str, DashboardMetric]:
        """Get system-wide metrics."""
        metrics = {}
        timestamp = datetime.now()

        # Get performance metrics
        perf_report = await performance_tracker.get_system_performance_report()

        # System performance
        metrics["total_agents"] = DashboardMetric(
            name="Total Agents",
            value=perf_report["agent_count"],
            unit="agents",
            timestamp=timestamp,
            status="healthy",
        )

        metrics["avg_inference_time"] = DashboardMetric(
            name="Avg Inference Time",
            value=perf_report["detailed_stats"]["inference_times"]["avg"],
            unit="ms",
            timestamp=timestamp,
            status=self._get_metric_status(
                "inference_time_ms",
                perf_report["detailed_stats"]["inference_times"]["avg"],
            ),
        )

        metrics["system_memory"] = DashboardMetric(
            name="Memory Usage",
            value=perf_report["detailed_stats"]["memory_usage"]["latest"],
            unit="MB",
            timestamp=timestamp,
            status=self._get_metric_status(
                "memory_usage_mb",
                perf_report["detailed_stats"]["memory_usage"]["latest"],
            ),
        )

        metrics["system_cpu"] = DashboardMetric(
            name="CPU Usage",
            value=perf_report["detailed_stats"]["cpu_usage"]["latest"],
            unit="%",
            timestamp=timestamp,
            status=self._get_metric_status(
                "cpu_usage_percent",
                perf_report["detailed_stats"]["cpu_usage"]["latest"],
            ),
        )

        # Coordination metrics
        coord_report = coordination_metrics.get_system_coordination_report()

        metrics["active_coalitions"] = DashboardMetric(
            name="Active Coalitions",
            value=coord_report["coalition_stats"]["active_coalitions"],
            unit="coalitions",
            timestamp=timestamp,
            status="healthy",
        )

        metrics["coordination_success_rate"] = DashboardMetric(
            name="Coordination Success Rate",
            value=coord_report["system_success_rate"] * 100,
            unit="%",
            timestamp=timestamp,
            status="healthy"
            if coord_report["system_success_rate"] > 0.8
            else "warning",
        )

        # Update history
        for metric_name, metric in metrics.items():
            self.metric_history[metric_name].append((timestamp, metric.value))

        return metrics

    async def _get_agent_dashboards(self) -> Dict[str, AgentDashboard]:
        """Get individual agent dashboards."""
        dashboards = {}

        # Get all known agents from performance tracker
        for agent_id in performance_tracker.agent_metrics.keys():
            try:
                # Get agent performance metrics
                agent_perf = (
                    await performance_tracker.get_agent_performance_summary(
                        agent_id
                    )
                )

                # Get belief statistics
                belief_stats = belief_monitoring_hooks.get_agent_statistics(
                    agent_id
                )

                # Get coordination statistics
                coord_stats = coordination_metrics.get_coordination_statistics(
                    agent_id
                )

                # Create agent metrics
                current_metrics = {
                    "inference_time": DashboardMetric(
                        name="Inference Time",
                        value=agent_perf["inference_performance"]["avg"],
                        unit="ms",
                        timestamp=datetime.now(),
                        status=self._get_metric_status(
                            "inference_time_ms",
                            agent_perf["inference_performance"]["avg"],
                        ),
                    ),
                    "belief_entropy": DashboardMetric(
                        name="Belief Entropy",
                        value=belief_stats.get("entropy", {}).get("mean", 0.0),
                        unit="bits",
                        timestamp=datetime.now(),
                        status=self._get_metric_status(
                            "belief_entropy",
                            belief_stats.get("entropy", {}).get("mean", 0.0),
                        ),
                    ),
                    "coordination_rate": DashboardMetric(
                        name="Coordination Rate",
                        value=coord_stats.get("total_coordinations", 0),
                        unit="ops",
                        timestamp=datetime.now(),
                        status="healthy",
                    ),
                    "error_rate": DashboardMetric(
                        name="Error Rate",
                        value=agent_perf["error_rate"]["count"],
                        unit="errors",
                        timestamp=datetime.now(),
                        status="healthy"
                        if agent_perf["error_rate"]["count"] == 0
                        else "warning",
                    ),
                }

                # Create time series data
                time_series = {}

                # Add agent dashboard
                dashboards[agent_id] = AgentDashboard(
                    agent_id=agent_id,
                    name=f"Agent {agent_id}",
                    status="active",
                    current_metrics=current_metrics,
                    time_series=time_series,
                    alerts=[],
                )

            except Exception as e:
                logger.error(
                    f"Failed to get dashboard for agent {agent_id}: {e}"
                )

        return dashboards

    async def _get_coalition_data(self) -> List[Dict[str, Any]]:
        """Get active coalition information."""
        coalitions = []

        for (
            coalition_id,
            coalition_metrics,
        ) in coordination_metrics.coalition_metrics.items():
            coalitions.append(
                {
                    "coalition_id": coalition_id,
                    "member_count": coalition_metrics.member_count,
                    "efficiency": coalition_metrics.coordination_efficiency,
                    "task_completion_rate": coalition_metrics.task_completion_rate,
                    "formation_time": coalition_metrics.formation_time.isoformat(),
                }
            )

        return coalitions

    async def _check_alerts(
        self,
        system_metrics: Dict[str, DashboardMetric],
        agent_dashboards: Dict[str, AgentDashboard],
    ) -> List[Dict[str, Any]]:
        """Check for alerts based on thresholds."""
        alerts = []
        timestamp = datetime.now()

        # Check system metrics
        for metric_name, metric in system_metrics.items():
            if metric.status == "warning":
                alerts.append(
                    {
                        "level": "warning",
                        "type": "system",
                        "metric": metric_name,
                        "value": metric.value,
                        "threshold": self.alert_thresholds.get(
                            metric_name, {}
                        ).get("warning"),
                        "timestamp": timestamp.isoformat(),
                        "message": f"System {metric.name} is above warning threshold",
                    }
                )
            elif metric.status == "critical":
                alerts.append(
                    {
                        "level": "critical",
                        "type": "system",
                        "metric": metric_name,
                        "value": metric.value,
                        "threshold": self.alert_thresholds.get(
                            metric_name, {}
                        ).get("critical"),
                        "timestamp": timestamp.isoformat(),
                        "message": f"System {metric.name} is above critical threshold",
                    }
                )

        # Check agent metrics
        for agent_id, dashboard in agent_dashboards.items():
            for metric_name, metric in dashboard.current_metrics.items():
                if metric.status == "warning":
                    alerts.append(
                        {
                            "level": "warning",
                            "type": "agent",
                            "agent_id": agent_id,
                            "metric": metric_name,
                            "value": metric.value,
                            "timestamp": timestamp.isoformat(),
                            "message": f"Agent {agent_id} {metric.name} is above warning threshold",
                        }
                    )
                elif metric.status == "critical":
                    alerts.append(
                        {
                            "level": "critical",
                            "type": "agent",
                            "agent_id": agent_id,
                            "metric": metric_name,
                            "value": metric.value,
                            "timestamp": timestamp.isoformat(),
                            "message": f"Agent {agent_id} {metric.name} is above critical threshold",
                        }
                    )

        # Store alerts in history
        for alert in alerts:
            self.alert_history.append(alert)

        return alerts

    def _get_metric_status(self, metric_type: str, value: float) -> str:
        """Get status for a metric based on thresholds."""
        thresholds = self.alert_thresholds.get(metric_type, {})

        if not thresholds:
            return "healthy"

        if value >= thresholds.get("critical", float("inf")):
            return "critical"
        elif value >= thresholds.get("warning", float("inf")):
            return "warning"
        else:
            return "healthy"

    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record a dashboard event."""
        self.event_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "data": data,
            }
        )

    def get_current_dashboard(self) -> Optional[Dict[str, Any]]:
        """Get current dashboard data as dictionary."""
        if not self.current_dashboard:
            return None

        return self._dashboard_to_dict(self.current_dashboard)

    def _dashboard_to_dict(self, dashboard: SystemDashboard) -> Dict[str, Any]:
        """Convert dashboard to dictionary for JSON serialization."""
        return {
            "timestamp": dashboard.timestamp.isoformat(),
            "system_metrics": {
                name: self._metric_to_dict(metric)
                for name, metric in dashboard.system_metrics.items()
            },
            "agent_dashboards": {
                agent_id: self._agent_dashboard_to_dict(agent_dashboard)
                for agent_id, agent_dashboard in dashboard.agent_dashboards.items()
            },
            "active_coalitions": dashboard.active_coalitions,
            "recent_events": dashboard.recent_events,
            "alerts": dashboard.alerts,
        }

    def _metric_to_dict(self, metric: DashboardMetric) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "timestamp": metric.timestamp.isoformat(),
            "trend": metric.trend,
            "status": metric.status,
        }

    def _agent_dashboard_to_dict(
        self, dashboard: AgentDashboard
    ) -> Dict[str, Any]:
        """Convert agent dashboard to dictionary."""
        return {
            "agent_id": dashboard.agent_id,
            "name": dashboard.name,
            "status": dashboard.status,
            "current_metrics": {
                name: self._metric_to_dict(metric)
                for name, metric in dashboard.current_metrics.items()
            },
            "alerts": dashboard.alerts,
        }

    def get_metric_history(
        self, metric_name: str, duration_minutes: int = 5
    ) -> Optional[TimeSeriesData]:
        """Get historical data for a metric."""
        if metric_name not in self.metric_history:
            return None

        history = self.metric_history[metric_name]
        if not history:
            return None

        # Filter by duration
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        filtered_data = [(ts, val) for ts, val in history if ts >= cutoff]

        if not filtered_data:
            return None

        timestamps, values = zip(*filtered_data)

        return TimeSeriesData(
            metric_name=metric_name,
            timestamps=list(timestamps),
            values=list(values),
            unit=self._get_metric_unit(metric_name),
        )

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for a metric."""
        units = {
            "total_agents": "agents",
            "avg_inference_time": "ms",
            "system_memory": "MB",
            "system_cpu": "%",
            "active_coalitions": "coalitions",
            "coordination_success_rate": "%",
            "inference_time": "ms",
            "belief_entropy": "bits",
            "coordination_rate": "ops",
            "error_rate": "errors",
        }
        return units.get(metric_name, "units")


# Global dashboard instance
monitoring_dashboard = MonitoringDashboard()


# Helper functions
async def start_monitoring_dashboard():
    """Start the monitoring dashboard."""
    await monitoring_dashboard.start()


async def stop_monitoring_dashboard():
    """Stop the monitoring dashboard."""
    await monitoring_dashboard.stop()


def get_dashboard_data() -> Optional[Dict[str, Any]]:
    """Get current dashboard data."""
    return monitoring_dashboard.get_current_dashboard()


def get_metric_time_series(
    metric_name: str, duration_minutes: int = 5
) -> Optional[Dict[str, Any]]:
    """Get time series data for a metric."""
    time_series = monitoring_dashboard.get_metric_history(
        metric_name, duration_minutes
    )
    if not time_series:
        return None

    return {
        "metric_name": time_series.metric_name,
        "timestamps": [ts.isoformat() for ts in time_series.timestamps],
        "values": time_series.values,
        "unit": time_series.unit,
    }


def record_dashboard_event(event_type: str, data: Dict[str, Any]):
    """Record an event for the dashboard."""
    monitoring_dashboard.record_event(event_type, data)

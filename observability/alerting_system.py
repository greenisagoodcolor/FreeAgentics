"""Alerting system for agent failures and performance degradation."""

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts."""

    AGENT_FAILURE = "agent_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    BELIEF_ANOMALY = "belief_anomaly"
    COORDINATION_FAILURE = "coordination_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SYSTEM_ERROR = "system_error"


@dataclass
class Alert:
    """Represents an alert."""

    alert_id: str
    timestamp: datetime
    level: AlertLevel
    alert_type: AlertType
    source: str
    title: str
    message: str
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class AlertingSystem:
    """Manages alerts for agent operations."""

    def __init__(self) -> None:
        """Initialize the alerting system."""
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque[Alert] = deque(maxlen=1000)
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_counts: defaultdict[str, int] = defaultdict(int)

        logger.info("Initialized alerting system")

    async def check_agent_health(self, agent_id: str, agent_data: Dict[str, Any]) -> None:
        """Check agent health and raise alerts if needed."""
        # Check for agent failure
        if agent_data.get("agent_status") == "failed":
            await self._create_alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.AGENT_FAILURE,
                source=agent_id,
                title=f"Agent {agent_id} has failed",
                message=f"Agent {agent_id} reported failure: {agent_data.get('error', 'Unknown error')}",
                metadata=agent_data,
            )

        # Check inference time
        inference_time = agent_data.get("inference_time_ms", 0)
        if inference_time > 200:
            await self._create_alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                source=agent_id,
                title=f"Critical inference time for agent {agent_id}",
                message=f"Agent {agent_id} inference time: {inference_time}ms (critical threshold: 200ms)",
                metadata=agent_data,
            )
        elif inference_time > 100:
            await self._create_alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                source=agent_id,
                title=f"High inference time for agent {agent_id}",
                message=f"Agent {agent_id} inference time: {inference_time}ms (threshold: 100ms)",
                metadata=agent_data,
            )

        # Check for belief anomaly
        if agent_data.get("belief_anomaly", False):
            await self._create_alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.BELIEF_ANOMALY,
                source=agent_id,
                title=f"Belief anomaly detected for agent {agent_id}",
                message=f"Agent {agent_id} belief KL divergence: {agent_data.get('kl_divergence', 'N/A')}",
                metadata=agent_data,
            )

    async def check_system_health(self, system_data: Dict[str, Any]) -> None:
        """Check system health and raise alerts if needed."""
        # Check memory usage
        memory_usage = system_data.get("memory_usage_mb", 0)
        if memory_usage > 1000:
            await self._create_alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                source="system",
                title="Critical memory usage detected",
                message=f"System memory usage: {memory_usage}MB (critical threshold: 1000MB)",
                metadata=system_data,
            )
        elif memory_usage > 800:
            await self._create_alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                source="system",
                title="High memory usage detected",
                message=f"System memory usage: {memory_usage}MB (threshold: 800MB)",
                metadata=system_data,
            )

    async def _create_alert(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        source: str,
        title: str,
        message: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Create and store an alert."""
        # Check cooldown
        alert_key = f"{alert_type.value}_{source}"
        last_alert_time = self.last_alert_times.get(alert_key)
        if last_alert_time:
            cooldown_elapsed = (datetime.now() - last_alert_time).total_seconds() / 60
            if cooldown_elapsed < 5:  # 5 minute cooldown
                return

        # Create alert
        alert_id = f"{alert_key}_{datetime.now().timestamp()}"
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            level=level,
            alert_type=alert_type,
            source=source,
            title=title,
            message=message,
            metadata=metadata,
        )

        # Store alert
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.alert_counts[alert_type.value] += 1
        self.last_alert_times[alert_key] = alert.timestamp

        # Log alert
        if level == AlertLevel.CRITICAL or level == AlertLevel.EMERGENCY:
            logger.error(f"ALERT {level.value}: {title} - {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"ALERT {level.value}: {title} - {message}")
        else:
            logger.info(f"ALERT {level.value}: {title} - {message}")

    def resolve_alert(self, alert_id: str, resolution_notes: Optional[str] = None) -> None:
        """Mark an alert as resolved."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            alert.resolution_notes = resolution_notes
            logger.info(f"Resolved alert: {alert_id}")

    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts."""
        return [a for a in self.alerts.values() if not a.resolved]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_alerts = self.get_active_alerts()

        # Count by level
        level_counts: defaultdict[str, int] = defaultdict(int)
        for alert in active_alerts:
            level_counts[alert.level.value] += 1

        # Count by type
        type_counts: defaultdict[str, int] = defaultdict(int)
        for alert in active_alerts:
            type_counts[alert.alert_type.value] += 1

        return {
            "total_active": len(active_alerts),
            "total_historical": len(self.alert_history),
            "by_level": dict(level_counts),
            "by_type": dict(type_counts),
            "total_counts": dict(self.alert_counts),
            "critical_alerts": [
                {
                    "alert_id": a.alert_id,
                    "title": a.title,
                    "timestamp": a.timestamp.isoformat(),
                    "source": a.source,
                }
                for a in active_alerts
                if a.level == AlertLevel.CRITICAL or a.level == AlertLevel.EMERGENCY
            ],
        }

    def export_alerts(self, format: str = "json") -> str:
        """Export alerts for analysis."""
        alerts = list(self.alert_history)

        if format == "json":
            export_data = {
                "export_time": datetime.now().isoformat(),
                "alert_count": len(alerts),
                "alerts": [
                    {
                        "alert_id": a.alert_id,
                        "timestamp": a.timestamp.isoformat(),
                        "level": a.level.value,
                        "type": a.alert_type.value,
                        "source": a.source,
                        "title": a.title,
                        "message": a.message,
                        "resolved": a.resolved,
                        "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                    }
                    for a in alerts
                ],
            }
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global alerting system instance
alerting_system = AlertingSystem()


# Helper functions
async def check_agent_alert(agent_id: str, metrics: Dict[str, Any]) -> None:
    """Check if agent metrics trigger any alerts."""
    await alerting_system.check_agent_health(agent_id, metrics)


async def check_system_alert(metrics: Dict[str, Any]) -> None:
    """Check if system metrics trigger any alerts."""
    await alerting_system.check_system_health(metrics)


def get_active_alerts() -> List[Dict[str, Any]]:
    """Get all active alerts."""
    alerts = alerting_system.get_active_alerts()
    return [
        {
            "alert_id": a.alert_id,
            "timestamp": a.timestamp.isoformat(),
            "level": a.level.value,
            "type": a.alert_type.value,
            "source": a.source,
            "title": a.title,
            "message": a.message,
        }
        for a in alerts
    ]


def resolve_alert(alert_id: str, notes: Optional[str] = None) -> None:
    """Resolve an alert."""
    alerting_system.resolve_alert(alert_id, notes)


def get_alert_stats() -> Dict[str, Any]:
    """Get alert statistics."""
    return alerting_system.get_alert_statistics()

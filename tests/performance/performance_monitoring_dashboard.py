"""
Performance Monitoring Dashboard with Real-Time Alerts
======================================================

This module provides a comprehensive performance monitoring dashboard featuring:
- Real-time performance metrics visualization
- Automated alerting system for SLA violations
- Performance trend analysis and prediction
- Interactive dashboard with live updates
- Configurable alert thresholds
- Performance baseline tracking
- Anomaly detection and notification
"""

import asyncio
import json
import logging
import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

from observability.performance_monitor import (
    get_performance_monitor,
)

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Configuration for a performance alert rule."""

    name: str
    metric: str
    threshold: float
    condition: str  # 'greater_than', 'less_than', 'equals'
    severity: str  # 'critical', 'warning', 'info'
    duration_seconds: int = 60  # How long condition must persist
    cooldown_seconds: int = 300  # Cooldown period after firing
    enabled: bool = True
    description: str = ""


@dataclass
class AlertNotification:
    """A triggered alert notification."""

    id: str
    rule_name: str
    metric: str
    current_value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""

    timestamp: datetime
    metrics: Dict[str, float]
    alerts_active: int
    system_health_score: float
    trend_indicators: Dict[str, str]  # 'improving', 'stable', 'degrading'


class PerformanceMonitoringDashboard:
    """Real-time performance monitoring dashboard with alerting."""

    def __init__(self, update_interval: float = 5.0):
        self.update_interval = update_interval
        self.performance_monitor = get_performance_monitor()
        self.process = psutil.Process()

        # Data storage
        self.metrics_history: deque = deque(maxlen=1000)  # Last 1000 snapshots
        self.alerts_history: deque = deque(maxlen=500)  # Last 500 alerts
        self.active_alerts: Dict[str, AlertNotification] = {}

        # Alert configuration
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_callbacks: List[Callable] = []

        # Dashboard state
        self.dashboard_running = False
        self.dashboard_thread = None
        self.last_alert_times: Dict[str, datetime] = {}

        # Performance baselines
        self.baselines: Dict[str, float] = {}
        self.baseline_window = (
            100  # Number of samples for baseline calculation
        )

        # Health score components
        self.health_score_weights = {
            'response_time': 0.25,
            'error_rate': 0.25,
            'resource_usage': 0.25,
            'throughput': 0.25,
        }

        # Initialize default alert rules
        self._create_default_alert_rules()

    def _create_default_alert_rules(self):
        """Create default alert rules for common performance issues."""
        default_rules = [
            AlertRule(
                name="high_response_time",
                metric="api_response_time_ms",
                threshold=3000.0,  # 3 seconds
                condition="greater_than",
                severity="critical",
                duration_seconds=60,
                description="API response time exceeds 3 seconds",
            ),
            AlertRule(
                name="high_error_rate",
                metric="api_error_rate",
                threshold=5.0,  # 5%
                condition="greater_than",
                severity="critical",
                duration_seconds=120,
                description="API error rate exceeds 5%",
            ),
            AlertRule(
                name="high_memory_usage",
                metric="memory_usage",
                threshold=85.0,  # 85%
                condition="greater_than",
                severity="warning",
                duration_seconds=300,
                description="Memory usage exceeds 85%",
            ),
            AlertRule(
                name="high_cpu_usage",
                metric="cpu_usage",
                threshold=80.0,  # 80%
                condition="greater_than",
                severity="warning",
                duration_seconds=180,
                description="CPU usage exceeds 80%",
            ),
            AlertRule(
                name="low_throughput",
                metric="api_requests_per_second",
                threshold=10.0,  # 10 RPS
                condition="less_than",
                severity="warning",
                duration_seconds=300,
                description="API throughput below 10 requests per second",
            ),
            AlertRule(
                name="database_slow_queries",
                metric="db_query_time_ms",
                threshold=500.0,  # 500ms
                condition="greater_than",
                severity="warning",
                duration_seconds=240,
                description="Database queries taking longer than 500ms",
            ),
            AlertRule(
                name="agent_slow_processing",
                metric="agent_step_time_ms",
                threshold=200.0,  # 200ms
                condition="greater_than",
                severity="warning",
                duration_seconds=180,
                description="Agent processing time exceeds 200ms",
            ),
            AlertRule(
                name="websocket_connection_drop",
                metric="websocket_connections",
                threshold=5.0,  # Less than 5 connections
                condition="less_than",
                severity="info",
                duration_seconds=120,
                description="WebSocket connections below expected threshold",
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.name] = rule

    def start_monitoring(self):
        """Start the performance monitoring dashboard."""
        if self.dashboard_running:
            logger.warning("Dashboard already running")
            return

        self.dashboard_running = True
        self.dashboard_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.dashboard_thread.start()

        # Start the underlying performance monitor
        self.performance_monitor.start_monitoring()

        logger.info("Performance monitoring dashboard started")

    def stop_monitoring(self):
        """Stop the performance monitoring dashboard."""
        self.dashboard_running = False

        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=10)

        self.performance_monitor.stop_monitoring()

        logger.info("Performance monitoring dashboard stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.dashboard_running:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()

                # Update baselines
                self._update_baselines(current_metrics)

                # Calculate health score
                health_score = self._calculate_health_score(current_metrics)

                # Calculate trend indicators
                trend_indicators = self._calculate_trends(current_metrics)

                # Create snapshot
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.now(),
                    metrics=current_metrics,
                    alerts_active=len(self.active_alerts),
                    system_health_score=health_score,
                    trend_indicators=trend_indicators,
                )

                # Store snapshot
                self.metrics_history.append(snapshot)

                # Check for alert conditions
                self._check_alert_conditions(current_metrics)

                # Clean up resolved alerts
                self._cleanup_resolved_alerts(current_metrics)

                # Log dashboard status
                self._log_dashboard_status(snapshot)

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)

    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        # Get metrics from performance monitor
        current_perf_metrics = self.performance_monitor.get_current_metrics()

        metrics = {}

        if current_perf_metrics:
            metrics.update(
                {
                    'cpu_usage': current_perf_metrics.cpu_usage,
                    'memory_usage': current_perf_metrics.memory_usage,
                    'memory_rss_mb': current_perf_metrics.memory_rss_mb,
                    'thread_count': current_perf_metrics.thread_count,
                    'api_requests_per_second': current_perf_metrics.api_requests_per_second,
                    'api_response_time_ms': current_perf_metrics.api_response_time_ms,
                    'api_error_rate': current_perf_metrics.api_error_rate,
                    'db_query_time_ms': current_perf_metrics.db_query_time_ms,
                    'db_connections': current_perf_metrics.db_connections,
                    'agent_count': current_perf_metrics.agent_count,
                    'agent_step_time_ms': current_perf_metrics.agent_step_time_ms,
                    'websocket_connections': current_perf_metrics.websocket_connections,
                    'websocket_messages_per_second': current_perf_metrics.websocket_messages_per_second,
                    'gil_contention': current_perf_metrics.gil_contention,
                }
            )

        # Add system metrics
        try:
            system_metrics = {
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'network_io_bytes_sent': psutil.net_io_counters().bytes_sent,
                'network_io_bytes_recv': psutil.net_io_counters().bytes_recv,
                'load_average_1m': psutil.getloadavg()[0]
                if hasattr(psutil, 'getloadavg')
                else 0,
                'swap_usage_percent': psutil.swap_memory().percent,
            }
            metrics.update(system_metrics)
        except Exception as e:
            logger.debug(f"Error collecting system metrics: {e}")

        return metrics

    def _update_baselines(self, current_metrics: Dict[str, float]):
        """Update performance baselines."""
        if len(self.metrics_history) < self.baseline_window:
            return

        # Calculate baselines from recent history
        recent_snapshots = list(self.metrics_history)[-self.baseline_window :]

        for metric_name in current_metrics.keys():
            values = []
            for snapshot in recent_snapshots:
                if metric_name in snapshot.metrics:
                    values.append(snapshot.metrics[metric_name])

            if values:
                # Use median for baseline to avoid outliers
                self.baselines[metric_name] = statistics.median(values)

    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0

        # Response time component
        response_time = metrics.get('api_response_time_ms', 0)
        if response_time > 0:
            if response_time > 5000:  # >5s is very bad
                score -= 40 * self.health_score_weights['response_time']
            elif response_time > 3000:  # >3s is bad
                score -= 25 * self.health_score_weights['response_time']
            elif response_time > 1000:  # >1s is concerning
                score -= 15 * self.health_score_weights['response_time']

        # Error rate component
        error_rate = metrics.get('api_error_rate', 0)
        if error_rate > 10:  # >10% error rate
            score -= 40 * self.health_score_weights['error_rate']
        elif error_rate > 5:  # >5% error rate
            score -= 25 * self.health_score_weights['error_rate']
        elif error_rate > 1:  # >1% error rate
            score -= 10 * self.health_score_weights['error_rate']

        # Resource usage component
        cpu_usage = metrics.get('cpu_usage', 0)
        memory_usage = metrics.get('memory_usage', 0)

        if cpu_usage > 90 or memory_usage > 90:
            score -= 35 * self.health_score_weights['resource_usage']
        elif cpu_usage > 75 or memory_usage > 75:
            score -= 20 * self.health_score_weights['resource_usage']
        elif cpu_usage > 60 or memory_usage > 60:
            score -= 10 * self.health_score_weights['resource_usage']

        # Throughput component
        throughput = metrics.get('api_requests_per_second', 0)
        if throughput < 5:  # Very low throughput
            score -= 25 * self.health_score_weights['throughput']
        elif throughput < 10:  # Low throughput
            score -= 15 * self.health_score_weights['throughput']
        elif throughput < 20:  # Moderate throughput
            score -= 5 * self.health_score_weights['throughput']

        return max(score, 0)

    def _calculate_trends(
        self, current_metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Calculate trend indicators for key metrics."""
        trends = {}

        if len(self.metrics_history) < 10:
            return {metric: 'stable' for metric in current_metrics.keys()}

        # Look at last 10 snapshots for trend analysis
        recent_snapshots = list(self.metrics_history)[-10:]

        for metric_name in current_metrics.keys():
            values = []
            for snapshot in recent_snapshots:
                if metric_name in snapshot.metrics:
                    values.append(snapshot.metrics[metric_name])

            if len(values) >= 5:
                # Calculate trend using linear regression slope
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]

                # Determine trend based on slope and metric type
                if metric_name in [
                    'api_response_time_ms',
                    'api_error_rate',
                    'cpu_usage',
                    'memory_usage',
                ]:
                    # For these metrics, increasing is bad
                    if slope > 0.1:
                        trends[metric_name] = 'degrading'
                    elif slope < -0.1:
                        trends[metric_name] = 'improving'
                    else:
                        trends[metric_name] = 'stable'
                else:
                    # For metrics like throughput, increasing is good
                    if slope > 0.1:
                        trends[metric_name] = 'improving'
                    elif slope < -0.1:
                        trends[metric_name] = 'degrading'
                    else:
                        trends[metric_name] = 'stable'
            else:
                trends[metric_name] = 'stable'

        return trends

    def _check_alert_conditions(self, metrics: Dict[str, float]):
        """Check all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            metric_value = metrics.get(rule.metric, 0)

            # Check if condition is met
            condition_met = False
            if rule.condition == 'greater_than':
                condition_met = metric_value > rule.threshold
            elif rule.condition == 'less_than':
                condition_met = metric_value < rule.threshold
            elif rule.condition == 'equals':
                condition_met = abs(metric_value - rule.threshold) < 0.01

            if condition_met:
                # Check if we're in cooldown period
                if rule_name in self.last_alert_times:
                    time_since_last = (
                        datetime.now() - self.last_alert_times[rule_name]
                    ).total_seconds()
                    if time_since_last < rule.cooldown_seconds:
                        continue

                # Check if condition has persisted long enough
                if rule_name not in self.active_alerts:
                    # Start tracking this potential alert
                    self._start_alert_tracking(rule_name, rule, metric_value)
                else:
                    # Check if duration threshold is met
                    alert = self.active_alerts[rule_name]
                    duration = (
                        datetime.now() - alert.timestamp
                    ).total_seconds()

                    if (
                        duration >= rule.duration_seconds
                        and not alert.acknowledged
                    ):
                        # Fire the alert
                        self._fire_alert(rule_name, rule, metric_value)
            else:
                # Condition not met, remove from active tracking
                if rule_name in self.active_alerts:
                    self._resolve_alert(rule_name, metrics)

    def _start_alert_tracking(
        self, rule_name: str, rule: AlertRule, metric_value: float
    ):
        """Start tracking a potential alert."""
        alert = AlertNotification(
            id=str(uuid.uuid4()),
            rule_name=rule_name,
            metric=rule.metric,
            current_value=metric_value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=f"{rule.description} - {rule.metric}: {metric_value:.2f} (threshold: {rule.threshold:.2f})",
        )

        self.active_alerts[rule_name] = alert
        logger.debug(f"Started tracking alert: {rule_name}")

    def _fire_alert(
        self, rule_name: str, rule: AlertRule, metric_value: float
    ):
        """Fire an alert notification."""
        alert = self.active_alerts[rule_name]
        alert.current_value = metric_value
        alert.message = f"{rule.description} - {rule.metric}: {metric_value:.2f} (threshold: {rule.threshold:.2f})"

        # Add to alerts history
        self.alerts_history.append(alert)

        # Update last alert time
        self.last_alert_times[rule_name] = datetime.now()

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        logger.warning(
            f"ALERT FIRED: {alert.severity.upper()} - {alert.message}"
        )

    def _resolve_alert(self, rule_name: str, metrics: Dict[str, float]):
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolution_time = datetime.now()

            # Remove from active alerts
            del self.active_alerts[rule_name]

            logger.info(f"ALERT RESOLVED: {alert.message}")

    def _cleanup_resolved_alerts(self, metrics: Dict[str, float]):
        """Clean up resolved alerts from active tracking."""
        # This is handled in _resolve_alert, but we can add additional cleanup logic here
        pass

    def _log_dashboard_status(self, snapshot: PerformanceSnapshot):
        """Log current dashboard status."""
        if (
            len(self.metrics_history) % 12 == 0
        ):  # Log every 12 updates (1 minute at 5s intervals)
            logger.info(
                f"Dashboard Status - Health: {snapshot.system_health_score:.1f}%, "
                f"Active Alerts: {snapshot.alerts_active}, "
                f"CPU: {snapshot.metrics.get('cpu_usage', 0):.1f}%, "
                f"Memory: {snapshot.metrics.get('memory_usage', 0):.1f}%, "
                f"Response Time: {snapshot.metrics.get('api_response_time_ms', 0):.1f}ms"
            )

    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")

    def add_alert_callback(
        self, callback: Callable[[AlertNotification], None]
    ):
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert.message}")
                break

    def get_current_status(self) -> Dict[str, Any]:
        """Get current dashboard status."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}

        latest_snapshot = self.metrics_history[-1]

        return {
            'timestamp': latest_snapshot.timestamp.isoformat(),
            'health_score': latest_snapshot.system_health_score,
            'active_alerts': len(self.active_alerts),
            'metrics': latest_snapshot.metrics,
            'trends': latest_snapshot.trend_indicators,
            'baselines': self.baselines.copy(),
            'alerts': [
                {
                    'id': alert.id,
                    'rule_name': alert.rule_name,
                    'metric': alert.metric,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged,
                }
                for alert in self.active_alerts.values()
            ],
        }

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        history = []
        for snapshot in self.metrics_history:
            if snapshot.timestamp >= cutoff_time:
                history.append(
                    {
                        'timestamp': snapshot.timestamp.isoformat(),
                        'metrics': snapshot.metrics,
                        'health_score': snapshot.system_health_score,
                        'alerts_active': snapshot.alerts_active,
                    }
                )

        return history

    def get_alerts_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        history = []
        for alert in self.alerts_history:
            if alert.timestamp >= cutoff_time:
                history.append(
                    {
                        'id': alert.id,
                        'rule_name': alert.rule_name,
                        'metric': alert.metric,
                        'current_value': alert.current_value,
                        'threshold': alert.threshold,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'acknowledged': alert.acknowledged,
                        'resolved': alert.resolved,
                        'resolution_time': alert.resolution_time.isoformat()
                        if alert.resolution_time
                        else None,
                    }
                )

        return history

    def generate_dashboard_report(self) -> Dict[str, Any]:
        """Generate a comprehensive dashboard report."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}

        latest_snapshot = self.metrics_history[-1]

        # Calculate summary statistics
        recent_snapshots = list(self.metrics_history)[
            -50:
        ]  # Last 50 snapshots

        avg_health_score = statistics.mean(
            [s.system_health_score for s in recent_snapshots]
        )
        avg_response_time = statistics.mean(
            [
                s.metrics.get('api_response_time_ms', 0)
                for s in recent_snapshots
            ]
        )
        avg_cpu_usage = statistics.mean(
            [s.metrics.get('cpu_usage', 0) for s in recent_snapshots]
        )
        avg_memory_usage = statistics.mean(
            [s.metrics.get('memory_usage', 0) for s in recent_snapshots]
        )

        # Alert statistics
        alert_counts = {}
        for alert in self.alerts_history:
            alert_counts[alert.severity] = (
                alert_counts.get(alert.severity, 0) + 1
            )

        # Trend analysis
        degrading_trends = [
            k
            for k, v in latest_snapshot.trend_indicators.items()
            if v == 'degrading'
        ]
        improving_trends = [
            k
            for k, v in latest_snapshot.trend_indicators.items()
            if v == 'improving'
        ]

        report = {
            'timestamp': datetime.now().isoformat(),
            'dashboard_summary': {
                'current_health_score': latest_snapshot.system_health_score,
                'average_health_score': avg_health_score,
                'active_alerts': len(self.active_alerts),
                'total_alerts_24h': len(self.get_alerts_history(24)),
                'monitoring_uptime_hours': len(self.metrics_history)
                * self.update_interval
                / 3600,
            },
            'performance_summary': {
                'current_response_time_ms': latest_snapshot.metrics.get(
                    'api_response_time_ms', 0
                ),
                'average_response_time_ms': avg_response_time,
                'current_cpu_usage': latest_snapshot.metrics.get(
                    'cpu_usage', 0
                ),
                'average_cpu_usage': avg_cpu_usage,
                'current_memory_usage': latest_snapshot.metrics.get(
                    'memory_usage', 0
                ),
                'average_memory_usage': avg_memory_usage,
                'current_throughput': latest_snapshot.metrics.get(
                    'api_requests_per_second', 0
                ),
                'error_rate': latest_snapshot.metrics.get('api_error_rate', 0),
            },
            'alert_summary': {
                'active_alerts': len(self.active_alerts),
                'critical_alerts': len(
                    [
                        a
                        for a in self.active_alerts.values()
                        if a.severity == 'critical'
                    ]
                ),
                'warning_alerts': len(
                    [
                        a
                        for a in self.active_alerts.values()
                        if a.severity == 'warning'
                    ]
                ),
                'info_alerts': len(
                    [
                        a
                        for a in self.active_alerts.values()
                        if a.severity == 'info'
                    ]
                ),
                'alert_counts_24h': alert_counts,
            },
            'trend_analysis': {
                'degrading_metrics': degrading_trends,
                'improving_metrics': improving_trends,
                'stable_metrics': [
                    k
                    for k, v in latest_snapshot.trend_indicators.items()
                    if v == 'stable'
                ],
            },
            'recommendations': self._generate_dashboard_recommendations(
                latest_snapshot
            ),
        }

        return report

    def _generate_dashboard_recommendations(
        self, snapshot: PerformanceSnapshot
    ) -> List[str]:
        """Generate recommendations based on current system state."""
        recommendations = []

        # Health score recommendations
        if snapshot.system_health_score < 70:
            recommendations.append(
                "System health score is low. Investigate active alerts and performance bottlenecks."
            )

        # Response time recommendations
        response_time = snapshot.metrics.get('api_response_time_ms', 0)
        if response_time > 3000:
            recommendations.append(
                "API response time is high. Consider implementing caching and optimizing slow endpoints."
            )

        # Resource usage recommendations
        cpu_usage = snapshot.metrics.get('cpu_usage', 0)
        memory_usage = snapshot.metrics.get('memory_usage', 0)

        if cpu_usage > 80:
            recommendations.append(
                "High CPU usage detected. Consider optimizing CPU-intensive operations or scaling horizontally."
            )

        if memory_usage > 85:
            recommendations.append(
                "High memory usage detected. Check for memory leaks and consider increasing memory limits."
            )

        # Alert recommendations
        if len(self.active_alerts) > 5:
            recommendations.append(
                "Multiple active alerts detected. Review alert thresholds and investigate system issues."
            )

        # Trend recommendations
        degrading_trends = [
            k for k, v in snapshot.trend_indicators.items() if v == 'degrading'
        ]
        if degrading_trends:
            recommendations.append(
                f"Degrading trends detected in: {', '.join(degrading_trends)}. Monitor these metrics closely."
            )

        if not recommendations:
            recommendations.append(
                "System is performing well. Continue monitoring for any changes."
            )

        return recommendations

    def export_metrics(self, filename: str):
        """Export metrics history to a file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics_history': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'metrics': snapshot.metrics,
                    'health_score': snapshot.system_health_score,
                    'alerts_active': snapshot.alerts_active,
                    'trends': snapshot.trend_indicators,
                }
                for snapshot in self.metrics_history
            ],
            'alerts_history': [
                {
                    'id': alert.id,
                    'rule_name': alert.rule_name,
                    'metric': alert.metric,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved,
                    'resolution_time': alert.resolution_time.isoformat()
                    if alert.resolution_time
                    else None,
                }
                for alert in self.alerts_history
            ],
            'baselines': self.baselines,
            'alert_rules': {
                name: {
                    'name': rule.name,
                    'metric': rule.metric,
                    'threshold': rule.threshold,
                    'condition': rule.condition,
                    'severity': rule.severity,
                    'duration_seconds': rule.duration_seconds,
                    'enabled': rule.enabled,
                    'description': rule.description,
                }
                for name, rule in self.alert_rules.items()
            },
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics exported to {filename}")


# Example usage and demonstration
def demo_alert_callback(alert: AlertNotification):
    """Demo alert callback function."""
    print(f"🚨 ALERT: {alert.severity.upper()} - {alert.message}")


async def run_performance_dashboard_demo():
    """Run performance monitoring dashboard demo."""
    print("=" * 80)
    print("PERFORMANCE MONITORING DASHBOARD DEMO")
    print("=" * 80)

    # Create dashboard
    dashboard = PerformanceMonitoringDashboard(update_interval=2.0)

    # Add alert callback
    dashboard.add_alert_callback(demo_alert_callback)

    # Add custom alert rule
    custom_rule = AlertRule(
        name="demo_high_load",
        metric="api_requests_per_second",
        threshold=100.0,
        condition="greater_than",
        severity="info",
        duration_seconds=30,
        description="High API load detected",
    )
    dashboard.add_alert_rule(custom_rule)

    try:
        # Start monitoring
        dashboard.start_monitoring()

        print("Dashboard started. Monitoring for 60 seconds...")
        print("You can check the dashboard status every 10 seconds.")

        # Run for 60 seconds
        for i in range(6):
            await asyncio.sleep(10)

            # Get current status
            status = dashboard.get_current_status()

            print(f"\n--- Status Update {i+1} ---")
            print(f"Health Score: {status['health_score']:.1f}%")
            print(f"Active Alerts: {status['active_alerts']}")
            print(f"CPU Usage: {status['metrics'].get('cpu_usage', 0):.1f}%")
            print(
                f"Memory Usage: {status['metrics'].get('memory_usage', 0):.1f}%"
            )
            print(
                f"Response Time: {status['metrics'].get('api_response_time_ms', 0):.1f}ms"
            )

            if status['alerts']:
                print("Active Alerts:")
                for alert in status['alerts']:
                    print(
                        f"  - {alert['severity'].upper()}: {alert['message']}"
                    )

        # Generate final report
        print("\n--- FINAL REPORT ---")
        report = dashboard.generate_dashboard_report()

        print(
            f"Average Health Score: {report['dashboard_summary']['average_health_score']:.1f}%"
        )
        print(
            f"Total Alerts (24h): {report['dashboard_summary']['total_alerts_24h']}"
        )
        print(
            f"Average Response Time: {report['performance_summary']['average_response_time_ms']:.1f}ms"
        )
        print(
            f"Average CPU Usage: {report['performance_summary']['average_cpu_usage']:.1f}%"
        )
        print(
            f"Average Memory Usage: {report['performance_summary']['average_memory_usage']:.1f}%"
        )

        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

        # Export metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_dashboard_metrics_{timestamp}.json"
        dashboard.export_metrics(filename)
        print(f"\nMetrics exported to: {filename}")

    finally:
        # Stop monitoring
        dashboard.stop_monitoring()
        print("\nDashboard stopped.")


if __name__ == "__main__":
    asyncio.run(run_performance_dashboard_demo())

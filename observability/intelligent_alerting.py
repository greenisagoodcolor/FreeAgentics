"""
Intelligent Alerting System for FreeAgentics Production Monitoring.

This module provides advanced alerting capabilities with machine learning-based
anomaly detection, adaptive thresholds, and intelligent alert routing.
"""

import asyncio
import json
import logging
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from observability.performance_metrics import performance_tracker

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


class AlertType(Enum):
    """Alert types."""

    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    CORRELATION = "correlation"
    PREDICTION = "prediction"


@dataclass
class AlertRule:
    """Alert rule definition."""

    id: str
    name: str
    description: str
    severity: AlertSeverity
    alert_type: AlertType
    metric_name: str
    conditions: Dict[str, Any]
    threshold_value: Optional[float] = None
    threshold_operator: str = ">"  # >, <, >=, <=, ==, !=
    time_window: int = 300  # seconds
    evaluation_frequency: int = 60  # seconds
    min_data_points: int = 5
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    runbook_url: Optional[str] = None
    dashboard_url: Optional[str] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


@dataclass
class Alert:
    """Alert instance."""

    id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    alert_type: AlertType
    message: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: Optional[float]
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    starts_at: datetime = field(default_factory=datetime.now)
    ends_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    suppressed_until: Optional[datetime] = None
    fingerprint: str = field(default="")

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for the alert."""
        data = f"{self.rule_id}:{self.metric_name}:{self.labels.get('instance', '')}"
        return str(hash(data))

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "alert_type": self.alert_type.value,
            "message": self.message,
            "description": self.description,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "labels": self.labels,
            "annotations": self.annotations,
            "starts_at": self.starts_at.isoformat(),
            "ends_at": self.ends_at.isoformat() if self.ends_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "suppressed_until": (
                self.suppressed_until.isoformat() if self.suppressed_until else None
            ),
            "fingerprint": self.fingerprint,
        }


class AnomalyDetector:
    """Machine learning-based anomaly detector."""

    def __init__(self, contamination: float = 0.1):
        """Initialize anomaly detector."""
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.training_data = []
        self.min_training_samples = 100

    def add_training_data(self, value: float, timestamp: datetime):
        """Add training data point."""
        self.training_data.append(
            {
                "value": value,
                "timestamp": timestamp,
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "minute": timestamp.minute,
            }
        )

        # Keep only recent data for training
        cutoff_time = datetime.now() - timedelta(days=7)
        self.training_data = [d for d in self.training_data if d["timestamp"] > cutoff_time]

    def train(self):
        """Train the anomaly detection model."""
        if len(self.training_data) < self.min_training_samples:
            return False

        # Prepare features
        features = []
        for data in self.training_data:
            features.append(
                [
                    data["value"],
                    data["hour"],
                    data["day_of_week"],
                    data["minute"],
                ]
            )

        features = np.array(features)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train model
        self.model.fit(features_scaled)
        self.trained = True

        return True

    def detect_anomaly(self, value: float, timestamp: datetime) -> Tuple[bool, float]:
        """Detect if value is anomalous."""
        if not self.trained:
            return False, 0.0

        # Prepare features
        features = np.array([[value, timestamp.hour, timestamp.weekday(), timestamp.minute]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict anomaly
        prediction = self.model.predict(features_scaled)[0]
        anomaly_score = self.model.decision_function(features_scaled)[0]

        is_anomaly = prediction == -1

        return is_anomaly, abs(anomaly_score)


class AdaptiveThresholdManager:
    """Manages adaptive thresholds based on historical data."""

    def __init__(self, window_size: int = 1000):
        """Initialize adaptive threshold manager."""
        self.window_size = window_size
        self.metric_data = {}
        self.thresholds = {}

    def add_data_point(self, metric_name: str, value: float, timestamp: datetime):
        """Add data point for metric."""
        if metric_name not in self.metric_data:
            self.metric_data[metric_name] = []

        self.metric_data[metric_name].append({"value": value, "timestamp": timestamp})

        # Keep only recent data
        if len(self.metric_data[metric_name]) > self.window_size:
            self.metric_data[metric_name].pop(0)

    def calculate_adaptive_thresholds(self, metric_name: str) -> Dict[str, float]:
        """Calculate adaptive thresholds for metric."""
        if metric_name not in self.metric_data:
            return {}

        values = [d["value"] for d in self.metric_data[metric_name]]

        if len(values) < 10:
            return {}

        # Calculate statistics
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        median = statistics.median(values)

        # Calculate percentiles
        p75 = np.percentile(values, 75)
        p90 = np.percentile(values, 90)
        p95 = np.percentile(values, 95)
        p99 = np.percentile(values, 99)

        # Calculate adaptive thresholds
        thresholds = {
            "warning": mean + 2 * std_dev,
            "critical": mean + 3 * std_dev,
            "p75": p75,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
        }

        self.thresholds[metric_name] = thresholds

        return thresholds

    def get_threshold(self, metric_name: str, threshold_type: str = "warning") -> Optional[float]:
        """Get threshold for metric."""
        if metric_name not in self.thresholds:
            self.calculate_adaptive_thresholds(metric_name)

        return self.thresholds.get(metric_name, {}).get(threshold_type)


class AlertCorrelationEngine:
    """Engine for detecting correlated alerts."""

    def __init__(self):
        """Initialize correlation engine."""
        self.correlation_rules = []
        self.alert_history = []
        self.correlation_window = 300  # 5 minutes

    def add_correlation_rule(self, rule: Dict[str, Any]):
        """Add correlation rule."""
        self.correlation_rules.append(rule)

    def analyze_correlations(self, new_alert: Alert) -> List[Dict[str, Any]]:
        """Analyze correlations for new alert."""
        correlations = []

        # Get recent alerts
        cutoff_time = datetime.now() - timedelta(seconds=self.correlation_window)
        recent_alerts = [
            alert
            for alert in self.alert_history
            if alert.starts_at > cutoff_time and alert.id != new_alert.id
        ]

        # Check correlation rules
        for rule in self.correlation_rules:
            if self._matches_correlation_rule(new_alert, recent_alerts, rule):
                correlations.append(
                    {
                        "rule": rule,
                        "related_alerts": [
                            alert.id
                            for alert in recent_alerts
                            if self._alert_matches_rule_criteria(alert, rule)
                        ],
                    }
                )

        return correlations

    def _matches_correlation_rule(
        self, alert: Alert, recent_alerts: List[Alert], rule: Dict[str, Any]
    ) -> bool:
        """Check if alert matches correlation rule."""
        # Implementation would depend on specific correlation rules
        # This is a simplified version

        required_alerts = rule.get("required_alerts", [])
        matching_alerts = 0

        for required_alert in required_alerts:
            for recent_alert in recent_alerts:
                if self._alert_matches_rule_criteria(recent_alert, required_alert):
                    matching_alerts += 1
                    break

        return matching_alerts >= len(required_alerts)

    def _alert_matches_rule_criteria(self, alert: Alert, criteria: Dict[str, Any]) -> bool:
        """Check if alert matches rule criteria."""
        if criteria.get("severity") and alert.severity.value != criteria["severity"]:
            return False

        if criteria.get("metric_name") and alert.metric_name != criteria["metric_name"]:
            return False

        if criteria.get("labels"):
            for key, value in criteria["labels"].items():
                if alert.labels.get(key) != value:
                    return False

        return True


class IntelligentAlertingSystem:
    """Main intelligent alerting system."""

    def __init__(self):
        """Initialize intelligent alerting system."""
        self.rules = {}
        self.active_alerts = {}
        self.alert_history = []
        self.alert_callbacks = []
        self.suppression_rules = []

        # Components
        self.anomaly_detectors = {}
        self.adaptive_threshold_manager = AdaptiveThresholdManager()
        self.correlation_engine = AlertCorrelationEngine()

        # Control
        self.running = False
        self.evaluation_task = None
        self.training_task = None

        # Statistics
        self.stats = {
            "total_alerts": 0,
            "alerts_by_severity": {severity.value: 0 for severity in AlertSeverity},
            "alerts_by_type": {alert_type.value: 0 for alert_type in AlertType},
            "false_positives": 0,
            "true_positives": 0,
            "suppressed_alerts": 0,
        }

        # Setup default rules
        self._setup_default_rules()

        logger.info("🧠 Intelligent alerting system initialized")

    def _setup_default_rules(self):
        """Setup default alerting rules."""
        # System resource rules
        self.add_rule(
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="System CPU usage is above threshold",
                severity=AlertSeverity.HIGH,
                alert_type=AlertType.THRESHOLD,
                metric_name="cpu_usage",
                threshold_value=80.0,
                threshold_operator=">",
                time_window=300,
                evaluation_frequency=60,
                runbook_url="https://docs.freeagentics.com/runbooks/high-cpu",
            )
        )

        self.add_rule(
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                description="System memory usage is above threshold",
                severity=AlertSeverity.HIGH,
                alert_type=AlertType.THRESHOLD,
                metric_name="memory_usage",
                threshold_value=85.0,
                threshold_operator=">",
                time_window=300,
                evaluation_frequency=60,
                runbook_url="https://docs.freeagentics.com/runbooks/high-memory",
            )
        )

        # Agent-specific rules
        self.add_rule(
            AlertRule(
                id="agent_coordination_limit",
                name="Agent Coordination Limit Exceeded",
                description="Number of active agents exceeds coordination limit",
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.THRESHOLD,
                metric_name="active_agents",
                threshold_value=50.0,
                threshold_operator=">",
                time_window=60,
                evaluation_frequency=30,
                runbook_url="https://docs.freeagentics.com/runbooks/agent-coordination",
            )
        )

        # Anomaly detection rules
        self.add_rule(
            AlertRule(
                id="api_response_time_anomaly",
                name="API Response Time Anomaly",
                description="API response time shows anomalous behavior",
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.ANOMALY,
                metric_name="api_response_time",
                time_window=600,
                evaluation_frequency=120,
                runbook_url="https://docs.freeagentics.com/runbooks/api-anomaly",
            )
        )

        # Business metrics rules
        self.add_rule(
            AlertRule(
                id="low_user_interaction_rate",
                name="Low User Interaction Rate",
                description="User interaction rate is below expected threshold",
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.THRESHOLD,
                metric_name="user_interaction_rate",
                threshold_value=0.01,
                threshold_operator="<",
                time_window=3600,
                evaluation_frequency=600,
                runbook_url="https://docs.freeagentics.com/runbooks/user-interactions",
            )
        )

        # Setup correlation rules
        self.correlation_engine.add_correlation_rule(
            {
                "id": "database_cascade",
                "name": "Database Cascade Failure",
                "required_alerts": [
                    {
                        "metric_name": "database_connections",
                        "severity": "high",
                    },
                    {"metric_name": "api_response_time", "severity": "high"},
                ],
                "correlation_type": "cascade",
            }
        )

    async def start(self):
        """Start the intelligent alerting system."""
        if self.running:
            logger.warning("Intelligent alerting system already running")
            return

        self.running = True

        # Start evaluation task
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())

        # Start training task
        self.training_task = asyncio.create_task(self._training_loop())

        logger.info("🚀 Intelligent alerting system started")

    async def stop(self):
        """Stop the intelligent alerting system."""
        if not self.running:
            logger.warning("Intelligent alerting system not running")
            return

        self.running = False

        # Stop tasks
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass

        if self.training_task:
            self.training_task.cancel()
            try:
                await self.training_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Intelligent alerting system stopped")

    async def _evaluation_loop(self):
        """Main evaluation loop."""
        while self.running:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
                await asyncio.sleep(30)

    async def _training_loop(self):
        """Background training loop for anomaly detectors."""
        while self.running:
            try:
                await self._train_anomaly_detectors()
                await asyncio.sleep(3600)  # Train every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(3600)

    async def _evaluate_all_rules(self):
        """Evaluate all alert rules."""
        for rule in self.rules.values():
            if rule.enabled:
                try:
                    await self._evaluate_rule(rule)
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.id}: {e}")

    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        # Get current metric value
        current_value = await self._get_metric_value(rule.metric_name)

        if current_value is None:
            return

        # Add to adaptive threshold manager
        self.adaptive_threshold_manager.add_data_point(
            rule.metric_name, current_value, datetime.now()
        )

        # Evaluate based on alert type
        should_alert = False
        alert_message = ""

        if rule.alert_type == AlertType.THRESHOLD:
            should_alert, alert_message = self._evaluate_threshold_rule(rule, current_value)
        elif rule.alert_type == AlertType.ANOMALY:
            should_alert, alert_message = await self._evaluate_anomaly_rule(rule, current_value)

        # Check if alert should be triggered
        if should_alert:
            existing_alert = self._find_existing_alert(rule.id, rule.metric_name)

            if not existing_alert:
                # Create new alert
                alert = Alert(
                    id=str(uuid.uuid4()),
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    alert_type=rule.alert_type,
                    message=alert_message,
                    description=rule.description,
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold_value=rule.threshold_value,
                    labels=rule.tags,
                    annotations={
                        "runbook_url": rule.runbook_url or "",
                        "dashboard_url": rule.dashboard_url or "",
                    },
                )

                await self._trigger_alert(alert)
        else:
            # Check if we should resolve existing alert
            existing_alert = self._find_existing_alert(rule.id, rule.metric_name)
            if existing_alert and existing_alert.status == AlertStatus.ACTIVE:
                await self._resolve_alert(existing_alert)

    def _evaluate_threshold_rule(self, rule: AlertRule, current_value: float) -> Tuple[bool, str]:
        """Evaluate threshold-based rule."""
        if rule.threshold_value is None:
            return False, ""

        # Check threshold condition
        should_alert = False

        if rule.threshold_operator == ">":
            should_alert = current_value > rule.threshold_value
        elif rule.threshold_operator == "<":
            should_alert = current_value < rule.threshold_value
        elif rule.threshold_operator == ">=":
            should_alert = current_value >= rule.threshold_value
        elif rule.threshold_operator == "<=":
            should_alert = current_value <= rule.threshold_value
        elif rule.threshold_operator == "==":
            should_alert = current_value == rule.threshold_value
        elif rule.threshold_operator == "!=":
            should_alert = current_value != rule.threshold_value

        if should_alert:
            message = f"{rule.name}: {rule.metric_name} = {current_value:.2f} {rule.threshold_operator} {rule.threshold_value}"
            return True, message

        return False, ""

    async def _evaluate_anomaly_rule(
        self, rule: AlertRule, current_value: float
    ) -> Tuple[bool, str]:
        """Evaluate anomaly-based rule."""
        detector = self._get_anomaly_detector(rule.metric_name)

        # Add training data
        detector.add_training_data(current_value, datetime.now())

        # Check if we can detect anomalies
        if not detector.trained:
            return False, ""

        # Detect anomaly
        is_anomaly, anomaly_score = detector.detect_anomaly(current_value, datetime.now())

        if is_anomaly:
            message = f"{rule.name}: Anomaly detected in {rule.metric_name} = {current_value:.2f} (score: {anomaly_score:.3f})"
            return True, message

        return False, ""

    def _get_anomaly_detector(self, metric_name: str) -> AnomalyDetector:
        """Get or create anomaly detector for metric."""
        if metric_name not in self.anomaly_detectors:
            self.anomaly_detectors[metric_name] = AnomalyDetector()

        return self.anomaly_detectors[metric_name]

    async def _train_anomaly_detectors(self):
        """Train all anomaly detectors."""
        for metric_name, detector in self.anomaly_detectors.items():
            try:
                success = detector.train()
                if success:
                    logger.info(f"Trained anomaly detector for {metric_name}")
            except Exception as e:
                logger.error(f"Error training anomaly detector for {metric_name}: {e}")

    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current metric value."""
        try:
            # Map metric names to actual data sources
            if metric_name == "cpu_usage":
                import psutil

                return psutil.cpu_percent(interval=0.1)
            elif metric_name == "memory_usage":
                import psutil

                return psutil.virtual_memory().percent
            elif metric_name == "active_agents":
                return len(performance_tracker.agent_metrics)
            elif metric_name == "api_response_time":
                # Get from performance tracker
                return performance_tracker.get_average_inference_time()
            elif metric_name == "user_interaction_rate":
                # This would come from business metrics
                return 0.05  # Placeholder
            else:
                logger.warning(f"Unknown metric: {metric_name}")
                return None
        except Exception as e:
            logger.error(f"Error getting metric value for {metric_name}: {e}")
            return None

    def _find_existing_alert(self, rule_id: str, metric_name: str) -> Optional[Alert]:
        """Find existing alert for rule and metric."""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.metric_name == metric_name:
                return alert
        return None

    async def _trigger_alert(self, alert: Alert):
        """Trigger a new alert."""
        # Check suppression rules
        if self._is_suppressed(alert):
            self.stats["suppressed_alerts"] += 1
            return

        # Check correlations
        correlations = self.correlation_engine.analyze_correlations(alert)
        if correlations:
            alert.annotations["correlations"] = json.dumps(correlations)

        # Add to active alerts
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        self.correlation_engine.alert_history.append(alert)

        # Update statistics
        self.stats["total_alerts"] += 1
        self.stats["alerts_by_severity"][alert.severity.value] += 1
        self.stats["alerts_by_type"][alert.alert_type.value] += 1

        # Log alert
        logger.warning(f"🚨 Alert triggered: {alert.message}")

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def _resolve_alert(self, alert: Alert):
        """Resolve an active alert."""
        alert.status = AlertStatus.RESOLVED
        alert.ends_at = datetime.now()

        # Remove from active alerts
        if alert.id in self.active_alerts:
            del self.active_alerts[alert.id]

        logger.info(f"✅ Alert resolved: {alert.message}")

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        for rule in self.suppression_rules:
            if self._matches_suppression_rule(alert, rule):
                return True
        return False

    def _matches_suppression_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches suppression rule."""
        # Implementation depends on suppression rule format
        return False

    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_id: str):
        """Remove alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)

    def add_suppression_rule(self, rule: Dict[str, Any]):
        """Add suppression rule."""
        self.suppression_rules.append(rule)

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by

            logger.info(f"Alert acknowledged: {alert.message} by {acknowledged_by}")

    def suppress_alert(self, alert_id: str, duration: int):
        """Suppress an alert for a duration."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = datetime.now() + timedelta(seconds=duration)

            logger.info(f"Alert suppressed: {alert.message} for {duration} seconds")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        return {
            **self.stats,
            "active_alerts": len(self.active_alerts),
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "anomaly_detectors": len(self.anomaly_detectors),
            "trained_detectors": len([d for d in self.anomaly_detectors.values() if d.trained]),
        }


# Global intelligent alerting system instance
intelligent_alerting = IntelligentAlertingSystem()


async def start_intelligent_alerting():
    """Start the intelligent alerting system."""
    await intelligent_alerting.start()


async def stop_intelligent_alerting():
    """Stop the intelligent alerting system."""
    await intelligent_alerting.stop()


# Integration with existing systems
async def integrate_with_prometheus_alertmanager():
    """Integrate with Prometheus AlertManager."""

    async def prometheus_alert_callback(alert: Alert):
        """Callback to send alerts to Prometheus AlertManager."""
        # This would send alerts to AlertManager API
        alert_data = {
            "labels": {
                "alertname": alert.rule_name,
                "severity": alert.severity.value,
                "instance": alert.labels.get("instance", "freeagentics"),
                "job": "freeagentics-intelligent-alerting",
                **alert.labels,
            },
            "annotations": {
                "summary": alert.message,
                "description": alert.description,
                **alert.annotations,
            },
            "startsAt": alert.starts_at.isoformat(),
            "endsAt": alert.ends_at.isoformat() if alert.ends_at else None,
            "generatorURL": f"https://freeagentics.com/alerts/{alert.id}",
        }

        logger.info(f"Sending alert to Prometheus AlertManager: {json.dumps(alert_data, indent=2)}")

        # In a real implementation, this would use HTTP requests to AlertManager API
        # requests.post('http://alertmanager:9093/api/v1/alerts', json=[alert_data])

    intelligent_alerting.add_alert_callback(prometheus_alert_callback)

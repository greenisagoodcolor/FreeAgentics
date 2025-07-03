"""
Module for FreeAgentics Active Inference implementation.
"""

import smtplib
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

from .monitoring import get_logger

"""
Alerting System for GNN Processing
This module provides alerting mechanisms for monitoring thresholds,
error conditions, and performance degradation.
"""
logger = get_logger().logger


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""

    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    SYSTEM_HEALTH = "system_health"
    QUEUE_BACKUP = "queue_backup"
    MODEL_FAILURE = "model_failure"


@dataclass
class Alert:
    """Alert data structure"""

    id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: AlertType
    title: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["severity"] = self.severity.value
        data["alert_type"] = self.alert_type.value
        data["timestamp"] = self.timestamp.isoformat()
        if self.resolved_at:
            data["resolved_at"] = self.resolved_at.isoformat()
        return data


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    alert_type: AlertType
    metric: str
    threshold: float
    comparison: str
    severity: AlertSeverity
    cooldown_minutes: int = 5
    consecutive_breaches: int = 1
    enabled: bool = True

    def check_condition(self, value: float) -> bool:
        """Check if condition is met"""
        if self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "eq":
            return value == self.threshold
        elif self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        else:
            return False


class AlertChannel:
    """Base class for alert channels"""

    def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        raise NotImplementedError


class LoggerChannel(AlertChannel):
    """Logger-based alert channel"""

    def send_alert(self, alert: Alert) -> bool:
        """Log the alert"""
        try:
            log_message = f"ALERT [{
                alert.severity.value.upper()}] - {
                alert.title}: {
                alert.message}"
            if alert.severity == AlertSeverity.CRITICAL:
                logger.critical(log_message)
            elif alert.severity == AlertSeverity.ERROR:
                logger.error(log_message)
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(log_message)
            else:
                logger.info(log_message)
            return True
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
            return False


class EmailChannel(AlertChannel):
    """Email-based alert channel"""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
        use_tls: bool = True,
    ) -> None:
        """
        Initialize email channel.
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: Sender email
            to_emails: List of recipient emails
            use_tls: Use TLS encryption
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
            body = f"\nAlert Details:\n--------------\nSeverity: {
                alert.severity.value}\nType: {
                alert.alert_type.value}\nTime: {
                alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\nMessage:\n{
                alert.message}\n\n"
            if alert.metric_value is not None:
                body += f"Metric Value: {alert.metric_value}\n"
            if alert.threshold is not None:
                body += f"Threshold: {alert.threshold}\n"
            if alert.context:
                body += "\nAdditional Context:\n"
                for key, value in alert.context.items():
                    body += f"  {key}: {value}\n"
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookChannel(AlertChannel):
    """Webhook-based alert channel"""

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize webhook channel.
        Args:
            webhook_url: Webhook URL
            headers: Optional headers
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        try:
            payload = alert.to_dict()
            response = requests.post(
                self.webhook_url, json=payload, headers=self.headers, timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class AlertManager:
    """
    Manages alerts, rules, and notification channels.
    Features:
    - Rule-based alerting
    - Multiple notification channels
    - Alert deduplication
    - Alert history
    - Auto-resolution
    """

    def __init__(self, max_history: int = 1000, check_interval: int = 30) -> None:
        """
        Initialize alert manager.
        Args:
            max_history: Maximum alerts to keep in history
            check_interval: Interval to check rules in seconds
        """
        self.max_history = max_history
        self.check_interval = check_interval
        self.alerts: deque = deque(maxlen=max_history)
        self.active_alerts: Dict[str, Alert] = {}
        self.rules: List[AlertRule] = []
        self.channels: List[AlertChannel] = []
        self.rule_state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._check_thread = None
        self._setup_default_rules()
        self.add_channel(LoggerChannel())

    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                alert_type=AlertType.ERROR_RATE,
                metric="error_rate",
                threshold=0.1,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                consecutive_breaches=3,
            ),
            AlertRule(
                name="critical_error_rate",
                alert_type=AlertType.ERROR_RATE,
                metric="error_rate",
                threshold=0.25,
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                consecutive_breaches=1,
            ),
            AlertRule(
                name="high_processing_time",
                alert_type=AlertType.PERFORMANCE,
                metric="avg_processing_time",
                threshold=60,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                consecutive_breaches=5,
            ),
            AlertRule(
                name="high_memory_usage",
                alert_type=AlertType.RESOURCE_USAGE,
                metric="memory_usage_mb",
                threshold=8192,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                consecutive_breaches=3,
            ),
            AlertRule(
                name="critical_memory_usage",
                alert_type=AlertType.RESOURCE_USAGE,
                metric="memory_usage_mb",
                threshold=15360,
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                consecutive_breaches=1,
            ),
            AlertRule(
                name="high_cpu_usage",
                alert_type=AlertType.RESOURCE_USAGE,
                metric="cpu_usage_percent",
                threshold=90,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                consecutive_breaches=5,
            ),
            AlertRule(
                name="queue_backup",
                alert_type=AlertType.QUEUE_BACKUP,
                metric="queue_size",
                threshold=100,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                consecutive_breaches=1,
            ),
        ]
        for rule in default_rules:
            self.add_rule(rule)

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule"""
        with self._lock:
            self.rules.append(rule)
            self.rule_state[rule.name] = {"breach_count": 0, "last_alert_time": None}

    def add_channel(self, channel: AlertChannel) -> None:
        """Add notification channel"""
        with self._lock:
            self.channels.append(channel)

    def create_alert(
        self,
        severity: AlertSeverity,
        alert_type: AlertType,
        title: str,
        message: str,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create and send an alert"""
        alert = Alert(
            id=f"alert_{int(time.time() * 1000)}",
            timestamp=datetime.utcnow(),
            severity=severity,
            alert_type=alert_type,
            title=title,
            message=message,
            metric_value=metric_value,
            threshold=threshold,
            context=context or {},
        )
        with self._lock:
            self.alerts.append(alert)
            alert_key = f"{alert_type.value}:{title}"
            self.active_alerts[alert_key] = alert
            self._send_alert(alert)
        return alert

    def _send_alert(self, alert: Alert):
        """Send alert through all channels"""
        for channel in self.channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                logger.error(f"Error sending alert through channel: {e}")

    def check_metrics(self, metrics: Dict[str, float]):
        """Check metrics against rules"""
        with self._lock:
            for rule in self.rules:
                if not rule.enabled:
                    continue
                metric_value = metrics.get(rule.metric)
                if metric_value is None:
                    continue
                if rule.check_condition(metric_value):
                    state = self.rule_state[rule.name]
                    state["breach_count"] += 1
                    if state["breach_count"] >= rule.consecutive_breaches:
                        if state["last_alert_time"]:
                            cooldown_end = state["last_alert_time"] + timedelta(
                                minutes=rule.cooldown_minutes
                            )
                            if datetime.utcnow() < cooldown_end:
                                continue
                        self.create_alert(
                            severity=rule.severity,
                            alert_type=rule.alert_type,
                            title=f"{
                                rule.name} threshold breached",
                            message=f"Metric '{
                                rule.metric}' value {
                                metric_value:.2f} {
                                rule.comparison} threshold {
                                rule.threshold}",
                            metric_value=metric_value,
                            threshold=rule.threshold,
                            context={"rule_name": rule.name},
                        )
                        state["last_alert_time"] = datetime.utcnow()
                        state["breach_count"] = 0
                else:
                    self.rule_state[rule.name]["breach_count"] = 0

    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id and (not alert.resolved):
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    alert_key = f"{alert.alert_type.value}:{alert.title}"
                    if alert_key in self.active_alerts:
                        del self.active_alerts[alert_key]
                    resolution_alert = Alert(
                        id=f"resolution_{alert_id}",
                        timestamp=datetime.utcnow(),
                        severity=AlertSeverity.INFO,
                        alert_type=alert.alert_type,
                        title=f"Alert Resolved: {
                            alert.title}",
                        message=f"The alert '{
                            alert.title}' has been resolved.",
                        context={"original_alert_id": alert_id},
                    )
                    self._send_alert(resolution_alert)
                    break

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(
        self,
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
    ) -> List[Alert]:
        """Get alert history with filters"""
        with self._lock:
            alerts = list(self.alerts)
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            if alert_type:
                alerts = [a for a in alerts if a.alert_type == alert_type]
            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            return alerts[:limit]

    def start(self):
        """Start alert manager"""
        self._running = True
        self._check_thread = threading.Thread(target=self._check_loop)
        self._check_thread.daemon = True
        self._check_thread.start()
        logger.info("Alert manager started")

    def _check_loop(self):
        """Background thread to check rules"""
        while self._running:
            try:
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in alert check loop: {e}")
                time.sleep(self.check_interval)

    def stop(self):
        """Stop alert manager"""
        self._running = False
        if self._check_thread and self._check_thread.is_alive():
            self._check_thread.join(timeout=5)
        logger.info("Alert manager stopped")


_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get or create alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


if __name__ == "__main__":
    manager = get_alert_manager()
    manager.start()
    alert = manager.create_alert(
        severity=AlertSeverity.WARNING,
        alert_type=AlertType.PERFORMANCE,
        title="High Processing Time",
        message="Average processing time exceeded threshold",
        metric_value=75.5,
        threshold=60.0,
    )
    print(f"Alert created: {alert.id}")
    test_metrics = {
        "error_rate": 0.15,
        "avg_processing_time": 45.0,
        "memory_usage_mb": 4096,
        "cpu_usage_percent": 75,
    }
    manager.check_metrics(test_metrics)
    active_alerts = manager.get_active_alerts()
    print(f"Active alerts: {len(active_alerts)}")
    manager.stop()

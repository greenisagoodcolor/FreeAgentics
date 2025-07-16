"""
Advanced Security Monitoring System for FreeAgentics.

This module provides comprehensive security event monitoring, threat detection,
and incident response capabilities for production environments.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import httpx
from prometheus_client import Counter, Gauge, Histogram

from auth.security_logging import (
    SecurityEventSeverity,
    SecurityEventType,
    security_auditor,
)
from observability.prometheus_metrics import (
    record_security_anomaly,
    security_anomaly_detections_total,
)

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(str, Enum):
    """Types of detected attacks."""

    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MALWARE = "malware"
    DATA_EXFILTRATION = "data_exfiltration"
    INSIDER_THREAT = "insider_threat"


@dataclass
class SecurityAlert:
    """Security alert data structure."""

    id: str
    timestamp: datetime
    alert_type: AttackType
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    evidence: Dict[str, Any]
    status: str = "active"
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class ThreatIndicator:
    """Threat indicator for pattern matching."""

    indicator_type: str
    pattern: str
    description: str
    severity: ThreatLevel
    confidence: float
    last_seen: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecurityMetrics:
    """Security metrics snapshot."""

    total_events: int
    alerts_generated: int
    threats_detected: int
    false_positives: int
    mean_detection_time: float
    mean_response_time: float
    top_attack_types: Dict[str, int]
    top_source_ips: Dict[str, int]
    threat_level_distribution: Dict[str, int]


class SecurityMonitoringSystem:
    """Advanced security monitoring and threat detection system."""

    def __init__(self):
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.threat_indicators: List[ThreatIndicator] = []
        self.security_events: deque = deque(maxlen=10000)
        self.blocked_ips: Set[str] = set()
        self.suspicious_users: Set[str] = set()

        # Threat detection state
        self.ip_activity: Dict[str, List[datetime]] = defaultdict(list)
        self.user_activity: Dict[str, List[datetime]] = defaultdict(list)
        self.failed_logins: Dict[str, List[datetime]] = defaultdict(list)
        self.api_requests: Dict[str, List[datetime]] = defaultdict(list)

        # Configuration
        self.brute_force_threshold = 5  # attempts per 15 minutes
        self.ddos_threshold = 1000  # requests per minute
        self.anomaly_threshold = 3.0  # standard deviations
        self.max_failed_logins = 10

        # Metrics
        self.security_events_total = Counter(
            "security_events_total", "Total security events processed", ["event_type", "severity"]
        )
        self.threats_detected_total = Counter(
            "threats_detected_total", "Total threats detected", ["threat_type", "severity"]
        )
        self.security_alerts_active = Gauge(
            "security_alerts_active", "Number of active security alerts"
        )
        self.threat_detection_time = Histogram(
            "threat_detection_time_seconds", "Time to detect threats", ["threat_type"]
        )

        # Initialize threat indicators
        self._initialize_threat_indicators()

        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("🛡️ Security monitoring system initialized")

    def _initialize_threat_indicators(self):
        """Initialize built-in threat indicators."""
        self.threat_indicators = [
            # SQL Injection patterns
            ThreatIndicator(
                indicator_type="sql_injection",
                pattern=r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|script)",
                description="SQL injection attempt detected",
                severity=ThreatLevel.HIGH,
                confidence=0.8,
            ),
            # XSS patterns
            ThreatIndicator(
                indicator_type="xss",
                pattern=r"(?i)(<script|javascript:|vbscript:|onload=|onerror=|onclick=)",
                description="Cross-site scripting attempt detected",
                severity=ThreatLevel.HIGH,
                confidence=0.7,
            ),
            # Directory traversal
            ThreatIndicator(
                indicator_type="directory_traversal",
                pattern=r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
                description="Directory traversal attempt detected",
                severity=ThreatLevel.MEDIUM,
                confidence=0.9,
            ),
            # Command injection
            ThreatIndicator(
                indicator_type="command_injection",
                pattern=r"(?i)(;|&&|\|\||`|\$\(|nc\s|netcat|wget|curl|powershell|cmd\.exe)",
                description="Command injection attempt detected",
                severity=ThreatLevel.CRITICAL,
                confidence=0.8,
            ),
            # Suspicious user agents
            ThreatIndicator(
                indicator_type="suspicious_user_agent",
                pattern=r"(?i)(sqlmap|nmap|nikto|burp|owasp|zap|w3af|metasploit)",
                description="Suspicious user agent detected",
                severity=ThreatLevel.MEDIUM,
                confidence=0.6,
            ),
        ]

    async def start_monitoring(self):
        """Start security monitoring tasks."""
        if self.running:
            return

        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("🔍 Security monitoring started")

    async def stop_monitoring(self):
        """Stop security monitoring tasks."""
        self.running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Security monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._analyze_security_events()
                await self._detect_threats()
                await self._update_metrics()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in security monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _cleanup_loop(self):
        """Cleanup old data and resolved alerts."""
        while self.running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

    async def process_security_event(self, event: Dict[str, Any]):
        """Process a security event and detect threats."""
        try:
            # Add to event queue
            self.security_events.append(event)

            # Update metrics
            event_type = event.get("event_type", "unknown")
            severity = event.get("severity", "info")
            self.security_events_total.labels(event_type=event_type, severity=severity).inc()

            # Extract relevant information
            source_ip = event.get("ip_address", "unknown")
            user_id = event.get("user_id")
            timestamp = datetime.fromisoformat(
                event.get("timestamp", datetime.utcnow().isoformat())
            )

            # Track activity patterns
            self.ip_activity[source_ip].append(timestamp)
            if user_id:
                self.user_activity[user_id].append(timestamp)

            # Check for immediate threats
            await self._check_for_threats(event)

        except Exception as e:
            logger.error(f"Error processing security event: {e}")

    async def _check_for_threats(self, event: Dict[str, Any]):
        """Check event for immediate threat indicators."""
        source_ip = event.get("ip_address", "unknown")
        user_id = event.get("user_id")
        event_type = event.get("event_type")
        message = event.get("message", "")
        user_agent = event.get("user_agent", "")
        endpoint = event.get("endpoint", "")

        # Check for brute force attacks
        if event_type == SecurityEventType.LOGIN_FAILURE:
            await self._check_brute_force(source_ip, user_id, event)

        # Check for DDoS attacks
        if event_type == SecurityEventType.API_ACCESS:
            await self._check_ddos(source_ip, event)

        # Check for injection attacks
        for indicator in self.threat_indicators:
            if indicator.indicator_type in ["sql_injection", "xss", "command_injection"]:
                if await self._match_pattern(indicator, message + " " + endpoint):
                    await self._generate_alert(
                        (
                            AttackType.SQL_INJECTION
                            if "sql" in indicator.indicator_type
                            else AttackType.XSS
                        ),
                        indicator.severity,
                        source_ip,
                        user_id,
                        f"{indicator.description}: {message}",
                        event,
                    )

        # Check for suspicious user agents
        for indicator in self.threat_indicators:
            if indicator.indicator_type == "suspicious_user_agent":
                if await self._match_pattern(indicator, user_agent):
                    await self._generate_alert(
                        AttackType.SUSPICIOUS_ACTIVITY,
                        indicator.severity,
                        source_ip,
                        user_id,
                        f"Suspicious user agent: {user_agent}",
                        event,
                    )

    async def _check_brute_force(self, source_ip: str, user_id: str, event: Dict[str, Any]):
        """Check for brute force attacks."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=15)

        # Clean old attempts
        key = f"{source_ip}:{user_id}" if user_id else source_ip
        self.failed_logins[key] = [
            attempt for attempt in self.failed_logins[key] if attempt > cutoff
        ]

        # Add current attempt
        self.failed_logins[key].append(now)

        # Check threshold
        if len(self.failed_logins[key]) >= self.brute_force_threshold:
            await self._generate_alert(
                AttackType.BRUTE_FORCE,
                ThreatLevel.HIGH,
                source_ip,
                user_id,
                f"Brute force attack detected from {source_ip}",
                event,
            )

            # Block IP
            self.blocked_ips.add(source_ip)
            if user_id:
                self.suspicious_users.add(user_id)

    async def _check_ddos(self, source_ip: str, event: Dict[str, Any]):
        """Check for DDoS attacks."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)

        # Clean old requests
        self.api_requests[source_ip] = [
            request for request in self.api_requests[source_ip] if request > cutoff
        ]

        # Add current request
        self.api_requests[source_ip].append(now)

        # Check threshold
        if len(self.api_requests[source_ip]) >= self.ddos_threshold:
            await self._generate_alert(
                AttackType.DDoS,
                ThreatLevel.CRITICAL,
                source_ip,
                None,
                f"DDoS attack detected from {source_ip}",
                event,
            )

            # Block IP
            self.blocked_ips.add(source_ip)

    async def _match_pattern(self, indicator: ThreatIndicator, text: str) -> bool:
        """Check if text matches threat indicator pattern."""
        import re

        try:
            return bool(re.search(indicator.pattern, text))
        except Exception:
            return False

    async def _generate_alert(
        self,
        attack_type: AttackType,
        threat_level: ThreatLevel,
        source_ip: str,
        user_id: Optional[str],
        description: str,
        evidence: Dict[str, Any],
    ):
        """Generate a security alert."""
        alert_id = f"{attack_type}_{source_ip}_{int(time.time())}"

        alert = SecurityAlert(
            id=alert_id,
            timestamp=datetime.utcnow(),
            alert_type=attack_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            evidence=evidence,
        )

        self.active_alerts[alert_id] = alert

        # Update metrics
        self.threats_detected_total.labels(threat_type=attack_type, severity=threat_level).inc()

        self.security_alerts_active.inc()

        # Record in Prometheus
        record_security_anomaly(attack_type, threat_level)

        # Log security event
        security_auditor.log_event(
            SecurityEventType.SUSPICIOUS_PATTERN,
            (
                SecurityEventSeverity.CRITICAL
                if threat_level == ThreatLevel.CRITICAL
                else SecurityEventSeverity.WARNING
            ),
            description,
            details={
                "alert_id": alert_id,
                "attack_type": attack_type,
                "threat_level": threat_level,
                "source_ip": source_ip,
                "user_id": user_id,
                "evidence": evidence,
            },
        )

        # Send alert notification
        await self._send_alert_notification(alert)

        logger.warning(f"🚨 Security alert generated: {alert_id} - {description}")

    async def _send_alert_notification(self, alert: SecurityAlert):
        """Send alert notification to security team."""
        try:
            # This would integrate with your notification system
            # For now, we'll just log it
            logger.critical(f"SECURITY ALERT: {alert.description}")

            # TODO: Implement actual notification (email, Slack, PagerDuty, etc.)
            # Example webhook notification:
            # await self._send_webhook_notification(alert)

        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")

    async def _analyze_security_events(self):
        """Analyze security events for patterns and anomalies."""
        if not self.security_events:
            return

        # Analyze recent events (last 100)
        recent_events = list(self.security_events)[-100:]

        # Look for patterns
        event_types = defaultdict(int)
        source_ips = defaultdict(int)
        error_patterns = defaultdict(int)

        for event in recent_events:
            event_types[event.get("event_type", "unknown")] += 1
            source_ips[event.get("ip_address", "unknown")] += 1

            if event.get("severity") in ["error", "critical"]:
                error_patterns[event.get("message", "")[:50]] += 1

        # Detect anomalies
        await self._detect_anomalies(event_types, source_ips, error_patterns)

    async def _detect_anomalies(self, event_types, source_ips, error_patterns):
        """Detect anomalies in security event patterns."""
        # High volume of failed logins
        if event_types.get(SecurityEventType.LOGIN_FAILURE, 0) > 50:
            await self._generate_alert(
                AttackType.BRUTE_FORCE,
                ThreatLevel.HIGH,
                "multiple",
                None,
                "High volume of failed login attempts detected",
                {"failed_login_count": event_types[SecurityEventType.LOGIN_FAILURE]},
            )

        # Unusual number of errors from single IP
        for ip, count in source_ips.items():
            if count > 20:
                await self._generate_alert(
                    AttackType.SUSPICIOUS_ACTIVITY,
                    ThreatLevel.MEDIUM,
                    ip,
                    None,
                    f"Unusual activity from IP {ip}",
                    {"request_count": count},
                )

    async def _detect_threats(self):
        """Run threat detection algorithms."""
        # Update threat indicators
        await self._update_threat_indicators()

        # Check for behavioral anomalies
        await self._check_behavioral_anomalies()

        # Check for privilege escalation
        await self._check_privilege_escalation()

    async def _update_threat_indicators(self):
        """Update threat indicators based on recent intelligence."""
        # TODO: Integrate with threat intelligence feeds
        # For now, we'll just update the last_seen timestamps
        for indicator in self.threat_indicators:
            indicator.last_seen = datetime.utcnow()

    async def _check_behavioral_anomalies(self):
        """Check for behavioral anomalies in user activity."""
        # Analyze user activity patterns
        for user_id, activities in self.user_activity.items():
            if len(activities) > 100:  # High activity user
                # Check for unusual patterns
                recent_activities = [
                    a for a in activities if a > datetime.utcnow() - timedelta(hours=1)
                ]
                if len(recent_activities) > 50:
                    await self._generate_alert(
                        AttackType.SUSPICIOUS_ACTIVITY,
                        ThreatLevel.MEDIUM,
                        "unknown",
                        user_id,
                        f"Unusual high activity from user {user_id}",
                        {"activity_count": len(recent_activities)},
                    )

    async def _check_privilege_escalation(self):
        """Check for privilege escalation attempts."""
        # This would integrate with your authorization system
        # For now, it's a placeholder
        pass

    async def _update_metrics(self):
        """Update security metrics."""
        self.security_alerts_active.set(len(self.active_alerts))

    async def _cleanup_old_data(self):
        """Clean up old data and resolved alerts."""
        cutoff = datetime.utcnow() - timedelta(days=7)

        # Clean up old IP activity
        for ip in list(self.ip_activity.keys()):
            self.ip_activity[ip] = [
                activity for activity in self.ip_activity[ip] if activity > cutoff
            ]
            if not self.ip_activity[ip]:
                del self.ip_activity[ip]

        # Clean up old user activity
        for user_id in list(self.user_activity.keys()):
            self.user_activity[user_id] = [
                activity for activity in self.user_activity[user_id] if activity > cutoff
            ]
            if not self.user_activity[user_id]:
                del self.user_activity[user_id]

        # Clean up resolved alerts older than 24 hours
        alert_cutoff = datetime.utcnow() - timedelta(hours=24)
        for alert_id in list(self.active_alerts.keys()):
            alert = self.active_alerts[alert_id]
            if alert.resolved_at and alert.resolved_at < alert_cutoff:
                del self.active_alerts[alert_id]

    def get_security_metrics(self) -> SecurityMetrics:
        """Get current security metrics."""
        # Calculate metrics
        total_events = len(self.security_events)
        alerts_generated = len(self.active_alerts)

        # Top attack types
        top_attack_types = defaultdict(int)
        for alert in self.active_alerts.values():
            top_attack_types[alert.alert_type] += 1

        # Top source IPs
        top_source_ips = defaultdict(int)
        for alert in self.active_alerts.values():
            top_source_ips[alert.source_ip] += 1

        # Threat level distribution
        threat_level_distribution = defaultdict(int)
        for alert in self.active_alerts.values():
            threat_level_distribution[alert.threat_level] += 1

        return SecurityMetrics(
            total_events=total_events,
            alerts_generated=alerts_generated,
            threats_detected=len([a for a in self.active_alerts.values() if a.status == "active"]),
            false_positives=len(
                [a for a in self.active_alerts.values() if a.status == "false_positive"]
            ),
            mean_detection_time=0.0,  # TODO: Calculate actual metrics
            mean_response_time=0.0,  # TODO: Calculate actual metrics
            top_attack_types=dict(top_attack_types),
            top_source_ips=dict(top_source_ips),
            threat_level_distribution=dict(threat_level_distribution),
        )

    def get_active_alerts(self) -> List[SecurityAlert]:
        """Get all active security alerts."""
        return [alert for alert in self.active_alerts.values() if alert.status == "active"]

    def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """Resolve a security alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = "resolved"
            alert.resolved_at = datetime.utcnow()
            alert.resolution_notes = resolution_notes

            self.security_alerts_active.dec()

            logger.info(f"🔍 Security alert resolved: {alert_id}")
            return True
        return False

    def mark_false_positive(self, alert_id: str, notes: str = "") -> bool:
        """Mark alert as false positive."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = "false_positive"
            alert.resolved_at = datetime.utcnow()
            alert.resolution_notes = notes

            self.security_alerts_active.dec()

            logger.info(f"🔍 Security alert marked as false positive: {alert_id}")
            return True
        return False

    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips

    def is_user_suspicious(self, user_id: str) -> bool:
        """Check if user is marked as suspicious."""
        return user_id in self.suspicious_users

    def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address."""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"🔓 IP unblocked: {ip}")
            return True
        return False

    def clear_user_suspicion(self, user_id: str) -> bool:
        """Clear user suspicion."""
        if user_id in self.suspicious_users:
            self.suspicious_users.remove(user_id)
            logger.info(f"🔓 User suspicion cleared: {user_id}")
            return True
        return False


# Global security monitoring instance
security_monitor = SecurityMonitoringSystem()


async def start_security_monitoring():
    """Start the security monitoring system."""
    await security_monitor.start_monitoring()


async def stop_security_monitoring():
    """Stop the security monitoring system."""
    await security_monitor.stop_monitoring()


def get_security_metrics() -> SecurityMetrics:
    """Get current security metrics."""
    return security_monitor.get_security_metrics()


def get_active_alerts() -> List[SecurityAlert]:
    """Get active security alerts."""
    return security_monitor.get_active_alerts()

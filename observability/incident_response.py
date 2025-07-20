"""
Automated Incident Response System for FreeAgentics.

This module provides automated incident response capabilities including
incident detection, classification, response automation, and escalation.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import httpx

from auth.security_logging import (
    SecurityEventSeverity,
    SecurityEventType,
    security_auditor,
)
from observability.security_monitoring import (
    AttackType,
    SecurityAlert,
    ThreatLevel,
)

logger = logging.getLogger(__name__)


class IncidentSeverity(str, Enum):
    """Incident severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    """Incident status levels."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINMENT = "containment"
    ERADICATION = "eradication"
    RECOVERY = "recovery"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ResponseAction(str, Enum):
    """Types of automated response actions."""

    BLOCK_IP = "block_ip"
    SUSPEND_USER = "suspend_user"
    RATE_LIMIT = "rate_limit"
    ALERT_TEAM = "alert_team"
    QUARANTINE_HOST = "quarantine_host"
    DISABLE_ENDPOINT = "disable_endpoint"
    COLLECT_EVIDENCE = "collect_evidence"
    NOTIFY_MANAGEMENT = "notify_management"
    ESCALATE = "escalate"
    LOG_ANALYSIS = "log_analysis"


@dataclass
class IncidentResponse:
    """Incident response data structure."""

    id: str
    incident_id: str
    action: ResponseAction
    status: str
    timestamp: datetime
    details: Dict[str, Any]
    success: bool = False
    error_message: Optional[str] = None
    execution_time: float = 0.0
    automated: bool = True


@dataclass
class SecurityIncident:
    """Security incident data structure."""

    id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    attack_type: AttackType
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    affected_users: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    responses: List[IncidentResponse] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    escalation_level: int = 0
    false_positive: bool = False
    lesson_learned: Optional[str] = None


@dataclass
class ResponsePlaybook:
    """Response playbook for specific attack types."""

    name: str
    attack_types: List[AttackType]
    severity_threshold: IncidentSeverity
    automated_actions: List[ResponseAction]
    manual_actions: List[str]
    escalation_timeout: int  # minutes
    description: str
    success_criteria: List[str]


class IncidentResponseSystem:
    """Automated incident response and management system."""

    def __init__(self):
        self.incidents: Dict[str, SecurityIncident] = {}
        self.response_history: List[IncidentResponse] = []
        self.blocked_ips: Set[str] = set()
        self.suspended_users: Set[str] = set()
        self.quarantined_hosts: Set[str] = set()
        self.disabled_endpoints: Set[str] = set()

        # Response playbooks
        self.playbooks: Dict[str, ResponsePlaybook] = {}
        self._initialize_playbooks()

        # Configuration
        self.auto_response_enabled = True
        self.escalation_timeout = 30  # minutes
        self.notification_channels = {"email": [], "slack": [], "webhook": []}

        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.escalation_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("🚨 Incident response system initialized")

    def _initialize_playbooks(self):
        """Initialize response playbooks."""
        self.playbooks = {
            "brute_force": ResponsePlaybook(
                name="Brute Force Attack Response",
                attack_types=[AttackType.BRUTE_FORCE],
                severity_threshold=IncidentSeverity.HIGH,
                automated_actions=[
                    ResponseAction.BLOCK_IP,
                    ResponseAction.COLLECT_EVIDENCE,
                    ResponseAction.ALERT_TEAM,
                    ResponseAction.LOG_ANALYSIS,
                ],
                manual_actions=[
                    "Review authentication logs",
                    "Check for successful logins from same IP",
                    "Verify user account security",
                    "Update password policies if needed",
                ],
                escalation_timeout=15,
                description="Automated response to brute force attacks",
                success_criteria=[
                    "Attacker IP blocked",
                    "No successful logins from attacker",
                    "Evidence collected and preserved",
                ],
            ),
            "ddos": ResponsePlaybook(
                name="DDoS Attack Response",
                attack_types=[AttackType.DDoS],
                severity_threshold=IncidentSeverity.CRITICAL,
                automated_actions=[
                    ResponseAction.BLOCK_IP,
                    ResponseAction.RATE_LIMIT,
                    ResponseAction.ALERT_TEAM,
                    ResponseAction.NOTIFY_MANAGEMENT,
                    ResponseAction.COLLECT_EVIDENCE,
                ],
                manual_actions=[
                    "Activate DDoS mitigation service",
                    "Scale up infrastructure if needed",
                    "Contact ISP for additional protection",
                    "Monitor application performance",
                ],
                escalation_timeout=5,
                description="Automated response to DDoS attacks",
                success_criteria=[
                    "Attack traffic blocked",
                    "Service availability maintained",
                    "Performance metrics stable",
                ],
            ),
            "sql_injection": ResponsePlaybook(
                name="SQL Injection Response",
                attack_types=[AttackType.SQL_INJECTION],
                severity_threshold=IncidentSeverity.HIGH,
                automated_actions=[
                    ResponseAction.BLOCK_IP,
                    ResponseAction.DISABLE_ENDPOINT,
                    ResponseAction.COLLECT_EVIDENCE,
                    ResponseAction.ALERT_TEAM,
                    ResponseAction.NOTIFY_MANAGEMENT,
                ],
                manual_actions=[
                    "Review application code for vulnerabilities",
                    "Check database for unauthorized access",
                    "Validate input sanitization",
                    "Update WAF rules",
                    "Perform security code review",
                ],
                escalation_timeout=10,
                description="Automated response to SQL injection attacks",
                success_criteria=[
                    "Vulnerable endpoint secured",
                    "No database compromise detected",
                    "Attack vector eliminated",
                ],
            ),
            "privilege_escalation": ResponsePlaybook(
                name="Privilege Escalation Response",
                attack_types=[AttackType.PRIVILEGE_ESCALATION],
                severity_threshold=IncidentSeverity.CRITICAL,
                automated_actions=[
                    ResponseAction.SUSPEND_USER,
                    ResponseAction.BLOCK_IP,
                    ResponseAction.COLLECT_EVIDENCE,
                    ResponseAction.ALERT_TEAM,
                    ResponseAction.NOTIFY_MANAGEMENT,
                    ResponseAction.ESCALATE,
                ],
                manual_actions=[
                    "Review user permissions and roles",
                    "Check for unauthorized access",
                    "Audit system configuration",
                    "Validate authorization controls",
                    "Investigate potential insider threat",
                ],
                escalation_timeout=5,
                description="Automated response to privilege escalation attempts",
                success_criteria=[
                    "User account secured",
                    "No unauthorized access confirmed",
                    "System integrity maintained",
                ],
            ),
            "data_exfiltration": ResponsePlaybook(
                name="Data Exfiltration Response",
                attack_types=[AttackType.DATA_EXFILTRATION],
                severity_threshold=IncidentSeverity.CRITICAL,
                automated_actions=[
                    ResponseAction.BLOCK_IP,
                    ResponseAction.SUSPEND_USER,
                    ResponseAction.QUARANTINE_HOST,
                    ResponseAction.COLLECT_EVIDENCE,
                    ResponseAction.ALERT_TEAM,
                    ResponseAction.NOTIFY_MANAGEMENT,
                    ResponseAction.ESCALATE,
                ],
                manual_actions=[
                    "Identify compromised data",
                    "Assess breach impact",
                    "Activate data breach response plan",
                    "Notify legal and compliance teams",
                    "Prepare breach notifications",
                ],
                escalation_timeout=5,
                description="Automated response to data exfiltration attempts",
                success_criteria=[
                    "Data transfer stopped",
                    "Breach contained",
                    "Legal requirements met",
                ],
            ),
        }

    async def start_monitoring(self):
        """Start incident response monitoring."""
        if self.running:
            return

        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.escalation_task = asyncio.create_task(self._escalation_loop())

        logger.info("🔍 Incident response monitoring started")

    async def stop_monitoring(self):
        """Stop incident response monitoring."""
        self.running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.escalation_task:
            self.escalation_task.cancel()
            try:
                await self.escalation_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Incident response monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for incident response."""
        while self.running:
            try:
                await self._check_incident_status()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in incident response monitoring: {e}")
                await asyncio.sleep(60)

    async def _escalation_loop(self):
        """Escalation monitoring loop."""
        while self.running:
            try:
                await self._check_escalations()
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in escalation monitoring: {e}")
                await asyncio.sleep(300)

    async def create_incident_from_alert(
        self, alert: SecurityAlert
    ) -> SecurityIncident:
        """Create a security incident from an alert."""
        incident_id = f"INC-{int(time.time())}"

        # Map alert severity to incident severity
        severity_mapping = {
            ThreatLevel.LOW: IncidentSeverity.LOW,
            ThreatLevel.MEDIUM: IncidentSeverity.MEDIUM,
            ThreatLevel.HIGH: IncidentSeverity.HIGH,
            ThreatLevel.CRITICAL: IncidentSeverity.CRITICAL,
        }

        incident = SecurityIncident(
            id=incident_id,
            title=f"{alert.alert_type.value.replace('_', ' ').title()} Incident",
            description=alert.description,
            severity=severity_mapping.get(
                alert.threat_level, IncidentSeverity.MEDIUM
            ),
            status=IncidentStatus.OPEN,
            attack_type=alert.alert_type,
            threat_level=alert.threat_level,
            source_ip=alert.source_ip,
            affected_users=[alert.user_id] if alert.user_id else [],
            indicators=[alert.source_ip] if alert.source_ip else [],
            evidence=alert.evidence,
            timeline=[
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "event": "Incident created from security alert",
                    "details": {"alert_id": alert.id},
                }
            ],
        )

        self.incidents[incident_id] = incident

        # Log incident creation
        security_auditor.log_event(
            SecurityEventType.SECURITY_CONFIG_CHANGE,
            SecurityEventSeverity.WARNING,
            f"Security incident created: {incident_id}",
            details={
                "incident_id": incident_id,
                "attack_type": alert.alert_type,
                "severity": incident.severity,
                "source_ip": alert.source_ip,
            },
        )

        # Trigger automated response
        if self.auto_response_enabled:
            await self._trigger_automated_response(incident)

        logger.warning(f"🚨 Security incident created: {incident_id}")
        return incident

    async def _trigger_automated_response(self, incident: SecurityIncident):
        """Trigger automated response based on incident type."""
        # Find applicable playbook
        playbook = None
        for pb in self.playbooks.values():
            if (
                incident.attack_type in pb.attack_types
                and incident.severity.value >= pb.severity_threshold.value
            ):
                playbook = pb
                break

        if not playbook:
            logger.warning(f"No playbook found for incident {incident.id}")
            return

        # Execute automated actions
        for action in playbook.automated_actions:
            try:
                response = await self._execute_response_action(
                    incident, action
                )
                incident.responses.append(response)

                # Update incident timeline
                incident.timeline.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "event": f"Automated response: {action.value}",
                        "details": {
                            "success": response.success,
                            "execution_time": response.execution_time,
                        },
                    }
                )

            except Exception as e:
                logger.error(
                    f"Failed to execute {action.value} for incident {incident.id}: {e}"
                )

        # Update incident status
        incident.status = IncidentStatus.INVESTIGATING
        incident.updated_at = datetime.utcnow()

        logger.info(f"Automated response completed for incident {incident.id}")

    async def _execute_response_action(
        self, incident: SecurityIncident, action: ResponseAction
    ) -> IncidentResponse:
        """Execute a specific response action."""
        start_time = time.time()
        response_id = f"RESP-{int(time.time())}"

        response = IncidentResponse(
            id=response_id,
            incident_id=incident.id,
            action=action,
            status="executing",
            timestamp=datetime.utcnow(),
            details={},
        )

        try:
            if action == ResponseAction.BLOCK_IP:
                await self._block_ip(incident.source_ip)
                response.details = {"blocked_ip": incident.source_ip}
                response.success = True

            elif action == ResponseAction.SUSPEND_USER:
                if incident.affected_users:
                    for user in incident.affected_users:
                        await self._suspend_user(user)
                    response.details = {
                        "suspended_users": incident.affected_users
                    }
                    response.success = True

            elif action == ResponseAction.RATE_LIMIT:
                await self._apply_rate_limit(incident.source_ip)
                response.details = {"rate_limited_ip": incident.source_ip}
                response.success = True

            elif action == ResponseAction.ALERT_TEAM:
                await self._alert_security_team(incident)
                response.details = {"notification_sent": True}
                response.success = True

            elif action == ResponseAction.QUARANTINE_HOST:
                if incident.source_ip:
                    await self._quarantine_host(incident.source_ip)
                    response.details = {"quarantined_host": incident.source_ip}
                    response.success = True

            elif action == ResponseAction.DISABLE_ENDPOINT:
                endpoints = self._extract_endpoints_from_evidence(
                    incident.evidence
                )
                for endpoint in endpoints:
                    await self._disable_endpoint(endpoint)
                response.details = {"disabled_endpoints": endpoints}
                response.success = True

            elif action == ResponseAction.COLLECT_EVIDENCE:
                evidence = await self._collect_evidence(incident)
                response.details = {"evidence_collected": len(evidence)}
                response.success = True

            elif action == ResponseAction.NOTIFY_MANAGEMENT:
                await self._notify_management(incident)
                response.details = {"management_notified": True}
                response.success = True

            elif action == ResponseAction.ESCALATE:
                await self._escalate_incident(incident)
                response.details = {
                    "escalation_level": incident.escalation_level
                }
                response.success = True

            elif action == ResponseAction.LOG_ANALYSIS:
                analysis = await self._perform_log_analysis(incident)
                response.details = {"analysis_results": analysis}
                response.success = True

            else:
                response.error_message = f"Unknown action: {action}"
                response.success = False

        except Exception as e:
            response.error_message = str(e)
            response.success = False
            logger.error(f"Failed to execute {action.value}: {e}")

        response.execution_time = time.time() - start_time
        response.status = "completed"

        return response

    async def _block_ip(self, ip: str):
        """Block an IP address."""
        if ip and ip != "unknown":
            self.blocked_ips.add(ip)
            logger.info(f"🚫 IP blocked: {ip}")

            # Here you would integrate with your firewall/WAF
            # For now, we'll just log it
            security_auditor.log_event(
                SecurityEventType.SECURITY_CONFIG_CHANGE,
                SecurityEventSeverity.WARNING,
                f"IP address blocked: {ip}",
                details={"blocked_ip": ip, "automated": True},
            )

    async def _suspend_user(self, user_id: str):
        """Suspend a user account."""
        if user_id:
            self.suspended_users.add(user_id)
            logger.info(f"👤 User suspended: {user_id}")

            # Here you would integrate with your user management system
            security_auditor.log_event(
                SecurityEventType.SECURITY_CONFIG_CHANGE,
                SecurityEventSeverity.WARNING,
                f"User account suspended: {user_id}",
                details={"suspended_user": user_id, "automated": True},
            )

    async def _apply_rate_limit(self, ip: str):
        """Apply rate limiting to an IP."""
        logger.info(f"⏱️ Rate limit applied to: {ip}")
        # Integration with rate limiting system would go here

    async def _alert_security_team(self, incident: SecurityIncident):
        """Alert the security team about an incident."""
        logger.info(f"📢 Security team alerted for incident: {incident.id}")

        # Here you would integrate with notification systems
        alert_message = {
            "incident_id": incident.id,
            "title": incident.title,
            "severity": incident.severity.value,
            "attack_type": incident.attack_type.value,
            "source_ip": incident.source_ip,
            "timestamp": incident.created_at.isoformat(),
        }

        # Send to configured channels
        for channel in self.notification_channels.get("webhook", []):
            await self._send_webhook_alert(channel, alert_message)

    async def _quarantine_host(self, host: str):
        """Quarantine a host."""
        if host:
            self.quarantined_hosts.add(host)
            logger.info(f"🔐 Host quarantined: {host}")

    async def _disable_endpoint(self, endpoint: str):
        """Disable a specific endpoint."""
        if endpoint:
            self.disabled_endpoints.add(endpoint)
            logger.info(f"🚫 Endpoint disabled: {endpoint}")

    async def _collect_evidence(
        self, incident: SecurityIncident
    ) -> Dict[str, Any]:
        """Collect evidence for an incident."""
        # Here you would collect various types of evidence
        # For now, we'll just return the existing evidence
        return incident.evidence

    async def _notify_management(self, incident: SecurityIncident):
        """Notify management about a critical incident."""
        logger.info(f"📧 Management notified for incident: {incident.id}")

        # Here you would send notifications to management
        # Email, SMS, or other high-priority channels

    async def _escalate_incident(self, incident: SecurityIncident):
        """Escalate an incident to higher severity."""
        incident.escalation_level += 1
        logger.info(
            f"⬆️ Incident escalated: {incident.id} (Level {incident.escalation_level})"
        )

    async def _perform_log_analysis(
        self, incident: SecurityIncident
    ) -> Dict[str, Any]:
        """Perform automated log analysis."""
        analysis = {
            "patterns_detected": [],
            "timeline_reconstructed": True,
            "related_events": [],
            "recommendations": [],
        }

        # Here you would perform actual log analysis
        return analysis

    async def _send_webhook_alert(
        self, webhook_url: str, alert_data: Dict[str, Any]
    ):
        """Send alert via webhook."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=alert_data)
                response.raise_for_status()
                logger.info(f"Webhook alert sent to {webhook_url}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def _extract_endpoints_from_evidence(
        self, evidence: Dict[str, Any]
    ) -> List[str]:
        """Extract endpoints from incident evidence."""
        endpoints = []

        # Extract endpoints from evidence
        if "endpoint" in evidence:
            endpoints.append(evidence["endpoint"])

        return endpoints

    async def _check_incident_status(self):
        """Check and update incident status."""
        for incident in self.incidents.values():
            if incident.status == IncidentStatus.INVESTIGATING:
                # Check if all automated responses completed successfully
                automated_responses = [
                    r for r in incident.responses if r.automated
                ]
                if automated_responses and all(
                    r.success for r in automated_responses
                ):
                    incident.status = IncidentStatus.CONTAINMENT
                    incident.updated_at = datetime.utcnow()

    async def _check_escalations(self):
        """Check for incidents that need escalation."""
        now = datetime.utcnow()

        for incident in self.incidents.values():
            if incident.status in [
                IncidentStatus.OPEN,
                IncidentStatus.INVESTIGATING,
            ]:
                # Check if escalation timeout has passed
                time_since_creation = now - incident.created_at
                escalation_timeout = timedelta(minutes=self.escalation_timeout)

                if time_since_creation > escalation_timeout:
                    await self._escalate_incident(incident)
                    await self._notify_management(incident)

    def get_incident_statistics(self) -> Dict[str, Any]:
        """Get incident statistics."""
        incidents = list(self.incidents.values())

        stats = {
            "total_incidents": len(incidents),
            "open_incidents": len(
                [i for i in incidents if i.status == IncidentStatus.OPEN]
            ),
            "resolved_incidents": len(
                [i for i in incidents if i.status == IncidentStatus.RESOLVED]
            ),
            "by_severity": {},
            "by_attack_type": {},
            "by_status": {},
            "average_response_time": 0.0,
            "escalated_incidents": len(
                [i for i in incidents if i.escalation_level > 0]
            ),
            "false_positives": len([i for i in incidents if i.false_positive]),
        }

        # Calculate statistics
        for incident in incidents:
            # Count by severity
            severity_key = incident.severity.value
            stats["by_severity"][severity_key] = (
                stats["by_severity"].get(severity_key, 0) + 1
            )

            # Count by attack type
            attack_type_key = incident.attack_type.value
            stats["by_attack_type"][attack_type_key] = (
                stats["by_attack_type"].get(attack_type_key, 0) + 1
            )

            # Count by status
            status_key = incident.status.value
            stats["by_status"][status_key] = (
                stats["by_status"].get(status_key, 0) + 1
            )

        return stats

    def get_recent_incidents(self, limit: int = 10) -> List[SecurityIncident]:
        """Get recent incidents."""
        incidents = sorted(
            self.incidents.values(), key=lambda x: x.created_at, reverse=True
        )
        return incidents[:limit]

    def get_incident(self, incident_id: str) -> Optional[SecurityIncident]:
        """Get a specific incident."""
        return self.incidents.get(incident_id)

    def resolve_incident(
        self, incident_id: str, resolution_notes: str = ""
    ) -> bool:
        """Resolve an incident."""
        if incident_id in self.incidents:
            incident = self.incidents[incident_id]
            incident.status = IncidentStatus.RESOLVED
            incident.resolved_at = datetime.utcnow()
            incident.updated_at = datetime.utcnow()
            incident.lesson_learned = resolution_notes

            logger.info(f"✅ Incident resolved: {incident_id}")
            return True
        return False

    def mark_false_positive(self, incident_id: str, notes: str = "") -> bool:
        """Mark incident as false positive."""
        if incident_id in self.incidents:
            incident = self.incidents[incident_id]
            incident.false_positive = True
            incident.status = IncidentStatus.CLOSED
            incident.resolved_at = datetime.utcnow()
            incident.updated_at = datetime.utcnow()
            incident.lesson_learned = f"False positive: {notes}"

            logger.info(f"🔍 Incident marked as false positive: {incident_id}")
            return True
        return False

    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips

    def is_user_suspended(self, user_id: str) -> bool:
        """Check if user is suspended."""
        return user_id in self.suspended_users

    def is_endpoint_disabled(self, endpoint: str) -> bool:
        """Check if endpoint is disabled."""
        return endpoint in self.disabled_endpoints

    def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address."""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"🔓 IP unblocked: {ip}")
            return True
        return False

    def enable_endpoint(self, endpoint: str) -> bool:
        """Enable a disabled endpoint."""
        if endpoint in self.disabled_endpoints:
            self.disabled_endpoints.remove(endpoint)
            logger.info(f"✅ Endpoint enabled: {endpoint}")
            return True
        return False


# Global incident response system instance
incident_response = IncidentResponseSystem()


async def start_incident_response():
    """Start the incident response system."""
    await incident_response.start_monitoring()


async def stop_incident_response():
    """Stop the incident response system."""
    await incident_response.stop_monitoring()


def get_incident_statistics() -> Dict[str, Any]:
    """Get incident statistics."""
    return incident_response.get_incident_statistics()


def get_recent_incidents(limit: int = 10) -> List[SecurityIncident]:
    """Get recent incidents."""
    return incident_response.get_recent_incidents(limit)

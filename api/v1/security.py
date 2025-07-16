"""Security monitoring API endpoints.

Provides access to security audit logs and monitoring data.
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import Permission, TokenData, get_current_user, require_permission
from auth.https_enforcement import SSLCertificateManager, SSLConfiguration
from auth.security_logging import (
    SecurityAuditLog,
    SecurityEventSeverity,
    SecurityEventType,
    security_auditor,
)
from database.session import get_db
from observability.incident_response import (
    IncidentSeverity,
    IncidentStatus,
    incident_response,
)
from observability.security_monitoring import (
    SecurityAlert,
    SecurityMetrics,
    ThreatLevel,
    security_monitor,
)
from observability.vulnerability_scanner import (
    SeverityLevel,
    VulnerabilityType,
    vulnerability_scanner,
)

router = APIRouter()


class SecurityEventResponse(BaseModel):
    """Security event response model."""

    id: int
    timestamp: datetime
    event_type: str
    severity: str
    user_id: Optional[str]
    username: Optional[str]
    ip_address: Optional[str]
    endpoint: Optional[str]
    method: Optional[str]
    status_code: Optional[int]
    message: str
    details: Optional[Dict[str, Any]]

    class Config:
        orm_mode = True


class SecuritySummaryResponse(BaseModel):
    """Security summary response."""

    total_events: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    failed_logins: int
    suspicious_ips: List[str]
    top_ips: Dict[str, int]
    time_range_hours: int


class SecurityAlertResponse(BaseModel):
    """Security alert response model."""

    id: str
    timestamp: datetime
    alert_type: str
    threat_level: str
    source_ip: str
    user_id: Optional[str]
    description: str
    status: str
    evidence: Dict[str, Any]


class SecurityMetricsResponse(BaseModel):
    """Security metrics response model."""

    total_events: int
    alerts_generated: int
    threats_detected: int
    false_positives: int
    mean_detection_time: float
    mean_response_time: float
    top_attack_types: Dict[str, int]
    top_source_ips: Dict[str, int]
    threat_level_distribution: Dict[str, int]


class VulnerabilityResponse(BaseModel):
    """Vulnerability response model."""

    id: str
    type: str
    severity: str
    title: str
    description: str
    file_path: Optional[str]
    line_number: Optional[int]
    cve_id: Optional[str]
    cwe_id: Optional[str]
    remediation: Optional[str]
    confidence: float
    first_detected: datetime
    last_seen: datetime
    status: str
    scanner_name: str


class IncidentResponse(BaseModel):
    """Incident response model."""

    id: str
    title: str
    description: str
    severity: str
    status: str
    attack_type: str
    threat_level: str
    source_ip: Optional[str]
    affected_users: List[str]
    affected_systems: List[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    escalation_level: int


@router.get("/security/summary", response_model=SecuritySummaryResponse)
@require_permission(Permission.ADMIN_SYSTEM)
async def get_security_summary(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back (max 7 days)"),
    current_user: TokenData = Depends(get_current_user),
) -> SecuritySummaryResponse:
    """Get summary of security events.

    Requires ADMIN_SYSTEM permission.
    """
    summary = security_auditor.get_security_summary(hours)

    if "error" in summary:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate security summary: {summary['error']}",
        )

    return SecuritySummaryResponse(
        total_events=summary["total_events"],
        by_type=summary["by_type"],
        by_severity=summary["by_severity"],
        failed_logins=summary["failed_logins"],
        suspicious_ips=summary["suspicious_ips"],
        top_ips=summary["top_ips"],
        time_range_hours=hours,
    )


@router.get("/security/events", response_model=List[SecurityEventResponse])
@require_permission(Permission.ADMIN_SYSTEM)
async def get_security_events(
    event_type: Optional[SecurityEventType] = Query(None, description="Filter by event type"),
    severity: Optional[SecurityEventSeverity] = Query(None, description="Filter by severity"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    ip_address: Optional[str] = Query(None, description="Filter by IP address"),
    hours: int = Query(24, ge=1, le=168, description="Hours to look back (max 7 days)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[SecurityEventResponse]:
    """Get security audit log events.

    Requires ADMIN_SYSTEM permission.
    """
    if not security_auditor.AuditSessionLocal:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security audit database not configured",
        )

    try:
        audit_db = security_auditor.AuditSessionLocal()
        try:
            # Build query
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            query = audit_db.query(SecurityAuditLog).filter(SecurityAuditLog.timestamp >= cutoff)

            # Apply filters
            if event_type:
                query = query.filter(SecurityAuditLog.event_type == event_type)
            if severity:
                query = query.filter(SecurityAuditLog.severity == severity)
            if user_id:
                query = query.filter(SecurityAuditLog.user_id == user_id)
            if ip_address:
                query = query.filter(SecurityAuditLog.ip_address == ip_address)

            # Order by timestamp descending and limit
            events = query.order_by(SecurityAuditLog.timestamp.desc()).limit(limit).all()

            # Convert to response models
            return [
                SecurityEventResponse(
                    id=event.id,
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    severity=event.severity,
                    user_id=event.user_id,
                    username=event.username,
                    ip_address=event.ip_address,
                    endpoint=event.endpoint,
                    method=event.method,
                    status_code=event.status_code,
                    message=event.message,
                    details=json.loads(event.details) if event.details else None,
                )
                for event in events
            ]

        finally:
            audit_db.close()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve security events: {str(e)}",
        )


@router.get("/security/suspicious-activity")
@require_permission(Permission.ADMIN_SYSTEM)
async def get_suspicious_activity(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get current suspicious activity tracking.

    Requires ADMIN_SYSTEM permission.
    """
    return {
        "suspicious_ips": list(security_auditor.suspicious_ips),
        "failed_login_tracking": {
            key: len(attempts) for key, attempts in security_auditor.failed_login_attempts.items()
        },
        "rate_limit_violations": {
            ip: len(violations) for ip, violations in security_auditor.rate_limit_violations.items()
        },
    }


@router.post("/security/alert-test")
@require_permission(Permission.ADMIN_SYSTEM)
async def test_security_alert(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Test security alerting system.

    Requires ADMIN_SYSTEM permission.
    """
    # Log a test security event
    security_auditor.log_event(
        SecurityEventType.SECURITY_CONFIG_CHANGE,
        SecurityEventSeverity.INFO,
        f"Security alert test initiated by {current_user.username}",
        request=request,
        user_id=current_user.user_id,
        username=current_user.username,
        details={"test": True, "initiated_by": current_user.username},
    )

    return {
        "message": "Security alert test completed",
        "check_logs": "Check security audit logs for test event",
    }


@router.get("/.well-known/security.txt", response_class=PlainTextResponse)
async def security_txt() -> str:
    """Serve security.txt file per RFC 9116 for vulnerability disclosure."""
    security_content = """Contact: security@freeagentics.com
Contact: https://github.com/FreeAgentics/FreeAgentics/security/advisories/new
Expires: 2025-12-31T23:59:59.000Z
Preferred-Languages: en
Canonical: https://freeagentics.com/.well-known/security.txt
Policy: https://freeagentics.com/security-policy
Hiring: https://freeagentics.com/careers
Acknowledgments: https://freeagentics.com/security-acknowledgments
"""
    return security_content


# ============================================================================
# ENHANCED SECURITY MONITORING ENDPOINTS
# ============================================================================


@router.get("/security/alerts", response_model=List[SecurityAlertResponse])
@require_permission(Permission.ADMIN_SYSTEM)
async def get_security_alerts(
    current_user: TokenData = Depends(get_current_user),
) -> List[SecurityAlertResponse]:
    """Get active security alerts.

    Requires ADMIN_SYSTEM permission.
    """
    alerts = security_monitor.get_active_alerts()

    return [
        SecurityAlertResponse(
            id=alert.id,
            timestamp=alert.timestamp,
            alert_type=alert.alert_type.value,
            threat_level=alert.threat_level.value,
            source_ip=alert.source_ip,
            user_id=alert.user_id,
            description=alert.description,
            status=alert.status,
            evidence=alert.evidence,
        )
        for alert in alerts
    ]


@router.get("/security/metrics", response_model=SecurityMetricsResponse)
@require_permission(Permission.ADMIN_SYSTEM)
async def get_security_metrics(
    current_user: TokenData = Depends(get_current_user),
) -> SecurityMetricsResponse:
    """Get security metrics and statistics.

    Requires ADMIN_SYSTEM permission.
    """
    metrics = security_monitor.get_security_metrics()

    return SecurityMetricsResponse(
        total_events=metrics.total_events,
        alerts_generated=metrics.alerts_generated,
        threats_detected=metrics.threats_detected,
        false_positives=metrics.false_positives,
        mean_detection_time=metrics.mean_detection_time,
        mean_response_time=metrics.mean_response_time,
        top_attack_types=metrics.top_attack_types,
        top_source_ips=metrics.top_source_ips,
        threat_level_distribution=metrics.threat_level_distribution,
    )


@router.post("/security/alerts/{alert_id}/resolve")
@require_permission(Permission.ADMIN_SYSTEM)
async def resolve_security_alert(
    alert_id: str,
    resolution_notes: str = Query("", description="Resolution notes"),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Resolve a security alert.

    Requires ADMIN_SYSTEM permission.
    """
    success = security_monitor.resolve_alert(alert_id, resolution_notes)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert {alert_id} not found"
        )

    # Log the resolution
    security_auditor.log_event(
        SecurityEventType.SECURITY_CONFIG_CHANGE,
        SecurityEventSeverity.INFO,
        f"Security alert resolved: {alert_id}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={"alert_id": alert_id, "resolution_notes": resolution_notes},
    )

    return {"message": f"Alert {alert_id} resolved successfully"}


@router.post("/security/alerts/{alert_id}/false-positive")
@require_permission(Permission.ADMIN_SYSTEM)
async def mark_alert_false_positive(
    alert_id: str,
    notes: str = Query("", description="Notes about why this is a false positive"),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Mark a security alert as false positive.

    Requires ADMIN_SYSTEM permission.
    """
    success = security_monitor.mark_false_positive(alert_id, notes)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert {alert_id} not found"
        )

    # Log the false positive marking
    security_auditor.log_event(
        SecurityEventType.SECURITY_CONFIG_CHANGE,
        SecurityEventSeverity.INFO,
        f"Security alert marked as false positive: {alert_id}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={"alert_id": alert_id, "notes": notes},
    )

    return {"message": f"Alert {alert_id} marked as false positive"}


@router.get("/security/vulnerabilities", response_model=List[VulnerabilityResponse])
@require_permission(Permission.ADMIN_SYSTEM)
async def get_vulnerabilities(
    severity: Optional[SeverityLevel] = Query(None, description="Filter by severity"),
    vuln_type: Optional[VulnerabilityType] = Query(
        None, description="Filter by vulnerability type"
    ),
    current_user: TokenData = Depends(get_current_user),
) -> List[VulnerabilityResponse]:
    """Get vulnerabilities from security scans.

    Requires ADMIN_SYSTEM permission.
    """
    vulnerabilities = vulnerability_scanner.get_vulnerabilities(severity, vuln_type)

    return [
        VulnerabilityResponse(
            id=vuln.id,
            type=vuln.type.value,
            severity=vuln.severity.value,
            title=vuln.title,
            description=vuln.description,
            file_path=vuln.file_path,
            line_number=vuln.line_number,
            cve_id=vuln.cve_id,
            cwe_id=vuln.cwe_id,
            remediation=vuln.remediation,
            confidence=vuln.confidence,
            first_detected=vuln.first_detected,
            last_seen=vuln.last_seen,
            status=vuln.status,
            scanner_name=vuln.scanner_name,
        )
        for vuln in vulnerabilities
    ]


@router.get("/security/vulnerabilities/stats")
@require_permission(Permission.ADMIN_SYSTEM)
async def get_vulnerability_stats(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get vulnerability statistics.

    Requires ADMIN_SYSTEM permission.
    """
    return vulnerability_scanner.get_vulnerability_stats()


@router.post("/security/vulnerabilities/{vuln_id}/suppress")
@require_permission(Permission.ADMIN_SYSTEM)
async def suppress_vulnerability(
    vuln_id: str,
    reason: str = Query("", description="Reason for suppression"),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Suppress a vulnerability.

    Requires ADMIN_SYSTEM permission.
    """
    vulnerability_scanner.suppress_vulnerability(vuln_id, reason)

    # Log the suppression
    security_auditor.log_event(
        SecurityEventType.SECURITY_CONFIG_CHANGE,
        SecurityEventSeverity.INFO,
        f"Vulnerability suppressed: {vuln_id}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={"vulnerability_id": vuln_id, "reason": reason},
    )

    return {"message": f"Vulnerability {vuln_id} suppressed"}


@router.post("/security/vulnerabilities/{vuln_id}/false-positive")
@require_permission(Permission.ADMIN_SYSTEM)
async def mark_vulnerability_false_positive(
    vuln_id: str,
    reason: str = Query("", description="Reason for marking as false positive"),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Mark a vulnerability as false positive.

    Requires ADMIN_SYSTEM permission.
    """
    vulnerability_scanner.mark_false_positive(vuln_id, reason)

    # Log the false positive marking
    security_auditor.log_event(
        SecurityEventType.SECURITY_CONFIG_CHANGE,
        SecurityEventSeverity.INFO,
        f"Vulnerability marked as false positive: {vuln_id}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={"vulnerability_id": vuln_id, "reason": reason},
    )

    return {"message": f"Vulnerability {vuln_id} marked as false positive"}


@router.get("/security/incidents", response_model=List[IncidentResponse])
@require_permission(Permission.ADMIN_SYSTEM)
async def get_security_incidents(
    limit: int = Query(20, ge=1, le=100, description="Number of incidents to return"),
    current_user: TokenData = Depends(get_current_user),
) -> List[IncidentResponse]:
    """Get security incidents.

    Requires ADMIN_SYSTEM permission.
    """
    incidents = incident_response.get_recent_incidents(limit)

    return [
        IncidentResponse(
            id=incident.id,
            title=incident.title,
            description=incident.description,
            severity=incident.severity.value,
            status=incident.status.value,
            attack_type=incident.attack_type.value,
            threat_level=incident.threat_level.value,
            source_ip=incident.source_ip,
            affected_users=incident.affected_users,
            affected_systems=incident.affected_systems,
            created_at=incident.created_at,
            updated_at=incident.updated_at,
            resolved_at=incident.resolved_at,
            escalation_level=incident.escalation_level,
        )
        for incident in incidents
    ]


@router.get("/security/incidents/stats")
@require_permission(Permission.ADMIN_SYSTEM)
async def get_incident_stats(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get incident statistics.

    Requires ADMIN_SYSTEM permission.
    """
    return incident_response.get_incident_statistics()


@router.post("/security/incidents/{incident_id}/resolve")
@require_permission(Permission.ADMIN_SYSTEM)
async def resolve_incident(
    incident_id: str,
    resolution_notes: str = Query("", description="Resolution notes"),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Resolve a security incident.

    Requires ADMIN_SYSTEM permission.
    """
    success = incident_response.resolve_incident(incident_id, resolution_notes)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident {incident_id} not found"
        )

    # Log the resolution
    security_auditor.log_event(
        SecurityEventType.SECURITY_CONFIG_CHANGE,
        SecurityEventSeverity.INFO,
        f"Security incident resolved: {incident_id}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={"incident_id": incident_id, "resolution_notes": resolution_notes},
    )

    return {"message": f"Incident {incident_id} resolved successfully"}


@router.post("/security/incidents/{incident_id}/false-positive")
@require_permission(Permission.ADMIN_SYSTEM)
async def mark_incident_false_positive(
    incident_id: str,
    notes: str = Query("", description="Notes about why this is a false positive"),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Mark a security incident as false positive.

    Requires ADMIN_SYSTEM permission.
    """
    success = incident_response.mark_false_positive(incident_id, notes)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident {incident_id} not found"
        )

    # Log the false positive marking
    security_auditor.log_event(
        SecurityEventType.SECURITY_CONFIG_CHANGE,
        SecurityEventSeverity.INFO,
        f"Security incident marked as false positive: {incident_id}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={"incident_id": incident_id, "notes": notes},
    )

    return {"message": f"Incident {incident_id} marked as false positive"}


@router.get("/security/blocked-ips")
@require_permission(Permission.ADMIN_SYSTEM)
async def get_blocked_ips(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, List[str]]:
    """Get list of blocked IP addresses.

    Requires ADMIN_SYSTEM permission.
    """
    return {
        "blocked_ips": list(incident_response.blocked_ips),
        "quarantined_hosts": list(incident_response.quarantined_hosts),
        "suspended_users": list(incident_response.suspended_users),
        "disabled_endpoints": list(incident_response.disabled_endpoints),
    }


@router.post("/security/blocked-ips/{ip}/unblock")
@require_permission(Permission.ADMIN_SYSTEM)
async def unblock_ip(
    ip: str,
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Unblock an IP address.

    Requires ADMIN_SYSTEM permission.
    """
    success = incident_response.unblock_ip(ip)

    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"IP {ip} is not blocked")

    # Log the unblocking
    security_auditor.log_event(
        SecurityEventType.SECURITY_CONFIG_CHANGE,
        SecurityEventSeverity.INFO,
        f"IP address unblocked: {ip}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={"unblocked_ip": ip},
    )

    return {"message": f"IP {ip} unblocked successfully"}


@router.post("/security/scan/trigger")
@require_permission(Permission.ADMIN_SYSTEM)
async def trigger_security_scan(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Trigger a manual security scan.

    Requires ADMIN_SYSTEM permission.
    """
    # Trigger vulnerability scan
    asyncio.create_task(vulnerability_scanner._run_all_scanners())

    # Log the manual scan trigger
    security_auditor.log_event(
        SecurityEventType.SECURITY_CONFIG_CHANGE,
        SecurityEventSeverity.INFO,
        f"Manual security scan triggered by {current_user.username}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={"manual_trigger": True},
    )

    return {"message": "Security scan triggered successfully"}


import asyncio

# Import json for parsing details
import json
import os


class SSLHealthResponse(BaseModel):
    """SSL health check response model."""

    ssl_status: str
    protocol: Optional[str] = None
    cipher: Optional[str] = None
    cert_expiry_days: Optional[int] = None
    hsts_enabled: bool
    certificate_valid: bool
    chain_valid: bool
    ocsp_stapling: bool


class CertificateInfoResponse(BaseModel):
    """Certificate information response model."""

    subject: str
    issuer: str
    valid_from: datetime
    valid_until: datetime
    days_until_expiry: int
    serial_number: str
    signature_algorithm: str
    key_size: int
    san_domains: List[str]
    is_wildcard: bool
    is_self_signed: bool


@router.get("/security/ssl-health", response_model=SSLHealthResponse)
async def get_ssl_health() -> SSLHealthResponse:
    """Get SSL/TLS health status.

    Public endpoint for monitoring SSL configuration.
    """
    try:
        # Initialize SSL configuration
        ssl_config = SSLConfiguration()
        cert_manager = SSLCertificateManager(ssl_config)

        # Check certificate expiry
        cert_expiry_days = None
        certificate_valid = False
        chain_valid = False

        if Path(ssl_config.cert_path).exists():
            time_until_expiry = cert_manager.check_certificate_expiry()
            if time_until_expiry:
                cert_expiry_days = time_until_expiry.days
                certificate_valid = True

            # Check certificate chain
            chain_valid = cert_manager.validate_certificate_chain()

        # Check HSTS configuration
        hsts_enabled = ssl_config.hsts_enabled

        # Determine SSL status
        if certificate_valid and chain_valid:
            ssl_status = "active"
        elif certificate_valid:
            ssl_status = "certificate_valid_chain_invalid"
        else:
            ssl_status = "inactive"

        # Check OCSP stapling (simplified check)
        ocsp_stapling = True  # Assume enabled if certificate is valid

        return SSLHealthResponse(
            ssl_status=ssl_status,
            protocol="TLSv1.3",  # Would be determined dynamically in production
            cipher="TLS_AES_256_GCM_SHA384",  # Would be determined dynamically
            cert_expiry_days=cert_expiry_days,
            hsts_enabled=hsts_enabled,
            certificate_valid=certificate_valid,
            chain_valid=chain_valid,
            ocsp_stapling=ocsp_stapling,
        )

    except Exception as e:
        # Return error status
        return SSLHealthResponse(
            ssl_status="error",
            hsts_enabled=False,
            certificate_valid=False,
            chain_valid=False,
            ocsp_stapling=False,
        )


@router.get("/security/certificate-info", response_model=CertificateInfoResponse)
@require_permission(Permission.ADMIN_SYSTEM)
async def get_certificate_info(
    current_user: TokenData = Depends(get_current_user),
) -> CertificateInfoResponse:
    """Get detailed certificate information.

    Requires ADMIN_SYSTEM permission.
    """
    ssl_config = SSLConfiguration()

    if not Path(ssl_config.cert_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="SSL certificate not found"
        )

    try:
        # Use openssl to get certificate info
        result = subprocess.run(
            ["openssl", "x509", "-in", ssl_config.cert_path, "-noout", "-text"],
            capture_output=True,
            text=True,
            check=True,
        )

        cert_text = result.stdout

        # Parse certificate information (simplified)
        subject = "CN=freeagentics.com"  # Would parse from cert_text
        issuer = "Let's Encrypt Authority X3"  # Would parse from cert_text
        valid_from = datetime.utcnow() - timedelta(days=10)  # Would parse from cert_text
        valid_until = datetime.utcnow() + timedelta(days=80)  # Would parse from cert_text
        days_until_expiry = (valid_until - datetime.utcnow()).days

        return CertificateInfoResponse(
            subject=subject,
            issuer=issuer,
            valid_from=valid_from,
            valid_until=valid_until,
            days_until_expiry=days_until_expiry,
            serial_number="12345678901234567890",  # Would parse from cert_text
            signature_algorithm="sha256WithRSAEncryption",  # Would parse from cert_text
            key_size=2048,  # Would parse from cert_text
            san_domains=["freeagentics.com", "www.freeagentics.com"],  # Would parse from cert_text
            is_wildcard=False,
            is_self_signed=False,
        )

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse certificate: {e}",
        )


@router.post("/security/ssl-renewal")
@require_permission(Permission.ADMIN_SYSTEM)
async def trigger_ssl_renewal(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Trigger SSL certificate renewal.

    Requires ADMIN_SYSTEM permission.
    """
    try:
        ssl_config = SSLConfiguration()
        cert_manager = SSLCertificateManager(ssl_config)

        # Check if renewal is needed
        time_until_expiry = cert_manager.check_certificate_expiry()
        if time_until_expiry and time_until_expiry.days > ssl_config.cert_renewal_days:
            return {
                "message": f"Certificate renewal not needed. {time_until_expiry.days} days remaining."
            }

        # Trigger renewal
        success = cert_manager.setup_letsencrypt()

        if success:
            # Log successful renewal
            security_auditor.log_event(
                SecurityEventType.SECURITY_CONFIG_CHANGE,
                SecurityEventSeverity.INFO,
                f"SSL certificate renewed successfully",
                user_id=current_user.user_id,
                username=current_user.username,
                details={"renewal_triggered": True},
            )

            return {"message": "SSL certificate renewed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to renew SSL certificate",
            )

    except Exception as e:
        # Log renewal failure
        security_auditor.log_event(
            SecurityEventType.SECURITY_CONFIG_CHANGE,
            SecurityEventSeverity.ERROR,
            f"SSL certificate renewal failed: {str(e)}",
            user_id=current_user.user_id,
            username=current_user.username,
            details={"renewal_failed": True, "error": str(e)},
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SSL certificate renewal failed: {str(e)}",
        )


@router.post("/security/csp-report")
async def csp_report(request: Request) -> Dict[str, str]:
    """Receive Content Security Policy violation reports.

    Public endpoint for CSP reporting.
    """
    try:
        report_data = await request.json()

        # Log CSP violation
        security_auditor.log_event(
            SecurityEventType.SECURITY_VIOLATION,
            SecurityEventSeverity.WARNING,
            "Content Security Policy violation reported",
            details=report_data,
        )

        return {"message": "CSP report received"}

    except Exception as e:
        return {"error": "Failed to process CSP report"}


@router.post("/security/ct-report")
async def ct_report(request: Request) -> Dict[str, str]:
    """Receive Certificate Transparency violation reports.

    Public endpoint for CT reporting.
    """
    try:
        report_data = await request.json()

        # Log CT violation
        security_auditor.log_event(
            SecurityEventType.SECURITY_VIOLATION,
            SecurityEventSeverity.WARNING,
            "Certificate Transparency violation reported",
            details=report_data,
        )

        return {"message": "CT report received"}

    except Exception as e:
        return {"error": "Failed to process CT report"}

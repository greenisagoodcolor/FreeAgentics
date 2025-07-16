"""Security audit logging implementation for FreeAgentics.

Provides comprehensive logging of security events including authentication,
authorization, access patterns, and suspicious activities.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import Request
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Security logger configuration
security_logger = logging.getLogger("security.audit")
security_logger.setLevel(logging.INFO)

# Ensure security logs go to separate file
if not security_logger.handlers:
    handler = logging.FileHandler("logs/security_audit.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    security_logger.addHandler(handler)

# Also log critical security events to main logger
critical_logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Types of security events to log."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_INVALID = "token_invalid"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHECK = "permission_check"
    PRIVILEGE_ESCALATION = "privilege_escalation"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RATE_LIMIT_WARNING = "rate_limit_warning"

    # Suspicious activity
    BRUTE_FORCE_DETECTED = "brute_force_detected"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    COMMAND_INJECTION_ATTEMPT = "command_injection_attempt"
    SUSPICIOUS_PATTERN = "suspicious_pattern"

    # API access
    API_ACCESS = "api_access"
    API_ERROR = "api_error"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

    # System events
    SECURITY_CONFIG_CHANGE = "security_config_change"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    USER_MODIFIED = "user_modified"
    PASSWORD_CHANGED = "password_changed"

    # MFA events
    MFA_ENROLLED = "mfa_enrolled"
    MFA_ENROLLMENT_FAILED = "mfa_enrollment_failed"
    MFA_SUCCESS = "mfa_success"
    MFA_FAILED = "mfa_failed"
    MFA_ERROR = "mfa_error"
    MFA_DISABLED = "mfa_disabled"
    MFA_LOCKOUT = "mfa_lockout"
    MFA_BACKUP_CODES_REGENERATED = "mfa_backup_codes_regenerated"


class SecurityEventSeverity(str, Enum):
    """Severity levels for security events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Database for security audit logs (separate from main DB)
Base = declarative_base()


class SecurityAuditLog(Base):
    """Security audit log database model."""

    __tablename__ = "security_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    event_type = Column(String(50), index=True)
    severity = Column(String(20), index=True)
    user_id = Column(String(100), nullable=True, index=True)
    username = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), index=True)
    user_agent = Column(Text)
    endpoint = Column(String(200), nullable=True)
    method = Column(String(10), nullable=True)
    status_code = Column(Integer, nullable=True)
    message = Column(Text)
    details = Column(Text)  # JSON string for additional data
    session_id = Column(String(100), nullable=True)
    request_id = Column(String(100), nullable=True)


# Create separate engine for audit logs
AUDIT_DB_URL = os.getenv("AUDIT_DATABASE_URL", os.getenv("DATABASE_URL"))
if AUDIT_DB_URL:
    # Configure audit engine based on database dialect
    audit_engine_args = {}

    if AUDIT_DB_URL.startswith("postgresql://") or AUDIT_DB_URL.startswith("postgres://"):
        # PostgreSQL-specific configuration
        audit_engine_args.update(
            {
                "pool_size": 5,
                "max_overflow": 10,
                "pool_pre_ping": True,
            }
        )
    elif AUDIT_DB_URL.startswith("sqlite://"):
        # SQLite-specific configuration
        audit_engine_args.update({"connect_args": {"check_same_thread": False}})

    audit_engine = create_engine(AUDIT_DB_URL, **audit_engine_args)
    AuditSessionLocal = sessionmaker(bind=audit_engine)
    Base.metadata.create_all(bind=audit_engine)
else:
    audit_engine = None
    AuditSessionLocal = None


class SecurityAuditor:
    """Security audit logging manager."""

    def __init__(self):
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.rate_limit_violations: Dict[str, List[datetime]] = {}
        self.suspicious_ips: set = set()

    def log_event(
        self,
        event_type: SecurityEventType,
        severity: SecurityEventSeverity,
        message: str,
        request: Optional[Request] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> None:
        """Log a security event."""
        try:
            # Prepare log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "severity": severity,
                "message": message,
                "user_id": user_id,
                "username": username,
                "details": details or {},
            }

            # Extract request information if available
            if request:
                log_entry.update(
                    {
                        "ip_address": self._get_client_ip(request),
                        "user_agent": request.headers.get("User-Agent", "Unknown"),
                        "endpoint": str(request.url.path),
                        "method": request.method,
                        "request_id": request.headers.get("X-Request-ID"),
                    }
                )

            if status_code:
                log_entry["status_code"] = status_code

            # Log to file
            self._write_to_log(log_entry, severity)

            # Store in database if available
            if AuditSessionLocal:
                self._write_to_database(log_entry)

            # Check for patterns requiring alerts
            self._check_alert_conditions(event_type, log_entry)

        except Exception as e:
            critical_logger.error(f"Failed to log security event: {e}")

    def _write_to_log(self, log_entry: Dict[str, Any], severity: SecurityEventSeverity) -> None:
        """Write security event to log file."""
        log_message = json.dumps(log_entry)

        if severity == SecurityEventSeverity.INFO:
            security_logger.info(log_message)
        elif severity == SecurityEventSeverity.WARNING:
            security_logger.warning(log_message)
        elif severity == SecurityEventSeverity.ERROR:
            security_logger.error(log_message)
        elif severity == SecurityEventSeverity.CRITICAL:
            security_logger.critical(log_message)
            # Also log critical events to main logger
            critical_logger.critical(f"SECURITY CRITICAL: {log_entry['message']}")

    def _write_to_database(self, log_entry: Dict[str, Any]) -> None:
        """Store security event in database."""
        try:
            db = AuditSessionLocal()
            try:
                audit_log = SecurityAuditLog(
                    event_type=log_entry["event_type"],
                    severity=log_entry["severity"],
                    user_id=log_entry.get("user_id"),
                    username=log_entry.get("username"),
                    ip_address=log_entry.get("ip_address"),
                    user_agent=log_entry.get("user_agent"),
                    endpoint=log_entry.get("endpoint"),
                    method=log_entry.get("method"),
                    status_code=log_entry.get("status_code"),
                    message=log_entry["message"],
                    details=json.dumps(log_entry.get("details", {})),
                    request_id=log_entry.get("request_id"),
                )
                db.add(audit_log)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            security_logger.error(f"Failed to write to audit database: {e}")

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        return request.client.host if request.client else "unknown"

    def _check_alert_conditions(
        self, event_type: SecurityEventType, log_entry: Dict[str, Any]
    ) -> None:
        """Check if event requires immediate alerting."""
        ip_address = log_entry.get("ip_address", "unknown")

        # Track failed login attempts
        if event_type == SecurityEventType.LOGIN_FAILURE:
            self._track_failed_login(ip_address, log_entry.get("username"))

        # Track rate limit violations
        elif event_type == SecurityEventType.RATE_LIMIT_EXCEEDED:
            self._track_rate_limit_violation(ip_address)

        # Critical events requiring immediate alert
        elif event_type in [
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventType.COMMAND_INJECTION_ATTEMPT,
            SecurityEventType.PRIVILEGE_ESCALATION,
            SecurityEventType.BRUTE_FORCE_DETECTED,
        ]:
            self._send_security_alert(event_type, log_entry)
            self.suspicious_ips.add(ip_address)

    def _track_failed_login(self, ip_address: str, username: Optional[str]) -> None:
        """Track failed login attempts for brute force detection."""
        key = f"{ip_address}:{username}" if username else ip_address
        now = datetime.utcnow()

        # Initialize or clean old attempts
        if key not in self.failed_login_attempts:
            self.failed_login_attempts[key] = []

        # Remove attempts older than 15 minutes
        cutoff = now - timedelta(minutes=15)
        self.failed_login_attempts[key] = [
            attempt for attempt in self.failed_login_attempts[key] if attempt > cutoff
        ]

        # Add current attempt
        self.failed_login_attempts[key].append(now)

        # Check for brute force (5 failures in 15 minutes)
        if len(self.failed_login_attempts[key]) >= 5:
            self.log_event(
                SecurityEventType.BRUTE_FORCE_DETECTED,
                SecurityEventSeverity.CRITICAL,
                f"Brute force detected from {ip_address} targeting {username or 'multiple users'}",
                details={
                    "attempts": len(self.failed_login_attempts[key]),
                    "ip_address": ip_address,
                    "username": username,
                },
            )

    def _track_rate_limit_violation(self, ip_address: str) -> None:
        """Track rate limit violations for abuse detection."""
        now = datetime.utcnow()

        if ip_address not in self.rate_limit_violations:
            self.rate_limit_violations[ip_address] = []

        # Remove old violations (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        self.rate_limit_violations[ip_address] = [
            violation for violation in self.rate_limit_violations[ip_address] if violation > cutoff
        ]

        # Add current violation
        self.rate_limit_violations[ip_address].append(now)

        # Alert if too many violations (10 in 1 hour)
        if len(self.rate_limit_violations[ip_address]) >= 10:
            self._send_security_alert(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                {
                    "ip_address": ip_address,
                    "violations": len(self.rate_limit_violations[ip_address]),
                    "message": f"Excessive rate limit violations from {ip_address}",
                },
            )

    def _send_security_alert(self, event_type: SecurityEventType, details: Dict[str, Any]) -> None:
        """Send immediate security alert (email, webhook, etc)."""
        # TODO: Implement actual alerting mechanism (email, Slack, PagerDuty, etc)
        critical_logger.critical(
            f"SECURITY ALERT: {event_type} - {details.get('message', 'Security incident detected')}"
        )

        # Log the alert
        self.log_event(
            SecurityEventType.SUSPICIOUS_PATTERN,
            SecurityEventSeverity.CRITICAL,
            f"Security alert triggered for {event_type}",
            details=details,
        )

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of security events for monitoring."""
        if not AuditSessionLocal:
            return {"error": "Audit database not configured"}

        try:
            db = AuditSessionLocal()
            try:
                cutoff = datetime.utcnow() - timedelta(hours=hours)

                # Query recent events
                events = (
                    db.query(SecurityAuditLog).filter(SecurityAuditLog.timestamp >= cutoff).all()
                )

                # Aggregate by type and severity
                summary = {
                    "total_events": len(events),
                    "by_type": {},
                    "by_severity": {},
                    "failed_logins": 0,
                    "suspicious_ips": list(self.suspicious_ips),
                    "top_ips": {},
                }

                for event in events:
                    # Count by type
                    event_type = event.event_type
                    summary["by_type"][event_type] = summary["by_type"].get(event_type, 0) + 1

                    # Count by severity
                    severity = event.severity
                    summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1

                    # Count failed logins
                    if event.event_type == SecurityEventType.LOGIN_FAILURE:
                        summary["failed_logins"] += 1

                    # Track top IPs
                    if event.ip_address:
                        summary["top_ips"][event.ip_address] = (
                            summary["top_ips"].get(event.ip_address, 0) + 1
                        )

                # Sort top IPs
                summary["top_ips"] = dict(
                    sorted(summary["top_ips"].items(), key=lambda x: x[1], reverse=True)[:10]
                )

                return summary

            finally:
                db.close()

        except Exception as e:
            security_logger.error(f"Failed to generate security summary: {e}")
            return {"error": str(e)}


# Global security auditor instance
security_auditor = SecurityAuditor()


# Convenience functions for common logging scenarios
def log_login_success(username: str, user_id: str, request: Request) -> None:
    """Log successful login."""
    security_auditor.log_event(
        SecurityEventType.LOGIN_SUCCESS,
        SecurityEventSeverity.INFO,
        f"User {username} logged in successfully",
        request=request,
        user_id=user_id,
        username=username,
    )


def log_login_failure(username: str, request: Request, reason: str = "Invalid credentials") -> None:
    """Log failed login attempt."""
    security_auditor.log_event(
        SecurityEventType.LOGIN_FAILURE,
        SecurityEventSeverity.WARNING,
        f"Failed login attempt for user {username}: {reason}",
        request=request,
        username=username,
        details={"reason": reason},
    )


def log_access_denied(
    user_id: str, username: str, resource: str, permission: str, request: Request
) -> None:
    """Log access denied event."""
    security_auditor.log_event(
        SecurityEventType.ACCESS_DENIED,
        SecurityEventSeverity.WARNING,
        f"Access denied for user {username} to {resource} (missing {permission})",
        request=request,
        user_id=user_id,
        username=username,
        details={
            "resource": resource,
            "required_permission": permission,
        },
    )


def log_suspicious_activity(
    event_type: SecurityEventType,
    message: str,
    request: Request,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log suspicious activity."""
    security_auditor.log_event(
        event_type,
        SecurityEventSeverity.CRITICAL,
        message,
        request=request,
        details=details,
    )


def log_api_access(
    request: Request,
    response_status: int,
    response_time: float,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
) -> None:
    """Log API access for monitoring."""
    # Only log non-successful responses or slow requests
    if response_status >= 400 or response_time > 1.0:
        severity = (
            SecurityEventSeverity.ERROR
            if response_status >= 500
            else (
                SecurityEventSeverity.WARNING
                if response_status >= 400
                else SecurityEventSeverity.INFO
            )
        )

        security_auditor.log_event(
            SecurityEventType.API_ERROR if response_status >= 400 else SecurityEventType.API_ACCESS,
            severity,
            f"API request to {request.url.path} returned {response_status} in {response_time:.2f}s",
            request=request,
            user_id=user_id,
            username=username,
            status_code=response_status,
            details={
                "response_time": response_time,
                "slow_request": response_time > 1.0,
            },
        )

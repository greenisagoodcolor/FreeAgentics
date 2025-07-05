"""Security monitoring API endpoints.

Provides access to security audit logs and monitoring data.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import Permission, TokenData, get_current_user, require_permission
from auth.security_logging import (
    SecurityAuditLog,
    SecurityEventSeverity,
    SecurityEventType,
    security_auditor,
)
from database.session import get_db

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


# Import json for parsing details
import json

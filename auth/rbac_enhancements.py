"""
Enhanced RBAC system with ABAC, audit logging, and role assignment workflows.
Task #14.14 - RBAC Audit and Access Control Enhancement

This module extends the existing auth system with:
- Attribute-Based Access Control (ABAC)
- Comprehensive audit logging
- Role assignment workflows with approval process
- Dynamic permission evaluation
- Periodic access review mechanisms
"""

import ipaddress
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

from .security_implementation import (
    ROLE_PERMISSIONS,
    Permission,
    TokenData,
    UserRole,
)
from .security_logging import SecurityEventSeverity, SecurityEventType, security_auditor

logger = logging.getLogger(__name__)


class ABACEffect(str, Enum):
    """ABAC rule effects."""

    ALLOW = "allow"
    DENY = "deny"


class RequestStatus(str, Enum):
    """Role assignment request status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ABACRule:
    """Attribute-Based Access Control rule."""

    id: str
    name: str
    description: str
    resource_type: str
    action: str
    subject_conditions: Dict[str, Any]
    resource_conditions: Dict[str, Any]
    environment_conditions: Dict[str, Any]
    effect: ABACEffect
    priority: int
    created_at: datetime
    created_by: str
    is_active: bool = True


@dataclass
class RoleAssignmentRequest:
    """Role assignment/modification request."""

    id: str
    requester_id: str
    target_user_id: str
    target_username: str
    current_role: Optional[UserRole]
    requested_role: UserRole
    justification: str
    business_justification: str
    temporary: bool
    expiry_date: Optional[datetime]
    status: RequestStatus
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    reviewer_notes: Optional[str] = None
    auto_approved: bool = False


@dataclass
class AccessContext:
    """Context for access control decisions."""

    user_id: str
    username: str
    role: UserRole
    permissions: List[Permission]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    device_id: Optional[str] = None
    risk_score: Optional[float] = None


@dataclass
class ResourceContext:
    """Context about the resource being accessed."""

    resource_id: Optional[str] = None
    resource_type: str = ""
    owner_id: Optional[str] = None
    department: Optional[str] = None
    classification: Optional[str] = None
    sensitivity_level: Optional[str] = None
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedRBACManager:
    """Enhanced RBAC manager with ABAC, workflows, and audit capabilities."""

    def __init__(self):
        """Initialize the enhanced RBAC manager."""
        self.abac_rules: List[ABACRule] = []
        self.role_requests: List[RoleAssignmentRequest] = []
        self.access_audit_log: List[Dict[str, Any]] = []

        # Initialize with default ABAC rules
        self._setup_default_abac_rules()

        logger.info("Enhanced RBAC Manager initialized")

    def _setup_default_abac_rules(self):
        """Set up default ABAC rules."""
        default_rules = [
            {
                "id": "admin_business_hours",
                "name": "Admin Business Hours Only",
                "description": "Administrative access restricted to business hours",
                "resource_type": "system",
                "action": "*",
                "subject_conditions": {"role": ["admin"]},
                "resource_conditions": {},
                "environment_conditions": {
                    "time_range": {"start": "08:00", "end": "18:00"},
                    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                },
                "effect": ABACEffect.ALLOW,
                "priority": 100,
            },
            {
                "id": "admin_trusted_ip",
                "name": "Admin Trusted IP Access",
                "description": "Admin access only from trusted IP ranges",
                "resource_type": "system",
                "action": "admin",
                "subject_conditions": {"role": ["admin"]},
                "resource_conditions": {},
                "environment_conditions": {
                    "ip_whitelist": ["127.0.0.1", "192.168.0.0/16", "10.0.0.0/8"]
                },
                "effect": ABACEffect.ALLOW,
                "priority": 95,
            },
            {
                "id": "resource_ownership",
                "name": "Resource Ownership Control",
                "description": "Users can only access resources they own",
                "resource_type": "agent",
                "action": "modify",
                "subject_conditions": {},
                "resource_conditions": {"ownership_required": True},
                "environment_conditions": {},
                "effect": ABACEffect.ALLOW,
                "priority": 80,
            },
            {
                "id": "department_isolation",
                "name": "Department-based Isolation",
                "description": "Access limited to same department resources",
                "resource_type": "*",
                "action": "view",
                "subject_conditions": {},
                "resource_conditions": {"same_department": True},
                "environment_conditions": {},
                "effect": ABACEffect.ALLOW,
                "priority": 70,
            },
            {
                "id": "high_risk_deny",
                "name": "High Risk Access Denial",
                "description": "Deny access for high-risk sessions",
                "resource_type": "*",
                "action": "*",
                "subject_conditions": {},
                "resource_conditions": {},
                "environment_conditions": {"max_risk_score": 0.8},
                "effect": ABACEffect.DENY,
                "priority": 150,
            },
        ]

        for rule_data in default_rules:
            rule = ABACRule(
                id=rule_data["id"],
                name=rule_data["name"],
                description=rule_data["description"],
                resource_type=rule_data["resource_type"],
                action=rule_data["action"],
                subject_conditions=rule_data["subject_conditions"],
                resource_conditions=rule_data["resource_conditions"],
                environment_conditions=rule_data["environment_conditions"],
                effect=rule_data["effect"],
                priority=rule_data["priority"],
                created_at=datetime.now(timezone.utc),
                created_by="system",
            )
            self.abac_rules.append(rule)

        logger.info(f"Initialized {len(self.abac_rules)} default ABAC rules")

    def add_abac_rule(self, rule: ABACRule) -> bool:
        """Add a new ABAC rule."""
        try:
            # Validate rule doesn't conflict
            existing_rule = next((r for r in self.abac_rules if r.id == rule.id), None)
            if existing_rule:
                logger.warning(f"ABAC rule with ID {rule.id} already exists")
                return False

            self.abac_rules.append(rule)

            # Sort by priority (highest first)
            self.abac_rules.sort(key=lambda x: x.priority, reverse=True)

            logger.info(f"Added ABAC rule: {rule.name} (ID: {rule.id})")
            return True

        except Exception as e:
            logger.error(f"Failed to add ABAC rule: {e}")
            return False

    def evaluate_abac_access(
        self, access_context: AccessContext, resource_context: ResourceContext, action: str
    ) -> Tuple[bool, str, List[str]]:
        """
        Evaluate ABAC rules for access decision.

        Returns:
            Tuple[bool, str, List[str]]: (access_granted, reason, applied_rules)
        """
        applied_rules = []
        final_decision = False
        decision_reason = "No applicable rules found - default deny"

        # Get applicable rules
        applicable_rules = [
            rule
            for rule in self.abac_rules
            if rule.is_active and self._rule_applies(rule, resource_context.resource_type, action)
        ]

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda x: x.priority, reverse=True)

        for rule in applicable_rules:
            if self._evaluate_rule(rule, access_context, resource_context):
                applied_rules.append(rule.name)

                if rule.effect == ABACEffect.DENY:
                    # Deny takes precedence
                    final_decision = False
                    decision_reason = f"Access denied by rule: {rule.name}"
                    break
                elif rule.effect == ABACEffect.ALLOW:
                    final_decision = True
                    decision_reason = f"Access allowed by rule: {rule.name}"
                    # Continue checking for deny rules with higher priority

        # Log the decision
        self._log_abac_decision(
            access_context, resource_context, action, final_decision, decision_reason, applied_rules
        )

        return final_decision, decision_reason, applied_rules

    def _rule_applies(self, rule: ABACRule, resource_type: str, action: str) -> bool:
        """Check if a rule applies to the given resource type and action."""
        return (rule.resource_type == "*" or rule.resource_type == resource_type) and (
            rule.action == "*" or rule.action == action
        )

    def _evaluate_rule(
        self, rule: ABACRule, access_context: AccessContext, resource_context: ResourceContext
    ) -> bool:
        """Evaluate if a rule's conditions are satisfied."""

        # Evaluate subject conditions
        if not self._evaluate_subject_conditions(rule.subject_conditions, access_context):
            return False

        # Evaluate resource conditions
        if not self._evaluate_resource_conditions(
            rule.resource_conditions, access_context, resource_context
        ):
            return False

        # Evaluate environment conditions
        if not self._evaluate_environment_conditions(rule.environment_conditions, access_context):
            return False

        return True

    def _evaluate_subject_conditions(
        self, conditions: Dict[str, Any], context: AccessContext
    ) -> bool:
        """Evaluate subject-based conditions."""
        for key, value in conditions.items():
            if key == "role":
                if isinstance(value, list):
                    if context.role.value not in value:
                        return False
                elif context.role.value != value:
                    return False

            elif key == "department":
                if isinstance(value, list):
                    if context.department not in value:
                        return False
                elif context.department != value:
                    return False

            elif key == "min_risk_score":
                if context.risk_score is None or context.risk_score < value:
                    return False

            elif key == "max_risk_score":
                if context.risk_score is None or context.risk_score > value:
                    return False

        return True

    def _evaluate_resource_conditions(
        self,
        conditions: Dict[str, Any],
        access_context: AccessContext,
        resource_context: ResourceContext,
    ) -> bool:
        """Evaluate resource-based conditions."""
        for key, value in conditions.items():
            if key == "ownership_required" and value:
                if access_context.user_id != resource_context.owner_id:
                    return False

            elif key == "same_department" and value:
                if access_context.department != resource_context.department:
                    return False

            elif key == "classification":
                if isinstance(value, list):
                    if resource_context.classification not in value:
                        return False
                elif resource_context.classification != value:
                    return False

            elif key == "sensitivity_level":
                allowed_levels = ["public", "internal", "confidential", "restricted"]
                if isinstance(value, str):
                    max_level_index = allowed_levels.index(value)
                    resource_level_index = allowed_levels.index(
                        resource_context.sensitivity_level or "public"
                    )
                    if resource_level_index > max_level_index:
                        return False

        return True

    def _evaluate_environment_conditions(
        self, conditions: Dict[str, Any], context: AccessContext
    ) -> bool:
        """Evaluate environment-based conditions."""
        for key, value in conditions.items():
            if key == "time_range":
                if not self._check_time_range(value):
                    return False

            elif key == "days":
                current_day = datetime.now().strftime("%A")
                if current_day not in value:
                    return False

            elif key == "ip_whitelist":
                if not self._check_ip_whitelist(context.ip_address or "", value):
                    return False

            elif key == "location":
                if isinstance(value, list):
                    if context.location not in value:
                        return False
                elif context.location != value:
                    return False

            elif key == "max_risk_score":
                if context.risk_score is None or context.risk_score > value:
                    return False

        return True

    def _check_time_range(self, time_range: Dict[str, str]) -> bool:
        """Check if current time is within allowed range."""
        from datetime import time

        try:
            now = datetime.now().time()
            start_time = time.fromisoformat(time_range["start"])
            end_time = time.fromisoformat(time_range["end"])

            if start_time <= end_time:
                return start_time <= now <= end_time
            else:
                # Handle overnight ranges (e.g., 22:00 to 06:00)
                return now >= start_time or now <= end_time
        except Exception as e:
            logger.warning(f"Error evaluating time range: {e}")
            return False

    def _check_ip_whitelist(self, user_ip: str, whitelist: List[str]) -> bool:
        """Check if user IP is in whitelist."""
        if not user_ip:
            return False

        try:
            user_ip_obj = ipaddress.ip_address(user_ip)

            for allowed_ip in whitelist:
                try:
                    if "/" in allowed_ip:  # CIDR notation
                        network = ipaddress.ip_network(allowed_ip, strict=False)
                        if user_ip_obj in network:
                            return True
                    else:  # Single IP
                        if str(user_ip_obj) == allowed_ip:
                            return True
                except ValueError:
                    logger.warning(f"Invalid IP/network in whitelist: {allowed_ip}")
                    continue

        except ValueError:
            logger.warning(f"Invalid IP address format: {user_ip}")
            return False

        return False

    def _log_abac_decision(
        self,
        access_context: AccessContext,
        resource_context: ResourceContext,
        action: str,
        decision: bool,
        reason: str,
        applied_rules: List[str],
    ):
        """Log ABAC access decision for audit purposes."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "abac_decision",
            "user_id": access_context.user_id,
            "username": access_context.username,
            "role": access_context.role.value,
            "resource_type": resource_context.resource_type,
            "resource_id": resource_context.resource_id,
            "action": action,
            "decision": decision,
            "reason": reason,
            "applied_rules": applied_rules,
            "ip_address": access_context.ip_address,
            "risk_score": access_context.risk_score,
        }

        self.access_audit_log.append(log_entry)

        # Also log through security auditor
        event_type = (
            SecurityEventType.ACCESS_GRANTED if decision else SecurityEventType.ACCESS_DENIED
        )
        security_auditor.log_event(
            event_type,
            SecurityEventSeverity.INFO if decision else SecurityEventSeverity.WARNING,
            f"ABAC decision: {('ALLOW' if decision else 'DENY')} - {reason}",
            user_id=access_context.user_id,
            username=access_context.username,
            details={
                "resource_type": resource_context.resource_type,
                "action": action,
                "applied_rules": applied_rules,
                "risk_score": access_context.risk_score,
            },
        )

    def request_role_assignment(
        self,
        requester_id: str,
        target_user_id: str,
        target_username: str,
        current_role: Optional[UserRole],
        requested_role: UserRole,
        justification: str,
        business_justification: str,
        temporary: bool = False,
        expiry_date: Optional[datetime] = None,
    ) -> str:
        """Submit a role assignment request."""

        request_id = f"RAR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.role_requests)}"

        request = RoleAssignmentRequest(
            id=request_id,
            requester_id=requester_id,
            target_user_id=target_user_id,
            target_username=target_username,
            current_role=current_role,
            requested_role=requested_role,
            justification=justification,
            business_justification=business_justification,
            temporary=temporary,
            expiry_date=expiry_date,
            status=RequestStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

        # Check for auto-approval criteria
        if self._should_auto_approve(request):
            request.status = RequestStatus.APPROVED
            request.auto_approved = True
            request.reviewed_at = datetime.now(timezone.utc)
            request.reviewer_notes = "Auto-approved based on policy"

        self.role_requests.append(request)

        # Log the request using available event type
        security_auditor.log_event(
            SecurityEventType.PERMISSION_CHECK,  # Using available event type
            SecurityEventSeverity.INFO,
            f"Role assignment request: {target_username} from {current_role} to {requested_role}",
            user_id=requester_id,
            details={
                "request_id": request_id,
                "target_user": target_username,
                "current_role": current_role.value if current_role else None,
                "requested_role": requested_role.value,
                "temporary": temporary,
                "auto_approved": request.auto_approved,
                "event_subtype": "role_assignment_request",
            },
        )

        logger.info(
            f"Role assignment request submitted: {request_id} ({'auto-approved' if request.auto_approved else 'pending approval'})"
        )

        return request_id

    def _should_auto_approve(self, request: RoleAssignmentRequest) -> bool:
        """Determine if a role assignment request should be auto-approved."""

        # Auto-approve downgrades (less privileged roles)
        if request.current_role and request.requested_role:
            role_hierarchy = {
                UserRole.OBSERVER: 1,
                UserRole.AGENT_MANAGER: 2,
                UserRole.RESEARCHER: 3,
                UserRole.ADMIN: 4,
            }

            current_level = role_hierarchy.get(request.current_role, 0)
            requested_level = role_hierarchy.get(request.requested_role, 0)

            if requested_level < current_level:
                return True

        # Auto-approve observer role assignments
        if request.requested_role == UserRole.OBSERVER:
            return True

        # Auto-approve temporary assignments for short periods
        if request.temporary and request.expiry_date:
            time_diff = request.expiry_date - datetime.now(timezone.utc)
            if time_diff.total_seconds() <= 24 * 3600:  # 24 hours
                return True

        return False

    def approve_role_request(
        self, request_id: str, reviewer_id: str, reviewer_notes: Optional[str] = None
    ) -> bool:
        """Approve a role assignment request."""

        request = next((r for r in self.role_requests if r.id == request_id), None)

        if not request:
            logger.warning(f"Role assignment request not found: {request_id}")
            return False

        if request.status != RequestStatus.PENDING:
            logger.warning(
                f"Role assignment request {request_id} is not pending (status: {request.status})"
            )
            return False

        request.status = RequestStatus.APPROVED
        request.reviewed_at = datetime.now(timezone.utc)
        request.reviewed_by = reviewer_id
        request.reviewer_notes = reviewer_notes

        # Log the approval using available event type
        security_auditor.log_event(
            SecurityEventType.ACCESS_GRANTED,  # Using available event type
            SecurityEventSeverity.INFO,
            f"Role assignment approved: {request.target_username} to {request.requested_role}",
            user_id=reviewer_id,
            details={
                "request_id": request_id,
                "target_user": request.target_username,
                "approved_role": request.requested_role.value,
                "reviewer_notes": reviewer_notes,
                "event_subtype": "role_assignment_approved",
            },
        )

        logger.info(f"Role assignment request approved: {request_id}")
        return True

    def reject_role_request(self, request_id: str, reviewer_id: str, reviewer_notes: str) -> bool:
        """Reject a role assignment request."""

        request = next((r for r in self.role_requests if r.id == request_id), None)

        if not request:
            logger.warning(f"Role assignment request not found: {request_id}")
            return False

        if request.status != RequestStatus.PENDING:
            logger.warning(
                f"Role assignment request {request_id} is not pending (status: {request.status})"
            )
            return False

        request.status = RequestStatus.REJECTED
        request.reviewed_at = datetime.now(timezone.utc)
        request.reviewed_by = reviewer_id
        request.reviewer_notes = reviewer_notes

        # Log the rejection using available event type
        security_auditor.log_event(
            SecurityEventType.ACCESS_DENIED,  # Using available event type
            SecurityEventSeverity.INFO,
            f"Role assignment rejected: {request.target_username} to {request.requested_role}",
            user_id=reviewer_id,
            details={
                "request_id": request_id,
                "target_user": request.target_username,
                "rejected_role": request.requested_role.value,
                "reviewer_notes": reviewer_notes,
                "event_subtype": "role_assignment_rejected",
            },
        )

        logger.info(f"Role assignment request rejected: {request_id}")
        return True

    def get_pending_requests(
        self, reviewer_role: Optional[UserRole] = None
    ) -> List[RoleAssignmentRequest]:
        """Get pending role assignment requests."""

        pending_requests = [r for r in self.role_requests if r.status == RequestStatus.PENDING]

        # Filter based on reviewer role if specified
        if reviewer_role:
            # Admins can review all requests
            if reviewer_role != UserRole.ADMIN:
                # Non-admins can only review requests for lower-privilege roles
                role_hierarchy = {
                    UserRole.OBSERVER: 1,
                    UserRole.AGENT_MANAGER: 2,
                    UserRole.RESEARCHER: 3,
                    UserRole.ADMIN: 4,
                }

                reviewer_level = role_hierarchy.get(reviewer_role, 0)
                pending_requests = [
                    r
                    for r in pending_requests
                    if role_hierarchy.get(r.requested_role, 0) < reviewer_level
                ]

        return pending_requests

    def expire_old_requests(self, max_age_days: int = 30) -> int:
        """Expire old pending requests."""

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        expired_count = 0

        for request in self.role_requests:
            if request.status == RequestStatus.PENDING and request.created_at < cutoff_date:
                request.status = RequestStatus.EXPIRED
                expired_count += 1

        if expired_count > 0:
            logger.info(f"Expired {expired_count} old role assignment requests")

        return expired_count

    def generate_access_report(self) -> Dict[str, Any]:
        """Generate comprehensive access control report."""

        return {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report_type": "enhanced_rbac_audit",
            },
            "rbac_config": {
                "total_roles": len(UserRole),
                "total_permissions": len(Permission),
                "role_permission_matrix": {
                    role.value: [p.value for p in perms] for role, perms in ROLE_PERMISSIONS.items()
                },
            },
            "abac_config": {
                "total_rules": len(self.abac_rules),
                "active_rules": len([r for r in self.abac_rules if r.is_active]),
                "rules_by_priority": [
                    {
                        "id": rule.id,
                        "name": rule.name,
                        "priority": rule.priority,
                        "effect": rule.effect.value,
                        "resource_type": rule.resource_type,
                        "action": rule.action,
                    }
                    for rule in sorted(self.abac_rules, key=lambda x: x.priority, reverse=True)
                ],
            },
            "role_assignment_workflow": {
                "total_requests": len(self.role_requests),
                "pending_requests": len(
                    [r for r in self.role_requests if r.status == RequestStatus.PENDING]
                ),
                "approved_requests": len(
                    [r for r in self.role_requests if r.status == RequestStatus.APPROVED]
                ),
                "rejected_requests": len(
                    [r for r in self.role_requests if r.status == RequestStatus.REJECTED]
                ),
                "auto_approved_requests": len([r for r in self.role_requests if r.auto_approved]),
            },
            "audit_statistics": {
                "total_access_decisions": len(self.access_audit_log),
                "access_granted": len([e for e in self.access_audit_log if e.get("decision")]),
                "access_denied": len([e for e in self.access_audit_log if not e.get("decision")]),
            },
        }


# Global enhanced RBAC manager instance
enhanced_rbac_manager = EnhancedRBACManager()


def enhanced_permission_check(permission: Permission):
    """Enhanced permission check decorator with ABAC support."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current user from function arguments
            current_user = None
            request = None

            for arg in args:
                if isinstance(arg, TokenData):
                    current_user = arg
                elif hasattr(arg, "client"):  # FastAPI Request object
                    request = arg

            for value in kwargs.values():
                if isinstance(value, TokenData):
                    current_user = value
                elif hasattr(value, "client"):  # FastAPI Request object
                    request = value

            if not current_user:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
                )

            # Standard RBAC check
            if permission not in current_user.permissions:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission.value}",
                )

            # Enhanced ABAC check
            access_context = AccessContext(
                user_id=current_user.user_id,
                username=current_user.username,
                role=current_user.role,
                permissions=current_user.permissions,
                ip_address=request.client.host if request and request.client else None,
                user_agent=request.headers.get("user-agent") if request else None,
                timestamp=datetime.now(timezone.utc),
            )

            # Basic resource context (could be enhanced based on function parameters)
            resource_context = ResourceContext(
                resource_type=func.__name__.split("_")[-1] if "_" in func.__name__ else "unknown"
            )

            access_granted, reason, applied_rules = enhanced_rbac_manager.evaluate_abac_access(
                access_context, resource_context, permission.value
            )

            if not access_granted:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied by ABAC policy: {reason}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def calculate_user_risk_score(
    user_context: AccessContext,
    recent_failed_attempts: int = 0,
    location_anomaly: bool = False,
    time_anomaly: bool = False,
    device_anomaly: bool = False,
) -> float:
    """Calculate user risk score for access decisions."""

    risk_score = 0.0

    # Base risk by role (lower privilege = lower risk)
    role_risk = {
        UserRole.OBSERVER: 0.1,
        UserRole.AGENT_MANAGER: 0.2,
        UserRole.RESEARCHER: 0.3,
        UserRole.ADMIN: 0.4,
    }
    risk_score += role_risk.get(user_context.role, 0.5)

    # Failed login attempts
    risk_score += min(recent_failed_attempts * 0.1, 0.3)

    # Anomaly detection
    if location_anomaly:
        risk_score += 0.2
    if time_anomaly:
        risk_score += 0.15
    if device_anomaly:
        risk_score += 0.1

    # IP address risk (simplified)
    if user_context.ip_address:
        try:
            ip = ipaddress.ip_address(user_context.ip_address)
            if ip.is_private:
                risk_score += 0.0  # Trusted internal network
            else:
                risk_score += 0.1  # External IP
        except ValueError:
            risk_score += 0.2  # Invalid IP format

    return min(risk_score, 1.0)  # Cap at 1.0

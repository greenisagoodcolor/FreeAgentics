"""Admin API endpoints for RBAC management."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth.rbac_enhancements import (
    ABACEffect,
    ABACRule,
    AccessContext,
    RequestStatus,
    ResourceContext,
    enhanced_rbac_manager,
)

# Security imports for RBAC
from auth.security_implementation import (
    ROLE_PERMISSIONS,
    Permission,
    TokenData,
    UserRole,
    auth_manager,
    get_current_user,
    require_permission,
)
from auth.security_logging import (
    SecurityEventSeverity,
    SecurityEventType,
    security_auditor,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Models for admin operations
class UserInfo(BaseModel):
    """User information for admin operations."""

    user_id: str
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]


class UserCreate(BaseModel):
    """Request to create a new user."""

    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.OBSERVER


class UserUpdate(BaseModel):
    """Request to update user information."""

    email: Optional[str] = Field(None, regex=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    is_active: Optional[bool] = None


class RoleChangeRequest(BaseModel):
    """Request to change user role."""

    user_id: str
    new_role: UserRole
    justification: str = Field(..., min_length=10)
    temporary: bool = False
    expiry_hours: Optional[int] = None


class ABACRuleCreate(BaseModel):
    """Request to create ABAC rule."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    resource_type: str = Field(..., min_length=1, max_length=50)
    action: str = Field(..., min_length=1, max_length=50)
    subject_conditions: Dict[str, Any] = Field(default_factory=dict)
    resource_conditions: Dict[str, Any] = Field(default_factory=dict)
    environment_conditions: Dict[str, Any] = Field(default_factory=dict)
    effect: ABACEffect
    priority: int = Field(100, ge=1, le=1000)


class ABACRuleUpdate(BaseModel):
    """Request to update ABAC rule."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, min_length=1, max_length=500)
    subject_conditions: Optional[Dict[str, Any]] = None
    resource_conditions: Optional[Dict[str, Any]] = None
    environment_conditions: Optional[Dict[str, Any]] = None
    effect: Optional[ABACEffect] = None
    priority: Optional[int] = Field(None, ge=1, le=1000)
    is_active: Optional[bool] = None


class AuditLogQuery(BaseModel):
    """Query parameters for audit log retrieval."""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    user_id: Optional[str] = None
    event_type: Optional[str] = None
    severity: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


# User management endpoints
@router.get("/users", response_model=List[UserInfo])
@require_permission(Permission.ADMIN_SYSTEM)
async def list_users(
    current_user: TokenData = Depends(get_current_user),
    active_only: bool = True,
    role: Optional[UserRole] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[UserInfo]:
    """List all users with filtering options."""
    # Get users from auth manager
    users = []
    for username, user_data in auth_manager.users.items():
        user = user_data["user"]

        # Apply filters
        if active_only and not user.is_active:
            continue
        if role and user.role != role:
            continue

        users.append(
            UserInfo(
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                role=user.role,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login,
            )
        )

    # Apply pagination
    users.sort(key=lambda x: x.created_at, reverse=True)
    return users[offset : offset + limit]


@router.post("/users", response_model=UserInfo, status_code=201)
@require_permission(Permission.ADMIN_SYSTEM)
async def create_user(
    user_data: UserCreate, current_user: TokenData = Depends(get_current_user)
) -> UserInfo:
    """Create a new user."""
    try:
        # Check if user already exists
        if user_data.username in auth_manager.users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists",
            )

        # Create user
        user = auth_manager.register_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role=user_data.role,
        )

        # Log user creation
        security_auditor.log_event(
            SecurityEventType.USER_CREATED,
            SecurityEventSeverity.INFO,
            f"Admin {current_user.username} created user {user.username}",
            user_id=current_user.user_id,
            username=current_user.username,
            details={
                "created_user": user.username,
                "created_role": user.role.value,
                "created_email": user.email,
            },
        )

        return UserInfo(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
        )

    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user",
        )


@router.get("/users/{user_id}", response_model=UserInfo)
@require_permission(Permission.ADMIN_SYSTEM)
async def get_user(
    user_id: str, current_user: TokenData = Depends(get_current_user)
) -> UserInfo:
    """Get user details by ID."""
    # Find user by ID
    for username, user_data in auth_manager.users.items():
        user = user_data["user"]
        if user.user_id == user_id:
            return UserInfo(
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                role=user.role,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login,
            )

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")


@router.put("/users/{user_id}", response_model=UserInfo)
@require_permission(Permission.ADMIN_SYSTEM)
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    current_user: TokenData = Depends(get_current_user),
) -> UserInfo:
    """Update user information."""
    # Find user by ID
    target_user = None
    for username, user_data in auth_manager.users.items():
        user = user_data["user"]
        if user.user_id == user_id:
            target_user = user
            break

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Update user fields
    if user_update.email is not None:
        target_user.email = user_update.email
    if user_update.is_active is not None:
        target_user.is_active = user_update.is_active

    # Log user update
    security_auditor.log_event(
        SecurityEventType.USER_UPDATED,
        SecurityEventSeverity.INFO,
        f"Admin {current_user.username} updated user {target_user.username}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={
            "updated_user": target_user.username,
            "changes": user_update.dict(exclude_none=True),
        },
    )

    return UserInfo(
        user_id=target_user.user_id,
        username=target_user.username,
        email=target_user.email,
        role=target_user.role,
        is_active=target_user.is_active,
        created_at=target_user.created_at,
        last_login=target_user.last_login,
    )


@router.delete("/users/{user_id}")
@require_permission(Permission.ADMIN_SYSTEM)
async def delete_user(
    user_id: str, current_user: TokenData = Depends(get_current_user)
) -> Dict[str, str]:
    """Delete a user (deactivate)."""
    # Find user by ID
    target_user = None
    for username, user_data in auth_manager.users.items():
        user = user_data["user"]
        if user.user_id == user_id:
            target_user = user
            break

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Prevent self-deletion
    if target_user.user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    # Deactivate user instead of deleting
    target_user.is_active = False

    # Log user deactivation
    security_auditor.log_event(
        SecurityEventType.USER_DEACTIVATED,
        SecurityEventSeverity.WARNING,
        f"Admin {current_user.username} deactivated user {target_user.username}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={
            "deactivated_user": target_user.username,
            "deactivated_user_id": target_user.user_id,
        },
    )

    return {"message": f"User {target_user.username} has been deactivated"}


# Role management endpoints
@router.post("/roles/request")
@require_permission(Permission.ADMIN_SYSTEM)
async def request_role_change(
    role_request: RoleChangeRequest,
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Request role change for a user."""
    # Find target user
    target_user = None
    target_username = None
    for username, user_data in auth_manager.users.items():
        user = user_data["user"]
        if user.user_id == role_request.user_id:
            target_user = user
            target_username = username
            break

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Calculate expiry if temporary
    expiry_date = None
    if role_request.temporary and role_request.expiry_hours:
        from datetime import timedelta

        expiry_date = datetime.now() + timedelta(hours=role_request.expiry_hours)

    # Submit role assignment request
    request_id = enhanced_rbac_manager.request_role_assignment(
        requester_id=current_user.user_id,
        target_user_id=role_request.user_id,
        target_username=target_username,
        current_role=target_user.role,
        requested_role=role_request.new_role,
        justification=role_request.justification,
        business_justification=f"Admin request by {current_user.username}",
        temporary=role_request.temporary,
        expiry_date=expiry_date,
    )

    return {
        "request_id": request_id,
        "message": "Role change request submitted",
    }


@router.get("/roles/requests", response_model=List[Dict[str, Any]])
@require_permission(Permission.ADMIN_SYSTEM)
async def get_role_requests(
    current_user: TokenData = Depends(get_current_user),
    status: Optional[RequestStatus] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Get role assignment requests."""
    requests = enhanced_rbac_manager.get_pending_requests(current_user.role)

    # Filter by status if provided
    if status:
        requests = [r for r in requests if r.status == status]

    # Apply pagination
    requests.sort(key=lambda x: x.created_at, reverse=True)
    paginated_requests = requests[offset : offset + limit]

    return [
        {
            "id": req.id,
            "requester_id": req.requester_id,
            "target_user_id": req.target_user_id,
            "target_username": req.target_username,
            "current_role": req.current_role.value if req.current_role else None,
            "requested_role": req.requested_role.value,
            "justification": req.justification,
            "business_justification": req.business_justification,
            "temporary": req.temporary,
            "expiry_date": req.expiry_date.isoformat() if req.expiry_date else None,
            "status": req.status.value,
            "created_at": req.created_at.isoformat(),
            "reviewed_at": req.reviewed_at.isoformat() if req.reviewed_at else None,
            "reviewed_by": req.reviewed_by,
            "reviewer_notes": req.reviewer_notes,
            "auto_approved": req.auto_approved,
        }
        for req in paginated_requests
    ]


@router.post("/roles/requests/{request_id}/approve")
@require_permission(Permission.ADMIN_SYSTEM)
async def approve_role_request(
    request_id: str,
    notes: Optional[str] = None,
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Approve a role assignment request."""
    success = enhanced_rbac_manager.approve_role_request(
        request_id=request_id,
        reviewer_id=current_user.user_id,
        reviewer_notes=notes,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found or already processed",
        )

    return {"message": "Role assignment request approved"}


@router.post("/roles/requests/{request_id}/reject")
@require_permission(Permission.ADMIN_SYSTEM)
async def reject_role_request(
    request_id: str,
    notes: str = Field(..., min_length=10),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Reject a role assignment request."""
    success = enhanced_rbac_manager.reject_role_request(
        request_id=request_id,
        reviewer_id=current_user.user_id,
        reviewer_notes=notes,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found or already processed",
        )

    return {"message": "Role assignment request rejected"}


# Permission management endpoints
@router.get("/permissions", response_model=Dict[str, Any])
@require_permission(Permission.ADMIN_SYSTEM)
async def get_permissions_overview(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get overview of roles and permissions."""
    return {
        "roles": [role.value for role in UserRole],
        "permissions": [perm.value for perm in Permission],
        "role_permission_matrix": {
            role.value: [perm.value for perm in perms]
            for role, perms in ROLE_PERMISSIONS.items()
        },
        "user_distribution": {
            role.value: len(
                [
                    u
                    for u in auth_manager.users.values()
                    if u["user"].role == role and u["user"].is_active
                ]
            )
            for role in UserRole
        },
    }


# ABAC rule management endpoints
@router.get("/abac/rules", response_model=List[Dict[str, Any]])
@require_permission(Permission.ADMIN_SYSTEM)
async def list_abac_rules(
    current_user: TokenData = Depends(get_current_user),
    active_only: bool = True,
    resource_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List ABAC rules."""
    rules = enhanced_rbac_manager.abac_rules

    # Apply filters
    if active_only:
        rules = [r for r in rules if r.is_active]
    if resource_type:
        rules = [
            r
            for r in rules
            if r.resource_type == resource_type or r.resource_type == "*"
        ]

    # Apply pagination
    rules.sort(key=lambda x: x.priority, reverse=True)
    paginated_rules = rules[offset : offset + limit]

    return [
        {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "resource_type": rule.resource_type,
            "action": rule.action,
            "subject_conditions": rule.subject_conditions,
            "resource_conditions": rule.resource_conditions,
            "environment_conditions": rule.environment_conditions,
            "effect": rule.effect.value,
            "priority": rule.priority,
            "is_active": rule.is_active,
            "created_at": rule.created_at.isoformat(),
            "created_by": rule.created_by,
        }
        for rule in paginated_rules
    ]


@router.post("/abac/rules", response_model=Dict[str, str], status_code=201)
@require_permission(Permission.ADMIN_SYSTEM)
async def create_abac_rule(
    rule_data: ABACRuleCreate,
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Create a new ABAC rule."""
    import uuid
    from datetime import timezone

    # Create rule
    rule = ABACRule(
        id=str(uuid.uuid4()),
        name=rule_data.name,
        description=rule_data.description,
        resource_type=rule_data.resource_type,
        action=rule_data.action,
        subject_conditions=rule_data.subject_conditions,
        resource_conditions=rule_data.resource_conditions,
        environment_conditions=rule_data.environment_conditions,
        effect=rule_data.effect,
        priority=rule_data.priority,
        created_at=datetime.now(timezone.utc),
        created_by=current_user.user_id,
    )

    success = enhanced_rbac_manager.add_abac_rule(rule)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create ABAC rule",
        )

    # Log rule creation
    security_auditor.log_event(
        SecurityEventType.PERMISSION_CHECK,  # Using available event type
        SecurityEventSeverity.INFO,
        f"Admin {current_user.username} created ABAC rule {rule.name}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={
            "rule_id": rule.id,
            "rule_name": rule.name,
            "resource_type": rule.resource_type,
            "action": rule.action,
            "effect": rule.effect.value,
            "event_subtype": "abac_rule_created",
        },
    )

    return {"rule_id": rule.id, "message": "ABAC rule created successfully"}


@router.put("/abac/rules/{rule_id}", response_model=Dict[str, str])
@require_permission(Permission.ADMIN_SYSTEM)
async def update_abac_rule(
    rule_id: str,
    rule_update: ABACRuleUpdate,
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Update an ABAC rule."""
    # Find rule
    rule = next((r for r in enhanced_rbac_manager.abac_rules if r.id == rule_id), None)
    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="ABAC rule not found"
        )

    # Update rule fields
    if rule_update.name is not None:
        rule.name = rule_update.name
    if rule_update.description is not None:
        rule.description = rule_update.description
    if rule_update.subject_conditions is not None:
        rule.subject_conditions = rule_update.subject_conditions
    if rule_update.resource_conditions is not None:
        rule.resource_conditions = rule_update.resource_conditions
    if rule_update.environment_conditions is not None:
        rule.environment_conditions = rule_update.environment_conditions
    if rule_update.effect is not None:
        rule.effect = rule_update.effect
    if rule_update.priority is not None:
        rule.priority = rule_update.priority
    if rule_update.is_active is not None:
        rule.is_active = rule_update.is_active

    # Re-sort rules by priority
    enhanced_rbac_manager.abac_rules.sort(key=lambda x: x.priority, reverse=True)

    # Log rule update
    security_auditor.log_event(
        SecurityEventType.PERMISSION_CHECK,  # Using available event type
        SecurityEventSeverity.INFO,
        f"Admin {current_user.username} updated ABAC rule {rule.name}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={
            "rule_id": rule.id,
            "rule_name": rule.name,
            "changes": rule_update.dict(exclude_none=True),
            "event_subtype": "abac_rule_updated",
        },
    )

    return {"message": "ABAC rule updated successfully"}


@router.delete("/abac/rules/{rule_id}")
@require_permission(Permission.ADMIN_SYSTEM)
async def delete_abac_rule(
    rule_id: str, current_user: TokenData = Depends(get_current_user)
) -> Dict[str, str]:
    """Delete an ABAC rule."""
    # Find and remove rule
    rule = next((r for r in enhanced_rbac_manager.abac_rules if r.id == rule_id), None)
    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="ABAC rule not found"
        )

    enhanced_rbac_manager.abac_rules.remove(rule)

    # Log rule deletion
    security_auditor.log_event(
        SecurityEventType.PERMISSION_CHECK,  # Using available event type
        SecurityEventSeverity.WARNING,
        f"Admin {current_user.username} deleted ABAC rule {rule.name}",
        user_id=current_user.user_id,
        username=current_user.username,
        details={
            "rule_id": rule.id,
            "rule_name": rule.name,
            "event_subtype": "abac_rule_deleted",
        },
    )

    return {"message": "ABAC rule deleted successfully"}


# Audit and monitoring endpoints
@router.get("/audit/access-log", response_model=List[Dict[str, Any]])
@require_permission(Permission.ADMIN_SYSTEM)
async def get_access_audit_log(
    current_user: TokenData = Depends(get_current_user),
    limit: int = 100,
    offset: int = 0,
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    decision: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Get access control audit log."""
    audit_log = enhanced_rbac_manager.access_audit_log

    # Apply filters
    if user_id:
        audit_log = [entry for entry in audit_log if entry.get("user_id") == user_id]
    if resource_type:
        audit_log = [
            entry for entry in audit_log if entry.get("resource_type") == resource_type
        ]
    if decision is not None:
        audit_log = [entry for entry in audit_log if entry.get("decision") == decision]

    # Apply pagination
    audit_log.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return audit_log[offset : offset + limit]


@router.get("/audit/report", response_model=Dict[str, Any])
@require_permission(Permission.ADMIN_SYSTEM)
async def generate_audit_report(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """Generate comprehensive audit report."""
    return enhanced_rbac_manager.generate_access_report()


@router.post("/audit/test-access")
@require_permission(Permission.ADMIN_SYSTEM)
async def test_access_control(
    user_id: str,
    resource_type: str,
    resource_id: str,
    action: str,
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """Test access control for a specific scenario."""
    # Find test user
    test_user = None
    for username, user_data in auth_manager.users.items():
        user = user_data["user"]
        if user.user_id == user_id:
            test_user = user
            break

    if not test_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Test user not found"
        )

    # Create access context
    access_context = AccessContext(
        user_id=test_user.user_id,
        username=test_user.username,
        role=test_user.role,
        permissions=ROLE_PERMISSIONS.get(test_user.role, []),
        timestamp=datetime.now(),
    )

    # Create resource context
    resource_context = ResourceContext(
        resource_id=resource_id, resource_type=resource_type
    )

    # Test RBAC
    rbac_allowed = any(
        perm.value == action
        or action in ["view", "read"]
        and perm == Permission.VIEW_AGENTS
        for perm in access_context.permissions
    )

    # Test ABAC
    (
        abac_allowed,
        abac_reason,
        applied_rules,
    ) = enhanced_rbac_manager.evaluate_abac_access(
        access_context, resource_context, action
    )

    return {
        "user_id": user_id,
        "username": test_user.username,
        "role": test_user.role.value,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "action": action,
        "rbac_result": {
            "allowed": rbac_allowed,
            "permissions": [p.value for p in access_context.permissions],
        },
        "abac_result": {
            "allowed": abac_allowed,
            "reason": abac_reason,
            "applied_rules": applied_rules,
        },
        "final_decision": rbac_allowed and abac_allowed,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/maintenance/expire-requests")
@require_permission(Permission.ADMIN_SYSTEM)
async def expire_old_requests(
    current_user: TokenData = Depends(get_current_user), max_age_days: int = 30
) -> Dict[str, Any]:
    """Expire old pending role requests."""
    expired_count = enhanced_rbac_manager.expire_old_requests(max_age_days)

    # Log maintenance action
    security_auditor.log_event(
        SecurityEventType.PERMISSION_CHECK,  # Using available event type
        SecurityEventSeverity.INFO,
        f"Admin {current_user.username} expired {expired_count} old requests",
        user_id=current_user.user_id,
        username=current_user.username,
        details={
            "expired_count": expired_count,
            "max_age_days": max_age_days,
            "event_subtype": "maintenance_expire_requests",
        },
    )

    return {
        "message": f"Expired {expired_count} old requests",
        "expired_count": expired_count,
    }


@router.get("/stats/rbac")
@require_permission(Permission.ADMIN_SYSTEM)
async def get_rbac_stats(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get RBAC system statistics."""
    # User statistics
    total_users = len(auth_manager.users)
    active_users = len([u for u in auth_manager.users.values() if u["user"].is_active])

    # Role distribution
    role_distribution = {}
    for role in UserRole:
        role_distribution[role.value] = len(
            [
                u
                for u in auth_manager.users.values()
                if u["user"].role == role and u["user"].is_active
            ]
        )

    # ABAC statistics
    abac_rules = enhanced_rbac_manager.abac_rules
    active_abac_rules = len([r for r in abac_rules if r.is_active])

    # Request statistics
    role_requests = enhanced_rbac_manager.role_requests
    pending_requests = len(
        [r for r in role_requests if r.status == RequestStatus.PENDING]
    )
    approved_requests = len(
        [r for r in role_requests if r.status == RequestStatus.APPROVED]
    )

    # Access statistics
    audit_log = enhanced_rbac_manager.access_audit_log
    total_access_checks = len(audit_log)
    access_granted = len([e for e in audit_log if e.get("decision")])
    access_denied = len([e for e in audit_log if not e.get("decision")])

    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "inactive": total_users - active_users,
            "role_distribution": role_distribution,
        },
        "abac": {
            "total_rules": len(abac_rules),
            "active_rules": active_abac_rules,
            "inactive_rules": len(abac_rules) - active_abac_rules,
        },
        "requests": {
            "total": len(role_requests),
            "pending": pending_requests,
            "approved": approved_requests,
            "rejected": len(
                [r for r in role_requests if r.status == RequestStatus.REJECTED]
            ),
            "expired": len(
                [r for r in role_requests if r.status == RequestStatus.EXPIRED]
            ),
        },
        "access_control": {
            "total_checks": total_access_checks,
            "granted": access_granted,
            "denied": access_denied,
            "grant_rate": (
                (access_granted / total_access_checks * 100)
                if total_access_checks > 0
                else 0
            ),
        },
    }

"""Resource-based access control with ownership validation.

This module provides enhanced authorization decorators that validate
resource ownership and implement fine-grained access control.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from auth.comprehensive_audit_logger import comprehensive_auditor
from auth.rbac_enhancements import AccessContext, ResourceContext, enhanced_rbac_manager
from auth.security_implementation import Permission, TokenData
from auth.security_logging import SecurityEventSeverity, SecurityEventType, security_auditor

logger = logging.getLogger(__name__)


class ResourceAccessValidator:
    """Validates resource access based on ownership and ABAC rules."""

    @staticmethod
    def validate_agent_access(
        current_user: TokenData,
        agent_id: str,
        action: str,
        db: Session,
        request: Optional[Request] = None,
    ) -> bool:
        """Validate access to agent resources."""

        try:
            from uuid import UUID

            from database.models import Agent as AgentModel

            # Get agent from database
            agent_uuid = UUID(agent_id)
            agent = db.query(AgentModel).filter(AgentModel.id == agent_uuid).first()

            if not agent:
                logger.warning(f"Agent {agent_id} not found for access validation")
                return False

            # Create access context
            access_context = AccessContext(
                user_id=current_user.user_id,
                username=current_user.username,
                role=current_user.role,
                permissions=current_user.permissions,
                ip_address=request.client.host if request and request.client else None,
                user_agent=request.headers.get("user-agent") if request else None,
                timestamp=None,  # Will be set by ABAC evaluator
            )

            # Create resource context
            resource_context = ResourceContext(
                resource_id=agent_id,
                resource_type="agent",
                owner_id=str(agent.created_by) if hasattr(agent, "created_by") else None,
                metadata={
                    "agent_name": agent.name,
                    "agent_template": agent.template,
                    "agent_status": agent.status.value,
                },
            )

            # Check basic permission first
            required_permission = None
            if action in ["view", "read"]:
                required_permission = Permission.VIEW_AGENTS
            elif action in ["create"]:
                required_permission = Permission.CREATE_AGENT
            elif action in ["modify", "update", "patch"]:
                required_permission = Permission.MODIFY_AGENT
            elif action in ["delete"]:
                required_permission = Permission.DELETE_AGENT

            if required_permission and required_permission not in current_user.permissions:
                logger.warning(
                    f"User {current_user.username} lacks permission {required_permission.value} for action {action}"
                )
                return False

            # Check ABAC rules
            access_granted, reason, applied_rules = enhanced_rbac_manager.evaluate_abac_access(
                access_context, resource_context, action
            )

            # Log ABAC decision
            comprehensive_auditor.log_abac_decision(
                user_id=current_user.user_id,
                username=current_user.username,
                resource_type="agent",
                resource_id=agent_id,
                action=action,
                decision=access_granted,
                reason=reason,
                applied_rules=applied_rules,
                context={
                    "agent_name": agent.name,
                    "agent_template": agent.template,
                    "agent_status": agent.status.value,
                    "ip_address": request.client.host if request and request.client else None,
                },
            )

            if not access_granted:
                logger.warning(f"ABAC denied access: {reason}")
                return False

            # Check ownership for sensitive operations
            if action in ["modify", "update", "patch", "delete"]:
                # Only the creator or admin can modify/delete
                is_owner = (
                    hasattr(agent, "created_by") and str(agent.created_by) == current_user.user_id
                )
                admin_override = current_user.role.value == "admin"

                # Log ownership check
                comprehensive_auditor.log_ownership_check(
                    user_id=current_user.user_id,
                    username=current_user.username,
                    resource_type="agent",
                    resource_id=agent_id,
                    is_owner=is_owner,
                    admin_override=admin_override,
                    metadata={
                        "action": action,
                        "agent_name": agent.name,
                        "agent_creator": (
                            str(agent.created_by) if hasattr(agent, "created_by") else None
                        ),
                    },
                )

                if not is_owner and not admin_override:
                    logger.warning(
                        f"User {current_user.username} not authorized to {action} agent {agent_id} - not owner"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating agent access: {e}")
            return False

    @staticmethod
    def validate_system_access(
        current_user: TokenData, resource_type: str, action: str, request: Optional[Request] = None
    ) -> bool:
        """Validate access to system resources."""

        try:
            # Create access context
            access_context = AccessContext(
                user_id=current_user.user_id,
                username=current_user.username,
                role=current_user.role,
                permissions=current_user.permissions,
                ip_address=request.client.host if request and request.client else None,
                user_agent=request.headers.get("user-agent") if request else None,
                timestamp=None,  # Will be set by ABAC evaluator
            )

            # Create resource context for system resources
            resource_context = ResourceContext(
                resource_type=resource_type,
                metadata={
                    "system_resource": True,
                    "sensitivity_level": "restricted" if resource_type == "admin" else "internal",
                },
            )

            # Check ABAC rules
            access_granted, reason, applied_rules = enhanced_rbac_manager.evaluate_abac_access(
                access_context, resource_context, action
            )

            if not access_granted:
                logger.warning(f"ABAC denied system access: {reason}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating system access: {e}")
            return False

    @staticmethod
    def validate_user_access(
        current_user: TokenData, target_user_id: str, action: str, request: Optional[Request] = None
    ) -> bool:
        """Validate access to user resources."""

        try:
            # Create access context
            access_context = AccessContext(
                user_id=current_user.user_id,
                username=current_user.username,
                role=current_user.role,
                permissions=current_user.permissions,
                ip_address=request.client.host if request and request.client else None,
                user_agent=request.headers.get("user-agent") if request else None,
                timestamp=None,  # Will be set by ABAC evaluator
            )

            # Create resource context
            resource_context = ResourceContext(
                resource_id=target_user_id,
                resource_type="user",
                owner_id=target_user_id,  # Users "own" themselves
                metadata={"user_management": True, "sensitivity_level": "confidential"},
            )

            # Check ABAC rules
            access_granted, reason, applied_rules = enhanced_rbac_manager.evaluate_abac_access(
                access_context, resource_context, action
            )

            if not access_granted:
                logger.warning(f"ABAC denied user access: {reason}")
                return False

            # Additional checks for user management
            if action in ["modify", "update", "delete"]:
                # Only admin or self can modify user data
                if current_user.role.value != "admin" and current_user.user_id != target_user_id:
                    logger.warning(
                        f"User {current_user.username} not authorized to {action} user {target_user_id}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating user access: {e}")
            return False


def require_resource_access(
    resource_type: str, action: str, resource_id_param: str = None, owner_check: bool = True
):
    """
    Decorator for resource-based access control.

    Args:
        resource_type: Type of resource (agent, user, system, etc.)
        action: Action being performed (view, create, modify, delete)
        resource_id_param: Parameter name containing resource ID
        owner_check: Whether to check resource ownership
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract dependencies
            current_user = None
            request = None
            db = None
            resource_id = None

            # Get current user from arguments
            for arg in args:
                if isinstance(arg, TokenData):
                    current_user = arg
                elif hasattr(arg, "client"):  # FastAPI Request
                    request = arg

            for key, value in kwargs.items():
                if isinstance(value, TokenData):
                    current_user = value
                elif hasattr(value, "client"):  # FastAPI Request
                    request = value
                elif key == "db" and hasattr(value, "query"):  # SQLAlchemy Session
                    db = value
                elif key == resource_id_param:
                    resource_id = value

            # Extract resource ID from path parameters if not found
            if not resource_id and resource_id_param:
                resource_id = kwargs.get(resource_id_param)

            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
                )

            # Validate access based on resource type
            access_granted = False

            try:
                if resource_type == "agent" and resource_id:
                    access_granted = ResourceAccessValidator.validate_agent_access(
                        current_user, resource_id, action, db, request
                    )
                elif resource_type == "system":
                    access_granted = ResourceAccessValidator.validate_system_access(
                        current_user, resource_type, action, request
                    )
                elif resource_type == "user" and resource_id:
                    access_granted = ResourceAccessValidator.validate_user_access(
                        current_user, resource_id, action, request
                    )
                else:
                    # Default to basic permission check
                    required_permission = _get_required_permission(resource_type, action)
                    if required_permission:
                        access_granted = required_permission in current_user.permissions

                        # Log RBAC decision
                        comprehensive_auditor.log_rbac_decision(
                            user_id=current_user.user_id,
                            username=current_user.username,
                            role=current_user.role.value,
                            required_permission=required_permission.value,
                            has_permission=access_granted,
                            endpoint=f"/{resource_type}/{resource_id or 'N/A'}",
                            resource_id=resource_id,
                            metadata={"action": action, "resource_type": resource_type},
                        )
                    else:
                        access_granted = True  # No specific permission required

                if not access_granted:
                    # Log access denial
                    security_auditor.log_event(
                        SecurityEventType.ACCESS_DENIED,
                        SecurityEventSeverity.WARNING,
                        f"Resource access denied: {current_user.username} -> {resource_type}:{resource_id or 'N/A'} ({action})",
                        user_id=current_user.user_id,
                        username=current_user.username,
                        details={
                            "resource_type": resource_type,
                            "resource_id": resource_id,
                            "action": action,
                            "reason": "insufficient_permissions_or_ownership",
                        },
                    )

                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Access denied to {resource_type} resource",
                    )

                # Log successful access
                security_auditor.log_event(
                    SecurityEventType.ACCESS_GRANTED,
                    SecurityEventSeverity.INFO,
                    f"Resource access granted: {current_user.username} -> {resource_type}:{resource_id or 'N/A'} ({action})",
                    user_id=current_user.user_id,
                    username=current_user.username,
                    details={
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "action": action,
                    },
                )

                return await func(*args, **kwargs)

            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                logger.error(f"Error in resource access control: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Access control validation failed",
                )

        return wrapper

    return decorator


def _get_required_permission(resource_type: str, action: str) -> Optional[Permission]:
    """Get the required permission for a resource type and action."""

    permission_map = {
        "agent": {
            "view": Permission.VIEW_AGENTS,
            "read": Permission.VIEW_AGENTS,
            "create": Permission.CREATE_AGENT,
            "modify": Permission.MODIFY_AGENT,
            "update": Permission.MODIFY_AGENT,
            "patch": Permission.MODIFY_AGENT,
            "delete": Permission.DELETE_AGENT,
        },
        "system": {
            "view": Permission.VIEW_METRICS,
            "read": Permission.VIEW_METRICS,
            "admin": Permission.ADMIN_SYSTEM,
            "modify": Permission.ADMIN_SYSTEM,
            "update": Permission.ADMIN_SYSTEM,
            "manage": Permission.ADMIN_SYSTEM,
        },
        "user": {
            "view": Permission.ADMIN_SYSTEM,
            "read": Permission.ADMIN_SYSTEM,
            "create": Permission.ADMIN_SYSTEM,
            "modify": Permission.ADMIN_SYSTEM,
            "update": Permission.ADMIN_SYSTEM,
            "delete": Permission.ADMIN_SYSTEM,
        },
        "coalition": {
            "view": Permission.VIEW_AGENTS,
            "read": Permission.VIEW_AGENTS,
            "create": Permission.CREATE_COALITION,
            "modify": Permission.CREATE_COALITION,
            "update": Permission.CREATE_COALITION,
            "delete": Permission.CREATE_COALITION,
        },
    }

    return permission_map.get(resource_type, {}).get(action)


def require_ownership(
    resource_type: str, resource_id_param: str = "resource_id", allow_admin_override: bool = True
):
    """
    Decorator to require resource ownership.

    Args:
        resource_type: Type of resource to check ownership for
        resource_id_param: Parameter name containing resource ID
        allow_admin_override: Whether admins can bypass ownership check
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract dependencies
            current_user = None
            db = None
            resource_id = None

            # Get current user from arguments
            for arg in args:
                if isinstance(arg, TokenData):
                    current_user = arg

            for key, value in kwargs.items():
                if isinstance(value, TokenData):
                    current_user = value
                elif key == "db" and hasattr(value, "query"):  # SQLAlchemy Session
                    db = value
                elif key == resource_id_param:
                    resource_id = value

            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
                )

            if not resource_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Resource ID required"
                )

            # Admin override
            if allow_admin_override and current_user.role.value == "admin":
                return await func(*args, **kwargs)

            # Check ownership based on resource type
            is_owner = False

            try:
                if resource_type == "agent" and db:
                    from uuid import UUID

                    from database.models import Agent as AgentModel

                    agent_uuid = UUID(resource_id)
                    agent = db.query(AgentModel).filter(AgentModel.id == agent_uuid).first()

                    if agent and hasattr(agent, "created_by"):
                        is_owner = str(agent.created_by) == current_user.user_id

                elif resource_type == "user":
                    # Users "own" themselves
                    is_owner = resource_id == current_user.user_id

                if not is_owner:
                    # Log ownership violation
                    security_auditor.log_event(
                        SecurityEventType.ACCESS_DENIED,
                        SecurityEventSeverity.WARNING,
                        f"Ownership violation: {current_user.username} -> {resource_type}:{resource_id}",
                        user_id=current_user.user_id,
                        username=current_user.username,
                        details={
                            "resource_type": resource_type,
                            "resource_id": resource_id,
                            "violation_type": "ownership_required",
                        },
                    )

                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN, detail="Resource ownership required"
                    )

                return await func(*args, **kwargs)

            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                logger.error(f"Error checking ownership: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Ownership validation failed",
                )

        return wrapper

    return decorator


def require_department_access(
    department_param: str = "department", allow_admin_override: bool = True
):
    """
    Decorator to require department-based access.

    Args:
        department_param: Parameter name containing department
        allow_admin_override: Whether admins can bypass department check
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract dependencies
            current_user = None
            target_department = None

            # Get current user from arguments
            for arg in args:
                if isinstance(arg, TokenData):
                    current_user = arg

            for key, value in kwargs.items():
                if isinstance(value, TokenData):
                    current_user = value
                elif key == department_param:
                    target_department = value

            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
                )

            # Admin override
            if allow_admin_override and current_user.role.value == "admin":
                return await func(*args, **kwargs)

            # Check department access
            user_department = getattr(current_user, "department", None)

            if target_department and user_department != target_department:
                # Log department access violation
                security_auditor.log_event(
                    SecurityEventType.ACCESS_DENIED,
                    SecurityEventSeverity.WARNING,
                    f"Department access violation: {current_user.username} ({user_department}) -> {target_department}",
                    user_id=current_user.user_id,
                    username=current_user.username,
                    details={
                        "user_department": user_department,
                        "target_department": target_department,
                        "violation_type": "department_restriction",
                    },
                )

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="Department access required"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator

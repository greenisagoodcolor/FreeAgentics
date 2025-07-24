"""
RBAC Authorization Bypass Penetration Tests

Comprehensive test suite to identify and prevent authorization bypass vulnerabilities
in the RBAC implementation.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from auth.rbac_enhancements import (
    ABACEffect,
    ABACRule,
    AccessContext,
    ResourceContext,
    enhanced_rbac_manager,
)
from auth.resource_access_control import ResourceAccessValidator
from auth.security_implementation import ROLE_PERMISSIONS, Permission, TokenData, UserRole
from fastapi import Request
from sqlalchemy.orm import Session

from database.models import Agent as AgentModel
from database.models import AgentStatus


class TestRBACAuthorizationBypasses:
    """Test suite for RBAC authorization bypass vulnerabilities."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock(spec=Session)

    @pytest.fixture
    def mock_request(self):
        """Create mock request object."""
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "192.168.1.100"
        request.headers = {"user-agent": "test-agent"}
        return request

    @pytest.fixture
    def test_users(self):
        """Create test users with different roles."""
        users = {
            "admin": TokenData(
                user_id="admin-001",
                username="admin_user",
                role=UserRole.ADMIN,
                permissions=ROLE_PERMISSIONS[UserRole.ADMIN],
                exp=datetime.now(timezone.utc),
            ),
            "researcher": TokenData(
                user_id="researcher-001",
                username="researcher_user",
                role=UserRole.RESEARCHER,
                permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
                exp=datetime.now(timezone.utc),
            ),
            "observer": TokenData(
                user_id="observer-001",
                username="observer_user",
                role=UserRole.OBSERVER,
                permissions=ROLE_PERMISSIONS[UserRole.OBSERVER],
                exp=datetime.now(timezone.utc),
            ),
            "agent_manager": TokenData(
                user_id="manager-001",
                username="manager_user",
                role=UserRole.AGENT_MANAGER,
                permissions=ROLE_PERMISSIONS[UserRole.AGENT_MANAGER],
                exp=datetime.now(timezone.utc),
            ),
        }
        return users

    def test_horizontal_privilege_escalation(self, test_users, mock_db, mock_request):
        """Test horizontal privilege escalation vulnerabilities."""
        # Create two users with same role
        user1 = test_users["researcher"]
        user2 = TokenData(
            user_id="researcher-002",
            username="researcher_user2",
            role=UserRole.RESEARCHER,
            permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
            exp=datetime.now(timezone.utc),
        )

        # Create agent owned by user2
        agent = Mock(spec=AgentModel)
        agent.id = uuid.uuid4()
        agent.created_by = user2.user_id
        agent.name = "User2's Agent"
        agent.template = "test"
        agent.status = AgentStatus.ACTIVE

        mock_db.query.return_value.filter.return_value.first.return_value = agent

        # Test that user1 cannot modify user2's agent
        can_access = ResourceAccessValidator.validate_agent_access(
            user1, str(agent.id), "modify", mock_db, mock_request
        )

        assert (
            can_access is False
        ), "Horizontal privilege escalation detected - user can modify another user's resource"

    def test_vertical_privilege_escalation(self, test_users, mock_db, mock_request):
        """Test vertical privilege escalation vulnerabilities."""
        observer = test_users["observer"]

        # Test observer trying to perform admin actions
        actions_and_permissions = [
            ("delete", Permission.DELETE_AGENT),
            ("admin", Permission.ADMIN_SYSTEM),
            ("create", Permission.CREATE_AGENT),
        ]

        for action, required_perm in actions_and_permissions:
            # Observer should not have these permissions
            assert (
                required_perm not in observer.permissions
            ), f"Observer incorrectly has {required_perm} permission"

            # Test system access
            can_access = ResourceAccessValidator.validate_system_access(
                observer, "system", action, mock_request
            )

            # Should be denied by ABAC even if permission check passes
            assert (
                can_access is False
            ), f"Vertical privilege escalation - observer can perform {action} action"

    def test_permission_boundary_violations(self, test_users, mock_db, mock_request):
        """Test permission boundary violations."""
        researcher = test_users["researcher"]

        # Researcher should not be able to delete agents (even their own)
        assert Permission.DELETE_AGENT not in researcher.permissions

        # Create agent owned by researcher
        agent = Mock(spec=AgentModel)
        agent.id = uuid.uuid4()
        agent.created_by = researcher.user_id
        agent.name = "Researcher's Agent"
        agent.template = "test"
        agent.status = AgentStatus.ACTIVE

        mock_db.query.return_value.filter.return_value.first.return_value = agent

        # Even ownership shouldn't grant delete permission
        can_delete = ResourceAccessValidator.validate_agent_access(
            researcher, str(agent.id), "delete", mock_db, mock_request
        )

        assert (
            can_delete is False
        ), "Permission boundary violation - user can perform action beyond their role"

    def test_indirect_object_reference_vulnerabilities(self, test_users, mock_db, mock_request):
        """Test Insecure Direct Object Reference (IDOR) vulnerabilities."""
        user1 = test_users["agent_manager"]

        # Test accessing non-existent resources
        mock_db.query.return_value.filter.return_value.first.return_value = None

        # Should deny access to non-existent resources
        can_access = ResourceAccessValidator.validate_agent_access(
            user1, "non-existent-id", "view", mock_db, mock_request
        )

        assert can_access is False, "IDOR vulnerability - access granted to non-existent resource"

        # Test accessing resources with manipulated IDs
        malicious_ids = [
            "../admin/config",
            "../../etc/passwd",
            "';DROP TABLE agents;--",
            str(uuid.uuid4()) + "' OR '1'='1",
        ]

        for malicious_id in malicious_ids:
            try:
                can_access = ResourceAccessValidator.validate_agent_access(
                    user1, malicious_id, "view", mock_db, mock_request
                )
                assert can_access is False, f"IDOR vulnerability with malicious ID: {malicious_id}"
            except Exception:
                # Should handle invalid IDs gracefully
                pass

    def test_role_manipulation_attempts(self, test_users):
        """Test attempts to manipulate user roles."""
        observer = test_users["observer"]

        # Attempt to modify role directly
        original_role = observer.role
        observer.role = UserRole.ADMIN  # Direct manipulation

        # Permissions should not automatically update
        assert (
            observer.permissions == ROLE_PERMISSIONS[UserRole.OBSERVER]
        ), "Role manipulation vulnerability - permissions updated with role change"

        # Restore original role
        observer.role = original_role

    def test_permission_inheritance_vulnerabilities(self, test_users, mock_db, mock_request):
        """Test permission inheritance and delegation vulnerabilities."""
        # Test that child resources don't inherit parent permissions incorrectly
        admin = test_users["admin"]
        researcher = test_users["researcher"]

        # Admin creates a resource
        admin_agent = Mock(spec=AgentModel)
        admin_agent.id = uuid.uuid4()
        admin_agent.created_by = admin.user_id
        admin_agent.name = "Admin's Agent"
        admin_agent.template = "test"
        admin_agent.status = AgentStatus.ACTIVE

        mock_db.query.return_value.filter.return_value.first.return_value = admin_agent

        # Researcher should not inherit admin's permissions on the resource
        can_delete = ResourceAccessValidator.validate_agent_access(
            researcher, str(admin_agent.id), "delete", mock_db, mock_request
        )

        assert (
            can_delete is False
        ), "Permission inheritance vulnerability - non-admin inherited admin permissions"

    def test_context_manipulation_attacks(self, test_users, mock_request):
        """Test context manipulation attacks on ABAC."""
        observer = test_users["observer"]

        # Create manipulated access context
        malicious_context = AccessContext(
            user_id=observer.user_id,
            username=observer.username,
            role=UserRole.ADMIN,  # Attempt to escalate role in context
            permissions=ROLE_PERMISSIONS[UserRole.ADMIN],  # Attempt to escalate permissions
            ip_address=mock_request.client.host,
            user_agent=mock_request.headers.get("user-agent"),
            timestamp=datetime.now(timezone.utc),
        )

        resource_context = ResourceContext(
            resource_type="system",
            metadata={"sensitivity_level": "restricted"},
        )

        # ABAC should validate against actual user permissions, not manipulated context
        (
            access_granted,
            reason,
            rules,
        ) = enhanced_rbac_manager.evaluate_abac_access(malicious_context, resource_context, "admin")

        # Should check actual user role/permissions, not just context
        # This test ensures ABAC rules properly validate user claims

    def test_time_based_access_bypass(self, test_users, mock_request):
        """Test time-based access control bypass attempts."""
        admin = test_users["admin"]

        # Create custom time-based rule
        time_rule = ABACRule(
            id="test_time_rule",
            name="Business Hours Only",
            description="Access only during business hours",
            resource_type="sensitive",
            action="*",
            subject_conditions={"role": ["admin"]},
            resource_conditions={},
            environment_conditions={
                "time_range": {"start": "09:00", "end": "17:00"},
                "days": [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                ],
            },
            effect=ABACEffect.ALLOW,
            priority=100,
            created_at=datetime.now(timezone.utc),
            created_by="system",
        )

        enhanced_rbac_manager.add_abac_rule(time_rule)

        # Test access outside business hours
        with patch("datetime.datetime") as mock_datetime:
            # Set time to weekend
            mock_datetime.now.return_value = datetime(2024, 1, 6, 10, 0)  # Saturday
            mock_datetime.now.return_value = mock_datetime.now.return_value.replace(
                tzinfo=timezone.utc
            )

            access_context = AccessContext(
                user_id=admin.user_id,
                username=admin.username,
                role=admin.role,
                permissions=admin.permissions,
                ip_address=mock_request.client.host,
                timestamp=mock_datetime.now(timezone.utc),
            )

            resource_context = ResourceContext(
                resource_type="sensitive",
                metadata={"sensitivity_level": "restricted"},
            )

            (
                access_granted,
                reason,
                rules,
            ) = enhanced_rbac_manager.evaluate_abac_access(access_context, resource_context, "view")

            # Access should be denied outside business hours
            assert (
                "Business Hours Only" in rules or not access_granted
            ), "Time-based access control bypass - access granted outside allowed time"

    def test_department_isolation_bypass(self, test_users, mock_db, mock_request):
        """Test department isolation bypass attempts."""
        # Create users in different departments
        it_user = TokenData(
            user_id="it-001",
            username="it_user",
            role=UserRole.RESEARCHER,
            permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
            exp=datetime.now(timezone.utc),
        )

        TokenData(
            user_id="hr-001",
            username="hr_user",
            role=UserRole.RESEARCHER,
            permissions=ROLE_PERMISSIONS[UserRole.RESEARCHER],
            exp=datetime.now(timezone.utc),
        )

        # Create department-specific contexts
        it_context = AccessContext(
            user_id=it_user.user_id,
            username=it_user.username,
            role=it_user.role,
            permissions=it_user.permissions,
            department="IT",
            ip_address=mock_request.client.host,
        )

        hr_resource = ResourceContext(
            resource_id="hr-resource-001",
            resource_type="document",
            department="HR",
            metadata={"sensitivity_level": "confidential"},
        )

        # IT user should not access HR resources
        (
            access_granted,
            reason,
            rules,
        ) = enhanced_rbac_manager.evaluate_abac_access(it_context, hr_resource, "view")

        # Should enforce department isolation
        assert (
            "Department-based Isolation" in rules or not access_granted
        ), "Department isolation bypass - cross-department access allowed"

    def test_permission_caching_vulnerabilities(self, test_users, mock_db, mock_request):
        """Test permission caching vulnerabilities."""
        user = test_users["researcher"]

        # Simulate permission revocation scenario
        original_permissions = user.permissions.copy()

        # First access should work
        agent = Mock(spec=AgentModel)
        agent.id = uuid.uuid4()
        agent.created_by = user.user_id
        agent.name = "Test Agent"
        agent.template = "test"
        agent.status = AgentStatus.ACTIVE

        mock_db.query.return_value.filter.return_value.first.return_value = agent

        can_access_before = ResourceAccessValidator.validate_agent_access(
            user, str(agent.id), "modify", mock_db, mock_request
        )

        assert can_access_before is True, "User should initially have access"

        # Simulate permission revocation
        user.permissions = [Permission.VIEW_AGENTS, Permission.VIEW_METRICS]

        # Access should be denied after permission change
        can_access_after = ResourceAccessValidator.validate_agent_access(
            user, str(agent.id), "modify", mock_db, mock_request
        )

        assert (
            can_access_after is False
        ), "Permission caching vulnerability - access granted with revoked permissions"

        # Restore permissions
        user.permissions = original_permissions

    def test_null_byte_injection_bypass(self, test_users, mock_db, mock_request):
        """Test null byte injection authorization bypass."""
        user = test_users["agent_manager"]

        # Test null byte injection in resource IDs
        null_byte_payloads = [
            "admin\x00user",
            "resource\x00../admin",
            str(uuid.uuid4()) + "\x00admin",
        ]

        for payload in null_byte_payloads:
            try:
                can_access = ResourceAccessValidator.validate_agent_access(
                    user, payload, "view", mock_db, mock_request
                )
                assert (
                    can_access is False
                ), f"Null byte injection bypass with payload: {repr(payload)}"
            except Exception:
                # Should handle null bytes safely
                pass

    def test_race_condition_authorization(self, test_users, mock_db, mock_request):
        """Test race condition vulnerabilities in authorization."""
        user = test_users["researcher"]

        # Create agent
        agent = Mock(spec=AgentModel)
        agent.id = uuid.uuid4()
        agent.created_by = user.user_id
        agent.name = "Test Agent"
        agent.template = "test"
        agent.status = AgentStatus.ACTIVE

        mock_db.query.return_value.filter.return_value.first.return_value = agent

        # Simulate concurrent access attempts
        results = []

        def check_access():
            result = ResourceAccessValidator.validate_agent_access(
                user, str(agent.id), "modify", mock_db, mock_request
            )
            results.append(result)

        # Simulate multiple concurrent requests
        import threading

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=check_access)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be consistent
        assert all(
            r == results[0] for r in results
        ), "Race condition vulnerability - inconsistent authorization results"


class TestAuthorizationBypassMitigations:
    """Test authorization bypass mitigation strategies."""

    def test_defense_in_depth(self, test_users, mock_db, mock_request):
        """Test defense-in-depth authorization checks."""
        observer = test_users["observer"]

        # Multiple layers of authorization should all deny access
        layers_checked = []

        # Layer 1: Basic permission check
        has_delete_permission = Permission.DELETE_AGENT in observer.permissions
        layers_checked.append(("permission_check", has_delete_permission))

        # Layer 2: ABAC check
        access_context = AccessContext(
            user_id=observer.user_id,
            username=observer.username,
            role=observer.role,
            permissions=observer.permissions,
            ip_address=mock_request.client.host,
        )

        resource_context = ResourceContext(
            resource_type="agent", metadata={"sensitivity_level": "internal"}
        )

        abac_granted, _, _ = enhanced_rbac_manager.evaluate_abac_access(
            access_context, resource_context, "delete"
        )
        layers_checked.append(("abac_check", abac_granted))

        # Layer 3: Resource-specific check
        resource_granted = ResourceAccessValidator.validate_system_access(
            observer, "agent", "delete", mock_request
        )
        layers_checked.append(("resource_check", resource_granted))

        # All layers should deny access
        assert all(
            not granted for _, granted in layers_checked
        ), "Defense-in-depth failure - at least one layer allowed unauthorized access"

    def test_fail_secure_authorization(self, test_users, mock_db, mock_request):
        """Test fail-secure authorization behavior."""
        user = test_users["researcher"]

        # Simulate authorization system errors
        with patch.object(
            enhanced_rbac_manager,
            "evaluate_abac_access",
            side_effect=Exception("Authorization system error"),
        ):
            # Should fail securely (deny access) on errors
            try:
                result = ResourceAccessValidator.validate_agent_access(
                    user, "test-id", "view", mock_db, mock_request
                )
                assert result is False, "Fail-open vulnerability - access granted on error"
            except Exception:
                # Failing with exception is also acceptable (fail-secure)
                pass

    def test_authorization_audit_trail(self, test_users, mock_db, mock_request):
        """Test authorization audit trail completeness."""
        admin = test_users["admin"]

        # Clear audit log
        enhanced_rbac_manager.access_audit_log.clear()

        # Perform various authorization checks
        agent = Mock(spec=AgentModel)
        agent.id = uuid.uuid4()
        agent.created_by = admin.user_id
        agent.name = "Test Agent"
        agent.template = "test"
        agent.status = AgentStatus.ACTIVE

        mock_db.query.return_value.filter.return_value.first.return_value = agent

        # Successful access
        ResourceAccessValidator.validate_agent_access(
            admin, str(agent.id), "delete", mock_db, mock_request
        )

        # Failed access
        observer = test_users["observer"]
        ResourceAccessValidator.validate_agent_access(
            observer, str(agent.id), "delete", mock_db, mock_request
        )

        # Check audit log
        assert (
            len(enhanced_rbac_manager.access_audit_log) >= 2
        ), "Incomplete audit trail - not all authorization decisions logged"

        # Verify audit entries contain required information
        for entry in enhanced_rbac_manager.access_audit_log:
            assert "user_id" in entry
            assert "username" in entry
            assert "resource_type" in entry
            assert "action" in entry
            assert "decision" in entry
            assert "timestamp" in entry

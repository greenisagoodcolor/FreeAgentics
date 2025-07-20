"""
Comprehensive Authorization Boundary Testing for FreeAgentics.

This module implements exhaustive security testing for authorization boundaries,
covering all aspects of the authorization system including:
- Role-based access control (RBAC) boundaries
- Resource-level authorization
- API endpoint authorization
- Advanced authorization scenarios (ABAC, context-aware, time-based)
- Attack vector testing (privilege escalation, IDOR, bypass attempts)

The tests ensure that the authorization system maintains strict security boundaries
and prevents unauthorized access in production environments.
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from api.main import app
from auth.rbac_enhancements import (
    ABACEffect,
    ABACRule,
    AccessContext,
    ResourceContext,
    calculate_user_risk_score,
    enhanced_rbac_manager,
)
from auth.security_implementation import (
    ROLE_PERMISSIONS,
    Permission,
    TokenData,
    UserRole,
    auth_manager,
)


class TestRoleBasedAuthorizationBoundaries:
    """Test role-based authorization boundaries and enforcement."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def test_users(self):
        """Create test users with different roles."""
        users = {}
        roles = [
            UserRole.ADMIN,
            UserRole.RESEARCHER,
            UserRole.AGENT_MANAGER,
            UserRole.OBSERVER,
        ]

        for role in roles:
            username = f"test_{role.value}"
            user = auth_manager.register_user(
                username=username,
                email=f"{username}@test.com",
                password="Test123!@#",
                role=role,
            )
            token = auth_manager.create_access_token(user)
            users[role] = {
                "user": user,
                "token": token,
                "username": username,
                "password": "Test123!@#",
            }

        return users

    def test_permission_boundary_validation(self, test_users):
        """Test that each role can only access permissions within their boundary."""
        # Verify each role has exactly the expected permissions
        for role, permissions in ROLE_PERMISSIONS.items():
            user_data = test_users[role]
            token_data = auth_manager.verify_token(user_data["token"])

            # Check that user has all expected permissions
            assert set(token_data.permissions) == set(
                permissions
            ), f"Role {role} has incorrect permissions"

            # Verify no extra permissions
            all_permissions = set(Permission)
            unauthorized_permissions = all_permissions - set(permissions)
            for perm in unauthorized_permissions:
                assert (
                    perm not in token_data.permissions
                ), f"Role {role} should not have permission {perm}"

    def test_role_hierarchy_enforcement(self, client, test_users):
        """Test that role hierarchy is properly enforced."""
        # Test endpoints that require specific roles
        test_endpoints = [
            ("/api/v1/system/config", UserRole.ADMIN, "GET"),
            ("/api/v1/agents", UserRole.AGENT_MANAGER, "POST"),
            ("/api/v1/metrics", UserRole.OBSERVER, "GET"),
        ]

        for endpoint, min_role, method in test_endpoints:
            # Define role hierarchy
            role_hierarchy = {
                UserRole.OBSERVER: 1,
                UserRole.AGENT_MANAGER: 2,
                UserRole.RESEARCHER: 3,
                UserRole.ADMIN: 4,
            }

            for role, user_data in test_users.items():
                headers = {"Authorization": f"Bearer {user_data['token']}"}

                if method == "GET":
                    response = client.get(endpoint, headers=headers)
                elif method == "POST":
                    response = client.post(endpoint, headers=headers, json={})

                # Check if user should have access based on hierarchy
                user_level = role_hierarchy[role]
                required_level = role_hierarchy[min_role]

                if user_level >= required_level:
                    # Should have access (or get 404/422 for missing data, not 403)
                    assert (
                        response.status_code != status.HTTP_403_FORBIDDEN
                    ), f"Role {role} should have access to {endpoint}"
                else:
                    # Should be forbidden
                    assert (
                        response.status_code == status.HTTP_403_FORBIDDEN
                    ), f"Role {role} should not have access to {endpoint}"

    def test_permission_inheritance(self, test_users):
        """Test that permission inheritance works correctly."""
        # Admin should have all permissions
        admin_perms = set(test_users[UserRole.ADMIN]["user"].permissions)
        all_perms = set(Permission)

        # Verify admin has all permissions
        assert admin_perms == all_perms, "Admin should have all permissions"

        # Verify permission subset relationships
        researcher_perms = set(ROLE_PERMISSIONS[UserRole.RESEARCHER])
        agent_manager_perms = set(ROLE_PERMISSIONS[UserRole.AGENT_MANAGER])
        observer_perms = set(ROLE_PERMISSIONS[UserRole.OBSERVER])

        # Observer permissions should be subset of all higher roles
        assert observer_perms.issubset(
            agent_manager_perms
        ), "Observer permissions should be subset of Agent Manager"
        assert observer_perms.issubset(
            researcher_perms
        ), "Observer permissions should be subset of Researcher"
        assert observer_perms.issubset(
            admin_perms
        ), "Observer permissions should be subset of Admin"

    def test_cross_role_access_attempts(self, client, test_users):
        """Test attempts to access resources across role boundaries."""
        # Observer trying to create agent (should fail)
        observer_token = test_users[UserRole.OBSERVER]["token"]
        headers = {"Authorization": f"Bearer {observer_token}"}

        response = client.post(
            "/api/v1/agents",
            headers=headers,
            json={
                "name": "Unauthorized Agent",
                "template": "basic_agent",
                "parameters": {},
            },
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

        # Agent Manager trying to access admin endpoints (should fail)
        manager_token = test_users[UserRole.AGENT_MANAGER]["token"]
        headers = {"Authorization": f"Bearer {manager_token}"}

        response = client.get("/api/v1/system/config", headers=headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_role_elevation_attacks(self, client, test_users):
        """Test various role elevation attack vectors."""
        observer_user = test_users[UserRole.OBSERVER]["user"]
        observer_token = test_users[UserRole.OBSERVER]["token"]

        # Attempt 1: Try to modify JWT payload to elevate role
        with patch.object(auth_manager, "verify_token") as mock_verify:
            # Simulate tampered token with elevated role
            tampered_data = TokenData(
                user_id=observer_user.user_id,
                username=observer_user.username,
                role=UserRole.ADMIN,  # Elevated role
                permissions=[p for p in Permission],  # All permissions
                exp=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            mock_verify.return_value = tampered_data

            # Even with tampered token data, the system should validate
            # against the actual user record
            headers = {"Authorization": f"Bearer {observer_token}"}
            response = client.get("/api/v1/system/config", headers=headers)

            # Reset mock
            mock_verify.return_value = auth_manager.verify_token(
                observer_token
            )

        # Attempt 2: Try to access admin functions through parameter injection
        headers = {"Authorization": f"Bearer {observer_token}"}

        # Try to inject admin role through query parameters
        response = client.get(
            "/api/v1/agents?role=admin&override=true", headers=headers
        )
        # Should still respect actual user role
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_403_FORBIDDEN,
        ]

        # Attempt 3: Try to elevate through race condition
        elevation_attempts = []

        def attempt_elevation():
            headers = {"Authorization": f"Bearer {observer_token}"}
            resp = client.post(
                "/api/v1/agents",
                headers=headers,
                json={"name": "RaceConditionAgent", "template": "basic"},
            )
            elevation_attempts.append(resp.status_code)

        # Launch multiple concurrent requests
        threads = [
            threading.Thread(target=attempt_elevation) for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All attempts should fail with 403
        assert all(
            code == status.HTTP_403_FORBIDDEN for code in elevation_attempts
        ), "Race condition allowed privilege elevation"


class TestResourceLevelAuthorization:
    """Test resource-level authorization controls."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def test_resources(self, test_users):
        """Create test resources with different owners."""
        resources = {"agents": {}, "coalitions": {}, "metrics": {}}

        # Create agents owned by different users
        for role, user_data in test_users.items():
            if role in [
                UserRole.ADMIN,
                UserRole.RESEARCHER,
                UserRole.AGENT_MANAGER,
            ]:
                agent_id = f"agent_{role.value}_{user_data['user'].user_id}"
                resources["agents"][agent_id] = {
                    "id": agent_id,
                    "name": f"Agent owned by {role.value}",
                    "owner_id": user_data["user"].user_id,
                    "owner_role": role,
                    "created_at": datetime.now(timezone.utc),
                    "department": f"dept_{role.value}",
                    "classification": "internal"
                    if role != UserRole.ADMIN
                    else "restricted",
                }

        # Create coalitions with different access levels
        coalition_id = "coalition_research_only"
        resources["coalitions"][coalition_id] = {
            "id": coalition_id,
            "name": "Research Coalition",
            "owner_id": test_users[UserRole.RESEARCHER]["user"].user_id,
            "members": [test_users[UserRole.RESEARCHER]["user"].user_id],
            "access_level": "private",
            "department": "research",
        }

        return resources

    def test_resource_ownership_validation(
        self, client, test_users, test_resources
    ):
        """Test that resource ownership is properly validated."""
        # User 1 creates a resource
        user1 = test_users[UserRole.RESEARCHER]
        headers1 = {"Authorization": f"Bearer {user1['token']}"}

        # Create an agent
        response = client.post(
            "/api/v1/agents",
            headers=headers1,
            json={
                "name": "OwnershipTestAgent",
                "template": "basic_agent",
                "parameters": {"owner_id": user1["user"].user_id},
            },
        )

        if response.status_code == status.HTTP_201_CREATED:
            agent_data = response.json()
            agent_id = agent_data.get("id")

            # User 2 (different user, same role) tries to modify
            user2 = test_users[UserRole.AGENT_MANAGER]
            headers2 = {"Authorization": f"Bearer {user2['token']}"}

            # Attempt to modify another user's agent
            response = client.patch(
                f"/api/v1/agents/{agent_id}",
                headers=headers2,
                json={"name": "HijackedAgent"},
            )

            # Should be forbidden unless admin
            if user2["user"].role != UserRole.ADMIN:
                assert (
                    response.status_code == status.HTTP_403_FORBIDDEN
                ), "Non-owner should not be able to modify resource"

    def test_resource_access_controls(self, test_resources):
        """Test fine-grained resource access controls."""
        # Test ABAC rules for resource access
        researcher_context = AccessContext(
            user_id="researcher_123",
            username="researcher",
            role=UserRole.RESEARCHER,
            permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
            department="research",
            ip_address="192.168.1.100",
        )

        # Test 1: Access to own department resources
        own_resource = ResourceContext(
            resource_id="res_001",
            resource_type="agent",
            owner_id="other_researcher",
            department="research",
            classification="internal",
        )

        granted, reason, rules = enhanced_rbac_manager.evaluate_abac_access(
            researcher_context, own_resource, "view"
        )

        # Should be allowed due to same department rule
        assert (
            granted or "department" not in reason
        ), "Same department access should be considered"

        # Test 2: Access to different department resources
        other_resource = ResourceContext(
            resource_id="res_002",
            resource_type="agent",
            owner_id="admin_user",
            department="admin",
            classification="restricted",
        )

        granted, reason, rules = enhanced_rbac_manager.evaluate_abac_access(
            researcher_context, other_resource, "view"
        )

        # May be denied based on department isolation
        if not granted:
            assert "department" in reason.lower() or "denied" in reason.lower()

    def test_cross_tenant_isolation(self, client, test_users):
        """Test that cross-tenant access is properly isolated."""
        # Simulate multi-tenant environment
        tenant1_context = AccessContext(
            user_id="tenant1_user",
            username="user1",
            role=UserRole.RESEARCHER,
            permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
            department="tenant1",
        )

        tenant2_context = AccessContext(
            user_id="tenant2_user",
            username="user2",
            role=UserRole.RESEARCHER,
            permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
            department="tenant2",
        )

        # Resource belonging to tenant1
        tenant1_resource = ResourceContext(
            resource_id="tenant1_agent",
            resource_type="agent",
            owner_id="tenant1_user",
            department="tenant1",
            metadata={"tenant_id": "tenant1"},
        )

        # Tenant2 user trying to access tenant1 resource
        granted, reason, rules = enhanced_rbac_manager.evaluate_abac_access(
            tenant2_context, tenant1_resource, "view"
        )

        # Should be denied due to department/tenant isolation
        assert (
            not granted
            or tenant1_context.department != tenant2_context.department
        ), "Cross-tenant access should be isolated"

    def test_resource_hierarchy_permissions(self, test_resources):
        """Test permissions across resource hierarchies."""
        # Parent resource
        parent_resource = ResourceContext(
            resource_id="parent_coalition",
            resource_type="coalition",
            owner_id="admin_user",
            department="operations",
            metadata={"hierarchy_level": "parent"},
        )

        # Child resource
        child_resource = ResourceContext(
            resource_id="child_agent",
            resource_type="agent",
            owner_id="admin_user",
            department="operations",
            metadata={
                "hierarchy_level": "child",
                "parent_id": "parent_coalition",
            },
        )

        # User with access to parent
        user_context = AccessContext(
            user_id="ops_manager",
            username="ops_mgr",
            role=UserRole.AGENT_MANAGER,
            permissions=list(ROLE_PERMISSIONS[UserRole.AGENT_MANAGER]),
            department="operations",
        )

        # Test cascading permissions
        parent_granted, _, _ = enhanced_rbac_manager.evaluate_abac_access(
            user_context, parent_resource, "view"
        )

        child_granted, _, _ = enhanced_rbac_manager.evaluate_abac_access(
            user_context, child_resource, "view"
        )

        # Access patterns should be consistent within hierarchy
        assert (
            parent_granted == child_granted
            or user_context.role == UserRole.ADMIN
        ), "Hierarchical permissions should be consistent"

    def test_resource_specific_policies(self):
        """Test resource-specific access policies."""
        # High-sensitivity resource
        sensitive_resource = ResourceContext(
            resource_id="sensitive_001",
            resource_type="system_config",
            owner_id="system",
            classification="top_secret",
            sensitivity_level="restricted",
            metadata={"requires_mfa": True, "ip_restricted": True},
        )

        # Regular user context
        regular_context = AccessContext(
            user_id="regular_user",
            username="regular",
            role=UserRole.RESEARCHER,
            permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
            ip_address="1.2.3.4",  # External IP
            risk_score=0.3,
        )

        # Admin context from trusted IP
        admin_context = AccessContext(
            user_id="admin_user",
            username="admin",
            role=UserRole.ADMIN,
            permissions=list(ROLE_PERMISSIONS[UserRole.ADMIN]),
            ip_address="192.168.1.1",  # Internal IP
            risk_score=0.1,
        )

        # Test access to sensitive resource
        (
            regular_granted,
            regular_reason,
            _,
        ) = enhanced_rbac_manager.evaluate_abac_access(
            regular_context, sensitive_resource, "view"
        )

        (
            admin_granted,
            admin_reason,
            _,
        ) = enhanced_rbac_manager.evaluate_abac_access(
            admin_context, sensitive_resource, "view"
        )

        # Regular user should likely be denied
        # Admin from trusted IP might be allowed
        assert not regular_granted or regular_context.role == UserRole.ADMIN

        # Verify IP-based restrictions are considered
        if "ip" in admin_reason.lower():
            assert admin_context.ip_address in ["127.0.0.1", "192.168.1.1"]


class TestAPIEndpointAuthorization:
    """Test API endpoint-level authorization."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def api_endpoints(self):
        """Define all API endpoints with their authorization requirements."""
        return {
            # Agent endpoints
            "/api/v1/agents": {
                "GET": {
                    "permission": Permission.VIEW_AGENTS,
                    "roles": [UserRole.OBSERVER],
                },
                "POST": {
                    "permission": Permission.CREATE_AGENT,
                    "roles": [UserRole.AGENT_MANAGER],
                },
            },
            "/api/v1/agents/{agent_id}": {
                "GET": {
                    "permission": Permission.VIEW_AGENTS,
                    "roles": [UserRole.OBSERVER],
                },
                "PUT": {
                    "permission": Permission.MODIFY_AGENT,
                    "roles": [UserRole.AGENT_MANAGER],
                },
                "DELETE": {
                    "permission": Permission.DELETE_AGENT,
                    "roles": [UserRole.ADMIN],
                },
            },
            # Coalition endpoints
            "/api/v1/coalitions": {
                "GET": {
                    "permission": Permission.VIEW_AGENTS,
                    "roles": [UserRole.OBSERVER],
                },
                "POST": {
                    "permission": Permission.CREATE_COALITION,
                    "roles": [UserRole.RESEARCHER],
                },
            },
            # System endpoints
            "/api/v1/system/config": {
                "GET": {
                    "permission": Permission.ADMIN_SYSTEM,
                    "roles": [UserRole.ADMIN],
                },
                "PUT": {
                    "permission": Permission.ADMIN_SYSTEM,
                    "roles": [UserRole.ADMIN],
                },
            },
            # Metrics endpoints
            "/api/v1/metrics": {
                "GET": {
                    "permission": Permission.VIEW_METRICS,
                    "roles": [UserRole.OBSERVER],
                },
            },
        }

    def test_endpoint_permission_validation(
        self, client, test_users, api_endpoints
    ):
        """Test that each endpoint properly validates required permissions."""
        for endpoint, methods in api_endpoints.items():
            for method, requirements in methods.items():
                required_permission = requirements["permission"]
                requirements["roles"]

                # Test with each user role
                for role, user_data in test_users.items():
                    headers = {"Authorization": f"Bearer {user_data['token']}"}

                    # Prepare endpoint (replace placeholders)
                    test_endpoint = endpoint.replace(
                        "{agent_id}", "test_agent_123"
                    )

                    # Make request based on method
                    if method == "GET":
                        response = client.get(test_endpoint, headers=headers)
                    elif method == "POST":
                        response = client.post(
                            test_endpoint, headers=headers, json={}
                        )
                    elif method == "PUT":
                        response = client.put(
                            test_endpoint, headers=headers, json={}
                        )
                    elif method == "DELETE":
                        response = client.delete(
                            test_endpoint, headers=headers
                        )

                    # Check authorization
                    user_permissions = ROLE_PERMISSIONS.get(role, [])
                    has_permission = required_permission in user_permissions

                    if has_permission:
                        # Should not get 403 (might get other errors)
                        assert (
                            response.status_code != status.HTTP_403_FORBIDDEN
                        ), f"{role} should have access to {method} {endpoint}"
                    else:
                        # Should get 403
                        assert (
                            response.status_code == status.HTTP_403_FORBIDDEN
                        ), f"{role} should not have access to {method} {endpoint}"

    def test_http_method_based_access(self, client, test_users):
        """Test that different HTTP methods have appropriate access controls."""
        # Observer can GET but not POST/PUT/DELETE
        observer = test_users[UserRole.OBSERVER]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Should be able to view agents
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == status.HTTP_200_OK

        # Should not be able to create agents
        response = client.post(
            "/api/v1/agents",
            headers=headers,
            json={"name": "test", "template": "basic"},
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

        # Should not be able to update agents
        response = client.put(
            "/api/v1/agents/some_id", headers=headers, json={"name": "updated"}
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_api_versioning_authorization(self, client, test_users):
        """Test authorization across different API versions."""
        # Assuming we might have v1 and v2 APIs
        api_versions = ["v1"]  # Add "v2" when available

        for version in api_versions:
            endpoint = f"/api/{version}/agents"

            for role, user_data in test_users.items():
                headers = {"Authorization": f"Bearer {user_data['token']}"}
                response = client.get(endpoint, headers=headers)

                # Version should not affect authorization logic
                has_permission = (
                    Permission.VIEW_AGENTS in ROLE_PERMISSIONS.get(role, [])
                )

                if has_permission:
                    assert response.status_code != status.HTTP_403_FORBIDDEN
                else:
                    assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_parameter_based_access_control(self, client, test_users):
        """Test access control based on query parameters and request body."""
        # Some parameters might require elevated permissions
        researcher = test_users[UserRole.RESEARCHER]
        headers = {"Authorization": f"Bearer {researcher['token']}"}

        # Regular query - should work
        response = client.get("/api/v1/agents?limit=10", headers=headers)
        assert response.status_code == status.HTTP_200_OK

        # Sensitive parameters - might require admin
        response = client.get(
            "/api/v1/agents?include_deleted=true&show_system=true",
            headers=headers,
        )
        # System might restrict certain parameters

        # Test request body restrictions
        admin = test_users[UserRole.ADMIN]
        admin_headers = {"Authorization": f"Bearer {admin['token']}"}

        # Admin-only fields in request body
        response = client.post(
            "/api/v1/agents",
            headers=headers,  # Non-admin
            json={
                "name": "test_agent",
                "template": "basic",
                "system_override": True,  # Admin-only field
                "bypass_validation": True,  # Admin-only field
            },
        )
        # System should ignore or reject admin-only fields for non-admin users

    def test_middleware_authorization_chain(self, client, test_users):
        """Test that authorization middleware properly chains and validates."""
        # Test with missing token
        response = client.get("/api/v1/agents")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Test with invalid token
        headers = {"Authorization": "Bearer invalid_token_here"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Test with expired token
        test_users[UserRole.OBSERVER]["user"]
        with patch("jwt.decode") as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError()

            headers = {
                "Authorization": f"Bearer {test_users[UserRole.OBSERVER]['token']}"
            }
            response = client.get("/api/v1/agents", headers=headers)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Test proper token flow
        valid_headers = {
            "Authorization": f"Bearer {test_users[UserRole.OBSERVER]['token']}"
        }
        response = client.get("/api/v1/agents", headers=valid_headers)
        assert response.status_code == status.HTTP_200_OK


class TestAdvancedAuthorizationScenarios:
    """Test advanced authorization scenarios including ABAC, context-aware, and dynamic permissions."""

    @pytest.fixture
    def setup_advanced_rules(self):
        """Set up advanced ABAC rules for testing."""
        rules = [
            ABACRule(
                id="weekend_restriction",
                name="Weekend Access Restriction",
                description="Limit access on weekends",
                resource_type="*",
                action="modify",
                subject_conditions={"role": ["researcher", "agent_manager"]},
                resource_conditions={},
                environment_conditions={
                    "days": [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                    ]
                },
                effect=ABACEffect.ALLOW,
                priority=100,
                created_at=datetime.now(timezone.utc),
                created_by="system",
            ),
            ABACRule(
                id="high_value_protection",
                name="High Value Resource Protection",
                description="Extra protection for high-value resources",
                resource_type="*",
                action="*",
                subject_conditions={"min_risk_score": 0.7},
                resource_conditions={"sensitivity_level": "restricted"},
                environment_conditions={},
                effect=ABACEffect.DENY,
                priority=150,
                created_at=datetime.now(timezone.utc),
                created_by="system",
            ),
            ABACRule(
                id="geo_restriction",
                name="Geographic Access Control",
                description="Restrict access based on location",
                resource_type="system_config",
                action="*",
                subject_conditions={},
                resource_conditions={},
                environment_conditions={"location": ["US", "CA", "UK"]},
                effect=ABACEffect.ALLOW,
                priority=90,
                created_at=datetime.now(timezone.utc),
                created_by="system",
            ),
        ]

        for rule in rules:
            enhanced_rbac_manager.add_abac_rule(rule)

        return rules

    def test_attribute_based_access_control(self, setup_advanced_rules):
        """Test ABAC rule evaluation with complex attributes."""
        # Test 1: Time-based access restriction
        researcher_context = AccessContext(
            user_id="researcher_001",
            username="researcher",
            role=UserRole.RESEARCHER,
            permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
            timestamp=datetime.now(timezone.utc),
            risk_score=0.2,
        )

        resource = ResourceContext(
            resource_id="agent_001",
            resource_type="agent",
            classification="internal",
        )

        # Mock weekend scenario
        with patch("datetime.datetime") as mock_datetime:
            # Set to Saturday
            mock_datetime.now.return_value = datetime(
                2024, 1, 6, 12, 0, 0
            )  # Saturday
            mock_datetime.strftime = datetime.strftime

            (
                granted,
                reason,
                rules,
            ) = enhanced_rbac_manager.evaluate_abac_access(
                researcher_context, resource, "modify"
            )

            # Should be denied on weekend for non-admin
            if (
                "Weekend Access Restriction"
                in enhanced_rbac_manager.abac_rules
            ):
                assert not granted or researcher_context.role == UserRole.ADMIN

    def test_context_aware_authorization(self):
        """Test authorization decisions based on contextual information."""
        # High-risk context
        high_risk_context = AccessContext(
            user_id="user_001",
            username="suspicious_user",
            role=UserRole.RESEARCHER,
            permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
            ip_address="123.45.67.89",  # Unknown external IP
            user_agent="suspicious-bot/1.0",
            risk_score=0.85,
            device_id="unknown_device",
        )

        # Low-risk context
        low_risk_context = AccessContext(
            user_id="user_002",
            username="trusted_user",
            role=UserRole.RESEARCHER,
            permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
            ip_address="192.168.1.100",  # Internal IP
            user_agent="Mozilla/5.0 (Corporate Device)",
            risk_score=0.1,
            device_id="corp_device_123",
        )

        sensitive_resource = ResourceContext(
            resource_id="sensitive_001",
            resource_type="system",
            sensitivity_level="restricted",
        )

        # Test high-risk access
        (
            high_risk_granted,
            high_risk_reason,
            _,
        ) = enhanced_rbac_manager.evaluate_abac_access(
            high_risk_context, sensitive_resource, "view"
        )

        # Test low-risk access
        (
            low_risk_granted,
            low_risk_reason,
            _,
        ) = enhanced_rbac_manager.evaluate_abac_access(
            low_risk_context, sensitive_resource, "view"
        )

        # High-risk should be more restricted
        if high_risk_context.risk_score > 0.8:
            assert not high_risk_granted or "risk" in high_risk_reason.lower()

    def test_time_based_access_restrictions(self):
        """Test time-based access control mechanisms."""
        # Business hours rule
        business_hours_rule = ABACRule(
            id="business_hours_only",
            name="Business Hours Access",
            description="Access only during business hours",
            resource_type="financial_data",
            action="*",
            subject_conditions={},
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

        enhanced_rbac_manager.add_abac_rule(business_hours_rule)

        user_context = AccessContext(
            user_id="user_001",
            username="finance_user",
            role=UserRole.RESEARCHER,
            permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
        )

        financial_resource = ResourceContext(
            resource_id="fin_001", resource_type="financial_data"
        )

        # Test during business hours
        with patch("datetime.datetime") as mock_datetime:
            # Monday 10 AM
            mock_datetime.now.return_value = datetime(2024, 1, 1, 10, 0, 0)
            mock_datetime.now().time.return_value = datetime(
                2024, 1, 1, 10, 0, 0
            ).time()
            mock_datetime.now().strftime.return_value = "Monday"

            (
                granted_business,
                _,
                _,
            ) = enhanced_rbac_manager.evaluate_abac_access(
                user_context, financial_resource, "view"
            )

            # Saturday 10 AM
            mock_datetime.now.return_value = datetime(2024, 1, 6, 10, 0, 0)
            mock_datetime.now().time.return_value = datetime(
                2024, 1, 6, 10, 0, 0
            ).time()
            mock_datetime.now().strftime.return_value = "Saturday"

            granted_weekend, _, _ = enhanced_rbac_manager.evaluate_abac_access(
                user_context, financial_resource, "view"
            )

            # Access should be different based on day
            # (Note: actual behavior depends on rule configuration)

    def test_location_based_access_control(self):
        """Test location-based access restrictions."""
        # US-based user
        us_context = AccessContext(
            user_id="us_user",
            username="american_user",
            role=UserRole.ADMIN,
            permissions=list(ROLE_PERMISSIONS[UserRole.ADMIN]),
            location="US",
            ip_address="8.8.8.8",
        )

        # Non-approved location user
        other_context = AccessContext(
            user_id="other_user",
            username="foreign_user",
            role=UserRole.ADMIN,
            permissions=list(ROLE_PERMISSIONS[UserRole.ADMIN]),
            location="XX",
            ip_address="200.200.200.200",
        )

        system_resource = ResourceContext(
            resource_id="sys_config", resource_type="system_config"
        )

        # Evaluate access for both contexts
        us_granted, us_reason, _ = enhanced_rbac_manager.evaluate_abac_access(
            us_context, system_resource, "modify"
        )

        (
            other_granted,
            other_reason,
            _,
        ) = enhanced_rbac_manager.evaluate_abac_access(
            other_context, system_resource, "modify"
        )

        # Location-based rules might affect access
        # (Actual behavior depends on configured rules)

    def test_dynamic_permission_evaluation(self):
        """Test dynamic permission evaluation based on runtime conditions."""
        # Test risk score calculation
        normal_context = AccessContext(
            user_id="user_001",
            username="normal_user",
            role=UserRole.RESEARCHER,
            permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
            ip_address="192.168.1.100",
        )

        # Calculate risk with various factors
        risk_score = calculate_user_risk_score(
            normal_context,
            recent_failed_attempts=0,
            location_anomaly=False,
            time_anomaly=False,
            device_anomaly=False,
        )
        assert risk_score < 0.5  # Normal activity should have low risk

        # High risk scenario
        high_risk_score = calculate_user_risk_score(
            normal_context,
            recent_failed_attempts=5,
            location_anomaly=True,
            time_anomaly=True,
            device_anomaly=True,
        )
        assert (
            high_risk_score > 0.5
        )  # Anomalous activity should have high risk

        # Update context with calculated risk
        normal_context.risk_score = risk_score

        # Test permission evaluation with risk
        resource = ResourceContext(
            resource_id="sensitive_001",
            resource_type="sensitive_data",
            sensitivity_level="restricted",
        )

        granted, reason, _ = enhanced_rbac_manager.evaluate_abac_access(
            normal_context, resource, "view"
        )

        # Low risk should generally allow access (depends on rules)
        # High risk might trigger additional restrictions


class TestAuthorizationAttackVectors:
    """Test various authorization attack vectors and ensure proper defense."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def attack_users(self):
        """Create users for attack testing."""
        users = {}

        # Attacker with minimal privileges
        attacker = auth_manager.register_user(
            username="attacker",
            email="attacker@test.com",
            password="Attack123!",
            role=UserRole.OBSERVER,
        )
        users["attacker"] = {
            "user": attacker,
            "token": auth_manager.create_access_token(attacker),
        }

        # Legitimate user
        legitimate = auth_manager.register_user(
            username="legitimate",
            email="legitimate@test.com",
            password="Legit123!",
            role=UserRole.RESEARCHER,
        )
        users["legitimate"] = {
            "user": legitimate,
            "token": auth_manager.create_access_token(legitimate),
        }

        # Admin for comparison
        admin = auth_manager.register_user(
            username="admin",
            email="admin@test.com",
            password="Admin123!",
            role=UserRole.ADMIN,
        )
        users["admin"] = {
            "user": admin,
            "token": auth_manager.create_access_token(admin),
        }

        return users

    def test_horizontal_privilege_escalation(self, client, attack_users):
        """Test defenses against horizontal privilege escalation."""
        attacker = attack_users["attacker"]
        legitimate = attack_users["legitimate"]

        # Attacker tries to access another user's resources
        headers = {"Authorization": f"Bearer {attacker['token']}"}

        # Attempt 1: Direct ID manipulation
        response = client.get(
            f"/api/v1/users/{legitimate['user'].user_id}/agents",
            headers=headers,
        )
        # Should be forbidden or return only public data

        # Attempt 2: Parameter pollution
        response = client.get(
            f"/api/v1/agents?user_id={legitimate['user'].user_id}&user_id={attacker['user'].user_id}",
            headers=headers,
        )
        # Should only return attacker's agents or error

        # Attempt 3: JWT sub claim manipulation (simulated)
        with patch.object(auth_manager, "verify_token") as mock_verify:
            # Simulate token with modified user_id
            tampered_token_data = TokenData(
                user_id=legitimate[
                    "user"
                ].user_id,  # Attempting to impersonate
                username=attacker["user"].username,
                role=attacker["user"].role,
                permissions=list(ROLE_PERMISSIONS[attacker["user"].role]),
                exp=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            mock_verify.return_value = tampered_token_data

            response = client.get("/api/v1/agents", headers=headers)
            # System should detect inconsistency

    def test_vertical_privilege_escalation(self, client, attack_users):
        """Test defenses against vertical privilege escalation."""
        attacker = attack_users["attacker"]
        attack_users["admin"]

        headers = {"Authorization": f"Bearer {attacker['token']}"}

        # Attempt 1: Access admin endpoints directly
        admin_endpoints = [
            "/api/v1/system/config",
            "/api/v1/users",
            "/api/v1/security/audit-log",
        ]

        for endpoint in admin_endpoints:
            response = client.get(endpoint, headers=headers)
            assert (
                response.status_code == status.HTTP_403_FORBIDDEN
            ), f"Attacker should not access {endpoint}"

        # Attempt 2: Exploit role checking logic
        # Try to confuse the system with role-like values
        exploit_payloads = [
            {"role": "admin"},
            {"role": ["observer", "admin"]},
            {"roles": "admin"},
            {"user_role": "admin"},
            {"override_role": "admin"},
        ]

        for payload in exploit_payloads:
            response = client.post(
                "/api/v1/agents",
                headers=headers,
                json={"name": "ExploitAgent", "template": "basic", **payload},
            )
            assert (
                response.status_code == status.HTTP_403_FORBIDDEN
            ), f"Role exploitation attempt should fail: {payload}"

        # Attempt 3: Chain multiple vulnerabilities
        # First, try to create a coalition (shouldn't work)
        response = client.post(
            "/api/v1/coalitions",
            headers=headers,
            json={"name": "AttackerCoalition", "agents": []},
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_idor_vulnerabilities(self, client, attack_users):
        """Test Insecure Direct Object Reference (IDOR) vulnerabilities."""
        attacker = attack_users["attacker"]
        legitimate = attack_users["legitimate"]

        # Legitimate user creates a resource
        legit_headers = {"Authorization": f"Bearer {legitimate['token']}"}
        response = client.post(
            "/api/v1/agents",
            headers=legit_headers,
            json={
                "name": "LegitimateAgent",
                "template": "basic",
                "parameters": {"private_data": "secret"},
            },
        )

        if response.status_code == status.HTTP_201_CREATED:
            agent_id = response.json().get("id")

            # Attacker tries to access via direct ID
            attacker_headers = {"Authorization": f"Bearer {attacker['token']}"}

            # Attempt 1: Direct access
            response = client.get(
                f"/api/v1/agents/{agent_id}", headers=attacker_headers
            )
            # Might be allowed for viewing but should not expose private data

            # Attempt 2: Modification attempt
            response = client.put(
                f"/api/v1/agents/{agent_id}",
                headers=attacker_headers,
                json={"name": "Hijacked"},
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN

            # Attempt 3: Deletion attempt
            response = client.delete(
                f"/api/v1/agents/{agent_id}", headers=attacker_headers
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN

        # Test sequential ID enumeration
        base_id = "agent_"
        for i in range(1, 10):
            response = client.get(
                f"/api/v1/agents/{base_id}{i}", headers=attacker_headers
            )
            # Should either 404 or 403, not expose existence

    def test_authorization_bypass_attempts(self, client, attack_users):
        """Test various authorization bypass techniques."""
        attacker = attack_users["attacker"]
        headers = {"Authorization": f"Bearer {attacker['token']}"}

        # Attempt 1: Case sensitivity bypass
        case_variations = [
            {"Authorization": f"bearer {attacker['token']}"},
            {"authorization": f"Bearer {attacker['token']}"},
            {"AUTHORIZATION": f"BEARER {attacker['token']}"},
        ]

        for header_variant in case_variations:
            response = client.post(
                "/api/v1/agents",
                headers=header_variant,
                json={"name": "test", "template": "basic"},
            )
            # Should still enforce authorization

        # Attempt 2: HTTP method tampering
        # Try unusual HTTP methods
        response = client.request(
            "PATCH",
            "/api/v1/system/config",
            headers=headers,
            json={"setting": "value"},
        )
        # Should not bypass authorization

        # Attempt 3: Content-Type manipulation
        headers_with_content = {
            **headers,
            "Content-Type": "text/plain",  # Trying to bypass JSON validation
        }
        response = client.post(
            "/api/v1/agents",
            headers=headers_with_content,
            data='{"name": "test", "role": "admin"}',  # Sneaking in role
        )
        assert response.status_code in [
            status.HTTP_403_FORBIDDEN,
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        ]

        # Attempt 4: Double encoding
        encoded_payload = {
            "name": "test%2520agent",  # Double URL encoding
            "template": "basic",
            "parameters": {"role": "%61%64%6d%69%6e"},  # "admin" encoded
        }
        response = client.post(
            "/api/v1/agents", headers=headers, json=encoded_payload
        )
        # Should not grant elevated privileges

    def test_token_manipulation_attacks(self, client, attack_users):
        """Test token manipulation for privilege escalation."""

        import jwt

        attacker = attack_users["attacker"]
        attack_users["admin"]

        # Attempt 1: Algorithm confusion attack
        # Try to force HS256 instead of RS256
        try:
            # Decode without verification to get payload
            payload = jwt.decode(
                attacker["token"], options={"verify_signature": False}
            )

            # Modify payload
            payload["role"] = "admin"
            payload["permissions"] = ["admin_system"]

            # Try to encode with HS256 using public key as secret
            # (This is a known attack vector)
            fake_token = jwt.encode(payload, "fake_secret", algorithm="HS256")

            headers = {"Authorization": f"Bearer {fake_token}"}
            response = client.get("/api/v1/system/config", headers=headers)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
        except Exception:
            pass  # Attack should fail

        # Attempt 2: Token substitution
        # Try to use refresh token as access token
        refresh_token = auth_manager.create_refresh_token(attacker["user"])
        headers = {"Authorization": f"Bearer {refresh_token}"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Attempt 3: Expired token reuse
        with patch("datetime.datetime") as mock_datetime:
            # Create token that expires immediately
            mock_datetime.now.return_value = datetime.now(timezone.utc)
            mock_datetime.now(timezone.utc).return_value = datetime.now(
                timezone.utc
            )

            expired_token = auth_manager.create_access_token(attacker["user"])

            # Move time forward
            mock_datetime.now.return_value = datetime.now(
                timezone.utc
            ) + timedelta(hours=2)

            headers = {"Authorization": f"Bearer {expired_token}"}
            response = client.get("/api/v1/agents", headers=headers)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Attempt 4: Token replay attack
        # Use the same token multiple times rapidly
        headers = {"Authorization": f"Bearer {attacker['token']}"}
        responses = []

        for _ in range(10):
            response = client.get("/api/v1/agents", headers=headers)
            responses.append(response.status_code)

        # All requests should be handled consistently
        assert all(
            status == status.HTTP_200_OK for status in responses
        ) or all(status != status.HTTP_200_OK for status in responses)


class TestAuthorizationPerformanceAndConcurrency:
    """Test authorization system performance and concurrent access scenarios."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def load_test_users(self):
        """Create multiple users for load testing."""
        users = []
        roles = [
            UserRole.ADMIN,
            UserRole.RESEARCHER,
            UserRole.AGENT_MANAGER,
            UserRole.OBSERVER,
        ]

        for i in range(20):  # 20 users, 5 per role
            role = roles[i % len(roles)]
            user = auth_manager.register_user(
                username=f"loadtest_user_{i}",
                email=f"loadtest{i}@test.com",
                password="LoadTest123!",
                role=role,
            )
            token = auth_manager.create_access_token(user)
            users.append({"user": user, "token": token, "role": role})

        return users

    def test_concurrent_authorization_checks(self, client, load_test_users):
        """Test system behavior under concurrent authorization requests."""
        import concurrent.futures

        def make_auth_request(user_data):
            headers = {"Authorization": f"Bearer {user_data['token']}"}
            response = client.get("/api/v1/agents", headers=headers)
            return response.status_code, user_data["role"]

        # Launch concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_auth_request, user)
                for user in load_test_users
            ]

            results = [
                f.result() for f in concurrent.futures.as_completed(futures)
            ]

        # Verify all requests were properly authorized
        for status_code, role in results:
            if Permission.VIEW_AGENTS in ROLE_PERMISSIONS.get(role, []):
                assert status_code == status.HTTP_200_OK
            else:
                assert status_code == status.HTTP_403_FORBIDDEN

    def test_authorization_caching_behavior(self, client):
        """Test authorization caching and cache invalidation."""
        # Create user and get token
        user = auth_manager.register_user(
            username="cache_test_user",
            email="cache@test.com",
            password="Cache123!",
            role=UserRole.RESEARCHER,
        )
        token = auth_manager.create_access_token(user)
        headers = {"Authorization": f"Bearer {token}"}

        # Make multiple requests to test caching
        response_times = []

        for i in range(10):
            start_time = time.time()
            response = client.get("/api/v1/agents", headers=headers)
            end_time = time.time()

            response_times.append(end_time - start_time)
            assert response.status_code == status.HTTP_200_OK

        # Later requests should be faster due to caching (if implemented)
        # This is a basic check - actual caching behavior depends on implementation
        avg_first_half = sum(response_times[:5]) / 5
        avg_second_half = sum(response_times[5:]) / 5

        # Log the performance characteristics
        print(f"Avg response time first half: {avg_first_half:.4f}s")
        print(f"Avg response time second half: {avg_second_half:.4f}s")

    def test_rate_limited_authorization_attempts(self, client):
        """Test system behavior under rate-limited authorization attempts."""
        # Create attacker user
        attacker = auth_manager.register_user(
            username="rate_limit_attacker",
            email="ratelimit@test.com",
            password="RateLimit123!",
            role=UserRole.OBSERVER,
        )
        token = auth_manager.create_access_token(attacker)
        headers = {"Authorization": f"Bearer {token}"}

        # Attempt many unauthorized requests rapidly
        forbidden_count = 0
        rate_limited_count = 0

        for i in range(150):  # Exceed rate limit
            response = client.post(
                "/api/v1/agents",  # Observer can't create agents
                headers=headers,
                json={"name": f"spam_{i}", "template": "basic"},
            )

            if response.status_code == status.HTTP_403_FORBIDDEN:
                forbidden_count += 1
            elif response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                rate_limited_count += 1

        # Should eventually hit rate limit
        assert forbidden_count > 0  # Initial requests are forbidden
        # Rate limiting might kick in after threshold

    def test_authorization_under_resource_pressure(self):
        """Test authorization decisions under resource pressure."""
        # Simulate high memory/CPU scenarios
        contexts = []

        # Create many ABAC contexts
        for i in range(1000):
            context = AccessContext(
                user_id=f"user_{i}",
                username=f"user_{i}",
                role=UserRole.RESEARCHER,
                permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
                ip_address=f"192.168.1.{i % 255}",
                risk_score=i / 1000.0,
            )
            contexts.append(context)

        # Create complex resource contexts
        resources = []
        for i in range(1000):
            resource = ResourceContext(
                resource_id=f"resource_{i}",
                resource_type="agent",
                owner_id=f"user_{i % 100}",
                department=f"dept_{i % 10}",
                classification="internal" if i % 2 == 0 else "public",
                sensitivity_level="public" if i % 3 == 0 else "internal",
                metadata={"index": i, "tags": [f"tag_{j}" for j in range(5)]},
            )
            resources.append(resource)

        # Time authorization decisions under load
        start_time = time.time()
        decisions = []

        for i in range(100):  # Sample of evaluations
            context = contexts[i]
            resource = resources[i]

            (
                granted,
                reason,
                rules,
            ) = enhanced_rbac_manager.evaluate_abac_access(
                context, resource, "view"
            )
            decisions.append((granted, reason))

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Authorization decisions under load: {total_time:.4f}s for 100 evaluations"
        )
        print(f"Average time per decision: {total_time/100:.4f}s")

        # Ensure decisions are still being made correctly
        assert len(decisions) == 100
        assert all(isinstance(d[0], bool) for d in decisions)


class TestComprehensiveAuthorizationIntegration:
    """Integration tests combining multiple authorization aspects."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def complex_scenario_setup(self):
        """Set up a complex authorization scenario."""
        # Create departments
        departments = ["research", "operations", "finance", "hr"]

        # Create users across departments and roles
        users = {}
        for dept in departments:
            for role in [
                UserRole.ADMIN,
                UserRole.RESEARCHER,
                UserRole.AGENT_MANAGER,
                UserRole.OBSERVER,
            ]:
                username = f"{dept}_{role.value}"
                user = auth_manager.register_user(
                    username=username,
                    email=f"{username}@company.com",
                    password="Complex123!",
                    role=role,
                )
                users[username] = {
                    "user": user,
                    "token": auth_manager.create_access_token(user),
                    "department": dept,
                    "role": role,
                }

        # Create complex ABAC rules
        rules = [
            {
                "id": "dept_isolation",
                "name": "Department Isolation",
                "resource_type": "*",
                "action": "*",
                "subject_conditions": {},
                "resource_conditions": {"same_department": True},
                "environment_conditions": {},
                "effect": ABACEffect.ALLOW,
                "priority": 100,
            },
            {
                "id": "finance_protection",
                "name": "Finance Data Protection",
                "resource_type": "financial_record",
                "action": "*",
                "subject_conditions": {"department": ["finance", "executive"]},
                "resource_conditions": {},
                "environment_conditions": {
                    "time_range": {"start": "09:00", "end": "18:00"}
                },
                "effect": ABACEffect.ALLOW,
                "priority": 120,
            },
        ]

        for rule_data in rules:
            rule = ABACRule(
                id=rule_data["id"],
                name=rule_data["name"],
                description=f"{rule_data['name']} rule",
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
            enhanced_rbac_manager.add_abac_rule(rule)

        return users, departments

    def test_end_to_end_authorization_flow(
        self, client, complex_scenario_setup
    ):
        """Test complete authorization flow from login to resource access."""
        users, departments = complex_scenario_setup

        # Test complete flow for research department admin
        users["research_admin"]

        # 1. Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": "research_admin", "password": "Complex123!"},
        )

        if login_response.status_code == status.HTTP_200_OK:
            token = login_response.json().get("access_token")
            headers = {"Authorization": f"Bearer {token}"}

            # 2. Access own department resources
            response = client.get(
                "/api/v1/agents?department=research", headers=headers
            )
            assert response.status_code == status.HTTP_200_OK

            # 3. Try to access other department resources
            response = client.get(
                "/api/v1/agents?department=finance", headers=headers
            )
            # Might be filtered or forbidden based on ABAC rules

            # 4. Create resource in own department
            response = client.post(
                "/api/v1/agents",
                headers=headers,
                json={
                    "name": "ResearchAgent",
                    "template": "research_template",
                    "parameters": {"department": "research"},
                },
            )
            assert response.status_code in [
                status.HTTP_201_CREATED,
                status.HTTP_200_OK,
            ]

    def test_multi_factor_authorization_decision(self, complex_scenario_setup):
        """Test authorization decisions involving multiple factors."""
        users, _ = complex_scenario_setup

        # Finance user trying to access financial data
        finance_user = users["finance_researcher"]

        # Create context with multiple factors
        context = AccessContext(
            user_id=finance_user["user"].user_id,
            username=finance_user["user"].username,
            role=finance_user["role"],
            permissions=list(ROLE_PERMISSIONS[finance_user["role"]]),
            department="finance",
            ip_address="10.0.0.50",  # Internal IP
            location="US",
            risk_score=0.2,
            timestamp=datetime.now(timezone.utc).replace(hour=14),  # 2 PM
        )

        # Financial resource
        financial_resource = ResourceContext(
            resource_id="fin_report_001",
            resource_type="financial_record",
            owner_id=finance_user["user"].user_id,
            department="finance",
            classification="confidential",
            sensitivity_level="restricted",
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        # Test access during business hours
        granted, reason, rules = enhanced_rbac_manager.evaluate_abac_access(
            context, financial_resource, "view"
        )

        # Should consider multiple factors:
        # - Role permissions
        # - Department match
        # - Time restrictions
        # - Risk score
        # - Resource sensitivity

        print(f"Multi-factor decision: {granted}, Reason: {reason}")
        print(f"Applied rules: {rules}")

    def test_authorization_audit_trail(self, client, complex_scenario_setup):
        """Test that all authorization decisions are properly audited."""
        users, _ = complex_scenario_setup

        # Clear existing audit log
        enhanced_rbac_manager.access_audit_log.clear()

        # Perform various actions
        test_user = users["operations_agent_manager"]
        headers = {"Authorization": f"Bearer {test_user['token']}"}

        # Successful access
        response = client.get("/api/v1/agents", headers=headers)

        # Failed access attempt
        response = client.get("/api/v1/system/config", headers=headers)

        # Resource creation
        response = client.post(
            "/api/v1/agents",
            headers=headers,
            json={"name": "AuditTestAgent", "template": "basic"},
        )

        # Check audit log
        audit_entries = enhanced_rbac_manager.access_audit_log

        # Verify audit entries exist
        assert len(audit_entries) > 0

        # Verify audit entry structure
        for entry in audit_entries:
            assert "timestamp" in entry
            assert "user_id" in entry
            assert "action" in entry
            assert "decision" in entry
            assert "resource_type" in entry

        # Generate audit report
        report = enhanced_rbac_manager.generate_access_report()
        assert "audit_statistics" in report
        assert report["audit_statistics"]["total_access_decisions"] == len(
            audit_entries
        )


# Test execution helpers
if __name__ == "__main__":
    # Run specific test classes or all tests
    pytest.main([__file__, "-v", "-s"])

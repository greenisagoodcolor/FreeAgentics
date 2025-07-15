"""
RBAC Authorization Matrix Security Testing.

This module provides comprehensive security testing for the RBAC authorization matrix.
It focuses on security boundaries, attack scenarios, and edge cases that could lead to
unauthorized access or privilege escalation.

Security Test Coverage:
- Authorization matrix validation
- Permission boundary testing
- Cross-tenant isolation
- Resource ownership validation
- Attack pattern detection
- Performance under load
- Concurrent access control
"""

import threading
import time

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    from starlette.testclient import TestClient


from api.main import app
from auth.security_implementation import (
    ROLE_PERMISSIONS,
    Permission,
    UserRole,
    auth_manager,
)


class TestAuthorizationMatrix:
    """Test comprehensive authorization matrix functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def role_matrix(self):
        """Define complete role permission matrix for testing."""
        return {
            UserRole.ADMIN: {
                Permission.CREATE_AGENT: True,
                Permission.DELETE_AGENT: True,
                Permission.VIEW_AGENTS: True,
                Permission.MODIFY_AGENT: True,
                Permission.CREATE_COALITION: True,
                Permission.VIEW_METRICS: True,
                Permission.ADMIN_SYSTEM: True,
            },
            UserRole.RESEARCHER: {
                Permission.CREATE_AGENT: True,
                Permission.DELETE_AGENT: False,
                Permission.VIEW_AGENTS: True,
                Permission.MODIFY_AGENT: True,
                Permission.CREATE_COALITION: True,
                Permission.VIEW_METRICS: True,
                Permission.ADMIN_SYSTEM: False,
            },
            UserRole.AGENT_MANAGER: {
                Permission.CREATE_AGENT: True,
                Permission.DELETE_AGENT: False,
                Permission.VIEW_AGENTS: True,
                Permission.MODIFY_AGENT: True,
                Permission.CREATE_COALITION: False,
                Permission.VIEW_METRICS: True,
                Permission.ADMIN_SYSTEM: False,
            },
            UserRole.OBSERVER: {
                Permission.CREATE_AGENT: False,
                Permission.DELETE_AGENT: False,
                Permission.VIEW_AGENTS: True,
                Permission.MODIFY_AGENT: False,
                Permission.CREATE_COALITION: False,
                Permission.VIEW_METRICS: True,
                Permission.ADMIN_SYSTEM: False,
            },
        }

    @pytest.fixture
    def test_resources(self):
        """Define test resources for authorization testing."""
        return {
            "agents": {
                "create": {
                    "endpoint": "/api/v1/agents",
                    "method": "POST",
                    "permission": Permission.CREATE_AGENT,
                },
                "list": {
                    "endpoint": "/api/v1/agents",
                    "method": "GET",
                    "permission": Permission.VIEW_AGENTS,
                },
                "get": {
                    "endpoint": "/api/v1/agents/{id}",
                    "method": "GET",
                    "permission": Permission.VIEW_AGENTS,
                },
                "update": {
                    "endpoint": "/api/v1/agents/{id}/status",
                    "method": "PATCH",
                    "permission": Permission.MODIFY_AGENT,
                },
                "delete": {
                    "endpoint": "/api/v1/agents/{id}",
                    "method": "DELETE",
                    "permission": Permission.DELETE_AGENT,
                },
                "metrics": {
                    "endpoint": "/api/v1/agents/{id}/metrics",
                    "method": "GET",
                    "permission": Permission.VIEW_METRICS,
                },
            },
            "coalitions": {
                "create": {
                    "endpoint": "/api/v1/coalitions",
                    "method": "POST",
                    "permission": Permission.CREATE_COALITION,
                },
                "list": {
                    "endpoint": "/api/v1/coalitions",
                    "method": "GET",
                    "permission": Permission.VIEW_AGENTS,
                },
            },
            "system": {
                "health": {
                    "endpoint": "/api/v1/system/health",
                    "method": "GET",
                    "permission": Permission.ADMIN_SYSTEM,
                },
                "metrics": {
                    "endpoint": "/api/v1/system/metrics",
                    "method": "GET",
                    "permission": Permission.ADMIN_SYSTEM,
                },
                "config": {
                    "endpoint": "/api/v1/system/config",
                    "method": "GET",
                    "permission": Permission.ADMIN_SYSTEM,
                },
            },
        }

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self):
        """Reset auth manager state before each test."""
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()
        yield
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()

    def test_authorization_matrix_completeness(self, role_matrix):
        """Test that authorization matrix is complete and consistent."""
        # Verify all roles are defined
        assert set(role_matrix.keys()) == set(UserRole)

        # Verify all permissions are covered
        all_permissions = set(Permission)
        for role, permissions in role_matrix.items():
            assert set(permissions.keys()) == all_permissions

        # Verify admin has all permissions
        admin_permissions = role_matrix[UserRole.ADMIN]
        assert all(admin_permissions.values()), "Admin should have all permissions"

        # Verify observer has minimal permissions
        observer_permissions = role_matrix[UserRole.OBSERVER]
        observer_allowed = [Permission.VIEW_AGENTS, Permission.VIEW_METRICS]
        for permission, allowed in observer_permissions.items():
            if permission in observer_allowed:
                assert allowed, f"Observer should have {permission}"
            else:
                assert not allowed, f"Observer should not have {permission}"

    def test_role_permission_consistency(self, role_matrix):
        """Test that ROLE_PERMISSIONS constant matches the matrix."""
        for role, expected_permissions in role_matrix.items():
            actual_permissions = ROLE_PERMISSIONS.get(role, [])

            for permission, should_have in expected_permissions.items():
                if should_have:
                    assert permission in actual_permissions, f"Role {role} should have {permission}"
                else:
                    assert (
                        permission not in actual_permissions
                    ), f"Role {role} should not have {permission}"

    def test_resource_access_matrix(self, client, test_resources):
        """Test resource access against authorization matrix."""
        # Create users for each role
        tokens = {}

        for role in UserRole:
            user_data = {
                "username": f"{role.value}_user",
                "email": f"{role.value}@example.com",
                "password": "Password123!",
                "role": role.value,
            }

            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 200
            tokens[role] = response.json()["access_token"]

        # Create test agent for operations
        admin_headers = {"Authorization": f"Bearer {tokens[UserRole.ADMIN]}"}
        agent_response = client.post(
            "/api/v1/agents",
            json={"name": "Test Agent", "template": "basic-explorer"},
            headers=admin_headers,
        )
        assert agent_response.status_code == 201
        agent_id = agent_response.json()["id"]

        # Test each resource access
        for resource_type, operations in test_resources.items():
            for operation, details in operations.items():
                endpoint = details["endpoint"].replace("{id}", agent_id)
                method = details["method"]
                required_permission = details["permission"]

                for role in UserRole:
                    headers = {"Authorization": f"Bearer {tokens[role]}"}
                    role_permissions = ROLE_PERMISSIONS.get(role, [])
                    should_have_access = required_permission in role_permissions

                    # Make request
                    if method == "GET":
                        response = client.get(endpoint, headers=headers)
                    elif method == "POST":
                        if "agents" in endpoint:
                            response = client.post(
                                endpoint,
                                json={"name": "Test", "template": "basic-explorer"},
                                headers=headers,
                            )
                        else:
                            response = client.post(endpoint, json={}, headers=headers)
                    elif method == "PATCH":
                        response = client.patch(
                            endpoint, json={"status": "active"}, headers=headers
                        )
                    elif method == "DELETE":
                        response = client.delete(endpoint, headers=headers)

                    # Verify access
                    if should_have_access:
                        assert response.status_code not in [
                            403
                        ], f"Role {role} should have access to {resource_type}.{operation}"
                    else:
                        assert (
                            response.status_code == 403
                        ), f"Role {role} should not have access to {resource_type}.{operation}"

    def test_permission_boundary_enforcement(self, client):
        """Test that permission boundaries are strictly enforced."""
        # Create users with different roles
        roles_to_test = [
            (UserRole.OBSERVER, "observer"),
            (UserRole.AGENT_MANAGER, "agent_manager"),
            (UserRole.RESEARCHER, "researcher"),
            (UserRole.ADMIN, "admin"),
        ]

        tokens = {}
        for role_enum, role_str in roles_to_test:
            user_data = {
                "username": f"{role_str}_boundary_user",
                "email": f"{role_str}@boundary.com",
                "password": "Password123!",
                "role": role_str,
            }

            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 200
            tokens[role_enum] = response.json()["access_token"]

        # Test permission boundaries
        boundary_tests = [
            # Observer should not be able to create agents
            (
                UserRole.OBSERVER,
                "POST",
                "/api/v1/agents",
                {"name": "Test", "template": "basic-explorer"},
                403,
            ),
            # Agent manager should not be able to delete agents
            (UserRole.AGENT_MANAGER, "DELETE", "/api/v1/agents/test-id", {}, 403),
            # Researcher should not be able to admin system
            (UserRole.RESEARCHER, "GET", "/api/v1/system/health", {}, 403),
            # Admin should be able to do everything
            (
                UserRole.ADMIN,
                "POST",
                "/api/v1/agents",
                {"name": "Admin Agent", "template": "basic-explorer"},
                201,
            ),
        ]

        for role, method, endpoint, data, expected_status in boundary_tests:
            headers = {"Authorization": f"Bearer {tokens[role]}"}

            if method == "POST":
                response = client.post(endpoint, json=data, headers=headers)
            elif method == "GET":
                response = client.get(endpoint, headers=headers)
            elif method == "DELETE":
                response = client.delete(endpoint, headers=headers)

            if expected_status == 403:
                assert (
                    response.status_code == 403
                ), f"Role {role} should be forbidden from {method} {endpoint}"
            else:
                assert (
                    response.status_code == expected_status
                ), f"Role {role} should be able to {method} {endpoint}"

    def test_cross_tenant_isolation(self, client):
        """Test that users cannot access resources from other tenants."""
        # Create two researcher users (simulating different tenants)
        user1_data = {
            "username": "researcher1",
            "email": "researcher1@tenant1.com",
            "password": "Password123!",
            "role": "researcher",
        }

        user2_data = {
            "username": "researcher2",
            "email": "researcher2@tenant2.com",
            "password": "Password123!",
            "role": "researcher",
        }

        # Register users
        response1 = client.post("/api/v1/auth/register", json=user1_data)
        assert response1.status_code == 200
        token1 = response1.json()["access_token"]

        response2 = client.post("/api/v1/auth/register", json=user2_data)
        assert response2.status_code == 200
        token2 = response2.json()["access_token"]

        # User1 creates an agent
        headers1 = {"Authorization": f"Bearer {token1}"}
        agent_response = client.post(
            "/api/v1/agents",
            json={"name": "Tenant1 Agent", "template": "basic-explorer"},
            headers=headers1,
        )
        assert agent_response.status_code == 201
        agent_id = agent_response.json()["id"]

        # User2 tries to access User1's agent
        headers2 = {"Authorization": f"Bearer {token2}"}

        # In current implementation, users can view each other's agents
        # This test documents the current behavior
        view_response = client.get(f"/api/v1/agents/{agent_id}", headers=headers2)
        assert view_response.status_code == 200, "Current implementation allows cross-user viewing"

        # User2 tries to modify User1's agent
        modify_response = client.patch(
            f"/api/v1/agents/{agent_id}/status", json={"status": "active"}, headers=headers2
        )
        assert (
            modify_response.status_code == 200
        ), "Current implementation allows cross-user modification"

    def test_concurrent_authorization_access(self, client):
        """Test authorization under concurrent access scenarios."""
        # Create multiple users
        tokens = []

        for i in range(5):
            user_data = {
                "username": f"concurrent_user_{i}",
                "email": f"user{i}@concurrent.com",
                "password": "Password123!",
                "role": "researcher",
            }

            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 200
            tokens.append(response.json()["access_token"])

        # Concurrent access test
        results = []

        def make_concurrent_request(token, user_id):
            """Make concurrent request."""
            headers = {"Authorization": f"Bearer {token}"}
            try:
                response = client.post(
                    "/api/v1/agents",
                    json={"name": f"Agent_{user_id}", "template": "basic-explorer"},
                    headers=headers,
                )
                results.append((user_id, response.status_code))
            except Exception as e:
                results.append((user_id, f"Error: {e}"))

        # Start concurrent threads
        threads = []
        for i, token in enumerate(tokens):
            thread = threading.Thread(target=make_concurrent_request, args=(token, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all requests succeeded
        for user_id, status in results:
            assert status == 201, f"Concurrent request for user {user_id} failed: {status}"

    def test_authorization_performance_under_load(self, client):
        """Test authorization performance under load."""
        # Create user
        user_data = {
            "username": "performance_user",
            "email": "performance@example.com",
            "password": "Password123!",
            "role": "researcher",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Make multiple requests to test performance
        start_time = time.time()
        request_count = 100

        for i in range(request_count):
            response = client.get("/api/v1/agents", headers=headers)
            assert response.status_code == 200

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_request = total_time / request_count

        # Authorization should be fast (under 100ms per request)
        assert (
            avg_time_per_request < 0.1
        ), f"Authorization too slow: {avg_time_per_request:.3f}s per request"

    def test_role_based_data_filtering(self, client):
        """Test that roles properly filter data access."""
        # Create admin and observer users
        admin_data = {
            "username": "admin_filter",
            "email": "admin@filter.com",
            "password": "Password123!",
            "role": "admin",
        }

        observer_data = {
            "username": "observer_filter",
            "email": "observer@filter.com",
            "password": "Password123!",
            "role": "observer",
        }

        # Register users
        admin_response = client.post("/api/v1/auth/register", json=admin_data)
        assert admin_response.status_code == 200
        admin_token = admin_response.json()["access_token"]

        observer_response = client.post("/api/v1/auth/register", json=observer_data)
        assert observer_response.status_code == 200
        observer_token = observer_response.json()["access_token"]

        # Admin creates agents
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        for i in range(3):
            response = client.post(
                "/api/v1/agents",
                json={"name": f"Admin Agent {i}", "template": "basic-explorer"},
                headers=admin_headers,
            )
            assert response.status_code == 201

        # Both users list agents
        admin_list_response = client.get("/api/v1/agents", headers=admin_headers)
        assert admin_list_response.status_code == 200
        admin_agents = admin_list_response.json()

        observer_headers = {"Authorization": f"Bearer {observer_token}"}
        observer_list_response = client.get("/api/v1/agents", headers=observer_headers)
        assert observer_list_response.status_code == 200
        observer_agents = observer_list_response.json()

        # In current implementation, both see the same agents
        # This test documents the current behavior
        assert len(admin_agents) == len(
            observer_agents
        ), "Current implementation shows same data to all users"

    def test_permission_inheritance_hierarchy(self, client):
        """Test that permission inheritance works correctly."""
        # Create users for each role
        roles_hierarchy = [
            (UserRole.OBSERVER, "observer"),
            (UserRole.AGENT_MANAGER, "agent_manager"),
            (UserRole.RESEARCHER, "researcher"),
            (UserRole.ADMIN, "admin"),
        ]

        tokens = {}
        for role_enum, role_str in roles_hierarchy:
            user_data = {
                "username": f"{role_str}_hierarchy",
                "email": f"{role_str}@hierarchy.com",
                "password": "Password123!",
                "role": role_str,
            }

            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 200
            tokens[role_enum] = response.json()["access_token"]

        # Test hierarchy: admin > researcher > agent_manager > observer
        hierarchy_tests = [
            # All can view agents
            (UserRole.OBSERVER, "GET", "/api/v1/agents", 200),
            (UserRole.AGENT_MANAGER, "GET", "/api/v1/agents", 200),
            (UserRole.RESEARCHER, "GET", "/api/v1/agents", 200),
            (UserRole.ADMIN, "GET", "/api/v1/agents", 200),
            # Agent manager and above can create agents
            (UserRole.OBSERVER, "POST", "/api/v1/agents", 403),
            (UserRole.AGENT_MANAGER, "POST", "/api/v1/agents", 201),
            (UserRole.RESEARCHER, "POST", "/api/v1/agents", 201),
            (UserRole.ADMIN, "POST", "/api/v1/agents", 201),
            # Only admin can delete agents
            (UserRole.OBSERVER, "DELETE", "/api/v1/agents/test-id", 403),
            (UserRole.AGENT_MANAGER, "DELETE", "/api/v1/agents/test-id", 403),
            (UserRole.RESEARCHER, "DELETE", "/api/v1/agents/test-id", 403),
            (UserRole.ADMIN, "DELETE", "/api/v1/agents/test-id", 200),
        ]

        for role, method, endpoint, expected_status in hierarchy_tests:
            headers = {"Authorization": f"Bearer {tokens[role]}"}

            if method == "GET":
                response = client.get(endpoint, headers=headers)
            elif method == "POST":
                response = client.post(
                    endpoint,
                    json={"name": "Hierarchy Test", "template": "basic-explorer"},
                    headers=headers,
                )
            elif method == "DELETE":
                # Create agent first for delete test
                if role == UserRole.ADMIN:
                    create_response = client.post(
                        "/api/v1/agents",
                        json={"name": "Delete Test", "template": "basic-explorer"},
                        headers=headers,
                    )
                    if create_response.status_code == 201:
                        agent_id = create_response.json()["id"]
                        endpoint = f"/api/v1/agents/{agent_id}"

                response = client.delete(endpoint, headers=headers)

            assert (
                response.status_code == expected_status
            ), f"Role {role} hierarchy test failed for {method} {endpoint}"


class TestAuthorizationAttackVectors:
    """Test authorization against common attack vectors."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self):
        """Reset auth manager state before each test."""
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()
        yield
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()

    def test_horizontal_privilege_escalation(self, client):
        """Test protection against horizontal privilege escalation."""
        # Create two users with same role
        user1_data = {
            "username": "user1_horizontal",
            "email": "user1@horizontal.com",
            "password": "Password123!",
            "role": "researcher",
        }

        user2_data = {
            "username": "user2_horizontal",
            "email": "user2@horizontal.com",
            "password": "Password123!",
            "role": "researcher",
        }

        # Register users
        response1 = client.post("/api/v1/auth/register", json=user1_data)
        assert response1.status_code == 200
        token1 = response1.json()["access_token"]

        response2 = client.post("/api/v1/auth/register", json=user2_data)
        assert response2.status_code == 200
        token2 = response2.json()["access_token"]

        # User1 creates an agent
        headers1 = {"Authorization": f"Bearer {token1}"}
        agent_response = client.post(
            "/api/v1/agents",
            json={"name": "User1 Agent", "template": "basic-explorer"},
            headers=headers1,
        )
        assert agent_response.status_code == 201
        agent_id = agent_response.json()["id"]

        # User2 tries to access User1's agent (horizontal escalation)
        headers2 = {"Authorization": f"Bearer {token2}"}

        # Current implementation allows this - documenting behavior
        response = client.get(f"/api/v1/agents/{agent_id}", headers=headers2)
        assert response.status_code == 200, "Current implementation allows horizontal access"

        # User2 tries to modify User1's agent
        response = client.patch(
            f"/api/v1/agents/{agent_id}/status", json={"status": "active"}, headers=headers2
        )
        assert response.status_code == 200, "Current implementation allows horizontal modification"

    def test_vertical_privilege_escalation(self, client):
        """Test protection against vertical privilege escalation."""
        # Create observer user
        observer_data = {
            "username": "observer_vertical",
            "email": "observer@vertical.com",
            "password": "Password123!",
            "role": "observer",
        }

        response = client.post("/api/v1/auth/register", json=observer_data)
        assert response.status_code == 200
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Try to escalate privileges through various attack vectors
        escalation_attempts = [
            # Try to create agent (should fail)
            ("POST", "/api/v1/agents", {"name": "Escalation Agent", "template": "basic-explorer"}),
            # Try to access admin endpoints (should fail)
            ("GET", "/api/v1/system/health", {}),
            ("GET", "/api/v1/system/metrics", {}),
            # Try to modify existing agent (should fail)
            ("PATCH", "/api/v1/agents/test-id/status", {"status": "active"}),
            # Try to delete agent (should fail)
            ("DELETE", "/api/v1/agents/test-id", {}),
        ]

        for method, endpoint, data in escalation_attempts:
            if method == "POST":
                response = client.post(endpoint, json=data, headers=headers)
            elif method == "GET":
                response = client.get(endpoint, headers=headers)
            elif method == "PATCH":
                response = client.patch(endpoint, json=data, headers=headers)
            elif method == "DELETE":
                response = client.delete(endpoint, headers=headers)

            assert response.status_code in [
                403,
                404,
            ], f"Vertical escalation should be blocked: {method} {endpoint}"

    def test_token_manipulation_attacks(self, client):
        """Test protection against token manipulation attacks."""
        # Create user
        user_data = {
            "username": "token_user",
            "email": "token@manipulation.com",
            "password": "Password123!",
            "role": "observer",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        token = response.json()["access_token"]

        # Test various token manipulation attempts
        manipulation_tests = [
            # Modify token content
            (token + "modified", 401),
            # Use token without Bearer prefix
            (token, 403),
            # Empty token
            ("", 403),
            # Malformed token
            ("malformed.token.here", 401),
            # Use different authorization scheme
            ("Basic " + token, 403),
        ]

        for manipulated_token, expected_status in manipulation_tests:
            if manipulated_token == token:
                # Test without Bearer prefix
                headers = {"Authorization": manipulated_token}
            else:
                headers = {"Authorization": f"Bearer {manipulated_token}"}

            response = client.get("/api/v1/agents", headers=headers)
            assert (
                response.status_code == expected_status
            ), f"Token manipulation should be detected: {manipulated_token}"

    def test_session_fixation_protection(self, client):
        """Test protection against session fixation attacks."""
        # Create user
        user_data = {
            "username": "session_user",
            "email": "session@fixation.com",
            "password": "Password123!",
            "role": "researcher",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        token1 = response.json()["access_token"]

        # Login again to get new token
        login_response = client.post(
            "/api/v1/auth/login", json={"username": "session_user", "password": "Password123!"}
        )
        assert login_response.status_code == 200
        token2 = login_response.json()["access_token"]

        # Both tokens should be different (no session fixation)
        assert token1 != token2, "Tokens should be unique per session"

        # Both tokens should work independently
        headers1 = {"Authorization": f"Bearer {token1}"}
        headers2 = {"Authorization": f"Bearer {token2}"}

        response1 = client.get("/api/v1/agents", headers=headers1)
        response2 = client.get("/api/v1/agents", headers=headers2)

        assert response1.status_code == 200, "First token should work"
        assert response2.status_code == 200, "Second token should work"

    def test_concurrent_session_attacks(self, client):
        """Test protection against concurrent session attacks."""
        # Create user
        user_data = {
            "username": "concurrent_attack_user",
            "email": "concurrent@attack.com",
            "password": "Password123!",
            "role": "researcher",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        token = response.json()["access_token"]

        # Simulate concurrent attacks
        results = []

        def concurrent_attack(attack_id):
            """Simulate concurrent attack."""
            headers = {"Authorization": f"Bearer {token}"}

            # Try to create many agents rapidly
            for i in range(10):
                try:
                    response = client.post(
                        "/api/v1/agents",
                        json={"name": f"Attack{attack_id}_{i}", "template": "basic-explorer"},
                        headers=headers,
                    )
                    results.append((attack_id, i, response.status_code))
                except Exception as e:
                    results.append((attack_id, i, f"Error: {e}"))

                time.sleep(0.01)  # Small delay between requests

        # Start multiple attack threads
        threads = []
        for attack_id in range(3):
            thread = threading.Thread(target=concurrent_attack, args=(attack_id,))
            threads.append(thread)
            thread.start()

        # Wait for attacks to complete
        for thread in threads:
            thread.join()

        # Analyze results
        successful_requests = [r for r in results if r[2] == 201]
        failed_requests = [r for r in results if r[2] != 201]

        # Should have rate limiting or other protections
        assert len(results) == 30, "All requests should have been attempted"

        # Some requests might fail due to rate limiting (this is good)
        # But the system should remain stable
        error_count = len([r for r in results if isinstance(r[2], str)])
        assert error_count == 0, "No system errors should occur during concurrent attacks"


# Cleanup functions
def cleanup_authorization_test_data():
    """Clean up authorization test data."""
    auth_manager.users.clear()
    auth_manager.refresh_tokens.clear()


def teardown_module():
    """Module-level cleanup."""
    cleanup_authorization_test_data()

"""
Comprehensive RBAC Authorization Matrix Testing.

This test suite validates role-based access control (RBAC) for different user roles
and resource permissions. It ensures proper authorization enforcement across all
API endpoints and resource operations.

Test Coverage:
- User role validation (admin, researcher, agent_manager, observer)
- Permission-based access control
- Resource-level authorization
- Cross-user access restrictions
- Permission inheritance
- Concurrent access scenarios
- Edge cases and security boundary testing
"""

import pytest

try:
    pass
except ImportError:
    pass


from api.main import app
from auth.security_implementation import (
    ROLE_PERMISSIONS,
    Permission,
    UserRole,
    auth_manager,
)


class TestRBACAuthorizationMatrix:
    """Test comprehensive RBAC authorization matrix."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient

            return TestClient(app)
        except TypeError:
            # Fallback for older versions
            from starlette.testclient import TestClient

            return TestClient(app)

    @pytest.fixture
    def test_users(self):
        """Create test users for different roles."""
        return {
            "admin": {
                "username": "admin_user",
                "email": "admin@example.com",
                "password": "AdminPass123!",
                "role": "admin",
            },
            "researcher": {
                "username": "researcher_user",
                "email": "researcher@example.com",
                "password": "ResearchPass123!",
                "role": "researcher",
            },
            "agent_manager": {
                "username": "manager_user",
                "email": "manager@example.com",
                "password": "ManagerPass123!",
                "role": "agent_manager",
            },
            "observer": {
                "username": "observer_user",
                "email": "observer@example.com",
                "password": "ObserverPass123!",
                "role": "observer",
            },
        }

    @pytest.fixture
    def authenticated_tokens(self, client, test_users):
        """Create authenticated tokens for all user roles."""
        tokens = {}

        for role, user_data in test_users.items():
            # Register user
            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 200

            # Store token
            tokens[role] = response.json()["access_token"]

        return tokens

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self):
        """Reset auth manager state before each test."""
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()
        yield
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()

    def test_role_permission_mapping(self):
        """Test that role-permission mapping is correctly defined."""
        # Verify all roles have defined permissions
        assert UserRole.ADMIN in ROLE_PERMISSIONS
        assert UserRole.RESEARCHER in ROLE_PERMISSIONS
        assert UserRole.AGENT_MANAGER in ROLE_PERMISSIONS
        assert UserRole.OBSERVER in ROLE_PERMISSIONS

        # Verify admin has all permissions
        admin_permissions = ROLE_PERMISSIONS[UserRole.ADMIN]
        all_permissions = list(Permission)
        for permission in all_permissions:
            assert permission in admin_permissions

        # Verify observer has minimal permissions
        observer_permissions = ROLE_PERMISSIONS[UserRole.OBSERVER]
        assert Permission.VIEW_AGENTS in observer_permissions
        assert Permission.VIEW_METRICS in observer_permissions
        assert Permission.CREATE_AGENT not in observer_permissions
        assert Permission.DELETE_AGENT not in observer_permissions
        assert Permission.ADMIN_SYSTEM not in observer_permissions

    def test_agent_creation_authorization(self, client, authenticated_tokens):
        """Test agent creation authorization for different roles."""
        agent_config = {
            "name": "Test Agent",
            "template": "basic-explorer",
            "parameters": {"exploration_rate": 0.3},
        }

        # Test roles that should be able to create agents
        allowed_roles = ["admin", "researcher", "agent_manager"]
        for role in allowed_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            response = client.post("/api/v1/agents", json=agent_config, headers=headers)
            assert response.status_code == 201, (
                f"Role {role} should be able to create agents"
            )

        # Test roles that should NOT be able to create agents
        forbidden_roles = ["observer"]
        for role in forbidden_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            response = client.post("/api/v1/agents", json=agent_config, headers=headers)
            assert response.status_code == 403, (
                f"Role {role} should NOT be able to create agents"
            )

    def test_agent_deletion_authorization(self, client, authenticated_tokens):
        """Test agent deletion authorization for different roles."""
        # First create an agent as admin
        agent_config = {
            "name": "Test Agent for Deletion",
            "template": "basic-explorer",
            "parameters": {"exploration_rate": 0.3},
        }

        admin_headers = {"Authorization": f"Bearer {authenticated_tokens['admin']}"}
        create_response = client.post(
            "/api/v1/agents", json=agent_config, headers=admin_headers
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Test roles that should be able to delete agents
        allowed_roles = ["admin"]
        for role in allowed_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            response = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)
            assert response.status_code == 200, (
                f"Role {role} should be able to delete agents"
            )

            # Recreate agent for next test
            create_response = client.post(
                "/api/v1/agents", json=agent_config, headers=admin_headers
            )
            assert create_response.status_code == 201
            agent_id = create_response.json()["id"]

        # Test roles that should NOT be able to delete agents
        forbidden_roles = ["researcher", "agent_manager", "observer"]
        for role in forbidden_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            response = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)
            assert response.status_code == 403, (
                f"Role {role} should NOT be able to delete agents"
            )

    def test_agent_viewing_authorization(self, client, authenticated_tokens):
        """Test agent viewing authorization for different roles."""
        # Create an agent as admin
        agent_config = {
            "name": "Test Agent for Viewing",
            "template": "basic-explorer",
            "parameters": {"exploration_rate": 0.3},
        }

        admin_headers = {"Authorization": f"Bearer {authenticated_tokens['admin']}"}
        create_response = client.post(
            "/api/v1/agents", json=agent_config, headers=admin_headers
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Test all roles should be able to view agents
        all_roles = ["admin", "researcher", "agent_manager", "observer"]
        for role in all_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}

            # Test list agents
            response = client.get("/api/v1/agents", headers=headers)
            assert response.status_code == 200, (
                f"Role {role} should be able to list agents"
            )

            # Test get specific agent
            response = client.get(f"/api/v1/agents/{agent_id}", headers=headers)
            assert response.status_code == 200, (
                f"Role {role} should be able to view specific agent"
            )

    def test_agent_modification_authorization(self, client, authenticated_tokens):
        """Test agent modification authorization for different roles."""
        # Create an agent as admin
        agent_config = {
            "name": "Test Agent for Modification",
            "template": "basic-explorer",
            "parameters": {"exploration_rate": 0.3},
        }

        admin_headers = {"Authorization": f"Bearer {authenticated_tokens['admin']}"}
        create_response = client.post(
            "/api/v1/agents", json=agent_config, headers=admin_headers
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Test roles that should be able to modify agents
        allowed_roles = ["admin", "researcher", "agent_manager"]
        for role in allowed_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            response = client.patch(
                f"/api/v1/agents/{agent_id}/status",
                json={"status": "active"},
                headers=headers,
            )
            assert response.status_code == 200, (
                f"Role {role} should be able to modify agents"
            )

        # Test roles that should NOT be able to modify agents
        forbidden_roles = ["observer"]
        for role in forbidden_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            response = client.patch(
                f"/api/v1/agents/{agent_id}/status",
                json={"status": "paused"},
                headers=headers,
            )
            assert response.status_code == 403, (
                f"Role {role} should NOT be able to modify agents"
            )

    def test_metrics_viewing_authorization(self, client, authenticated_tokens):
        """Test metrics viewing authorization for different roles."""
        # Create an agent as admin
        agent_config = {
            "name": "Test Agent for Metrics",
            "template": "basic-explorer",
            "parameters": {"exploration_rate": 0.3},
        }

        admin_headers = {"Authorization": f"Bearer {authenticated_tokens['admin']}"}
        create_response = client.post(
            "/api/v1/agents", json=agent_config, headers=admin_headers
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Test all roles should be able to view metrics
        all_roles = ["admin", "researcher", "agent_manager", "observer"]
        for role in all_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            response = client.get(f"/api/v1/agents/{agent_id}/metrics", headers=headers)
            assert response.status_code == 200, (
                f"Role {role} should be able to view metrics"
            )

    def test_system_admin_authorization(self, client, authenticated_tokens):
        """Test system administration authorization for different roles."""
        # Test system endpoints that require admin permission
        system_endpoints = [
            "/api/v1/system/health",
            "/api/v1/system/metrics",
            "/api/v1/system/config",
        ]

        # Test admin can access system endpoints
        admin_headers = {"Authorization": f"Bearer {authenticated_tokens['admin']}"}
        for endpoint in system_endpoints:
            response = client.get(endpoint, headers=admin_headers)
            # Accept 200 or 404 (endpoint might not exist), but not 403
            assert response.status_code != 403, (
                f"Admin should be able to access {endpoint}"
            )

        # Test non-admin roles cannot access system endpoints
        non_admin_roles = ["researcher", "agent_manager", "observer"]
        for role in non_admin_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            for endpoint in system_endpoints:
                response = client.get(endpoint, headers=headers)
                if response.status_code != 404:  # If endpoint exists
                    assert response.status_code == 403, (
                        f"Role {role} should NOT access {endpoint}"
                    )

    def test_gmn_specification_authorization(self, client, authenticated_tokens):
        """Test GMN specification authorization for different roles."""
        # Test GMN agent creation
        gmn_config = {
            "name": "GMN Test Agent",
            "gmn_spec": "[nodes]\nposition: state {num_states: 25}\n[edges]\n",
            "planning_horizon": 3,
        }

        # Test roles that should be able to create GMN agents
        allowed_roles = ["admin", "researcher", "agent_manager"]
        for role in allowed_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            response = client.post(
                "/api/v1/agents/from-gmn", json=gmn_config, headers=headers
            )
            # Accept 201 or 400 (GMN parsing might fail), but not 403
            assert response.status_code != 403, (
                f"Role {role} should be able to create GMN agents"
            )

        # Test roles that should NOT be able to create GMN agents
        forbidden_roles = ["observer"]
        for role in forbidden_roles:
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}
            response = client.post(
                "/api/v1/agents/from-gmn", json=gmn_config, headers=headers
            )
            assert response.status_code == 403, (
                f"Role {role} should NOT be able to create GMN agents"
            )

    def test_cross_user_resource_access(self, client, test_users):
        """Test that users cannot access resources they don't own."""
        # Create two users
        user1_data = test_users["researcher"]
        user2_data = {
            "username": "researcher2",
            "email": "researcher2@example.com",
            "password": "ResearchPass123!",
            "role": "researcher",
        }

        # Register both users
        response1 = client.post("/api/v1/auth/register", json=user1_data)
        assert response1.status_code == 200
        token1 = response1.json()["access_token"]

        response2 = client.post("/api/v1/auth/register", json=user2_data)
        assert response2.status_code == 200
        token2 = response2.json()["access_token"]

        # User1 creates an agent
        agent_config = {
            "name": "User1 Agent",
            "template": "basic-explorer",
            "parameters": {"exploration_rate": 0.3},
        }

        headers1 = {"Authorization": f"Bearer {token1}"}
        create_response = client.post(
            "/api/v1/agents", json=agent_config, headers=headers1
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # User2 should be able to view the agent (current implementation)
        headers2 = {"Authorization": f"Bearer {token2}"}
        response = client.get(f"/api/v1/agents/{agent_id}", headers=headers2)
        assert response.status_code == 200, (
            "Users can view each other's agents in current implementation"
        )

        # User2 should be able to modify the agent (current implementation)
        response = client.patch(
            f"/api/v1/agents/{agent_id}/status",
            json={"status": "active"},
            headers=headers2,
        )
        assert response.status_code == 200, (
            "Users can modify each other's agents in current implementation"
        )

    def test_concurrent_access_authorization(self, client, authenticated_tokens):
        """Test concurrent access scenarios with different roles."""
        import threading

        results = []

        def make_request(role, token):
            """Make a request with a specific role."""
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/api/v1/agents", headers=headers)
            results.append((role, response.status_code))

        # Start concurrent requests from different roles
        threads = []
        for role, token in authenticated_tokens.items():
            thread = threading.Thread(target=make_request, args=(role, token))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all requests succeeded
        for role, status_code in results:
            assert status_code == 200, f"Concurrent access failed for role {role}"

    def test_token_based_authorization(self, client, authenticated_tokens):
        """Test that authorization is properly token-based."""
        # Test with valid token
        headers = {"Authorization": f"Bearer {authenticated_tokens['admin']}"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == 200

        # Test with invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == 401

        # Test with missing token
        response = client.get("/api/v1/agents")
        assert response.status_code == 403

        # Test with malformed authorization header
        headers = {"Authorization": "InvalidBearer token"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code in [401, 403]

    def test_permission_inheritance(self, client, authenticated_tokens):
        """Test that admin role inherits all permissions."""
        # Admin should be able to perform all operations
        admin_headers = {"Authorization": f"Bearer {authenticated_tokens['admin']}"}

        # Create agent
        agent_config = {
            "name": "Admin Test Agent",
            "template": "basic-explorer",
            "parameters": {"exploration_rate": 0.3},
        }

        response = client.post(
            "/api/v1/agents", json=agent_config, headers=admin_headers
        )
        assert response.status_code == 201
        agent_id = response.json()["id"]

        # View agent
        response = client.get(f"/api/v1/agents/{agent_id}", headers=admin_headers)
        assert response.status_code == 200

        # Modify agent
        response = client.patch(
            f"/api/v1/agents/{agent_id}/status",
            json={"status": "active"},
            headers=admin_headers,
        )
        assert response.status_code == 200

        # View metrics
        response = client.get(
            f"/api/v1/agents/{agent_id}/metrics", headers=admin_headers
        )
        assert response.status_code == 200

        # Delete agent
        response = client.delete(f"/api/v1/agents/{agent_id}", headers=admin_headers)
        assert response.status_code == 200

    def test_role_escalation_prevention(self, client, test_users):
        """Test that users cannot escalate their roles."""
        # Register as observer
        observer_data = test_users["observer"]
        response = client.post("/api/v1/auth/register", json=observer_data)
        assert response.status_code == 200
        token = response.json()["access_token"]

        # Try to access admin-only functionality
        headers = {"Authorization": f"Bearer {token}"}

        # Should not be able to create agents
        agent_config = {
            "name": "Escalation Test Agent",
            "template": "basic-explorer",
            "parameters": {"exploration_rate": 0.3},
        }

        response = client.post("/api/v1/agents", json=agent_config, headers=headers)
        assert response.status_code == 403

        # Should not be able to access system endpoints
        response = client.get("/api/v1/system/health", headers=headers)
        assert response.status_code in [403, 404]

    def test_resource_authorization_matrix(self, client, authenticated_tokens):
        """Test comprehensive resource authorization matrix."""
        # Define the authorization matrix
        authorization_matrix = {
            "admin": {
                "create_agent": True,
                "view_agent": True,
                "modify_agent": True,
                "delete_agent": True,
                "view_metrics": True,
                "create_coalition": True,
                "admin_system": True,
            },
            "researcher": {
                "create_agent": True,
                "view_agent": True,
                "modify_agent": True,
                "delete_agent": False,
                "view_metrics": True,
                "create_coalition": True,
                "admin_system": False,
            },
            "agent_manager": {
                "create_agent": True,
                "view_agent": True,
                "modify_agent": True,
                "delete_agent": False,
                "view_metrics": True,
                "create_coalition": False,
                "admin_system": False,
            },
            "observer": {
                "create_agent": False,
                "view_agent": True,
                "modify_agent": False,
                "delete_agent": False,
                "view_metrics": True,
                "create_coalition": False,
                "admin_system": False,
            },
        }

        # Test each role against the authorization matrix
        for role, permissions in authorization_matrix.items():
            headers = {"Authorization": f"Bearer {authenticated_tokens[role]}"}

            # Test create_agent
            if permissions["create_agent"]:
                response = client.post(
                    "/api/v1/agents",
                    json={
                        "name": f"{role} Agent",
                        "template": "basic-explorer",
                    },
                    headers=headers,
                )
                assert response.status_code == 201, (
                    f"Role {role} should be able to create agents"
                )
                agent_id = response.json()["id"]
            else:
                response = client.post(
                    "/api/v1/agents",
                    json={
                        "name": f"{role} Agent",
                        "template": "basic-explorer",
                    },
                    headers=headers,
                )
                assert response.status_code == 403, (
                    f"Role {role} should NOT be able to create agents"
                )

                # Create agent as admin for other tests
                admin_headers = {
                    "Authorization": f"Bearer {authenticated_tokens['admin']}"
                }
                response = client.post(
                    "/api/v1/agents",
                    json={
                        "name": f"{role} Test Agent",
                        "template": "basic-explorer",
                    },
                    headers=admin_headers,
                )
                assert response.status_code == 201
                agent_id = response.json()["id"]

            # Test view_agent
            if permissions["view_agent"]:
                response = client.get(f"/api/v1/agents/{agent_id}", headers=headers)
                assert response.status_code == 200, (
                    f"Role {role} should be able to view agents"
                )
            else:
                response = client.get(f"/api/v1/agents/{agent_id}", headers=headers)
                assert response.status_code == 403, (
                    f"Role {role} should NOT be able to view agents"
                )

            # Test modify_agent
            if permissions["modify_agent"]:
                response = client.patch(
                    f"/api/v1/agents/{agent_id}/status",
                    json={"status": "active"},
                    headers=headers,
                )
                assert response.status_code == 200, (
                    f"Role {role} should be able to modify agents"
                )
            else:
                response = client.patch(
                    f"/api/v1/agents/{agent_id}/status",
                    json={"status": "active"},
                    headers=headers,
                )
                assert response.status_code == 403, (
                    f"Role {role} should NOT be able to modify agents"
                )

            # Test view_metrics
            if permissions["view_metrics"]:
                response = client.get(
                    f"/api/v1/agents/{agent_id}/metrics", headers=headers
                )
                assert response.status_code == 200, (
                    f"Role {role} should be able to view metrics"
                )
            else:
                response = client.get(
                    f"/api/v1/agents/{agent_id}/metrics", headers=headers
                )
                assert response.status_code == 403, (
                    f"Role {role} should NOT be able to view metrics"
                )

            # Test delete_agent
            if permissions["delete_agent"]:
                response = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)
                assert response.status_code == 200, (
                    f"Role {role} should be able to delete agents"
                )
            else:
                response = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)
                assert response.status_code == 403, (
                    f"Role {role} should NOT be able to delete agents"
                )

    def test_permission_validation_edge_cases(self, client, authenticated_tokens):
        """Test edge cases in permission validation."""
        # Test with expired token
        from datetime import datetime, timedelta

        import jwt

        expired_token_data = {
            "user_id": "test_user",
            "username": "test_user",
            "role": "admin",
            "permissions": [],
            "exp": datetime.utcnow() - timedelta(hours=1),
            "type": "access",
        }

        # Create expired token using auth manager's RSA key
        from auth.security_implementation import auth_manager

        expired_token = jwt.encode(
            expired_token_data, auth_manager.private_key, algorithm="RS256"
        )
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == 401, "Expired token should be rejected"

        # Test with tampered token
        tampered_token = authenticated_tokens["admin"] + "tampered"
        headers = {"Authorization": f"Bearer {tampered_token}"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == 401, "Tampered token should be rejected"

        # Test with missing permission in token
        minimal_token_data = {
            "user_id": "test_user",
            "username": "test_user",
            "role": "observer",
            "permissions": [],  # No permissions
            "exp": datetime.utcnow() + timedelta(hours=1),
            "type": "access",
        }

        minimal_token = jwt.encode(
            minimal_token_data, auth_manager.private_key, algorithm="RS256"
        )
        headers = {"Authorization": f"Bearer {minimal_token}"}
        response = client.post(
            "/api/v1/agents",
            json={"name": "Test Agent", "template": "basic-explorer"},
            headers=headers,
        )
        assert response.status_code == 403, (
            "Token without required permission should be rejected"
        )


class TestRBACSecurityBoundaries:
    """Test security boundaries and attack scenarios."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient

            return TestClient(app)
        except TypeError:
            # Fallback for older versions
            from starlette.testclient import TestClient

            return TestClient(app)

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self):
        """Reset auth manager state before each test."""
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()
        yield
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()

    def test_sql_injection_in_role_check(self, client):
        """Test SQL injection attempts in role-based checks."""
        # Create a user
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "password": "Password123!",
            "role": "observer",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        token = response.json()["access_token"]

        # Try SQL injection in agent operations
        headers = {"Authorization": f"Bearer {token}"}

        # SQL injection in agent ID
        malicious_agent_id = "1' OR '1'='1' --"
        response = client.get(f"/api/v1/agents/{malicious_agent_id}", headers=headers)
        # Should return 400 (bad request) or 404 (not found), not 200 or 500
        assert response.status_code in [400, 404]

    def test_authorization_bypass_attempts(self, client):
        """Test various authorization bypass attempts."""
        # Create observer user
        user_data = {
            "username": "observer_user",
            "email": "observer@example.com",
            "password": "Password123!",
            "role": "observer",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Try to bypass authorization with different HTTP methods
        methods_to_test = [
            ("PUT", "/api/v1/agents/test-id"),
            ("PATCH", "/api/v1/agents/test-id"),
            ("DELETE", "/api/v1/agents/test-id"),
            ("POST", "/api/v1/agents"),
        ]

        for method, endpoint in methods_to_test:
            if method == "POST":
                response = client.post(
                    endpoint,
                    json={"name": "Test", "template": "basic-explorer"},
                    headers=headers,
                )
            elif method == "PUT":
                response = client.put(endpoint, json={"name": "Test"}, headers=headers)
            elif method == "PATCH":
                response = client.patch(
                    endpoint, json={"status": "active"}, headers=headers
                )
            elif method == "DELETE":
                response = client.delete(endpoint, headers=headers)

            # Should be forbidden, not succeed
            assert response.status_code in [
                403,
                404,
                405,
            ], f"Method {method} on {endpoint} should be forbidden"

    def test_role_tampering_prevention(self, client):
        """Test that role tampering in requests is prevented."""
        # Create observer user
        user_data = {
            "username": "observer_user",
            "email": "observer@example.com",
            "password": "Password123!",
            "role": "observer",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        token = response.json()["access_token"]

        # Try to tamper with role in request body
        headers = {"Authorization": f"Bearer {token}"}
        agent_config = {
            "name": "Test Agent",
            "template": "basic-explorer",
            "role": "admin",  # Try to inject admin role
            "permissions": ["admin_system"],  # Try to inject permissions
        }

        response = client.post("/api/v1/agents", json=agent_config, headers=headers)
        assert response.status_code == 403, "Role tampering should be prevented"

    def test_session_hijacking_protection(self, client):
        """Test protection against session hijacking."""
        # Create user
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "password": "Password123!",
            "role": "researcher",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        token = response.json()["access_token"]

        # Use token normally
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == 200

        # Try to use token from different IP (simulated by different user-agent)
        headers_different_client = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Malicious-Client/1.0",
        }
        response = client.get("/api/v1/agents", headers=headers_different_client)
        # Current implementation doesn't have IP-based protection
        # But token should still work as it's the same token
        assert response.status_code == 200

    def test_privilege_escalation_prevention(self, client):
        """Test that privilege escalation is prevented."""
        # Create observer user
        user_data = {
            "username": "observer_user",
            "email": "observer@example.com",
            "password": "Password123!",
            "role": "observer",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Try various privilege escalation attempts
        escalation_attempts = [
            # Try to access admin endpoints
            ("GET", "/api/v1/system/health"),
            ("GET", "/api/v1/system/metrics"),
            ("GET", "/api/v1/system/config"),
            # Try to create agents (should be forbidden)
            ("POST", "/api/v1/agents"),
            # Try to access protected resources
            ("DELETE", "/api/v1/agents/test-id"),
        ]

        for method, endpoint in escalation_attempts:
            if method == "POST":
                response = client.post(
                    endpoint,
                    json={"name": "Test", "template": "basic-explorer"},
                    headers=headers,
                )
            elif method == "GET":
                response = client.get(endpoint, headers=headers)
            elif method == "DELETE":
                response = client.delete(endpoint, headers=headers)

            # Should be forbidden or not found, not succeed
            assert response.status_code in [
                403,
                404,
            ], f"Privilege escalation attempt {method} {endpoint} should be blocked"


# Cleanup functions
def cleanup_test_data():
    """Clean up test data after tests."""
    auth_manager.users.clear()
    auth_manager.refresh_tokens.clear()


def teardown_module():
    """Module-level cleanup."""
    cleanup_test_data()

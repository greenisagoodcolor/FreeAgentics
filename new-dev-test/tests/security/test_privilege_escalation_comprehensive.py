"""
Comprehensive Privilege Escalation Testing Module.

This module provides exhaustive tests for all privilege escalation vectors including:
- Vertical privilege escalation (role elevation)
- Horizontal privilege escalation (cross-user access)
- Token-based escalation attacks
- API endpoint escalation
- Database-level escalation

Tests validate that the system properly defends against all privilege escalation attempts.
"""

import base64
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt
import pytest
from api.main import app
from auth.security_implementation import Permission, UserRole, auth_manager
from fastapi import status
from fastapi.testclient import TestClient


class TestVerticalPrivilegeEscalation:
    """Test vertical privilege escalation attacks (role elevation)."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def setup_vertical_escalation(self):
        """Set up test scenario for vertical escalation testing."""
        # Create users with different roles
        users = {}

        # Observer (lowest privilege)
        observer = auth_manager.register_user(
            username="observer_user",
            email="observer@test.com",
            password="Observer123!",
            role=UserRole.OBSERVER,
        )
        users["observer"] = {
            "user": observer,
            "token": auth_manager.create_access_token(observer),
            "role": UserRole.OBSERVER,
        }

        # Researcher (medium privilege)
        researcher = auth_manager.register_user(
            username="researcher_user",
            email="researcher@test.com",
            password="Researcher123!",
            role=UserRole.RESEARCHER,
        )
        users["researcher"] = {
            "user": researcher,
            "token": auth_manager.create_access_token(researcher),
            "role": UserRole.RESEARCHER,
        }

        # Admin (highest privilege)
        admin = auth_manager.register_user(
            username="admin_user",
            email="admin@test.com",
            password="Admin123!",
            role=UserRole.ADMIN,
        )
        users["admin"] = {
            "user": admin,
            "token": auth_manager.create_access_token(admin),
            "role": UserRole.ADMIN,
        }

        return users

    def test_role_elevation_via_registration(self, client):
        """Test attempting to register with elevated privileges."""
        # Attempt to register as admin directly
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "attacker_admin",
                "email": "attacker@admin.com",
                "password": "Attack123!",
                "role": "admin",  # Attempting to set admin role
            },
        )

        # Should either reject or default to lowest privilege
        if response.status_code == status.HTTP_201_CREATED:
            data = response.json()
            # Verify user was not created with admin role
            user = auth_manager.get_user(data.get("username"))
            assert user.role != UserRole.ADMIN
            assert user.role in [UserRole.OBSERVER, UserRole.RESEARCHER]

    def test_role_elevation_via_profile_update(self, client, setup_vertical_escalation):
        """Test attempting to elevate role through profile update."""
        users = setup_vertical_escalation
        observer = users["observer"]

        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Attempt to update own role to admin
        response = client.put(
            f"/api/v1/users/{observer['user'].id}",
            headers=headers,
            json={"role": "admin"},
        )

        # Should be forbidden or ignored
        assert response.status_code in [
            status.HTTP_403_FORBIDDEN,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

        # Verify role hasn't changed
        user = auth_manager.get_user(observer["user"].username)
        assert user.role == UserRole.OBSERVER

    def test_permission_injection_attack(self, client, setup_vertical_escalation):
        """Test attempting to inject additional permissions."""
        users = setup_vertical_escalation
        observer = users["observer"]

        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Attempt various permission injection techniques
        injection_attempts = [
            {"permissions": ["admin_system"]},
            {"permission": "admin_system"},
            {"roles": ["admin"]},
            {"role": "admin"},
            {"is_admin": True},
            {"admin": True},
            {"superuser": True},
            {"elevated": True},
        ]

        for payload in injection_attempts:
            # Try through user update endpoint
            response = client.put(
                f"/api/v1/users/{observer['user'].id}",
                headers=headers,
                json=payload,
            )

            # Should not succeed in granting elevated permissions
            if response.status_code == status.HTTP_200_OK:
                # Verify permissions haven't changed
                current_user = auth_manager.get_current_user(observer["token"])
                assert not auth_manager.has_permission(current_user, Permission.ADMIN_SYSTEM)

    def test_administrative_function_access(self, client, setup_vertical_escalation):
        """Test accessing admin-only functions with lower privileges."""
        users = setup_vertical_escalation

        # Admin-only endpoints
        admin_endpoints = [
            ("/api/v1/system/config", "GET"),
            ("/api/v1/system/config", "PUT"),
            ("/api/v1/users", "GET"),  # List all users
            ("/api/v1/system/logs", "GET"),
            ("/api/v1/system/audit", "GET"),
            ("/api/v1/database/backup", "POST"),
            ("/api/v1/security/keys/rotate", "POST"),
        ]

        # Test each role trying to access admin endpoints
        for role_name, user_data in users.items():
            if role_name == "admin":
                continue  # Skip admin user

            headers = {"Authorization": f"Bearer {user_data['token']}"}

            for endpoint, method in admin_endpoints:
                if method == "GET":
                    response = client.get(endpoint, headers=headers)
                elif method == "POST":
                    response = client.post(endpoint, headers=headers, json={})
                elif method == "PUT":
                    response = client.put(endpoint, headers=headers, json={})

                # Should be forbidden for non-admin users
                assert response.status_code in [
                    status.HTTP_403_FORBIDDEN,
                    status.HTTP_401_UNAUTHORIZED,
                ], f"{role_name} should not access {endpoint}"

    def test_system_level_privilege_escalation(self, client, setup_vertical_escalation):
        """Test attempts to gain system-level privileges."""
        users = setup_vertical_escalation
        observer = users["observer"]

        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Attempt to access system internals
        system_attacks = [
            # Environment variable access
            {"action": "get_env", "key": "SECRET_KEY"},
            {"action": "set_env", "key": "DEBUG", "value": "true"},
            # Configuration manipulation
            {"action": "reload_config"},
            {
                "action": "set_config",
                "key": "security.enabled",
                "value": False,
            },
            # Process control
            {"action": "restart_service"},
            {"action": "execute_command", "cmd": "whoami"},
        ]

        for attack in system_attacks:
            response = client.post("/api/v1/system/execute", headers=headers, json=attack)

            # Should be completely blocked
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,  # Endpoint might not exist
                status.HTTP_401_UNAUTHORIZED,
            ]


class TestHorizontalPrivilegeEscalation:
    """Test horizontal privilege escalation attacks (cross-user access)."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def setup_horizontal_escalation(self):
        """Set up test scenario with multiple users and their resources."""
        users = {}

        # Create multiple users with same role but different data
        for i in range(3):
            username = f"user_{i}"
            user = auth_manager.register_user(
                username=username,
                email=f"{username}@test.com",
                password=f"User{i}Pass123!",
                role=UserRole.RESEARCHER,
            )

            token = auth_manager.create_access_token(user)
            users[username] = {"user": user, "token": token, "resources": []}

            # Create resources for each user
            client = TestClient(app)
            headers = {"Authorization": f"Bearer {token}"}

            # Create agents
            for j in range(2):
                response = client.post(
                    "/api/v1/agents",
                    headers=headers,
                    json={
                        "name": f"Agent_{i}_{j}",
                        "template": "basic",
                        "parameters": {
                            "secret_data": f"confidential_{i}_{j}",
                            "user_specific": f"user_{i}_private_info",
                        },
                    },
                )
                if response.status_code == status.HTTP_201_CREATED:
                    users[username]["resources"].append(response.json())

        return users

    def test_cross_user_data_access(self, client, setup_horizontal_escalation):
        """Test accessing another user's data."""
        users = setup_horizontal_escalation

        user_0 = users["user_0"]
        user_1 = users["user_1"]

        # User 0 tries to access User 1's resources
        headers = {"Authorization": f"Bearer {user_0['token']}"}

        for resource in user_1["resources"]:
            # Try to read another user's agent
            response = client.get(f"/api/v1/agents/{resource['id']}", headers=headers)

            # Should be forbidden or not found
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]

            # Try to modify another user's agent
            response = client.put(
                f"/api/v1/agents/{resource['id']}",
                headers=headers,
                json={"name": "Hijacked_Agent"},
            )

            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]

            # Try to delete another user's agent
            response = client.delete(f"/api/v1/agents/{resource['id']}", headers=headers)

            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]

    def test_resource_ownership_bypass(self, client, setup_horizontal_escalation):
        """Test bypassing resource ownership checks."""
        users = setup_horizontal_escalation

        user_0 = users["user_0"]
        user_1 = users["user_1"]

        headers_0 = {"Authorization": f"Bearer {user_0['token']}"}

        # Ownership bypass attempts
        bypass_attempts = [
            # Direct ID manipulation
            {"agent_id": user_1["resources"][0]["id"]},
            # Wildcard attempts
            {"agent_id": "*"},
            {"agent_id": "%"},
            # Array injection
            {
                "agent_id": [
                    user_0["resources"][0]["id"],
                    user_1["resources"][0]["id"],
                ]
            },
            # SQL-like patterns
            {"agent_id": f"{user_1['resources'][0]['id']}' OR '1'='1"},
            # Path traversal
            {"agent_id": f"../users/{user_1['user'].id}/agents/{user_1['resources'][0]['id']}"},
        ]

        for attempt in bypass_attempts:
            # Try various endpoints with bypass attempts
            response = client.post(
                "/api/v1/agents/action",
                headers=headers_0,
                json={**attempt, "action": "view"},
            )

            # Should not allow access to other user's resources
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                # Ensure no data from user_1 is returned
                assert "user_1_private_info" not in str(data)
                assert user_1["user"].username not in str(data)

    def test_multi_tenant_isolation_bypass(self, client, setup_horizontal_escalation):
        """Test bypassing multi-tenant isolation."""
        users = setup_horizontal_escalation

        # Create tenant-like structure

        # User from tenant B tries to access tenant A data
        attacker = users["user_2"]
        headers = {"Authorization": f"Bearer {attacker['token']}"}

        # Attempt to list all resources (should be filtered)
        response = client.get("/api/v1/agents", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            agents = response.json()
            # Verify only sees own agents
            for agent in agents:
                assert "user_2" in agent.get("name", "")
                assert "user_0" not in agent.get("name", "")
                assert "user_1" not in agent.get("name", "")

    def test_session_hijacking_attempts(self, client, setup_horizontal_escalation):
        """Test session hijacking and impersonation attempts."""
        users = setup_horizontal_escalation

        user_0 = users["user_0"]
        user_1 = users["user_1"]

        # Attempt to use user_1's session/data with user_0's token
        headers = {"Authorization": f"Bearer {user_0['token']}"}

        # Session hijacking attempts
        hijack_attempts = [
            # Try to set session cookies
            {"Cookie": f"session={user_1['token']}"},
            # Try to add impersonation headers
            {"X-Impersonate-User": user_1["user"].username},
            {"X-Act-As": user_1["user"].id},
            {"X-Sudo-User": user_1["user"].username},
            # Try to override user context
            {"X-User-Id": str(user_1["user"].id)},
            {"X-User-Context": json.dumps({"user_id": user_1["user"].id})},
        ]

        for hijack_header in hijack_attempts:
            # Merge with auth header
            attack_headers = {**headers, **hijack_header}

            # Try to access as hijacked user
            response = client.get("/api/v1/users/me", headers=attack_headers)

            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                # Verify still returns user_0's data, not user_1's
                assert data.get("username") == user_0["user"].username
                assert data.get("username") != user_1["user"].username


class TestTokenBasedEscalation:
    """Test token manipulation and escalation attacks."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def setup_token_escalation(self):
        """Set up token-based testing scenario."""
        # Create users with different roles
        observer = auth_manager.register_user(
            username="token_observer",
            email="token_observer@test.com",
            password="Observer123!",
            role=UserRole.OBSERVER,
        )

        admin = auth_manager.register_user(
            username="token_admin",
            email="token_admin@test.com",
            password="Admin123!",
            role=UserRole.ADMIN,
        )

        return {
            "observer": {
                "user": observer,
                "token": auth_manager.create_access_token(observer),
                "refresh": auth_manager.create_refresh_token(observer),
            },
            "admin": {
                "user": admin,
                "token": auth_manager.create_access_token(admin),
                "refresh": auth_manager.create_refresh_token(admin),
            },
        }

    def test_jwt_role_manipulation(self, client, setup_token_escalation):
        """Test manipulating JWT to change role."""
        users = setup_token_escalation
        observer = users["observer"]

        # Decode the observer token
        original_token = observer["token"]

        # Try various JWT manipulation attacks
        manipulation_attempts = []

        # Attempt 1: Change algorithm to 'none'
        try:
            header = {"alg": "none", "typ": "JWT"}
            decoded = jwt.decode(original_token, options={"verify_signature": False})
            decoded["role"] = "admin"
            manipulation_attempts.append(
                base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
                + "."
                + base64.urlsafe_b64encode(json.dumps(decoded).encode()).decode().rstrip("=")
                + "."
            )
        except Exception:
            pass

        # Attempt 2: Use HS256 with public key as secret (if RS256 is used)
        try:
            decoded = jwt.decode(original_token, options={"verify_signature": False})
            decoded["role"] = "admin"
            # Try to sign with a known/guessable secret
            for secret in ["secret", "key", "jwt_secret", "dev_jwt_secret"]:
                manipulation_attempts.append(jwt.encode(decoded, secret, algorithm="HS256"))
        except Exception:
            pass

        # Attempt 3: Modify existing token
        try:
            parts = original_token.split(".")
            if len(parts) == 3:
                payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
                payload["role"] = "admin"
                payload["permissions"] = ["admin_system"]
                new_payload = (
                    base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
                )
                manipulation_attempts.append(f"{parts[0]}.{new_payload}.{parts[2]}")
        except Exception:
            pass

        # Test each manipulated token
        for manipulated_token in manipulation_attempts:
            headers = {"Authorization": f"Bearer {manipulated_token}"}

            # Try to access admin endpoint
            response = client.get("/api/v1/system/config", headers=headers)

            # Should be rejected
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
            ]

    def test_token_replay_attacks(self, client, setup_token_escalation):
        """Test token replay and reuse attacks."""
        users = setup_token_escalation
        admin = users["admin"]

        # Simulate token revocation/logout
        headers = {"Authorization": f"Bearer {admin['token']}"}
        client.post("/api/v1/auth/logout", headers=headers)

        # Try to reuse the logged-out token
        response = client.get("/api/v1/users/me", headers=headers)

        # Should be rejected
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Test expired token replay
        with patch("auth.security_implementation.datetime") as mock_datetime:
            # Set time to future (after token expiry)
            future_time = datetime.now(timezone.utc) + timedelta(days=30)
            mock_datetime.now.return_value = future_time
            mock_datetime.utcnow.return_value = future_time

            response = client.get("/api/v1/users/me", headers=headers)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_token_substitution_attacks(self, client, setup_token_escalation):
        """Test substituting tokens between different contexts."""
        users = setup_token_escalation

        observer = users["observer"]
        admin = users["admin"]

        # Try to use refresh token as access token
        headers = {"Authorization": f"Bearer {observer['refresh']}"}
        response = client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Try to use access token as refresh token
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": observer["token"]})
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_400_BAD_REQUEST,
        ]

        # Try to mix tokens from different users
        mixed_token_attempts = [
            # Observer token with admin's signature
            observer["token"].rsplit(".", 1)[0] + "." + admin["token"].rsplit(".", 1)[1],
            # Admin header with observer payload
            admin["token"].split(".", 1)[0] + "." + observer["token"].split(".", 1)[1],
        ]

        for mixed_token in mixed_token_attempts:
            headers = {"Authorization": f"Bearer {mixed_token}"}
            response = client.get("/api/v1/system/config", headers=headers)
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
            ]

    def test_refresh_token_abuse(self, client, setup_token_escalation):
        """Test refresh token abuse scenarios."""
        users = setup_token_escalation
        observer = users["observer"]

        # Test infinite refresh attempt
        refresh_token = observer["refresh"]
        used_tokens = set()

        for i in range(10):  # Try to refresh multiple times
            response = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})

            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                new_access = data.get("access_token")
                new_refresh = data.get("refresh_token")

                # Check for token reuse
                assert new_access not in used_tokens
                used_tokens.add(new_access)

                # Update refresh token if rotating
                if new_refresh and new_refresh != refresh_token:
                    refresh_token = new_refresh
            else:
                # Should eventually fail or rate limit
                assert response.status_code in [
                    status.HTTP_401_UNAUTHORIZED,
                    status.HTTP_429_TOO_MANY_REQUESTS,
                ]
                break

        # Test using old refresh token after rotation
        if observer["refresh"] != refresh_token:
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": observer["refresh"]},  # Old token
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestAPIEndpointEscalation:
    """Test API endpoint privilege escalation attacks."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def setup_api_escalation(self):
        """Set up API testing scenario."""
        observer = auth_manager.register_user(
            username="api_observer",
            email="api_observer@test.com",
            password="Observer123!",
            role=UserRole.OBSERVER,
        )

        return {
            "observer": {
                "user": observer,
                "token": auth_manager.create_access_token(observer),
            }
        }

    def test_parameter_manipulation(self, client, setup_api_escalation):
        """Test parameter manipulation for privilege escalation."""
        users = setup_api_escalation
        observer = users["observer"]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Parameter pollution attempts
        param_attacks = [
            # Duplicate parameters
            "/api/v1/agents?role=observer&role=admin",
            # Array injection
            "/api/v1/agents?permissions[]=view&permissions[]=admin",
            # Object injection
            "/api/v1/agents?user[role]=admin",
            # Special characters
            "/api/v1/agents?role=observer%00admin",
            # Unicode tricks
            "/api/v1/agents?role=obs%E2%80%8Eerver&admin=true",
            # Parameter override
            "/api/v1/agents?_method=DELETE",
        ]

        for attack_url in param_attacks:
            response = client.get(attack_url, headers=headers)

            # Should not grant elevated access
            if response.status_code == status.HTTP_200_OK:
                # Verify response doesn't include admin-level data
                data = response.json()
                assert "admin" not in str(data).lower()

    def test_http_method_tampering(self, client, setup_api_escalation):
        """Test HTTP method tampering attacks."""
        users = setup_api_escalation
        observer = users["observer"]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Admin endpoint that should only allow specific methods
        admin_endpoint = "/api/v1/system/config"

        # Try various HTTP methods
        methods = [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "HEAD",
            "OPTIONS",
            "TRACE",
            "CONNECT",
        ]
        custom_methods = ["GETS", "GETX", "RETRIEVE", "GRAB", "FETCH"]

        for method in methods + custom_methods:
            # Use raw request to test custom methods
            if method in [
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "PATCH",
                "HEAD",
                "OPTIONS",
            ]:
                response = client.request(method, admin_endpoint, headers=headers)
            else:
                # Custom method via override header
                override_headers = {
                    **headers,
                    "X-HTTP-Method-Override": method,
                    "X-Method-Override": method,
                    "X-HTTP-Method": method,
                }
                response = client.post(admin_endpoint, headers=override_headers)

            # Should not allow unauthorized access
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
                status.HTTP_405_METHOD_NOT_ALLOWED,
                status.HTTP_404_NOT_FOUND,
            ]

    def test_header_injection_attacks(self, client, setup_api_escalation):
        """Test header injection for privilege escalation."""
        users = setup_api_escalation
        observer = users["observer"]
        base_headers = {"Authorization": f"Bearer {observer['token']}"}

        # Header injection attempts
        header_attacks = [
            # Role override attempts
            {"X-User-Role": "admin"},
            {"X-Override-Role": "admin"},
            {"X-Forwarded-Role": "admin"},
            # Permission injection
            {"X-User-Permissions": "admin_system,delete_all"},
            {"X-Add-Permission": "admin_system"},
            # Identity spoofing
            {"X-Real-User": "admin"},
            {"X-Forwarded-User": "admin_user"},
            {"X-Original-User": "system"},
            # Internal headers
            {"X-Internal-Request": "true"},
            {"X-Admin-Override": "true"},
            {"X-Bypass-Auth": "true"},
            # Source spoofing
            {"X-Forwarded-For": "127.0.0.1"},
            {"X-Real-IP": "::1"},
            {"X-Originating-IP": "localhost"},
        ]

        for attack_headers in header_attacks:
            headers = {**base_headers, **attack_headers}

            # Try to access admin endpoint
            response = client.get("/api/v1/system/config", headers=headers)

            # Should not grant access
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
            ]

    def test_path_traversal_attacks(self, client, setup_api_escalation):
        """Test path traversal for accessing unauthorized endpoints."""
        users = setup_api_escalation
        observer = users["observer"]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Path traversal attempts
        path_attacks = [
            # Directory traversal
            "/api/v1/../admin/system/config",
            "/api/v1/agents/../../../admin/config",
            "/api/v1/agents/%2e%2e%2f%2e%2e%2fadmin/config",
            # Path confusion
            "/api/v1//system/config",
            "/api/v1/./system/config",
            "/api/v1/agents/;/system/config",
            # URL encoding tricks
            "/api/v1%2fsystem%2fconfig",
            "/api/v1/system%2fconfig",
            # Unicode normalization
            "/api/v\u0131/system/config",  # Unicode character
            # Case variations (if case-insensitive)
            "/API/V1/SYSTEM/CONFIG",
            "/Api/V1/System/Config",
        ]

        for attack_path in path_attacks:
            response = client.get(attack_path, headers=headers)

            # Should not allow access to admin endpoints
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]


class TestDatabaseLevelEscalation:
    """Test database-level privilege escalation attacks."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def setup_db_escalation(self):
        """Set up database testing scenario."""
        observer = auth_manager.register_user(
            username="db_observer",
            email="db_observer@test.com",
            password="Observer123!",
            role=UserRole.OBSERVER,
        )

        researcher = auth_manager.register_user(
            username="db_researcher",
            email="db_researcher@test.com",
            password="Researcher123!",
            role=UserRole.RESEARCHER,
        )

        return {
            "observer": {
                "user": observer,
                "token": auth_manager.create_access_token(observer),
            },
            "researcher": {
                "user": researcher,
                "token": auth_manager.create_access_token(researcher),
            },
        }

    def test_sql_injection_privilege_escalation(self, client, setup_db_escalation):
        """Test SQL injection attempts for privilege escalation."""
        users = setup_db_escalation
        observer = users["observer"]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # SQL injection payloads targeting privilege escalation
        sql_injections = [
            # Union-based privilege escalation
            "' UNION SELECT * FROM users WHERE role='admin'--",
            "' UNION SELECT username,password,role FROM users--",
            # Subquery injection
            "'; UPDATE users SET role='admin' WHERE username='db_observer'--",
            # Boolean-based extraction
            "' OR role='admin' AND '1'='1",
            # Time-based extraction
            "' OR IF(role='admin', SLEEP(5), 0)--",
            # Stacked queries
            "'; INSERT INTO user_roles VALUES ('db_observer', 'admin'); --",
            # Comment injection
            "' /* admin */ OR '1'='1",
            # Function injection
            "' OR CURRENT_USER()='root'--",
        ]

        # Test various endpoints with SQL injection
        for payload in sql_injections:
            # Search endpoint
            response = client.get(f"/api/v1/agents/search?query={payload}", headers=headers)

            # Name field
            response = client.post(
                "/api/v1/agents",
                headers=headers,
                json={"name": payload, "template": "basic"},
            )

            # Filter parameters
            response = client.get(f"/api/v1/agents?filter={payload}", headers=headers)

            # Should not expose admin data or change roles
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert "admin" not in str(data).lower()

                # Verify role hasn't changed
                me_response = client.get("/api/v1/users/me", headers=headers)
                if me_response.status_code == status.HTTP_200_OK:
                    user_data = me_response.json()
                    assert user_data.get("role") == "observer"

    def test_database_role_manipulation(self, client, setup_db_escalation):
        """Test direct database role manipulation attempts."""
        users = setup_db_escalation
        observer = users["observer"]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Attempts to manipulate database roles
        role_attacks = [
            # Direct role update
            {"action": "update_role", "role": "admin"},
            # Role array manipulation
            {"roles": ["observer", "admin"]},
            # Permission set manipulation
            {"permissions": {"add": ["admin_system"]}},
            # Group membership
            {"groups": ["admins", "superusers"]},
            # Privilege flags
            {"is_superuser": True, "is_staff": True},
        ]

        for attack in role_attacks:
            # Try through user update
            response = client.patch(
                f"/api/v1/users/{observer['user'].id}",
                headers=headers,
                json=attack,
            )

            # Should not succeed
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]

    def test_stored_procedure_abuse(self, client, setup_db_escalation):
        """Test abuse of stored procedures for escalation."""
        users = setup_db_escalation
        observer = users["observer"]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Stored procedure calls that might exist
        proc_attacks = [
            {"procedure": "grant_admin", "params": {"user": "db_observer"}},
            {
                "procedure": "add_role",
                "params": {"user": "db_observer", "role": "admin"},
            },
            {
                "procedure": "execute_as_admin",
                "params": {"command": "SELECT * FROM users"},
            },
            {
                "procedure": "backup_database",
                "params": {"include_passwords": True},
            },
        ]

        for attack in proc_attacks:
            response = client.post("/api/v1/database/execute", headers=headers, json=attack)

            # Should be blocked
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_401_UNAUTHORIZED,
            ]

    def test_database_connection_hijacking(self, client, setup_db_escalation):
        """Test database connection hijacking attempts."""
        users = setup_db_escalation
        observer = users["observer"]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Connection hijacking attempts
        connection_attacks = [
            # Connection string injection
            {"db_url": "postgresql://admin:password@localhost/db"},
            # Connection pooling manipulation
            {"connection_pool": "admin_pool"},
            # Schema switching
            {"schema": "admin_schema"},
            {"database": "admin_db"},
            # Transaction isolation manipulation
            {"isolation_level": "READ_UNCOMMITTED"},
            {"autocommit": True},
        ]

        for attack in connection_attacks:
            # Try to manipulate connection settings
            response = client.post("/api/v1/database/connection", headers=headers, json=attack)

            # Should not allow connection manipulation
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_401_UNAUTHORIZED,
            ]

    @pytest.mark.asyncio
    async def test_orm_bypass_attacks(self, client, setup_db_escalation):
        """Test ORM bypass attempts for direct database access."""
        users = setup_db_escalation
        observer = users["observer"]

        # Test raw SQL execution attempts
        with patch("database.session.get_db") as mock_db:
            # Mock database session
            mock_session = MagicMock()
            mock_db.return_value = mock_session

            # Ensure ORM prevents raw SQL execution by unauthorized users
            headers = {"Authorization": f"Bearer {observer['token']}"}

            # Attempt raw query execution
            response = client.post(
                "/api/v1/database/raw",
                headers=headers,
                json={"query": "UPDATE users SET role='admin' WHERE username='db_observer'"},
            )

            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]


class TestAdvancedEscalationScenarios:
    """Test advanced and combined privilege escalation scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_chained_escalation_attack(self, client):
        """Test chained attacks combining multiple techniques."""
        # Register as basic user
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "chained_attacker",
                "email": "chained@test.com",
                "password": "Chained123!",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED

        # Login
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "chained_attacker", "password": "Chained123!"},
        )

        token = response.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}

        # Chain 1: Try information disclosure to find admin users
        response = client.get("/api/v1/users", headers=headers)
        if response.status_code == status.HTTP_200_OK:
            # Should not see other users as basic user
            users = response.json()
            admin_users = [u for u in users if u.get("role") == "admin"]
            assert len(admin_users) == 0

        # Chain 2: Try to enumerate valid endpoints
        admin_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/admin/system",
            "/admin/api/v1/users",
            "/internal/admin/users",
        ]

        for endpoint in admin_endpoints:
            response = client.get(endpoint, headers=headers)
            # Should not reveal admin endpoints
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]

    def test_race_condition_privilege_escalation(self, client):
        """Test race condition attacks for privilege escalation."""
        import threading

        # Register user
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "race_attacker",
                "email": "race@test.com",
                "password": "Race123!",
            },
        )

        user_id = response.json().get("id")

        # Login
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "race_attacker", "password": "Race123!"},
        )

        token = response.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}

        results = []

        def attempt_escalation():
            """Attempt to escalate privileges during profile update."""
            response = client.patch(
                f"/api/v1/users/{user_id}",
                headers=headers,
                json={"email": "race_new@test.com", "role": "admin"},
            )
            results.append(response.status_code)

        # Launch multiple concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=attempt_escalation)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no successful privilege escalation
        response = client.get("/api/v1/users/me", headers=headers)
        if response.status_code == status.HTTP_200_OK:
            user_data = response.json()
            assert user_data.get("role") != "admin"

    def test_timing_attack_privilege_disclosure(self, client):
        """Test timing attacks to discover privileged users."""
        import time

        # Create users with different roles
        users_to_test = [
            ("timing_admin", "admin@timing.com", "Admin123!", UserRole.ADMIN),
            ("timing_user", "user@timing.com", "User123!", UserRole.OBSERVER),
        ]

        for username, email, password, role in users_to_test:
            auth_manager.register_user(username=username, email=email, password=password, role=role)

        # Measure login times
        timings = {}

        for username, _, password, _ in users_to_test:
            start_time = time.time()
            client.post(
                "/api/v1/auth/login",
                json={"username": username, "password": password},
            )
            end_time = time.time()

            timings[username] = end_time - start_time

        # Timing differences should be negligible (no role-based timing leaks)
        time_diff = abs(timings["timing_admin"] - timings["timing_user"])
        assert time_diff < 0.1  # Less than 100ms difference

    def test_cache_poisoning_escalation(self, client):
        """Test cache poisoning for privilege escalation."""
        # Register users
        observer = auth_manager.register_user(
            username="cache_observer",
            email="cache_observer@test.com",
            password="Observer123!",
            role=UserRole.OBSERVER,
        )

        auth_manager.register_user(
            username="cache_admin",
            email="cache_admin@test.com",
            password="Admin123!",
            role=UserRole.ADMIN,
        )

        observer_token = auth_manager.create_access_token(observer)

        # Attempt cache poisoning via headers
        poison_headers = {
            "Authorization": f"Bearer {observer_token}",
            "X-Cache-Key": "user:cache_admin",
            "X-Forward-Cache": "role:admin",
            "Cache-Control": "private, role=admin",
        }

        # Make request with poisoned headers
        response = client.get("/api/v1/users/me", headers=poison_headers)

        # Verify response is for correct user
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data.get("username") == "cache_observer"
            assert data.get("role") == "observer"

        # Make subsequent request to check if cache was poisoned
        normal_headers = {"Authorization": f"Bearer {observer_token}"}
        response = client.get("/api/v1/users/me", headers=normal_headers)

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data.get("role") == "observer"  # Should not be escalated


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

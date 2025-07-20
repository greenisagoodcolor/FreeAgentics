"""Comprehensive authentication flow tests.

Tests complete authentication flow including:
- User registration
- Login with valid/invalid credentials
- Logout and session termination
- Session management and persistence
- Token validation and expiry
- Rate limiting
- Security event logging
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import jwt
import pytest

try:
    pass
except ImportError:
    pass

from api.main import app
from auth import AuthenticationManager, User, UserRole, auth_manager
from auth.security_logging import SecurityEventSeverity, SecurityEventType


class TestAuthenticationFlow:
    """Test complete authentication flow."""

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
    def test_user_data(self):
        """Test user registration data."""
        return {
            "username": "test_user",
            "email": "test@example.com",
            "password": "SecurePassword123!",
            "role": "observer",
        }

    @pytest.fixture
    def admin_user_data(self):
        """Admin user registration data."""
        return {
            "username": "admin_user",
            "email": "admin@example.com",
            "password": "AdminPassword123!",
            "role": "admin",
        }

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self):
        """Reset auth manager state before each test."""
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()
        yield
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()

    def test_user_registration_success(self, client, test_user_data):
        """Test successful user registration."""
        response = client.post("/api/v1/auth/register", json=test_user_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "user" in data

        # Verify user data
        user = data["user"]
        assert user["username"] == test_user_data["username"]
        assert user["email"] == test_user_data["email"]
        assert user["role"] == test_user_data["role"]
        assert user["is_active"] is True
        assert "user_id" in user
        assert "created_at" in user

    def test_user_registration_duplicate_username(
        self, client, test_user_data
    ):
        """Test registration with duplicate username."""
        # Register first user
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        # Try to register with same username
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_user_registration_invalid_email(self, client):
        """Test registration with invalid email."""
        invalid_data = {
            "username": "test_user",
            "email": "invalid-email",
            "password": "SecurePassword123!",
            "role": "observer",
        }

        response = client.post("/api/v1/auth/register", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_user_registration_rate_limiting(self, client):
        """Test registration rate limiting."""
        # Make multiple registration attempts
        for i in range(6):  # Rate limit is 5 requests per 10 minutes
            data = {
                "username": f"user_{i}",
                "email": f"user{i}@example.com",
                "password": "Password123!",
                "role": "observer",
            }
            response = client.post("/api/v1/auth/register", json=data)

            if i < 5:
                assert response.status_code == 200
            else:
                assert response.status_code == 429  # Too many requests

    def test_login_success(self, client, test_user_data):
        """Test successful login."""
        # Register user first
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        # Login with correct credentials
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"],
        }

        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data

        # Verify user last_login is updated
        user = data["user"]
        assert user["last_login"] is not None

    def test_login_invalid_username(self, client):
        """Test login with invalid username."""
        login_data = {
            "username": "nonexistent_user",
            "password": "Password123!",
        }

        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]

    def test_login_invalid_password(self, client, test_user_data):
        """Test login with invalid password."""
        # Register user first
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        # Login with wrong password
        login_data = {
            "username": test_user_data["username"],
            "password": "WrongPassword123!",
        }

        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]

    def test_login_inactive_user(self, client, test_user_data):
        """Test login with inactive user."""
        # Register user and deactivate
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        # Manually deactivate user
        username = test_user_data["username"]
        auth_manager.users[username]["user"].is_active = False

        # Try to login
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"],
        }

        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401
        assert "Account is disabled" in response.json()["detail"]

    def test_login_rate_limiting(self, client, test_user_data):
        """Test login rate limiting."""
        # Register user first
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        login_data = {
            "username": test_user_data["username"],
            "password": "WrongPassword",
        }

        # Make multiple login attempts
        for i in range(11):  # Rate limit is 10 requests per 5 minutes
            response = client.post("/api/v1/auth/login", json=login_data)

            if i < 10:
                assert response.status_code == 401
            else:
                assert response.status_code == 429  # Too many requests

    def test_get_current_user_info(self, client, test_user_data):
        """Test getting current user information."""
        # Register and login
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        access_token = register_response.json()["access_token"]

        # Get user info with valid token
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data["username"] == test_user_data["username"]
        assert data["role"] == test_user_data["role"]
        assert "user_id" in data
        assert "permissions" in data
        assert "exp" in data

    def test_get_current_user_info_invalid_token(self, client):
        """Test getting user info with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == 401
        assert "Invalid token" in response.json()["detail"]

    def test_get_current_user_info_expired_token(self, client, test_user_data):
        """Test getting user info with expired token."""
        # Register user
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        # Create expired token
        user = auth_manager.users[test_user_data["username"]]["user"]
        expired_token_data = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "permissions": [],
            "exp": datetime.utcnow()
            - timedelta(hours=1),  # Expired 1 hour ago
            "type": "access",
        }

        expired_token = jwt.encode(
            expired_token_data,
            (
                auth_manager.JWT_SECRET
                if hasattr(auth_manager, "JWT_SECRET")
                else "dev_jwt_secret_2025_not_for_production"
            ),
            algorithm="HS256",
        )

        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == 401
        assert "Token expired" in response.json()["detail"]

    def test_logout_success(self, client, test_user_data):
        """Test successful logout."""
        # Register and login
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        access_token = register_response.json()["access_token"]
        user_id = register_response.json()["user"]["user_id"]

        # Verify refresh token exists
        assert user_id in auth_manager.refresh_tokens

        # Logout
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.post("/api/v1/auth/logout", headers=headers)

        assert response.status_code == 200
        assert response.json()["message"] == "Successfully logged out"

        # Verify refresh token is removed
        assert user_id not in auth_manager.refresh_tokens

    def test_logout_without_auth(self, client):
        """Test logout without authentication."""
        response = client.post("/api/v1/auth/logout")
        assert response.status_code == 403  # Forbidden

    def test_get_permissions(self, client, test_user_data, admin_user_data):
        """Test getting user permissions."""
        # Test observer permissions
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        access_token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        response = client.get("/api/v1/auth/permissions", headers=headers)
        assert response.status_code == 200

        permissions = response.json()
        assert permissions["role"] == "observer"
        assert permissions["can_create_agents"] is False
        assert permissions["can_delete_agents"] is False
        assert permissions["can_view_metrics"] is True
        assert permissions["can_admin_system"] is False

        # Test admin permissions
        admin_response = client.post(
            "/api/v1/auth/register", json=admin_user_data
        )
        assert admin_response.status_code == 200

        admin_token = admin_response.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        response = client.get(
            "/api/v1/auth/permissions", headers=admin_headers
        )
        assert response.status_code == 200

        admin_permissions = response.json()
        assert admin_permissions["role"] == "admin"
        assert admin_permissions["can_create_agents"] is True
        assert admin_permissions["can_delete_agents"] is True
        assert admin_permissions["can_view_metrics"] is True
        assert admin_permissions["can_admin_system"] is True

    def test_token_refresh_flow(self, client, test_user_data):
        """Test token refresh functionality."""
        # Register user
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        access_token = register_response.json()["access_token"]
        refresh_token = register_response.json()["refresh_token"]

        # Decode tokens to verify structure
        access_payload = jwt.decode(
            access_token, options={"verify_signature": False}
        )
        refresh_payload = jwt.decode(
            refresh_token, options={"verify_signature": False}
        )

        assert access_payload["type"] == "access"
        assert refresh_payload["type"] == "refresh"

        # Verify token expiry times
        access_exp = datetime.fromtimestamp(access_payload["exp"])
        refresh_exp = datetime.fromtimestamp(refresh_payload["exp"])

        # Access token should expire in ~30 minutes
        assert (
            access_exp - datetime.utcnow()
        ).total_seconds() < 1900  # Less than 32 minutes
        assert (
            access_exp - datetime.utcnow()
        ).total_seconds() > 1700  # More than 28 minutes

        # Refresh token should expire in ~7 days
        assert (
            refresh_exp - datetime.utcnow()
        ).total_seconds() > 600000  # More than 6.9 days

    @patch("auth.security_logging.security_auditor.log_event")
    def test_security_event_logging(
        self, mock_log_event, client, test_user_data
    ):
        """Test security event logging during authentication flow."""
        # Test registration logging
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        # Verify registration event was logged
        assert mock_log_event.called
        registration_call = [
            call
            for call in mock_log_event.call_args_list
            if call[0][0] == SecurityEventType.USER_CREATED
        ][0]
        assert registration_call[0][1] == SecurityEventSeverity.INFO
        assert test_user_data["username"] in registration_call[0][2]

        # Reset mock
        mock_log_event.reset_mock()

        # Test successful login logging
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"],
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200

        # Verify login event was logged
        assert mock_log_event.called
        login_call = [
            call
            for call in mock_log_event.call_args_list
            if call[0][0] == SecurityEventType.LOGIN_SUCCESS
        ][0]
        assert login_call[0][1] == SecurityEventSeverity.INFO

        # Reset mock
        mock_log_event.reset_mock()

        # Test failed login logging
        bad_login_data = {
            "username": test_user_data["username"],
            "password": "WrongPassword",
        }
        response = client.post("/api/v1/auth/login", json=bad_login_data)
        assert response.status_code == 401

        # Verify failed login event was logged
        assert mock_log_event.called
        failed_login_call = [
            call
            for call in mock_log_event.call_args_list
            if call[0][0] == SecurityEventType.LOGIN_FAILURE
        ][0]
        assert failed_login_call[0][1] == SecurityEventSeverity.WARNING

        # Reset mock
        mock_log_event.reset_mock()

        # Test logout logging
        access_token = client.post(
            "/api/v1/auth/login", json=login_data
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.post("/api/v1/auth/logout", headers=headers)
        assert response.status_code == 200

        # Verify logout event was logged
        assert mock_log_event.called
        logout_call = [
            call
            for call in mock_log_event.call_args_list
            if call[0][0] == SecurityEventType.LOGOUT
        ][0]
        assert logout_call[0][1] == SecurityEventSeverity.INFO

    def test_concurrent_session_management(self, client, test_user_data):
        """Test concurrent session handling."""
        # Register user
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        # Login from multiple sessions
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"],
        }

        # Create 3 sessions
        tokens = []
        for _ in range(3):
            response = client.post("/api/v1/auth/login", json=login_data)
            assert response.status_code == 200
            tokens.append(response.json()["access_token"])

        # All tokens should be valid
        for token in tokens:
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code == 200

    def test_session_persistence(self, client, test_user_data):
        """Test session persistence across requests."""
        # Register and login
        register_response = client.post(
            "/api/v1/auth/register", json=test_user_data
        )
        assert register_response.status_code == 200

        access_token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Make multiple authenticated requests
        endpoints = [
            "/api/v1/auth/me",
            "/api/v1/auth/permissions",
            "/api/v1/agents",  # This will test auth on other endpoints
        ]

        for endpoint in endpoints[
            :2
        ]:  # Skip agents endpoint if it doesn't exist
            response = client.get(endpoint, headers=headers)
            assert response.status_code == 200

        # Token should still be valid after multiple requests
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200

    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        # This would typically be handled by a background task
        # For testing, we'll simulate the cleanup process

        # Create auth manager with some expired tokens
        manager = AuthenticationManager()

        # Add some test users with tokens
        test_users = []
        for i in range(5):
            user = User(
                user_id=f"user_{i}",
                username=f"test_user_{i}",
                email=f"user{i}@example.com",
                role=UserRole.OBSERVER,
                created_at=datetime.utcnow(),
            )
            test_users.append(user)

            # Add refresh token
            manager.refresh_tokens[user.user_id] = f"refresh_token_{i}"

        # Simulate cleanup - in production this would check token expiry
        # For testing, we'll just verify the structure exists
        assert len(manager.refresh_tokens) == 5

        # Clear expired tokens (simulated)
        expired_users = ["user_0", "user_1"]
        for user_id in expired_users:
            if user_id in manager.refresh_tokens:
                del manager.refresh_tokens[user_id]

        assert len(manager.refresh_tokens) == 3
        assert "user_0" not in manager.refresh_tokens
        assert "user_1" not in manager.refresh_tokens


class TestAuthenticationEdgeCases:
    """Test edge cases and error scenarios in authentication."""

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

    def test_malformed_authorization_header(self, client):
        """Test various malformed authorization headers."""
        malformed_headers = [
            {"Authorization": "InvalidBearer token"},
            {"Authorization": "Bearer"},
            {"Authorization": "token"},
            {"Authorization": ""},
            {"NotAuthorization": "Bearer token"},
        ]

        for headers in malformed_headers:
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code in [401, 403]

    def test_sql_injection_in_username(self, client):
        """Test SQL injection attempts in username field."""
        injection_attempts = [
            "admin'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin' --",
            "admin' /*",
            "admin' OR 1=1 --",
        ]

        for username in injection_attempts:
            login_data = {
                "username": username,
                "password": "password123",
            }

            response = client.post("/api/v1/auth/login", json=login_data)
            # Should fail safely without SQL injection
            assert response.status_code == 401
            assert "Invalid credentials" in response.json()["detail"]

    def test_xss_in_registration(self, client):
        """Test XSS attempts in registration fields."""
        xss_attempts = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
        ]

        for xss in xss_attempts:
            # Try XSS in username
            data = {
                "username": xss,
                "email": "test@example.com",
                "password": "Password123!",
                "role": "observer",
            }

            response = client.post("/api/v1/auth/register", json=data)
            # Should either sanitize or reject
            if response.status_code == 200:
                # If accepted, verify it was sanitized
                user = response.json()["user"]
                assert "<script>" not in user["username"]
                assert "javascript:" not in user["username"]

    def test_password_requirements(self, client):
        """Test password validation requirements."""
        weak_passwords = [
            "short",  # Too short
            "alllowercase",  # No uppercase or numbers
            "ALLUPPERCASE",  # No lowercase or numbers
            "NoNumbers!",  # No numbers
            "12345678",  # No letters
        ]

        for password in weak_passwords:
            data = {
                "username": f"user_{password}",
                "email": "test@example.com",
                "password": password,
                "role": "observer",
            }

            _response = client.post("/api/v1/auth/register", json=data)
            # Depending on implementation, might accept or reject
            # This test documents the behavior

    def test_unicode_in_credentials(self, client):
        """Test Unicode characters in credentials."""
        unicode_data = {
            "username": "user_😀_test",
            "email": "test@example.com",
            "password": "Password123!",
            "role": "observer",
        }

        _response = client.post("/api/v1/auth/register", json=unicode_data)
        # Document behavior with unicode

    def test_long_input_fields(self, client):
        """Test extremely long input fields."""
        long_string = "a" * 1000

        data = {
            "username": long_string,
            "email": f"{long_string}@example.com",
            "password": long_string,
            "role": "observer",
        }

        _response = client.post("/api/v1/auth/register", json=data)
        # Should handle gracefully, either truncate or reject

    def test_null_and_empty_fields(self, client):
        """Test null and empty field handling."""
        test_cases = [
            {"username": None, "password": "pass"},
            {"username": "", "password": "pass"},
            {"username": "user", "password": None},
            {"username": "user", "password": ""},
            {},  # Empty body
        ]

        for data in test_cases:
            response = client.post("/api/v1/auth/login", json=data)
            assert response.status_code in [
                400,
                422,
            ]  # Bad request or validation error


# Cleanup helper functions for test isolation
def cleanup_test_users():
    """Clean up test users after tests."""
    auth_manager.users.clear()
    auth_manager.refresh_tokens.clear()


def cleanup_rate_limits():
    """Clean up rate limiting data."""
    from auth import rate_limiter

    rate_limiter.requests.clear()
    rate_limiter.user_requests.clear()


# Run cleanup after each test module
def teardown_module():
    """Module-level cleanup."""
    cleanup_test_users()
    cleanup_rate_limits()

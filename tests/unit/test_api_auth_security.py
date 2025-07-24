"""Security-critical tests for API authentication endpoints following TDD principles.

This test suite covers API authentication security:
- Login endpoint security
- Token refresh security
- Logout and revocation
- Rate limiting
- CSRF protection
- Security headers
"""

import time
from unittest.mock import patch

import pytest
from fastapi import Depends, status
from fastapi.testclient import TestClient

from api.main import app
from auth.jwt_handler import jwt_handler
from auth.security_implementation import get_current_user


class TestLoginEndpointSecurity:
    """Test login endpoint security measures."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_auth_manager(self):
        """Mock auth manager for testing."""
        with patch("api.v1.auth.auth_manager") as mock:
            yield mock

    def test_successful_login_with_valid_credentials(self, client, mock_auth_manager):
        """Test successful login returns tokens."""
        # Arrange
        from unittest.mock import MagicMock
        mock_user = MagicMock()
        mock_user.user_id = "user-123"
        mock_user.username = "testuser"
        mock_user.role = "user"
        mock_user.permissions = ["read"]
        mock_user.dict.return_value = {
            "user_id": "user-123",
            "username": "testuser", 
            "role": "user",
            "permissions": ["read"]
        }
        mock_auth_manager.authenticate_user.return_value = mock_user
        mock_auth_manager.create_access_token.return_value = "test_access_token"
        mock_auth_manager.create_refresh_token.return_value = "test_refresh_token"
        mock_auth_manager.set_token_cookie.return_value = None
        mock_auth_manager.set_csrf_cookie.return_value = None
        
        # Mock jwt_handler for fingerprint generation
        mock_auth_manager.jwt_handler.generate_fingerprint.return_value = "test_fingerprint"
        
        # Mock csrf_protection
        mock_auth_manager.csrf_protection.generate_csrf_token.return_value = "test_csrf_token"

        login_data = {"username": "testuser", "password": "SecurePass123!"}

        # Act
        response = client.post("/api/v1/login", json=login_data)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

    def test_login_fails_with_invalid_credentials(self, client, mock_auth_manager):
        """Test login failure with invalid credentials."""
        # Arrange
        mock_auth_manager.authenticate_user.return_value = None

        login_data = {"username": "testuser", "password": "WrongPassword"}

        # Act
        response = client.post("/api/v1/login", json=login_data)

        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "access_token" not in data
        assert "detail" in data

    def test_login_validates_input_format(self, client):
        """Test that login validates input format."""
        # Arrange - Invalid input formats
        invalid_inputs = [
            {},  # Empty
            {"username": "test"},  # Missing password
            {"password": "test"},  # Missing username
            {"username": "", "password": "test"},  # Empty username
            {"username": "test", "password": ""},  # Empty password
            {"username": "a" * 256, "password": "test"},  # Too long
        ]

        for invalid_input in invalid_inputs:
            # Act
            response = client.post("/api/v1/login", json=invalid_input)

            # Assert
            assert response.status_code in [
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_401_UNAUTHORIZED,  # Empty username/password may pass validation but fail auth
            ]

    def test_login_prevents_timing_attacks(self, client, mock_auth_manager):
        """Test constant-time comparison prevents timing attacks."""
        # Arrange
        mock_auth_manager.authenticate_user.return_value = None

        # Measure time for valid username
        start1 = time.time()
        client.post(
            "/api/v1/login",
            json={"username": "validuser", "password": "wrongpass"},
        )
        time1 = time.time() - start1

        # Measure time for invalid username
        start2 = time.time()
        client.post(
            "/api/v1/login",
            json={"username": "invaliduser", "password": "wrongpass"},
        )
        time2 = time.time() - start2

        # Assert - Times should be similar (constant-time)
        assert abs(time1 - time2) < 0.1  # Within 100ms

    def test_login_rate_limiting(self, client, mock_auth_manager):
        """Test that login endpoint has rate limiting."""
        # Arrange
        mock_auth_manager.authenticate_user.return_value = None
        login_data = {"username": "test", "password": "wrong"}

        # Act - Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.post("/api/v1/login", json=login_data)
            responses.append(response.status_code)

        # Assert - Should hit rate limit
        assert status.HTTP_429_TOO_MANY_REQUESTS in responses

    def test_login_logs_failed_attempts(self, client, mock_auth_manager):
        """Test that failed login attempts are logged."""
        # Arrange
        mock_auth_manager.authenticate_user.return_value = None

        with patch("api.v1.auth.logger") as mock_logger:
            # Act
            response = client.post(
                "/api/v1/login",
                json={"username": "attacker", "password": "malicious"},
            )

        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        mock_logger.warning.assert_called()
        warning_msg = str(mock_logger.warning.call_args)
        assert "failed login" in warning_msg.lower()


class TestTokenRefreshSecurity:
    """Test token refresh endpoint security."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def authenticated_client(self, client):
        """Create client with valid tokens."""
        # Generate valid tokens
        access_token = jwt_handler.create_access_token(
            user_id="user-123", username="testuser", role="user", permissions=["read"]
        )

        refresh_token, family_id = jwt_handler.create_refresh_token("user-123")

        return client, access_token, refresh_token, family_id

    def test_successful_token_refresh(self, authenticated_client):
        """Test successful token refresh."""
        client, _, refresh_token, _ = authenticated_client

        # Act
        response = client.post("/api/v1/refresh", json={"refresh_token": refresh_token})

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["refresh_token"] != refresh_token  # New token

    def test_refresh_with_invalid_token_fails(self, client):
        """Test refresh with invalid token fails."""
        # Act
        response = client.post("/api/v1/refresh", json={"refresh_token": "invalid-token"})

        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_refresh_token_reuse_detection(self, authenticated_client):
        """Test that reused refresh tokens are detected."""
        client, _, refresh_token, _ = authenticated_client

        # First refresh - should succeed
        response1 = client.post("/api/v1/refresh", json={"refresh_token": refresh_token})
        assert response1.status_code == status.HTTP_200_OK

        # Attempt to reuse old token - should fail
        response2 = client.post("/api/v1/refresh", json={"refresh_token": refresh_token})
        assert response2.status_code == status.HTTP_401_UNAUTHORIZED

        # New token from first refresh should also be invalidated
        new_refresh = response1.json()["refresh_token"]
        response3 = client.post("/api/v1/refresh", json={"refresh_token": new_refresh})
        assert response3.status_code == status.HTTP_401_UNAUTHORIZED

    def test_refresh_validates_token_type(self, client):
        """Test that only refresh tokens can be used for refresh."""
        # Create access token (wrong type)
        access_token = jwt_handler.create_access_token(
            user_id="user-123", username="testuser", role="user", permissions=[]
        )

        # Act
        response = client.post("/api/v1/refresh", json={"refresh_token": access_token})

        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestLogoutSecurity:
    """Test logout endpoint security."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Create authorization headers."""
        access_token = jwt_handler.create_access_token(
            user_id="user-123", username="testuser", role="user", permissions=["read"]
        )
        return {"Authorization": f"Bearer {access_token}"}

    def test_successful_logout(self, client, auth_headers):
        """Test successful logout revokes tokens."""
        # Act
        response = client.post("/api/v1/logout", headers=auth_headers)

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Verify token is revoked
        response2 = client.get("/api/v1/me", headers=auth_headers)
        assert response2.status_code == status.HTTP_401_UNAUTHORIZED

    def test_logout_requires_authentication(self, client):
        """Test that logout requires valid authentication."""
        # Act - No auth headers
        response = client.post("/api/v1/logout")

        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_logout_with_invalid_token_fails(self, client):
        """Test logout with invalid token fails."""
        # Act
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/api/v1/logout", headers=headers)

        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestAuthorizationHeaders:
    """Test authorization header security."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def protected_endpoint(self, client):
        """Create a protected test endpoint."""

        @app.get("/test/protected")
        async def protected(current_user=Depends(get_current_user)):
            return {"user_id": current_user["user_id"]}

        return client

    def test_bearer_token_format_validation(self, protected_endpoint):
        """Test that bearer token format is validated."""
        invalid_formats = [
            "invalid-token",  # No Bearer prefix
            "Bearer",  # No token
            "Bearer  ",  # Empty token
            "Basic dGVzdDp0ZXN0",  # Wrong auth type
            "Bearer token1 token2",  # Multiple tokens
        ]

        for invalid_format in invalid_formats:
            headers = {"Authorization": invalid_format}
            response = protected_endpoint.get("/test/protected", headers=headers)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_token_fingerprint_validation(self, client):
        """Test token fingerprint validation."""
        # Create token with fingerprint
        fingerprint = jwt_handler.generate_fingerprint()
        access_token = jwt_handler.create_access_token(
            user_id="user-123",
            username="testuser",
            role="user",
            permissions=[],
            fingerprint=fingerprint,
        )

        # Set fingerprint in cookie
        client.cookies.set("fingerprint", fingerprint)

        # Valid request with matching fingerprint
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/me", headers=headers)
        assert response.status_code == status.HTTP_200_OK

        # Invalid request with wrong fingerprint
        client.cookies.set("fingerprint", "wrong-fingerprint")
        response2 = client.get("/api/v1/me", headers=headers)
        assert response2.status_code == status.HTTP_401_UNAUTHORIZED


class TestSecurityHeaders:
    """Test security headers on auth endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_security_headers_present(self, client):
        """Test that security headers are set."""
        response = client.post("/api/v1/login", json={"username": "test", "password": "test"})

        # Check security headers
        headers = response.headers

        # Strict-Transport-Security
        assert "strict-transport-security" in headers
        assert "max-age=" in headers["strict-transport-security"]

        # X-Content-Type-Options
        assert headers.get("x-content-type-options") == "nosniff"

        # X-Frame-Options
        assert headers.get("x-frame-options") == "DENY"

        # Content-Security-Policy
        assert "content-security-policy" in headers

        # X-XSS-Protection (legacy but still good)
        assert headers.get("x-xss-protection") == "1; mode=block"

    def test_cors_configuration(self, client):
        """Test CORS is properly configured."""
        # Preflight request
        response = client.options(
            "/api/v1/login",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Should either block (no CORS) or have proper headers
        if response.status_code == status.HTTP_200_OK:
            # If CORS is enabled, verify proper configuration
            assert "access-control-allow-origin" in response.headers
            assert "access-control-allow-credentials" in response.headers
            assert response.headers["access-control-allow-credentials"] == "true"

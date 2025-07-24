"""
Integration tests for MFA API endpoints.

Tests cover:
- MFA enrollment flow
- Token verification
- Rate limiting
- Security headers
- Error handling
- API response formats
"""

from unittest.mock import Mock, patch

import pytest
from api.main import app
from auth.mfa_service import MFAResponse
from fastapi.testclient import TestClient


class TestMFAAPI:
    """Integration tests for MFA API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        return {
            "user_id": "test_user_123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"],
        }

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for requests."""
        return {
            "Authorization": "Bearer test_token",
            "X-CSRF-Token": "test_csrf_token",
        }

    def test_enroll_mfa_totp_success(self, client, mock_user, auth_headers):
        """Test successful MFA enrollment with TOTP."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Mock MFA service
            mock_service = Mock()
            mock_service.enroll_user.return_value = MFAResponse(
                success=True,
                message="TOTP MFA enrollment initiated. Please scan the QR code and verify.",
                qr_code_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                backup_codes=["ABCD1234", "EFGH5678", "IJKL9012"],
            )
            mock_get_service.return_value = mock_service

            # Test enrollment
            response = client.post(
                "/api/v1/mfa/enroll",
                json={
                    "user_id": "test_user_123",
                    "user_email": "test@example.com",
                    "method": "totp",
                },
                headers=auth_headers,
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["qr_code_url"] is not None
            assert data["backup_codes"] is not None
            assert len(data["backup_codes"]) == 3

    def test_enroll_mfa_unauthorized_user(self, client, mock_user, auth_headers):
        """Test MFA enrollment for different user (should fail)."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Test enrollment for different user
            response = client.post(
                "/api/v1/mfa/enroll",
                json={
                    "user_id": "different_user",
                    "user_email": "test@example.com",
                    "method": "totp",
                },
                headers=auth_headers,
            )

            # Verify response
            assert response.status_code == 403
            assert "Can only enroll MFA for your own account" in response.json()["detail"]

    def test_enroll_mfa_already_enabled(self, client, mock_user, auth_headers):
        """Test MFA enrollment when already enabled."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Mock MFA service
            mock_service = Mock()
            mock_service.enroll_user.return_value = MFAResponse(
                success=False, message="MFA is already enabled for this user"
            )
            mock_get_service.return_value = mock_service

            # Test enrollment
            response = client.post(
                "/api/v1/mfa/enroll",
                json={
                    "user_id": "test_user_123",
                    "user_email": "test@example.com",
                    "method": "totp",
                },
                headers=auth_headers,
            )

            # Verify response
            assert response.status_code == 400
            assert "already enabled" in response.json()["detail"]

    def test_verify_mfa_success(self, client, mock_user, auth_headers):
        """Test successful MFA verification."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Mock MFA service
            mock_service = Mock()
            mock_service.verify_mfa.return_value = MFAResponse(
                success=True, message="MFA verification successful"
            )
            mock_get_service.return_value = mock_service

            # Test verification
            response = client.post(
                "/api/v1/mfa/verify",
                json={
                    "user_id": "test_user_123",
                    "token": "123456",
                    "method": "totp",
                },
                headers=auth_headers,
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "MFA verification successful"

    def test_verify_mfa_invalid_token(self, client, mock_user, auth_headers):
        """Test MFA verification with invalid token."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Mock MFA service
            mock_service = Mock()
            mock_service.verify_mfa.return_value = MFAResponse(
                success=False, message="Invalid MFA token"
            )
            mock_get_service.return_value = mock_service

            # Test verification
            response = client.post(
                "/api/v1/mfa/verify",
                json={
                    "user_id": "test_user_123",
                    "token": "000000",
                    "method": "totp",
                },
                headers=auth_headers,
            )

            # Verify response
            assert response.status_code == 400
            assert "Invalid MFA token" in response.json()["detail"]

    def test_verify_mfa_rate_limited(self, client, mock_user, auth_headers):
        """Test MFA verification when rate limited."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Mock MFA service
            mock_service = Mock()
            mock_service.verify_mfa.return_value = MFAResponse(
                success=False,
                message="Too many failed attempts. Please try again later.",
            )
            mock_get_service.return_value = mock_service

            # Test verification
            response = client.post(
                "/api/v1/mfa/verify",
                json={
                    "user_id": "test_user_123",
                    "token": "123456",
                    "method": "totp",
                },
                headers=auth_headers,
            )

            # Verify response
            assert response.status_code == 429
            assert "too many failed attempts" in response.json()["detail"].lower()

    def test_get_mfa_status_enabled(self, client, mock_user, auth_headers):
        """Test getting MFA status for enabled user."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user

            # Mock MFA service
            mock_service = Mock()
            mock_service.get_mfa_status.return_value = {
                "enabled": True,
                "methods": ["totp"],
                "backup_codes_remaining": 8,
                "locked_until": None,
                "failed_attempts": 0,
                "last_used": "2025-01-16T12:00:00Z",
            }
            mock_get_service.return_value = mock_service

            # Test status
            response = client.get(
                "/api/v1/mfa/status",
                headers={"Authorization": "Bearer test_token"},
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is True
            assert "totp" in data["methods"]
            assert data["backup_codes_remaining"] == 8

    def test_get_mfa_status_disabled(self, client, mock_user, auth_headers):
        """Test getting MFA status for disabled user."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user

            # Mock MFA service
            mock_service = Mock()
            mock_service.get_mfa_status.return_value = {
                "enabled": False,
                "methods": [],
                "backup_codes_remaining": 0,
                "locked_until": None,
                "failed_attempts": 0,
                "last_used": None,
            }
            mock_get_service.return_value = mock_service

            # Test status
            response = client.get(
                "/api/v1/mfa/status",
                headers={"Authorization": "Bearer test_token"},
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is False
            assert data["methods"] == []
            assert data["backup_codes_remaining"] == 0

    def test_disable_mfa_success(self, client, mock_user, auth_headers):
        """Test successful MFA disable."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Mock MFA service
            mock_service = Mock()
            mock_service.disable_mfa.return_value = MFAResponse(
                success=True, message="MFA has been disabled"
            )
            mock_get_service.return_value = mock_service

            # Test disable
            response = client.post("/api/v1/mfa/disable", headers=auth_headers)

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "MFA has been disabled"

    def test_regenerate_backup_codes_success(self, client, mock_user, auth_headers):
        """Test successful backup code regeneration."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Mock MFA service
            mock_service = Mock()
            mock_service.regenerate_backup_codes.return_value = MFAResponse(
                success=True,
                message="Backup codes regenerated successfully",
                backup_codes=["NEW1234", "NEW5678", "NEW9012"],
            )
            mock_get_service.return_value = mock_service

            # Test regeneration
            response = client.post("/api/v1/mfa/regenerate-backup-codes", headers=auth_headers)

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["backup_codes"] is not None
            assert len(data["backup_codes"]) == 3

    def test_get_available_methods(self, client):
        """Test getting available MFA methods."""
        response = client.get("/api/v1/mfa/methods")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "methods" in data
        assert "recommended_method" in data
        assert "backup_method" in data

        # Check TOTP method is available
        totp_method = next((m for m in data["methods"] if m["name"] == "totp"), None)
        assert totp_method is not None
        assert totp_method["enabled"] is True

    def test_test_mfa_token_valid(self, client, mock_user, auth_headers):
        """Test MFA token testing with valid token."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user

            # Mock MFA service
            mock_service = Mock()
            mock_service.get_mfa_status.return_value = {
                "enabled": False,  # Not enabled yet (enrollment in progress)
                "methods": [],
                "backup_codes_remaining": 0,
                "locked_until": None,
            }
            mock_service.verify_mfa.return_value = MFAResponse(
                success=True, message="MFA verification successful"
            )
            mock_get_service.return_value = mock_service

            # Test token
            response = client.post(
                "/api/v1/mfa/test-token",
                json={"token": "123456"},
                headers={"Authorization": "Bearer test_token"},
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["ready_for_enrollment"] is True

    def test_test_mfa_token_invalid(self, client, mock_user, auth_headers):
        """Test MFA token testing with invalid token."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user

            # Mock MFA service
            mock_service = Mock()
            mock_service.get_mfa_status.return_value = {
                "enabled": False,
                "methods": [],
                "backup_codes_remaining": 0,
                "locked_until": None,
            }
            mock_service.verify_mfa.return_value = MFAResponse(
                success=False, message="Invalid MFA token"
            )
            mock_get_service.return_value = mock_service

            # Test token
            response = client.post(
                "/api/v1/mfa/test-token",
                json={"token": "000000"},
                headers={"Authorization": "Bearer test_token"},
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert data["ready_for_enrollment"] is False

    def test_mfa_health_check_healthy(self, client):
        """Test MFA health check when healthy."""
        with patch("api.v1.mfa.get_mfa_service") as mock_get_service:
            # Mock MFA service
            mock_service = Mock()
            mock_service._encrypt_secret.return_value = "encrypted_test"
            mock_service._decrypt_secret.return_value = "test_secret_for_health_check"
            mock_get_service.return_value = mock_service

            # Test health check
            response = client.get("/api/v1/mfa/health")

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["encryption"] == "ok"

    def test_mfa_health_check_unhealthy(self, client):
        """Test MFA health check when unhealthy."""
        with patch("api.v1.mfa.get_mfa_service") as mock_get_service:
            # Mock MFA service to raise exception
            mock_service = Mock()
            mock_service._encrypt_secret.side_effect = Exception("Encryption failed")
            mock_get_service.return_value = mock_service

            # Test health check
            response = client.get("/api/v1/mfa/health")

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data

    def test_mfa_endpoints_require_authentication(self, client):
        """Test that MFA endpoints require authentication."""
        endpoints = [
            ("/api/v1/mfa/enroll", "POST"),
            ("/api/v1/mfa/verify", "POST"),
            ("/api/v1/mfa/status", "GET"),
            ("/api/v1/mfa/disable", "POST"),
            ("/api/v1/mfa/regenerate-backup-codes", "POST"),
            ("/api/v1/mfa/test-token", "POST"),
        ]

        for endpoint, method in endpoints:
            response = client.request(method, endpoint)
            # Should return 401 or 403 for unauthenticated requests
            assert response.status_code in [401, 403]

    def test_mfa_endpoints_require_csrf_token(self, client, mock_user):
        """Test that MFA endpoints require CSRF token."""
        with patch("api.v1.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = mock_user

            # Test without CSRF token
            response = client.post(
                "/api/v1/mfa/enroll",
                json={
                    "user_id": "test_user_123",
                    "user_email": "test@example.com",
                    "method": "totp",
                },
                headers={"Authorization": "Bearer test_token"},
            )

            # Should return 403 for missing CSRF token
            assert response.status_code == 403

    @pytest.mark.parametrize("method", ["totp", "sms", "email", "hardware_key"])
    def test_enroll_mfa_method_validation(self, client, mock_user, auth_headers, method):
        """Test MFA enrollment method validation."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Mock MFA service
            mock_service = Mock()
            if method == "totp":
                mock_service.enroll_user.return_value = MFAResponse(
                    success=True, message="TOTP enrollment successful"
                )
            else:
                mock_service.enroll_user.return_value = MFAResponse(
                    success=False,
                    message=f"Method {method} not yet implemented",
                )
            mock_get_service.return_value = mock_service

            # Test enrollment
            response = client.post(
                "/api/v1/mfa/enroll",
                json={
                    "user_id": "test_user_123",
                    "user_email": "test@example.com",
                    "method": method,
                },
                headers=auth_headers,
            )

            # Verify response
            if method == "totp":
                assert response.status_code == 200
            else:
                assert response.status_code == 400
                assert "not yet implemented" in response.json()["detail"]

    def test_verify_mfa_backup_code(self, client, mock_user, auth_headers):
        """Test MFA verification with backup code."""
        with (
            patch("api.v1.mfa.get_current_user") as mock_get_user,
            patch("api.v1.mfa.validate_csrf_token") as mock_csrf,
            patch("api.v1.mfa.get_mfa_service") as mock_get_service,
        ):
            # Mock authentication
            mock_get_user.return_value = mock_user
            mock_csrf.return_value = "test_csrf_token"

            # Mock MFA service
            mock_service = Mock()
            mock_service.verify_mfa.return_value = MFAResponse(
                success=True, message="MFA verification successful"
            )
            mock_get_service.return_value = mock_service

            # Test verification with backup code
            response = client.post(
                "/api/v1/mfa/verify",
                json={
                    "user_id": "test_user_123",
                    "token": "",
                    "method": "backup_code",
                    "backup_code": "ABCD1234",
                },
                headers=auth_headers,
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

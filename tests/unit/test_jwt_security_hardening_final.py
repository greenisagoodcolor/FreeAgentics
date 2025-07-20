"""
Unit tests for JWT Security Hardening (Task 14.3).

Tests all security requirements without needing the full app context:
1. RS256 algorithm with asymmetric keys
2. Short-lived access tokens (15 min) with secure refresh tokens (7 days)
3. JTI (JWT ID) for token revocation capability
4. Token binding to prevent replay attacks
5. Proper logout with token blacklisting
6. Validation of all claims including iss, aud, exp, nbf
7. CSRF protection implementation
8. Refresh token rotation
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from auth import UserRole, auth_manager
from auth.jwt_handler import jwt_handler


@pytest.mark.slow
class TestJWTSecurityHardening:
    """Unit tests for JWT security hardening."""

    @pytest.fixture
    def test_user(self):
        """Create a test user."""
        # Clean up any existing user
        if "testuser" in auth_manager.users:
            del auth_manager.users["testuser"]

        return auth_manager.register_user(
            username="testuser",
            email="test@example.com",
            password="SecurePassword123!",
            role=UserRole.RESEARCHER,
        )

    def test_jwt_uses_rs256_algorithm(self):
        """Verify JWT uses RS256 algorithm."""
        # Check algorithm configuration
        key_info = jwt_handler.get_key_info()
        assert key_info["algorithm"] == "RS256"

        # Verify keys are RSA
        assert isinstance(jwt_handler.private_key, rsa.RSAPrivateKey)
        assert isinstance(jwt_handler.public_key, rsa.RSAPublicKey)

        # Verify key size is strong (4096 bits)
        assert key_info["key_size"] == 4096

    def test_access_token_expiration_15_minutes(self, test_user):
        """Verify access tokens expire in 15 minutes."""
        token = auth_manager.create_access_token(test_user)

        # Decode without verification to check expiration
        payload = jwt.decode(token, options={"verify_signature": False})

        exp_time = datetime.fromtimestamp(payload["exp"], timezone.utc)
        iat_time = datetime.fromtimestamp(payload["iat"], timezone.utc)

        # Should be exactly 15 minutes
        duration = exp_time - iat_time
        assert (
            890 <= duration.total_seconds() <= 910
        )  # 15 minutes ± 10 seconds

    def test_refresh_token_expiration_7_days(self, test_user):
        """Verify refresh tokens expire in 7 days."""
        token = auth_manager.create_refresh_token(test_user)

        # Decode without verification to check expiration
        payload = jwt.decode(token, options={"verify_signature": False})

        exp_time = datetime.fromtimestamp(payload["exp"], timezone.utc)
        iat_time = datetime.fromtimestamp(payload["iat"], timezone.utc)

        # Should be exactly 7 days
        duration = exp_time - iat_time
        expected_seconds = 7 * 24 * 60 * 60
        assert (
            abs(duration.total_seconds() - expected_seconds) < 3600
        )  # Within 1 hour

    def test_jwt_includes_all_required_claims(self, test_user):
        """Verify JWT includes all required claims."""
        token = auth_manager.create_access_token(test_user)
        payload = jwt.decode(token, options={"verify_signature": False})

        required_claims = [
            "iss",
            "aud",
            "exp",
            "nbf",
            "iat",
            "jti",
            "user_id",
            "username",
            "role",
            "type",
        ]
        for claim in required_claims:
            assert claim in payload, f"Missing required claim: {claim}"

        # Verify claim values
        assert payload["iss"] == "freeagentics-auth"
        assert payload["aud"] == "freeagentics-api"
        assert payload["type"] == "access"

        # Verify JTI is unique and sufficient length
        assert len(payload["jti"]) >= 32

    def test_token_includes_unique_jti(self, test_user):
        """Test that each token has a unique JTI."""
        token1 = auth_manager.create_access_token(test_user)
        token2 = auth_manager.create_access_token(test_user)

        payload1 = jwt.decode(token1, options={"verify_signature": False})
        payload2 = jwt.decode(token2, options={"verify_signature": False})

        # JTIs should be different
        assert payload1["jti"] != payload2["jti"]

    def test_token_revocation_via_jti(self, test_user):
        """Test token revocation using JTI."""
        token = auth_manager.create_access_token(test_user)

        # Token should be valid initially
        token_data = auth_manager.verify_token(token)
        assert token_data.user_id == test_user.user_id

        # Revoke the token
        jwt_handler.revoke_token(token)

        # Token should now be invalid
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(token)
        assert exc_info.value.status_code == 401
        assert "revoked" in str(exc_info.value.detail).lower()

    def test_token_binding_with_fingerprint(self, test_user):
        """Test token binding to prevent replay attacks."""
        fingerprint = jwt_handler.generate_fingerprint()

        # Create token with fingerprint
        token = auth_manager.create_access_token(
            test_user, client_fingerprint=fingerprint
        )

        # Decode to verify fingerprint is included (as hash)
        payload = jwt.decode(token, options={"verify_signature": False})
        assert "fingerprint" in payload

        # Should succeed with correct fingerprint
        token_data = auth_manager.verify_token(
            token, client_fingerprint=fingerprint
        )
        assert token_data.user_id == test_user.user_id

        # Should fail with wrong fingerprint
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(
                token, client_fingerprint="wrong_fingerprint"
            )
        assert exc_info.value.status_code == 401
        assert "fingerprint" in str(exc_info.value.detail).lower()

    def test_csrf_token_generation_and_validation(self, test_user):
        """Test CSRF token generation and validation."""
        csrf_protection = auth_manager.csrf_protection
        session_id = test_user.user_id

        # Generate CSRF token
        csrf_token = csrf_protection.generate_csrf_token(session_id)
        assert len(csrf_token) >= 32

        # Validate correct token
        assert (
            csrf_protection.verify_csrf_token(session_id, csrf_token) is True
        )

        # Invalid token should fail
        assert (
            csrf_protection.verify_csrf_token(session_id, "wrong_token")
            is False
        )

        # Non-existent session should fail
        assert (
            csrf_protection.verify_csrf_token("wrong_session", csrf_token)
            is False
        )

    def test_refresh_token_rotation(self, test_user):
        """Test refresh token rotation prevents reuse."""
        # Create initial tokens
        access_token = auth_manager.create_access_token(test_user)
        refresh_token = auth_manager.create_refresh_token(test_user)

        # Use refresh token to get new tokens
        new_access, new_refresh = auth_manager.refresh_access_token(
            refresh_token
        )

        # New tokens should be different
        assert new_access != access_token
        assert new_refresh != refresh_token

        # Decode to verify new tokens are valid
        new_access_payload = jwt.decode(
            new_access, options={"verify_signature": False}
        )
        assert new_access_payload["user_id"] == test_user.user_id

    def test_logout_revokes_tokens(self, test_user):
        """Test that logout revokes tokens."""
        token = auth_manager.create_access_token(test_user)

        # Token should be valid before logout
        token_data = auth_manager.verify_token(token)
        assert token_data.user_id == test_user.user_id

        # Logout
        auth_manager.logout(token, user_id=test_user.user_id)

        # Token should be revoked
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(token)
        assert exc_info.value.status_code == 401

    def test_token_validation_checks_not_before(self, test_user):
        """Test that tokens with future NBF are rejected."""
        # Create payload with future NBF
        future_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        payload = {
            "user_id": test_user.user_id,
            "username": test_user.username,
            "role": test_user.role.value,
            "permissions": [],
            "type": "access",
            "jti": "test-jti",
            "iss": "freeagentics-auth",
            "aud": "freeagentics-api",
            "nbf": future_time.timestamp(),
            "exp": (future_time + timedelta(minutes=15)).timestamp(),
            "iat": datetime.now(timezone.utc).timestamp(),
        }

        # Create token with future NBF
        future_token = jwt.encode(
            payload, jwt_handler.private_key, algorithm="RS256"
        )

        # Should fail validation
        with pytest.raises(HTTPException) as exc_info:
            payload = jwt_handler.verify_access_token(future_token)
        assert exc_info.value.status_code == 401

    def test_blacklist_cleanup_removes_expired_entries(self):
        """Test that blacklist cleanup removes only expired entries."""
        # Clear existing blacklist
        jwt_handler.blacklist._blacklist.clear()

        # Add expired entries
        past_time = datetime.now(timezone.utc) - timedelta(days=2)
        jwt_handler.blacklist._blacklist["expired-1"] = past_time.timestamp()
        jwt_handler.blacklist._blacklist["expired-2"] = past_time.timestamp()

        # Add current entry
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        jwt_handler.blacklist._blacklist["current"] = future_time.timestamp()

        # Force cleanup by setting last cleanup time to past
        jwt_handler.blacklist._last_cleanup = time.time() - 7200  # 2 hours ago

        # Trigger cleanup
        jwt_handler.blacklist._cleanup()

        # Check results
        assert "expired-1" not in jwt_handler.blacklist._blacklist
        assert "expired-2" not in jwt_handler.blacklist._blacklist
        assert "current" in jwt_handler.blacklist._blacklist

    def test_secure_cookie_configuration(self):
        """Test secure cookie helper methods."""
        from fastapi import Response

        response = Response()

        # Test access token cookie
        auth_manager.set_token_cookie(response, "test_token", secure=True)

        # Test CSRF cookie
        auth_manager.set_csrf_cookie(response, "csrf_token", secure=True)

        # Verify methods exist and don't raise errors
        assert hasattr(auth_manager, "set_token_cookie")
        assert hasattr(auth_manager, "set_csrf_cookie")

    def test_key_rotation_warning(self):
        """Test that key rotation warnings are generated."""
        key_info = jwt_handler.get_key_info()

        # Check that key info includes rotation information
        assert "key_age_days" in key_info
        assert "rotation_required" in key_info
        assert "rotation_warning" in key_info

        # Keys should be fresh in test environment
        assert key_info["rotation_required"] is False

    def test_comprehensive_token_validation(self, test_user):
        """Test comprehensive validation of all token claims."""
        token = auth_manager.create_access_token(test_user)

        # Manually decode to inspect all claims
        payload = jwt.decode(
            token,
            jwt_handler.public_key,
            algorithms=["RS256"],
            audience="freeagentics-api",
            issuer="freeagentics-auth",
        )

        # Verify all required claims are present and valid
        assert payload["user_id"] == test_user.user_id
        assert payload["username"] == test_user.username
        assert payload["role"] == test_user.role.value
        assert payload["type"] == "access"

        # Verify timestamps
        now = datetime.now(timezone.utc).timestamp()
        assert payload["iat"] <= now
        assert payload["nbf"] <= now
        assert payload["exp"] > now

        # Verify issuer and audience
        assert payload["iss"] == "freeagentics-auth"
        assert payload["aud"] == "freeagentics-api"

        # Verify JTI
        assert len(payload["jti"]) >= 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

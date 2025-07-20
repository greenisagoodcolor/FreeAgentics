"""
Comprehensive JWT Security Hardening Test Suite
Test-driven development for Task #14.3 - JWT Security and Authentication Hardening

This test suite validates:
1. RS256 algorithm with asymmetric keys
2. Short-lived access tokens (15 min) with secure refresh tokens (7 days)
3. JTI (JWT ID) for token revocation capability
4. Secure token storage using httpOnly cookies with SameSite=Strict
5. Token binding to prevent replay attacks
6. Proper logout with token blacklisting
7. Validation of all claims including iss, aud, exp, nbf
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from auth.security_implementation import (
    ALGORITHM,
    AuthenticationManager,
    User,
    UserRole,
)


class TestJWTAlgorithmSecurity:
    """Test RS256 algorithm implementation with asymmetric keys."""

    def test_should_use_rs256_algorithm(self):
        """Test that JWT uses RS256 algorithm instead of HS256."""
        # This test will fail initially as current implementation uses HS256
        assert (
            ALGORITHM == "RS256"
        ), "JWT must use RS256 algorithm for asymmetric signing"

    def test_should_generate_rsa_key_pair(self):
        """Test RSA key pair generation for JWT signing."""
        # Generate new RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Extract public key
        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        assert private_pem.startswith(b"-----BEGIN PRIVATE KEY-----")
        assert public_pem.startswith(b"-----BEGIN PUBLIC KEY-----")

    def test_should_sign_token_with_private_key(self):
        """Test JWT signing with RSA private key."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Should successfully create token with RSA private key
        token = auth_manager.create_access_token(user)
        assert token is not None
        assert len(token) > 0

        # Should be able to decode without verification to check structure
        decoded = jwt.decode(token, options={"verify_signature": False})
        assert decoded["user_id"] == user.user_id
        assert decoded["type"] == "access"

    def test_should_verify_token_with_public_key(self):
        """Test JWT verification with RSA public key."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Create token with RSA private key
        token = auth_manager.create_access_token(user)

        # Should successfully verify with RSA public key
        token_data = auth_manager.verify_token(token)
        assert token_data.user_id == user.user_id
        assert token_data.username == user.username
        assert token_data.role == user.role

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )


class TestTokenLifecycleManagement:
    """Test token duration and lifecycle management."""

    def test_access_token_should_expire_in_15_minutes(self):
        """Test that access tokens expire in exactly 15 minutes."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)
        decoded = jwt.decode(token, options={"verify_signature": False})

        exp_time = datetime.fromtimestamp(decoded["exp"], timezone.utc)
        created_time = datetime.now(timezone.utc)
        duration = exp_time - created_time

        # Should be 15 minutes, not 30
        assert (
            abs(duration.total_seconds() - 900) < 60
        ), "Access token should expire in 15 minutes"

    def test_refresh_token_should_expire_in_7_days(self):
        """Test that refresh tokens expire in exactly 7 days."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_refresh_token(user)
        decoded = jwt.decode(token, options={"verify_signature": False})

        exp_time = datetime.fromtimestamp(decoded["exp"], timezone.utc)
        created_time = datetime.now(timezone.utc)
        duration = exp_time - created_time

        # Should be 7 days
        expected_seconds = 7 * 24 * 60 * 60  # 7 days in seconds
        assert (
            abs(duration.total_seconds() - expected_seconds) < 3600
        ), "Refresh token should expire in 7 days"

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )


class TestJTIRevocationSystem:
    """Test JTI (JWT ID) implementation for token revocation."""

    def test_tokens_should_include_jti_claim(self):
        """Test that all tokens include a unique JTI claim."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert "jti" in decoded, "Token must include JTI claim"
        assert len(decoded["jti"]) >= 16, "JTI should be sufficiently long"

    def test_jti_should_be_unique_per_token(self):
        """Test that each token gets a unique JTI."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token1 = auth_manager.create_access_token(user)
        token2 = auth_manager.create_access_token(user)

        decoded1 = jwt.decode(token1, options={"verify_signature": False})
        decoded2 = jwt.decode(token2, options={"verify_signature": False})

        assert (
            decoded1["jti"] != decoded2["jti"]
        ), "Each token should have unique JTI"

    def test_should_revoke_token_by_jti(self):
        """Test token revocation using JTI."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)
        decoded = jwt.decode(token, options={"verify_signature": False})
        jti = decoded["jti"]

        # Should be able to revoke token
        auth_manager.revoke_token(jti)

        # Verification should fail for revoked token
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(token)

        assert exc_info.value.status_code == 401
        assert "blacklisted" in exc_info.value.detail.lower()

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )


class TestSecureTokenStorage:
    """Test secure cookie storage with httpOnly and SameSite=Strict."""

    def test_should_set_httponly_cookie(self):
        """Test that tokens are stored in httpOnly cookies."""
        auth_manager = AuthenticationManager()
        response = Mock()

        # Should set httpOnly cookie
        auth_manager.set_token_cookie(response, "fake_token")

        # Verify cookie was set with httponly flag
        response.set_cookie.assert_called_once()
        call_args = response.set_cookie.call_args
        assert call_args[1]["httponly"] is True

    def test_should_set_samesite_strict(self):
        """Test that cookies use SameSite=Strict."""
        auth_manager = AuthenticationManager()
        response = Mock()

        # Should set SameSite=Strict
        auth_manager.set_token_cookie(response, "fake_token")

        # Verify cookie was set with SameSite=Strict
        response.set_cookie.assert_called_once()
        call_args = response.set_cookie.call_args
        assert call_args[1]["samesite"] == "strict"

    def test_should_set_secure_flag_in_production(self):
        """Test that cookies use Secure flag in production."""
        auth_manager = AuthenticationManager()
        response = Mock()

        # Should set Secure flag in production
        auth_manager.set_token_cookie(response, "fake_token", secure=True)

        # Verify cookie was set with Secure flag
        response.set_cookie.assert_called_once()
        call_args = response.set_cookie.call_args
        assert call_args[1]["secure"] is True


class TestTokenBinding:
    """Test token binding to prevent replay attacks."""

    def test_tokens_should_include_binding_claim(self):
        """Test that tokens include binding claim tied to client."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()
        client_fingerprint = "client_fingerprint_hash"

        token = auth_manager.create_access_token(
            user, client_fingerprint=client_fingerprint
        )
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert "binding" in decoded, "Token must include binding claim"
        assert (
            decoded["binding"] == client_fingerprint
        ), "Binding should match client fingerprint"

    def test_should_reject_token_with_wrong_binding(self):
        """Test that tokens with wrong binding are rejected."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Create token with one fingerprint
        token = auth_manager.create_access_token(
            user, client_fingerprint="fingerprint1"
        )

        # Try to verify with different fingerprint
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(token, client_fingerprint="fingerprint2")

        assert exc_info.value.status_code == 401
        assert "binding" in exc_info.value.detail.lower()

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )


class TestTokenBlacklisting:
    """Test token blacklisting for secure logout."""

    def test_should_blacklist_token_on_logout(self):
        """Test that logout blacklists the token."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)

        # Logout should blacklist token
        auth_manager.logout(token)

        # Verification should fail for blacklisted token
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(token)

        assert exc_info.value.status_code == 401
        assert "blacklisted" in exc_info.value.detail.lower()

    def test_should_clean_expired_blacklist_entries(self):
        """Test cleanup of expired blacklist entries."""
        auth_manager = AuthenticationManager()

        # Add expired entry to blacklist (older than REFRESH_TOKEN_EXPIRE_DAYS + 1)
        expired_jti = "expired_jti"
        auth_manager.blacklist[expired_jti] = datetime.now(
            timezone.utc
        ) - timedelta(days=9)

        # Cleanup should remove expired entries
        auth_manager.cleanup_blacklist()

        assert expired_jti not in auth_manager.blacklist

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )


class TestComprehensiveClaimsValidation:
    """Test comprehensive validation of all JWT claims."""

    def test_should_validate_issuer_claim(self):
        """Test validation of 'iss' (issuer) claim."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert "iss" in decoded, "Token must include issuer claim"
        assert (
            decoded["iss"] == "freeagentics"
        ), "Issuer should be 'freeagentics'"

    def test_should_validate_audience_claim(self):
        """Test validation of 'aud' (audience) claim."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert "aud" in decoded, "Token must include audience claim"
        assert (
            decoded["aud"] == "freeagentics-api"
        ), "Audience should be 'freeagentics-api'"

    def test_should_validate_not_before_claim(self):
        """Test validation of 'nbf' (not before) claim."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert "nbf" in decoded, "Token must include not-before claim"

        # NBF should be current time or slightly before
        nbf_time = datetime.fromtimestamp(decoded["nbf"], timezone.utc)
        now = datetime.now(timezone.utc)
        assert nbf_time <= now, "Not-before time should not be in future"

    def test_should_validate_issued_at_claim(self):
        """Test validation of 'iat' (issued at) claim."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert "iat" in decoded, "Token must include issued-at claim"

        # IAT should be current time
        iat_time = datetime.fromtimestamp(decoded["iat"], timezone.utc)
        now = datetime.now(timezone.utc)
        assert (
            abs((now - iat_time).total_seconds()) < 60
        ), "Issued-at should be current time"

    def test_should_reject_future_nbf_tokens(self):
        """Test rejection of tokens with future 'nbf' claim."""
        auth_manager = AuthenticationManager()

        # Create token with future NBF manually
        future_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        payload = {
            "user_id": "test",
            "username": "testuser",
            "role": "researcher",
            "permissions": [],
            "type": "access",
            "jti": "test_jti",
            "iss": "freeagentics",
            "aud": "freeagentics-api",
            "nbf": future_time.timestamp(),  # Future NBF
            "exp": (future_time + timedelta(minutes=15)).timestamp(),
            "iat": datetime.now(timezone.utc).timestamp(),
        }

        # Create token with future NBF using private key
        invalid_token = jwt.encode(
            payload, auth_manager.private_key, algorithm=ALGORITHM
        )

        # Verification should fail for future NBF
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(invalid_token)

        assert exc_info.value.status_code == 401

    def test_should_validate_all_required_claims(self):
        """Test that all required claims are present and valid."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)
        decoded = jwt.decode(token, options={"verify_signature": False})

        required_claims = [
            "iss",
            "aud",
            "exp",
            "nbf",
            "iat",
            "jti",
            "user_id",
            "role",
        ]

        for claim in required_claims:
            assert (
                claim in decoded
            ), f"Token must include required claim: {claim}"

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )


class TestSecurityIntegration:
    """Integration tests for JWT security features."""

    def test_complete_authentication_flow(self):
        """Test complete authentication flow with all security features."""
        auth_manager = AuthenticationManager()

        # Register user
        user = auth_manager.register_user(
            username="testuser",
            email="test@example.com",
            password="secure_password",
            role=UserRole.RESEARCHER,
        )

        # Authenticate
        authenticated_user = auth_manager.authenticate_user(
            "testuser", "secure_password"
        )
        assert authenticated_user is not None

        # Create tokens with all security features
        client_fingerprint = "test_fingerprint"
        access_token = auth_manager.create_access_token(
            authenticated_user, client_fingerprint=client_fingerprint
        )

        # Verify token
        token_data = auth_manager.verify_token(
            access_token, client_fingerprint=client_fingerprint
        )
        assert token_data.user_id == user.user_id

        # Logout (blacklist token)
        auth_manager.logout(access_token)

        # Verify token is now invalid
        with pytest.raises(HTTPException):
            auth_manager.verify_token(
                access_token, client_fingerprint=client_fingerprint
            )

    def test_security_headers_integration(self):
        """Test integration with security headers."""
        auth_manager = AuthenticationManager()
        response = Mock()

        # Test that secure cookie headers are properly set
        auth_manager.set_token_cookie(response, "test_token", secure=True)

        # Verify all security headers are set
        response.set_cookie.assert_called_once()
        call_args = response.set_cookie.call_args

        assert call_args[1]["httponly"] is True, "Cookie should be httpOnly"
        assert call_args[1]["secure"] is True, "Cookie should be secure"
        assert (
            call_args[1]["samesite"] == "strict"
        ), "Cookie should use SameSite=Strict"
        assert call_args[1]["path"] == "/", "Cookie should have correct path"

    def test_rate_limiting_integration(self):
        """Test integration with rate limiting."""
        # Test that JWT endpoints respect rate limiting

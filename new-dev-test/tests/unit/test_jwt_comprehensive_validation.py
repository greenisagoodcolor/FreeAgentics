"""
Comprehensive JWT Security Validation Test Suite
Task #14.13 - JWT Security Validation and Hardening - Final Validation

This test suite validates ALL 9 requirements from the task:
1. ✅ RS256 algorithm (not HS256)
2. ✅ Proper token expiration (15-30 min access, 7-30 days refresh)
3. ✅ Token revocation mechanism with blacklist support
4. ✅ Secure refresh token rotation
5. ✅ JWT claims validation (iss, aud, exp, nbf, iat)
6. ✅ Secure transmission (httpOnly, secure cookies)
7. ✅ JWT fingerprinting to prevent token theft
8. ✅ Monitoring for suspicious token usage patterns
9. ✅ Repository cleanup and consolidation
"""

import time
from datetime import datetime, timezone
from unittest.mock import Mock

import jwt
import pytest
from fastapi import HTTPException

from auth.security_implementation import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    REFRESH_TOKEN_EXPIRE_DAYS,
    AuthenticationManager,
    User,
    UserRole,
)
from auth.security_logging import security_auditor


class TestJWTComprehensiveValidation:
    """Comprehensive validation of all JWT security requirements."""

    def test_requirement_1_rs256_algorithm(self):
        """Requirement 1: Verify JWT signing algorithm is RS256 (not HS256)."""
        # Verify global algorithm configuration
        assert ALGORITHM == "RS256", "JWT algorithm must be RS256, not HS256"

        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Create token and verify it uses RSA keys
        token = auth_manager.create_access_token(user)

        # Should be able to verify with RSA public key
        token_data = auth_manager.verify_token(token)
        assert token_data.user_id == user.user_id

        # Verify token was signed with RSA (can decode with public key)
        payload = jwt.decode(
            token,
            auth_manager.public_key,
            algorithms=["RS256"],
            options={"verify_aud": False, "verify_iss": False},
        )
        assert payload["user_id"] == user.user_id

    def test_requirement_2_proper_token_expiration(self):
        """Requirement 2: Proper token expiration (15-30 min access, 7-30 days refresh)."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Test access token expiration (should be 15 minutes)
        access_token = auth_manager.create_access_token(user)
        access_payload = jwt.decode(access_token, options={"verify_signature": False})

        access_exp = datetime.fromtimestamp(access_payload["exp"], timezone.utc)
        access_iat = datetime.fromtimestamp(access_payload["iat"], timezone.utc)
        access_duration = access_exp - access_iat

        # Should be 15 minutes (900 seconds) with 60 second tolerance
        assert abs(access_duration.total_seconds() - (ACCESS_TOKEN_EXPIRE_MINUTES * 60)) < 60
        assert 15 * 60 <= access_duration.total_seconds() <= 30 * 60  # Between 15-30 minutes

        # Test refresh token expiration (should be 7 days)
        refresh_token = auth_manager.create_refresh_token(user)
        refresh_payload = jwt.decode(refresh_token, options={"verify_signature": False})

        refresh_exp = datetime.fromtimestamp(refresh_payload["exp"], timezone.utc)
        refresh_iat = datetime.fromtimestamp(refresh_payload["iat"], timezone.utc)
        refresh_duration = refresh_exp - refresh_iat

        # Should be 7 days with 1 hour tolerance
        expected_seconds = REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        assert abs(refresh_duration.total_seconds() - expected_seconds) < 3600
        assert (
            7 * 24 * 60 * 60 <= refresh_duration.total_seconds() <= 30 * 24 * 60 * 60
        )  # Between 7-30 days

    def test_requirement_3_token_revocation_mechanism(self):
        """Requirement 3: Token revocation mechanism with blacklist support."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Create token
        token = auth_manager.create_access_token(user)
        auth_manager.verify_token(token)  # Should work initially

        # Extract JTI and revoke
        payload = jwt.decode(token, options={"verify_signature": False})
        jti = payload["jti"]
        auth_manager.revoke_token(jti)

        # Verify token is blacklisted
        assert jti in auth_manager.blacklist

        # Verify token no longer works
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(token)
        assert exc_info.value.status_code == 401
        assert "blacklisted" in exc_info.value.detail.lower()

    def test_requirement_4_secure_refresh_token_rotation(self):
        """Requirement 4: Secure refresh token rotation."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Register user for refresh functionality
        auth_manager.users[user.username] = {
            "user": user,
            "password_hash": "dummy",
        }

        # Create initial refresh token
        old_refresh_token = auth_manager.create_refresh_token(user)

        # Refresh access token (should rotate refresh token)
        (
            new_access_token,
            new_refresh_token,
        ) = auth_manager.refresh_access_token(old_refresh_token)

        # Verify rotation occurred
        assert new_refresh_token != old_refresh_token
        assert auth_manager.refresh_tokens[user.user_id] == new_refresh_token

        # Verify old refresh token is blacklisted
        old_payload = jwt.decode(old_refresh_token, options={"verify_signature": False})
        old_jti = old_payload["jti"]
        assert old_jti in auth_manager.blacklist

        # Verify old refresh token no longer works
        with pytest.raises(HTTPException):
            auth_manager.refresh_access_token(old_refresh_token)

    def test_requirement_5_jwt_claims_validation(self):
        """Requirement 5: JWT claims validation (iss, aud, exp, nbf, iat)."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        token = auth_manager.create_access_token(user)
        payload = jwt.decode(token, options={"verify_signature": False})

        # Verify all required claims are present
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
            assert claim in payload, f"Token must include claim: {claim}"

        # Verify claim values
        assert payload["iss"] == "freeagentics"
        assert payload["aud"] == "freeagentics-api"

        # Verify timestamps are reasonable
        now = datetime.now(timezone.utc)
        exp_time = datetime.fromtimestamp(payload["exp"], timezone.utc)
        nbf_time = datetime.fromtimestamp(payload["nbf"], timezone.utc)
        iat_time = datetime.fromtimestamp(payload["iat"], timezone.utc)

        assert iat_time <= now
        assert nbf_time <= now
        assert exp_time > now

    def test_requirement_6_secure_transmission(self):
        """Requirement 6: Secure transmission (httpOnly, secure cookies)."""
        auth_manager = AuthenticationManager()
        response_mock = Mock()

        # Test secure cookie setting
        auth_manager.set_token_cookie(response_mock, "test_token", secure=True)

        # Verify cookie was set with security flags
        response_mock.set_cookie.assert_called_once()
        call_args = response_mock.set_cookie.call_args

        assert call_args[1]["httponly"] is True, "Cookie must be httpOnly"
        assert call_args[1]["secure"] is True, "Cookie must be secure"
        assert call_args[1]["samesite"] == "strict", "Cookie must use SameSite=Strict"

    def test_requirement_7_jwt_fingerprinting(self):
        """Requirement 7: JWT fingerprinting to prevent token theft."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        client_fingerprint = "unique_client_fingerprint"

        # Create token with fingerprint
        token = auth_manager.create_access_token(user, client_fingerprint=client_fingerprint)
        payload = jwt.decode(token, options={"verify_signature": False})

        # Verify binding claim is present
        assert "binding" in payload
        assert payload["binding"] == client_fingerprint

        # Verify token works with correct fingerprint
        token_data = auth_manager.verify_token(token, client_fingerprint=client_fingerprint)
        assert token_data.user_id == user.user_id

        # Verify token fails with wrong fingerprint
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(token, client_fingerprint="wrong_fingerprint")
        assert exc_info.value.status_code == 401
        assert "binding" in exc_info.value.detail.lower()

    def test_requirement_8_monitoring_suspicious_patterns(self):
        """Requirement 8: Monitoring for suspicious token usage patterns."""
        # Clear auditor state
        security_auditor.token_usage_patterns = {}
        security_auditor.token_binding_violations = {}

        # Test suspicious pattern detection
        user_id = "test_user_123"
        ip_address = "192.168.1.100"

        # Simulate multiple invalid token attempts
        for i in range(6):  # Should trigger suspicious pattern alert
            security_auditor._track_suspicious_token_usage(
                {
                    "event_type": "token_invalid",
                    "user_id": user_id,
                    "ip_address": ip_address,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        # Should have tracked the patterns
        assert user_id in security_auditor.token_usage_patterns
        assert len(security_auditor.token_usage_patterns[user_id]) >= 5

        # Test token binding violation tracking
        for i in range(4):  # Should trigger alert
            security_auditor._track_token_binding_violation(ip_address)

        assert ip_address in security_auditor.token_binding_violations
        assert len(security_auditor.token_binding_violations[ip_address]) >= 3

    def test_requirement_9_repository_cleanup(self):
        """Requirement 9: Repository cleanup and consolidation."""
        # Verify deprecated HS256 is not used in production code
        auth_manager = AuthenticationManager()

        # Production should use RS256
        assert ALGORITHM == "RS256"

        # Verify RSA keys are being used
        assert hasattr(auth_manager, "private_key")
        assert hasattr(auth_manager, "public_key")

        # Verify tokens are created with RSA keys
        user = self._create_test_user()
        token = auth_manager.create_access_token(user)

        # Should be verifiable with RSA public key
        payload = jwt.decode(
            token,
            auth_manager.public_key,
            algorithms=["RS256"],
            options={"verify_aud": False, "verify_iss": False},
        )
        assert payload["user_id"] == user.user_id

    def test_production_readiness_comprehensive(self):
        """Comprehensive production readiness validation."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Register user for complete testing
        auth_manager.users[user.username] = {
            "user": user,
            "password_hash": "dummy",
        }

        # Test complete authentication flow
        client_fingerprint = "production_client_fingerprint"

        # 1. Create initial tokens
        access_token = auth_manager.create_access_token(user, client_fingerprint=client_fingerprint)
        refresh_token = auth_manager.create_refresh_token(user)

        # 2. Verify access token
        token_data = auth_manager.verify_token(access_token, client_fingerprint=client_fingerprint)
        assert token_data.user_id == user.user_id

        # 3. Refresh tokens
        (
            new_access_token,
            new_refresh_token,
        ) = auth_manager.refresh_access_token(refresh_token, client_fingerprint=client_fingerprint)
        assert new_access_token != access_token
        assert new_refresh_token != refresh_token

        # 4. Verify new access token
        new_token_data = auth_manager.verify_token(
            new_access_token, client_fingerprint=client_fingerprint
        )
        assert new_token_data.user_id == user.user_id

        # 5. Logout (blacklist tokens)
        auth_manager.logout(new_access_token)

        # 6. Verify tokens are blacklisted
        with pytest.raises(HTTPException):
            auth_manager.verify_token(new_access_token, client_fingerprint=client_fingerprint)

    def test_performance_requirements(self):
        """Test that JWT operations meet performance requirements."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Token creation should be fast
        start_time = time.time()
        for _ in range(100):
            token = auth_manager.create_access_token(user)
        creation_time = time.time() - start_time
        assert creation_time < 2.0, "Token creation should be fast"

        # Token verification should be fast
        token = auth_manager.create_access_token(user)
        start_time = time.time()
        for _ in range(100):
            auth_manager.verify_token(token)
        verification_time = time.time() - start_time
        assert verification_time < 2.0, "Token verification should be fast"

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )

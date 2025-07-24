"""
Comprehensive JWT Lifecycle Test Suite

Tests all aspects of JWT token generation, validation, expiration, and refresh flows
for the FreeAgentics authentication system.
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
import pytest
from auth.security_implementation import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    REFRESH_TOKEN_EXPIRE_DAYS,
    AuthenticationManager,
    User,
    UserRole,
)


class TestJWTLifecycle:
    """Test JWT token lifecycle comprehensively."""

    @pytest.fixture
    def auth_manager(self):
        """Create fresh AuthenticationManager instance."""
        return AuthenticationManager()

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            user_id="test_user_123",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
            is_active=True,
        )

    def test_access_token_generation(self, auth_manager, test_user):
        """Test access token generation with all required claims."""
        token = auth_manager.create_access_token(test_user)

        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are long

        # Decode token to verify claims
        payload = jwt.decode(token, options={"verify_signature": False})

        # Verify required claims
        assert payload["user_id"] == test_user.user_id
        assert payload["username"] == test_user.username
        assert payload["role"] == test_user.role.value
        assert payload["type"] == "access"
        assert payload["iss"] == "freeagentics"
        assert payload["aud"] == "freeagentics-api"
        assert "jti" in payload
        assert "exp" in payload
        assert "nbf" in payload
        assert "iat" in payload
        assert "permissions" in payload

        # Verify expiration is set correctly
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        now = datetime.now(timezone.utc)
        expected_exp = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        assert abs((exp_time - expected_exp).total_seconds()) < 5  # Allow 5 second tolerance

    def test_refresh_token_generation(self, auth_manager, test_user):
        """Test refresh token generation with proper claims."""
        token = auth_manager.create_refresh_token(test_user)

        assert isinstance(token, str)
        assert len(token) > 50

        # Decode token to verify claims
        payload = jwt.decode(token, options={"verify_signature": False})

        # Verify required claims
        assert payload["user_id"] == test_user.user_id
        assert payload["type"] == "refresh"
        assert payload["iss"] == "freeagentics"
        assert payload["aud"] == "freeagentics-api"
        assert "jti" in payload
        assert "exp" in payload
        assert "nbf" in payload
        assert "iat" in payload

        # Verify expiration is set correctly
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        now = datetime.now(timezone.utc)
        expected_exp = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        assert abs((exp_time - expected_exp).total_seconds()) < 5

    def test_access_token_validation(self, auth_manager, test_user):
        """Test access token validation with proper verification."""
        token = auth_manager.create_access_token(test_user)

        # Verify token
        token_data = auth_manager.verify_token(token)

        assert token_data.user_id == test_user.user_id
        assert token_data.username == test_user.username
        assert token_data.role == test_user.role
        assert len(token_data.permissions) > 0
        assert isinstance(token_data.exp, datetime)

    def test_refresh_token_validation(self, auth_manager, test_user):
        """Test refresh token validation flow."""
        # Register the user first
        auth_manager.users[test_user.username] = {
            "user": test_user,
            "password_hash": "dummy_hash",
        }

        refresh_token = auth_manager.create_refresh_token(test_user)

        # Attempt to refresh access token
        (
            new_access_token,
            new_refresh_token,
        ) = auth_manager.refresh_access_token(refresh_token)

        assert isinstance(new_access_token, str)
        assert isinstance(new_refresh_token, str)
        assert new_access_token != refresh_token
        assert new_refresh_token != refresh_token

        # Verify new access token is valid
        token_data = auth_manager.verify_token(new_access_token)
        assert token_data.user_id == test_user.user_id

    def test_token_expiration_handling(self, auth_manager, test_user):
        """Test token expiration and rejection of expired tokens."""
        # Create token with very short expiration
        with patch("auth.security_implementation.ACCESS_TOKEN_EXPIRE_MINUTES", 0):
            token = auth_manager.create_access_token(test_user)

            # Wait for token to expire
            time.sleep(1)

            # Verify token is rejected
            with pytest.raises(Exception) as exc_info:
                auth_manager.verify_token(token)

            assert "expired" in str(exc_info.value).lower()

    def test_invalid_token_signature(self, auth_manager, test_user):
        """Test rejection of tokens with invalid signatures."""
        token = auth_manager.create_access_token(test_user)

        # Tamper with token signature
        tampered_token = token[:-10] + "tampered123"

        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(tampered_token)

        assert "invalid" in str(exc_info.value).lower()

    def test_token_blacklisting(self, auth_manager, test_user):
        """Test token blacklisting functionality."""
        token = auth_manager.create_access_token(test_user)

        # Verify token is valid initially
        token_data = auth_manager.verify_token(token)
        assert token_data.user_id == test_user.user_id

        # Extract JTI and blacklist the token
        payload = jwt.decode(token, options={"verify_signature": False})
        jti = payload["jti"]
        auth_manager.revoke_token(jti)

        # Verify token is now rejected
        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(token)

        assert "blacklisted" in str(exc_info.value).lower()

    def test_refresh_token_rotation(self, auth_manager, test_user):
        """Test refresh token rotation security feature."""
        # Register the user first
        auth_manager.users[test_user.username] = {
            "user": test_user,
            "password_hash": "dummy_hash",
        }

        # Create initial refresh token
        refresh_token_1 = auth_manager.create_refresh_token(test_user)

        # Use it to get new tokens
        new_access_token, refresh_token_2 = auth_manager.refresh_access_token(refresh_token_1)

        # Verify new refresh token is different
        assert refresh_token_2 != refresh_token_1

        # Verify old refresh token is no longer valid
        with pytest.raises(Exception):
            auth_manager.refresh_access_token(refresh_token_1)

    def test_token_with_client_binding(self, auth_manager, test_user):
        """Test token client binding security feature."""
        client_fingerprint = "test-client-fingerprint-123"

        # Create token with client binding
        token = auth_manager.create_access_token(test_user, client_fingerprint=client_fingerprint)

        # Verify token works with correct fingerprint
        token_data = auth_manager.verify_token(token, client_fingerprint=client_fingerprint)
        assert token_data.user_id == test_user.user_id

        # Verify token fails with wrong fingerprint
        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(token, client_fingerprint="wrong-fingerprint")

        assert "binding" in str(exc_info.value).lower()

    def test_token_without_jti_rejection(self, auth_manager, test_user):
        """Test rejection of tokens without JTI claim."""
        # Create token payload without JTI
        now = datetime.now(timezone.utc)
        payload = {
            "user_id": test_user.user_id,
            "username": test_user.username,
            "role": test_user.role.value,
            "type": "access",
            "iss": "freeagentics",
            "aud": "freeagentics-api",
            "exp": now + timedelta(minutes=15),
            "nbf": now,
            "iat": now,
            "permissions": ["view_agents"],
        }

        # Sign token without JTI
        token = jwt.encode(payload, auth_manager.private_key, algorithm=ALGORITHM)

        # Verify token is rejected
        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(token)

        assert "jti" in str(exc_info.value).lower()

    def test_wrong_token_type_rejection(self, auth_manager, test_user):
        """Test rejection of tokens with wrong type."""
        refresh_token = auth_manager.create_refresh_token(test_user)

        # Try to use refresh token as access token
        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(refresh_token)

        assert "type" in str(exc_info.value).lower()

    def test_token_issued_in_future_rejection(self, auth_manager, test_user):
        """Test rejection of tokens issued in the future."""
        # Create token with future issued time
        now = datetime.now(timezone.utc)
        future_time = now + timedelta(hours=1)

        payload = {
            "user_id": test_user.user_id,
            "username": test_user.username,
            "role": test_user.role.value,
            "type": "access",
            "jti": "test-jti",
            "iss": "freeagentics",
            "aud": "freeagentics-api",
            "exp": future_time + timedelta(minutes=15),
            "nbf": future_time,
            "iat": future_time,
            "permissions": ["view_agents"],
        }

        # Sign token with future time
        token = jwt.encode(payload, auth_manager.private_key, algorithm=ALGORITHM)

        # Verify token is rejected
        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(token)

        assert (
            "not yet valid" in str(exc_info.value).lower() or "iat" in str(exc_info.value).lower()
        )

    def test_token_wrong_audience_rejection(self, auth_manager, test_user):
        """Test rejection of tokens with wrong audience."""
        # Create token with wrong audience
        now = datetime.now(timezone.utc)
        payload = {
            "user_id": test_user.user_id,
            "username": test_user.username,
            "role": test_user.role.value,
            "type": "access",
            "jti": "test-jti",
            "iss": "freeagentics",
            "aud": "wrong-audience",
            "exp": now + timedelta(minutes=15),
            "nbf": now,
            "iat": now,
            "permissions": ["view_agents"],
        }

        # Sign token with wrong audience
        token = jwt.encode(payload, auth_manager.private_key, algorithm=ALGORITHM)

        # Verify token is rejected
        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(token)

        assert "audience" in str(exc_info.value).lower()

    def test_token_wrong_issuer_rejection(self, auth_manager, test_user):
        """Test rejection of tokens with wrong issuer."""
        # Create token with wrong issuer
        now = datetime.now(timezone.utc)
        payload = {
            "user_id": test_user.user_id,
            "username": test_user.username,
            "role": test_user.role.value,
            "type": "access",
            "jti": "test-jti",
            "iss": "wrong-issuer",
            "aud": "freeagentics-api",
            "exp": now + timedelta(minutes=15),
            "nbf": now,
            "iat": now,
            "permissions": ["view_agents"],
        }

        # Sign token with wrong issuer
        token = jwt.encode(payload, auth_manager.private_key, algorithm=ALGORITHM)

        # Verify token is rejected
        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(token)

        assert "issuer" in str(exc_info.value).lower()

    def test_blacklist_cleanup(self, auth_manager, test_user):
        """Test blacklist cleanup removes expired entries."""
        # Create and blacklist a token
        token = auth_manager.create_access_token(test_user)
        payload = jwt.decode(token, options={"verify_signature": False})
        jti = payload["jti"]

        auth_manager.revoke_token(jti)
        assert jti in auth_manager.blacklist

        # Manually set old timestamp
        auth_manager.blacklist[jti] = datetime.now(timezone.utc) - timedelta(
            days=REFRESH_TOKEN_EXPIRE_DAYS + 2
        )

        # Run cleanup
        auth_manager.cleanup_blacklist()

        # Verify expired entry is removed
        assert jti not in auth_manager.blacklist

    def test_logout_blacklists_token(self, auth_manager, test_user):
        """Test logout properly blacklists the token."""
        token = auth_manager.create_access_token(test_user)
        payload = jwt.decode(token, options={"verify_signature": False})
        jti = payload["jti"]

        # Logout should blacklist the token
        auth_manager.logout(token)
        assert jti in auth_manager.blacklist

        # Verify token is now rejected
        with pytest.raises(Exception):
            auth_manager.verify_token(token)

    def test_concurrent_token_operations(self, auth_manager, test_user):
        """Test thread safety of token operations."""
        import threading

        tokens = []
        errors = []

        def create_and_verify_token():
            try:
                token = auth_manager.create_access_token(test_user)
                tokens.append(token)
                token_data = auth_manager.verify_token(token)
                assert token_data.user_id == test_user.user_id
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_and_verify_token)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0
        assert len(tokens) == 10

        # Verify all tokens are unique
        assert len(set(tokens)) == 10

    def test_token_permissions_validation(self, auth_manager, test_user):
        """Test token contains correct permissions for user role."""
        token = auth_manager.create_access_token(test_user)
        token_data = auth_manager.verify_token(token)

        # Verify permissions match role
        from auth.security_implementation import ROLE_PERMISSIONS

        expected_permissions = ROLE_PERMISSIONS[test_user.role]

        assert len(token_data.permissions) == len(expected_permissions)
        for permission in expected_permissions:
            assert permission in token_data.permissions

    def test_malformed_token_rejection(self, auth_manager):
        """Test rejection of malformed tokens."""
        malformed_tokens = [
            "not.a.token",
            "invalid-token",
            "",
            "a.b",  # Too few parts
            "a.b.c.d",  # Too many parts
            "eyJhbGciOiJIUzI1NiJ9.invalid.signature",  # Invalid middle part
        ]

        for malformed_token in malformed_tokens:
            with pytest.raises(Exception):
                auth_manager.verify_token(malformed_token)

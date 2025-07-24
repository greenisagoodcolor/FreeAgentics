"""
JWT Refresh Token Rotation Tests
Task #14.13 - JWT Security Validation and Hardening - Refresh Token Component

This test suite validates:
1. Refresh token automatic rotation on use
2. Old refresh token revocation
3. Refresh token security validation
4. Token refresh API endpoint functionality
"""

import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from auth.security_implementation import AuthenticationManager, User, UserRole
from fastapi import HTTPException


class TestRefreshTokenRotation:
    """Test refresh token rotation mechanism."""

    def test_should_rotate_refresh_token_on_creation(self):
        """Test that creating a new refresh token rotates the old one."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Create first refresh token
        first_refresh_token = auth_manager.create_refresh_token(user)
        assert user.user_id in auth_manager.refresh_tokens
        assert auth_manager.refresh_tokens[user.user_id] == first_refresh_token

        # Create second refresh token (should rotate/revoke first)
        second_refresh_token = auth_manager.create_refresh_token(user)
        assert auth_manager.refresh_tokens[user.user_id] == second_refresh_token
        assert first_refresh_token != second_refresh_token

        # First token should be blacklisted
        first_payload = jwt.decode(first_refresh_token, options={"verify_signature": False})
        first_jti = first_payload.get("jti")
        assert first_jti in auth_manager.blacklist

    def test_should_create_new_tokens_on_refresh(self):
        """Test that refresh operation creates new access and refresh tokens."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Register user for authentication context
        auth_manager.users[user.username] = {
            "user": user,
            "password_hash": "dummy",
        }

        # Create initial tokens
        original_access_token = auth_manager.create_access_token(user)
        original_refresh_token = auth_manager.create_refresh_token(user)

        # Refresh tokens
        (
            new_access_token,
            new_refresh_token,
        ) = auth_manager.refresh_access_token(original_refresh_token)

        # Should get new tokens
        assert new_access_token != original_access_token
        assert new_refresh_token != original_refresh_token

        # New access token should be valid
        token_data = auth_manager.verify_token(new_access_token)
        assert token_data.user_id == user.user_id

        # Original refresh token should be blacklisted
        original_payload = jwt.decode(original_refresh_token, options={"verify_signature": False})
        original_jti = original_payload.get("jti")
        assert original_jti in auth_manager.blacklist

    def test_should_reject_old_refresh_token_after_rotation(self):
        """Test that old refresh token is rejected after rotation."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Register user
        auth_manager.users[user.username] = {
            "user": user,
            "password_hash": "dummy",
        }

        # Create initial refresh token
        old_refresh_token = auth_manager.create_refresh_token(user)

        # Refresh once (this rotates the token)
        (
            new_access_token,
            new_refresh_token,
        ) = auth_manager.refresh_access_token(old_refresh_token)

        # Try to use old refresh token again - should fail
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.refresh_access_token(old_refresh_token)

        assert exc_info.value.status_code == 401
        assert "revoked" in exc_info.value.detail.lower()

    def test_should_reject_invalid_refresh_token(self):
        """Test rejection of invalid refresh token."""
        auth_manager = AuthenticationManager()

        invalid_tokens = [
            "invalid.token.string",
            "",
            None,
        ]

        for invalid_token in invalid_tokens:
            if invalid_token is None:
                continue
            with pytest.raises(HTTPException) as exc_info:
                auth_manager.refresh_access_token(invalid_token)
            assert exc_info.value.status_code == 401

    def test_should_reject_access_token_as_refresh_token(self):
        """Test that access tokens cannot be used as refresh tokens."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Create access token
        access_token = auth_manager.create_access_token(user)

        # Try to use access token as refresh token - should fail
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.refresh_access_token(access_token)

        assert exc_info.value.status_code == 401
        assert "type" in exc_info.value.detail.lower()

    def test_should_reject_expired_refresh_token(self):
        """Test rejection of expired refresh token."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Create expired refresh token manually
        expired_payload = {
            "user_id": user.user_id,
            "type": "refresh",
            "jti": "expired_refresh_jti",
            "iss": "freeagentics",
            "aud": "freeagentics-api",
            "exp": (datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp(),  # Expired
            "nbf": (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp(),
            "iat": (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp(),
        }

        expired_refresh_token = jwt.encode(
            expired_payload, auth_manager.private_key, algorithm="RS256"
        )

        # Try to refresh with expired token - should fail
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.refresh_access_token(expired_refresh_token)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_should_maintain_user_context_across_refresh(self):
        """Test that user context is maintained across token refresh."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Register user
        auth_manager.users[user.username] = {
            "user": user,
            "password_hash": "dummy",
        }

        # Create initial tokens
        original_access_token = auth_manager.create_access_token(user)
        refresh_token = auth_manager.create_refresh_token(user)

        # Get original token data
        original_token_data = auth_manager.verify_token(original_access_token)

        # Refresh tokens
        (
            new_access_token,
            new_refresh_token,
        ) = auth_manager.refresh_access_token(refresh_token)

        # Get new token data
        new_token_data = auth_manager.verify_token(new_access_token)

        # User context should be the same
        assert new_token_data.user_id == original_token_data.user_id
        assert new_token_data.username == original_token_data.username
        assert new_token_data.role == original_token_data.role
        assert new_token_data.permissions == original_token_data.permissions

    def test_should_support_client_fingerprinting_on_refresh(self):
        """Test that client fingerprinting works with token refresh."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Register user
        auth_manager.users[user.username] = {
            "user": user,
            "password_hash": "dummy",
        }

        client_fingerprint = "test_client_fingerprint"

        # Create tokens with client fingerprint
        auth_manager.create_access_token(user, client_fingerprint=client_fingerprint)
        refresh_token = auth_manager.create_refresh_token(user)

        # Refresh with same fingerprint
        (
            new_access_token,
            new_refresh_token,
        ) = auth_manager.refresh_access_token(refresh_token, client_fingerprint=client_fingerprint)

        # Should work fine
        token_data = auth_manager.verify_token(
            new_access_token, client_fingerprint=client_fingerprint
        )
        assert token_data.user_id == user.user_id

    def test_refresh_token_rotation_performance(self):
        """Test that refresh token rotation doesn't significantly impact performance."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Register user
        auth_manager.users[user.username] = {
            "user": user,
            "password_hash": "dummy",
        }

        # Measure performance of token refresh
        refresh_token = auth_manager.create_refresh_token(user)

        start_time = time.time()
        for _ in range(10):
            (
                new_access_token,
                refresh_token,
            ) = auth_manager.refresh_access_token(refresh_token)
        end_time = time.time()

        # Should complete 10 refreshes quickly (under 1 second)
        assert (end_time - start_time) < 1.0

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )


class TestRefreshTokenSecurityValidation:
    """Test refresh token security validation."""

    def test_should_validate_refresh_token_claims(self):
        """Test validation of all refresh token claims."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        refresh_token = auth_manager.create_refresh_token(user)
        payload = jwt.decode(refresh_token, options={"verify_signature": False})

        # Check all required claims
        required_claims = [
            "user_id",
            "type",
            "jti",
            "iss",
            "aud",
            "exp",
            "nbf",
            "iat",
        ]
        for claim in required_claims:
            assert claim in payload, f"Refresh token must include claim: {claim}"

        # Check claim values
        assert payload["type"] == "refresh"
        assert payload["iss"] == "freeagentics"
        assert payload["aud"] == "freeagentics-api"
        assert payload["user_id"] == user.user_id

    def test_should_prevent_refresh_token_reuse(self):
        """Test that refresh tokens cannot be reused after consumption."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Register user
        auth_manager.users[user.username] = {
            "user": user,
            "password_hash": "dummy",
        }

        # Create refresh token
        refresh_token = auth_manager.create_refresh_token(user)

        # Use refresh token once
        (
            new_access_token,
            new_refresh_token,
        ) = auth_manager.refresh_access_token(refresh_token)

        # Try to use the same refresh token again - should fail
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.refresh_access_token(refresh_token)

        assert exc_info.value.status_code == 401

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )

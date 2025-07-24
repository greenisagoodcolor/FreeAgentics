"""Security-critical tests for JWT handler following TDD principles.

This test suite covers OWASP security requirements for JWT handling:
- Token validation and verification
- Token expiration and revocation
- Refresh token rotation
- Fingerprint validation
- Blacklist functionality
- Key rotation warnings
"""

import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import jwt
import pytest
from auth.jwt_handler import JWTHandler, RefreshTokenStore, TokenBlacklist
from fastapi import HTTPException


class TestTokenBlacklist:
    """Test token blacklist functionality for secure revocation."""

    def test_add_token_to_blacklist(self):
        """Test adding a token to the blacklist."""
        # Arrange
        blacklist = TokenBlacklist()
        jti = "test-jti-123"
        exp = datetime.now(timezone.utc) + timedelta(hours=1)

        # Act
        blacklist.add(jti, exp)

        # Assert
        assert blacklist.is_blacklisted(jti) is True

    def test_non_blacklisted_token_returns_false(self):
        """Test that non-blacklisted tokens return False."""
        # Arrange
        blacklist = TokenBlacklist()

        # Act & Assert
        assert blacklist.is_blacklisted("non-existent-jti") is False

    def test_expired_tokens_are_cleaned_up(self):
        """Test that expired tokens are removed from blacklist."""
        # Arrange
        blacklist = TokenBlacklist()
        blacklist._cleanup_interval = 0  # Force immediate cleanup

        # Add expired token
        expired_jti = "expired-jti"
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        blacklist.add(expired_jti, expired_time)

        # Add valid token
        valid_jti = "valid-jti"
        valid_time = datetime.now(timezone.utc) + timedelta(hours=1)
        blacklist.add(valid_jti, valid_time)

        # Act - Force cleanup by checking a token
        blacklist.is_blacklisted("any-token")

        # Assert
        assert blacklist.is_blacklisted(expired_jti) is False
        assert blacklist.is_blacklisted(valid_jti) is True


class TestRefreshTokenStore:
    """Test refresh token storage and rotation security."""

    def test_store_refresh_token(self):
        """Test storing a refresh token securely."""
        # Arrange
        store = RefreshTokenStore()
        user_id = "user-123"
        token = "refresh-token-abc"

        # Act
        family_id = store.store(user_id, token)

        # Assert
        assert family_id is not None
        assert len(family_id) > 20  # Ensure secure random ID

    def test_verify_valid_refresh_token(self):
        """Test verifying a valid refresh token."""
        # Arrange
        store = RefreshTokenStore()
        user_id = "user-123"
        token = "refresh-token-abc"
        family_id = store.store(user_id, token)

        # Act
        result_family_id = store.verify_and_rotate(user_id, token)

        # Assert
        assert result_family_id == family_id

    def test_verify_invalid_refresh_token(self):
        """Test that invalid refresh tokens are rejected."""
        # Arrange
        store = RefreshTokenStore()
        user_id = "user-123"
        valid_token = "valid-token"
        invalid_token = "invalid-token"
        store.store(user_id, valid_token)

        # Act
        result = store.verify_and_rotate(user_id, invalid_token)

        # Assert
        assert result is None

    def test_token_theft_detection_invalidates_family(self):
        """Test that token theft detection invalidates entire token family."""
        # Arrange
        store = RefreshTokenStore()
        user_id = "user-123"
        token1 = "token-1"
        family_id = store.store(user_id, token1)

        # Simulate token rotation
        token2 = "token-2"
        store.store(user_id, token2, family_id)

        # Act - Attempt to use old token (theft scenario)
        result = store.verify_and_rotate(user_id, token1)

        # Assert
        assert result is None
        # Verify entire family is invalidated
        assert store.verify_and_rotate(user_id, token2) is None

    def test_invalidate_user_tokens(self):
        """Test invalidating all tokens for a user."""
        # Arrange
        store = RefreshTokenStore()
        user_id = "user-123"
        token = "refresh-token"
        store.store(user_id, token)

        # Act
        store.invalidate(user_id)

        # Assert
        assert store.verify_and_rotate(user_id, token) is None


class TestJWTHandler:
    """Test JWT handler security functionality."""

    @pytest.fixture
    def jwt_handler(self, tmp_path):
        """Create JWT handler with temporary keys."""
        # Create temporary key directory
        keys_dir = tmp_path / "keys"
        keys_dir.mkdir()

        # Patch key paths
        with patch("auth.jwt_handler.PRIVATE_KEY_PATH", str(keys_dir / "private.pem")):
            with patch("auth.jwt_handler.PUBLIC_KEY_PATH", str(keys_dir / "public.pem")):
                handler = JWTHandler()
                yield handler

    def test_create_access_token_with_required_claims(self, jwt_handler):
        """Test that access tokens contain all required security claims."""
        # Arrange
        user_id = "user-123"
        username = "testuser"
        role = "user"
        permissions = ["read", "write"]

        # Act
        token = jwt_handler.create_access_token(
            user_id=user_id, username=username, role=role, permissions=permissions
        )

        # Assert - Decode without verification to check claims
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["user_id"] == user_id
        assert payload["username"] == username
        assert payload["role"] == role
        assert payload["permissions"] == permissions
        assert payload["type"] == "access"
        assert "jti" in payload  # Unique token ID
        assert "iat" in payload  # Issued at
        assert "exp" in payload  # Expiration
        assert "nbf" in payload  # Not before
        assert payload["iss"] == "freeagentics-auth"  # Issuer
        assert payload["aud"] == "freeagentics-api"  # Audience

    def test_access_token_expiration(self, jwt_handler):
        """Test that access tokens expire after configured time."""
        # Arrange
        token = jwt_handler.create_access_token(
            user_id="user-123", username="testuser", role="user", permissions=[]
        )

        # Decode to check expiration
        payload = jwt.decode(token, options={"verify_signature": False})
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat_time = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)

        # Assert - Should expire in 15 minutes
        assert (exp_time - iat_time).total_seconds() == 15 * 60

    def test_verify_valid_access_token(self, jwt_handler):
        """Test verifying a valid access token."""
        # Arrange
        user_id = "user-123"
        token = jwt_handler.create_access_token(
            user_id=user_id, username="testuser", role="user", permissions=["read"]
        )

        # Act
        payload = jwt_handler.verify_access_token(token)

        # Assert
        assert payload["user_id"] == user_id
        assert payload["type"] == "access"

    def test_verify_expired_access_token_raises_401(self, jwt_handler):
        """Test that expired tokens are rejected with 401."""
        # Arrange - Create token with past expiration
        now = datetime.now(timezone.utc)
        expired_payload = {
            "user_id": "user-123",
            "type": "access",
            "iat": now - timedelta(hours=2),
            "exp": now - timedelta(hours=1),
            "nbf": now - timedelta(hours=2),
            "iss": "freeagentics-auth",
            "aud": "freeagentics-api",
            "jti": "test-jti",
        }

        expired_token = jwt.encode(expired_payload, jwt_handler.private_key, algorithm="RS256")

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            jwt_handler.verify_access_token(expired_token)

        assert exc_info.value.status_code == 401
        assert "expired" in str(exc_info.value.detail).lower()

    def test_verify_blacklisted_token_raises_401(self, jwt_handler):
        """Test that blacklisted tokens are rejected."""
        # Arrange
        token = jwt_handler.create_access_token(
            user_id="user-123", username="testuser", role="user", permissions=[]
        )

        # Blacklist the token
        jwt_handler.revoke_token(token)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            jwt_handler.verify_access_token(token)

        assert exc_info.value.status_code == 401
        assert "revoked" in str(exc_info.value.detail).lower()

    def test_token_fingerprint_validation(self, jwt_handler):
        """Test that token fingerprints prevent token theft."""
        # Arrange
        fingerprint = jwt_handler.generate_fingerprint()
        token = jwt_handler.create_access_token(
            user_id="user-123",
            username="testuser",
            role="user",
            permissions=[],
            fingerprint=fingerprint,
        )

        # Act & Assert - Valid fingerprint
        payload = jwt_handler.verify_access_token(token, fingerprint)
        assert payload["user_id"] == "user-123"

        # Act & Assert - Invalid fingerprint
        with pytest.raises(HTTPException) as exc_info:
            jwt_handler.verify_access_token(token, "wrong-fingerprint")

        assert exc_info.value.status_code == 401
        assert "fingerprint" in str(exc_info.value.detail).lower()

    def test_refresh_token_rotation(self, jwt_handler):
        """Test secure refresh token rotation."""
        # Arrange
        user_id = "user-123"

        # Create initial refresh token
        refresh_token, family_id = jwt_handler.create_refresh_token(user_id)

        # Act - Rotate token
        new_access, new_refresh, new_family_id = jwt_handler.rotate_refresh_token(
            refresh_token, user_id
        )

        # Assert
        assert new_access is not None
        assert new_refresh is not None
        assert new_refresh != refresh_token  # New token generated
        assert new_family_id == family_id  # Same family

        # Verify old token is blacklisted
        payload = jwt.decode(refresh_token, options={"verify_signature": False})
        assert jwt_handler.blacklist.is_blacklisted(payload["jti"])

    def test_refresh_token_reuse_detection(self, jwt_handler):
        """Test that reused refresh tokens trigger security response."""
        # Arrange
        user_id = "user-123"

        # Create and rotate token
        refresh_token1, _ = jwt_handler.create_refresh_token(user_id)
        _, refresh_token2, _ = jwt_handler.rotate_refresh_token(refresh_token1, user_id)

        # Act & Assert - Attempt to reuse old token
        with pytest.raises(HTTPException) as exc_info:
            jwt_handler.rotate_refresh_token(refresh_token1, user_id)

        assert exc_info.value.status_code == 401
        assert "invalid" in str(exc_info.value.detail).lower()

    def test_verify_wrong_token_type_raises_error(self, jwt_handler):
        """Test that using wrong token type is rejected."""
        # Arrange - Create refresh token
        refresh_token, _ = jwt_handler.create_refresh_token("user-123")

        # Act & Assert - Try to use as access token
        with pytest.raises(HTTPException) as exc_info:
            jwt_handler.verify_access_token(refresh_token)

        assert exc_info.value.status_code == 401
        assert "type" in str(exc_info.value.detail).lower()

    def test_key_rotation_warning(self, jwt_handler, tmp_path, monkeypatch):
        """Test that old keys trigger rotation warnings."""

        # Arrange - Mock file age
        def mock_stat(path):
            stat_result = Mock()
            # Set key age to 85 days (warning threshold is 83 days)
            stat_result.st_mtime = time.time() - (85 * 86400)
            return stat_result

        monkeypatch.setattr(os, "stat", mock_stat)

        # Act - Check key rotation with logging capture
        with patch("auth.jwt_handler.logger") as mock_logger:
            jwt_handler._check_key_rotation()

        # Assert
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "85 days old" in warning_msg
        assert "rotation recommended" in warning_msg

    def test_revoke_user_tokens(self, jwt_handler):
        """Test revoking all tokens for a user."""
        # Arrange
        user_id = "user-123"

        # Create tokens
        jwt_handler.create_access_token(
            user_id=user_id, username="testuser", role="user", permissions=[]
        )
        refresh_token, _ = jwt_handler.create_refresh_token(user_id)

        # Act
        jwt_handler.revoke_user_tokens(user_id)

        # Assert - Refresh token should be invalidated
        with pytest.raises(HTTPException):
            jwt_handler.verify_refresh_token(refresh_token, user_id)

    def test_generate_secure_fingerprint(self, jwt_handler):
        """Test that fingerprints are cryptographically secure."""
        # Act
        fingerprint1 = jwt_handler.generate_fingerprint()
        fingerprint2 = jwt_handler.generate_fingerprint()

        # Assert
        assert len(fingerprint1) >= 32  # Sufficient entropy
        assert fingerprint1 != fingerprint2  # Unique each time
        assert isinstance(fingerprint1, str)  # URL-safe string

    def test_get_key_info(self, jwt_handler):
        """Test retrieving key information for monitoring."""
        # Act
        info = jwt_handler.get_key_info()

        # Assert
        assert info["algorithm"] == "RS256"
        assert info["key_size"] == 4096  # Strong key size
        assert "key_age_days" in info
        assert "rotation_required" in info
        assert "rotation_warning" in info

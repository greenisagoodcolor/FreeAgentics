"""Characterization tests for JWT Handler following Michael Feathers' principles.

These tests capture the CURRENT behavior of the JWT handler to establish
a safety net before refactoring. They document how the system actually works,
not how we think it should work.

Coverage areas:
- Token creation and structure
- Token verification flows
- Blacklist behavior
- Refresh token rotation
- Key management
- Error handling patterns
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from auth.jwt_handler import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    TOKEN_AUDIENCE,
    TOKEN_ISSUER,
    JWTHandler,
    RefreshTokenStore,
    TokenBlacklist,
)


@pytest.fixture
def temp_keys(tmp_path):
    """Create temporary RSA keys for testing."""
    # Generate test keys
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048  # Smaller for tests
    )
    public_key = private_key.public_key()

    # Create temporary key files
    private_path = tmp_path / "jwt_private.pem"
    public_path = tmp_path / "jwt_public.pem"

    # Write keys
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    public_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    return str(private_path), str(public_path), private_key, public_key


@pytest.fixture
def jwt_handler_with_temp_keys(temp_keys, monkeypatch):
    """Create JWT handler with temporary keys."""
    private_path, public_path, _, _ = temp_keys
    monkeypatch.setattr("auth.jwt_handler.PRIVATE_KEY_PATH", private_path)
    monkeypatch.setattr("auth.jwt_handler.PUBLIC_KEY_PATH", public_path)
    return JWTHandler()


class TestJWTHandlerCharacterization:
    """Characterization tests capturing current JWT handler behavior."""

    def test_token_creation_structure(self, jwt_handler_with_temp_keys):
        """Characterize the structure of created access tokens."""
        # Given
        handler = jwt_handler_with_temp_keys
        user_id = "test-user-123"
        username = "testuser"
        role = "user"
        permissions = ["read", "write"]

        # When
        token = handler.create_access_token(user_id, username, role, permissions)

        # Then - Characterize actual token structure
        assert isinstance(token, str)
        assert len(token) > 100  # JWT tokens are typically long

        # Decode to inspect structure
        decoded = jwt.decode(
            token,
            handler.public_key,
            algorithms=[JWT_ALGORITHM],
            audience=TOKEN_AUDIENCE,  # Must provide audience
            issuer=TOKEN_ISSUER,  # Must provide issuer
            options={"verify_exp": False},  # Don't verify expiration for characterization
        )

        # Document actual payload structure
        assert decoded["user_id"] == user_id
        assert decoded["username"] == username
        assert decoded["role"] == role
        assert decoded["permissions"] == permissions
        assert decoded["type"] == "access"
        assert decoded["iss"] == TOKEN_ISSUER
        assert decoded["aud"] == TOKEN_AUDIENCE
        assert "iat" in decoded
        assert "exp" in decoded
        assert "nbf" in decoded
        assert "jti" in decoded
        assert len(decoded["jti"]) > 20  # Random token ID

    def test_token_expiration_timing(self, jwt_handler_with_temp_keys):
        """Characterize token expiration behavior."""
        # Given
        handler = jwt_handler_with_temp_keys

        # When
        token = handler.create_access_token("user", "username", "role", [])
        decoded = jwt.decode(
            token,
            handler.public_key,
            algorithms=[JWT_ALGORITHM],
            audience=TOKEN_AUDIENCE,
            issuer=TOKEN_ISSUER,
            options={"verify_exp": False},
        )

        # Then - Document actual expiration logic
        iat = decoded["iat"]
        exp = decoded["exp"]
        actual_duration_minutes = (exp - iat) / 60

        assert actual_duration_minutes == ACCESS_TOKEN_EXPIRE_MINUTES

    def test_fingerprint_hashing_behavior(self, jwt_handler_with_temp_keys):
        """Characterize fingerprint hashing in tokens."""
        # Given
        handler = jwt_handler_with_temp_keys
        fingerprint = "test-fingerprint-12345"

        # When
        token = handler.create_access_token("user", "username", "role", [], fingerprint=fingerprint)
        decoded = jwt.decode(
            token,
            handler.public_key,
            algorithms=[JWT_ALGORITHM],
            audience=TOKEN_AUDIENCE,
            issuer=TOKEN_ISSUER,
            options={"verify_exp": False},
        )

        # Then - Document how fingerprints are stored
        assert "fingerprint" in decoded
        # Verify it's hashed, not plain text
        assert decoded["fingerprint"] != fingerprint
        # Verify it's SHA256 (64 hex chars)
        assert len(decoded["fingerprint"]) == 64
        assert decoded["fingerprint"] == hashlib.sha256(fingerprint.encode()).hexdigest()

    def test_refresh_token_family_tracking(self, jwt_handler_with_temp_keys):
        """Characterize refresh token family behavior."""
        # Given
        handler = jwt_handler_with_temp_keys
        user_id = "user-123"

        # When - Create initial refresh token
        token1, family_id1 = handler.create_refresh_token(user_id)

        # Then
        assert isinstance(token1, str)
        assert isinstance(family_id1, str)
        assert len(family_id1) > 20  # Secure random ID

        # When - Create token with same family
        token2, family_id2 = handler.create_refresh_token(user_id, family_id1)

        # Then - Same family ID is maintained
        assert family_id2 == family_id1

    def test_token_verification_success_flow(self, jwt_handler_with_temp_keys):
        """Characterize successful token verification."""
        # Given
        handler = jwt_handler_with_temp_keys
        user_id = "user-123"
        username = "testuser"
        role = "admin"
        permissions = ["all"]

        # When
        token = handler.create_access_token(user_id, username, role, permissions)
        result = handler.verify_access_token(token)

        # Then - Document what's returned on success
        assert isinstance(result, dict)
        assert result["user_id"] == user_id
        assert result["username"] == username
        assert result["role"] == role
        assert result["permissions"] == permissions
        assert result["type"] == "access"

    def test_token_verification_with_fingerprint(self, jwt_handler_with_temp_keys):
        """Characterize fingerprint verification behavior."""
        # Given
        handler = jwt_handler_with_temp_keys
        fingerprint = "device-fingerprint-xyz"

        # When - Create token with fingerprint
        token = handler.create_access_token("user", "username", "role", [], fingerprint=fingerprint)

        # Then - Verification succeeds with correct fingerprint
        result = handler.verify_access_token(token, fingerprint=fingerprint)
        assert result["user_id"] == "user"

        # When - Wrong fingerprint
        with pytest.raises(HTTPException) as exc_info:
            handler.verify_access_token(token, fingerprint="wrong-fingerprint")

        # Then - Document error behavior
        assert exc_info.value.status_code == 401
        assert "Invalid token fingerprint" in exc_info.value.detail

    def test_blacklist_integration(self, jwt_handler_with_temp_keys):
        """Characterize token blacklist behavior."""
        # Given
        handler = jwt_handler_with_temp_keys
        token = handler.create_access_token("user", "username", "role", [])

        # When - Token works before blacklisting
        result = handler.verify_access_token(token)
        assert result["user_id"] == "user"

        # When - Revoke token
        handler.revoke_token(token)

        # Then - Token is rejected
        with pytest.raises(HTTPException) as exc_info:
            handler.verify_access_token(token)

        assert exc_info.value.status_code == 401
        assert "Token has been revoked" in exc_info.value.detail

    def test_refresh_token_rotation_flow(self, jwt_handler_with_temp_keys):
        """Characterize refresh token rotation behavior."""
        # Given
        handler = jwt_handler_with_temp_keys
        user_id = "user-123"

        # When - Create initial tokens
        access1 = handler.create_access_token(user_id, "user", "role", [])
        refresh1, family_id = handler.create_refresh_token(user_id)

        # When - Rotate tokens
        access2, refresh2, returned_family = handler.rotate_refresh_token(refresh1, user_id)

        # Then - Document rotation behavior
        assert isinstance(access2, str)
        assert isinstance(refresh2, str)
        assert access2 != access1  # New access token
        assert refresh2 != refresh1  # New refresh token
        assert returned_family == family_id  # Same family

        # Old refresh token is invalidated (it was revoked)
        with pytest.raises(HTTPException) as exc_info:
            handler.rotate_refresh_token(refresh1, user_id)
        assert "Token has been revoked" in exc_info.value.detail

    def test_refresh_token_theft_detection(self, jwt_handler_with_temp_keys):
        """Characterize refresh token theft detection."""
        # Given
        handler = jwt_handler_with_temp_keys
        user_id = "user-123"

        # When - Normal flow
        refresh1, family_id = handler.create_refresh_token(user_id)
        _, refresh2, _ = handler.rotate_refresh_token(refresh1, user_id)

        # When - Attempt to reuse old token (theft scenario)
        with pytest.raises(HTTPException):
            handler.rotate_refresh_token(refresh1, user_id)

        # Then - Entire family is invalidated
        with pytest.raises(HTTPException) as exc_info:
            handler.rotate_refresh_token(refresh2, user_id)
        assert "Invalid or reused refresh token" in exc_info.value.detail

    def test_expired_token_handling(self, jwt_handler_with_temp_keys):
        """Characterize expired token behavior."""
        # Given
        handler = jwt_handler_with_temp_keys

        # Create an already-expired token by mocking time
        # We'll decode and re-encode with past expiration
        now = datetime.now(timezone.utc)
        past = now - timedelta(minutes=30)

        # Create token payload manually
        payload = {
            "user_id": "user",
            "username": "username",
            "role": "role",
            "permissions": [],
            "type": "access",
            "iat": past,
            "exp": past + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),  # Still expired
            "nbf": past,
            "iss": TOKEN_ISSUER,
            "aud": TOKEN_AUDIENCE,
            "jti": secrets.token_urlsafe(32),
        }

        # Encode with handler's key
        expired_token = jwt.encode(payload, handler.private_key, algorithm=JWT_ALGORITHM)

        # Then - Verification fails with specific error
        with pytest.raises(HTTPException) as exc_info:
            handler.verify_access_token(expired_token)

        assert exc_info.value.status_code == 401
        assert "Access token expired" in exc_info.value.detail

    def test_key_generation_when_missing(self, tmp_path, monkeypatch):
        """Characterize key generation behavior."""
        # Given - No existing keys
        missing_private = tmp_path / "missing_private.pem"
        missing_public = tmp_path / "missing_public.pem"

        monkeypatch.setattr("auth.jwt_handler.PRIVATE_KEY_PATH", str(missing_private))
        monkeypatch.setattr("auth.jwt_handler.PUBLIC_KEY_PATH", str(missing_public))

        # When - Initialize handler
        handler = JWTHandler()

        # Then - Keys are generated
        assert missing_private.exists()
        assert missing_public.exists()

        # Verify key properties
        assert handler.private_key.key_size == 4096  # Strong key

        # Verify file permissions
        private_stat = missing_private.stat()
        assert oct(private_stat.st_mode)[-3:] == "600"  # Owner read/write only

    def test_key_rotation_warnings(self, jwt_handler_with_temp_keys, monkeypatch, caplog):
        """Characterize key rotation warning behavior."""
        # Given - Old key file
        handler = jwt_handler_with_temp_keys
        old_time = time.time() - (91 * 86400)  # 91 days ago

        # Mock file age
        mock_stat = Mock()
        mock_stat.st_mtime = old_time
        monkeypatch.setattr("os.stat", lambda x: mock_stat)

        # When - Check rotation
        handler._check_key_rotation()

        # Then - Error logged for required rotation
        assert "rotation required" in caplog.text

    def test_invalid_token_error_messages(self, jwt_handler_with_temp_keys):
        """Characterize error messages for various invalid tokens."""
        handler = jwt_handler_with_temp_keys

        # Test cases documenting actual error messages
        test_cases = [
            ("not-a-jwt", "Invalid access token"),
            ("", "Token verification failed"),
            (secrets.token_urlsafe(32), "Invalid access token"),
        ]

        for invalid_token, expected_msg_part in test_cases:
            with pytest.raises(HTTPException) as exc_info:
                handler.verify_access_token(invalid_token)

            assert exc_info.value.status_code == 401
            assert expected_msg_part in exc_info.value.detail

    def test_user_token_revocation(self, jwt_handler_with_temp_keys):
        """Characterize user-level token revocation."""
        # Given
        handler = jwt_handler_with_temp_keys
        user_id = "user-to-revoke"

        # Create tokens
        access_token = handler.create_access_token(user_id, "user", "role", [])
        refresh_token, _ = handler.create_refresh_token(user_id)

        # When - Revoke all user tokens
        handler.revoke_user_tokens(user_id)

        # Then - Refresh token is invalidated
        with pytest.raises(HTTPException):
            handler.verify_refresh_token(refresh_token, user_id)

        # Note: Access token still works (documented limitation)
        result = handler.verify_access_token(access_token)
        assert result["user_id"] == user_id


class TestTokenBlacklistCharacterization:
    """Characterization tests for TokenBlacklist behavior."""

    def test_cleanup_interval_behavior(self):
        """Characterize cleanup interval logic."""
        # Given
        blacklist = TokenBlacklist()
        initial_cleanup_time = blacklist._last_cleanup

        # When - Add token (triggers cleanup check)
        blacklist.add("jti1", datetime.now(timezone.utc) + timedelta(hours=1))

        # Then - No cleanup yet (interval not reached)
        assert blacklist._last_cleanup == initial_cleanup_time

        # When - Force interval to pass
        blacklist._last_cleanup = time.time() - 3601  # Over 1 hour ago
        blacklist.add("jti2", datetime.now(timezone.utc) + timedelta(hours=1))

        # Then - Cleanup occurred
        assert blacklist._last_cleanup > initial_cleanup_time

    def test_concurrent_cleanup_safety(self):
        """Characterize behavior during concurrent access."""
        # Given
        blacklist = TokenBlacklist()

        # Add many tokens
        for i in range(100):
            exp = datetime.now(timezone.utc) + timedelta(hours=1)
            blacklist.add(f"jti-{i}", exp)

        # When - Multiple checks (simulating concurrent access)
        results = []
        for i in range(100):
            results.append(blacklist.is_blacklisted(f"jti-{i}"))

        # Then - All checks succeed
        assert all(results)


class TestRefreshTokenStoreCharacterization:
    """Characterization tests for RefreshTokenStore behavior."""

    def test_token_hashing_method(self):
        """Characterize how tokens are hashed for storage."""
        # Given
        store = RefreshTokenStore()
        token = "my-refresh-token"

        # When
        expected_hash = hashlib.sha256(token.encode()).hexdigest()
        actual_hash = store._hash_token(token)

        # Then - SHA256 is used
        assert actual_hash == expected_hash
        assert len(actual_hash) == 64  # SHA256 produces 64 hex chars

    def test_family_tracking_structure(self):
        """Characterize family tracking data structure."""
        # Given
        store = RefreshTokenStore()

        # When - Store tokens in same family
        family_id = store.store("user1", "token1")
        store.store("user2", "token2", family_id)
        store.store("user3", "token3", family_id)

        # Then - Document internal structure
        assert len(store._token_families[family_id]) == 3
        assert all(len(token_hash) == 64 for token_hash in store._token_families[family_id])

    def test_family_invalidation_cascade(self):
        """Characterize family invalidation behavior."""
        # Given
        store = RefreshTokenStore()

        # Create family with multiple tokens
        family_id = store.store("user1", "token1")
        store.store("user2", "token2", family_id)
        store.store("user3", "token3", family_id)

        # When - Invalidate one user
        store.invalidate("user1")

        # Then - Entire family is invalidated
        assert family_id not in store._token_families
        assert "user1" not in store._tokens
        assert "user2" not in store._tokens
        assert "user3" not in store._tokens

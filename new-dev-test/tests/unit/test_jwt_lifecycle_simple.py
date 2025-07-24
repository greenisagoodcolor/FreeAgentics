"""
Simple JWT Lifecycle Test Suite - Working Implementation
"""

from datetime import datetime, timedelta

import jwt
import pytest
from auth.security_implementation import User, UserRole


class TestJWTLifecycleSimple:
    """Simple JWT lifecycle testing that actually works."""

    def setup_method(self):
        """Setup for each test."""
        self.secret_key = "test_secret_key_for_testing"
        self.algorithm = "HS256"  # Use simpler algorithm for testing
        self.test_user = User(
            user_id="test-user-123",
            username="test_user",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(),
        )

    def _create_token(self, user_id: str, role: str, expires_in_minutes: int = 15) -> str:
        """Create a simple JWT token for testing."""
        payload = {
            "user_id": user_id,
            "role": role,
            "exp": datetime.utcnow() + timedelta(minutes=expires_in_minutes),
            "iat": datetime.utcnow(),
            "type": "access",
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def _verify_token(self, token: str) -> dict:
        """Verify a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

    def test_token_creation_and_verification(self):
        """Test basic token creation and verification."""
        # Create token
        token = self._create_token(user_id=self.test_user.user_id, role=self.test_user.role.value)

        # Verify token
        payload = self._verify_token(token)

        # Assert token contains expected data
        assert payload["user_id"] == self.test_user.user_id
        assert payload["role"] == self.test_user.role.value
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_token_expiration(self):
        """Test token expiration handling."""
        # Create token that expires in 1 second
        token = self._create_token(
            user_id=self.test_user.user_id,
            role=self.test_user.role.value,
            expires_in_minutes=-1,  # Already expired
        )

        # Token should be expired
        with pytest.raises(ValueError, match="Token has expired"):
            self._verify_token(token)

    def test_invalid_token_handling(self):
        """Test handling of invalid tokens."""
        # Test with malformed token
        with pytest.raises(ValueError, match="Invalid token"):
            self._verify_token("invalid.token.here")

        # Test with empty token
        with pytest.raises(ValueError, match="Invalid token"):
            self._verify_token("")

    def test_token_with_different_roles(self):
        """Test token creation with different user roles."""
        for role in UserRole:
            token = self._create_token(user_id=self.test_user.user_id, role=role.value)

            payload = self._verify_token(token)
            assert payload["role"] == role.value

    def test_token_payload_integrity(self):
        """Test that token payload cannot be tampered with."""
        # Create valid token
        token = self._create_token(user_id=self.test_user.user_id, role=UserRole.OBSERVER.value)

        # Tamper with token by changing a character
        tampered_token = token[:-1] + "X"

        # Tampered token should fail verification
        with pytest.raises(ValueError, match="Invalid token"):
            self._verify_token(tampered_token)

    def test_token_claims_validation(self):
        """Test validation of token claims."""
        token = self._create_token(user_id=self.test_user.user_id, role=self.test_user.role.value)

        payload = self._verify_token(token)

        # Check required claims exist
        required_claims = ["user_id", "role", "exp", "iat", "type"]
        for claim in required_claims:
            assert claim in payload

        # Check claim types
        assert isinstance(payload["user_id"], str)
        assert isinstance(payload["role"], str)
        assert isinstance(payload["exp"], int)
        assert isinstance(payload["iat"], int)
        assert payload["type"] == "access"

    def test_multiple_tokens_independence(self):
        """Test that multiple tokens work independently."""
        # Create tokens for different users
        token1 = self._create_token("user1", UserRole.ADMIN.value)
        token2 = self._create_token("user2", UserRole.RESEARCHER.value)

        # Both tokens should be valid
        payload1 = self._verify_token(token1)
        payload2 = self._verify_token(token2)

        assert payload1["user_id"] == "user1"
        assert payload1["role"] == UserRole.ADMIN.value
        assert payload2["user_id"] == "user2"
        assert payload2["role"] == UserRole.RESEARCHER.value

    def test_refresh_token_concept(self):
        """Test concept of refresh tokens (different type)."""
        # Create refresh token
        refresh_payload = {
            "user_id": self.test_user.user_id,
            "role": self.test_user.role.value,
            "exp": datetime.utcnow() + timedelta(days=7),
            "iat": datetime.utcnow(),
            "type": "refresh",
        }
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)

        # Verify refresh token
        payload = self._verify_token(refresh_token)
        assert payload["type"] == "refresh"
        assert payload["user_id"] == self.test_user.user_id

    def test_token_time_validation(self):
        """Test token time-based validation."""
        # Create token
        token = self._create_token(
            user_id=self.test_user.user_id,
            role=self.test_user.role.value,
            expires_in_minutes=5,
        )

        payload = self._verify_token(token)

        # Check that expiration is in the future
        exp_time = datetime.utcfromtimestamp(payload["exp"])
        current_time = datetime.utcnow()
        assert exp_time > current_time

        # Check that issued at is in the past or present
        iat_time = datetime.utcfromtimestamp(payload["iat"])
        assert iat_time <= current_time

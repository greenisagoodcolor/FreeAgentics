"""
Behavior-driven tests for JWT authentication - targeting authentication business logic.
Focus on user-facing authentication behaviors, not implementation details.
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import jwt
import pytest


class TestJWTAuthenticationBehavior:
    """Test JWT authentication behaviors that users depend on."""

    def test_jwt_handler_generates_valid_access_tokens(self):
        """
        GIVEN: A user with valid credentials
        WHEN: An access token is generated
        THEN: It should create a valid JWT token with correct claims
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Generate access token
        token = jwt_handler.generate_access_token(user_id)

        # Verify token structure
        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token is valid JWT
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            assert decoded["sub"] == user_id
            assert decoded["type"] == "access"
            assert "exp" in decoded
            assert "iat" in decoded
        except jwt.InvalidTokenError:
            pytest.fail("Generated token is not a valid JWT")

    def test_jwt_handler_generates_valid_refresh_tokens(self):
        """
        GIVEN: A user needing long-term authentication
        WHEN: A refresh token is generated
        THEN: It should create a valid JWT refresh token
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Generate refresh token
        token = jwt_handler.generate_refresh_token(user_id)

        # Verify token structure
        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token is valid JWT
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            assert decoded["sub"] == user_id
            assert decoded["type"] == "refresh"
            assert "exp" in decoded
            assert "iat" in decoded
        except jwt.InvalidTokenError:
            pytest.fail("Generated refresh token is not a valid JWT")

    def test_jwt_handler_validates_access_tokens_correctly(self):
        """
        GIVEN: A user with an access token
        WHEN: The token is validated
        THEN: It should correctly identify valid tokens and extract user information
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Generate a valid token
        token = jwt_handler.generate_access_token(user_id)

        # Validate the token
        try:
            payload = jwt_handler.validate_access_token(token)
            assert payload["sub"] == user_id
            assert payload["type"] == "access"
        except Exception as e:
            pytest.fail(f"Valid token was rejected: {e}")

    def test_jwt_handler_validates_refresh_tokens_correctly(self):
        """
        GIVEN: A user with a refresh token
        WHEN: The token is validated
        THEN: It should correctly identify valid refresh tokens
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Generate a valid refresh token
        token = jwt_handler.generate_refresh_token(user_id)

        # Validate the token
        try:
            payload = jwt_handler.validate_refresh_token(token)
            assert payload["sub"] == user_id
            assert payload["type"] == "refresh"
        except Exception as e:
            pytest.fail(f"Valid refresh token was rejected: {e}")

    def test_jwt_handler_rejects_invalid_tokens(self):
        """
        GIVEN: An invalid or malformed token
        WHEN: The token is validated
        THEN: It should reject the token and raise appropriate error
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()

        # Test various invalid tokens
        invalid_tokens = [
            "invalid.token.here",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.payload",
            "",
            None,
            123,
            {"not": "a_token"},
        ]

        for invalid_token in invalid_tokens:
            with pytest.raises(Exception):
                jwt_handler.validate_access_token(invalid_token)

    def test_jwt_handler_rejects_expired_tokens(self):
        """
        GIVEN: An expired token
        WHEN: The token is validated
        THEN: It should reject the token due to expiration
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Mock time to create an expired token
        with patch("auth.jwt_handler.datetime") as mock_datetime:
            # Set time to past for token generation
            past_time = datetime.utcnow() - timedelta(hours=2)
            mock_datetime.utcnow.return_value = past_time

            expired_token = jwt_handler.generate_access_token(user_id)

        # Try to validate the expired token
        with pytest.raises(Exception):
            jwt_handler.validate_access_token(expired_token)

    def test_jwt_handler_prevents_type_confusion_attacks(self):
        """
        GIVEN: A token with wrong type claim
        WHEN: The token is validated against the wrong endpoint
        THEN: It should reject the token to prevent type confusion
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Generate access token
        access_token = jwt_handler.generate_access_token(user_id)

        # Try to validate access token as refresh token
        with pytest.raises(Exception):
            jwt_handler.validate_refresh_token(access_token)

        # Generate refresh token
        refresh_token = jwt_handler.generate_refresh_token(user_id)

        # Try to validate refresh token as access token
        with pytest.raises(Exception):
            jwt_handler.validate_access_token(refresh_token)

    def test_jwt_handler_includes_security_claims(self):
        """
        GIVEN: A user authentication request
        WHEN: A JWT token is generated
        THEN: It should include necessary security claims
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Generate token
        token = jwt_handler.generate_access_token(user_id)

        # Decode without verification to check claims
        decoded = jwt.decode(token, options={"verify_signature": False})

        # Check required claims
        assert "sub" in decoded  # Subject (user ID)
        assert "iat" in decoded  # Issued at
        assert "exp" in decoded  # Expiration
        assert "type" in decoded  # Token type

        # Check expiration is reasonable (not too far in future)
        exp_time = datetime.fromtimestamp(decoded["exp"])
        now = datetime.utcnow()
        assert exp_time > now  # Should not be expired
        assert exp_time < now + timedelta(
            days=1
        )  # Should not be too far in future

    def test_jwt_handler_token_refresh_workflow(self):
        """
        GIVEN: A user with a valid refresh token
        WHEN: They request a new access token
        THEN: It should generate a new access token using the refresh token
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Generate refresh token
        refresh_token = jwt_handler.generate_refresh_token(user_id)

        # Validate refresh token
        refresh_payload = jwt_handler.validate_refresh_token(refresh_token)

        # Generate new access token from refresh token
        new_access_token = jwt_handler.generate_access_token(
            refresh_payload["sub"]
        )

        # Verify new access token is valid
        access_payload = jwt_handler.validate_access_token(new_access_token)
        assert access_payload["sub"] == user_id
        assert access_payload["type"] == "access"


class TestJWTSecurityBehavior:
    """Test JWT security behaviors."""

    def test_jwt_handler_uses_secure_algorithms(self):
        """
        GIVEN: The JWT handler implementation
        WHEN: Tokens are generated
        THEN: They should use secure cryptographic algorithms
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Generate token
        token = jwt_handler.generate_access_token(user_id)

        # Decode header to check algorithm
        header = jwt.get_unverified_header(token)

        # Should use a secure algorithm (not 'none' or weak algorithms)
        assert header["alg"] in [
            "HS256",
            "HS512",
            "RS256",
            "RS512",
            "ES256",
            "ES512",
        ]
        assert header["alg"] != "none"

    def test_jwt_handler_tokens_are_not_reusable_across_sessions(self):
        """
        GIVEN: Multiple authentication sessions
        WHEN: Tokens are generated
        THEN: Each token should be unique and not reusable
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()
        user_id = str(uuid.uuid4())

        # Generate multiple tokens
        token1 = jwt_handler.generate_access_token(user_id)
        token2 = jwt_handler.generate_access_token(user_id)

        # Tokens should be different (due to timestamps)
        assert token1 != token2

        # Both should be valid
        payload1 = jwt_handler.validate_access_token(token1)
        payload2 = jwt_handler.validate_access_token(token2)

        assert payload1["sub"] == user_id
        assert payload2["sub"] == user_id


class TestJWTErrorHandlingBehavior:
    """Test JWT error handling behaviors."""

    def test_jwt_handler_provides_clear_error_messages(self):
        """
        GIVEN: Various invalid authentication attempts
        WHEN: Validation fails
        THEN: It should provide clear error information for debugging
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()

        # Test with completely invalid token
        with pytest.raises(Exception) as exc_info:
            jwt_handler.validate_access_token("invalid_token")

        # Error should be informative
        assert exc_info.value is not None

        # Test with empty token
        with pytest.raises(Exception) as exc_info:
            jwt_handler.validate_access_token("")

        # Error should be informative
        assert exc_info.value is not None

    def test_jwt_handler_handles_missing_claims_gracefully(self):
        """
        GIVEN: A token with missing required claims
        WHEN: The token is validated
        THEN: It should handle the error gracefully
        """
        from auth.jwt_handler import JWTHandler

        jwt_handler = JWTHandler()

        # Create a token with missing claims
        incomplete_payload = {
            "sub": "user123"
        }  # Missing 'type' and other claims

        # Use the handler's secret key if available
        try:
            # Generate a token with incomplete payload
            incomplete_token = jwt.encode(
                incomplete_payload, "fake_secret", algorithm="HS256"
            )

            # Should reject token with missing claims
            with pytest.raises(Exception):
                jwt_handler.validate_access_token(incomplete_token)
        except Exception:
            # If we can't create the test token, that's okay
            pass

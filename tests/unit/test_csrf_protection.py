"""
Tests for CSRF protection implementation.

Verifies that CSRF protection is properly implemented on state-changing endpoints.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from auth import TokenData, UserRole, auth_manager, require_csrf_token


class TestCSRFProtection:
    """Test CSRF protection functionality."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = MagicMock(spec=Request)
        request.headers = {}
        request.cookies = {}
        return request

    @pytest.fixture
    def test_token_data(self):
        """Create test token data."""
        return TokenData(
            user_id="test-user-123",
            username="testuser",
            role=UserRole.RESEARCHER,
            permissions=[],
            exp=datetime.now(timezone.utc),
        )

    async def test_csrf_token_required_decorator(self, mock_request):
        """Test that CSRF token is required for protected endpoints."""

        # Create a protected function
        @require_csrf_token
        async def protected_endpoint(request: Request):
            return {"status": "success"}

        # Test without CSRF token - should fail
        mock_request.headers = {"Authorization": "Bearer fake-token"}

        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint(mock_request)

        assert exc_info.value.status_code == 403
        assert "CSRF token required" in exc_info.value.detail

    async def test_csrf_token_validation_success(
        self, mock_request, test_token_data
    ):
        """Test successful CSRF token validation."""
        # Generate CSRF token
        csrf_token = auth_manager.csrf_protection.generate_csrf_token(
            test_token_data.user_id
        )

        # Create protected function
        @require_csrf_token
        async def protected_endpoint(
            request: Request, current_user: TokenData
        ):
            return {"status": "success", "user": current_user.username}

        # Add CSRF token to request
        mock_request.headers = {
            "Authorization": f"Bearer fake-token",
            "X-CSRF-Token": csrf_token,
        }

        # Mock JWT decode to return user_id
        import jwt

        with pytest.mock.patch.object(jwt, "decode") as mock_decode:
            mock_decode.return_value = {"user_id": test_token_data.user_id}

            # Should succeed with valid CSRF token
            result = await protected_endpoint(mock_request, test_token_data)
            assert result["status"] == "success"

    async def test_csrf_token_validation_failure(
        self, mock_request, test_token_data
    ):
        """Test CSRF token validation with wrong token."""
        # Generate CSRF token but use wrong one in request
        auth_manager.csrf_protection.generate_csrf_token(
            test_token_data.user_id
        )

        # Create protected function
        @require_csrf_token
        async def protected_endpoint(
            request: Request, current_user: TokenData
        ):
            return {"status": "success"}

        # Add wrong CSRF token
        mock_request.headers = {
            "Authorization": f"Bearer fake-token",
            "X-CSRF-Token": "wrong-csrf-token",
        }

        # Mock JWT decode
        import jwt

        with pytest.mock.patch.object(jwt, "decode") as mock_decode:
            mock_decode.return_value = {"user_id": test_token_data.user_id}

            # Should fail with wrong CSRF token
            with pytest.raises(HTTPException) as exc_info:
                await protected_endpoint(mock_request, test_token_data)

            assert exc_info.value.status_code == 403
            assert "Invalid CSRF token" in exc_info.value.detail

    def test_csrf_token_generation_unique(self):
        """Test that CSRF tokens are unique per session."""
        session1 = "user-session-1"
        session2 = "user-session-2"

        token1 = auth_manager.csrf_protection.generate_csrf_token(session1)
        token2 = auth_manager.csrf_protection.generate_csrf_token(session2)

        # Tokens should be different
        assert token1 != token2

        # Each token should work only for its session
        assert auth_manager.csrf_protection.verify_csrf_token(session1, token1)
        assert not auth_manager.csrf_protection.verify_csrf_token(
            session1, token2
        )
        assert auth_manager.csrf_protection.verify_csrf_token(session2, token2)
        assert not auth_manager.csrf_protection.verify_csrf_token(
            session2, token1
        )

    def test_csrf_token_invalidation(self):
        """Test CSRF token invalidation on logout."""
        session_id = "test-session"

        # Generate token
        csrf_token = auth_manager.csrf_protection.generate_csrf_token(
            session_id
        )
        assert auth_manager.csrf_protection.verify_csrf_token(
            session_id, csrf_token
        )

        # Invalidate token
        auth_manager.csrf_protection.invalidate_csrf_token(session_id)

        # Token should no longer be valid
        assert not auth_manager.csrf_protection.verify_csrf_token(
            session_id, csrf_token
        )

    async def test_csrf_form_data_support(self, mock_request):
        """Test CSRF token can be extracted from form data."""
        # Create mock form data
        form_data = MagicMock()
        form_data.get.return_value = "csrf-token-from-form"

        async def mock_form():
            return form_data

        mock_request.form = mock_form
        mock_request.headers = {"Authorization": "Bearer fake-token"}

        # Create protected function
        @require_csrf_token
        async def protected_endpoint(request: Request):
            return {"status": "success"}

        # Mock JWT decode
        import jwt

        with pytest.mock.patch.object(jwt, "decode") as mock_decode:
            mock_decode.return_value = {"user_id": "test-user"}

            # Generate matching CSRF token
            auth_manager.csrf_protection.generate_csrf_token("test-user")
            auth_manager.csrf_protection._token_store[
                "test-user"
            ] = "csrf-token-from-form"

            # Should succeed with CSRF token from form
            result = await protected_endpoint(mock_request)
            assert result["status"] == "success"

    def test_csrf_protection_integration_with_auth_manager(self):
        """Test CSRF protection is integrated with auth manager."""
        # Verify auth manager has CSRF protection
        assert hasattr(auth_manager, "csrf_protection")
        assert auth_manager.csrf_protection is not None

        # Verify CSRF methods are available
        assert hasattr(auth_manager, "set_csrf_cookie")
        assert callable(auth_manager.set_csrf_cookie)

    def test_csrf_token_length_and_randomness(self):
        """Test CSRF tokens are sufficiently long and random."""
        tokens = set()

        # Generate multiple tokens
        for i in range(100):
            token = auth_manager.csrf_protection.generate_csrf_token(
                f"session-{i}"
            )
            tokens.add(token)

            # Each token should be at least 32 characters
            assert len(token) >= 32

        # All tokens should be unique
        assert len(tokens) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

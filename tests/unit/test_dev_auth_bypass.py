"""Unit tests for dev mode authentication bypass."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from auth.dev_bypass import (
    get_current_user_optional,
    get_dev_user,
    is_dev_token,
    _DEV_USER
)
from auth.security_implementation import TokenData, UserRole


@pytest.mark.asyncio
class TestDevBypass:
    """Test dev mode authentication bypass."""

    @patch('auth.dev_bypass.environment')
    async def test_get_current_user_optional_dev_mode(self, mock_env):
        """Test that dev user is returned in dev mode without auth."""
        # Setup
        mock_env.is_development = True
        mock_env.config.auth_required = False
        
        request = Mock(spec=Request)
        
        # Execute
        result = await get_current_user_optional(request, None)
        
        # Verify
        assert result == _DEV_USER
        assert result.user_id == "dev_user"
        assert result.username == "dev_user"
        assert result.role == UserRole.ADMIN

    @patch('auth.dev_bypass.environment')
    @patch('auth.security_implementation.get_current_user')
    async def test_get_current_user_optional_production(self, mock_get_user, mock_env):
        """Test that real auth is used when not in dev mode."""
        # Setup
        mock_env.is_development = False
        mock_env.config.auth_required = True
        
        expected_user = Mock(spec=TokenData)
        mock_get_user.return_value = expected_user
        
        request = Mock(spec=Request)
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        
        # Execute
        result = await get_current_user_optional(request, credentials)
        
        # Verify
        assert result == expected_user
        mock_get_user.assert_called_once_with(request, credentials)

    @patch('auth.dev_bypass.environment')
    def test_get_dev_user_success(self, mock_env):
        """Test getting dev user in dev mode."""
        # Setup
        mock_env.is_development = True
        mock_env.config.auth_required = False
        
        # Execute
        result = get_dev_user()
        
        # Verify
        assert result == _DEV_USER

    @patch('auth.dev_bypass.environment')
    def test_get_dev_user_fails_in_production(self, mock_env):
        """Test that getting dev user fails in production."""
        # Setup
        mock_env.is_development = False
        mock_env.config.auth_required = True
        
        # Execute & Verify
        with pytest.raises(RuntimeError, match="Dev user only available"):
            get_dev_user()

    @patch('auth.dev_bypass.environment')
    def test_is_dev_token_valid(self, mock_env):
        """Test dev token validation in dev mode."""
        # Setup
        mock_env.is_development = True
        mock_env.config.auth_required = False
        
        # Execute & Verify
        assert is_dev_token("dev") is True
        assert is_dev_token("not-dev") is False

    @patch('auth.dev_bypass.environment')
    def test_is_dev_token_invalid_in_production(self, mock_env):
        """Test dev token is invalid in production."""
        # Setup
        mock_env.is_development = False
        mock_env.config.auth_required = True
        
        # Execute & Verify
        assert is_dev_token("dev") is False

    def test_dev_user_has_admin_permissions(self):
        """Test that dev user has all admin permissions."""
        assert _DEV_USER.role == UserRole.ADMIN
        assert "create_agent" in [p.value for p in _DEV_USER.permissions]
        assert "admin_system" in [p.value for p in _DEV_USER.permissions]
        
    def test_dev_user_token_not_expired(self):
        """Test that dev user token has long expiry."""
        # exp is a datetime object
        expiry_time = _DEV_USER.exp
        now = datetime.utcnow()
        
        # Should be at least 300 days in the future
        assert (expiry_time - now).days > 300
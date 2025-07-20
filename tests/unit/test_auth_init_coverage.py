"""Tests for auth/__init__.py to boost coverage."""

import pytest


class TestAuthInit:
    """Test auth package imports."""

    def test_auth_imports(self):
        """Test that all auth components can be imported."""
        # This will execute the auth/__init__.py file
        from auth import (
            ALGORITHM,
            CSRF_COOKIE_NAME,
            CSRF_HEADER_NAME,
            AuthenticationManager,
            CSRFProtection,
            Permission,
            SecurityMiddleware,
            SecurityValidator,
            TokenData,
            User,
            UserRole,
            auth_manager,
            get_current_user,
            rate_limit,
            rate_limiter,
            require_csrf_token,
            require_permission,
            require_role,
            secure_database_query,
            security_validator,
            validate_csrf_token,
        )

        # Verify imports exist
        assert ALGORITHM is not None
        assert CSRF_COOKIE_NAME is not None
        assert CSRF_HEADER_NAME is not None
        assert AuthenticationManager is not None
        assert CSRFProtection is not None
        assert Permission is not None
        assert SecurityMiddleware is not None
        assert SecurityValidator is not None
        assert TokenData is not None
        assert User is not None
        assert UserRole is not None
        assert auth_manager is not None
        assert get_current_user is not None
        assert rate_limit is not None
        assert rate_limiter is not None
        assert require_csrf_token is not None
        assert require_permission is not None
        assert require_role is not None
        assert secure_database_query is not None
        assert security_validator is not None
        assert validate_csrf_token is not None

    def test_auth_all_exports(self):
        """Test __all__ exports are correct."""
        import auth

        expected_exports = [
            "ALGORITHM",
            "AuthenticationManager",
            "auth_manager",
            "rate_limiter",
            "security_validator",
            "get_current_user",
            "require_permission",
            "require_role",
            "require_csrf_token",
            "rate_limit",
            "secure_database_query",
            "SecurityMiddleware",
            "User",
            "UserRole",
            "Permission",
            "TokenData",
            "SecurityValidator",
            "CSRFProtection",
            "CSRF_HEADER_NAME",
            "CSRF_COOKIE_NAME",
            "validate_csrf_token",
        ]

        assert hasattr(auth, "__all__")
        assert set(auth.__all__) == set(expected_exports)

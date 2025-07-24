"""
pytest configuration for security tests.

Provides shared fixtures and configuration for all security tests.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Import the main app
from api.main import app


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(scope="session")
def production_environment():
    """Set up production environment variables for testing."""
    original_env = os.environ.copy()

    # Set production environment variables
    test_env = {
        "PRODUCTION": "true",
        "SECRET_KEY": "test_secret_key_for_production_testing_only_very_long_and_secure",
        "JWT_SECRET": "test_jwt_secret_for_production_testing_only_very_long_and_secure",
        "DATABASE_URL": "postgresql://testuser:testpassword@localhost:5432/testdb",
        "REDIS_URL": "redis://localhost:6379/0",
        "ENVIRONMENT": "production",
        "DEBUG": "false",
        "LOG_LEVEL": "INFO",
    }

    # Apply test environment
    for key, value in test_env.items():
        os.environ[key] = value

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_database_error():
    """Mock database errors for testing error handling."""
    from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

    def raise_db_error(error_type="operational"):
        if error_type == "operational":
            raise OperationalError("Connection to database failed", None, None)
        elif error_type == "integrity":
            raise IntegrityError("Duplicate key violates unique constraint", None, None)
        else:
            raise SQLAlchemyError("Generic database error")

    return raise_db_error


@pytest.fixture
def mock_auth_manager():
    """Mock authentication manager for testing."""

    with patch("auth.security_implementation.auth_manager") as mock_auth:
        mock_auth.verify_token.side_effect = Exception("JWT verification failed")
        mock_auth.authenticate_user.return_value = None
        yield mock_auth


@pytest.fixture
def temporary_log_file():
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as log_file:
        yield log_file.name

    # Clean up
    try:
        os.unlink(log_file.name)
    except OSError:
        pass


@pytest.fixture
def security_test_data():
    """Provide test data for security tests."""
    return {
        "sql_injection_payloads": [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; SELECT * FROM information_schema.tables; --",
            "' UNION SELECT NULL, version(), NULL --",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
        ],
        "xss_payloads": [
            "<script>alert('xss')</script>",
            '"><script>alert("xss")</script>',
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
        ],
        "command_injection_payloads": [
            "; ls -la",
            "&& cat /etc/passwd",
            "| whoami",
            "`id`",
            "$(whoami)",
        ],
        "path_traversal_payloads": [
            "../../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "/var/log/auth.log",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
        ],
        "sensitive_files": [
            ".env",
            "config.ini",
            "settings.json",
            "database.yml",
            "secrets.txt",
            "private.key",
            "id_rsa",
            "credentials.json",
        ],
    }


@pytest.fixture
def mock_security_headers():
    """Mock security headers configuration."""
    from auth.security_headers import SecurityHeadersManager, SecurityPolicy

    # Create a test security policy
    test_policy = SecurityPolicy(
        enable_hsts=True,
        hsts_max_age=31536000,
        csp_policy="default-src 'self'",
        x_frame_options="DENY",
        production_mode=True,
    )

    return SecurityHeadersManager(test_policy)


@pytest.fixture(autouse=True)
def reset_authentication_state():
    """Reset authentication state before each test."""
    from auth.security_implementation import auth_manager

    # Clear any existing state
    if hasattr(auth_manager, "users"):
        auth_manager.users.clear()
    if hasattr(auth_manager, "blacklist"):
        auth_manager.blacklist.clear()
    if hasattr(auth_manager, "refresh_tokens"):
        auth_manager.refresh_tokens.clear()

    yield

    # Clean up after test
    if hasattr(auth_manager, "users"):
        auth_manager.users.clear()
    if hasattr(auth_manager, "blacklist"):
        auth_manager.blacklist.clear()
    if hasattr(auth_manager, "refresh_tokens"):
        auth_manager.refresh_tokens.clear()


@pytest.fixture
def capture_logs():
    """Capture log output for testing."""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)

    # Add handler to root logger
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)

    yield log_capture

    # Clean up
    root_logger.removeHandler(handler)
    root_logger.setLevel(original_level)


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing."""
    from auth.security_implementation import rate_limiter

    with patch.object(rate_limiter, "is_rate_limited") as mock_rate_limit:
        mock_rate_limit.return_value = False
        yield mock_rate_limit


# Test markers for categorizing tests
pytest_markers = [
    "security: marks tests as security tests",
    "error_handling: marks tests for error handling",
    "authentication: marks tests for authentication security",
    "api_security: marks tests for API security",
    "production_hardening: marks tests for production hardening",
    "slow: marks tests as slow running",
    "integration: marks tests as integration tests",
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add security marker to all tests in security directory
        if "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)

        # Add specific markers based on test file names
        if "error_handling" in str(item.fspath):
            item.add_marker(pytest.mark.error_handling)
        elif "authentication" in str(item.fspath):
            item.add_marker(pytest.mark.authentication)
        elif "api_security" in str(item.fspath):
            item.add_marker(pytest.mark.api_security)
        elif "production_hardening" in str(item.fspath):
            item.add_marker(pytest.mark.production_hardening)

        # Mark tests that might be slow
        if any(
            keyword in item.name.lower()
            for keyword in ["comprehensive", "all_tests", "integration"]
        ):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def test_report_directory():
    """Create directory for test reports."""
    report_dir = "/home/green/FreeAgentics/tests/security/reports"
    os.makedirs(report_dir, exist_ok=True)
    return report_dir

"""Global test configuration and fixtures."""

import os

import pytest
from sqlalchemy import MetaData

# Set development mode to enable SQLite fallback for tests
os.environ["DEVELOPMENT_MODE"] = "true"
os.environ["ENVIRONMENT"] = "development"

# Set test database URL to use SQLite for tests
os.environ["DATABASE_URL"] = "sqlite:///./test_freeagentics.db"

# Disable Redis for tests
os.environ["REDIS_ENABLED"] = "false"
os.environ["TESTING"] = "true"

# JWT config for tests
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["JWT_ACCESS_TOKEN_EXPIRE_MINUTES"] = "30"


@pytest.fixture(scope="session", autouse=True)
def disable_redis():
    """Disable Redis for all tests."""
    os.environ["REDIS_ENABLED"] = "false"
    yield


@pytest.fixture(autouse=True)
def mock_redis_for_api_tests(request):
    """Automatically mock Redis for API tests that need it."""
    # Only apply to API tests that import redis
    if "test_api" in request.node.name:
        try:
            from tests.mocks.mock_redis import patch_redis_imports

            patch_redis_imports()
        except ImportError:
            pass  # Mock not needed
    yield


@pytest.fixture
def mock_redis_client():
    """Provide a mock Redis client."""
    from tests.mocks.mock_redis import create_mock_redis_client

    return create_mock_redis_client()


@pytest.fixture(scope="module")
def _clear_tables_fixture():
    """Module-level fixture to handle SQLAlchemy table definitions.
    
    This helps prevent "Table already defined" errors when multiple test modules
    import the same models by ensuring clean imports at the module level.
    """
    yield

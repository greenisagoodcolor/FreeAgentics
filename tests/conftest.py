"""Global test configuration and fixtures."""

import os

import pytest

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

"""
FreeAgentics Test Configuration
===============================

This conftest.py file automatically configures the test environment for all pytest runs.
It ensures complete test isolation by loading test-specific environment variables
and setting up proper fixtures.

Key features:
- Automatically loads .env.test environment variables
- Ensures in-memory database for complete test isolation 
- Sets up mock LLM providers for deterministic responses
- Configures test-optimized settings for fast execution

Committee consensus: Complete test isolation with zero external dependencies.
"""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv

# Load test environment variables before any imports
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest to use test environment settings."""
    # Get the project root directory
    project_root = Path(__file__).parent
    test_env_path = project_root / ".env.test"
    
    # Clear any existing DATABASE_URL first
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]
    
    if test_env_path.exists():
        # Load test environment variables
        load_dotenv(test_env_path, override=True)
    
    # Always force test settings, regardless of what .env.test contains
    os.environ["TESTING"] = "true"
    os.environ["ENVIRONMENT"] = "test"
    
    # Force in-memory database - MUST override any existing value
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["TEST_DATABASE_URL"] = "sqlite:///:memory:"
    
    # Force mock LLM provider
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["ANTHROPIC_API_KEY"] = ""
    
    # Disable external services
    os.environ["REDIS_URL"] = ""
    os.environ["ENABLE_REAL_LLM_CALLS"] = "false"
    
    print("✓ Test environment loaded")
    print(f"✓ Database: {os.environ.get('DATABASE_URL', 'not set')}")
    print(f"✓ LLM Provider: {os.environ.get('LLM_PROVIDER', 'not set')}")
    print(f"✓ Environment: {os.environ.get('ENVIRONMENT', 'not set')}")


def pytest_sessionstart(session: pytest.Session) -> None:
    """Called after the Session object has been created."""
    print("🧪 Starting test session")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Called after whole test run finished."""
    if exitstatus == 0:
        print("✅ All tests passed - test environment worked correctly")
    else:
        print(f"❌ Tests failed with exit status {exitstatus}")


@pytest.fixture(scope="session", autouse=True)
def ensure_test_environment() -> Generator[None, None, None]:
    """
    Automatically applied fixture that ensures test environment is properly configured.
    
    This fixture runs once per test session and validates that:
    - We're in test mode
    - Database is in-memory
    - LLM provider is mocked
    - No external services are enabled
    """
    # Validate test environment
    assert os.environ.get("TESTING") == "true", "TESTING environment variable not set"
    assert os.environ.get("ENVIRONMENT") == "test", "ENVIRONMENT must be 'test'"
    assert "memory" in os.environ.get("DATABASE_URL", ""), "Database must be in-memory for tests"
    assert os.environ.get("LLM_PROVIDER") == "mock", "LLM provider must be 'mock' for tests"
    
    # Ensure no real API keys are set
    assert not os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be empty in test environment"
    assert not os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY must be empty in test environment"
    
    yield
    
    # Cleanup after tests (if needed)
    # In-memory database automatically cleans up
    pass


@pytest.fixture(scope="function")
def isolated_test_env() -> Generator[dict[str, str], None, None]:
    """
    Provides a completely isolated environment for individual tests.
    
    This fixture can be used by tests that need to modify environment variables
    without affecting other tests.
    """
    # Save current environment
    original_env = dict(os.environ)
    
    try:
        yield dict(os.environ)
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


# Performance monitoring for test execution
@pytest.fixture(autouse=True)
def monitor_test_performance(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Monitor test execution time and warn about slow tests."""
    import time
    
    start_time = time.time()
    yield
    duration = time.time() - start_time
    
    # Warn about slow tests (over 5 seconds)
    if duration > 5.0:
        print(f"⚠️  Slow test detected: {request.node.name} took {duration:.2f}s")


# Database fixtures for tests that need database access
@pytest.fixture(scope="function")
def test_db():
    """
    Provides a fresh in-memory database for each test.
    
    This fixture ensures complete isolation between tests by providing
    a new database instance for each test function.
    """
    from database.session import get_db_session
    from database.base import Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.close()


# Mock fixtures for external dependencies
@pytest.fixture
def mock_llm_provider():
    """Provides a mock LLM provider for tests."""
    from inference.llm.mock_provider import MockLLMProvider
    return MockLLMProvider()


@pytest.fixture
def mock_redis():
    """Provides a mock Redis instance for tests."""
    from tests.mocks.mock_redis import MockRedis
    return MockRedis()


# Test data fixtures
@pytest.fixture
def sample_agent_data():
    """Provides sample agent data for tests."""
    return {
        "name": "test_agent",
        "description": "A test agent for unit testing",
        "capabilities": ["reasoning", "memory"],
        "parameters": {
            "temperature": 0.0,
            "max_tokens": 100
        }
    }


@pytest.fixture
def sample_user_data():
    """Provides sample user data for tests."""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "test_password_123"
    }
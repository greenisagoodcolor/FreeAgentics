"""Test configuration and fixtures."""

import asyncio
import os
import sys
from unittest.mock import MagicMock

import pytest

# Set test environment variables before importing any application modules
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ASYNC_DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["ENVIRONMENT"] = "test"
os.environ["SECRET_KEY"] = "test_secret_key_for_testing_only"
os.environ["JWT_SECRET"] = "test_jwt_secret_for_testing_only"
os.environ["API_KEY"] = "test_api_key"
os.environ["DEVELOPMENT_MODE"] = "false"
os.environ["PRODUCTION"] = "false"

# Add project root to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_database():
    """Mock database for testing without actual DB connection."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.base import Base

    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
async def async_mock_database():
    """Async mock database for testing."""
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from database.base import Base

    # Create async in-memory SQLite database
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_maker() as session:
        yield session
        await session.close()

    await engine.dispose()


@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph without database dependency."""
    mock_kg = MagicMock()
    mock_kg.nodes = MagicMock(return_value=[])
    mock_kg.edges = MagicMock(return_value=[])
    mock_kg.add_node = MagicMock(return_value=True)
    mock_kg.add_edge = MagicMock(return_value=True)
    return mock_kg


@pytest.fixture
def mock_storage_manager():
    """Mock storage manager without database dependency."""
    mock_sm = MagicMock()
    mock_sm.save = MagicMock(return_value=True)
    mock_sm.load = MagicMock(return_value={})
    return mock_sm


# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    from fastapi.testclient import TestClient

    from api.main import app
    from auth.security_implementation import get_current_user
    from database.session import get_db
    from tests.test_helpers.auth_helpers import mock_auth_dependency
    from tests.test_helpers.db_helpers import get_test_db

    # Override dependencies
    app.dependency_overrides[get_db] = get_test_db
    app.dependency_overrides[get_current_user] = mock_auth_dependency

    with TestClient(app) as test_client:
        yield test_client

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def authenticated_client(client):
    """Create a test client with authentication headers."""
    client.headers["Authorization"] = "Bearer test_token"
    return client

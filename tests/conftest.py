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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from database.base import Base

    # Create async in-memory SQLite database
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

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

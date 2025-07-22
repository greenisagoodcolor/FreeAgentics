"""Test fixtures for database testing with transaction isolation.

Provides pytest fixtures and utilities for isolated database tests,
ensuring each test runs in its own transaction that gets rolled back.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .test_config import (
    create_test_engine,
    setup_test_database,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def db_engine() -> Generator[Engine, None, None]:
    """Create database engine for the test session.

    This fixture is session-scoped, so the engine is created once
    and reused across all tests.
    """
    engine = create_test_engine()
    setup_test_database(engine)

    yield engine

    # Cleanup after all tests
    engine.dispose()


@pytest.fixture(scope="session")
def sqlite_engine() -> Generator[Engine, None, None]:
    """Create SQLite in-memory engine for fast unit tests.

    This fixture is session-scoped and uses an in-memory SQLite database.
    """
    engine = create_test_engine(use_sqlite=True)
    setup_test_database(engine)

    yield engine

    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Generator[Session, None, None]:
    """Provide a transactional database session for tests.

    Each test gets its own transaction that is rolled back after the test,
    ensuring complete isolation between tests.
    """
    connection = db_engine.connect()
    transaction = connection.begin()

    # Create session bound to the connection
    session = Session(bind=connection)

    # Begin a nested transaction (savepoint)
    nested = connection.begin_nested()

    # If the test prematurely exits, ensure rollback
    @event.listens_for(session, "after_transaction_end")
    def end_savepoint(session, transaction):
        nonlocal nested
        if not nested.is_active:
            nested = connection.begin_nested()

    yield session

    # Rollback everything
    session.close()
    if nested.is_active:
        nested.rollback()
    if transaction.is_active:
        transaction.rollback()
    connection.close()


@pytest.fixture
def sqlite_session(sqlite_engine: Engine) -> Generator[Session, None, None]:
    """Provide a transactional SQLite session for unit tests.

    Similar to db_session but uses SQLite for faster execution.
    """
    connection = sqlite_engine.connect()
    transaction = connection.begin()

    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def async_db_session(db_engine: Engine) -> Generator[Session, None, None]:
    """Provide an async-compatible database session.

    For tests that need to simulate async database operations.
    """
    # For now, we use sync sessions even in async tests
    # In production, you'd use asyncpg or similar
    connection = db_engine.connect()
    transaction = connection.begin()

    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@contextmanager
def isolated_db_test(engine: Optional[Engine] = None):
    """Context manager for isolated database tests.

    Usage:
        with isolated_db_test() as session:
            # Your test code here
            pass
        # Transaction is automatically rolled back
    """
    if engine is None:
        engine = create_test_engine()

    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@contextmanager
def persistent_db_test(engine: Optional[Engine] = None):
    """Context manager for tests that need persistent data.

    Unlike isolated_db_test, this commits the transaction.
    Use sparingly and clean up after yourself!
    """
    if engine is None:
        engine = create_test_engine()

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class DatabaseTestCase:
    """Base class for database test cases with common functionality."""

    @pytest.fixture(autouse=True)
    def setup_db(self, db_session):
        """Automatically inject db_session into test methods."""
        self.db_session = db_session
        self.setup_test_data()
        yield
        self.cleanup_test_data()

    def setup_test_data(self):
        """Override to set up test data before each test."""
        pass

    def cleanup_test_data(self):
        """Override to clean up after each test."""
        pass


class PerformanceTestCase(DatabaseTestCase):
    """Base class for performance tests with timing utilities."""

    def __init__(self):
        self.timings = {}

    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time database operations.

        Usage:
            with self.time_operation("create_agents"):
                # Your operation here
                pass
        """
        import time

        start_time = time.time()

        yield

        duration = time.time() - start_time
        if operation_name not in self.timings:
            self.timings[operation_name] = []
        self.timings[operation_name].append(duration)

        logger.info(f"{operation_name}: {duration:.3f}s")

    def get_timing_stats(self, operation_name: str) -> dict:
        """Get timing statistics for an operation."""
        if operation_name not in self.timings:
            return {}

        timings = self.timings[operation_name]
        if not timings:
            return {}

        import statistics

        return {
            "count": len(timings),
            "total": sum(timings),
            "mean": statistics.mean(timings),
            "median": statistics.median(timings),
            "min": min(timings),
            "max": max(timings),
            "stdev": statistics.stdev(timings) if len(timings) > 1 else 0,
        }


# Fixtures for specific test scenarios


@pytest.fixture
def empty_db(db_session: Session) -> Session:
    """Provide a completely empty database session."""
    # The transaction rollback ensures it's empty
    return db_session


@pytest.fixture
def populated_db(db_session: Session) -> Session:
    """Provide a database session with sample data."""
    from .factories import AgentFactory, KnowledgeGraphFactory

    # Create some agents
    _ = AgentFactory.create_batch(db_session, count=10)

    # Create a knowledge graph
    _ = KnowledgeGraphFactory.create_connected_graph(db_session, num_nodes=20, connectivity=0.3)

    return db_session


@pytest.fixture
def multi_agent_db(db_session: Session) -> Session:
    """Provide a database with a multi-agent scenario."""
    from .factories import TestDataGenerator

    TestDataGenerator.create_multi_agent_scenario(db_session, num_agents=50, num_coalitions=5)

    return db_session


# Markers for test categorization


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "db_test: mark test as requiring database")
    config.addinivalue_line("markers", "slow_db_test: mark test as slow database test")
    config.addinivalue_line("markers", "postgres_only: mark test as requiring PostgreSQL")
    config.addinivalue_line("markers", "sqlite_compatible: mark test as compatible with SQLite")


# Utility functions for common test operations


def assert_transaction_rolled_back(session: Session, model_class, count_before: int):
    """Assert that a transaction was properly rolled back.

    Args:
        session: Database session
        model_class: SQLAlchemy model class to check
        count_before: Expected count (should match current count)
    """
    current_count = session.query(model_class).count()
    assert current_count == count_before, (
        "Transaction not rolled back properly. "
        f"Expected {count_before} {model_class.__name__} records, found {current_count}"
    )


def assert_in_transaction(session: Session):
    """Assert that we're currently in a transaction."""
    assert session.in_transaction(), "Not in a transaction"


def assert_data_persisted(engine: Engine, model_class, expected_count: int):
    """Assert that data was persisted to the database.

    Creates a new session to verify data outside current transaction.
    """
    SessionLocal = sessionmaker(bind=engine)
    verify_session = SessionLocal()
    try:
        actual_count = verify_session.query(model_class).count()
        assert actual_count == expected_count, (
            f"Data not persisted. Expected {expected_count} "
            f"{model_class.__name__} records, found {actual_count}"
        )
    finally:
        verify_session.close()


if __name__ == "__main__":
    # Example usage
    print("Testing database fixtures...")

    # Test isolated database operations
    with isolated_db_test() as session:
        from database.models import Agent

        from .factories import AgentFactory

        print("\nTesting isolated transaction...")

        # Create an agent
        agent = AgentFactory.create(session, name="IsolatedTestAgent")
        print(f"✅ Created agent: {agent.name}")

        # Verify it exists in this session
        count = session.query(Agent).count()
        print(f"✅ Agent count in session: {count}")

        assert_in_transaction(session)
        print("✅ Confirmed in transaction")

    # After exiting context, transaction is rolled back
    print("✅ Transaction rolled back - data not persisted")

    # Test timing utilities
    print("\nTesting performance utilities...")
    perf_test = PerformanceTestCase()

    with perf_test.time_operation("test_operation"):
        import time

        time.sleep(0.1)  # Simulate work

    stats = perf_test.get_timing_stats("test_operation")
    print(f"✅ Operation stats: {stats}")

    print("\n✅ All fixture tests passed!")

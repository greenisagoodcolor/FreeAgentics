"""Test database configuration and setup utilities.

Provides centralized configuration for test database operations,
including connection management, schema setup, and teardown.
"""

import logging
import os
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)

# Test database configuration
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://freeagentics:freeagentics123@localhost:5432/freeagentics_test",
)

# Alternative SQLite URL for unit tests
SQLITE_TEST_URL = "sqlite:///./test.db"
SQLITE_MEMORY_URL = "sqlite:///:memory:"


class DatabaseTestConfig:
    """Configuration for test database operations."""

    def __init__(self, database_url: Optional[str] = None, use_sqlite: bool = False):
        """Initialize test database configuration.

        Args:
            database_url: Override database URL
            use_sqlite: Use SQLite instead of PostgreSQL
        """
        if use_sqlite:
            self.database_url = SQLITE_MEMORY_URL
            self.pool_class = NullPool  # SQLite doesn't support connection pooling
            self.connect_args = {"check_same_thread": False}
        else:
            self.database_url = database_url or TEST_DATABASE_URL
            self.pool_class = QueuePool
            self.connect_args = {}

        self.pool_size = 20
        self.max_overflow = 30
        self.pool_pre_ping = True
        self.echo = False

    def create_engine(self) -> Engine:
        """Create database engine with test configuration."""
        engine_kwargs = {
            "poolclass": self.pool_class,
            "echo": self.echo,
            "connect_args": self.connect_args,
        }
        
        # Only add pool parameters for non-SQLite databases
        if self.pool_class != NullPool:
            engine_kwargs.update({
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_pre_ping": self.pool_pre_ping,
            })
        
        return create_engine(self.database_url, **engine_kwargs)


def create_test_engine(use_sqlite: bool = False) -> Engine:
    """Create test database engine with proper configuration.

    Args:
        use_sqlite: Use SQLite for faster unit tests

    Returns:
        Configured SQLAlchemy engine
    """
    config = DatabaseTestConfig(use_sqlite=use_sqlite)
    return config.create_engine()


def setup_test_database(engine: Optional[Engine] = None) -> Engine:
    """Setup test database schema.

    Args:
        engine: Optional engine to use, creates new one if not provided

    Returns:
        Database engine used for setup
    """
    if engine is None:
        engine = create_test_engine()

    try:
        # Import models to ensure they're registered
        from database.base import Base

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Test database schema created successfully")

        return engine

    except Exception as e:
        logger.error(f"Failed to setup test database: {e}")
        raise


def teardown_test_database(engine: Optional[Engine] = None):
    """Clean up test database.

    Args:
        engine: Optional engine to use, creates new one if not provided
    """
    if engine is None:
        engine = create_test_engine()

    try:
        from database.base import Base

        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
        logger.info("Test database cleaned up successfully")

    except Exception as e:
        logger.error(f"Failed to teardown test database: {e}")
        raise


def reset_test_database(engine: Optional[Engine] = None) -> Engine:
    """Reset test database to clean state.

    Args:
        engine: Optional engine to use, creates new one if not provided

    Returns:
        Database engine used for reset
    """
    if engine is None:
        engine = create_test_engine()

    teardown_test_database(engine)
    return setup_test_database(engine)


def verify_test_database_connection(engine: Optional[Engine] = None) -> bool:
    """Verify test database connection is working.

    Args:
        engine: Optional engine to use, creates new one if not provided

    Returns:
        True if connection is successful, False otherwise
    """
    if engine is None:
        engine = create_test_engine()

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


class SessionTestManager:
    """Manages test database sessions with proper cleanup."""

    def __init__(self, engine: Optional[Engine] = None):
        """Initialize session manager.

        Args:
            engine: Optional engine to use, creates new one if not provided
        """
        self.engine = engine or create_test_engine()
        self.session_factory = sessionmaker(bind=self.engine)

    def create_session(self) -> Session:
        """Create a new database session."""
        return self.session_factory()

    def create_scoped_session(self):
        """Create a scoped session for thread-local operations."""
        from sqlalchemy.orm import scoped_session

        return scoped_session(self.session_factory)


# Convenience functions for common test scenarios
def get_test_session() -> Session:
    """Get a test database session."""
    engine = create_test_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def with_test_database(func):
    """Decorator to run a test with a fresh database.

    Sets up database before test and tears it down after.
    """

    def wrapper(*args, **kwargs):
        engine = create_test_engine()
        setup_test_database(engine)
        try:
            return func(*args, **kwargs)
        finally:
            teardown_test_database(engine)

    return wrapper


if __name__ == "__main__":
    # Test database connection
    print("Testing database configuration...")

    # Test PostgreSQL connection
    print("\nTesting PostgreSQL connection:")
    pg_engine = create_test_engine(use_sqlite=False)
    if verify_test_database_connection(pg_engine):
        print("✅ PostgreSQL connection successful")
    else:
        print("❌ PostgreSQL connection failed")

    # Test SQLite connection
    print("\nTesting SQLite connection:")
    sqlite_engine = create_test_engine(use_sqlite=True)
    if verify_test_database_connection(sqlite_engine):
        print("✅ SQLite connection successful")
    else:
        print("❌ SQLite connection failed")

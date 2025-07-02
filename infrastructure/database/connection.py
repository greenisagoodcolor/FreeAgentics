"""
Database connection management for FreeAgentics.

Handles database connections, sessions, and connection pooling.
"""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

# Get database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://freeagentics:dev_password@localhost:5432/freeagentics_dev",
)

# Configure connection pooling based on environment
POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "5"))
MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))

# Create engine with appropriate settings
if os.getenv("TESTING", "false").lower() == "true":
    # Use NullPool for testing to avoid connection issues
    engine = create_engine(DATABASE_URL, poolclass=NullPool)
else:
    # Use connection pooling for production/development
    engine = create_engine(
        DATABASE_URL,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT,
        pool_pre_ping=True,  # Verify connections before using
        echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.

    Yields:
        Session: SQLAlchemy database session

    Example:
        ```python
        from fastapi import Depends
        from infrastructure.database import get_db

        @app.get("/agents")
        def get_agents(db: Session = Depends(get_db)):
            return db.query(Agent).all()
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db():
    """

    Async version of database session dependency.
    For future use with async SQLAlchemy.
    """
    # TODO: Implement async database support
    raise NotImplementedError("Async database support not yet implemented")


class DatabaseManager:
    """
    Database manager class for chaos testing and connection management.

    Provides a class-based interface to the database connection functionality
    while maintaining compatibility with the existing function-based approach.
    """

    def __init__(self):
        """Initialize the database manager."""
        self.engine = engine
        self.session_factory = SessionLocal

    def get_session(self) -> Session:
        """
        Get a database session.

        Returns:
            Session: SQLAlchemy database session
        """
        return self.session_factory()

    async def get_connection(self) -> Session:
        """
        Async method to get database connection for chaos testing.

        Returns:
            Session: Database session

        Raises:
            Exception: If database connection fails
        """
        try:
            session = self.get_session()
            # Test the connection
            session.execute("SELECT 1")
            return session
        except Exception as e:
            raise Exception(f"Database connection failed: {str(e)}")

    def close_all_connections(self):
        """Close all database connections."""
        self.engine.dispose()

    def is_connected(self) -> bool:
        """
        Check if database is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

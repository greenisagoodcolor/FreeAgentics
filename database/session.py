"""Database session management for FreeAgentics.

This module provides SQLAlchemy session management and connection pooling
for the PostgreSQL database. NO IN-MEMORY STORAGE.
"""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.base import Base

# Get database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://freeagentics:freeagentics123@localhost:5432/freeagentics"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    # Connection pool settings for production
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=os.getenv("DEBUG_SQL", "false").lower() == "true",  # SQL logging
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """Get database session for dependency injection.

    Yields:
        Database session that auto-closes after use
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database by creating all tables.

    This should be called on application startup or via migration.
    """
    Base.metadata.create_all(bind=engine)


def drop_all_tables() -> None:
    """Drop all tables - DANGEROUS, only for development.

    This will DELETE ALL DATA. Use with extreme caution.
    """
    if os.getenv("DEVELOPMENT_MODE", "false").lower() != "true":
        raise RuntimeError("Cannot drop tables in production mode")

    Base.metadata.drop_all(bind=engine)

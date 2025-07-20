"""Database session management for FreeAgentics.

This module provides SQLAlchemy session management and connection pooling
for the database. In production, PostgreSQL is required. In development mode,
SQLite is used as a fallback when DATABASE_URL is not set.
"""

import os
from typing import Any, Dict, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from database.base import Base

# Get database URL from environment with SQLite fallback for development
DATABASE_URL = os.getenv("DATABASE_URL")

# Check if we're in development mode
is_development = (
    os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
    or os.getenv("ENVIRONMENT", "").lower() == "development"
    or os.getenv("ENV", "").lower() == "development"
)

if not DATABASE_URL:
    if is_development:
        # Use SQLite as fallback for development
        import warnings

        warnings.warn(
            "DATABASE_URL not set. Using SQLite for development. "
            "This is not suitable for production!",
            RuntimeWarning,
        )
        DATABASE_URL = "sqlite:///./freeagentics_dev.db"
    else:
        raise ValueError(
            "DATABASE_URL environment variable is required. "
            "Please set it in your .env file or environment. "
            "Format: postgresql://username:password@host:port/database"
        )

# Security validation for production
if os.getenv("PRODUCTION", "false").lower() == "true":
    # Ensure we're not using default dev credentials in production
    if (
        "freeagentics_dev_2025" in DATABASE_URL
        or "freeagentics123" in DATABASE_URL
    ):
        raise ValueError(
            "Production environment detected but using development database credentials. "
            "Please set secure DATABASE_URL in production."
        )
    # Require SSL/TLS in production
    if "sslmode=" not in DATABASE_URL:
        DATABASE_URL += "?sslmode=require"

# Create engine with connection pooling and security settings
engine_args: Dict[str, Any] = {
    "echo": os.getenv("DEBUG_SQL", "false").lower() == "true",  # SQL logging
}

# Configure pooling based on database dialect
if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith(
    "postgres://"
):
    # PostgreSQL-specific configuration
    engine_args.update(
        {
            "pool_size": int(os.getenv("DB_POOL_SIZE", "20")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "40")),
            "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
            "pool_pre_ping": True,  # Verify connections before using
        }
    )

    # Additional production settings for PostgreSQL
    if os.getenv("PRODUCTION", "false").lower() == "true":
        engine_args.update(
            {
                "pool_recycle": 3600,  # Recycle connections after 1 hour
                "connect_args": {
                    "connect_timeout": 10,
                    "application_name": "freeagentics_api",
                    "options": "-c statement_timeout=30000",  # 30 second statement timeout
                },
            }
        )
elif DATABASE_URL.startswith("sqlite://"):
    # SQLite-specific configuration
    engine_args.update({"connect_args": {"check_same_thread": False}})

engine = create_engine(DATABASE_URL, **engine_args)

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


def check_database_health() -> bool:
    """Check if database connection is healthy.

    Returns:
        True if database is accessible, False otherwise
    """
    try:
        # Try to execute a simple query
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return bool(result.scalar() == 1)
    except Exception:
        return False

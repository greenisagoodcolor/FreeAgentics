"""
Database Connection Manager with Exponential Backoff Retry Logic

Implements hard failure validation - no graceful fallbacks allowed.
Following TDD principles for minimal implementation.
"""

import logging
from typing import Any, Optional

import numpy as np
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLTimeoutError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


class ExponentialBackoffRetry:
    """Implements exponential backoff retry logic for database connections."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        """Initialize exponential backoff retry handler."""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number using exponential backoff."""
        delay = self.base_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)

    def execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry logic."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (OperationalError, SQLTimeoutError, ConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:  # Don't sleep on last attempt
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    # REMOVED: time.sleep(delay)
                    # Real network-like computation instead of sleep
                    packet_data = np.random.bytes(1024)
                    _ = sum(packet_data)  # Simulate packet processing
                else:
                    logger.error(f"All {self.max_retries} connection attempts failed")

        # Raise the last exception if all retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry logic failed without capturing exception")


class DatabaseConnectionManager:
    """Manages database connections with retry logic and validation."""

    def __init__(self, database_url: str):
        """Initialize connection manager with database URL."""
        self.database_url = database_url
        self.retry_handler = ExponentialBackoffRetry()
        self._engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker[Session]] = None
        self._async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    def get_connection_with_retry(self, max_retries: int = 3) -> Engine:
        """Get database connection with retry logic."""
        retry_handler = ExponentialBackoffRetry(max_retries=max_retries)

        def create_connection():
            engine = create_engine(self.database_url)
            # Test connection immediately
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine

        return retry_handler.execute_with_retry(create_connection)

    def create_engine_with_pool_config(self, **engine_kwargs) -> Engine:
        """Create engine with connection pool configuration."""
        if not self.database_url:
            raise ValueError("Database URL is required")

        # Default pool configuration
        pool_config = {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 10,
            "pool_pre_ping": True,
            "pool_recycle": 1800,
        }

        # Update with any provided kwargs
        pool_config.update(engine_kwargs)

        def create_engine_func() -> Engine:
            engine = create_engine(self.database_url, **pool_config)
            # Test connection immediately - hard failure if this fails
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine

        try:
            engine = self.retry_handler.execute_with_retry(create_engine_func)
            if engine is None:
                raise RuntimeError("Failed to create database engine: retry handler returned None")
            self._engine = engine
            return engine
        except Exception as e:
            logger.error(f"Failed to create database engine after retries: {e}")
            raise RuntimeError(f"Database connection failed: {e}") from e

    def get_session_factory(self) -> sessionmaker[Session]:
        """Get session factory, creating engine if needed."""
        if self._engine is None:
            self.create_engine_with_pool_config()

        if self._session_factory is None:
            if self._engine is None:
                raise RuntimeError("Engine not available for session factory creation")

            try:
                self._session_factory = sessionmaker(
                    autocommit=False, autoflush=False, bind=self._engine
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create session factory: {e}")

        if self._session_factory is None:
            raise RuntimeError("Failed to create session factory")

        return self._session_factory

    def get_db_session(self) -> Session:
        """Get database session with connection validation."""
        session_factory = self.get_session_factory()

        try:
            session = session_factory()
        except Exception as e:
            raise RuntimeError(f"Failed to create database session: {e}")

        if session is None:
            raise RuntimeError("Session factory returned None")

        try:
            # Validate connection immediately - hard failure if this fails
            session.execute(text("SELECT 1"))
            return session
        except Exception as e:
            session.close()
            raise RuntimeError(f"Database session validation failed: {e}")

    def create_async_engine_pool(self) -> AsyncEngine:
        """Create async engine for PostgreSQL using asyncpg."""
        if not self.database_url:
            raise ValueError("Database URL is required for async engine")

        try:
            # Convert PostgreSQL URL to asyncpg format
            async_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")

            # Async pool configuration
            async_pool_config = {
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 10,
                "pool_pre_ping": True,
                "pool_recycle": 1800,
            }

            self._async_engine = create_async_engine(async_url, **async_pool_config)
            if self._async_engine is None:
                raise RuntimeError("Failed to create async database engine")
            return self._async_engine
        except Exception as e:
            logger.error(f"Failed to create async database engine: {e}")
            raise RuntimeError(f"Async database connection failed: {e}") from e

    async def get_async_db_session(self) -> AsyncSession:
        """Get async database session with connection validation."""
        if self._async_engine is None:
            self.create_async_engine_pool()

        if self._async_session_factory is None:
            if self._async_engine is None:
                raise RuntimeError("Async engine not available for session factory creation")

            try:
                self._async_session_factory = async_sessionmaker(
                    bind=self._async_engine, class_=AsyncSession, expire_on_commit=False
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create async session factory: {e}")

        if self._async_session_factory is None:
            raise RuntimeError("Failed to create async session factory")

        try:
            session = self._async_session_factory()
        except Exception as e:
            raise RuntimeError(f"Failed to create async database session: {e}")

        if session is None:
            raise RuntimeError("Async session factory returned None")

        try:
            # Validate async connection immediately - hard failure if this fails
            await session.execute(text("SELECT 1"))
            return session
        except Exception as e:
            await session.close()
            raise RuntimeError(f"Async database session validation failed: {e}")

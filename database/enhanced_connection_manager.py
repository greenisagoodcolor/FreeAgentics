"""
Enhanced Database Connection Manager with Connection Pooling Integration.

Integrates with the connection pool manager to provide optimized database
connections with monitoring, circuit breaker patterns, and resource lifecycle management.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from agents.connection_pool_manager import (
    ConnectionPoolConfig,
    EnhancedConnectionPoolManager,
)

logger = logging.getLogger(__name__)


class EnhancedDatabaseConnectionManager:
    """Enhanced database connection manager with connection pooling."""

    def __init__(
        self, database_url: str, config: Optional[ConnectionPoolConfig] = None
    ):
        """Initialize enhanced database connection manager."""
        self.database_url = database_url
        self.config = config or ConnectionPoolConfig()

        # Initialize connection pool manager
        self.pool_manager = EnhancedConnectionPoolManager(
            self.config, database_url
        )

        # Traditional SQLAlchemy components for sync operations
        self._engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker[Session]] = None
        self._async_session_factory: Optional[
            async_sessionmaker[AsyncSession]
        ] = None

        # Connection metrics
        self.connection_metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "pool_hits": 0,
            "pool_misses": 0,
        }

        logger.info(
            f"Enhanced database connection manager initialized for {database_url}"
        )

    async def initialize(self):
        """Initialize the enhanced connection manager."""
        await self.pool_manager.initialize()
        logger.info("Enhanced database connection manager fully initialized")

    def get_sync_engine(self) -> Engine:
        """Get synchronous database engine with connection pooling."""
        if self._engine is None:
            # Enhanced pool configuration
            pool_config = {
                "pool_size": self.config.db_pool_size,
                "max_overflow": self.config.db_max_overflow,
                "pool_timeout": self.config.db_pool_timeout,
                "pool_pre_ping": self.config.db_pool_pre_ping,
                "pool_recycle": self.config.db_pool_recycle,
                "echo": False,  # Set to True for SQL logging in dev
                "echo_pool": False,  # Set to True for pool logging in dev
            }

            try:
                self._engine = create_engine(self.database_url, **pool_config)

                # Test connection immediately
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

                self.connection_metrics["total_connections"] += 1
                logger.info(
                    f"Synchronous database engine created with pool size {self.config.db_pool_size}"
                )

            except Exception as e:
                self.connection_metrics["failed_connections"] += 1
                logger.error(
                    f"Failed to create synchronous database engine: {e}"
                )
                raise

        return self._engine

    def get_async_engine(self) -> AsyncEngine:
        """Get asynchronous database engine with connection pooling."""
        if self._async_engine is None:
            # Convert to async URL format
            async_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )

            # Enhanced async pool configuration
            async_pool_config = {
                "pool_size": self.config.db_pool_size,
                "max_overflow": self.config.db_max_overflow,
                "pool_timeout": self.config.db_pool_timeout,
                "pool_pre_ping": self.config.db_pool_pre_ping,
                "pool_recycle": self.config.db_pool_recycle,
                "echo": False,  # Set to True for SQL logging in dev
                "echo_pool": False,  # Set to True for pool logging in dev
            }

            try:
                self._async_engine = create_async_engine(
                    async_url, **async_pool_config
                )
                self.connection_metrics["total_connections"] += 1
                logger.info(
                    f"Asynchronous database engine created with pool size {self.config.db_pool_size}"
                )

            except Exception as e:
                self.connection_metrics["failed_connections"] += 1
                logger.error(
                    f"Failed to create asynchronous database engine: {e}"
                )
                raise

        return self._async_engine

    def get_session_factory(self) -> sessionmaker[Session]:
        """Get session factory for synchronous operations."""
        if self._session_factory is None:
            engine = self.get_sync_engine()
            self._session_factory = sessionmaker(
                autocommit=False, autoflush=False, bind=engine
            )

        return self._session_factory

    def get_async_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get session factory for asynchronous operations."""
        if self._async_session_factory is None:
            engine = self.get_async_engine()
            self._async_session_factory = async_sessionmaker(
                autocommit=False, autoflush=False, bind=engine
            )

        return self._async_session_factory

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection using enhanced connection pooling."""
        async with self.pool_manager.get_database_connection() as connection:
            self.connection_metrics["active_connections"] += 1
            self.connection_metrics["pool_hits"] += 1

            try:
                yield connection
            finally:
                self.connection_metrics["active_connections"] -= 1

    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session with enhanced connection management."""
        session_factory = self.get_async_session_factory()

        session = session_factory()
        self.connection_metrics["active_connections"] += 1

        try:
            # Validate connection
            await session.execute(text("SELECT 1"))
            yield session

        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise

        finally:
            await session.close()
            self.connection_metrics["active_connections"] -= 1

    def get_sync_session(self) -> Session:
        """Get synchronous database session."""
        session_factory = self.get_session_factory()
        session = session_factory()

        try:
            # Validate connection
            session.execute(text("SELECT 1"))
            self.connection_metrics["active_connections"] += 1
            return session

        except Exception as e:
            session.close()
            self.connection_metrics["failed_connections"] += 1
            logger.error(f"Synchronous database session error: {e}")
            raise

    def close_sync_session(self, session: Session):
        """Close synchronous database session."""
        if session:
            session.close()
            self.connection_metrics["active_connections"] -= 1

    def get_connection_metrics(self) -> Dict[str, Any]:
        """Get comprehensive connection metrics."""
        system_metrics = self.pool_manager.get_system_metrics()

        return {
            "database_metrics": self.connection_metrics,
            "system_metrics": system_metrics,
            "pool_config": {
                "db_pool_size": self.config.db_pool_size,
                "db_max_overflow": self.config.db_max_overflow,
                "db_pool_timeout": self.config.db_pool_timeout,
                "db_pool_recycle": self.config.db_pool_recycle,
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "database_connection": False,
            "async_connection": False,
            "pool_status": "unknown",
            "connection_metrics": self.connection_metrics,
        }

        # Test synchronous connection
        try:
            with self.get_sync_session() as session:
                session.execute(text("SELECT 1"))
                health_status["database_connection"] = True
        except Exception as e:
            logger.error(f"Synchronous database health check failed: {e}")
            health_status["database_connection"] = False

        # Test asynchronous connection
        try:
            async with self.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                health_status["async_connection"] = True
        except Exception as e:
            logger.error(f"Asynchronous database health check failed: {e}")
            health_status["async_connection"] = False

        # Get pool status
        try:
            system_metrics = self.pool_manager.get_system_metrics()
            health_status["pool_status"] = (
                "healthy" if system_metrics else "degraded"
            )
        except Exception as e:
            logger.error(f"Pool status check failed: {e}")
            health_status["pool_status"] = "failed"

        return health_status

    async def close(self):
        """Close all connections and cleanup resources."""
        try:
            # Close async engine
            if self._async_engine:
                await self._async_engine.dispose()
                logger.info("Async database engine disposed")

            # Close sync engine
            if self._engine:
                self._engine.dispose()
                logger.info("Sync database engine disposed")

            # Close pool manager
            await self.pool_manager.close()
            logger.info("Enhanced database connection manager closed")

        except Exception as e:
            logger.error(
                f"Error closing enhanced database connection manager: {e}"
            )
            raise


# Global instance - initialized when needed
_global_db_manager: Optional[EnhancedDatabaseConnectionManager] = None


def get_enhanced_db_manager(
    database_url: Optional[str] = None,
) -> EnhancedDatabaseConnectionManager:
    """Get global enhanced database connection manager."""
    global _global_db_manager

    if _global_db_manager is None:
        if database_url is None:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError(
                    "DATABASE_URL environment variable is required"
                )

        _global_db_manager = EnhancedDatabaseConnectionManager(database_url)

    return _global_db_manager


async def initialize_global_db_manager(database_url: Optional[str] = None):
    """Initialize global database connection manager."""
    db_manager = get_enhanced_db_manager(database_url)
    await db_manager.initialize()
    logger.info("Global enhanced database connection manager initialized")


async def close_global_db_manager():
    """Close global database connection manager."""
    global _global_db_manager

    if _global_db_manager:
        await _global_db_manager.close()
        _global_db_manager = None
        logger.info("Global enhanced database connection manager closed")


# Dependency injection helpers for FastAPI
async def get_db_session():
    """FastAPI dependency for database session."""
    db_manager = get_enhanced_db_manager()

    async with db_manager.get_async_session() as session:
        yield session


async def get_db_connection():
    """FastAPI dependency for database connection."""
    db_manager = get_enhanced_db_manager()

    async with db_manager.get_connection() as connection:
        yield connection

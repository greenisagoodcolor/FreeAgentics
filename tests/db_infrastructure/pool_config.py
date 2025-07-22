"""Database connection pool configuration for load testing."""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from sqlalchemy import create_engine, pool, text
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


class DatabasePool:
    """Thread-safe PostgreSQL connection pool for load testing using SQLAlchemy."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "freeagentics_test",
        user: str = "freeagentics",
        password: str = "freeagentics123",
        min_connections: int = 5,
        max_connections: int = 50,
        **kwargs,
    ):
        """Initialize connection pool with configurable parameters."""
        # Build database URL
        self.database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        # Create engine with QueuePool for thread safety
        self.engine = create_engine(
            self.database_url,
            poolclass=pool.QueuePool,
            pool_size=min_connections,
            max_overflow=max_connections - min_connections,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,  # Set to True for SQL logging
            **kwargs,
        )

        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.min_connections = min_connections
        self.max_connections = max_connections

        logger.info(
            f"Database pool initialized: {min_connections}-{max_connections} connections "
            f"to {user}@{host}:{port}/{database}"
        )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def get_connection(self, dict_cursor: bool = True):
        """Get a raw connection for compatibility with existing code."""
        connection = self.engine.raw_connection()
        try:
            # Set connection parameters for optimal performance
            with connection.cursor() as cursor:
                cursor.execute("SET synchronous_commit = OFF")
                cursor.execute("SET work_mem = '256MB'")
                cursor.execute("SET maintenance_work_mem = '512MB'")

            cursor = connection.cursor()
            yield connection, cursor

            connection.commit()

        except Exception as e:
            connection.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
            connection.close()

    @contextmanager
    def transaction(self, isolation_level=None):
        """Execute operations within a transaction with optional isolation level."""
        with self.get_session() as session:
            if isolation_level:
                session.connection().execution_options(isolation_level=isolation_level)
            yield session

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a single query and return results."""
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            if result.returns_rows:
                return [dict(row._mapping) for row in result]
            return result.rowcount

    def execute_many(self, query: str, params_list: list) -> int:
        """Execute a query multiple times with different parameters."""
        with self.get_session() as session:
            stmt = text(query)
            for params in params_list:
                session.execute(stmt, params)
            return len(params_list)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        pool_impl = self.engine.pool
        return {
            "min_connections": pool_impl.size(),
            "max_connections": pool_impl.size() + pool_impl.overflow(),
            "checked_out_connections": pool_impl.checkedout(),
            "overflow": pool_impl.overflow(),
            "total": pool_impl.size() + pool_impl.overflow(),
            "available": pool_impl.size() + pool_impl.overflow() - pool_impl.checkedout(),
        }

    def close(self):
        """Close all connections in the pool."""
        if hasattr(self, "engine"):
            self.engine.dispose()
            logger.info("Database pool closed")


class PerformancePool(DatabasePool):
    """Extended pool with performance monitoring capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_stats = {}

    @contextmanager
    def monitored_session(self, operation_name: str):
        """Get a session that monitors query performance."""
        import time

        start_time = time.time()

        with self.get_session() as session:
            # Enable query timing
            session.execute(text("SET log_duration = ON"))
            session.execute(text("SET log_statement = 'all'"))

            yield session

            # Record timing
            duration = time.time() - start_time

            if operation_name not in self.query_stats:
                self.query_stats[operation_name] = {
                    "count": 0,
                    "total_time": 0,
                    "min_time": float("inf"),
                    "max_time": 0,
                }

            stats = self.query_stats[operation_name]
            stats["count"] += 1
            stats["total_time"] += duration
            stats["min_time"] = min(stats["min_time"], duration)
            stats["max_time"] = max(stats["max_time"], duration)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all monitored operations."""
        stats = {}

        for operation, data in self.query_stats.items():
            if data["count"] > 0:
                stats[operation] = {
                    **data,
                    "avg_time": data["total_time"] / data["count"],
                }

        return stats


# Global pool instances for different test scenarios
_pools = {}


def get_pool(
    pool_name: str = "default",
    min_connections: int = 5,
    max_connections: int = 50,
    **kwargs,
) -> DatabasePool:
    """Get or create a named connection pool."""
    if pool_name not in _pools:
        # Get configuration from environment or use defaults
        config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "database": os.getenv("DB_NAME", "freeagentics_test"),
            "user": os.getenv("DB_USER", "freeagentics"),
            "password": os.getenv("DB_PASSWORD", "freeagentics123"),
            "min_connections": min_connections,
            "max_connections": max_connections,
            **kwargs,
        }

        _pools[pool_name] = PerformancePool(**config)

    return _pools[pool_name]


def close_all_pools():
    """Close all active connection pools."""
    for pool_name, pool_instance in _pools.items():
        pool_instance.close()
        logger.info(f"Closed pool: {pool_name}")
    _pools.clear()

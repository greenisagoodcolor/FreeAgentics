"""Database connection pooling with resilient error handling."""

import asyncio
from typing import Any, Dict, Optional

import asyncpg

# Configuration constants
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


class ConnectionPoolError(Exception):
    """Exception raised when connection pool operations fail."""

    pass


# Global connection pool (initialized by create_pool)
_pool: Optional[asyncpg.Pool] = None


async def create_pool(database_url: str, **kwargs) -> asyncpg.Pool:
    """Create a connection pool with retry logic.

    Args:
        database_url: Database URL for connection
        **kwargs: Additional pool configuration

    Returns:
        Connection pool instance

    Raises:
        ConnectionPoolError: If pool creation fails after retries
    """
    global _pool

    if _pool is not None:
        return _pool

    for attempt in range(MAX_RETRIES):
        try:
            _pool = await asyncpg.create_pool(database_url, **kwargs)
            return _pool
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise ConnectionPoolError(
                    f"Failed to create connection pool: {e}"
                )
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    raise ConnectionPoolError("Max retries exceeded")


async def get_connection() -> asyncpg.Connection:
    """Get a connection from the pool with retry logic.

    Returns:
        Database connection

    Raises:
        ConnectionPoolError: If connection acquisition fails
    """
    if _pool is None:
        raise ConnectionPoolError("Connection pool not initialized")

    for attempt in range(MAX_RETRIES):
        try:
            return await _pool.acquire()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise ConnectionPoolError(f"Failed to acquire connection: {e}")
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    raise ConnectionPoolError("Max retries exceeded")


async def execute_query(query: str, *args) -> Any:
    """Execute a query with connection retry logic.

    Args:
        query: SQL query to execute
        *args: Query parameters

    Returns:
        Query result

    Raises:
        ConnectionPoolError: If query execution fails
    """
    connection = await get_connection()
    try:
        return await connection.fetch(query, *args)
    except Exception as e:
        raise ConnectionPoolError(f"Query execution failed: {e}")
    finally:
        await _pool.release(connection)


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

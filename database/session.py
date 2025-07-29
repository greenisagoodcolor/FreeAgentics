# database/session.py
import asyncio
from typing import Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Global state for initialization tracking
_db_initialized = False
_init_lock = asyncio.Lock()


async def init_db() -> Dict[str, Any]:
    """Initial implementation without proper guards - this will cause race conditions"""
    global _db_initialized
    # Simulate database initialization without proper locking
    await asyncio.sleep(0.01)  # Simulate init time
    _db_initialized = True
    return {"status": "initialized"}


async def init_db_safe() -> Dict[str, Any]:
    """Safe initialization with proper guards and race condition handling"""
    global _db_initialized
    
    if _db_initialized:
        return {"status": "already_initialized"}
    
    # Use lock to prevent race conditions during initialization
    async with _init_lock:
        # Double-check pattern: check again after acquiring lock
        if _db_initialized:
            return {"status": "already_initialized"}
        
        return await init_db()


def get_db_health() -> Dict[str, Any]:
    """Get database health status"""
    return {
        "status": "healthy" if _db_initialized else "initializing",
        "db_ready": _db_initialized
    }


def get_db():
    """Get database session with graceful error handling"""
    try:
        if not _db_initialized:
            raise Exception("Database not initialized")
        # Mock database session
        return {"session": "mock_session"}
    except Exception as e:
        # Log error but don't crash the application
        print(f"Database session error: {e}")
        raise
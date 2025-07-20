"""Mock Redis implementation for testing without Redis dependency."""

import asyncio
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, Mock


class MockRedis:
    """Mock Redis client for testing."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
        
    async def get(self, key: str) -> Optional[bytes]:
        """Get a value from the mock store."""
        if key in self.data:
            # Check expiry
            if key in self.expiry and asyncio.get_event_loop().time() > self.expiry[key]:
                del self.data[key]
                del self.expiry[key]
                return None
            value = self.data[key]
            if isinstance(value, str):
                return value.encode()
            return value
        return None
    
    async def set(self, key: str, value: Union[str, bytes], ex: Optional[int] = None) -> bool:
        """Set a value in the mock store."""
        if isinstance(value, bytes):
            value = value.decode()
        self.data[key] = value
        if ex:
            self.expiry[key] = asyncio.get_event_loop().time() + ex
        return True
    
    async def delete(self, *keys: str) -> int:
        """Delete keys from the mock store."""
        deleted = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                if key in self.expiry:
                    del self.expiry[key]
                deleted += 1
        return deleted
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        return sum(1 for key in keys if key in self.data)
    
    async def incr(self, key: str) -> int:
        """Increment a counter."""
        current = int(self.data.get(key, 0))
        self.data[key] = str(current + 1)
        return current + 1
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiry on a key."""
        if key in self.data:
            self.expiry[key] = asyncio.get_event_loop().time() + seconds
            return True
        return False
    
    async def ttl(self, key: str) -> int:
        """Get time to live for a key."""
        if key not in self.data:
            return -2  # Key doesn't exist
        if key not in self.expiry:
            return -1  # No expiry
        ttl = int(self.expiry[key] - asyncio.get_event_loop().time())
        return max(0, ttl)
    
    async def ping(self) -> bool:
        """Ping the mock Redis."""
        return True
    
    async def close(self):
        """Close the mock connection."""
        pass
    
    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a set."""
        if key not in self.data:
            self.data[key] = set()
        elif not isinstance(self.data[key], set):
            self.data[key] = set()
        
        added = 0
        for member in members:
            if member not in self.data[key]:
                self.data[key].add(member)
                added += 1
        return added
    
    async def smembers(self, key: str) -> set:
        """Get all members of a set."""
        if key in self.data and isinstance(self.data[key], set):
            return self.data[key]
        return set()
    
    async def sismember(self, key: str, member: str) -> bool:
        """Check if member is in set."""
        if key in self.data and isinstance(self.data[key], set):
            return member in self.data[key]
        return False


class MockRedisPool:
    """Mock Redis connection pool."""
    
    def __init__(self):
        self.redis = MockRedis()
    
    async def __aenter__(self):
        return self.redis
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @classmethod
    def from_url(cls, url: str, **kwargs):
        """Create mock pool from URL."""
        return cls()


def create_mock_redis_client():
    """Create a mock Redis client for testing."""
    return MockRedis()


def patch_redis_imports():
    """Patch Redis imports for testing."""
    import sys
    
    # Create mock module
    mock_redis_module = Mock()
    mock_redis_module.asyncio = Mock()
    mock_redis_module.asyncio.Redis = MockRedis
    mock_redis_module.asyncio.ConnectionPool = MockRedisPool
    mock_redis_module.asyncio.from_url = lambda url, **kwargs: MockRedis()
    mock_redis_module.Redis = MockRedis
    mock_redis_module.ConnectionPool = MockRedisPool
    mock_redis_module.from_url = lambda url, **kwargs: MockRedis()
    
    # Inject into sys.modules
    sys.modules['redis'] = mock_redis_module
    sys.modules['redis.asyncio'] = mock_redis_module.asyncio
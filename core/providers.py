"""Provider factory pattern for unified dev/demo/prod environments.

This module implements dependency injection for swappable providers:
- Database: PostgreSQL (prod) → SQLite (dev)
- Cache/Rate-limit: Redis (prod) → In-memory (dev)
- LLM: OpenAI/Anthropic (prod) → Mock (dev)
- Auth: Full JWT (prod) → Dev token (dev)

The goal is to maintain the same API surface across all environments
while allowing graceful fallbacks for local development.
"""

import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from database.base import Base

logger = logging.getLogger(__name__)


class ProviderMode:
    """Detect and report current provider mode."""

    @staticmethod
    def get_mode() -> str:
        """Determine current mode based on environment."""
        # Production mode requires explicit flag
        if os.getenv("PRODUCTION", "false").lower() == "true":
            return "production"

        # Check for database URL
        if os.getenv("DATABASE_URL"):
            return "development"

        # No database = demo/local mode
        return "demo"

    @staticmethod
    def log_mode():
        """Log current provider mode for visibility."""
        mode = ProviderMode.get_mode()
        emoji = {"production": "🏭", "development": "🔧", "demo": "🎯"}.get(mode, "❓")

        logger.info(f"{emoji} Provider Mode: {mode.upper()}")

        if mode == "demo":
            logger.info("  📦 Database: SQLite (in-memory)")
            logger.info("  💾 Cache: In-memory dictionary")
            logger.info("  🤖 LLM: Mock responses")
            logger.info("  🔑 Auth: Auto-generated dev token")
        elif mode == "development":
            db_type = "PostgreSQL" if "postgres" in os.getenv("DATABASE_URL", "") else "SQLite"
            logger.info(f"  📦 Database: {db_type}")
            logger.info("  💾 Cache: Redis (if available) or in-memory")
            logger.info("  🤖 LLM: Real provider (if keys set) or mock")
            logger.info("  🔑 Auth: Standard JWT")


# Database Provider
class DatabaseProvider(ABC):
    """Abstract base for database providers."""

    @abstractmethod
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session."""
        pass

    @abstractmethod
    def init_db(self) -> None:
        """Initialize database schema."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if database is available."""
        pass


class PostgreSQLProvider(DatabaseProvider):
    """PostgreSQL database provider."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Generator[Session, None, None]:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def init_db(self) -> None:
        Base.metadata.create_all(bind=self.engine)

    def is_available(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False


class SQLiteProvider(DatabaseProvider):
    """SQLite database provider for development."""

    def __init__(self, path: str = ":memory:"):
        self.path = path
        connect_args = {"check_same_thread": False}

        if path == ":memory:":
            # Use StaticPool for in-memory to share across threads
            self.engine = create_engine(
                f"sqlite:///{path}",
                connect_args=connect_args,
                poolclass=StaticPool,
            )
        else:
            self.engine = create_engine(
                f"sqlite:///{path}",
                connect_args=connect_args,
            )

        self.SessionLocal = sessionmaker(bind=self.engine)
        # Auto-initialize for SQLite
        self.init_db()

    def get_session(self) -> Generator[Session, None, None]:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def init_db(self) -> None:
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"SQLite database initialized at {self.path}")

    def is_available(self) -> bool:
        return True  # SQLite is always available


# Rate Limiter Provider
class RateLimiterProvider(ABC):
    """Abstract base for rate limiting providers."""

    @abstractmethod
    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if request is within rate limit."""
        pass

    @abstractmethod
    async def increment(self, key: str) -> int:
        """Increment counter for key."""
        pass


class RedisRateLimiter(RateLimiterProvider):
    """Redis-based rate limiter."""

    def __init__(self, redis_url: str):
        # Import here to avoid dependency if Redis not used
        import redis.asyncio as redis

        self.redis = redis.from_url(redis_url)

    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, window)
        return count <= limit

    async def increment(self, key: str) -> int:
        return await self.redis.incr(key)


class InMemoryRateLimiter(RateLimiterProvider):
    """In-memory rate limiter for development."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._cleanup_interval = 60  # seconds

    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        import time

        now = time.time()

        if key not in self._store:
            self._store[key] = {"count": 1, "reset_at": now + window}
            return True

        entry = self._store[key]
        if now > entry["reset_at"]:
            entry["count"] = 1
            entry["reset_at"] = now + window
            return True

        entry["count"] += 1
        return entry["count"] <= limit

    async def increment(self, key: str) -> int:
        if key not in self._store:
            self._store[key] = {"count": 1, "reset_at": 0}
        else:
            self._store[key]["count"] += 1
        return self._store[key]["count"]


# LLM Provider
class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion from prompt."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for development."""

    async def complete(self, prompt: str, **kwargs) -> str:
        """Return deterministic mock responses."""
        if "agent" in prompt.lower():
            return "I am a mock Active Inference agent ready to assist."
        elif "explain" in prompt.lower():
            return "This is a mock explanation of the requested concept."
        else:
            return f"Mock response for: {prompt[:50]}..."

    async def embed(self, text: str) -> list[float]:
        """Return deterministic mock embedding."""
        # Simple hash-based embedding for consistency
        import hashlib

        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        # Generate 384-dim embedding (standard for many models)
        return [(hash_val >> i & 1) * 0.1 for i in range(384)]


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: str):
        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 500),
        )
        return response.choices[0].message.content

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        return response.data[0].embedding


# Factory Functions
_db_provider: Optional[DatabaseProvider] = None
_rate_limiter: Optional[RateLimiterProvider] = None
_llm_provider: Optional[LLMProvider] = None


def reset_providers():
    """Reset all providers - useful for testing."""
    global _db_provider, _rate_limiter, _llm_provider
    _db_provider = None
    _rate_limiter = None
    _llm_provider = None


def get_database() -> DatabaseProvider:
    """Get database provider based on environment."""
    global _db_provider

    if _db_provider is None:
        database_url = os.getenv("DATABASE_URL")

        if database_url:
            if database_url.startswith(("postgresql://", "postgres://")):
                _db_provider = PostgreSQLProvider(database_url)
                logger.info("Using PostgreSQL database")
            elif database_url.startswith("sqlite:///"):
                # Extract file path from SQLite URL
                path = database_url.replace("sqlite:///", "")
                _db_provider = SQLiteProvider(path)
                logger.info(f"Using SQLite database: {path}")
            else:
                # Assume direct path
                _db_provider = SQLiteProvider(database_url)
                logger.info(f"Using SQLite database: {database_url}")
        else:
            # Demo mode - in-memory SQLite
            _db_provider = SQLiteProvider(":memory:")
            logger.info("Using in-memory SQLite (demo mode)")

    return _db_provider


def get_rate_limiter() -> RateLimiterProvider:
    """Get rate limiter based on environment."""
    global _rate_limiter

    if _rate_limiter is None:
        redis_url = os.getenv("REDIS_URL")

        if redis_url:
            try:
                _rate_limiter = RedisRateLimiter(redis_url)
                logger.info("Using Redis rate limiter")
            except ImportError:
                logger.warning("Redis not installed, falling back to in-memory")
                _rate_limiter = InMemoryRateLimiter()
        else:
            _rate_limiter = InMemoryRateLimiter()
            logger.info("Using in-memory rate limiter")

    return _rate_limiter


def get_llm() -> LLMProvider:
    """Get LLM provider based on environment."""
    global _llm_provider

    if _llm_provider is None:
        openai_key = os.getenv("OPENAI_API_KEY")

        if openai_key:
            try:
                _llm_provider = OpenAIProvider(openai_key)
                logger.info("Using OpenAI LLM provider")
            except ImportError:
                logger.warning("OpenAI library not installed, using mock")
                _llm_provider = MockLLMProvider()
        else:
            _llm_provider = MockLLMProvider()
            logger.info("Using mock LLM provider")

    return _llm_provider


# Dependency injection helpers for FastAPI
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session."""
    provider = get_database()
    yield from provider.get_session()


async def get_rate_limit_checker() -> RateLimiterProvider:
    """FastAPI dependency for rate limiter."""
    return get_rate_limiter()


async def get_llm_client() -> LLMProvider:
    """FastAPI dependency for LLM client."""
    return get_llm()


# Initialize providers on import
def init_providers():
    """Initialize all providers and log status."""
    ProviderMode.log_mode()

    # Initialize database
    db = get_database()
    if db.is_available():
        logger.info("✅ Database provider ready")
    else:
        logger.error("❌ Database provider failed to initialize")

    # Check rate limiter
    get_rate_limiter()
    logger.info("✅ Rate limiter ready")

    # Check LLM
    get_llm()
    logger.info("✅ LLM provider ready")

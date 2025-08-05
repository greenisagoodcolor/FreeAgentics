"""Response caching implementations for performance optimization."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .models import ResponseData

logger = logging.getLogger(__name__)


class ResponseCache(ABC):
    """Abstract base class for response caches."""

    @abstractmethod
    async def get(self, key: str) -> Optional[ResponseData]:
        """Get cached response by key.

        Args:
            key: Cache key

        Returns:
            Cached response data or None if not found
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: ResponseData,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set cached response.

        Args:
            key: Cache key
            value: Response data to cache
            ttl_seconds: Time to live in seconds (None for default)
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cached response.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached responses."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class InMemoryResponseCache(ResponseCache):
    """In-memory response cache with TTL support and LRU eviction.

    This implementation follows the performance patterns established in the
    knowledge graph query cache, providing fast lookups with memory bounds.
    """

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 300):
        """Initialize the in-memory cache.

        Args:
            max_size: Maximum number of entries to cache
            default_ttl_seconds: Default TTL for cached entries
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds

        # Cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: Dict[str, float] = {}  # For LRU tracking

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "size": 0,
        }

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.debug(f"InMemoryResponseCache initialized with max_size={max_size}")

    async def get(self, key: str) -> Optional[ResponseData]:
        """Get cached response by key."""
        async with self._lock:
            now = time.time()

            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if entry["expires_at"] <= now:
                # Expired entry - remove it
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
                self._stats["size"] -= 1
                self._stats["misses"] += 1
                return None

            # Update access time for LRU
            self._access_order[key] = now
            self._stats["hits"] += 1

            try:
                # Deserialize response data
                response_data = self._deserialize_response(entry["data"])
                return response_data
            except Exception as e:
                logger.warning(f"Failed to deserialize cached response: {e}")
                # Remove corrupted entry
                await self._remove_entry(key)
                self._stats["misses"] += 1
                return None

    async def set(
        self,
        key: str,
        value: ResponseData,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set cached response."""
        async with self._lock:
            now = time.time()
            ttl = ttl_seconds or self.default_ttl_seconds

            try:
                # Serialize response data
                serialized_data = self._serialize_response(value)

                # Check if we need to evict entries
                if key not in self._cache and len(self._cache) >= self.max_size:
                    await self._evict_lru()

                # Store entry
                self._cache[key] = {
                    "data": serialized_data,
                    "created_at": now,
                    "expires_at": now + ttl,
                }

                self._access_order[key] = now

                if key not in self._cache or self._stats["size"] < len(self._cache):
                    self._stats["size"] = len(self._cache)

                self._stats["sets"] += 1

            except Exception as e:
                logger.error(f"Failed to cache response: {e}")
                # Don't raise - caching failures should not break the application

    async def delete(self, key: str) -> bool:
        """Delete cached response."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                self._stats["deletes"] += 1
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached responses."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats["size"] = 0
            logger.debug("Response cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._stats.copy()

        # Add computed metrics
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
            stats["miss_rate"] = stats["misses"] / total_requests
        else:
            stats["hit_rate"] = 0.0
            stats["miss_rate"] = 0.0

        stats["max_size"] = self.max_size
        stats["default_ttl_seconds"] = self.default_ttl_seconds

        return stats

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return

        # Find LRU key
        lru_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])

        # Remove it
        await self._remove_entry(lru_key)
        self._stats["evictions"] += 1

        logger.debug(f"Evicted LRU entry: {lru_key}")

    async def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and access tracking."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            del self._access_order[key]
        self._stats["size"] = len(self._cache)

    def _serialize_response(self, response: ResponseData) -> Dict[str, Any]:
        """Serialize response data for caching."""
        try:
            # Convert to dictionary
            data = response.to_dict()

            # Handle any non-serializable fields
            # (The ResponseData model should already be JSON-serializable)
            return data

        except Exception as e:
            logger.error(f"Response serialization failed: {e}")
            raise

    def _deserialize_response(self, data: Dict[str, Any]) -> ResponseData:
        """Deserialize response data from cache."""
        try:
            # Since we stored the full dict representation, we need to reconstruct
            # the ResponseData object with all its nested objects
            from .models import (
                ActionExplanation,
                BeliefSummary,
                ConfidenceRating,
                ResponseMetadata,
                ResponseType,
            )
            from datetime import datetime

            # Reconstruct nested objects
            action_explanation = ActionExplanation(**data["action_explanation"])
            belief_summary = BeliefSummary(**data["belief_summary"])

            # Handle confidence rating with enum conversion
            conf_data = data["confidence_rating"]
            confidence_rating = ConfidenceRating(
                overall=conf_data["overall"],
                level=conf_data["level"],  # Will be converted by enum
                action_confidence=conf_data["action_confidence"],
                belief_confidence=conf_data["belief_confidence"],
                model_confidence=conf_data.get("model_confidence", 1.0),
                factors=conf_data.get("factors", {}),
            )

            # Handle metadata with datetime conversion
            meta_data = data["metadata"]
            metadata = ResponseMetadata(
                response_id=meta_data["response_id"],
                generation_time_ms=meta_data["generation_time_ms"],
                cached=meta_data.get("cached", False),
                cache_key=meta_data.get("cache_key"),
                nlg_enhanced=meta_data.get("nlg_enhanced", False),
                streaming=meta_data.get("streaming", False),
                formatting_time_ms=meta_data.get("formatting_time_ms", 0.0),
                nlg_time_ms=meta_data.get("nlg_time_ms", 0.0),
                cache_lookup_time_ms=meta_data.get("cache_lookup_time_ms", 0.0),
                template_used=meta_data.get("template_used"),
                fallback_used=meta_data.get("fallback_used", False),
                errors=meta_data.get("errors", []),
                timestamp=datetime.fromisoformat(meta_data["timestamp"]),
                trace_id=meta_data.get("trace_id"),
                conversation_id=meta_data.get("conversation_id"),
            )

            # Reconstruct main response
            response = ResponseData(
                message=data["message"],
                action_explanation=action_explanation,
                belief_summary=belief_summary,
                confidence_rating=confidence_rating,
                knowledge_graph_updates=data.get("knowledge_graph_updates"),
                related_concepts=data.get("related_concepts", []),
                suggested_actions=data.get("suggested_actions", []),
                metadata=metadata,
                response_type=ResponseType(data["response_type"]),
                format_version=data.get("format_version", "1.0"),
            )

            return response

        except Exception as e:
            logger.error(f"Response deserialization failed: {e}")
            raise


class RedisResponseCache(ResponseCache):
    """Redis-based response cache for production deployments.

    This implementation would use Redis for distributed caching across
    multiple application instances. Currently a placeholder for future
    implementation when Redis is available in the infrastructure.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "freeagentics:response:",
        default_ttl_seconds: int = 300,
    ):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for cache keys
            default_ttl_seconds: Default TTL for cached entries
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl_seconds = default_ttl_seconds

        # Stats tracking
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

        # Redis client would be initialized here
        self._redis = None

        logger.warning(
            "RedisResponseCache initialized but Redis client not implemented. "
            "This is a placeholder for future Redis integration."
        )

    async def get(self, key: str) -> Optional[ResponseData]:
        """Get cached response by key."""
        # TODO: Implement Redis get operation
        logger.debug(f"Redis cache get not implemented: {key}")
        self._stats["misses"] += 1
        return None

    async def set(
        self,
        key: str,
        value: ResponseData,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set cached response."""
        # TODO: Implement Redis set operation
        logger.debug(f"Redis cache set not implemented: {key}")
        self._stats["sets"] += 1

    async def delete(self, key: str) -> bool:
        """Delete cached response."""
        # TODO: Implement Redis delete operation
        logger.debug(f"Redis cache delete not implemented: {key}")
        self._stats["deletes"] += 1
        return False

    async def clear(self) -> None:
        """Clear all cached responses."""
        # TODO: Implement Redis clear operation
        logger.debug("Redis cache clear not implemented")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._stats.copy()
        stats["implementation"] = "redis_placeholder"
        stats["redis_url"] = self.redis_url
        return stats

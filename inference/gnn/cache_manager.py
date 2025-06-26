import concurrent.futures
import gzip
import hashlib
import json
import pickle
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .monitoring import get_logger

"""
Caching Manager for GNN Processing
This module provides caching mechanisms for intermediate results
in GNN processing pipelines to improve performance.
"""
logger = get_logger().logger


@dataclass
class CacheEntry:
    """Represents a single cache entry"""

    key: str
    data: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheManager:
    """
    Main cache manager for GNN processing.
    Features:
    - Multi-level caching (memory, disk, distributed)
    - Automatic eviction policies
    - Cache warming and preloading
    - Compression for disk storage
    """

    def __init__(
        self,
        cache_dir: str = ".cache/gnn",
        max_memory_mb: int = 1024,
        max_disk_gb: int = 10,
        ttl_hours: int = 24,
        compression: bool = True,
    ) -> None:
        """
        Initialize cache manager.
        Args:
            cache_dir: Directory for disk cache
            max_memory_mb: Maximum memory cache size in MB
            max_disk_gb: Maximum disk cache size in GB
            ttl_hours: Time to live for cache entries in hours
            compression: Whether to compress disk cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_gb * 1024 * 1024 * 1024
        self.ttl = timedelta(hours=ttl_hours)
        self.compression = compression
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_size = 0
        self._lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_size": 0,
            "disk_size": 0,
        }
        self._init_disk_cache()

    def _init_disk_cache(self):
        """Initialize disk cache and load metadata"""
        self.metadata_file = self.cache_dir / "metadata.json"
        self.disk_metadata = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    self.disk_metadata = json.load(f)
                self._clean_expired_disk_entries()
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {e}")

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(
        self,
        key: str,
        loader_fn: Optional[Callable] = None,
        loader_args: Optional[tuple] = None,
        loader_kwargs: Optional[dict] = None,
    ) -> Optional[Any]:
        """
        Get item from cache or load it.
        Args:
            key: Cache key
            loader_fn: Function to load data if not cached
            loader_args: Arguments for loader function
            loader_kwargs: Keyword arguments for loader function
        Returns:
            Cached data or None
        """
        data = self._get_from_memory(key)
        if data is not None:
            self.stats["hits"] += 1
            return data
        data = self._get_from_disk(key)
        if data is not None:
            self.stats["hits"] += 1
            self._add_to_memory(key, data)
            return data
        self.stats["misses"] += 1
        if loader_fn is not None:
            loader_args = loader_args or ()
            loader_kwargs = loader_kwargs or {}
            try:
                data = loader_fn(*loader_args, **loader_kwargs)
                self.set(key, data)
                return data
            except Exception as e:
                logger.error(f"Failed to load data for key {key}: {e}")
        return None

    def set(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set item in cache.
        Args:
            key: Cache key
            data: Data to cache
            metadata: Optional metadata
        """
        size_bytes = self._estimate_size(data)
        now = datetime.now()
        entry = CacheEntry(
            key=key,
            data=data,
            size_bytes=size_bytes,
            created_at=now,
            last_accessed=now,
            metadata=metadata or {},
        )
        self._add_to_memory(key, data, entry)
        if (
            size_bytes > self.max_memory_bytes * 0.1
            or self._memory_size > self.max_memory_bytes * 0.8
        ):
            self._add_to_disk(key, data, entry)

    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get item from memory cache"""
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if datetime.now() - entry.created_at > self.ttl:
                    del self._memory_cache[key]
                    self._memory_size -= entry.size_bytes
                    return None
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._memory_cache.move_to_end(key)
                return entry.data
        return None

    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get item from disk cache"""
        if key not in self.disk_metadata:
            return None
        metadata = self.disk_metadata[key]
        created_at = datetime.fromisoformat(metadata["created_at"])
        if datetime.now() - created_at > self.ttl:
            self._remove_from_disk(key)
            return None
        cache_file = self.cache_dir / f"{key}.pkl"
        if not cache_file.exists():
            del self.disk_metadata[key]
            self._save_disk_metadata()
            return None
        try:
            with open(cache_file, "rb") as f:
                if self.compression:
                    data = pickle.loads(gzip.decompress(f.read()))
                else:
                    data = pickle.load(f)
            self.disk_metadata[key]["last_accessed"] = datetime.now().isoformat()
            self.disk_metadata[key]["access_count"] += 1
            self._save_disk_metadata()
            return data
        except Exception as e:
            logger.error(f"Failed to load from disk cache: {e}")
            self._remove_from_disk(key)
            return None

    def _add_to_memory(self, key: str, data: Any, entry: Optional[CacheEntry] = None):
        """Add item to memory cache"""
        with self._lock:
            if entry is None:
                size_bytes = self._estimate_size(data)
                now = datetime.now()
                entry = CacheEntry(
                    key=key,
                    data=data,
                    size_bytes=size_bytes,
                    created_at=now,
                    last_accessed=now,
                )
            while (
                self._memory_size + entry.size_bytes > self.max_memory_bytes and self._memory_cache
            ):
                self._evict_from_memory()
            if key in self._memory_cache:
                old_entry = self._memory_cache[key]
                self._memory_size -= old_entry.size_bytes
            self._memory_cache[key] = entry
            self._memory_size += entry.size_bytes
            self.stats["memory_size"] = self._memory_size

    def _add_to_disk(self, key: str, data: Any, entry: CacheEntry):
        """Add item to disk cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            current_disk_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            if current_disk_size + entry.size_bytes > self.max_disk_bytes:
                self._evict_from_disk()
            with open(cache_file, "wb") as f:
                if self.compression:
                    f.write(gzip.compress(pickle.dumps(data)))
                else:
                    pickle.dump(data, f)
            self.disk_metadata[key] = {
                "size_bytes": entry.size_bytes,
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat(),
                "access_count": entry.access_count,
                "metadata": entry.metadata,
            }
            self._save_disk_metadata()
            self.stats["disk_size"] = current_disk_size + entry.size_bytes
        except Exception as e:
            logger.error(f"Failed to save to disk cache: {e}")

    def _evict_from_memory(self):
        """Evict least recently used item from memory"""
        if not self._memory_cache:
            return
        key, entry = next(iter(self._memory_cache.items()))
        del self._memory_cache[key]
        self._memory_size -= entry.size_bytes
        self.stats["evictions"] += 1
        logger.debug(f"Evicted {key} from memory cache")

    def _evict_from_disk(self):
        """Evict least recently used items from disk"""
        sorted_entries = sorted(self.disk_metadata.items(), key=lambda x: x[1]["last_accessed"])
        num_to_evict = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:num_to_evict]:
            self._remove_from_disk(key)
            self.stats["evictions"] += 1

    def _remove_from_disk(self, key: str):
        """Remove item from disk cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
        if key in self.disk_metadata:
            del self.disk_metadata[key]
            self._save_disk_metadata()

    def _clean_expired_disk_entries(self):
        """Remove expired entries from disk"""
        now = datetime.now()
        keys_to_remove = []
        for key, metadata in self.disk_metadata.items():
            created_at = datetime.fromisoformat(metadata["created_at"])
            if now - created_at > self.ttl:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self._remove_from_disk(key)

    def _save_disk_metadata(self):
        """Save disk cache metadata"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.disk_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes"""
        if isinstance(data, torch.Tensor):
            return data.element_size() * data.nelement()
        elif isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_size(item) for item in data)
        elif isinstance(data, dict):
            return sum((self._estimate_size(k) + self._estimate_size(v) for k, v in data.items()))
        else:
            try:
                return len(pickle.dumps(data))
            except Exception:
                return 1024

    def clear(self, memory_only: bool = False) -> None:
        """
        Clear cache.
        Args:
            memory_only: If True, only clear memory cache
        """
        with self._lock:
            self._memory_cache.clear()
            self._memory_size = 0
            self.stats["memory_size"] = 0
            if not memory_only:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.disk_metadata = {}
                self._save_disk_metadata()
                self.stats["disk_size"] = 0
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "memory_size_mb": self.stats["memory_size"] / 1024 / 1024,
            "disk_size_mb": self.stats["disk_size"] / 1024 / 1024,
            "memory_entries": len(self._memory_cache),
            "disk_entries": len(self.disk_metadata),
        }

    def warm_cache(self, keys: List[str], loader_fn: Callable, parallel: bool = True):
        """
        Warm cache with specified keys.
        Args:
            keys: List of cache keys to warm
            loader_fn: Function to load data for each key
            parallel: Whether to load in parallel
        """
        logger.info(f"Warming cache with {len(keys)} keys")
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.get, key, loader_fn, (key,)): key for key in keys}
                for future in concurrent.futures.as_completed(futures):
                    key = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Failed to warm cache for key {key}: {e}")
        else:
            for key in keys:
                try:
                    self.get(key, loader_fn, (key,))
                except Exception as e:
                    logger.error(f"Failed to warm cache for key {key}: {e}")
        logger.info("Cache warming completed")


class GraphFeatureCache(CacheManager):
    """Specialized cache for graph features"""

    def cache_graph_features(
        self,
        graph_id: str,
        feature_type: str,
        features: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Cache graph features with specific handling"""
        key = f"features_{graph_id}_{feature_type}"
        if metadata is None:
            metadata = {}
        metadata.update(
            {
                "graph_id": graph_id,
                "feature_type": feature_type,
                "shape": list(features.shape),
                "dtype": str(features.dtype),
            }
        )
        self.set(key, features, metadata)

    def get_graph_features(self, graph_id: str, feature_type: str) -> Optional[torch.Tensor]:
        """Get cached graph features"""
        key = f"features_{graph_id}_{feature_type}"
        return self.get(key)


class ModelCache(CacheManager):
    """Specialized cache for trained models"""

    def cache_model(
        self,
        model_id: str,
        model_state: Dict[str, Any],
        performance_metrics: Dict[str, float],
        config: Dict[str, Any],
    ):
        """Cache trained model with metadata"""
        key = f"model_{model_id}"
        data = {
            "state_dict": model_state,
            "metrics": performance_metrics,
            "config": config,
            "cached_at": datetime.now().isoformat(),
        }
        self.set(key, data, {"model_id": model_id})

    def get_model(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get cached model"""
        key = f"model_{model_id}"
        return self.get(key)

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """List all cached models with their metadata"""
        models = []
        with self._lock:
            for key, entry in self._memory_cache.items():
                if key.startswith("model_"):
                    models.append(
                        {
                            "model_id": entry.metadata.get("model_id", key[6:]),
                            "cached_at": entry.created_at.isoformat(),
                            "size_mb": entry.size_bytes / 1024 / 1024,
                            "location": "memory",
                        }
                    )
        for key, metadata in self.disk_metadata.items():
            if key.startswith("model_"):
                models.append(
                    {
                        "model_id": metadata["metadata"].get("model_id", key[6:]),
                        "cached_at": metadata["created_at"],
                        "size_mb": metadata["size_bytes"] / 1024 / 1024,
                        "location": "disk",
                    }
                )
        return models


def cached(cache_manager: CacheManager, key_prefix: str = ""):
    """Decorator to cache function results"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = cache_manager._generate_key(key_prefix, func.__name__, *args, **kwargs)
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result)
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    cache = CacheManager(cache_dir=".cache/gnn_example", max_memory_mb=512, max_disk_gb=5)

    @cached(cache, "computation")
    def expensive_computation(x: int, y: int) -> int:
        print(f"Computing {x} + {y}...")
        time.sleep(1)
        return x + y

    result1 = expensive_computation(5, 3)
    print(f"Result 1: {result1}")
    result2 = expensive_computation(5, 3)
    print(f"Result 2: {result2}")
    print(f"\nCache stats: {cache.get_stats()}")
    feature_cache = GraphFeatureCache()
    features = torch.randn(100, 64)
    feature_cache.cache_graph_features("graph_1", "node_embeddings", features)
    cached_features = feature_cache.get_graph_features("graph_1", "node_embeddings")
    print(
        f"\nCached features shape: {(cached_features.shape if cached_features is not None else 'None')}"
    )
    cache.clear()
    feature_cache.clear()

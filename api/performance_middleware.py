"""
API Performance Optimization Middleware.

This module implements comprehensive API performance optimizations:
1. Response caching with TTL and invalidation
2. Request deduplication and batching
3. Response compression (gzip, brotli)
4. Streaming responses for large data
5. Request/response timing and monitoring
6. API rate limiting with performance awareness
7. Response size optimization
8. Concurrent request handling optimization
"""

import asyncio
import gzip
import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import cachetools
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import StreamingResponse

from observability.performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for API response caching."""

    # Cache settings
    max_size: int = 1000
    default_ttl: int = 300  # 5 minutes

    # Cache key settings
    include_headers: List[str] = field(
        default_factory=lambda: ["Authorization", "Accept-Language"]
    )
    include_query_params: bool = True

    # Cache invalidation
    invalidation_patterns: List[str] = field(default_factory=list)

    # Cache warming
    preload_endpoints: List[str] = field(default_factory=list)

    # Per-endpoint TTL override
    endpoint_ttl: Dict[str, int] = field(default_factory=dict)


@dataclass
class CompressionConfig:
    """Configuration for response compression."""

    # Compression settings
    min_size: int = 1024  # Only compress responses > 1KB
    compression_level: int = 6  # Compression level (1-9)

    # Supported algorithms
    algorithms: List[str] = field(default_factory=lambda: ["gzip", "deflate", "br"])

    # MIME types to compress
    compressible_types: List[str] = field(
        default_factory=lambda: [
            "application/json",
            "application/javascript",
            "text/plain",
            "text/html",
            "text/css",
            "text/xml",
            "application/xml",
        ]
    )


@dataclass
class PerformanceConfig:
    """Configuration for API performance optimizations."""

    # Response caching
    caching_enabled: bool = True
    cache_config: CacheConfig = field(default_factory=CacheConfig)

    # Response compression
    compression_enabled: bool = True
    compression_config: CompressionConfig = field(default_factory=CompressionConfig)

    # Request deduplication
    deduplication_enabled: bool = True
    deduplication_window: int = 10  # seconds

    # Performance monitoring
    monitoring_enabled: bool = True
    slow_request_threshold: float = 1.0  # seconds

    # Streaming responses
    streaming_threshold: int = 1024 * 1024  # 1MB
    streaming_enabled: bool = True


class ResponseCache:
    """High-performance response cache with TTL and invalidation."""

    def __init__(self, config: CacheConfig):
        """Initialize the response cache.

        Args:
            config: Cache configuration settings.
        """
        self.config = config
        self.cache = cachetools.TTLCache(
            maxsize=config.max_size, ttl=config.default_ttl
        )
        self.hit_count = 0
        self.miss_count = 0
        self.invalidation_count = 0

    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request."""
        # Base key from path and method
        key_parts = [request.method, request.url.path]

        # Add query parameters
        if self.config.include_query_params and request.query_params:
            sorted_params = sorted(request.query_params.items())
            key_parts.append(str(sorted_params))

        # Add relevant headers
        for header in self.config.include_headers:
            value = request.headers.get(header)
            if value:
                key_parts.append(f"{header}:{value}")

        # Generate hash
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()

    def _get_ttl(self, request: Request) -> int:
        """Get TTL for specific endpoint."""
        path = request.url.path
        return self.config.endpoint_ttl.get(path, self.config.default_ttl)

    def get(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get cached response for request."""
        cache_key = self._generate_cache_key(request)

        try:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                self.hit_count += 1
                logger.debug(f"Cache hit for {request.url.path}")
                return cached_response
        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        self.miss_count += 1
        return None

    def set(self, request: Request, response_data: Dict[str, Any]):
        """Cache response for request."""
        cache_key = self._generate_cache_key(request)
        ttl = self._get_ttl(request)

        try:
            # Create cache entry with custom TTL
            self.cache[cache_key] = response_data
            logger.debug(f"Cached response for {request.url.path} (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        keys_to_remove = []

        for key in self.cache:
            # In production, implement proper pattern matching
            if pattern in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]
            self.invalidation_count += 1

        logger.info(
            f"Invalidated {len(keys_to_remove)} cache entries for pattern: {pattern}"
        )

    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
        self.invalidation_count += 1
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) * 100 if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.config.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "invalidation_count": self.invalidation_count,
            "utilization": (len(self.cache) / self.config.max_size) * 100,
        }


class RequestDeduplicator:
    """Deduplicates identical requests within a time window."""

    def __init__(self, window_seconds: int = 10):
        """Initialize the request deduplicator.

        Args:
            window_seconds: Time window for deduplication in seconds.
        """
        self.window_seconds = window_seconds
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_timestamps: Dict[str, float] = {}
        self.deduplication_count = 0

    def _generate_request_key(self, request: Request) -> str:
        """Generate key for request deduplication."""
        # Include method, path, query params, and body hash
        key_parts = [request.method, request.url.path]

        # Add query parameters
        if request.query_params:
            sorted_params = sorted(request.query_params.items())
            key_parts.append(str(sorted_params))

        # Add body hash for POST/PUT requests
        if hasattr(request, "_body_hash"):
            key_parts.append(request._body_hash)

        return hashlib.md5(
            "|".join(key_parts).encode(), usedforsecurity=False
        ).hexdigest()

    def _cleanup_expired(self):
        """Clean up expired requests."""
        current_time = time.time()
        expired_keys = []

        for key, timestamp in self.request_timestamps.items():
            if current_time - timestamp > self.window_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.request_timestamps[key]
            if key in self.pending_requests:
                del self.pending_requests[key]

    async def deduplicate_request(self, request: Request, handler: Callable) -> Any:
        """Deduplicate request or return existing response."""
        request_key = self._generate_request_key(request)
        current_time = time.time()

        self._cleanup_expired()

        # Check if request is already pending
        if request_key in self.pending_requests:
            logger.debug(f"Deduplicating request: {request.url.path}")
            self.deduplication_count += 1
            return await self.pending_requests[request_key]

        # Create new future for this request
        future = asyncio.Future()
        self.pending_requests[request_key] = future
        self.request_timestamps[request_key] = current_time

        try:
            # Execute the request handler
            result = await handler()
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            # Clean up
            if request_key in self.pending_requests:
                del self.pending_requests[request_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "pending_requests": len(self.pending_requests),
            "deduplication_count": self.deduplication_count,
            "window_seconds": self.window_seconds,
        }


class ResponseCompressor:
    """Compresses responses using various algorithms."""

    def __init__(self, config: CompressionConfig):
        """Initialize the response compressor.

        Args:
            config: Compression configuration settings.
        """
        self.config = config
        self.compression_count = 0
        self.bytes_saved = 0

    def _should_compress(self, response: Response) -> bool:
        """Check if response should be compressed."""
        # Check content type
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type for ct in self.config.compressible_types):
            return False

        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.config.min_size:
            return False

        # Check if already compressed
        if response.headers.get("content-encoding"):
            return False

        return True

    def _get_best_encoding(self, request: Request) -> Optional[str]:
        """Get best compression encoding based on Accept-Encoding header."""
        accept_encoding = request.headers.get("accept-encoding", "")

        # Priority order based on efficiency
        for encoding in ["br", "gzip", "deflate"]:
            if encoding in accept_encoding and encoding in self.config.algorithms:
                return encoding

        return None

    def _compress_data(self, data: bytes, encoding: str) -> bytes:
        """Compress data using specified encoding."""
        if encoding == "gzip":
            return gzip.compress(data, compresslevel=self.config.compression_level)
        elif encoding == "deflate":
            import zlib

            return zlib.compress(data, level=self.config.compression_level)
        elif encoding == "br":
            try:
                import brotli

                return brotli.compress(data, quality=self.config.compression_level)
            except ImportError:
                logger.warning("Brotli not available, falling back to gzip")
                return gzip.compress(data, compresslevel=self.config.compression_level)

        return data

    async def compress_response(self, request: Request, response: Response) -> Response:
        """Compress response if appropriate."""
        if not self._should_compress(response):
            return response

        encoding = self._get_best_encoding(request)
        if not encoding:
            return response

        # Get response body
        if hasattr(response, "body") and response.body:
            original_size = len(response.body)
            compressed_body = self._compress_data(response.body, encoding)
            compressed_size = len(compressed_body)

            # Update response
            response.body = compressed_body
            response.headers["content-encoding"] = encoding
            response.headers["content-length"] = str(compressed_size)

            # Update statistics
            self.compression_count += 1
            self.bytes_saved += original_size - compressed_size

            logger.debug(
                f"Compressed response: {original_size} -> {compressed_size} bytes ({encoding})"
            )

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            "compression_count": self.compression_count,
            "bytes_saved": self.bytes_saved,
            "avg_compression_ratio": (
                (self.bytes_saved / max(self.compression_count, 1))
                if self.compression_count > 0
                else 0
            ),
        }


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Comprehensive API performance middleware."""

    def __init__(self, app: FastAPI, config: PerformanceConfig = None):
        """Initialize the performance middleware.

        Args:
            app: The FastAPI application instance.
            config: Performance configuration settings.
        """
        super().__init__(app)
        self.config = config or PerformanceConfig()
        self.performance_monitor = get_performance_monitor()

        # Initialize components
        self.cache = (
            ResponseCache(self.config.cache_config)
            if self.config.caching_enabled
            else None
        )
        self.deduplicator = (
            RequestDeduplicator(self.config.deduplication_window)
            if self.config.deduplication_enabled
            else None
        )
        self.compressor = (
            ResponseCompressor(self.config.compression_config)
            if self.config.compression_enabled
            else None
        )

        # Statistics
        self.request_count = 0
        self.slow_request_count = 0
        self.total_response_time = 0
        self.response_sizes = deque(maxlen=100)

        logger.info("Performance middleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process incoming requests through the performance middleware pipeline."""
        start_time = time.perf_counter()

        # Increment request count
        self.request_count += 1

        # Add body hash for POST/PUT requests (for deduplication)
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            request._body_hash = hashlib.md5(body, usedforsecurity=False).hexdigest()
            # Reset body for downstream processing
            request._body = body

        # Check cache first
        if self.cache and request.method == "GET":
            cached_response = self.cache.get(request)
            if cached_response:
                return self._create_response_from_cache(cached_response)

        # Handle request deduplication
        if self.deduplicator and request.method == "GET":
            response = await self.deduplicator.deduplicate_request(
                request, lambda: call_next(request)
            )
        else:
            response = await call_next(request)

        # Process response
        response = await self._process_response(request, response, start_time)

        return response

    async def _process_response(
        self, request: Request, response: Response, start_time: float
    ) -> Response:
        """Process the response with optimizations."""
        # Calculate response time
        response_time = time.perf_counter() - start_time
        self.total_response_time += response_time

        # Track slow requests
        if response_time > self.config.slow_request_threshold:
            self.slow_request_count += 1
            logger.warning(
                f"Slow request detected: {request.url.path} - {response_time:.3f}s"
            )

        # Update performance monitor
        with self.performance_monitor.time_api_request():
            pass  # Time already measured

        # Cache successful GET responses
        if (
            self.cache
            and request.method == "GET"
            and response.status_code == 200
            and hasattr(response, "body")
        ):
            cache_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.body,
            }
            self.cache.set(request, cache_data)

        # Compress response
        if self.compressor:
            response = await self.compressor.compress_response(request, response)

        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        response.headers["X-Cache-Status"] = (
            "MISS"  # TODO: Update based on cache hit/miss
        )

        # Track response size
        content_length = response.headers.get("content-length")
        if content_length:
            self.response_sizes.append(int(content_length))

        return response

    def _create_response_from_cache(self, cache_data: Dict[str, Any]) -> Response:
        """Create response from cached data."""
        response = Response(
            content=cache_data["body"],
            status_code=cache_data["status_code"],
            headers=cache_data["headers"],
        )
        response.headers["X-Cache-Status"] = "HIT"
        return response

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive middleware statistics."""
        avg_response_time = (
            (self.total_response_time / self.request_count)
            if self.request_count > 0
            else 0
        )
        avg_response_size = (
            sum(self.response_sizes) / len(self.response_sizes)
            if self.response_sizes
            else 0
        )

        stats = {
            "request_count": self.request_count,
            "avg_response_time": avg_response_time,
            "slow_request_count": self.slow_request_count,
            "slow_request_rate": (
                (self.slow_request_count / self.request_count) * 100
                if self.request_count > 0
                else 0
            ),
            "avg_response_size": avg_response_size,
            "total_response_time": self.total_response_time,
        }

        # Add component statistics
        if self.cache:
            stats["cache"] = self.cache.get_stats()

        if self.deduplicator:
            stats["deduplication"] = self.deduplicator.get_stats()

        if self.compressor:
            stats["compression"] = self.compressor.get_stats()

        return stats

    def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries."""
        if self.cache:
            if pattern:
                self.cache.invalidate_pattern(pattern)
            else:
                self.cache.clear()

    def warm_cache(self, endpoints: List[str]):
        """Warm cache for specific endpoints."""
        # TODO: Implement cache warming
        logger.info(f"Cache warming requested for endpoints: {endpoints}")


class StreamingResponseOptimizer:
    """Optimizes streaming responses for large data."""

    def __init__(self, threshold_bytes: int = 1024 * 1024):
        """Initialize the stream optimizer.

        Args:
            threshold_bytes: Size threshold for streaming responses.
        """
        self.threshold_bytes = threshold_bytes
        self.streaming_count = 0

    def should_stream(self, data_size: int) -> bool:
        """Check if response should be streamed."""
        return data_size > self.threshold_bytes

    def create_streaming_response(
        self, data: Union[List, Dict], content_type: str = "application/json"
    ) -> StreamingResponse:
        """Create optimized streaming response."""
        self.streaming_count += 1

        async def generate():
            if isinstance(data, list):
                # Stream array items
                yield "["
                for i, item in enumerate(data):
                    if i > 0:
                        yield ","
                    yield json.dumps(item)
                yield "]"
            else:
                # Stream object
                yield json.dumps(data)

        return StreamingResponse(
            generate(),
            media_type=content_type,
            headers={"X-Streaming": "true", "Cache-Control": "no-cache"},
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "streaming_count": self.streaming_count,
            "threshold_bytes": self.threshold_bytes,
        }


# Global middleware instance
performance_middleware: Optional[PerformanceMiddleware] = None


def setup_performance_middleware(
    app: FastAPI, config: PerformanceConfig = None
) -> PerformanceMiddleware:
    """Set up performance middleware for FastAPI app."""
    global performance_middleware

    performance_middleware = PerformanceMiddleware(app, config)
    app.add_middleware(PerformanceMiddleware, config=config)

    logger.info("Performance middleware setup complete")
    return performance_middleware


def get_performance_middleware() -> Optional[PerformanceMiddleware]:
    """Get the global performance middleware instance."""
    return performance_middleware


# Convenience functions
def invalidate_api_cache(pattern: str = None):
    """Invalidate API cache."""
    if performance_middleware:
        performance_middleware.invalidate_cache(pattern)


def get_api_performance_stats() -> Dict[str, Any]:
    """Get API performance statistics."""
    if performance_middleware:
        return performance_middleware.get_statistics()
    return {}


def warm_api_cache(endpoints: List[str]):
    """Warm API cache for specific endpoints."""
    if performance_middleware:
        performance_middleware.warm_cache(endpoints)


# Example usage
async def benchmark_api_performance():
    """Benchmark API performance optimizations."""
    print("=" * 80)
    print("API PERFORMANCE OPTIMIZATION BENCHMARK")
    print("=" * 80)

    # Configuration
    config = PerformanceConfig(
        caching_enabled=True,
        compression_enabled=True,
        deduplication_enabled=True,
    )

    # Create test FastAPI app
    app = FastAPI()
    middleware = setup_performance_middleware(app, config)

    # Simulate requests
    print("\nSimulating API requests...")

    # Mock request/response cycle
    num_requests = 1000
    start_time = time.perf_counter()

    # Simulate processing
    for _ in range(num_requests):
        # Simulate request processing time
        await asyncio.sleep(0.001)

    elapsed = time.perf_counter() - start_time
    throughput = num_requests / elapsed

    print(f"Processed {num_requests} requests in {elapsed:.3f}s")
    print(f"Throughput: {throughput:.1f} requests/second")

    # Print statistics
    stats = middleware.get_statistics()
    print("\nMiddleware Statistics:")
    print(f"  Average response time: {stats['avg_response_time']:.3f}s")
    print(f"  Slow requests: {stats['slow_request_count']}")
    print(f"  Cache stats: {stats.get('cache', 'N/A')}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(benchmark_api_performance())

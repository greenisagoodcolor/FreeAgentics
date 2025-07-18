"""DDoS Protection and Rate Limiting Middleware for FastAPI.

This module implements comprehensive rate limiting and DDoS protection using Redis
for distributed storage and FastAPI middleware for request processing.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Set, Tuple

import redis.asyncio as aioredis
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from auth.security_logging import (
    SecurityEventSeverity,
    SecurityEventType,
    security_auditor,
)

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """Configuration for rate limiting rules."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10,
        window_size: int = 60,
        block_duration: int = 300,  # 5 minutes
        ddos_threshold: int = 1000,  # requests per minute for DDoS detection
        ddos_block_duration: int = 3600,  # 1 hour
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit
        self.window_size = window_size
        self.block_duration = block_duration
        self.ddos_threshold = ddos_threshold
        self.ddos_block_duration = ddos_block_duration


class EndpointRateLimits:
    """Define rate limits for different endpoint types."""

    # Authentication endpoints - stricter limits
    AUTH_ENDPOINTS = RateLimitConfig(
        requests_per_minute=5,
        requests_per_hour=100,
        burst_limit=3,
        block_duration=600,  # 10 minutes
    )

    # API endpoints - standard limits
    API_ENDPOINTS = RateLimitConfig(
        requests_per_minute=100,
        requests_per_hour=2000,
        burst_limit=20,
        block_duration=300,  # 5 minutes
    )

    # WebSocket endpoints - higher limits for real-time
    WEBSOCKET_ENDPOINTS = RateLimitConfig(
        requests_per_minute=200,
        requests_per_hour=5000,
        burst_limit=50,
        block_duration=60,  # 1 minute
    )

    # Static/health endpoints - more lenient
    STATIC_ENDPOINTS = RateLimitConfig(
        requests_per_minute=200,
        requests_per_hour=10000,
        burst_limit=100,
        block_duration=60,  # 1 minute
    )


class RateLimiter:
    """Redis-based distributed rate limiter."""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, int] = {}

    async def _get_client_key(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # Try to get real IP from headers (for proxy/load balancer)
        real_ip = (
            request.headers.get("X-Real-IP")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.client.host
            if request.client
            else "unknown"
        )

        # Include user ID if authenticated for more precise limiting
        user_id = None
        if hasattr(request.state, "user"):
            user_id = getattr(request.state.user, "user_id", None)

        if user_id:
            return f"rate_limit:user:{user_id}"
        else:
            return f"rate_limit:ip:{real_ip}"

    async def _get_endpoint_config(self, path: str) -> RateLimitConfig:
        """Get rate limit configuration for endpoint."""
        if path.startswith("/api/v1/auth"):
            return EndpointRateLimits.AUTH_ENDPOINTS
        elif path.startswith("/api/v1/websocket") or path.startswith("/ws"):
            return EndpointRateLimits.WEBSOCKET_ENDPOINTS
        elif path in ["/health", "/", "/docs", "/redoc"]:
            return EndpointRateLimits.STATIC_ENDPOINTS
        else:
            return EndpointRateLimits.API_ENDPOINTS

    async def _is_blocked(self, client_key: str, ip: str) -> bool:
        """Check if client is currently blocked."""
        # Check Redis for blocks
        blocked = await self.redis.get(f"blocked:{client_key}")
        if blocked:
            return True

        # Check Redis for DDoS blocks
        ddos_blocked = await self.redis.get(f"ddos_blocked:{ip}")
        if ddos_blocked:
            return True

        return False

    async def _record_request(
        self, client_key: str, config: RateLimitConfig
    ) -> Tuple[int, int]:
        """Record request and return current counts."""
        now = int(time.time())
        minute_key = f"{client_key}:minute:{now // 60}"
        hour_key = f"{client_key}:hour:{now // 3600}"

        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()

        # Increment counters
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)

        results = await pipe.execute()
        minute_count = results[0]
        hour_count = results[2]

        return minute_count, hour_count

    async def _detect_ddos(
        self, ip: str, minute_count: int, config: RateLimitConfig
    ) -> bool:
        """Detect potential DDoS attack."""
        if minute_count >= config.ddos_threshold:
            # Block IP for extended period
            await self.redis.setex(
                f"ddos_blocked:{ip}",
                config.ddos_block_duration,
                json.dumps(
                    {
                        "blocked_at": datetime.now().isoformat(),
                        "reason": "DDoS_DETECTION",
                        "requests_per_minute": minute_count,
                        "threshold": config.ddos_threshold,
                    }
                ),
            )

            # Log DDoS event
            security_auditor.log_event(
                SecurityEventType.DDOS_ATTACK,
                SecurityEventSeverity.CRITICAL,
                f"DDoS attack detected from IP {ip}",
                details={
                    "ip": ip,
                    "requests_per_minute": minute_count,
                    "threshold": config.ddos_threshold,
                    "block_duration": config.ddos_block_duration,
                },
            )

            return True

        return False

    async def _block_client(
        self, client_key: str, ip: str, config: RateLimitConfig, reason: str
    ) -> None:
        """Block client for rate limit violation."""
        block_data = {
            "blocked_at": datetime.now().isoformat(),
            "reason": reason,
            "expires_at": (
                datetime.now() + timedelta(seconds=config.block_duration)
            ).isoformat(),
        }

        await self.redis.setex(
            f"blocked:{client_key}",
            config.block_duration,
            json.dumps(block_data),
        )

        # Log rate limit violation
        security_auditor.log_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecurityEventSeverity.WARNING,
            f"Rate limit exceeded for {client_key}",
            details={
                "client_key": client_key,
                "ip": ip,
                "reason": reason,
                "block_duration": config.block_duration,
            },
        )

    async def check_rate_limit(
        self, request: Request
    ) -> Optional[JSONResponse]:
        """Check if request should be rate limited."""
        client_key = await self._get_client_key(request)
        config = await self._get_endpoint_config(request.url.path)

        # Extract IP for DDoS detection
        ip = (
            request.headers.get("X-Real-IP")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.client.host
            if request.client
            else "unknown"
        )

        # Check if already blocked
        if await self._is_blocked(client_key, ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Your IP has been temporarily blocked due to excessive requests",
                    "retry_after": 300,
                },
                headers={"Retry-After": "300"},
            )

        # Record request and get counts
        minute_count, hour_count = await self._record_request(
            client_key, config
        )

        # Check for DDoS
        if await self._detect_ddos(ip, minute_count, config):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "DDoS protection activated",
                    "message": "Suspicious activity detected. IP blocked.",
                    "retry_after": config.ddos_block_duration,
                },
                headers={"Retry-After": str(config.ddos_block_duration)},
            )

        # Check rate limits
        if minute_count > config.requests_per_minute:
            await self._block_client(
                client_key, ip, config, "REQUESTS_PER_MINUTE_EXCEEDED"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests per minute. Limit: {config.requests_per_minute}",
                    "retry_after": 60,
                },
                headers={"Retry-After": "60"},
            )

        if hour_count > config.requests_per_hour:
            await self._block_client(
                client_key, ip, config, "REQUESTS_PER_HOUR_EXCEEDED"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests per hour. Limit: {config.requests_per_hour}",
                    "retry_after": 3600,
                },
                headers={"Retry-After": "3600"},
            )

        # Add rate limit headers for client awareness
        response_headers = {
            "X-RateLimit-Limit-Minute": str(config.requests_per_minute),
            "X-RateLimit-Limit-Hour": str(config.requests_per_hour),
            "X-RateLimit-Remaining-Minute": str(
                max(0, config.requests_per_minute - minute_count)
            ),
            "X-RateLimit-Remaining-Hour": str(
                max(0, config.requests_per_hour - hour_count)
            ),
            "X-RateLimit-Reset": str(int(time.time()) + 60),
        }

        # Store headers for response
        request.state.rate_limit_headers = response_headers

        return None  # No rate limit violation


class DDoSProtectionMiddleware(BaseHTTPMiddleware):
    """Middleware for DDoS protection and rate limiting."""

    def __init__(self, app, redis_url: str = None):
        super().__init__(app)
        self.redis_url = redis_url or "redis://localhost:6379"
        self.redis_client = None
        self.rate_limiter = None
        self._connection_pool = None

    async def _get_redis_client(self) -> aioredis.Redis:
        """Get or create Redis client with connection pooling."""
        if self.redis_client is None:
            try:
                # Create connection pool for better performance
                self._connection_pool = aioredis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30,
                )

                self.redis_client = aioredis.Redis(
                    connection_pool=self._connection_pool
                )

                # Test connection
                await self.redis_client.ping()
                logger.info("Connected to Redis for rate limiting")

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                # Fallback to in-memory (not recommended for production)
                self.redis_client = None

        return self.redis_client

    async def _get_rate_limiter(self) -> Optional[RateLimiter]:
        """Get or create rate limiter."""
        if self.rate_limiter is None:
            redis_client = await self._get_redis_client()
            if redis_client:
                self.rate_limiter = RateLimiter(redis_client)

        return self.rate_limiter

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request with DDoS protection and rate limiting."""
        # Skip rate limiting for internal health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Get rate limiter
        rate_limiter = await self._get_rate_limiter()

        if rate_limiter:
            # Check rate limits
            rate_limit_response = await rate_limiter.check_rate_limit(request)
            if rate_limit_response:
                return rate_limit_response
        else:
            # Log warning if Redis is not available
            logger.warning("Rate limiting disabled - Redis not available")

        # Process request
        response = await call_next(request)

        # Add rate limit headers if available
        if hasattr(request.state, "rate_limit_headers"):
            for header, value in request.state.rate_limit_headers.items():
                response.headers[header] = value

        return response

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_redis_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._connection_pool:
            await self._connection_pool.disconnect()


class WebSocketRateLimiter:
    """Rate limiter specifically for WebSocket connections."""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.connection_limits = {
            "max_connections_per_ip": 10,
            "max_messages_per_minute": 100,
            "max_data_per_minute": 1024 * 1024,  # 1MB
        }

    async def check_connection_limit(self, ip: str) -> bool:
        """Check if IP can establish new WebSocket connection."""
        connection_key = f"ws_connections:{ip}"
        current_connections = await self.redis.get(connection_key)

        if current_connections:
            current_count = int(current_connections)
            if (
                current_count
                >= self.connection_limits["max_connections_per_ip"]
            ):
                return False

        # Increment connection count
        await self.redis.incr(connection_key)
        await self.redis.expire(connection_key, 300)  # 5 minutes

        return True

    async def check_message_rate(self, ip: str, message_size: int) -> bool:
        """Check if message rate is within limits."""
        now = int(time.time())
        minute_key = f"ws_messages:{ip}:{now // 60}"
        data_key = f"ws_data:{ip}:{now // 60}"

        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.incrby(data_key, message_size)
        pipe.expire(data_key, 60)

        results = await pipe.execute()
        message_count = results[0]
        data_count = results[2]

        if message_count > self.connection_limits["max_messages_per_minute"]:
            return False

        if data_count > self.connection_limits["max_data_per_minute"]:
            return False

        return True

    async def release_connection(self, ip: str):
        """Release WebSocket connection for IP."""
        connection_key = f"ws_connections:{ip}"
        await self.redis.decr(connection_key)

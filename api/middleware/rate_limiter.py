"""Comprehensive rate limiting and DDoS protection middleware.

This module implements:
- Redis-based distributed rate limiting
- Sliding window algorithm
- Per-endpoint configuration
- IP-based and user-based limits
- DDoS protection features
- Automatic blocking for suspicious patterns
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import redis.asyncio as redis
import yaml
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Import metrics if available
try:
    from observability.rate_limiting_metrics import rate_limiting_metrics

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""

    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    TOKEN_BUCKET = "token_bucket"  # nosec B105
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitType(str, Enum):
    """Types of rate limits."""

    IP_BASED = "ip_based"
    USER_BASED = "user_based"
    ENDPOINT_BASED = "endpoint_based"
    GLOBAL = "global"


class BlockReason(str, Enum):
    """Reasons for blocking."""

    RATE_LIMIT = "rate_limit"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    BLACKLISTED = "blacklisted"
    DDOS_ATTACK = "ddos_attack"


class RateLimitConfig:
    """Rate limit configuration."""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW,
        burst_size: Optional[int] = None,
    ):
        """Initialize rate limit configuration.

        Args:
            max_requests: Maximum number of requests allowed.
            window_seconds: Time window in seconds.
            algorithm: Rate limiting algorithm to use.
            burst_size: Maximum burst size for token bucket algorithm.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.algorithm = algorithm
        self.burst_size = burst_size or max_requests * 2


class EndpointConfig:
    """Endpoint-specific rate limit configuration."""

    def __init__(
        self,
        path_pattern: str,
        anonymous_limit: RateLimitConfig,
        authenticated_limit: RateLimitConfig,
        priority: int = 0,
    ):
        """Initialize endpoint-specific rate limit configuration.

        Args:
            path_pattern: URL path pattern to match.
            anonymous_limit: Rate limit configuration for anonymous users.
            authenticated_limit: Rate limit configuration for authenticated users.
            priority: Priority for matching (higher values are checked first).
        """
        self.path_pattern = path_pattern
        self.anonymous_limit = anonymous_limit
        self.authenticated_limit = authenticated_limit
        self.priority = priority


class SuspiciousPatternDetector:
    """Detects suspicious request patterns for DDoS protection."""

    def __init__(self):
        """Initialize suspicious pattern detector with predefined patterns."""
        self.patterns = {
            "rapid_404": {
                "threshold": 10,
                "window": 60,
            },  # 10 404s in 60 seconds
            "rapid_errors": {
                "threshold": 20,
                "window": 60,
            },  # 20 errors in 60 seconds
            "path_scanning": {
                "threshold": 15,
                "window": 30,
            },  # 15 different paths in 30 seconds
            "large_requests": {
                "threshold": 5,
                "window": 60,
            },  # 5 large requests in 60 seconds
        }
        self.metrics = defaultdict(lambda: defaultdict(list))

    async def check_pattern(
        self, identifier: str, pattern_type: str, redis_client: redis.Redis
    ) -> bool:
        """Check if identifier matches suspicious pattern."""
        key = f"pattern:{pattern_type}:{identifier}"
        now = time.time()
        window = self.patterns[pattern_type]["window"]
        threshold = self.patterns[pattern_type]["threshold"]

        # Get recent events
        await redis_client.zremrangebyscore(key, 0, now - window)
        count = await redis_client.zcard(key)

        if count >= threshold:
            return True

        # Add current event
        await redis_client.zadd(key, {str(now): now})
        await redis_client.expire(key, window + 60)

        return False


class RateLimiter:
    """Redis-based distributed rate limiter with DDoS protection."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config_file: Optional[str] = None,
        default_anonymous_limit: RateLimitConfig = None,
        default_authenticated_limit: RateLimitConfig = None,
    ):
        """Initialize the Redis-based rate limiter.

        Args:
            redis_url: Redis connection URL.
            config_file: Path to configuration file.
            default_anonymous_limit: Default limits for anonymous users.
            default_authenticated_limit: Default limits for authenticated users.
        """
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.config_file = config_file
        self.endpoint_configs: List[EndpointConfig] = []
        self.blacklist: Set[str] = set()
        self.whitelist: Set[str] = set()
        self.ip_networks_blacklist: List[ipaddress.IPv4Network] = []
        self.ip_networks_whitelist: List[ipaddress.IPv4Network] = []
        self.pattern_detector = SuspiciousPatternDetector()
        self.blocked_identifiers: Dict[str, Tuple[datetime, BlockReason]] = {}

        # Default limits
        self.default_anonymous_limit = (
            default_anonymous_limit
            or RateLimitConfig(max_requests=60, window_seconds=60)
        )
        self.default_authenticated_limit = (
            default_authenticated_limit
            or RateLimitConfig(max_requests=300, window_seconds=60)
        )

        # DDoS protection settings
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_header_size = 8192  # 8KB
        self.connection_limit_per_ip = 50
        self.block_duration = timedelta(minutes=30)

        # Load configuration if provided
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str):
        """Load rate limit configuration from YAML file."""
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Load endpoint configurations
            for endpoint in config.get("endpoints", []):
                self.endpoint_configs.append(
                    EndpointConfig(
                        path_pattern=endpoint["path"],
                        anonymous_limit=RateLimitConfig(
                            **endpoint["anonymous"]
                        ),
                        authenticated_limit=RateLimitConfig(
                            **endpoint["authenticated"]
                        ),
                        priority=endpoint.get("priority", 0),
                    )
                )

            # Sort by priority
            self.endpoint_configs.sort(key=lambda x: x.priority, reverse=True)

            # Load blacklist/whitelist
            self.blacklist = set(config.get("blacklist", {}).get("ips", []))
            self.whitelist = set(config.get("whitelist", {}).get("ips", []))

            # Load IP networks
            for network in config.get("blacklist", {}).get("networks", []):
                self.ip_networks_blacklist.append(
                    ipaddress.ip_network(network)
                )

            for network in config.get("whitelist", {}).get("networks", []):
                self.ip_networks_whitelist.append(
                    ipaddress.ip_network(network)
                )

            # Load DDoS protection settings
            ddos_config = config.get("ddos_protection", {})
            self.max_request_size = ddos_config.get(
                "max_request_size", self.max_request_size
            )
            self.max_header_size = ddos_config.get(
                "max_header_size", self.max_header_size
            )
            self.connection_limit_per_ip = ddos_config.get(
                "connection_limit_per_ip", self.connection_limit_per_ip
            )
            self.block_duration = timedelta(
                minutes=ddos_config.get("block_duration_minutes", 30)
            )

        except Exception as e:
            logger.error(f"Failed to load rate limit config: {e}")

    async def connect(self):
        """Connect to Redis."""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                self.redis_url, decode_responses=True
            )

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for proxy headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP in the chain
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fallback to direct connection
        if request.client:
            return request.client.host

        return "unknown"

    def is_ip_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        if ip in self.whitelist:
            return True

        try:
            ip_addr = ipaddress.ip_address(ip)
            for network in self.ip_networks_whitelist:
                if ip_addr in network:
                    return True
        except ValueError:
            pass

        return False

    def is_ip_blacklisted(self, ip: str) -> bool:
        """Check if IP is blacklisted."""
        if ip in self.blacklist:
            return True

        try:
            ip_addr = ipaddress.ip_address(ip)
            for network in self.ip_networks_blacklist:
                if ip_addr in network:
                    return True
        except ValueError:
            pass

        return False

    async def is_blocked(
        self, identifier: str
    ) -> Tuple[bool, Optional[BlockReason]]:
        """Check if identifier is currently blocked."""
        if identifier in self.blocked_identifiers:
            blocked_until, reason = self.blocked_identifiers[identifier]
            if datetime.utcnow() < blocked_until:
                return True, reason
            else:
                del self.blocked_identifiers[identifier]

        # Check Redis for distributed blocks
        if self.redis_client:
            block_key = f"blocked:{identifier}"
            block_data = await self.redis_client.get(block_key)
            if block_data:
                block_info = json.loads(block_data)
                return True, BlockReason(block_info["reason"])

        return False, None

    async def block_identifier(
        self,
        identifier: str,
        reason: BlockReason,
        duration: Optional[timedelta] = None,
    ):
        """Block an identifier for a specified duration."""
        duration = duration or self.block_duration
        blocked_until = datetime.utcnow() + duration

        # Store locally
        self.blocked_identifiers[identifier] = (blocked_until, reason)

        # Record metrics if enabled
        if METRICS_ENABLED:
            block_type = "user" if identifier.startswith("user:") else "ip"
            rate_limiting_metrics.record_block(block_type, reason.value)

        # Store in Redis for distributed blocking
        if self.redis_client:
            block_key = f"blocked:{identifier}"
            block_data = {
                "reason": reason.value,
                "blocked_until": blocked_until.isoformat(),
                "blocked_at": datetime.utcnow().isoformat(),
            }
            await self.redis_client.setex(
                block_key,
                int(duration.total_seconds()),
                json.dumps(block_data),
            )

    async def check_sliding_window(
        self, identifier: str, config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using sliding window algorithm."""
        if not self.redis_client:
            return True, {"allowed": True, "reason": "Redis not connected"}

        # Time the operation if metrics enabled
        if METRICS_ENABLED:
            start_time = time.time()

        key = f"rate_limit:sliding:{identifier}"
        now = time.time()
        window_start = now - config.window_seconds

        # Remove old entries
        await self.redis_client.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        request_count = await self.redis_client.zcard(key)

        if METRICS_ENABLED:
            rate_limiting_metrics.time_redis_operation("zcard")
            elapsed = time.time() - start_time
            rate_limiting_metrics.rate_limit_check_duration.labels(
                algorithm="sliding_window"
            ).observe(elapsed)

        if request_count >= config.max_requests:
            # Get oldest request time to calculate retry after
            oldest_request = await self.redis_client.zrange(
                key, 0, 0, withscores=True
            )
            if oldest_request:
                retry_after = int(
                    oldest_request[0][1] + config.window_seconds - now
                )
            else:
                retry_after = config.window_seconds

            return False, {
                "allowed": False,
                "request_count": request_count,
                "max_requests": config.max_requests,
                "window_seconds": config.window_seconds,
                "retry_after": retry_after,
            }

        # Add current request
        await self.redis_client.zadd(key, {str(now): now})
        await self.redis_client.expire(key, config.window_seconds + 60)

        return True, {
            "allowed": True,
            "request_count": request_count + 1,
            "max_requests": config.max_requests,
            "window_seconds": config.window_seconds,
        }

    async def check_token_bucket(
        self, identifier: str, config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using token bucket algorithm."""
        if not self.redis_client:
            return True, {"allowed": True, "reason": "Redis not connected"}

        key = f"rate_limit:token_bucket:{identifier}"
        now = time.time()
        rate = config.max_requests / config.window_seconds
        burst = config.burst_size

        # Get current bucket state
        bucket_data = await self.redis_client.hgetall(key)

        if bucket_data:
            tokens = float(bucket_data.get("tokens", burst))
            last_update = float(bucket_data.get("last_update", now))
        else:
            tokens = burst
            last_update = now

        # Calculate tokens to add
        time_passed = now - last_update
        tokens = min(burst, tokens + time_passed * rate)

        if tokens >= 1:
            # Consume a token
            tokens -= 1
            await self.redis_client.hset(
                key, mapping={"tokens": tokens, "last_update": now}
            )
            await self.redis_client.expire(key, config.window_seconds * 2)

            return True, {
                "allowed": True,
                "tokens_remaining": int(tokens),
                "burst_size": burst,
            }
        else:
            # Calculate retry after
            retry_after = int((1 - tokens) / rate)

            return False, {
                "allowed": False,
                "tokens_remaining": 0,
                "burst_size": burst,
                "retry_after": retry_after,
            }

    async def check_rate_limit(
        self, identifier: str, config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using configured algorithm."""
        if config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self.check_sliding_window(identifier, config)
        elif config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self.check_token_bucket(identifier, config)
        else:
            # Default to sliding window
            return await self.check_sliding_window(identifier, config)

    def get_endpoint_config(self, path: str) -> Optional[EndpointConfig]:
        """Get rate limit configuration for endpoint."""
        for config in self.endpoint_configs:
            # Simple pattern matching (can be enhanced with regex)
            if (
                path.startswith(config.path_pattern)
                or config.path_pattern == "*"
            ):
                return config
        return None

    async def check_ddos_patterns(
        self, request: Request, ip: str, response_status: Optional[int] = None
    ) -> Optional[BlockReason]:
        """Check for DDoS attack patterns."""
        # Check rapid 404s
        if response_status == 404:
            if await self.pattern_detector.check_pattern(
                ip, "rapid_404", self.redis_client
            ):
                if METRICS_ENABLED:
                    rate_limiting_metrics.record_suspicious_pattern(
                        "rapid_404"
                    )
                return BlockReason.SUSPICIOUS_PATTERN

        # Check rapid errors
        if response_status and response_status >= 400:
            if await self.pattern_detector.check_pattern(
                ip, "rapid_errors", self.redis_client
            ):
                if METRICS_ENABLED:
                    rate_limiting_metrics.record_suspicious_pattern(
                        "rapid_errors"
                    )
                return BlockReason.SUSPICIOUS_PATTERN

        # Check path scanning
        path_key = f"paths:{ip}"
        await self.redis_client.sadd(path_key, request.url.path)
        await self.redis_client.expire(path_key, 30)
        path_count = await self.redis_client.scard(path_key)
        if path_count > 15:
            if METRICS_ENABLED:
                rate_limiting_metrics.record_suspicious_pattern(
                    "path_scanning"
                )
            return BlockReason.SUSPICIOUS_PATTERN

        # Check large requests
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            if await self.pattern_detector.check_pattern(
                ip, "large_requests", self.redis_client
            ):
                if METRICS_ENABLED:
                    rate_limiting_metrics.record_suspicious_pattern(
                        "large_requests"
                    )
                    rate_limiting_metrics.record_ddos_attack(
                        "large_request_flood", ip
                    )
                return BlockReason.DDOS_ATTACK

        return None

    async def process_request(
        self, request: Request, user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[Response]]:
        """Process request through rate limiter."""
        ip = self.get_client_ip(request)

        # Check whitelist
        if self.is_ip_whitelisted(ip):
            return True, None

        # Check blacklist
        if self.is_ip_blacklisted(ip):
            return False, JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"},
            )

        # Check if blocked
        is_blocked, block_reason = await self.is_blocked(ip)
        if is_blocked:
            return False, JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"Blocked: {block_reason.value}"},
            )

        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return False, JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request too large"},
            )

        # Check header size
        header_size = sum(len(k) + len(v) for k, v in request.headers.items())
        if header_size > self.max_header_size:
            return False, JSONResponse(
                status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE,
                content={"detail": "Headers too large"},
            )

        # Get endpoint configuration
        endpoint_config = self.get_endpoint_config(request.url.path)

        # Determine rate limit config based on authentication
        if user_id:
            # Authenticated user
            identifier = f"user:{user_id}"
            config = (
                endpoint_config.authenticated_limit
                if endpoint_config
                else self.default_authenticated_limit
            )
        else:
            # Anonymous user
            identifier = f"ip:{ip}"
            config = (
                endpoint_config.anonymous_limit
                if endpoint_config
                else self.default_anonymous_limit
            )

        # Check rate limit
        allowed, info = await self.check_rate_limit(identifier, config)

        if not allowed:
            # Check for repeated violations
            violation_key = f"violations:{identifier}"
            violations = await self.redis_client.incr(violation_key)
            await self.redis_client.expire(violation_key, 3600)

            if violations > 10:
                await self.block_identifier(identifier, BlockReason.RATE_LIMIT)

            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": info.get("retry_after", 60),
                },
            )
            response.headers["Retry-After"] = str(info.get("retry_after", 60))
            response.headers["X-RateLimit-Limit"] = str(config.max_requests)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time()) + info.get("retry_after", 60)
            )

            return False, response

        # Add rate limit headers to successful requests
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(config.max_requests),
            "X-RateLimit-Remaining": str(
                config.max_requests - info.get("request_count", 0)
            ),
            "X-RateLimit-Reset": str(int(time.time()) + config.window_seconds),
        }

        return True, None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting and DDoS protection."""

    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: RateLimiter,
        get_user_id: Optional[callable] = None,
    ):
        """Initialize the rate limiting middleware.

        Args:
            app: The ASGI application.
            rate_limiter: The rate limiter instance.
            get_user_id: Optional function to extract user ID from request.
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.get_user_id = get_user_id

    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiter."""
        # Connect to Redis if not connected
        await self.rate_limiter.connect()

        # Extract user ID if authenticated
        user_id = None
        if self.get_user_id:
            try:
                user_id = await self.get_user_id(request)
            except Exception as e:
                # Log failed user ID extraction but continue with anonymous rate limiting
                logger.debug(
                    f"Failed to extract user ID for rate limiting: {e}"
                )
                user_id = None

        # Process through rate limiter
        allowed, error_response = await self.rate_limiter.process_request(
            request, user_id
        )

        if not allowed:
            return error_response

        # Process request
        try:
            response = await call_next(request)

            # Add rate limit headers
            if hasattr(request.state, "rate_limit_headers"):
                for header, value in request.state.rate_limit_headers.items():
                    response.headers[header] = value

            # Check for DDoS patterns
            ip = self.rate_limiter.get_client_ip(request)
            block_reason = await self.rate_limiter.check_ddos_patterns(
                request, ip, response.status_code
            )

            if block_reason:
                await self.rate_limiter.block_identifier(ip, block_reason)

            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise


# Helper functions for easy integration
def create_rate_limiter(
    redis_url: str = "redis://localhost:6379",
    config_file: Optional[str] = None,
) -> RateLimiter:
    """Create and configure rate limiter instance."""
    return RateLimiter(redis_url=redis_url, config_file=config_file)


def create_middleware(
    app: ASGIApp,
    redis_url: str = "redis://localhost:6379",
    config_file: Optional[str] = None,
    get_user_id: Optional[callable] = None,
) -> RateLimitMiddleware:
    """Create rate limit middleware for FastAPI."""
    rate_limiter = create_rate_limiter(redis_url, config_file)
    return RateLimitMiddleware(app, rate_limiter, get_user_id)


# Global rate limiter instance
_global_rate_limiter = None


def get_global_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = create_rate_limiter()
    return _global_rate_limiter


def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW,
):
    """Create a decorator for rate limiting individual endpoints."""

    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            rate_limiter = get_global_rate_limiter()
            await rate_limiter.connect()

            # Create temporary config for this endpoint
            config = RateLimitConfig(
                max_requests=max_requests,
                window_seconds=window_seconds,
                algorithm=algorithm,
            )

            ip = rate_limiter.get_client_ip(request)
            identifier = f"ip:{ip}"

            # Check rate limit
            allowed, info = await rate_limiter.check_rate_limit(
                identifier, config
            )

            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(info.get("retry_after", 60))},
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator

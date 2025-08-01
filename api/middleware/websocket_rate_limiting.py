"""WebSocket Rate Limiting Integration.

This module provides rate limiting specifically for WebSocket connections
to prevent abuse of real-time communication endpoints.
"""

import logging
import os
from typing import Callable, Dict, Optional

import redis.asyncio as aioredis
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from auth.security_logging import SecurityEventSeverity, SecurityEventType, security_auditor

from .ddos_protection import WebSocketRateLimiter

logger = logging.getLogger(__name__)

# Check if we're in dev mode (no database)
DEV_MODE = os.getenv("DATABASE_URL") is None


class WebSocketRateLimitManager:
    """Manages rate limiting for WebSocket connections."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize WebSocket rate limit manager."""
        self.redis_url = redis_url  # Don't default to localhost if not provided
        self.redis_client: Optional[aioredis.Redis] = None
        self.rate_limiter: Optional["WebSocketRateLimiter"] = None
        self.active_connections: Dict[str, WebSocket] = {}

    async def _get_redis_client(self) -> Optional[aioredis.Redis]:
        """Get or create Redis client."""
        if self.redis_client is None and self.redis_url:
            try:
                self.redis_client = aioredis.from_url(
                    self.redis_url, max_connections=20, retry_on_timeout=True
                )
                if self.redis_client:
                    await self.redis_client.ping()
                    logger.info("WebSocket rate limiter connected to Redis")
            except Exception as e:
                if not DEV_MODE:
                    logger.error(f"Failed to connect to Redis for WebSocket rate limiting: {e}")
                else:
                    logger.debug(
                        f"Redis not available in demo mode for WebSocket rate limiting: {e}"
                    )
                self.redis_client = None
        elif not self.redis_url:
            logger.debug("No Redis URL configured for WebSocket rate limiting")

        return self.redis_client

    async def _get_rate_limiter(self) -> Optional[WebSocketRateLimiter]:
        """Get or create WebSocket rate limiter."""
        if self.rate_limiter is None:
            redis_client = await self._get_redis_client()
            if redis_client:
                self.rate_limiter = WebSocketRateLimiter(redis_client)

        return self.rate_limiter

    def _get_client_ip(self, websocket: WebSocket) -> str:
        """Extract client IP from WebSocket connection."""
        # Try to get real IP from headers
        real_ip = None
        if websocket.headers:
            real_ip = (
                websocket.headers.get("x-real-ip")
                or websocket.headers.get("x-forwarded-for", "").split(",")[0].strip()
            )

        # Fallback to client host
        if not real_ip and websocket.client:
            real_ip = websocket.client.host

        return real_ip or "unknown"

    async def check_connection_allowed(self, websocket: WebSocket) -> bool:
        """Check if WebSocket connection is allowed based on rate limits."""
        # Skip rate limiting entirely in demo mode
        if DEV_MODE:
            return True

        rate_limiter = await self._get_rate_limiter()
        if not rate_limiter:
            logger.warning("WebSocket rate limiting disabled - Redis not available")
            return True

        client_ip = self._get_client_ip(websocket)

        # Check connection limit
        allowed = await rate_limiter.check_connection_limit(client_ip)

        if not allowed:
            # Log rate limit violation
            security_auditor.log_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                SecurityEventSeverity.WARNING,
                f"WebSocket connection limit exceeded for IP {client_ip}",
                details={
                    "ip": client_ip,
                    "connection_type": "websocket",
                    "limit_type": "max_connections_per_ip",
                },
            )

            # Send rate limit message before closing
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Connection limit exceeded. Too many connections from your IP.",
                        "code": "RATE_LIMIT_EXCEEDED",
                    }
                )
            except Exception as e:
                # Log failed message send, but continue (websocket may already be closed)
                logger.debug(f"Failed to send rate limit message to websocket: {e}")

        return allowed

    async def check_message_allowed(self, websocket: WebSocket, message: str) -> bool:
        """Check if WebSocket message is allowed based on rate limits."""
        # Skip rate limiting entirely in demo mode
        if DEV_MODE:
            return True

        rate_limiter = await self._get_rate_limiter()
        if not rate_limiter:
            return True

        client_ip = self._get_client_ip(websocket)
        message_size = len(message.encode("utf-8"))

        # Check message rate
        allowed = await rate_limiter.check_message_rate(client_ip, message_size)

        if not allowed:
            # Log rate limit violation
            security_auditor.log_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                SecurityEventSeverity.WARNING,
                f"WebSocket message rate limit exceeded for IP {client_ip}",
                details={
                    "ip": client_ip,
                    "message_size": message_size,
                    "limit_type": "message_rate",
                },
            )

            # Send rate limit message
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Message rate limit exceeded. Please slow down.",
                        "code": "MESSAGE_RATE_LIMIT_EXCEEDED",
                    }
                )
            except Exception as e:
                # Log failed message send, but continue (websocket may already be closed)
                logger.debug(f"Failed to send message rate limit error to websocket: {e}")

        return allowed

    async def register_connection(self, websocket: WebSocket, connection_id: str):
        """Register a WebSocket connection."""
        self.active_connections[connection_id] = websocket

        client_ip = self._get_client_ip(websocket)
        logger.info(f"WebSocket connection registered: {connection_id} from {client_ip}")

    async def unregister_connection(self, connection_id: str):
        """Unregister a WebSocket connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            client_ip = self._get_client_ip(websocket)

            # Release connection in rate limiter
            rate_limiter = await self._get_rate_limiter()
            if rate_limiter:
                await rate_limiter.release_connection(client_ip)

            del self.active_connections[connection_id]
            logger.info(f"WebSocket connection unregistered: {connection_id} from {client_ip}")

    async def handle_websocket_with_rate_limiting(
        self,
        websocket: WebSocket,
        connection_id: str,
        message_handler: Callable,
    ):
        """Handle WebSocket connection with rate limiting."""
        # Skip rate limiting entirely in demo mode
        if not DEV_MODE:
            # Check if connection is allowed
            if not await self.check_connection_allowed(websocket):
                await websocket.close(code=1008, reason="Rate limit exceeded")
                return

        try:
            # Accept connection
            await websocket.accept()
            await self.register_connection(websocket, connection_id)

            # Handle messages
            while True:
                try:
                    # Receive message
                    message = await websocket.receive_text()

                    # Check if message is allowed (skip in demo mode)
                    if not DEV_MODE and not await self.check_message_allowed(websocket, message):
                        # Send warning but don't close connection
                        await websocket.send_json(
                            {
                                "type": "warning",
                                "message": "Rate limit exceeded. Message dropped.",
                                "code": "MESSAGE_DROPPED",
                            }
                        )
                        continue

                    # Process message
                    await message_handler(websocket, message)

                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected: {connection_id}")
                    break
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    try:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Internal server error",
                                "code": "INTERNAL_ERROR",
                            }
                        )
                    except Exception as e:
                        # Log failed error message send
                        logger.debug(f"Failed to send internal error message to websocket: {e}")
                    break

        except Exception as e:
            logger.error(f"Error in WebSocket connection {connection_id}: {e}")

        finally:
            # Cleanup
            await self.unregister_connection(connection_id)

            # Ensure connection is closed
            if websocket.client_state != WebSocketState.DISCONNECTED:
                try:
                    await websocket.close()
                except Exception as e:
                    # Log failed websocket close attempt
                    logger.debug(f"Failed to close websocket connection {connection_id}: {e}")


# Global instance for use across the application
# Only pass REDIS_URL if it's set in environment
websocket_rate_limit_manager = WebSocketRateLimitManager(redis_url=os.getenv("REDIS_URL"))


def get_websocket_rate_limit_manager() -> WebSocketRateLimitManager:
    """Get the global WebSocket rate limit manager."""
    return websocket_rate_limit_manager


async def websocket_rate_limit_dependency(websocket: WebSocket) -> bool:
    """FastAPI dependency for WebSocket rate limiting."""
    manager = get_websocket_rate_limit_manager()
    return await manager.check_connection_allowed(websocket)

"""
WebSocket Authentication Handler

Implements secure WebSocket authentication using JWT tokens from the existing auth system.
Provides token validation, refresh, rate limiting, and security measures for WebSocket connections.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from auth.security_implementation import (
    AuthenticationManager,
    Permission,
    TokenData,
    UserRole,
)

logger = logging.getLogger(__name__)

# Global auth manager instance
auth_manager = AuthenticationManager()


# WebSocket-specific error codes
class WebSocketErrorCode:
    """WebSocket close codes for authentication errors."""

    AUTHENTICATION_FAILED = 4001
    TOKEN_EXPIRED = 4002
    PERMISSION_DENIED = 4003
    RATE_LIMITED = 4004
    INVALID_MESSAGE = 4005


class WebSocketAuthConfig(BaseModel):
    """Configuration for WebSocket authentication."""

    token_refresh_interval: int = Field(
        default=300,
        description="Token refresh interval in seconds (5 minutes)",
    )
    max_connections_per_user: int = Field(
        default=5, description="Maximum concurrent connections per user"
    )
    heartbeat_interval: int = Field(
        default=30, description="Heartbeat interval in seconds"
    )
    heartbeat_timeout: int = Field(
        default=60, description="Heartbeat timeout in seconds"
    )
    allow_query_token: bool = Field(
        default=True, description="Allow token in query parameters"
    )
    allow_header_token: bool = Field(
        default=True, description="Allow token in headers"
    )
    origin_whitelist: Optional[Set[str]] = Field(
        default=None, description="Allowed origins (None = allow all)"
    )


class ConnectionState(BaseModel):
    """State information for a WebSocket connection."""

    client_id: str
    user_data: Optional[TokenData] = None
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    permissions: Set[Permission] = Field(default_factory=set)
    metadata: Dict = Field(default_factory=dict)


class WebSocketAuthHandler:
    """Handles WebSocket authentication and connection management."""

    def __init__(self, config: Optional[WebSocketAuthConfig] = None):
        """Initialize the auth handler with configuration."""
        self.config = config or WebSocketAuthConfig()
        self.connections: Dict[str, ConnectionState] = {}
        self.user_connections: Dict[
            str, Set[str]
        ] = {}  # user_id -> set of client_ids
        self.rate_limiter: Dict[
            str, list
        ] = {}  # IP -> list of connection times

    async def authenticate_connection(
        self, websocket: WebSocket, client_id: str, token: Optional[str] = None
    ) -> TokenData:
        """
        Authenticate a WebSocket connection using JWT token.

        Args:
            websocket: The WebSocket connection
            client_id: Unique identifier for this connection
            token: JWT token (can be from query params or headers)

        Returns:
            TokenData if authentication succeeds

        Raises:
            WebSocketDisconnect with appropriate error code
        """
        try:
            # Extract token if not provided
            if not token:
                token = await self._extract_token(websocket)

            if not token:
                logger.warning(
                    f"No token provided for WebSocket connection: {client_id}"
                )
                await websocket.close(
                    code=WebSocketErrorCode.AUTHENTICATION_FAILED
                )
                raise WebSocketDisconnect(
                    code=WebSocketErrorCode.AUTHENTICATION_FAILED
                )

            # Verify origin if whitelist is configured
            if not await self._verify_origin(websocket):
                logger.warning(
                    f"Invalid origin for WebSocket connection: {client_id}"
                )
                await websocket.close(
                    code=WebSocketErrorCode.AUTHENTICATION_FAILED
                )
                raise WebSocketDisconnect(
                    code=WebSocketErrorCode.AUTHENTICATION_FAILED
                )

            # Check rate limiting
            if not await self._check_rate_limit(websocket):
                logger.warning(
                    f"Rate limit exceeded for WebSocket connection: {client_id}"
                )
                await websocket.close(code=WebSocketErrorCode.RATE_LIMITED)
                raise WebSocketDisconnect(code=WebSocketErrorCode.RATE_LIMITED)

            # Verify JWT token
            try:
                user_data = auth_manager.verify_token(token)
            except HTTPException as e:
                if "expired" in str(e.detail).lower():
                    logger.warning(
                        f"Expired token for WebSocket connection: {client_id}"
                    )
                    await websocket.close(
                        code=WebSocketErrorCode.TOKEN_EXPIRED
                    )
                    raise WebSocketDisconnect(
                        code=WebSocketErrorCode.TOKEN_EXPIRED
                    )
                else:
                    logger.warning(
                        f"Invalid token for WebSocket connection: {client_id}"
                    )
                    await websocket.close(
                        code=WebSocketErrorCode.AUTHENTICATION_FAILED
                    )
                    raise WebSocketDisconnect(
                        code=WebSocketErrorCode.AUTHENTICATION_FAILED
                    )

            # Check concurrent connections limit
            if not await self._check_connection_limit(
                user_data.user_id, client_id
            ):
                logger.warning(
                    f"Connection limit exceeded for user: {user_data.username}"
                )
                await websocket.close(code=WebSocketErrorCode.RATE_LIMITED)
                raise WebSocketDisconnect(code=WebSocketErrorCode.RATE_LIMITED)

            # Store connection state
            connection_state = ConnectionState(
                client_id=client_id,
                user_data=user_data,
                permissions=set(user_data.permissions),
                metadata={
                    "user_id": user_data.user_id,
                    "username": user_data.username,
                    "role": user_data.role.value,
                    "authenticated": True,
                    "ip_address": self._get_client_ip(websocket),
                },
            )

            self.connections[client_id] = connection_state

            # Track user connections
            if user_data.user_id not in self.user_connections:
                self.user_connections[user_data.user_id] = set()
            self.user_connections[user_data.user_id].add(client_id)

            logger.info(
                f"WebSocket authenticated for user: {user_data.username} (client: {client_id})"
            )
            return user_data

        except WebSocketDisconnect:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during WebSocket authentication: {e}"
            )
            await websocket.close(
                code=WebSocketErrorCode.AUTHENTICATION_FAILED
            )
            raise WebSocketDisconnect(
                code=WebSocketErrorCode.AUTHENTICATION_FAILED
            )

    async def _extract_token(self, websocket: WebSocket) -> Optional[str]:
        """Extract JWT token from WebSocket connection."""
        token = None

        # Try to get token from query parameters
        if self.config.allow_query_token:
            query_params = parse_qs(urlparse(str(websocket.url)).query)
            if "token" in query_params:
                token = query_params["token"][0]

        # Try to get token from headers
        if not token and self.config.allow_header_token:
            auth_header = websocket.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]

        return token

    async def _verify_origin(self, websocket: WebSocket) -> bool:
        """Verify the origin of the WebSocket connection."""
        if not self.config.origin_whitelist:
            return True

        origin = websocket.headers.get("origin", "")
        return origin in self.config.origin_whitelist

    async def _check_rate_limit(self, websocket: WebSocket) -> bool:
        """Check rate limiting for the connection."""
        ip = self._get_client_ip(websocket)
        now = datetime.utcnow()

        if ip not in self.rate_limiter:
            self.rate_limiter[ip] = []

        # Clean old entries (older than 1 minute)
        self.rate_limiter[ip] = [
            t for t in self.rate_limiter[ip] if (now - t).total_seconds() < 60
        ]

        # Check if too many connections in the last minute
        if (
            len(self.rate_limiter[ip]) >= 10
        ):  # Max 10 connections per minute per IP
            return False

        self.rate_limiter[ip].append(now)
        return True

    async def _check_connection_limit(
        self, user_id: str, client_id: str
    ) -> bool:
        """Check if user has exceeded connection limit."""
        if user_id not in self.user_connections:
            return True

        current_connections = len(self.user_connections[user_id])
        if client_id in self.user_connections[user_id]:
            # This connection is already counted
            return True

        return current_connections < self.config.max_connections_per_user

    def _get_client_ip(self, websocket: WebSocket) -> str:
        """Get client IP address from WebSocket."""
        # Check for forwarded IP
        forwarded = websocket.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Use direct client IP
        if websocket.client:
            return websocket.client.host

        return "unknown"

    async def refresh_token(
        self, client_id: str, refresh_token: str
    ) -> Tuple[str, TokenData]:
        """
        Refresh access token for a WebSocket connection.

        Args:
            client_id: The client connection ID
            refresh_token: The refresh token

        Returns:
            Tuple of (new_access_token, updated_user_data)
        """
        if client_id not in self.connections:
            raise ValueError("Connection not found")

        try:
            # Get new tokens
            (
                new_access_token,
                new_refresh_token,
            ) = auth_manager.refresh_access_token(refresh_token)

            # Verify the new access token to get updated user data
            user_data = auth_manager.verify_token(new_access_token)

            # Update connection state
            self.connections[client_id].user_data = user_data
            self.connections[client_id].permissions = set(
                user_data.permissions
            )

            logger.info(f"Token refreshed for client: {client_id}")
            return new_access_token, user_data

        except HTTPException as e:
            logger.error(
                f"Token refresh failed for client {client_id}: {e.detail}"
            )
            raise

    async def verify_permission(
        self, client_id: str, required_permission: Permission
    ) -> bool:
        """
        Verify if a connection has the required permission.

        Args:
            client_id: The client connection ID
            required_permission: The permission to check

        Returns:
            True if permission is granted, False otherwise
        """
        if client_id not in self.connections:
            return False

        connection = self.connections[client_id]
        if not connection.user_data:
            return False

        return required_permission in connection.permissions

    async def update_heartbeat(self, client_id: str) -> None:
        """Update the last heartbeat time for a connection."""
        if client_id in self.connections:
            self.connections[client_id].last_heartbeat = datetime.utcnow()

    async def check_heartbeat_timeout(self) -> Set[str]:
        """
        Check for connections that have timed out.

        Returns:
            Set of client IDs that have timed out
        """
        now = datetime.utcnow()
        timeout_seconds = self.config.heartbeat_timeout
        timed_out = set()

        for client_id, connection in self.connections.items():
            if (
                now - connection.last_heartbeat
            ).total_seconds() > timeout_seconds:
                timed_out.add(client_id)

        return timed_out

    async def disconnect(self, client_id: str) -> None:
        """Clean up connection state on disconnect."""
        if client_id not in self.connections:
            return

        connection = self.connections[client_id]

        # Remove from user connections
        if (
            connection.user_data
            and connection.user_data.user_id in self.user_connections
        ):
            self.user_connections[connection.user_data.user_id].discard(
                client_id
            )
            if not self.user_connections[connection.user_data.user_id]:
                del self.user_connections[connection.user_data.user_id]

        # Remove connection state
        del self.connections[client_id]

        logger.info(f"WebSocket disconnected: {client_id}")

    def get_connection_info(self, client_id: str) -> Optional[Dict]:
        """Get information about a specific connection."""
        if client_id not in self.connections:
            return None

        connection = self.connections[client_id]
        return {
            "client_id": client_id,
            "user_id": connection.user_data.user_id
            if connection.user_data
            else None,
            "username": connection.user_data.username
            if connection.user_data
            else None,
            "role": connection.user_data.role.value
            if connection.user_data
            else None,
            "connected_at": connection.connected_at.isoformat(),
            "last_heartbeat": connection.last_heartbeat.isoformat(),
            "permissions": [p.value for p in connection.permissions],
            "metadata": connection.metadata,
        }

    def get_user_connections(self, user_id: str) -> Set[str]:
        """Get all connection IDs for a specific user."""
        return self.user_connections.get(user_id, set()).copy()

    async def broadcast_to_user(self, user_id: str, message: Dict) -> int:
        """
        Broadcast a message to all connections of a specific user.

        Returns:
            Number of connections the message was sent to
        """
        client_ids = self.get_user_connections(user_id)
        sent_count = 0

        for client_id in client_ids:
            if client_id in self.connections:
                # This would integrate with the ConnectionManager
                # to actually send the message
                sent_count += 1

        return sent_count


# Global WebSocket auth handler instance
ws_auth_handler = WebSocketAuthHandler()


async def websocket_auth(
    websocket: WebSocket, token: Optional[str] = None
) -> TokenData:
    """
    Authenticate a WebSocket connection.

    This is the main entry point for WebSocket authentication that integrates
    with the existing WebSocket endpoint.
    """
    client_id = str(websocket.url).split("/")[-1]  # Extract client_id from URL
    return await ws_auth_handler.authenticate_connection(
        websocket, client_id, token
    )


async def handle_token_refresh(client_id: str, refresh_token: str) -> Dict:
    """Handle token refresh request from WebSocket client."""
    try:
        new_access_token, user_data = await ws_auth_handler.refresh_token(
            client_id, refresh_token
        )
        return {
            "type": "token_refreshed",
            "access_token": new_access_token,
            "expires_at": user_data.exp.isoformat(),
        }
    except Exception as e:
        return {
            "type": "error",
            "code": "TOKEN_REFRESH_FAILED",
            "message": str(e),
        }


async def websocket_heartbeat_monitor():
    """
    Background task to monitor WebSocket heartbeats and disconnect timed-out connections.

    This should be started as a background task when the server starts.
    """
    while True:
        try:
            timed_out_clients = await ws_auth_handler.check_heartbeat_timeout()

            for client_id in timed_out_clients:
                logger.warning(
                    f"WebSocket heartbeat timeout for client: {client_id}"
                )
                # The actual disconnection would be handled by the ConnectionManager
                await ws_auth_handler.disconnect(client_id)

            await asyncio.sleep(10)  # Check every 10 seconds

        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}")
            await asyncio.sleep(10)

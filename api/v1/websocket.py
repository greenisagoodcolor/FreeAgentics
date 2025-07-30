"""WebSocket endpoints for real-time agent monitoring and communication."""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator

from api.middleware.websocket_rate_limiting import websocket_rate_limit_manager
from auth.security_implementation import Permission
from websocket_server.auth_handler import (
    WebSocketErrorCode,
    handle_token_refresh,
    ws_auth_handler,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class WebSocketMessage(BaseModel):
    """Standard WebSocket message format."""

    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)

    @validator("type")
    def validate_type(cls, v):
        """Validate message type to prevent injection."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError("Invalid message type format")
        return v

    @validator("data")
    def validate_data(cls, v):
        """Validate data payload size."""
        # Prevent excessively large payloads
        if len(str(v)) > 100000:  # 100KB limit
            raise ValueError("Message data too large")
        return v


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize connection manager."""
        # Active connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}
        # Subscriptions: event_type -> set of client IDs
        self.subscriptions: Dict[str, Set[str]] = {}
        # Connection metadata
        self.connection_metadata: Dict[str, dict] = {}

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        metadata: Optional[dict] = None,
    ):
        """Accept and register a new WebSocket connection."""
        # Connection is already accepted by the endpoint after authentication
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = metadata or {}
        logger.info(f"WebSocket client {client_id} connected")

        # Send connection acknowledgment with auth info
        await self.send_personal_message(
            {
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "authenticated": True,
                "user": metadata.get("username") if metadata else None,
                "role": metadata.get("role") if metadata else None,
            },
            client_id,
        )

    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]

            # Remove from all subscriptions
            for event_type in self.subscriptions:
                self.subscriptions[event_type].discard(client_id)

            logger.info(f"WebSocket client {client_id} disconnected")

    async def send_personal_message(self, message: dict, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict, event_type: Optional[str] = None):
        """Broadcast a message to all connected clients or subscribers of an event type."""
        if event_type and event_type in self.subscriptions:
            # Send only to subscribers of this event type
            client_ids = list(self.subscriptions[event_type])
        else:
            # Send to all connected clients
            client_ids = list(self.active_connections.keys())

        # Send messages concurrently
        tasks = []
        for client_id in client_ids:
            if client_id in self.active_connections:
                tasks.append(self.send_personal_message(message, client_id))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def subscribe(self, client_id: str, event_type: str):
        """Subscribe a client to an event type."""
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = set()
        self.subscriptions[event_type].add(client_id)
        logger.info(f"Client {client_id} subscribed to {event_type}")

    def unsubscribe(self, client_id: str, event_type: str):
        """Unsubscribe a client from an event type."""
        if event_type in self.subscriptions:
            self.subscriptions[event_type].discard(client_id)
            logger.info(f"Client {client_id} unsubscribed from {event_type}")


# Global connection manager instance
manager = ConnectionManager()

# Import environment for dev mode check
from core.environment import environment


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    token: Optional[str] = Query(None, description="JWT token for authentication"),
):
    """Main WebSocket endpoint for real-time communication with authentication."""
    try:
        # Check rate limiting first
        if not await websocket_rate_limit_manager.check_connection_allowed(websocket):
            await websocket.close(
                code=WebSocketErrorCode.RATE_LIMITED,
                reason="Rate limit exceeded",
            )
            return

        # Authenticate the connection
        user_data = await ws_auth_handler.authenticate_connection(websocket, client_id, token)

        # Accept the connection after successful authentication
        await websocket.accept()

        # Register with rate limiter
        await websocket_rate_limit_manager.register_connection(websocket, client_id)

        # Connect to manager with user metadata
        metadata = {
            "user_id": user_data.user_id,
            "username": user_data.username,
            "role": user_data.role.value,
            "authenticated": True,
            "permissions": [p.value for p in user_data.permissions],
        }
        await manager.connect(websocket, client_id, metadata)

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat_handler(websocket, client_id))

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()

                # Check message rate limiting
                if not await websocket_rate_limit_manager.check_message_allowed(websocket, data):
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": "Message rate limit exceeded",
                        },
                        client_id,
                    )
                    continue

                try:
                    # Validate and parse message
                    message = json.loads(data)

                    # Validate message structure
                    if not isinstance(message, dict) or "type" not in message:
                        raise ValueError("Invalid message structure")

                    # Create validated message
                    validated_msg = WebSocketMessage(
                        type=message.get("type"), data=message.get("data", {})
                    )

                    # Handle special authentication messages
                    if validated_msg.type == "refresh_token":
                        refresh_token = validated_msg.data.get("refresh_token")
                        if refresh_token:
                            result = await handle_token_refresh(client_id, refresh_token)
                            await manager.send_personal_message(result, client_id)
                        else:
                            await manager.send_personal_message(
                                {
                                    "type": "error",
                                    "code": "MISSING_REFRESH_TOKEN",
                                    "message": "Refresh token required",
                                },
                                client_id,
                            )
                    elif validated_msg.type == "heartbeat":
                        # Update heartbeat
                        await ws_auth_handler.update_heartbeat(client_id)
                        await manager.send_personal_message(
                            {
                                "type": "heartbeat_ack",
                                "timestamp": datetime.now().isoformat(),
                            },
                            client_id,
                        )
                    else:
                        # Handle regular messages with permission checks
                        await handle_client_message_with_auth(client_id, message)

                except json.JSONDecodeError:
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "code": "INVALID_JSON",
                            "message": "Invalid JSON format",
                        },
                        client_id,
                    )
                except ValueError as e:
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "code": "VALIDATION_ERROR",
                            "message": str(e),
                        },
                        client_id,
                    )
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "code": "INTERNAL_ERROR",
                            "message": "An error occurred processing your message",
                        },
                        client_id,
                    )

        finally:
            # Cancel heartbeat task
            heartbeat_task.cancel()

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        # Cleanup
        await ws_auth_handler.disconnect(client_id)
        await websocket_rate_limit_manager.unregister_connection(client_id)
        manager.disconnect(client_id)


async def heartbeat_handler(websocket: WebSocket, client_id: str):
    """Send periodic heartbeats to keep connection alive and verify client is responsive."""
    try:
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            await manager.send_personal_message(
                {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "require_response": True,
                },
                client_id,
            )
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Heartbeat error for {client_id}: {e}")


async def handle_client_message_with_auth(client_id: str, message: dict):
    """Handle incoming messages from WebSocket clients with permission checks."""
    msg_type = message.get("type", "unknown")

    if msg_type == "subscribe":
        # Check permission for subscriptions
        if not await ws_auth_handler.verify_permission(client_id, Permission.VIEW_AGENTS):
            await manager.send_personal_message(
                {
                    "type": "error",
                    "code": "PERMISSION_DENIED",
                    "message": "Insufficient permissions for subscription",
                },
                client_id,
            )
            return

        # Handle subscription requests
        event_types = message.get("event_types", [])
        # Validate event types
        valid_event_types = []
        for event_type in event_types:
            if isinstance(event_type, str) and re.match(r"^[a-zA-Z0-9:_-]+$", event_type):
                valid_event_types.append(event_type)
            else:
                logger.warning(f"Invalid event type rejected: {event_type}")

        for event_type in valid_event_types:
            manager.subscribe(client_id, event_type)

        await manager.send_personal_message(
            {
                "type": "subscription_confirmed",
                "event_types": valid_event_types,
            },
            client_id,
        )

    elif msg_type == "unsubscribe":
        # Handle unsubscription requests
        event_types = message.get("event_types", [])
        for event_type in event_types:
            manager.unsubscribe(client_id, event_type)

        await manager.send_personal_message(
            {"type": "unsubscription_confirmed", "event_types": event_types},
            client_id,
        )

    elif msg_type == "ping":
        # Handle ping/pong for connection health check
        await manager.send_personal_message(
            {"type": "pong", "timestamp": datetime.now().isoformat()},
            client_id,
        )

    elif msg_type == "agent_command":
        # Check permission for agent commands
        command_data = message.get("data", {})
        command = command_data.get("command")

        # Determine required permission based on command
        required_permission = Permission.VIEW_AGENTS
        if command in ["start", "stop", "restart", "update"]:
            required_permission = Permission.MODIFY_AGENT
        elif command in ["create", "delete"]:
            required_permission = Permission.CREATE_AGENT

        if not await ws_auth_handler.verify_permission(client_id, required_permission):
            await manager.send_personal_message(
                {
                    "type": "error",
                    "code": "PERMISSION_DENIED",
                    "message": f"Insufficient permissions for command: {command}",
                },
                client_id,
            )
            return

        # Handle agent control commands
        await handle_agent_command(client_id, command_data)

    elif msg_type == "query":
        # Check permission for queries
        if not await ws_auth_handler.verify_permission(client_id, Permission.VIEW_AGENTS):
            await manager.send_personal_message(
                {
                    "type": "error",
                    "code": "PERMISSION_DENIED",
                    "message": "Insufficient permissions for query",
                },
                client_id,
            )
            return

        # Handle real-time queries
        await handle_query(client_id, message.get("data", {}))

    elif msg_type == "prompt_submitted":
        # Handle prompt submission notifications
        prompt_id = message.get("prompt_id")
        prompt_text = message.get("prompt")
        conversation_id = message.get("conversation_id")
        
        logger.info(
            f"Prompt submitted via WebSocket - client: {client_id}, "
            f"prompt_id: {prompt_id}, conversation_id: {conversation_id}"
        )
        
        # Acknowledge receipt
        await manager.send_personal_message(
            {
                "type": "prompt_acknowledged",
                "prompt_id": prompt_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Prompt received and processing started"
            },
            client_id,
        )
        
        # Broadcast to other clients if needed
        await broadcast_agent_event(
            agent_id="system",
            event_type="prompt_processing",
            data={
                "prompt_id": prompt_id,
                "conversation_id": conversation_id,
                "status": "processing"
            }
        )

    elif msg_type == "clear_conversation":
        # Handle conversation clearing
        conversation_data = message.get("data", {})
        conversation_id = conversation_data.get("conversationId")
        
        logger.info(
            f"Clear conversation request - client: {client_id}, "
            f"conversation_id: {conversation_id}"
        )
        
        # Acknowledge clearing
        await manager.send_personal_message(
            {
                "type": "conversation_cleared",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Conversation history cleared"
            },
            client_id,
        )
        
        # Notify other components if needed
        await broadcast_system_event(
            event_type="conversation_cleared",
            data={
                "conversation_id": conversation_id,
                "cleared_by": client_id
            }
        )

    else:
        await manager.send_personal_message(
            {
                "type": "error",
                "code": "UNKNOWN_MESSAGE_TYPE",
                "message": f"Unknown message type: {msg_type}",
            },
            client_id,
        )


async def handle_agent_command(client_id: str, command_data: dict):
    """Handle agent control commands received via WebSocket."""
    command = command_data.get("command")
    agent_id = command_data.get("agent_id")

    # Validate inputs
    if not command or not agent_id:
        await manager.send_personal_message(
            {
                "type": "error",
                "code": "INVALID_COMMAND",
                "message": "Missing command or agent_id",
            },
            client_id,
        )
        return

    # Validate agent_id format to prevent injection
    if not re.match(r"^[a-zA-Z0-9_-]+$", agent_id):
        await manager.send_personal_message(
            {
                "type": "error",
                "code": "INVALID_AGENT_ID",
                "message": "Invalid agent_id format",
            },
            client_id,
        )
        return

    # Validate command
    valid_commands = [
        "start",
        "stop",
        "restart",
        "status",
        "update",
        "create",
        "delete",
    ]
    if command not in valid_commands:
        await manager.send_personal_message(
            {
                "type": "error",
                "code": "INVALID_COMMAND",
                "message": f"Invalid command: {command}",
            },
            client_id,
        )
        return

    # Process command (integrate with agent manager)
    # This is a placeholder - actual implementation would call agent manager
    result = {
        "type": "agent_command_result",
        "agent_id": agent_id,
        "command": command,
        "status": "acknowledged",
        "timestamp": datetime.now().isoformat(),
    }

    await manager.send_personal_message(result, client_id)


async def handle_query(client_id: str, query_data: dict):
    """Handle real-time queries from clients."""
    query_type = query_data.get("query_type")

    if query_type == "agent_status":
        # Return real-time agent status
        result = {
            "type": "query_result",
            "query_type": "agent_status",
            "data": {
                # Placeholder - would fetch actual agent status
                "agents": [],
                "timestamp": datetime.now().isoformat(),
            },
        }
        await manager.send_personal_message(result, client_id)

    elif query_type == "world_state":
        # Return current world state
        result = {
            "type": "query_result",
            "query_type": "world_state",
            "data": {
                # Placeholder - would fetch actual world state
                "world": {},
                "timestamp": datetime.now().isoformat(),
            },
        }
        await manager.send_personal_message(result, client_id)

    else:
        await manager.send_personal_message(
            {"type": "error", "message": f"Unknown query type: {query_type}"},
            client_id,
        )


# Event broadcasting functions for other parts of the system
async def broadcast_agent_event(agent_id: str, event_type: str, data: dict):
    """Broadcast an agent-related event to subscribers."""
    message = {
        "type": "agent_event",
        "event_type": event_type,
        "agent_id": agent_id,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast(message, event_type=f"agent:{event_type}")


async def broadcast_world_event(event_type: str, data: dict):
    """Broadcast a world-related event to subscribers."""
    message = {
        "type": "world_event",
        "event_type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast(message, event_type=f"world:{event_type}")


async def broadcast_system_event(event_type: str, data: dict):
    """Broadcast a system-wide event to all connected clients."""
    message = {
        "type": "system_event",
        "event_type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast(message)


# WebSocket monitoring endpoints
@router.get("/ws/connections")
async def get_active_connections():
    """Get information about active WebSocket connections."""
    # Get enhanced connection info from auth handler
    connections_info = []
    for client_id in manager.active_connections:
        conn_info = ws_auth_handler.get_connection_info(client_id)
        if conn_info:
            # Add subscription info
            conn_info["subscriptions"] = [
                event_type
                for event_type, subscribers in manager.subscriptions.items()
                if client_id in subscribers
            ]
            connections_info.append(conn_info)

    return {
        "total_connections": len(manager.active_connections),
        "authenticated_connections": len([c for c in connections_info if c.get("user_id")]),
        "connections": connections_info,
    }


@router.get("/ws/subscriptions")
async def get_subscriptions():
    """Get information about event subscriptions."""
    return {
        "subscriptions": {
            event_type: list(subscribers)
            for event_type, subscribers in manager.subscriptions.items()
        }
    }


@router.websocket("/ws/dev")
async def websocket_dev_endpoint(websocket: WebSocket):
    """Dev WebSocket endpoint without authentication for development mode.
    
    This endpoint is only available when running in dev mode (no DATABASE_URL).
    It provides basic WebSocket functionality for UI development and testing.
    """
    if not (environment.is_development and not environment.config.auth_required):
        await websocket.close(
            code=4003,
            reason="Dev endpoint only available in dev mode"
        )
        return
    
    # Generate a dev client ID
    client_id = f"dev_{datetime.now().timestamp()}"
    
    try:
        # Accept connection immediately (no auth required)
        await websocket.accept()
        
        # Connect to manager with dev metadata
        await manager.connect(
            websocket,
            client_id,
            metadata={
                "username": "dev_user",
                "role": "dev",
                "dev_mode": True
            }
        )
        
        # Send initial dev data
        await manager.send_personal_message(
            {
                "type": "dev_welcome",
                "message": "Connected to FreeAgentics dev WebSocket",
                "features": [
                    "Agent creation simulation",
                    "Real-time updates",
                    "Knowledge graph visualization"
                ],
                "timestamp": datetime.now().isoformat()
            },
            client_id
        )
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle demo-specific message types
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        },
                        client_id
                    )
                elif message.get("type") == "agent_create":
                    # Simulate agent creation
                    await manager.send_personal_message(
                        {
                            "type": "agent_created",
                            "agent": {
                                "id": f"dev_agent_{datetime.now().timestamp()}",
                                "name": message.get("data", {}).get("name", "Dev Agent"),
                                "type": message.get("data", {}).get("type", "explorer"),
                                "status": "active",
                                "created_at": datetime.now().isoformat()
                            }
                        },
                        client_id
                    )
                    
                    # Broadcast to all dev connections
                    await manager.broadcast(
                        {
                            "type": "agent_update",
                            "action": "created",
                            "agent_id": f"dev_agent_{datetime.now().timestamp()}"
                        }
                    )
                elif message.get("type") == "prompt_submitted":
                    # Handle prompt submission in dev mode
                    prompt_id = message.get("prompt_id")
                    prompt_text = message.get("prompt")
                    conversation_id = message.get("conversation_id")
                    
                    logger.info(
                        f"[Dev] Prompt submitted - prompt_id: {prompt_id}, "
                        f"conversation_id: {conversation_id}"
                    )
                    
                    await manager.send_personal_message(
                        {
                            "type": "prompt_acknowledged",
                            "prompt_id": prompt_id,
                            "conversation_id": conversation_id,
                            "timestamp": datetime.now().isoformat(),
                            "message": "Dev mode: Prompt received"
                        },
                        client_id
                    )
                elif message.get("type") == "clear_conversation":
                    # Handle conversation clearing in dev mode
                    conversation_data = message.get("data", {})
                    conversation_id = conversation_data.get("conversationId")
                    
                    logger.info(f"[Dev] Clear conversation - id: {conversation_id}")
                    
                    await manager.send_personal_message(
                        {
                            "type": "conversation_cleared",
                            "conversation_id": conversation_id,
                            "timestamp": datetime.now().isoformat(),
                            "message": "Dev mode: Conversation cleared"
                        },
                        client_id
                    )
                else:
                    # Echo back for demo purposes
                    await manager.send_personal_message(
                        {
                            "type": "echo",
                            "original": message,
                            "timestamp": datetime.now().isoformat()
                        },
                        client_id
                    )
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Invalid JSON format"
                    },
                    client_id
                )
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Dev WebSocket error: {e}")
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Internal error occurred"
                    },
                    client_id
                )
                
    except Exception as e:
        logger.error(f"Dev WebSocket connection error: {e}")
    finally:
        manager.disconnect(client_id)

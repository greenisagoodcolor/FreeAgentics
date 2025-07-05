"""WebSocket endpoints for real-time agent monitoring and communication."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class WebSocketMessage(BaseModel):
    """Standard WebSocket message format."""

    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: dict = Field(default_factory=dict)


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

    async def connect(self, websocket: WebSocket, client_id: str, metadata: Optional[dict] = None):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = metadata or {}
        logger.info(f"WebSocket client {client_id} connected")

        # Send connection acknowledgment
        await self.send_personal_message(
            {
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
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


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Main WebSocket endpoint for real-time communication."""
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handle_client_message(client_id, message)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON format"}, client_id
                )
            except Exception as e:
                logger.error(f"Error handling message from {client_id}: {e}")
                await manager.send_personal_message({"type": "error", "message": str(e)}, client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)


async def handle_client_message(client_id: str, message: dict):
    """Handle incoming messages from WebSocket clients."""
    msg_type = message.get("type", "unknown")

    if msg_type == "subscribe":
        # Handle subscription requests
        event_types = message.get("event_types", [])
        for event_type in event_types:
            manager.subscribe(client_id, event_type)

        await manager.send_personal_message(
            {"type": "subscription_confirmed", "event_types": event_types}, client_id
        )

    elif msg_type == "unsubscribe":
        # Handle unsubscription requests
        event_types = message.get("event_types", [])
        for event_type in event_types:
            manager.unsubscribe(client_id, event_type)

        await manager.send_personal_message(
            {"type": "unsubscription_confirmed", "event_types": event_types}, client_id
        )

    elif msg_type == "ping":
        # Handle ping/pong for connection health check
        await manager.send_personal_message(
            {"type": "pong", "timestamp": datetime.now().isoformat()}, client_id
        )

    elif msg_type == "agent_command":
        # Handle agent control commands
        await handle_agent_command(client_id, message.get("data", {}))

    elif msg_type == "query":
        # Handle real-time queries
        await handle_query(client_id, message.get("data", {}))

    else:
        await manager.send_personal_message(
            {"type": "error", "message": f"Unknown message type: {msg_type}"}, client_id
        )


async def handle_agent_command(client_id: str, command_data: dict):
    """Handle agent control commands received via WebSocket."""
    command = command_data.get("command")
    agent_id = command_data.get("agent_id")

    if not command or not agent_id:
        await manager.send_personal_message(
            {"type": "error", "message": "Missing command or agent_id"}, client_id
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
            {"type": "error", "message": f"Unknown query type: {query_type}"}, client_id
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
    return {
        "total_connections": len(manager.active_connections),
        "connections": [
            {
                "client_id": client_id,
                "metadata": manager.connection_metadata.get(client_id, {}),
                "subscriptions": [
                    event_type
                    for event_type, subscribers in manager.subscriptions.items()
                    if client_id in subscribers
                ],
            }
            for client_id in manager.active_connections
        ],
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

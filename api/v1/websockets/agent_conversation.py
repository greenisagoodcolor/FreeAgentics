"""
WebSocket handler for Agent Conversations

Real-time WebSocket endpoint for agent conversation updates and status notifications.
Implements TaskMaster task 28.4 requirements.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ConversationWebSocketMessage(BaseModel):
    """Standard WebSocket message format for agent conversations."""

    type: str
    conversation_id: str
    timestamp: str
    data: Dict[str, Any]


class ConversationConnectionManager:
    """Manages WebSocket connections for agent conversation updates."""

    def __init__(self):
        """Initialize connection manager."""
        # Active connections by conversation ID
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        conversation_id: str,
        client_info: Optional[Dict[str, Any]] = None,
    ):
        """Accept and register a new WebSocket connection for a conversation."""

        await websocket.accept()

        # Add to conversation connections
        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = set()
        self.active_connections[conversation_id].add(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            "conversation_id": conversation_id,
            "connected_at": datetime.now(),
            "client_info": client_info or {},
        }

        logger.info(f"WebSocket connected to conversation {conversation_id}")

        # Send connection acknowledgment
        await self.send_to_connection(
            websocket,
            {
                "type": "connection_established",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to agent conversation stream",
            },
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""

        if websocket in self.connection_metadata:
            conversation_id = self.connection_metadata[websocket]["conversation_id"]

            # Remove from conversation connections
            if conversation_id in self.active_connections:
                self.active_connections[conversation_id].discard(websocket)

                # Clean up empty conversation sets
                if not self.active_connections[conversation_id]:
                    del self.active_connections[conversation_id]

            # Remove metadata
            del self.connection_metadata[websocket]

            logger.info(f"WebSocket disconnected from conversation {conversation_id}")

    async def send_to_connection(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a specific WebSocket connection."""

        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast_to_conversation(self, conversation_id: str, message: Dict[str, Any]):
        """Broadcast a message to all connections for a specific conversation."""

        if conversation_id not in self.active_connections:
            logger.debug(f"No active connections for conversation {conversation_id}")
            return

        # Get all connections for this conversation
        connections = list(self.active_connections[conversation_id])

        # Send messages concurrently
        tasks = []
        for websocket in connections:
            tasks.append(self.send_to_connection(websocket, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(
                f"Broadcasted message to {len(tasks)} connections for conversation {conversation_id}"
            )

    async def send_status_update(
        self,
        conversation_id: str,
        status: str,
        progress: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Send a status update to all connections for a conversation."""

        message = {
            "type": "status_update",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "data": {"status": status, "progress": progress, "details": details or {}},
        }

        await self.broadcast_to_conversation(conversation_id, message)

    async def send_agent_message(
        self, conversation_id: str, agent_id: str, agent_name: str, content: str, turn_number: int
    ):
        """Send an agent message to all connections for a conversation."""

        message = {
            "type": "agent_message",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "content": content,
                "turn_number": turn_number,
            },
        }

        await self.broadcast_to_conversation(conversation_id, message)

    async def send_conversation_complete(
        self, conversation_id: str, total_turns: int, total_messages: int
    ):
        """Send conversation completion notification."""

        message = {
            "type": "conversation_complete",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "total_turns": total_turns,
                "total_messages": total_messages,
                "status": "completed",
            },
        }

        await self.broadcast_to_conversation(conversation_id, message)

    async def send_error(
        self,
        conversation_id: str,
        error_code: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Send error notification to all connections for a conversation."""

        message = {
            "type": "error",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "error_code": error_code,
                "error_message": error_message,
                "details": details or {},
            },
        }

        await self.broadcast_to_conversation(conversation_id, message)

    def get_connection_count(self, conversation_id: str) -> int:
        """Get the number of active connections for a conversation."""

        return len(self.active_connections.get(conversation_id, set()))

    def list_active_conversations(self) -> Dict[str, int]:
        """List all active conversations and their connection counts."""

        return {
            conv_id: len(connections) for conv_id, connections in self.active_connections.items()
        }


# Global connection manager instance
conversation_manager = ConversationConnectionManager()


@router.websocket("/ws/agent/{conversation_id}")
async def agent_conversation_websocket(
    websocket: WebSocket,
    conversation_id: str,
    client_id: Optional[str] = Query(None, description="Optional client identifier"),
):
    """
    WebSocket endpoint for real-time agent conversation updates.

    Endpoint: /api/v1/ws/agent/{conversation_id}

    This WebSocket connection provides real-time updates for:
    - Agent initialization status
    - Conversation turn messages
    - Progress updates
    - Error notifications
    - Completion status
    """

    client_info = {
        "client_id": client_id,
        "user_agent": websocket.headers.get("user-agent", "unknown"),
    }

    try:
        # Connect to the conversation stream
        await conversation_manager.connect(websocket, conversation_id, client_info)

        # Send initial status
        await conversation_manager.send_status_update(
            conversation_id=conversation_id,
            status="connected",
            details={"message": "WebSocket connection established"},
        )

        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (heartbeat, requests, etc.)
                data = await websocket.receive_text()

                try:
                    client_message = json.loads(data)
                    await handle_client_message(websocket, conversation_id, client_message)
                except json.JSONDecodeError:
                    await conversation_manager.send_to_connection(
                        websocket,
                        {
                            "type": "error",
                            "conversation_id": conversation_id,
                            "timestamp": datetime.now().isoformat(),
                            "data": {
                                "error_code": "INVALID_JSON",
                                "error_message": "Invalid JSON format in client message",
                            },
                        },
                    )

            except WebSocketDisconnect:
                logger.info(f"Client disconnected from conversation {conversation_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await conversation_manager.send_to_connection(
                    websocket,
                    {
                        "type": "error",
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "data": {"error_code": "MESSAGE_HANDLING_ERROR", "error_message": str(e)},
                    },
                )

    except Exception as e:
        logger.error(f"WebSocket connection error for conversation {conversation_id}: {e}")

    finally:
        # Cleanup connection
        conversation_manager.disconnect(websocket)


async def handle_client_message(
    websocket: WebSocket, conversation_id: str, message: Dict[str, Any]
):
    """Handle incoming messages from WebSocket clients."""

    message_type = message.get("type", "unknown")

    if message_type == "ping":
        # Handle heartbeat ping
        await conversation_manager.send_to_connection(
            websocket,
            {
                "type": "pong",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "data": {"message": "pong"},
            },
        )

    elif message_type == "get_status":
        # Request current conversation status
        # This could integrate with the ConversationService to get real status
        await conversation_manager.send_to_connection(
            websocket,
            {
                "type": "status_response",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "conversation_id": conversation_id,
                    "connection_count": conversation_manager.get_connection_count(conversation_id),
                    "status": "active",  # This would come from ConversationService
                },
            },
        )

    elif message_type == "subscribe_updates":
        # Client wants to subscribe to specific update types
        update_types = message.get("update_types", [])
        await conversation_manager.send_to_connection(
            websocket,
            {
                "type": "subscription_confirmed",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "data": {"subscribed_updates": update_types, "message": "Subscription confirmed"},
            },
        )

    else:
        # Unknown message type
        await conversation_manager.send_to_connection(
            websocket,
            {
                "type": "error",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "error_code": "UNKNOWN_MESSAGE_TYPE",
                    "error_message": f"Unknown message type: {message_type}",
                },
            },
        )


# Integration functions for other services to use


async def notify_conversation_status(
    conversation_id: str,
    status: str,
    progress: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """Notify WebSocket clients about conversation status changes."""

    await conversation_manager.send_status_update(
        conversation_id=conversation_id, status=status, progress=progress, details=details
    )


async def notify_agent_message(
    conversation_id: str, agent_id: str, agent_name: str, content: str, turn_number: int
):
    """Notify WebSocket clients about new agent messages."""

    await conversation_manager.send_agent_message(
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        content=content,
        turn_number=turn_number,
    )


async def notify_conversation_complete(conversation_id: str, total_turns: int, total_messages: int):
    """Notify WebSocket clients that a conversation has completed."""

    await conversation_manager.send_conversation_complete(
        conversation_id=conversation_id, total_turns=total_turns, total_messages=total_messages
    )


async def notify_conversation_error(
    conversation_id: str,
    error_code: str,
    error_message: str,
    details: Optional[Dict[str, Any]] = None,
):
    """Notify WebSocket clients about conversation errors."""

    await conversation_manager.send_error(
        conversation_id=conversation_id,
        error_code=error_code,
        error_message=error_message,
        details=details,
    )


# Health check endpoint for WebSocket connections
@router.get("/ws/agent/health")
async def websocket_health():
    """Health check endpoint for agent conversation WebSockets."""

    active_conversations = conversation_manager.list_active_conversations()
    total_connections = sum(active_conversations.values())

    return {
        "status": "healthy",
        "active_conversations": len(active_conversations),
        "total_connections": total_connections,
        "conversations": active_conversations,
        "timestamp": datetime.now().isoformat(),
    }

"""
Real-Time Conversation Monitoring WebSocket Endpoint

Implements real-time WebSocket monitoring for multi-agent conversation events
following ADR-008 WebSocket patterns and ADR-002 canonical structure.
Provides live conversation updates with clear attribution, metadata, and
thread relationships.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass
class ConversationEvent:
    """Real-time conversation event data structure."""

    # Event types: 'message_created', 'message_updated', 'conversation_started',
    # 'conversation_ended', 'agent_typing', 'agent_stopped_typing'
    type: str
    timestamp: datetime
    conversation_id: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class ConversationSubscription:
    """Client subscription configuration."""

    conversation_ids: set[str]
    agent_ids: set[str]
    message_types: set[str]
    include_typing: bool = True
    include_system_messages: bool = False
    include_metadata: bool = True


class ConversationWebSocketManager:
    """Manages WebSocket connections for real-time conversation monitoring."""

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, dict] = {}
        self.subscriptions: Dict[WebSocket, ConversationSubscription] = {}
        # conversation_id -> agent_id -> timestamp
        self.typing_indicators: Dict[str, Dict[str, datetime]] = {}

    async def connect(
            self,
            websocket: WebSocket,
            client_id: Optional[str] = None):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)

        # Initialize connection metadata
        self.connection_metadata[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.utcnow(),
            "message_count": 0,
            "last_ping": datetime.utcnow(),
        }

        # Initialize empty subscription
        self.subscriptions[websocket] = ConversationSubscription(
            conversation_ids=set(), agent_ids=set(), message_types=set()
        )

        logger.info(
            f"Conversation monitoring client connected: {client_id}, "
            f"total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        """Disconnect and unregister a WebSocket connection."""
        if websocket in self.active_connections:
            client_info = self.connection_metadata.get(websocket, {})
            self.active_connections.remove(websocket)

            # Clean up metadata and subscriptions
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
            if websocket in self.subscriptions:
                del self.subscriptions[websocket]

            logger.info(
                f"Conversation monitoring client disconnected: "
                f"{client_info.get('client_id')}, "
                f"total connections: {len(self.active_connections)}"
            )

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_json(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast_conversation_event(self, event: ConversationEvent):
        """Broadcast a conversation event to subscribed clients."""
        if not self.active_connections:
            return

        disconnected = []
        event_dict = event.to_dict()

        for websocket in self.active_connections:
            try:
                subscription = self.subscriptions.get(websocket)
                if subscription and self._should_send_event(
                        event, subscription):
                    await websocket.send_json(event_dict)
                    if websocket in self.connection_metadata:
                        self.connection_metadata[websocket]["message_count"] += 1
            except Exception as e:
                logger.error(f"Error broadcasting conversation event: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)

    def _should_send_event(
        self, event: ConversationEvent, subscription: ConversationSubscription
    ) -> bool:
        """Determine if an event should be sent to a subscribed client."""
        # Check conversation filter
        if (
            subscription.conversation_ids
            and event.conversation_id not in subscription.conversation_ids
        ):
            return False

        # Check agent filter
        agent_id = event.data.get("senderId") or event.data.get("agentId")
        if subscription.agent_ids and agent_id not in subscription.agent_ids:
            return False

        # Check message type filter
        message_type = event.data.get("type") or event.data.get(
            "metadata", {}).get("type")
        if subscription.message_types and message_type not in subscription.message_types:
            return False

        # Check typing events
        if (
            event.type in ["agent_typing", "agent_stopped_typing"]
            and not subscription.include_typing
        ):
            return False

        # Check system messages
        is_system = event.data.get(
            "metadata", {}).get(
            "isSystemMessage", False)
        if is_system and not subscription.include_system_messages:
            return False

        return True

    async def update_subscription(
            self,
            websocket: WebSocket,
            subscription_data: dict):
        """Update client subscription preferences."""
        if websocket not in self.subscriptions:
            return

        subscription = self.subscriptions[websocket]

        # Update conversation IDs
        if "conversation_ids" in subscription_data:
            subscription.conversation_ids = set(
                subscription_data["conversation_ids"])

        # Update agent IDs
        if "agent_ids" in subscription_data:
            subscription.agent_ids = set(subscription_data["agent_ids"])

        # Update message types
        if "message_types" in subscription_data:
            subscription.message_types = set(
                subscription_data["message_types"])

        # Update flags
        if "include_typing" in subscription_data:
            subscription.include_typing = subscription_data["include_typing"]

        if "include_system_messages" in subscription_data:
            subscription.include_system_messages = subscription_data["include_system_messages"]

        if "include_metadata" in subscription_data:
            subscription.include_metadata = subscription_data["include_metadata"]

    def update_typing_indicator(
            self,
            conversation_id: str,
            agent_id: str,
            is_typing: bool) -> None:
        """Update typing indicator state and broadcast changes."""
        if conversation_id not in self.typing_indicators:
            self.typing_indicators[conversation_id] = {}

        if is_typing:
            self.typing_indicators[conversation_id][agent_id] = datetime.utcnow(
            )
        else:
            self.typing_indicators[conversation_id].pop(agent_id, None)

    def get_typing_agents(self, conversation_id: str) -> List[str]:
        """Get list of agents currently typing in a conversation."""
        if conversation_id not in self.typing_indicators:
            return []

        # Remove stale typing indicators (older than 30 seconds)
        cutoff_time = datetime.utcnow()
        stale_agents = []

        for agent_id, timestamp in self.typing_indicators[conversation_id].items(
        ):
            if (cutoff_time - timestamp).total_seconds() > 30:
                stale_agents.append(agent_id)

        for agent_id in stale_agents:
            del self.typing_indicators[conversation_id][agent_id]

        return list(self.typing_indicators[conversation_id].keys())

    def get_connection_stats(self) -> dict:
        """Get statistics about current connections."""
        active_typing = [
            cid for cid,
            agents in self.typing_indicators.items() if agents]

        empty_sub = ConversationSubscription(set(), set(), set())

        return {
            "total_connections": len(self.active_connections),
            "active_typing_conversations": len(active_typing),
            "connections": [
                {
                    "client_id": metadata["client_id"],
                    "connected_at": metadata["connected_at"].isoformat(),
                    "message_count": metadata["message_count"],
                    "subscribed_conversations": len(
                        self.subscriptions.get(ws, empty_sub).conversation_ids
                    ),
                    "subscribed_agents": len(self.subscriptions.get(ws, empty_sub).agent_ids),
                }
                for ws, metadata in self.connection_metadata.items()
            ],
        }


# Global WebSocket manager
ws_manager = ConversationWebSocketManager()


@router.websocket("/ws/conversations")
async def conversation_websocket_endpoint(
        websocket: WebSocket,
        client_id: Optional[str] = None):
    """
    WebSocket endpoint for real-time conversation monitoring.

    Clients connect to this endpoint to receive real-time updates about:
    - New messages in conversations
    - Agent typing indicators
    - Conversation lifecycle events
    - Message queue status
    """
    await ws_manager.connect(websocket, client_id)

    try:
        # Send initial connection confirmation
        await ws_manager.send_personal_message(
            {
                "type": "connection_established",
                "message": "Connected to conversation monitoring",
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": client_id,
                "supported_events": [
                    "message_created",
                    "message_updated",
                    "message_deleted",
                    "conversation_started",
                    "conversation_ended",
                    "agent_typing",
                    "agent_stopped_typing",
                    "agent_joined",
                    "agent_left",
                    "message_queue_updated",
                ],
            },
            websocket,
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (subscriptions, commands, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)

                await handle_client_message(websocket, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await ws_manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await ws_manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Internal server error",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {e}")
    finally:
        ws_manager.disconnect(websocket)


async def handle_client_message(websocket: WebSocket, message: dict):
    """Handle incoming messages from WebSocket clients."""
    message_type = message.get("type", "")

    if message_type == "ping":
        await ws_manager.send_personal_message(
            {"type": "pong", "timestamp": datetime.utcnow().isoformat()}, websocket
        )

        # Update last ping time
        if websocket in ws_manager.connection_metadata:
            ws_manager.connection_metadata[websocket]["last_ping"] = datetime.utcnow(
            )

    elif message_type == "subscribe":
        await ws_manager.update_subscription(websocket, message.get("subscription", {}))
        await ws_manager.send_personal_message(
            {
                "type": "subscription_updated",
                "timestamp": datetime.utcnow().isoformat(),
                "subscription": message.get("subscription", {}),
            },
            websocket,
        )

    elif message_type == "get_typing_status":
        conversation_id = message.get("conversation_id")
        if conversation_id:
            typing_agents = ws_manager.get_typing_agents(conversation_id)
            await ws_manager.send_personal_message(
                {
                    "type": "typing_status",
                    "conversation_id": conversation_id,
                    "typing_agents": typing_agents,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                websocket,
            )

    elif message_type == "set_typing":
        conversation_id = message.get("conversation_id")
        agent_id = message.get("agent_id")
        is_typing = message.get("is_typing", False)

        if conversation_id and agent_id:
            ws_manager.update_typing_indicator(
                conversation_id, agent_id, is_typing)

            # Broadcast typing change
            event = ConversationEvent(
                type="agent_typing" if is_typing else "agent_stopped_typing",
                timestamp=datetime.utcnow(),
                conversation_id=conversation_id,
                data={"agentId": agent_id, "isTyping": is_typing},
            )
            await ws_manager.broadcast_conversation_event(event)

    elif message_type == "get_stats":
        stats = ws_manager.get_connection_stats()
        await ws_manager.send_personal_message(
            {
                "type": "connection_stats",
                "stats": stats,
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )

    else:
        await ws_manager.send_personal_message(
            {
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )


# API functions for triggering conversation events
# (called by other parts of the system)


async def broadcast_message_created(conversation_id: str, message_data: dict):
    """Broadcast a message creation event."""
    event = ConversationEvent(
        type="message_created",
        timestamp=datetime.utcnow(),
        conversation_id=conversation_id,
        data=message_data,
    )
    await ws_manager.broadcast_conversation_event(event)


async def broadcast_message_updated(conversation_id: str, message_data: dict):
    """Broadcast a message update event."""
    event = ConversationEvent(
        type="message_updated",
        timestamp=datetime.utcnow(),
        conversation_id=conversation_id,
        data=message_data,
    )
    await ws_manager.broadcast_conversation_event(event)


async def broadcast_conversation_started(
    conversation_id: str, participants: List[str], metadata: dict = None
):
    """Broadcast a conversation start event."""
    event = ConversationEvent(
        type="conversation_started",
        timestamp=datetime.utcnow(),
        conversation_id=conversation_id,
        data={"participants": participants, "metadata": metadata or {}},
    )
    await ws_manager.broadcast_conversation_event(event)


async def broadcast_conversation_ended(
        conversation_id: str,
        summary: dict = None):
    """Broadcast a conversation end event."""
    event = ConversationEvent(
        type="conversation_ended",
        timestamp=datetime.utcnow(),
        conversation_id=conversation_id,
        data={"summary": summary or {}},
    )
    await ws_manager.broadcast_conversation_event(event)


async def broadcast_agent_joined(conversation_id: str, agent_id: str):
    """Broadcast an agent joining event."""
    event = ConversationEvent(
        type="agent_joined",
        timestamp=datetime.utcnow(),
        conversation_id=conversation_id,
        data={"agentId": agent_id},
    )
    await ws_manager.broadcast_conversation_event(event)


async def broadcast_agent_left(conversation_id: str, agent_id: str):
    """Broadcast an agent leaving event."""
    event = ConversationEvent(
        type="agent_left",
        timestamp=datetime.utcnow(),
        conversation_id=conversation_id,
        data={"agentId": agent_id},
    )
    await ws_manager.broadcast_conversation_event(event)


async def broadcast_message_queue_updated(
        conversation_id: str, queue_status: dict):
    """Broadcast message queue status update."""
    event = ConversationEvent(
        type="message_queue_updated",
        timestamp=datetime.utcnow(),
        conversation_id=conversation_id,
        data=queue_status,
    )
    await ws_manager.broadcast_conversation_event(event)


# Export the router and manager for use in main application
__all__ = [
    "router",
    "ws_manager",
    "broadcast_message_created",
    "broadcast_message_updated",
    "broadcast_conversation_started",
    "broadcast_conversation_ended",
    "broadcast_agent_joined",
    "broadcast_agent_left",
    "broadcast_message_queue_updated",
]

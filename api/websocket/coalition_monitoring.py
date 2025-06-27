"""
Coalition Monitoring WebSocket Endpoint

Implements real-time WebSocket monitoring for coalition formation events
following ADR-008 WebSocket patterns and ADR-002 canonical structure.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

from coalitions.formation.monitoring_integration import (
    CoalitionFormationMonitor,
    CoalitionMonitoringEvent,
    create_coalition_monitoring_system,
)

logger = logging.getLogger(__name__)

# Global coalition monitoring system
coalition_monitor = create_coalition_monitoring_system()

router = APIRouter()


class CoalitionWebSocketManager:
    """Manages WebSocket connections for coalition monitoring."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.utcnow(),
            "message_count": 0,
        }

        # Note: websocket_adapter integration would go here in full implementation

        logger.info(
            f"Coalition monitoring client connected: {client_id}, "
            f"total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        """Disconnect and unregister a WebSocket connection."""
        if websocket in self.active_connections:
            client_info = self.connection_metadata.get(websocket, {})
            self.active_connections.remove(websocket)
            del self.connection_metadata[websocket]

            # Note: websocket_adapter cleanup would go here in full implementation

            logger.info(
                f"Coalition monitoring client disconnected: {client_info.get('client_id')}, "
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

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["message_count"] += 1
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)

    def get_connection_stats(self) -> dict:
        """Get statistics about current connections."""
        return {
            "total_connections": len(self.active_connections),
            "connections": [
                {
                    "client_id": metadata["client_id"],
                    "connected_at": metadata["connected_at"].isoformat(),
                    "message_count": metadata["message_count"],
                }
                for metadata in self.connection_metadata.values()
            ],
        }


# Global WebSocket manager
ws_manager = CoalitionWebSocketManager()


@router.websocket("/ws/coalitions")
async def coalition_websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None):
    """
    WebSocket endpoint for real-time coalition formation monitoring.

    Clients connect to this endpoint to receive real-time updates about:
    - Coalition formation events
    - Business value calculations
    - Formation progress and results
    """
    await ws_manager.connect(websocket, client_id)

    try:
        # Send initial connection confirmation
        await ws_manager.send_personal_message(
            {
                "type": "connection_established",
                "message": "Connected to coalition formation monitoring",
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": client_id,
            },
            websocket,
        )

        # Send current formation status
        active_formations = coalition_monitor.get_active_formations()
        if active_formations:
            await ws_manager.send_personal_message(
                {
                    "type": "active_formations",
                    "formations": active_formations,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                websocket,
            )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (ping, requests, etc.)
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

    elif message_type == "get_formation_history":
        limit = message.get("limit", 50)
        history = coalition_monitor.get_formation_history(limit)
        await ws_manager.send_personal_message(
            {
                "type": "formation_history",
                "history": history,
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )

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


# Function to trigger coalition formation (for testing/manual use)
async def trigger_coalition_formation(agents_data: List[dict], strategy: str = "active_inference"):
    """

    Trigger a coalition formation process and broadcast events.
    This function can be called from other parts of the system.
    """
    try:
        # This would integrate with the actual agent system
        # For now, we'll broadcast a mock formation event

        formation_event = {
            "type": "coalition_formation_event",
            "event_type": "formation_started",
            "coalition_id": None,
            "timestamp": datetime.utcnow().isoformat(),
            "strategy_used": strategy,
            "participants": [agent.get("id", f"agent_{i}") for i, agent in enumerate(agents_data)],
            "business_value": None,
            "metadata": {
                "formation_id": f'test_formation_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}',
                "agent_count": len(agents_data),
            },
        }

        await ws_manager.broadcast(formation_event)
        logger.info(f"Triggered coalition formation broadcast for {len(agents_data)} agents")

    except Exception as e:
        logger.error(f"Error triggering coalition formation: {e}")


# Export the router and manager for use in main application
__all__ = ["router", "ws_manager", "coalition_monitor", "trigger_coalition_formation"]

"""
Markov Blanket Real-Time Monitoring WebSocket Endpoint

Implements real-time WebSocket monitoring for Markov Blanket boundary violations
and agent state updates following ADR-008 WebSocket patterns and
ADR-002 canonical structure. Integrates with the boundary monitoring service
for comprehensive safety monitoring.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

from agents.base.markov_blanket import BoundaryViolationEvent
from infrastructure.safety.boundary_monitoring_service import (
    BoundaryMonitoringService,
    MonitoringEvent,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass
class MarkovBlanketEvent:
    """Real-time Markov Blanket monitoring event data structure."""

    # Event types: 'boundary_violation', 'state_update', 'agent_registered',
    # 'agent_unregistered', 'monitoring_started', 'monitoring_stopped',
    # 'threshold_breach', 'integrity_update'
    type: str
    timestamp: datetime
    agent_id: str
    data: Dict[str, Any]
    severity: str = "info"  # info, warning, error, critical
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class MarkovBlanketSubscription:
    """Client subscription configuration for Markov Blanket monitoring."""

    agent_ids: Set[str]
    event_types: Set[str]
    severity_levels: Set[str]
    include_mathematical_proofs: bool = False
    include_detailed_metrics: bool = True
    violation_alerts_only: bool = False
    real_time_updates: bool = True


class MarkovBlanketWebSocketManager:
    """Manages WebSocket connections for real-time Markov Blanket monitoring
    ."""

    def __init__(self, boundary_service: BoundaryMonitoringService) -> None:
        self.boundary_service = boundary_service
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self.subscriptions: Dict[WebSocket, MarkovBlanketSubscription] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.monitoring_stats: Dict[str, Any] = {
            "total_events_sent": 0,
            "active_violations": 0,
            "monitored_agents": 0,
            "uptime_start": datetime.utcnow(),
        }

        # Register handlers with boundary service
        # Note: Handlers will be called synchronously from the monitoring
        # service

    async def connect(
            self,
            websocket: WebSocket,
            client_id: Optional[str] = None):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)

        # Initialize connection metadata
        self.connection_metadata[websocket] = {
            "client_id": client_id or f"markov_client_{len(self.active_connections)}",
            "connected_at": datetime.utcnow(),
            "message_count": 0,
            "last_ping": datetime.utcnow(),
            "events_sent": 0,
        }

        # Initialize empty subscription
        self.subscriptions[websocket] = MarkovBlanketSubscription(
            agent_ids=set(),
            event_types=set(),
            severity_levels={"info", "warning", "error", "critical"},
        )

        logger.info(
            f"Markov Blanket monitoring client connected: {client_id}, "
            f"total connections: {len(self.active_connections)}"
        )

        # Send current monitoring status
        await self._send_monitoring_status(websocket)

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
                f"Markov Blanket monitoring client disconnected: "
                f"{client_info.get('client_id')}, "
                f"total connections: {len(self.active_connections)}"
            )

    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_json(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1
                self.connection_metadata[websocket]["events_sent"] += 1
            self.monitoring_stats["total_events_sent"] += 1
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast_markov_event(self, event: MarkovBlanketEvent):
        """Broadcast a Markov Blanket event to subscribed clients."""
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
                        self.connection_metadata[websocket]["events_sent"] += 1
                    self.monitoring_stats["total_events_sent"] += 1
            except Exception as e:
                logger.error(f"Error broadcasting Markov Blanket event: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)

    def _should_send_event(
            self,
            event: MarkovBlanketEvent,
            subscription: MarkovBlanketSubscription) -> bool:
        """Determine if an event should be sent to a subscribed client."""
        # Check agent filter
        if subscription.agent_ids and event.agent_id not in subscription.agent_ids:
            return False

        # Check event type filter
        if subscription.event_types and event.type not in subscription.event_types:
            return False

        # Check severity filter
        if subscription.severity_levels and event.severity not in subscription.severity_levels:
            return False

        # Check violation alerts only mode
        if subscription.violation_alerts_only and event.type != "boundary_violation":
            return False

        return True

    async def update_subscription(
            self,
            websocket: WebSocket,
            subscription_data: Dict):
        """Update client subscription preferences."""
        if websocket not in self.subscriptions:
            return

        subscription = self.subscriptions[websocket]

        # Update agent IDs
        if "agent_ids" in subscription_data:
            subscription.agent_ids = set(subscription_data["agent_ids"])

        # Update event types
        if "event_types" in subscription_data:
            subscription.event_types = set(subscription_data["event_types"])

        # Update severity levels
        if "severity_levels" in subscription_data:
            subscription.severity_levels = set(
                subscription_data["severity_levels"])

        # Update flags
        if "include_mathematical_proofs" in subscription_data:
            subscription.include_mathematical_proofs = subscription_data[
                "include_mathematical_proofs"
            ]

        if "include_detailed_metrics" in subscription_data:
            subscription.include_detailed_metrics = subscription_data["include_detailed_metrics"]

        if "violation_alerts_only" in subscription_data:
            subscription.violation_alerts_only = subscription_data["violation_alerts_only"]

        if "real_time_updates" in subscription_data:
            subscription.real_time_updates = subscription_data["real_time_updates"]

        logger.info(
            f"Updated subscription for client {
                self.connection_metadata.get(
                    websocket,
                    {}).get('client_id')}")

    async def _handle_boundary_violation(
            self, violation: BoundaryViolationEvent):
        """Handle boundary violations from the monitoring service."""
        event = MarkovBlanketEvent(
            type="boundary_violation",
            timestamp=violation.timestamp,
            agent_id=violation.agent_id,
            severity=violation.severity.value.lower(),
            data={
                "violation_type": violation.violation_type,
                "independence_measure": violation.independence_measure,
                "threshold": violation.threshold,
                "mathematical_justification": violation.mathematical_justification,
                "evidence": violation.evidence,
            },
            metadata={
                "violation_id": f"boundary_{
                    violation.agent_id}_{
                    violation.timestamp}",
                "requires_immediate_attention": violation.severity.value in [
                    "HIGH",
                    "CRITICAL"],
            },
        )

        await self.broadcast_markov_event(event)
        self.monitoring_stats["active_violations"] += 1

    async def _handle_monitoring_event(
            self, monitoring_event: MonitoringEvent):
        """Handle general monitoring events from the boundary service."""
        # Map monitoring event to Markov Blanket event
        event_type_mapping = {
            "boundary_check": "state_update",
            "boundary_violation_alert": "boundary_violation",
            "monitoring_error": "monitoring_error",
        }

        event = MarkovBlanketEvent(
            type=event_type_mapping.get(
                monitoring_event.event_type,
                "monitoring_update"),
            timestamp=monitoring_event.timestamp,
            agent_id=monitoring_event.agent_id,
            severity=monitoring_event.severity.value.lower(),
            data=monitoring_event.data,
            metadata={
                "event_id": monitoring_event.event_id,
                "mathematical_evidence": monitoring_event.mathematical_evidence,
                "processed": monitoring_event.processed,
            },
        )

        await self.broadcast_markov_event(event)

    async def _send_monitoring_status(self, websocket: WebSocket):
        """Send current monitoring status to a client."""
        status = self.boundary_service.get_monitoring_status()

        await self.send_personal_message(
            {
                "type": "monitoring_status",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "monitoring_active": status.get("monitoring_active", False),
                    "monitored_agents": status.get("active_agents", []),
                    "total_violations": status.get("total_violations", 0),
                    "system_uptime": status.get("system_uptime", 0),
                    "last_check": status.get("last_check", ""),
                },
            },
            websocket,
        )

    def get_connection_stats(self) -> Dict:
        """Get statistics about current connections and monitoring."""
        uptime = (
            datetime.utcnow() -
            self.monitoring_stats["uptime_start"]).total_seconds()

        return {
            "total_connections": len(self.active_connections),
            "total_events_sent": self.monitoring_stats["total_events_sent"],
            "active_violations": self.monitoring_stats["active_violations"],
            "monitored_agents": len(self.boundary_service.active_agents),
            "system_uptime": uptime,
            "connections": [
                {
                    "client_id": metadata["client_id"],
                    "connected_at": metadata["connected_at"].isoformat(),
                    "events_sent": metadata["events_sent"],
                    "subscribed_agents": len(
                        self.subscriptions.get(
                            ws, MarkovBlanketSubscription(set(), set(), set())
                        ).agent_ids
                    ),
                }
                for ws, metadata in self.connection_metadata.items()
            ],
        }


# Global boundary monitoring service and WebSocket manager
boundary_service = BoundaryMonitoringService()
ws_manager = MarkovBlanketWebSocketManager(boundary_service)


@router.websocket("/ws/markov-blanket")
async def markov_blanket_websocket_endpoint(
        websocket: WebSocket,
        client_id: Optional[str] = None):
    """
    WebSocket endpoint for real-time Markov Blanket monitoring.

    Clients connect to this endpoint to receive real-time updates about:
    - Boundary violations and integrity breaches
    - Agent state changes and belief updates
    - Mathematical proofs and independence measures
    - Monitoring system status and alerts
    """
    await ws_manager.connect(websocket, client_id)

    try:
        # Send initial connection confirmation
        await ws_manager.send_personal_message(
            {
                "type": "connection_established",
                "message": "Connected to Markov Blanket monitoring",
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": client_id,
                "supported_events": [
                    "boundary_violation",
                    "state_update",
                    "agent_registered",
                    "agent_unregistered",
                    "monitoring_started",
                    "monitoring_stopped",
                    "threshold_breach",
                    "integrity_update",
                    "monitoring_error",
                ],
                "supported_severities": ["info", "warning", "error", "critical"],
            },
            websocket,
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (ping, subscription updates, etc.)
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
        logger.error(
            f"Unexpected error in Markov Blanket WebSocket connection: {e}")
    finally:
        ws_manager.disconnect(websocket)


class MessageHandler:
    """Base class for WebSocket message handlers using Command pattern"""

    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        raise NotImplementedError


class PingHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        await ws_manager.send_personal_message(
            {"type": "pong", "timestamp": datetime.utcnow().isoformat()}, websocket)


class SubscribeHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        subscription_data = message.get("subscription", {})
        await ws_manager.update_subscription(websocket, subscription_data)
        await ws_manager.send_personal_message(
            {
                "type": "subscription_updated",
                "subscription": subscription_data,
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )


class MonitoringStatusHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        await ws_manager._send_monitoring_status(websocket)


class AgentViolationsHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        agent_id = message.get("agent_id")
        if agent_id:
            violations = boundary_service.get_agent_violations(agent_id)
            await ws_manager.send_personal_message(
                {
                    "type": "agent_violations",
                    "agent_id": agent_id,
                    "violations": [violation.__dict__ for violation in violations],
                    "timestamp": datetime.utcnow().isoformat(),
                },
                websocket,
            )


class RegisterAgentHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        agent_id = message.get("agent_id")
        if agent_id:
            boundary_service.register_agent(agent_id)
            event = MarkovBlanketEvent(
                type="agent_registered",
                timestamp=datetime.utcnow(),
                agent_id=agent_id,
                data={
                    "message": f"Agent {agent_id} registered for monitoring"},
                severity="info",
            )
            await ws_manager.broadcast_markov_event(event)


class UnregisterAgentHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        agent_id = message.get("agent_id")
        if agent_id:
            boundary_service.unregister_agent(agent_id)
            event = MarkovBlanketEvent(
                type="agent_unregistered",
                timestamp=datetime.utcnow(),
                agent_id=agent_id,
                data={
                    "message": f"Agent {agent_id} unregistered from monitoring"},
                severity="info",
            )
            await ws_manager.broadcast_markov_event(event)


class StartMonitoringHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        if not boundary_service.monitoring_active:
            await boundary_service.start_monitoring()
            event = MarkovBlanketEvent(
                type="monitoring_started",
                timestamp=datetime.utcnow(),
                agent_id="system",
                data={"message": "Boundary monitoring service started"},
                severity="info",
            )
            await ws_manager.broadcast_markov_event(event)


class StopMonitoringHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        if boundary_service.monitoring_active:
            await boundary_service.stop_monitoring()
            event = MarkovBlanketEvent(
                type="monitoring_stopped",
                timestamp=datetime.utcnow(),
                agent_id="system",
                data={"message": "Boundary monitoring service stopped"},
                severity="warning",
            )
            await ws_manager.broadcast_markov_event(event)


class StatsHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        stats = ws_manager.get_connection_stats()
        await ws_manager.send_personal_message(
            {
                "type": "connection_stats",
                "stats": stats,
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )


class ComplianceReportHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        agent_id = message.get("agent_id")
        report = boundary_service.export_compliance_report(agent_id)
        await ws_manager.send_personal_message(
            {
                "type": "compliance_report",
                "agent_id": agent_id,
                "report": report,
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )


class UnknownMessageHandler(MessageHandler):
    async def handle(self, websocket: WebSocket, message: Dict) -> None:
        message_type = message.get("type", "")
        await ws_manager.send_personal_message(
            {
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )


# Message handler registry using Command pattern
MESSAGE_HANDLERS = {
    "ping": PingHandler(),
    "subscribe": SubscribeHandler(),
    "get_monitoring_status": MonitoringStatusHandler(),
    "get_agent_violations": AgentViolationsHandler(),
    "register_agent": RegisterAgentHandler(),
    "unregister_agent": UnregisterAgentHandler(),
    "start_monitoring": StartMonitoringHandler(),
    "stop_monitoring": StopMonitoringHandler(),
    "get_stats": StatsHandler(),
    "get_compliance_report": ComplianceReportHandler(),
}


async def handle_client_message(websocket: WebSocket, message: Dict):
    """Handle incoming messages from WebSocket clients using Command pattern"""
    message_type = message.get("type", "")
    handler = MESSAGE_HANDLERS.get(message_type, UnknownMessageHandler())
    await handler.handle(websocket, message)


# Public API functions for broadcasting events
async def broadcast_boundary_violation(agent_id: str, violation_data: Dict):
    """Broadcast a boundary violation event to all connected clients."""
    event = MarkovBlanketEvent(
        type="boundary_violation",
        timestamp=datetime.utcnow(),
        agent_id=agent_id,
        severity="error",
        data=violation_data,
    )
    await ws_manager.broadcast_markov_event(event)


async def broadcast_state_update(agent_id: str, state_data: Dict):
    """Broadcast an agent state update to all connected clients."""
    event = MarkovBlanketEvent(
        type="state_update",
        timestamp=datetime.utcnow(),
        agent_id=agent_id,
        severity="info",
        data=state_data,
    )
    await ws_manager.broadcast_markov_event(event)


async def broadcast_integrity_update(agent_id: str, integrity_data: Dict):
    """Broadcast a boundary integrity update to all connected clients."""
    event = MarkovBlanketEvent(
        type="integrity_update",
        timestamp=datetime.utcnow(),
        agent_id=agent_id,
        severity="info" if integrity_data.get(
            "boundary_integrity",
            0) > 0.8 else "warning",
        data=integrity_data,
    )
    await ws_manager.broadcast_markov_event(event)


# Initialize the monitoring service
async def initialize_markov_blanket_monitoring():
    """Initialize the Markov Blanket monitoring system."""
    if not boundary_service.monitoring_active:
        await boundary_service.start_monitoring()
        logger.info("Markov Blanket monitoring system initialized")


# Cleanup function
async def cleanup_markov_blanket_monitoring():
    """Cleanup the Markov Blanket monitoring system."""
    if boundary_service.monitoring_active:
        await boundary_service.stop_monitoring()
        logger.info("Markov Blanket monitoring system cleaned up")

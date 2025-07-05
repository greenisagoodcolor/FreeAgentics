"""Example WebSocket client for FreeAgentics real-time monitoring."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FreeAgenticsWebSocketClient:
    """WebSocket client for FreeAgentics real-time communication."""

    def __init__(self, base_url: str = "ws://localhost:8000", client_id: str = "example_client"):
        """Initialize WebSocket client."""
        self.base_url = base_url
        self.client_id = client_id
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.monitoring_websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            # Connect to main WebSocket endpoint
            ws_url = f"{self.base_url}/api/v1/ws/{self.client_id}"
            self.websocket = await websockets.connect(ws_url)
            logger.info(f"Connected to WebSocket: {ws_url}")

            # Connect to monitoring WebSocket endpoint
            monitor_url = f"{self.base_url}/api/v1/ws/monitor/{self.client_id}"
            self.monitoring_websocket = await websockets.connect(monitor_url)
            logger.info(f"Connected to monitoring WebSocket: {monitor_url}")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from main WebSocket")

        if self.monitoring_websocket:
            await self.monitoring_websocket.close()
            logger.info("Disconnected from monitoring WebSocket")

    async def subscribe_to_events(self, event_types: list):
        """Subscribe to specific event types."""
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket")

        message = {"type": "subscribe", "event_types": event_types}

        await self.websocket.send(json.dumps(message))
        logger.info(f"Subscribed to events: {event_types}")

    async def start_monitoring(self, metrics: list, agents: list = None, sample_rate: float = 1.0):
        """Start real-time monitoring."""
        if not self.monitoring_websocket:
            raise RuntimeError("Not connected to monitoring WebSocket")

        config = {
            "metrics": metrics,
            "agents": agents or [],
            "sample_rate": sample_rate,
        }

        message = {"type": "start_monitoring", "config": config}

        await self.monitoring_websocket.send(json.dumps(message))
        logger.info(f"Started monitoring: {metrics}")

    async def send_agent_command(self, agent_id: str, command: str, params: dict = None):
        """Send a command to an agent."""
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket")

        message = {
            "type": "agent_command",
            "data": {"agent_id": agent_id, "command": command, "params": params or {}},
        }

        await self.websocket.send(json.dumps(message))
        logger.info(f"Sent command '{command}' to agent {agent_id}")

    async def query_agent_status(self):
        """Query current agent status."""
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket")

        message = {"type": "query", "data": {"query_type": "agent_status"}}

        await self.websocket.send(json.dumps(message))
        logger.info("Queried agent status")

    async def listen_for_messages(self):
        """Listen for messages from the WebSocket server."""
        try:
            # Listen on both WebSocket connections
            while True:
                # Check main WebSocket
                if self.websocket:
                    try:
                        message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                        data = json.loads(message)
                        await self.handle_message(data, "main")
                    except asyncio.TimeoutError:
                        pass
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Main WebSocket connection closed")
                        self.websocket = None

                # Check monitoring WebSocket
                if self.monitoring_websocket:
                    try:
                        message = await asyncio.wait_for(
                            self.monitoring_websocket.recv(), timeout=0.1
                        )
                        data = json.loads(message)
                        await self.handle_message(data, "monitoring")
                    except asyncio.TimeoutError:
                        pass
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Monitoring WebSocket connection closed")
                        self.monitoring_websocket = None

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in message listener: {e}")

    async def handle_message(self, message: dict, source: str):
        """Handle incoming WebSocket messages."""
        msg_type = message.get("type", "unknown")
        timestamp = message.get("timestamp", datetime.now().isoformat())

        if msg_type == "connection_established":
            logger.info(f"[{source}] Connection established: {message.get('client_id')}")

        elif msg_type == "subscription_confirmed":
            logger.info(f"[{source}] Subscription confirmed: {message.get('event_types')}")

        elif msg_type == "agent_event":
            event_type = message.get("event_type")
            agent_id = message.get("agent_id")
            data = message.get("data", {})
            logger.info(f"[{source}] Agent event - {event_type} from {agent_id}: {data}")

        elif msg_type == "world_event":
            event_type = message.get("event_type")
            data = message.get("data", {})
            logger.info(f"[{source}] World event - {event_type}: {data}")

        elif msg_type == "system_event":
            event_type = message.get("event_type")
            data = message.get("data", {})
            logger.info(f"[{source}] System event - {event_type}: {data}")

        elif msg_type == "metrics_update":
            metrics = message.get("metrics", {})
            counters = message.get("counters", {})

            # Log key metrics
            if metrics:
                logger.info(f"[{source}] Metrics update:")
                for metric_type, value in metrics.items():
                    if isinstance(value, dict):
                        logger.info(f"  {metric_type}: {value.get('value', 'N/A')}")
                    else:
                        logger.info(f"  {metric_type}: {value}")

            if counters:
                logger.info(f"[{source}] Counters: {counters}")

        elif msg_type == "monitoring_started":
            session_id = message.get("session_id")
            logger.info(f"[{source}] Monitoring started - session: {session_id}")

        elif msg_type == "error":
            error_msg = message.get("message", "Unknown error")
            logger.error(f"[{source}] Error: {error_msg}")

        else:
            logger.info(f"[{source}] {msg_type}: {message}")

    async def run_example(self):
        """Run example WebSocket client interactions."""
        try:
            # Connect to server
            await self.connect()

            # Subscribe to events
            await self.subscribe_to_events(
                [
                    "agent:created",
                    "agent:started",
                    "agent:stopped",
                    "agent:action",
                    "world:updated",
                ]
            )

            # Start monitoring
            await self.start_monitoring(
                metrics=["cpu_usage", "memory_usage", "inference_rate", "agent_count"],
                sample_rate=2.0,  # Every 2 seconds
            )

            # Query agent status
            await self.query_agent_status()

            # Send some example agent commands
            await asyncio.sleep(2)
            await self.send_agent_command("agent_1", "start")

            await asyncio.sleep(2)
            await self.send_agent_command("agent_1", "move", {"direction": "up"})

            # Listen for messages
            logger.info("Listening for messages... (Press Ctrl+C to stop)")
            await self.listen_for_messages()

        finally:
            await self.disconnect()


async def main():
    """Main entry point."""
    client = FreeAgenticsWebSocketClient()
    await client.run_example()


if __name__ == "__main__":
    asyncio.run(main())

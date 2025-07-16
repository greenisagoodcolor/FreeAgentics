"""
Secure WebSocket Client Example

Demonstrates proper WebSocket authentication, token refresh, heartbeat handling,
and secure message exchange with the FreeAgentics WebSocket API.
"""

import asyncio
import json
import logging
import ssl
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

import websockets
from websockets.exceptions import WebSocketException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureWebSocketClient:
    """Secure WebSocket client with authentication and automatic reconnection."""

    def __init__(self, base_url: str, access_token: str, refresh_token: Optional[str] = None):
        """
        Initialize the secure WebSocket client.

        Args:
            base_url: WebSocket server URL (e.g., "ws://localhost:8000")
            access_token: JWT access token for authentication
            refresh_token: Optional refresh token for token renewal
        """
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.client_id = f"client_{datetime.now().timestamp()}"
        self.websocket = None
        self.running = False
        self.heartbeat_task = None
        self.receive_task = None
        self.last_heartbeat_ack = datetime.now()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds

    async def connect(self):
        """Establish WebSocket connection with authentication."""
        try:
            # Build URL with token in query parameters
            params = urlencode({"token": self.access_token})
            url = f"{self.base_url}/api/v1/ws/{self.client_id}?{params}"

            logger.info(f"Connecting to WebSocket at {self.base_url}")

            # Create SSL context if using wss://
            ssl_context = None
            if url.startswith("wss://"):
                ssl_context = ssl.create_default_context()
                # For development, you might need to disable cert verification
                # ssl_context.check_hostname = False
                # ssl_context.verify_mode = ssl.CERT_NONE

            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(url, ssl=ssl_context), timeout=10.0
            )

            logger.info("WebSocket connection established")
            self.running = True
            self.reconnect_attempts = 0

            # Start heartbeat and receive tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.receive_task = asyncio.create_task(self._receive_loop())

        except asyncio.TimeoutError:
            logger.error("Connection timeout")
            raise
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    async def disconnect(self):
        """Gracefully disconnect from WebSocket."""
        self.running = False

        # Cancel tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.receive_task:
            self.receive_task.cancel()

        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        logger.info("Disconnected from WebSocket")

    async def send_message(self, message_type: str, data: dict = None):
        """
        Send a message to the WebSocket server.

        Args:
            message_type: Type of message to send
            data: Optional data payload
        """
        if not self.websocket or not self.running:
            raise RuntimeError("Not connected to WebSocket")

        message = {
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "data": data or {},
        }

        try:
            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent message: {message_type}")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    async def subscribe_to_events(self, event_types: list):
        """Subscribe to specific event types."""
        await self.send_message("subscribe", {"event_types": event_types})

    async def send_agent_command(self, agent_id: str, command: str, params: dict = None):
        """Send a command to control an agent."""
        await self.send_message(
            "agent_command", {"agent_id": agent_id, "command": command, "params": params or {}}
        )

    async def query_agent_status(self):
        """Query the status of all agents."""
        await self.send_message("query", {"query_type": "agent_status"})

    async def refresh_auth_token(self):
        """Refresh the authentication token."""
        if not self.refresh_token:
            logger.warning("No refresh token available")
            return False

        await self.send_message("refresh_token", {"refresh_token": self.refresh_token})
        return True

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to keep connection alive."""
        try:
            while self.running:
                # Send heartbeat
                await self.send_message("heartbeat")

                # Wait for heartbeat interval
                await asyncio.sleep(30)

                # Check if we received acknowledgment
                if (datetime.now() - self.last_heartbeat_ack).total_seconds() > 60:
                    logger.warning("Heartbeat timeout - no acknowledgment received")
                    await self._handle_connection_loss()
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            await self._handle_connection_loss()

    async def _receive_loop(self):
        """Receive and process messages from the WebSocket server."""
        try:
            while self.running:
                message = await self.websocket.recv()
                await self._handle_message(json.loads(message))

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
            await self._handle_connection_loss()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive error: {e}")
            await self._handle_connection_loss()

    async def _handle_message(self, message: dict):
        """Handle incoming messages from the server."""
        msg_type = message.get("type")

        if msg_type == "connection_established":
            logger.info(f"Connected as {message.get('user')} with role {message.get('role')}")

        elif msg_type == "heartbeat_ack":
            self.last_heartbeat_ack = datetime.now()
            logger.debug("Heartbeat acknowledged")

        elif msg_type == "heartbeat":
            # Server-initiated heartbeat, respond with heartbeat message
            await self.send_message("heartbeat")

        elif msg_type == "token_refreshed":
            self.access_token = message.get("access_token")
            logger.info("Authentication token refreshed")

        elif msg_type == "subscription_confirmed":
            logger.info(f"Subscribed to events: {message.get('event_types')}")

        elif msg_type == "agent_event":
            await self._handle_agent_event(message)

        elif msg_type == "error":
            code = message.get("code", "UNKNOWN")
            error_msg = message.get("message", "Unknown error")
            logger.error(f"Server error [{code}]: {error_msg}")

            # Handle specific error codes
            if code == "TOKEN_EXPIRED":
                logger.info("Token expired, attempting refresh")
                await self.refresh_auth_token()
            elif code == "RATE_LIMIT_EXCEEDED":
                logger.warning("Rate limit exceeded, slowing down")
                await asyncio.sleep(5)

        else:
            logger.info(f"Received message: {msg_type}")

    async def _handle_agent_event(self, message: dict):
        """Handle agent-related events."""
        event_type = message.get("event_type")
        agent_id = message.get("agent_id")
        data = message.get("data", {})

        logger.info(f"Agent event: {event_type} for {agent_id}")
        # Process agent events as needed

    async def _handle_connection_loss(self):
        """Handle loss of WebSocket connection."""
        self.running = False

        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(
                f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}"
            )

            # Wait before reconnecting
            await asyncio.sleep(self.reconnect_delay * self.reconnect_attempts)

            try:
                await self.connect()
                logger.info("Reconnection successful")
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                await self._handle_connection_loss()
        else:
            logger.error("Maximum reconnection attempts reached")
            await self.disconnect()

    async def run_interactive(self):
        """Run an interactive session."""
        print("\nSecure WebSocket Client")
        print("Commands:")
        print("  subscribe <event_types> - Subscribe to events (comma-separated)")
        print("  agent <id> <command> - Send agent command")
        print("  status - Query agent status")
        print("  refresh - Refresh authentication token")
        print("  quit - Disconnect and exit")
        print()

        try:
            while self.running:
                try:
                    # Get user input with timeout to allow message processing
                    command = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, input, "> "), timeout=1.0
                    )

                    if command.startswith("subscribe "):
                        event_types = command[10:].split(",")
                        await self.subscribe_to_events([e.strip() for e in event_types])

                    elif command.startswith("agent "):
                        parts = command[6:].split()
                        if len(parts) >= 2:
                            agent_id, cmd = parts[0], parts[1]
                            await self.send_agent_command(agent_id, cmd)

                    elif command == "status":
                        await self.query_agent_status()

                    elif command == "refresh":
                        await self.refresh_auth_token()

                    elif command == "quit":
                        break

                    else:
                        print(f"Unknown command: {command}")

                except asyncio.TimeoutError:
                    # Timeout is normal, allows message processing
                    continue
                except EOFError:
                    # Handle Ctrl+D
                    break

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            await self.disconnect()


async def main():
    """Main function demonstrating secure WebSocket usage."""
    # Example configuration
    WS_URL = "ws://localhost:8000"

    # In a real application, you would obtain these tokens through proper authentication
    # For testing, you might get them from your auth endpoint first
    ACCESS_TOKEN = "your_access_token_here"
    REFRESH_TOKEN = "your_refresh_token_here"

    # Create secure client
    client = SecureWebSocketClient(WS_URL, ACCESS_TOKEN, REFRESH_TOKEN)

    try:
        # Connect to WebSocket
        await client.connect()

        # Subscribe to events
        await client.subscribe_to_events(
            ["agent:status_change", "agent:task_complete", "world:state_change"]
        )

        # Run interactive session
        await client.run_interactive()

    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

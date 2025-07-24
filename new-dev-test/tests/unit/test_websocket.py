"""
Test suite for WebSocket functionality.

Tests the WebSocket endpoints, connection management, and event broadcasting.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.v1.websocket import ConnectionManager, WebSocketMessage, broadcast_agent_event
from fastapi import WebSocket
from fastapi.testclient import TestClient


class TestConnectionManager:
    """Test the ConnectionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a ConnectionManager instance."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_json = AsyncMock()
        return websocket

    @pytest.mark.asyncio
    async def test_connect(self, manager, mock_websocket):
        """Test WebSocket connection."""
        client_id = "test_client"
        metadata = {"user": "test_user"}

        await manager.connect(mock_websocket, client_id, metadata)

        # Check connection was established
        assert client_id in manager.active_connections
        assert manager.active_connections[client_id] == mock_websocket
        assert manager.connection_metadata[client_id] == metadata

        # Check accept was called
        mock_websocket.accept.assert_called_once()

        # Check connection acknowledgment was sent
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "connection_established"
        assert call_args["client_id"] == client_id

    def test_disconnect(self, manager):
        """Test WebSocket disconnection."""
        client_id = "test_client"
        manager.active_connections[client_id] = MagicMock()
        manager.connection_metadata[client_id] = {}
        manager.subscriptions["test_event"] = {client_id}

        manager.disconnect(client_id)

        # Check connection was removed
        assert client_id not in manager.active_connections
        assert client_id not in manager.connection_metadata
        assert client_id not in manager.subscriptions["test_event"]

    @pytest.mark.asyncio
    async def test_send_personal_message(self, manager, mock_websocket):
        """Test sending message to specific client."""
        client_id = "test_client"
        manager.active_connections[client_id] = mock_websocket

        message = {"type": "test", "data": "hello"}
        await manager.send_personal_message(message, client_id)

        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_personal_message_error_handling(self, manager, mock_websocket):
        """Test error handling when sending message fails."""
        client_id = "test_client"
        mock_websocket.send_json.side_effect = Exception("Connection error")
        manager.active_connections[client_id] = mock_websocket
        manager.connection_metadata[client_id] = {}

        message = {"type": "test"}
        await manager.send_personal_message(message, client_id)

        # Check client was disconnected on error
        assert client_id not in manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, manager):
        """Test broadcasting to all clients."""
        # Setup multiple clients
        clients = {}
        for i in range(3):
            client_id = f"client_{i}"
            websocket = AsyncMock()
            manager.active_connections[client_id] = websocket
            clients[client_id] = websocket

        message = {"type": "broadcast", "data": "announcement"}
        await manager.broadcast(message)

        # Check all clients received the message
        for websocket in clients.values():
            websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(self, manager):
        """Test broadcasting to event subscribers only."""
        # Setup clients with different subscriptions
        event_type = "agent:created"

        # Subscribed clients
        for i in range(2):
            client_id = f"subscribed_{i}"
            websocket = AsyncMock()
            manager.active_connections[client_id] = websocket
            manager.subscribe(client_id, event_type)

        # Non-subscribed client
        other_client = "other_client"
        other_websocket = AsyncMock()
        manager.active_connections[other_client] = other_websocket

        message = {"type": "event", "event_type": event_type}
        await manager.broadcast(message, event_type=event_type)

        # Check only subscribers received the message
        for client_id in [f"subscribed_{i}" for i in range(2)]:
            manager.active_connections[client_id].send_json.assert_called_once_with(message)

        # Check non-subscriber didn't receive it
        other_websocket.send_json.assert_not_called()

    def test_subscribe(self, manager):
        """Test event subscription."""
        client_id = "test_client"
        event_type = "agent:action"

        manager.subscribe(client_id, event_type)

        assert event_type in manager.subscriptions
        assert client_id in manager.subscriptions[event_type]

    def test_unsubscribe(self, manager):
        """Test event unsubscription."""
        client_id = "test_client"
        event_type = "agent:action"
        manager.subscriptions[event_type] = {client_id, "other_client"}

        manager.unsubscribe(client_id, event_type)

        assert client_id not in manager.subscriptions[event_type]
        assert "other_client" in manager.subscriptions[event_type]


class TestWebSocketMessage:
    """Test WebSocketMessage model."""

    def test_message_creation(self):
        """Test creating WebSocket message."""
        msg = WebSocketMessage(type="test", data={"key": "value"})

        assert msg.type == "test"
        assert msg.data == {"key": "value"}
        assert msg.timestamp is not None

    def test_message_defaults(self):
        """Test message default values."""
        msg = WebSocketMessage(type="test")

        assert msg.data == {}
        assert msg.timestamp is not None


class TestWebSocketEndpoint:
    """Test WebSocket endpoint integration."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test establishing WebSocket connection."""
        from main import app

        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/test_client") as websocket:
                # Should receive connection acknowledgment
                data = websocket.receive_json()
                assert data["type"] == "connection_established"
                assert data["client_id"] == "test_client"

    @pytest.mark.asyncio
    async def test_websocket_subscription(self):
        """Test event subscription via WebSocket."""
        from main import app

        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/test_client") as websocket:
                # Skip connection message
                websocket.receive_json()

                # Send subscription request
                websocket.send_json(
                    {
                        "type": "subscribe",
                        "event_types": ["agent:created", "agent:started"],
                    }
                )

                # Should receive confirmation
                data = websocket.receive_json()
                assert data["type"] == "subscription_confirmed"
                assert data["event_types"] == [
                    "agent:created",
                    "agent:started",
                ]

    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self):
        """Test ping/pong for connection health."""
        from main import app

        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/test_client") as websocket:
                # Skip connection message
                websocket.receive_json()

                # Send ping
                websocket.send_json({"type": "ping"})

                # Should receive pong
                data = websocket.receive_json()
                assert data["type"] == "pong"
                assert "timestamp" in data


class TestEventBroadcasting:
    """Test event broadcasting functions."""

    @pytest.mark.asyncio
    async def test_broadcast_agent_event(self):
        """Test broadcasting agent events."""
        from api.v1.websocket import manager

        # Mock the manager broadcast method
        with patch.object(manager, "broadcast", new_callable=AsyncMock) as mock_broadcast:
            await broadcast_agent_event("agent_123", "created", {"name": "TestAgent"})

            # Check broadcast was called correctly
            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args[0][0]
            assert call_args["type"] == "agent_event"
            assert call_args["event_type"] == "created"
            assert call_args["agent_id"] == "agent_123"
            assert call_args["data"]["name"] == "TestAgent"

            # Check event type filtering
            assert mock_broadcast.call_args[1]["event_type"] == "agent:created"


class TestWebSocketMonitoring:
    """Test WebSocket monitoring endpoints."""

    def test_get_active_connections(self):
        """Test getting active connections info."""
        from main import app

        with TestClient(app) as client:
            response = client.get("/api/v1/ws/connections")

            assert response.status_code == 200
            data = response.json()
            assert "total_connections" in data
            assert "connections" in data
            assert isinstance(data["connections"], list)

    def test_get_subscriptions(self):
        """Test getting subscription info."""
        from main import app

        with TestClient(app) as client:
            response = client.get("/api/v1/ws/subscriptions")

            assert response.status_code == 200
            data = response.json()
            assert "subscriptions" in data
            assert isinstance(data["subscriptions"], dict)

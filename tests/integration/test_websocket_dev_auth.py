"""Integration tests for WebSocket dev mode authentication."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app


@pytest.mark.asyncio
class TestWebSocketDevAuth:
    """Test WebSocket authentication in dev mode."""

    @patch('core.environment.environment.is_development', True)
    @patch('core.environment.environment.config.auth_required', False)
    def test_websocket_dev_endpoint_accepts_dev_token(self):
        """Test that dev WebSocket endpoint accepts 'dev' token."""
        client = TestClient(app)
        
        with client.websocket_connect("/api/v1/ws/dev?token=dev") as websocket:
            # Should connect successfully
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            assert data["authenticated"] is True
            assert data["user"] == "dev_user"
            assert data["role"] == "admin"
            
            # Test ping/pong
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"

    @patch('core.environment.environment.is_development', True)
    @patch('core.environment.environment.config.auth_required', False)
    def test_websocket_dev_endpoint_accepts_no_token(self):
        """Test that dev WebSocket endpoint accepts connection without token."""
        client = TestClient(app)
        
        with client.websocket_connect("/api/v1/ws/dev") as websocket:
            # Should connect successfully even without token
            data = websocket.receive_json()
            assert data["type"] == "connection_established"

    @patch('core.environment.environment.is_development', False)
    @patch('core.environment.environment.config.auth_required', True)
    def test_websocket_dev_endpoint_rejects_in_production(self):
        """Test that dev WebSocket endpoint is disabled in production."""
        client = TestClient(app)
        
        with pytest.raises(Exception) as exc_info:
            with client.websocket_connect("/api/v1/ws/dev?token=dev"):
                pass
        
        # Should get a 403 or similar error
        assert "403" in str(exc_info.value) or "4003" in str(exc_info.value)

    @patch('core.environment.environment.is_development', True)
    @patch('core.environment.environment.config.auth_required', False)
    def test_websocket_regular_endpoint_with_dev_token(self):
        """Test that regular WebSocket endpoint accepts dev token in dev mode."""
        client = TestClient(app)
        
        with client.websocket_connect("/api/v1/ws/test-client?token=dev") as websocket:
            # Should accept the connection with dev token
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"

    @patch('core.environment.environment.is_development', True)
    @patch('core.environment.environment.config.auth_required', False)
    def test_websocket_broadcasts_work_in_dev_mode(self):
        """Test that WebSocket broadcasts work in dev mode."""
        client = TestClient(app)
        
        with client.websocket_connect("/api/v1/ws/dev?token=dev") as websocket:
            # Clear initial connection message
            websocket.receive_json()
            
            # Subscribe to agent events
            websocket.send_json({
                "type": "subscribe",
                "event_types": ["agent:created"]
            })
            
            # Should receive subscription confirmation
            response = websocket.receive_json()
            assert response["type"] == "subscription_confirmed"
            assert "agent:created" in response["event_types"]
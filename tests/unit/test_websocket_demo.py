"""Test demo WebSocket endpoint.

Following TDD principles from CLAUDE.md:
1. Write failing test first (RED)
2. Implement minimal code to pass (GREEN)
3. Refactor if needed
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app


class TestWebSocketDemo:
    """Test demo WebSocket endpoint functionality."""

    def test_demo_websocket_available_in_demo_mode(self):
        """Test that demo WebSocket is available when DATABASE_URL is not set."""
        with patch("api.v1.websocket.DEMO_MODE", True):
            client = TestClient(app)
            
            with client.websocket_connect("/api/v1/ws/demo") as websocket:
                # Should receive welcome message
                data = websocket.receive_json()
                assert data["type"] == "demo_welcome"
                assert "Connected to FreeAgentics demo WebSocket" in data["message"]
                assert "features" in data
                
    def test_demo_websocket_unavailable_in_production(self):
        """Test that demo WebSocket is NOT available when DATABASE_URL is set."""
        with patch("api.v1.websocket.DEMO_MODE", False):
            client = TestClient(app)
            
            with pytest.raises(Exception) as exc_info:
                with client.websocket_connect("/api/v1/ws/demo") as websocket:
                    pass
            
            # Should be rejected with specific code
            assert "4003" in str(exc_info.value)
    
    def test_demo_websocket_ping_pong(self):
        """Test ping/pong functionality in demo WebSocket."""
        with patch("api.v1.websocket.DEMO_MODE", True):
            client = TestClient(app)
            
            with client.websocket_connect("/api/v1/ws/demo") as websocket:
                # Skip welcome message
                websocket.receive_json()
                
                # Send ping
                websocket.send_json({"type": "ping"})
                
                # Should receive pong
                data = websocket.receive_json()
                assert data["type"] == "pong"
                assert "timestamp" in data
    
    def test_demo_websocket_agent_creation(self):
        """Test simulated agent creation in demo WebSocket."""
        with patch("api.v1.websocket.DEMO_MODE", True):
            client = TestClient(app)
            
            with client.websocket_connect("/api/v1/ws/demo") as websocket:
                # Skip welcome message
                websocket.receive_json()
                
                # Create agent
                websocket.send_json({
                    "type": "agent_create",
                    "data": {
                        "name": "Test Agent",
                        "type": "explorer"
                    }
                })
                
                # Should receive agent_created response
                data = websocket.receive_json()
                assert data["type"] == "agent_created"
                assert data["agent"]["name"] == "Test Agent"
                assert data["agent"]["type"] == "explorer"
                assert data["agent"]["status"] == "active"
                assert "id" in data["agent"]
                assert "created_at" in data["agent"]
    
    def test_demo_websocket_echo(self):
        """Test echo functionality for unknown message types."""
        with patch("api.v1.websocket.DEMO_MODE", True):
            client = TestClient(app)
            
            with client.websocket_connect("/api/v1/ws/demo") as websocket:
                # Skip welcome message
                websocket.receive_json()
                
                # Send unknown message type
                test_message = {
                    "type": "custom_test",
                    "data": {"foo": "bar"}
                }
                websocket.send_json(test_message)
                
                # Should receive echo
                data = websocket.receive_json()
                assert data["type"] == "echo"
                assert data["original"] == test_message
                assert "timestamp" in data
    
    def test_demo_websocket_invalid_json(self):
        """Test handling of invalid JSON in demo WebSocket."""
        with patch("api.v1.websocket.DEMO_MODE", True):
            client = TestClient(app)
            
            with client.websocket_connect("/api/v1/ws/demo") as websocket:
                # Skip welcome message
                websocket.receive_json()
                
                # Send invalid JSON
                websocket.send_text("not valid json")
                
                # Should receive error
                data = websocket.receive_json()
                assert data["type"] == "error"
                assert data["message"] == "Invalid JSON format"
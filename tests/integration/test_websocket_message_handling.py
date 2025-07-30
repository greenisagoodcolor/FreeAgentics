"""Integration tests for WebSocket message handling.

This test captures the current behavior where prompt_submitted and 
clear_conversation messages return UNKNOWN_MESSAGE_TYPE errors.
"""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from api.main import app


class TestWebSocketMessageHandling:
    """Test WebSocket message handling for all message types."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def dev_websocket_url(self):
        """Get dev WebSocket URL."""
        return "ws://localhost:8000/api/v1/ws/dev"

    def test_prompt_submitted_message_is_acknowledged(self, client):
        """Test that prompt_submitted messages are properly acknowledged."""
        with client.websocket_connect("/api/v1/ws/dev") as websocket:
            # Send prompt_submitted message
            message = {
                "type": "prompt_submitted",
                "prompt_id": "test-prompt-123",
                "prompt": "Test prompt",
                "conversation_id": "test-conv-456"
            }
            websocket.send_json(message)
            
            # Skip initial connection messages and find acknowledgment
            ack_received = False
            for _ in range(5):  # Check up to 5 messages
                response = websocket.receive_json()
                if response.get("type") == "prompt_acknowledged":
                    ack_received = True
                    # Verify response contains expected fields
                    assert response["prompt_id"] == "test-prompt-123"
                    assert response["conversation_id"] == "test-conv-456"
                    assert "timestamp" in response
                    assert "message" in response
                    break
            
            assert ack_received, "Expected prompt_acknowledged response"

    def test_clear_conversation_message_is_acknowledged(self, client):
        """Test that clear_conversation messages are properly acknowledged."""
        with client.websocket_connect("/api/v1/ws/dev") as websocket:
            # Send clear_conversation message
            message = {
                "type": "clear_conversation",
                "data": {
                    "conversationId": "test-conv-789"
                }
            }
            websocket.send_json(message)
            
            # Skip initial connection messages and find acknowledgment
            clear_received = False
            for _ in range(5):  # Check up to 5 messages
                response = websocket.receive_json()
                if response.get("type") == "conversation_cleared":
                    clear_received = True
                    # Verify response contains expected fields
                    assert response["conversation_id"] == "test-conv-789"
                    assert "timestamp" in response
                    assert "message" in response
                    break
            
            assert clear_received, "Expected conversation_cleared response"

    def test_existing_ping_message_works(self, client):
        """Test that existing message types like ping still work."""
        with client.websocket_connect("/api/v1/ws/dev") as websocket:
            # Send ping message
            message = {"type": "ping"}
            websocket.send_json(message)
            
            # Skip initial connection messages and find pong
            pong_received = False
            for _ in range(5):  # Check up to 5 messages
                response = websocket.receive_json()
                if response.get("type") == "pong":
                    pong_received = True
                    break
            
            assert pong_received, "Expected pong response to ping message"

    @pytest.mark.asyncio
    async def test_websocket_with_settings_context(self):
        """Test that WebSocket messages can include settings context.
        
        This test is forward-looking for when we add settings support.
        Currently it just verifies the message is received.
        """
        # This will be implemented when we add settings context support
        pass
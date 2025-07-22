"""
Test suite for Agent Manager integration with UI compatibility layer.

This tests the connection between the UI API and the real agent manager,
ensuring that agents are created, managed, and synchronized properly.

Following TDD approach from CLAUDE.md:
- RED: Write failing tests first
- GREEN: Implement minimal code to pass
- REFACTOR: Improve code quality while keeping tests green
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.v1.agents import Agent as V1Agent
from auth.jwt_handler import jwt_handler


class TestAgentManagerIntegration:
    """Test integration between UI API and Agent Manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.token = jwt_handler.create_access_token(
            user_id="test-user",
            username="testuser",
            role="admin",
            permissions=[
                "create_agent",
                "view_agents",
                "modify_agent",
                "delete_agent",
            ],
        )
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @patch("api.ui_compatibility.v1_create_agent")
    @patch("api.ui_compatibility.broadcast_agent_created_event")
    def test_agent_creation_creates_real_agent(self, mock_broadcast, mock_create):
        """Test that UI agent creation creates real agent in manager."""
        # Mock the V1 API response
        mock_agent = V1Agent(
            id="test-agent-123",
            name="Test Explorer",
            template="basic-explorer",
            status="pending",
            created_at=datetime.now(),
            parameters={"description": "A test explorer agent"},
            inference_count=0,
        )
        mock_create.return_value = mock_agent
        mock_broadcast.return_value = None

        # Mock the agent manager
        with (
            patch("api.v1.agents.agent_manager") as mock_manager,
            patch("api.v1.agents.AGENT_MANAGER_AVAILABLE", True),
        ):
            mock_manager.create_agent.return_value = "real-agent-id"
            mock_manager.start_agent.return_value = True

            # Create agent via UI API
            response = self.client.post(
                "/api/agents",
                json={"description": "A test explorer agent"},
                headers=self.headers,
            )

            assert response.status_code == 201
            agent_data = response.json()

            # Verify agent manager was called
            mock_manager.create_agent.assert_called_once()
            call_args = mock_manager.create_agent.call_args

            # Check that correct parameters were passed
            assert call_args[1]["agent_type"] == "explorer"
            assert call_args[1]["name"] == "A Test Explorer"
            assert call_args[1]["description"] == "A test explorer agent"

            # Verify agent was started
            mock_manager.start_agent.assert_called_once_with("real-agent-id")

            # Verify response format
            assert agent_data["status"] == "active"
            assert agent_data["type"] == "explorer"

            # Verify WebSocket event was broadcasted
            mock_broadcast.assert_called_once()

    @patch("api.ui_compatibility.v1_update_agent_status")
    @patch("api.ui_compatibility.get_agent_ui")
    @patch("api.ui_compatibility.broadcast_agent_updated_event")
    def test_agent_status_updates_manager(self, mock_broadcast, mock_get, mock_update):
        """Test that status updates propagate to agent manager."""
        # Mock the responses
        mock_update.return_value = {
            "agent_id": "test-agent-123",
            "status": "active",
        }
        mock_updated_agent = MagicMock()
        mock_updated_agent.id = "test-agent-123"
        mock_get.return_value = mock_updated_agent
        mock_broadcast.return_value = None

        # Mock the agent manager
        with (
            patch("api.v1.agents.agent_manager") as mock_manager,
            patch("api.v1.agents.AGENT_MANAGER_AVAILABLE", True),
        ):
            mock_manager.start_agent.return_value = True
            mock_manager.stop_agent.return_value = True

            # Test starting agent
            response = self.client.patch(
                "/api/agents/test-agent-123/status",
                json={"status": "active"},
                headers=self.headers,
            )

            assert response.status_code == 200
            mock_manager.start_agent.assert_called_once_with("test-agent-123")

            # Test stopping agent
            response = self.client.patch(
                "/api/agents/test-agent-123/status",
                json={"status": "stopped"},
                headers=self.headers,
            )

            assert response.status_code == 200
            mock_manager.stop_agent.assert_called_once_with("test-agent-123")

            # Verify WebSocket events were broadcasted
            assert mock_broadcast.call_count == 2

    @patch("api.ui_compatibility.v1_delete_agent")
    @patch("api.ui_compatibility.broadcast_agent_deleted_event")
    def test_agent_deletion_removes_from_manager(self, mock_broadcast, mock_delete):
        """Test that agent deletion removes agent from manager."""
        mock_delete.return_value = {"message": "Agent deleted"}
        mock_broadcast.return_value = None

        # Mock the agent manager
        with (
            patch("api.v1.agents.agent_manager") as mock_manager,
            patch("api.v1.agents.AGENT_MANAGER_AVAILABLE", True),
        ):
            mock_manager.delete_agent.return_value = True

            # Delete agent via UI API
            response = self.client.delete("/api/agents/test-agent-123", headers=self.headers)

            assert response.status_code == 200

            # Verify agent manager was called
            mock_manager.delete_agent.assert_called_once_with("test-agent-123")

            # Verify WebSocket event was broadcasted
            mock_broadcast.assert_called_once_with("test-agent-123")

    def test_agent_manager_unavailable_graceful_handling(self):
        """Test graceful handling when agent manager is unavailable."""
        # Mock V1 API to return an agent
        mock_agent = V1Agent(
            id="test-agent-123",
            name="Test Explorer",
            template="basic-explorer",
            status="pending",
            created_at=datetime.now(),
            parameters={"description": "A test explorer agent"},
            inference_count=0,
        )

        with (
            patch("api.ui_compatibility.v1_create_agent") as mock_create,
            patch("api.ui_compatibility.broadcast_agent_created_event") as mock_broadcast,
            patch("api.v1.agents.AGENT_MANAGER_AVAILABLE", False),
        ):
            mock_create.return_value = mock_agent
            mock_broadcast.return_value = None

            # Create agent via UI API
            response = self.client.post(
                "/api/agents",
                json={"description": "A test explorer agent"},
                headers=self.headers,
            )

            assert response.status_code == 201
            agent_data = response.json()

            # Should gracefully handle unavailable manager
            assert agent_data["status"] == "pending"  # Not active since manager unavailable

            # Should still broadcast event
            mock_broadcast.assert_called_once()

    @patch("api.ui_compatibility.v1_create_agent")
    @patch("api.ui_compatibility.broadcast_agent_created_event")
    def test_agent_manager_error_handling(self, mock_broadcast, mock_create):
        """Test error handling when agent manager operations fail."""
        # Mock the V1 API response
        mock_agent = V1Agent(
            id="test-agent-123",
            name="Test Explorer",
            template="basic-explorer",
            status="pending",
            created_at=datetime.now(),
            parameters={"description": "A test explorer agent"},
            inference_count=0,
        )
        mock_create.return_value = mock_agent
        mock_broadcast.return_value = None

        # Mock the agent manager to raise an exception
        with (
            patch("api.v1.agents.agent_manager") as mock_manager,
            patch("api.v1.agents.AGENT_MANAGER_AVAILABLE", True),
        ):
            mock_manager.create_agent.side_effect = Exception("Manager error")

            # Create agent via UI API
            response = self.client.post(
                "/api/agents",
                json={"description": "A test explorer agent"},
                headers=self.headers,
            )

            assert response.status_code == 201
            agent_data = response.json()

            # Should handle error gracefully
            assert agent_data["status"] == "pending"  # Falls back to pending

            # Should still broadcast event
            mock_broadcast.assert_called_once()


class TestWebSocketEventIntegration:
    """Test WebSocket event broadcasting integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.token = jwt_handler.create_access_token(
            user_id="test-user",
            username="testuser",
            role="admin",
            permissions=[
                "create_agent",
                "view_agents",
                "modify_agent",
                "delete_agent",
            ],
        )
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @patch("api.v1.websocket.broadcast_agent_event")
    def test_agent_created_event_broadcast(self, mock_broadcast):
        """Test that agent creation broadcasts WebSocket event."""
        # Mock the broadcast function
        mock_broadcast.return_value = None

        # Mock the V1 API and agent manager
        mock_agent = V1Agent(
            id="test-agent-123",
            name="Test Explorer",
            template="basic-explorer",
            status="pending",
            created_at=datetime.now(),
            parameters={"description": "A test explorer agent"},
            inference_count=0,
        )

        with (
            patch("api.ui_compatibility.v1_create_agent") as mock_create,
            patch("api.v1.agents.AGENT_MANAGER_AVAILABLE", False),
        ):
            mock_create.return_value = mock_agent

            # Create agent via UI API
            response = self.client.post(
                "/api/agents",
                json={"description": "A test explorer agent"},
                headers=self.headers,
            )

            assert response.status_code == 201

            # Verify WebSocket event was broadcasted
            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args

            # Check event parameters - broadcast_agent_event uses keyword arguments
            assert call_args.kwargs["agent_id"] == "test-agent-123"  # agent_id
            assert call_args.kwargs["event_type"] == "created"  # event_type
            assert "agent" in call_args.kwargs["data"]  # event data
            assert "timestamp" in call_args.kwargs["data"]

    @patch("api.v1.websocket.broadcast_agent_event")
    def test_websocket_broadcast_error_handling(self, mock_broadcast):
        """Test graceful handling of WebSocket broadcast errors."""
        # Mock the broadcast function to raise an exception
        mock_broadcast.side_effect = Exception("WebSocket error")

        # Mock the V1 API
        mock_agent = V1Agent(
            id="test-agent-123",
            name="Test Explorer",
            template="basic-explorer",
            status="pending",
            created_at=datetime.now(),
            parameters={"description": "A test explorer agent"},
            inference_count=0,
        )

        with (
            patch("api.ui_compatibility.v1_create_agent") as mock_create,
            patch("api.v1.agents.AGENT_MANAGER_AVAILABLE", False),
        ):
            mock_create.return_value = mock_agent

            # Create agent via UI API
            response = self.client.post(
                "/api/agents",
                json={"description": "A test explorer agent"},
                headers=self.headers,
            )

            # Should still succeed despite WebSocket error
            assert response.status_code == 201

            # Verify broadcast was attempted
            mock_broadcast.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

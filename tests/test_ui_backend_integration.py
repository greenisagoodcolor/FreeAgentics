"""
Test suite for UI-Backend integration.

This test describes the end-to-end flow that should work:
1. UI creates agent via simple API call
2. Backend creates agent and starts it in agent manager
3. WebSocket broadcasts agent events
4. UI receives real-time updates

Following TDD approach from CLAUDE.md:
- Write failing test first
- Then implement just enough code to make it pass
- Refactor while keeping tests green
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from auth.jwt_handler import jwt_handler


class TestUIBackendIntegration:
    """Test suite for UI-Backend integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

        # Create test user token
        self.test_user_data = {
            "user_id": "test-user",
            "username": "testuser",
            "role": "admin",
            "permissions": ["create_agent", "view_agents", "modify_agent"],
        }

        self.access_token = jwt_handler.create_access_token(
            user_id=self.test_user_data["user_id"],
            username=self.test_user_data["username"],
            role=self.test_user_data["role"],
            permissions=self.test_user_data["permissions"],
        )
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

        # Set up database
        self.setup_database()

    def setup_database(self):
        """Set up test database with tables."""
        from database.base import Base
        from database.session import engine

        # Create all tables
        Base.metadata.create_all(bind=engine)

    def test_simple_agent_creation_flow(self):
        """
        Test the simple agent creation flow expected by UI.

        This test WILL FAIL initially because:
        1. UI calls /api/agents but backend has /api/v1/agents
        2. UI sends {description: "..."} but backend expects {name: "...", template: "..."}
        3. Response format mismatch
        """
        # Mock the database operations to focus on the API layer
        from datetime import datetime
        from unittest.mock import patch

        from api.v1.agents import Agent as V1Agent

        # Create a mock agent that would be returned by the database
        mock_agent = V1Agent(
            id="test-agent-123",
            name="An Explorer Agent",
            template="basic-explorer",
            status="active",
            created_at=datetime.now(),
            parameters={"description": "An explorer agent that searches for resources"},
            inference_count=0,
        )

        # Mock the database operations
        with (
            patch("api.ui_compatibility.v1_create_agent") as mock_create,
            patch("api.v1.agents.agent_manager") as mock_agent_manager,
        ):
            # Configure the mock to return our test agent
            mock_create.return_value = mock_agent
            mock_agent_manager.agents = {}

            # Step 1: UI attempts to create agent with simple description
            ui_request = {
                "description": "An explorer agent that searches for resources"
            }

            # Call the endpoint
            response = self.client.post(
                "/api/agents", json=ui_request, headers=self.headers
            )

            # What the UI expects to receive
            assert response.status_code == 201
            agent_data = response.json()

            # UI expects this simple format
            assert "id" in agent_data
            assert "name" in agent_data
            assert "type" in agent_data
            assert "status" in agent_data
            assert agent_data["status"] == "active"  # Should be immediately active

            # Verify the backend was called with correct parameters
            assert mock_create.called
            call_args = mock_create.call_args
            config = call_args[0][0]  # First argument is the config

            # Verify conversion from description to proper format
            assert config.name == "An Explorer Agent"
            assert config.template == "basic-explorer"
            assert (
                config.parameters["description"]
                == "An explorer agent that searches for resources"
            )

    def test_agent_list_simple_format(self):
        """Test that agent list returns UI-expected format."""
        from unittest.mock import patch

        from api.v1.agents import Agent as V1Agent

        # Create mock agents
        mock_agents = [
            V1Agent(
                id="test-agent-1",
                name="Explorer Agent",
                template="basic-explorer",
                status="active",
                created_at=datetime.now(),
                parameters={"description": "First agent"},
                inference_count=0,
            ),
            V1Agent(
                id="test-agent-2",
                name="Collector Agent",
                template="basic-explorer",
                status="idle",
                created_at=datetime.now(),
                parameters={"description": "Second agent"},
                inference_count=0,
            ),
        ]

        # Mock the V1 list endpoint
        with patch("api.ui_compatibility.v1_list_agents") as mock_list:
            mock_list.return_value = mock_agents

            # UI calls simple endpoint
            response = self.client.get("/api/agents", headers=self.headers)

            assert response.status_code == 200
            data = response.json()

            # UI expects simple format
            assert "agents" in data
            assert isinstance(data["agents"], list)
            assert len(data["agents"]) == 2

            # Check agent format
            agent = data["agents"][0]
            assert "id" in agent
            assert "name" in agent
            assert "type" in agent
            assert "status" in agent
            assert agent["type"] == "explorer"  # Should be derived from description

    @pytest.mark.asyncio
    async def test_websocket_agent_events(self):
        """Test WebSocket integration for agent events."""
        # RED: This will fail initially

        # This test is more complex, but shows the integration
        # For now, we'll use a mock WebSocket connection

        with patch("api.v1.websocket.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()

            # Create agent via API
            ui_request = {"description": "Test agent"}
            self.client.post("/api/agents", json=ui_request, headers=self.headers)

            # Should broadcast agent creation event
            mock_manager.broadcast.assert_called_once()

            # Event should be in format UI expects
            call_args = mock_manager.broadcast.call_args
            event_data = call_args[0][0]  # First argument

            assert event_data["type"] == "agent_created"
            assert "data" in event_data
            assert "id" in event_data["data"]

    def test_agent_status_update_flow(self):
        """Test agent status updates work through the API."""
        # RED: This will fail initially

        # Create agent first
        ui_request = {"description": "Test agent"}
        create_response = self.client.post(
            "/api/agents", json=ui_request, headers=self.headers
        )
        assert create_response.status_code == 201

        agent_data = create_response.json()
        agent_id = agent_data["id"]

        # Update status
        status_update = {"status": "idle"}
        update_response = self.client.patch(
            f"/api/agents/{agent_id}/status",
            json=status_update,
            headers=self.headers,
        )

        assert update_response.status_code == 200

        # Verify status changed
        get_response = self.client.get(f"/api/agents/{agent_id}", headers=self.headers)
        assert get_response.status_code == 200
        assert get_response.json()["status"] == "idle"

    def test_agent_deletion_flow(self):
        """Test agent deletion works and cleans up agent manager."""
        # RED: This will fail initially

        # Create agent first
        ui_request = {"description": "Test agent"}
        create_response = self.client.post(
            "/api/agents", json=ui_request, headers=self.headers
        )
        assert create_response.status_code == 201

        agent_data = create_response.json()
        agent_id = agent_data["id"]

        # Delete agent
        delete_response = self.client.delete(
            f"/api/agents/{agent_id}", headers=self.headers
        )
        assert delete_response.status_code == 200

        # Verify agent is gone
        get_response = self.client.get(f"/api/agents/{agent_id}", headers=self.headers)
        assert get_response.status_code == 404

        # Agent should be removed from agent manager
        from api.v1.agents import agent_manager

        assert agent_id not in agent_manager.agents


class TestAgentManagerIntegration:
    """Test integration between API and AgentManager."""

    def test_agent_creation_creates_real_agent(self):
        """Test that API agent creation creates real agent in manager."""
        # RED: This will fail initially

        from api.v1.agents import agent_manager

        # Mock the agent manager to track calls
        with patch.object(agent_manager, "create_agent") as mock_create:
            mock_create.return_value = "test_agent_id"

            client = TestClient(app)

            # Create test token
            test_user_data = {
                "user_id": "test-user",
                "username": "testuser",
                "role": "admin",
                "permissions": ["create_agent"],
            }
            access_token = jwt_handler.create_access_token(
                user_id=test_user_data["user_id"],
                username=test_user_data["username"],
                role=test_user_data["role"],
                permissions=test_user_data["permissions"],
            )
            headers = {"Authorization": f"Bearer {access_token}"}

            # Call API
            ui_request = {"description": "Test explorer agent"}
            client.post("/api/agents", json=ui_request, headers=headers)

            # Should call agent manager with correct parameters
            mock_create.assert_called_once()
            call_args = mock_create.call_args

            # Should convert description to proper agent type
            assert "explorer" in call_args[0] or "explorer" in call_args[1]

    def test_agent_status_updates_manager(self):
        """Test that status updates propagate to agent manager."""
        # RED: This will fail initially

        from api.v1.agents import agent_manager

        with (
            patch.object(agent_manager, "start_agent") as mock_start,
            patch.object(agent_manager, "stop_agent") as mock_stop,
        ):
            client = TestClient(app)

            # Create test token
            test_user_data = {
                "user_id": "test-user",
                "username": "testuser",
                "role": "admin",
                "permissions": ["modify_agent"],
            }
            access_token = jwt_handler.create_access_token(
                user_id=test_user_data["user_id"],
                username=test_user_data["username"],
                role=test_user_data["role"],
                permissions=test_user_data["permissions"],
            )
            headers = {"Authorization": f"Bearer {access_token}"}

            # Mock existing agent
            agent_id = "test-agent-id"

            # Start agent
            client.patch(
                f"/api/agents/{agent_id}/status",
                json={"status": "active"},
                headers=headers,
            )

            # Should call agent manager
            mock_start.assert_called_once_with(agent_id)

            # Stop agent
            client.patch(
                f"/api/agents/{agent_id}/status",
                json={"status": "stopped"},
                headers=headers,
            )

            # Should call agent manager
            mock_stop.assert_called_once_with(agent_id)


if __name__ == "__main__":
    # Run the tests to see them fail (RED phase)
    pytest.main([__file__, "-v"])

"""
Test suite for Agents API endpoints with real database integration.

Tests the FastAPI agents endpoints using proper database models
and real Active Inference implementation.
"""

import uuid

import pytest
from sqlalchemy.orm import Session

# Import the app from the API module
from api.main import app
from database.models import Agent as AgentModel
from database.session import get_db
from tests.fixtures.fixtures import db_session, test_engine
from tests.helpers import get_auth_headers
from tests.test_client_compat import TestClient


class TestAgentsAPI:
    """Test agents API endpoints with real database."""

    @pytest.fixture
    def client(self, db_session: Session):
        """Create test client with database override."""

        def override_get_db():
            try:
                yield db_session
            finally:
                pass  # Session cleanup handled by fixture

        app.dependency_overrides[get_db] = override_get_db

        with TestClient(app) as test_client:
            yield test_client

        # Clean up override
        app.dependency_overrides.clear()

    def test_create_agent(self, client: TestClient, db_session: Session):
        """Test creating a new agent."""
        agent_data = {
            "name": "Test Explorer",
            "template": "basic-explorer",
            "parameters": {"grid_size": 10},
        }

        response = client.post("/api/v1/agents", json=agent_data, headers=get_auth_headers())

        if response.status_code != 201:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Explorer"
        assert data["template"] == "basic-explorer"
        assert data["status"] == "pending"
        assert "id" in data

        # Verify in database
        db_agent = db_session.query(AgentModel).filter_by(id=uuid.UUID(data["id"])).first()
        assert db_agent is not None
        assert db_agent.name == "Test Explorer"

    def test_get_agent(self, client: TestClient, db_session: Session):
        """Test retrieving an agent."""
        # Create agent first
        agent_data = {"name": "Test Agent", "template": "basic-explorer"}
        create_response = client.post("/api/v1/agents", json=agent_data, headers=get_auth_headers())
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Get the agent
        response = client.get(f"/api/v1/agents/{agent_id}", headers=get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == agent_id
        assert data["name"] == "Test Agent"

    def test_list_agents(self, client: TestClient, db_session: Session):
        """Test listing all agents."""
        # Create multiple agents
        for i in range(3):
            agent_data = {
                "name": f"Test Agent {i}",
                "template": "basic-explorer",
            }
            response = client.post("/api/v1/agents", json=agent_data, headers=get_auth_headers())
            assert response.status_code == 201

        # List agents
        response = client.get("/api/v1/agents", headers=get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        # API returns List[Agent] directly, not wrapped in object
        assert len(data) == 3
        assert all("id" in agent for agent in data)

    def test_update_agent(self, client: TestClient, db_session: Session):
        """Test updating an agent."""
        # Create agent
        agent_data = {"name": "Original Name", "template": "basic-explorer"}
        create_response = client.post("/api/v1/agents", json=agent_data, headers=get_auth_headers())
        agent_id = create_response.json()["id"]

        # Update agent status (API only supports PATCH for status)
        response = client.patch(
            f"/api/v1/agents/{agent_id}/status?status=active",
            headers=get_auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert data["agent_id"] == agent_id

    def test_delete_agent(self, client: TestClient, db_session: Session):
        """Test deleting an agent."""
        # Create agent
        agent_data = {"name": "To Delete", "template": "basic-explorer"}
        create_response = client.post("/api/v1/agents", json=agent_data, headers=get_auth_headers())
        agent_id = create_response.json()["id"]

        # Delete agent
        response = client.delete(f"/api/v1/agents/{agent_id}", headers=get_auth_headers())
        assert response.status_code == 200  # API returns 200 with message, not 204
        data = response.json()
        assert "deleted successfully" in data["message"]

        # Verify deletion
        get_response = client.get(f"/api/v1/agents/{agent_id}", headers=get_auth_headers())
        assert get_response.status_code == 404

    def test_agent_not_found(self, client: TestClient):
        """Test accessing non-existent agent."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/agents/{fake_id}", headers=get_auth_headers())
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_invalid_agent_data(self, client: TestClient):
        """Test creating agent with invalid data."""
        # Missing required fields
        response = client.post("/api/v1/agents", json={}, headers=get_auth_headers())
        assert response.status_code == 422

        # Empty name (should fail min_length validation)
        invalid_data = {"name": "", "template": "basic-explorer"}
        response = client.post("/api/v1/agents", json=invalid_data, headers=get_auth_headers())
        assert response.status_code == 422

    def test_agent_creation_with_parameters(self, client: TestClient, db_session: Session):
        """Test agent creation with custom parameters."""
        # Create agent with parameters
        agent_data = {
            "name": "AI Agent",
            "template": "basic-explorer",
            "parameters": {"test": True, "exploration_rate": 0.5},
        }
        response = client.post("/api/v1/agents", json=agent_data, headers=get_auth_headers())

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "AI Agent"
        assert data["template"] == "basic-explorer"
        assert data["parameters"]["test"] is True
        assert data["parameters"]["exploration_rate"] == 0.5

    def test_agent_status_transitions(self, client: TestClient, db_session: Session):
        """Test agent status state machine."""
        # Create agent
        agent_data = {"name": "State Test", "template": "basic-explorer"}
        create_response = client.post("/api/v1/agents", json=agent_data, headers=get_auth_headers())
        agent_id = create_response.json()["id"]

        # Valid transitions
        valid_transitions = [
            ("pending", "active"),
            ("active", "paused"),
            ("paused", "active"),
            ("active", "stopped"),
        ]

        current_status = "pending"
        for from_status, to_status in valid_transitions:
            if current_status == from_status:
                response = client.patch(
                    f"/api/v1/agents/{agent_id}/status?status={to_status}",
                    headers=get_auth_headers(),
                )
                assert response.status_code == 200
                assert response.json()["status"] == to_status
                current_status = to_status

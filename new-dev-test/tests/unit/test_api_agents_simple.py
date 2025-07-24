"""
Test suite for agent API endpoints following TDD principles.

This test file verifies core agent API endpoints functionality.
Following TDD: tests fail first, then we implement minimal code to make them pass.
"""

import pytest
from api.main import app
from auth.security_implementation import get_current_user
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from database.session import get_db
from tests.test_helpers.auth_helpers import mock_auth_dependency


class TestAgentEndpoints:
    """Test class for agent API endpoints."""

    @pytest.fixture
    def client(self, db_session: Session):
        """Create test client with mocked authentication and database."""
        # Override the auth dependency to bypass authentication for testing
        app.dependency_overrides[get_current_user] = mock_auth_dependency

        # Override database dependency
        def override_get_db():
            try:
                yield db_session
            finally:
                pass  # Session cleanup handled by fixture

        app.dependency_overrides[get_db] = override_get_db

        with TestClient(app) as test_client:
            yield test_client

        # Clean up the overrides after test
        app.dependency_overrides.clear()

    @pytest.fixture
    def sample_agent_config(self):
        """Create sample agent configuration."""
        return {
            "name": "Test Agent",
            "template": "explorer",
            "parameters": {"grid_size": 10},
            "gmn_spec": None,
            "use_pymdp": True,
        }

    def test_create_agent_endpoint_exists(self, client):
        """Test that POST /api/v1/agents endpoint exists."""
        # Following TDD: Start with the simplest test
        response = client.post("/api/v1/agents", json={})
        # We expect 422 (validation error) not 404 (not found)
        assert response.status_code != 404

    def test_create_agent_returns_validation_error_for_empty_body(self, client):
        """Test that empty request body returns validation error."""
        response = client.post("/api/v1/agents", json={})
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_create_agent_with_valid_data(self, client, sample_agent_config):
        """Test creating agent with valid configuration."""
        response = client.post("/api/v1/agents", json=sample_agent_config)
        assert response.status_code == 201
        data = response.json()

        # Verify response structure
        assert "id" in data
        assert data["name"] == sample_agent_config["name"]
        assert data["template"] == sample_agent_config["template"]
        assert data["status"] == "pending"  # Default status

    def test_get_agent_by_id(self, client, sample_agent_config):
        """Test retrieving agent by ID."""
        # First create an agent
        create_response = client.post("/api/v1/agents", json=sample_agent_config)
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Now retrieve it
        response = client.get(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 200
        data = response.json()

        assert data["id"] == agent_id
        assert data["name"] == sample_agent_config["name"]

    def test_get_nonexistent_agent_returns_404(self, client):
        """Test that requesting non-existent agent returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/agents/{fake_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_agents_endpoint(self, client, sample_agent_config):
        """Test listing all agents."""
        # Create a few agents
        for i in range(3):
            config = sample_agent_config.copy()
            config["name"] = f"Test Agent {i}"
            response = client.post("/api/v1/agents", json=config)
            assert response.status_code == 201

        # List all agents
        response = client.get("/api/v1/agents")
        assert response.status_code == 200
        data = response.json()

        assert "agents" in data
        assert len(data["agents"]) >= 3  # At least our 3 agents
        assert "total" in data

    def test_update_agent_status(self, client, sample_agent_config):
        """Test updating agent status."""
        # Create agent
        create_response = client.post("/api/v1/agents", json=sample_agent_config)
        agent_id = create_response.json()["id"]

        # Update status
        update_data = {"status": "active"}
        response = client.put(f"/api/v1/agents/{agent_id}", json=update_data)
        assert response.status_code == 200
        assert response.json()["status"] == "active"

    def test_delete_agent(self, client, sample_agent_config):
        """Test deleting an agent."""
        # Create agent
        create_response = client.post("/api/v1/agents", json=sample_agent_config)
        agent_id = create_response.json()["id"]

        # Delete agent
        response = client.delete(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_response = client.get(f"/api/v1/agents/{agent_id}")
        assert get_response.status_code == 404

    def test_agent_template_validation(self, client):
        """Test that invalid template names are rejected."""
        invalid_config = {
            "name": "Test Agent",
            "template": "invalid_template_name",
            "parameters": {},
        }
        response = client.post("/api/v1/agents", json=invalid_config)
        # Should get validation error
        assert response.status_code == 422

    def test_agent_parameters_persistence(self, client, sample_agent_config):
        """Test that agent parameters are properly stored and retrieved."""
        # Create agent with specific parameters
        config = sample_agent_config.copy()
        config["parameters"] = {
            "grid_size": 20,
            "learning_rate": 0.01,
            "custom_setting": "test_value",
        }

        create_response = client.post("/api/v1/agents", json=config)
        agent_id = create_response.json()["id"]

        # Retrieve and verify parameters
        response = client.get(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["parameters"]["grid_size"] == 20
        assert data["parameters"]["learning_rate"] == 0.01
        assert data["parameters"]["custom_setting"] == "test_value"

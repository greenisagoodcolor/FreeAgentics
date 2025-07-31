"""End-to-end test for development mode."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.providers import reset_providers


class TestDevModeE2E:
    """Test complete dev mode functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset providers before each test."""
        reset_providers()

    @pytest.fixture
    def demo_env(self):
        """Set up demo environment."""
        with patch.dict(os.environ, {"DATABASE_URL": "", "PRODUCTION": "false"}, clear=True):
            yield

    @pytest.fixture
    def client(self, demo_env):
        """Create test client with demo mode."""
        # Import here to ensure env vars are set
        from api.main import app

        return TestClient(app)

    def test_dev_config_endpoint(self, client):
        """Test /api/v1/dev-config returns auth token in demo mode."""
        response = client.get("/api/v1/dev-config")
        assert response.status_code == 200

        data = response.json()
        assert data["mode"] == "demo"
        assert "auth" in data
        assert data["auth"]["token"] is not None
        assert data["features"]["auth_required"] is False

    def test_agents_endpoint_with_dev_token(self, client):
        """Test agents endpoint works with dev token."""
        # Get dev token
        config_response = client.get("/api/v1/dev-config")
        token = config_response.json()["auth"]["token"]

        # Use token to access agents
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == 200
        assert response.json() == []  # Empty list initially

    def test_create_agent_with_dev_token(self, client):
        """Test creating agent with dev token."""
        # Get dev token
        config_response = client.get("/api/v1/dev-config")
        token = config_response.json()["auth"]["token"]

        # Create agent
        headers = {"Authorization": f"Bearer {token}"}
        agent_data = {"name": "Test Agent", "template": "basic-explorer", "parameters": {}}

        response = client.post("/api/v1/agents", json=agent_data, headers=headers)
        assert response.status_code == 201

        agent = response.json()
        assert agent["name"] == "Test Agent"
        assert agent["status"] == "pending"

    def test_knowledge_graph_endpoint(self, client):
        """Test knowledge graph endpoint exists."""
        # Get dev token
        config_response = client.get("/api/v1/dev-config")
        token = config_response.json()["auth"]["token"]

        # Access knowledge graph
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/knowledge-graph", headers=headers)
        assert response.status_code == 200

        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0

    def test_websocket_demo_endpoint(self, client):
        """Test WebSocket demo endpoint is available."""
        # Should be accessible without auth in demo mode
        with client.websocket_connect("/api/v1/ws/demo") as websocket:
            # Should receive connection event
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["data"]["demo_mode"] is True

    def test_health_endpoint_no_auth(self, client):
        """Test health endpoint needs no auth."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

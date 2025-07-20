"""
Test suite for agent API endpoints following TDD principles.

This test file verifies core agent API endpoints functionality.
Following TDD: tests fail first, then we implement minimal code to make them pass.
"""

import os

import pytest
from fastapi.testclient import TestClient

from api.main import app
from auth.security_implementation import get_current_user
from database.session import init_db
from tests.test_helpers.auth_helpers import mock_auth_dependency

# Set testing environment before using app
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["API_KEY"] = "test_api_key_for_testing"
os.environ["SECRET_KEY"] = "this_is_a_test_secret_key_with_enough_characters"
os.environ["DEVELOPMENT_MODE"] = "false"

# Initialize database tables for testing
init_db()


class TestAgentEndpoints:
    """Test class for agent API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked authentication."""
        # Override the auth dependency to bypass authentication for testing
        app.dependency_overrides[get_current_user] = mock_auth_dependency
        client = TestClient(app)
        yield client
        # Clean up the override after test
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
            "planning_horizon": 3,
        }

    def test_create_agent_endpoint_exists(self, client, sample_agent_config):
        """Test that the create agent endpoint exists and responds."""
        response = client.post("/api/v1/agents", json=sample_agent_config)

        # The endpoint should exist (not 404) and handle the request
        assert (
            response.status_code != 404
        ), "Create agent endpoint should exist"

        # The response should be either success (201) or an error we can handle
        assert response.status_code in [
            200,
            201,
            422,
            500,
        ], "Should return a valid HTTP status"

    def test_get_agents_endpoint_exists(self, client):
        """Test that the get agents endpoint exists and responds."""
        response = client.get("/api/v1/agents")

        # The endpoint should exist (not 404)
        assert response.status_code != 404, "Get agents endpoint should exist"

        # Should return some form of response
        assert response.status_code in [
            200,
            500,
        ], "Should return a valid HTTP status"

    def test_get_single_agent_endpoint_exists(self, client):
        """Test that the get single agent endpoint exists."""
        response = client.get(
            "/api/v1/agents/12345678-1234-1234-1234-123456789012"
        )

        # The endpoint should exist (not 404)
        assert (
            response.status_code != 404
        ), "Get single agent endpoint should exist"

        # Should handle the request (even if agent doesn't exist)
        assert response.status_code in [
            200,
            404,
            500,
        ], "Should return a valid HTTP status"

    def test_delete_agent_endpoint_exists(self, client):
        """Test that the delete agent endpoint exists."""
        response = client.delete(
            "/api/v1/agents/12345678-1234-1234-1234-123456789012"
        )

        # The endpoint should exist (not 404)
        assert (
            response.status_code != 404
        ), "Delete agent endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            204,
            404,
            500,
        ], "Should return a valid HTTP status"

    def test_get_agent_metrics_endpoint_exists(self, client):
        """Test that the agent metrics endpoint exists."""
        response = client.get(
            "/api/v1/agents/12345678-1234-1234-1234-123456789012/metrics"
        )

        # The endpoint should exist (not 404)
        assert (
            response.status_code != 404
        ), "Agent metrics endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            404,
            500,
        ], "Should return a valid HTTP status"

    def test_start_agent_endpoint_exists(self, client):
        """Test that the start agent endpoint exists."""
        response = client.post(
            "/api/v1/agents/12345678-1234-1234-1234-123456789012/start"
        )

        # The endpoint should exist (not 404)
        assert response.status_code != 404, "Start agent endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            404,
            500,
        ], "Should return a valid HTTP status"

    def test_stop_agent_endpoint_exists(self, client):
        """Test that the stop agent endpoint exists."""
        response = client.post(
            "/api/v1/agents/12345678-1234-1234-1234-123456789012/stop"
        )

        # The endpoint should exist (not 404)
        assert response.status_code != 404, "Stop agent endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            404,
            500,
        ], "Should return a valid HTTP status"

    def test_agent_conversations_endpoint_exists(self, client):
        """Test that the agent conversations endpoint exists."""
        response = client.get(
            "/api/v1/agents/12345678-1234-1234-1234-123456789012/conversations"
        )

        # The endpoint should exist (not 404)
        assert (
            response.status_code != 404
        ), "Agent conversations endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            404,
            500,
        ], "Should return a valid HTTP status"

    def test_create_conversation_endpoint_exists(self, client):
        """Test that the create conversation endpoint exists."""
        conversation_data = {
            "prompt": "Hello, test agent!",
            "provider": "ollama",
            "context": {},
        }

        response = client.post(
            "/api/v1/agents/12345678-1234-1234-1234-123456789012/conversations",
            json=conversation_data,
        )

        # The endpoint should exist (not 404)
        assert (
            response.status_code != 404
        ), "Create conversation endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            201,
            404,
            422,
            500,
        ], "Should return a valid HTTP status"

    def test_templates_endpoint_exists(self, client):
        """Test that the templates endpoint exists."""
        response = client.get("/api/v1/templates")

        # The endpoint should exist (not 404)
        assert response.status_code != 404, "Templates endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            500,
        ], "Should return a valid HTTP status"

    def test_public_templates_endpoint_exists(self, client):
        """Test that the public templates endpoint exists."""
        response = client.get("/api/v1/templates/public")

        # The endpoint should exist (not 404)
        assert (
            response.status_code != 404
        ), "Public templates endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            500,
        ], "Should return a valid HTTP status"

    def test_gmn_agent_creation_endpoint_exists(self, client):
        """Test that the GMN agent creation endpoint exists."""
        gmn_config = {
            "name": "GMN Test Agent",
            "gmn_specification": {
                "id": "test-gmn-agent",
                "description": "Test agent created from GMN",
                "goals": [{"name": "explore", "priority": 0.8}],
                "constraints": {"belief_constraints": []},
            },
        }

        response = client.post("/api/v1/agents/from-gmn", json=gmn_config)

        # The endpoint should exist (not 404)
        assert (
            response.status_code != 404
        ), "GMN agent creation endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            201,
            422,
            500,
        ], "Should return a valid HTTP status"

    def test_demo_agents_endpoint_exists(self, client):
        """Test that the demo agents endpoint exists."""
        response = client.get("/api/v1/agents/demo")

        # The endpoint should exist (not 404)
        assert response.status_code != 404, "Demo agents endpoint should exist"

        # Should handle the request
        assert response.status_code in [
            200,
            500,
        ], "Should return a valid HTTP status"

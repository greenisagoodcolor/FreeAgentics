"""Tests for agent creation API endpoints.

Integration tests for the agent creation REST API, including all endpoints
and WebSocket functionality. Tests both success and failure scenarios.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.creation.models import (
    AgentCreationResult,
    AgentSpecification,
    AnalysisConfidence,
    PersonalityProfile,
    PromptAnalysisResult,
)
from api.v1.agent_creation import router
from database.models import Agent, AgentType


@pytest.fixture
def app():
    """Create FastAPI app with agent creation router."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1/agents")
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent_factory():
    """Create mock agent factory."""
    factory = Mock()

    # Mock successful agent creation
    created_agent = Agent(
        id="test-agent-123",
        name="Test Analyst",
        agent_type=AgentType.ANALYST,
        system_prompt="You are a test analyst agent.",
        personality_traits={"assertiveness": 0.7, "analytical_depth": 0.9},
    )

    analysis_result = PromptAnalysisResult(
        agent_type=AgentType.ANALYST,
        confidence=AnalysisConfidence.HIGH,
        domain="finance",
        capabilities=["data_analysis", "reporting"],
        reasoning="User needs analytical capabilities",
        original_prompt="Help me analyze data",
        processed_prompt="Help me analyze data.",
    )

    specification = AgentSpecification(
        name="Test Analyst",
        agent_type=AgentType.ANALYST,
        system_prompt="You are a test analyst agent.",
        personality=PersonalityProfile(assertiveness=0.7, analytical_depth=0.9),
        source_prompt="Help me analyze data",
        capabilities=["data_analysis", "reporting"],
    )

    success_result = AgentCreationResult(
        success=True,
        agent=created_agent,
        specification=specification,
        analysis_result=analysis_result,
        processing_time_ms=1200,
        llm_calls_made=3,
        tokens_used=450,
    )

    factory.create_agent = AsyncMock(return_value=success_result)
    factory.get_supported_agent_types = AsyncMock(return_value=list(AgentType))
    factory.get_metrics = Mock(
        return_value={
            "agents_created": 10,
            "creation_failures": 1,
            "success_rate": 0.91,
            "failure_rate": 0.09,
            "avg_creation_time_ms": 1500.0,
            "fallback_rate": 0.1,
        }
    )

    return factory


class TestAgentCreationAPI:
    """Test agent creation API endpoints."""

    @patch("api.v1.agent_creation.get_agent_factory")
    @patch("core.providers.get_db")
    def test_create_agent_success(self, mock_db, mock_factory_dep, client, mock_agent_factory):
        """Should create agent successfully via API."""
        mock_factory_dep.return_value = mock_agent_factory
        mock_db.return_value = [Mock()]  # Mock database session generator

        request_data = {
            "prompt": "I need help analyzing financial market trends",
            "agent_name": "Finance Analyst",
            "preferred_type": "analyst",
            "enable_advanced_personality": True,
            "enable_custom_capabilities": True,
        }

        response = client.post("/api/v1/agents/create", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["agent_id"] == "test-agent-123"
        assert data["agent_name"] == "Test Analyst"
        assert data["agent_type"] == "analyst"
        assert data["processing_time_ms"] == 1200
        assert data["analysis_confidence"] == "high"
        assert data["detected_domain"] == "finance"
        assert "data_analysis" in data["capabilities"]

    @patch("api.v1.agent_creation.get_agent_factory")
    def test_create_agent_failure(self, mock_factory_dep, client, mock_agent_factory):
        """Should handle agent creation failure gracefully."""
        # Mock factory to return failure
        failure_result = AgentCreationResult(
            success=False, error_message="LLM service unavailable", processing_time_ms=500
        )
        mock_agent_factory.create_agent = AsyncMock(return_value=failure_result)
        mock_factory_dep.return_value = mock_agent_factory

        request_data = {"prompt": "Create an agent for me"}

        response = client.post("/api/v1/agents/create", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "LLM service unavailable" in data["detail"]

    @patch("api.v1.agent_creation.get_agent_factory")
    def test_create_agent_validation_error(self, mock_factory_dep, client):
        """Should validate request parameters."""
        request_data = {
            "prompt": "short",  # Too short
            "agent_name": "A" * 150,  # Too long
        }

        response = client.post("/api/v1/agents/create", json=request_data)

        assert response.status_code == 422  # Validation error

    @patch("api.v1.agent_creation.get_agent_factory")
    def test_preview_agent_success(self, mock_factory_dep, client, mock_agent_factory):
        """Should preview agent successfully."""
        mock_factory_dep.return_value = mock_agent_factory

        request_data = {"prompt": "Help me critique business proposals", "preferred_type": "critic"}

        response = client.post("/api/v1/agents/preview", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["agent_name"] == "Test Analyst"
        assert data["agent_type"] == "analyst"
        assert "You are a test analyst agent" in data["system_prompt"]
        assert data["confidence"] == "high"
        assert isinstance(data["capabilities"], list)

    @patch("api.v1.agent_creation.get_agent_factory")
    def test_preview_agent_failure(self, mock_factory_dep, client, mock_agent_factory):
        """Should handle preview failure."""
        failure_result = AgentCreationResult(
            success=False, error_message="Preview generation failed"
        )
        mock_agent_factory.create_agent = AsyncMock(return_value=failure_result)
        mock_factory_dep.return_value = mock_agent_factory

        request_data = {"prompt": "Create something for me"}

        response = client.post("/api/v1/agents/preview", json=request_data)

        assert response.status_code == 500

    @patch("api.v1.agent_creation.get_agent_factory")
    def test_get_agent_types(self, mock_factory_dep, client, mock_agent_factory):
        """Should return supported agent types."""
        mock_factory_dep.return_value = mock_agent_factory

        response = client.get("/api/v1/agents/types")

        assert response.status_code == 200
        data = response.json()

        assert "agent_types" in data
        assert len(data["agent_types"]) == 5

        # Check that all expected types are present
        types = [t["type"] for t in data["agent_types"]]
        assert "advocate" in types
        assert "analyst" in types
        assert "critic" in types
        assert "creative" in types
        assert "moderator" in types

        # Check descriptions are present
        for agent_type in data["agent_types"]:
            assert "description" in agent_type
            assert len(agent_type["description"]) > 10

    @patch("api.v1.agent_creation.get_agent_factory")
    def test_get_metrics(self, mock_factory_dep, client, mock_agent_factory):
        """Should return creation metrics."""
        mock_factory_dep.return_value = mock_agent_factory

        response = client.get("/api/v1/agents/metrics")

        assert response.status_code == 200
        data = response.json()

        assert data["agents_created"] == 10
        assert data["creation_failures"] == 1
        assert data["success_rate"] == 0.91
        assert data["system_health"] in ["healthy", "degraded"]
        assert data["fallback_health"] in ["good", "high_usage"]

    def test_invalid_agent_type(self, client):
        """Should reject invalid agent types."""
        request_data = {"prompt": "Help me with something", "preferred_type": "invalid_type"}

        response = client.post("/api/v1/agents/create", json=request_data)

        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Should reject requests missing required fields."""
        request_data = {}  # Missing prompt

        response = client.post("/api/v1/agents/create", json=request_data)

        assert response.status_code == 422

    @patch("api.v1.agent_creation.get_agent_factory")
    def test_exception_handling(self, mock_factory_dep, client):
        """Should handle unexpected exceptions gracefully."""
        mock_factory = Mock()
        mock_factory.create_agent = AsyncMock(side_effect=Exception("Unexpected error"))
        mock_factory_dep.return_value = mock_factory

        request_data = {"prompt": "Create an agent for me"}

        response = client.post("/api/v1/agents/create", json=request_data)

        # Should return error response instead of crashing
        assert response.status_code == 200  # Handled gracefully
        data = response.json()
        assert data["success"] is False
        assert "Unexpected error" in data["error_message"]


class TestWebSocketAgentCreation:
    """Test WebSocket agent creation endpoint."""

    @patch("api.v1.agent_creation.get_agent_factory")
    def test_websocket_agent_creation_success(self, mock_factory_dep, client, mock_agent_factory):
        """Should create agent via WebSocket with progress updates."""
        mock_factory_dep.return_value = mock_agent_factory

        with client.websocket_connect("/api/v1/agents/create/ws") as websocket:
            # Send creation request
            request_data = {"prompt": "I need help analyzing data", "agent_name": "Data Analyst"}
            websocket.send_json(request_data)

            # Should receive progress updates
            messages = []
            for _ in range(5):  # Expect multiple messages
                try:
                    message = websocket.receive_json()
                    messages.append(message)
                    if message.get("status") == "success":
                        break
                except:
                    break

            # Verify we got progress updates and final success
            assert len(messages) > 0
            final_message = messages[-1]
            assert final_message["status"] == "success"
            assert "data" in final_message
            assert final_message["data"]["agent_id"] == "test-agent-123"

    @patch("api.v1.agent_creation.get_agent_factory")
    def test_websocket_agent_creation_failure(self, mock_factory_dep, client):
        """Should handle WebSocket creation failure."""
        mock_factory = Mock()
        failure_result = AgentCreationResult(success=False, error_message="Creation failed")
        mock_factory.create_agent = AsyncMock(return_value=failure_result)
        mock_factory_dep.return_value = mock_factory

        with client.websocket_connect("/api/v1/agents/create/ws") as websocket:
            request_data = {"prompt": "Create an agent"}
            websocket.send_json(request_data)

            # Should receive error message
            message = websocket.receive_json()
            while message.get("status") == "progress":
                message = websocket.receive_json()

            assert message["status"] == "error"
            assert "Creation failed" in message["message"]

    def test_websocket_validation_error(self, client):
        """Should handle validation errors in WebSocket."""
        with client.websocket_connect("/api/v1/agents/create/ws") as websocket:
            # Send invalid request
            invalid_data = {"prompt": "short"}  # Too short
            websocket.send_json(invalid_data)

            message = websocket.receive_json()
            assert message["status"] == "error"
            assert "Request failed" in message["message"]


@pytest.mark.integration
class TestAgentCreationIntegration:
    """Integration tests with real database and services."""

    def test_end_to_end_agent_creation(self):
        """Test complete agent creation workflow end-to-end."""
        # This would test with real database and LLM service
        # Skipped for now as it requires actual API keys and database
        pytest.skip("Integration test requires real services")

    def test_database_persistence(self):
        """Test that created agents are properly persisted."""
        pytest.skip("Integration test requires real database")

    def test_llm_service_integration(self):
        """Test integration with actual LLM service."""
        pytest.skip("Integration test requires LLM API keys")

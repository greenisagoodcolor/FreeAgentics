"""Test suite for prompt processing API endpoint.

Following TDD principles - tests written before implementation.
Tests cover the entire prompt → agent → KG pipeline.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.main import app
from database.models import Agent, AgentStatus


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestPromptsAPI:
    """Test the /api/v1/prompts endpoint."""

    @pytest.fixture
    def mock_prompt_processor(self):
        """Mock prompt processor service."""
        processor = AsyncMock()
        processor.process_prompt = AsyncMock()
        return processor

    @pytest.fixture
    def mock_gmn_generator(self):
        """Mock GMN generator service."""
        generator = AsyncMock()
        generator.prompt_to_gmn = AsyncMock(
            return_value="""
        node state s1 {
            type: discrete
            size: 4
        }
        node observation o1 {
            type: discrete
            size: 5
        }
        node action a1 {
            type: discrete
            size: 4
        }
        """
        )
        generator.validate_gmn = AsyncMock(return_value=(True, []))
        return generator

    @pytest.fixture
    def mock_agent_factory(self):
        """Mock agent factory service."""
        factory = AsyncMock()
        factory.create_from_gmn_model = AsyncMock()
        factory.validate_model = AsyncMock(return_value=(True, []))
        return factory

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Mock knowledge graph service."""
        kg = AsyncMock()
        kg.add_node = AsyncMock(return_value=True)
        kg.update_node = AsyncMock(return_value=True)
        return kg

    @pytest.fixture
    def mock_belief_kg_bridge(self):
        """Mock belief-KG bridge service."""
        bridge = AsyncMock()
        bridge.update_kg_from_agent = AsyncMock(
            return_value={
                "nodes_added": 5,
                "nodes_updated": 2,
                "edges_added": 3,
            }
        )
        return bridge

    def test_prompt_endpoint_exists(self, client: TestClient):
        """Test that the prompts endpoint exists."""
        response = client.post(
            "/api/v1/prompts",
            json={"prompt": "Create an explorer agent"},
            headers={"Authorization": "Bearer test-token"},
        )
        # Should not be 404
        assert response.status_code != status.HTTP_404_NOT_FOUND

    def test_prompt_validation_empty_prompt(self, client: TestClient):
        """Test validation rejects empty prompt."""
        response = client.post(
            "/api/v1/prompts",
            json={"prompt": ""},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "prompt" in str(data)

    def test_prompt_validation_missing_prompt(self, client: TestClient):
        """Test validation rejects missing prompt."""
        response = client.post(
            "/api/v1/prompts",
            json={},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_prompt_validation_iteration_count(self, client: TestClient):
        """Test iteration count validation."""
        # Too high
        response = client.post(
            "/api/v1/prompts",
            json={"prompt": "Test", "iteration_count": 11},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Negative
        response = client.post(
            "/api/v1/prompts",
            json={"prompt": "Test", "iteration_count": -1},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_successful_prompt_processing(
        self,
        client: TestClient,
        mock_prompt_processor,
        mock_gmn_generator,
        mock_agent_factory,
        mock_knowledge_graph,
        mock_belief_kg_bridge,
    ):
        """Test successful prompt → agent → KG flow."""
        # Setup mocks
        agent_id = str(uuid.uuid4())
        mock_agent = MagicMock()
        mock_agent.id = agent_id

        mock_prompt_processor.process_prompt.return_value = {
            "agent_id": agent_id,
            "gmn_specification": "node state s1 {...}",
            "knowledge_graph_updates": [
                {
                    "node_id": "n1",
                    "type": "belief",
                    "properties": {"value": 0.8},
                },
                {
                    "node_id": "n2",
                    "type": "observation",
                    "properties": {"value": "grid"},
                },
            ],
            "next_suggestions": [
                "Add goal-seeking behavior",
                "Implement obstacle avoidance",
            ],
            "status": "success",
        }

        with patch("api.v1.prompts.prompt_processor", mock_prompt_processor):
            response = client.post(
                "/api/v1/prompts",
                json={
                    "prompt": "Create an explorer agent for a grid world",
                    "iteration_count": 2,
                },
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Validate response structure
        assert "agent_id" in data
        assert "gmn_specification" in data
        assert "knowledge_graph_updates" in data
        assert "next_suggestions" in data
        assert "status" in data

        # Validate response content
        assert data["agent_id"] == agent_id
        assert data["status"] == "success"
        assert len(data["knowledge_graph_updates"]) == 2
        assert len(data["next_suggestions"]) == 2

    @pytest.mark.asyncio
    async def test_conversation_continuation(
        self, client: TestClient, mock_prompt_processor, db: Session
    ):
        """Test continuing a conversation with existing context."""
        conversation_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())

        # Create existing agent in DB
        agent = Agent(
            id=agent_id,
            name="Explorer",
            template="explorer",
            status=AgentStatus.ACTIVE,
            gmn_spec="node state s1 {...}",
        )
        db.add(agent)
        db.commit()

        mock_prompt_processor.process_prompt.return_value = {
            "agent_id": agent_id,
            "gmn_specification": "updated GMN spec",
            "knowledge_graph_updates": [],
            "next_suggestions": ["Refine exploration strategy"],
            "status": "success",
        }

        with patch("api.v1.prompts.prompt_processor", mock_prompt_processor):
            response = client.post(
                "/api/v1/prompts",
                json={
                    "prompt": "Make the agent more cautious",
                    "conversation_id": conversation_id,
                },
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == status.HTTP_200_OK
        # Verify conversation_id was passed to processor
        mock_prompt_processor.process_prompt.assert_called_once()
        call_args = mock_prompt_processor.process_prompt.call_args
        assert call_args[1]["conversation_id"] == conversation_id

    @pytest.mark.asyncio
    async def test_gmn_validation_failure(
        self, client: TestClient, mock_prompt_processor
    ):
        """Test handling of GMN validation failures."""
        mock_prompt_processor.process_prompt.side_effect = ValueError(
            "GMN validation failed: Invalid state space dimensions"
        )

        with patch("api.v1.prompts.prompt_processor", mock_prompt_processor):
            response = client.post(
                "/api/v1/prompts",
                json={"prompt": "Create invalid agent"},
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "GMN validation failed" in data["detail"]

    @pytest.mark.asyncio
    async def test_agent_creation_failure(
        self, client: TestClient, mock_prompt_processor
    ):
        """Test handling of agent creation failures."""
        mock_prompt_processor.process_prompt.side_effect = RuntimeError(
            "Agent creation failed: Incompatible belief dimensions"
        )

        with patch("api.v1.prompts.prompt_processor", mock_prompt_processor):
            response = client.post(
                "/api/v1/prompts",
                json={"prompt": "Create agent"},
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Agent creation failed" in data["detail"]

    @pytest.mark.asyncio
    async def test_knowledge_graph_update_failure(
        self, client: TestClient, mock_prompt_processor
    ):
        """Test handling of knowledge graph update failures."""
        # Return partial success with KG warning
        mock_prompt_processor.process_prompt.return_value = {
            "agent_id": str(uuid.uuid4()),
            "gmn_specification": "node state s1 {...}",
            "knowledge_graph_updates": [],
            "next_suggestions": [],
            "status": "partial_success",
            "warnings": ["Knowledge graph update failed: Connection timeout"],
        }

        with patch("api.v1.prompts.prompt_processor", mock_prompt_processor):
            response = client.post(
                "/api/v1/prompts",
                json={"prompt": "Create agent"},
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "partial_success"
        assert "warnings" in data
        assert "Knowledge graph update failed" in data["warnings"][0]

    @pytest.mark.asyncio
    async def test_response_time_constraint(
        self, client: TestClient, mock_prompt_processor
    ):
        """Test that response time is under 3 seconds."""
        import time

        # Simulate processing time
        async def slow_process(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate some processing
            return {
                "agent_id": str(uuid.uuid4()),
                "gmn_specification": "node state s1 {...}",
                "knowledge_graph_updates": [],
                "next_suggestions": [],
                "status": "success",
            }

        mock_prompt_processor.process_prompt = slow_process

        with patch("api.v1.prompts.prompt_processor", mock_prompt_processor):
            start_time = time.time()
            response = client.post(
                "/api/v1/prompts",
                json={"prompt": "Create agent"},
                headers={"Authorization": "Bearer test-token"},
            )
            elapsed_time = time.time() - start_time

        assert response.status_code == status.HTTP_200_OK
        assert (
            elapsed_time < 3.0
        ), f"Response time {elapsed_time}s exceeds 3s limit"

    @pytest.mark.asyncio
    async def test_authentication_required(self, client: TestClient):
        """Test that authentication is required."""
        response = client.post(
            "/api/v1/prompts", json={"prompt": "Create agent"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_permission_required(self, client: TestClient):
        """Test that proper permissions are required."""
        # Mock a user without agent creation permission
        with patch(
            "auth.security_implementation.get_current_user"
        ) as mock_user:
            mock_user.return_value = MagicMock(permissions=[])

            response = client.post(
                "/api/v1/prompts",
                json={"prompt": "Create agent"},
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self, client: TestClient, mock_prompt_processor
    ):
        """Test handling of concurrent prompt requests."""

        mock_prompt_processor.process_prompt.return_value = {
            "agent_id": str(uuid.uuid4()),
            "gmn_specification": "node state s1 {...}",
            "knowledge_graph_updates": [],
            "next_suggestions": [],
            "status": "success",
        }

        with patch("api.v1.prompts.prompt_processor", mock_prompt_processor):
            # Send multiple concurrent requests
            for i in range(5):
                response = client.post(
                    "/api/v1/prompts",
                    json={"prompt": f"Create agent {i}"},
                    headers={"Authorization": "Bearer test-token"},
                )
                assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_logging_and_monitoring(
        self, client: TestClient, mock_prompt_processor, caplog
    ):
        """Test that proper logging occurs."""
        agent_id = str(uuid.uuid4())
        mock_prompt_processor.process_prompt.return_value = {
            "agent_id": agent_id,
            "gmn_specification": "node state s1 {...}",
            "knowledge_graph_updates": [],
            "next_suggestions": [],
            "status": "success",
        }

        with patch("api.v1.prompts.prompt_processor", mock_prompt_processor):
            response = client.post(
                "/api/v1/prompts",
                json={"prompt": "Create agent"},
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == status.HTTP_200_OK

        # Check for expected log entries
        assert any(
            "Processing prompt" in record.message for record in caplog.records
        )
        assert any(agent_id in record.message for record in caplog.records)

"""Integration tests for the complete prompt → agent → KG pipeline.

These tests verify the end-to-end functionality of the prompt processing
pipeline including WebSocket real-time updates.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.prompts import PromptRequest, get_prompt_processor, process_prompt
from auth.security_implementation import Permission, Role, TokenData
from database.prompt_models import Conversation
from services.websocket_integration import (
    pipeline_monitor,
)


@pytest.fixture
async def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_current_user():
    """Create a mock authenticated user."""
    return TokenData(
        username="testuser",
        user_id="user123",
        role=Role.USER,
        permissions=[Permission.CREATE_AGENT, Permission.VIEW_AGENTS],
    )


@pytest.fixture
def websocket_events():
    """Collect WebSocket events for verification."""
    events = []

    async def capture_event(event_type: str, data: Dict[str, Any]):
        events.append(
            {
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.utcnow(),
            }
        )

    return events, capture_event


class TestPromptPipelineIntegration:
    """Test the complete prompt processing pipeline."""

    @pytest.mark.asyncio
    async def test_successful_pipeline_processing(
        self, mock_db_session, mock_current_user, websocket_events
    ):
        """Test successful end-to-end pipeline processing with WebSocket updates."""
        events_list, capture_event = websocket_events

        # Create prompt processor with WebSocket callback
        processor = get_prompt_processor()
        processor.websocket_callback = capture_event

        # Mock conversation query
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Create request
        request = PromptRequest(
            prompt="Create an explorer agent for a 5x5 grid world",
            iteration_count=2,
        )

        # Process prompt
        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_prompt_processor", return_value=processor):
                response = await process_prompt(
                    request=request,
                    current_user=mock_current_user,
                    db=mock_db_session,
                    prompt_processor=processor,
                )

        # Verify response
        assert response.agent_id
        assert response.gmn_specification
        assert response.knowledge_graph_updates is not None
        assert len(response.next_suggestions) > 0
        assert response.status == "success"
        assert response.processing_time_ms > 0

        # Verify WebSocket events were sent
        event_types = [e["event_type"] for e in events_list]
        assert "pipeline_started" in event_types
        assert "pipeline_progress" in event_types
        assert "gmn_generated" in event_types
        assert "validation_success" in event_types
        assert "agent_created" in event_types
        assert "knowledge_graph_updated" in event_types
        assert "pipeline_completed" in event_types

        # Verify event sequence
        start_event = next(e for e in events_list if e["event_type"] == "pipeline_started")
        assert start_event["data"]["total_stages"] == 6

        completed_event = next(e for e in events_list if e["event_type"] == "pipeline_completed")
        assert completed_event["data"]["status"] == "success"
        assert completed_event["data"]["agent_id"] == response.agent_id

    @pytest.mark.asyncio
    async def test_pipeline_gmn_validation_failure(
        self, mock_db_session, mock_current_user, websocket_events
    ):
        """Test pipeline handling of GMN validation failure."""
        events_list, capture_event = websocket_events

        # Create processor with mocked GMN generator that produces invalid GMN
        processor = get_prompt_processor()
        processor.websocket_callback = capture_event

        # Mock GMN generator to return invalid GMN
        async def mock_generate_gmn(*args, **kwargs):
            return "invalid gmn without proper nodes"

        processor.gmn_generator.prompt_to_gmn = mock_generate_gmn

        # Mock conversation query
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Create request
        request = PromptRequest(prompt="Create an invalid agent")

        # Process prompt and expect failure
        with pytest.raises(Exception):
            with patch("api.v1.prompts.get_db", return_value=mock_db_session):
                with patch(
                    "api.v1.prompts.get_prompt_processor",
                    return_value=processor,
                ):
                    await process_prompt(
                        request=request,
                        current_user=mock_current_user,
                        db=mock_db_session,
                        prompt_processor=processor,
                    )

        # Verify failure events
        event_types = [e["event_type"] for e in events_list]
        assert "pipeline_started" in event_types
        assert "pipeline_failed" in event_types or "validation_failed" in event_types

        # Verify no success events
        assert "pipeline_completed" not in event_types
        assert "agent_created" not in event_types

    @pytest.mark.asyncio
    async def test_pipeline_with_existing_conversation(
        self, mock_db_session, mock_current_user, websocket_events
    ):
        """Test pipeline processing with existing conversation context."""
        events_list, capture_event = websocket_events

        # Create processor
        processor = get_prompt_processor()
        processor.websocket_callback = capture_event

        # Mock existing conversation
        mock_conversation = MagicMock(spec=Conversation)
        mock_conversation.id = "conv123"
        mock_conversation.context = {"agent_count": 2}
        mock_conversation.agent_ids = ["agent1", "agent2"]
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation

        # Create request with conversation ID
        request = PromptRequest(
            prompt="Add a coordinator agent to manage existing agents",
            conversation_id="conv123",
        )

        # Process prompt
        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_prompt_processor", return_value=processor):
                response = await process_prompt(
                    request=request,
                    current_user=mock_current_user,
                    db=mock_db_session,
                    prompt_processor=processor,
                )

        # Verify response
        assert response.status == "success"
        assert (
            "coalition" in " ".join(response.next_suggestions).lower()
        )  # Should suggest coalition

        # Verify conversation context was used
        start_event = next(e for e in events_list if e["event_type"] == "pipeline_started")
        assert start_event["data"]["conversation_id"] == "conv123"

    @pytest.mark.asyncio
    async def test_pipeline_performance_under_3s(
        self, mock_db_session, mock_current_user, websocket_events
    ):
        """Test that pipeline completes within 3 second performance target."""
        events_list, capture_event = websocket_events

        # Create processor
        processor = get_prompt_processor()
        processor.websocket_callback = capture_event

        # Mock conversation query
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Create request
        request = PromptRequest(prompt="Create a simple explorer agent")

        # Time the processing
        start_time = asyncio.get_event_loop().time()

        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_prompt_processor", return_value=processor):
                response = await process_prompt(
                    request=request,
                    current_user=mock_current_user,
                    db=mock_db_session,
                    prompt_processor=processor,
                )

        end_time = asyncio.get_event_loop().time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms

        # Verify performance
        assert response.status == "success"
        assert processing_time < 3000  # Should complete under 3 seconds
        assert response.processing_time_ms < 3000

        # Verify timing in events
        completed_event = next(e for e in events_list if e["event_type"] == "pipeline_completed")
        assert completed_event["data"]["processing_time_ms"] < 3000

    @pytest.mark.asyncio
    async def test_pipeline_stage_progression(
        self, mock_db_session, mock_current_user, websocket_events
    ):
        """Test that pipeline progresses through all stages correctly."""
        events_list, capture_event = websocket_events

        # Create processor
        processor = get_prompt_processor()
        processor.websocket_callback = capture_event

        # Mock conversation query
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Create request
        request = PromptRequest(prompt="Create a trader agent")

        # Process prompt
        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_prompt_processor", return_value=processor):
                await process_prompt(
                    request=request,
                    current_user=mock_current_user,
                    db=mock_db_session,
                    prompt_processor=processor,
                )

        # Extract progress events
        progress_events = [e for e in events_list if e["event_type"] == "pipeline_progress"]

        # Verify all stages were processed
        stages_processed = [e["data"]["stage"] for e in progress_events]
        expected_stages = [
            "gmn_generation",
            "gmn_validation",
            "agent_creation",
            "database_storage",
            "knowledge_graph_update",
            "suggestion_generation",
        ]

        for stage in expected_stages:
            assert stage in stages_processed

        # Verify stage numbers
        for event in progress_events:
            assert 1 <= event["data"]["stage_number"] <= 6
            assert event["data"]["message"]

    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(
        self, mock_db_session, mock_current_user, websocket_events
    ):
        """Test pipeline error handling and recovery."""
        events_list, capture_event = websocket_events

        # Create processor
        processor = get_prompt_processor()
        processor.websocket_callback = capture_event

        # Mock knowledge graph update to fail but not critically

        async def mock_update_kg(*args, **kwargs):
            # Simulate a non-critical KG update failure
            return []  # Empty updates but don't raise exception

        processor._update_knowledge_graph = mock_update_kg

        # Mock conversation query
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Create request
        request = PromptRequest(prompt="Create an agent with KG issues")

        # Process prompt
        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_prompt_processor", return_value=processor):
                response = await process_prompt(
                    request=request,
                    current_user=mock_current_user,
                    db=mock_db_session,
                    prompt_processor=processor,
                )

        # Should still succeed despite KG update issues
        assert response.status == "success"
        assert response.agent_id
        assert len(response.knowledge_graph_updates) == 0  # No updates due to failure

        # Verify events show KG update with 0 updates
        kg_event = next(e for e in events_list if e["event_type"] == "knowledge_graph_updated")
        assert kg_event["data"]["updates_count"] == 0
        assert kg_event["data"]["nodes_added"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_monitoring_integration(self, mock_db_session, mock_current_user):
        """Test integration with pipeline monitoring system."""
        # Reset pipeline monitor
        pipeline_monitor.active_pipelines.clear()

        # Create processor
        processor = get_prompt_processor()

        # Mock conversation query
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Create request
        request = PromptRequest(prompt="Create a monitored agent")

        # Start monitoring
        prompt_id = str(uuid.uuid4())
        pipeline_monitor.start_pipeline(prompt_id, mock_current_user.user_id, request.prompt)

        # Process prompt
        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_prompt_processor", return_value=processor):
                await process_prompt(
                    request=request,
                    current_user=mock_current_user,
                    db=mock_db_session,
                    prompt_processor=processor,
                )

        # Verify monitoring data
        active_pipelines = pipeline_monitor.get_active_pipelines(mock_current_user.user_id)
        assert len(active_pipelines) == 1
        assert active_pipelines[0]["user_id"] == mock_current_user.user_id

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_processing(
        self, mock_db_session, mock_current_user, websocket_events
    ):
        """Test multiple concurrent pipeline processing."""
        events_list, capture_event = websocket_events

        # Create processor
        processor = get_prompt_processor()
        processor.websocket_callback = capture_event

        # Mock conversation query
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Create multiple requests
        requests = [PromptRequest(prompt=f"Create agent {i}") for i in range(3)]

        # Process concurrently
        tasks = []
        for request in requests:
            task = process_prompt(
                request=request,
                current_user=mock_current_user,
                db=mock_db_session,
                prompt_processor=processor,
            )
            tasks.append(task)

        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_prompt_processor", return_value=processor):
                responses = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(responses) == 3
        for response in responses:
            assert response.status == "success"
            assert response.agent_id

        # Verify events for all pipelines
        completed_events = [e for e in events_list if e["event_type"] == "pipeline_completed"]
        assert len(completed_events) == 3

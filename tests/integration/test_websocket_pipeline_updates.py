"""Integration tests for WebSocket real-time pipeline updates.

These tests verify that WebSocket clients receive proper real-time updates
during prompt processing pipeline execution.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.prompts import websocket_pipeline_callback
from api.v1.websocket import (
    ConnectionManager,
    broadcast_agent_event,
    broadcast_system_event,
)
from auth.security_implementation import Permission, Role, TokenData
from services.websocket_integration import (
    PipelineEventType,
    PipelineStage,
    broadcast_pipeline_event,
    create_pipeline_progress_message,
    pipeline_monitor,
)


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock(spec=WebSocket)
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def connection_manager():
    """Create a connection manager instance."""
    manager = ConnectionManager()
    manager.active_connections.clear()
    manager.subscriptions.clear()
    return manager


@pytest.fixture
def mock_client_metadata():
    """Create mock client metadata."""
    return {
        "user_id": "user123",
        "username": "testuser",
        "role": "user",
        "authenticated": True,
        "permissions": ["CREATE_AGENT", "VIEW_AGENTS"],
    }


class TestWebSocketPipelineUpdates:
    """Test WebSocket updates during pipeline processing."""

    @pytest.mark.asyncio
    async def test_websocket_callback_broadcasts_events(
        self, connection_manager
    ):
        """Test that the WebSocket callback properly broadcasts pipeline events."""
        # Track broadcast calls
        broadcast_calls = []

        async def mock_broadcast(message, event_type=None):
            broadcast_calls.append(
                {"message": message, "event_type": event_type}
            )

        # Patch broadcast methods
        with patch('api.v1.prompts.broadcast_system_event', mock_broadcast):
            with patch('api.v1.prompts.broadcast_agent_event', mock_broadcast):
                # Test pipeline started event
                await websocket_pipeline_callback(
                    "pipeline_started",
                    {
                        "prompt_id": "prompt123",
                        "user_id": "user123",
                        "stage": "initialization",
                    },
                )

                # Test agent created event
                await websocket_pipeline_callback(
                    "agent_created",
                    {
                        "prompt_id": "prompt123",
                        "agent_id": "agent456",
                        "agent_type": "explorer",
                    },
                )

                # Test KG updated event
                await websocket_pipeline_callback(
                    "knowledge_graph_updated",
                    {
                        "prompt_id": "prompt123",
                        "updates_count": 5,
                        "nodes_added": 3,
                    },
                )

        # Verify broadcasts
        assert len(broadcast_calls) >= 3

        # Check specific events
        event_types = [
            call["event_type"]
            for call in broadcast_calls
            if call["event_type"]
        ]
        assert "pipeline:pipeline_started" in event_types
        assert any("knowledge_graph:updated" in et for et in event_types if et)

    @pytest.mark.asyncio
    async def test_pipeline_subscriber_receives_updates(
        self, connection_manager, mock_websocket, mock_client_metadata
    ):
        """Test that subscribed clients receive pipeline updates."""
        client_id = "client123"
        prompt_id = "prompt456"

        # Connect client
        await connection_manager.connect(
            mock_websocket, client_id, mock_client_metadata
        )

        # Subscribe to pipeline
        pipeline_monitor.subscribe_to_pipeline(prompt_id, client_id)

        # Send pipeline event
        await broadcast_pipeline_event(
            PipelineEventType.PIPELINE_PROGRESS,
            prompt_id,
            {
                "stage": "gmn_generation",
                "stage_number": 1,
                "message": "Generating GMN...",
            },
            connection_manager,
        )

        # Verify message sent to client
        mock_websocket.send_json.assert_called()
        sent_message = mock_websocket.send_json.call_args[0][0]
        assert sent_message["type"] == "pipeline:pipeline_progress"
        assert sent_message["prompt_id"] == prompt_id
        assert sent_message["data"]["stage"] == "gmn_generation"

    @pytest.mark.asyncio
    async def test_multiple_clients_receive_updates(self, connection_manager):
        """Test that multiple clients receive pipeline updates."""
        # Connect multiple clients
        clients = []
        for i in range(3):
            ws = AsyncMock(spec=WebSocket)
            ws.send_json = AsyncMock()
            client_id = f"client{i}"
            await connection_manager.connect(
                ws, client_id, {"user_id": f"user{i}"}
            )
            clients.append((client_id, ws))

        # Subscribe all to same pipeline event type
        for client_id, _ in clients:
            connection_manager.subscribe(client_id, "pipeline:updates")

        # Broadcast event
        await connection_manager.broadcast(
            {"type": "pipeline_progress", "data": {"stage": "agent_creation"}},
            event_type="pipeline:updates",
        )

        # Verify all clients received the message
        for _, ws in clients:
            ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_progress_messages(self):
        """Test creation of standardized progress messages."""
        # Test progress message creation
        msg = create_pipeline_progress_message(
            stage=PipelineStage.GMN_VALIDATION,
            stage_number=2,
            total_stages=6,
            message="Validating GMN specification",
            details={"errors": 0, "warnings": 1},
        )

        assert msg["stage"] == "gmn_validation"
        assert msg["stage_number"] == 2
        assert msg["progress_percentage"] == (2 / 6) * 100
        assert msg["message"] == "Validating GMN specification"
        assert msg["details"]["warnings"] == 1

    @pytest.mark.asyncio
    async def test_pipeline_monitoring_lifecycle(self):
        """Test complete pipeline monitoring lifecycle."""
        prompt_id = str(uuid.uuid4())
        user_id = "user123"

        # Start pipeline
        pipeline_info = pipeline_monitor.start_pipeline(
            prompt_id, user_id, "Create an explorer agent"
        )

        assert pipeline_info["prompt_id"] == prompt_id
        assert (
            pipeline_info["current_stage"]
            == PipelineStage.INITIALIZATION.value
        )

        # Update stages
        stage_updates = [
            (PipelineStage.GMN_GENERATION, "Generating GMN"),
            (PipelineStage.GMN_VALIDATION, "Validating GMN"),
            (PipelineStage.AGENT_CREATION, "Creating agent"),
            (PipelineStage.DATABASE_STORAGE, "Storing in database"),
            (PipelineStage.KNOWLEDGE_GRAPH_UPDATE, "Updating KG"),
            (PipelineStage.SUGGESTION_GENERATION, "Generating suggestions"),
        ]

        for stage, message in stage_updates:
            update_info = pipeline_monitor.update_stage(
                prompt_id, stage, message
            )
            assert update_info["current_stage"] == stage.value

        # Complete pipeline
        completion_info = pipeline_monitor.complete_pipeline(
            prompt_id,
            agent_id="agent789",
            processing_time_ms=1500.0,
            result_data={
                "next_suggestions": ["Explore", "Trade", "Coordinate"],
                "knowledge_graph_updates": [1, 2, 3],
            },
        )

        assert completion_info["agent_id"] == "agent789"
        assert completion_info["processing_time_ms"] == 1500.0
        assert completion_info["stages_completed"] == 7  # Including completion

        # Verify pipeline status
        status = pipeline_monitor.get_pipeline_status(prompt_id)
        assert status["current_stage"] == PipelineStage.COMPLETED.value
        assert status["agent_id"] == "agent789"

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """Test pipeline error recording and broadcasting."""
        prompt_id = str(uuid.uuid4())

        # Start pipeline
        pipeline_monitor.start_pipeline(prompt_id, "user123", "Test prompt")

        # Record error
        error_info = pipeline_monitor.record_error(
            prompt_id,
            "GMN validation failed: missing state nodes",
            PipelineStage.GMN_VALIDATION,
            "validation_error",
        )

        assert (
            error_info["error"] == "GMN validation failed: missing state nodes"
        )
        assert error_info["stage"] == PipelineStage.GMN_VALIDATION.value

        # Verify pipeline marked as failed
        status = pipeline_monitor.get_pipeline_status(prompt_id)
        assert status["current_stage"] == PipelineStage.FAILED.value
        assert len(status["errors"]) == 1

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_monitoring(self):
        """Test monitoring multiple concurrent pipelines."""
        user_id = "user123"
        pipeline_ids = []

        # Start multiple pipelines
        for i in range(5):
            prompt_id = str(uuid.uuid4())
            pipeline_ids.append(prompt_id)
            pipeline_monitor.start_pipeline(prompt_id, user_id, f"Prompt {i}")

        # Get active pipelines for user
        active = pipeline_monitor.get_active_pipelines(user_id)
        assert len(active) == 5

        # Complete some pipelines
        for i in range(2):
            pipeline_monitor.complete_pipeline(
                pipeline_ids[i], f"agent{i}", 1000.0 + i * 100, {}
            )

        # Check remaining active
        remaining_active = [
            p
            for p in pipeline_monitor.get_active_pipelines(user_id)
            if p["current_stage"] != PipelineStage.COMPLETED.value
        ]
        assert len(remaining_active) == 3

    @pytest.mark.asyncio
    async def test_websocket_disconnection_cleanup(
        self, connection_manager, mock_websocket
    ):
        """Test cleanup when WebSocket client disconnects."""
        client_id = "client123"
        prompt_id = "prompt456"

        # Connect and subscribe
        await connection_manager.connect(mock_websocket, client_id, {})
        pipeline_monitor.subscribe_to_pipeline(prompt_id, client_id)
        connection_manager.subscribe(client_id, "pipeline:updates")

        # Verify subscriptions
        assert client_id in pipeline_monitor.get_pipeline_subscribers(
            prompt_id
        )
        assert client_id in connection_manager.subscriptions.get(
            "pipeline:updates", set()
        )

        # Disconnect
        connection_manager.disconnect(client_id)
        pipeline_monitor.unsubscribe_from_pipeline(prompt_id, client_id)

        # Verify cleanup
        assert client_id not in connection_manager.active_connections
        assert client_id not in pipeline_monitor.get_pipeline_subscribers(
            prompt_id
        )

    @pytest.mark.asyncio
    async def test_pipeline_stage_timing_calculation(self):
        """Test calculation of stage timings."""
        prompt_id = str(uuid.uuid4())

        # Start pipeline
        pipeline_monitor.start_pipeline(prompt_id, "user123", "Test prompt")

        # Simulate stage progression with delays
        stages = [
            PipelineStage.GMN_GENERATION,
            PipelineStage.GMN_VALIDATION,
            PipelineStage.AGENT_CREATION,
        ]

        for stage in stages:
            pipeline_monitor.update_stage(prompt_id, stage)
            await asyncio.sleep(0.1)  # Small delay to simulate processing

        # Complete pipeline
        completion_info = pipeline_monitor.complete_pipeline(
            prompt_id, "agent123", 600.0, {}
        )

        # Check stage timings
        assert "stage_timings" in completion_info
        timings = completion_info["stage_timings"]

        # Should have timing for each processing stage
        for stage in stages:
            assert stage.value in timings
            assert timings[stage.value] > 0

    @pytest.mark.asyncio
    async def test_selective_event_broadcasting(self, connection_manager):
        """Test that events are only sent to relevant subscribers."""
        # Connect clients
        client1_id = "client1"
        client2_id = "client2"

        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)

        await connection_manager.connect(ws1, client1_id, {})
        await connection_manager.connect(ws2, client2_id, {})

        # Client 1 subscribes to agent events
        connection_manager.subscribe(client1_id, "agent:created")

        # Client 2 subscribes to KG events
        connection_manager.subscribe(client2_id, "knowledge_graph:updated")

        # Broadcast agent event
        await connection_manager.broadcast(
            {"type": "agent_created", "data": {"agent_id": "123"}},
            event_type="agent:created",
        )

        # Only client 1 should receive it
        ws1.send_json.assert_called_once()
        ws2.send_json.assert_not_called()

        # Reset mocks
        ws1.send_json.reset_mock()
        ws2.send_json.reset_mock()

        # Broadcast KG event
        await connection_manager.broadcast(
            {"type": "kg_updated", "data": {"nodes": 5}},
            event_type="knowledge_graph:updated",
        )

        # Only client 2 should receive it
        ws1.send_json.assert_not_called()
        ws2.send_json.assert_called_once()

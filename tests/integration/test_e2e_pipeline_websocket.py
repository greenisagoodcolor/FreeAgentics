"""End-to-end integration test for the complete pipeline with WebSocket updates.

This test simulates a real-world scenario where a frontend client connects via
WebSocket and processes a prompt while receiving real-time updates.
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from api.main import app
from auth.security_implementation import Permission, Role, TokenData
from services.websocket_integration import pipeline_monitor


class MockWebSocketClient:
    """Mock WebSocket client that captures all received messages."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.connected = False
        self.client_id = None

    async def connect(self, client_id: str, token: str):
        """Simulate WebSocket connection."""
        self.client_id = client_id
        self.connected = True
        return {"type": "connection_established", "client_id": client_id}

    async def receive_message(self):
        """Simulate receiving a message."""
        if self.messages:
            return self.messages.pop(0)
        return None

    def add_message(self, message: Dict[str, Any]):
        """Add a message to be received by the client."""
        self.messages.append(message)

    async def disconnect(self):
        """Simulate disconnection."""
        self.connected = False


@pytest.fixture
def test_client():
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_auth_token():
    """Create a mock JWT token."""
    return "mock.jwt.token"


@pytest.fixture
async def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()

    # Mock conversation query
    session.execute.return_value.scalar_one_or_none = AsyncMock(return_value=None)

    return session


class TestEndToEndPipelineWebSocket:
    """Test complete end-to-end pipeline with WebSocket updates."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_flow_with_websocket(
        self, test_client, mock_auth_token, mock_db_session
    ):
        """Test the complete flow from WebSocket connection to pipeline completion."""
        # Setup
        client_id = "test_client_123"
        ws_client = MockWebSocketClient()
        received_events = []

        # Mock WebSocket message handler
        async def capture_ws_event(message: dict, client_id: str):
            received_events.append(message)
            ws_client.add_message(message)

        # Patch dependencies
        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_current_user") as mock_get_user:
                with patch(
                    "api.v1.websocket.manager.send_personal_message",
                    capture_ws_event,
                ):
                    # Setup authenticated user
                    mock_get_user.return_value = TokenData(
                        username="testuser",
                        user_id="user123",
                        role=Role.USER,
                        permissions=[
                            Permission.CREATE_AGENT,
                            Permission.VIEW_AGENTS,
                        ],
                    )

                    # Step 1: Connect WebSocket
                    await ws_client.connect(client_id, mock_auth_token)

                    # Step 2: Subscribe to pipeline events

                    # Step 3: Send prompt request
                    prompt_data = {
                        "prompt": "Create an explorer agent for a 10x10 grid world with curiosity",
                        "iteration_count": 2,
                    }

                    response = test_client.post(
                        "/api/v1/prompts",
                        json=prompt_data,
                        headers={"Authorization": f"Bearer {mock_auth_token}"},
                    )

                    # Verify HTTP response
                    assert response.status_code == 200
                    result = response.json()
                    assert result["status"] == "success"
                    assert result["agent_id"]
                    assert result["gmn_specification"]
                    assert len(result["next_suggestions"]) > 0

                    # Step 4: Verify WebSocket events received
                    await asyncio.sleep(0.1)  # Allow async events to process

                    # Check event types received
                    event_types = [
                        e.get("type", e.get("event_type")) for e in received_events
                    ]

                    # Should have received pipeline events
                    expected_events = [
                        "pipeline:pipeline_started",
                        "pipeline:pipeline_progress",
                        "pipeline:gmn_generated",
                        "pipeline:validation_success",
                        "pipeline:agent_created",
                        "pipeline:knowledge_graph_updated",
                        "pipeline:pipeline_completed",
                    ]

                    for expected in expected_events:
                        assert any(expected in str(e) for e in event_types), (
                            f"Missing event: {expected}"
                        )

                    # Verify event sequence and content
                    pipeline_events = [
                        e
                        for e in received_events
                        if "pipeline" in str(e.get("type", ""))
                    ]

                    # Check pipeline started event
                    start_event = next(
                        (
                            e
                            for e in pipeline_events
                            if "pipeline_started" in str(e.get("type", ""))
                        ),
                        None,
                    )
                    assert start_event
                    assert start_event["data"]["total_stages"] == 6

                    # Check pipeline completed event
                    complete_event = next(
                        (
                            e
                            for e in pipeline_events
                            if "pipeline_completed" in str(e.get("type", ""))
                        ),
                        None,
                    )
                    assert complete_event
                    assert complete_event["data"]["status"] == "success"
                    assert complete_event["data"]["agent_id"] == result["agent_id"]
                    assert (
                        complete_event["data"]["processing_time_ms"] < 3000
                    )  # Under 3s requirement

    @pytest.mark.asyncio
    async def test_realtime_progress_tracking(
        self, test_client, mock_auth_token, mock_db_session
    ):
        """Test real-time progress tracking through WebSocket updates."""
        progress_updates = []

        # Capture progress events
        async def capture_progress(message: dict, client_id: str):
            if "pipeline_progress" in message.get("type", ""):
                progress_updates.append(message["data"])

        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_current_user") as mock_get_user:
                with patch(
                    "api.v1.websocket.manager.send_personal_message",
                    capture_progress,
                ):
                    # Setup user
                    mock_get_user.return_value = TokenData(
                        username="testuser",
                        user_id="user123",
                        role=Role.USER,
                        permissions=[Permission.CREATE_AGENT],
                    )

                    # Process prompt
                    response = test_client.post(
                        "/api/v1/prompts",
                        json={"prompt": "Create a simple agent"},
                        headers={"Authorization": f"Bearer {mock_auth_token}"},
                    )

                    assert response.status_code == 200

                    # Verify progress updates
                    await asyncio.sleep(0.1)

                    # Should have progress for each stage
                    assert len(progress_updates) >= 6

                    # Verify stage progression
                    stage_numbers = [u["stage_number"] for u in progress_updates]
                    assert sorted(stage_numbers) == list(
                        range(1, len(stage_numbers) + 1)
                    )

                    # Verify each update has required fields
                    for update in progress_updates:
                        assert "stage" in update
                        assert "stage_number" in update
                        assert "message" in update
                        assert update["stage_number"] <= 6

    @pytest.mark.asyncio
    async def test_error_handling_with_websocket_notification(
        self, test_client, mock_auth_token, mock_db_session
    ):
        """Test that errors are properly communicated via WebSocket."""
        error_events = []

        # Capture error events
        async def capture_errors(message: dict, client_id: str):
            if "failed" in message.get("type", "") or "error" in message.get(
                "type", ""
            ):
                error_events.append(message)

        # Force an error in GMN generation
        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_current_user") as mock_get_user:
                with patch(
                    "api.v1.websocket.manager.send_personal_message",
                    capture_errors,
                ):
                    with patch(
                        "services.gmn_generator.GMNGenerator.prompt_to_gmn"
                    ) as mock_gmn:
                        # Setup user
                        mock_get_user.return_value = TokenData(
                            username="testuser",
                            user_id="user123",
                            role=Role.USER,
                            permissions=[Permission.CREATE_AGENT],
                        )

                        # Make GMN generation fail
                        mock_gmn.side_effect = ValueError(
                            "Invalid prompt: cannot generate GMN"
                        )

                        # Process prompt
                        response = test_client.post(
                            "/api/v1/prompts",
                            json={"prompt": "This will fail"},
                            headers={"Authorization": f"Bearer {mock_auth_token}"},
                        )

                        # Should return error
                        assert response.status_code == 400

                        # Verify error event sent via WebSocket
                        await asyncio.sleep(0.1)
                        assert len(error_events) > 0

                        error_event = error_events[0]
                        assert "error" in error_event["data"]
                        assert "validation_error" in str(
                            error_event["data"].get("error_type", "")
                        )

    @pytest.mark.asyncio
    async def test_performance_monitoring_via_websocket(
        self, test_client, mock_auth_token, mock_db_session
    ):
        """Test that performance metrics are communicated via WebSocket."""
        performance_events = []

        # Capture all events
        async def capture_all(message: dict, client_id: str):
            performance_events.append(message)

        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_current_user") as mock_get_user:
                with patch(
                    "api.v1.websocket.manager.send_personal_message",
                    capture_all,
                ):
                    # Setup user
                    mock_get_user.return_value = TokenData(
                        username="testuser",
                        user_id="user123",
                        role=Role.USER,
                        permissions=[Permission.CREATE_AGENT],
                    )

                    # Time the request
                    start_time = time.time()

                    response = test_client.post(
                        "/api/v1/prompts",
                        json={"prompt": "Create a fast agent"},
                        headers={"Authorization": f"Bearer {mock_auth_token}"},
                    )

                    end_time = time.time()
                    request_time = (end_time - start_time) * 1000

                    assert response.status_code == 200

                    # Check performance in response
                    result = response.json()
                    assert result["processing_time_ms"] < 3000
                    assert request_time < 3000  # Total request under 3s

                    # Check performance in WebSocket events
                    await asyncio.sleep(0.1)

                    # Find completion event
                    completion_event = next(
                        (
                            e
                            for e in performance_events
                            if "pipeline_completed" in str(e.get("type", ""))
                        ),
                        None,
                    )

                    assert completion_event
                    assert completion_event["data"]["processing_time_ms"] < 3000

    @pytest.mark.asyncio
    async def test_multiple_clients_monitoring_same_pipeline(
        self, test_client, mock_auth_token, mock_db_session
    ):
        """Test multiple WebSocket clients monitoring the same pipeline."""
        client_events = {"client1": [], "client2": [], "client3": []}

        # Route events to different clients
        async def route_events(message: dict, client_id: str):
            if client_id in client_events:
                client_events[client_id].append(message)

        with patch("api.v1.prompts.get_db", return_value=mock_db_session):
            with patch("api.v1.prompts.get_current_user") as mock_get_user:
                with patch(
                    "api.v1.websocket.manager.send_personal_message",
                    route_events,
                ):
                    with patch("api.v1.websocket.manager.broadcast") as mock_broadcast:
                        # Setup broadcast to simulate sending to all clients
                        async def broadcast_to_all(message, event_type=None):
                            for client_id in client_events:
                                await route_events(message, client_id)

                        mock_broadcast.side_effect = broadcast_to_all

                        # Setup user
                        mock_get_user.return_value = TokenData(
                            username="testuser",
                            user_id="user123",
                            role=Role.USER,
                            permissions=[Permission.CREATE_AGENT],
                        )

                        # Process prompt
                        response = test_client.post(
                            "/api/v1/prompts",
                            json={"prompt": "Create a shared agent"},
                            headers={"Authorization": f"Bearer {mock_auth_token}"},
                        )

                        assert response.status_code == 200

                        # Wait for events
                        await asyncio.sleep(0.1)

                        # Verify all clients received events
                        for client_id, events in client_events.items():
                            assert len(events) > 0, (
                                f"Client {client_id} received no events"
                            )

                            # Each client should receive similar events
                            event_types = [e.get("type", "") for e in events]
                            assert any("pipeline" in et for et in event_types)

    @pytest.mark.asyncio
    async def test_pipeline_monitoring_dashboard_data(self):
        """Test that pipeline monitoring provides dashboard-ready data."""
        # Start multiple pipelines
        pipeline_ids = []
        for i in range(3):
            pid = f"prompt_{i}"
            pipeline_ids.append(pid)
            pipeline_monitor.start_pipeline(pid, "user123", f"Test prompt {i}")

            # Simulate progress
            if i == 0:
                # Complete first pipeline
                pipeline_monitor.update_stage(pid, PipelineStage.GMN_GENERATION)
                pipeline_monitor.update_stage(pid, PipelineStage.AGENT_CREATION)
                pipeline_monitor.complete_pipeline(pid, f"agent_{i}", 1200.0, {})
            elif i == 1:
                # Leave second in progress
                pipeline_monitor.update_stage(pid, PipelineStage.GMN_GENERATION)
                pipeline_monitor.update_stage(pid, PipelineStage.GMN_VALIDATION)
            else:
                # Mark third as failed
                pipeline_monitor.update_stage(pid, PipelineStage.GMN_GENERATION)
                pipeline_monitor.record_error(
                    pid, "Validation failed", PipelineStage.GMN_VALIDATION
                )

        # Get dashboard data
        active_pipelines = pipeline_monitor.get_active_pipelines("user123")

        # Analyze pipeline states
        states = {}
        for p in active_pipelines:
            state = p["current_stage"]
            states[state] = states.get(state, 0) + 1

        # Should have one of each state
        assert states.get("completed", 0) >= 1
        assert states.get("failed", 0) >= 1
        assert sum(states.values()) >= 3

        # Get specific pipeline details
        completed_pipeline = pipeline_monitor.get_pipeline_status(pipeline_ids[0])
        assert completed_pipeline["current_stage"] == "completed"
        assert completed_pipeline["agent_id"] == "agent_0"
        assert completed_pipeline["processing_time_ms"] == 1200.0

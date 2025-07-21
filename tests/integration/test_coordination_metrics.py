"""Test coordination metrics integration."""

import asyncio
import logging
import time

import pytest

from agents.coalition_coordinator import CoalitionCoordinatorAgent
from observability.coordination_metrics import (
    coordination_metrics,
    get_agent_coordination_stats,
    get_system_coordination_report,
    record_coalition_event,
    record_coordination,
)


@pytest.fixture
def coordinator_agent():
    """Create a coalition coordinator agent."""
    agent = CoalitionCoordinatorAgent("coordinator-1", "Coordinator 1")
    agent.start()
    yield agent
    agent.stop()


@pytest.mark.asyncio
async def test_coordination_metrics_basic():
    """Test basic coordination metrics collection."""
    coordinator_id = "test-coordinator"
    participant_ids = ["agent-1", "agent-2", "agent-3"]

    # Record a coordination operation
    start_time = time.time()
    await asyncio.sleep(0.01)  # Simulate coordination time

    await record_coordination(
        coordinator_id=coordinator_id,
        participant_ids=participant_ids,
        coordination_type="task_assignment",
        start_time=start_time,
        success=True,
        metadata={"task": "explore"},
    )

    # Check statistics
    stats = get_agent_coordination_stats(coordinator_id)

    assert stats["agent_id"] == coordinator_id
    assert stats["coordinations_initiated"] == 1
    assert stats["success_rate"] == 1.0
    assert stats["avg_coordination_time_ms"] > 0


@pytest.mark.asyncio
async def test_coalition_formation_metrics():
    """Test coalition formation metrics."""
    coalition_id = "coalition-1"
    coordinator_id = "coordinator-1"
    member_ids = ["agent-1", "agent-2", "agent-3", "agent-4"]

    # Record coalition formation
    await record_coalition_event(
        event_type="formation",
        coalition_id=coalition_id,
        coordinator_id=coordinator_id,
        member_ids=member_ids,
        duration_ms=150.0,
        success=True,
    )

    # Get coalition statistics
    report = get_system_coordination_report()
    coalition_stats = report["coalition_stats"]

    assert coalition_stats["active_coalitions"] == 1
    assert coalition_stats["avg_member_count"] == 4
    assert coalition_stats["avg_efficiency"] == 1.0  # Initial efficiency


@pytest.mark.asyncio
async def test_coalition_efficiency_updates():
    """Test coalition efficiency metric updates."""
    coalition_id = "coalition-2"
    coordinator_id = "coordinator-2"
    member_ids = ["agent-5", "agent-6"]

    # Form coalition
    await record_coalition_event(
        event_type="formation",
        coalition_id=coalition_id,
        coordinator_id=coordinator_id,
        member_ids=member_ids,
        duration_ms=100.0,
        success=True,
    )

    # Update coalition efficiency
    await coordination_metrics.update_coalition_efficiency(
        coalition_id=coalition_id,
        task_completed=True,
        coordination_overhead_ms=30.0,
    )

    # Check that metrics were updated
    coalition = coordination_metrics.coalition_metrics.get(coalition_id)
    assert coalition is not None
    assert coalition.task_completion_rate > 0
    assert coalition.communication_overhead > 0


@pytest.mark.asyncio
async def test_inter_agent_messaging_metrics():
    """Test inter-agent messaging metrics."""
    sender_id = "agent-sender"
    receiver_id = "agent-receiver"

    # Record multiple messages
    for i in range(5):
        await coordination_metrics.record_inter_agent_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type="coordination_request",
            message_size_bytes=128,
            latency_ms=5.0 + i,
        )

    # Check statistics
    sender_stats = get_agent_coordination_stats(sender_id)
    receiver_stats = get_agent_coordination_stats(receiver_id)

    assert sender_stats["messages_sent"] == 5
    assert receiver_stats["messages_received"] == 5


@pytest.mark.asyncio
async def test_failed_coordination_metrics():
    """Test metrics for failed coordinations."""
    coordinator_id = "test-coordinator-fail"
    participant_ids = ["agent-fail-1", "agent-fail-2"]

    # Record successful coordination
    start_time = time.time()
    await record_coordination(
        coordinator_id=coordinator_id,
        participant_ids=participant_ids,
        coordination_type="task_assignment",
        start_time=start_time,
        success=True,
    )

    # Record failed coordination
    start_time = time.time()
    await record_coordination(
        coordinator_id=coordinator_id,
        participant_ids=participant_ids,
        coordination_type="task_assignment",
        start_time=start_time,
        success=False,
    )

    # Check statistics
    stats = get_agent_coordination_stats(coordinator_id)

    assert stats["coordinations_initiated"] == 2
    assert stats["success_rate"] == 0.5


@pytest.mark.asyncio
async def test_coalition_dissolution_metrics():
    """Test coalition dissolution metrics."""
    coalition_id = "coalition-dissolve"
    coordinator_id = "coordinator-dissolve"
    member_ids = ["agent-d1", "agent-d2"]

    # Form coalition
    await record_coalition_event(
        event_type="formation",
        coalition_id=coalition_id,
        coordinator_id=coordinator_id,
        member_ids=member_ids,
        duration_ms=100.0,
        success=True,
    )

    # Verify coalition exists
    assert coalition_id in coordination_metrics.coalition_metrics

    # Dissolve coalition
    await record_coalition_event(
        event_type="dissolution",
        coalition_id=coalition_id,
        coordinator_id=coordinator_id,
        member_ids=member_ids,
        duration_ms=50.0,
        success=True,
    )

    # Verify coalition was removed
    assert coalition_id not in coordination_metrics.coalition_metrics


@pytest.mark.asyncio
async def test_system_coordination_report():
    """Test system-wide coordination report."""
    # Generate some coordination activity
    coordinators = ["coord-1", "coord-2", "coord-3"]

    for i, coord_id in enumerate(coordinators):
        participants = [f"agent-{i}-{j}" for j in range(3)]

        # Record coordinations
        start_time = time.time()
        await record_coordination(
            coordinator_id=coord_id,
            participant_ids=participants,
            coordination_type="exploration",
            start_time=start_time,
            success=i != 1,  # Make second one fail
        )

    # Get system report
    report = get_system_coordination_report()

    assert report["active_agents"] >= 3
    assert report["total_coordinations"] >= 3
    assert 0 < report["system_success_rate"] < 1.0  # Some failed
    assert "coalition_stats" in report
    assert "performance_thresholds" in report


@pytest.mark.asyncio
async def test_coordination_with_agent_integration(coordinator_agent):
    """Test coordination metrics with actual coordinator agent."""
    # Simulate agent discovering other agents
    coordinator_agent.known_agents = {
        "agent-1": {
            "position": [1, 1],
            "type": "explorer",
            "status": "active",
        },
        "agent-2": {
            "position": [2, 2],
            "type": "explorer",
            "status": "active",
        },
    }

    # Record coordination initiated by the agent
    start_time = time.time()

    # Simulate agent forming a coalition
    action = coordinator_agent.select_action()

    # Record the coordination
    await record_coordination(
        coordinator_id=coordinator_agent.agent_id,
        participant_ids=list(coordinator_agent.known_agents.keys()),
        coordination_type="coalition_formation",
        start_time=start_time,
        success=True,
        metadata={"action": action},
    )

    # Check statistics
    stats = get_agent_coordination_stats(coordinator_agent.agent_id)
    assert stats["coordinations_initiated"] > 0


@pytest.mark.asyncio
async def test_performance_threshold_warnings(caplog):
    """Test that performance warnings are logged for slow operations."""
    coordinator_id = "slow-coordinator"
    participant_ids = ["agent-1", "agent-2"]

    # Set low threshold for testing
    coordination_metrics.thresholds["coordination_time_ms"] = 10.0

    # Record slow coordination
    start_time = time.time()
    await asyncio.sleep(0.025)  # 25ms delay

    with caplog.at_level(logging.WARNING):
        await record_coordination(
            coordinator_id=coordinator_id,
            participant_ids=participant_ids,
            coordination_type="slow_operation",
            start_time=start_time,
            success=True,
        )

    # Check that warning was logged
    assert any("Slow coordination detected" in record.message for record in caplog.records)

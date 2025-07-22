"""Test agent inference metrics integration."""

import asyncio
import time

import numpy as np
import pytest

from agents.base_agent import BasicExplorerAgent
from observability.performance_metrics import (
    get_agent_report,
    get_performance_report,
    performance_tracker,
    record_belief_metric,
    record_inference_metric,
    record_step_metric,
    start_performance_tracking,
    stop_performance_tracking,
)


@pytest.fixture
async def metrics_tracker():
    """Setup and cleanup performance tracker."""
    await start_performance_tracking()
    yield performance_tracker
    await stop_performance_tracking()


@pytest.mark.asyncio
async def test_inference_metrics_collection(metrics_tracker):
    """Test that inference metrics are collected during agent operations."""
    # Create test agent
    agent = BasicExplorerAgent("test-agent-1", "Explorer 1")
    agent.start()

    # Simulate inference with timing
    start_time = time.time()
    observation = {"grid": np.zeros((10, 10))}
    agent.perceive(observation)
    agent.update_beliefs()
    agent.select_action()
    inference_time_ms = (time.time() - start_time) * 1000

    # Record the metric
    await record_inference_metric(
        agent_id=agent.agent_id,
        inference_time_ms=inference_time_ms,
        success=True,
    )

    # Allow time for metric collection
    await asyncio.sleep(0.1)

    # Verify metrics were collected
    agent_report = await get_agent_report(agent.agent_id)
    assert agent_report["agent_id"] == "test-agent-1"
    assert agent_report["inference_performance"]["count"] > 0
    assert agent_report["inference_performance"]["avg"] > 0

    agent.stop()


@pytest.mark.asyncio
async def test_belief_update_metrics(metrics_tracker):
    """Test belief update metrics collection."""
    agent = BasicExplorerAgent("test-agent-2", "Explorer 2")
    agent.start()

    # Perform belief update
    start_time = time.time()
    observation = {"grid": np.zeros((10, 10))}
    agent.perceive(observation)
    agent.update_beliefs()
    update_time_ms = (time.time() - start_time) * 1000

    # Compute free energy
    free_energy_components = agent.compute_free_energy()
    free_energy = free_energy_components.get("total_free_energy", 0.0)

    # Record belief update metric
    await record_belief_metric(
        agent_id=agent.agent_id,
        update_time_ms=update_time_ms,
        free_energy=free_energy,
    )

    await asyncio.sleep(0.1)

    # Verify metrics
    agent_report = await get_agent_report(agent.agent_id)
    assert agent_report["belief_update_rate"]["count"] > 0

    system_report = await get_performance_report()
    assert system_report["system_counters"]["total_belief_updates"] > 0

    agent.stop()


@pytest.mark.asyncio
async def test_agent_step_metrics(metrics_tracker):
    """Test agent step metrics collection."""
    agent = BasicExplorerAgent("test-agent-3", "Explorer 3")
    agent.start()

    # Perform complete agent step
    start_time = time.time()
    observation = {"grid": np.zeros((10, 10))}
    agent.step(observation)
    step_time_ms = (time.time() - start_time) * 1000

    # Record step metric
    await record_step_metric(agent_id=agent.agent_id, step_time_ms=step_time_ms)

    await asyncio.sleep(0.1)

    # Verify metrics
    agent_report = await get_agent_report(agent.agent_id)
    assert agent_report["action_rate"]["count"] > 0
    assert agent_report["total_steps"] > 0

    agent.stop()


@pytest.mark.asyncio
async def test_performance_alerts(metrics_tracker):
    """Test performance alert generation."""
    agent = BasicExplorerAgent("test-agent-4", "Explorer 4")
    agent.start()

    # Set low baseline to trigger alert
    performance_tracker.update_baselines({"inference_time_ms": 1.0})

    # Simulate slow inference
    await record_inference_metric(
        agent_id=agent.agent_id,
        inference_time_ms=10.0,
        success=True,  # 10x baseline
    )

    # Force alert check
    await performance_tracker._check_performance_alerts()

    # Check system report for alerts
    system_report = await get_performance_report()
    assert system_report is not None

    agent.stop()


@pytest.mark.asyncio
async def test_error_metrics_collection(metrics_tracker):
    """Test error metrics collection during failed operations."""
    agent = BasicExplorerAgent("test-agent-5", "Explorer 5")
    agent.start()

    # Record failed inference
    await record_inference_metric(
        agent_id=agent.agent_id,
        inference_time_ms=5.0,
        success=False,
        error="Test error",
    )

    await asyncio.sleep(0.1)

    # Verify error metrics
    agent_report = await get_agent_report(agent.agent_id)
    assert agent_report["error_rate"]["count"] > 0
    assert agent_report["total_errors"] > 0

    system_report = await get_performance_report()
    assert system_report["system_counters"]["total_errors"] > 0

    agent.stop()


@pytest.mark.asyncio
async def test_multi_agent_metrics(metrics_tracker):
    """Test metrics collection with multiple agents."""
    agents = []

    # Create multiple agents
    for i in range(3):
        agent = BasicExplorerAgent(f"test-agent-multi-{i}", f"Explorer {i}")
        agent.start()
        agents.append(agent)

    # Simulate operations on all agents
    for i, agent in enumerate(agents):
        observation = {"grid": np.zeros((10, 10))}
        agent.step(observation)

        await record_inference_metric(
            agent_id=agent.agent_id, inference_time_ms=5.0 + i, success=True
        )

    await asyncio.sleep(0.1)

    # Verify system-wide metrics
    system_report = await get_performance_report()
    assert system_report["agent_count"] >= 3
    assert system_report["system_counters"]["total_inferences"] >= 3

    # Verify individual agent metrics
    for i, agent in enumerate(agents):
        agent_report = await get_agent_report(agent.agent_id)
        assert agent_report["agent_id"] == f"test-agent-multi-{i}"
        assert agent_report["inference_performance"]["count"] > 0

    # Cleanup
    for agent in agents:
        agent.stop()


@pytest.mark.asyncio
async def test_metrics_export(metrics_tracker):
    """Test metrics export functionality."""
    agent = BasicExplorerAgent("test-agent-export", "Explorer Export")
    agent.start()

    # Generate some metrics
    for i in range(5):
        await record_inference_metric(
            agent_id=agent.agent_id, inference_time_ms=5.0 + i, success=True
        )

    await asyncio.sleep(0.1)

    # Export metrics
    export_data = performance_tracker.export_metrics(format="json")
    assert export_data is not None
    assert "system_counters" in export_data
    assert "metrics_summary" in export_data

    agent.stop()


@pytest.mark.asyncio
async def test_performance_snapshot(metrics_tracker):
    """Test performance snapshot generation."""
    agent = BasicExplorerAgent("test-agent-snapshot", "Explorer Snapshot")
    agent.start()

    # Generate activity
    observation = {"grid": np.zeros((10, 10))}
    agent.step(observation)

    await record_inference_metric(agent_id=agent.agent_id, inference_time_ms=5.0, success=True)

    await asyncio.sleep(0.1)

    # Get snapshot
    snapshot = await performance_tracker.get_current_performance_snapshot()

    assert snapshot.timestamp is not None
    assert snapshot.active_agents > 0
    assert hasattr(snapshot, "inference_time_ms")
    assert hasattr(snapshot, "memory_usage_mb")
    assert hasattr(snapshot, "cpu_usage_percent")

    agent.stop()

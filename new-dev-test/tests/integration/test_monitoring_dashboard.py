"""Test monitoring dashboard functionality."""

import asyncio
from datetime import datetime

import pytest
from observability.monitoring_dashboard import (
    MonitoringDashboard,
    get_dashboard_data,
    get_metric_time_series,
    monitoring_dashboard,
    record_dashboard_event,
    start_monitoring_dashboard,
    stop_monitoring_dashboard,
)
from observability.performance_metrics import (
    record_inference_metric,
    start_performance_tracking,
    stop_performance_tracking,
)

from agents.base_agent import BasicExplorerAgent


@pytest.fixture
async def dashboard():
    """Setup and cleanup monitoring dashboard."""
    # Start performance tracking first
    await start_performance_tracking()
    await start_monitoring_dashboard()
    yield monitoring_dashboard
    await stop_monitoring_dashboard()
    await stop_performance_tracking()


@pytest.mark.asyncio
async def test_dashboard_startup_shutdown():
    """Test dashboard startup and shutdown."""
    dashboard = MonitoringDashboard()

    assert not dashboard.running

    await dashboard.start()
    assert dashboard.running

    await dashboard.stop()
    assert not dashboard.running


@pytest.mark.asyncio
async def test_dashboard_metrics_collection(dashboard):
    """Test that dashboard collects metrics."""
    # Create and run an agent
    agent = BasicExplorerAgent("dashboard-test-agent", "Test Agent")
    agent.start()

    # Generate some metrics
    for i in range(3):
        await record_inference_metric(
            agent_id=agent.agent_id, inference_time_ms=10.0 + i, success=True
        )

    # Wait for dashboard update (give it more time on first run)
    await asyncio.sleep(2.5)

    # Get dashboard data
    data = get_dashboard_data()

    assert data is not None
    assert "system_metrics" in data
    assert "agent_dashboards" in data
    assert "timestamp" in data

    # Check system metrics
    assert "total_agents" in data["system_metrics"]
    assert data["system_metrics"]["total_agents"]["value"] >= 1

    agent.stop()


@pytest.mark.asyncio
async def test_metric_status_calculation(dashboard):
    """Test metric status calculation based on thresholds."""
    # Test status calculation - use the global instance
    assert monitoring_dashboard._get_metric_status("inference_time_ms", 30) == "healthy"
    assert monitoring_dashboard._get_metric_status("inference_time_ms", 60) == "warning"
    assert monitoring_dashboard._get_metric_status("inference_time_ms", 110) == "critical"

    assert monitoring_dashboard._get_metric_status("cpu_usage_percent", 50) == "healthy"
    assert monitoring_dashboard._get_metric_status("cpu_usage_percent", 75) == "warning"
    assert monitoring_dashboard._get_metric_status("cpu_usage_percent", 95) == "critical"


@pytest.mark.asyncio
async def test_dashboard_alerts(dashboard):
    """Test alert generation."""
    # Create agent with high latency
    agent = BasicExplorerAgent("alert-test-agent", "Alert Test")
    agent.start()

    # Generate high latency metrics
    await record_inference_metric(
        agent_id=agent.agent_id,
        inference_time_ms=150.0,
        success=True,  # Above critical threshold
    )

    # Wait for dashboard update
    await asyncio.sleep(1.5)

    # Check alerts
    data = get_dashboard_data()
    assert data is not None
    assert "alerts" in data

    # Should have at least one alert for high inference time
    alerts = data["alerts"]
    critical_alerts = [a for a in alerts if a["level"] == "critical"]
    assert len(critical_alerts) > 0

    agent.stop()


@pytest.mark.asyncio
async def test_dashboard_events(dashboard):
    """Test dashboard event recording."""
    # Record some events
    record_dashboard_event("agent_created", {"agent_id": "test-1"})
    record_dashboard_event("coalition_formed", {"coalition_id": "coal-1"})
    record_dashboard_event("task_completed", {"task_id": "task-1"})

    # Wait a bit
    await asyncio.sleep(0.1)

    # Get dashboard data
    data = get_dashboard_data()
    assert data is not None
    assert "recent_events" in data

    # Check events
    events = data["recent_events"]
    event_types = [e["type"] for e in events]

    assert "agent_created" in event_types
    assert "coalition_formed" in event_types
    assert "task_completed" in event_types


@pytest.mark.asyncio
async def test_metric_history(dashboard):
    """Test metric history tracking."""
    # Generate metrics over time
    agent = BasicExplorerAgent("history-test-agent", "History Test")
    agent.start()

    for i in range(5):
        await record_inference_metric(
            agent_id=agent.agent_id,
            inference_time_ms=10.0 + i * 2,
            success=True,
        )
        await asyncio.sleep(0.5)

    # Wait for final update
    await asyncio.sleep(1.0)

    # Get metric history
    history = get_metric_time_series("avg_inference_time", duration_minutes=5)

    assert history is not None
    assert "timestamps" in history
    assert "values" in history
    assert len(history["timestamps"]) > 0
    assert len(history["values"]) == len(history["timestamps"])

    agent.stop()


@pytest.mark.asyncio
async def test_dashboard_agent_details(dashboard):
    """Test individual agent dashboard details."""
    # Create multiple agents
    agents = []
    for i in range(3):
        agent = BasicExplorerAgent(f"detail-agent-{i}", f"Agent {i}")
        agent.start()
        agents.append(agent)

        # Generate different metrics for each
        await record_inference_metric(
            agent_id=agent.agent_id,
            inference_time_ms=10.0 * (i + 1),
            success=True,
        )

    # Wait for dashboard update
    await asyncio.sleep(1.5)

    # Get dashboard data
    data = get_dashboard_data()
    assert data is not None
    assert "agent_dashboards" in data

    # Check each agent has a dashboard
    agent_dashboards = data["agent_dashboards"]
    for agent in agents:
        assert agent.agent_id in agent_dashboards

        agent_dash = agent_dashboards[agent.agent_id]
        assert "current_metrics" in agent_dash
        assert "inference_time" in agent_dash["current_metrics"]
        assert agent_dash["status"] == "active"

    # Cleanup
    for agent in agents:
        agent.stop()


@pytest.mark.asyncio
async def test_dashboard_coalition_tracking(dashboard):
    """Test coalition tracking in dashboard."""
    from observability.coordination_metrics import record_coalition_event

    # Record coalition formation
    await record_coalition_event(
        event_type="formation",
        coalition_id="test-coalition-1",
        coordinator_id="coordinator-1",
        member_ids=["agent-1", "agent-2", "agent-3"],
        duration_ms=100.0,
        success=True,
    )

    # Wait for dashboard update
    await asyncio.sleep(1.5)

    # Get dashboard data
    data = get_dashboard_data()
    assert data is not None
    assert "active_coalitions" in data

    # Check coalition is tracked
    coalitions = data["active_coalitions"]
    assert len(coalitions) > 0

    coalition = coalitions[0]
    assert coalition["coalition_id"] == "test-coalition-1"
    assert coalition["member_count"] == 3
    assert coalition["efficiency"] == 1.0  # Initial efficiency


@pytest.mark.asyncio
async def test_dashboard_performance(dashboard):
    """Test dashboard performance with many agents."""
    # Create many agents
    agents = []
    for i in range(10):
        agent = BasicExplorerAgent(f"perf-agent-{i}", f"Perf Agent {i}")
        agent.start()
        agents.append(agent)

    # Generate lots of metrics
    datetime.now()

    for _ in range(5):
        for agent in agents:
            await record_inference_metric(
                agent_id=agent.agent_id, inference_time_ms=10.0, success=True
            )
        await asyncio.sleep(0.1)

    # Wait for dashboard update
    await asyncio.sleep(1.5)

    # Measure dashboard update time
    update_start = datetime.now()
    data = get_dashboard_data()
    update_time = (datetime.now() - update_start).total_seconds()

    assert data is not None
    assert update_time < 0.1  # Should be fast
    assert len(data["agent_dashboards"]) >= 10

    # Cleanup
    for agent in agents:
        agent.stop()


@pytest.mark.asyncio
async def test_dashboard_error_handling(dashboard):
    """Test dashboard handles errors gracefully."""
    # Test with no data
    await stop_monitoring_dashboard()
    data = get_dashboard_data()
    assert data is None

    # Restart
    await start_monitoring_dashboard()

    # Test with invalid metric name
    history = get_metric_time_series("invalid_metric_name")
    assert history is None

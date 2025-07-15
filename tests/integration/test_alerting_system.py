"""Test alerting system for agent failures and performance issues."""

import asyncio

import pytest

from agents.base_agent import BasicExplorerAgent
from observability.alerting_system import (
    AlertLevel,
    AlertType,
    alerting_system,
    check_agent_alert,
    check_system_alert,
    get_active_alerts,
    get_alert_stats,
    resolve_alert,
)
from observability.performance_metrics import (
    record_inference_metric,
    start_performance_tracking,
    stop_performance_tracking,
)


@pytest.fixture
async def setup_alerting():
    """Setup alerting and performance tracking."""
    await start_performance_tracking()
    # Clear any existing alerts
    alerting_system.alerts.clear()
    alerting_system.alert_history.clear()
    alerting_system.last_alert_times.clear()
    yield alerting_system
    await stop_performance_tracking()


@pytest.mark.asyncio
async def test_agent_failure_alert(setup_alerting):
    """Test alert generation for agent failures."""
    agent_id = "test-fail-agent"

    # Simulate agent failure
    await check_agent_alert(
        agent_id, {"agent_id": agent_id, "agent_status": "failed", "error": "Test failure"}
    )

    # Check that alert was created
    alerts = get_active_alerts()
    assert len(alerts) > 0

    # Find the failure alert
    failure_alerts = [a for a in alerts if a["type"] == AlertType.AGENT_FAILURE.value]
    assert len(failure_alerts) == 1

    alert = failure_alerts[0]
    assert alert["level"] == AlertLevel.CRITICAL.value
    assert agent_id in alert["title"]
    assert "Test failure" in alert["message"]


@pytest.mark.asyncio
async def test_performance_degradation_alert(setup_alerting):
    """Test alert generation for performance degradation."""
    agent_id = "test-slow-agent"

    # Simulate high inference time
    await check_agent_alert(
        agent_id,
        {
            "agent_id": agent_id,
            "inference_time_ms": 150,  # Above warning threshold
            "agent_status": "active",
        },
    )

    # Check alerts
    alerts = get_active_alerts()
    perf_alerts = [a for a in alerts if a["type"] == AlertType.PERFORMANCE_DEGRADATION.value]
    assert len(perf_alerts) == 1

    alert = perf_alerts[0]
    assert alert["level"] == AlertLevel.WARNING.value
    assert "150" in alert["message"]


@pytest.mark.asyncio
async def test_critical_performance_alert(setup_alerting):
    """Test critical performance alerts."""
    agent_id = "test-critical-agent"

    # Simulate very high inference time
    await check_agent_alert(
        agent_id,
        {
            "agent_id": agent_id,
            "inference_time_ms": 250,  # Above critical threshold
            "agent_status": "active",
        },
    )

    # Check alerts
    alerts = get_active_alerts()
    critical_alerts = [
        a
        for a in alerts
        if a["type"] == AlertType.PERFORMANCE_DEGRADATION.value
        and a["level"] == AlertLevel.CRITICAL.value
    ]
    assert len(critical_alerts) == 1


@pytest.mark.asyncio
async def test_belief_anomaly_alert(setup_alerting):
    """Test belief anomaly alerts."""
    agent_id = "test-anomaly-agent"

    # Simulate belief anomaly
    await check_agent_alert(
        agent_id,
        {
            "agent_id": agent_id,
            "belief_anomaly": True,
            "kl_divergence": 5.2,
            "agent_status": "active",
        },
    )

    # Check alerts
    alerts = get_active_alerts()
    anomaly_alerts = [a for a in alerts if a["type"] == AlertType.BELIEF_ANOMALY.value]
    assert len(anomaly_alerts) == 1

    alert = anomaly_alerts[0]
    assert alert["level"] == AlertLevel.WARNING.value
    assert "5.2" in alert["message"]


@pytest.mark.asyncio
async def test_resource_exhaustion_alert(setup_alerting):
    """Test resource exhaustion alerts."""
    # Simulate high memory usage
    await check_system_alert(
        {"memory_usage_mb": 850, "cpu_usage_percent": 60}  # Above warning threshold
    )

    # Check alerts
    alerts = get_active_alerts()
    resource_alerts = [a for a in alerts if a["type"] == AlertType.RESOURCE_EXHAUSTION.value]
    assert len(resource_alerts) == 1

    alert = resource_alerts[0]
    assert alert["level"] == AlertLevel.WARNING.value
    assert "850MB" in alert["message"]


@pytest.mark.asyncio
async def test_alert_cooldown(setup_alerting):
    """Test alert cooldown mechanism."""
    agent_id = "test-cooldown-agent"

    # Generate first alert
    await check_agent_alert(
        agent_id, {"agent_id": agent_id, "inference_time_ms": 150, "agent_status": "active"}
    )

    initial_count = len(get_active_alerts())

    # Try to generate same alert immediately
    await check_agent_alert(
        agent_id, {"agent_id": agent_id, "inference_time_ms": 160, "agent_status": "active"}
    )

    # Should not create new alert due to cooldown
    assert len(get_active_alerts()) == initial_count


@pytest.mark.asyncio
async def test_alert_resolution(setup_alerting):
    """Test alert resolution."""
    agent_id = "test-resolve-agent"

    # Generate alert
    await check_agent_alert(
        agent_id, {"agent_id": agent_id, "agent_status": "failed", "error": "Test error"}
    )

    alerts = get_active_alerts()
    assert len(alerts) > 0

    alert_id = alerts[0]["alert_id"]

    # Resolve alert
    resolve_alert(alert_id, "Issue fixed")

    # Check active alerts - the resolved alert should not be in the list
    active_alerts = get_active_alerts()
    active_alert_ids = [a["alert_id"] for a in active_alerts]
    assert alert_id not in active_alert_ids

    # Check that alert is marked as resolved
    resolved_alert = alerting_system.alerts[alert_id]
    assert resolved_alert.resolved
    assert resolved_alert.resolution_notes == "Issue fixed"


@pytest.mark.asyncio
async def test_alert_statistics(setup_alerting):
    """Test alert statistics."""
    # Generate various alerts
    await check_agent_alert(
        "agent-1", {"agent_id": "agent-1", "agent_status": "failed", "error": "Error 1"}
    )

    await check_agent_alert("agent-2", {"agent_id": "agent-2", "inference_time_ms": 150})

    await check_system_alert({"memory_usage_mb": 850})

    # Get statistics
    stats = get_alert_stats()

    assert stats["total_active"] >= 3
    assert stats["by_level"]["critical"] >= 1
    assert stats["by_level"]["warning"] >= 2
    assert stats["by_type"][AlertType.AGENT_FAILURE.value] >= 1
    assert stats["by_type"][AlertType.PERFORMANCE_DEGRADATION.value] >= 1
    assert stats["by_type"][AlertType.RESOURCE_EXHAUSTION.value] >= 1


@pytest.mark.asyncio
async def test_alert_export(setup_alerting):
    """Test alert export functionality."""
    # Generate some alerts
    for i in range(3):
        await check_agent_alert(
            f"agent-{i}", {"agent_id": f"agent-{i}", "inference_time_ms": 100 + i * 50}
        )

    # Export alerts
    export_data = alerting_system.export_alerts(format="json")

    assert export_data is not None
    import json

    data = json.loads(export_data)

    assert "alerts" in data
    assert data["alert_count"] >= 3
    assert len(data["alerts"]) >= 3


@pytest.mark.asyncio
async def test_integration_with_agent(setup_alerting):
    """Test alerting integration with actual agent operations."""
    agent = BasicExplorerAgent("alert-integration-agent", "Alert Test Agent")
    agent.start()

    # Simulate slow inference
    await record_inference_metric(
        agent_id=agent.agent_id, inference_time_ms=220.0, success=True  # Above critical threshold
    )

    # Wait for alert processing
    await asyncio.sleep(0.1)

    # Check for alerts
    alerts = get_active_alerts()
    agent_alerts = [a for a in alerts if a["source"] == agent.agent_id]

    assert len(agent_alerts) > 0

    # Should have critical performance alert
    critical_perf_alerts = [
        a
        for a in agent_alerts
        if a["type"] == AlertType.PERFORMANCE_DEGRADATION.value
        and a["level"] == AlertLevel.CRITICAL.value
    ]
    assert len(critical_perf_alerts) > 0

    agent.stop()

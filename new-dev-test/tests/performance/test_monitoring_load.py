"""Test monitoring system under load conditions."""

import asyncio
import random
import time

import numpy as np
import pytest
from observability.alerting_system import get_active_alerts, get_alert_stats
from observability.belief_monitoring import get_all_belief_statistics, monitor_belief_update
from observability.coordination_metrics import (
    get_system_coordination_report,
    record_coalition_event,
    record_coordination,
)
from observability.monitoring_dashboard import (
    get_dashboard_data,
    start_monitoring_dashboard,
    stop_monitoring_dashboard,
)
from observability.performance_metrics import (
    get_performance_report,
    record_belief_metric,
    record_inference_metric,
    record_step_metric,
    start_performance_tracking,
    stop_performance_tracking,
)

from agents.base_agent import BasicExplorerAgent
from agents.coalition_coordinator import CoalitionCoordinatorAgent


@pytest.fixture(scope="module")
async def monitoring_setup():
    """Setup all monitoring systems."""
    await start_performance_tracking()
    await start_monitoring_dashboard()
    yield
    await stop_monitoring_dashboard()
    await stop_performance_tracking()


async def simulate_agent_operations(agent_id: str, num_operations: int, error_rate: float = 0.05):
    """Simulate agent operations with metrics."""
    for i in range(num_operations):
        # Simulate inference with variable time
        inference_time = random.gauss(30, 15)  # Mean 30ms, std 15ms
        if random.random() < 0.1:  # 10% chance of slow inference
            inference_time = random.gauss(150, 50)

        success = random.random() > error_rate

        await record_inference_metric(
            agent_id=agent_id,
            inference_time_ms=max(1, inference_time),
            success=success,
            error="Simulated error" if not success else None,
        )

        # Simulate belief update
        if i % 5 == 0:
            beliefs = {"qs": [np.random.dirichlet([1, 1, 1, 1])]}
            free_energy = random.gauss(2.0, 0.5)

            await monitor_belief_update(
                agent_id=agent_id,
                beliefs=beliefs,
                free_energy=free_energy,
                metadata={"step": i},
            )

            await record_belief_metric(
                agent_id=agent_id,
                update_time_ms=random.gauss(5, 2),
                free_energy=free_energy,
            )

        # Simulate step
        await record_step_metric(agent_id=agent_id, step_time_ms=random.gauss(50, 20))

        # Small delay between operations
        await asyncio.sleep(0.001)


async def simulate_coordination_operations(coordinator_id: str, num_operations: int):
    """Simulate coordination operations."""
    for i in range(num_operations):
        participant_ids = [f"agent-{j}" for j in range(random.randint(2, 5))]

        start_time = time.time()
        await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate coordination time

        success = random.random() > 0.1  # 90% success rate

        await record_coordination(
            coordinator_id=coordinator_id,
            participant_ids=participant_ids,
            coordination_type=random.choice(
                ["task_assignment", "resource_allocation", "formation"]
            ),
            start_time=start_time,
            success=success,
            metadata={"operation": i},
        )

        # Occasionally form/dissolve coalitions
        if i % 10 == 0:
            coalition_id = f"coalition_{coordinator_id}_{i}"
            await record_coalition_event(
                event_type="formation" if i % 20 == 0 else "dissolution",
                coalition_id=coalition_id,
                coordinator_id=coordinator_id,
                member_ids=participant_ids,
                duration_ms=random.gauss(100, 30),
                success=True,
            )


@pytest.mark.asyncio
async def test_monitoring_under_light_load(monitoring_setup):
    """Test monitoring with light load (5 agents)."""
    num_agents = 5
    operations_per_agent = 50

    # Create agents
    agents = []
    for i in range(num_agents):
        agent = BasicExplorerAgent(f"load-test-agent-{i}", f"Load Agent {i}")
        agent.start()
        agents.append(agent)

    # Run operations concurrently
    start_time = time.time()

    tasks = [simulate_agent_operations(agent.agent_id, operations_per_agent) for agent in agents]

    await asyncio.gather(*tasks)

    duration = time.time() - start_time

    # Verify monitoring is working
    perf_report = await get_performance_report()
    assert perf_report["system_counters"]["total_inferences"] >= num_agents * operations_per_agent

    # Wait for dashboard to update
    await asyncio.sleep(2)

    # Check dashboard
    dashboard = get_dashboard_data()
    assert dashboard is not None
    assert len(dashboard["agent_dashboards"]) >= num_agents

    # Performance assertions
    assert duration < 10  # Should complete within 10 seconds

    # Cleanup
    for agent in agents:
        agent.stop()


@pytest.mark.asyncio
async def test_monitoring_under_moderate_load(monitoring_setup):
    """Test monitoring with moderate load (20 agents)."""
    num_agents = 20
    operations_per_agent = 100

    # Create mixed agent types
    agents = []
    for i in range(num_agents):
        if i % 4 == 0:
            agent = CoalitionCoordinatorAgent(f"coordinator-{i}", f"Coordinator {i}")
        else:
            agent = BasicExplorerAgent(f"moderate-agent-{i}", f"Agent {i}")
        agent.start()
        agents.append(agent)

    # Run operations with coordinators
    start_time = time.time()

    tasks = []
    for i, agent in enumerate(agents):
        if isinstance(agent, CoalitionCoordinatorAgent):
            tasks.append(
                simulate_coordination_operations(agent.agent_id, operations_per_agent // 2)
            )
        tasks.append(simulate_agent_operations(agent.agent_id, operations_per_agent))

    await asyncio.gather(*tasks)

    time.time() - start_time

    # Verify all systems are functioning
    perf_report = await get_performance_report()
    coord_report = get_system_coordination_report()
    belief_stats = get_all_belief_statistics()
    alert_stats = get_alert_stats()

    assert perf_report["agent_count"] >= num_agents
    assert coord_report["active_agents"] > 0
    assert len(belief_stats) > 0

    # Check for performance degradation
    avg_inference_time = perf_report["detailed_stats"]["inference_times"]["avg"]
    assert avg_inference_time < 100  # Should maintain reasonable performance

    # Check alerts
    if alert_stats["total_active"] > 0:
        # Ensure we're not getting overwhelmed with alerts
        assert alert_stats["total_active"] < 50

    # Cleanup
    for agent in agents:
        agent.stop()


@pytest.mark.asyncio
async def test_monitoring_under_heavy_load(monitoring_setup):
    """Test monitoring with heavy load (50 agents)."""
    num_agents = 50
    operations_per_agent = 200

    # Create agents in batches to avoid overwhelming the system
    agents = []
    for batch in range(5):
        batch_agents = []
        for i in range(10):
            agent_id = batch * 10 + i
            agent = BasicExplorerAgent(f"heavy-agent-{agent_id}", f"Heavy Agent {agent_id}")
            agent.start()
            batch_agents.append(agent)
        agents.extend(batch_agents)
        await asyncio.sleep(0.1)  # Small delay between batches

    # Run operations with controlled concurrency
    start_time = time.time()

    # Use semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(20)  # Max 20 concurrent agent simulations

    async def run_with_semaphore(agent_id, ops):
        async with semaphore:
            await simulate_agent_operations(agent_id, ops, error_rate=0.1)

    tasks = [run_with_semaphore(agent.agent_id, operations_per_agent) for agent in agents]

    await asyncio.gather(*tasks)

    time.time() - start_time

    # Verify system stability
    perf_report = await get_performance_report()

    # System should handle the load
    assert (
        perf_report["system_counters"]["total_inferences"]
        >= num_agents * operations_per_agent * 0.9
    )

    # Check memory usage didn't explode
    memory_usage = perf_report["detailed_stats"]["memory_usage"]["max"]
    assert memory_usage < 2000  # Should stay under 2GB

    # Check that monitoring systems are still responsive
    dashboard_start = time.time()
    dashboard = get_dashboard_data()
    dashboard_time = time.time() - dashboard_start

    assert dashboard is not None
    assert dashboard_time < 1.0  # Dashboard should respond quickly

    # Cleanup
    for agent in agents:
        agent.stop()


@pytest.mark.asyncio
async def test_monitoring_burst_load(monitoring_setup):
    """Test monitoring with burst load patterns."""
    num_agents = 30

    # Create agents
    agents = []
    for i in range(num_agents):
        agent = BasicExplorerAgent(f"burst-agent-{i}", f"Burst Agent {i}")
        agent.start()
        agents.append(agent)

    # Simulate burst pattern
    for burst in range(3):
        # Burst phase - high activity
        burst_tasks = []
        for agent in agents:
            burst_tasks.append(simulate_agent_operations(agent.agent_id, 50, error_rate=0.2))

        await asyncio.gather(*burst_tasks)

        # Check system during burst
        perf_report = await get_performance_report()
        get_active_alerts()

        # System should handle bursts without crashing
        assert perf_report is not None

        # Quiet phase
        await asyncio.sleep(1)

    # Verify recovery after bursts
    final_report = await get_performance_report()
    get_alert_stats()

    # System should stabilize
    assert final_report["detailed_stats"]["cpu_usage"]["latest"] < 80

    # Cleanup
    for agent in agents:
        agent.stop()


@pytest.mark.asyncio
async def test_monitoring_sustained_load(monitoring_setup):
    """Test monitoring under sustained load for extended period."""
    num_agents = 15
    duration_seconds = 30

    # Create agents
    agents = []
    for i in range(num_agents):
        agent = BasicExplorerAgent(f"sustained-agent-{i}", f"Sustained Agent {i}")
        agent.start()
        agents.append(agent)

    # Run sustained load
    start_time = time.time()
    end_time = start_time + duration_seconds

    async def sustained_operations(agent_id):
        operations = 0
        while time.time() < end_time:
            await simulate_agent_operations(agent_id, 10)
            operations += 10
            await asyncio.sleep(0.1)
        return operations

    # Run all agents
    results = await asyncio.gather(*[sustained_operations(agent.agent_id) for agent in agents])

    total_operations = sum(results)

    # Verify sustained performance
    perf_report = await get_performance_report()

    # Check operation throughput
    actual_duration = time.time() - start_time
    ops_per_second = total_operations / actual_duration

    assert ops_per_second > 100  # Should maintain good throughput

    # Check system didn't degrade
    inference_stats = perf_report["detailed_stats"]["inference_times"]
    assert inference_stats["p99"] < 200  # 99th percentile should be reasonable

    # Check monitoring data completeness
    dashboard = get_dashboard_data()
    assert dashboard is not None
    assert len(dashboard["agent_dashboards"]) >= num_agents

    # Cleanup
    for agent in agents:
        agent.stop()


@pytest.mark.asyncio
async def test_monitoring_error_scenarios(monitoring_setup):
    """Test monitoring behavior under error conditions."""
    # Create agents that will fail
    failing_agents = []
    for i in range(10):
        agent = BasicExplorerAgent(f"failing-agent-{i}", f"Failing Agent {i}")
        agent.start()
        failing_agents.append(agent)

    # Simulate high error rate
    tasks = [
        simulate_agent_operations(agent.agent_id, 50, error_rate=0.5) for agent in failing_agents
    ]

    await asyncio.gather(*tasks)

    # Check alert generation
    alert_stats = get_alert_stats()
    assert alert_stats["total_active"] > 0
    assert alert_stats["by_type"].get("agent_failure", 0) > 0

    # Verify monitoring still works despite errors
    perf_report = await get_performance_report()
    assert perf_report["system_counters"]["total_errors"] > 0

    # Dashboard should show error states
    dashboard = get_dashboard_data()
    assert dashboard is not None

    # Check some agents show errors
    error_agents = [
        d
        for d in dashboard["agent_dashboards"].values()
        if d["current_metrics"]["error_rate"]["value"] > 0
    ]
    assert len(error_agents) > 0

    # Cleanup
    for agent in failing_agents:
        agent.stop()

"""Test belief state monitoring integration."""

import numpy as np
import pytest

from agents.base_agent import BasicExplorerAgent
from observability.belief_monitoring import (
    BeliefMonitor,
    belief_monitoring_hooks,
    get_all_belief_statistics,
    get_belief_statistics,
    monitor_belief_update,
)


@pytest.fixture
def belief_monitor():
    """Create a belief monitor for testing."""
    monitor = BeliefMonitor("test-agent")
    yield monitor
    monitor.reset()


@pytest.mark.asyncio
async def test_belief_monitoring_basic(belief_monitor):
    """Test basic belief monitoring functionality."""
    # Create test beliefs
    beliefs = {"qs": [np.array([0.25, 0.25, 0.25, 0.25])]}  # Uniform beliefs

    # Record belief update
    snapshot = await belief_monitor.record_belief_update(beliefs=beliefs, free_energy=1.5)

    assert snapshot.agent_id == "test-agent"
    assert snapshot.free_energy == 1.5
    assert snapshot.entropy is not None
    assert snapshot.kl_divergence == 0.0  # First update has no KL


@pytest.mark.asyncio
async def test_belief_entropy_calculation(belief_monitor):
    """Test entropy calculation for different belief distributions."""
    # Test uniform distribution (high entropy)
    uniform_beliefs = {"qs": [np.array([0.25, 0.25, 0.25, 0.25])]}
    snapshot1 = await belief_monitor.record_belief_update(uniform_beliefs)

    # Test peaked distribution (low entropy)
    peaked_beliefs = {"qs": [np.array([0.9, 0.05, 0.03, 0.02])]}
    snapshot2 = await belief_monitor.record_belief_update(peaked_beliefs)

    # Uniform distribution should have higher entropy
    assert snapshot1.entropy > snapshot2.entropy


@pytest.mark.asyncio
async def test_kl_divergence_calculation(belief_monitor):
    """Test KL divergence calculation between belief updates."""
    # First update
    beliefs1 = {"qs": [np.array([0.25, 0.25, 0.25, 0.25])]}
    await belief_monitor.record_belief_update(beliefs1)

    # Second update (same distribution)
    beliefs2 = {"qs": [np.array([0.25, 0.25, 0.25, 0.25])]}
    snapshot2 = await belief_monitor.record_belief_update(beliefs2)

    # Third update (different distribution)
    beliefs3 = {"qs": [np.array([0.7, 0.1, 0.1, 0.1])]}
    snapshot3 = await belief_monitor.record_belief_update(beliefs3)

    # KL divergence should be near 0 for identical distributions
    assert snapshot2.kl_divergence is not None
    assert snapshot2.kl_divergence < 0.01

    # KL divergence should be larger for different distributions
    assert snapshot3.kl_divergence > 0.1


@pytest.mark.asyncio
async def test_anomaly_detection(belief_monitor):
    """Test anomaly detection in belief updates."""
    # Generate consistent belief updates
    for i in range(15):
        beliefs = {
            "qs": [
                np.array([0.25 + i * 0.01, 0.25 - i * 0.003, 0.25 - i * 0.003, 0.25 - i * 0.004])
            ]
        }
        await belief_monitor.record_belief_update(beliefs)

    # Now add an anomalous update
    anomalous_beliefs = {"qs": [np.array([0.01, 0.01, 0.01, 0.97])]}  # Sudden spike

    # Record current anomaly count
    initial_anomaly_count = belief_monitor.anomaly_count

    # This should trigger an anomaly
    await belief_monitor.record_belief_update(anomalous_beliefs)

    # Check that anomaly was detected
    assert belief_monitor.anomaly_count > initial_anomaly_count


@pytest.mark.asyncio
async def test_belief_statistics(belief_monitor):
    """Test belief statistics computation."""
    # Generate some belief updates
    free_energies = [1.5, 1.3, 1.4, 1.2, 1.6]

    for i, fe in enumerate(free_energies):
        beliefs = {
            "qs": [np.array([0.25 + i * 0.05, 0.25 - i * 0.02, 0.25 - i * 0.02, 0.25 - i * 0.01])]
        }
        await belief_monitor.record_belief_update(beliefs, free_energy=fe)

    # Get statistics
    stats = belief_monitor.get_belief_statistics()

    assert stats["total_updates"] == 5
    assert stats["entropy"]["mean"] > 0
    assert stats["free_energy"]["mean"] == np.mean(free_energies)
    assert stats["kl_divergence"]["mean"] > 0


@pytest.mark.asyncio
async def test_belief_monitoring_hooks():
    """Test global belief monitoring hooks."""
    agent_id = "test-agent-hooks"

    # Record belief updates through hooks
    beliefs1 = {"qs": [np.array([0.3, 0.3, 0.2, 0.2])]}
    snapshot1 = await monitor_belief_update(agent_id, beliefs1, free_energy=1.2)

    assert snapshot1 is not None
    assert snapshot1.agent_id == agent_id

    # Get statistics
    stats = get_belief_statistics(agent_id)
    assert stats["total_updates"] == 1

    # Record more updates
    beliefs2 = {"qs": [np.array([0.4, 0.3, 0.2, 0.1])]}
    await monitor_belief_update(agent_id, beliefs2, free_energy=1.1)

    # Get all statistics
    all_stats = get_all_belief_statistics()
    assert agent_id in all_stats
    assert all_stats[agent_id]["total_updates"] == 2


@pytest.mark.asyncio
async def test_belief_monitoring_with_agent():
    """Test belief monitoring integration with actual agent."""
    agent = BasicExplorerAgent("test-agent-integration", "Explorer")
    agent.start()

    # Enable belief monitoring for this agent
    belief_monitoring_hooks.get_monitor(agent.agent_id)

    # Perform agent step
    observation = {"grid": np.zeros((10, 10))}
    agent.perceive(observation)

    # Get current beliefs if using PyMDP
    if agent.pymdp_agent and hasattr(agent.pymdp_agent, "qs"):
        beliefs = {"qs": agent.pymdp_agent.qs}

        # Compute free energy
        fe_components = agent.compute_free_energy()
        free_energy = fe_components.get("total_free_energy")

        # Monitor the belief update
        await monitor_belief_update(agent.agent_id, beliefs, free_energy=free_energy)

    # Update beliefs
    agent.update_beliefs()

    # Monitor again after update
    if agent.pymdp_agent and hasattr(agent.pymdp_agent, "qs"):
        beliefs = {"qs": agent.pymdp_agent.qs}
        fe_components = agent.compute_free_energy()
        free_energy = fe_components.get("total_free_energy")

        await monitor_belief_update(agent.agent_id, beliefs, free_energy=free_energy)

    # Check statistics
    stats = get_belief_statistics(agent.agent_id)
    assert stats["total_updates"] >= 1

    agent.stop()


@pytest.mark.asyncio
async def test_belief_history(belief_monitor):
    """Test belief history tracking."""
    # Generate some belief updates
    for i in range(5):
        beliefs = {
            "qs": [np.array([0.2 + i * 0.05, 0.3 - i * 0.02, 0.3 - i * 0.02, 0.2 - i * 0.01])]
        }
        await belief_monitor.record_belief_update(beliefs)

    # Get history
    history = belief_monitor.get_belief_history()
    assert len(history) == 5

    # Get limited history
    limited_history = belief_monitor.get_belief_history(limit=3)
    assert len(limited_history) == 3

    # Check that history is ordered (oldest to newest)
    for i in range(1, len(history)):
        assert history[i].timestamp > history[i - 1].timestamp


@pytest.mark.asyncio
async def test_monitor_reset(belief_monitor):
    """Test monitor reset functionality."""
    # Add some data
    beliefs = {"qs": [np.array([0.25, 0.25, 0.25, 0.25])]}
    await belief_monitor.record_belief_update(beliefs)

    assert belief_monitor.total_updates == 1
    assert len(belief_monitor.belief_history) == 1

    # Reset
    belief_monitor.reset()

    assert belief_monitor.total_updates == 0
    assert len(belief_monitor.belief_history) == 0
    assert belief_monitor.anomaly_count == 0

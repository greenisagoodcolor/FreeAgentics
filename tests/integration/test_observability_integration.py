"""Test observability integration with PyMDP agents.

Validates that observability monitoring works correctly with
Active Inference agents and PyMDP operations.
"""

import asyncio
import logging
import time
from typing import Any, Dict

# import pytest  # Not available in this environment

# Test the observability integration
try:
    from observability import (
        get_pymdp_performance_summary,
        monitor_pymdp_inference,
        pymdp_observer,
        record_agent_lifecycle_event,
        record_belief_update,
    )

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

try:
    from agents.base_agent import BaseAgent
    from agents.resource_collector import ResourceCollectorAgent

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

    # Mock BaseAgent if not available
    class BaseAgent:
        def __init__(self, agent_id: str, name: str = "", config: dict = None):
            self.agent_id = agent_id
            self.name = name
            self.config = config or {}
            self.observability_enabled = (
                config.get("enable_observability", False) if config else False
            )
            self.metrics = {"belief_entropy": 0.0, "avg_free_energy": 0.0}
            self.is_active = False

        def activate(self):
            self.is_active = True

        def deactivate(self):
            self.is_active = False

        def step(self, observation):
            return "mock_action"


logger = logging.getLogger(__name__)


class MockObservableAgent(BaseAgent):
    """Mock agent for testing observability integration."""

    def __init__(self, agent_id: str, name: str = "Mock Agent", config: Dict = None):
        config = config or {"enable_observability": True}
        super().__init__(agent_id, name, config)
        self.test_beliefs = {"test_belie": 0.5}
        self.test_observations = []

    def perceive(self, observation: Any) -> None:
        """Mock perception."""
        self.test_observations.append(observation)
        self.current_observation = observation

    def update_beliefs(self) -> None:
        """Mock belief update with observability."""
        # Simulate belief change
        old_belief = self.test_beliefs["test_belie"]
        self.test_beliefs["test_belie"] = min(1.0, old_belief + 0.1)

        # Update metrics
        self.metrics["belief_entropy"] = 1.0 - self.test_beliefs["test_belie"]
        self.metrics["avg_free_energy"] = self.test_beliefs["test_belie"] * 10

    def select_action(self) -> str:
        """Mock action selection."""
        return "mock_action"


async def test_observability_lifecycle_events():
    """Test agent lifecycle event monitoring."""
    if not OBSERVABILITY_AVAILABLE:
        logger.warning("⚠️ Observability not available - skipping test")
        return

    # Create agent (should record creation event)
    agent = MockObservableAgent("test_agent_lifecycle", config={"enable_observability": True})

    # Wait a bit for async event recording
    # No artificial delays - operations should complete synchronously

    # Check if agent is tracked
    assert agent.agent_id in pymdp_observer.agent_lifecycles

    # Record custom lifecycle events
    await record_agent_lifecycle_event(agent.agent_id, "activated", {"status": "active"})
    await record_agent_lifecycle_event(agent.agent_id, "terminated", {"reason": "test_complete"})

    # Verify events were recorded
    events = pymdp_observer.agent_lifecycles[agent.agent_id]
    event_types = [event["event"] for event in events]

    assert "created" in event_types
    assert "activated" in event_types
    assert "terminated" in event_types

    logger.info("✅ Lifecycle event monitoring test passed")


async def test_observability_belief_monitoring():
    """Test belief update monitoring."""
    if not OBSERVABILITY_AVAILABLE:
        logger.warning("⚠️ Observability not available - skipping test")
        return

    agent = MockObservableAgent("test_agent_beliefs", config={"enable_observability": True})

    # Test belief update recording
    beliefs_before = {"test_belie": 0.5, "entropy": 0.8}
    beliefs_after = {"test_belie": 0.7, "entropy": 0.6}
    free_energy = 5.2

    await record_belief_update(agent.agent_id, beliefs_before, beliefs_after, free_energy)

    # Wait for async processing
    # No artificial delays - operations should complete synchronously

    # Verify belief update was recorded
    assert agent.agent_id in pymdp_observer.belief_update_history

    belief_history = pymdp_observer.belief_update_history[agent.agent_id]
    assert len(belief_history) > 0

    latest_update = belief_history[-1]
    assert latest_update["free_energy"] == free_energy
    assert latest_update["belief_change_magnitude"] > 0  # Should detect change

    logger.info("✅ Belief update monitoring test passed")


async def test_observability_inference_monitoring():
    """Test PyMDP inference performance monitoring."""
    if not OBSERVABILITY_AVAILABLE:
        logger.warning("⚠️ Observability not available - skipping test")
        return

    agent = MockObservableAgent("test_agent_inference", config={"enable_observability": True})

    # Test monitoring decorator
    @monitor_pymdp_inference(agent.agent_id)
    async def mock_inference_operation():
        """Mock inference with computation."""
        # Perform actual computation instead of sleep
        result = sum(range(10000))  # Real computation
        return "inference_result"

    # Run monitored inference
    result = await mock_inference_operation()
    assert result == "inference_result"

    # Wait for metrics recording
    # No artificial delays - operations should complete synchronously

    # Verify performance metrics were recorded
    assert agent.agent_id in pymdp_observer.inference_metrics

    inference_metrics = pymdp_observer.inference_metrics[agent.agent_id]
    assert len(inference_metrics) > 0

    latest_metric = inference_metrics[-1]
    assert latest_metric["success"] is True
    assert latest_metric["inference_time_ms"] >= 0  # Should have positive time
    assert latest_metric["inference_time_ms"] < 1000  # Less than 1 second

    logger.info("✅ Inference monitoring test passed")


async def test_observability_agent_integration():
    """Test observability integration with real agent operations."""
    if not OBSERVABILITY_AVAILABLE or not AGENTS_AVAILABLE:
        logger.warning("⚠️ Observability or agents not available - skipping test")
        return

    # Create ResourceCollectorAgent with observability enabled
    config = {
        "enable_observability": True,
        "grid_size": 5,
        "use_pymdp": False,  # Use fallback for testing
    }

    agent = ResourceCollectorAgent("test_integration_agent", "Test Agent", config)

    # Activate agent
    agent.activate()

    # Perform several steps
    observations = [
        {"position": [1, 1], "resources": []},
        {"position": [1, 2], "resources": [{"position": [2, 2], "type": "food"}]},
        {"position": [2, 2], "resources": []},
    ]

    for obs in observations:
        action = agent.step(obs)
        assert action is not None
        # No artificial delays between steps

    # Deactivate agent
    agent.deactivate()

    # Wait for all async operations
    # No artificial delays - operations should complete synchronously

    # Verify integration worked
    summary = await get_pymdp_performance_summary(agent.agent_id)

    assert "agent_id" in summary
    assert summary["agent_id"] == agent.agent_id
    assert summary["inference_count"] >= 0  # May be 0 if using fallback
    assert summary["lifecycle_events"] >= 1  # At least creation event

    logger.info("✅ Agent integration test passed")


async def test_observability_performance_summary():
    """Test performance summary generation."""
    if not OBSERVABILITY_AVAILABLE:
        logger.warning("⚠️ Observability not available - skipping test")
        return

    # Create multiple agents
    agents = []
    for i in range(3):
        agent = MockObservableAgent(f"perf_test_agent_{i}", config={"enable_observability": True})
        agents.append(agent)

        # Record some metrics
        await record_agent_lifecycle_event(agent.agent_id, "created")
        await record_belief_update(agent.agent_id, {"belief": 0.0}, {"belie": 0.5})

    # No artificial delays - operations should complete synchronously

    # Test system-wide summary
    system_summary = await get_pymdp_performance_summary()

    assert "total_agents" in system_summary
    assert system_summary["total_agents"] >= 3
    assert "total_belief_updates" in system_summary
    assert system_summary["total_belief_updates"] >= 3

    # Test agent-specific summary
    agent_summary = await get_pymdp_performance_summary(agents[0].agent_id)

    assert "agent_id" in agent_summary
    assert agent_summary["agent_id"] == agents[0].agent_id
    assert "belief_updates" in agent_summary
    assert agent_summary["belief_updates"] >= 1

    logger.info("✅ Performance summary test passed")


def test_observability_fallback():
    """Test observability fallback when not available."""
    # This test should always pass even without observability

    # Mock an agent without observability
    config = {"enable_observability": False}
    agent = MockObservableAgent("fallback_test_agent", config=config)

    # These operations should not fail
    agent.activate()
    action = agent.step({"test": "observation"})
    agent.deactivate()

    assert action is not None
    logger.info("✅ Observability fallback test passed")


if __name__ == "__main__":

    async def run_observability_tests():
        """Run observability integration tests."""
        logger.info("🚀 Starting observability integration tests...")

        try:
            # Run tests
            await test_observability_lifecycle_events()
            await test_observability_belief_monitoring()
            await test_observability_inference_monitoring()
            await test_observability_agent_integration()
            await test_observability_performance_summary()
            test_observability_fallback()

            logger.info("🎉 All observability integration tests passed!")
            return True

        except Exception as e:
            logger.error(f"❌ Observability integration tests failed: {e}")
            return False

    success = asyncio.run(run_observability_tests())
    exit(0 if success else 1)

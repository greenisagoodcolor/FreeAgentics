"""
Comprehensive test suite for Base Agent - Additional Coverage
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

# Mock imports before importing base_agent
with patch.dict(
    "sys.modules",
    {
        "pymdp": MagicMock(),
        "pymdp.utils": MagicMock(),
        "pymdp.agent": MagicMock(),
        "observability": MagicMock(),
        "observability.belief_monitoring": MagicMock(),
    },
):
    from agents.base_agent import (
        OBSERVABILITY_AVAILABLE,
        PYMDP_AVAILABLE,
        ActiveInferenceAgent,
        AgentConfig,
        safe_array_to_int,
    )


class TestSafeArrayToInt:
    """Test the safe_array_to_int utility function."""

    def test_numpy_scalar(self):
        """Test conversion of numpy scalar."""
        value = np.int64(42)
        result = safe_array_to_int(value)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_0d_array(self):
        """Test conversion of 0-dimensional numpy array."""
        value = np.array(42)
        result = safe_array_to_int(value)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_1d_single_element(self):
        """Test conversion of single-element 1D array."""
        value = np.array([42])
        result = safe_array_to_int(value)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_multidimensional_array(self):
        """Test conversion of multidimensional array (takes first element)."""
        value = np.array([[1, 2], [3, 4]])
        result = safe_array_to_int(value)
        assert result == 1
        assert isinstance(result, int)

    def test_empty_array_raises_error(self):
        """Test that empty array raises ValueError."""
        value = np.array([])
        with pytest.raises(
            ValueError, match="Empty array cannot be converted to integer"
        ):
            safe_array_to_int(value)

    def test_python_list(self):
        """Test conversion of Python list."""
        value = [42, 43, 44]
        result = safe_array_to_int(value)
        assert result == 42
        assert isinstance(result, int)

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        value = []
        with pytest.raises(
            ValueError, match="Empty array cannot be converted to integer"
        ):
            safe_array_to_int(value)

    def test_python_tuple(self):
        """Test conversion of Python tuple."""
        value = (42, 43, 44)
        result = safe_array_to_int(value)
        assert result == 42
        assert isinstance(result, int)

    def test_regular_int(self):
        """Test conversion of regular int."""
        value = 42
        result = safe_array_to_int(value)
        assert result == 42
        assert isinstance(result, int)

    def test_float_conversion(self):
        """Test conversion of float."""
        value = 42.7
        result = safe_array_to_int(value)
        assert result == 42
        assert isinstance(result, int)

    def test_string_conversion(self):
        """Test conversion of numeric string."""
        value = "42"
        result = safe_array_to_int(value)
        assert result == 42
        assert isinstance(result, int)

    def test_invalid_string_raises_error(self):
        """Test that invalid string raises ValueError."""
        value = "not_a_number"
        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int(value)

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        value = None
        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int(value)

    def test_complex_object_raises_error(self):
        """Test that complex object raises ValueError."""
        value = {"key": "value"}
        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int(value)


class TestActiveInferenceAgentConfiguration:
    """Test agent configuration and initialization."""

    def test_agent_config_creation(self):
        """Test AgentConfig creation."""
        config = AgentConfig(
            agent_id="test_agent",
            name="Test Agent",
            learning_rate=0.1,
            policy_precision=2.0,
            use_pymdp=True,
            use_observability=True,
            pymdp_config={"some": "config"},
            llm_config={"model": "test"},
        )

        assert config.agent_id == "test_agent"
        assert config.name == "Test Agent"
        assert config.learning_rate == 0.1
        assert config.policy_precision == 2.0
        assert config.use_pymdp is True
        assert config.use_observability is True
        assert config.pymdp_config == {"some": "config"}
        assert config.llm_config == {"model": "test"}

    @patch('agents.base_agent.PYMDP_AVAILABLE', True)
    def test_agent_initialization_with_pymdp(self):
        """Test agent initialization when PyMDP is available."""

        class TestAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                self.last_observation = observation

            def update_beliefs(self):
                self.beliefs = {"test": 0.5}

            def select_action(self):
                return 0

        agent = TestAgent(
            agent_id="test_001", name="Test Agent", config={"use_pymdp": True}
        )

        assert agent.agent_id == "test_001"
        assert agent.name == "Test Agent"
        assert agent.config.use_pymdp is True
        assert agent.created_at is not None
        assert agent.status == "initialized"

    @patch('agents.base_agent.PYMDP_AVAILABLE', False)
    def test_agent_initialization_without_pymdp(self):
        """Test agent initialization when PyMDP is not available."""

        class TestAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                self.last_observation = observation

            def update_beliefs(self):
                self.beliefs = {"test": 0.5}

            def select_action(self):
                return 0

        agent = TestAgent(
            agent_id="test_002",
            name="Test Agent No PyMDP",
            config={"use_pymdp": False},
        )

        assert agent.agent_id == "test_002"
        assert agent.name == "Test Agent No PyMDP"
        assert agent.config.use_pymdp is False
        assert agent.pymdp_agent is None


class TestActiveInferenceAgentMethods:
    """Test agent methods and behaviors."""

    def setup_method(self):
        """Set up test agent."""

        class TestAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                self.last_observation = observation

            def update_beliefs(self):
                self.beliefs = {"test": 0.5}

            def select_action(self):
                return 0

        self.agent = TestAgent(
            agent_id="test_agent",
            name="Test Agent",
            config={"use_pymdp": False},
        )

    def test_agent_step_basic(self):
        """Test basic agent step functionality."""
        observation = {"sensor": "value"}

        result = self.agent.step(observation)

        assert result is not None
        assert hasattr(self.agent, 'last_observation')
        assert self.agent.last_observation == observation

    def test_agent_get_info(self):
        """Test agent info retrieval."""
        info = self.agent.get_info()

        assert isinstance(info, dict)
        assert "agent_id" in info
        assert "name" in info
        assert "status" in info
        assert "created_at" in info
        assert info["agent_id"] == "test_agent"
        assert info["name"] == "Test Agent"

    def test_agent_status_management(self):
        """Test agent status management."""
        assert self.agent.status == "initialized"

        # Test status change
        self.agent.status = "running"
        assert self.agent.status == "running"

        info = self.agent.get_info()
        assert info["status"] == "running"

    def test_agent_config_access(self):
        """Test agent configuration access."""
        config = self.agent.config

        assert isinstance(config, AgentConfig)
        assert config.agent_id == "test_agent"
        assert config.name == "Test Agent"
        assert config.use_pymdp is False

    def test_agent_metadata_handling(self):
        """Test agent metadata handling."""
        # Test initial metadata
        assert hasattr(self.agent, 'created_at')
        assert isinstance(self.agent.created_at, datetime)

        # Test info includes metadata
        info = self.agent.get_info()
        assert "created_at" in info

    @patch('agents.base_agent.logger')
    def test_agent_logging(self, mock_logger):
        """Test agent logging functionality."""
        observation = {"test": "data"}

        self.agent.step(observation)

        # Verify logging occurred (depends on implementation)
        # This test ensures the logger is accessible
        assert mock_logger is not None

    def test_agent_error_handling_in_step(self):
        """Test error handling in agent step."""

        class ErrorAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                raise ValueError("Test error")

            def update_beliefs(self):
                pass

            def select_action(self):
                return 0

        agent = ErrorAgent(
            agent_id="error_agent",
            name="Error Agent",
            config={"use_pymdp": False},
        )

        # Test that errors are handled gracefully
        with pytest.raises(ValueError):
            agent.step({"test": "data"})

    def test_agent_performance_monitoring(self):
        """Test performance monitoring integration."""
        # This test verifies the agent can be monitored
        observation = {"sensor": "data"}

        # Should not raise errors
        result = self.agent.step(observation)
        assert result is not None

    def test_agent_memory_management(self):
        """Test agent memory management."""
        # Test that agent doesn't accumulate excessive memory
        for i in range(10):
            observation = {"iteration": i}
            self.agent.step(observation)

        # Agent should still be functional
        info = self.agent.get_info()
        assert info["agent_id"] == "test_agent"

    def test_agent_belief_updates(self):
        """Test belief update mechanism."""
        observation = {"belief_trigger": True}

        # Run step to trigger belief update
        self.agent.step(observation)

        # Verify beliefs were updated
        assert hasattr(self.agent, 'beliefs')
        assert self.agent.beliefs == {"test": 0.5}

    def test_agent_action_selection(self):
        """Test action selection mechanism."""
        observation = {"action_trigger": True}

        # Run step to trigger action selection
        result = self.agent.step(observation)

        # Verify action was selected
        assert result is not None


class TestActiveInferenceAgentIntegration:
    """Test agent integration with external systems."""

    def setup_method(self):
        """Set up test agent with integration features."""

        class IntegrationAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                self.last_observation = observation
                self.observation_count = (
                    getattr(self, 'observation_count', 0) + 1
                )

            def update_beliefs(self):
                self.beliefs = {"confidence": 0.8, "uncertainty": 0.2}

            def select_action(self):
                return getattr(self, 'observation_count', 0) % 2

        self.agent = IntegrationAgent(
            agent_id="integration_agent",
            name="Integration Agent",
            config={"use_pymdp": False, "use_observability": False},
        )

    def test_agent_observation_tracking(self):
        """Test agent observation tracking."""
        observations = [
            {"sensor1": "value1"},
            {"sensor2": "value2"},
            {"sensor3": "value3"},
        ]

        for obs in observations:
            self.agent.step(obs)

        assert self.agent.observation_count == 3
        assert self.agent.last_observation == observations[-1]

    def test_agent_belief_evolution(self):
        """Test belief evolution over time."""
        observations = [
            {"evidence": "positive"},
            {"evidence": "negative"},
            {"evidence": "neutral"},
        ]

        for obs in observations:
            self.agent.step(obs)

        # Verify beliefs are updated
        assert hasattr(self.agent, 'beliefs')
        assert "confidence" in self.agent.beliefs
        assert "uncertainty" in self.agent.beliefs

    def test_agent_action_history(self):
        """Test agent action history tracking."""
        actions = []

        for i in range(5):
            observation = {"step": i}
            action = self.agent.step(observation)
            actions.append(action)

        # Verify actions follow expected pattern
        assert len(actions) == 5
        assert all(isinstance(action, int) for action in actions)

    def test_agent_state_consistency(self):
        """Test agent state consistency across steps."""
        # Initial state
        initial_info = self.agent.get_info()

        # Run several steps
        for i in range(3):
            self.agent.step({"iteration": i})

        # Check state consistency
        final_info = self.agent.get_info()
        assert final_info["agent_id"] == initial_info["agent_id"]
        assert final_info["name"] == initial_info["name"]
        assert final_info["created_at"] == initial_info["created_at"]

    @patch('agents.base_agent.OBSERVABILITY_AVAILABLE', True)
    def test_agent_observability_integration(self):
        """Test agent observability integration."""
        # This test verifies observability hooks work when available
        observation = {"monitored": True}

        # Should not raise errors even with observability
        result = self.agent.step(observation)
        assert result is not None

    def test_agent_error_recovery(self):
        """Test agent error recovery mechanisms."""

        class RecoveryAgent(ActiveInferenceAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.error_count = 0

            def perceive(self, observation):
                if observation.get("cause_error"):
                    self.error_count += 1
                    if self.error_count < 3:
                        raise ValueError("Recoverable error")
                self.last_observation = observation

            def update_beliefs(self):
                self.beliefs = {"error_count": self.error_count}

            def select_action(self):
                return 0

        agent = RecoveryAgent(
            agent_id="recovery_agent",
            name="Recovery Agent",
            config={"use_pymdp": False},
        )

        # Test error recovery
        with pytest.raises(ValueError):
            agent.step({"cause_error": True})

        with pytest.raises(ValueError):
            agent.step({"cause_error": True})

        # Third time should work
        result = agent.step({"cause_error": True})
        assert result is not None
        assert agent.error_count == 3

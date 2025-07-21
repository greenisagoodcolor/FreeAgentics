"""Comprehensive tests for base_agent.py targeting 80%+ coverage."""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from agents.base_agent import (
    AgentConfig,
    BasicExplorerAgent,
    safe_array_to_int,
)
from agents.error_handling import (
    ActionSelectionError,
    AgentError,
    InferenceError,
)


class TestSafeArrayToInt:
    """Test safe_array_to_int function comprehensively."""

    def test_numpy_scalar_array(self):
        """Test with 0-dimensional numpy array."""
        value = np.array(5)
        assert safe_array_to_int(value) == 5

    def test_single_element_array(self):
        """Test with single element numpy array."""
        value = np.array([3])
        assert safe_array_to_int(value) == 3

    def test_multi_element_array(self):
        """Test with multi-element array - should take first element."""
        value = np.array([7, 8, 9])
        assert safe_array_to_int(value) == 7

    def test_multidimensional_array(self):
        """Test with multidimensional array."""
        value = np.array([[1, 2], [3, 4]])
        assert safe_array_to_int(value) == 1

    def test_empty_array(self):
        """Test with empty array - should raise ValueError."""
        value = np.array([])
        with pytest.raises(ValueError, match="Empty array cannot be converted"):
            safe_array_to_int(value)

    def test_list_input(self):
        """Test with list input."""
        value = [4, 5, 6]
        assert safe_array_to_int(value) == 4

    def test_empty_list(self):
        """Test with empty list - should raise ValueError."""
        value = []
        with pytest.raises(ValueError, match="Empty array cannot be converted"):
            safe_array_to_int(value)

    def test_numpy_scalar(self):
        """Test with numpy scalar."""
        value = np.int64(8)
        assert safe_array_to_int(value) == 8

    def test_regular_scalar(self):
        """Test with regular Python scalar."""
        assert safe_array_to_int(5) == 5
        assert safe_array_to_int(5.7) == 5

    def test_invalid_input(self):
        """Test with invalid input."""
        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int("invalid")


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        config = AgentConfig(name="test_agent")
        assert config.name == "test_agent"
        assert config.use_pymdp is True
        assert config.planning_horizon == 3
        assert config.precision == 1.0
        assert config.lr == 0.1
        assert config.gmn_spec is None
        assert config.llm_config is None

    def test_custom_initialization(self):
        """Test custom initialization."""
        config = AgentConfig(
            name="custom",
            use_pymdp=False,
            planning_horizon=5,
            precision=2.0,
            lr=0.05,
            gmn_spec="test_spec",
            llm_config={"model": "test"},
        )

        assert config.name == "custom"
        assert config.use_pymdp is False
        assert config.planning_horizon == 5
        assert config.precision == 2.0
        assert config.lr == 0.05
        assert config.gmn_spec == "test_spec"
        assert config.llm_config == {"model": "test"}


class TestBasicExplorerAgent:
    """Test BasicExplorerAgent implementation."""

    @patch("agents.base_agent.PYMDP_AVAILABLE", False)
    def test_initialization_without_pymdp(self):
        """Test initialization when PyMDP is not available."""
        AgentConfig(name="test_agent")
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        assert agent.agent_id == "test_agent"
        assert agent.num_states == [4]
        assert agent.num_actions == [3]
        assert agent.num_observations == [4]
        # Should initialize simple beliefs
        assert agent.beliefs.shape == (4,)
        assert np.allclose(agent.beliefs, 0.25)

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_initialization_with_pymdp(self, mock_pymdp_agent):
        """Test initialization when PyMDP is available."""
        # Mock PyMDP agent
        mock_agent_instance = Mock()
        mock_pymdp_agent.return_value = mock_agent_instance

        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        assert agent.pymdp_agent == mock_agent_instance
        mock_pymdp_agent.assert_called_once()

    @patch("agents.base_agent.PYMDP_AVAILABLE", False)
    def test_update_beliefs_without_pymdp(self):
        """Test belief update without PyMDP."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Initial beliefs should be uniform
        assert np.allclose(agent.beliefs, 0.25)

        # Update beliefs
        agent.update_beliefs(1)

        # Beliefs should change
        assert agent.beliefs[1] > 0.25
        assert np.allclose(agent.beliefs.sum(), 1.0)

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_update_beliefs_with_pymdp(self, mock_pymdp_agent):
        """Test belief update with PyMDP."""
        # Mock PyMDP agent
        mock_agent_instance = Mock()
        mock_agent_instance.infer_states.return_value = np.array([0.7, 0.1, 0.1, 0.1])
        mock_pymdp_agent.return_value = mock_agent_instance

        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        agent.update_beliefs(1)

        assert np.array_equal(agent.beliefs, np.array([0.7, 0.1, 0.1, 0.1]))
        mock_agent_instance.infer_states.assert_called_once_with(1)

    @patch("agents.base_agent.PYMDP_AVAILABLE", False)
    def test_select_action_without_pymdp(self):
        """Test action selection without PyMDP."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Set specific beliefs to make action selection deterministic
        agent.beliefs = np.array([0.7, 0.1, 0.1, 0.1])

        action = agent.select_action()
        assert 0 <= action < 3

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_select_action_with_pymdp(self, mock_pymdp_agent):
        """Test action selection with PyMDP."""
        # Mock PyMDP agent
        mock_agent_instance = Mock()
        mock_agent_instance.infer_policies.return_value = None
        mock_agent_instance.sample_action.return_value = np.array([2])
        mock_pymdp_agent.return_value = mock_agent_instance

        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        action = agent.select_action()

        assert action == 2
        mock_agent_instance.infer_policies.assert_called_once()
        mock_agent_instance.sample_action.assert_called_once()

    def test_learn_method(self):
        """Test the learn method."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        agent.preferences = np.array([1.0, 0.0, 0.0, 0.0])
        initial_preferences = agent.preferences.copy()
        agent.learn(observation=1, reward=0.5)

        # Preferences should be updated
        assert not np.array_equal(agent.preferences, initial_preferences)

    def test_reset_method(self):
        """Test the reset method."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Set some state
        agent.beliefs = np.array([0.8, 0.1, 0.05, 0.05])
        agent.action_history = [1, 2, 0]
        agent.observation_history = [0, 1, 2]

        agent.reset()

        # State should be cleared
        assert agent.action_history == []
        assert agent.observation_history == []
        # Beliefs should be reset to uniform
        assert np.allclose(agent.beliefs, 0.25)

    def test_step_method(self):
        """Test the step method."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Perform step
        action = agent.step(2)

        # Check action is valid
        assert 0 <= action < 3

        # Check observation was recorded
        assert agent.observation_history[-1] == 2
        assert agent.action_history[-1] == action

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_pymdp_initialization_error(self, mock_pymdp_agent):
        """Test handling of PyMDP initialization errors."""
        # Make PyMDP agent initialization fail
        mock_pymdp_agent.side_effect = Exception("PyMDP init error")

        # Should fall back to simple implementation
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        assert agent.pymdp_agent is None
        assert agent.beliefs is not None

    def test_invalid_observation(self):
        """Test handling of invalid observations."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Out of range observation
        with pytest.raises(AgentError):
            agent.step(10)

        # Invalid type
        with pytest.raises(AgentError):
            agent.step("invalid")

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_belief_update_error_recovery(self, mock_pymdp_agent):
        """Test recovery from belief update errors."""
        # Mock PyMDP agent
        mock_agent_instance = Mock()
        mock_agent_instance.infer_states.side_effect = Exception("Inference error")
        mock_pymdp_agent.return_value = mock_agent_instance

        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Should raise InferenceError
        with pytest.raises(InferenceError):
            agent.update_beliefs(1)

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_select_action_with_pymdp_error(self, mock_pymdp_agent):
        """Test action selection with PyMDP error."""
        # Mock PyMDP agent that raises error
        mock_agent_instance = Mock()
        mock_agent_instance.infer_policies.side_effect = Exception("PyMDP error")
        mock_pymdp_agent.return_value = mock_agent_instance

        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        with pytest.raises(ActionSelectionError):
            agent.select_action()

    @patch("agents.base_agent.OBSERVABILITY_AVAILABLE", True)
    @patch("agents.base_agent.record_agent_lifecycle_event")
    def test_lifecycle_event_recording(self, mock_record_event):
        """Test that lifecycle events are recorded."""
        # Mock as coroutine
        mock_record_event.return_value = AsyncMock()

        BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Should have recorded creation event
        mock_record_event.assert_called()
        call_args = mock_record_event.call_args[1]
        assert call_args["agent_id"] == "test_agent"
        assert call_args["event_type"] == "created"

    @patch("agents.base_agent.BELIEF_MONITORING_AVAILABLE", True)
    @patch("agents.base_agent.monitor_belief_update")
    def test_belief_monitoring(self, mock_monitor):
        """Test belief monitoring."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Update beliefs
        agent.update_beliefs(1)

        # Should have monitored belief update
        mock_monitor.assert_called_once()

    @patch("agents.base_agent.performance_monitor")
    def test_performance_monitoring(self, mock_monitor):
        """Test that performance is monitored."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Perform operations
        agent.update_beliefs(1)
        agent.select_action()

        # Should have monitored performance
        assert mock_monitor.call_count >= 2

    def test_history_tracking(self):
        """Test that action and observation history are tracked."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
        )

        # Perform several steps
        for i in range(3):
            action = agent.step(i)
            assert len(agent.observation_history) == i + 1
            assert len(agent.action_history) == i + 1
            assert agent.observation_history[-1] == i
            assert agent.action_history[-1] == action

    def test_preference_learning_convergence(self):
        """Test that preference learning converges."""
        agent = BasicExplorerAgent(
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            agent_id="test_agent",
            learning_rate=0.1,
        )

        # Initialize preferences
        agent.preferences = np.zeros(4)

        # Repeatedly reward observation 2
        for _ in range(10):
            agent.learn(observation=2, reward=1.0)

        # Preference for observation 2 should be highest
        assert agent.preferences[2] > agent.preferences[0]
        assert agent.preferences[2] > agent.preferences[1]
        assert agent.preferences[2] > agent.preferences[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

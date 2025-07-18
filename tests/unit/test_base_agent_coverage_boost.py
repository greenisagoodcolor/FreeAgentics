"""Comprehensive tests for base_agent.py to boost coverage to 80%+."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from agents.base_agent import (
    safe_array_to_int,
    ActiveInferenceAgent,
    BasicExplorerAgent,
    AgentConfig,
    PYMDP_AVAILABLE,
    OBSERVABILITY_AVAILABLE,
)
from agents.error_handling import (
    ActionSelectionError,
    InferenceError,
    AgentError,
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
        config = AgentConfig()
        assert config.agent_id == "default"
        assert config.num_states == []
        assert config.num_actions == []
        assert config.num_observations == []
        assert config.initial_beliefs is None
        assert config.learning_rate == 0.01
        assert config.planning_horizon == 1
        assert config.enable_monitoring is True

    def test_custom_initialization(self):
        """Test custom initialization."""
        beliefs = np.array([0.25, 0.25, 0.25, 0.25])
        config = AgentConfig(
            agent_id="custom",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            initial_beliefs=beliefs,
            learning_rate=0.05,
            planning_horizon=3,
            enable_monitoring=False,
        )
        
        assert config.agent_id == "custom"
        assert config.num_states == [4]
        assert config.num_actions == [3]
        assert config.num_observations == [4]
        assert np.array_equal(config.initial_beliefs, beliefs)
        assert config.learning_rate == 0.05
        assert config.planning_horizon == 3
        assert config.enable_monitoring is False


class TestActiveInferenceAgent:
    """Test ActiveInferenceAgent abstract class."""

    def test_initialization_from_config(self):
        """Test initialization from AgentConfig."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        
        # Create a concrete implementation for testing
        class ConcreteAgent(ActiveInferenceAgent):
            def update_beliefs(self, observation):
                pass

            def select_action(self):
                return 0

            def learn(self, observation, reward=None):
                pass

        agent = ConcreteAgent(config)
        assert agent.agent_id == "test_agent"
        assert agent.num_states == [4]
        assert agent.num_actions == [3]
        assert agent.num_observations == [4]

    def test_step_method(self):
        """Test the step method."""
        class ConcreteAgent(ActiveInferenceAgent):
            def __init__(self, config):
                super().__init__(config)
                self.update_beliefs_called = False
                self.select_action_called = False

            def update_beliefs(self, observation):
                self.update_beliefs_called = True

            def select_action(self):
                self.select_action_called = True
                return 1

            def learn(self, observation, reward=None):
                pass

        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = ConcreteAgent(config)

        action = agent.step(2)
        assert action == 1
        assert agent.update_beliefs_called
        assert agent.select_action_called

    def test_reset_method(self):
        """Test the reset method."""
        class ConcreteAgent(ActiveInferenceAgent):
            def update_beliefs(self, observation):
                self.beliefs = np.ones(4) * 0.25

            def select_action(self):
                return 0

            def learn(self, observation, reward=None):
                pass

        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = ConcreteAgent(config)

        # Set some beliefs
        agent.beliefs = np.array([0.7, 0.1, 0.1, 0.1])

        # Reset should clear beliefs
        agent.reset()
        assert agent.beliefs is None


class TestBasicExplorerAgent:
    """Test BasicExplorerAgent implementation."""

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_initialization_with_pymdp(self, mock_pymdp_agent):
        """Test initialization when PyMDP is available."""
        # Mock PyMDP agent
        mock_agent_instance = Mock()
        mock_pymdp_agent.return_value = mock_agent_instance

        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        assert agent.pymdp_agent == mock_agent_instance
        mock_pymdp_agent.assert_called_once()

    @patch("agents.base_agent.PYMDP_AVAILABLE", False)
    def test_initialization_without_pymdp(self):
        """Test initialization when PyMDP is not available."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        assert agent.pymdp_agent is None
        # Should initialize simple beliefs
        assert agent.beliefs.shape == (4,)
        assert np.allclose(agent.beliefs, 0.25)

    @patch("agents.base_agent.PYMDP_AVAILABLE", False)
    def test_update_beliefs_without_pymdp(self):
        """Test belief update without PyMDP."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

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

        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        agent.update_beliefs(1)

        assert np.array_equal(agent.beliefs, np.array([0.7, 0.1, 0.1, 0.1]))
        mock_agent_instance.infer_states.assert_called_once_with(1)

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    @patch("agents.base_agent.monitor_belief_update")
    def test_update_beliefs_with_monitoring(self, mock_monitor, mock_pymdp_agent):
        """Test belief update with monitoring enabled."""
        # Mock PyMDP agent
        mock_agent_instance = Mock()
        mock_agent_instance.infer_states.return_value = np.array([0.7, 0.1, 0.1, 0.1])
        mock_pymdp_agent.return_value = mock_agent_instance

        with patch("agents.base_agent.BELIEF_MONITORING_AVAILABLE", True):
            config = AgentConfig(
                agent_id="test_agent",
                num_states=[4],
                num_actions=[3],
                num_observations=[4],
            )
            agent = BasicExplorerAgent(config)

            agent.update_beliefs(1)

            mock_monitor.assert_called_once()

    @patch("agents.base_agent.PYMDP_AVAILABLE", False)
    def test_select_action_without_pymdp(self):
        """Test action selection without PyMDP."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

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

        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        action = agent.select_action()

        assert action == 2
        mock_agent_instance.infer_policies.assert_called_once()
        mock_agent_instance.sample_action.assert_called_once()

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_select_action_with_pymdp_error(self, mock_pymdp_agent):
        """Test action selection with PyMDP error."""
        # Mock PyMDP agent that raises error
        mock_agent_instance = Mock()
        mock_agent_instance.infer_policies.side_effect = Exception("PyMDP error")
        mock_pymdp_agent.return_value = mock_agent_instance

        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        with pytest.raises(ActionSelectionError):
            agent.select_action()

    def test_learn_method(self):
        """Test the learn method."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        agent.preferences = np.array([1.0, 0.0, 0.0, 0.0])
        initial_preferences = agent.preferences.copy()
        agent.learn(observation=1, reward=0.5)

        # Preferences should be updated
        assert not np.array_equal(agent.preferences, initial_preferences)

    def test_reset_method(self):
        """Test the reset method."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

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


class TestErrorHandling:
    """Test error handling in base agent."""

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_pymdp_initialization_error(self, mock_pymdp_agent):
        """Test handling of PyMDP initialization errors."""
        # Make PyMDP agent initialization fail
        mock_pymdp_agent.side_effect = Exception("PyMDP init error")

        # Should fall back to simple implementation
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        assert agent.pymdp_agent is None
        assert agent.beliefs is not None

    def test_invalid_observation(self):
        """Test handling of invalid observations."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

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

        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        # Should raise InferenceError
        with pytest.raises(InferenceError):
            agent.update_beliefs(1)


class TestObservabilityIntegration:
    """Test observability integration."""

    @patch("agents.base_agent.OBSERVABILITY_AVAILABLE", True)
    @patch("agents.base_agent.record_agent_lifecycle_event")
    def test_lifecycle_event_recording(self, mock_record_event):
        """Test that lifecycle events are recorded."""
        # Mock as coroutine
        mock_record_event.return_value = AsyncMock()
        
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        # Should have recorded creation event
        mock_record_event.assert_called()
        call_args = mock_record_event.call_args[1]
        assert call_args["agent_id"] == "test_agent"
        assert call_args["event_type"] == "created"

    @patch("agents.base_agent.OBSERVABILITY_AVAILABLE", True)
    @patch("agents.base_agent.monitor_pymdp_inference")
    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_inference_monitoring(self, mock_pymdp_agent, mock_monitor):
        """Test that inference is monitored."""
        # Mock PyMDP agent
        mock_agent_instance = Mock()
        mock_agent_instance.infer_states.return_value = np.array([0.7, 0.1, 0.1, 0.1])
        mock_agent_instance.infer_policies.return_value = None
        mock_agent_instance.sample_action.return_value = np.array([1])
        mock_pymdp_agent.return_value = mock_agent_instance

        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        # Update beliefs
        agent.update_beliefs(1)

        # Select action
        agent.select_action()

        # Should have monitored inference
        assert mock_monitor.call_count >= 1


class TestPerformanceOptimization:
    """Test performance optimization features."""

    @patch("agents.base_agent.performance_monitor")
    def test_performance_monitoring(self, mock_monitor):
        """Test that performance is monitored."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        # Perform operations
        agent.update_beliefs(1)
        agent.select_action()

        # Should have monitored performance
        assert mock_monitor.call_count >= 2

    def test_belief_caching(self):
        """Test belief caching optimization."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        # Update beliefs multiple times with same observation
        agent.update_beliefs(1)
        beliefs1 = agent.beliefs.copy()

        agent.update_beliefs(1)
        beliefs2 = agent.beliefs.copy()

        # Beliefs should be consistent
        assert np.allclose(beliefs1, beliefs2)


class TestMatrixValidation:
    """Test matrix validation in PyMDP operations."""

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_valid_matrix_dimensions(self, mock_pymdp_agent):
        """Test that matrix dimensions are validated."""
        # Create agent with specific dimensions
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4, 3],  # Multi-factor states
            num_actions=[5],
            num_observations=[4, 2],  # Multi-modality observations
        )
        
        # Mock should be called with correct dimensions
        agent = BasicExplorerAgent(config)
        
        # Verify PyMDP agent was initialized
        assert mock_pymdp_agent.called


class TestActionHistoryTracking:
    """Test action and observation history tracking."""

    def test_history_tracking(self):
        """Test that action and observation history are tracked."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        # Perform several steps
        for i in range(3):
            action = agent.step(i)
            assert len(agent.observation_history) == i + 1
            assert len(agent.action_history) == i + 1
            assert agent.observation_history[-1] == i
            assert agent.action_history[-1] == action

    def test_history_reset(self):
        """Test that history is cleared on reset."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
        )
        agent = BasicExplorerAgent(config)

        # Build up history
        for i in range(3):
            agent.step(i)

        # Reset should clear history
        agent.reset()
        assert agent.action_history == []
        assert agent.observation_history == []


class TestComplexScenarios:
    """Test complex scenarios and edge cases."""

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    def test_multi_factor_states(self, mock_pymdp_agent):
        """Test handling of multi-factor state spaces."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[3, 4, 2],  # 3 factors
            num_actions=[5],
            num_observations=[3, 3],  # 2 modalities
        )
        
        mock_agent_instance = Mock()
        mock_pymdp_agent.return_value = mock_agent_instance
        
        agent = BasicExplorerAgent(config)
        
        # Verify multi-factor handling
        assert agent.num_states == [3, 4, 2]

    def test_preference_learning_convergence(self):
        """Test that preference learning converges."""
        config = AgentConfig(
            agent_id="test_agent",
            num_states=[4],
            num_actions=[3],
            num_observations=[4],
            learning_rate=0.1,
        )
        agent = BasicExplorerAgent(config)
        
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
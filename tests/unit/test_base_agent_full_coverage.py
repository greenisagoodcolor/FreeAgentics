"""Comprehensive tests for base_agent.py module to achieve 100% coverage."""

import logging

# Mock modules before import
import sys
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

sys.modules["pymdp"] = MagicMock()
sys.modules["pymdp.utils"] = MagicMock()
sys.modules["pymdp.agent"] = MagicMock()
sys.modules["observability"] = MagicMock()
sys.modules["observability.agent_metrics_integration"] = MagicMock()
sys.modules["observability.belief_monitoring"] = MagicMock()

# Now import after mocking
from agents.base_agent import (
    ActiveInferenceAgent,
    AgentConfig,
    BasicExplorerAgent,
    safe_array_to_int,
)


class TestSafeArrayToInt:
    """Test safe_array_to_int function comprehensively."""

    def test_numpy_scalar_array(self):
        """Test conversion of 0-dimensional numpy array."""
        value = np.array(42)
        assert safe_array_to_int(value) == 42

    def test_single_element_array(self):
        """Test conversion of single element numpy array."""
        value = np.array([42])
        assert safe_array_to_int(value) == 42

    def test_multi_element_array(self):
        """Test conversion of multi-element array (takes first element)."""
        value = np.array([42, 43, 44])
        assert safe_array_to_int(value) == 42

    def test_multi_dimensional_array(self):
        """Test conversion of multi-dimensional array."""
        value = np.array([[42, 43], [44, 45]])
        assert safe_array_to_int(value) == 42

    def test_empty_numpy_array(self):
        """Test empty numpy array raises ValueError."""
        value = np.array([])
        with pytest.raises(ValueError, match="Empty array cannot be converted to integer"):
            safe_array_to_int(value)

    def test_list_input(self):
        """Test conversion of regular Python list."""
        value = [42, 43]
        assert safe_array_to_int(value) == 42

    def test_empty_list(self):
        """Test empty list raises ValueError."""
        value = []
        with pytest.raises(ValueError, match="Empty array cannot be converted to integer"):
            safe_array_to_int(value)

    def test_tuple_input(self):
        """Test conversion of tuple."""
        value = (42, 43)
        assert safe_array_to_int(value) == 42

    def test_numpy_scalar(self):
        """Test numpy scalar with item() method."""
        value = np.int64(42)
        assert safe_array_to_int(value) == 42

    def test_regular_int(self):
        """Test regular integer passes through."""
        assert safe_array_to_int(42) == 42

    def test_regular_float(self):
        """Test float conversion to int."""
        assert safe_array_to_int(42.7) == 42

    def test_invalid_type(self):
        """Test invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int("not a number")

    def test_none_value(self):
        """Test None value raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int(None)

    def test_object_with_item_method(self):
        """Test object with item() method."""

        class MockScalar:
            def item(self):
                return 42

        value = MockScalar()
        assert safe_array_to_int(value) == 42

    def test_object_with_len_and_getitem(self):
        """Test object with __len__ and __getitem__."""

        class MockArray:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return [42, 43][idx]

        value = MockArray()
        assert safe_array_to_int(value) == 42

    def test_exception_in_conversion(self):
        """Test exception during conversion."""

        class BadObject:
            def __int__(self):
                raise RuntimeError("Bad conversion")

        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int(BadObject())


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_config_creation(self):
        """Test configuration creation with required name."""
        config = AgentConfig(name="test_agent")
        assert config.name == "test_agent"
        assert config.use_pymdp is True
        assert config.planning_horizon == 3
        assert config.precision == 1.0
        assert config.lr == 0.1
        assert config.gmn_spec is None
        assert config.llm_config is None

    def test_custom_config(self):
        """Test custom configuration values."""
        llm_config = {"provider": "ollama", "model": "llama2"}
        gmn_spec = "test_spec"

        config = AgentConfig(
            name="custom_agent",
            use_pymdp=False,
            planning_horizon=5,
            precision=2.0,
            lr=0.05,
            llm_config=llm_config,
            gmn_spec=gmn_spec,
        )

        assert config.name == "custom_agent"
        assert config.use_pymdp is False
        assert config.planning_horizon == 5
        assert config.precision == 2.0
        assert config.lr == 0.05
        assert config.llm_config == llm_config
        assert config.gmn_spec == gmn_spec


class TestActiveInferenceAgent:
    """Test abstract ActiveInferenceAgent class."""

    @pytest.fixture
    def mock_agent_class(self):
        """Create a concrete implementation of ActiveInferenceAgent for testing."""

        class TestAgent(ActiveInferenceAgent):
            def setup_generative_model(self, *args, **kwargs):
                self.A = np.ones((2, 2))
                self.B = np.ones((2, 2, 1))
                self.C = np.ones((2, 1))
                self.D = np.ones((2,))
                self.pA = None
                self.pB = None

            def act(self, observation: int) -> int:
                self.update_beliefs(observation)
                return self.select_action()

        return TestAgent

    def test_agent_initialization(self, mock_agent_class):
        """Test agent initialization."""
        config = AgentConfig(name="test_agent")
        agent = mock_agent_class("test_id", config)

        assert agent.agent_id == "test_id"
        assert agent.config == config
        assert agent.name == "test_agent"
        assert agent.step_count == 0
        assert agent.observation_history == []
        assert agent.action_history == []
        assert hasattr(agent, "created_at")
        assert hasattr(agent, "metrics")
        assert agent.is_active is False

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.PyMDPAgent")
    @patch("agents.base_agent.validate_pymdp_matrices")
    def test_agent_with_pymdp(self, mock_validate, mock_pymdp_agent, mock_agent_class):
        """Test agent initialization with PyMDP available."""
        mock_validate.return_value = True
        mock_instance = MagicMock()
        mock_pymdp_agent.return_value = mock_instance

        config = AgentConfig(name="test", use_pymdp=True)
        agent = mock_agent_class("test_id", config)

        # Verify PyMDP agent is created
        assert hasattr(agent, "pymdp_agent")
        assert agent.pymdp_agent == mock_instance

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.validate_pymdp_matrices")
    def test_agent_with_pymdp_validation_failure(self, mock_validate, mock_agent_class):
        """Test agent initialization when PyMDP validation fails."""
        mock_validate.return_value = False

        config = AgentConfig(name="test", use_pymdp=True)
        agent = mock_agent_class("test_id", config)

        # PyMDP agent should not be created when validation fails
        assert not hasattr(agent, "pymdp_agent") or agent.pymdp_agent is None

    def test_initialization_method(self, mock_agent_class):
        """Test initialization method."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        # Initialization happens in __init__, so just verify state
        assert hasattr(agent, "A")
        assert hasattr(agent, "B")
        assert hasattr(agent, "C")
        assert hasattr(agent, "D")

    @patch("agents.base_agent.GMN_AVAILABLE", True)
    @patch("agents.base_agent.parse_gmn_spec")
    def test_gmn_initialization(self, mock_parse, mock_agent_class):
        """Test initialization with GMN spec."""
        gmn_spec = {"nodes": [], "edges": []}
        mock_parse.return_value = {
            "A": [np.eye(2)],
            "B": [np.ones((2, 2, 1))],
            "C": [np.ones((2, 1))],
            "D": [np.ones(2) / 2],
            "num_states": [2],
            "num_obs": [2],
            "num_actions": 1,
        }

        config = AgentConfig(name="test", gmn_spec=gmn_spec)
        agent = mock_agent_class("test_id", config)

        # GMN parsing should have been called
        mock_parse.assert_called_once()

    @patch("agents.base_agent.LLM_AVAILABLE", True)
    @patch("agents.base_agent.LocalLLMManager")
    def test_llm_initialization(self, mock_llm_manager, mock_agent_class):
        """Test initialization with LLM config."""
        llm_config = {"provider": "ollama", "model": "llama2"}
        mock_manager = MagicMock()
        mock_llm_manager.return_value = mock_manager

        config = AgentConfig(name="test", llm_config=llm_config)
        agent = mock_agent_class("test_id", config)

        assert agent.llm_manager == mock_manager

    def test_start_stop_methods(self, mock_agent_class):
        """Test start and stop methods."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        assert agent.is_active is False

        agent.start()
        assert agent.is_active is True

        agent.stop()
        assert agent.is_active is False

    def test_reset_method(self, mock_agent_class):
        """Test reset method."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        # Set some state
        agent.step_count = 5
        agent.observation_history = [1, 2, 3]
        agent.action_history = [0, 1, 0]

        # Reset
        with patch("agents.base_agent.record_agent_lifecycle_event") as mock_record:
            agent.reset()

        assert agent.step_count == 0
        assert agent.observation_history == []
        assert agent.action_history == []

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    def test_reset_with_pymdp(self, mock_agent_class):
        """Test reset with PyMDP agent."""
        config = AgentConfig(name="test", use_pymdp=True)
        agent = mock_agent_class("test_id", config)

        mock_pymdp = MagicMock()
        agent.pymdp_agent = mock_pymdp

        agent.reset()

        mock_pymdp.reset.assert_called_once()

    def test_update_beliefs(self, mock_agent_class):
        """Test belief update."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        with patch("agents.base_agent.monitor_belief_state") as mock_monitor:
            with patch("agents.base_agent.measure_belief_update") as mock_measure:
                agent.update_beliefs(1)

        assert 1 in agent.observation_history

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    def test_update_beliefs_with_pymdp(self, mock_agent_class):
        """Test belief update with PyMDP."""
        config = AgentConfig(name="test", use_pymdp=True)
        agent = mock_agent_class("test_id", config)

        mock_pymdp = MagicMock()
        mock_pymdp.infer_states.return_value = (np.array([0.7, 0.3]), {})
        agent.pymdp_agent = mock_pymdp

        agent.update_beliefs(1)

        mock_pymdp.infer_states.assert_called_once_with(1)
        assert hasattr(agent, "qs_current")
        assert np.array_equal(agent.qs_current, np.array([0.7, 0.3]))

    def test_select_action(self, mock_agent_class):
        """Test action selection."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        with patch("agents.base_agent.measure_inference_time") as mock_measure:
            action = agent.select_action()

        assert isinstance(action, int)
        assert action in agent.action_history

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    def test_select_action_with_pymdp(self, mock_agent_class):
        """Test action selection with PyMDP."""
        config = AgentConfig(name="test", use_pymdp=True)
        agent = mock_agent_class("test_id", config)

        mock_pymdp = MagicMock()
        mock_pymdp.sample_action.return_value = np.array([1])
        agent.pymdp_agent = mock_pymdp
        agent.qs_current = np.array([0.5, 0.5])

        action = agent.select_action()

        assert action == 1
        mock_pymdp.sample_action.assert_called_once()

    def test_select_action_error_handling(self, mock_agent_class):
        """Test action selection error handling."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        with patch("agents.base_agent.safe_pymdp_operation", side_effect=Exception("Test error")):
            action = agent.select_action()

        # Should return a valid action even on error
        assert isinstance(action, int)

    def test_step_method(self, mock_agent_class):
        """Test step method."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        with patch("agents.base_agent.measure_agent_step") as mock_measure:
            agent.step(1)

        assert agent.step_count == 1

    def test_step_with_error(self, mock_agent_class):
        """Test step with error handling."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        with patch.object(agent, "update_beliefs", side_effect=Exception("Test error")):
            agent.step(1)

        # Step count should still increment
        assert agent.step_count == 1

    def test_log_step(self, mock_agent_class):
        """Test log_step method."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        agent.observation_history = [1]
        agent.action_history = [0]

        # Should not raise
        agent.log_step()

    def test_get_state(self, mock_agent_class):
        """Test get_state method."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        agent.step_count = 5
        agent.observation_history = [1, 2]
        agent.action_history = [0, 1]

        state = agent.get_state()

        assert state["agent_id"] == "test_id"
        assert state["step_count"] == 5
        assert state["observation_history"] == [1, 2]
        assert state["action_history"] == [0, 1]
        assert "created_at" in state

    def test_set_state(self, mock_agent_class):
        """Test set_state method."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        state = {
            "agent_id": "test_id",
            "step_count": 10,
            "observation_history": [1, 2, 3],
            "action_history": [0, 1, 0],
            "created_at": datetime.now().isoformat(),
        }

        agent.set_state(state)

        assert agent.step_count == 10
        assert agent.observation_history == [1, 2, 3]
        assert agent.action_history == [0, 1, 0]

    def test_get_metrics(self, mock_agent_class):
        """Test get_metrics method."""
        config = AgentConfig(name="test")
        agent = mock_agent_class("test_id", config)

        agent.step_count = 5
        agent.observation_history = [1, 2, 3]
        agent.action_history = [0, 1]

        metrics = agent.get_metrics()

        assert metrics["step_count"] == 5
        assert metrics["total_observations"] == 3
        assert metrics["total_actions"] == 2


class TestBasicExplorerAgent:
    """Test BasicExplorerAgent implementation."""

    def test_initialization(self):
        """Test BasicExplorerAgent initialization."""
        config = AgentConfig(name="explorer")
        agent = BasicExplorerAgent("explorer_id", config, num_states=4, num_actions=2)

        assert agent.agent_id == "explorer_id"
        assert agent.num_states == 4
        assert agent.num_actions == 2
        assert agent.A.shape == (4, 4)
        assert agent.B.shape == (4, 4, 2)
        assert agent.C.shape == (4, 1)
        assert agent.D.shape == (4,)

    def test_transition_matrix_normalization(self):
        """Test that transition matrices are properly normalized."""
        config = AgentConfig(name="explorer")
        agent = BasicExplorerAgent("explorer", config, num_states=4, num_actions=2)

        # Each column should sum to 1
        for action in range(agent.num_actions):
            for state in range(agent.num_states):
                assert np.isclose(agent.B[:, state, action].sum(), 1.0)

    def test_act_method(self):
        """Test act method returns valid action."""
        config = AgentConfig(name="explorer")
        agent = BasicExplorerAgent("explorer", config, num_states=4, num_actions=2)

        action = agent.act(2)
        assert 0 <= action < agent.num_actions

    @patch("agents.base_agent.PYMDP_AVAILABLE", False)
    def test_act_without_pymdp(self):
        """Test agent behavior when PyMDP is not available."""
        config = AgentConfig(name="explorer", use_pymdp=False)
        agent = BasicExplorerAgent("explorer", config, num_states=4, num_actions=2)

        action = agent.act(1)
        assert 0 <= action < agent.num_actions

    def test_error_handling_in_act(self):
        """Test error handling in act method."""
        config = AgentConfig(name="explorer")
        agent = BasicExplorerAgent("explorer", config, num_states=4, num_actions=2)

        with patch.object(agent, "update_beliefs", side_effect=Exception("Test error")):
            action = agent.act(1)
            assert 0 <= action < agent.num_actions

    def test_act_with_invalid_observation(self):
        """Test act with out of bounds observation."""
        config = AgentConfig(name="explorer")
        agent = BasicExplorerAgent("explorer", config, num_states=4, num_actions=2)

        # Large observation index
        action = agent.act(999)
        assert 0 <= action < agent.num_actions


class TestObservabilityIntegration:
    """Test observability integration."""

    @patch("agents.base_agent.OBSERVABILITY_AVAILABLE", True)
    def test_lifecycle_event_recording(self):
        """Test lifecycle events are recorded."""
        config = AgentConfig(name="test")
        agent = BasicExplorerAgent("test", config, num_states=2, num_actions=1)

        with patch("agents.base_agent.record_agent_lifecycle_event") as mock_record:
            agent.reset()
            mock_record.assert_called_with(agent.agent_id, "reset")

    @patch("agents.base_agent.OBSERVABILITY_AVAILABLE", True)
    def test_belief_update_monitoring(self):
        """Test belief updates are monitored."""
        config = AgentConfig(name="test")
        agent = BasicExplorerAgent("test", config, num_states=2, num_actions=1)

        with patch("agents.base_agent.monitor_belief_state") as mock_monitor:
            agent.update_beliefs(1)
            assert mock_monitor.called

    @patch("agents.base_agent.OBSERVABILITY_AVAILABLE", True)
    def test_step_performance_monitoring(self):
        """Test step performance is monitored."""
        config = AgentConfig(name="test")
        agent = BasicExplorerAgent("test", config, num_states=2, num_actions=1)

        with patch("agents.base_agent.measure_agent_step") as mock_measure:
            agent.step(1)
            assert mock_measure.called

    @patch("agents.base_agent.OBSERVABILITY_AVAILABLE", False)
    def test_no_observability(self):
        """Test behavior when observability is not available."""
        config = AgentConfig(name="test")
        agent = BasicExplorerAgent("test", config, num_states=2, num_actions=1)

        # Should work normally without observability
        agent.step(1)
        agent.reset()
        assert agent.step_count == 0


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_matrix_validation_error(self):
        """Test handling of invalid generative model matrices."""
        with patch("agents.base_agent.validate_pymdp_matrices", return_value=False):
            config = AgentConfig(name="test")
            agent = BasicExplorerAgent("test", config, num_states=2, num_actions=1)
            assert agent is not None

    def test_pymdp_import_error(self):
        """Test handling when PyMDP import fails."""
        with patch("agents.base_agent.PYMDP_AVAILABLE", False):
            config = AgentConfig(name="test", use_pymdp=True)
            agent = BasicExplorerAgent("test", config, num_states=2, num_actions=1)
            assert not hasattr(agent, "pymdp_agent")

    def test_step_with_monitoring_error(self):
        """Test step continues even if monitoring fails."""
        config = AgentConfig(name="test")
        agent = BasicExplorerAgent("test", config, num_states=2, num_actions=1)

        with patch("agents.base_agent.measure_agent_step", side_effect=Exception("Monitor error")):
            agent.step(1)
            assert agent.step_count == 1

"""
Simple working tests for Base Agent functionality
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Mock imports before importing base_agent
with patch.dict(
    "sys.modules",
    {
        "pymdp": Mock(),
        "pymdp.utils": Mock(),
        "pymdp.agent": Mock(),
        "observability": Mock(),
        "observability.belief_monitoring": Mock(),
    },
):
    from agents.base_agent import (
        ActiveInferenceAgent,
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


class TestActiveInferenceAgentBasic:
    """Test basic ActiveInferenceAgent functionality."""

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

    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.agent_id == "test_agent"
        assert self.agent.name == "Test Agent"
        assert self.agent.config["use_pymdp"] is False
        assert self.agent.is_active is False
        assert self.agent.created_at is not None
        assert isinstance(self.agent.created_at, datetime)
        assert self.agent.total_steps == 0

    def test_agent_activation(self):
        """Test agent activation."""
        assert self.agent.is_active is False

        # Activate agent
        self.agent.is_active = True
        assert self.agent.is_active is True

    def test_agent_step_counter(self):
        """Test that agent step counter works."""
        initial_steps = self.agent.total_steps
        assert initial_steps == 0

        # Simulate step increment
        self.agent.total_steps += 1
        assert self.agent.total_steps == 1

    def test_agent_config_access(self):
        """Test agent configuration access."""
        config = self.agent.config
        assert isinstance(config, dict)
        assert "use_pymdp" in config
        assert config["use_pymdp"] is False

    def test_agent_performance_mode(self):
        """Test agent performance mode setting."""
        # Default performance mode
        assert self.agent.performance_mode == "balanced"

        # Create agent with custom performance mode
        class CustomAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                pass

            def update_beliefs(self):
                pass

            def select_action(self):
                return 0

        custom_agent = CustomAgent(
            agent_id="custom_agent",
            name="Custom Agent",
            config={"performance_mode": "fast"},
        )

        assert custom_agent.performance_mode == "fast"

    def test_agent_matrix_cache(self):
        """Test agent matrix cache functionality."""
        # Initial cache should be empty
        assert isinstance(self.agent.matrix_cache, dict)
        assert len(self.agent.matrix_cache) == 0

        # Add something to cache
        self.agent.matrix_cache["test_matrix"] = np.eye(2)
        assert "test_matrix" in self.agent.matrix_cache
        assert isinstance(self.agent.matrix_cache["test_matrix"], np.ndarray)

    def test_agent_beliefs_initialization(self):
        """Test agent beliefs initialization."""
        assert isinstance(self.agent.beliefs, dict)

        # Update beliefs
        self.agent.update_beliefs()
        assert "test" in self.agent.beliefs
        assert self.agent.beliefs["test"] == 0.5

    def test_agent_preferences_initialization(self):
        """Test agent preferences initialization."""
        assert isinstance(self.agent.preferences, dict)

        # Add preferences
        self.agent.preferences["preferred_state"] = 1.0
        assert "preferred_state" in self.agent.preferences
        assert self.agent.preferences["preferred_state"] == 1.0

    def test_agent_policies_initialization(self):
        """Test agent policies initialization."""
        assert isinstance(self.agent.policies, list)
        assert len(self.agent.policies) == 0

        # Add policy
        self.agent.policies.append([0, 1, 0])
        assert len(self.agent.policies) == 1
        assert self.agent.policies[0] == [0, 1, 0]

    def test_agent_gmn_spec(self):
        """Test agent GMN specification."""
        assert self.agent.gmn_spec is None

        # Set GMN spec
        self.agent.gmn_spec = "test_gmn_spec"
        assert self.agent.gmn_spec == "test_gmn_spec"

    def test_agent_perceive_method(self):
        """Test agent perceive method."""
        observation = {"sensor": "value"}

        # Call perceive
        self.agent.perceive(observation)

        # Check that observation was stored
        assert hasattr(self.agent, 'last_observation')
        assert self.agent.last_observation == observation

    def test_agent_select_action_method(self):
        """Test agent select_action method."""
        action = self.agent.select_action()
        assert action == 0
        assert isinstance(action, int)

    def test_agent_pymdp_integration(self):
        """Test agent PyMDP integration."""
        # Initially no PyMDP agent
        assert self.agent.pymdp_agent is None

        # Can set PyMDP agent
        mock_pymdp = Mock()
        self.agent.pymdp_agent = mock_pymdp
        assert self.agent.pymdp_agent == mock_pymdp

    def test_agent_timing_tracking(self):
        """Test agent timing tracking."""
        assert self.agent.last_action_at is None

        # Set action time
        action_time = datetime.now()
        self.agent.last_action_at = action_time
        assert self.agent.last_action_at == action_time

    def test_agent_selective_update_interval(self):
        """Test agent selective update interval."""
        # Default interval
        assert self.agent.selective_update_interval == 1

        # Create agent with custom interval
        class IntervalAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                pass

            def update_beliefs(self):
                pass

            def select_action(self):
                return 0

        interval_agent = IntervalAgent(
            agent_id="interval_agent",
            name="Interval Agent",
            config={"selective_update_interval": 5},
        )

        assert interval_agent.selective_update_interval == 5

    def test_agent_with_empty_config(self):
        """Test agent with empty configuration."""

        class EmptyConfigAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                pass

            def update_beliefs(self):
                pass

            def select_action(self):
                return 0

        empty_agent = EmptyConfigAgent(
            agent_id="empty_agent", name="Empty Agent", config={}
        )

        assert isinstance(empty_agent.config, dict)
        assert len(empty_agent.config) == 0
        assert empty_agent.performance_mode == "balanced"

    def test_agent_with_partial_config(self):
        """Test agent with partial configuration."""

        class PartialConfigAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                pass

            def update_beliefs(self):
                pass

            def select_action(self):
                return 0

        partial_agent = PartialConfigAgent(
            agent_id="partial_agent",
            name="Partial Agent",
            config={"custom_param": "value"},
        )

        assert "custom_param" in partial_agent.config
        assert partial_agent.config["custom_param"] == "value"
        assert partial_agent.performance_mode == "balanced"  # Default value

    def test_agent_multiple_instances(self):
        """Test multiple agent instances."""

        class MultiAgent(ActiveInferenceAgent):
            def perceive(self, observation):
                pass

            def update_beliefs(self):
                pass

            def select_action(self):
                return 0

        agent1 = MultiAgent("agent1", "Agent 1", {"param": "value1"})
        agent2 = MultiAgent("agent2", "Agent 2", {"param": "value2"})

        assert agent1.agent_id != agent2.agent_id
        assert agent1.name != agent2.name
        assert agent1.config["param"] != agent2.config["param"]
        assert agent1.created_at != agent2.created_at  # Different timestamps

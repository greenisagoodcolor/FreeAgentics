"""Coverage tests for agents/base_agent.py focusing on untested code paths."""

import importlib
from unittest.mock import Mock, patch

import numpy as np
import pytest

from agents.base_agent import (
    ActiveInferenceAgent,
    AgentConfig,
    BasicExplorerAgent,
    _get_llm_manager,
    _get_pymdp_components,
    safe_array_to_int,
)


class TestSafeArrayToInt:
    """Test safe_array_to_int function."""

    def test_numpy_scalar(self):
        """Test with numpy scalar."""
        value = np.array(42)
        assert safe_array_to_int(value) == 42

    def test_numpy_array_single(self):
        """Test with single element array."""
        value = np.array([42])
        assert safe_array_to_int(value) == 42

    def test_numpy_array_multi(self):
        """Test with multi-element array."""
        value = np.array([42, 10, 5])
        assert safe_array_to_int(value) == 42

    def test_numpy_2d_array(self):
        """Test with 2D array."""
        value = np.array([[42, 10], [5, 3]])
        assert safe_array_to_int(value) == 42

    def test_empty_numpy_array(self):
        """Test with empty array."""
        value = np.array([])
        with pytest.raises(ValueError, match="Empty array"):
            safe_array_to_int(value)

    def test_python_list(self):
        """Test with Python list."""
        assert safe_array_to_int([42, 10]) == 42

    def test_empty_list(self):
        """Test with empty list."""
        with pytest.raises(ValueError, match="Empty array"):
            safe_array_to_int([])

    def test_regular_int(self):
        """Test with regular int."""
        assert safe_array_to_int(42) == 42

    def test_regular_float(self):
        """Test with float."""
        assert safe_array_to_int(42.7) == 42

    def test_numpy_int64(self):
        """Test with numpy int64."""
        value = np.int64(42)
        assert safe_array_to_int(value) == 42

    def test_invalid_type(self):
        """Test with invalid type."""
        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int("not a number")


class TestPyMDPComponents:
    """Test PyMDP component loading."""

    def test_get_pymdp_components_success(self):
        """Test successful PyMDP import."""
        # Reset the module-level variables
        import agents.base_agent

        agents.base_agent.PYMDP_AVAILABLE = None
        agents.base_agent._pymdp_utils = None
        agents.base_agent._PyMDPAgent = None

        with patch.object(importlib, "import_module") as mock_import:
            mock_pymdp = Mock()
            mock_pymdp.utils = Mock()
            mock_pymdp.agent.Agent = Mock
            mock_import.return_value = mock_pymdp

            available, utils, agent_class = _get_pymdp_components()

            assert available is True
            assert utils is not None
            assert agent_class is not None

    def test_get_pymdp_components_failure(self):
        """Test PyMDP import failure."""
        import agents.base_agent

        agents.base_agent.PYMDP_AVAILABLE = None

        with patch.object(importlib, "import_module", side_effect=ImportError):
            available, utils, agent_class = _get_pymdp_components()

            assert available is False
            assert utils is None
            assert agent_class is None


class TestLLMManager:
    """Test LLM manager loading."""

    def test_get_llm_manager_success(self):
        """Test successful LLM import."""
        mock_agent = Mock()
        mock_agent.agent_id = "test"

        with patch.object(importlib, "import_module") as mock_import:
            mock_llm = Mock()
            mock_llm.LocalLLMManager = Mock
            mock_import.return_value = mock_llm

            import agents.base_agent

            agents.base_agent.LLM_MANAGER = None

            manager = _get_llm_manager(mock_agent)
            assert manager is not None

    def test_get_llm_manager_failure(self):
        """Test LLM import failure."""
        mock_agent = Mock()

        with patch.object(importlib, "import_module", side_effect=ImportError):
            import agents.base_agent

            agents.base_agent.LLM_MANAGER = None

            manager = _get_llm_manager(mock_agent)
            assert manager is None


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_config_creation(self):
        """Test creating config."""
        config = AgentConfig(
            name="TestAgent", use_pymdp=True, planning_horizon=5, precision=2.0, lr=0.01
        )
        assert config.name == "TestAgent"
        assert config.use_pymdp is True
        assert config.planning_horizon == 5

    def test_config_defaults(self):
        """Test config defaults."""
        config = AgentConfig(name="Test")
        assert config.use_pymdp is True
        assert config.planning_horizon == 3
        assert config.precision == 1.0
        assert config.lr == 0.1
        assert config.gmn_spec is None
        assert config.llm_config is None


class TestActiveInferenceAgent:
    """Test ActiveInferenceAgent abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            ActiveInferenceAgent("test", "Test Agent")

    def test_subclass_implementation(self):
        """Test creating a concrete subclass."""

        class ConcreteAgent(ActiveInferenceAgent):
            def perceive(self, obs):
                return obs

            def update_beliefs(self, obs):
                return {}

            def plan_actions(self, beliefs):
                return ["stay"]

            def select_action(self, policies):
                return "stay"

            def execute_action(self, action):
                return action

        agent = ConcreteAgent("test_id", "Test Agent")
        assert agent.agent_id == "test_id"
        assert agent.name == "Test Agent"
        assert agent.is_active is False
        assert agent.total_steps == 0


class TestBasicExplorerAgent:
    """Test BasicExplorerAgent implementation."""

    def test_initialization(self):
        """Test BasicExplorerAgent initialization."""
        agent = BasicExplorerAgent("explorer1", "Explorer", {"use_pymdp": False})
        assert agent.agent_id == "explorer1"
        assert agent.name == "Explorer"

    def test_perceive(self):
        """Test perceive method."""
        agent = BasicExplorerAgent("explorer", "Explorer")
        obs = {"position": [1, 2], "walls": [[0, 0]]}

        result = agent.perceive(obs)
        assert result == obs

    def test_update_beliefs_no_pymdp(self):
        """Test update_beliefs without PyMDP."""
        agent = BasicExplorerAgent("explorer", "Explorer", {"use_pymdp": False})

        obs = {"position": [1, 2]}
        beliefs = agent.update_beliefs(obs)

        assert isinstance(beliefs, dict)
        assert "position" in beliefs

    @patch("agents.base_agent._get_pymdp_components")
    def test_update_beliefs_with_pymdp(self, mock_get_pymdp):
        """Test update_beliefs with PyMDP available."""
        # Mock PyMDP components
        mock_utils = Mock()
        mock_agent_class = Mock()
        mock_get_pymdp.return_value = (True, mock_utils, mock_agent_class)

        agent = BasicExplorerAgent("explorer", "Explorer", {"use_pymdp": True})
        agent.pymdp_agent = Mock()
        agent.pymdp_agent.infer_states.return_value = (np.array([0.2, 0.8]), None)

        obs = {"position": [1, 2]}
        beliefs = agent.update_beliefs(obs)

        assert isinstance(beliefs, dict)

    def test_plan_actions(self):
        """Test plan_actions method."""
        agent = BasicExplorerAgent("explorer", "Explorer")

        beliefs = {"position": [1, 2], "confidence": 0.8}
        actions = agent.plan_actions(beliefs)

        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_select_action_high_confidence(self):
        """Test action selection with high confidence."""
        agent = BasicExplorerAgent("explorer", "Explorer")

        # High confidence - should choose first policy
        agent.last_confidence = 0.9
        policies = ["up", "down", "left", "right"]

        action = agent.select_action(policies)
        assert action == "up"

    def test_select_action_low_confidence(self):
        """Test action selection with low confidence - exploration."""
        agent = BasicExplorerAgent("explorer", "Explorer")

        # Low confidence - should explore
        agent.last_confidence = 0.3
        policies = ["up", "down", "left", "right"]

        # Run multiple times to verify randomness
        actions = set()
        for _ in range(20):
            action = agent.select_action(policies)
            actions.add(action)

        # Should have variety due to exploration
        assert len(actions) > 1

    def test_execute_action(self):
        """Test execute_action method."""
        agent = BasicExplorerAgent("explorer", "Explorer")

        result = agent.execute_action("up")
        assert result == {"action": "up", "timestamp": result["timestamp"]}
        assert agent.total_steps == 1
        assert agent.last_action_at is not None

    def test_step_full_cycle(self):
        """Test full agent step cycle."""
        agent = BasicExplorerAgent("explorer", "Explorer", {"use_pymdp": False})

        obs = {"position": [5, 5], "walls": []}
        action_result = agent.step(obs)

        assert "action" in action_result
        assert "timestamp" in action_result
        assert agent.total_steps == 1

    def test_activate_deactivate(self):
        """Test agent activation/deactivation."""
        agent = BasicExplorerAgent("explorer", "Explorer")

        assert agent.is_active is False

        agent.activate()
        assert agent.is_active is True

        agent.deactivate()
        assert agent.is_active is False

    def test_get_state(self):
        """Test getting agent state."""
        agent = BasicExplorerAgent("explorer", "Explorer")
        agent.total_steps = 5
        agent.last_confidence = 0.75

        state = agent.get_state()

        assert state["agent_id"] == "explorer"
        assert state["name"] == "Explorer"
        assert state["is_active"] is False
        assert state["total_steps"] == 5
        assert state["performance_mode"] == "balanced"

    def test_handle_error(self):
        """Test error handling in step."""
        agent = BasicExplorerAgent("explorer", "Explorer")

        # Force an error by mocking perceive to raise
        with patch.object(agent, "perceive", side_effect=Exception("Test error")):
            result = agent.step({"position": [1, 2]})

            assert result["action"] == "stay"
            assert "error" in result


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_safe_array_index(self):
        """Test safe_array_index if it exists."""
        try:
            from agents.pymdp_error_handling import safe_array_index

            arr = np.array([1, 2, 3])
            assert safe_array_index(arr, 0) == 1
            assert safe_array_index(arr, 5, default=-1) == -1
        except ImportError:
            pytest.skip("pymdp_error_handling module not available")

    def test_validate_observation_integration(self):
        """Test validate_observation usage."""
        agent = BasicExplorerAgent("test", "Test")

        # Valid observation
        obs = {"position": [1, 2]}
        processed = agent.perceive(obs)
        assert processed == obs

        # None observation should be handled
        processed = agent.perceive(None)
        assert processed is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

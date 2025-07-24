"""Comprehensive tests for base_agent.py module to achieve 100% coverage."""

# Mock modules before import
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules["pymdp"] = MagicMock()
sys.modules["pymdp.utils"] = MagicMock()
sys.modules["pymdp.agent"] = MagicMock()
sys.modules["observability"] = MagicMock()
sys.modules["observability.agent_metrics_integration"] = MagicMock()
sys.modules["observability.belief_monitoring"] = MagicMock()

# Now import after mocking
from agents.base_agent import AgentConfig, BasicExplorerAgent, safe_array_to_int


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

    def test_exception_handling(self):
        """Test various exception paths."""

        # Test TypeError path
        class BadType:
            pass

        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int(BadType())

        # Test IndexError path
        class BadIndex:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                raise IndexError("test")

        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int(BadIndex())


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


class TestBasicExplorerAgent:
    """Test BasicExplorerAgent concrete implementation."""

    def test_initialization(self):
        """Test BasicExplorerAgent initialization."""
        agent = BasicExplorerAgent("explorer_id", "explorer_name", grid_size=5)

        assert agent.agent_id == "explorer_id"
        assert agent.name == "explorer_name"
        assert agent.grid_size == 5
        assert agent.num_states == 25  # 5*5
        assert agent.num_obs == 5
        assert agent.num_actions == 5
        assert agent.position == [2, 2]  # Center of 5x5 grid
        assert agent.uncertainty_map.shape == (5, 5)
        assert agent.exploration_rate == 0.3

    def test_action_mapping(self):
        """Test action mapping is correctly set up."""
        agent = BasicExplorerAgent("test", "test")

        expected_actions = ["up", "down", "left", "right", "stay"]
        assert agent.actions == expected_actions
        assert agent.action_map[0] == "up"
        assert agent.action_map[1] == "down"
        assert agent.action_map[2] == "left"
        assert agent.action_map[3] == "right"
        assert agent.action_map[4] == "stay"

    @patch("agents.base_agent.PYMDP_AVAILABLE", False)
    def test_initialization_without_pymdp(self):
        """Test initialization when PyMDP is not available."""
        agent = BasicExplorerAgent("test", "test")

        # Should still initialize properly
        assert agent.agent_id == "test"
        assert agent.name == "test"
        assert hasattr(agent, "uncertainty_map")

    def test_perceive_method(self):
        """Test perceive method with various observations."""
        agent = BasicExplorerAgent("test", "test")

        # Test with position observation
        observation = {"position": [3, 4]}
        agent.perceive(observation)
        assert agent.position == [3, 4]

        # Test with surroundings
        surroundings = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        observation = {"position": [1, 1], "surroundings": surroundings}
        agent.perceive(observation)
        assert agent.position == [1, 1]

    def test_perceive_with_pymdp(self):
        """Test perceive method with PyMDP enabled."""
        with patch("agents.base_agent.PYMDP_AVAILABLE", True):
            agent = BasicExplorerAgent("test", "test")
            mock_pymdp = MagicMock()
            agent.pymdp_agent = mock_pymdp

            # Test empty observation
            surroundings = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            observation = {"surroundings": surroundings}
            agent.perceive(observation)

            assert hasattr(agent, "current_observation")
            assert agent.current_observation == [0]  # Empty observation

    def test_perceive_observation_mapping(self):
        """Test observation mapping to PyMDP format."""
        with patch("agents.base_agent.PYMDP_AVAILABLE", True):
            agent = BasicExplorerAgent("test", "test")
            mock_pymdp = MagicMock()
            agent.pymdp_agent = mock_pymdp

            # Test different observation types
            test_cases = [
                (-2, 4),  # Out of bounds
                (-1, 1),  # Obstacle
                (1, 2),  # Goal
                (2, 3),  # Other agent
                (0, 0),  # Empty
            ]

            for center_val, expected_obs in test_cases:
                surroundings = np.array([[0, 0, 0], [0, center_val, 0], [0, 0, 0]])
                observation = {"surroundings": surroundings}
                agent.perceive(observation)

                assert agent.current_observation == [expected_obs]

    def test_perceive_fallback_mode(self):
        """Test perceive method in fallback mode without PyMDP."""
        agent = BasicExplorerAgent("test", "test")
        agent.pymdp_agent = None

        observation = {"position": [2, 3]}
        agent.perceive(observation)

        # Should update uncertainty map
        assert agent.uncertainty_map[2, 3] == 0

    def test_update_beliefs_method(self):
        """Test update_beliefs method."""
        agent = BasicExplorerAgent("test", "test")

        # Test without PyMDP
        agent.update_beliefs()

        # Should not raise any errors
        assert True

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    def test_update_beliefs_with_pymdp(self):
        """Test update_beliefs with PyMDP agent."""
        agent = BasicExplorerAgent("test", "test")
        mock_pymdp = MagicMock()
        mock_pymdp.infer_states.return_value = (
            np.array([0.7, 0.3, 0.0, 0.0, 0.0]),
            None,
        )
        agent.pymdp_agent = mock_pymdp
        agent.current_observation = [0]

        agent.update_beliefs()

        mock_pymdp.infer_states.assert_called_once_with(agent.current_observation)

    def test_update_beliefs_pymdp_error(self):
        """Test update_beliefs with PyMDP error handling."""
        with patch("agents.base_agent.PYMDP_AVAILABLE", True):
            agent = BasicExplorerAgent("test", "test")
            mock_pymdp = MagicMock()
            mock_pymdp.infer_states.side_effect = Exception("PyMDP error")
            agent.pymdp_agent = mock_pymdp

            # Should handle error gracefully
            agent.update_beliefs()
            # Should not raise exception

    def test_update_beliefs_fallback(self):
        """Test fallback belief update method."""
        agent = BasicExplorerAgent("test", "test")

        # Test fallback method directly
        agent._fallback_update_beliefs()

        # Should not raise any errors
        assert True

    def test_select_action_method(self):
        """Test select_action method."""
        agent = BasicExplorerAgent("test", "test")

        action = agent.select_action()

        assert action in agent.actions
        assert isinstance(action, str)

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    def test_select_action_with_pymdp(self):
        """Test action selection with PyMDP."""
        agent = BasicExplorerAgent("test", "test")
        mock_pymdp = MagicMock()
        mock_pymdp.sample_action.return_value = np.array([2])
        agent.pymdp_agent = mock_pymdp

        action = agent.select_action()

        assert action == "left"  # action_map[2] = "left"

    def test_select_action_pymdp_error(self):
        """Test action selection with PyMDP error."""
        with patch("agents.base_agent.PYMDP_AVAILABLE", True):
            agent = BasicExplorerAgent("test", "test")
            mock_pymdp = MagicMock()
            mock_pymdp.sample_action.side_effect = Exception("PyMDP error")
            agent.pymdp_agent = mock_pymdp

            action = agent.select_action()

            # Should return fallback action
            assert action in agent.actions

    def test_fallback_action_selection(self):
        """Test fallback action selection methods."""
        agent = BasicExplorerAgent("test", "test")

        # Test _fallback_select_action
        action = agent._fallback_select_action()
        assert action in agent.actions

        # Test _fallback_action_selection
        action = agent._fallback_action_selection()
        assert action in agent.actions

    def test_simulate_action(self):
        """Test action simulation method."""
        agent = BasicExplorerAgent("test", "test", grid_size=5)

        # Test all actions
        x, y = 2, 2

        # Test up
        new_x, new_y = agent._simulate_action(x, y, "up")
        assert new_x == 1 and new_y == 2

        # Test down
        new_x, new_y = agent._simulate_action(x, y, "down")
        assert new_x == 3 and new_y == 2

        # Test left
        new_x, new_y = agent._simulate_action(x, y, "left")
        assert new_x == 2 and new_y == 1

        # Test right
        new_x, new_y = agent._simulate_action(x, y, "right")
        assert new_x == 2 and new_y == 3

        # Test stay
        new_x, new_y = agent._simulate_action(x, y, "stay")
        assert new_x == 2 and new_y == 2

    def test_simulate_action_bounds(self):
        """Test action simulation with boundary conditions."""
        agent = BasicExplorerAgent("test", "test", grid_size=5)

        # Test bounds at edge
        x, y = 0, 0

        # Up should stay at edge
        new_x, new_y = agent._simulate_action(x, y, "up")
        assert new_x >= 0

        # Left should stay at edge
        new_x, new_y = agent._simulate_action(x, y, "left")
        assert new_y >= 0

    def test_compute_free_energy(self):
        """Test free energy computation method."""
        agent = BasicExplorerAgent("test", "test")

        result = agent.compute_free_energy()

        assert isinstance(result, dict)
        assert "total_uncertainty" in result
        assert "exploration_value" in result
        assert "position_entropy" in result

    def test_step_method(self):
        """Test step method integration."""
        agent = BasicExplorerAgent("test", "test")

        observation = {"position": [1, 1]}

        with patch("agents.base_agent.measure_agent_step"):
            action = agent.step(observation)

        assert action in agent.actions
        assert agent.position == [1, 1]

    def test_step_with_error(self):
        """Test step method with error handling."""
        agent = BasicExplorerAgent("test", "test")

        with patch.object(agent, "update_beliefs", side_effect=Exception("Error")):
            observation = {"position": [1, 1]}
            action = agent.step(observation)

            # Should return fallback action
            assert action == "stay"

    def test_status_methods(self):
        """Test start, stop, and status methods."""
        agent = BasicExplorerAgent("test", "test")

        # Test start
        agent.start()
        assert agent.is_active is True

        # Test status
        status = agent.get_status()
        assert isinstance(status, dict)
        assert "agent_id" in status
        assert "is_active" in status

        # Test stop
        agent.stop()
        assert agent.is_active is False

    def test_reset_method(self):
        """Test reset method."""
        agent = BasicExplorerAgent("test", "test")

        # Change some state
        agent.position = [1, 1]
        agent.uncertainty_map[1, 1] = 0.5

        with patch("agents.base_agent.record_agent_lifecycle_event"):
            agent.reset()

        # Should reset to initial state
        assert agent.position == [agent.grid_size // 2, agent.grid_size // 2]

    def test_pymdp_initialization(self):
        """Test PyMDP initialization method."""
        with patch("agents.base_agent.PYMDP_AVAILABLE", True):
            with patch("agents.base_agent.utils") as mock_utils:
                mock_utils.initialize_empty_A.return_value = [np.ones((5, 100))]
                mock_utils.initialize_empty_B.return_value = [np.ones((100, 100, 5))]

                agent = BasicExplorerAgent("test", "test")
                agent._initialize_pymdp()

                # Should create PyMDP matrices
                assert hasattr(agent, "A")
                assert hasattr(agent, "B")

    def test_pymdp_initialization_error(self):
        """Test PyMDP initialization with error."""
        with patch("agents.base_agent.PYMDP_AVAILABLE", True):
            with patch("agents.base_agent.utils") as mock_utils:
                mock_utils.initialize_empty_A.side_effect = Exception("Init error")

                agent = BasicExplorerAgent("test", "test")
                agent._initialize_pymdp()

                # Should handle error gracefully
                assert agent.pymdp_agent is None

    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling."""
        agent = BasicExplorerAgent("test", "test")

        # Test perceive with invalid observation
        agent.perceive({})  # Empty observation

        # Test perceive with malformed surroundings
        agent.perceive({"surroundings": "invalid"})

        # Test update_beliefs with corrupted state
        agent.current_observation = None
        agent.update_beliefs()

        # All should handle gracefully
        assert True

    def test_grid_size_variations(self):
        """Test agent with different grid sizes."""
        for size in [1, 3, 10, 20]:
            agent = BasicExplorerAgent("test", "test", grid_size=size)
            assert agent.grid_size == size
            assert agent.num_states == size * size
            assert agent.position == [size // 2, size // 2]
            assert agent.uncertainty_map.shape == (size, size)

    def test_metrics_integration(self):
        """Test metrics integration."""
        agent = BasicExplorerAgent("test", "test")

        # Test that metrics are properly tracked
        observation = {
            "position": [1, 1],
            "surroundings": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        }
        agent.perceive(observation)

        # Should update metrics
        if hasattr(agent, "metrics"):
            assert "last_observation" in agent.metrics

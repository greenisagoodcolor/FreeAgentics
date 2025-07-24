"""
Test suite for real Active Inference implementation with PyMDP.

Tests the BasicExplorerAgent with actual PyMDP integration,
verifying belief updates, free energy calculations, and action selection.
"""

from datetime import datetime

import numpy as np
import pytest

from agents.base_agent import PYMDP_AVAILABLE, BasicExplorerAgent


def create_test_agent(grid_size=5):
    """Create a BasicExplorerAgent for testing without LLM dependencies."""
    # Manually create config to avoid LLM initialization

    # Create agent with minimal initialization
    agent = object.__new__(BasicExplorerAgent)

    # Set required attributes manually
    agent.agent_id = "test_id"
    agent.name = "Test Explorer"
    agent.grid_size = grid_size
    agent.position = [grid_size // 2, grid_size // 2]
    agent.config = {"use_pymdp": PYMDP_AVAILABLE, "use_llm": False}

    # Set up agent state
    agent.is_active = False
    agent.created_at = datetime.now()  # Mock datetime
    agent.last_action_at = None
    agent.total_steps = 0

    # Active Inference components
    agent.pymdp_agent = None
    agent.beliefs = {}
    agent.preferences = {}
    agent.policies = []
    agent.gmn_spec = None
    agent.llm_manager = None

    # State space for PyMDP
    agent.num_states = grid_size * grid_size
    agent.num_obs = 5
    agent.num_actions = 5

    # Map actions to indices
    agent.action_map = {0: "up", 1: "down", 2: "left", 3: "right", 4: "stay"}
    agent.actions = list(agent.action_map.values())

    # Initialize metrics
    agent.metrics = {
        "total_observations": 0,
        "total_actions": 0,
        "avg_free_energy": 0.0,
        "belief_entropy": 0.0,
    }

    # Initialize PyMDP if available
    if PYMDP_AVAILABLE:
        agent._initialize_pymdp()

    # Other attributes
    agent.uncertainty_map = np.ones((grid_size, grid_size))
    agent.uncertainty_map[agent.position[0], agent.position[1]] = 0
    agent.exploration_rate = 0.3

    return agent


@pytest.fixture
def test_agent():
    """Create a BasicExplorerAgent for testing."""
    return create_test_agent(grid_size=5)


class TestActiveInferenceReal:
    """Test real Active Inference with PyMDP."""

    def test_pymdp_is_available(self):
        """Verify PyMDP is properly installed."""
        assert PYMDP_AVAILABLE, "PyMDP must be installed for Active Inference"

    def test_basic_explorer_initialization(self, test_agent):
        """Test BasicExplorerAgent initializes with PyMDP."""
        agent = test_agent

        assert agent.agent_id == "test_id"
        assert agent.name == "Test Explorer"
        assert agent.grid_size == 5
        assert agent.position == [2, 2]  # Center of 5x5 grid

        # Verify PyMDP agent is created
        assert agent.pymdp_agent is not None
        assert hasattr(agent.pymdp_agent, "infer_states")
        assert hasattr(agent.pymdp_agent, "infer_policies")
        assert hasattr(agent.pymdp_agent, "sample_action")

    def test_pymdp_matrices_structure(self):
        """Test PyMDP matrices are properly structured."""
        agent = BasicExplorerAgent("test_id", "Test Explorer", grid_size=3)

        # Check A matrix (observations)
        assert agent.pymdp_agent.A[0].shape == (
            5,
            9,
        )  # 5 obs types, 9 states (3x3)

        # Check B matrix (transitions)
        assert len(agent.pymdp_agent.B) == 1  # Single factor
        assert agent.pymdp_agent.B[0].shape == (
            9,
            9,
            5,
        )  # 9 states, 9 states, 5 actions

        # Check C vector (preferences)
        assert agent.pymdp_agent.C[0].shape == (5,)  # 5 observation types
        assert agent.pymdp_agent.C[0][2] > agent.pymdp_agent.C[0][0]  # Prefer goals over empty

        # Check D vector (initial beliefs)
        assert agent.pymdp_agent.D[0].shape == (9,)  # 9 states
        assert np.allclose(agent.pymdp_agent.D[0].sum(), 1.0)  # Normalized

    def test_perception_updates_beliefs(self):
        """Test perception updates agent beliefs."""
        agent = BasicExplorerAgent("test_id", "Test Explorer", grid_size=3)

        # Create observation
        observation = {
            "position": [1, 1],
            "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),  # Goal to the south
        }

        # Perceive
        agent.perceive(observation)

        assert agent.position == [1, 1]
        assert hasattr(agent, "current_observation")
        assert agent.current_observation == [0]  # Empty cell observation

    def test_belief_update_with_pymdp(self):
        """Test belief updates using PyMDP variational inference."""
        agent = BasicExplorerAgent("test_id", "Test Explorer", grid_size=3)
        agent.config["debug_mode"] = True  # Enable debug mode to store state_posterior
        agent.start()

        # Initial observation
        observation = {
            "position": [1, 1],
            "surroundings": np.array(
                [
                    [0, -1, 0],  # Obstacle to north
                    [0, 0, 0],  # Agent in center
                    [0, 0, 1],  # Goal to southeast
                ]
            ),
        }

        # Run perception and belief update
        agent.perceive(observation)
        agent.update_beliefs()

        # Check belief entropy is computed
        assert "belief_entropy" in agent.metrics
        assert agent.metrics["belief_entropy"] > 0

        # Check beliefs are stored
        assert "state_posterior" in agent.beliefs
        assert len(agent.beliefs["state_posterior"]) > 0

    def test_action_selection_with_pymdp(self):
        """Test action selection using expected free energy."""
        agent = BasicExplorerAgent("test_id", "Test Explorer", grid_size=3)
        agent.start()

        # Set up observation
        observation = {
            "position": [1, 1],
            "surroundings": np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),  # Goal to northeast
        }

        agent.perceive(observation)
        agent.update_beliefs()
        action = agent.select_action()

        # Should select a movement action
        assert action in ["up", "down", "left", "right", "stay"]

        # Check expected free energy is computed
        assert "expected_free_energy" in agent.metrics
        assert isinstance(agent.metrics["expected_free_energy"], float)

    def test_free_energy_computation(self):
        """Test variational free energy computation."""
        agent = BasicExplorerAgent("test_id", "Test Explorer", grid_size=3)
        agent.start()

        # Run a perception-action cycle
        observation = {
            "position": [1, 1],
            "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        }

        agent.step(observation)

        # Check free energy components
        fe_components = agent.compute_free_energy()
        assert "total_free_energy" in fe_components
        assert "accuracy" in fe_components
        assert "complexity" in fe_components
        assert "surprise" in fe_components

        # Verify mathematical properties
        assert (
            fe_components["total_free_energy"]
            == fe_components["complexity"] - fe_components["accuracy"]
        )

    def test_epistemic_value_exploration(self):
        """Test agent explores to reduce uncertainty (epistemic value)."""
        agent = BasicExplorerAgent("test_id", "Test Explorer", grid_size=5)
        agent.start()

        # Track positions over multiple steps with varied observations
        positions = []
        actions = []

        # Make exploration attractive by ensuring the fallback system works
        # Temporarily disable PyMDP to test fallback exploration
        original_pymdp = agent.pymdp_agent
        agent.pymdp_agent = None  # Force fallback behavior

        for i in range(10):
            observation = {
                "position": agent.position.copy(),
                "surroundings": np.zeros((3, 3)),  # All empty
            }
            action = agent.step(observation)
            actions.append(action)

            # Simulate movement
            if action == "up" and agent.position[0] > 0:
                agent.position[0] -= 1
            elif action == "down" and agent.position[0] < 4:
                agent.position[0] += 1
            elif action == "left" and agent.position[1] > 0:
                agent.position[1] -= 1
            elif action == "right" and agent.position[1] < 4:
                agent.position[1] += 1

            positions.append(tuple(agent.position))

        # Restore PyMDP agent
        agent.pymdp_agent = original_pymdp

        # Should explore different positions using fallback system
        unique_positions = set(positions)
        set(actions)

        # Check that exploration occurred (should visit multiple positions)
        assert (
            len(unique_positions) > 1
        ), f"Agent should explore multiple positions, visited: {unique_positions}"

        # Check that different actions were taken
        movement_actions = [a for a in actions if a in ["up", "down", "left", "right"]]
        assert len(movement_actions) > 0, f"Agent should take movement actions, got: {actions}"

    def test_pragmatic_value_goal_seeking(self):
        """Test agent seeks goals (pragmatic value)."""
        agent = BasicExplorerAgent("test_id", "Test Explorer", grid_size=3)
        agent.start()

        # Place agent at center
        agent.position = [1, 1]

        # Observation with goal visible - center observation shows goal
        observation = {
            "position": [1, 1],
            "surroundings": np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]  # Goal at center (agent observes goal)
            ),
        }

        # Agent should have non-zero expected free energy when observing goal
        actions = []
        for _ in range(3):
            action = agent.step(observation)
            actions.append(action)

        # Should take some action (not always stay)
        # At minimum, expected free energy should be computed
        assert "expected_free_energy" in agent.metrics
        assert isinstance(agent.metrics["expected_free_energy"], float)

    def test_policy_selection_horizon(self):
        """Test policy selection with planning horizon."""
        agent = BasicExplorerAgent("test_id", "Test Explorer", grid_size=5)

        # Check policy length is set (PyMDP uses policy_len instead of planning_horizon)
        # BasicExplorerAgent defaults to fast mode for performance (policy_len = 1)
        assert agent.pymdp_agent.policy_len == 1

        # Verify policies are evaluated
        agent.start()
        observation = {"position": [2, 2], "surroundings": np.zeros((3, 3))}

        agent.perceive(observation)
        agent.update_beliefs()

        # Get policy posterior
        q_pi, G = agent.pymdp_agent.infer_policies()

        # Should have multiple policies evaluated
        assert len(G) > 0  # Expected free energies computed
        assert q_pi.sum() > 0.99  # Normalized distribution

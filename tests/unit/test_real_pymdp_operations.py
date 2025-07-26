"""Reality checkpoint test: Run belief update and inference operations with real data.

Task 1.4: Reality checkpoint - Run operations with real data to ensure they work.
"""

import numpy as np
import pytest
from pymdp import utils
from pymdp.agent import Agent as PyMDPAgent

from agents.pymdp_adapter import PyMDPCompatibilityAdapter
from agents.resource_collector import ResourceCollectorAgent


class TestRealPyMDPOperations:
    """Test PyMDP operations with real data to ensure bulletproof return handling."""

    def test_full_agent_cycle_with_real_data(self):
        """Test complete agent cycle: creation, belief update, action selection."""
        # Create a real active inference agent using ResourceCollectorAgent
        agent = ResourceCollectorAgent(agent_id="test_agent", name="TestAgent", grid_size=10)

        # Verify agent has PyMDP initialized
        assert agent.pymdp_agent is not None, "PyMDP agent should be initialized"
        assert hasattr(agent.pymdp_agent, "A"), "PyMDP agent should have A matrix"
        assert hasattr(agent.pymdp_agent, "B"), "PyMDP agent should have B matrix"
        assert hasattr(agent.pymdp_agent, "C"), "PyMDP agent should have C matrix"
        assert hasattr(agent.pymdp_agent, "D"), "PyMDP agent should have D matrix"

        # Activate the agent
        agent.is_active = True

        # Update beliefs with real observation using perceive
        observation = {
            "type": "empty",
            "position": agent.position,
            "visible_cells": [],
        }
        agent.perceive(observation)

        # Select action - this tests the full pipeline
        action = agent.select_action()

        # Verify action is a valid string
        assert isinstance(action, str), f"Expected string action, got {type(action)}"
        assert action in [
            "up",
            "down",
            "left",
            "right",
            "collect",
            "return_to_base",
        ], f"Invalid action: {action}"

        # Verify that PyMDP agent has G (expected free energy) after inference
        assert hasattr(agent.pymdp_agent, "G"), "PyMDP agent should have G after inference"
        assert agent.pymdp_agent.G is not None, "G should not be None after inference"

        # Verify we can access G directly without defensive checks
        # This would have failed with the old defensive code
        min_free_energy = float(np.min(agent.pymdp_agent.G))
        assert isinstance(min_free_energy, float), "Should be able to compute min free energy"

    def test_pymdp_return_value_consistency(self):
        """Test that PyMDP operations consistently return expected types."""
        # Create minimal PyMDP agent
        num_obs = [4]
        num_states = [4]
        num_actions = 5

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_actions)
        C = np.zeros(num_obs[0])
        D = np.ones(num_states[0]) / num_states[0]

        agent = PyMDPAgent(A, B, C, D)
        adapter = PyMDPCompatibilityAdapter()

        # Test multiple cycles to ensure consistency
        for i in range(10):
            # Update beliefs
            obs = [i % 4]  # Cycle through observations
            beliefs_result = agent.infer_states(obs)

            # Verify belief format
            assert isinstance(beliefs_result, np.ndarray), f"Cycle {i}: Expected ndarray"
            # PyMDP returns object array containing belief arrays directly
            if beliefs_result.dtype == np.object_:
                # For single-factor agents, beliefs_result contains the belief array
                if beliefs_result.shape == (1,):
                    belief = beliefs_result[0]
                    assert isinstance(
                        belief, np.ndarray
                    ), f"Cycle {i}: Expected belief to be ndarray"
            else:
                # Direct array return
                belief = beliefs_result

            # Infer policies
            q_pi, G = agent.infer_policies()

            # Verify policy format
            assert isinstance(q_pi, np.ndarray), f"Cycle {i}: q_pi should be ndarray"
            assert isinstance(G, np.ndarray), f"Cycle {i}: G should be ndarray"
            assert G.size > 0, f"Cycle {i}: G should not be empty"

            # Sample action
            action_idx = adapter.sample_action(agent)

            # Verify action format
            assert isinstance(action_idx, int), f"Cycle {i}: Expected int, got {type(action_idx)}"
            assert 0 <= action_idx < num_actions, f"Cycle {i}: Invalid action {action_idx}"

    def test_edge_cases_in_return_values(self):
        """Test edge cases in PyMDP return value handling."""
        # Create agent with extreme parameters
        num_obs = [2]  # Minimal observations
        num_states = [10]  # More states than observations
        num_actions = 2  # Binary actions

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_actions)
        C = np.array([1.0, -1.0])  # Strong preferences
        D = np.array([0.9] + [0.1 / (num_states[0] - 1)] * (num_states[0] - 1))  # Peaked prior

        agent = PyMDPAgent(A, B, C, D)
        adapter = PyMDPCompatibilityAdapter()

        # Test with edge case observations
        test_cases = [
            [0],  # First observation
            [1],  # Last observation
        ]

        for obs in test_cases:
            # Update beliefs
            agent.infer_states(obs)

            # Infer policies
            q_pi, G = agent.infer_policies()

            # Sample action
            action_idx = adapter.sample_action(agent)

            # All operations should succeed without fallbacks
            assert isinstance(action_idx, int), f"For obs {obs}: Expected int"
            assert action_idx in [
                0,
                1,
            ], f"For obs {obs}: Invalid binary action {action_idx}"

    def test_no_graceful_degradation(self):
        """Test that invalid operations fail hard, not gracefully."""
        adapter = PyMDPCompatibilityAdapter()

        # Test with non-PyMDP object
        with pytest.raises(TypeError, match="Expected PyMDPAgent"):
            adapter.sample_action("not_an_agent")

        # Test with None
        with pytest.raises(TypeError, match="Expected PyMDPAgent"):
            adapter.sample_action(None)

        # Create a mock object that looks like PyMDP but returns wrong type
        class FakePyMDP:
            def sample_action(self):
                return "wrong_type"  # String instead of array

        fake_agent = FakePyMDP()
        with pytest.raises((TypeError, RuntimeError), match="Expected numpy.ndarray|returned"):
            adapter.sample_action(fake_agent)

    def test_direct_attribute_access(self):
        """Test that we access attributes directly without getattr fallbacks."""
        agent = ResourceCollectorAgent(agent_id="test_direct", name="DirectTest")

        # Direct attribute access should work
        assert agent.agent_id == "test_direct"
        assert agent.name == "DirectTest"
        assert agent.position == [
            5,
            5,
        ]  # ResourceCollectorAgent sets position to grid_size//2

        # PyMDP attributes should exist
        assert agent.pymdp_agent.A is not None
        assert agent.pymdp_agent.B is not None
        assert agent.pymdp_agent.C is not None
        assert agent.pymdp_agent.D is not None

        # Accessing non-existent attribute should raise AttributeError
        with pytest.raises(AttributeError):
            _ = agent.non_existent_attribute


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

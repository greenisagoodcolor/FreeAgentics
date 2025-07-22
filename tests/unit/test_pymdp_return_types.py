"""Test PyMDP return types to verify API behavior.

Task 1.4: Write tests verifying expected return types from PyMDP operations
"""

import numpy as np
import pytest
from pymdp import utils
from pymdp.agent import Agent as PyMDPAgent

from agents.pymdp_adapter import PyMDPCompatibilityAdapter


class TestPyMDPReturnTypes:
    """Test actual PyMDP return types for critical operations."""

    def test_sample_action_return_type(self):
        """Test that sample_action returns the expected type."""
        # Create minimal PyMDP agent
        num_obs = [4]
        num_states = [4]
        num_actions = 5

        # Create properly shaped matrices for single modality
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_actions)

        # For single modality, C should be a 1D array
        C = np.zeros(num_obs[0])

        # D should be a normalized distribution over initial states
        D = np.ones(num_states[0]) / num_states[0]  # Uniform distribution

        agent = PyMDPAgent(A, B, C, D)

        # Initialize agent state
        obs = [0]
        agent.infer_states(obs)
        agent.infer_policies()

        # Test sample_action return type
        action_result = agent.sample_action()

        # Verify it's a numpy array with shape (1,)
        assert isinstance(
            action_result, np.ndarray
        ), f"Expected numpy.ndarray, got {type(action_result)}"
        assert action_result.shape == (1,), f"Expected shape (1,), got {action_result.shape}"
        assert action_result.dtype in [
            np.float64,
            np.float32,
            np.int64,
            np.int32,
        ], f"Unexpected dtype: {action_result.dtype}"

        # Verify the value is a valid action index
        action_idx = int(action_result.item())
        assert 0 <= action_idx < num_actions, f"Invalid action index: {action_idx}"

    def test_infer_policies_return_type(self):
        """Test that infer_policies returns the expected types."""
        # Create minimal PyMDP agent
        num_obs = [4]
        num_states = [4]
        num_actions = 5

        # Create properly shaped matrices for single modality
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_actions)

        # For single modality, C should be a 1D array
        C = np.zeros(num_obs[0])

        # D should be a normalized distribution over initial states
        D = np.ones(num_states[0]) / num_states[0]  # Uniform distribution

        agent = PyMDPAgent(A, B, C, D)

        # Initialize agent state
        obs = [0]
        agent.infer_states(obs)

        # Test infer_policies return type
        result = agent.infer_policies()

        # Verify it's a tuple of length 2
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"

        q_pi, G = result

        # Verify q_pi is numpy array
        assert isinstance(q_pi, np.ndarray), f"Expected q_pi to be numpy.ndarray, got {type(q_pi)}"
        assert np.issubdtype(
            q_pi.dtype, np.floating
        ), f"Expected floating dtype for q_pi, got {q_pi.dtype}"

        # Verify G is numpy array
        assert isinstance(G, np.ndarray), f"Expected G to be numpy.ndarray, got {type(G)}"
        assert np.issubdtype(G.dtype, np.floating), f"Expected floating dtype for G, got {G.dtype}"

    def test_infer_states_return_type(self):
        """Test that infer_states returns the expected types."""
        # Create minimal PyMDP agent
        num_obs = [4]
        num_states = [4]
        num_actions = 5

        # Create properly shaped matrices for single modality
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_actions)

        # For single modality, C should be a 1D array
        C = np.zeros(num_obs[0])

        # D should be a normalized distribution over initial states
        D = np.ones(num_states[0]) / num_states[0]  # Uniform distribution

        agent = PyMDPAgent(A, B, C, D)

        # Test infer_states return type
        obs = [0]
        result = agent.infer_states(obs)

        # PyMDP returns numpy.ndarray with dtype=object containing list of belief arrays
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result)}"
        assert result.dtype == np.object_, f"Expected dtype object, got {result.dtype}"
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"

        # Extract the actual list of beliefs
        beliefs_list = result.item()
        assert isinstance(
            beliefs_list, list
        ), f"Expected list in object array, got {type(beliefs_list)}"

        # Verify each belief array
        for i, belief in enumerate(beliefs_list):
            assert isinstance(
                belief, np.ndarray
            ), f"Belief {i} expected numpy.ndarray, got {type(belief)}"
            assert np.issubdtype(
                belief.dtype, np.floating
            ), f"Belief {i} expected floating dtype, got {belief.dtype}"
            # Beliefs should sum to 1 (probability distribution)
            assert np.isclose(np.sum(belief), 1.0), f"Belief {i} doesn't sum to 1: {np.sum(belief)}"

    def test_adapter_sample_action_conversion(self):
        """Test that the adapter correctly converts sample_action return value."""
        # Create minimal PyMDP agent
        num_obs = [4]
        num_states = [4]
        num_actions = 5

        # Create properly shaped matrices for single modality
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_actions)

        # For single modality, C should be a 1D array
        C = np.zeros(num_obs[0])

        # D should be a normalized distribution over initial states
        D = np.ones(num_states[0]) / num_states[0]  # Uniform distribution

        agent = PyMDPAgent(A, B, C, D)

        # Initialize agent state
        obs = [0]
        agent.infer_states(obs)
        agent.infer_policies()

        # Test adapter conversion
        adapter = PyMDPCompatibilityAdapter()
        action_idx = adapter.sample_action(agent)

        # Verify it's exactly int type
        assert isinstance(action_idx, int), f"Expected int, got {type(action_idx)}"
        assert not isinstance(action_idx, np.integer), "Should be Python int, not numpy integer"
        assert 0 <= action_idx < num_actions, f"Invalid action index: {action_idx}"

    def test_no_tuple_unpacking_needed(self):
        """Test that sample_action doesn't return a tuple that needs unpacking."""
        # Create minimal PyMDP agent
        num_obs = [4]
        num_states = [4]
        num_actions = 5

        # Create properly shaped matrices for single modality
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_actions)

        # For single modality, C should be a 1D array
        C = np.zeros(num_obs[0])

        # D should be a normalized distribution over initial states
        D = np.ones(num_states[0]) / num_states[0]  # Uniform distribution

        agent = PyMDPAgent(A, B, C, D)

        # Initialize agent state
        obs = [0]
        agent.infer_states(obs)
        agent.infer_policies()

        # Test that we don't need tuple unpacking
        action_result = agent.sample_action()

        # This should NOT be a tuple
        assert not isinstance(
            action_result, tuple
        ), f"sample_action should not return tuple, got {type(action_result)}"

        # Direct conversion should work
        action_idx = int(action_result.item())
        assert isinstance(action_idx, int), f"Expected int after conversion, got {type(action_idx)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

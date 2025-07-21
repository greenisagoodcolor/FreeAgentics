"""
PyMDP Active Inference Functionality Validation Tests.

These tests validate that PyMDP Active Inference operations work correctly
with real data and without any graceful fallbacks or mock implementations.
Tests MUST fail if PyMDP is not properly installed or functional.
"""

import numpy as np

# Import PyMDP - NO graceful fallbacks, MUST fail if unavailable
import pymdp
import pytest
from pymdp import utils
from pymdp.agent import Agent as PyMDPAgent


class TestPyMDPActiveInferenceFunctionality:
    """Test suite to validate PyMDP Active Inference functionality."""

    def setup_method(self):
        """Set up test environment with real PyMDP components."""
        # Create simple 2-state, 2-observation model
        self.num_states = [3, 3]  # Two factors with 3 states each
        self.num_observations = [3, 3]  # Two observation modalities
        self.num_controls = [4, 4]  # Two control factors

        # Generate generative model matrices
        self.A = utils.random_A_matrix(self.num_observations, self.num_states)
        self.B = utils.random_B_matrix(self.num_states, self.num_controls)
        self.C = utils.obj_array_uniform(self.num_observations)

        # Create agent
        self.agent = PyMDPAgent(A=self.A, B=self.B, C=self.C, policy_len=3, inference_horizon=1)

    def test_pymdp_belief_state_updates_with_real_data(self):
        """Test belief state updates work with real observations."""
        # Generate real observation
        obs = [1, 2]  # Specific observation values

        # Initial belief state
        initial_beliefs = [qs.copy() for qs in self.agent.qs]

        # Perform belief update
        self.agent.infer_states(obs)

        # Verify belief state has changed
        for i, (initial, updated) in enumerate(zip(initial_beliefs, self.agent.qs)):
            assert not np.allclose(initial, updated), f"Belief state {i} did not update"

        # Verify beliefs are valid probability distributions
        for i, qs in enumerate(self.agent.qs):
            assert np.allclose(np.sum(qs), 1.0), f"Belief state {i} is not normalized"
            assert np.all(qs >= 0), f"Belief state {i} has negative probabilities"

    def test_pymdp_policy_computation_executes_properly(self):
        """Test policy computation and Expected Free Energy calculations."""
        # Set observation
        obs = [0, 1]
        self.agent.infer_states(obs)

        # Perform policy inference
        self.agent.infer_policies()

        # Verify policies were computed
        assert hasattr(self.agent, "G"), "Expected Free Energy (G) not computed"
        assert self.agent.G is not None, "G values are None"
        assert len(self.agent.G) > 0, "No policies evaluated"

        # Verify G values are finite
        assert np.all(np.isfinite(self.agent.G)), "G values contain NaN or inf"

        # Verify policy probabilities
        assert hasattr(self.agent, "q_pi"), "Policy probabilities not computed"
        assert np.allclose(np.sum(self.agent.q_pi), 1.0), "Policy probabilities not normalized"
        assert np.all(self.agent.q_pi >= 0), "Policy probabilities are negative"

    def test_pymdp_action_selection_operates_correctly(self):
        """Test action selection produces valid actions."""
        # Set observation and infer states
        obs = [1, 0]
        self.agent.infer_states(obs)

        # Infer policies
        self.agent.infer_policies()

        # Sample action
        action = self.agent.sample_action()

        # Verify action is valid
        assert isinstance(action, (list, np.ndarray)), "Action is not list or array"
        assert len(action) == len(self.num_controls), "Action dimensionality incorrect"

        # Verify action values are within valid range
        for i, (act, num_ctrl) in enumerate(zip(action, self.num_controls)):
            assert 0 <= act < num_ctrl, f"Action {i} value {act} out of range [0, {num_ctrl})"

    def test_pymdp_full_inference_cycle(self):
        """Test complete inference cycle: observe -> infer states -> infer policies -> act."""
        observations_sequence = [[0, 0], [1, 1], [2, 0], [0, 2]]

        actions_taken = []
        beliefs_history = []

        for obs in observations_sequence:
            # Belief update
            self.agent.infer_states(obs)

            # Store beliefs
            beliefs_history.append([qs.copy() for qs in self.agent.qs])

            # Policy inference
            self.agent.infer_policies()

            # Action selection
            action = self.agent.sample_action()
            actions_taken.append(action)

        # Verify we got valid actions for all observations
        assert len(actions_taken) == len(observations_sequence), "Missing actions"

        # Verify beliefs evolved over time
        for t in range(1, len(beliefs_history)):
            for i in range(len(beliefs_history[t])):
                # At least some beliefs should change over time
                if not np.allclose(beliefs_history[t - 1][i], beliefs_history[t][i], atol=1e-6):
                    break
            else:
                # If we get here, no beliefs changed at all between timesteps
                pytest.fail(f"Beliefs did not evolve between timesteps {t - 1} and {t}")

    def test_pymdp_with_different_model_configurations(self):
        """Test PyMDP works with various model configurations."""
        test_configs = [
            {"num_states": [2], "num_observations": [2], "num_controls": [2]},
            {
                "num_states": [4, 2],
                "num_observations": [3, 4],
                "num_controls": [3, 2],
            },
            {
                "num_states": [5, 3, 2],
                "num_observations": [4, 3, 3],
                "num_controls": [3, 2, 2],
            },
        ]

        for config in test_configs:
            # Create generative model
            A = utils.random_A_matrix(config["num_observations"], config["num_states"])
            B = utils.random_B_matrix(config["num_states"], config["num_controls"])
            C = utils.obj_array_uniform(config["num_observations"])

            # Create agent
            agent = PyMDPAgent(A=A, B=B, C=C)

            # Test inference cycle
            obs = [np.random.randint(0, num_obs) for num_obs in config["num_observations"]]

            agent.infer_states(obs)
            agent.infer_policies()
            action = agent.sample_action()

            # Verify valid action
            assert len(action) == len(config["num_controls"])
            for act, num_ctrl in zip(action, config["num_controls"]):
                assert 0 <= act < num_ctrl

    def test_pymdp_error_handling_with_invalid_inputs(self):
        """Test PyMDP fails gracefully with invalid inputs."""
        # Test with wrong observation dimensionality
        with pytest.raises((ValueError, IndexError, TypeError)):
            self.agent.infer_states([0])  # Too few observations

        with pytest.raises((ValueError, IndexError, TypeError)):
            self.agent.infer_states([0, 1, 2])  # Too many observations

        # Test with invalid observation values
        with pytest.raises((ValueError, IndexError)):
            self.agent.infer_states([10, 10])  # Out of range observations

    def test_pymdp_deterministic_behavior(self):
        """Test PyMDP produces consistent results with same inputs."""
        obs = [1, 1]

        # Set random seed for reproducibility
        np.random.seed(42)

        # Run inference twice with same setup
        agent1 = PyMDPAgent(A=self.A, B=self.B, C=self.C)
        agent1.infer_states(obs)
        beliefs1 = [qs.copy() for qs in agent1.qs]

        np.random.seed(42)
        agent2 = PyMDPAgent(A=self.A, B=self.B, C=self.C)
        agent2.infer_states(obs)
        beliefs2 = [qs.copy() for qs in agent2.qs]

        # Verify same beliefs (within numerical precision)
        for b1, b2 in zip(beliefs1, beliefs2):
            assert np.allclose(b1, b2, atol=1e-10), "PyMDP inference not deterministic"

    def test_pymdp_memory_and_performance_characteristics(self):
        """Test PyMDP memory usage and performance characteristics."""
        import os
        import time

        import psutil

        process = psutil.Process(os.getpid())

        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create larger model and run inference
        large_states = [10, 8]
        large_obs = [8, 10]
        large_controls = [5, 4]

        A_large = utils.random_A_matrix(large_obs, large_states)
        B_large = utils.random_B_matrix(large_states, large_controls)
        C_large = utils.obj_array_uniform(large_obs)

        agent_large = PyMDPAgent(A=A_large, B=B_large, C=C_large, policy_len=5)

        # Time multiple inference cycles
        start_time = time.time()
        for _ in range(10):
            obs = [np.random.randint(0, num_obs) for num_obs in large_obs]
            agent_large.infer_states(obs)
            agent_large.infer_policies()
            agent_large.sample_action()
        end_time = time.time()

        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        # Verify reasonable performance characteristics
        total_time = end_time - start_time
        avg_time_per_cycle = total_time / 10
        memory_used = memory_after - memory_before

        # These are sanity checks, not hard requirements
        assert avg_time_per_cycle < 5.0, f"Inference too slow: {avg_time_per_cycle:.3f}s per cycle"
        assert memory_used < 100, f"Excessive memory usage: {memory_used:.1f}MB"

        print(f"Performance: {avg_time_per_cycle:.3f}s per cycle, {memory_used:.1f}MB memory")


def test_pymdp_import_and_basic_functionality():
    """Test PyMDP can be imported and basic functionality works."""
    # Test basic PyMDP utilities
    assert hasattr(pymdp, "utils"), "PyMDP utils module not available"
    assert hasattr(utils, "random_A_matrix"), "random_A_matrix function not available"
    assert hasattr(utils, "random_B_matrix"), "random_B_matrix function not available"
    assert hasattr(utils, "obj_array_uniform"), "obj_array_uniform function not available"

    # Test basic matrix generation
    A = utils.random_A_matrix([2, 2], [3, 3])
    assert len(A) == 2, "A matrix wrong structure"

    B = utils.random_B_matrix([3, 3], [2, 2])
    assert len(B) == 2, "B matrix wrong structure"

    # Test agent creation
    agent = PyMDPAgent(A, B)
    assert hasattr(agent, "infer_states"), "Agent missing infer_states method"
    assert hasattr(agent, "infer_policies"), "Agent missing infer_policies method"
    assert hasattr(agent, "sample_action"), "Agent missing sample_action method"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

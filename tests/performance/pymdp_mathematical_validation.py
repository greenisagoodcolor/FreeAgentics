"""
PyMDP Mathematical Correctness Validation Suite
Validates that PyMDP computations are mathematically correct and performance tests are realistic.
"""

import time
from typing import Dict, Tuple

import numpy as np

# PyMDP required for mathematical validation - hard failure, no graceful degradation
import pymdp
import pytest
from pymdp import utils

PYMDP_AVAILABLE = True


class PyMDPMathematicalValidator:
    """Validator for PyMDP mathematical correctness."""

    def __init__(self):
        self.tolerance = 1e-10

    def create_simple_agent(self) -> Tuple[object, np.ndarray, np.ndarray]:
        """Create a simple agent with known matrices for validation."""
        if not PYMDP_AVAILABLE:
            raise ImportError("PyMDP required for validation")

        # Simple 2-state, 2-observation model
        num_obs = [2]
        num_states = [2]
        num_controls = [2]

        # Create A matrix (observation model) - Identity-like
        A = utils.obj_array_zeros([(obs, *num_states) for obs in num_obs])
        A[0][0, 0] = 0.9  # High probability of observing 0 when in state 0
        A[0][1, 0] = 0.1  # Low probability of observing 1 when in state 0
        A[0][0, 1] = 0.1  # Low probability of observing 0 when in state 1
        A[0][1, 1] = 0.9  # High probability of observing 1 when in state 1

        # Create B matrix (transition model)
        B = utils.obj_array_zeros([(s, s, num_controls[i]) for i, s in enumerate(num_states)])
        # Action 0: Stay in same state (deterministic)
        B[0][0, 0, 0] = 1.0
        B[0][1, 0, 0] = 0.0
        B[0][0, 1, 0] = 0.0
        B[0][1, 1, 0] = 1.0
        # Action 1: Switch states (deterministic)
        B[0][0, 0, 1] = 0.0
        B[0][1, 0, 1] = 1.0
        B[0][0, 1, 1] = 1.0
        B[0][1, 1, 1] = 0.0

        agent = pymdp.agent.Agent(A=A, B=B, policy_len=1)

        return agent, A, B

    def test_probability_matrix_normalization(self):
        """Test that probability matrices are properly normalized."""
        agent, A, B = self.create_simple_agent()

        # Check A matrix normalization (should sum to 1 across observations)
        for state_combo in range(A[0].shape[1]):
            obs_probs = A[0][:, state_combo]
            assert abs(obs_probs.sum() - 1.0) < self.tolerance, (
                f"A matrix not normalized: {obs_probs.sum()}"
            )

        # Check B matrix normalization (should sum to 1 across next states)
        for state in range(B[0].shape[1]):
            for action in range(B[0].shape[2]):
                trans_probs = B[0][:, state, action]
                assert abs(trans_probs.sum() - 1.0) < self.tolerance, (
                    f"B matrix not normalized: {trans_probs.sum()}"
                )

        print("✅ Probability matrices are correctly normalized")

    def test_belief_state_normalization(self):
        """Test that belief states are properly normalized."""
        agent, A, B = self.create_simple_agent()

        # Test multiple observations
        observations = [[0], [1], [0], [1]]

        for obs in observations:
            qs = agent.infer_states(obs)

            # Check belief state normalization
            for factor_idx, belief in enumerate(qs):
                belief_sum = belief.sum()
                assert abs(belief_sum - 1.0) < self.tolerance, (
                    f"Belief state not normalized: {belief_sum}"
                )

                # Check no negative probabilities
                assert (belief >= 0).all(), "Negative probabilities in belief state"

        print("✅ Belief states are correctly normalized")

    def test_bayesian_inference_correctness(self):
        """Test that Bayesian inference produces correct results."""
        agent, A, B = self.create_simple_agent()

        # Test with deterministic observation
        # If we observe 0, we should have higher belief in state 0
        obs = [0]
        qs = agent.infer_states(obs)
        belief_state = qs[0]

        # Given our A matrix, observing 0 should make state 0 more likely
        assert belief_state[0] > belief_state[1], (
            f"Incorrect inference: belief in state 0 ({belief_state[0]}) should be > state 1 ({belief_state[1]})"
        )

        # Test with other observation
        obs = [1]
        qs = agent.infer_states(obs)
        belief_state = qs[0]

        # Observing 1 should make state 1 more likely
        assert belief_state[1] > belief_state[0], (
            f"Incorrect inference: belief in state 1 ({belief_state[1]}) should be > state 0 ({belief_state[0]})"
        )

        print("✅ Bayesian inference produces correct results")

    def test_policy_evaluation_correctness(self):
        """Test that policy evaluation produces sensible results."""
        agent, A, B = self.create_simple_agent()

        # First, establish a belief state
        obs = [0]
        agent.infer_states(obs)

        # Evaluate policies
        q_pi, G = agent.infer_policies()

        # Check that policy probabilities are normalized
        assert abs(q_pi.sum() - 1.0) < self.tolerance, (
            f"Policy probabilities not normalized: {q_pi.sum()}"
        )

        # Check no negative probabilities
        assert (q_pi >= 0).all(), "Negative policy probabilities"

        # Check that G (expected free energy) has reasonable values
        assert np.isfinite(G).all(), "Non-finite values in expected free energy"

        print("✅ Policy evaluation produces correct results")

    def test_action_sampling_correctness(self):
        """Test that action sampling produces valid actions."""
        agent, A, B = self.create_simple_agent()

        # Establish belief and policies
        obs = [0]
        agent.infer_states(obs)
        q_pi, G = agent.infer_policies()

        # Sample multiple actions
        actions_sampled = []
        for _ in range(100):
            action = agent.sample_action()
            actions_sampled.append(action)

            # Check action is valid
            assert isinstance(action, (list, np.ndarray)), "Action not in correct format"
            if isinstance(action, list):
                assert len(action) == 1, "Incorrect action dimensionality"
                assert action[0] in [
                    0,
                    1,
                ], f"Invalid action value: {action[0]}"

        print("✅ Action sampling produces valid actions")

    def test_performance_benchmark_realism(self):
        """Test that performance benchmarks measure realistic operations."""
        start_time = time.perf_counter()

        # Create agent (this should take non-zero time)
        agent, A, B = self.create_simple_agent()
        creation_time = time.perf_counter() - start_time

        assert creation_time > 0, "Agent creation should take measurable time"

        # Test inference timing
        start_time = time.perf_counter()
        obs = [0]
        agent.infer_states(obs)
        q_pi, G = agent.infer_policies()
        agent.sample_action()
        inference_time = time.perf_counter() - start_time

        assert inference_time > 0, "Inference should take measurable time"

        # Multiple inferences should take more time than single inference
        start_time = time.perf_counter()
        for i in range(10):
            obs = [i % 2]
            agent.infer_states(obs)
            q_pi, G = agent.infer_policies()
            agent.sample_action()
        multi_inference_time = time.perf_counter() - start_time

        assert multi_inference_time > inference_time, (
            "Multiple inferences should take more time than single inference"
        )

        print("✅ Performance benchmarks measure realistic operations")

    def test_mathematical_consistency(self):
        """Test mathematical consistency across multiple runs."""
        agent, A, B = self.create_simple_agent()

        # Test that same observation produces same result (deterministic inference)
        obs = [0]

        results = []
        for _ in range(5):
            # Reset agent to same state
            agent, A, B = self.create_simple_agent()
            qs = agent.infer_states(obs)
            results.append(qs[0].copy())

        # Check consistency across runs
        for i in range(1, len(results)):
            diff = np.abs(results[0] - results[i])
            assert (diff < self.tolerance).all(), (
                f"Inconsistent results across runs: max diff = {diff.max()}"
            )

        print("✅ Mathematical operations are consistent")

    def run_full_validation(self) -> Dict[str, bool]:
        """Run the complete mathematical validation suite."""
        results = {}

        tests = [
            "test_probability_matrix_normalization",
            "test_belief_state_normalization",
            "test_bayesian_inference_correctness",
            "test_policy_evaluation_correctness",
            "test_action_sampling_correctness",
            "test_performance_benchmark_realism",
            "test_mathematical_consistency",
        ]

        for test_name in tests:
            try:
                test_method = getattr(self, test_name)
                test_method()
                results[test_name] = True
            except Exception as e:
                print(f"❌ {test_name} failed: {e}")
                results[test_name] = False

        return results


def test_pymdp_mathematical_validation():
    """Run PyMDP mathematical validation tests."""
    if not PYMDP_AVAILABLE:
        pytest.fail(
            "PyMDP not available - test cannot proceed without PyMDP mathematical validation"
        )

    validator = PyMDPMathematicalValidator()
    results = validator.run_full_validation()

    # Check that all tests passed
    failed_tests = [test for test, passed in results.items() if not passed]
    assert len(failed_tests) == 0, f"Failed tests: {failed_tests}"

    print(f"✅ All {len(results)} mathematical validation tests passed")


if __name__ == "__main__":
    print("Running PyMDP Mathematical Validation Suite")
    print(f"PyMDP Available: {PYMDP_AVAILABLE}")

    if PYMDP_AVAILABLE:
        validator = PyMDPMathematicalValidator()
        results = validator.run_full_validation()

        print("\nValidation Results:")
        passed = sum(results.values())
        total = len(results)
        print(f"Passed: {passed}/{total}")

        if passed == total:
            print("🎉 ALL MATHEMATICAL VALIDATION TESTS PASSED")
        else:
            failed = [test for test, result in results.items() if not result]
            print(f"❌ Failed tests: {failed}")
    else:
        print("❌ PyMDP not available - cannot run validation")

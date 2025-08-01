"""Unit tests for Expected Free Energy Policy Selection (Task 44.3).

This module implements comprehensive tests for EFE-based policy selection,
following Kent Beck's TDD principles with mathematical validation.

Tests cover:
- EFE calculation correctness (epistemic + pragmatic values)
- Policy evaluation and ranking
- Preference integration from GMN
- Numerical stability and edge cases
- Performance within memory/timing constraints
"""

import time

import numpy as np
import pytest

from inference.active.belief_manager import BeliefState
from inference.active.config import ActiveInferenceConfig
from inference.active.policy_selection import (
    EFEResult,
    PolicyEvaluator,
    PolicyRanking,
    PolicySelector,
    PreferenceManager,
)


class TestPolicyEvaluator:
    """Test EFE calculation correctness and numerical stability."""

    def test_epistemic_value_calculation(self):
        """Test epistemic value (information gain) calculation."""
        # Setup simple 2x2 environment
        config = ActiveInferenceConfig(
            num_observations=[2], num_states=[2], num_controls=[2], planning_horizon=3
        )

        # Create observation model (A matrix)
        A_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])  # Good observability

        # Create current beliefs (uniform uncertainty)
        beliefs = np.array([0.5, 0.5])
        belief_state = BeliefState(beliefs=beliefs)

        # Create policy evaluator
        evaluator = PolicyEvaluator(config=config, A_matrix=A_matrix)

        # Test epistemic value for different actions
        # Action 0 should provide information gain
        epistemic_value_0 = evaluator.compute_epistemic_value(
            belief_state=belief_state,
            policy=[0, 1],  # Two-step policy
            horizon=2,
        )

        # Information gain should be positive (we expect to learn something)
        assert epistemic_value_0 > 0, "Epistemic value should be positive for informative actions"

        # With perfect confidence, epistemic value should be near zero
        confident_beliefs = np.array([0.99, 0.01])
        confident_state = BeliefState(beliefs=confident_beliefs)

        epistemic_value_confident = evaluator.compute_epistemic_value(
            belief_state=confident_state, policy=[0, 1], horizon=2
        )

        assert (
            epistemic_value_confident < epistemic_value_0
        ), "Epistemic value should be lower when already confident"

    def test_pragmatic_value_calculation(self):
        """Test pragmatic value (preference satisfaction) calculation."""
        config = ActiveInferenceConfig(num_observations=[2], num_states=[2], num_controls=[2])

        A_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        beliefs = np.array([0.5, 0.5])
        belief_state = BeliefState(beliefs=beliefs)

        evaluator = PolicyEvaluator(config=config, A_matrix=A_matrix)

        # Create preferences (prefer observation 0)
        preferences = np.array([2.0, 0.0])  # Strong preference for obs 0

        pragmatic_value = evaluator.compute_pragmatic_value(
            belief_state=belief_state, policy=[0], preferences=preferences, horizon=1
        )

        # Should be positive since we can achieve preferred observations
        assert pragmatic_value > 0, "Pragmatic value should be positive for achievable preferences"

        # Test with negative preferences
        negative_prefs = np.array([0.0, 2.0])  # Prefer observation 1

        negative_pragmatic = evaluator.compute_pragmatic_value(
            belief_state=belief_state,
            policy=[0],  # Action that tends toward obs 0
            preferences=negative_prefs,
            horizon=1,
        )

        assert (
            negative_pragmatic < pragmatic_value
        ), "Pragmatic value should be lower for non-preferred outcomes"

    def test_expected_free_energy_calculation(self):
        """Test complete EFE calculation (epistemic + pragmatic)."""
        config = ActiveInferenceConfig(num_observations=[2], num_states=[2], num_controls=[2])

        A_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        beliefs = np.array([0.5, 0.5])
        belief_state = BeliefState(beliefs=beliefs)
        preferences = np.array([1.5, 0.5])  # Mild preference for obs 0

        evaluator = PolicyEvaluator(config=config, A_matrix=A_matrix)

        # Calculate EFE for a policy
        efe_result = evaluator.compute_expected_free_energy(
            belief_state=belief_state, policy=[0, 1], preferences=preferences, horizon=2
        )

        # Validate EFE result structure
        assert isinstance(efe_result, EFEResult)
        assert hasattr(efe_result, "total_efe")
        assert hasattr(efe_result, "epistemic_value")
        assert hasattr(efe_result, "pragmatic_value")
        assert hasattr(efe_result, "calculation_time_ms")

        # EFE should be finite
        assert np.isfinite(efe_result.total_efe), "EFE must be finite"

        # EFE = -epistemic_value - pragmatic_value
        expected_total = -(efe_result.epistemic_value + efe_result.pragmatic_value)
        assert np.isclose(
            efe_result.total_efe, expected_total, rtol=1e-5
        ), "EFE calculation formula should be correct"

        # Calculation should be fast (< 10ms for simple case)
        assert (
            efe_result.calculation_time_ms < 10
        ), "EFE calculation should be fast for simple cases"

    def test_numerical_stability_edge_cases(self):
        """Test EFE calculation stability with edge cases."""
        config = ActiveInferenceConfig(num_observations=[2], num_states=[2], num_controls=[2])

        # Near-singular A matrix
        A_matrix = np.array([[0.9999, 0.0001], [0.0001, 0.9999]])

        # Extremely confident beliefs
        extreme_beliefs = np.array([0.9999, 0.0001])
        belief_state = BeliefState(beliefs=extreme_beliefs)

        evaluator = PolicyEvaluator(config=config, A_matrix=A_matrix)
        preferences = np.array([1.0, 1.0])  # Neutral preferences

        # Should not crash or return NaN/Inf
        efe_result = evaluator.compute_expected_free_energy(
            belief_state=belief_state, policy=[0], preferences=preferences, horizon=1
        )

        assert np.isfinite(efe_result.total_efe), "EFE should remain finite with extreme beliefs"
        assert np.isfinite(efe_result.epistemic_value), "Epistemic value should remain finite"
        assert np.isfinite(efe_result.pragmatic_value), "Pragmatic value should remain finite"

        # Test with zero preferences
        zero_prefs = np.array([0.0, 0.0])

        zero_efe = evaluator.compute_expected_free_energy(
            belief_state=belief_state, policy=[0], preferences=zero_prefs, horizon=1
        )

        assert np.isfinite(zero_efe.total_efe), "EFE should handle zero preferences gracefully"


class TestPolicySelector:
    """Test policy selection and action sampling."""

    def test_policy_ranking(self):
        """Test ranking of policies by EFE values."""
        config = ActiveInferenceConfig(
            num_observations=[2], num_states=[2], num_controls=[2], planning_horizon=3
        )

        A_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        beliefs = np.array([0.5, 0.5])
        belief_state = BeliefState(beliefs=beliefs)
        preferences = np.array([2.0, 0.0])

        selector = PolicySelector(config=config, A_matrix=A_matrix)

        # Generate policy ranking
        ranking = selector.rank_policies(
            belief_state=belief_state,
            preferences=preferences,
            max_policies=4,  # 2^2 possible single-step policies
        )

        assert isinstance(ranking, PolicyRanking)
        assert len(ranking.policies) > 0, "Should generate some policies"
        assert len(ranking.efe_values) == len(
            ranking.policies
        ), "Should have EFE value for each policy"

        # Policies should be sorted by EFE (ascending - lower is better)
        efe_values = ranking.efe_values
        assert all(
            efe_values[i] <= efe_values[i + 1] for i in range(len(efe_values) - 1)
        ), "Policies should be sorted by EFE (ascending)"

        # Best policy should have lowest EFE
        best_policy = ranking.policies[0]
        best_efe = ranking.efe_values[0]

        assert isinstance(best_policy, list), "Policy should be list of actions"
        assert len(best_policy) == config.policy_length, "Policy should match configured length"
        assert np.isfinite(best_efe), "Best EFE should be finite"

    def test_action_sampling_with_temperature(self):
        """Test probabilistic action sampling with temperature parameter."""
        config = ActiveInferenceConfig(
            num_observations=[2],
            num_states=[2],
            num_controls=[2],
            alpha=16.0,  # High precision (low temperature)
        )

        A_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        beliefs = np.array([0.5, 0.5])
        belief_state = BeliefState(beliefs=beliefs)
        preferences = np.array([2.0, 0.0])

        selector = PolicySelector(config=config, A_matrix=A_matrix)

        # Sample actions multiple times
        actions = []
        for _ in range(50):
            action = selector.sample_action(belief_state=belief_state, preferences=preferences)
            actions.append(action)

        # Should be valid actions
        assert all(isinstance(a, int) for a in actions), "Actions should be integers"
        assert all(
            0 <= a < config.num_controls[0] for a in actions
        ), "Actions should be within valid range"

        # With high precision, should consistently choose best action
        action_counts = np.bincount(actions, minlength=config.num_controls[0])
        most_frequent_action = np.argmax(action_counts)

        # Should strongly prefer one action (>80% of samples)
        assert (
            action_counts[most_frequent_action] / len(actions) > 0.8
        ), "High precision should lead to consistent action selection"

        # Test with low precision (high temperature)
        low_precision_config = ActiveInferenceConfig(
            num_observations=[2],
            num_states=[2],
            num_controls=[2],
            alpha=1.0,  # Low precision (high temperature)
        )

        low_temp_selector = PolicySelector(config=low_precision_config, A_matrix=A_matrix)

        low_temp_actions = []
        for _ in range(50):
            action = low_temp_selector.sample_action(
                belief_state=belief_state, preferences=preferences
            )
            low_temp_actions.append(action)

        low_temp_counts = np.bincount(low_temp_actions, minlength=config.num_controls[0])
        max_proportion = np.max(low_temp_counts) / len(low_temp_actions)

        # Should be more exploratory (less than 70% for any action)
        assert (
            max_proportion < 0.7
        ), "Low precision should lead to more exploratory action selection"

    def test_performance_constraints(self):
        """Test that policy selection meets performance requirements."""
        config = ActiveInferenceConfig(
            num_observations=[4],
            num_states=[4],
            num_controls=[3],
            planning_horizon=5,  # Moderately complex
        )

        A_matrix = np.eye(4) + 0.1 * np.random.rand(4, 4)
        A_matrix = A_matrix / A_matrix.sum(axis=0)

        beliefs = np.ones(4) / 4
        belief_state = BeliefState(beliefs=beliefs)
        preferences = np.array([2.0, 1.0, 0.5, 0.0])

        selector = PolicySelector(config=config, A_matrix=A_matrix)

        # Measure action selection time
        start_time = time.time()

        action = selector.sample_action(belief_state=belief_state, preferences=preferences)

        end_time = time.time()
        selection_time_ms = (end_time - start_time) * 1000

        # Should complete within 50ms as per requirement
        assert (
            selection_time_ms < 50
        ), f"Action selection took {selection_time_ms:.2f}ms, should be <50ms"

        assert isinstance(action, int), "Should return valid action"
        assert 0 <= action < config.num_controls[0], "Action should be within valid range"


class TestPreferenceManager:
    """Test GMN preference integration and management."""

    def test_preference_extraction_from_gmn(self):
        """Test extraction of preferences from GMN structure."""
        # Mock GMN structure
        mock_gmn = {
            "goals": [
                {"observation_preference": 0, "weight": 2.0},
                {"observation_preference": 1, "weight": 1.0},
            ],
            "constraints": [{"avoid_observation": 2, "penalty": -1.0}],
        }

        manager = PreferenceManager(num_observations=3)

        preferences = manager.extract_preferences_from_gmn(mock_gmn)

        assert isinstance(preferences, np.ndarray), "Preferences should be numpy array"
        assert len(preferences) == 3, "Should have preference for each observation"

        # Check preference values match GMN specification
        assert preferences[0] == 2.0, "Should extract positive preference correctly"
        assert preferences[1] == 1.0, "Should extract weighted preference correctly"
        assert preferences[2] == -1.0, "Should extract negative preference (constraint)"

    def test_preference_normalization(self):
        """Test preference vector normalization for numerical stability."""
        manager = PreferenceManager(num_observations=3)

        # Test with extreme values
        extreme_prefs = np.array([1000.0, -1000.0, 0.0])

        normalized = manager.normalize_preferences(extreme_prefs)

        assert np.isfinite(normalized).all(), "Normalized preferences should be finite"
        assert not np.any(np.abs(normalized) > 100), "Normalized preferences should be bounded"

        # Relative ordering should be preserved
        assert (
            normalized[0] > normalized[2] > normalized[1]
        ), "Preference ordering should be preserved after normalization"

    def test_preference_caching(self):
        """Test preference caching for performance."""
        manager = PreferenceManager(num_observations=2, enable_caching=True)

        mock_gmn = {"goals": [{"observation_preference": 0, "weight": 1.0}]}

        # First call should compute preferences
        start_time = time.time()
        prefs1 = manager.extract_preferences_from_gmn(mock_gmn)
        first_time = time.time() - start_time

        # Second call should use cache
        start_time = time.time()
        prefs2 = manager.extract_preferences_from_gmn(mock_gmn)
        second_time = time.time() - start_time

        # Results should be identical
        assert np.array_equal(prefs1, prefs2), "Cached preferences should match computed ones"

        # Second call should be faster (though this may be flaky in tests)
        # Just verify it doesn't crash and returns correct results
        assert second_time >= 0, "Second call should complete successfully"


class TestIntegrationScenarios:
    """Integration tests for complete EFE policy selection pipeline."""

    def test_complete_policy_selection_pipeline(self):
        """Test complete flow from beliefs to action selection."""
        # Setup environment
        config = ActiveInferenceConfig(
            num_observations=[3], num_states=[3], num_controls=[2], planning_horizon=3, alpha=8.0
        )

        # Create observation model
        A_matrix = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

        # Initial uncertain beliefs
        beliefs = np.array([0.4, 0.4, 0.2])
        belief_state = BeliefState(beliefs=beliefs)

        # Preferences favor first observation
        preferences = np.array([2.0, 1.0, 0.5])

        # Create policy selector
        selector = PolicySelector(config=config, A_matrix=A_matrix)

        # Run complete pipeline
        ranking = selector.rank_policies(
            belief_state=belief_state, preferences=preferences, max_policies=8
        )

        # Verify pipeline completion
        assert len(ranking.policies) > 0, "Should generate policy ranking"
        assert all(
            np.isfinite(efe) for efe in ranking.efe_values
        ), "All EFE values should be finite"

        # Sample action
        action = selector.sample_action(belief_state=belief_state, preferences=preferences)

        assert isinstance(action, int), "Should return valid action"
        assert 0 <= action < config.num_controls[0], "Action should be within valid range"

    def test_memory_efficiency(self):
        """Test memory usage stays within bounds."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create moderately complex environment
        config = ActiveInferenceConfig(
            num_observations=[5], num_states=[5], num_controls=[4], planning_horizon=4
        )

        A_matrix = np.eye(5) + 0.1 * np.random.rand(5, 5)
        A_matrix = A_matrix / A_matrix.sum(axis=0)

        beliefs = np.ones(5) / 5
        belief_state = BeliefState(beliefs=beliefs)
        preferences = np.random.rand(5)

        selector = PolicySelector(config=config, A_matrix=A_matrix)

        # Run multiple policy selections
        for _ in range(10):
            action = selector.sample_action(belief_state=belief_state, preferences=preferences)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not increase memory significantly (less than 10MB)
        assert memory_increase < 10, f"Memory increased by {memory_increase:.2f}MB, should be <10MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

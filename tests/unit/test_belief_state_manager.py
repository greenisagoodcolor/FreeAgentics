"""Tests for Belief State Management System (Task 44.2).

This module implements comprehensive tests for the belief state management
system following the Nemesis Committee's TDD recommendations.

Based on:
- Kent Beck: TDD with mathematical correctness verification
- Michael Feathers: Characterization tests and proper seams
- Sindre Sorhus: Property-based testing and comprehensive validation
- Jessica Kerr: Observability testing with metrics validation
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from inference.active.belief_manager import (
    BeliefState,
    BeliefStateManager,
    BeliefUpdateResult,
    InMemoryBeliefRepository,
)
from inference.active.config import ActiveInferenceConfig


class TestBeliefState:
    """Test the BeliefState class for correct probabilistic behavior."""

    def test_belief_state_creation_valid(self):
        """Test creating valid belief states."""
        # Uniform distribution
        beliefs = np.array([0.5, 0.5])
        state = BeliefState(beliefs)

        assert np.allclose(state.beliefs, beliefs)
        assert state.entropy > 0  # Uniform has positive entropy
        assert state.max_confidence == 0.5
        assert state.effective_states == 2
        assert isinstance(state.timestamp, float)
        assert state.observation_history == []

    def test_belief_state_creation_with_history(self):
        """Test creating belief state with observation history."""
        beliefs = np.array([0.8, 0.2])
        history = [0, 1, 0]

        state = BeliefState(beliefs, observation_history=history)

        assert state.observation_history == history
        assert state.max_confidence == 0.8
        assert state.effective_states == 2  # Both states have >1% probability

    def test_belief_state_validation_sum_not_one(self):
        """Test that beliefs must sum to 1."""
        beliefs = np.array([0.6, 0.6])  # Sum = 1.2

        with pytest.raises(ValueError, match="must sum to 1.0"):
            BeliefState(beliefs)

    def test_belief_state_validation_negative_values(self):
        """Test that beliefs cannot be negative."""
        beliefs = np.array([1.1, -0.1])  # Negative value

        with pytest.raises(ValueError, match="must be non-negative"):
            BeliefState(beliefs)

    def test_belief_state_validation_not_finite(self):
        """Test that beliefs must be finite."""
        beliefs = np.array([0.5, np.inf])

        with pytest.raises(ValueError, match="must be finite"):
            BeliefState(beliefs)

    def test_belief_state_validation_wrong_dimensions(self):
        """Test that beliefs must be 1-dimensional."""
        beliefs = np.array([[0.5, 0.5], [0.3, 0.7]])  # 2D array

        with pytest.raises(ValueError, match="must be 1-dimensional"):
            BeliefState(beliefs)

    def test_belief_state_entropy_calculation(self):
        """Test entropy calculation for different distributions."""
        # Uniform distribution (maximum entropy for 2 states)
        uniform = BeliefState(np.array([0.5, 0.5]))

        # Deterministic distribution (minimum entropy)
        deterministic = BeliefState(np.array([1.0, 0.0]))

        # Skewed distribution
        skewed = BeliefState(np.array([0.9, 0.1]))

        assert uniform.entropy > skewed.entropy > deterministic.entropy
        assert deterministic.entropy < 1e-10  # Essentially zero uncertainty

    def test_belief_state_kl_divergence(self):
        """Test KL divergence calculation between belief states."""
        p = BeliefState(np.array([0.8, 0.2]))
        q = BeliefState(np.array([0.6, 0.4]))

        kl_div = p.kl_divergence_from(q)

        assert kl_div >= 0  # KL divergence is non-negative
        assert isinstance(kl_div, float)

        # Self-divergence should be 0
        self_div = p.kl_divergence_from(p)
        assert abs(self_div) < 1e-10

    def test_belief_state_most_likely_state(self):
        """Test identification of most likely state."""
        beliefs = np.array([0.3, 0.7, 0.0])
        state = BeliefState(beliefs)

        assert state.most_likely_state() == 1  # Index of max probability

    def test_belief_state_effective_states(self):
        """Test effective states calculation."""
        # All states significant
        all_significant = BeliefState(np.array([0.4, 0.3, 0.3]))
        assert all_significant.effective_states == 3

        # One state at threshold (exactly 1%)
        at_threshold = BeliefState(np.array([0.5, 0.49, 0.01]))
        assert at_threshold.effective_states == 2  # 0.01 is not > 0.01

        # One state very insignificant
        very_insignificant = BeliefState(np.array([0.995, 0.004, 0.001]))
        assert very_insignificant.effective_states == 1


class TestInMemoryBeliefRepository:
    """Test the in-memory belief repository implementation."""

    def test_repository_save_and_retrieve(self):
        """Test basic save and retrieve operations."""
        repo = InMemoryBeliefRepository()
        belief_state = BeliefState(np.array([0.6, 0.4]))
        agent_id = "test_agent"

        # Save belief state
        repo.save_belief_state(belief_state, agent_id)

        # Retrieve current state
        retrieved = repo.get_current_belief_state(agent_id)

        assert retrieved is not None
        assert np.allclose(retrieved.beliefs, belief_state.beliefs)

    def test_repository_history_tracking(self):
        """Test belief history tracking."""
        repo = InMemoryBeliefRepository(max_history_per_agent=3)
        agent_id = "test_agent"

        # Save multiple belief states
        states = [
            BeliefState(np.array([1.0, 0.0])),
            BeliefState(np.array([0.8, 0.2])),
            BeliefState(np.array([0.6, 0.4])),
            BeliefState(np.array([0.4, 0.6])),  # Should evict first state
        ]

        for state in states:
            repo.save_belief_state(state, agent_id)

        history = repo.get_belief_history(agent_id)

        # Should only have last 3 states due to max_history limit
        assert len(history) == 3
        assert np.allclose(history[-1].beliefs, states[-1].beliefs)

    def test_repository_clear_history(self):
        """Test clearing agent history."""
        repo = InMemoryBeliefRepository()
        agent_id = "test_agent"

        # Save some states
        repo.save_belief_state(BeliefState(np.array([0.5, 0.5])), agent_id)
        repo.save_belief_state(BeliefState(np.array([0.7, 0.3])), agent_id)

        # Clear history
        repo.clear_history(agent_id)

        assert repo.get_current_belief_state(agent_id) is None
        assert repo.get_belief_history(agent_id) == []


class TestBeliefStateManager:
    """Test the core belief state management system."""

    def create_test_manager(self) -> BeliefStateManager:
        """Create a test belief state manager with simple 2x2 environment."""
        config = ActiveInferenceConfig(num_observations=[2], num_states=[2], num_controls=[2])

        # Perfect observation model (identity matrix)
        A_matrix = np.array(
            [
                [0.9, 0.1],  # state 0 -> obs 0 with 0.9 prob
                [0.1, 0.9],
            ]
        )  # state 1 -> obs 1 with 0.9 prob

        repository = InMemoryBeliefRepository()

        return BeliefStateManager(
            config=config, repository=repository, A_matrix=A_matrix, agent_id="test_agent"
        )

    def test_manager_initialization(self):
        """Test belief state manager initialization."""
        manager = self.create_test_manager()

        current_beliefs = manager.get_current_beliefs()

        # Should start with uniform beliefs
        assert np.allclose(current_beliefs.beliefs, [0.5, 0.5])
        assert current_beliefs.observation_history == []
        assert current_beliefs.entropy > 0  # Should have uncertainty initially

    def test_manager_belief_update_correct_mathematics(self):
        """Test mathematically correct Bayesian belief updates."""
        manager = self.create_test_manager()

        # Initial uniform beliefs: [0.5, 0.5]
        # A matrix: [[0.9, 0.1], [0.1, 0.9]]
        # Observe 0: P(s|o=0) ∝ P(o=0|s) * P(s) = [0.9, 0.1] * [0.5, 0.5] = [0.45, 0.05]
        # Normalized: [0.9, 0.1]

        result = manager.update_beliefs(observation=0)

        # Check mathematical correctness
        expected_beliefs = np.array([0.9, 0.1])
        assert np.allclose(result.new_belief_state.beliefs, expected_beliefs, rtol=1e-10)

        # Check observability metrics
        assert result.observation == 0
        assert result.update_time_ms > 0
        assert result.entropy_change < 0  # Entropy should decrease (less uncertainty)
        assert result.kl_divergence > 0  # Beliefs should change
        assert result.confidence_change > 0  # Confidence should increase

    def test_manager_sequential_belief_updates(self):
        """Test sequential belief updates accumulate correctly."""
        manager = self.create_test_manager()

        # First observation: 0
        result1 = manager.update_beliefs(0)
        beliefs_after_first = result1.new_belief_state.beliefs

        # Second observation: 0 again
        result2 = manager.update_beliefs(0)
        beliefs_after_second = result2.new_belief_state.beliefs

        # After two observations of 0, belief in state 0 should be even higher
        assert beliefs_after_second[0] > beliefs_after_first[0]
        assert beliefs_after_second[1] < beliefs_after_first[1]

        # Check observation history
        final_beliefs = manager.get_current_beliefs()
        assert final_beliefs.observation_history == [0, 0]

    def test_manager_belief_update_invalid_observation(self):
        """Test error handling for invalid observations."""
        manager = self.create_test_manager()

        # Test negative observation
        with pytest.raises(ValueError, match="out of range"):
            manager.update_beliefs(-1)

        # Test too large observation
        with pytest.raises(ValueError, match="out of range"):
            manager.update_beliefs(2)

        # Test non-integer observation
        with pytest.raises(ValueError, match="must be integer"):
            manager.update_beliefs(0.5)

    def test_manager_belief_reset(self):
        """Test belief state reset functionality."""
        manager = self.create_test_manager()

        # Update beliefs to non-uniform state
        manager.update_beliefs(0)
        assert not np.allclose(manager.get_current_beliefs().beliefs, [0.5, 0.5])

        # Reset to uniform
        manager.reset_beliefs()

        reset_state = manager.get_current_beliefs()
        assert np.allclose(reset_state.beliefs, [0.5, 0.5])
        assert reset_state.observation_history == []

    def test_manager_custom_belief_reset(self):
        """Test reset with custom initial beliefs."""
        manager = self.create_test_manager()

        custom_beliefs = np.array([0.8, 0.2])
        manager.reset_beliefs(custom_beliefs)

        current = manager.get_current_beliefs()
        assert np.allclose(current.beliefs, custom_beliefs)

    def test_manager_belief_summary(self):
        """Test belief summary generation for monitoring."""
        manager = self.create_test_manager()

        # Update beliefs
        manager.update_beliefs(0)
        manager.update_beliefs(1)

        summary = manager.get_belief_summary()

        # Check required fields
        required_fields = [
            "entropy",
            "max_confidence",
            "effective_states",
            "most_likely_state",
            "num_observations",
            "belief_distribution",
            "timestamp",
        ]

        for field in required_fields:
            assert field in summary

        assert summary["num_observations"] == 2
        assert isinstance(summary["belief_distribution"], list)
        assert len(summary["belief_distribution"]) == 2

    def test_manager_zero_posterior_handling(self):
        """Test handling of zero posterior probability scenarios."""
        config = ActiveInferenceConfig()

        # Create A matrix where observation 0 is impossible from state 1
        A_matrix = np.array(
            [
                [1.0, 0.0],  # obs 0 only from state 0
                [0.0, 1.0],
            ]
        )  # obs 1 only from state 1

        repository = InMemoryBeliefRepository()
        manager = BeliefStateManager(config, repository, A_matrix, "test")

        # Start with belief only in state 1
        manager.reset_beliefs(np.array([0.0, 1.0]))

        # Observe 0 (impossible from state 1)
        with patch("inference.active.belief_manager.logger") as mock_logger:
            result = manager.update_beliefs(0)

            # Should keep previous beliefs and log warning
            mock_logger.warning.assert_called_once()
            assert np.allclose(result.new_belief_state.beliefs, [0.0, 1.0])


class TestBeliefUpdateResult:
    """Test the belief update result class for observability."""

    def test_update_result_metrics(self):
        """Test computation of update result metrics."""
        # Create before and after states
        before = BeliefState(np.array([0.5, 0.5]))
        after = BeliefState(np.array([0.8, 0.2]))

        result = BeliefUpdateResult(
            new_belief_state=after, previous_belief_state=before, observation=0, update_time_ms=5.0
        )

        # Check metrics
        assert result.entropy_change < 0  # Entropy decreased
        assert result.kl_divergence > 0  # States are different
        assert result.confidence_change > 0  # Confidence increased
        assert result.update_time_ms == 5.0
        assert result.observation == 0


# Property-based tests as recommended by Sindre Sorhus
class TestBeliefStateProperties:
    """Property-based tests for belief state management."""

    def test_belief_normalization_property(self):
        """Property: belief states always remain normalized after updates."""
        manager = self.create_test_manager()

        # Test with random observation sequences
        np.random.seed(42)  # Reproducible
        observations = np.random.randint(0, 2, size=20)

        for obs in observations:
            manager.update_beliefs(int(obs))
            beliefs = manager.get_current_beliefs().beliefs

            # Property: beliefs always sum to 1
            assert abs(np.sum(beliefs) - 1.0) < 1e-10
            # Property: all beliefs are non-negative
            assert np.all(beliefs >= 0)

    def test_entropy_monotonicity_property(self):
        """Property: entropy decreases with informative observations."""
        manager = self.create_test_manager()

        initial_entropy = manager.get_current_beliefs().entropy

        # Make informative observation (state 0 more likely given obs 0)
        result = manager.update_beliefs(0)

        # Property: informative observations should reduce entropy
        assert result.new_belief_state.entropy < initial_entropy

    def test_belief_update_associativity_property(self):
        """Property: independent observations can be processed in any order."""
        # This is a simplified test - true associativity requires independent observations
        config = ActiveInferenceConfig()
        A_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])

        # Manager 1: observe 0 then 1
        repo1 = InMemoryBeliefRepository()
        manager1 = BeliefStateManager(config, repo1, A_matrix, "agent1")
        manager1.update_beliefs(0)
        manager1.update_beliefs(1)
        beliefs1 = manager1.get_current_beliefs().beliefs

        # Manager 2: observe 1 then 0
        repo2 = InMemoryBeliefRepository()
        manager2 = BeliefStateManager(config, repo2, A_matrix, "agent2")
        manager2.update_beliefs(1)
        manager2.update_beliefs(0)
        beliefs2 = manager2.get_current_beliefs().beliefs

        # Property: order shouldn't matter for same observation set
        # (This is approximate due to numerical precision)
        assert np.allclose(beliefs1, beliefs2, rtol=1e-10)

    def create_test_manager(self) -> BeliefStateManager:
        """Helper to create test manager."""
        config = ActiveInferenceConfig()
        A_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        repository = InMemoryBeliefRepository()
        return BeliefStateManager(config, repository, A_matrix, "test")


# Performance tests as recommended by Addy Osmani
class TestBeliefStatePerformance:
    """Performance tests for belief state operations."""

    def test_belief_update_performance(self):
        """Test that belief updates complete within SLA requirements."""
        config = ActiveInferenceConfig()
        A_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        repository = InMemoryBeliefRepository()
        manager = BeliefStateManager(config, repository, A_matrix, "perf_test")

        # Measure update time
        start_time = time.time()
        result = manager.update_beliefs(0)
        end_time = time.time()

        actual_time_ms = (end_time - start_time) * 1000

        # SLA requirement: updates should complete within 10ms
        assert actual_time_ms < 10, f"Update took {actual_time_ms}ms, exceeding 10ms SLA"
        assert result.update_time_ms < 10

    def test_large_state_space_performance(self):
        """Test performance with larger state spaces."""
        config = ActiveInferenceConfig(num_observations=[10], num_states=[10], num_controls=[5])

        # Create 10x10 observation model
        A_matrix = np.random.rand(10, 10)
        A_matrix = A_matrix / A_matrix.sum(axis=0)  # Normalize columns

        repository = InMemoryBeliefRepository()
        manager = BeliefStateManager(config, repository, A_matrix, "large_test")

        # Test multiple updates
        times = []
        for i in range(50):
            obs = i % 10
            result = manager.update_beliefs(obs)
            times.append(result.update_time_ms)

        avg_time = np.mean(times)
        max_time = np.max(times)

        # Performance requirements for larger spaces
        assert avg_time < 20, f"Average update time {avg_time}ms too high"
        assert max_time < 50, f"Max update time {max_time}ms too high"

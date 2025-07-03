"""
Comprehensive tests for Epistemic Value Calculation Engine.

Tests all aspects of epistemic value calculations, knowledge propagation,
and collective intelligence metrics with mathematical rigor.
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest

from agents.base.epistemic_value_engine import (
    CollectiveIntelligenceMetrics,
    EpistemicState,
    EpistemicValueCalculationEngine,
    EpistemicValueType,
    KnowledgePropagationEvent,
    epistemic_engine,
)


class TestEpistemicValueCalculationEngine:
    """Test the main epistemic value calculation engine."""

    @pytest.fixture
    def engine(self):
        """Create a fresh engine instance for each test."""
        return EpistemicValueCalculationEngine()

    @pytest.fixture
    def sample_belief_distribution(self):
        """Sample normalized belief distribution."""
        beliefs = np.array([0.1, 0.3, 0.4, 0.2])
        return beliefs / np.sum(beliefs)  # Ensure normalized

    @pytest.fixture
    def uniform_belief_distribution(self):
        """Uniform belief distribution for testing."""
        return np.array([0.25, 0.25, 0.25, 0.25])

    @pytest.fixture
    def concentrated_belief_distribution(self):
        """Highly concentrated belief distribution."""
        return np.array([0.9, 0.05, 0.03, 0.02])

    @pytest.fixture
    def sample_observations(self):
        """Sample observation data."""
        return np.array([1, 0, 1, 0])

    def test_engine_initialization(self, engine):
        """Test engine initializes with empty state."""
        assert len(engine.agent_states) == 0
        assert len(engine.propagation_history) == 0
        assert len(engine.network_metrics_history) == 0

    def test_calculate_epistemic_value_basic(
        self, engine, sample_belief_distribution, sample_observations
    ):
        """Test basic epistemic value calculation."""
        agent_id = "test_agent_1"

        result = engine.calculate_epistemic_value(
            agent_id=agent_id,
            belief_distribution=sample_belief_distribution,
            observations=sample_observations,
        )

        # Verify result structure
        assert isinstance(result, EpistemicState)
        assert result.agent_id == agent_id
        assert np.array_equal(result.belief_distribution, sample_belief_distribution)
        assert isinstance(result.knowledge_entropy, float)
        assert isinstance(result.confidence_level, float)
        assert isinstance(result.certainty_measure, float)
        assert isinstance(result.epistemic_value, float)

        # Verify value ranges
        assert 0.0 <= result.confidence_level <= 1.0
        assert 0.0 <= result.certainty_measure <= 1.0
        assert 0.0 <= result.epistemic_value <= 1.0
        assert result.knowledge_entropy >= 0.0

        # Verify agent state is stored
        assert agent_id in engine.agent_states
        assert engine.agent_states[agent_id] == result

    def test_calculate_epistemic_value_with_prior(
        self, engine, sample_belief_distribution, uniform_belief_distribution, sample_observations
    ):
        """Test epistemic value calculation with explicit prior."""
        agent_id = "test_agent_2"

        result = engine.calculate_epistemic_value(
            agent_id=agent_id,
            belief_distribution=sample_belief_distribution,
            observations=sample_observations,
            prior_distribution=uniform_belief_distribution,
        )

        assert isinstance(result, EpistemicState)
        assert result.agent_id == agent_id

        # With a prior, information gain should be calculated differently
        assert result.epistemic_value > 0.0

    def test_knowledge_entropy_calculation(
        self, engine, uniform_belief_distribution, concentrated_belief_distribution
    ):
        """Test knowledge entropy calculation for different distributions."""
        # Uniform distribution should have higher entropy
        uniform_entropy = engine._calculate_knowledge_entropy(uniform_belief_distribution)
        concentrated_entropy = engine._calculate_knowledge_entropy(concentrated_belief_distribution)

        assert uniform_entropy > concentrated_entropy
        assert uniform_entropy > 0.0
        assert concentrated_entropy >= 0.0

    def test_information_gain_calculation(
        self, engine, sample_belief_distribution, uniform_belief_distribution
    ):
        """Test information gain (KL divergence) calculation."""
        # Information gain from uniform to sample distribution
        info_gain = engine._calculate_information_gain(
            sample_belief_distribution, uniform_belief_distribution
        )

        assert isinstance(info_gain, float)
        assert info_gain >= 0.0  # KL divergence is always non-negative

        # Self-information gain should be zero
        self_gain = engine._calculate_information_gain(
            sample_belief_distribution, sample_belief_distribution
        )
        assert abs(self_gain) < 1e-10  # Should be essentially zero

    def test_confidence_level_calculation(
        self, engine, uniform_belief_distribution, concentrated_belief_distribution
    ):
        """Test confidence level calculation."""
        uniform_confidence = engine._calculate_confidence_level(uniform_belief_distribution)
        concentrated_confidence = engine._calculate_confidence_level(
            concentrated_belief_distribution
        )

        # Concentrated distribution should have higher confidence
        assert concentrated_confidence > uniform_confidence
        assert 0.0 <= uniform_confidence <= 1.0
        assert 0.0 <= concentrated_confidence <= 1.0

    def test_certainty_measure_calculation(
        self, engine, uniform_belief_distribution, concentrated_belief_distribution
    ):
        """Test certainty measure calculation."""
        uniform_certainty = engine._calculate_certainty_measure(uniform_belief_distribution)
        concentrated_certainty = engine._calculate_certainty_measure(
            concentrated_belief_distribution
        )

        # Concentrated distribution should have higher certainty
        assert concentrated_certainty > uniform_certainty
        assert 0.0 <= uniform_certainty <= 1.0
        assert 0.0 <= concentrated_certainty <= 1.0

    def test_overall_epistemic_value_calculation(self, engine):
        """Test overall epistemic value calculation."""
        entropy = 1.5
        info_gain = 0.8
        confidence = 0.7
        certainty = 0.6

        epistemic_value = engine._calculate_overall_epistemic_value(
            entropy, info_gain, confidence, certainty
        )

        assert isinstance(epistemic_value, float)
        assert 0.0 <= epistemic_value <= 1.0

    def test_calculate_knowledge_propagation(
        self, engine, sample_belief_distribution, uniform_belief_distribution, sample_observations
    ):
        """Test knowledge propagation between agents."""
        # Set up two agents
        source_id = "source_agent"
        target_id = "target_agent"

        # Create initial states for both agents
        engine.calculate_epistemic_value(source_id, sample_belief_distribution, sample_observations)
        engine.calculate_epistemic_value(
            target_id, uniform_belief_distribution, sample_observations
        )

        # Test knowledge propagation
        shared_info = np.array([0.2, 0.3, 0.3, 0.2])

        propagation_event = engine.calculate_knowledge_propagation(
            source_agent_id=source_id, target_agent_id=target_id, shared_information=shared_info
        )

        # Verify propagation event structure
        assert isinstance(propagation_event, KnowledgePropagationEvent)
        assert propagation_event.source_agent == source_id
        assert propagation_event.target_agent == target_id
        assert isinstance(propagation_event.information_transferred, float)
        assert isinstance(propagation_event.epistemic_gain, float)
        assert isinstance(propagation_event.propagation_efficiency, float)
        assert propagation_event.belief_change is not None

        # Verify ranges
        assert 0.0 <= propagation_event.propagation_efficiency <= 1.0
        assert propagation_event.information_transferred >= 0.0

        # Verify event is stored
        assert len(engine.propagation_history) == 1
        assert engine.propagation_history[0] == propagation_event

        # Verify target agent state was updated
        updated_target = engine.agent_states[target_id]
        assert not np.array_equal(updated_target.belief_distribution, uniform_belief_distribution)

    def test_knowledge_propagation_invalid_agents(self, engine):
        """Test knowledge propagation with invalid agent IDs."""
        shared_info = np.array([0.25, 0.25, 0.25, 0.25])

        with pytest.raises(ValueError, match="Both agents must have existing epistemic states"):
            engine.calculate_knowledge_propagation("nonexistent1", "nonexistent2", shared_info)

    def test_belief_update_bayesian(self, engine):
        """Test Bayesian belief update."""
        target_beliefs = np.array([0.4, 0.3, 0.2, 0.1])
        shared_info = np.array([0.1, 0.5, 0.3, 0.1])

        updated = engine._update_beliefs_with_shared_info(
            target_beliefs, shared_info, "bayesian_update"
        )

        # Should be normalized
        assert abs(np.sum(updated) - 1.0) < 1e-10
        assert len(updated) == len(target_beliefs)
        assert np.all(updated >= 0.0)

    def test_belief_update_weighted_average(self, engine):
        """Test weighted average belief update."""
        target_beliefs = np.array([0.4, 0.3, 0.2, 0.1])
        shared_info = np.array([0.1, 0.5, 0.3, 0.1])

        updated = engine._update_beliefs_with_shared_info(
            target_beliefs, shared_info, "weighted_average"
        )

        # Should be normalized
        assert abs(np.sum(updated) - 1.0) < 1e-10
        assert len(updated) == len(target_beliefs)
        assert np.all(updated >= 0.0)

    def test_propagation_efficiency_calculation(self, engine):
        """Test propagation efficiency calculation."""
        # Normal case
        efficiency = engine._calculate_propagation_efficiency(0.5, 0.3)
        assert 0.0 <= efficiency <= 1.0

        # Zero information transfer
        efficiency_zero = engine._calculate_propagation_efficiency(0.0, 0.3)
        assert efficiency_zero == 0.0

    def test_calculate_collective_intelligence_metrics(
        self, engine, sample_belief_distribution, uniform_belief_distribution, sample_observations
    ):
        """Test collective intelligence metrics calculation."""
        # Set up multiple agents
        agent_ids = ["agent1", "agent2", "agent3"]
        distributions = [
            sample_belief_distribution,
            uniform_belief_distribution,
            np.array([0.2, 0.2, 0.3, 0.3]),
        ]

        for agent_id, dist in zip(agent_ids, distributions):
            engine.calculate_epistemic_value(agent_id, dist, sample_observations)

        # Define network topology
        network = {
            "agent1": ["agent2", "agent3"],
            "agent2": ["agent1", "agent3"],
            "agent3": ["agent1", "agent2"],
        }

        metrics = engine.calculate_collective_intelligence_metrics(network)

        # Verify metrics structure
        assert isinstance(metrics, CollectiveIntelligenceMetrics)
        assert isinstance(metrics.network_entropy, float)
        assert isinstance(metrics.information_diversity, float)
        assert isinstance(metrics.consensus_level, float)
        assert isinstance(metrics.knowledge_distribution, float)
        assert isinstance(metrics.epistemic_efficiency, float)
        assert isinstance(metrics.collective_accuracy, float)
        assert isinstance(metrics.emergence_indicator, float)
        assert isinstance(metrics.stability_measure, float)

        # Verify ranges
        assert metrics.network_entropy >= 0.0
        assert metrics.information_diversity >= 0.0
        assert 0.0 <= metrics.consensus_level <= 1.0
        assert 0.0 <= metrics.epistemic_efficiency <= 1.0
        assert 0.0 <= metrics.collective_accuracy <= 1.0
        assert metrics.emergence_indicator >= 0.0
        assert 0.0 <= metrics.stability_measure <= 1.0

        # Verify metrics are stored
        assert len(engine.network_metrics_history) == 1
        assert engine.network_metrics_history[0] == metrics

    def test_collective_intelligence_insufficient_agents(self, engine):
        """Test collective intelligence with insufficient agents."""
        # No agents
        network = {}
        metrics = engine.calculate_collective_intelligence_metrics(network)

        # Should return zero metrics
        assert metrics.network_entropy == 0.0
        assert metrics.information_diversity == 0.0
        assert metrics.consensus_level == 0.0

        # One agent
        engine.calculate_epistemic_value(
            "solo_agent", np.array([0.25, 0.25, 0.25, 0.25]), np.array([1, 0, 1, 0])
        )
        metrics_solo = engine.calculate_collective_intelligence_metrics({"solo_agent": []})

        # Can't calculate diversity with one agent
        assert metrics_solo.information_diversity == 0.0

    def test_get_network_analytics(self, engine, sample_belief_distribution, sample_observations):
        """Test network analytics generation."""
        # Add some agents and propagation events
        engine.calculate_epistemic_value("agent1", sample_belief_distribution, sample_observations)
        engine.calculate_epistemic_value("agent2", sample_belief_distribution, sample_observations)

        # Add a propagation event
        shared_info = np.array([0.2, 0.3, 0.3, 0.2])
        engine.calculate_knowledge_propagation("agent1", "agent2", shared_info)

        # Add metrics history
        dummy_metrics = CollectiveIntelligenceMetrics(
            network_entropy=1.0,
            information_diversity=0.5,
            consensus_level=0.7,
            knowledge_distribution=0.6,
            epistemic_efficiency=0.8,
            collective_accuracy=0.7,
            emergence_indicator=0.1,
            stability_measure=0.9,
        )
        engine.network_metrics_history.append(dummy_metrics)

        analytics = engine.get_network_analytics()

        # Verify analytics structure
        assert isinstance(analytics, dict)
        assert "agent_count" in analytics
        assert "total_propagation_events" in analytics
        assert "average_epistemic_value" in analytics
        assert "network_diversity" in analytics
        assert "consensus_level" in analytics
        assert "emergence_indicator" in analytics
        assert "recent_propagation_efficiency" in analytics
        assert "top_epistemic_agents" in analytics

        # Verify values
        assert analytics["agent_count"] == 2
        assert analytics["total_propagation_events"] == 1
        assert isinstance(analytics["average_epistemic_value"], (int, float))
        assert analytics["network_diversity"] == 0.5
        assert analytics["consensus_level"] == 0.7
        assert analytics["emergence_indicator"] == 0.1
        assert isinstance(analytics["recent_propagation_efficiency"], (int, float))
        assert isinstance(analytics["top_epistemic_agents"], list)
        assert len(analytics["top_epistemic_agents"]) <= 5


class TestEpistemicValueDataClasses:
    """Test the data classes used in epistemic value calculations."""

    def test_epistemic_state_creation(self):
        """Test EpistemicState creation and defaults."""
        belief_dist = np.array([0.2, 0.3, 0.3, 0.2])

        state = EpistemicState(
            agent_id="test_agent",
            belief_distribution=belief_dist,
            knowledge_entropy=1.5,
            confidence_level=0.7,
        )

        assert state.agent_id == "test_agent"
        assert np.array_equal(state.belief_distribution, belief_dist)
        assert state.knowledge_entropy == 1.5
        assert state.confidence_level == 0.7
        assert state.information_sources == []  # Default
        assert isinstance(state.timestamp, datetime)
        assert state.certainty_measure == 0.0  # Default
        assert state.epistemic_value == 0.0  # Default

    def test_knowledge_propagation_event_creation(self):
        """Test KnowledgePropagationEvent creation."""
        belief_change = np.array([0.1, -0.1, 0.05, -0.05])

        event = KnowledgePropagationEvent(
            source_agent="agent1",
            target_agent="agent2",
            information_transferred=0.5,
            epistemic_gain=0.3,
            propagation_efficiency=0.6,
            belief_change=belief_change,
        )

        assert event.source_agent == "agent1"
        assert event.target_agent == "agent2"
        assert event.information_transferred == 0.5
        assert event.epistemic_gain == 0.3
        assert event.propagation_efficiency == 0.6
        assert np.array_equal(event.belief_change, belief_change)
        assert isinstance(event.timestamp, datetime)

    def test_collective_intelligence_metrics_creation(self):
        """Test CollectiveIntelligenceMetrics creation."""
        metrics = CollectiveIntelligenceMetrics(
            network_entropy=1.2,
            information_diversity=0.8,
            consensus_level=0.6,
            knowledge_distribution=0.7,
            epistemic_efficiency=0.75,
            collective_accuracy=0.8,
            emergence_indicator=0.1,
            stability_measure=0.9,
        )

        assert metrics.network_entropy == 1.2
        assert metrics.information_diversity == 0.8
        assert metrics.consensus_level == 0.6
        assert metrics.knowledge_distribution == 0.7
        assert metrics.epistemic_efficiency == 0.75
        assert metrics.collective_accuracy == 0.8
        assert metrics.emergence_indicator == 0.1
        assert metrics.stability_measure == 0.9


class TestEpistemicValueType:
    """Test the EpistemicValueType enum."""

    def test_epistemic_value_type_enum(self):
        """Test all enum values are accessible."""
        assert EpistemicValueType.INFORMATION_GAIN.value == "information_gain"
        assert EpistemicValueType.KNOWLEDGE_ENTROPY.value == "knowledge_entropy"
        assert EpistemicValueType.BELIEF_DIVERGENCE.value == "belief_divergence"
        assert EpistemicValueType.COLLECTIVE_INTELLIGENCE.value == "collective_intelligence"
        assert EpistemicValueType.KNOWLEDGE_PROPAGATION.value == "knowledge_propagation"
        assert EpistemicValueType.EPISTEMIC_CONVERGENCE.value == "epistemic_convergence"


class TestGlobalEpistemicEngine:
    """Test the global epistemic engine instance."""

    def test_global_engine_exists(self):
        """Test that global engine instance exists."""
        assert epistemic_engine is not None
        assert isinstance(epistemic_engine, EpistemicValueCalculationEngine)

    def test_global_engine_functionality(self):
        """Test that global engine works correctly."""
        # Clear any existing state
        epistemic_engine.agent_states.clear()
        epistemic_engine.propagation_history.clear()
        epistemic_engine.network_metrics_history.clear()

        # Test basic functionality
        belief_dist = np.array([0.2, 0.3, 0.3, 0.2])
        observations = np.array([1, 0, 1, 0])

        result = epistemic_engine.calculate_epistemic_value(
            agent_id="global_test_agent", belief_distribution=belief_dist, observations=observations
        )

        assert isinstance(result, EpistemicState)
        assert result.agent_id == "global_test_agent"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_zero_sum_belief_distribution(self):
        """Test handling of zero-sum belief distribution."""
        engine = EpistemicValueCalculationEngine()
        zero_beliefs = np.array([0.0, 0.0, 0.0, 0.0])
        observations = np.array([1, 0, 1, 0])

        # Should handle gracefully without crashing
        result = engine.calculate_epistemic_value("test_agent", zero_beliefs, observations)
        assert isinstance(result, EpistemicState)

    def test_single_element_distribution(self):
        """Test handling of single-element distribution."""
        engine = EpistemicValueCalculationEngine()
        single_belief = np.array([1.0])
        observations = np.array([1])

        result = engine.calculate_epistemic_value("test_agent", single_belief, observations)
        assert isinstance(result, EpistemicState)
        assert result.confidence_level == 1.0  # Should be maximally confident

    def test_very_small_probabilities(self):
        """Test handling of very small probabilities."""
        engine = EpistemicValueCalculationEngine()
        small_beliefs = np.array([1e-15, 1e-15, 1e-15, 1.0 - 3e-15])
        observations = np.array([1, 0, 1, 0])

        result = engine.calculate_epistemic_value("test_agent", small_beliefs, observations)
        assert isinstance(result, EpistemicState)
        assert np.isfinite(result.knowledge_entropy)
        assert np.isfinite(result.epistemic_value)

    @patch("agents.base.epistemic_value_engine.logger")
    def test_logging_functionality(self, mock_logger):
        """Test that logging works correctly."""
        engine = EpistemicValueCalculationEngine()
        belief_dist = np.array([0.2, 0.3, 0.3, 0.2])
        observations = np.array([1, 0, 1, 0])

        # Test epistemic value calculation logging
        engine.calculate_epistemic_value("test_agent", belief_dist, observations)
        mock_logger.debug.assert_called()

        # Test knowledge propagation logging
        engine.calculate_epistemic_value("agent2", belief_dist, observations)
        shared_info = np.array([0.25, 0.25, 0.25, 0.25])

        engine.calculate_knowledge_propagation("test_agent", "agent2", shared_info)
        mock_logger.info.assert_called()

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values."""
        engine = EpistemicValueCalculationEngine()

        # Test with very concentrated distribution
        extreme_beliefs = np.array([0.9999999, 1e-7, 1e-7, 1e-7])
        observations = np.array([1, 0, 1, 0])

        result = engine.calculate_epistemic_value("extreme_agent", extreme_beliefs, observations)

        # All values should be finite and within expected ranges
        assert np.isfinite(result.knowledge_entropy)
        assert np.isfinite(result.confidence_level)
        assert np.isfinite(result.certainty_measure)
        assert np.isfinite(result.epistemic_value)
        assert 0.0 <= result.confidence_level <= 1.0
        assert 0.0 <= result.certainty_measure <= 1.0
        assert 0.0 <= result.epistemic_value <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])

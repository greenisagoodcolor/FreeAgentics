"""
Comprehensive tests for Epistemic Value Calculation Engine.

Tests the epistemic value calculations for multi-agent networks including
information gain, knowledge propagation, collective intelligence metrics,
and network-level epistemic efficiency measurements.
"""

from agents.base.epistemic_value_engine import (
    CollectiveIntelligenceMetrics,
    EpistemicState,
    EpistemicValueCalculationEngine,
    EpistemicValueType,
    KnowledgePropagationEvent,
    epistemic_engine,
)
import os
import sys
from datetime import datetime

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import directly to avoid the full agents package


class TestEpistemicValueType:
    """Test EpistemicValueType enum."""

    def test_epistemic_value_type_values(self):
        """Test epistemic value type enum values."""
        assert EpistemicValueType.INFORMATION_GAIN.value == "information_gain"
        assert EpistemicValueType.KNOWLEDGE_ENTROPY.value == "knowledge_entropy"
        assert EpistemicValueType.BELIEF_DIVERGENCE.value == "belief_divergence"
        assert EpistemicValueType.COLLECTIVE_INTELLIGENCE.value == "collective_intelligence"
        assert EpistemicValueType.KNOWLEDGE_PROPAGATION.value == "knowledge_propagation"
        assert EpistemicValueType.EPISTEMIC_CONVERGENCE.value == "epistemic_convergence"

    def test_epistemic_value_type_count(self):
        """Test correct number of epistemic value types."""
        value_types = list(EpistemicValueType)
        assert len(value_types) == 6


class TestEpistemicState:
    """Test EpistemicState dataclass."""

    def test_epistemic_state_creation(self):
        """Test creating epistemic state with all fields."""
        belief_dist = np.array([0.2, 0.3, 0.5])

        state = EpistemicState(
            agent_id="agent_1",
            belief_distribution=belief_dist,
            knowledge_entropy=1.0,
            confidence_level=0.8,
            information_sources=["agent_1", "agent_2"],
            certainty_measure=0.7,
            epistemic_value=0.85,
        )

        assert state.agent_id == "agent_1"
        assert np.array_equal(state.belief_distribution, belief_dist)
        assert state.knowledge_entropy == 1.0
        assert state.confidence_level == 0.8
        assert state.information_sources == ["agent_1", "agent_2"]
        assert state.certainty_measure == 0.7
        assert state.epistemic_value == 0.85
        assert isinstance(state.timestamp, datetime)

    def test_epistemic_state_defaults(self):
        """Test default values for optional fields."""
        state = EpistemicState(
            agent_id="agent_1",
            belief_distribution=np.array([0.5, 0.5]),
            knowledge_entropy=0.69,
            confidence_level=0.5,
        )

        assert state.information_sources == []
        assert state.certainty_measure == 0.0
        assert state.epistemic_value == 0.0
        assert isinstance(state.timestamp, datetime)


class TestKnowledgePropagationEvent:
    """Test KnowledgePropagationEvent dataclass."""

    def test_propagation_event_creation(self):
        """Test creating propagation event with all fields."""
        belief_change = np.array([0.1, -0.05, -0.05])

        event = KnowledgePropagationEvent(
            source_agent="agent_1",
            target_agent="agent_2",
            information_transferred=0.8,
            epistemic_gain=0.3,
            propagation_efficiency=0.375,
            belief_change=belief_change,
        )

        assert event.source_agent == "agent_1"
        assert event.target_agent == "agent_2"
        assert event.information_transferred == 0.8
        assert event.epistemic_gain == 0.3
        assert event.propagation_efficiency == 0.375
        assert np.array_equal(event.belief_change, belief_change)
        assert isinstance(event.timestamp, datetime)

    def test_propagation_event_defaults(self):
        """Test default values for optional fields."""
        event = KnowledgePropagationEvent(
            source_agent="agent_1",
            target_agent="agent_2",
            information_transferred=0.5,
            epistemic_gain=0.2,
            propagation_efficiency=0.4,
        )

        assert event.belief_change is None
        assert isinstance(event.timestamp, datetime)


class TestCollectiveIntelligenceMetrics:
    """Test CollectiveIntelligenceMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating collective intelligence metrics."""
        metrics = CollectiveIntelligenceMetrics(
            network_entropy=2.5,
            information_diversity=0.8,
            consensus_level=0.6,
            knowledge_distribution=0.7,
            epistemic_efficiency=0.75,
            collective_accuracy=0.85,
            emergence_indicator=0.2,
            stability_measure=0.9,
        )

        assert metrics.network_entropy == 2.5
        assert metrics.information_diversity == 0.8
        assert metrics.consensus_level == 0.6
        assert metrics.knowledge_distribution == 0.7
        assert metrics.epistemic_efficiency == 0.75
        assert metrics.collective_accuracy == 0.85
        assert metrics.emergence_indicator == 0.2
        assert metrics.stability_measure == 0.9


class TestEpistemicValueCalculationEngine:
    """Test EpistemicValueCalculationEngine class."""

    def setup_method(self):
        """Set up test engine."""
        self.engine = EpistemicValueCalculationEngine()

    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.agent_states == {}
        assert self.engine.propagation_history == []
        assert self.engine.network_metrics_history == []

    def test_calculate_epistemic_value_basic(self):
        """Test basic epistemic value calculation."""
        agent_id = "agent_1"
        belief_dist = np.array([0.2, 0.3, 0.5])
        observations = np.array([1, 0, 0])

        state = self.engine.calculate_epistemic_value(
            agent_id, belief_dist, observations)

        assert state.agent_id == agent_id
        assert np.array_equal(state.belief_distribution, belief_dist)
        assert isinstance(state.knowledge_entropy, float)
        assert 0 <= state.confidence_level <= 1
        assert 0 <= state.certainty_measure <= 1
        assert 0 <= state.epistemic_value <= 1
        assert state.information_sources == [agent_id]

        # Check state was stored
        assert agent_id in self.engine.agent_states
        assert self.engine.agent_states[agent_id] == state

    def test_calculate_epistemic_value_with_prior(self):
        """Test epistemic value calculation with prior distribution."""
        agent_id = "agent_2"
        belief_dist = np.array([0.1, 0.2, 0.7])
        observations = np.array([0, 0, 1])
        prior_dist = np.array([0.33, 0.33, 0.34])

        state = self.engine.calculate_epistemic_value(
            agent_id, belief_dist, observations, prior_dist
        )

        assert state.agent_id == agent_id
        # Information gain should be positive (moved from uniform prior)
        assert state.epistemic_value > 0

    def test_calculate_knowledge_entropy(self):
        """Test knowledge entropy calculation."""
        # Uniform distribution has maximum entropy
        uniform_dist = np.array([0.25, 0.25, 0.25, 0.25])
        entropy_uniform = self.engine._calculate_knowledge_entropy(
            uniform_dist)

        # Peaked distribution has lower entropy
        peaked_dist = np.array([0.1, 0.1, 0.7, 0.1])
        entropy_peaked = self.engine._calculate_knowledge_entropy(peaked_dist)

        assert entropy_uniform > entropy_peaked
        assert entropy_uniform == pytest.approx(np.log(4), rel=1e-5)

    def test_calculate_information_gain(self):
        """Test information gain (KL divergence) calculation."""
        prior = np.array([0.5, 0.5])
        posterior = np.array([0.9, 0.1])

        info_gain = self.engine._calculate_information_gain(posterior, prior)

        assert info_gain > 0  # Should have gained information
        # Manual calculation: 0.9*log(0.9/0.5) + 0.1*log(0.1/0.5)
        expected = 0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)
        assert info_gain == pytest.approx(expected, rel=1e-5)

    def test_calculate_confidence_level(self):
        """Test confidence level calculation."""
        # High entropy = low confidence
        uniform_dist = np.array([0.25, 0.25, 0.25, 0.25])
        confidence_uniform = self.engine._calculate_confidence_level(
            uniform_dist)

        # Low entropy = high confidence
        peaked_dist = np.array([0.01, 0.01, 0.97, 0.01])
        confidence_peaked = self.engine._calculate_confidence_level(
            peaked_dist)

        assert 0 <= confidence_uniform <= 1
        assert 0 <= confidence_peaked <= 1
        assert confidence_peaked > confidence_uniform

    def test_calculate_certainty_measure(self):
        """Test certainty measure calculation."""
        # Uniform distribution has low certainty
        uniform_dist = np.array([0.25, 0.25, 0.25, 0.25])
        certainty_uniform = self.engine._calculate_certainty_measure(
            uniform_dist)

        # Concentrated distribution has high certainty
        concentrated_dist = np.array([0.01, 0.02, 0.03, 0.94])
        certainty_concentrated = self.engine._calculate_certainty_measure(
            concentrated_dist)

        assert 0 <= certainty_uniform <= 1
        assert 0 <= certainty_concentrated <= 1
        assert certainty_concentrated > certainty_uniform

    def test_calculate_knowledge_propagation_basic(self):
        """Test basic knowledge propagation between agents."""
        # Set up two agents
        self.engine.calculate_epistemic_value(
            "agent_1", np.array([0.7, 0.2, 0.1]), np.array([1, 0, 0])
        )
        self.engine.calculate_epistemic_value(
            "agent_2", np.array([0.33, 0.33, 0.34]), np.array([0, 1, 0])
        )

        # Share information
        shared_info = np.array([0.6, 0.3, 0.1])

        event = self.engine.calculate_knowledge_propagation(
            "agent_1", "agent_2", shared_info)

        assert event.source_agent == "agent_1"
        assert event.target_agent == "agent_2"
        assert event.information_transferred >= 0
        assert isinstance(event.epistemic_gain, float)
        assert 0 <= event.propagation_efficiency <= 1
        assert event.belief_change is not None

        # Check propagation history
        assert len(self.engine.propagation_history) == 1
        assert self.engine.propagation_history[0] == event

        # Check target agent was updated
        assert "agent_1" in self.engine.agent_states["agent_2"].information_sources

    def test_calculate_knowledge_propagation_invalid_agents(self):
        """Test propagation with non-existent agents."""
        with pytest.raises(ValueError) as excinfo:
            self.engine.calculate_knowledge_propagation(
                "unknown_1", "unknown_2", np.array([0.5, 0.5])
            )
        assert "Both agents must have existing epistemic states" in str(
            excinfo.value)

    def test_calculate_knowledge_propagation_models(self):
        """Test different propagation models."""
        # Set up agents
        self.engine.calculate_epistemic_value(
            "agent_1", np.array([0.8, 0.2]), np.array([1, 0]))
        self.engine.calculate_epistemic_value(
            "agent_2", np.array([0.5, 0.5]), np.array([0, 1]))

        shared_info = np.array([0.7, 0.3])

        # Test Bayesian update model
        event_bayes = self.engine.calculate_knowledge_propagation(
            "agent_1", "agent_2", shared_info, "bayesian_update"
        )

        # Reset agent 2
        self.engine.calculate_epistemic_value(
            "agent_2", np.array([0.5, 0.5]), np.array([0, 1]))

        # Test default (weighted average) model
        event_default = self.engine.calculate_knowledge_propagation(
            "agent_1", "agent_2", shared_info, "weighted_average"
        )

        # Both should produce valid events
        assert event_bayes.propagation_efficiency >= 0
        assert event_default.propagation_efficiency >= 0

    def test_calculate_collective_intelligence_metrics_insufficient_agents(
            self):
        """Test collective metrics with insufficient agents."""
        # No agents
        metrics = self.engine.calculate_collective_intelligence_metrics({})
        assert metrics.network_entropy == 0.0
        assert metrics.information_diversity == 0.0

        # Only one agent
        self.engine.calculate_epistemic_value(
            "agent_1", np.array([0.5, 0.5]), np.array([1, 0]))

        metrics = self.engine.calculate_collective_intelligence_metrics({
                                                                        "agent_1": []})
        assert metrics.network_entropy == 0.0
        assert metrics.information_diversity == 0.0

    def test_calculate_collective_intelligence_metrics_basic(self):
        """Test basic collective intelligence metrics calculation."""
        # Set up network with 3 agents
        self.engine.calculate_epistemic_value(
            "agent_1", np.array([0.7, 0.2, 0.1]), np.array([1, 0, 0])
        )
        self.engine.calculate_epistemic_value(
            "agent_2", np.array([0.2, 0.6, 0.2]), np.array([0, 1, 0])
        )
        self.engine.calculate_epistemic_value(
            "agent_3", np.array([0.3, 0.3, 0.4]), np.array([0, 0, 1])
        )

        network = {
            "agent_1": ["agent_2"],
            "agent_2": ["agent_1", "agent_3"],
            "agent_3": ["agent_2"],
        }

        metrics = self.engine.calculate_collective_intelligence_metrics(
            network)

        assert metrics.network_entropy > 0
        assert metrics.information_diversity > 0
        assert 0 <= metrics.consensus_level <= 1
        assert metrics.epistemic_efficiency > 0
        assert 0 <= metrics.collective_accuracy <= 1
        assert metrics.emergence_indicator >= 0
        assert 0 <= metrics.stability_measure <= 1

        # Check metrics history
        assert len(self.engine.network_metrics_history) == 1
        assert self.engine.network_metrics_history[0] == metrics

    def test_calculate_collective_intelligence_with_propagation(self):
        """Test collective metrics after knowledge propagation."""
        # Set up agents
        for i in range(4):
            beliefs = np.random.dirichlet([1, 1, 1])
            self.engine.calculate_epistemic_value(
                f"agent_{i}", beliefs, np.random.choice([0, 1], 3)
            )

        # Perform some propagations
        shared_info = np.array([0.4, 0.4, 0.2])
        self.engine.calculate_knowledge_propagation(
            "agent_0", "agent_1", shared_info)
        self.engine.calculate_knowledge_propagation(
            "agent_1", "agent_2", shared_info)

        network = {
            "agent_0": ["agent_1"],
            "agent_1": ["agent_0", "agent_2"],
            "agent_2": ["agent_1", "agent_3"],
            "agent_3": ["agent_2"],
        }

        metrics = self.engine.calculate_collective_intelligence_metrics(
            network)

        # With propagation history, knowledge distribution should be > 0
        assert metrics.knowledge_distribution > 0

    def test_information_transfer_calculation(self):
        """Test information transfer calculation."""
        source_beliefs = np.array([0.7, 0.2, 0.1])
        target_beliefs = np.array([0.3, 0.4, 0.3])
        shared_info = np.array([0.5, 0.3, 0.2])

        transfer = self.engine._calculate_information_transfer(
            source_beliefs, target_beliefs, shared_info
        )

        assert transfer >= 0
        # Should be dot product
        expected = np.sum(source_beliefs * shared_info)
        assert transfer == pytest.approx(expected)

    def test_update_beliefs_bayesian(self):
        """Test Bayesian belief update."""
        target_beliefs = np.array([0.4, 0.4, 0.2])
        shared_info = np.array([0.8, 0.1, 0.1])

        updated = self.engine._update_beliefs_with_shared_info(
            target_beliefs, shared_info, "bayesian_update"
        )

        # Should be normalized product
        assert np.allclose(np.sum(updated), 1.0)
        # First element should have increased (high in shared info)
        assert updated[0] > target_beliefs[0]

    def test_update_beliefs_weighted_average(self):
        """Test weighted average belief update."""
        target_beliefs = np.array([0.4, 0.4, 0.2])
        shared_info = np.array([0.8, 0.1, 0.1])

        updated = self.engine._update_beliefs_with_shared_info(
            target_beliefs, shared_info, "weighted_average"
        )

        assert np.allclose(np.sum(updated), 1.0)
        # Should be between original and shared info
        assert target_beliefs[0] < updated[0] < shared_info[0]

    def test_propagation_efficiency_calculation(self):
        """Test propagation efficiency calculation."""
        # Zero information transferred
        efficiency = self.engine._calculate_propagation_efficiency(0.0, 0.5)
        assert efficiency == 0.0

        # Normal case
        efficiency = self.engine._calculate_propagation_efficiency(2.0, 0.5)
        assert efficiency == 0.25  # 0.5 / 2.0

        # Clamped to [0, 1]
        efficiency = self.engine._calculate_propagation_efficiency(0.1, 0.5)
        assert efficiency == 1.0  # Clamped

    def test_network_entropy_calculation(self):
        """Test network entropy calculation."""
        # Empty network
        entropy = self.engine._calculate_network_entropy()
        assert entropy == 0.0

        # Add agents with different entropies
        self.engine.calculate_epistemic_value(
            "agent_1", np.array([0.9, 0.1]), np.array([1, 0])  # Low entropy
        )
        self.engine.calculate_epistemic_value(
            "agent_2", np.array([0.5, 0.5]), np.array([0, 1])  # High entropy
        )

        entropy = self.engine._calculate_network_entropy()
        assert entropy > 0
        # Should be average of individual entropies
        expected = (
            self.engine.agent_states["agent_1"].knowledge_entropy
            + self.engine.agent_states["agent_2"].knowledge_entropy
        ) / 2
        assert entropy == pytest.approx(expected)

    def test_information_diversity_calculation(self):
        """Test information diversity calculation."""
        # Similar beliefs = low diversity
        self.engine.calculate_epistemic_value(
            "agent_1", np.array([0.7, 0.2, 0.1]), np.array([1, 0, 0])
        )
        self.engine.calculate_epistemic_value(
            "agent_2", np.array([0.6, 0.3, 0.1]), np.array([1, 0, 0])
        )

        diversity_similar = self.engine._calculate_information_diversity()

        # Different beliefs = high diversity
        self.engine.calculate_epistemic_value(
            "agent_3", np.array([0.1, 0.1, 0.8]), np.array([0, 0, 1])
        )

        diversity_different = self.engine._calculate_information_diversity()

        assert diversity_similar >= 0
        assert diversity_different > diversity_similar

    def test_consensus_calculation(self):
        """Test consensus level calculation."""
        # High diversity = low consensus
        self.engine.calculate_epistemic_value(
            "agent_1", np.array([0.9, 0.1]), np.array([1, 0]))
        self.engine.calculate_epistemic_value(
            "agent_2", np.array([0.1, 0.9]), np.array([0, 1]))

        consensus = self.engine._calculate_consensus_level()
        assert 0 <= consensus <= 1
        assert consensus < 0.8  # Should be moderate due to opposite beliefs

    def test_emergence_indicator(self):
        """Test emergence indicator calculation."""
        # Set up agents
        for i in range(3):
            self.engine.calculate_epistemic_value(
                f"agent_{i}", np.array([0.4, 0.3, 0.3]), np.array([1, 0, 0])
            )

        # Calculate metrics to establish history
        self.engine.calculate_collective_intelligence_metrics({})

        emergence = self.engine._calculate_emergence_indicator()
        assert emergence >= 0
        # With similar agents, emergence should be low
        assert emergence < 0.5

    def test_network_stability(self):
        """Test network stability calculation."""
        # No history = stable
        stability = self.engine._calculate_network_stability()
        assert stability == 1.0

        # Add metrics history with varying consensus
        for consensus in [0.5, 0.6, 0.4, 0.7, 0.3]:
            metrics = CollectiveIntelligenceMetrics(
                network_entropy=1.0,
                information_diversity=0.5,
                consensus_level=consensus,
                knowledge_distribution=0.5,
                epistemic_efficiency=0.5,
                collective_accuracy=0.5,
                emergence_indicator=0.0,
                stability_measure=0.5,
            )
            self.engine.network_metrics_history.append(metrics)

        stability = self.engine._calculate_network_stability()
        assert 0 <= stability <= 1
        # With moderate variance in consensus, stability should still be high but
        # not perfect
        assert stability < 1.0

    def test_get_network_analytics(self):
        """Test network analytics generation."""
        # Empty network
        analytics = self.engine.get_network_analytics()
        assert analytics["agent_count"] == 0
        assert analytics["total_propagation_events"] == 0
        assert analytics["average_epistemic_value"] == 0.0

        # Add agents and propagation
        self.engine.calculate_epistemic_value(
            "agent_1", np.array([0.7, 0.3]), np.array([1, 0]))
        self.engine.calculate_epistemic_value(
            "agent_2", np.array([0.4, 0.6]), np.array([0, 1]))

        self.engine.calculate_knowledge_propagation(
            "agent_1", "agent_2", np.array([0.6, 0.4]))

        # Calculate collective metrics
        self.engine.calculate_collective_intelligence_metrics(
            {"agent_1": ["agent_2"], "agent_2": ["agent_1"]}
        )

        analytics = self.engine.get_network_analytics()

        assert analytics["agent_count"] == 2
        assert analytics["total_propagation_events"] == 1
        assert analytics["average_epistemic_value"] > 0
        assert analytics["network_diversity"] >= 0
        assert analytics["consensus_level"] >= 0
        assert analytics["emergence_indicator"] >= 0
        assert analytics["recent_propagation_efficiency"] >= 0
        assert len(analytics["top_epistemic_agents"]) == 2

        # Check top agents are sorted
        top_agents = analytics["top_epistemic_agents"]
        assert top_agents[0][1] >= top_agents[1][1]

    def test_global_epistemic_engine(self):
        """Test global epistemic engine instance."""
        assert isinstance(epistemic_engine, EpistemicValueCalculationEngine)
        assert epistemic_engine.agent_states == {}
        assert epistemic_engine.propagation_history == []
        assert epistemic_engine.network_metrics_history == []

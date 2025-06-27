."""
Epistemic Value Calculation Engine for Multi-Agent Networks

Implements mathematically rigorous epistemic value calculations using pymdp
for multi-agent knowledge propagation and collective intelligence metrics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EpistemicValueType(Enum):
    """Types of epistemic value calculations."""

    INFORMATION_GAIN = "information_gain"
    KNOWLEDGE_ENTROPY = "knowledge_entropy"
    BELIEF_DIVERGENCE = "belief_divergence"
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"
    KNOWLEDGE_PROPAGATION = "knowledge_propagation"
    EPISTEMIC_CONVERGENCE = "epistemic_convergence"


@dataclass
class EpistemicState:
    """Represents the epistemic state of an agent."""

    agent_id: str
    belief_distribution: np.ndarray
    knowledge_entropy: float
    confidence_level: float
    information_sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    certainty_measure: float = 0.0
    epistemic_value: float = 0.0


@dataclass
class KnowledgePropagationEvent:
    """Represents a knowledge propagation event between agents."""

    source_agent: str
    target_agent: str
    information_transferred: float
    epistemic_gain: float
    propagation_efficiency: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    belief_change: Optional[np.ndarray] = None


@dataclass
class CollectiveIntelligenceMetrics:
    """Metrics for collective intelligence assessment."""

    network_entropy: float
    information_diversity: float
    consensus_level: float
    knowledge_distribution: float
    epistemic_efficiency: float
    collective_accuracy: float
    emergence_indicator: float
    stability_measure: float


class EpistemicValueCalculationEngine:
    """
    Engine for calculating epistemic values in multi-agent networks.

    Uses pymdp-based calculations for mathematical correctness and implements
    sophisticated metrics for knowledge propagation and collective intelligence.
    """

    def __init__(self) -> None:
        """Initialize."""
        self.agent_states: Dict[str, EpistemicState] = {}
        self.propagation_history: List[KnowledgePropagationEvent] = []
        self.network_metrics_history: List[CollectiveIntelligenceMetrics] = []

    def calculate_epistemic_value(
        self,
        agent_id: str,
        belief_distribution: np.ndarray,
        observations: np.ndarray,
        prior_distribution: Optional[np.ndarray] = None,
    ) -> EpistemicState:
        """
        Calculate comprehensive epistemic value for an agent.

        Args:
            agent_id: Unique identifier for the agent
            belief_distribution: Current belief state (posterior)
            observations: Recent observations
            prior_distribution: Prior belief distribution

        Returns:
            EpistemicState with all calculated metrics
        """

        # Calculate knowledge entropy H[Q(s)]
        knowledge_entropy = (
            self._calculate_knowledge_entropy(belief_distribution))

        # Calculate information gain relative to prior
        if prior_distribution is not None:
            information_gain = self._calculate_information_gain(
                belief_distribution, prior_distribution
            )
        else:
            # Use uniform prior if not provided
            uniform_prior = (
                np.ones_like(belief_distribution) / len(belief_distribution))
            information_gain = (
                self._calculate_information_gain(belief_distribution, uniform_prior))

        # Calculate confidence level (inverse of uncertainty)
        confidence_level = (
            self._calculate_confidence_level(belief_distribution))

        # Calculate certainty measure
        certainty_measure = (
            self._calculate_certainty_measure(belief_distribution))

        # Calculate overall epistemic value
        epistemic_value = self._calculate_overall_epistemic_value(
            knowledge_entropy, information_gain, confidence_level,
                certainty_measure
        )

        # Create epistemic state
        epistemic_state = EpistemicState(
            agent_id=agent_id,
            belief_distribution=belief_distribution,
            knowledge_entropy=knowledge_entropy,
            confidence_level=confidence_level,
            certainty_measure=certainty_measure,
            epistemic_value=epistemic_value,
            information_sources= (
                [agent_id],  # Will be updated with network interactions)
        )

        # Store state
        self.agent_states[agent_id] = epistemic_state

        logger.debug(f"Calculated epistemic value for agent {agent_id}: {epistemic_value:.4f}")

        return epistemic_state

    def calculate_knowledge_propagation(
        self,
        source_agent_id: str,
        target_agent_id: str,
        shared_information: np.ndarray,
        propagation_model: str = "bayesian_update",
    ) -> KnowledgePropagationEvent:
        """
        Calculate knowledge propagation between two agents.

        Args:
            source_agent_id: Agent sharing knowledge
            target_agent_id: Agent receiving knowledge
            shared_information: Information being shared
            propagation_model: Model for information propagation

        Returns:
            KnowledgePropagationEvent with propagation metrics
        """

        if source_agent_id not in self.agent_states or target_agent_id not in self.agent_states:
            raise ValueError("Both agents must have existing epistemic states")

        source_state = self.agent_states[source_agent_id]
        target_state = self.agent_states[target_agent_id]

        # Calculate information transfer efficiency
        information_transferred = self._calculate_information_transfer(
            source_state.belief_distribution, target_state.belief_distribution,
                shared_information
        )

        # Calculate epistemic gain for target agent
        original_entropy = target_state.knowledge_entropy

        # Simulate belief update with shared information
        updated_beliefs = self._update_beliefs_with_shared_info(
            target_state.belief_distribution, shared_information,
                propagation_model
        )

        new_entropy = self._calculate_knowledge_entropy(updated_beliefs)
        epistemic_gain = (
            original_entropy - new_entropy  # Reduction in entropy = gain)

        # Calculate propagation efficiency
        propagation_efficiency = self._calculate_propagation_efficiency(
            information_transferred, epistemic_gain
        )

        # Calculate belief change magnitude
        belief_change = updated_beliefs - target_state.belief_distribution

        # Create propagation event
        propagation_event = KnowledgePropagationEvent(
            source_agent=source_agent_id,
            target_agent=target_agent_id,
            information_transferred=information_transferred,
            epistemic_gain=epistemic_gain,
            propagation_efficiency=propagation_efficiency,
            belief_change=belief_change,
        )

        # Update target agent's state
        target_state.belief_distribution = updated_beliefs
        target_state.knowledge_entropy = new_entropy
        target_state.confidence_level = (
            self._calculate_confidence_level(updated_beliefs))
        target_state.epistemic_value = self._calculate_overall_epistemic_value(
            new_entropy,
            epistemic_gain,
            target_state.confidence_level,
            target_state.certainty_measure,
        )
        target_state.information_sources.append(source_agent_id)

        # Store propagation event
        self.propagation_history.append(propagation_event)

        logger.info(
            f"Knowledge propagation from {source_agent_id} to {target_agent_id}: "
            f"gain= (
                {epistemic_gain:.4f}, efficiency={propagation_efficiency:.4f}")
        )

        return propagation_event

    def calculate_collective_intelligence_metrics(
        self, agent_network: Dict[str, List[str]]  # agent_id -> list of connected agents
    ) -> CollectiveIntelligenceMetrics:
        """
        Calculate collective intelligence metrics for the agent network.

        Args:
            agent_network: Network topology (adjacency list format)

        Returns:
            CollectiveIntelligenceMetrics with network-level intelligence measures
        """

        if len(self.agent_states) < 2:
            # Cannot calculate collective metrics with fewer than 2 agents
            return CollectiveIntelligenceMetrics(
                network_entropy=0.0,
                information_diversity=0.0,
                consensus_level=0.0,
                knowledge_distribution=0.0,
                epistemic_efficiency=0.0,
                collective_accuracy=0.0,
                emergence_indicator=0.0,
                stability_measure=0.0,
            )

        # Calculate network-level entropy
        network_entropy = self._calculate_network_entropy()

        # Calculate information diversity across agents
        information_diversity = self._calculate_information_diversity()

        # Calculate consensus level (agreement between agents)
        consensus_level = self._calculate_consensus_level()

        # Calculate knowledge distribution efficiency
        knowledge_distribution = (
            self._calculate_knowledge_distribution_efficiency(agent_network))

        # Calculate epistemic efficiency (information flow efficiency)
        epistemic_efficiency = self._calculate_epistemic_efficiency()

        # Calculate collective accuracy (if ground truth available)
        collective_accuracy = self._calculate_collective_accuracy()

        # Calculate emergence indicator (non-linear intelligence gains)
        emergence_indicator = self._calculate_emergence_indicator()

        # Calculate stability measure
        stability_measure = self._calculate_network_stability()

        metrics = CollectiveIntelligenceMetrics(
            network_entropy=network_entropy,
            information_diversity=information_diversity,
            consensus_level=consensus_level,
            knowledge_distribution=knowledge_distribution,
            epistemic_efficiency=epistemic_efficiency,
            collective_accuracy=collective_accuracy,
            emergence_indicator=emergence_indicator,
            stability_measure=stability_measure,
        )

        # Store metrics
        self.network_metrics_history.append(metrics)

        logger.info(
            f"Calculated collective intelligence metrics: "
            f"entropy= (
                {network_entropy:.4f}, diversity={information_diversity:.4f}, ")
            f"consensus={consensus_level:.4f}"
        )

        return metrics

    # Private calculation methods

    def _calculate_knowledge_entropy(self, belief_distribution: np.ndarray) -> float:
        """Calculate Shannon entropy of belief distribution."""
        # Ensure no zero probabilities for numerical stability
        safe_beliefs = np.maximum(belief_distribution, 1e-16)
        entropy = -np.sum(safe_beliefs * np.log(safe_beliefs))
        return float(entropy)

    def _calculate_information_gain(self, posterior: np.ndarray,
        prior: np.ndarray) -> float:
        """Calculate information gain (KL divergence from prior to
        posterior)."""
        safe_posterior = np.maximum(posterior, 1e-16)
        safe_prior = np.maximum(prior, 1e-16)
        kl_divergence = (
            np.sum(safe_posterior * np.log(safe_posterior / safe_prior)))
        return float(kl_divergence)

    def _calculate_confidence_level(self, belief_distribution: np.ndarray) -> float:
        """Calculate confidence level as inverse of normalized entropy."""
        max_entropy = np.log(len(belief_distribution))
        current_entropy = (
            self._calculate_knowledge_entropy(belief_distribution))
        confidence = 1.0 - (current_entropy / max_entropy)
        return float(np.clip(confidence, 0.0, 1.0))

    def _calculate_certainty_measure(self, belief_distribution: np.ndarray) -> float:
        """Calculate certainty as concentration of probability mass."""
        # Use Gini coefficient-like measure
        sorted_beliefs = np.sort(belief_distribution)
        n = len(sorted_beliefs)
        index = np.arange(1, n + 1)
        certainty = (
            (2 * np.sum(index * sorted_beliefs)) / (n * np.sum(sorted_beliefs)) - ()
            n + 1
        ) / n
        return float(np.clip(certainty, 0.0, 1.0))

    def _calculate_overall_epistemic_value(
        self, entropy: float, information_gain: float, confidence: float,
            certainty: float
    ) -> float:
        """Calculate overall epistemic value as weighted combination of
        metrics."""
        # Normalize entropy (lower entropy = higher value)
        max_entropy = 10.0  # Assumed maximum entropy for normalization
        normalized_entropy = 1.0 - min(entropy / max_entropy, 1.0)

        # Normalize information gain
        max_info_gain = 5.0  # Assumed maximum information gain
        normalized_info_gain = min(information_gain / max_info_gain, 1.0)

        # Weighted combination
        epistemic_value = (
            0.3 * normalized_entropy  # Knowledge organization
            + 0.3 * normalized_info_gain  # Learning progress
            + 0.2 * confidence  # Confidence level
            + 0.2 * certainty  # Certainty measure
        )

        return float(np.clip(epistemic_value, 0.0, 1.0))

    def _calculate_information_transfer(
        self, source_beliefs: np.ndarray, target_beliefs: np.ndarray,
            shared_info: np.ndarray
    ) -> float:
        """Calculate amount of information transferred."""
        # Simplified: mutual information between source and shared information
        # In practice, this would use more sophisticated information theory
        overlap: float = float(np.sum(source_beliefs * shared_info))
        return overlap

    def _update_beliefs_with_shared_info(
        self, target_beliefs: np.ndarray, shared_info: np.ndarray, model: str
    ) -> np.ndarray:
        """Update target beliefs with shared information."""
        if model == "bayesian_update":
            # Simple Bayesian update
            updated: np.ndarray = target_beliefs * shared_info
            updated = updated / np.sum(updated)  # Normalize
            return updated
        else:
            # Default: weighted average
            alpha = 0.3  # Learning rate
            updated = (1 - alpha) * target_beliefs + alpha * shared_info
            return updated / np.sum(updated)

    def _calculate_propagation_efficiency(
        self, info_transferred: float, epistemic_gain: float
    ) -> float:
        """Calculate efficiency of knowledge propagation."""
        if info_transferred == 0:
            return 0.0
        efficiency = epistemic_gain / info_transferred
        return float(np.clip(efficiency, 0.0, 1.0))

    def _calculate_network_entropy(self) -> float:
        """Calculate entropy across the entire network."""
        if not self.agent_states:
            return 0.0

        # Average entropy across all agents
        entropies = (
            [state.knowledge_entropy for state in self.agent_states.values()])
        return float(np.mean(entropies))

    def _calculate_information_diversity(self) -> float:
        """Calculate diversity of information across agents."""
        if len(self.agent_states) < 2:
            return 0.0

        # Calculate pairwise KL divergences
        agent_ids = list(self.agent_states.keys())
        divergences = []

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                belief_i = self.agent_states[agent_ids[i]].belief_distribution
                belief_j = self.agent_states[agent_ids[j]].belief_distribution

                # Symmetric KL divergence
                kl_ij = self._calculate_information_gain(belief_i, belief_j)
                kl_ji = self._calculate_information_gain(belief_j, belief_i)
                symmetric_kl = (kl_ij + kl_ji) / 2.0

                divergences.append(symmetric_kl)

        return float(np.mean(divergences))

    def _calculate_consensus_level(self) -> float:
        """Calculate level of consensus (agreement) between agents."""
        diversity = self._calculate_information_diversity()
        # Consensus is inverse of diversity
        max_diversity = 5.0  # Assumed maximum diversity
        consensus = 1.0 - min(diversity / max_diversity, 1.0)
        return float(consensus)

    def _calculate_knowledge_distribution_efficiency(self, network: Dict[str,
        List[str]]) -> float:
        """Calculate efficiency of knowledge distribution in network."""
        if not self.propagation_history:
            return 0.0

        # Average propagation efficiency
        efficiencies = (
            [event.propagation_efficiency for event in self.propagation_history])
        return float(np.mean(efficiencies))

    def _calculate_epistemic_efficiency(self) -> float:
        """Calculate overall epistemic efficiency of the network."""
        if not self.agent_states:
            return 0.0

        # Average epistemic value across agents
        epistemic_values = (
            [state.epistemic_value for state in self.agent_states.values()])
        return float(np.mean(epistemic_values))

    def _calculate_collective_accuracy(self) -> float:
        """Calculate collective accuracy (placeholder for ground truth
        comparison)."""
        # This would require ground truth data in practice
        # For now, return average confidence as proxy
        if not self.agent_states:
            return 0.0

        confidences = (
            [state.confidence_level for state in self.agent_states.values()])
        return float(np.mean(confidences))

    def _calculate_emergence_indicator(self) -> float:
        """Calculate indicator of emergent intelligence."""
        if len(self.network_metrics_history) < 2:
            return 0.0

        # Compare current collective performance to individual sum
        current_efficiency = self._calculate_epistemic_efficiency()
        individual_sum = (
            sum(state.epistemic_value for state in self.agent_states.values()))

        if individual_sum == 0:
            return 0.0

        # Emergence = collective performance / sum of individual performances
        emergence = (
            current_efficiency * len(self.agent_states) / individual_sum)

        # Emergence > 1.0 indicates synergistic effects
        return float(max(0.0, emergence - 1.0))

    def _calculate_network_stability(self) -> float:
        """Calculate stability of the network's epistemic state."""
        if len(self.network_metrics_history) < 2:
            return 1.0  # Assume stable if no history

        # Calculate variance in recent metrics
        recent_metrics = (
            self.network_metrics_history[-5:]  # Last 5 measurements)
        consensus_variance = (
            np.var([m.consensus_level for m in recent_metrics]))

        # Stability is inverse of variance
        stability = 1.0 / (1.0 + consensus_variance)
        return float(stability)

    def get_network_analytics(self) -> Dict[str, Any]:
        """Get comprehensive network analytics for dashboard display."""

        latest_metrics = (
            self.network_metrics_history[-1] if self.network_metrics_history else None)

        return {
            "agent_count": len(self.agent_states),
            "total_propagation_events": len(self.propagation_history),
            "average_epistemic_value": (
                np.mean([s.epistemic_value for s in self.agent_states.values()])
                if self.agent_states
                else 0.0
            ),
            "network_diversity": latest_metrics.information_diversity if latest_metrics else 0.0,
            "consensus_level": latest_metrics.consensus_level if latest_metrics else 0.0,
            "emergence_indicator": latest_metrics.emergence_indicator if latest_metrics else 0.0,
            "recent_propagation_efficiency": (
                np.mean([e.propagation_efficiency for e in self.propagation_history[-10:]])
                if self.propagation_history
                else 0.0
            ),
            "top_epistemic_agents": sorted(
                [
                    (agent_id, state.epistemic_value)
                    for agent_id, state in self.agent_states.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }


# Global epistemic value engine instance
epistemic_engine = EpistemicValueCalculationEngine()

__all__ = [
    "EpistemicValueCalculationEngine",
    "EpistemicState",
    "KnowledgePropagationEvent",
    "CollectiveIntelligenceMetrics",
    "EpistemicValueType",
    "epistemic_engine",
]

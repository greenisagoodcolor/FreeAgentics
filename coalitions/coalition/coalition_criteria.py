import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

"""
Coalition Formation Criteria
Defines the parameters and rules that determine when and how agents form coalitions.
"""
logger = logging.getLogger(__name__)


class FormationTrigger(Enum):
    """Triggers that can initiate coalition formation."""

    OPPORTUNITY_DETECTED = "opportunity_detected"
    RESOURCE_SHORTAGE = "resource_shortage"
    EXTERNAL_THREAT = "external_threat"
    CAPABILITY_GAP = "capability_gap"
    EXPLICIT_REQUEST = "explicit_request"
    SCHEDULED = "scheduled"


class DissolutionCondition(Enum):
    """Conditions that can trigger coalition dissolution."""

    GOAL_ACHIEVED = "goal_achieved"
    TIMEOUT = "timeout"
    MEMBER_DEPARTURE = "member_departure"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    CONSENSUS_LOST = "consensus_lost"
    RESOURCE_DEPLETION = "resource_depletion"


@dataclass
class CompatibilityMetric:
    """Metric for measuring agent compatibility."""

    name: str
    weight: float = 1.0
    threshold: float = 0.5
    bidirectional: bool = True

    def calculate(self, agent1_profile: Dict[str, Any], agent2_profile: Dict[str, Any]) -> float:
        """
        Calculate compatibility score between two agents.
        Returns:
            Score between 0 and 1
        """
        # This is a base implementation - specific metrics override this
        return 0.5


class GoalAlignmentMetric(CompatibilityMetric):
    """Measures how well agent goals align."""

    def __init__(self) -> None:
        super().__init__(
            name="goal_alignment",
            weight=2.0,  # High importance
            threshold=0.6,
        )

    def calculate(self, agent1_profile: Dict[str, Any], agent2_profile: Dict[str, Any]) -> float:
        """Calculate goal alignment score."""
        goals1 = set(agent1_profile.get("goals", []))
        goals2 = set(agent2_profile.get("goals", []))
        if not goals1 or not goals2:
            return 0.0
        # Jaccard similarity
        intersection = goals1.intersection(goals2)
        union = goals1.union(goals2)
        return len(intersection) / len(union) if union else 0.0


class CapabilityComplementarityMetric(CompatibilityMetric):
    """Measures how well agent capabilities complement each other."""

    def __init__(self) -> None:
        super().__init__(
            name="capability_complementarity",
            weight=1.5,
            threshold=0.4,
            bidirectional=False,  # A complements B doesn't mean B complements A
        )

    def calculate(self, agent1_profile: Dict[str, Any], agent2_profile: Dict[str, Any]) -> float:
        """Calculate capability complementarity score."""
        caps1 = set(agent1_profile.get("capabilities", []))
        caps2 = set(agent2_profile.get("capabilities", []))
        needs1 = set(agent1_profile.get("capability_gaps", []))
        needs2 = set(agent2_profile.get("capability_gaps", []))
        # How well does agent2 fill agent1's gaps?
        filled_gaps = needs1.intersection(caps2)
        complementarity = len(filled_gaps) / len(needs1) if needs1 else 0.0
        # Bonus for non-overlapping capabilities
        unique_caps = len(caps1.symmetric_difference(caps2)) / (len(caps1) + len(caps2))
        return 0.7 * complementarity + 0.3 * unique_caps


class ResourceBalanceMetric(CompatibilityMetric):
    """Measures resource balance between agents."""

    def __init__(self) -> None:
        super().__init__(name="resource_balance", weight=1.0, threshold=0.3)

    def calculate(self, agent1_profile: Dict[str, Any], agent2_profile: Dict[str, Any]) -> float:
        """Calculate resource balance score."""
        resources1 = agent1_profile.get("resources", {})
        resources2 = agent2_profile.get("resources", {})
        if not resources1 or not resources2:
            return 0.5
        # Compare resource levels
        balance_scores = []
        for resource_type in set(resources1.keys()).union(resources2.keys()):
            r1 = resources1.get(resource_type, 0)
            r2 = resources2.get(resource_type, 0)
            if r1 + r2 > 0:
                # Higher score for complementary levels (one high, one low)
                balance = 1 - abs(r1 - r2) / (r1 + r2)
                complementarity = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 0
                balance_scores.append(0.3 * balance + 0.7 * (1 - complementarity))
        return np.mean(balance_scores) if balance_scores else 0.5


class TrustMetric(CompatibilityMetric):
    """Measures trust level between agents based on past interactions."""

    def __init__(self) -> None:
        super().__init__(name="trust", weight=1.8, threshold=0.5)

    def calculate(self, agent1_profile: Dict[str, Any], agent2_profile: Dict[str, Any]) -> float:
        """Calculate trust score based on interaction history."""
        # Get interaction history
        history = agent1_profile.get("interaction_history", {})
        agent2_id = agent2_profile.get("agent_id")
        if not agent2_id or agent2_id not in history:
            # No history - neutral trust
            return 0.5
        interactions = history[agent2_id]
        # Calculate trust based on past interactions
        positive = interactions.get("positive_interactions", 0)
        negative = interactions.get("negative_interactions", 0)
        total = positive + negative
        if total == 0:
            return 0.5
        # Basic trust calculation with recency weighting
        base_trust = positive / total
        # Recent interactions matter more
        recent_positive = interactions.get("recent_positive", 0)
        recent_negative = interactions.get("recent_negative", 0)
        recent_total = recent_positive + recent_negative
        if recent_total > 0:
            recent_trust = recent_positive / recent_total
            # Weight recent interactions more heavily
            return 0.3 * base_trust + 0.7 * recent_trust
        return base_trust


@dataclass
class CoalitionFormationCriteria:
    """
    Comprehensive criteria for coalition formation.
    Defines all parameters and rules for when and how coalitions form.
    """

    # Size constraints
    min_members: int = 2
    max_members: int = 10
    optimal_size: int = 4
    # Compatibility requirements
    compatibility_metrics: List[CompatibilityMetric] = field(default_factory=list)
    min_compatibility_score: float = 0.5
    # Formation triggers
    formation_triggers: Set[FormationTrigger] = field(
        default_factory=lambda: {FormationTrigger.OPPORTUNITY_DETECTED}
    )
    # Dissolution conditions
    dissolution_conditions: Set[DissolutionCondition] = field(
        default_factory=lambda: {
            DissolutionCondition.GOAL_ACHIEVED,
            DissolutionCondition.TIMEOUT,
        }
    )
    # Time constraints
    max_formation_time: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    min_coalition_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    max_coalition_duration: timedelta = field(default_factory=lambda: timedelta(days=7))
    # Performance thresholds
    min_performance_score: float = 0.3
    performance_evaluation_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))
    # Resource requirements
    min_combined_resources: Dict[str, float] = field(default_factory=dict)
    # Voting and consensus
    consensus_threshold: float = 0.66  # 2/3 majority
    min_participation_rate: float = 0.8  # 80% must vote

    def __post_init__(self):
        """Initialize default compatibility metrics if none provided."""
        if not self.compatibility_metrics:
            self.compatibility_metrics = [
                GoalAlignmentMetric(),
                CapabilityComplementarityMetric(),
                ResourceBalanceMetric(),
                TrustMetric(),
            ]

    def calculate_compatibility(
        self, agent1_profile: Dict[str, Any], agent2_profile: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall compatibility between two agents.
        Returns:
            Tuple of (overall_score, individual_metric_scores)
        """
        scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        for metric in self.compatibility_metrics:
            score = metric.calculate(agent1_profile, agent2_profile)
            scores[metric.name] = score
            # Only count if above threshold
            if score >= metric.threshold:
                weighted_sum += score * metric.weight
                total_weight += metric.weight
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return overall_score, scores

    def is_compatible_pair(
        self, agent1_profile: Dict[str, Any], agent2_profile: Dict[str, Any]
    ) -> bool:
        """Check if two agents are compatible for coalition formation."""
        overall_score, _ = self.calculate_compatibility(agent1_profile, agent2_profile)
        return overall_score >= self.min_compatibility_score

    def evaluate_coalition_viability(
        self, member_profiles: List[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if a proposed coalition meets formation criteria.
        Returns:
            Tuple of (is_viable, evaluation_details)
        """
        evaluation = {
            "member_count": len(member_profiles),
            "size_valid": self.min_members <= len(member_profiles) <= self.max_members,
            "compatibility_scores": {},
            "resource_totals": {},
            "capability_coverage": set(),
            "issues": [],
        }
        # Check size constraints
        if not evaluation["size_valid"]:
            evaluation["issues"].append(
                f"Size {len(member_profiles)} outside range [{self.min_members}, {self.max_members}]"
            )
        # Calculate pairwise compatibility
        compatibility_sum = 0.0
        compatibility_count = 0
        for i in range(len(member_profiles)):
            for j in range(i + 1, len(member_profiles)):
                score, _ = self.calculate_compatibility(member_profiles[i], member_profiles[j])
                pair_key = f"{member_profiles[i]['agent_id']}-{member_profiles[j]['agent_id']}"
                evaluation["compatibility_scores"][pair_key] = score
                compatibility_sum += score
                compatibility_count += 1
                if score < self.min_compatibility_score:
                    evaluation["issues"].append(
                        f"Low compatibility ({score:.2f}) between {pair_key}"
                    )
        # Average compatibility
        avg_compatibility = (
            compatibility_sum / compatibility_count if compatibility_count > 0 else 0
        )
        evaluation["average_compatibility"] = avg_compatibility
        # Aggregate resources
        for profile in member_profiles:
            for resource, amount in profile.get("resources", {}).items():
                evaluation["resource_totals"][resource] = (
                    evaluation["resource_totals"].get(resource, 0) + amount
                )
        # Check resource requirements
        for resource, min_amount in self.min_combined_resources.items():
            if evaluation["resource_totals"].get(resource, 0) < min_amount:
                evaluation["issues"].append(
                    f"Insufficient {resource}: {evaluation['resource_totals'].get(
                        resource,
                        0
                    )} < {min_amount}"
                )
        # Aggregate capabilities
        for profile in member_profiles:
            evaluation["capability_coverage"].update(profile.get("capabilities", []))
        # Overall viability
        is_viable = (
            evaluation["size_valid"]
            and avg_compatibility >= self.min_compatibility_score
            and len(evaluation["issues"]) == 0
        )
        evaluation["is_viable"] = is_viable
        return is_viable, evaluation

    def check_dissolution_conditions(
        self, coalition_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[DissolutionCondition]]:
        """
        Check if any dissolution conditions are met.
        Returns:
            Tuple of (should_dissolve, condition_met)
        """
        # Check timeout
        if DissolutionCondition.TIMEOUT in self.dissolution_conditions:
            duration = datetime.utcnow() - coalition_state.get("start_time", datetime.utcnow())
            if duration > self.max_coalition_duration:
                return True, DissolutionCondition.TIMEOUT
        # Check goal achievement
        if DissolutionCondition.GOAL_ACHIEVED in self.dissolution_conditions:
            if coalition_state.get("goals_achieved", False):
                return True, DissolutionCondition.GOAL_ACHIEVED
        # Check performance
        if DissolutionCondition.PERFORMANCE_THRESHOLD in self.dissolution_conditions:
            if coalition_state.get("performance_score", 1.0) < self.min_performance_score:
                return True, DissolutionCondition.PERFORMANCE_THRESHOLD
        # Check member count
        if DissolutionCondition.MEMBER_DEPARTURE in self.dissolution_conditions:
            if coalition_state.get("member_count", 0) < self.min_members:
                return True, DissolutionCondition.MEMBER_DEPARTURE
        # Check consensus
        if DissolutionCondition.CONSENSUS_LOST in self.dissolution_conditions:
            if coalition_state.get("consensus_score", 1.0) < self.consensus_threshold:
                return True, DissolutionCondition.CONSENSUS_LOST
        return False, None

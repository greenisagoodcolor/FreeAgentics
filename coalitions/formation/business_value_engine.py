"""
Business Value Calculation Engine for Coalition Formation

This module implements transparent calculation of coalition business value metrics
including synergy, risk reduction, market positioning, and sustainability.
All calculations are documented for business intelligence review and
    investor readiness.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..coalition.coalition_models import Coalition
from .coalition_formation_algorithms import AgentProfile, FormationResult

logger = logging.getLogger(__name__)


class BusinessMetricType(Enum):
    """Types of business value metrics calculated"""

    SYNERGY = "synergy"
    RISK_REDUCTION = "risk_reduction"
    MARKET_POSITIONING = "market_positioning"
    SUSTAINABILITY = "sustainability"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    INNOVATION_POTENTIAL = "innovation_potential"


@dataclass
class BusinessValueMetrics:
    """Comprehensive business value metrics for a coalition"""

    synergy_score: float = 0.0
    risk_reduction: float = 0.0
    market_positioning: float = 0.0
    sustainability_score: float = 0.0
    operational_efficiency: float = 0.0
    innovation_potential: float = 0.0
    total_value: float = 0.0
    confidence_level: float = 0.0
    calculation_timestamp: datetime = field(default_factory=datetime.utcnow)
    methodology_notes: Dict[str, str] = field(default_factory=dict)


class BusinessValueCalculationEngine:
    """
    Engine for calculating comprehensive business value metrics for coalitions.

    All calculations are transparent, documented, and suitable for investor presentations.
    Uses existing coalition models and follows ADR-006 business logic patterns.
    """

    def __init__(self) -> None:
        """Initialize"""
        self.metric_weights = {
            BusinessMetricType.SYNERGY: 0.25,
            BusinessMetricType.RISK_REDUCTION: 0.20,
            BusinessMetricType.MARKET_POSITIONING: 0.20,
            BusinessMetricType.SUSTAINABILITY: 0.15,
            BusinessMetricType.OPERATIONAL_EFFICIENCY: 0.10,
            BusinessMetricType.INNOVATION_POTENTIAL: 0.10,
        }
        self.calculation_history: List[BusinessValueMetrics] = []

    def calculate_business_value(
        self,
        coalition: Coalition,
        formation_result: FormationResult,
        agent_profiles: List[AgentProfile],
        market_context: Optional[Dict[str, Any]] = None,
    ) -> BusinessValueMetrics:
        """
        Calculate comprehensive business value for a coalition.

        Args:
            coalition: The formed coalition
            formation_result: Result from formation algorithm
            agent_profiles: Profiles of coalition members
            market_context: Optional market data for positioning calculations

        Returns:
            BusinessValueMetrics with all calculated values
        """
        metrics = BusinessValueMetrics()

        try:
            # Calculate individual metrics
            metrics.synergy_score = self._calculate_synergy(
                coalition, formation_result, agent_profiles
            )
            metrics.methodology_notes["synergy"] = (
                "Coalition value exceeds sum of individual agent values through "
                "capability complementarity")

            metrics.risk_reduction = self._calculate_risk_reduction(
                coalition, agent_profiles)
            metrics.methodology_notes["risk_reduction"] = (
                "Diversification across capabilities and resources reduces operational risk"
            )

            metrics.market_positioning = self._calculate_market_positioning(
                coalition, formation_result, market_context
            )
            metrics.methodology_notes["market_positioning"] = (
                "Strategic position based on formation strategy and market fit"
            )

            metrics.sustainability_score = self._calculate_sustainability(
                coalition, formation_result, agent_profiles
            )
            metrics.methodology_notes["sustainability"] = (
                "Long-term viability based on resource efficiency and formation quality"
            )

            metrics.operational_efficiency = self._calculate_operational_efficiency(
                coalition, agent_profiles)
            metrics.methodology_notes["operational_efficiency"] = (
                "Resource utilization and capability optimization"
            )

            metrics.innovation_potential = self._calculate_innovation_potential(
                coalition, agent_profiles)
            metrics.methodology_notes["innovation_potential"] = (
                "Capability diversity and novel combinations"
            )

            # Calculate total value as weighted sum
            metrics.total_value = self._calculate_total_value(metrics)

            # Calculate confidence level
            metrics.confidence_level = self._calculate_confidence(
                formation_result, agent_profiles)

            # Store in history
            self.calculation_history.append(metrics)

            logger.info(
                f"Calculated business value for coalition {
                    coalition.coalition_id}: " f"total={
                    metrics.total_value:.3f}, confidence={
                    metrics.confidence_level:.3f}")

        except Exception as e:
            logger.error(f"Error calculating business value: {e}")
            metrics = BusinessValueMetrics()  # Return zeros on error

        return metrics

    def _calculate_synergy(
        self,
        coalition: Coalition,
        formation_result: FormationResult,
        agent_profiles: List[AgentProfile],
    ) -> float:
        """
        Calculate synergy score: Coalition value > sum of individual values.

        Methodology:
        1. Estimate individual agent values based on capabilities and resources
        2. Estimate coalition value from formation score and complementarity
        3. Calculate synergy as (coalition_value - sum_individual) / sum_individual
        """
        if not agent_profiles:
            return 0.0

        # Calculate sum of individual agent values
        individual_values = []
        for profile in agent_profiles:
            capability_value = len(profile.capabilities) * 2.0
            resource_value = sum(profile.resources.values()) * \
                0.1 if profile.resources else 0.0
            reliability_value = profile.reliability_score * 5.0
            individual_value = capability_value + resource_value + reliability_value
            individual_values.append(individual_value)

        sum_individual = sum(individual_values)

        # Estimate coalition value
        # Base value from formation score
        coalition_base_value = formation_result.score * \
            10.0 if formation_result.success else 0.0

        # Capability complementarity bonus
        all_capabilities = set()
        for profile in agent_profiles:
            all_capabilities.update(profile.capabilities)

        unique_capabilities = len(all_capabilities)
        total_individual_capabilities = sum(
            len(p.capabilities) for p in agent_profiles)
        complementarity_ratio = unique_capabilities / \
            max(1, total_individual_capabilities)
        complementarity_bonus = complementarity_ratio * 20.0

        # Resource diversity bonus
        all_resource_types = set()
        for profile in agent_profiles:
            all_resource_types.update(profile.resources.keys())

        resource_diversity_bonus = len(all_resource_types) * 3.0

        coalition_value = coalition_base_value + \
            complementarity_bonus + resource_diversity_bonus

        # Calculate synergy
        if sum_individual > 0:
            synergy = max(
                0.0, (coalition_value - sum_individual) / sum_individual)
            return min(1.0, synergy)  # Cap at 100% synergy
        else:
            return 0.0

    def _calculate_risk_reduction(
        self, coalition: Coalition, agent_profiles: List[AgentProfile]
    ) -> float:
        """
        Calculate risk reduction through diversification.

        Methodology:
        1. Measure capability diversification (Herfindahl-Hirschman Index)
        2. Measure resource diversification
        3. Factor in reliability distribution
        4. Higher diversification = lower risk = higher score
        """
        if not agent_profiles:
            return 0.0

        # Capability diversification using Herfindahl-Hirschman Index
        capability_counts = {}
        total_capabilities = 0

        for profile in agent_profiles:
            for capability in profile.capabilities:
                capability_counts[capability] = capability_counts.get(
                    capability, 0) + 1
                total_capabilities += 1

        if total_capabilities > 0:
            # Calculate HHI (lower = more diversified)
            hhi = sum((count / total_capabilities) **
                      2 for count in capability_counts.values())
            capability_diversity = 1.0 - hhi  # Convert to diversity score
        else:
            capability_diversity = 0.0

        # Resource diversification
        resource_types = set()
        for profile in agent_profiles:
            resource_types.update(profile.resources.keys())

        max_possible_resource_types = 10  # Assume max 10 resource types
        resource_diversity = len(resource_types) / max_possible_resource_types

        # Reliability distribution (prefer mix of high/medium reliability)
        reliabilities = [p.reliability_score for p in agent_profiles]
        if reliabilities:
            # Calculate standard deviation (some variance is good for risk
            # management)
            mean_reliability = sum(reliabilities) / len(reliabilities)
            variance = sum((r - mean_reliability) **
                           2 for r in reliabilities) / len(reliabilities)
            reliability_diversity = min(
                1.0, variance * 4.0)  # Scale appropriately
        else:
            reliability_diversity = 0.0

        # Combine diversification measures
        risk_reduction = (
            capability_diversity *
            0.5 +
            resource_diversity *
            0.3 +
            reliability_diversity *
            0.2)

        return min(1.0, risk_reduction)

    def _calculate_market_positioning(
        self,
        coalition: Coalition,
        formation_result: FormationResult,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate market positioning strength.

        Methodology:
        1. Base score from formation strategy effectiveness
        2. Formation speed bonus (faster = more agile)
        3. Market context alignment (if provided)
        4. Coalition size optimization
        """
        # Base score from formation strategy
        strategy_scores = {
            "active_inference": 0.9,  # Most advanced
            "capability_based": 0.8,
            "resource_optimization": 0.7,
            "preference_matching": 0.6,
            "stability_maximization": 0.7,
            "business_opportunity": 1.0,
        }

        strategy_name = (
            formation_result.strategy_used.value if formation_result.strategy_used else "unknown")
        base_score = strategy_scores.get(strategy_name, 0.5)

        # Formation speed bonus (agility indicator)
        speed_bonus = 0.0
        if formation_result.formation_time > 0:
            # Faster formation = higher agility = better positioning
            # Scale: under 1 second = max bonus, over 10 seconds = no bonus
            speed_factor = max(
                0.0, (10.0 - formation_result.formation_time) / 10.0)
            speed_bonus = speed_factor * 0.2

        # Market context alignment
        market_bonus = 0.0
        if market_context:
            # This would integrate with actual market data
            # For now, provide a placeholder
            market_readiness = market_context.get("readiness_score", 0.5)
            market_bonus = market_readiness * 0.2

        # Coalition size optimization
        size_bonus = 0.0
        if hasattr(coalition, "members"):
            size = len(coalition.members)
            # Optimal size is 3-5 members for most use cases
            if 3 <= size <= 5:
                size_bonus = 0.1
            elif size == 2 or size == 6:
                size_bonus = 0.05

        market_positioning = base_score + speed_bonus + market_bonus + size_bonus
        return min(1.0, market_positioning)

    def _calculate_sustainability(
        self,
        coalition: Coalition,
        formation_result: FormationResult,
        agent_profiles: List[AgentProfile],
    ) -> float:
        """
        Calculate long-term sustainability score.

        Methodology:
        1. Formation quality (higher formation score = more stable)
        2. Resource balance (sufficient resources for operation)
        3. Capability coverage (comprehensive skill set)
        4. Member availability and commitment
        """
        # Formation quality factor
        quality_factor = 0.0
        if formation_result.success and formation_result.score > 0:
            # Normalize formation score to 0-1 range
            quality_factor = min(1.0, formation_result.score / 10.0)

        # Resource balance factor
        resource_balance = 0.0
        if agent_profiles:
            total_resources = {}
            for profile in agent_profiles:
                for resource, amount in profile.resources.items():
                    total_resources[resource] = total_resources.get(
                        resource, 0) + amount

            if total_resources:
                # Check if we have diverse, sufficient resources
                resource_types = len(total_resources)
                avg_resource_amount = sum(
                    total_resources.values()) / len(total_resources)

                diversity_score = min(
                    1.0, resource_types / 5.0)  # Assume 5 types is good
                sufficiency_score = min(
                    1.0, avg_resource_amount / 10.0)  # Assume 10 is sufficient

                resource_balance = (diversity_score + sufficiency_score) / 2.0

        # Capability coverage factor
        capability_coverage = 0.0
        if agent_profiles:
            all_capabilities = set()
            for profile in agent_profiles:
                all_capabilities.update(profile.capabilities)

            # Assume 8-10 capabilities is comprehensive coverage
            capability_coverage = min(1.0, len(all_capabilities) / 8.0)

        # Member availability factor
        availability_factor = 0.0
        if agent_profiles:
            total_availability = sum(p.availability for p in agent_profiles)
            avg_availability = total_availability / len(agent_profiles)
            availability_factor = avg_availability

        # Combine factors
        sustainability = (
            quality_factor * 0.3
            + resource_balance * 0.25
            + capability_coverage * 0.25
            + availability_factor * 0.2
        )

        return min(1.0, sustainability)

    def _calculate_operational_efficiency(
        self, coalition: Coalition, agent_profiles: List[AgentProfile]
    ) -> float:
        """Calculate operational efficiency based on resource utilization"""
        if not agent_profiles:
            return 0.0

        # Resource utilization efficiency
        total_resources = sum(sum(profile.resources.values())
                              for profile in agent_profiles)
        total_agents = len(agent_profiles)

        if total_agents > 0:
            resource_per_agent = total_resources / total_agents
            # Higher resource per agent = higher efficiency potential
            efficiency = min(
                1.0,
                resource_per_agent /
                20.0)  # Assume 20 is optimal
        else:
            efficiency = 0.0

        return efficiency

    def _calculate_innovation_potential(
        self, coalition: Coalition, agent_profiles: List[AgentProfile]
    ) -> float:
        """Calculate innovation potential from capability diversity"""
        if not agent_profiles:
            return 0.0

        # Unique capability combinations
        all_capabilities = set()
        for profile in agent_profiles:
            all_capabilities.update(profile.capabilities)

        # Innovation potential increases with capability diversity
        # Assume 10+ is highly innovative
        innovation_score = min(1.0, len(all_capabilities) / 10.0)

        return innovation_score

    def _calculate_total_value(self, metrics: BusinessValueMetrics) -> float:
        """Calculate total business value as weighted sum of all metrics"""
        total = (metrics.synergy_score *
                 self.metric_weights[BusinessMetricType.SYNERGY] +
                 metrics.risk_reduction *
                 self.metric_weights[BusinessMetricType.RISK_REDUCTION] +
                 metrics.market_positioning *
                 self.metric_weights[BusinessMetricType.MARKET_POSITIONING] +
                 metrics.sustainability_score *
                 self.metric_weights[BusinessMetricType.SUSTAINABILITY] +
                 metrics.operational_efficiency *
                 self.metric_weights[BusinessMetricType.OPERATIONAL_EFFICIENCY] +
                 metrics.innovation_potential *
                 self.metric_weights[BusinessMetricType.INNOVATION_POTENTIAL])

        return max(0.0, min(1.0, total))

    def _calculate_confidence(
            self,
            formation_result: FormationResult,
            agent_profiles: List[AgentProfile]) -> float:
        """Calculate confidence level in the business value calculation"""
        confidence_factors = []

        # Data completeness
        if agent_profiles:
            data_completeness = sum(
                1.0 if (p.capabilities and p.resources) else 0.5 for p in agent_profiles
            ) / len(agent_profiles)
            confidence_factors.append(data_completeness)

        # Formation result quality
        if formation_result.success:
            result_quality = min(1.0, formation_result.score / 5.0)
            confidence_factors.append(result_quality)
        else:
            # Low confidence for failed formations
            confidence_factors.append(0.1)

        # Sample size factor
        # 3+ agents for good confidence
        sample_factor = min(1.0, len(agent_profiles) / 3.0)
        confidence_factors.append(sample_factor)

        if confidence_factors:
            confidence = sum(confidence_factors) / len(confidence_factors)
            return max(0.0, min(1.0, confidence))
        else:
            return 0.0

    def get_calculation_history(
            self, limit: int = 100) -> List[BusinessValueMetrics]:
        """Get recent business value calculations"""
        return self.calculation_history[-limit:]

    def export_metrics_for_investors(
            self, metrics: BusinessValueMetrics) -> Dict[str, Any]:
        """Export metrics in investor-friendly format"""
        return {
            "executive_summary": {
                "total_business_value": f"{metrics.total_value:.1%}",
                "confidence_level": f"{metrics.confidence_level:.1%}",
                "key_strengths": self._identify_key_strengths(metrics),
                "calculated_at": metrics.calculation_timestamp.isoformat(),
            },
            "detailed_metrics": {
                "synergy_potential": f"{metrics.synergy_score:.1%}",
                "risk_mitigation": f"{metrics.risk_reduction:.1%}",
                "market_position": f"{metrics.market_positioning:.1%}",
                "sustainability": f"{metrics.sustainability_score:.1%}",
                "operational_efficiency": f"{metrics.operational_efficiency:.1%}",
                "innovation_potential": f"{metrics.innovation_potential:.1%}",
            },
            "methodology": metrics.methodology_notes,
            "investment_readiness": (
                "HIGH"
                if metrics.total_value > 0.7
                else "MEDIUM" if metrics.total_value > 0.4 else "LOW"
            ),
        }

    def _identify_key_strengths(
            self, metrics: BusinessValueMetrics) -> List[str]:
        """Identify the top strengths of the coalition"""
        metric_scores = [
            ("Synergy", metrics.synergy_score),
            ("Risk Reduction", metrics.risk_reduction),
            ("Market Positioning", metrics.market_positioning),
            ("Sustainability", metrics.sustainability_score),
            ("Operational Efficiency", metrics.operational_efficiency),
            ("Innovation Potential", metrics.innovation_potential),
        ]

        # Sort by score and return top 3
        sorted_metrics = sorted(
            metric_scores,
            key=lambda x: x[1],
            reverse=True)
        return [name for name, score in sorted_metrics[:3] if score > 0.6]


# Global business value engine instance
business_value_engine = BusinessValueCalculationEngine()

# Export for use by monitoring system
__all__ = [
    "BusinessValueCalculationEngine",
    "BusinessValueMetrics",
    "business_value_engine"]

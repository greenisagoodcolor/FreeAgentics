"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..coalition.coalition_criteria import CoalitionFormationCriteria
from ..coalition.coalition_models import Coalition, CoalitionGoal, CoalitionRole

"""
Coalition Formation Algorithms
Advanced algorithms for forming optimal coalitions based on various criteria:
- Active inference preference matching
- Capability complementarity
- Resource optimization
- Stability analysis
"""
logger = logging.getLogger(__name__)


class FormationStrategy(Enum):
    """Different coalition formation strategies"""

    ACTIVE_INFERENCE = "active_inference"
    CAPABILITY_BASED = "capability_based"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    PREFERENCE_MATCHING = "preference_matching"
    STABILITY_MAXIMIZATION = "stability_maximization"
    BUSINESS_OPPORTUNITY = "business_opportunity"


@dataclass
class AgentProfile:
    """Profile of an agent available for coalition formation"""

    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    resources: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, float] = field(default_factory=dict)  # agent_id -> preference score
    # Active inference parameters
    beliefs: Dict[str, float] = field(default_factory=dict)
    observations: Dict[str, float] = field(default_factory=dict)
    # Performance metrics
    reliability_score: float = 1.0
    cooperation_history: Dict[str, float] = field(default_factory=dict)  # agent_id -> success rate
    # Constraints
    max_coalitions: int = 3
    current_coalitions: int = 0
    availability: float = 1.0  # 0.0 to 1.0
    # Personality traits (for behavioral modeling)
    personality_traits: Dict[str, float] = field(default_factory=dict)


@dataclass
class FormationResult:
    """Result of a coalition formation attempt"""

    coalition: Optional[Coalition]
    success: bool
    score: float
    formation_time: float
    strategy_used: FormationStrategy
    participants: List[str]
    rejected_agents: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class ActiveInferenceFormation:
    """
    Coalition formation based on Active Inference principles.
    Agents form coalitions by minimizing free energy through belief alignment.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def calculate_free_energy(
        self, agent_beliefs: Dict[str, float], coalition_observations: Dict[str, float]
    ) -> float:
        """
        Calculate free energy F = -log P(o|s) + KL[Q(s)||P(s)]
        Simplified for coalition context.
        """
        if not agent_beliefs or not coalition_observations:
            return float("inf")
        # Likelihood term: how well agent beliefs predict coalition observations
        likelihood = 0.0
        for key in coalition_observations:
            if key in agent_beliefs:
                # Higher alignment = lower free energy
                diff = abs(agent_beliefs[key] - coalition_observations[key])
                likelihood += -math.log(max(0.001, 1.0 - diff))
        # Prior term: deviation from neutral beliefs (simplified KL divergence)
        prior = 0.0
        for belief_value in agent_beliefs.values():
            # Neutral belief is 0.5, deviation increases prior cost
            prior += abs(belief_value - 0.5) * 2.0
        return likelihood + prior

    def calculate_coalition_beliefs(self, agent_profiles: List[AgentProfile]) -> Dict[str, float]:
        """Calculate emergent coalition beliefs from member beliefs"""
        if not agent_profiles:
            return {}
        coalition_beliefs = {}
        belief_keys = set()
        # Collect all belief keys
        for profile in agent_profiles:
            belief_keys.update(profile.beliefs.keys())
        # Calculate weighted average beliefs
        for key in belief_keys:
            total_weight = 0.0
            weighted_sum = 0.0
            for profile in agent_profiles:
                if key in profile.beliefs:
                    weight = profile.reliability_score
                    weighted_sum += profile.beliefs[key] * weight
                    total_weight += weight
            if total_weight > 0:
                coalition_beliefs[key] = weighted_sum / total_weight
        return coalition_beliefs

    def evaluate_coalition_fit(
        self, agent: AgentProfile, existing_members: List[AgentProfile]
    ) -> float:
        """Evaluate how well an agent fits with existing coalition members"""
        if not existing_members:
            return 0.0  # Neutral for first member
        # Calculate current coalition beliefs
        coalition_beliefs = self.calculate_coalition_beliefs(existing_members)
        # Calculate free energy for agent joining
        free_energy = self.calculate_free_energy(agent.beliefs, coalition_beliefs)
        # Convert to fitness score (lower free energy = higher fitness)
        fitness = math.exp(-free_energy / self.temperature)
        return fitness

    def form_coalition(
        self,
        agents: List[AgentProfile],
        goal: Optional[CoalitionGoal] = None,
        max_size: int = 5,
    ) -> FormationResult:
        """Form coalition using active inference principles"""
        start_time = datetime.utcnow()
        if not agents:
            return FormationResult(
                coalition=None,
                success=False,
                score=0.0,
                formation_time=0.0,
                strategy_used=FormationStrategy.ACTIVE_INFERENCE,
                participants=[],
            )

        # Start with the agent with strongest beliefs (lowest entropy)
        def belief_strength(agent: AgentProfile) -> float:
            if not agent.beliefs:
                return 0.0
            return sum(abs(b - 0.5) for b in agent.beliefs.values())

        selected_agents = [max(agents, key=belief_strength)]
        remaining_agents = [a for a in agents if a.agent_id != selected_agents[0].agent_id]
        # Iteratively add agents that minimize collective free energy
        while len(selected_agents) < max_size and remaining_agents:
            best_agent = None
            best_score = -float("inf")
            for candidate in remaining_agents:
                # Check availability
                if candidate.current_coalitions >= candidate.max_coalitions:
                    continue
                fitness = self.evaluate_coalition_fit(candidate, selected_agents)
                # Bonus for complementary capabilities
                existing_capabilities = set()
                for member in selected_agents:
                    existing_capabilities.update(member.capabilities)
                new_capabilities = candidate.capabilities - existing_capabilities
                capability_bonus = len(new_capabilities) * 0.1
                total_score = fitness + capability_bonus
                if total_score > best_score:
                    best_score = total_score
                    best_agent = candidate
            if best_agent and best_score > 0.1:  # Minimum threshold
                selected_agents.append(best_agent)
                remaining_agents.remove(best_agent)
            else:
                break
        # Create coalition if we have enough members
        if len(selected_agents) >= 2:
            coalition = Coalition(
                coalition_id=f"coalition_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                name=f"Active Inference Coalition",
                description=("Coalition formed through active inference belief alignment",),
            )
            # Add members
            for i, agent in enumerate(selected_agents):
                role = CoalitionRole.LEADER if i == 0 else CoalitionRole.CONTRIBUTOR
                coalition.add_member(
                    agent_id=agent.agent_id,
                    role=role,
                    capabilities=agent.capabilities,
                    resources=agent.resources,
                )
            # Add goal if provided
            if goal:
                coalition.add_goal(goal)
            formation_time = (datetime.utcnow() - start_time).total_seconds()
            return FormationResult(
                coalition=coalition,
                success=True,
                score=best_score,
                formation_time=formation_time,
                strategy_used=FormationStrategy.ACTIVE_INFERENCE,
                participants=[a.agent_id for a in selected_agents],
                rejected_agents=[a.agent_id for a in remaining_agents],
            )
        formation_time = (datetime.utcnow() - start_time).total_seconds()
        return FormationResult(
            coalition=None,
            success=False,
            score=0.0,
            formation_time=formation_time,
            strategy_used=FormationStrategy.ACTIVE_INFERENCE,
            participants=[],
            rejected_agents=[a.agent_id for a in agents],
        )


class CapabilityBasedFormation:
    """Coalition formation based on capability complementarity"""

    def __init__(self, coverage_weight: float = 0.6, redundancy_penalty: float = 0.3) -> None:
        self.coverage_weight = coverage_weight
        self.redundancy_penalty = redundancy_penalty

    def calculate_capability_score(
        self, agents: List[AgentProfile], required_capabilities: Optional[Set[str]] = None
    ) -> float:
        """Calculate how well a group of agents covers required capabilities"""
        if not agents:
            return 0.0
        # Collect all capabilities
        all_capabilities = set()
        capability_counts = {}
        for agent in agents:
            for cap in agent.capabilities:
                all_capabilities.add(cap)
                capability_counts[cap] = capability_counts.get(cap, 0) + 1
        # Coverage score
        coverage_score = len(all_capabilities)
        if required_capabilities:
            covered_required = len(all_capabilities & required_capabilities)
            coverage_score = (
                covered_required / len(required_capabilities) if required_capabilities else 1.0
            )
        # Redundancy penalty
        redundancy_score = 0.0
        for count in capability_counts.values():
            if count > 1:
                redundancy_score += (count - 1) * self.redundancy_penalty
        return coverage_score * self.coverage_weight - redundancy_score

    def form_coalition(
        self,
        agents: List[AgentProfile],
        required_capabilities: Optional[Set[str]] = None,
        max_size: int = 5,
    ) -> FormationResult:
        """Form coalition by optimizing capability coverage"""
        start_time = datetime.utcnow()
        if not agents:
            return FormationResult(
                coalition=None,
                success=False,
                score=0.0,
                formation_time=0.0,
                strategy_used=FormationStrategy.CAPABILITY_BASED,
                participants=[],
            )
        # Greedy algorithm: iteratively add agents that improve capability score
        selected_agents = []
        remaining_agents = [a for a in agents if a.current_coalitions < a.max_coalitions]
        while len(selected_agents) < max_size and remaining_agents:
            best_agent = None
            best_score = -float("inf")
            for candidate in remaining_agents:
                test_group = selected_agents + [candidate]
                score = self.calculate_capability_score(test_group, required_capabilities)
                if score > best_score:
                    best_score = score
                    best_agent = candidate
            if best_agent and (
                not selected_agents
                or best_score
                > self.calculate_capability_score(selected_agents, required_capabilities)
            ):
                selected_agents.append(best_agent)
                remaining_agents.remove(best_agent)
            else:
                break
        # Create coalition if viable
        if len(selected_agents) >= 2:
            coalition = Coalition(
                coalition_id=f"coalition_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                name="Capability-Based Coalition",
                description="Coalition formed based on capability complementarity",
            )
            # Assign roles based on capability breadth
            sorted_agents = sorted(selected_agents, key=lambda a: len(a.capabilities), reverse=True)
            for i, agent in enumerate(sorted_agents):
                role = CoalitionRole.LEADER if i == 0 else CoalitionRole.CONTRIBUTOR
                coalition.add_member(
                    agent_id=agent.agent_id,
                    role=role,
                    capabilities=agent.capabilities,
                    resources=agent.resources,
                )
            formation_time = (datetime.utcnow() - start_time).total_seconds()
            return FormationResult(
                coalition=coalition,
                success=True,
                score=best_score,
                formation_time=formation_time,
                strategy_used=FormationStrategy.CAPABILITY_BASED,
                participants=[a.agent_id for a in selected_agents],
                rejected_agents=[a.agent_id for a in remaining_agents],
            )
        formation_time = (datetime.utcnow() - start_time).total_seconds()
        return FormationResult(
            coalition=None,
            success=False,
            score=0.0,
            formation_time=formation_time,
            strategy_used=FormationStrategy.CAPABILITY_BASED,
            participants=[],
            rejected_agents=[a.agent_id for a in agents],
        )


class ResourceOptimizationFormation:
    """Coalition formation based on resource optimization and efficiency"""

    def __init__(self, efficiency_weight: float = 0.5, balance_weight: float = 0.3) -> None:
        self.efficiency_weight = efficiency_weight
        self.balance_weight = balance_weight

    def calculate_resource_efficiency(self, agents: List[AgentProfile]) -> float:
        """Calculate resource utilization efficiency"""
        if not agents:
            return 0.0
        # Total resources available
        total_resources = {}
        for agent in agents:
            for resource, amount in agent.resources.items():
                total_resources[resource] = total_resources.get(resource, 0) + amount
        # Efficiency based on resource diversity and quantity
        diversity_score = len(total_resources)
        quantity_score = sum(total_resources.values())
        # Balance score - how evenly distributed resources are
        if total_resources:
            mean_amount = quantity_score / len(total_resources)
            variance = sum(
                (amount - mean_amount) ** 2 for amount in total_resources.values()
            ) / len(total_resources)
            balance_score = 1.0 / (1.0 + variance / max(1.0, mean_amount))
        else:
            balance_score = 0.0
        return (
            diversity_score * 10 + quantity_score
        ) * self.efficiency_weight + balance_score * self.balance_weight

    def form_coalition(
        self,
        agents: List[AgentProfile],
        resource_requirements: Optional[Dict[str, float]] = None,
        max_size: int = 5,
    ) -> FormationResult:
        """Form coalition by optimizing resource allocation"""
        start_time = datetime.utcnow()
        if not agents:
            return FormationResult(
                coalition=None,
                success=False,
                score=0.0,
                formation_time=0.0,
                strategy_used=FormationStrategy.RESOURCE_OPTIMIZATION,
                participants=[],
            )

        # Start with agent having the most diverse resources
        def resource_diversity(agent: AgentProfile) -> float:
            return len(agent.resources) + sum(agent.resources.values()) * 0.1

        selected_agents = [max(agents, key=resource_diversity)]
        remaining_agents = [
            a
            for a in agents
            if a.agent_id != selected_agents[0].agent_id and a.current_coalitions < a.max_coalitions
        ]
        while len(selected_agents) < max_size and remaining_agents:
            best_agent = None
            best_score = -float("inf")
            for candidate in remaining_agents:
                test_group = selected_agents + [candidate]
                score = self.calculate_resource_efficiency(test_group)
                # Bonus for meeting resource requirements
                if resource_requirements:
                    total_resources = {}
                    for agent in test_group:
                        for resource, amount in agent.resources.items():
                            total_resources[resource] = total_resources.get(resource, 0) + amount
                    requirement_bonus = 0.0
                    for resource, required in resource_requirements.items():
                        if resource in total_resources:
                            coverage = min(1.0, total_resources[resource] / required)
                            requirement_bonus += (
                                coverage * 50
                            )  # Significant bonus for meeting requirements
                    score += requirement_bonus
                if score > best_score:
                    best_score = score
                    best_agent = candidate
            if best_agent and best_score > self.calculate_resource_efficiency(selected_agents):
                selected_agents.append(best_agent)
                remaining_agents.remove(best_agent)
            else:
                break
        # Create coalition
        if len(selected_agents) >= 2:
            coalition = Coalition(
                coalition_id=f"coalition_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                name="Resource-Optimized Coalition",
                description=("Coalition formed for optimal resource utilization",),
            )
            # Assign roles based on resource contribution
            sorted_agents = sorted(
                selected_agents, key=lambda a: sum(a.resources.values()), reverse=True
            )
            for i, agent in enumerate(sorted_agents):
                role = CoalitionRole.LEADER if i == 0 else CoalitionRole.CONTRIBUTOR
                coalition.add_member(
                    agent_id=agent.agent_id,
                    role=role,
                    capabilities=agent.capabilities,
                    resources=agent.resources,
                )
            formation_time = (datetime.utcnow() - start_time).total_seconds()
            return FormationResult(
                coalition=coalition,
                success=True,
                score=best_score,
                formation_time=formation_time,
                strategy_used=FormationStrategy.RESOURCE_OPTIMIZATION,
                participants=[a.agent_id for a in selected_agents],
                rejected_agents=[a.agent_id for a in remaining_agents],
            )
        formation_time = (datetime.utcnow() - start_time).total_seconds()
        return FormationResult(
            coalition=None,
            success=False,
            score=0.0,
            formation_time=formation_time,
            strategy_used=FormationStrategy.RESOURCE_OPTIMIZATION,
            participants=[],
            rejected_agents=[a.agent_id for a in agents],
        )


class CoalitionFormationEngine:
    """
    Main engine for coalition formation using multiple strategies.
    """

    def __init__(self) -> None:
        self.strategies = {
            FormationStrategy.ACTIVE_INFERENCE: ActiveInferenceFormation(),
            FormationStrategy.CAPABILITY_BASED: CapabilityBasedFormation(),
            FormationStrategy.RESOURCE_OPTIMIZATION: ResourceOptimizationFormation(),
        }
        self.formation_history: List[FormationResult] = []

    def register_strategy(self, strategy: FormationStrategy, implementation: Any) -> None:
        """Register a custom formation strategy"""
        self.strategies[strategy] = implementation

    def form_coalition(
        self,
        agents: List[AgentProfile],
        strategy: FormationStrategy = FormationStrategy.ACTIVE_INFERENCE,
        goal: Optional[CoalitionGoal] = None,
        criteria: Optional[CoalitionFormationCriteria] = None,
        max_size: int = 5,
    ) -> FormationResult:
        """
        Form a coalition using the specified strategy.
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown formation strategy: {strategy}")
        formation_impl = self.strategies[strategy]
        # Prepare parameters based on strategy and criteria
        kwargs = {"max_size": max_size}
        if strategy == FormationStrategy.ACTIVE_INFERENCE and goal:
            kwargs["goal"] = goal
        elif strategy == FormationStrategy.CAPABILITY_BASED and criteria:
            kwargs["required_capabilities"] = criteria.required_capabilities
        elif strategy == FormationStrategy.RESOURCE_OPTIMIZATION and criteria:
            kwargs["resource_requirements"] = criteria.resource_requirements
        # Execute formation
        result = formation_impl.form_coalition(agents, **kwargs)
        # Store result
        self.formation_history.append(result)
        # Log result
        if result.success:
            logger.info(
                f"Successfully formed coalition with {len(result.participants)} members "
                f"using {strategy.value} strategy (score: {result.score:.2f})"
            )
        else:
            logger.warning(f"Failed to form coalition using {strategy.value} strategy")
        return result

    def try_multiple_strategies(
        self,
        agents: List[AgentProfile],
        strategies: Optional[List[FormationStrategy]] = None,
        goal: Optional[CoalitionGoal] = None,
        criteria: Optional[CoalitionFormationCriteria] = None,
        max_size: int = 5,
    ) -> FormationResult:
        """
        Try multiple formation strategies and return the best result.
        """
        if strategies is None:
            strategies = [
                FormationStrategy.ACTIVE_INFERENCE,
                FormationStrategy.CAPABILITY_BASED,
                FormationStrategy.RESOURCE_OPTIMIZATION,
            ]
        best_result = None
        for strategy in strategies:
            try:
                result = self.form_coalition(
                    agents=agents,
                    strategy=strategy,
                    goal=goal,
                    criteria=criteria,
                    max_size=max_size,
                )
                if result.success and (best_result is None or result.score > best_result.score):
                    best_result = result
            except Exception as e:
                logger.error(f"Error with strategy {strategy.value}: {e}")
                continue
        return best_result or FormationResult(
            coalition=None,
            success=False,
            score=0.0,
            formation_time=0.0,
            strategy_used=(strategies[0] if strategies else FormationStrategy.ACTIVE_INFERENCE),
            participants=[],
            rejected_agents=[a.agent_id for a in agents],
        )

    def get_formation_statistics(self) -> Dict[str, Any]:
        """Get statistics about coalition formation performance"""
        if not self.formation_history:
            return {}
        successful_formations = [r for r in self.formation_history if r.success]
        stats = {
            "total_attempts": len(self.formation_history),
            "successful_formations": len(successful_formations),
            "success_rate": len(successful_formations) / len(self.formation_history),
            "average_formation_time": sum(r.formation_time for r in self.formation_history)
            / len(self.formation_history),
            "average_coalition_size": sum(len(r.participants) for r in successful_formations)
            / max(1, len(successful_formations)),
            "strategy_performance": {},
        }
        # Strategy-specific statistics
        for strategy in FormationStrategy:
            strategy_results = [r for r in self.formation_history if r.strategy_used == strategy]
            if strategy_results:
                successful = [r for r in strategy_results if r.success]
                stats["strategy_performance"][strategy.value] = {
                    "attempts": len(strategy_results),
                    "successes": len(successful),
                    "success_rate": len(successful) / len(strategy_results),
                    "average_score": sum(r.score for r in successful) / max(1, len(successful)),
                }
        return stats

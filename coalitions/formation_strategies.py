"""Coalition formation strategies for multi-agent systems."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from coalitions.coalition import Coalition, CoalitionObjective, CoalitionRole

logger = logging.getLogger(__name__)


@dataclass
class AgentProfile:
    """Profile of an agent for coalition formation."""

    agent_id: str
    capabilities: List[str]
    capacity: float  # How much work the agent can handle (0.0 to 1.0)
    reputation: float  # Agent's reputation score (0.0 to 1.0)
    preferences: Dict[str, float]  # Preferences for working with other agents
    current_coalitions: List[str]  # IDs of coalitions agent is currently in
    max_coalitions: int = 3  # Maximum number of coalitions agent can join


@dataclass
class FormationResult:
    """Result of a coalition formation process."""

    coalitions: List[Coalition]
    unassigned_agents: List[str]
    formation_time: float
    objective_coverage: float  # Percentage of objectives that can be achieved
    formation_score: float  # Overall quality of the formation


class FormationStrategy(ABC):
    """Abstract base class for coalition formation strategies."""

    def __init__(self, name: str):
        """Initialize formation strategy.

        Args:
            name: Name of the strategy
        """
        self.name = name

    @abstractmethod
    def form_coalitions(
        self,
        agents: List[AgentProfile],
        objectives: List[CoalitionObjective],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> FormationResult:
        """Form coalitions based on agents and objectives.

        Args:
            agents: Available agents
            objectives: Objectives to achieve
            constraints: Additional constraints for formation

        Returns:
            Formation result
        """
        pass

    def _calculate_capability_match(
        self, agent_capabilities: Set[str], required_capabilities: Set[str]
    ) -> float:
        """Calculate how well agent capabilities match requirements.

        Args:
            agent_capabilities: Set of agent capabilities
            required_capabilities: Set of required capabilities

        Returns:
            Match score (0.0 to 1.0)
        """
        if not required_capabilities:
            return 1.0

        intersection = agent_capabilities.intersection(required_capabilities)
        return len(intersection) / len(required_capabilities)

    def _calculate_coalition_value(
        self, agents: List[AgentProfile], objective: CoalitionObjective
    ) -> float:
        """Calculate the value of a potential coalition for an objective.

        Args:
            agents: Agents in the potential coalition
            objective: Objective to achieve

        Returns:
            Coalition value score
        """
        # Get all capabilities
        all_capabilities = set()
        total_capacity = 0.0
        avg_reputation = 0.0

        for agent in agents:
            all_capabilities.update(agent.capabilities)
            total_capacity += agent.capacity
            avg_reputation += agent.reputation

        if agents:
            avg_reputation /= len(agents)

        # Calculate capability coverage
        required_capabilities = set(objective.required_capabilities)
        capability_coverage = (
            len(all_capabilities.intersection(required_capabilities)) / len(required_capabilities)
            if required_capabilities
            else 1.0
        )

        # Calculate synergy (agents working well together)
        synergy_score = self._calculate_synergy(agents)

        # Combine factors
        value = (
            0.4 * capability_coverage
            + 0.2 * min(total_capacity, 1.0)  # Cap at 1.0
            + 0.2 * avg_reputation
            + 0.2 * synergy_score
        ) * objective.priority

        return value

    def _calculate_synergy(self, agents: List[AgentProfile]) -> float:
        """Calculate synergy between agents.

        Args:
            agents: List of agents

        Returns:
            Synergy score (0.0 to 1.0)
        """
        if len(agents) <= 1:
            return 1.0

        synergy_sum = 0.0
        pair_count = 0

        for i, agent1 in enumerate(agents):
            for agent2 in agents[i + 1 :]:
                # Check if agent1 has preference for agent2
                preference = agent1.preferences.get(agent2.agent_id, 0.5)  # Default neutral
                synergy_sum += preference
                pair_count += 1

        return synergy_sum / pair_count if pair_count > 0 else 1.0


class GreedyFormation(FormationStrategy):
    """Greedy coalition formation strategy."""

    def __init__(self) -> None:
        """Initialize the greedy formation strategy."""
        super().__init__("Greedy Formation")

    def form_coalitions(
        self,
        agents: List[AgentProfile],
        objectives: List[CoalitionObjective],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> FormationResult:
        """Form coalitions using greedy approach.

        Greedily assigns agents to coalitions based on immediate utility.
        """
        import time

        start_time = time.time()

        constraints = constraints or {}
        max_coalition_size = constraints.get("max_coalition_size", 10)

        coalitions = []
        available_agents = agents.copy()
        assigned_agents = set()

        # Sort objectives by priority (highest first)
        sorted_objectives = sorted(objectives, key=lambda x: x.priority, reverse=True)

        for obj_idx, objective in enumerate(sorted_objectives):
            # Create coalition for this objective
            coalition_id = f"coalition_{obj_idx + 1}"
            coalition = Coalition(
                coalition_id=coalition_id,
                name=f"Coalition for {objective.description}",
                objectives=[objective],
                max_size=max_coalition_size,
            )

            # Find best agents for this objective
            candidate_agents = [a for a in available_agents if a.agent_id not in assigned_agents]

            # Score each agent for this objective
            agent_scores = []
            for agent in candidate_agents:
                agent_caps = set(agent.capabilities)
                required_caps = set(objective.required_capabilities)

                capability_match = self._calculate_capability_match(agent_caps, required_caps)
                score = capability_match * agent.reputation * agent.capacity

                agent_scores.append((agent, score))

            # Sort by score (highest first)
            agent_scores.sort(key=lambda x: x[1], reverse=True)

            # Add agents to coalition until requirements are met or no more agents
            required_capabilities = set(objective.required_capabilities)
            covered_capabilities: set[str] = set()

            for agent, score in agent_scores:
                if len(coalition.members) >= max_coalition_size:
                    break

                # Check if agent can still join more coalitions
                if len(agent.current_coalitions) >= agent.max_coalitions:
                    continue

                # Add agent if they bring new capabilities or we still need coverage
                agent_caps = set(agent.capabilities)
                new_capabilities = agent_caps - covered_capabilities

                if (
                    new_capabilities and not required_capabilities.issubset(covered_capabilities)
                ) or len(coalition.members) == 0:  # Always add at least one agent
                    role = (
                        CoalitionRole.LEADER
                        if len(coalition.members) == 0
                        else CoalitionRole.MEMBER
                    )

                    if coalition.add_member(agent.agent_id, role, agent.capabilities):
                        assigned_agents.add(agent.agent_id)
                        agent.current_coalitions.append(coalition_id)
                        covered_capabilities.update(agent_caps)

                        # Stop if all required capabilities are covered
                        if required_capabilities.issubset(covered_capabilities):
                            break

            # Only add coalition if it has members and can make progress
            if coalition.members:
                coalition.activate()
                coalitions.append(coalition)
                logger.info(
                    f"Formed coalition {coalition_id} with {len(coalition.members)} members"
                )

        formation_time = time.time() - start_time
        unassigned_agents = [a.agent_id for a in agents if a.agent_id not in assigned_agents]

        # Calculate objective coverage
        achievable_objectives = 0
        for coalition in coalitions:
            for objective in coalition.objectives:
                if coalition.can_achieve_objective(objective):
                    achievable_objectives += 1

        objective_coverage = achievable_objectives / len(objectives) if objectives else 1.0

        # Calculate formation score
        total_value = sum(
            self._calculate_coalition_value(
                [a for a in agents if a.agent_id in coal.members],
                coal.objectives[0],
            )
            for coal in coalitions
            if coal.objectives
        )
        formation_score = total_value / len(objectives) if objectives else 0.0

        return FormationResult(
            coalitions=coalitions,
            unassigned_agents=unassigned_agents,
            formation_time=formation_time,
            objective_coverage=objective_coverage,
            formation_score=formation_score,
        )


class OptimalFormation(FormationStrategy):
    """Optimal coalition formation using exhaustive search (for small problems)."""

    def __init__(self, max_search_size: int = 8):
        """Initialize the optimal formation strategy."""
        super().__init__("Optimal Formation")
        self.max_search_size = max_search_size

    def form_coalitions(
        self,
        agents: List[AgentProfile],
        objectives: List[CoalitionObjective],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> FormationResult:
        """Form coalitions using optimal search.

        Uses exhaustive search for small problems, falls back to greedy for large ones.
        """
        import time

        start_time = time.time()

        # Fall back to greedy if problem is too large
        if len(agents) > self.max_search_size or len(objectives) > self.max_search_size:
            logger.warning("Problem too large for optimal search, falling back to greedy")
            greedy = GreedyFormation()
            return greedy.form_coalitions(agents, objectives, constraints)

        constraints = constraints or {}
        max_coalition_size = constraints.get("max_coalition_size", len(agents))

        best_formation = None
        best_score = -1.0

        # Generate all possible ways to assign agents to objectives
        for assignment in self._generate_assignments(agents, objectives, max_coalition_size):
            coalitions = self._create_coalitions_from_assignment(assignment, objectives)
            score = self._evaluate_formation(coalitions, objectives)

            if score > best_score:
                best_score = score
                best_formation = coalitions

        if best_formation is None:
            best_formation = []

        formation_time = time.time() - start_time

        # Calculate metrics
        assigned_agents: set[str] = set()
        for coalition in best_formation:
            assigned_agents.update(coalition.members.keys())

        unassigned_agents = [a.agent_id for a in agents if a.agent_id not in assigned_agents]

        achievable_objectives = sum(
            1
            for coalition in best_formation
            for objective in coalition.objectives
            if coalition.can_achieve_objective(objective)
        )
        objective_coverage = achievable_objectives / len(objectives) if objectives else 1.0

        return FormationResult(
            coalitions=best_formation,
            unassigned_agents=unassigned_agents,
            formation_time=formation_time,
            objective_coverage=objective_coverage,
            formation_score=best_score,
        )

    def _generate_assignments(
        self,
        agents: List[AgentProfile],
        objectives: List[CoalitionObjective],
        max_size: int,
    ) -> Any:
        """Generate all possible assignments of agents to objectives."""
        # For each objective, generate all possible coalitions of agents
        # This is computationally expensive but optimal for small problems

        for obj_idx, objective in enumerate(objectives):
            # Generate all possible subsets of agents for this objective
            agent_subsets = []
            for size in range(1, min(max_size + 1, len(agents) + 1)):
                for combo in combinations(agents, size):
                    agent_subsets.append(combo)

            # Yield each valid assignment
            yield [(obj_idx, list(combo)) for combo in agent_subsets]

    def _create_coalitions_from_assignment(
        self,
        assignment: List[Tuple[int, List[AgentProfile]]],
        objectives: List[CoalitionObjective],
    ) -> List[Coalition]:
        """Create coalitions from an assignment."""
        coalitions = []

        for obj_idx, agent_group in assignment:
            if not agent_group:
                continue

            objective = objectives[obj_idx]
            coalition_id = f"optimal_coalition_{obj_idx}"

            coalition = Coalition(
                coalition_id=coalition_id,
                name=f"Optimal Coalition for {objective.description}",
                objectives=[objective],
            )

            for i, agent in enumerate(agent_group):
                role = CoalitionRole.LEADER if i == 0 else CoalitionRole.MEMBER
                coalition.add_member(agent.agent_id, role, agent.capabilities)

            coalition.activate()
            coalitions.append(coalition)

        return coalitions

    def _evaluate_formation(
        self, coalitions: List[Coalition], objectives: List[CoalitionObjective]
    ) -> float:
        """Evaluate the quality of a formation."""
        if not coalitions:
            return 0.0

        total_score = 0.0

        for coalition in coalitions:
            for objective in coalition.objectives:
                if coalition.can_achieve_objective(objective):
                    # Base score from objective priority
                    score = objective.priority

                    # Bonus for efficient coalition size
                    required_caps = set(objective.required_capabilities)
                    available_caps = coalition.get_capabilities()

                    if required_caps.issubset(available_caps):
                        # Prefer smaller coalitions that still meet requirements
                        efficiency_bonus = 1.0 / len(coalition.members)
                        score += efficiency_bonus * 0.2

                    total_score += score

        return total_score


class HierarchicalFormation(FormationStrategy):
    """Hierarchical coalition formation strategy."""

    def __init__(self) -> None:
        """Initialize the hierarchical formation strategy."""
        super().__init__("Hierarchical Formation")

    def form_coalitions(
        self,
        agents: List[AgentProfile],
        objectives: List[CoalitionObjective],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> FormationResult:
        """Form coalitions using hierarchical approach.

        Creates hierarchical coalitions with coordinators and sub-groups.
        """
        import time

        start_time = time.time()

        constraints = constraints or {}
        max_coalition_size = constraints.get("max_coalition_size", 15)
        min_subgroup_size = constraints.get("min_subgroup_size", 3)

        coalitions = []
        assigned_agents = set()

        # Group objectives by similarity/dependencies
        objective_groups = self._group_objectives(objectives)

        for group_idx, obj_group in enumerate(objective_groups):
            coalition_id = f"hierarchical_coalition_{group_idx + 1}"

            # Create main coalition
            coalition = Coalition(
                coalition_id=coalition_id,
                name=f"Hierarchical Coalition {group_idx + 1}",
                objectives=obj_group,
                max_size=max_coalition_size,
            )

            # Find best coordinator (agent with highest reputation and broad capabilities)
            available_agents = [a for a in agents if a.agent_id not in assigned_agents]

            if not available_agents:
                continue

            coordinator = max(
                available_agents,
                key=lambda a: a.reputation * len(a.capabilities),
            )

            coalition.add_member(
                coordinator.agent_id,
                CoalitionRole.LEADER,
                coordinator.capabilities,
            )
            assigned_agents.add(coordinator.agent_id)
            coordinator.current_coalitions.append(coalition_id)

            # For each objective in the group, create specialized sub-teams
            for objective in obj_group:
                required_caps = set(objective.required_capabilities)
                covered_caps = set(coordinator.capabilities)

                # Find agents that can contribute to this objective
                specialist_candidates = []
                for agent in available_agents:
                    if agent.agent_id in assigned_agents:
                        continue

                    agent_caps = set(agent.capabilities)
                    contribution = len(agent_caps.intersection(required_caps))

                    if contribution > 0:
                        specialist_candidates.append((agent, contribution))

                # Sort by contribution and add best specialists
                specialist_candidates.sort(key=lambda x: x[1], reverse=True)

                added_count = 0
                for agent, contribution in specialist_candidates:
                    if (
                        len(coalition.members) >= max_coalition_size
                        or added_count >= min_subgroup_size
                    ):
                        break

                    if len(agent.current_coalitions) >= agent.max_coalitions:
                        continue

                    coalition.add_member(
                        agent.agent_id,
                        CoalitionRole.MEMBER,
                        agent.capabilities,
                    )
                    assigned_agents.add(agent.agent_id)
                    agent.current_coalitions.append(coalition_id)
                    added_count += 1

                    # Update covered capabilities
                    covered_caps.update(agent.capabilities)

                    # Stop if we have all required capabilities
                    if required_caps.issubset(covered_caps):
                        break

            # Add coordinators for large coalitions
            if len(coalition.members) > 6:
                # Promote some members to coordinators
                members = list(coalition.members.values())
                members.sort(key=lambda m: m.contribution_score, reverse=True)

                for i in range(min(2, len(members) // 3)):
                    if members[i].role == CoalitionRole.MEMBER:
                        members[i].role = CoalitionRole.COORDINATOR

            if coalition.members:
                coalition.activate()
                coalitions.append(coalition)
                logger.info(
                    f"Formed hierarchical coalition {coalition_id} with {len(coalition.members)} members"
                )

        formation_time = time.time() - start_time
        unassigned_agents = [a.agent_id for a in agents if a.agent_id not in assigned_agents]

        # Calculate metrics
        achievable_objectives = 0
        for coalition in coalitions:
            for objective in coalition.objectives:
                if coalition.can_achieve_objective(objective):
                    achievable_objectives += 1

        objective_coverage = achievable_objectives / len(objectives) if objectives else 1.0

        # Calculate formation score with hierarchy bonus
        total_value = 0.0
        for coalition in coalitions:
            for objective in coalition.objectives:
                coalition_agents = [a for a in agents if a.agent_id in coalition.members]
                value = self._calculate_coalition_value(coalition_agents, objective)

                # Hierarchy bonus for well-structured coalitions
                hierarchy_bonus = self._calculate_hierarchy_bonus(coalition)
                total_value += value * (1.0 + hierarchy_bonus)

        formation_score = total_value / len(objectives) if objectives else 0.0

        return FormationResult(
            coalitions=coalitions,
            unassigned_agents=unassigned_agents,
            formation_time=formation_time,
            objective_coverage=objective_coverage,
            formation_score=formation_score,
        )

    def _group_objectives(
        self, objectives: List[CoalitionObjective]
    ) -> List[List[CoalitionObjective]]:
        """Group objectives by similarity or dependencies."""
        if not objectives:
            return []

        # Simple grouping by capability overlap
        groups: List[List[CoalitionObjective]] = []

        for objective in objectives:
            objective_caps = set(objective.required_capabilities)

            # Find existing group with similar capabilities
            best_group = None
            best_overlap = 0.0

            for group in groups:
                group_caps = set()
                for obj in group:
                    group_caps.update(obj.required_capabilities)

                overlap = len(objective_caps.intersection(group_caps)) / len(
                    objective_caps.union(group_caps)
                )

                if overlap > best_overlap and overlap > 0.3:  # Threshold for grouping
                    best_group = group
                    best_overlap = overlap

            if best_group:
                best_group.append(objective)
            else:
                groups.append([objective])

        return groups

    def _calculate_hierarchy_bonus(self, coalition: Coalition) -> float:
        """Calculate bonus for well-structured hierarchy."""
        member_count = len(coalition.members)

        if member_count <= 3:
            return 0.0  # No hierarchy needed

        # Count roles
        role_counts: Dict[str, int] = {}
        for member in coalition.members.values():
            role_counts[member.role.value] = role_counts.get(member.role.value, 0) + 1

        # Ideal hierarchy ratios
        leaders = role_counts.get(CoalitionRole.LEADER.value, 0)
        coordinators = role_counts.get(CoalitionRole.COORDINATOR.value, 0)
        members = role_counts.get(CoalitionRole.MEMBER.value, 0)

        # Should have 1 leader, some coordinators, and members
        ideal_coordinators = max(1, member_count // 5)  # 1 coordinator per 5 members

        hierarchy_score = 0.0

        # Check leader count (should be exactly 1)
        if leaders == 1:
            hierarchy_score += 0.4

        # Check coordinator ratio
        if coordinators > 0:
            coordinator_ratio = abs(coordinators - ideal_coordinators) / max(ideal_coordinators, 1)
            hierarchy_score += 0.3 * max(0.0, 1.0 - coordinator_ratio)

        # Check span of control (not too many direct reports)
        if coordinators > 0:
            avg_span = members / coordinators
            if 3 <= avg_span <= 7:  # Ideal span of control
                hierarchy_score += 0.3

        return min(hierarchy_score, 0.5)  # Cap at 50% bonus

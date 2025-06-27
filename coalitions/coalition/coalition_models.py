"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

"""
Coalition Models
Core data models for coalition formation, management, and lifecycle.
"""
logger = logging.getLogger(__name__)


class CoalitionStatus(Enum):
    """Status of a coalition."""

    FORMING = "forming"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISSOLVING = "dissolving"
    DISSOLVED = "dissolved"


class CoalitionRole(Enum):
    """Roles that agents can have within a coalition."""

    LEADER = "leader"
    COORDINATOR = "coordinator"
    CONTRIBUTOR = "contributor"
    SPECIALIST = "specialist"
    OBSERVER = "observer"


class CoalitionGoalStatus(Enum):
    """Status of coalition goals."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class CoalitionGoal:
    """Represents a goal that the coalition is working toward."""

    goal_id: str
    title: str
    description: str
    priority: float = 0.5  # 0.0 (low) to 1.0 (high)
    # Goal metrics
    target_value: float = 0.0
    current_progress: float = 0.0
    success_threshold: float = 0.8
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    # Status and tracking
    status: CoalitionGoalStatus = CoalitionGoalStatus.PROPOSED
    assigned_members: Set[str] = field(default_factory=set)
    required_capabilities: Set[str] = field(default_factory=set)
    # Resource requirements
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    # Voting and consensus
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)
    consensus_required: bool = True

    def __post_init__(self):
        if not self.goal_id:
            self.goal_id = f"goal_{uuid.uuid4().hex[:8]}"

    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        return min(100.0, self.current_progress * 100.0)

    @property
    def is_overdue(self) -> bool:
        """Check if goal is overdue."""
        if self.deadline:
            return datetime.utcnow() > self.deadline
        return False

    @property
    def is_completed(self) -> bool:
        """Check if goal is successfully completed."""
        return (
            self.status == CoalitionGoalStatus.COMPLETED
            and self.current_progress >= self.success_threshold
        )

    def add_vote(self, member_id: str, support: bool) -> None:
        """Add a vote for or against the goal."""
        if support:
            self.votes_for.add(member_id)
            self.votes_against.discard(member_id)
        else:
            self.votes_against.add(member_id)
            self.votes_for.discard(member_id)

    def calculate_consensus(self, total_members: int) -> float:
        """Calculate consensus level (0-1)."""
        total_votes = len(self.votes_for) + len(self.votes_against)
        if total_votes == 0:
            return 0.0
        return len(self.votes_for) / total_votes

    def update_progress(self, progress: float, member_id: Optional[str] = None) -> None:
        """Update goal progress."""
        self.current_progress = max(0.0, min(1.0, progress))
        # Auto-update status based on progress
        if self.current_progress >= self.success_threshold:
            if self.status == CoalitionGoalStatus.IN_PROGRESS:
                self.status = CoalitionGoalStatus.COMPLETED
        elif self.current_progress > 0.0:
            if self.status == CoalitionGoalStatus.ACCEPTED:
                self.status = CoalitionGoalStatus.IN_PROGRESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "target_value": self.target_value,
            "current_progress": self.current_progress,
            "success_threshold": self.success_threshold,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "estimated_completion": (
                self.estimated_completion.isoformat() if self.estimated_completion else None
            ),
            "status": self.status.value,
            "assigned_members": list(self.assigned_members),
            "required_capabilities": list(self.required_capabilities),
            "resource_requirements": self.resource_requirements,
            "allocated_resources": self.allocated_resources,
            "votes_for": list(self.votes_for),
            "votes_against": list(self.votes_against),
            "consensus_required": self.consensus_required,
            "progress_percentage": self.progress_percentage,
            "is_overdue": self.is_overdue,
            "is_completed": self.is_completed,
        }


@dataclass
class CoalitionMember:
    """Represents a member of a coalition."""

    agent_id: str
    role: CoalitionRole = CoalitionRole.CONTRIBUTOR
    joined_at: datetime = field(default_factory=datetime.utcnow)
    # Contributions and commitments
    contribution_score: float = 0.0
    resource_commitments: Dict[str, float] = field(default_factory=dict)
    capability_contributions: Set[str] = field(default_factory=set)
    # Participation tracking
    voting_weight: float = 1.0
    participation_rate: float = 1.0
    last_active: datetime = field(default_factory=datetime.utcnow)
    # Performance metrics
    goals_completed: int = 0
    goals_failed: int = 0
    reliability_score: float = 1.0
    # Status
    is_active: bool = True
    departure_intent: bool = False
    departure_reason: Optional[str] = None

    @property
    def tenure_days(self) -> int:
        """Get member tenure in days."""
        return (datetime.utcnow() - self.joined_at).days

    @property
    def success_rate(self) -> float:
        """Calculate success rate for completed goals."""
        total_goals = self.goals_completed + self.goals_failed
        if total_goals == 0:
            return 1.0
        return self.goals_completed / total_goals

    @property
    def is_leader(self) -> bool:
        """Check if member is in a leadership role."""
        return self.role in [CoalitionRole.LEADER, CoalitionRole.COORDINATOR]

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_active = datetime.utcnow()

    def signal_departure(self, reason: Optional[str] = None) -> None:
        """Signal intent to leave coalition."""
        self.departure_intent = True
        self.departure_reason = reason
        logger.info(f"Member {self.agent_id} signaled departure: {reason}")

    def complete_goal(self, success: bool) -> None:
        """Record goal completion."""
        if success:
            self.goals_completed += 1
        else:
            self.goals_failed += 1
        # Update reliability based on recent performance
        recent_success_rate = self.success_rate
        self.reliability_score = 0.7 * self.reliability_score + 0.3 * recent_success_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "joined_at": self.joined_at.isoformat(),
            "contribution_score": self.contribution_score,
            "resource_commitments": self.resource_commitments,
            "capability_contributions": list(self.capability_contributions),
            "voting_weight": self.voting_weight,
            "participation_rate": self.participation_rate,
            "last_active": self.last_active.isoformat(),
            "goals_completed": self.goals_completed,
            "goals_failed": self.goals_failed,
            "reliability_score": self.reliability_score,
            "is_active": self.is_active,
            "departure_intent": self.departure_intent,
            "departure_reason": self.departure_reason,
            "tenure_days": self.tenure_days,
            "success_rate": self.success_rate,
            "is_leader": self.is_leader,
        }


@dataclass
class Coalition:
    """
    Main coalition class representing a group of agents working together.
    """

    coalition_id: str
    name: str
    description: str = ""
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: CoalitionStatus = CoalitionStatus.FORMING
    # Membership
    members: Dict[str, CoalitionMember] = field(default_factory=dict)
    max_members: int = 10
    min_members: int = 2
    # Goals and objectives
    goals: Dict[str, CoalitionGoal] = field(default_factory=dict)
    primary_goal_id: Optional[str] = None
    # Governance
    consensus_threshold: float = 0.66  # 2/3 majority
    voting_system: str = "weighted"  # weighted, equal, proportional
    leader_id: Optional[str] = None
    # Resources
    shared_resources: Dict[str, float] = field(default_factory=dict)
    resource_allocation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Performance tracking
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0
    efficiency_score: float = 0.0
    # Communication and coordination
    communication_protocols: Dict[str, Any] = field(default_factory=dict)
    coordination_mechanisms: List[str] = field(default_factory=list)
    # Business context
    business_opportunity_id: Optional[str] = None
    value_distribution_model: str = "proportional"  # proportional, equal, contribution-based

    def __post_init__(self):
        if not self.coalition_id:
            self.coalition_id = f"coalition_{uuid.uuid4().hex[:8]}"

    @property
    def member_count(self) -> int:
        """Get current active member count."""
        return len([m for m in self.members.values() if m.is_active])

    @property
    def is_viable(self) -> bool:
        """Check if coalition meets minimum viability requirements."""
        active_members = self.member_count
        return self.min_members <= active_members <= self.max_members and self.status in [
            CoalitionStatus.FORMING,
            CoalitionStatus.ACTIVE,
        ]

    @property
    def combined_capabilities(self) -> Set[str]:
        """Get all capabilities available in the coalition."""
        capabilities = set()
        for member in self.members.values():
            if member.is_active:
                capabilities.update(member.capability_contributions)
        return capabilities

    @property
    def total_resources(self) -> Dict[str, float]:
        """Calculate total available resources."""
        totals = dict(self.shared_resources)
        for member in self.members.values():
            if member.is_active:
                for resource, amount in member.resource_commitments.items():
                    totals[resource] = totals.get(resource, 0) + amount
        return totals

    def add_member(
        self,
        agent_id: str,
        role: CoalitionRole = CoalitionRole.CONTRIBUTOR,
        capabilities: Optional[Set[str]] = None,
        resources: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Add a new member to the coalition."""
        if self.member_count >= self.max_members:
            logger.warning(f"Cannot add member {agent_id}: coalition at capacity")
            return False
        if agent_id in self.members:
            logger.warning(f"Agent {agent_id} already a member")
            return False
        member = CoalitionMember(
            agent_id=agent_id,
            role=role,
            capability_contributions=capabilities or set(),
            resource_commitments=resources or {},
        )
        self.members[agent_id] = member
        # Set as leader if first member and no leader assigned
        if not self.leader_id and role == CoalitionRole.LEADER:
            self.leader_id = agent_id
        # Update status if we now meet minimum requirements
        if self.status == CoalitionStatus.FORMING and self.member_count >= self.min_members:
            self.status = CoalitionStatus.ACTIVE
        logger.info(
            f"Added member {agent_id} with role {role.value} to coalition {self.coalition_id}"
        )
        return True

    def remove_member(self, agent_id: str, reason: str = "voluntary") -> bool:
        """Remove a member from the coalition."""
        if agent_id not in self.members:
            return False
        member = self.members[agent_id]
        member.is_active = False
        member.departure_reason = reason
        # Handle leadership transition
        if self.leader_id == agent_id:
            self._elect_new_leader()
        # Check if coalition is still viable
        if self.member_count < self.min_members:
            self.status = CoalitionStatus.DISSOLVING
            logger.warning(f"Coalition {self.coalition_id} below minimum members, dissolving")
        logger.info(f"Removed member {agent_id} from coalition {self.coalition_id}: {reason}")
        return True

    def add_goal(self, goal: CoalitionGoal) -> None:
        """Add a goal to the coalition."""
        self.goals[goal.goal_id] = goal
        # Set as primary goal if it's the first one
        if not self.primary_goal_id:
            self.primary_goal_id = goal.goal_id
        logger.info(f"Added goal {goal.goal_id} to coalition {self.coalition_id}")

    def vote_on_goal(self, goal_id: str, member_id: str, support: bool) -> None:
        """Record a member's vote on a goal."""
        if goal_id not in self.goals or member_id not in self.members:
            return
        goal = self.goals[goal_id]
        member = self.members[member_id]
        if not member.is_active:
            return
        goal.add_vote(member_id, support)
        # Check if consensus is reached
        consensus = goal.calculate_consensus(self.member_count)
        if consensus >= self.consensus_threshold:
            if goal.status == CoalitionGoalStatus.PROPOSED:
                goal.status = CoalitionGoalStatus.ACCEPTED
                logger.info(f"Goal {goal_id} accepted by coalition {self.coalition_id}")

    def update_goal_progress(
        self, goal_id: str, progress: float, member_id: Optional[str] = None
    ) -> None:
        """Update progress on a coalition goal."""
        if goal_id in self.goals:
            self.goals[goal_id].update_progress(progress, member_id)
            self._update_performance_metrics()

    def allocate_resources(self, allocation: Dict[str, Dict[str, float]]) -> bool:
        """Allocate resources for goals or members."""
        # Validate allocation doesn't exceed available resources
        total_allocated = {}
        for allocations in allocation.values():
            for resource, amount in allocations.items():
                total_allocated[resource] = total_allocated.get(resource, 0) + amount
        available = self.total_resources
        for resource, amount in total_allocated.items():
            if amount > available.get(resource, 0):
                logger.warning(
                    f"Insufficient {resource}: need {amount}, " f"have {available.get(resource, 0)}"
                )
                return False
        self.resource_allocation = allocation
        return True

    def calculate_member_contributions(self) -> Dict[str, float]:
        """Calculate contribution scores for all members."""
        contributions = {}
        for member_id, member in self.members.items():
            if not member.is_active:
                continue
            # Base contribution from resources
            resource_contribution = sum(member.resource_commitments.values())
            # Capability contribution (estimated value)
            capability_contribution = len(member.capability_contributions) * 100
            # Performance contribution
            performance_contribution = member.success_rate * member.reliability_score * 1000
            # Leadership bonus
            leadership_bonus = 500 if member.is_leader else 0
            total_contribution = (
                resource_contribution
                + capability_contribution
                + performance_contribution
                + leadership_bonus
            )
            contributions[member_id] = total_contribution
            member.contribution_score = total_contribution
        return contributions

    def distribute_value(self, total_value: float) -> Dict[str, float]:
        """Distribute value among members based on the distribution model."""
        if not self.members:
            return {}
        active_members = [m for m in self.members.values() if m.is_active]
        distribution = {}
        if self.value_distribution_model == "equal":
            # Equal distribution
            per_member = total_value / len(active_members)
            for member in active_members:
                distribution[member.agent_id] = per_member
        elif self.value_distribution_model == "contribution-based":
            # Distribution based on contribution scores
            contributions = self.calculate_member_contributions()
            total_contribution = sum(contributions.values())
            if total_contribution > 0:
                for member_id, contribution in contributions.items():
                    distribution[member_id] = total_value * (contribution / total_contribution)
            else:
                # Fallback to equal distribution
                per_member = total_value / len(active_members)
                for member in active_members:
                    distribution[member.agent_id] = per_member
        else:  # proportional (default)
            # Distribution based on resource commitments and voting weight
            total_weight = sum(m.voting_weight for m in active_members)
            for member in active_members:
                distribution[member.agent_id] = total_value * (member.voting_weight / total_weight)
        return distribution

    def _elect_new_leader(self) -> None:
        """Elect a new leader when current leader leaves."""
        candidates = [
            m
            for m in self.members.values()
            if m.is_active and m.role in [CoalitionRole.COORDINATOR, CoalitionRole.CONTRIBUTOR]
        ]
        if candidates:
            # Select member with highest contribution score
            new_leader = max(candidates, key=lambda m: m.contribution_score)
            new_leader.role = CoalitionRole.LEADER
            self.leader_id = new_leader.agent_id
            logger.info(
                f"Elected {new_leader.agent_id} as new leader of coalition {self.coalition_id}"
            )
        else:
            self.leader_id = None

    def _update_performance_metrics(self) -> None:
        """Update coalition performance metrics."""
        if not self.goals:
            return
        # Calculate success rate
        completed_goals = sum(1 for g in self.goals.values() if g.is_completed)
        failed_goals = sum(1 for g in self.goals.values() if g.status == CoalitionGoalStatus.FAILED)
        total_finished = completed_goals + failed_goals
        if total_finished > 0:
            self.success_rate = completed_goals / total_finished
        # Calculate efficiency (progress rate)
        total_progress = sum(g.current_progress for g in self.goals.values())
        goal_count = len(self.goals)
        self.efficiency_score = total_progress / goal_count if goal_count > 0 else 0.0
        # Update individual performance metrics
        self.performance_metrics.update(
            {
                "average_member_contribution": sum(
                    m.contribution_score for m in self.members.values()
                )
                / len(self.members),
                "average_participation": sum(m.participation_rate for m in self.members.values())
                / len(self.members),
                "resource_utilization": len(self.resource_allocation)
                / max(1, len(self.total_resources)),
                "capability_coverage": len(self.combined_capabilities),
            }
        )

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        active_members = [m for m in self.members.values() if m.is_active]
        return {
            "coalition_id": self.coalition_id,
            "name": self.name,
            "status": self.status.value,
            "member_count": len(active_members),
            "leader": self.leader_id,
            "goals": {
                "total": len(self.goals),
                "completed": sum(1 for g in self.goals.values() if g.is_completed),
                "in_progress": sum(
                    1 for g in self.goals.values() if g.status == CoalitionGoalStatus.IN_PROGRESS
                ),
                "failed": sum(
                    1 for g in self.goals.values() if g.status == CoalitionGoalStatus.FAILED
                ),
            },
            "performance": {
                "success_rate": self.success_rate,
                "efficiency_score": self.efficiency_score,
            },
            "resources": {
                "total_types": len(self.total_resources),
                "allocated_goals": len(self.resource_allocation),
            },
            "capabilities": list(self.combined_capabilities),
            "is_viable": self.is_viable,
            "age_days": (datetime.utcnow() - self.created_at).days,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "coalition_id": self.coalition_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "members": {mid: member.to_dict() for mid, member in self.members.items()},
            "max_members": self.max_members,
            "min_members": self.min_members,
            "goals": {gid: goal.to_dict() for gid, goal in self.goals.items()},
            "primary_goal_id": self.primary_goal_id,
            "consensus_threshold": self.consensus_threshold,
            "voting_system": self.voting_system,
            "leader_id": self.leader_id,
            "shared_resources": self.shared_resources,
            "resource_allocation": self.resource_allocation,
            "performance_metrics": self.performance_metrics,
            "success_rate": self.success_rate,
            "efficiency_score": self.efficiency_score,
            "communication_protocols": self.communication_protocols,
            "coordination_mechanisms": self.coordination_mechanisms,
            "business_opportunity_id": self.business_opportunity_id,
            "value_distribution_model": self.value_distribution_model,
            "status_summary": self.get_status_summary(),
        }

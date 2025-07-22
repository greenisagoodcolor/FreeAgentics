"""Core coalition data structures and functionality."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CoalitionStatus(Enum):
    """Status of a coalition."""

    FORMING = "forming"
    ACTIVE = "active"
    DISBANDING = "disbanding"
    DISSOLVED = "dissolved"


class CoalitionRole(Enum):
    """Roles within a coalition."""

    LEADER = "leader"
    COORDINATOR = "coordinator"
    MEMBER = "member"
    OBSERVER = "observer"


@dataclass
class CoalitionMember:
    """Represents a member of a coalition."""

    agent_id: str
    role: CoalitionRole
    capabilities: List[str] = field(default_factory=list)
    contribution_score: float = 0.0
    join_time: datetime = field(default_factory=datetime.now)
    last_activity: Optional[datetime] = None
    trust_score: float = 1.0  # 0.0 to 1.0
    active: bool = True


@dataclass
class CoalitionObjective:
    """Represents an objective that a coalition aims to achieve."""

    objective_id: str
    description: str
    required_capabilities: List[str]
    priority: float  # 0.0 to 1.0
    deadline: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class Coalition:
    """Represents a coalition of agents working together."""

    def __init__(
        self,
        coalition_id: str,
        name: str,
        objectives: Optional[List[CoalitionObjective]] = None,
        max_size: Optional[int] = None,
    ):
        """Initialize a coalition.

        Args:
            coalition_id: Unique identifier for the coalition
            name: Human-readable name
            objectives: List of objectives for this coalition
            max_size: Maximum number of members allowed
        """
        self.coalition_id = coalition_id
        self.name = name
        self.objectives = objectives or []
        self.max_size = max_size

        # Member management
        self.members: Dict[str, CoalitionMember] = {}
        self.leader_id: Optional[str] = None

        # Coalition state
        self.status = CoalitionStatus.FORMING
        self.created_at = datetime.now()
        self.last_modified = datetime.now()

        # Performance metrics
        self.performance_score = 0.0
        self.coordination_efficiency = 0.0
        self.objective_completion_rate = 0.0

        # Communication and coordination
        self.communication_history: List[Dict[str, Any]] = []
        self.decision_log: List[Dict[str, Any]] = []

        logger.info(f"Created coalition {coalition_id}: {name}")

    def add_member(
        self,
        agent_id: str,
        role: CoalitionRole = CoalitionRole.MEMBER,
        capabilities: Optional[List[str]] = None,
    ) -> bool:
        """Add a member to the coalition.

        Args:
            agent_id: ID of the agent to add
            role: Role in the coalition
            capabilities: List of capabilities the agent brings

        Returns:
            True if member was added successfully
        """
        if self.max_size and len(self.members) >= self.max_size:
            logger.warning(f"Coalition {self.coalition_id} is at maximum capacity")
            return False

        if agent_id in self.members:
            logger.warning(f"Agent {agent_id} is already in coalition {self.coalition_id}")
            return False

        member = CoalitionMember(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities or [],
        )

        self.members[agent_id] = member

        # Set as leader if first member or explicitly assigned
        if role == CoalitionRole.LEADER or (not self.leader_id and len(self.members) == 1):
            self.leader_id = agent_id
            member.role = CoalitionRole.LEADER

        self.last_modified = datetime.now()
        self._log_decision(f"Added member {agent_id} with role {role.value}")

        logger.info(f"Added {agent_id} to coalition {self.coalition_id}")
        return True

    def remove_member(self, agent_id: str) -> bool:
        """Remove a member from the coalition.

        Args:
            agent_id: ID of the agent to remove

        Returns:
            True if member was removed successfully
        """
        if agent_id not in self.members:
            logger.warning(f"Agent {agent_id} not found in coalition {self.coalition_id}")
            return False

        was_leader = self.leader_id == agent_id
        del self.members[agent_id]

        # Choose new leader if needed
        if was_leader:
            self._elect_new_leader()

        self.last_modified = datetime.now()
        self._log_decision(f"Removed member {agent_id}")

        logger.info(f"Removed {agent_id} from coalition {self.coalition_id}")

        # Dissolve coalition if empty
        if not self.members:
            self.status = CoalitionStatus.DISSOLVED
            self._log_decision("Coalition dissolved - no members remaining")

        return True

    def _elect_new_leader(self) -> None:
        """Elect a new leader for the coalition."""
        if not self.members:
            self.leader_id = None
            return

        # Find best candidate based on contribution score and trust
        best_candidate = max(
            self.members.values(),
            key=lambda m: (m.contribution_score * m.trust_score, m.join_time),
            default=None,
        )

        if best_candidate:
            # Demote old leader if still in coalition
            if self.leader_id and self.leader_id in self.members:
                self.members[self.leader_id].role = CoalitionRole.MEMBER

            # Promote new leader
            best_candidate.role = CoalitionRole.LEADER
            self.leader_id = best_candidate.agent_id

            self._log_decision(f"Elected new leader: {self.leader_id}")
            logger.info(f"Elected new leader for coalition {self.coalition_id}: {self.leader_id}")

    def add_objective(self, objective: CoalitionObjective) -> bool:
        """Add an objective to the coalition.

        Args:
            objective: Objective to add

        Returns:
            True if objective was added successfully
        """
        # Check if we have required capabilities
        available_capabilities = set()
        for member in self.members.values():
            available_capabilities.update(member.capabilities)

        required_capabilities = set(objective.required_capabilities)
        if not required_capabilities.issubset(available_capabilities):
            missing = required_capabilities - available_capabilities
            logger.warning(
                f"Coalition {self.coalition_id} missing capabilities for objective "
                f"{objective.objective_id}: {missing}"
            )
            return False

        self.objectives.append(objective)
        self.last_modified = datetime.now()
        self._log_decision(f"Added objective {objective.objective_id}")

        logger.info(f"Added objective {objective.objective_id} to coalition {self.coalition_id}")
        return True

    def update_objective_progress(self, objective_id: str, progress: float) -> bool:
        """Update progress on an objective.

        Args:
            objective_id: ID of the objective
            progress: Progress value (0.0 to 1.0)

        Returns:
            True if update was successful
        """
        for objective in self.objectives:
            if objective.objective_id == objective_id:
                objective.progress = max(0.0, min(1.0, progress))

                if objective.progress >= 1.0:
                    objective.completed = True
                    self._log_decision(f"Completed objective {objective_id}")
                    logger.info(
                        f"Completed objective {objective_id} in coalition {self.coalition_id}"
                    )

                self.last_modified = datetime.now()
                self._update_performance_metrics()
                return True

        logger.warning(f"Objective {objective_id} not found in coalition {self.coalition_id}")
        return False

    def _update_performance_metrics(self) -> None:
        """Update coalition performance metrics."""
        if not self.objectives:
            self.objective_completion_rate = 1.0
        else:
            completed = sum(1 for obj in self.objectives if obj.completed)
            self.objective_completion_rate = completed / len(self.objectives)

        # Calculate coordination efficiency based on member activity
        active_members = sum(1 for member in self.members.values() if member.active)
        total_members = len(self.members)

        if total_members > 0:
            self.coordination_efficiency = active_members / total_members
        else:
            self.coordination_efficiency = 0.0

        # Overall performance score combines completion rate and coordination
        self.performance_score = (
            0.7 * self.objective_completion_rate + 0.3 * self.coordination_efficiency
        )

    def get_capabilities(self) -> Set[str]:
        """Get all capabilities available in the coalition.

        Returns:
            Set of all member capabilities
        """
        capabilities = set()
        for member in self.members.values():
            capabilities.update(member.capabilities)
        return capabilities

    def get_member_by_role(self, role: CoalitionRole) -> List[CoalitionMember]:
        """Get all members with a specific role.

        Args:
            role: Role to search for

        Returns:
            List of members with the specified role
        """
        return [member for member in self.members.values() if member.role == role]

    def can_achieve_objective(self, objective: CoalitionObjective) -> bool:
        """Check if coalition can achieve an objective.

        Args:
            objective: Objective to check

        Returns:
            True if coalition has required capabilities
        """
        available_capabilities = self.get_capabilities()
        required_capabilities = set(objective.required_capabilities)
        return required_capabilities.issubset(available_capabilities)

    def _log_decision(self, decision: str) -> None:
        """Log a decision made by the coalition."""
        self.decision_log.append(
            {
                "timestamp": datetime.now(),
                "decision": decision,
                "leader_id": self.leader_id,
                "member_count": len(self.members),
            }
        )

    def add_communication(
        self,
        sender_id: str,
        message: str,
        recipients: Optional[List[str]] = None,
    ) -> None:
        """Add a communication record.

        Args:
            sender_id: ID of the sender
            message: Message content
            recipients: List of recipient IDs (None for broadcast)
        """
        comm_record = {
            "timestamp": datetime.now(),
            "sender_id": sender_id,
            "message": message,
            "recipients": recipients or list(self.members.keys()),
            "broadcast": recipients is None,
        }

        self.communication_history.append(comm_record)

        # Update last activity for sender
        if sender_id in self.members:
            self.members[sender_id].last_activity = datetime.now()

    def activate(self) -> None:
        """Activate the coalition."""
        if self.status == CoalitionStatus.FORMING and self.members:
            self.status = CoalitionStatus.ACTIVE
            self.last_modified = datetime.now()
            self._log_decision("Coalition activated")
            logger.info(f"Activated coalition {self.coalition_id}")

    def disband(self) -> None:
        """Initiate coalition disbanding."""
        self.status = CoalitionStatus.DISBANDING
        self.last_modified = datetime.now()
        self._log_decision("Coalition disbanding initiated")
        logger.info(f"Disbanding coalition {self.coalition_id}")

    def get_status(self) -> Dict[str, Any]:
        """Get current coalition status.

        Returns:
            Status dictionary
        """
        return {
            "coalition_id": self.coalition_id,
            "name": self.name,
            "status": self.status.value,
            "member_count": len(self.members),
            "leader_id": self.leader_id,
            "objectives_count": len(self.objectives),
            "completed_objectives": sum(1 for obj in self.objectives if obj.completed),
            "performance_score": self.performance_score,
            "coordination_efficiency": self.coordination_efficiency,
            "objective_completion_rate": self.objective_completion_rate,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "capabilities": list(self.get_capabilities()),
        }

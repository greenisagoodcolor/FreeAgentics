"""Coalition manager for coordinating coalition formation and lifecycle."""

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from coalitions.coalition import Coalition, CoalitionObjective, CoalitionRole, CoalitionStatus
from coalitions.formation_strategies import (
    AgentProfile,
    FormationResult,
    FormationStrategy,
    GreedyFormation,
    HierarchicalFormation,
    OptimalFormation,
)

logger = logging.getLogger(__name__)


@dataclass
class CoalitionEvent:
    """Event related to coalition lifecycle."""

    event_type: str  # created, activated, disbanded, member_added, etc.
    coalition_id: str
    timestamp: datetime
    details: Dict[str, Any]


class CoalitionManager:
    """Manages coalition formation, lifecycle, and coordination."""

    def __init__(self):
        """Initialize the coalition manager."""
        self.coalitions: Dict[str, Coalition] = {}
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.pending_objectives: List[CoalitionObjective] = []

        # Formation strategies
        self.strategies: Dict[str, FormationStrategy] = {
            "greedy": GreedyFormation(),
            "optimal": OptimalFormation(),
            "hierarchical": HierarchicalFormation(),
        }
        self.default_strategy = "greedy"

        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_history: List[CoalitionEvent] = []

        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitor_interval = 30.0  # seconds

        # Performance tracking
        self.formation_stats = {
            "total_formations": 0,
            "successful_formations": 0,
            "average_formation_time": 0.0,
            "average_coalition_size": 0.0,
        }

        logger.info("Coalition manager initialized")

    def register_agent(
        self,
        agent_id: str,
        capabilities: List[str],
        capacity: float = 1.0,
        reputation: float = 1.0,
        preferences: Optional[Dict[str, float]] = None,
        max_coalitions: int = 3,
    ) -> bool:
        """Register an agent with the coalition manager.

        Args:
            agent_id: Unique identifier for the agent
            capabilities: List of agent capabilities
            capacity: Agent's work capacity (0.0 to 1.0)
            reputation: Agent's reputation score (0.0 to 1.0)
            preferences: Preferences for working with other agents
            max_coalitions: Maximum number of coalitions agent can join

        Returns:
            True if registration was successful
        """
        if agent_id in self.agent_profiles:
            logger.warning(f"Agent {agent_id} already registered")
            return False

        profile = AgentProfile(
            agent_id=agent_id,
            capabilities=capabilities,
            capacity=max(0.0, min(1.0, capacity)),
            reputation=max(0.0, min(1.0, reputation)),
            preferences=preferences or {},
            current_coalitions=[],
            max_coalitions=max_coalitions,
        )

        self.agent_profiles[agent_id] = profile
        self._emit_event("agent_registered", "", {"agent_id": agent_id})

        logger.info(f"Registered agent {agent_id} with {len(capabilities)} capabilities")
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the coalition manager.

        Args:
            agent_id: Agent to unregister

        Returns:
            True if unregistration was successful
        """
        if agent_id not in self.agent_profiles:
            logger.warning(f"Agent {agent_id} not found")
            return False

        profile = self.agent_profiles[agent_id]

        # Remove agent from all current coalitions
        for coalition_id in profile.current_coalitions.copy():
            if coalition_id in self.coalitions:
                self.coalitions[coalition_id].remove_member(agent_id)

        del self.agent_profiles[agent_id]
        self._emit_event("agent_unregistered", "", {"agent_id": agent_id})

        logger.info(f"Unregistered agent {agent_id}")
        return True

    def add_objective(self, objective: CoalitionObjective) -> bool:
        """Add an objective for coalition formation.

        Args:
            objective: Objective to add

        Returns:
            True if objective was added successfully
        """
        # Check if any existing coalition can handle this objective
        for coalition in self.coalitions.values():
            if (
                coalition.status == CoalitionStatus.ACTIVE
                and coalition.can_achieve_objective(objective)
                and len(coalition.objectives) < 3
            ):  # Limit objectives per coalition

                if coalition.add_objective(objective):
                    self._emit_event(
                        "objective_assigned",
                        coalition.coalition_id,
                        {"objective_id": objective.objective_id},
                    )
                    logger.info(
                        f"Assigned objective {objective.objective_id} to existing coalition {coalition.coalition_id}"
                    )
                    return True

        # Add to pending objectives for next formation round
        self.pending_objectives.append(objective)
        self._emit_event("objective_added", "", {"objective_id": objective.objective_id})

        logger.info(f"Added objective {objective.objective_id} to pending list")
        return True

    def form_coalitions(
        self,
        strategy_name: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        objectives: Optional[List[CoalitionObjective]] = None,
    ) -> FormationResult:
        """Form coalitions for pending objectives.

        Args:
            strategy_name: Name of formation strategy to use
            constraints: Additional constraints for formation
            objectives: Specific objectives to form for (uses pending if None)

        Returns:
            Formation result
        """
        strategy_name = strategy_name or self.default_strategy

        if strategy_name not in self.strategies:
            logger.error(f"Unknown strategy: {strategy_name}")
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = self.strategies[strategy_name]

        # Use specified objectives or pending ones
        objectives_to_form = objectives or self.pending_objectives.copy()

        if not objectives_to_form:
            logger.info("No objectives to form coalitions for")
            return FormationResult([], [], 0.0, 1.0, 0.0)

        # Get available agents
        available_agents = []
        for agent_profile in self.agent_profiles.values():
            # Only include agents that can join more coalitions
            if len(agent_profile.current_coalitions) < agent_profile.max_coalitions:
                available_agents.append(agent_profile)

        if not available_agents:
            logger.warning("No available agents for coalition formation")
            return FormationResult([], [], 0.0, 0.0, 0.0)

        logger.info(
            f"Forming coalitions using {strategy_name} strategy for {len(objectives_to_form)} objectives"
        )

        # Perform formation
        start_time = datetime.now()
        result = strategy.form_coalitions(available_agents, objectives_to_form, constraints)
        formation_time = (datetime.now() - start_time).total_seconds()

        # Register new coalitions
        for coalition in result.coalitions:
            if coalition.coalition_id not in self.coalitions:
                self.coalitions[coalition.coalition_id] = coalition
                self._emit_event(
                    "coalition_created",
                    coalition.coalition_id,
                    {
                        "member_count": len(coalition.members),
                        "objectives_count": len(coalition.objectives),
                    },
                )

        # Remove formed objectives from pending list
        if objectives is None:  # Only remove from pending if we used pending objectives
            formed_objective_ids = set()
            for coalition in result.coalitions:
                for obj in coalition.objectives:
                    formed_objective_ids.add(obj.objective_id)

            self.pending_objectives = [
                obj
                for obj in self.pending_objectives
                if obj.objective_id not in formed_objective_ids
            ]

        # Update statistics
        self._update_formation_stats(result, formation_time)

        self._emit_event(
            "formation_completed",
            "",
            {
                "strategy": strategy_name,
                "coalitions_formed": len(result.coalitions),
                "formation_time": formation_time,
                "objective_coverage": result.objective_coverage,
            },
        )

        logger.info(
            f"Formation completed: {len(result.coalitions)} coalitions formed, "
            f"{result.objective_coverage:.1%} objective coverage"
        )

        return result

    def get_coalition(self, coalition_id: str) -> Optional[Coalition]:
        """Get a coalition by ID.

        Args:
            coalition_id: Coalition identifier

        Returns:
            Coalition object or None if not found
        """
        return self.coalitions.get(coalition_id)

    def get_agent_coalitions(self, agent_id: str) -> List[Coalition]:
        """Get all coalitions an agent is part of.

        Args:
            agent_id: Agent identifier

        Returns:
            List of coalitions the agent belongs to
        """
        coalitions = []

        for coalition in self.coalitions.values():
            if agent_id in coalition.members:
                coalitions.append(coalition)

        return coalitions

    def dissolve_coalition(self, coalition_id: str, reason: str = "Manual dissolution") -> bool:
        """Dissolve a coalition.

        Args:
            coalition_id: Coalition to dissolve
            reason: Reason for dissolution

        Returns:
            True if dissolution was successful
        """
        if coalition_id not in self.coalitions:
            logger.warning(f"Coalition {coalition_id} not found")
            return False

        coalition = self.coalitions[coalition_id]

        # Remove agents from coalition
        for agent_id in list(coalition.members.keys()):
            coalition.remove_member(agent_id)

            # Update agent profile
            if agent_id in self.agent_profiles:
                profile = self.agent_profiles[agent_id]
                if coalition_id in profile.current_coalitions:
                    profile.current_coalitions.remove(coalition_id)

        # Move incomplete objectives back to pending
        for objective in coalition.objectives:
            if not objective.completed:
                self.pending_objectives.append(objective)

        coalition.status = CoalitionStatus.DISSOLVED
        del self.coalitions[coalition_id]

        self._emit_event("coalition_dissolved", coalition_id, {"reason": reason})

        logger.info(f"Dissolved coalition {coalition_id}: {reason}")
        return True

    def update_agent_reputation(self, agent_id: str, reputation: float) -> bool:
        """Update an agent's reputation score.

        Args:
            agent_id: Agent to update
            reputation: New reputation score (0.0 to 1.0)

        Returns:
            True if update was successful
        """
        if agent_id not in self.agent_profiles:
            logger.warning(f"Agent {agent_id} not found")
            return False

        old_reputation = self.agent_profiles[agent_id].reputation
        self.agent_profiles[agent_id].reputation = max(0.0, min(1.0, reputation))

        self._emit_event(
            "agent_reputation_updated",
            "",
            {
                "agent_id": agent_id,
                "old_reputation": old_reputation,
                "new_reputation": reputation,
            },
        )

        logger.info(
            f"Updated reputation for agent {agent_id}: {old_reputation:.2f} -> {reputation:.2f}"
        )
        return True

    def add_agent_preference(
        self, agent_id: str, preferred_agent_id: str, preference: float
    ) -> bool:
        """Add or update an agent's preference for working with another agent.

        Args:
            agent_id: Agent setting the preference
            preferred_agent_id: Agent being rated
            preference: Preference score (0.0 to 1.0)

        Returns:
            True if preference was set successfully
        """
        if agent_id not in self.agent_profiles:
            logger.warning(f"Agent {agent_id} not found")
            return False

        self.agent_profiles[agent_id].preferences[preferred_agent_id] = max(
            0.0, min(1.0, preference)
        )

        logger.info(f"Set preference for {agent_id} -> {preferred_agent_id}: {preference:.2f}")
        return True

    def start_monitoring(self) -> bool:
        """Start background monitoring of coalitions.

        Returns:
            True if monitoring was started successfully
        """
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return False

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()

        logger.info("Started coalition monitoring")
        return True

    def stop_monitoring(self) -> bool:
        """Stop background monitoring of coalitions.

        Returns:
            True if monitoring was stopped successfully
        """
        if not self._monitoring_active:
            logger.warning("Monitoring not active")
            return False

        self._monitoring_active = False

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)

        logger.info("Stopped coalition monitoring")
        return True

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self._monitor_coalitions()
                threading.Event().wait(self._monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                threading.Event().wait(self._monitor_interval)

    def _monitor_coalitions(self):
        """Monitor coalition health and performance."""
        current_time = datetime.now()

        for coalition_id, coalition in list(self.coalitions.items()):
            # Check for inactive coalitions
            time_since_modification = current_time - coalition.last_modified

            if time_since_modification > timedelta(hours=24):
                if coalition.status == CoalitionStatus.ACTIVE:
                    logger.warning(f"Coalition {coalition_id} has been inactive for 24 hours")
                    self._emit_event(
                        "coalition_inactive",
                        coalition_id,
                        {"inactive_hours": time_since_modification.total_seconds() / 3600},
                    )

            # Check for coalitions with completed objectives
            if all(obj.completed for obj in coalition.objectives) and coalition.objectives:
                if coalition.status == CoalitionStatus.ACTIVE:
                    coalition.status = CoalitionStatus.DISBANDING
                    self._emit_event("coalition_objectives_completed", coalition_id, {})

                    # Schedule dissolution after a grace period
                    logger.info(f"Coalition {coalition_id} completed all objectives")

            # Check for coalitions with expired objectives
            expired_objectives = []
            for obj in coalition.objectives:
                if obj.deadline and current_time > obj.deadline and not obj.completed:
                    expired_objectives.append(obj.objective_id)

            if expired_objectives:
                self._emit_event(
                    "objectives_expired", coalition_id, {"expired_objectives": expired_objectives}
                )
                logger.warning(
                    f"Coalition {coalition_id} has expired objectives: {expired_objectives}"
                )

            # Update coalition performance metrics
            coalition._update_performance_metrics()

    def add_event_handler(self, event_type: str, handler: Callable[[CoalitionEvent], None]):
        """Add an event handler for coalition events.

        Args:
            event_type: Type of event to handle
            handler: Callback function to handle the event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)
        logger.info(f"Added event handler for {event_type}")

    def _emit_event(self, event_type: str, coalition_id: str, details: Dict[str, Any]):
        """Emit a coalition event.

        Args:
            event_type: Type of event
            coalition_id: Related coalition ID
            details: Event details
        """
        event = CoalitionEvent(
            event_type=event_type,
            coalition_id=coalition_id,
            timestamp=datetime.now(),
            details=details,
        )

        self.event_history.append(event)

        # Keep event history manageable
        if len(self.event_history) > 10000:
            self.event_history = self.event_history[-5000:]

        # Call event handlers
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")

    def _update_formation_stats(self, result: FormationResult, formation_time: float):
        """Update formation statistics."""
        self.formation_stats["total_formations"] += 1

        if result.coalitions:
            self.formation_stats["successful_formations"] += 1

            # Update average formation time
            old_avg_time = self.formation_stats["average_formation_time"]
            total_formations = self.formation_stats["total_formations"]
            self.formation_stats["average_formation_time"] = (
                old_avg_time * (total_formations - 1) + formation_time
            ) / total_formations

            # Update average coalition size
            total_members = sum(len(c.members) for c in result.coalitions)
            avg_size = total_members / len(result.coalitions)

            old_avg_size = self.formation_stats["average_coalition_size"]
            successful_formations = self.formation_stats["successful_formations"]
            self.formation_stats["average_coalition_size"] = (
                old_avg_size * (successful_formations - 1) + avg_size
            ) / successful_formations

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status.

        Returns:
            Status dictionary with system metrics
        """
        active_coalitions = sum(
            1 for c in self.coalitions.values() if c.status == CoalitionStatus.ACTIVE
        )

        total_agents = len(self.agent_profiles)
        assigned_agents = len(
            set(
                agent_id
                for coalition in self.coalitions.values()
                for agent_id in coalition.members.keys()
            )
        )

        return {
            "total_agents": total_agents,
            "assigned_agents": assigned_agents,
            "available_agents": total_agents - assigned_agents,
            "total_coalitions": len(self.coalitions),
            "active_coalitions": active_coalitions,
            "pending_objectives": len(self.pending_objectives),
            "monitoring_active": self._monitoring_active,
            "formation_stats": self.formation_stats.copy(),
            "event_count": len(self.event_history),
        }

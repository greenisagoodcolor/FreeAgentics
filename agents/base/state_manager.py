import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .data_model import Agent, AgentStatus, Position

"""
Agent State Management System
This module provides a robust state management system for agents with:
- Event-driven architecture for state changes
- Thread-safe operations for concurrent access
- State transition validation
- State history tracking
- Observer pattern for state change notifications
"""
logger = logging.getLogger(__name__)


class StateTransitionError(Exception):
    """Exception raised when an invalid state transition is attempted"""

    pass


class StateEventType(Enum):
    """Types of state events"""

    STATUS_CHANGE = "status_change"
    POSITION_UPDATE = "position_update"
    RESOURCE_UPDATE = "resource_update"
    GOAL_UPDATE = "goal_update"
    INTERACTION = "interaction"


@dataclass
class StateEvent:
    """Represents a state change event"""

    event_type: StateEventType
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """Represents a snapshot of agent state at a point in time"""

    agent_id: str
    timestamp: datetime
    status: AgentStatus
    position: Position
    energy: float
    health: float
    current_goal_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateTransitionValidator:
    """Validates state transitions"""

    VALID_TRANSITIONS: Dict[AgentStatus, set[AgentStatus]] = {
        AgentStatus.IDLE: {
            AgentStatus.MOVING,
            AgentStatus.INTERACTING,
            AgentStatus.PLANNING,
            AgentStatus.LEARNING,
            AgentStatus.OFFLINE,
        },
        AgentStatus.MOVING: {
            AgentStatus.IDLE,
            AgentStatus.INTERACTING,
            AgentStatus.OFFLINE,
            AgentStatus.ERROR,
        },
        AgentStatus.INTERACTING: {
            AgentStatus.IDLE,
            AgentStatus.MOVING,
            AgentStatus.OFFLINE,
            AgentStatus.ERROR,
        },
        AgentStatus.PLANNING: {
            AgentStatus.IDLE,
            AgentStatus.MOVING,
            AgentStatus.OFFLINE,
            AgentStatus.ERROR,
        },
        AgentStatus.LEARNING: {
            AgentStatus.IDLE,
            AgentStatus.PLANNING,
            AgentStatus.OFFLINE,
            AgentStatus.ERROR,
        },
        AgentStatus.OFFLINE: {AgentStatus.IDLE},
        AgentStatus.ERROR: {AgentStatus.IDLE, AgentStatus.OFFLINE},
    }

    @classmethod
    def is_valid_transition(cls, from_status: AgentStatus, to_status: AgentStatus) -> bool:
        """Check if a state transition is valid"""
        if from_status == to_status:
            return True
        valid_targets = cls.VALID_TRANSITIONS.get(from_status, set())
        return to_status in valid_targets

    @classmethod
    def get_valid_transitions(cls, from_status: AgentStatus) -> set[AgentStatus]:
        """Get all valid transitions from a given status"""
        return cls.VALID_TRANSITIONS.get(from_status, set()).copy()


class StateObserver:
    """Base class for state observers"""

    def on_state_change(self, event: StateEvent) -> None:
        """Called when a state change occurs"""
        raise NotImplementedError


class LoggingObserver(StateObserver):
    """Observer that logs state changes"""

    def on_state_change(self, event: StateEvent) -> None:
        """Log state change event"""
        logger.info(
            f"State change: Agent {event.agent_id} - {event.event_type.value} - "
            f"Old: {event.old_value}, New: {event.new_value}"
        )


class AsyncStateObserver(StateObserver):
    """Base class for async state observers"""

    async def on_state_change_async(self, event: StateEvent) -> None:
        """Async handler for state changes"""
        raise NotImplementedError

    def on_state_change(self, event: StateEvent) -> None:
        """Sync wrapper that schedules async handler"""
        asyncio.create_task(self.on_state_change_async(event))


class AgentStateManager:
    """Main state management system for agents"""

    def __init__(self, max_history_size: int = 1000) -> None:
        """
        Initialize the state manager
        Args:
            max_history_size: Maximum number of state events to keep in history
        """
        self._agents: Dict[str, Agent] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        self._observers: List[StateObserver] = []
        self._event_history: deque = deque(maxlen=max_history_size)
        self._state_snapshots: Dict[str, List[StateSnapshot]] = {}
        self._transition_callbacks: Dict[str, List[Dict[str, Any]]] = {}
        self.add_observer(LoggingObserver())

    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the state manager
        Args:
            agent: The agent to register
        """
        with self._global_lock:
            if agent.agent_id in self._agents:
                logger.warning(f"Agent {agent.agent_id} already registered")
                return
            self._agents[agent.agent_id] = agent
            self._locks[agent.agent_id] = threading.RLock()
            self._state_snapshots[agent.agent_id] = []
            self._create_snapshot(agent)
            logger.info(f"Registered agent {agent.agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the state manager
        Args:
            agent_id: ID of the agent to unregister
        """
        with self._global_lock:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not registered")
                return
            del self._agents[agent_id]
            del self._locks[agent_id]
            del self._state_snapshots[agent_id]
            self._transition_callbacks.pop(agent_id, None)
            logger.info(f"Unregistered agent {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by ID
        Args:
            agent_id: The agent's ID
        Returns:
            The agent or None if not found
        """
        return self._agents.get(agent_id)

    def update_agent_status(
        self, agent_id: str, new_status: AgentStatus, force: bool = False
    ) -> None:
        """
        Update an agent's status with validation
        Args:
            agent_id: The agent's ID
            new_status: The new status to set
            force: If True, bypass transition validation
        Raises:
            StateTransitionError: If the transition is invalid
        """
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        with self._locks[agent_id]:
            old_status = agent.status
            if not force and not StateTransitionValidator.is_valid_transition(
                old_status, new_status
            ):
                raise StateTransitionError(
                    f"Invalid transition from {old_status.value} to {new_status.value}"
                )
            agent.update_status(new_status)
            event = StateEvent(
                event_type=StateEventType.STATUS_CHANGE,
                agent_id=agent_id,
                old_value=old_status,
                new_value=new_status,
            )
            self._notify_observers(event)
            self._execute_transition_callbacks(agent_id, old_status, new_status)
            self._create_snapshot(agent)

    def update_agent_position(self, agent_id: str, new_position: Position) -> None:
        """
        Update an agent's position
        Args:
            agent_id: The agent's ID
            new_position: The new position
        """
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        with self._locks[agent_id]:
            old_position = agent.position
            agent.update_position(new_position)
            if agent.status == AgentStatus.IDLE and old_position.distance_to(new_position) > 0.1:
                self.update_agent_status(agent_id, AgentStatus.MOVING)
            event = StateEvent(
                event_type=StateEventType.POSITION_UPDATE,
                agent_id=agent_id,
                old_value=old_position,
                new_value=new_position,
            )
            self._notify_observers(event)

    def update_agent_resources(
        self,
        agent_id: str,
        energy_delta: Optional[float] = None,
        health_delta: Optional[float] = None,
    ) -> None:
        """
        Update an agent's resources
        Args:
            agent_id: The agent's ID
            energy_delta: Change in energy (positive or negative)
            health_delta: Change in health (positive or negative)
        """
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        with self._locks[agent_id]:
            old_energy = agent.resources.energy
            old_health = agent.resources.health
            if energy_delta is not None:
                if energy_delta < 0:
                    agent.resources.consume_energy(-energy_delta)
                else:
                    agent.resources.restore_energy(energy_delta)
            if health_delta is not None:
                new_health = max(0.0, min(100.0, agent.resources.health + health_delta))
                agent.resources.health = new_health
            if agent.resources.energy <= 0 or agent.resources.health <= 0:
                self.update_agent_status(agent_id, AgentStatus.ERROR, force=True)
            event = StateEvent(
                event_type=StateEventType.RESOURCE_UPDATE,
                agent_id=agent_id,
                old_value={"energy": old_energy, "health": old_health},
                new_value={"energy": agent.resources.energy, "health": agent.resources.health},
            )
            self._notify_observers(event)

    def register_transition_callback(
        self,
        agent_id: str,
        from_status: AgentStatus,
        to_status: AgentStatus,
        callback: Callable[[Agent], None],
    ) -> None:
        """
        Register a callback for specific state transitions
        Args:
            agent_id: The agent's ID
            from_status: The source status
            to_status: The target status
            callback: Function to call when transition occurs
        """
        if agent_id not in self._transition_callbacks:
            self._transition_callbacks[agent_id] = []
        self._transition_callbacks[agent_id].append(
            {"from": from_status, "to": to_status, "callback": callback}
        )

    def add_observer(self, observer: StateObserver) -> None:
        """Add an observer for state changes"""
        self._observers.append(observer)

    def remove_observer(self, observer: StateObserver) -> None:
        """Remove an observer"""
        if observer in self._observers:
            self._observers.remove(observer)

    def get_state_history(
        self,
        agent_id: Optional[str] = None,
        event_type: Optional[StateEventType] = None,
        limit: int = 100,
    ) -> List[StateEvent]:
        """
        Get state change history
        Args:
            agent_id: Filter by agent ID (None for all agents)
            event_type: Filter by event type (None for all types)
            limit: Maximum number of events to return
        Returns:
            List of state events
        """
        events = list(self._event_history)
        if agent_id is not None:
            events = [e for e in events if e.agent_id == agent_id]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def get_state_snapshots(self, agent_id: str, limit: int = 10) -> List[StateSnapshot]:
        """
        Get state snapshots for an agent
        Args:
            agent_id: The agent's ID
            limit: Maximum number of snapshots to return
        Returns:
            List of state snapshots
        """
        snapshots = self._state_snapshots.get(agent_id, [])
        return snapshots[-limit:]

    def get_agent_state_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Get a summary of an agent's current state
        Args:
            agent_id: The agent's ID
        Returns:
            Dictionary containing state summary
        """
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        with self._locks[agent_id]:
            return {
                "agent_id": agent_id,
                "name": agent.name,
                "status": agent.status.value,
                "position": {"x": agent.position.x, "y": agent.position.y, "z": agent.position.z},
                "resources": {
                    "energy": agent.resources.energy,
                    "health": agent.resources.health,
                    "memory_used": agent.resources.memory_used,
                    "memory_capacity": agent.resources.memory_capacity,
                },
                "current_goal": agent.current_goal.description if agent.current_goal else None,
                "active_goals": len([g for g in agent.goals if not g.completed]),
                "relationships": len(agent.relationships),
                "experience_count": agent.experience_count,
                "last_updated": agent.last_updated.isoformat(),
            }

    def batch_update(self, updates: List[Dict[str, Any]]) -> None:
        """
        Perform multiple state updates atomically
        Args:
            updates: List of update operations
        Example:
            updates = [
                {"type": "status", "agent_id": "123", "value": AgentStatus.MOVING},
                {"type": "position", "agent_id": "123", "value": Position(1, 2, 0)}
            ]
        """
        sorted_updates = sorted(updates, key=lambda u: u["agent_id"])
        agent_ids = list({u["agent_id"] for u in sorted_updates})
        locks = [self._locks.get(aid) for aid in agent_ids if aid in self._locks]
        for lock in locks:
            if lock is not None:
                lock.acquire()
        try:
            for update in sorted_updates:
                update_type = update["type"]
                agent_id = update["agent_id"]
                value = update["value"]
                if update_type == "status":
                    self.update_agent_status(agent_id, value)
                elif update_type == "position":
                    self.update_agent_position(agent_id, value)
                elif update_type == "resources":
                    self.update_agent_resources(
                        agent_id,
                        energy_delta=value.get("energy_delta"),
                        health_delta=value.get("health_delta"),
                    )
        finally:
            for lock in reversed(locks):
                if lock is not None:
                    lock.release()

    def _notify_observers(self, event: StateEvent) -> None:
        """Notify all observers of a state change"""
        self._event_history.append(event)
        for observer in self._observers:
            try:
                observer.on_state_change(event)
            except Exception as e:
                logger.error(f"Observer error: {e}")

    def _execute_transition_callbacks(
        self, agent_id: str, from_status: AgentStatus, to_status: AgentStatus
    ) -> None:
        """Execute callbacks for state transitions"""
        callbacks = self._transition_callbacks.get(agent_id, [])
        agent = self._agents[agent_id]
        for cb_info in callbacks:
            if cb_info["from"] == from_status and cb_info["to"] == to_status:
                try:
                    cb_info["callback"](agent)
                except Exception as e:
                    logger.error(f"Transition callback error: {e}")

    def _create_snapshot(self, agent: Agent) -> None:
        """Create a state snapshot for an agent"""
        snapshot = StateSnapshot(
            agent_id=agent.agent_id,
            timestamp=datetime.now(),
            status=agent.status,
            position=agent.position,
            energy=agent.resources.energy,
            health=agent.resources.health,
            current_goal_id=agent.current_goal.goal_id if agent.current_goal else None,
        )
        if agent.agent_id not in self._state_snapshots:
            self._state_snapshots[agent.agent_id] = []
        self._state_snapshots[agent.agent_id].append(snapshot)
        if len(self._state_snapshots[agent.agent_id]) > 100:
            self._state_snapshots[agent.agent_id] = self._state_snapshots[agent.agent_id][-50:]


class StateCondition:
    """Represents a condition that can be monitored"""

    def __init__(self, name: str, check_func: Callable[[Agent], bool]) -> None:
        """
        Initialize a state condition
        Args:
            name: Name of the condition
            check_func: Function that checks if condition is met
        """
        self.name = name
        self.check_func = check_func

    def is_met(self, agent: Agent) -> bool:
        """Check if the condition is met"""
        return self.check_func(agent)


class StateMonitor:
    """Monitors agent states for specific conditions"""

    def __init__(self, state_manager: AgentStateManager) -> None:
        """
        Initialize the state monitor
        Args:
            state_manager: The state manager to monitor
        """
        self.state_manager = state_manager
        self._conditions: Dict[str, List[StateCondition]] = {}
        self._condition_callbacks: Dict[str, List[Callable]] = {}
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 1.0

    def add_condition(
        self,
        agent_id: str,
        condition: StateCondition,
        callback: Callable[[Agent, StateCondition], None],
    ) -> None:
        """
        Add a condition to monitor for an agent
        Args:
            agent_id: The agent's ID
            condition: The condition to monitor
            callback: Function to call when condition is met
        """
        if agent_id not in self._conditions:
            self._conditions[agent_id] = []
            self._condition_callbacks[agent_id] = []
        self._conditions[agent_id].append(condition)
        self._condition_callbacks[agent_id].append(callback)

    def start_monitoring(self, interval: float = 1.0) -> None:
        """
        Start monitoring conditions
        Args:
            interval: Check interval in seconds
        """
        self._monitor_interval = interval
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop monitoring conditions"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            for agent_id, conditions in self._conditions.items():
                agent = self.state_manager.get_agent(agent_id)
                if not agent:
                    continue
                for i, condition in enumerate(conditions):
                    try:
                        if condition.is_met(agent):
                            callback = self._condition_callbacks[agent_id][i]
                            callback(agent, condition)
                    except Exception as e:
                        logger.error(f"Condition check error: {e}")
            threading.Event().wait(self._monitor_interval)

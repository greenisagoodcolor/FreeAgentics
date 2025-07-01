"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

"""
World Integration for Agent System
Provides interfaces and implementations for agents to perceive, act upon,
and be affected by environmental changes in the H3 world system.
"""

# World imports - made optional for testing
try:
    from world.h3_world import Biome, H3World, HexCell, TerrainType
    from world.spatial.spatial_api import ResourceType, SpatialAPI, SpatialCoordinate
except ImportError:
    # Mock classes for testing when dependencies aren't available
    H3World = type("H3World", (), {})
    HexCell = type("HexCell", (), {})
    Biome = type("Biome", (), {})
    TerrainType = type("TerrainType", (), {})
    SpatialAPI = type("SpatialAPI", (), {})
    SpatialCoordinate = type("SpatialCoordinate", (), {})
    ResourceType = type("ResourceType", (), {})
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions agents can perform in the world"""

    MOVE = "move"
    OBSERVE = "observe"
    HARVEST_RESOURCE = "harvest_resource"
    DEPOSIT_RESOURCE = "deposit_resource"
    MODIFY_TERRAIN = "modify_terrain"
    COMMUNICATE = "communicate"
    BUILD_STRUCTURE = "build_structure"
    TRADE = "trade"


class EventType(Enum):
    """Types of world events that agents can respond to"""

    AGENT_MOVED = "agent_moved"
    RESOURCE_DEPLETED = "resource_depleted"
    RESOURCE_DISCOVERED = "resource_discovered"
    WEATHER_CHANGED = "weather_changed"
    STRUCTURE_BUILT = "structure_built"
    AGENT_INTERACTION = "agent_interaction"
    TERRITORY_CLAIMED = "territory_claimed"


@dataclass
class WorldEvent:
    """Represents an event that occurred in the world"""

    event_type: EventType
    location: str  # hex_id where event occurred
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    affected_agents: set[str] = field(default_factory=set)


@dataclass
class Perception:
    """What an agent perceives about its environment"""

    current_location: str
    visible_cells: List[Any]  # Will be HexCell objects
    nearby_agents: Dict[str, str]  # agent_id -> hex_id
    available_resources: Dict[str, float]  # resources at current location
    movement_options: List[str]  # valid adjacent hex_ids to move to
    environmental_conditions: Dict[str, Any]
    recent_events: List[WorldEvent]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ActionResult:
    """Result of an agent action in the world"""

    success: bool
    action_type: ActionType
    cost: float = 0.0
    effects: Dict[str, Any] = field(default_factory=dict)
    generated_events: List[WorldEvent] = field(default_factory=list)
    message: str = ""


class IWorldPerceptionInterface(ABC):
    """Interface for agent world perception"""

    @abstractmethod
    def perceive_environment(self, agent_id: str) -> Perception:
        """Get agent's perception of the environment"""
        pass

    @abstractmethod
    def get_agent_location(self, agent_id: str) -> Optional[str]:
        """Get current location of an agent"""
        pass

    @abstractmethod
    def get_visible_agents(self, observer_id: str) -> Dict[str, str]:
        """Get other agents visible to the observer"""
        pass

    @abstractmethod
    def get_available_actions(self, agent_id: str) -> List[ActionType]:
        """Get list of actions available to agent at current location"""
        pass


class IWorldActionInterface(ABC):
    """Interface for agent world actions"""

    @abstractmethod
    def perform_action(
        self, agent_id: str, action_type: ActionType, parameters: Dict[str, Any]
    ) -> ActionResult:
        """Perform an action in the world"""
        pass

    @abstractmethod
    def validate_action(
        self, agent_id: str, action_type: ActionType, parameters: Dict[str, Any]
    ) -> bool:
        """Check if an action is valid"""
        pass

    @abstractmethod
    def get_action_cost(
        self, agent_id: str, action_type: ActionType, parameters: Dict[str, Any]
    ) -> float:
        """Get energy cost of performing an action"""
        pass


class IWorldEventSystem(ABC):
    """Interface for world event handling"""

    @abstractmethod
    def subscribe_to_events(
        self, agent_id: str, event_types: List[EventType], callback: Callable[[WorldEvent], None]
    ) -> None:
        """Subscribe an agent to specific types of world events"""
        pass

    @abstractmethod
    def unsubscribe_from_events(self, agent_id: str, event_types: List[EventType]) -> None:
        """Unsubscribe an agent from event types"""
        pass

    @abstractmethod
    def publish_event(self, event: WorldEvent) -> None:
        """Publish an event to all subscribed agents"""
        pass

    @abstractmethod
    def get_recent_events(self, location: str, time_window_minutes: int = 10) -> List[WorldEvent]:
        """Get recent events near a location"""
        pass


class WorldEventSystem(IWorldEventSystem):
    """Implementation of world event system"""

    def __init__(self) -> None:
        self.subscribers: Dict[EventType, Dict[str, Callable]] = defaultdict(dict)
        self.event_history: List[WorldEvent] = []
        self.max_history_size = 10000

    def subscribe_to_events(
        self, agent_id: str, event_types: List[EventType], callback: Callable[[WorldEvent], None]
    ) -> None:
        """Subscribe an agent to specific types of world events"""
        for event_type in event_types:
            self.subscribers[event_type][agent_id] = callback
        logger.debug(f"Agent {agent_id} subscribed to events: {[e.value for e in event_types]}")

    def unsubscribe_from_events(self, agent_id: str, event_types: List[EventType]) -> None:
        """Unsubscribe an agent from event types"""
        for event_type in event_types:
            if agent_id in self.subscribers[event_type]:
                del self.subscribers[event_type][agent_id]
        logger.debug(f"Agent {agent_id} unsubscribed from events: {[e.value for e in event_types]}")

    def publish_event(self, event: WorldEvent) -> None:
        """Publish an event to all subscribed agents"""
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        # Notify subscribers
        subscribers = self.subscribers.get(event.event_type, {})
        for agent_id, callback in subscribers.items():
            # Only notify if agent is affected or in vicinity
            if (
                not event.affected_agents
                or agent_id in event.affected_agents
                or agent_id == event.agent_id
            ):
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error notifying agent {agent_id} of event: {e}")
        logger.debug(f"Published event {event.event_type.value} to {len(subscribers)} subscribers")

    def get_recent_events(self, location: str, time_window_minutes: int = 10) -> List[WorldEvent]:
        """Get recent events near a location"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (time_window_minutes * 60)
        recent_events = []
        for event in reversed(self.event_history):  # Most recent first
            if event.timestamp.timestamp() < cutoff_time:
                break
            # Check if event is near the location (within 2 hexes)
            if event.location == location:
                recent_events.append(event)
            # You could add more sophisticated distance checking here
        return recent_events


class AgentWorldManager:
    """
    Main manager for agent-world integration.
    Handles agent positioning, perception, actions, and event coordination.
    """

    def __init__(self, world, spatial_api=None) -> None:
        self.world = world
        self.spatial_api = spatial_api
        self.event_system = WorldEventSystem()
        # Agent state tracking
        self.agent_locations: Dict[str, str] = {}  # agent_id -> hex_id
        self.agent_resources: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.agent_energy: Dict[str, float] = defaultdict(lambda: 100.0)  # Default energy
        # World state modifications
        self.modified_cells: Dict[str, Any] = {}  # Track modifications to world cells
        self.structures: Dict[str, Dict[str, Any]] = defaultdict(dict)  # hex_id -> structures
        logger.info("Initialized AgentWorldManager")

    def place_agent(self, agent_id: str, hex_id: str) -> bool:
        """Place an agent at a specific location"""
        cell = self.world.get_cell(hex_id)
        if not cell:
            logger.warning(f"Cannot place agent {agent_id} at invalid location {hex_id}")
            return False
        old_location = self.agent_locations.get(agent_id)
        self.agent_locations[agent_id] = hex_id
        # Initialize agent energy if new agent
        _ = self.agent_energy[agent_id]  # Trigger defaultdict initialization
        # Publish movement event
        event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location=hex_id,
            agent_id=agent_id,
            data={"from": old_location, "to": hex_id},
        )
        self.event_system.publish_event(event)
        logger.debug(f"Placed agent {agent_id} at {hex_id}")
        return True

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the world"""
        if agent_id in self.agent_locations:
            del self.agent_locations[agent_id]
        if agent_id in self.agent_resources:
            del self.agent_resources[agent_id]
        if agent_id in self.agent_energy:
            del self.agent_energy[agent_id]
        logger.debug(f"Removed agent {agent_id} from world")

    def perceive_environment(self, agent_id: str) -> Optional[Perception]:
        """Get agent's perception of the environment"""
        location = self.agent_locations.get(agent_id)
        if not location:
            return None
        current_cell = self.world.get_cell(location)
        if not current_cell:
            return None
        # Get visible cells
        visible_cells = self.world.get_visible_cells(location)
        # Get nearby agents
        nearby_agents = {}
        for other_id, other_location in self.agent_locations.items():
            if other_id != agent_id and other_location in [cell.hex_id for cell in visible_cells]:
                nearby_agents[other_id] = other_location
        # Get available resources at current location
        available_resources = current_cell.resources.copy()
        # Get movement options
        neighbors = self.world.get_neighbors(location)
        movement_options = [
            neighbor.hex_id for neighbor in neighbors if neighbor.movement_cost < float("inf")
        ]  # Exclude impassable terrain
        # Environmental conditions
        environmental_conditions = {
            "temperature": current_cell.temperature,
            "moisture": current_cell.moisture,
            "elevation": current_cell.elevation,
            "biome": current_cell.biome.value,
            "terrain": current_cell.terrain.value,
            "weather": "clear",  # Could be enhanced with weather system
        }
        # Recent events
        recent_events = self.event_system.get_recent_events(location)
        return Perception(
            current_location=location,
            visible_cells=visible_cells,
            nearby_agents=nearby_agents,
            available_resources=available_resources,
            movement_options=movement_options,
            environmental_conditions=environmental_conditions,
            recent_events=recent_events,
        )

    def get_available_actions(self, agent_id: str) -> List[ActionType]:
        """Get list of actions available to agent at current location"""
        location = self.agent_locations.get(agent_id)
        if not location:
            return []
        actions = [ActionType.OBSERVE, ActionType.COMMUNICATE]
        # Movement available if there are valid neighbors
        neighbors = self.world.get_neighbors(location)
        if neighbors:
            actions.append(ActionType.MOVE)
        # Resource actions
        cell = self.world.get_cell(location)
        if cell and any(amount > 0 for amount in cell.resources.values()):
            actions.append(ActionType.HARVEST_RESOURCE)
        if any(amount > 0 for amount in self.agent_resources[agent_id].values()):
            actions.append(ActionType.DEPOSIT_RESOURCE)
        # Building/trading actions could be conditional on other factors
        actions.extend([ActionType.BUILD_STRUCTURE, ActionType.TRADE])
        return actions

    def perform_action(
        self, agent_id: str, action_type: ActionType, parameters: Dict[str, Any]
    ) -> ActionResult:
        """Perform an action in the world"""
        location = self.agent_locations.get(agent_id)
        if not location:
            return ActionResult(
                success=False, action_type=action_type, message="Agent not found in world"
            )
        # Calculate action cost
        cost = self._calculate_action_cost(agent_id, action_type, parameters)
        # Check if agent has enough energy
        if self.agent_energy[agent_id] < cost:
            return ActionResult(
                success=False, action_type=action_type, cost=cost, message="Insufficient energy"
            )
        # Perform the specific action
        if action_type == ActionType.MOVE:
            return self._perform_move_action(agent_id, parameters, cost)
        elif action_type == ActionType.HARVEST_RESOURCE:
            return self._perform_harvest_action(agent_id, parameters, cost)
        elif action_type == ActionType.DEPOSIT_RESOURCE:
            return self._perform_deposit_action(agent_id, parameters, cost)
        elif action_type == ActionType.OBSERVE:
            return self._perform_observe_action(agent_id, parameters, cost)
        elif action_type == ActionType.COMMUNICATE:
            return self._perform_communicate_action(agent_id, parameters, cost)
        elif action_type == ActionType.BUILD_STRUCTURE:
            return self._perform_build_action(agent_id, parameters, cost)
        else:
            return ActionResult(
                success=False,
                action_type=action_type,
                message=f"Action type {action_type.value} not implemented",
            )

    def _calculate_action_cost(
        self, agent_id: str, action_type: ActionType, parameters: Dict[str, Any]
    ) -> float:
        """Calculate energy cost of performing an action"""
        base_costs = {
            ActionType.MOVE: 10.0,
            ActionType.OBSERVE: 5.0,
            ActionType.HARVEST_RESOURCE: 15.0,
            ActionType.DEPOSIT_RESOURCE: 5.0,
            ActionType.COMMUNICATE: 2.0,
            ActionType.BUILD_STRUCTURE: 25.0,
            ActionType.TRADE: 5.0,
        }
        base_cost = base_costs.get(action_type, 10.0)
        # Modify based on terrain and other factors
        location = self.agent_locations.get(agent_id)
        if location:
            cell = self.world.get_cell(location)
            if cell:
                # Movement cost affected by terrain
                if action_type == ActionType.MOVE:
                    target_hex = parameters.get("target_hex")
                    if target_hex:
                        target_cell = self.world.get_cell(target_hex)
                        if target_cell:
                            base_cost *= target_cell.movement_cost
        return base_cost

    def _perform_move_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform movement action"""
        target_hex = parameters.get("target_hex")
        if not target_hex:
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE,
                cost=cost,
                message="No target_hex specified",
            )
        current_location = self.agent_locations[agent_id]
        # Validate movement
        neighbors = self.world.get_neighbors(current_location)
        valid_targets = [neighbor.hex_id for neighbor in neighbors]
        if target_hex not in valid_targets:
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE,
                cost=cost,
                message="Target location not adjacent or invalid",
            )
        # Perform movement
        self.agent_locations[agent_id] = target_hex
        self.agent_energy[agent_id] -= cost
        # Generate event
        event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location=target_hex,
            agent_id=agent_id,
            data={"from": current_location, "to": target_hex},
        )
        return ActionResult(
            success=True,
            action_type=ActionType.MOVE,
            cost=cost,
            effects={"new_location": target_hex},
            generated_events=[event],
            message=f"Moved from {current_location} to {target_hex}",
        )

    def _perform_harvest_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform resource harvesting action"""
        resource_type = parameters.get("resource_type")
        amount = parameters.get("amount", 10.0)
        if not resource_type:
            return ActionResult(
                success=False,
                action_type=ActionType.HARVEST_RESOURCE,
                cost=cost,
                message="No resource_type specified",
            )
        location = self.agent_locations[agent_id]
        cell = self.world.get_cell(location)
        if not cell or resource_type not in cell.resources:
            return ActionResult(
                success=False,
                action_type=ActionType.HARVEST_RESOURCE,
                cost=cost,
                message=f"Resource {resource_type} not available at location",
            )
        # Calculate actual harvest amount
        available = cell.resources[resource_type]
        harvested = min(amount, available)
        if harvested <= 0:
            return ActionResult(
                success=False,
                action_type=ActionType.HARVEST_RESOURCE,
                cost=cost,
                message=f"No {resource_type} available to harvest",
            )
        # Update world state
        cell.resources[resource_type] -= harvested
        # Update agent resources
        if resource_type not in self.agent_resources[agent_id]:
            self.agent_resources[agent_id][resource_type] = 0
        self.agent_resources[agent_id][resource_type] += harvested
        # Consume energy
        self.agent_energy[agent_id] -= cost
        # Generate events
        events = []
        if cell.resources[resource_type] <= 0:
            events.append(
                WorldEvent(
                    event_type=EventType.RESOURCE_DEPLETED,
                    location=location,
                    agent_id=agent_id,
                    data={"resource_type": resource_type},
                )
            )
        return ActionResult(
            success=True,
            action_type=ActionType.HARVEST_RESOURCE,
            cost=cost,
            effects={"harvested": harvested, "resource_type": resource_type},
            generated_events=events,
            message=f"Harvested {harvested} {resource_type}",
        )

    def _perform_observe_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform observation action"""
        # Consume energy
        self.agent_energy[agent_id] -= cost
        # Get enhanced perception
        perception = self.perceive_environment(agent_id)
        return ActionResult(
            success=True,
            action_type=ActionType.OBSERVE,
            cost=cost,
            effects={"perception": perception},
            message="Observed environment",
        )

    def _perform_communicate_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform communication action"""
        target_agent = parameters.get("target_agent")
        message = parameters.get("message", "")
        # Consume energy
        self.agent_energy[agent_id] -= cost
        # Generate communication event
        event = WorldEvent(
            event_type=EventType.AGENT_INTERACTION,
            location=self.agent_locations[agent_id],
            agent_id=agent_id,
            data={"target_agent": target_agent, "message": message},
            affected_agents={target_agent} if target_agent else set(),
        )
        return ActionResult(
            success=True,
            action_type=ActionType.COMMUNICATE,
            cost=cost,
            effects={"message_sent": True},
            generated_events=[event],
            message=f"Sent communication: {message}",
        )

    def _perform_build_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform building action"""
        structure_type = parameters.get("structure_type", "basic_shelter")
        location = self.agent_locations[agent_id]
        # Check if already has structure
        if structure_type in self.structures[location]:
            return ActionResult(
                success=False,
                action_type=ActionType.BUILD_STRUCTURE,
                cost=cost,
                message=f"Structure {structure_type} already exists at location",
            )
        # Consume energy
        self.agent_energy[agent_id] -= cost
        # Add structure
        self.structures[location][structure_type] = {
            "built_by": agent_id,
            "built_at": datetime.now(timezone.utc),
            "durability": 100,
        }
        # Generate event
        event = WorldEvent(
            event_type=EventType.STRUCTURE_BUILT,
            location=location,
            agent_id=agent_id,
            data={"structure_type": structure_type},
        )
        return ActionResult(
            success=True,
            action_type=ActionType.BUILD_STRUCTURE,
            cost=cost,
            effects={"structure_built": structure_type},
            generated_events=[event],
            message=f"Built {structure_type}",
        )

    # === Agent State Management ===
    def get_agent_energy(self, agent_id: str) -> float:
        """Get current energy level of an agent"""
        return self.agent_energy[agent_id]  # Use defaultdict behavior

    def restore_agent_energy(self, agent_id: str, amount: float) -> None:
        """Restore energy to an agent"""
        current = self.agent_energy[agent_id]  # Use defaultdict behavior
        self.agent_energy[agent_id] = min(100.0, current + amount)

    def get_agent_resources(self, agent_id: str) -> Dict[str, float]:
        """Get resources carried by an agent"""
        return self.agent_resources.get(agent_id, {}).copy()

    def get_world_state_summary(self) -> Dict[str, Any]:
        """Get summary of current world state"""
        return {
            "num_agents": len(self.agent_locations),
            "total_energy": sum(self.agent_energy.values()),
            "num_structures": sum(len(structs) for structs in self.structures.values()),
            "modified_cells": len(self.modified_cells),
            "recent_events": len(self.event_system.event_history),
        }


class AgentWorldManager:
    """
    Main manager for agent-world integration.
    Handles agent positioning, perception, actions, and event coordination.
    """

    def __init__(self, world: H3World, spatial_api: Optional[SpatialAPI] = None) -> None:
        self.world = world
        try:
            self.spatial_api = spatial_api or SpatialAPI(resolution=world.resolution)
        except (TypeError, AttributeError):
            # Fallback for testing or when SpatialAPI isn't available
            self.spatial_api = SpatialAPI()
        self.event_system = WorldEventSystem()
        # Agent state tracking
        self.agent_locations: Dict[str, str] = {}  # agent_id -> hex_id
        self.agent_resources: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.agent_energy: Dict[str, float] = defaultdict(lambda: 100.0)  # Default energy
        # World state modifications
        self.modified_cells: Dict[str, HexCell] = {}  # Track modifications to world cells
        self.structures: Dict[str, Dict[str, Any]] = defaultdict(dict)  # hex_id -> structures
        logger.info("Initialized AgentWorldManager")

    def place_agent(self, agent_id: str, hex_id: str) -> bool:
        """Place an agent at a specific location"""
        cell = self.world.get_cell(hex_id)
        if not cell:
            logger.warning(f"Cannot place agent {agent_id} at invalid location {hex_id}")
            return False
        old_location = self.agent_locations.get(agent_id)
        self.agent_locations[agent_id] = hex_id
        # Initialize agent energy if new agent
        _ = self.agent_energy[agent_id]  # Trigger defaultdict initialization
        # Publish movement event
        event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location=hex_id,
            agent_id=agent_id,
            data={"from": old_location, "to": hex_id},
        )
        self.event_system.publish_event(event)
        logger.debug(f"Placed agent {agent_id} at {hex_id}")
        return True

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the world"""
        if agent_id in self.agent_locations:
            del self.agent_locations[agent_id]
        if agent_id in self.agent_resources:
            del self.agent_resources[agent_id]
        if agent_id in self.agent_energy:
            del self.agent_energy[agent_id]
        logger.debug(f"Removed agent {agent_id} from world")

    def perceive_environment(self, agent_id: str) -> Optional[Perception]:
        """Get agent's perception of the environment"""
        location = self.agent_locations.get(agent_id)
        if not location:
            return None
        current_cell = self.world.get_cell(location)
        if not current_cell:
            return None
        # Get visible cells
        visible_cells = self.world.get_visible_cells(location)
        # Get nearby agents
        nearby_agents = {}
        for other_id, other_location in self.agent_locations.items():
            if other_id != agent_id and other_location in [cell.hex_id for cell in visible_cells]:
                nearby_agents[other_id] = other_location
        # Get available resources at current location
        available_resources = current_cell.resources.copy()
        # Get movement options
        neighbors = self.world.get_neighbors(location)
        movement_options = [
            neighbor.hex_id for neighbor in neighbors if neighbor.movement_cost < float("inf")
        ]  # Exclude impassable terrain
        # Environmental conditions
        environmental_conditions = {
            "temperature": current_cell.temperature,
            "moisture": current_cell.moisture,
            "elevation": current_cell.elevation,
            "biome": current_cell.biome.value,
            "terrain": current_cell.terrain.value,
            "weather": "clear",  # Could be enhanced with weather system
        }
        # Recent events
        recent_events = self.event_system.get_recent_events(location)
        return Perception(
            current_location=location,
            visible_cells=visible_cells,
            nearby_agents=nearby_agents,
            available_resources=available_resources,
            movement_options=movement_options,
            environmental_conditions=environmental_conditions,
            recent_events=recent_events,
        )

    def get_available_actions(self, agent_id: str) -> List[ActionType]:
        """Get list of actions available to agent at current location"""
        location = self.agent_locations.get(agent_id)
        if not location:
            return []
        actions = [ActionType.OBSERVE, ActionType.COMMUNICATE]
        # Movement available if there are valid neighbors
        neighbors = self.world.get_neighbors(location)
        if neighbors:
            actions.append(ActionType.MOVE)
        # Resource actions
        cell = self.world.get_cell(location)
        if cell and any(amount > 0 for amount in cell.resources.values()):
            actions.append(ActionType.HARVEST_RESOURCE)
        if any(amount > 0 for amount in self.agent_resources[agent_id].values()):
            actions.append(ActionType.DEPOSIT_RESOURCE)
        # Building/trading actions could be conditional on other factors
        actions.extend([ActionType.BUILD_STRUCTURE, ActionType.TRADE])
        return actions

    def perform_action(
        self, agent_id: str, action_type: ActionType, parameters: Dict[str, Any]
    ) -> ActionResult:
        """Perform an action in the world"""
        location = self.agent_locations.get(agent_id)
        if not location:
            return ActionResult(
                success=False, action_type=action_type, message="Agent not found in world"
            )
        # Calculate action cost
        cost = self._calculate_action_cost(agent_id, action_type, parameters)
        # Check if agent has enough energy
        if self.agent_energy[agent_id] < cost:
            return ActionResult(
                success=False, action_type=action_type, cost=cost, message="Insufficient energy"
            )
        # Perform the specific action
        if action_type == ActionType.MOVE:
            return self._perform_move_action(agent_id, parameters, cost)
        elif action_type == ActionType.HARVEST_RESOURCE:
            return self._perform_harvest_action(agent_id, parameters, cost)
        elif action_type == ActionType.DEPOSIT_RESOURCE:
            return self._perform_deposit_action(agent_id, parameters, cost)
        elif action_type == ActionType.OBSERVE:
            return self._perform_observe_action(agent_id, parameters, cost)
        elif action_type == ActionType.COMMUNICATE:
            return self._perform_communicate_action(agent_id, parameters, cost)
        elif action_type == ActionType.BUILD_STRUCTURE:
            return self._perform_build_action(agent_id, parameters, cost)
        else:
            return ActionResult(
                success=False,
                action_type=action_type,
                message=f"Action type {action_type.value} not implemented",
            )

    def _calculate_action_cost(
        self, agent_id: str, action_type: ActionType, parameters: Dict[str, Any]
    ) -> float:
        """Calculate energy cost of performing an action"""
        base_costs = {
            ActionType.MOVE: 10.0,
            ActionType.OBSERVE: 5.0,
            ActionType.HARVEST_RESOURCE: 15.0,
            ActionType.DEPOSIT_RESOURCE: 5.0,
            ActionType.COMMUNICATE: 2.0,
            ActionType.BUILD_STRUCTURE: 25.0,
            ActionType.TRADE: 5.0,
        }
        base_cost = base_costs.get(action_type, 10.0)
        # Modify based on terrain and other factors
        location = self.agent_locations.get(agent_id)
        if location:
            cell = self.world.get_cell(location)
            if cell:
                # Movement cost affected by terrain
                if action_type == ActionType.MOVE:
                    target_hex = parameters.get("target_hex")
                    if target_hex:
                        target_cell = self.world.get_cell(target_hex)
                        if target_cell:
                            base_cost *= target_cell.movement_cost
        return base_cost

    def _perform_move_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform movement action"""
        target_hex = parameters.get("target_hex")
        if not target_hex:
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE,
                cost=cost,
                message="No target_hex specified",
            )
        current_location = self.agent_locations[agent_id]
        # Validate movement
        neighbors = self.world.get_neighbors(current_location)
        valid_targets = [neighbor.hex_id for neighbor in neighbors]
        if target_hex not in valid_targets:
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE,
                cost=cost,
                message="Target location not adjacent or invalid",
            )
        # Perform movement
        self.agent_locations[agent_id] = target_hex
        self.agent_energy[agent_id] -= cost
        # Generate event
        event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location=target_hex,
            agent_id=agent_id,
            data={"from": current_location, "to": target_hex},
        )
        return ActionResult(
            success=True,
            action_type=ActionType.MOVE,
            cost=cost,
            effects={"new_location": target_hex},
            generated_events=[event],
            message=f"Moved from {current_location} to {target_hex}",
        )

    def _perform_harvest_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform resource harvesting action"""
        resource_type = parameters.get("resource_type")
        amount = parameters.get("amount", 10.0)
        if not resource_type:
            return ActionResult(
                success=False,
                action_type=ActionType.HARVEST_RESOURCE,
                cost=cost,
                message="No resource_type specified",
            )
        location = self.agent_locations[agent_id]
        cell = self.world.get_cell(location)
        if not cell or resource_type not in cell.resources:
            return ActionResult(
                success=False,
                action_type=ActionType.HARVEST_RESOURCE,
                cost=cost,
                message=f"Resource {resource_type} not available at location",
            )
        # Calculate actual harvest amount
        available = cell.resources[resource_type]
        harvested = min(amount, available)
        if harvested <= 0:
            return ActionResult(
                success=False,
                action_type=ActionType.HARVEST_RESOURCE,
                cost=cost,
                message=f"No {resource_type} available to harvest",
            )
        # Update world state
        cell.resources[resource_type] -= harvested
        # Update agent resources
        if resource_type not in self.agent_resources[agent_id]:
            self.agent_resources[agent_id][resource_type] = 0
        self.agent_resources[agent_id][resource_type] += harvested
        # Consume energy
        self.agent_energy[agent_id] -= cost
        # Generate events
        events = []
        if cell.resources[resource_type] <= 0:
            events.append(
                WorldEvent(
                    event_type=EventType.RESOURCE_DEPLETED,
                    location=location,
                    agent_id=agent_id,
                    data={"resource_type": resource_type},
                )
            )
        return ActionResult(
            success=True,
            action_type=ActionType.HARVEST_RESOURCE,
            cost=cost,
            effects={"harvested": harvested, "resource_type": resource_type},
            generated_events=events,
            message=f"Harvested {harvested} {resource_type}",
        )

    def _perform_deposit_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform resource deposit action"""
        resource_type = parameters.get("resource_type")
        amount = parameters.get("amount", 10.0)
        if not resource_type:
            return ActionResult(
                success=False,
                action_type=ActionType.DEPOSIT_RESOURCE,
                cost=cost,
                message="No resource_type specified",
            )
        agent_resources = self.agent_resources[agent_id]
        if resource_type not in agent_resources or agent_resources[resource_type] < amount:
            return ActionResult(
                success=False,
                action_type=ActionType.DEPOSIT_RESOURCE,
                cost=cost,
                message=f"Insufficient {resource_type} to deposit",
            )
        location = self.agent_locations[agent_id]
        cell = self.world.get_cell(location)
        # Update world state
        if resource_type not in cell.resources:
            cell.resources[resource_type] = 0
        cell.resources[resource_type] += amount
        # Update agent resources
        agent_resources[resource_type] -= amount
        # Consume energy
        self.agent_energy[agent_id] -= cost
        return ActionResult(
            success=True,
            action_type=ActionType.DEPOSIT_RESOURCE,
            cost=cost,
            effects={"deposited": amount, "resource_type": resource_type},
            message=f"Deposited {amount} {resource_type}",
        )

    def _perform_observe_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform observation action"""
        # Consume energy
        self.agent_energy[agent_id] -= cost
        # Get enhanced perception
        perception = self.perceive_environment(agent_id)
        return ActionResult(
            success=True,
            action_type=ActionType.OBSERVE,
            cost=cost,
            effects={"perception": perception},
            message="Observed environment",
        )

    def _perform_communicate_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform communication action"""
        target_agent = parameters.get("target_agent")
        message = parameters.get("message", "")
        # Consume energy
        self.agent_energy[agent_id] -= cost
        # Generate communication event
        event = WorldEvent(
            event_type=EventType.AGENT_INTERACTION,
            location=self.agent_locations[agent_id],
            agent_id=agent_id,
            data={"target_agent": target_agent, "message": message},
            affected_agents={target_agent} if target_agent else set(),
        )
        return ActionResult(
            success=True,
            action_type=ActionType.COMMUNICATE,
            cost=cost,
            effects={"message_sent": True},
            generated_events=[event],
            message=f"Sent communication: {message}",
        )

    def _perform_build_action(
        self, agent_id: str, parameters: Dict[str, Any], cost: float
    ) -> ActionResult:
        """Perform building action"""
        structure_type = parameters.get("structure_type", "basic_shelter")
        location = self.agent_locations[agent_id]
        # Check if already has structure
        if structure_type in self.structures[location]:
            return ActionResult(
                success=False,
                action_type=ActionType.BUILD_STRUCTURE,
                cost=cost,
                message=f"Structure {structure_type} already exists at location",
            )
        # Consume energy
        self.agent_energy[agent_id] -= cost
        # Add structure
        self.structures[location][structure_type] = {
            "built_by": agent_id,
            "built_at": datetime.now(timezone.utc),
            "durability": 100,
        }
        # Generate event
        event = WorldEvent(
            event_type=EventType.STRUCTURE_BUILT,
            location=location,
            agent_id=agent_id,
            data={"structure_type": structure_type},
        )
        return ActionResult(
            success=True,
            action_type=ActionType.BUILD_STRUCTURE,
            cost=cost,
            effects={"structure_built": structure_type},
            generated_events=[event],
            message=f"Built {structure_type}",
        )

    # === Agent State Management ===
    def get_agent_energy(self, agent_id: str) -> float:
        """Get current energy level of an agent"""
        return self.agent_energy[agent_id]  # Use defaultdict behavior

    def restore_agent_energy(self, agent_id: str, amount: float) -> None:
        """Restore energy to an agent"""
        current = self.agent_energy[agent_id]  # Use defaultdict behavior
        self.agent_energy[agent_id] = min(100.0, current + amount)

    def get_agent_resources(self, agent_id: str) -> Dict[str, float]:
        """Get resources carried by an agent"""
        return self.agent_resources.get(agent_id, {}).copy()

    def get_world_state_summary(self) -> Dict[str, Any]:
        """Get summary of current world state"""
        return {
            "num_agents": len(self.agent_locations),
            "total_energy": sum(self.agent_energy.values()),
            "num_structures": sum(len(structs) for structs in self.structures.values()),
            "modified_cells": len(self.modified_cells),
            "recent_events": len(self.event_system.event_history),
        }

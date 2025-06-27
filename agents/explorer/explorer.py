"""
Explorer Agent for FreeAgentics

This module provides the Explorer agent type, specialized in exploration,
discovery, pathfinding, and resource discovery. Explorers are curious,
adaptable, and excel at navigating unknown territories.
"""

import math
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Import base agent system
from agents.base import Agent, AgentCapability, BaseAgent, Position, get_default_factory
from agents.base.behaviors import BaseBehavior, BehaviorPriority


class ExplorationStatus(Enum):
    """Status of exploration activities"""

    IDLE = "idle"
    EXPLORING = "exploring"
    MAPPING = "mapping"
    INVESTIGATING = "investigating"
    RETURNING = "returning"


class DiscoveryType(Enum):
    """Types of discoveries explorers can make"""

    RESOURCE = "resource"
    LOCATION = "location"
    AGENT = "agent"
    ANOMALY = "anomaly"
    PATH = "path"
    TERRITORY = "territory"


class Discovery:
    """Represents a discovery made by an explorer"""

    def __init__(
        self,
        discovery_type: DiscoveryType,
        position: Position,
        description: str,
        value: float = 1.0,
        confidence: float = 1.0,
    ) -> None:
        self.discovery_type = discovery_type
        self.position = position
        self.description = description
        self.value = value  # Estimated value/importance
        self.confidence = confidence  # Confidence in the discovery
        self.discovered_at = datetime.now()
        self.discovery_id = (
            f"{discovery_type.value}_{position.x:.1f}_{position.y:.1f}_{datetime.now().timestamp()}"
        )
        self.verified = False
        self.shared = False


class ExplorationMap:
    """Maps and tracks explored territories"""

    def __init__(self, grid_size: float = 5.0) -> None:
        self.grid_size = grid_size
        self.explored_cells: set[tuple[int, int]] = set()
        self.discoveries: Dict[str, Discovery] = {}
        self.paths: Dict[tuple[int, int], dict[tuple[int, int], float]] = (
            {}
        )  # From -> To -> Distance
        self.last_updated = datetime.now()

    def get_cell(self, position: Position) -> tuple[int, int]:
        """Convert position to grid cell"""
        return (int(position.x // self.grid_size), int(position.y // self.grid_size))

    def mark_explored(self, position: Position) -> bool:
        """Mark a position as explored, returns True if newly explored"""
        cell = self.get_cell(position)
        if cell not in self.explored_cells:
            self.explored_cells.add(cell)
            self.last_updated = datetime.now()
            return True
        return False

    def is_explored(self, position: Position) -> bool:
        """Check if a position has been explored"""
        return self.get_cell(position) in self.explored_cells

    def add_discovery(self, discovery: Discovery) -> None:
        """Add a new discovery to the map"""
        self.discoveries[discovery.discovery_id] = discovery
        self.mark_explored(discovery.position)
        self.last_updated = datetime.now()

    def get_unexplored_directions(self, position: Position, radius: int = 3) -> List[float]:
        """Get directions to unexplored areas"""
        current_cell = self.get_cell(position)
        unexplored_directions = []

        # Check 8 directions
        directions = [i * math.pi / 4 for i in range(8)]

        for direction in directions:
            # Check cells in this direction within radius
            unexplored_count = 0
            total_count = 0

            for distance in range(1, radius + 1):
                check_x = current_cell[0] + int(distance * math.cos(direction))
                check_y = current_cell[1] + int(distance * math.sin(direction))

                if (check_x, check_y) not in self.explored_cells:
                    unexplored_count += 1
                total_count += 1

            # If more than 50% of cells in this direction are unexplored
            if total_count > 0 and unexplored_count / total_count > 0.5:
                unexplored_directions.append(direction)

        return unexplored_directions

    def get_exploration_score(self) -> float:
        """Get a score representing exploration progress"""
        if not self.explored_cells:
            return 0.0

        # Calculate area covered (rough estimate)
        min_x = min(cell[0] for cell in self.explored_cells)
        max_x = max(cell[0] for cell in self.explored_cells)
        min_y = min(cell[1] for cell in self.explored_cells)
        max_y = max(cell[1] for cell in self.explored_cells)

        total_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        explored_ratio = len(self.explored_cells) / total_area if total_area > 0 else 1.0

        return min(1.0, explored_ratio)


class AdvancedExplorationBehavior(BaseBehavior):
    """Advanced exploration behavior for explorer agents"""

    def __init__(self) -> None:
        super().__init__(
            "advanced_exploration",
            BehaviorPriority.HIGH,
            {AgentCapability.MOVEMENT, AgentCapability.PERCEPTION},
            timedelta(seconds=1),
        )
        self.exploration_radius = 15.0
        self.investigation_chance = 0.3

    def _can_execute_custom(self, agent: Agent, context: Dict[str, Any]) -> bool:
        # Can always explore unless agent is critically low on energy
        return agent.resources.energy > 10.0

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        exploration_map = agent.metadata.get("exploration_map")
        if not exploration_map:
            return {"success": False, "reason": "no_exploration_map"}

        # Get unexplored directions
        unexplored_directions = exploration_map.get_unexplored_directions(agent.position)

        if unexplored_directions:
            # Choose direction with least exploration
            chosen_direction = random.choice(unexplored_directions)
        else:
            # Random direction if all nearby areas explored
            chosen_direction = random.uniform(0, 2 * math.pi)

        # Calculate movement distance based on personality
        personality_profile = agent.metadata.get("personality_profile")
        base_distance = 8.0

        if personality_profile:
            risk_tolerance = personality_profile.get_trait_value("risk_tolerance")
            curiosity = personality_profile.get_trait_value("curiosity")
            distance_modifier = (risk_tolerance + curiosity) / 2.0
            distance = base_distance * (0.5 + distance_modifier)
        else:
            distance = base_distance

        # Calculate new position
        new_position = Position(
            agent.position.x + distance * math.cos(chosen_direction),
            agent.position.y + distance * math.sin(chosen_direction),
            agent.position.z,
        )

        # Check for discoveries
        discoveries = self._check_for_discoveries(agent, new_position, context)

        # Mark area as explored
        newly_explored = exploration_map.mark_explored(new_position)

        # Create movement action
        from agents.base.decision_making import Action, ActionType

        action = Action(
            action_type=ActionType.MOVE,
            target_position=new_position,
            parameters={
                "movement_type": "advanced_exploration",
                "direction": chosen_direction,
                "discoveries": len(discoveries),
                "newly_explored": newly_explored,
            },
        )

        # Update agent status
        agent.metadata["exploration_status"] = ExplorationStatus.EXPLORING

        return {
            "success": True,
            "action": action,
            "target_position": new_position,
            "discoveries": discoveries,
            "newly_explored": newly_explored,
            "exploration_direction": chosen_direction,
        }

    def _check_for_discoveries(
        self, agent: Agent, position: Position, context: Dict[str, Any]
    ) -> List[Discovery]:
        """Check for potential discoveries at a position"""
        discoveries = []
        exploration_map = agent.metadata.get("exploration_map")

        # Discovery chance based on personality
        personality_profile = agent.metadata.get("personality_profile")
        base_discovery_chance = 0.1

        if personality_profile:
            curiosity = personality_profile.get_trait_value("curiosity")
            openness = personality_profile.get_trait_value("openness")
            perception_modifier = (curiosity + openness) / 2.0
            discovery_chance = base_discovery_chance * (1 + perception_modifier)
        else:
            discovery_chance = base_discovery_chance

        # Check for various types of discoveries
        if random.random() < discovery_chance:
            discovery_type = random.choice(list(DiscoveryType))

            discovery = Discovery(
                discovery_type=discovery_type,
                position=position,
                description=(
                    f"Discovered {discovery_type.value} at {position.x:.1f}, {position.y:.1f}"
                ),
                value=random.uniform(0.5, 2.0),
                confidence=random.uniform(0.7, 1.0),
            )

            discoveries.append(discovery)
            exploration_map.add_discovery(discovery)

            # Add to agent memory
            agent.add_to_memory(
                {
                    "event": "discovery",
                    "discovery_type": discovery_type.value,
                    "position": position.to_array().tolist(),
                    "value": discovery.value,
                    "confidence": discovery.confidence,
                },
                is_important=True,
            )

        return discoveries

    def _get_priority_modifier(self, agent: Agent, context: Dict[str, Any]) -> float:
        # Higher priority for explorers with active exploration goals
        if agent.agent_type == "explorer":
            exploration_goals = [g for g in agent.goals if "explor" in g.description.lower()]
            return 1.5 if exploration_goals else 1.2

        return 0.8


class PathfindingBehavior(BaseBehavior):
    """Pathfinding and route optimization behavior"""

    def __init__(self) -> None:
        super().__init__(
            "pathfinding",
            BehaviorPriority.MEDIUM,
            {AgentCapability.MOVEMENT, AgentCapability.PLANNING},
        )

    def _can_execute_custom(self, agent: Agent, context: Dict[str, Any]) -> bool:
        # Can execute if agent has a specific destination goal
        destination_goals = [g for g in agent.goals if g.target_position and not g.completed]
        return len(destination_goals) > 0

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        # Find the closest destination goal
        destination_goals = [g for g in agent.goals if g.target_position and not g.completed]
        if not destination_goals:
            return {"success": False, "reason": "no_destination_goals"}

        # Sort by distance
        current_pos = agent.position
        destination_goals.sort(key=(lambda g: current_pos.distance_to(g.target_position)))

        target_goal = destination_goals[0]
        target_position = target_goal.target_position

        # Calculate optimal path (simplified - could use A* in the future)
        distance = current_pos.distance_to(target_position)

        if distance < 2.0:
            # Close enough - mark goal as completed
            target_goal.completed = True
            target_goal.progress = 1.0

            return {
                "success": True,
                "goal_completed": True,
                "goal": target_goal.description,
            }

        # Move towards target
        dx = target_position.x - current_pos.x
        dy = target_position.y - current_pos.y
        unit_dx = dx / distance
        unit_dy = dy / distance

        # Explorer movement speed
        movement_speed = 5.0
        personality_profile = agent.metadata.get("personality_profile")

        if personality_profile:
            adaptability = personality_profile.get_trait_value("adaptability")
            movement_speed *= 0.8 + adaptability * 0.4  # 0.8 to 1.2x speed

        move_distance = min(movement_speed, distance)

        new_position = Position(
            current_pos.x + unit_dx * move_distance,
            current_pos.y + unit_dy * move_distance,
            current_pos.z,
        )

        # Update goal progress
        target_goal.progress = 1.0 - (distance - move_distance) / distance

        from agents.base.decision_making import Action, ActionType

        action = Action(
            action_type=ActionType.MOVE,
            target_position=new_position,
            parameters={
                "movement_type": "pathfinding",
                "target_goal": target_goal.goal_id,
                "remaining_distance": distance - move_distance,
            },
        )

        return {
            "success": True,
            "action": action,
            "target_position": new_position,
            "goal_progress": target_goal.progress,
        }


class ExplorerAgent(BaseAgent):
    """Specialized Explorer Agent with enhanced exploration capabilities"""

    def __init__(self, **kwargs) -> None:
        # Set default explorer capabilities
        default_capabilities = {
            AgentCapability.MOVEMENT,
            AgentCapability.PERCEPTION,
            AgentCapability.MEMORY,
            AgentCapability.LEARNING,
            AgentCapability.PLANNING,
        }

        # Merge with provided capabilities
        if "capabilities" in kwargs:
            kwargs["capabilities"] = kwargs["capabilities"].union(default_capabilities)
        else:
            kwargs["capabilities"] = default_capabilities

        # Set explorer agent type
        kwargs["agent_type"] = "explorer"

        # Initialize base agent
        super().__init__(**kwargs)

        # Initialize explorer-specific components
        self._setup_explorer_components()

    def _setup_explorer_components(self):
        """Setup explorer-specific components"""
        # Initialize exploration map
        self.data.metadata["exploration_map"] = ExplorationMap()
        self.data.metadata["exploration_status"] = ExplorationStatus.IDLE
        self.data.metadata["total_discoveries"] = 0
        self.data.metadata["exploration_efficiency"] = 0.0

        # Add explorer-specific behaviors
        behavior_tree = self.get_component("behavior_tree")
        if behavior_tree:
            behavior_tree.add_behavior(AdvancedExplorationBehavior())
            behavior_tree.add_behavior(PathfindingBehavior())

        # Set explorer goals
        self._setup_default_goals()

    def _setup_default_goals(self):
        """Setup default exploration goals"""
        from agents.base.data_model import AgentGoal

        # Exploration goal
        exploration_goal = AgentGoal(
            goal_id="explore_territory",
            description="Explore and map unknown territories",
            priority=0.8,
            target_position=None,  # Dynamic target
            deadline=datetime.now() + timedelta(hours=24),
        )
        self.data.add_goal(exploration_goal)

        # Discovery goal
        discovery_goal = AgentGoal(
            goal_id="make_discoveries",
            description="Discover resources, locations, and points of interest",
            priority=0.7,
            target_position=None,
            deadline=datetime.now() + timedelta(hours=48),
        )
        self.data.add_goal(discovery_goal)

    def get_exploration_map(self) -> ExplorationMap:
        """Get the agent's exploration map"""
        return self.data.metadata.get("exploration_map")

    def get_discoveries(self) -> List[Discovery]:
        """Get all discoveries made by this explorer"""
        exploration_map = self.get_exploration_map()
        if exploration_map:
            return list(exploration_map.discoveries.values())
        return []

    def get_exploration_status(self) -> ExplorationStatus:
        """Get current exploration status"""
        return self.data.metadata.get("exploration_status", ExplorationStatus.IDLE)

    def set_exploration_target(
        self, target_position: Position, description: str = "Explore target area"
    ) -> None:
        """Set a specific exploration target"""
        from agents.base.data_model import AgentGoal

        target_goal = AgentGoal(
            goal_id=f"explore_target_{datetime.now().timestamp()}",
            description=description,
            priority=0.9,
            target_position=target_position,
            deadline=datetime.now() + timedelta(hours=12),
        )

        self.data.add_goal(target_goal)

    def share_discovery(self, discovery_id: str, target_agent_id: str) -> bool:
        """Share a discovery with another agent"""
        exploration_map = self.get_exploration_map()
        if not exploration_map or discovery_id not in exploration_map.discoveries:
            return False

        discovery = exploration_map.discoveries[discovery_id]

        # Mark as shared
        discovery.shared = True

        # Add to memory
        self.data.add_to_memory(
            {
                "event": "discovery_shared",
                "discovery_id": discovery_id,
                "target_agent": target_agent_id,
                "discovery_type": discovery.discovery_type.value,
                "position": discovery.position.to_array().tolist(),
            }
        )

        return True

    def get_exploration_efficiency(self) -> float:
        """Calculate exploration efficiency score"""
        exploration_map = self.get_exploration_map()
        if not exploration_map:
            return 0.0

        # Base on exploration coverage and discoveries
        exploration_score = exploration_map.get_exploration_score()
        discovery_count = len(exploration_map.discoveries)

        # Efficiency combines coverage with discovery rate
        efficiency = (exploration_score * 0.7) + (min(discovery_count / 10.0, 1.0) * 0.3)

        self.data.metadata["exploration_efficiency"] = efficiency
        return efficiency


def create_explorer_agent(
    name: str = "Explorer", position: Optional[Position] = None, **kwargs
) -> ExplorerAgent:
    """Factory function to create an explorer agent"""

    # Set default position if not provided
    if position is None:
        position = Position(0.0, 0.0, 0.0)

    # Create explorer with default configuration
    explorer = ExplorerAgent(name=name, position=position, **kwargs)

    return explorer


def register_explorer_type() -> None:
    """Register the explorer type with the default factory"""
    factory = get_default_factory()

    def _create_explorer(**kwargs):
        return create_explorer_agent(**kwargs)

    factory.register_type("explorer", _create_explorer)

    # Set explorer-specific default config
    factory.set_default_config(
        "explorer",
        {
            "agent_type": "explorer",
            "capabilities": {
                AgentCapability.MOVEMENT,
                AgentCapability.PERCEPTION,
                AgentCapability.MEMORY,
                AgentCapability.LEARNING,
                AgentCapability.PLANNING,
            },
        },
    )


# Auto-register when module is imported
register_explorer_type()

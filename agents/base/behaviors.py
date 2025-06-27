"""
Module for FreeAgentics Active Inference implementation.
"""

import math
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .data_model import Agent, AgentCapability, AgentStatus, Position, SocialRelationship
from .interfaces import IAgentBehavior, IBehaviorTree

if TYPE_CHECKING:
    from .decision_making import Action, ActionType

"""
Agent Behavior System for FreeAgentics
This module provides the behavior management system for agents, including
behavior trees, concrete behavior implementations, and behavior composition patterns.
"""


class BehaviorPriority(Enum):
    """Behavior priority levels"""

    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    BACKGROUND = 0.2


class BehaviorStatus(Enum):
    """Behavior execution status"""

    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
    READY = "ready"


class BaseBehavior(IAgentBehavior):
    """Base class for all agent behaviors"""

    def __init__(
        self,
        name: str,
        priority: BehaviorPriority = BehaviorPriority.MEDIUM,
        required_capabilities: Optional[set[AgentCapability]] = None,
        cooldown_time: timedelta = timedelta(seconds=0),
    ):
        self.name = name
        self.priority = priority
        self.required_capabilities = required_capabilities or set()
        self.cooldown_time = cooldown_time
        self.last_execution_time: Optional[datetime] = None
        self.status = BehaviorStatus.READY

    def can_execute(self, agent: Agent, context: Dict[str, Any]) -> bool:
        """Check if this behavior can be executed"""
        # Check cooldown
        if self.last_execution_time and self.cooldown_time > timedelta(0):
            if datetime.now() - self.last_execution_time < self.cooldown_time:
                return False
        # Check required capabilities
        if not all(agent.has_capability(cap) for cap in self.required_capabilities):
            return False
        # Check energy requirements
        if agent.resources.energy < self.get_energy_cost(agent, context):
            return False
        # Custom validation
        return self._can_execute_custom(agent, context)

    def execute(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the behavior"""
        self.last_execution_time = datetime.now()
        self.status = BehaviorStatus.RUNNING
        try:
            # Consume energy
            energy_cost = self.get_energy_cost(agent, context)
            agent.resources.consume_energy(energy_cost)
            # Execute custom logic
            result = self._execute_custom(agent, context)
            # Update status based on result
            if result.get("success", False):
                self.status = BehaviorStatus.SUCCESS
            else:
                self.status = BehaviorStatus.FAILURE
            return result
        except Exception as e:
            self.status = BehaviorStatus.FAILURE
            return {"success": False, "error": str(e), "behavior": self.name}

    def get_priority(self, agent: Agent, context: Dict[str, Any]) -> float:
        """Get the priority of this behavior"""
        base_priority = self.priority.value
        # Apply dynamic priority modifiers
        priority_modifier = self._get_priority_modifier(agent, context)
        # Apply personality-based modifier
        personality_modifier = self._get_personality_modifier(agent)
        final_priority = base_priority * priority_modifier * personality_modifier
        return min(1.0, max(0.0, final_priority))

    def _get_personality_modifier(self, agent: Agent) -> float:
        """Get personality-based modifier for this behavior"""
        # Get personality profile from agent metadata
        personality_profile = agent.metadata.get("personality_profile")
        if personality_profile is None:
            return 1.0  # No personality modification
        # Map behavior name to personality trait influence
        behavior_type = self.name.lower().replace("_", "")
        modifier = personality_profile.get_behavior_modifier(behavior_type)
        return modifier

    def get_energy_cost(self, agent: Agent, context: Dict[str, Any]) -> float:
        """Get the energy cost for executing this behavior"""
        return 1.0  # Default energy cost

    def _can_execute_custom(self, agent: Agent, context: Dict[str, Any]) -> bool:
        """Custom execution check - override in subclasses"""
        return True

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Custom execution logic - override in subclasses"""
        return {"success": True, "behavior": self.name}

    def _get_priority_modifier(self, agent: Agent, context: Dict[str, Any]) -> float:
        """Get dynamic priority modifier - override in subclasses"""
        return 1.0


class IdleBehavior(BaseBehavior):
    """Default idle behavior"""

    def __init__(self) -> None:
        super().__init__("idle", BehaviorPriority.BACKGROUND)

    def can_execute(self, agent: Agent, context: Dict[str, Any]) -> bool:
        return True  # Idle can always execute

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        # Restore some energy while idle
        agent.resources.restore_energy(0.5)
        return {"success": True, "action": "idle", "energy_restored": 0.5}


class WanderBehavior(BaseBehavior):
    """Random wandering behavior"""

    def __init__(self) -> None:
        super().__init__(
            "wander",
            BehaviorPriority.LOW,
            {AgentCapability.MOVEMENT},
            timedelta(seconds=2),
        )
        self.max_wander_distance = 5.0

    def _can_execute_custom(self, agent: Agent, context: Dict[str, Any]) -> bool:
        return agent.status in [AgentStatus.IDLE, AgentStatus.MOVING]

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        # Generate random movement direction
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(1.0, self.max_wander_distance)
        new_x = agent.position.x + distance * math.cos(angle)
        new_y = agent.position.y + distance * math.sin(angle)
        new_position = Position(new_x, new_y, agent.position.z)
        # Check if movement is valid through world interface
        world_interface = context.get("world_interface")
        if world_interface and not world_interface.can_move_to(agent, new_position):
            return {"success": False, "reason": "invalid_position"}
        action = Action(
            action_type=ActionType.MOVE,
            target_position=new_position,
            parameters={"movement_type": "wander"},
        )
        return {"success": True, "action": action, "target_position": new_position}

    def get_energy_cost(self, agent: Agent, context: Dict[str, Any]) -> float:
        return 2.0


class GoalSeekingBehavior(BaseBehavior):
    """Behavior for pursuing agent goals"""

    def __init__(self) -> None:
        super().__init__(
            "goal_seeking",
            BehaviorPriority.HIGH,
            {AgentCapability.MOVEMENT, AgentCapability.PLANNING},
        )

    def _can_execute_custom(self, agent: Agent, context: Dict[str, Any]) -> bool:
        # Can execute if agent has active goals
        return len(agent.goals) > 0 and agent.select_next_goal() is not None

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = agent.select_next_goal()
        if not goal:
            return {"success": False, "reason": "no_active_goals"}
        # If goal has a target position, move towards it
        if goal.target_position:
            current_pos = agent.position
            target_pos = goal.target_position
            # Calculate direction to goal
            dx = target_pos.x - current_pos.x
            dy = target_pos.y - current_pos.y
            distance = math.sqrt(dx * dx + dy * dy)
            # If close enough, consider goal accomplished
            if distance < 1.0:
                goal.completed = True
                goal.progress = 1.0
                agent.add_to_memory(
                    {
                        "event": "goal_completed",
                        "goal": goal.description,
                        "position": current_pos.to_array().tolist(),
                    },
                    is_important=True,
                )
                return {
                    "success": True,
                    "goal_completed": True,
                    "goal": goal.description,
                }
            # Move towards goal
            move_distance = min(3.0, distance)  # Maximum movement per step
            unit_dx = dx / distance
            unit_dy = dy / distance
            new_position = Position(
                current_pos.x + unit_dx * move_distance,
                current_pos.y + unit_dy * move_distance,
                current_pos.z,
            )
            # Update goal progress
            goal.progress = min(1.0, 1.0 - (distance - move_distance) / distance)
            action = Action(
                action_type=ActionType.MOVE,
                target_position=new_position,
                parameters={"goal_id": goal.goal_id, "movement_type": "goal_seeking"},
            )
            return {
                "success": True,
                "action": action,
                "goal": goal.description,
                "progress": goal.progress,
            }
        return {"success": False, "reason": "goal_has_no_position"}

    def _get_priority_modifier(self, agent: Agent, context: Dict[str, Any]) -> float:
        # Higher priority if agent has urgent goals
        active_goals = [g for g in agent.goals if not g.completed and not g.is_expired()]
        if not active_goals:
            return 0.0
        # Increase priority based on goal urgency and importance
        highest_priority = max(goal.priority for goal in active_goals)
        return 1.0 + highest_priority  # Boost priority based on goal importance

    def get_energy_cost(self, agent: Agent, context: Dict[str, Any]) -> float:
        return 3.0


class SocialInteractionBehavior(BaseBehavior):
    """Behavior for social interactions between agents"""

    def __init__(self) -> None:
        super().__init__(
            "social_interaction",
            BehaviorPriority.MEDIUM,
            {AgentCapability.SOCIAL_INTERACTION, AgentCapability.COMMUNICATION},
        )
        self.interaction_radius = 5.0

    def _can_execute_custom(self, agent: Agent, context: Dict[str, Any]) -> bool:
        # Check if there are other agents nearby
        world_interface = context.get("world_interface")
        if not world_interface:
            return False
        nearby_objects = world_interface.get_nearby_objects(agent.position, self.interaction_radius)
        nearby_agents = [obj for obj in nearby_objects if obj.get("type") == "agent"]
        return len(nearby_agents) > 0

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        world_interface = context.get("world_interface")
        if not world_interface:
            return {"success": False, "reason": "no_world_interface"}
        # Find nearby agents
        nearby_objects = world_interface.get_nearby_objects(agent.position, self.interaction_radius)
        nearby_agents = [obj for obj in nearby_objects if obj.get("type") == "agent"]
        if not nearby_agents:
            return {"success": False, "reason": "no_nearby_agents"}
        # Select an agent to interact with
        target_agent = random.choice(nearby_agents)
        target_agent_id = target_agent["agent_id"]
        # Check existing relationship
        relationship = agent.get_relationship(target_agent_id)
        # Create or update relationship
        if not relationship:
            relationship = SocialRelationship(
                target_agent_id=target_agent_id,
                relationship_type="neutral",
                trust_level=0.5,
                interaction_count=0,
            )
            agent.add_relationship(relationship)
        # Perform interaction based on personality
        interaction_outcome = self._perform_interaction(agent, relationship)
        # Update relationship
        relationship.interaction_count += 1
        relationship.last_interaction = datetime.now()
        relationship.update_trust(interaction_outcome["trust_change"])
        # Add to memory
        agent.add_to_memory(
            {
                "event": "social_interaction",
                "target_agent": target_agent_id,
                "outcome": interaction_outcome,
                "relationship_type": relationship.relationship_type,
                "trust_level": relationship.trust_level,
            }
        )
        return {
            "success": True,
            "interaction_type": "social",
            "target_agent": target_agent_id,
            "outcome": interaction_outcome,
        }

    def _perform_interaction(
        self, agent: Agent, relationship: SocialRelationship
    ) -> Dict[str, Any]:
        """Perform the actual social interaction"""
        # Interaction outcome based on personality and existing relationship
        personality = agent.personality
        # Calculate interaction success based on personality traits
        social_factor = (personality.extraversion + personality.agreeableness) / 2.0
        trust_factor = relationship.trust_level
        success_probability = (social_factor + trust_factor) / 2.0
        success = random.random() < success_probability
        if success:
            trust_change = random.uniform(0.01, 0.05)  # Small positive change
            interaction_type = "positive"
        else:
            trust_change = random.uniform(-0.02, 0.01)  # Small negative change
            interaction_type = "neutral"
        return {
            "type": interaction_type,
            "success": success,
            "trust_change": trust_change,
        }

    def get_energy_cost(self, agent: Agent, context: Dict[str, Any]) -> float:
        # Social interactions cost more energy for introverted agents
        introversion_factor = 1.0 - agent.personality.extraversion
        return 2.0 + introversion_factor * 2.0


class ExplorationBehavior(BaseBehavior):
    """Behavior for exploring unknown areas"""

    def __init__(self) -> None:
        super().__init__(
            "exploration",
            BehaviorPriority.MEDIUM,
            {AgentCapability.MOVEMENT, AgentCapability.PERCEPTION},
        )
        self.exploration_radius = 10.0

    def _can_execute_custom(self, agent: Agent, context: Dict[str, Any]) -> bool:
        # Can execute if agent has high openness or is an explorer type
        return (
            agent.personality.openness > 0.6
            or agent.agent_type == "explorer"
            or len(agent.goals) == 0  # Explore when no goals
        )

    def _execute_custom(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        # Choose exploration direction based on least explored areas
        # For now, use a simple approach of moving to unexplored directions
        explored_areas = self._get_explored_areas(agent)
        # Find least explored direction
        best_direction = self._find_best_exploration_direction(agent, explored_areas)
        # Move in that direction
        distance = random.uniform(5.0, self.exploration_radius)
        new_position = Position(
            agent.position.x + distance * math.cos(best_direction),
            agent.position.y + distance * math.sin(best_direction),
            agent.position.z,
        )
        # Check validity
        world_interface = context.get("world_interface")
        if world_interface and not world_interface.can_move_to(agent, new_position):
            # Try a different direction
            best_direction += math.pi / 4
            new_position = Position(
                agent.position.x + distance * math.cos(best_direction),
                agent.position.y + distance * math.sin(best_direction),
                agent.position.z,
            )
        action = Action(
            action_type=ActionType.MOVE,
            target_position=new_position,
            parameters={"movement_type": "exploration", "direction": best_direction},
        )
        # Add exploration memory
        agent.add_to_memory(
            {
                "event": "exploration",
                "direction": best_direction,
                "target_position": new_position.to_array().tolist(),
            }
        )
        return {
            "success": True,
            "action": action,
            "exploration_direction": best_direction,
        }

    def _get_explored_areas(self, agent: Agent) -> List[Position]:
        """Get areas the agent has explored based on memory"""
        explored_positions = []
        for memory in agent.short_term_memory + agent.long_term_memory:
            if "position" in memory.get("experience", {}):
                pos_data = memory["experience"]["position"]
                if isinstance(pos_data, list) and len(pos_data) >= 2:
                    explored_positions.append(
                        Position(
                            pos_data[0],
                            pos_data[1],
                            pos_data[2] if len(pos_data) > 2 else 0,
                        )
                    )
        return explored_positions

    def _find_best_exploration_direction(
        self, agent: Agent, explored_areas: List[Position]
    ) -> float:
        """Find the direction with least exploration"""
        if not explored_areas:
            return random.uniform(0, 2 * math.pi)
        # Test 8 directions and find the one with least nearby explored areas
        directions = [i * math.pi / 4 for i in range(8)]
        direction_scores = []
        for direction in directions:
            # Count explored areas in this direction
            test_position = Position(
                agent.position.x + 5.0 * math.cos(direction),
                agent.position.y + 5.0 * math.sin(direction),
                agent.position.z,
            )
            nearby_count = sum(1 for pos in explored_areas if pos.distance_to(test_position) < 3.0)
            direction_scores.append((direction, nearby_count))
        # Choose direction with minimum explored areas
        best_direction = min(direction_scores, key=lambda x: x[1])[0]
        return best_direction

    def _get_priority_modifier(self, agent: Agent, context: Dict[str, Any]) -> float:
        # Higher priority for explorer types and agents with high openness
        if agent.agent_type == "explorer":
            return 1.5
        return 0.5 + agent.personality.openness

    def get_energy_cost(self, agent: Agent, context: Dict[str, Any]) -> float:
        return 4.0


class BehaviorTreeManager(IBehaviorTree):
    """Manages and evaluates behavior trees for agents"""

    def __init__(self) -> None:
        self.behaviors: List[IAgentBehavior] = []
        self.default_behaviors = [
            IdleBehavior(),
            WanderBehavior(),
            GoalSeekingBehavior(),
            SocialInteractionBehavior(),
            ExplorationBehavior(),
        ]
        # Add default behaviors
        for behavior in self.default_behaviors:
            self.add_behavior(behavior)

    def add_behavior(self, behavior: IAgentBehavior) -> None:
        """Add a behavior to the tree"""
        if behavior not in self.behaviors:
            self.behaviors.append(behavior)

    def remove_behavior(self, behavior: IAgentBehavior) -> None:
        """Remove a behavior from the tree"""
        if behavior in self.behaviors:
            self.behaviors.remove(behavior)

    def evaluate(self, agent: Agent, context: Dict[str, Any]) -> Optional[IAgentBehavior]:
        """Evaluate the tree and return the best behavior to execute"""
        valid_behaviors = []
        # Filter behaviors that can execute
        for behavior in self.behaviors:
            if behavior.can_execute(agent, context):
                priority = behavior.get_priority(agent, context)
                valid_behaviors.append((behavior, priority))
        if not valid_behaviors:
            return None
        # Sort by priority (highest first)
        valid_behaviors.sort(key=lambda x: x[1], reverse=True)
        # Return highest priority behavior
        return valid_behaviors[0][0]


# Factory function for creating behavior trees
def create_behavior_tree(agent_type: str = "basic") -> BehaviorTreeManager:
    """
    Create a behavior tree customized for specific agent types.
    Args:
        agent_type: Type of agent to create behavior tree for
    Returns:
        BehaviorTreeManager instance
    """
    tree = BehaviorTreeManager()
    # Add specialized behaviors based on agent type
    if agent_type == "explorer":
        # Explorers prioritize exploration and movement
        tree.add_behavior(ExplorationBehavior())
    elif agent_type == "merchant":
        # Merchants would have trading behaviors (to be implemented)
        pass
    elif agent_type == "scholar":
        # Scholars would have learning behaviors (to be implemented)
        pass
    elif agent_type == "guardian":
        # Guardians would have protection behaviors (to be implemented)
        pass
    return tree

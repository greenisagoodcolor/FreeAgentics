"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from .agent import BaseAgent, create_agent
from .behaviors import create_behavior_tree
from .data_model import Agent as AgentData
from .data_model import AgentCapability, AgentPersonality, Position
from .interfaces import IAgentEventHandler, IAgentFactory, IAgentRegistry

"""
Agent Factory and Registry for FreeAgentics
This module provides factory and registry implementations for creating and managing agents,
following the ADR-002 canonical structure and ADR-003 dependency rules.
"""

# Import personality system (avoiding circular imports)
try:
    from .personality_system import create_personality_profile
except ImportError:
    # Handle case where personality_system is not available
    create_personality_profile = None


class AgentFactory(IAgentFactory):
    """Factory for creating different types of agents"""

    def __init__(self) -> None:
        self._agent_types: Dict[str, Callable[..., BaseAgent]] = {}
        self._default_configs: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("agent.factory")
        # Register default agent types
        self._register_default_types()

    def _register_default_types(self) -> None:
        """Register default agent types with their configurations"""
        # Basic agent
        self.register_type("basic", self._create_basic_agent)
        self._default_configs["basic"] = {
            "capabilities": {
                AgentCapability.MOVEMENT,
                AgentCapability.PERCEPTION,
                AgentCapability.COMMUNICATION,
                AgentCapability.MEMORY,
            },
            "personality": AgentPersonality(
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.3,
            ),
        }
        # Explorer agent
        self.register_type("explorer", self._create_explorer_agent)
        self._default_configs["explorer"] = {
            "capabilities": {
                AgentCapability.MOVEMENT,
                AgentCapability.PERCEPTION,
                AgentCapability.MEMORY,
                AgentCapability.LEARNING,
            },
            "personality": AgentPersonality(
                openness=0.9,  # High openness for exploration
                conscientiousness=0.7,
                extraversion=0.6,
                agreeableness=0.5,
                neuroticism=0.2,  # Low neuroticism for risk-taking
            ),
        }
        # Merchant agent
        self.register_type("merchant", self._create_merchant_agent)
        self._default_configs["merchant"] = {
            "capabilities": {
                AgentCapability.COMMUNICATION,
                AgentCapability.RESOURCE_MANAGEMENT,
                AgentCapability.SOCIAL_INTERACTION,
                AgentCapability.MEMORY,
            },
            "personality": AgentPersonality(
                openness=0.6,
                conscientiousness=0.8,  # High conscientiousness for business
                extraversion=0.8,  # High extraversion for trading
                agreeableness=0.6,
                neuroticism=0.3,
            ),
        }
        # Scholar agent
        self.register_type("scholar", self._create_scholar_agent)
        self._default_configs["scholar"] = {
            "capabilities": {
                AgentCapability.LEARNING,
                AgentCapability.MEMORY,
                AgentCapability.PERCEPTION,
                AgentCapability.PLANNING,
            },
            "personality": AgentPersonality(
                openness=0.9,  # High openness for learning
                conscientiousness=0.9,  # High conscientiousness for study
                extraversion=0.3,  # Lower extraversion (more focused)
                agreeableness=0.7,
                neuroticism=0.4,
            ),
        }
        # Guardian agent
        self.register_type("guardian", self._create_guardian_agent)
        self._default_configs["guardian"] = {
            "capabilities": {
                AgentCapability.MOVEMENT,
                AgentCapability.PERCEPTION,
                AgentCapability.PLANNING,
                AgentCapability.COMMUNICATION,
            },
            "personality": AgentPersonality(
                openness=0.4,  # Lower openness (more conservative)
                conscientiousness=0.9,  # High conscientiousness for duty
                extraversion=0.5,
                agreeableness=0.8,  # High agreeableness for protection
                neuroticism=0.2,  # Low neuroticism for stability
            ),
        }

    def create_agent(self, agent_type: str, **kwargs) -> BaseAgent:
        """Create an agent of the specified type"""
        if agent_type not in self._agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        # Merge default config with provided kwargs
        config = self._default_configs.get(agent_type, {}).copy()
        config.update(kwargs)
        # Create the agent
        agent = self._agent_types[agent_type](**config)
        # Create and attach personality profile
        self._attach_personality_profile(agent, agent_type, config.get("personality_traits"))
        self.logger.info(f"Created {agent_type} agent: {agent.agent_id}")
        return agent

    def _attach_personality_profile(
        self,
        agent: BaseAgent,
        agent_type: str,
        trait_values: Optional[Dict[str, float]] = None,
    ) -> None:
        """Attach a personality profile to an agent"""
        try:
            # Import here to avoid circular imports
            # Create personality profile for the agent
            personality_profile = create_personality_profile(
                agent_type=agent_type, trait_values=trait_values
            )
            # Store the personality profile in agent metadata
            agent.data.metadata["personality_profile"] = personality_profile
            # Also update the legacy personality field for backward compatibility
            big_five_traits = [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ]
            for trait_name in big_five_traits:
                if hasattr(agent.data.personality, trait_name):
                    setattr(
                        agent.data.personality,
                        trait_name,
                        personality_profile.get_trait_value(trait_name),
                    )
        except ImportError:
            # Fallback if personality system is not available
            self.logger.warning(
                "Personality system not available, skipping personality profile creation"
            )
            pass

    def get_supported_types(self) -> List[str]:
        ."""Get list of supported agent types."""
        return list(self._agent_types.keys())

    def register_type(self, agent_type: str, factory_func: Callable[..., BaseAgent]) -> None:
        ."""Register a new agent type with its factory function."""
        self._agent_types[agent_type] = factory_func
        self.logger.info(f"Registered agent type: {agent_type}")

    def set_default_config(self, agent_type: str, config: Dict[str, Any]) -> None:
        ."""Set default configuration for an agent type."""
        self._default_configs[agent_type] = config

    def get_default_config(self, agent_type: str) -> Dict[str, Any]:
        ."""Get default configuration for an agent type."""
        return self._default_configs.get(agent_type, {})

    # Agent creation methods
    def _create_basic_agent(self, **kwargs) -> BaseAgent:
        ."""Create a basic agent."""
        return create_agent(agent_type="basic", **kwargs)

    def _create_explorer_agent(self, **kwargs) -> BaseAgent:
        """Create an explorer agent"""
        agent = create_agent(agent_type="explorer", **kwargs)
        # Add exploration-specific behaviors
        behavior_tree = create_behavior_tree("explorer")
        agent.get_component("behavior_tree").behaviors = behavior_tree.behaviors
        return agent

    def _create_merchant_agent(self, **kwargs) -> BaseAgent:
        """Create a merchant agent"""
        agent = create_agent(agent_type="merchant", **kwargs)
        # Add merchant-specific setup
        if isinstance(agent.data, AgentData):
            # Convert to ResourceAgent if needed
            pass
        return agent

    def _create_scholar_agent(self, **kwargs) -> BaseAgent:
        """Create a scholar agent"""
        agent = create_agent(agent_type="scholar", **kwargs)
        # Add scholar-specific setup
        # Enhanced memory and learning capabilities
        memory_system = agent.get_component("memory")
        if memory_system:
            # Increase memory capacity for scholars
            agent.data.resources.memory_capacity = 200.0
        return agent

    def _create_guardian_agent(self, **kwargs) -> BaseAgent:
        """Create a guardian agent"""
        agent = create_agent(agent_type="guardian", **kwargs)
        # Add guardian-specific setup
        # Enhanced health and protective behaviors
        agent.data.resources.health = 150.0  # Higher health for guardians
        return agent


class AgentRegistry(IAgentRegistry):
    """Registry for managing active agents"""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self._agents_by_type: Dict[str, List[BaseAgent]] = defaultdict(list)
        self._spatial_index: Dict[tuple, List[BaseAgent]] = defaultdict(list)
        self._event_handlers: List[IAgentEventHandler] = []
        self._lock = threading.RLock()
        self.logger = logging.getLogger("agent.registry")
        # Spatial indexing parameters
        self._spatial_resolution = 10.0  # Grid cell size for spatial indexing

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent in the registry"""
        with self._lock:
            if agent.agent_id in self._agents:
                self.logger.warning(f"Agent {agent.agent_id} is already registered")
                return
            # Add to main registry
            self._agents[agent.agent_id] = agent
            # Add to type index
            self._agents_by_type[agent.data.agent_type].append(agent)
            # Add to spatial index
            self._update_spatial_index(agent)
            # Notify event handlers
            for handler in self._event_handlers:
                handler.on_agent_created(agent.data)
            self.logger.info(f"Registered agent: {agent.agent_id} ({agent.data.agent_type})")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the registry"""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                self.logger.warning(f"Agent {agent_id} not found in registry")
                return
            # Remove from main registry
            del self._agents[agent_id]
            # Remove from type index
            if agent in self._agents_by_type[agent.data.agent_type]:
                self._agents_by_type[agent.data.agent_type].remove(agent)
            # Remove from spatial index
            self._remove_from_spatial_index(agent)
            # Notify event handlers
            for handler in self._event_handlers:
                handler.on_agent_destroyed(agent.data)
            self.logger.info(f"Unregistered agent: {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        ."""Get an agent by ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def get_all_agents(self) -> List[BaseAgent]:
        ."""Get all registered agents."""
        with self._lock:
            return list(self._agents.values())

    def find_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        ."""Find agents by type."""
        with self._lock:
            return list(self._agents_by_type.get(agent_type, []))

    def find_agents_in_range(self, position: Position, radius: float) -> List[BaseAgent]:
        """Find agents within a specified range of a position"""
        with self._lock:
            nearby_agents = []
            # Get spatial grid cells that might contain nearby agents
            cells_to_check = self._get_spatial_cells_in_range(position, radius)
            # Check agents in relevant cells
            candidate_agents = set()
            for cell in cells_to_check:
                candidate_agents.update(self._spatial_index.get(cell, []))
            # Filter by actual distance
            for agent in candidate_agents:
                if position.distance_to(agent.data.position) <= radius:
                    nearby_agents.append(agent)
            return nearby_agents

    def get_agent_count(self) -> int:
        ."""Get total number of registered agents."""
        with self._lock:
            return len(self._agents)

    def get_agent_count_by_type(self) -> Dict[str, int]:
        ."""Get agent count by type."""
        with self._lock:
            return {agent_type: len(agents) for agent_type, agents in self._agents_by_type.items()}

    def update_agent_position(self, agent_id: str, new_position: Position) -> None:
        """Update an agent's position in the spatial index"""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return
            # Remove from old spatial location
            self._remove_from_spatial_index(agent)
            # Update position
            old_position = agent.data.position
            agent.data.update_position(new_position)
            # Add to new spatial location
            self._update_spatial_index(agent)
            # Notify event handlers
            for handler in self._event_handlers:
                handler.on_agent_moved(agent.data, old_position, new_position)

    def add_event_handler(self, handler: IAgentEventHandler) -> None:
        ."""Add an event handler."""
        self._event_handlers.append(handler)

    def remove_event_handler(self, handler: IAgentEventHandler) -> None:
        ."""Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    def _update_spatial_index(self, agent: BaseAgent) -> None:
        ."""Update agent in spatial index."""
        cell = self._get_spatial_cell(agent.data.position)
        if agent not in self._spatial_index[cell]:
            self._spatial_index[cell].append(agent)

    def _remove_from_spatial_index(self, agent: BaseAgent) -> None:
        """Remove agent from spatial index"""
        cell = self._get_spatial_cell(agent.data.position)
        if agent in self._spatial_index[cell]:
            self._spatial_index[cell].remove(agent)
        # Clean up empty cells
        if not self._spatial_index[cell]:
            del self._spatial_index[cell]

    def _get_spatial_cell(self, position: Position) -> tuple:
        ."""Get spatial grid cell for a position."""
        cell_x = int(position.x // self._spatial_resolution)
        cell_y = int(position.y // self._spatial_resolution)
        return (cell_x, cell_y)

    def _get_spatial_cells_in_range(self, position: Position, radius: float) -> List[tuple]:
        """Get all spatial cells that might contain agents within range"""
        cells = []
        # Calculate cell range
        cell_radius = int((radius / self._spatial_resolution) + 1)
        center_cell = self._get_spatial_cell(position)
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                cells.append(cell)
        return cells

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            return {
                "total_agents": len(self._agents),
                "agents_by_type": self.get_agent_count_by_type(),
                "spatial_cells_used": len(self._spatial_index),
                "average_agents_per_cell": (
                    (
                        sum(len(agents) for agents in self._spatial_index.values())
                        / len(self._spatial_index)
                    )
                    if self._spatial_index
                    else 0
                ),
            }


# Global instances for convenience
_default_factory = None
_default_registry = None


def get_default_factory() -> AgentFactory:
    """Get the default agent factory instance"""
    global _default_factory
    if _default_factory is None:
        _default_factory = AgentFactory()
    return _default_factory


def get_default_registry() -> AgentRegistry:
    """Get the default agent registry instance"""
    global _default_registry
    if _default_registry is None:
        _default_registry = AgentRegistry()
    return _default_registry


# Convenience functions
def create_and_register_agent(agent_type: str, **kwargs) -> BaseAgent:
    """Create and register an agent in one step"""
    factory = get_default_factory()
    registry = get_default_registry()
    agent = factory.create_agent(agent_type, **kwargs)
    registry.register_agent(agent)
    return agent


def quick_agent_setup(
    agent_type: str, name: str, position: Optional[Position] = None, **kwargs
) -> BaseAgent:
    """Quick setup for creating and registering an agent"""
    return create_and_register_agent(
        agent_type=agent_type,
        name=name,
        position=position or Position(0.0, 0.0, 0.0),
        **kwargs,
    )

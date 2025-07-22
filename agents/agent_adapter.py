"""Adapter to bridge ActiveInferenceAgent and GridWorld Agent types."""

import logging
from typing import Any, Dict, Optional

from agents.base_agent import ActiveInferenceAgent
from world.grid_world import Agent as GridAgent
from world.grid_world import Position

logger = logging.getLogger(__name__)


class ActiveInferenceGridAdapter:
    """Adapter that bridges ActiveInferenceAgent with GridWorld Agent."""

    def __init__(self):
        """Initialize the adapter."""
        # Maps AI agent IDs to grid agents
        self.ai_to_grid: Dict[str, GridAgent] = {}
        # Maps grid agent IDs to AI agents
        self.grid_to_ai: Dict[str, ActiveInferenceAgent] = {}

    def register_agent(
        self, ai_agent: ActiveInferenceAgent, initial_position: Position
    ) -> GridAgent:
        """Register an ActiveInferenceAgent and create corresponding GridAgent.

        Args:
            ai_agent: The ActiveInferenceAgent instance
            initial_position: Starting position in the grid

        Returns:
            Created GridAgent instance
        """
        # Create grid agent with same ID
        grid_agent = GridAgent(
            id=ai_agent.agent_id,
            position=initial_position,
            energy=100.0,
            resources={},
        )

        # Store mappings
        self.ai_to_grid[ai_agent.agent_id] = grid_agent
        self.grid_to_ai[ai_agent.agent_id] = ai_agent

        # Synchronize AI agent's position with grid position
        if hasattr(ai_agent, "position"):
            ai_agent.position = initial_position

        logger.info(f"Registered agent {ai_agent.agent_id} at position {initial_position}")
        return grid_agent

    def unregister_agent(self, agent_id: str):
        """Remove agent from adapter mappings.

        Args:
            agent_id: ID of agent to remove
        """
        if agent_id in self.ai_to_grid:
            del self.ai_to_grid[agent_id]
        if agent_id in self.grid_to_ai:
            del self.grid_to_ai[agent_id]
        logger.info(f"Unregistered agent {agent_id}")

    def get_grid_agent(self, ai_agent_id: str) -> Optional[GridAgent]:
        """Get GridAgent for given ActiveInferenceAgent ID.

        Args:
            ai_agent_id: ActiveInferenceAgent ID

        Returns:
            Corresponding GridAgent or None
        """
        return self.ai_to_grid.get(ai_agent_id)

    def get_ai_agent(self, grid_agent_id: str) -> Optional[ActiveInferenceAgent]:
        """Get ActiveInferenceAgent for given GridAgent ID.

        Args:
            grid_agent_id: GridAgent ID

        Returns:
            Corresponding ActiveInferenceAgent or None
        """
        return self.grid_to_ai.get(grid_agent_id)

    def convert_observation(self, grid_observation: Dict[str, Any]) -> Dict[str, Any]:
        """Convert GridWorld observation to ActiveInferenceAgent format.

        Args:
            grid_observation: Observation from GridWorld

        Returns:
            Observation in AI agent format
        """
        if not grid_observation:
            return {}

        # Extract position and surroundings
        agent_pos = grid_observation.get("agent_position")
        local_grid = grid_observation.get("local_grid", {})

        # Convert to AI agent format
        ai_observation = {
            "position": [agent_pos.x, agent_pos.y] if agent_pos else [0, 0],
            "surroundings": local_grid,
            "energy": grid_observation.get("energy", 100.0),
            "resources": grid_observation.get("resources", {}),
            "nearby_agents": grid_observation.get("nearby_agents", []),
            "goals": grid_observation.get("goals", []),
        }

        return ai_observation

    def convert_action(self, ai_action: str, current_position: Position) -> Position:
        """Convert AI agent action to GridWorld position.

        Args:
            ai_action: Action from AI agent (up/down/left/right/stay)
            current_position: Current position in grid

        Returns:
            New position after action
        """
        x, y = current_position.x, current_position.y

        if ai_action == "up":
            return Position(x, y - 1)
        elif ai_action == "down":
            return Position(x, y + 1)
        elif ai_action == "left":
            return Position(x - 1, y)
        elif ai_action == "right":
            return Position(x + 1, y)
        else:  # stay or unknown action
            return current_position

    def sync_agent_state(self, agent_id: str):
        """Synchronize state between AI and Grid agents.

        Args:
            agent_id: Agent ID to sync
        """
        ai_agent = self.grid_to_ai.get(agent_id)
        grid_agent = self.ai_to_grid.get(agent_id)

        if ai_agent and grid_agent:
            # Update AI agent's internal position tracking
            if hasattr(ai_agent, "position"):
                ai_agent.position = [
                    grid_agent.position.x,
                    grid_agent.position.y,
                ]

            # Update metrics if available
            if hasattr(ai_agent, "metrics"):
                ai_agent.metrics["energy"] = grid_agent.energy
                ai_agent.metrics["resources_collected"] = len(grid_agent.resources)

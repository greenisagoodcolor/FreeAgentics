"""Comprehensive grid world environment for agent interaction.

This module provides a fully-featured 2D grid world environment
for multi-agent Active Inference simulations.
"""

import heapq
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class CellType(Enum):
    """Types of cells in the grid world."""

    EMPTY = "empty"
    WALL = "wall"
    GOAL = "goal"
    HAZARD = "hazard"
    RESOURCE = "resource"


@dataclass
class Position:
    """2D position in the grid world."""

    x: int
    y: int

    def __eq__(self, other) -> bool:
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __lt__(self, other) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position."""
        return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))

    def manhattan_distance(self, other: "Position") -> int:
        """Calculate Manhattan distance to another position."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def get_neighbors(self) -> List["Position"]:
        """Get neighboring positions (up, down, left, right)."""
        return [
            Position(self.x - 1, self.y),  # Left
            Position(self.x + 1, self.y),  # Right
            Position(self.x, self.y - 1),  # Up
            Position(self.x, self.y + 1),  # Down
        ]


@dataclass
class Cell:
    """A cell in the grid world."""

    type: CellType
    position: Position
    value: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)

    def is_passable(self) -> bool:
        """Check if the cell can be moved through."""
        return self.type != CellType.WALL


@dataclass
class Agent:
    """An agent in the grid world."""

    id: str
    position: Position
    energy: float = 100.0
    resources: Dict[str, int] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridWorldConfig:
    """Configuration for the grid world."""

    width: int = 10
    height: int = 10
    wrap_edges: bool = False
    max_agents: int = 10
    enable_collisions: bool = True
    resource_respawn: bool = False
    step_penalty: float = -0.1


class GridWorld:
    """2D grid world environment for agent interaction.

    This is a comprehensive environment where agents can move around,
    observe their surroundings, collect resources, and interact with
    various cell types.
    """

    def __init__(self, config: GridWorldConfig):
        """Initialize grid world.

        Args:
            config: Configuration for the world
        """
        self.config = config
        self.width = config.width
        self.height = config.height

        # Initialize grid with empty cells
        self.grid: List[List[Cell]] = []
        for x in range(self.width):
            row = []
            for y in range(self.height):
                cell = Cell(CellType.EMPTY, Position(x, y))
                row.append(cell)
            self.grid.append(row)

        # Track agents in the world
        self.agents: Dict[str, Agent] = {}

        # World state
        self.step_count = 0
        self.created_at = datetime.now()

        logger.info(f"Created {self.width}x{self.height} grid world")

    def is_valid_position(self, position: Position) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def add_agent(self, agent: Agent) -> bool:
        """Add an agent to the world.

        Args:
            agent: Agent to add

        Returns:
            True if agent was added successfully
        """
        if agent.id in self.agents:
            logger.warning(f"Agent {agent.id} already in world")
            return False

        if not self.is_valid_position(agent.position):
            logger.error(f"Invalid position: {agent.position}")
            return False

        if len(self.agents) >= self.config.max_agents:
            logger.error("Maximum number of agents reached")
            return False

        # Check for collision if enabled
        if self.config.enable_collisions:
            existing_agent = self.get_agent_at_position(agent.position)
            if existing_agent is not None:
                logger.error(f"Position {agent.position} already occupied")
                return False

        self.agents[agent.id] = agent
        logger.info(f"Added agent {agent.id} at position {agent.position}")
        return True

    def remove_agent(self, agent_id: str) -> Optional[Agent]:
        """Remove an agent from the world.

        Args:
            agent_id: Agent to remove

        Returns:
            Removed agent or None if not found
        """
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            logger.info(f"Removed agent {agent_id}")
            return agent
        return None

    def get_agent_at_position(self, position: Position) -> Optional[Agent]:
        """Get agent at the specified position.

        Args:
            position: Position to check

        Returns:
            Agent at position or None
        """
        for agent in self.agents.values():
            if agent.position == position:
                return agent
        return None

    def move_agent(self, agent_id: str, new_position: Position) -> bool:
        """Move an agent to a new position.

        Args:
            agent_id: Agent to move
            new_position: Target position

        Returns:
            True if move was successful
        """
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]

        if not self.is_valid_position(new_position):
            return False

        # Check if target cell is passable
        target_cell = self.get_cell(new_position)
        if target_cell and not target_cell.is_passable():
            return False

        # Check for collision if enabled
        if self.config.enable_collisions:
            existing_agent = self.get_agent_at_position(new_position)
            if existing_agent is not None and existing_agent.id != agent_id:
                return False

        # Move agent
        agent.position = new_position
        return True

    def set_cell(
        self,
        position: Position,
        cell_type: CellType,
        value: float = 0.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set the type of a cell at the specified position.

        Args:
            position: Position to modify
            cell_type: New cell type
            value: Cell value
            properties: Additional properties

        Returns:
            True if cell was set successfully
        """
        if not self.is_valid_position(position):
            return False

        cell = self.grid[position.x][position.y]
        cell.type = cell_type
        cell.value = value
        if properties:
            cell.properties.update(properties)

        return True

    def get_cell(self, position: Position) -> Optional[Cell]:
        """Get the cell at the specified position.

        Args:
            position: Position to check

        Returns:
            Cell at position or None if invalid
        """
        if not self.is_valid_position(position):
            return None

        return self.grid[position.x][position.y]

    def get_neighbors(self, position: Position) -> List[Position]:
        """Get valid neighboring positions.

        Args:
            position: Center position

        Returns:
            List of valid neighboring positions
        """
        neighbors = []

        for neighbor_pos in position.get_neighbors():
            if self.config.wrap_edges:
                # Wrap around edges
                wrapped_x = neighbor_pos.x % self.width
                wrapped_y = neighbor_pos.y % self.height
                neighbors.append(Position(wrapped_x, wrapped_y))
            elif self.is_valid_position(neighbor_pos):
                neighbors.append(neighbor_pos)

        return neighbors

    def get_observation(self, agent_id: str, radius: int = 1) -> Optional[Dict[str, Any]]:
        """Get observation for an agent.

        Args:
            agent_id: Agent requesting observation
            radius: Observation radius

        Returns:
            Observation dictionary or None if agent not found
        """
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]

        # Get local grid around agent
        local_grid = []
        for dx in range(-radius, radius + 1):
            row = []
            for dy in range(-radius, radius + 1):
                check_pos = Position(agent.position.x + dx, agent.position.y + dy)
                cell = self.get_cell(check_pos)
                if cell:
                    row.append(
                        {
                            "type": cell.type.value,
                            "value": cell.value,
                            "properties": cell.properties,
                        }
                    )
                else:
                    row.append({"type": "out_of_bounds", "value": 0.0, "properties": {}})
            local_grid.append(row)

        # Get nearby agents
        nearby_agents = []
        for other_agent in self.agents.values():
            if other_agent.id != agent_id:
                distance = agent.position.distance_to(other_agent.position)
                if distance <= radius * np.sqrt(2):  # Within observation radius
                    nearby_agents.append(
                        {
                            "id": other_agent.id,
                            "position": {"x": other_agent.position.x, "y": other_agent.position.y},
                            "distance": distance,
                        }
                    )

        return {
            "agent_position": agent.position,
            "local_grid": local_grid,
            "nearby_agents": nearby_agents,
            "step_count": self.step_count,
        }

    def step(self):
        """Advance world by one time step."""
        self.step_count += 1

        # Apply step penalty to all agents
        for agent in self.agents.values():
            agent.energy += self.config.step_penalty

        # Could add other world dynamics here
        logger.debug(f"World step {self.step_count}")

    def reset(self):
        """Reset the world to initial state."""
        self.step_count = 0
        self.agents.clear()

        # Reset all cells to empty
        for x in range(self.width):
            for y in range(self.height):
                self.grid[x][y].type = CellType.EMPTY
                self.grid[x][y].value = 0.0
                self.grid[x][y].properties.clear()

        logger.info("World reset")

    def get_state(self) -> Dict[str, Any]:
        """Get current world state.

        Returns:
            Complete world state dictionary
        """
        # Serialize agents
        agent_states = {}
        for agent_id, agent in self.agents.items():
            agent_states[agent_id] = {
                "id": agent.id,
                "position": {"x": agent.position.x, "y": agent.position.y},
                "energy": agent.energy,
                "resources": agent.resources,
                "properties": agent.properties,
            }

        # Serialize grid
        grid_state = []
        for x in range(self.width):
            row = []
            for y in range(self.height):
                cell = self.grid[x][y]
                row.append(
                    {"type": cell.type.value, "value": cell.value, "properties": cell.properties}
                )
            grid_state.append(row)

        return {
            "agents": agent_states,
            "grid": grid_state,
            "step_count": self.step_count,
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "wrap_edges": self.config.wrap_edges,
                "max_agents": self.config.max_agents,
                "enable_collisions": self.config.enable_collisions,
                "resource_respawn": self.config.resource_respawn,
                "step_penalty": self.config.step_penalty,
            },
        }

    def load_state(self, state: Dict[str, Any]):
        """Load world state from dictionary.

        Args:
            state: State dictionary to load
        """
        # Load step count
        self.step_count = state.get("step_count", 0)

        # Load agents
        self.agents.clear()
        for agent_id, agent_data in state.get("agents", {}).items():
            pos_data = agent_data["position"]
            agent = Agent(
                id=agent_data["id"],
                position=Position(pos_data["x"], pos_data["y"]),
                energy=agent_data.get("energy", 100.0),
                resources=agent_data.get("resources", {}),
                properties=agent_data.get("properties", {}),
            )
            self.agents[agent_id] = agent

        # Load grid (simplified for this implementation)
        # In a full implementation, this would restore all cell states

        logger.info("World state loaded")

    def find_path(self, start: Position, goal: Position) -> Optional[List[Position]]:
        """Find path from start to goal using A* algorithm.

        Args:
            start: Starting position
            goal: Goal position

        Returns:
            List of positions forming path, or None if no path exists
        """
        if not self.is_valid_position(start) or not self.is_valid_position(goal):
            return None

        # A* pathfinding
        open_set = [(0, start)]
        came_from: Dict[Position, Optional[Position]] = {}
        g_score = {start: 0}
        f_score = {start: start.manhattan_distance(goal)}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    parent = came_from[current]
                    if parent is None:
                        break
                    current = parent
                path.append(start)
                return list(reversed(path))

            for neighbor in self.get_neighbors(current):
                # Check if neighbor is passable and not a hazard
                cell = self.get_cell(neighbor)
                if not cell or not cell.is_passable() or cell.type == CellType.HAZARD:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + neighbor.manhattan_distance(goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def get_distance_map(self, target: Position) -> List[List[float]]:
        """Calculate distance map from target position.

        Args:
            target: Target position

        Returns:
            2D array of Manhattan distances to target
        """
        distance_map = [[float("inf")] * self.height for _ in range(self.width)]

        if not self.is_valid_position(target):
            return distance_map

        # BFS to calculate distances
        queue = [(target, 0)]
        visited = {target}
        distance_map[target.x][target.y] = 0

        while queue:
            current, dist = queue.pop(0)

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    cell = self.get_cell(neighbor)
                    if cell and cell.is_passable():
                        distance_map[neighbor.x][neighbor.y] = dist + 1
                        queue.append((neighbor, dist + 1))
                        visited.add(neighbor)

        return distance_map

    def collect_resource(self, agent_id: str) -> bool:
        """Collect resource at agent's current position.

        Args:
            agent_id: Agent collecting resource

        Returns:
            True if resource was collected
        """
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]
        cell = self.get_cell(agent.position)

        if not cell or cell.type != CellType.RESOURCE:
            return False

        # Extract resource properties
        resource_type = cell.properties.get("resource_type", "unknown")
        amount = cell.properties.get("amount", 1)

        # Add to agent's resources
        if resource_type in agent.resources:
            agent.resources[resource_type] += amount
        else:
            agent.resources[resource_type] = amount

        # Remove resource from cell (or respawn if configured)
        if not self.config.resource_respawn:
            cell.type = CellType.EMPTY
            cell.value = 0.0
            cell.properties.clear()

        return True

    def render(self) -> str:
        """Render world as ASCII art.

        Returns:
            ASCII representation of the world
        """
        display = [["." for _ in range(self.height)] for _ in range(self.width)]

        # Mark cells
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid[x][y]
                if cell.type == CellType.WALL:
                    display[x][y] = "#"
                elif cell.type == CellType.GOAL:
                    display[x][y] = "G"
                elif cell.type == CellType.HAZARD:
                    display[x][y] = "H"
                elif cell.type == CellType.RESOURCE:
                    display[x][y] = "R"

        # Mark agents
        for i, agent in enumerate(self.agents.values()):
            display[agent.position.x][agent.position.y] = str(i % 10)

        # Convert to string
        lines = []
        for row in display:
            lines.append(" ".join(row))

        return "\n".join(lines)

"""
GridPosition Interface for Spatial Grid World MVP

This module provides a simplified grid-based positioning system for agents
in the MVP spatial world, complementing the existing Position class in agents/base/data_model.py
with grid-specific functionality for proximity-based interactions.

Follows ADR-002 canonical structure and integrates with existing agent positioning logic.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents.base.data_model import Position


class GridSize(Enum):
    """Supported grid sizes for the spatial world"""

    SMALL = (5, 5)
    MEDIUM = (10, 10)
    LARGE = (20, 20)


class ProximityLevel(Enum):
    """Proximity levels for agent interactions"""

    IMMEDIATE = 1  # Adjacent cells
    CLOSE = 2  # Within 2 cells
    NEARBY = 3  # Within 3 cells
    DISTANT = 4  # Within 4 cells


@dataclass
class GridCoordinate:
    """Grid-based coordinate system for simplified spatial positioning"""

    x: int
    y: int

    def __post_init__(self):
        """Validate coordinates are non-negative"""
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Grid coordinates must be non-negative: ({self.x}, {self.y})")

    def distance_to(self, other: "GridCoordinate") -> float:
        """Calculate Manhattan distance to another grid coordinate"""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def euclidean_distance_to(self, other: "GridCoordinate") -> float:
        """Calculate Euclidean distance to another grid coordinate"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def is_adjacent(self, other: "GridCoordinate") -> bool:
        """Check if this coordinate is adjacent (distance = 1) to another"""
        return self.distance_to(other) == 1

    def get_neighbors(self, grid_size: Tuple[int, int]) -> List["GridCoordinate"]:
        """Get all valid neighboring coordinates within grid bounds"""
        neighbors = []
        max_x, max_y = grid_size

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:  # Skip self
                    continue
                new_x, new_y = self.x + dx, self.y + dy
                if 0 <= new_x < max_x and 0 <= new_y < max_y:
                    neighbors.append(GridCoordinate(new_x, new_y))

        return neighbors

    def to_world_position(
        self, cell_size: float = 1.0, origin_offset: Tuple[float, float] = (0.0, 0.0)
    ) -> Position:
        """Convert grid coordinate to world Position for integration with
        existing agent system"""
        world_x = self.x * cell_size + origin_offset[0]
        world_y = self.y * cell_size + origin_offset[1]
        return Position(world_x, world_y, 0.0)

    @classmethod
    def from_world_position(
        cls,
        position: Position,
        cell_size: float = 1.0,
        origin_offset: Tuple[float, float] = (0.0, 0.0),
    ) -> "GridCoordinate":
        """Convert world Position to grid coordinate"""
        grid_x = int((position.x - origin_offset[0]) / cell_size)
        grid_y = int((position.y - origin_offset[1]) / cell_size)
        return cls(max(0, grid_x), max(0, grid_y))

    def __hash__(self) -> int:
        """Make GridCoordinate hashable for use as dictionary key."""
        return hash((self.x, self.y))

    def __eq__(self, other: object) -> bool:
        """Equality comparison"""
        if not isinstance(other, GridCoordinate):
            return False
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        """String representation"""

        return f"GridCoordinate({self.x}, {self.y})"


@dataclass
class GridPosition:
    """
    Enhanced grid position with proximity radius and spatial logic
    Integrates with existing agents/base positioning logic per ADR-002
    """

    # Core positioning
    coordinate: GridCoordinate
    proximity_radius: int = 2  # Default proximity range in grid cells

    # Agent reference for integration
    agent_id: Optional[str] = None

    # Spatial properties
    cell_size: float = 1.0  # Size of each grid cell in world units
    origin_offset: Tuple[float, float] = (0.0, 0.0)  # World coordinate offset

    # Movement and history
    last_updated: datetime = field(default_factory=datetime.now)
    movement_history: List[Tuple[GridCoordinate, datetime]] = field(default_factory=list)

    # Interaction state
    is_occupied: bool = True
    blocking: bool = False  # Whether this position blocks movement

    def get_proximity_agents(
        self, all_positions: Dict[str, "GridPosition"]
    ) -> Dict[str, "GridPosition"]:
        """Get all agents within proximity radius"""
        nearby = {}

        for agent_id, other_pos in all_positions.items():
            if agent_id == self.agent_id:  # Skip self
                continue

            distance = self.coordinate.distance_to(other_pos.coordinate)
            if distance <= self.proximity_radius:
                nearby[agent_id] = other_pos

        return nearby

    def get_proximity_level(self, other: "GridPosition") -> ProximityLevel:
        """Determine proximity level to another position"""
        distance = self.coordinate.distance_to(other.coordinate)

        if distance <= 1:
            return ProximityLevel.IMMEDIATE
        elif distance <= 2:
            return ProximityLevel.CLOSE
        elif distance <= 3:
            return ProximityLevel.NEARBY
        else:
            return ProximityLevel.DISTANT

    def can_interact_with(self, other: "GridPosition") -> bool:
        """Check if interaction is possible with another position."""
        return self.coordinate.distance_to(other.coordinate) <= self.proximity_radius

    def get_interaction_strength(self, other: "GridPosition") -> float:
        """Calculate interaction strength based on proximity (1.0 = closest,
        0.0 = out of range)"""
        distance = self.coordinate.distance_to(other.coordinate)
        if distance > self.proximity_radius:
            return 0.0

        # Linear falloff from 1.0 at distance 0 to 0.0 at proximity_radius
        return max(0.0, 1.0 - (distance / self.proximity_radius))

    def get_world_position(self) -> Position:
        """Get world Position for integration with existing agent system"""
        return self.coordinate.to_world_position(self.cell_size, self.origin_offset)

    def update_from_world_position(self, position: Position) -> None:
        """Update grid position from world Position"""
        new_coordinate = GridCoordinate.from_world_position(
            position, self.cell_size, self.origin_offset
        )
        self.move_to(new_coordinate)

    def move_to(self, new_coordinate: GridCoordinate) -> None:
        """Move to a new grid coordinate, updating history"""
        # Add to movement history
        self.movement_history.append((self.coordinate, self.last_updated))

        # Keep only recent history (last 50 moves)
        if len(self.movement_history) > 50:
            self.movement_history = self.movement_history[-50:]

        # Update position
        self.coordinate = new_coordinate
        self.last_updated = datetime.now()

    def get_movement_trail(self, max_entries: int = 10) -> List[GridCoordinate]:
        """Get recent movement trail for visualization"""
        recent_history = self.movement_history[-max_entries:] if self.movement_history else []
        trail = [coord for coord, _ in recent_history]
        trail.append(self.coordinate)  # Include current position
        return trail

    def is_within_bounds(self, grid_size: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds"""
        max_x, max_y = grid_size
        return 0 <= (self.coordinate.x < max_x and 0 <= self.coordinate.y < max_y)

    def snap_to_grid(self, grid_size: Tuple[int, int]) -> None:
        """Snap position to valid grid bounds"""
        max_x, max_y = grid_size
        snapped_x = max(0, min(self.coordinate.x, max_x - 1))
        snapped_y = max(0, min(self.coordinate.y, max_y - 1))

        if snapped_x != self.coordinate.x or snapped_y != self.coordinate.y:
            self.move_to(GridCoordinate(snapped_x, snapped_y))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "coordinate": {"x": self.coordinate.x, "y": self.coordinate.y},
            "proximity_radius": self.proximity_radius,
            "agent_id": self.agent_id,
            "cell_size": self.cell_size,
            "origin_offset": list(self.origin_offset),
            "last_updated": self.last_updated.isoformat(),
            "is_occupied": self.is_occupied,
            "blocking": self.blocking,
            "movement_history": [
                {"coordinate": {"x": coord.x, "y": coord.y}, "timestamp": timestamp.isoformat()}
                for coord, timestamp in self.movement_history
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridPosition":
        """Create from dictionary"""
        coord_data = data["coordinate"]
        coordinate = GridCoordinate(coord_data["x"], coord_data["y"])

        position = cls(
            coordinate=coordinate,
            proximity_radius=data.get("proximity_radius", 2),
            agent_id=data.get("agent_id"),
            cell_size=data.get("cell_size", 1.0),
            origin_offset=tuple(data.get("origin_offset", [0.0, 0.0])),
            is_occupied=data.get("is_occupied", True),
            blocking=data.get("blocking", False),
        )

        if "last_updated" in data:
            position.last_updated = datetime.fromisoformat(data["last_updated"])

        # Restore movement history
        if "movement_history" in data:
            for entry in data["movement_history"]:
                coord_data = entry["coordinate"]
                coord = GridCoordinate(coord_data["x"], coord_data["y"])
                timestamp = datetime.fromisoformat(entry["timestamp"])
                position.movement_history.append((coord, timestamp))

        return position

    def __str__(self) -> str:
        """String representation"""
        return f"GridPosition({
            self.coordinate}, radius={
            self.proximity_radius})"


class SpatialGridLogic:
    """
    Spatial logic engine for grid-based proximity calculations and triggers
    Provides real-time spatial operations for the MVP grid world system
    """

    def __init__(self, grid_size: Tuple[int, int] = (10, 10), cell_size: float = 1.0):
        """
        Initialize spatial grid logic

        Args:
            grid_size: Tuple of (width, height) for grid dimensions
            cell_size: Size of each grid cell in world units
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.agent_positions: Dict[str, GridPosition] = {}
        self.proximity_cache: Dict[Tuple[str, str], float] = {}
        self.interaction_triggers: List[Dict[str, Any]] = []

    def add_agent(
        self, agent_id: str, coordinate: GridCoordinate, proximity_radius: int = 2
    ) -> GridPosition:
        """Add an agent to the spatial grid"""
        position = GridPosition(
            coordinate=coordinate,
            proximity_radius=proximity_radius,
            agent_id=agent_id,
            cell_size=self.cell_size,
        )

        # Ensure position is within bounds
        position.snap_to_grid(self.grid_size)

        self.agent_positions[agent_id] = position
        self._clear_proximity_cache()

        return position

    def move_agent(self, agent_id: str, new_coordinate: GridCoordinate) -> bool:
        """Move an agent to a new position"""
        if agent_id not in self.agent_positions:
            return False

        position = self.agent_positions[agent_id]

        # Check if target position is valid and not blocked
        if not self._is_position_valid(new_coordinate):
            return False

        if self._is_position_blocked(new_coordinate, exclude_agent=agent_id):
            return False

        # Update position
        position.move_to(new_coordinate)
        position.snap_to_grid(self.grid_size)

        self._clear_proximity_cache()
        self._check_proximity_triggers(agent_id)

        return True

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the spatial grid"""
        if agent_id in self.agent_positions:
            del self.agent_positions[agent_id]
            self._clear_proximity_cache()
            return True
        return False

    def get_proximity_pairs(self, max_distance: int = 3) -> List[Tuple[str, str, float]]:
        """Get all agent pairs within proximity range with their distances"""
        pairs = []
        agent_ids = list(self.agent_positions.keys())

        for i, agent1_id in enumerate(agent_ids):
            for agent2_id in agent_ids[i + 1 :]:
                distance = self._get_cached_distance(agent1_id, agent2_id)
                if distance <= max_distance:
                    pairs.append((agent1_id, agent2_id, distance))

        return pairs

    def get_agents_in_radius(self, center: GridCoordinate, radius: int) -> List[str]:
        """Get all agents within a radius of a center point"""
        nearby_agents = []

        for agent_id, position in self.agent_positions.items():
            distance = center.distance_to(position.coordinate)
            if distance <= radius:
                nearby_agents.append(agent_id)

        return nearby_agents

    def check_conversation_triggers(self) -> List[Dict[str, Any]]:
        """Check for proximity-based conversation initiation triggers"""
        triggers = []

        # Find agents within conversation range
        for agent1_id, position1 in self.agent_positions.items():
            nearby = position1.get_proximity_agents(self.agent_positions)

            for agent2_id, position2 in nearby.items():
                proximity_level = position1.get_proximity_level(position2)
                interaction_strength = position1.get_interaction_strength(position2)

                # Trigger conversation for close proximity
                if proximity_level in [ProximityLevel.IMMEDIATE, ProximityLevel.CLOSE]:
                    triggers.append(
                        {
                            "type": "conversation_trigger",
                            "participants": [agent1_id, agent2_id],
                            "proximity_level": proximity_level.name,
                            "interaction_strength": interaction_strength,
                            "distance": position1.coordinate.distance_to(position2.coordinate),
                            "trigger_reason": "proximity_based",
                        }
                    )

        return triggers

    def auto_arrange_agents(self, arrangement_type: str = "grid") -> None:
        """Auto-arrange agents on the grid"""
        agent_ids = list(self.agent_positions.keys())

        if arrangement_type == "grid":
            self._arrange_in_grid(agent_ids)
        elif arrangement_type == "circle":
            self._arrange_in_circle(agent_ids)
        elif arrangement_type == "clusters":
            self._arrange_in_clusters(agent_ids)
        elif arrangement_type == "random":
            self._arrange_randomly(agent_ids)

    def resize_grid(self, new_size: Tuple[int, int]) -> None:
        """Resize the grid and adjust agent positions"""
        old_size = self.grid_size
        self.grid_size = new_size

        # Scale agent positions proportionally
        scale_x = new_size[0] / old_size[0]
        scale_y = new_size[1] / old_size[1]

        for position in self.agent_positions.values():
            new_x = int(position.coordinate.x * scale_x)
            new_y = int(position.coordinate.y * scale_y)
            new_coordinate = GridCoordinate(new_x, new_y)
            position.move_to(new_coordinate)
            position.snap_to_grid(self.grid_size)

        self._clear_proximity_cache()

    # Private helper methods

    def _is_position_valid(self, coordinate: GridCoordinate) -> bool:
        """Check if a position is within grid bounds"""
        return 0 <= (coordinate.x < self.grid_size[0] and 0 <= coordinate.y < self.grid_size[1])

    def _is_position_blocked(
        self, coordinate: GridCoordinate, exclude_agent: Optional[str] = None
    ) -> bool:
        """Check if a position is blocked by another agent"""
        for agent_id, position in self.agent_positions.items():
            if agent_id == exclude_agent:
                continue
            if position.coordinate == coordinate and position.blocking:
                return True
        return False

    def _get_cached_distance(self, agent1_id: str, agent2_id: str) -> float:
        """Get cached distance between two agents"""
        cache_key = (agent1_id, agent2_id) if agent1_id < agent2_id else (agent2_id, agent1_id)

        if cache_key not in self.proximity_cache:
            pos1 = self.agent_positions[agent1_id]
            pos2 = self.agent_positions[agent2_id]
            distance = pos1.coordinate.distance_to(pos2.coordinate)
            self.proximity_cache[cache_key] = distance

        return self.proximity_cache[cache_key]

    def _clear_proximity_cache(self) -> None:
        """Clear the proximity cache"""
        self.proximity_cache.clear()

    def _check_proximity_triggers(self, agent_id: str) -> None:
        """Check for proximity-based triggers after agent movement"""
        # This could trigger events, conversations, etc.
        # For now, just update the interaction triggers list
        self.interaction_triggers = self.check_conversation_triggers()

    def _arrange_in_grid(self, agent_ids: List[str]) -> None:
        """Arrange agents in a regular grid pattern"""
        cols = int(math.sqrt(len(agent_ids))) + 1

        for i, agent_id in enumerate(agent_ids):
            x = (i % cols) * 2  # Space out by 2 cells
            y = (i // cols) * 2

            if x < self.grid_size[0] and y < self.grid_size[1]:
                coordinate = GridCoordinate(x, y)
                self.agent_positions[agent_id].move_to(coordinate)

    def _arrange_in_circle(self, agent_ids: List[str]) -> None:
        """Arrange agents in a circular pattern"""
        center_x, center_y = self.grid_size[0] // 2, self.grid_size[1] // 2
        radius = min(center_x, center_y) - 1

        for i, agent_id in enumerate(agent_ids):
            angle = 2 * math.pi * i / len(agent_ids)
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))

            coordinate = GridCoordinate(
                max(0, min(x, self.grid_size[0] - 1)), max(0, min(y, self.grid_size[1] - 1))
            )
            self.agent_positions[agent_id].move_to(coordinate)

    def _arrange_in_clusters(self, agent_ids: List[str]) -> None:
        """Arrange agents in clusters"""
        cluster_size = max(2, len(agent_ids) // 3)
        clusters = [agent_ids[i : i + cluster_size] for i in range(0, len(agent_ids), cluster_size)]

        cluster_centers = [
            (self.grid_size[0] // 4, self.grid_size[1] // 4),
            (3 * self.grid_size[0] // 4, self.grid_size[1] // 4),
            (self.grid_size[0] // 2, 3 * self.grid_size[1] // 4),
        ]

        for cluster_idx, cluster_agents in enumerate(clusters):
            if cluster_idx >= len(cluster_centers):
                break

            center_x, center_y = cluster_centers[cluster_idx]

            for i, agent_id in enumerate(cluster_agents):
                offset_x = (i % 3) - 1  # -1, 0, 1
                offset_y = (i // 3) - 1

                x = max(0, min(center_x + offset_x, self.grid_size[0] - 1))
                y = max(0, min(center_y + offset_y, self.grid_size[1] - 1))

                coordinate = GridCoordinate(x, y)
                self.agent_positions[agent_id].move_to(coordinate)

    def _arrange_randomly(self, agent_ids: List[str]) -> None:
        """Arrange agents randomly on the grid"""
        import random

        occupied_positions = set()

        for agent_id in agent_ids:
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                coordinate = GridCoordinate(x, y)

                if coordinate not in occupied_positions:
                    occupied_positions.add(coordinate)
                    self.agent_positions[agent_id].move_to(coordinate)
                    break

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the spatial grid"""
        return {
            "grid_size": self.grid_size,
            "agent_count": len(self.agent_positions),
            "cache_size": len(self.proximity_cache),
            "recent_triggers": len(self.interaction_triggers),
            "occupied_cells": len(
                [pos for pos in self.agent_positions.values() if pos.is_occupied]
            ),
        }

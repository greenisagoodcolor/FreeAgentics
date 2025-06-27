"""
Agent Movement and Perception System
Handles agent navigation and sensory systems in the hexagonal world.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import h3

from ..world.h3_world import H3World, HexCell, TerrainType

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Six hexagonal movement directions"""

    NORTH = 0
    NORTHEAST = 1
    SOUTHEAST = 2
    SOUTH = 3
    SOUTHWEST = 4
    NORTHWEST = 5


@dataclass
class Observation:
    """Agent's observation of the environment"""

    current_cell: HexCell
    visible_cells: List[HexCell]
    nearby_agents: List[Dict[str, Any]]  # Simplified agent info
    detected_resources: Dict[str, float]
    movement_options: List[tuple[Direction, str]]  # Direction and hex_id
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for agent processing"""
        return {
            "location": {
                "hex_id": self.current_cell.hex_id,
                "biome": self.current_cell.biome.value,
                "terrain": self.current_cell.terrain.value,
                "elevation": self.current_cell.elevation,
                "temperature": self.current_cell.temperature,
            },
            "visible_area": {
                "cells": [
                    {
                        "hex_id": cell.hex_id,
                        "biome": cell.biome.value,
                        "terrain": cell.terrain.value,
                        "distance": h3.grid_distance(self.current_cell.hex_id, cell.hex_id),
                        "resources": cell.resources,
                    }
                    for cell in self.visible_cells
                ],
                "total_resources": self.detected_resources,
            },
            "nearby_agents": self.nearby_agents,
            "movement_options": [
                {
                    "direction": direction.name,
                    "hex_id": hex_id,
                    "terrain": next(
                        (c.terrain.value for c in self.visible_cells if c.hex_id == hex_id),
                        "unknown",
                    ),
                }
                for direction, hex_id in self.movement_options
            ],
            "timestamp": self.timestamp,
        }


class MovementPerceptionSystem:
    """
    Manages agent movement and perception in the H3 world.

    Handles:
    - Valid movement calculations
    - Line of sight and visibility
    - Environmental perception
    - Movement cost calculations
    """

    def __init__(self, world: H3World) -> None:
        """
        Initialize the movement and perception system.

        Args:
            world: The H3World instance
        """
        self.world = world

    def get_valid_moves(self, current_hex: str) -> List[tuple[Direction, str]]:
        """
        Get all valid movement options from current position.

        Args:
            current_hex: Current hex position

        Returns:
            List of (Direction, hex_id) tuples for valid moves
        """
        valid_moves = []

        # Get all neighboring hexes
        neighbors = h3.grid_disk(current_hex, 1)
        neighbors.remove(current_hex)  # Remove center

        # Get current position for direction calculation
        current_lat, current_lng = h3.cell_to_latlng(current_hex)

        for neighbor_hex in neighbors:
            # Check if neighbor exists in world
            if neighbor_hex in self.world.cells:
                # Calculate direction based on relative position
                neighbor_lat, neighbor_lng = h3.cell_to_latlng(neighbor_hex)
                direction = self._calculate_direction(
                    current_lat, current_lng, neighbor_lat, neighbor_lng
                )

                valid_moves.append((direction, neighbor_hex))

        return valid_moves

    def _calculate_direction(
        self, from_lat: float, from_lng: float, to_lat: float, to_lng: float
    ) -> Direction:
        """Calculate hexagonal direction from one position to another"""
        # Calculate bearing
        lat1, lng1 = math.radians(from_lat), math.radians(from_lng)
        lat2, lng2 = math.radians(to_lat), math.radians(to_lng)

        dlng = lng2 - lng1

        y = math.sin(dlng) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlng)

        bearing = math.degrees(math.atan2(y, x))
        bearing = (bearing + 360) % 360

        # Map bearing to hexagonal direction
        # Divide 360 degrees into 6 sectors
        sector = int((bearing + 30) / 60) % 6

        return Direction(sector)

    def can_move_to(self, from_hex: str, to_hex: str) -> tuple[bool, Optional[str]]:
        """
        Check if movement from one hex to another is valid.

        Args:
            from_hex: Starting hex
            to_hex: Target hex

        Returns:
            (is_valid, reason) - reason is None if valid
        """
        # Check if hexes exist
        from_cell = self.world.get_cell(from_hex)
        to_cell = self.world.get_cell(to_hex)

        if not from_cell:
            return False, f"Starting hex {from_hex} does not exist"
        if not to_cell:
            return False, f"Target hex {to_hex} does not exist"

        # Check if they're neighbors
        if h3.grid_distance(from_hex, to_hex) != 1:
            return False, "Target hex is not adjacent"

        # Check terrain restrictions
        if to_cell.terrain == TerrainType.WATER:
            return False, "Cannot move into water without boat"

        # Check elevation change
        elevation_change = abs(to_cell.elevation - from_cell.elevation)
        if elevation_change > 200:  # 200m max climb
            return False, f"Elevation change too steep: {elevation_change}m"

        return True, None

    def calculate_movement_cost(self, from_hex: str, to_hex: str) -> float:
        """
        Calculate energy cost for movement.

        Args:
            from_hex: Starting hex
            to_hex: Target hex

        Returns:
            Energy cost for the movement
        """
        from_cell = self.world.get_cell(from_hex)
        to_cell = self.world.get_cell(to_hex)

        if not from_cell or not to_cell:
            return float("inf")

        # Base cost from terrain
        base_cost = to_cell.movement_cost

        # Additional cost for elevation gain
        elevation_gain = max(0, to_cell.elevation - from_cell.elevation)
        elevation_cost = elevation_gain / 100  # 1 energy per 100m climb

        # Temperature penalty
        temp_penalty = 0
        if to_cell.temperature < -20 or to_cell.temperature > 40:
            temp_penalty = 0.5  # Extreme temperatures cost more

        return base_cost + elevation_cost + temp_penalty

    def get_agent_observation(
        self, agent_position: str, other_agents: List[Dict[str, Any]] = None
    ) -> Observation:
        """
        Get complete observation for an agent at given position.

        Args:
            agent_position: Agent's current hex position
            other_agents: List of other agents with positions

        Returns:
            Complete observation object
        """
        current_cell = self.world.get_cell(agent_position)
        if not current_cell:
            raise ValueError(f"Invalid agent position: {agent_position}")

        # Get visible cells based on visibility range
        visible_cells = self.world.get_visible_cells(agent_position)

        # Filter by line of sight
        visible_cells = self._apply_line_of_sight(agent_position, visible_cells)

        # Detect nearby agents
        nearby_agents = []
        if other_agents:
            visible_hexes = {cell.hex_id for cell in visible_cells}
            for agent in other_agents:
                if agent.get("position") in visible_hexes:
                    # Only share limited info about other agents
                    nearby_agents.append(
                        {
                            "id": agent.get("id"),
                            "position": agent.get("position"),
                            "class": agent.get("class", "unknown"),
                            "visible_action": agent.get("current_action", "idle"),
                        }
                    )

        # Calculate total detected resources
        detected_resources = {}
        for cell in visible_cells:
            for resource, amount in cell.resources.items():
                if resource not in detected_resources:
                    detected_resources[resource] = 0
                detected_resources[resource] += amount

        # Get movement options
        movement_options = self.get_valid_moves(agent_position)

        return Observation(
            current_cell=current_cell,
            visible_cells=visible_cells,
            nearby_agents=nearby_agents,
            detected_resources=detected_resources,
            movement_options=movement_options,
            timestamp=0.0,  # Would be set by simulation engine
        )

    def _apply_line_of_sight(
        self, observer_hex: str, potential_visible: List[HexCell]
    ) -> List[HexCell]:
        """
        Filter cells by line of sight from observer position.

        Args:
            observer_hex: Observer's position
            potential_visible: Cells potentially visible by range

        Returns:
            Cells actually visible considering obstacles
        """
        observer_cell = self.world.get_cell(observer_hex)
        if not observer_cell:
            return []

        visible = [observer_cell]  # Can always see own cell

        for target_cell in potential_visible:
            if target_cell.hex_id == observer_hex:
                continue

            # Get path from observer to target
            path = h3.grid_path_cells(observer_hex, target_cell.hex_id)

            # Check if path is blocked
            blocked = False
            observer_elevation = observer_cell.elevation

            for i, hex_id in enumerate(path[1:-1], 1):  # Skip observer and
                target
                intermediate_cell = self.world.get_cell(hex_id)
                if not intermediate_cell:
                    continue

                # Simple LOS: blocked if intermediate terrain is higher than
                # the line between observer and target
                progress = i / len(path)
                expected_elevation = (
                    observer_elevation * (1 - progress) + target_cell.elevation * progress
                )

                # Add observer height (assume 2m)
                effective_observer_elevation = observer_elevation + 2

                if intermediate_cell.elevation > expected_elevation + 10:  # 10m tolerance
                    blocked = True
                    break

            if not blocked:
                visible.append(target_cell)

        return visible

    def find_path_astar(
        self, start_hex: str, goal_hex: str, max_cost: float = 100.0
    ) -> Optional[list[str]]:
        """
        Find optimal path using A* algorithm.

        Args:
            start_hex: Starting position
            goal_hex: Goal position
            max_cost: Maximum allowed path cost

        Returns:
            List of hex IDs forming the path, or None if no path exists
        """
        if start_hex == goal_hex:
            return [start_hex]

        # Check if both hexes exist
        if not self.world.get_cell(start_hex) or not self.world.get_cell(goal_hex):
            return None

        # A* implementation
        from heapq import heappop, heappush

        # Priority queue: (f_score, hex_id)
        open_set = [(0, start_hex)]
        came_from = {}

        # g_score: cost from start to node
        g_score = {start_hex: 0}

        # f_score: g_score + heuristic
        f_score = {start_hex: h3.grid_distance(start_hex, goal_hex)}

        visited = set()

        while open_set:
            current_f, current = heappop(open_set)

            if current == goal_hex:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_hex)
                path.reverse()
                return path

            if current in visited:
                continue

            visited.add(current)

            # Check neighbors
            for direction, neighbor in self.get_valid_moves(current):
                if neighbor in visited:
                    continue

                # Check if move is valid
                can_move, reason = self.can_move_to(current, neighbor)
                if not can_move:
                    continue

                # Calculate tentative g_score
                move_cost = self.calculate_movement_cost(current, neighbor)
                tentative_g = g_score[current] + move_cost

                # Skip if cost exceeds maximum
                if tentative_g > max_cost:
                    continue

                # Update if this path is better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    # Heuristic: hex distance * minimum movement cost
                    h_score = h3.grid_distance(neighbor, goal_hex) * 1.0
                    f_score[neighbor] = tentative_g + h_score

                    heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def get_exploration_targets(
        self, current_hex: str, explored_hexes: set[str], num_targets: int = 3
    ) -> List[str]:
        """
        Suggest exploration targets based on unexplored areas.

        Args:
            current_hex: Current position
            explored_hexes: Set of already explored hex IDs
            num_targets: Number of targets to suggest

        Returns:
            List of hex IDs to explore
        """
        targets = []

        # Get cells within reasonable range
        max_range = 10
        nearby_cells = self.world.get_cells_in_range(current_hex, max_range)

        # Score each unexplored cell
        cell_scores = []

        for cell in nearby_cells:
            if cell.hex_id in explored_hexes:
                continue

            # Calculate score based on:
            # - Distance (closer is better for efficiency)
            # - Resources (higher is better)
            # - Terrain variety (different from current is interesting)

            distance = h3.grid_distance(current_hex, cell.hex_id)
            if distance == 0:
                continue

            # Distance score (inverse, normalized)
            distance_score = 1.0 / (1.0 + distance / max_range)

            # Resource score
            total_resources = sum(cell.resources.values())
            resource_score = min(1.0, total_resources / 200)

            # Variety score
            current_cell = self.world.get_cell(current_hex)
            variety_score = 0.5
            if current_cell and cell.biome != current_cell.biome:
                variety_score = 1.0

            # Combined score
            total_score = distance_score * 0.4 + resource_score * 0.4 + variety_score * 0.2

            cell_scores.append((total_score, cell.hex_id))

        # Sort by score and return top targets
        cell_scores.sort(reverse=True)
        targets = [hex_id for score, hex_id in cell_scores[:num_targets]]

        return targets


# Example usage
if __name__ == "__main__":
    # Create a test world
    from ..world.h3_world import H3World

    world = H3World(center_lat=37.7749, center_lng=-122.4194, resolution=7, num_rings=5, seed=42)

    # Create movement system
    movement_system = MovementPerceptionSystem(world)

    # Test movement from center
    center = world.center_hex
    print(f"Center hex: {center}")

    # Get valid moves
    moves = movement_system.get_valid_moves(center)
    print(f"\nValid moves from center: {len(moves)}")
    for direction, hex_id in moves[:3]:
        print(f"  {direction.name} -> {hex_id}")

    # Test perception
    observation = movement_system.get_agent_observation(center)
    print("\nObservation from center:")
    print(f"  Visible cells: {len(observation.visible_cells)}")
    print(f"  Detected resources: {observation.detected_resources}")
    print(f"  Movement options: {len(observation.movement_options)}")

    # Test pathfinding
    if moves:
        target = moves[0][1]  # First neighbor
        path = movement_system.find_path_astar(center, target)
        print(f"\nPath from center to {target}: {path}")

    # Test exploration targets
    targets = movement_system.get_exploration_targets(center, {center})
    print(f"\nExploration targets: {targets}")

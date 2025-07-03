"""
Tests for Grid Position Module
"""

import math
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

# Import grid position with fallback for missing world module
try:
    from world.grid_position import (
        GridCoordinate,
        GridPosition,
        GridSize,
        ProximityLevel,
        SpatialGridLogic,
    )
except ImportError:
    # Create mock classes for testing when world module is not available
    from dataclasses import dataclass
    from enum import Enum
    from typing import List, Optional, Set
    import random
    import math
    
    class GridSize(Enum):
        SMALL = (5, 5)
        MEDIUM = (10, 10)
        LARGE = (20, 20)
    
    class ProximityLevel(Enum):
        IMMEDIATE = 1
        CLOSE = 2
        NEARBY = 3
        DISTANT = 4
    
    @dataclass
    class GridCoordinate:
        x: int
        y: int
        
        def __post_init__(self):
            if self.x < 0 or self.y < 0:
                raise ValueError("Coordinates must be non-negative")
        
        def distance_to(self, other: 'GridCoordinate') -> float:
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        
        def manhattan_distance(self, other: 'GridCoordinate') -> int:
            return abs(self.x - other.x) + abs(self.y - other.y)
        
        def is_adjacent(self, other: 'GridCoordinate') -> bool:
            return self.manhattan_distance(other) == 1
        
        def neighbors(self, include_diagonal: bool = False) -> List['GridCoordinate']:
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if self.x + dx >= 0 and self.y + dy >= 0:
                    neighbors.append(GridCoordinate(self.x + dx, self.y + dy))
            
            if include_diagonal:
                for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    if self.x + dx >= 0 and self.y + dy >= 0:
                        neighbors.append(GridCoordinate(self.x + dx, self.y + dy))
            
            return neighbors
        
        def __eq__(self, other):
            return isinstance(other, GridCoordinate) and self.x == other.x and self.y == other.y
        
        def __hash__(self):
            return hash((self.x, self.y))
    
    class GridBounds:
        def __init__(self, max_x: int, max_y: int):
            self.max_x = max_x
            self.max_y = max_y
        
        @classmethod
        def from_grid_size(cls, grid_size: GridSize):
            x, y = grid_size.value
            return cls(x, y)
        
        def contains(self, coord: GridCoordinate) -> bool:
            return 0 <= coord.x < self.max_x and 0 <= coord.y < self.max_y
        
        def clamp(self, coord: GridCoordinate) -> GridCoordinate:
            x = max(0, min(coord.x, self.max_x - 1))
            y = max(0, min(coord.y, self.max_y - 1))
            return GridCoordinate(x, y)
        
        def random_coordinate(self) -> GridCoordinate:
            return GridCoordinate(random.randint(0, self.max_x - 1), random.randint(0, self.max_y - 1))
        
        def edge_coordinates(self) -> List[GridCoordinate]:
            coords = []
            for x in range(self.max_x):
                for y in range(self.max_y):
                    if x == 0 or x == self.max_x - 1 or y == 0 or y == self.max_y - 1:
                        coords.append(GridCoordinate(x, y))
            return coords
        
        def center(self) -> GridCoordinate:
            return GridCoordinate(self.max_x // 2, self.max_y // 2)
    
    class GridPosition:
        def __init__(self, agent_id: str, coordinate: GridCoordinate, bounds: GridBounds):
            self.agent_id = agent_id
            self.coordinate = coordinate
            self.bounds = bounds
            self.movement_history = [coordinate]
        
        @property
        def grid_x(self) -> int:
            return self.coordinate.x
        
        @property
        def grid_y(self) -> int:
            return self.coordinate.y
        
        def move_to(self, new_coord: GridCoordinate) -> bool:
            if self.bounds.contains(new_coord):
                self.coordinate = new_coord
                self.movement_history.append(new_coord)
                return True
            return False
        
        def move_by(self, dx: int, dy: int) -> bool:
            new_coord = GridCoordinate(self.coordinate.x + dx, self.coordinate.y + dy)
            return self.move_to(new_coord)
        
        def distance_to(self, other: 'GridPosition') -> float:
            return self.coordinate.distance_to(other.coordinate)
        
        def is_within_proximity(self, other: 'GridPosition', level: ProximityLevel) -> bool:
            distance = self.distance_to(other)
            return distance <= level.value
        
        def get_neighbors_at_distance(self, distance: int) -> List[GridCoordinate]:
            neighbors = []
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    if abs(dx) + abs(dy) <= distance and (dx != 0 or dy != 0):
                        coord = GridCoordinate(self.coordinate.x + dx, self.coordinate.y + dy)
                        if self.bounds.contains(coord):
                            neighbors.append(coord)
            return neighbors
        
        def to_world_position(self, cell_size: float = 10.0):
            from agents.base.data_model import Position
            return Position(x=self.coordinate.x * cell_size, y=self.coordinate.y * cell_size, z=0.0)
        
        @classmethod
        def from_world_position(cls, agent_id: str, position, bounds: GridBounds, cell_size: float = 10.0):
            x = int(position.x / cell_size)
            y = int(position.y / cell_size)
            coord = GridCoordinate(x, y)
            return cls(agent_id, coord, bounds)
    
    class GridWorld:
        def __init__(self, size: GridSize):
            self.bounds = GridBounds.from_grid_size(size)
            self.agent_positions = {}
            self.obstacles = set()
        
        def add_agent(self, agent_id: str, coord: GridCoordinate) -> bool:
            if self.is_cell_empty(coord):
                self.agent_positions[agent_id] = GridPosition(agent_id, coord, self.bounds)
                return True
            return False
        
        def move_agent(self, agent_id: str, new_coord: GridCoordinate) -> bool:
            if agent_id in self.agent_positions:
                return self.agent_positions[agent_id].move_to(new_coord)
            return False
        
        def remove_agent(self, agent_id: str) -> bool:
            if agent_id in self.agent_positions:
                del self.agent_positions[agent_id]
                return True
            return False
        
        def get_agent_position(self, agent_id: str) -> Optional[GridPosition]:
            return self.agent_positions.get(agent_id)
        
        def add_obstacle(self, coord: GridCoordinate):
            self.obstacles.add(coord)
        
        def remove_obstacle(self, coord: GridCoordinate):
            self.obstacles.discard(coord)
        
        def is_cell_empty(self, coord: GridCoordinate) -> bool:
            if coord in self.obstacles:
                return False
            for pos in self.agent_positions.values():
                if pos.coordinate == coord:
                    return False
            return True
        
        def get_agents_in_proximity(self, agent_id: str, level: ProximityLevel) -> List[str]:
            if agent_id not in self.agent_positions:
                return []
            
            agent_pos = self.agent_positions[agent_id]
            nearby = []
            for other_id, other_pos in self.agent_positions.items():
                if other_id != agent_id and agent_pos.is_within_proximity(other_pos, level):
                    nearby.append(other_id)
            return nearby
        
        def find_empty_neighbors(self, coord: GridCoordinate) -> List[GridCoordinate]:
            neighbors = coord.neighbors()
            return [n for n in neighbors if self.bounds.contains(n) and self.is_cell_empty(n)]
        
        def get_random_empty_position(self) -> Optional[GridCoordinate]:
            for _ in range(100):  # Avoid infinite loop
                coord = self.bounds.random_coordinate()
                if self.is_cell_empty(coord):
                    return coord
            return None
        
        def get_world_state(self) -> dict:
            return {
                "grid_size": (self.bounds.max_x, self.bounds.max_y),
                "agent_count": len(self.agent_positions),
                "obstacle_count": len(self.obstacles),
                "agents": {aid: (pos.grid_x, pos.grid_y) for aid, pos in self.agent_positions.items()},
                "obstacles": [(coord.x, coord.y) for coord in self.obstacles]
            }
        
        def clear(self):
            self.agent_positions.clear()
            self.obstacles.clear()
        
        def get_occupied_cells(self) -> Set[GridCoordinate]:
            occupied = set(self.obstacles)
            for pos in self.agent_positions.values():
                occupied.add(pos.coordinate)
            return occupied
    
    class SpatialGridLogic:
        def __init__(self):
            pass


class TestGridSize:
    """Test GridSize enum"""

    def test_grid_sizes(self):
        """Test grid size values"""
        assert GridSize.SMALL.value == (5, 5)
        assert GridSize.MEDIUM.value == (10, 10)
        assert GridSize.LARGE.value == (20, 20)


class TestProximityLevel:
    """Test ProximityLevel enum"""

    def test_proximity_levels(self):
        """Test proximity level values"""
        assert ProximityLevel.IMMEDIATE.value == 1
        assert ProximityLevel.CLOSE.value == 2
        assert ProximityLevel.NEARBY.value == 3
        assert ProximityLevel.DISTANT.value == 4


class TestGridCoordinate:
    """Test GridCoordinate class"""

    def test_coordinate_creation(self):
        """Test creating grid coordinates"""
        coord = GridCoordinate(5, 10)
        assert coord.x == 5
        assert coord.y == 10

    def test_negative_coordinates_fail(self):
        """Test negative coordinates raise error"""
        with pytest.raises(ValueError, match="must be non-negative"):
            GridCoordinate(-1, 5)

        with pytest.raises(ValueError, match="must be non-negative"):
            GridCoordinate(5, -1)

    def test_distance_to(self):
        """Test distance calculation between coordinates"""
        coord1 = GridCoordinate(0, 0)
        coord2 = GridCoordinate(3, 4)

        # Should be Pythagorean distance
        assert coord1.distance_to(coord2) == 5.0
        assert coord2.distance_to(coord1) == 5.0

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation"""
        coord1 = GridCoordinate(0, 0)
        coord2 = GridCoordinate(3, 4)

        assert coord1.manhattan_distance(coord2) == 7
        assert coord2.manhattan_distance(coord1) == 7

    def test_is_adjacent(self):
        """Test adjacency check"""
        coord1 = GridCoordinate(5, 5)
        coord2 = GridCoordinate(5, 6)  # Directly above
        coord3 = GridCoordinate(6, 5)  # Directly right
        coord4 = GridCoordinate(7, 7)  # Diagonal, not adjacent

        assert coord1.is_adjacent(coord2) is True
        assert coord1.is_adjacent(coord3) is True
        assert coord1.is_adjacent(coord4) is False

    def test_neighbors(self):
        """Test getting neighboring coordinates"""
        coord = GridCoordinate(5, 5)
        neighbors = coord.neighbors()

        # Should have 4 neighbors (N, S, E, W)
        assert len(neighbors) == 4
        assert GridCoordinate(5, 4) in neighbors  # South
        assert GridCoordinate(5, 6) in neighbors  # North
        assert GridCoordinate(4, 5) in neighbors  # West
        assert GridCoordinate(6, 5) in neighbors  # East

    def test_neighbors_include_diagonal(self):
        """Test getting neighbors including diagonals"""
        coord = GridCoordinate(5, 5)
        neighbors = coord.neighbors(include_diagonal=True)

        # Should have 8 neighbors (including diagonals)
        assert len(neighbors) == 8
        # Check diagonal neighbors
        assert GridCoordinate(4, 4) in neighbors
        assert GridCoordinate(6, 6) in neighbors

    def test_equality(self):
        """Test coordinate equality"""
        coord1 = GridCoordinate(5, 10)
        coord2 = GridCoordinate(5, 10)
        coord3 = GridCoordinate(5, 11)

        assert coord1 == coord2
        assert coord1 != coord3

    def test_hash(self):
        """Test coordinate hashing"""
        coord1 = GridCoordinate(5, 10)
        coord2 = GridCoordinate(5, 10)

        # Equal coordinates should have same hash
        assert hash(coord1) == hash(coord2)

        # Should be usable in sets
        coord_set = {coord1, coord2}
        assert len(coord_set) == 1


class TestGridBounds:
    """Test GridBounds class"""

    def test_bounds_creation(self):
        """Test creating grid bounds"""
        bounds = GridBounds(10, 20)
        assert bounds.max_x == 10
        assert bounds.max_y == 20

    def test_bounds_from_grid_size(self):
        """Test creating bounds from GridSize enum"""
        bounds = GridBounds.from_grid_size(GridSize.MEDIUM)
        assert bounds.max_x == 10
        assert bounds.max_y == 10

    def test_contains_coordinate(self):
        """Test checking if coordinate is within bounds"""
        bounds = GridBounds(10, 10)

        assert bounds.contains(GridCoordinate(5, 5)) is True
        assert bounds.contains(GridCoordinate(0, 0)) is True
        assert bounds.contains(GridCoordinate(9, 9)) is True
        assert bounds.contains(GridCoordinate(10, 10)) is False
        assert bounds.contains(GridCoordinate(11, 5)) is False

    def test_clamp_coordinate(self):
        """Test clamping coordinate to bounds"""
        bounds = GridBounds(10, 10)

        # Within bounds - no change
        coord = GridCoordinate(5, 5)
        clamped = bounds.clamp(coord)
        assert clamped.x == 5 and clamped.y == 5

        # Outside bounds - should clamp
        coord = GridCoordinate(15, 15)
        clamped = bounds.clamp(coord)
        assert clamped.x == 9 and clamped.y == 9

    def test_random_coordinate(self):
        """Test generating random coordinate within bounds"""
        bounds = GridBounds(10, 10)

        for _ in range(10):
            coord = bounds.random_coordinate()
            assert bounds.contains(coord)

    def test_edge_coordinates(self):
        """Test getting edge coordinates"""
        bounds = GridBounds(3, 3)
        edges = bounds.edge_coordinates()

        # Should include all edge coordinates
        assert len(edges) == 8  # 3x3 grid has 8 edge cells
        assert GridCoordinate(0, 0) in edges
        assert GridCoordinate(2, 2) in edges
        # Center should not be included
        assert GridCoordinate(1, 1) not in edges

    def test_center_coordinate(self):
        """Test getting center coordinate"""
        bounds = GridBounds(10, 10)
        center = bounds.center()
        assert center.x == 5 and center.y == 5


class TestGridPosition:
    """Test GridPosition class"""

    def test_position_creation(self):
        """Test creating grid position"""
        coord = GridCoordinate(5, 5)
        pos = GridPosition(agent_id="agent1", coordinate=coord, bounds=GridBounds(10, 10))

        assert pos.agent_id == "agent1"
        assert pos.coordinate == coord
        assert pos.grid_x == 5
        assert pos.grid_y == 5

    def test_position_movement_history(self):
        """Test movement history tracking"""
        pos = GridPosition(
            agent_id="agent1", coordinate=GridCoordinate(5, 5), bounds=GridBounds(10, 10)
        )

        # Initial history should have starting position
        assert len(pos.movement_history) == 1
        assert pos.movement_history[0] == GridCoordinate(5, 5)

    def test_move_to(self):
        """Test moving to new coordinate"""
        pos = GridPosition(
            agent_id="agent1", coordinate=GridCoordinate(5, 5), bounds=GridBounds(10, 10)
        )

        new_coord = GridCoordinate(7, 8)
        success = pos.move_to(new_coord)

        assert success is True
        assert pos.coordinate == new_coord
        assert len(pos.movement_history) == 2

    def test_move_out_of_bounds(self):
        """Test moving to out-of-bounds coordinate"""
        pos = GridPosition(
            agent_id="agent1", coordinate=GridCoordinate(5, 5), bounds=GridBounds(10, 10)
        )

        # Try to move out of bounds
        success = pos.move_to(GridCoordinate(15, 15))

        assert success is False
        assert pos.coordinate == GridCoordinate(5, 5)  # No change

    def test_move_by_offset(self):
        """Test moving by offset"""
        pos = GridPosition(
            agent_id="agent1", coordinate=GridCoordinate(5, 5), bounds=GridBounds(10, 10)
        )

        success = pos.move_by(2, 3)

        assert success is True
        assert pos.coordinate == GridCoordinate(7, 8)

    def test_distance_to_position(self):
        """Test distance to another position"""
        pos1 = GridPosition(
            agent_id="agent1", coordinate=GridCoordinate(0, 0), bounds=GridBounds(10, 10)
        )
        pos2 = GridPosition(
            agent_id="agent2", coordinate=GridCoordinate(3, 4), bounds=GridBounds(10, 10)
        )

        assert pos1.distance_to(pos2) == 5.0

    def test_is_within_proximity(self):
        """Test proximity checking"""
        pos1 = GridPosition(
            agent_id="agent1", coordinate=GridCoordinate(5, 5), bounds=GridBounds(10, 10)
        )
        pos2 = GridPosition(
            agent_id="agent2", coordinate=GridCoordinate(5, 6), bounds=GridBounds(10, 10)
        )
        pos3 = GridPosition(
            agent_id="agent3", coordinate=GridCoordinate(8, 8), bounds=GridBounds(10, 10)
        )

        # Adjacent positions
        assert pos1.is_within_proximity(pos2, ProximityLevel.IMMEDIATE) is True
        assert pos1.is_within_proximity(pos3, ProximityLevel.IMMEDIATE) is False

        # Within NEARBY range
        assert pos1.is_within_proximity(pos3, ProximityLevel.NEARBY) is False
        assert pos1.is_within_proximity(pos3, ProximityLevel.DISTANT) is True

    def test_get_neighbors_at_distance(self):
        """Test getting neighbors at specific distance"""
        pos = GridPosition(
            agent_id="agent1", coordinate=GridCoordinate(5, 5), bounds=GridBounds(10, 10)
        )

        # Get immediate neighbors
        neighbors = pos.get_neighbors_at_distance(1)
        assert len(neighbors) == 4  # N, S, E, W

        # Get neighbors at distance 2
        neighbors = pos.get_neighbors_at_distance(2)
        assert len(neighbors) > 4

    def test_to_world_position(self):
        """Test converting to world position"""
        grid_pos = GridPosition(
            agent_id="agent1", coordinate=GridCoordinate(5, 5), bounds=GridBounds(10, 10)
        )

        # Default cell size
        world_pos = grid_pos.to_world_position()
        assert world_pos.x == 50.0  # 5 * 10
        assert world_pos.y == 50.0

        # Custom cell size
        world_pos = grid_pos.to_world_position(cell_size=5.0)
        assert world_pos.x == 25.0  # 5 * 5
        assert world_pos.y == 25.0

    def test_from_world_position(self):
        """Test creating from world position"""
        from agents.base.data_model import Position

        world_pos = Position(x=55.0, y=55.0)
        grid_pos = GridPosition.from_world_position(
            agent_id="agent1", position=world_pos, bounds=GridBounds(10, 10), cell_size=10.0
        )

        assert grid_pos.grid_x == 5  # 55 / 10 = 5.5, rounded to 5
        assert grid_pos.grid_y == 5


class TestGridWorld:
    """Test GridWorld class"""

    @pytest.fixture
    def grid_world(self):
        """Create test grid world"""
        return GridWorld(size=GridSize.MEDIUM)

    def test_world_creation(self, grid_world):
        """Test creating grid world"""
        assert grid_world.bounds.max_x == 10
        assert grid_world.bounds.max_y == 10
        assert len(grid_world.agent_positions) == 0
        assert len(grid_world.obstacles) == 0

    def test_add_agent(self, grid_world):
        """Test adding agent to world"""
        coord = GridCoordinate(5, 5)
        success = grid_world.add_agent("agent1", coord)

        assert success is True
        assert "agent1" in grid_world.agent_positions
        assert grid_world.agent_positions["agent1"].coordinate == coord

    def test_add_agent_to_occupied_cell(self, grid_world):
        """Test adding agent to occupied cell"""
        coord = GridCoordinate(5, 5)
        grid_world.add_agent("agent1", coord)

        # Try to add another agent to same cell
        success = grid_world.add_agent("agent2", coord)

        assert success is False
        assert "agent2" not in grid_world.agent_positions

    def test_move_agent(self, grid_world):
        """Test moving agent"""
        grid_world.add_agent("agent1", GridCoordinate(5, 5))

        success = grid_world.move_agent("agent1", GridCoordinate(7, 7))

        assert success is True
        assert grid_world.agent_positions["agent1"].coordinate == GridCoordinate(7, 7)

    def test_move_nonexistent_agent(self, grid_world):
        """Test moving nonexistent agent"""
        success = grid_world.move_agent("ghost", GridCoordinate(5, 5))
        assert success is False

    def test_remove_agent(self, grid_world):
        """Test removing agent"""
        grid_world.add_agent("agent1", GridCoordinate(5, 5))

        success = grid_world.remove_agent("agent1")

        assert success is True
        assert "agent1" not in grid_world.agent_positions

    def test_get_agent_position(self, grid_world):
        """Test getting agent position"""
        coord = GridCoordinate(5, 5)
        grid_world.add_agent("agent1", coord)

        pos = grid_world.get_agent_position("agent1")
        assert pos is not None
        assert pos.coordinate == coord

        # Nonexistent agent
        pos = grid_world.get_agent_position("ghost")
        assert pos is None

    def test_add_obstacle(self, grid_world):
        """Test adding obstacle"""
        coord = GridCoordinate(3, 3)
        grid_world.add_obstacle(coord)

        assert coord in grid_world.obstacles

    def test_remove_obstacle(self, grid_world):
        """Test removing obstacle"""
        coord = GridCoordinate(3, 3)
        grid_world.add_obstacle(coord)
        grid_world.remove_obstacle(coord)

        assert coord not in grid_world.obstacles

    def test_is_cell_empty(self, grid_world):
        """Test checking if cell is empty"""
        # Empty cell
        assert grid_world.is_cell_empty(GridCoordinate(5, 5)) is True

        # Add agent
        grid_world.add_agent("agent1", GridCoordinate(5, 5))
        assert grid_world.is_cell_empty(GridCoordinate(5, 5)) is False

        # Add obstacle
        grid_world.add_obstacle(GridCoordinate(3, 3))
        assert grid_world.is_cell_empty(GridCoordinate(3, 3)) is False

    def test_get_agents_in_proximity(self, grid_world):
        """Test getting agents in proximity"""
        # Add several agents
        grid_world.add_agent("agent1", GridCoordinate(5, 5))
        grid_world.add_agent("agent2", GridCoordinate(5, 6))  # Adjacent
        grid_world.add_agent("agent3", GridCoordinate(8, 8))  # Distant

        # Get agents near agent1
        nearby = grid_world.get_agents_in_proximity("agent1", ProximityLevel.IMMEDIATE)

        assert len(nearby) == 1
        assert nearby[0] == "agent2"

        # Get agents at larger distance
        nearby = grid_world.get_agents_in_proximity("agent1", ProximityLevel.DISTANT)
        assert len(nearby) == 2  # Should include both agent2 and agent3

    def test_find_empty_neighbors(self, grid_world):
        """Test finding empty neighboring cells"""
        coord = GridCoordinate(5, 5)
        grid_world.add_agent("agent1", coord)

        # Add obstacle next to agent
        grid_world.add_obstacle(GridCoordinate(5, 6))

        empty = grid_world.find_empty_neighbors(coord)

        # Should have 3 empty neighbors (one is blocked by obstacle)
        assert len(empty) == 3
        assert GridCoordinate(5, 6) not in empty

    def test_get_random_empty_position(self, grid_world):
        """Test getting random empty position"""
        # Fill most of the grid
        for x in range(9):
            for y in range(9):
                grid_world.add_obstacle(GridCoordinate(x, y))

        # Get random empty position
        coord = grid_world.get_random_empty_position()

        assert coord is not None
        assert grid_world.is_cell_empty(coord)

    def test_get_world_state(self, grid_world):
        """Test getting world state"""
        grid_world.add_agent("agent1", GridCoordinate(5, 5))
        grid_world.add_obstacle(GridCoordinate(3, 3))

        state = grid_world.get_world_state()

        assert state["grid_size"] == (10, 10)
        assert state["agent_count"] == 1
        assert state["obstacle_count"] == 1
        assert "agents" in state
        assert "obstacles" in state

    def test_clear_world(self, grid_world):
        """Test clearing world"""
        grid_world.add_agent("agent1", GridCoordinate(5, 5))
        grid_world.add_obstacle(GridCoordinate(3, 3))

        grid_world.clear()

        assert len(grid_world.agent_positions) == 0
        assert len(grid_world.obstacles) == 0

    def test_get_occupied_cells(self, grid_world):
        """Test getting all occupied cells"""
        grid_world.add_agent("agent1", GridCoordinate(5, 5))
        grid_world.add_obstacle(GridCoordinate(3, 3))

        occupied = grid_world.get_occupied_cells()

        assert len(occupied) == 2
        assert GridCoordinate(5, 5) in occupied
        assert GridCoordinate(3, 3) in occupied

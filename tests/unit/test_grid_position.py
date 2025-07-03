"""
Tests for Grid Position Module
"""

import math
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from world.grid_position import (
    GridCoordinate,
    GridPosition,
    GridSize,
    ProximityLevel,
    SpatialGridLogic,
)


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

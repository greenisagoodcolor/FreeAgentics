"""
Comprehensive tests for Movement and Perception System.

Tests the agent navigation and sensory systems in the hexagonal world,
including movement validation, pathfinding, line of sight calculations,
environmental perception, and exploration target selection.
"""

import math
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import h3

from agents.core.movement_perception import (
    Direction,
    Observation,
    MovementPerceptionSystem,
)
from world.h3_world import H3World, HexCell, TerrainType, Biome


class TestDirection:
    """Test Direction enum."""
    
    def test_direction_enum_values(self):
        """Test direction enum has correct values."""
        assert Direction.NORTH.value == 0
        assert Direction.NORTHEAST.value == 1
        assert Direction.SOUTHEAST.value == 2
        assert Direction.SOUTH.value == 3
        assert Direction.SOUTHWEST.value == 4
        assert Direction.NORTHWEST.value == 5
    
    def test_direction_enum_count(self):
        """Test correct number of directions."""
        directions = list(Direction)
        assert len(directions) == 6


class TestObservation:
    """Test Observation dataclass."""
    
    def setup_method(self):
        """Set up test observation data."""
        # Create mock hex cells with valid H3 IDs
        self.current_cell = HexCell(
            hex_id="872830828ffffff",
            biome=Biome.FOREST,
            terrain=TerrainType.FLAT,
            elevation=100.0,
            temperature=20.0,
            moisture=50.0,
            resources={"water": 10.0, "food": 5.0}
        )
        
        self.visible_cell = HexCell(
            hex_id="87283082effffff",
            biome=Biome.FOREST,
            terrain=TerrainType.HILLS,
            elevation=120.0,
            temperature=18.0,
            moisture=45.0,
            resources={"minerals": 15.0}
        )
        
        self.nearby_agents = [
            {
                "id": "agent_1",
                "position": "87283082effffff",
                "class": "explorer",
                "current_action": "exploring"
            }
        ]
        
        self.detected_resources = {"water": 10.0, "food": 5.0, "minerals": 15.0}
        self.movement_options = [(Direction.NORTH, "87283082effffff")]
    
    def test_observation_creation(self):
        """Test creating observation with all fields."""
        timestamp = 123.456
        
        observation = Observation(
            current_cell=self.current_cell,
            visible_cells=[self.current_cell, self.visible_cell],
            nearby_agents=self.nearby_agents,
            detected_resources=self.detected_resources,
            movement_options=self.movement_options,
            timestamp=timestamp
        )
        
        assert observation.current_cell == self.current_cell
        assert len(observation.visible_cells) == 2
        assert observation.nearby_agents == self.nearby_agents
        assert observation.detected_resources == self.detected_resources
        assert observation.movement_options == self.movement_options
        assert observation.timestamp == timestamp
    
    def test_observation_to_dict(self):
        """Test converting observation to dictionary."""
        observation = Observation(
            current_cell=self.current_cell,
            visible_cells=[self.current_cell, self.visible_cell],
            nearby_agents=self.nearby_agents,
            detected_resources=self.detected_resources,
            movement_options=self.movement_options,
            timestamp=100.0
        )
        
        obs_dict = observation.to_dict()
        
        # Check structure
        assert "location" in obs_dict
        assert "visible_area" in obs_dict
        assert "nearby_agents" in obs_dict
        assert "movement_options" in obs_dict
        assert "timestamp" in obs_dict
        
        # Check location data
        location = obs_dict["location"]
        assert location["hex_id"] == "872830828ffffff"
        assert location["biome"] == "forest"
        assert location["terrain"] == "flat"
        assert location["elevation"] == 100.0
        assert location["temperature"] == 20.0
        
        # Check visible area
        visible_area = obs_dict["visible_area"]
        assert len(visible_area["cells"]) == 2
        assert visible_area["total_resources"] == self.detected_resources
        
        # Check cell data includes distance
        for cell_data in visible_area["cells"]:
            assert "distance" in cell_data
            assert "hex_id" in cell_data
            assert "biome" in cell_data
            assert "terrain" in cell_data
            assert "resources" in cell_data
        
        # Check nearby agents
        assert obs_dict["nearby_agents"] == self.nearby_agents
        
        # Check movement options
        movement_opts = obs_dict["movement_options"]
        assert len(movement_opts) == 1
        assert movement_opts[0]["direction"] == "NORTH"
        assert movement_opts[0]["hex_id"] == "87283082effffff"
        assert "terrain" in movement_opts[0]
        
        assert obs_dict["timestamp"] == 100.0


class TestMovementPerceptionSystem:
    """Test MovementPerceptionSystem class."""
    
    def setup_method(self):
        """Set up test movement perception system."""
        # Create mock world
        self.mock_world = Mock(spec=H3World)
        
        # Create test cells
        self.center_hex = "872830828ffffff"
        self.neighbor_hex = "87283082effffff"
        
        self.center_cell = HexCell(
            hex_id=self.center_hex,
            biome=Biome.FOREST,
            terrain=TerrainType.FLAT,
            elevation=100.0,
            temperature=20.0,
            moisture=50.0,
            resources={"water": 10.0}
        )
        
        self.neighbor_cell = HexCell(
            hex_id=self.neighbor_hex,
            biome=Biome.FOREST,
            terrain=TerrainType.HILLS,
            elevation=120.0,
            temperature=18.0,
            moisture=45.0,
            resources={"food": 5.0}
        )
        
        # Configure world mock
        self.mock_world.cells = {
            self.center_hex: self.center_cell,
            self.neighbor_hex: self.neighbor_cell
        }
        
        def mock_get_cell(hex_id):
            return self.mock_world.cells.get(hex_id)
        
        self.mock_world.get_cell.side_effect = mock_get_cell
        
        # Create system
        self.system = MovementPerceptionSystem(self.mock_world)
    
    def test_system_initialization(self):
        """Test movement perception system initialization."""
        assert self.system.world == self.mock_world
    
    @patch('h3.grid_disk')
    @patch('h3.cell_to_latlng')
    def test_get_valid_moves_basic(self, mock_cell_to_latlng, mock_grid_disk):
        """Test getting valid moves from current position."""
        # Mock h3 functions
        mock_grid_disk.return_value = {self.center_hex, self.neighbor_hex}
        mock_cell_to_latlng.side_effect = [
            (37.7749, -122.4194),  # center
            (37.7750, -122.4195),  # neighbor
        ]
        
        valid_moves = self.system.get_valid_moves(self.center_hex)
        
        assert len(valid_moves) == 1
        direction, hex_id = valid_moves[0]
        assert isinstance(direction, Direction)
        assert hex_id == self.neighbor_hex
        
        # Verify h3 calls
        mock_grid_disk.assert_called_once_with(self.center_hex, 1)
        assert mock_cell_to_latlng.call_count == 2
    
    @patch('h3.grid_disk')
    def test_get_valid_moves_no_neighbors_in_world(self, mock_grid_disk):
        """Test getting valid moves when neighbors don't exist in world."""
        # Mock h3 to return non-existent neighbor
        mock_grid_disk.return_value = {self.center_hex, "87283082aaaaaab"}
        
        valid_moves = self.system.get_valid_moves(self.center_hex)
        
        assert len(valid_moves) == 0
    
    def test_calculate_direction_north(self):
        """Test calculating direction for northward movement."""
        # Northward movement (higher latitude)
        direction = self.system._calculate_direction(37.0, -122.0, 38.0, -122.0)
        
        assert isinstance(direction, Direction)
    
    def test_calculate_direction_south(self):
        """Test calculating direction for southward movement."""
        # Southward movement (lower latitude)
        direction = self.system._calculate_direction(38.0, -122.0, 37.0, -122.0)
        
        assert isinstance(direction, Direction)
    
    def test_calculate_direction_east(self):
        """Test calculating direction for eastward movement."""
        # Eastward movement (higher longitude)
        direction = self.system._calculate_direction(37.0, -122.0, 37.0, -121.0)
        
        assert isinstance(direction, Direction)
    
    def test_calculate_direction_west(self):
        """Test calculating direction for westward movement."""
        # Westward movement (lower longitude)
        direction = self.system._calculate_direction(37.0, -121.0, 37.0, -122.0)
        
        assert isinstance(direction, Direction)
    
    def test_can_move_to_valid_move(self):
        """Test checking valid move between adjacent hexes."""
        # Mock h3 distance
        with patch('h3.grid_distance', return_value=1):
            can_move, reason = self.system.can_move_to(self.center_hex, self.neighbor_hex)
        
        assert can_move is True
        assert reason is None
    
    def test_can_move_to_nonexistent_source(self):
        """Test checking move from non-existent hex."""
        self.mock_world.get_cell.return_value = None
        
        can_move, reason = self.system.can_move_to("nonexistent", self.neighbor_hex)
        
        assert can_move is False
        assert "does not exist" in reason
    
    def test_can_move_to_nonexistent_target(self):
        """Test checking move to non-existent hex."""
        def mock_get_cell_selective(hex_id):
            if hex_id == self.center_hex:
                return self.center_cell
            return None
        
        self.mock_world.get_cell.side_effect = mock_get_cell_selective
        
        can_move, reason = self.system.can_move_to(self.center_hex, "nonexistent")
        
        assert can_move is False
        assert "does not exist" in reason
    
    @patch('h3.grid_distance')
    def test_can_move_to_not_adjacent(self, mock_grid_distance):
        """Test checking move to non-adjacent hex."""
        mock_grid_distance.return_value = 2  # Not adjacent
        
        can_move, reason = self.system.can_move_to(self.center_hex, self.neighbor_hex)
        
        assert can_move is False
        assert "not adjacent" in reason
    
    @patch('h3.grid_distance')
    def test_can_move_to_water_terrain(self, mock_grid_distance):
        """Test checking move to water terrain."""
        mock_grid_distance.return_value = 1
        
        # Create water cell
        water_cell = HexCell(
            hex_id="water_hex",
            biome=Biome.OCEAN,
            terrain=TerrainType.WATER,
            elevation=0.0,
            temperature=15.0,
            moisture=100.0,
            resources={}
        )
        
        self.mock_world.cells["water_hex"] = water_cell
        
        can_move, reason = self.system.can_move_to(self.center_hex, "water_hex")
        
        assert can_move is False
        assert "water" in reason.lower()
    
    @patch('h3.grid_distance')
    def test_can_move_to_steep_elevation(self, mock_grid_distance):
        """Test checking move with steep elevation change."""
        mock_grid_distance.return_value = 1
        
        # Create high elevation cell
        high_cell = HexCell(
            hex_id="high_hex",
            biome=Biome.MOUNTAIN,
            terrain=TerrainType.HILLS,
            elevation=350.0,  # 250m higher than center (100m)
            temperature=10.0,
            moisture=30.0,
            resources={}
        )
        
        self.mock_world.cells["high_hex"] = high_cell
        
        can_move, reason = self.system.can_move_to(self.center_hex, "high_hex")
        
        assert can_move is False
        assert "steep" in reason.lower()
    
    def test_calculate_movement_cost_basic(self):
        """Test calculating basic movement cost."""
        # Set movement cost on target cell
        self.neighbor_cell.movement_cost = 2.0
        
        cost = self.system.calculate_movement_cost(self.center_hex, self.neighbor_hex)
        
        # Base cost (2.0) + elevation cost + temperature penalty
        expected_base = 2.0
        elevation_gain = max(0, 120.0 - 100.0)  # 20m gain
        elevation_cost = elevation_gain / 100  # 0.2
        temp_penalty = 0  # 18°C is within normal range
        
        expected_cost = expected_base + elevation_cost + temp_penalty
        assert cost == expected_cost
    
    def test_calculate_movement_cost_with_elevation_gain(self):
        """Test movement cost calculation with elevation gain."""
        # Create high elevation target
        high_cell = HexCell(
            hex_id="high_hex",
            biome=Biome.MOUNTAIN,
            terrain=TerrainType.HILLS,
            elevation=200.0,  # 100m higher
            temperature=15.0,
            moisture=30.0,
            resources={},
            movement_cost=1.5
        )
        
        self.mock_world.cells["high_hex"] = high_cell
        
        cost = self.system.calculate_movement_cost(self.center_hex, "high_hex")
        
        # Base cost (1.5) + elevation cost (1.0) + temp penalty (0)
        expected_cost = 1.5 + 1.0 + 0
        assert cost == expected_cost
    
    def test_calculate_movement_cost_with_temperature_penalty(self):
        """Test movement cost with extreme temperature penalty."""
        # Create extreme cold cell
        cold_cell = HexCell(
            hex_id="cold_hex",
            biome=Biome.TUNDRA,
            terrain=TerrainType.FLAT,
            elevation=100.0,  # Same elevation
            temperature=-25.0,  # Below -20°C
            moisture=20.0,
            resources={},
            movement_cost=1.0
        )
        
        self.mock_world.cells["cold_hex"] = cold_cell
        
        cost = self.system.calculate_movement_cost(self.center_hex, "cold_hex")
        
        # Base cost (1.0) + elevation cost (0) + temp penalty (0.5)
        expected_cost = 1.0 + 0 + 0.5
        assert cost == expected_cost
    
    def test_calculate_movement_cost_nonexistent_cells(self):
        """Test movement cost calculation with non-existent cells."""
        cost = self.system.calculate_movement_cost("nonexistent1", "nonexistent2")
        
        assert cost == float("inf")
    
    def test_get_agent_observation_basic(self):
        """Test getting basic agent observation."""
        # Mock world methods
        visible_cells = [self.center_cell, self.neighbor_cell]
        self.mock_world.get_visible_cells.return_value = visible_cells
        
        # Mock valid moves
        with patch.object(self.system, 'get_valid_moves') as mock_get_moves:
            mock_get_moves.return_value = [(Direction.NORTH, self.neighbor_hex)]
            
            # Mock line of sight filtering
            with patch.object(self.system, '_apply_line_of_sight') as mock_los:
                mock_los.return_value = visible_cells
                
                observation = self.system.get_agent_observation(self.center_hex)
        
        assert isinstance(observation, Observation)
        assert observation.current_cell == self.center_cell
        assert len(observation.visible_cells) == 2
        assert len(observation.nearby_agents) == 0  # No other agents
        assert observation.detected_resources == {"water": 10.0, "food": 5.0}
        assert len(observation.movement_options) == 1
        assert observation.timestamp == 0.0
    
    def test_get_agent_observation_with_other_agents(self):
        """Test getting observation with other agents visible."""
        other_agents = [
            {
                "id": "agent_1",
                "position": self.neighbor_hex,
                "class": "explorer",
                "current_action": "exploring"
            },
            {
                "id": "agent_2",
                "position": "far_away_hex",  # Not visible
                "class": "gatherer",
                "current_action": "gathering"
            }
        ]
        
        visible_cells = [self.center_cell, self.neighbor_cell]
        self.mock_world.get_visible_cells.return_value = visible_cells
        
        with patch.object(self.system, 'get_valid_moves') as mock_get_moves:
            mock_get_moves.return_value = [(Direction.NORTH, self.neighbor_hex)]
            
            with patch.object(self.system, '_apply_line_of_sight') as mock_los:
                mock_los.return_value = visible_cells
                
                observation = self.system.get_agent_observation(
                    self.center_hex, other_agents
                )
        
        # Should only see agent_1 (in visible area)
        assert len(observation.nearby_agents) == 1
        nearby_agent = observation.nearby_agents[0]
        assert nearby_agent["id"] == "agent_1"
        assert nearby_agent["position"] == self.neighbor_hex
        assert nearby_agent["class"] == "explorer"
        assert nearby_agent["visible_action"] == "exploring"
    
    def test_get_agent_observation_invalid_position(self):
        """Test getting observation from invalid position."""
        self.mock_world.get_cell.return_value = None
        
        with pytest.raises(ValueError, match="Invalid agent position"):
            self.system.get_agent_observation("invalid_hex")
    
    def test_apply_line_of_sight_direct_visibility(self):
        """Test line of sight with direct visibility."""
        observer_hex = self.center_hex
        potential_visible = [self.center_cell, self.neighbor_cell]
        
        # Mock h3 path
        with patch('h3.grid_path_cells') as mock_path:
            mock_path.return_value = [self.center_hex, self.neighbor_hex]
            
            visible = self.system._apply_line_of_sight(observer_hex, potential_visible)
        
        # Should see both cells (observer + target)
        assert len(visible) == 2
        assert self.center_cell in visible
        assert self.neighbor_cell in visible
    
    def test_apply_line_of_sight_blocked_by_elevation(self):
        """Test line of sight blocked by elevation."""
        observer_hex = self.center_hex
        
        # Create target cell
        target_cell = HexCell(
            hex_id="target_hex",
            biome=Biome.FOREST,
            terrain=TerrainType.FLAT,
            elevation=100.0,
            temperature=20.0,
            moisture=50.0,
            resources={}
        )
        
        # Create blocking cell with high elevation
        blocking_cell = HexCell(
            hex_id="blocking_hex",
            biome=Biome.MOUNTAIN,
            terrain=TerrainType.HILLS,
            elevation=200.0,  # High elevation blocks view
            temperature=15.0,
            moisture=30.0,
            resources={}
        )
        
        self.mock_world.cells["target_hex"] = target_cell
        self.mock_world.cells["blocking_hex"] = blocking_cell
        
        potential_visible = [self.center_cell, target_cell]
        
        # Mock h3 path to include blocking cell
        with patch('h3.grid_path_cells') as mock_path:
            mock_path.return_value = [self.center_hex, "blocking_hex", "target_hex"]
            
            visible = self.system._apply_line_of_sight(observer_hex, potential_visible)
        
        # Should only see observer cell (target blocked)
        assert len(visible) == 1
        assert self.center_cell in visible
        assert target_cell not in visible
    
    def test_apply_line_of_sight_nonexistent_observer(self):
        """Test line of sight from non-existent observer."""
        self.mock_world.get_cell.return_value = None
        
        visible = self.system._apply_line_of_sight("nonexistent", [self.neighbor_cell])
        
        assert len(visible) == 0
    
    @patch('h3.grid_distance')
    def test_find_path_astar_same_position(self, mock_grid_distance):
        """Test A* pathfinding from position to itself."""
        path = self.system.find_path_astar(self.center_hex, self.center_hex)
        
        assert path == [self.center_hex]
    
    @patch('h3.grid_distance')
    def test_find_path_astar_nonexistent_start(self, mock_grid_distance):
        """Test A* pathfinding from non-existent start."""
        self.mock_world.get_cell.side_effect = lambda x: None if x == "nonexistent" else self.center_cell
        
        path = self.system.find_path_astar("nonexistent", self.center_hex)
        
        assert path is None
    
    @patch('h3.grid_distance')
    def test_find_path_astar_nonexistent_goal(self, mock_grid_distance):
        """Test A* pathfinding to non-existent goal."""
        self.mock_world.get_cell.side_effect = lambda x: self.center_cell if x == self.center_hex else None
        
        path = self.system.find_path_astar(self.center_hex, "nonexistent")
        
        assert path is None
    
    @patch('h3.grid_distance')
    def test_find_path_astar_simple_path(self, mock_grid_distance):
        """Test A* pathfinding with simple path."""
        # Mock distance function
        def mock_distance(hex1, hex2):
            if hex1 == hex2:
                return 0
            return 1
        
        mock_grid_distance.side_effect = mock_distance
        
        # Mock valid moves and movement checks
        with patch.object(self.system, 'get_valid_moves') as mock_get_moves:
            with patch.object(self.system, 'can_move_to') as mock_can_move:
                with patch.object(self.system, 'calculate_movement_cost') as mock_cost:
                    
                    # Setup mocks
                    mock_get_moves.return_value = [(Direction.NORTH, self.neighbor_hex)]
                    mock_can_move.return_value = (True, None)
                    mock_cost.return_value = 1.0
                    
                    path = self.system.find_path_astar(self.center_hex, self.neighbor_hex)
        
        assert path is not None
        assert path[0] == self.center_hex
        assert path[-1] == self.neighbor_hex
    
    @patch('h3.grid_distance')
    def test_find_path_astar_exceeds_max_cost(self, mock_grid_distance):
        """Test A* pathfinding that exceeds max cost."""
        mock_grid_distance.side_effect = lambda x, y: 0 if x == y else 1
        
        with patch.object(self.system, 'get_valid_moves') as mock_get_moves:
            with patch.object(self.system, 'can_move_to') as mock_can_move:
                with patch.object(self.system, 'calculate_movement_cost') as mock_cost:
                    
                    mock_get_moves.return_value = [(Direction.NORTH, self.neighbor_hex)]
                    mock_can_move.return_value = (True, None)
                    mock_cost.return_value = 150.0  # Exceeds max_cost of 100
                    
                    path = self.system.find_path_astar(
                        self.center_hex, self.neighbor_hex, max_cost=100.0
                    )
        
        assert path is None
    
    @patch('h3.grid_distance')
    def test_get_exploration_targets_basic(self, mock_grid_distance):
        """Test getting exploration targets."""
        # Create additional cells for exploration
        target_cells = []
        for i in range(3):
            cell = HexCell(
                hex_id=f"target_{i}",
                biome=Biome.FOREST,
                terrain=TerrainType.FLAT,
                elevation=100.0,
                temperature=20.0,
                moisture=50.0,
                resources={"minerals": 10.0 + i * 5.0}  # Varying resources
            )
            target_cells.append(cell)
            self.mock_world.cells[f"target_{i}"] = cell
        
        # Mock world method
        all_cells = [self.center_cell] + target_cells
        self.mock_world.get_cells_in_range.return_value = all_cells
        
        # Mock distance calculations
        def mock_distance(hex1, hex2):
            if hex1 == hex2:
                return 0
            if hex2 == "target_0":
                return 1
            elif hex2 == "target_1":
                return 2
            elif hex2 == "target_2":
                return 3
            return 5
        
        mock_grid_distance.side_effect = mock_distance
        
        explored_hexes = {self.center_hex}  # Only center explored
        
        targets = self.system.get_exploration_targets(
            self.center_hex, explored_hexes, num_targets=2
        )
        
        assert len(targets) <= 2
        assert self.center_hex not in targets  # Already explored
        
        # Should prefer closer targets with more resources
        for target in targets:
            assert target in [f"target_{i}" for i in range(3)]
    
    @patch('h3.grid_distance')
    def test_get_exploration_targets_all_explored(self, mock_grid_distance):
        """Test getting exploration targets when all areas explored."""
        # Mock world method
        self.mock_world.get_cells_in_range.return_value = [self.center_cell, self.neighbor_cell]
        
        # Mark all cells as explored
        explored_hexes = {self.center_hex, self.neighbor_hex}
        
        targets = self.system.get_exploration_targets(
            self.center_hex, explored_hexes, num_targets=3
        )
        
        assert len(targets) == 0
    
    @patch('h3.grid_distance')
    def test_get_exploration_targets_with_biome_variety(self, mock_grid_distance):
        """Test exploration targets considering biome variety."""
        # Create cell with different biome
        desert_cell = HexCell(
            hex_id="desert_target",
            biome=Biome.DESERT,  # Different from current (FOREST)
            terrain=TerrainType.FLAT,
            elevation=100.0,
            temperature=30.0,
            moisture=20.0,
            resources={"minerals": 5.0}
        )
        
        # Create cell with same biome
        forest_cell = HexCell(
            hex_id="forest_target",
            biome=Biome.FOREST,  # Same as current
            terrain=TerrainType.FLAT,
            elevation=100.0,
            temperature=20.0,
            moisture=50.0,
            resources={"minerals": 5.0}
        )
        
        self.mock_world.cells["desert_target"] = desert_cell
        self.mock_world.cells["forest_target"] = forest_cell
        
        all_cells = [self.center_cell, desert_cell, forest_cell]
        self.mock_world.get_cells_in_range.return_value = all_cells
        
        # Mock equal distances
        mock_grid_distance.side_effect = lambda x, y: 0 if x == y else 2
        
        explored_hexes = {self.center_hex}
        
        targets = self.system.get_exploration_targets(
            self.center_hex, explored_hexes, num_targets=2
        )
        
        # Desert target should be preferred due to variety bonus
        assert "desert_target" in targets


class TestIntegrationScenarios:
    """Test integrated scenarios with movement and perception."""
    
    def setup_method(self):
        """Set up integration test scenario."""
        # Create a more realistic world setup
        self.world = Mock(spec=H3World)
        self.system = MovementPerceptionSystem(self.world)
        
        # Create a small network of connected cells with valid H3 IDs
        self.cells = {}
        cell_configs = [
            ("872830828ffffff", Biome.FOREST, TerrainType.FLAT, 100.0, 20.0, 50.0),
            ("87283082effffff", Biome.FOREST, TerrainType.HILLS, 120.0, 18.0, 45.0),
            ("87283082affffff", Biome.FOREST, TerrainType.FLAT, 95.0, 22.0, 55.0),
            ("87283082bffffff", Biome.DESERT, TerrainType.FLAT, 110.0, 30.0, 20.0),
            ("872830829ffffff", Biome.OCEAN, TerrainType.WATER, 0.0, 15.0, 100.0),
        ]
        
        for hex_id, biome, terrain, elevation, temperature, moisture in cell_configs:
            self.cells[hex_id] = HexCell(
                hex_id=hex_id,
                biome=biome,
                terrain=terrain,
                elevation=elevation,
                temperature=temperature,
                moisture=moisture,
                resources={"food": 5.0} if hex_id != "872830829ffffff" else {}
            )
        
        self.world.cells = self.cells
        self.world.get_cell.side_effect = lambda x: self.cells.get(x)
    
    @patch('h3.grid_disk')
    @patch('h3.cell_to_latlng')
    @patch('h3.grid_distance')
    def test_complete_movement_scenario(self, mock_distance, mock_cell_to_latlng, mock_grid_disk):
        """Test complete movement scenario with planning and execution."""
        # Setup mocks
        mock_grid_disk.return_value = {"872830828ffffff", "87283082effffff", "87283082affffff", "87283082bffffff"}
        mock_cell_to_latlng.side_effect = [
            (37.7749, -122.4194),  # center
            (37.7750, -122.4194),  # north
            (37.7748, -122.4194),  # south
            (37.7749, -122.4193),  # east
        ]
        mock_distance.side_effect = lambda x, y: 0 if x == y else 1
        
        # Step 1: Get valid moves from center
        valid_moves = self.system.get_valid_moves("872830828ffffff")
        
        assert len(valid_moves) == 3  # north, south, east (water excluded)
        
        # Step 2: Check movement validity
        can_move_north, _ = self.system.can_move_to("872830828ffffff", "87283082effffff")
        can_move_water, reason = self.system.can_move_to("872830828ffffff", "872830829ffffff")
        
        assert can_move_north is True
        assert can_move_water is False
        assert "water" in reason.lower()
        
        # Step 3: Calculate movement costs
        cost_north = self.system.calculate_movement_cost("872830828ffffff", "87283082effffff")
        cost_east = self.system.calculate_movement_cost("872830828ffffff", "87283082bffffff")
        
        # North has elevation gain (20m) and normal temperature (18°C)
        assert cost_north > 1.0  # Base cost + elevation cost
        
        # East has no elevation gain but normal temperature (30°C, no penalty)
        assert cost_east < cost_north  # Lower due to no elevation gain
    
    def test_perception_with_multiple_agents(self):
        """Test perception system with multiple agents in view."""
        other_agents = [
            {"id": "explorer_1", "position": "87283082effffff", "class": "explorer", "current_action": "exploring"},
            {"id": "gatherer_1", "position": "87283082affffff", "class": "gatherer", "current_action": "gathering"},
            {"id": "distant_agent", "position": "far_away", "class": "trader", "current_action": "trading"},
        ]
        
        visible_cells = [self.cells["872830828ffffff"], self.cells["87283082effffff"], self.cells["87283082affffff"]]
        self.world.get_visible_cells.return_value = visible_cells
        
        with patch.object(self.system, 'get_valid_moves') as mock_moves:
            with patch.object(self.system, '_apply_line_of_sight') as mock_los:
                mock_moves.return_value = [(Direction.NORTH, "87283082effffff")]
                mock_los.return_value = visible_cells
                
                observation = self.system.get_agent_observation("872830828ffffff", other_agents)
        
        # Should see explorer_1 and gatherer_1 (both in visible area)
        assert len(observation.nearby_agents) == 2
        
        agent_ids = [agent["id"] for agent in observation.nearby_agents]
        assert "explorer_1" in agent_ids
        assert "gatherer_1" in agent_ids
        assert "distant_agent" not in agent_ids  # Not in visible area
        
        # Check resource detection from multiple cells
        expected_resources = {"food": 15.0}  # 3 cells * 5.0 food each
        assert observation.detected_resources == expected_resources
    
    @patch('h3.grid_distance')
    def test_exploration_target_selection_strategy(self, mock_distance):
        """Test strategic exploration target selection."""
        # Create diverse exploration targets
        exploration_cells = []
        for i in range(5):
            cell = HexCell(
                hex_id=f"explore_{i}",
                biome=Biome.DESERT if i % 2 == 0 else Biome.FOREST,
                terrain=TerrainType.FLAT,
                elevation=100.0 + i * 10,
                temperature=20.0 + i * 5,
                moisture=50.0 - i * 5,
                resources={"minerals": 20.0 - i * 3}  # Decreasing resources with distance
            )
            exploration_cells.append(cell)
            self.cells[f"explore_{i}"] = cell
        
        all_cells = list(self.cells.values())
        self.world.get_cells_in_range.return_value = all_cells
        
        # Mock distances (increasing with index)
        def mock_distance_calc(hex1, hex2):
            if hex1 == hex2:
                return 0
            if hex2.startswith("explore_"):
                idx = int(hex2.split("_")[1])
                return idx + 1
            return 1
        
        mock_distance.side_effect = mock_distance_calc
        
        explored_hexes = {"872830828ffffff", "87283082effffff", "87283082affffff", "87283082bffffff", "872830829ffffff"}
        
        targets = self.system.get_exploration_targets("872830828ffffff", explored_hexes, num_targets=3)
        
        # Should prioritize closer targets with more resources and biome variety
        assert len(targets) <= 3
        
        # explore_0 should be highly ranked (close, high resources, different biome)
        assert "explore_0" in targets
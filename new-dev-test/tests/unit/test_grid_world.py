"""
Comprehensive test suite for GridWorld environment.

Tests the GridWorld environment functionality for multi-agent
Active Inference simulations.
"""

import numpy as np
import pytest

from world.grid_world import Agent, Cell, CellType, GridWorld, GridWorldConfig, Position


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test position creation."""
        pos = Position(3, 4)
        assert pos.x == 3
        assert pos.y == 4

    def test_position_equality(self):
        """Test position equality."""
        pos1 = Position(2, 3)
        pos2 = Position(2, 3)
        pos3 = Position(3, 2)

        assert pos1 == pos2
        assert pos1 != pos3

    def test_position_distance(self):
        """Test position distance calculation."""
        pos1 = Position(0, 0)
        pos2 = Position(3, 4)

        distance = pos1.distance_to(pos2)
        expected = np.sqrt(3**2 + 4**2)
        assert abs(distance - expected) < 1e-6

    def test_position_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        pos1 = Position(1, 1)
        pos2 = Position(4, 5)

        manhattan = pos1.manhattan_distance(pos2)
        assert manhattan == 7  # |4-1| + |5-1|

    def test_position_neighbors(self):
        """Test getting neighboring positions."""
        pos = Position(2, 2)
        neighbors = pos.get_neighbors()

        expected = [
            Position(1, 2),  # Left
            Position(3, 2),  # Right
            Position(2, 1),  # Up
            Position(2, 3),  # Down
        ]

        assert len(neighbors) == 4
        for expected_pos in expected:
            assert expected_pos in neighbors


class TestCellType:
    """Test CellType enum."""

    def test_cell_types(self):
        """Test all cell types are defined."""
        assert CellType.EMPTY.value == "empty"
        assert CellType.WALL.value == "wall"
        assert CellType.GOAL.value == "goal"
        assert CellType.HAZARD.value == "hazard"
        assert CellType.RESOURCE.value == "resource"


class TestCell:
    """Test Cell dataclass."""

    def test_cell_creation(self):
        """Test cell creation."""
        cell = Cell(
            type=CellType.GOAL,
            position=Position(5, 5),
            value=10.0,
            properties={"color": "green"},
        )

        assert cell.type == CellType.GOAL
        assert cell.position == Position(5, 5)
        assert cell.value == 10.0
        assert cell.properties["color"] == "green"

    def test_cell_defaults(self):
        """Test cell with default values."""
        cell = Cell(CellType.EMPTY, Position(0, 0))

        assert cell.type == CellType.EMPTY
        assert cell.value == 0.0
        assert cell.properties == {}

    def test_cell_is_passable(self):
        """Test cell passability."""
        empty_cell = Cell(CellType.EMPTY, Position(0, 0))
        wall_cell = Cell(CellType.WALL, Position(1, 0))
        goal_cell = Cell(CellType.GOAL, Position(2, 0))

        assert empty_cell.is_passable() is True
        assert wall_cell.is_passable() is False
        assert goal_cell.is_passable() is True


class TestAgent:
    """Test Agent dataclass."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = Agent(
            id="agent_1",
            position=Position(1, 1),
            energy=100.0,
            resources={"gold": 5},
            properties={"speed": 1.5},
        )

        assert agent.id == "agent_1"
        assert agent.position == Position(1, 1)
        assert agent.energy == 100.0
        assert agent.resources["gold"] == 5
        assert agent.properties["speed"] == 1.5

    def test_agent_defaults(self):
        """Test agent with default values."""
        agent = Agent("test", Position(0, 0))

        assert agent.energy == 100.0
        assert agent.resources == {}
        assert agent.properties == {}


class TestGridWorldConfig:
    """Test GridWorldConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = GridWorldConfig()

        assert config.width == 10
        assert config.height == 10
        assert config.wrap_edges is False
        assert config.max_agents == 10
        assert config.enable_collisions is True
        assert config.resource_respawn is False
        assert config.step_penalty == -0.1

    def test_config_custom(self):
        """Test custom configuration."""
        config = GridWorldConfig(
            width=20,
            height=15,
            wrap_edges=True,
            max_agents=5,
            enable_collisions=False,
            resource_respawn=True,
            step_penalty=-0.5,
        )

        assert config.width == 20
        assert config.height == 15
        assert config.wrap_edges is True
        assert config.max_agents == 5
        assert config.enable_collisions is False
        assert config.resource_respawn is True
        assert config.step_penalty == -0.5


class TestGridWorld:
    """Test GridWorld environment."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return GridWorldConfig(width=5, height=5, max_agents=3)

    @pytest.fixture
    def world(self, config):
        """Create test world."""
        return GridWorld(config)

    def test_world_initialization(self, world, config):
        """Test world initialization."""
        assert world.config == config
        assert world.width == 5
        assert world.height == 5
        assert len(world.grid) == 5
        assert len(world.grid[0]) == 5
        assert world.agents == {}
        assert world.step_count == 0

    def test_grid_initialization(self, world):
        """Test grid initialization."""
        for x in range(world.width):
            for y in range(world.height):
                cell = world.grid[x][y]
                assert cell.type == CellType.EMPTY
                assert cell.position == Position(x, y)

    def test_is_valid_position(self, world):
        """Test position validation."""
        assert world.is_valid_position(Position(0, 0)) is True
        assert world.is_valid_position(Position(4, 4)) is True
        assert world.is_valid_position(Position(-1, 0)) is False
        assert world.is_valid_position(Position(0, -1)) is False
        assert world.is_valid_position(Position(5, 0)) is False
        assert world.is_valid_position(Position(0, 5)) is False

    def test_add_agent(self, world):
        """Test adding agents."""
        agent = Agent("test_agent", Position(1, 1))
        success = world.add_agent(agent)

        assert success is True
        assert "test_agent" in world.agents
        assert world.agents["test_agent"] == agent

    def test_add_agent_invalid_position(self, world):
        """Test adding agent to invalid position."""
        agent = Agent("test", Position(-1, -1))
        success = world.add_agent(agent)

        assert success is False
        assert "test" not in world.agents

    def test_add_agent_occupied_position(self, world):
        """Test adding agent to occupied position."""
        agent1 = Agent("agent1", Position(2, 2))
        agent2 = Agent("agent2", Position(2, 2))

        world.add_agent(agent1)
        success = world.add_agent(agent2)

        assert success is False
        assert "agent2" not in world.agents

    def test_remove_agent(self, world):
        """Test removing agents."""
        agent = Agent("removable", Position(3, 3))
        world.add_agent(agent)

        assert "removable" in world.agents

        removed = world.remove_agent("removable")

        assert removed == agent
        assert "removable" not in world.agents

    def test_remove_nonexistent_agent(self, world):
        """Test removing non-existent agent."""
        removed = world.remove_agent("nonexistent")
        assert removed is None

    def test_get_agent_at_position(self, world):
        """Test getting agent at position."""
        agent = Agent("positioned", Position(2, 3))
        world.add_agent(agent)

        found_agent = world.get_agent_at_position(Position(2, 3))
        assert found_agent == agent

        no_agent = world.get_agent_at_position(Position(1, 1))
        assert no_agent is None

    def test_move_agent(self, world):
        """Test moving agents."""
        agent = Agent("mover", Position(1, 1))
        world.add_agent(agent)

        success = world.move_agent("mover", Position(2, 2))

        assert success is True
        assert agent.position == Position(2, 2)

    def test_move_agent_invalid_destination(self, world):
        """Test moving agent to invalid position."""
        agent = Agent("stuck", Position(1, 1))
        world.add_agent(agent)

        success = world.move_agent("stuck", Position(-1, -1))

        assert success is False
        assert agent.position == Position(1, 1)

    def test_move_agent_collision(self, world):
        """Test agent collision during movement."""
        agent1 = Agent("blocker", Position(2, 2))
        agent2 = Agent("mover", Position(1, 1))

        world.add_agent(agent1)
        world.add_agent(agent2)

        success = world.move_agent("mover", Position(2, 2))

        assert success is False
        assert agent2.position == Position(1, 1)

    def test_move_agent_with_collisions_disabled(self, world):
        """Test movement with collisions disabled."""
        world.config.enable_collisions = False

        agent1 = Agent("blocker", Position(2, 2))
        agent2 = Agent("mover", Position(1, 1))

        world.add_agent(agent1)
        world.add_agent(agent2)

        success = world.move_agent("mover", Position(2, 2))

        assert success is True
        assert agent2.position == Position(2, 2)

    def test_set_cell(self, world):
        """Test setting cell types."""
        world.set_cell(Position(2, 2), CellType.WALL)

        cell = world.get_cell(Position(2, 2))
        assert cell.type == CellType.WALL

    def test_set_cell_invalid_position(self, world):
        """Test setting cell at invalid position."""
        success = world.set_cell(Position(-1, -1), CellType.WALL)
        assert success is False

    def test_get_cell(self, world):
        """Test getting cells."""
        cell = world.get_cell(Position(1, 1))
        assert cell.type == CellType.EMPTY
        assert cell.position == Position(1, 1)

    def test_get_cell_invalid_position(self, world):
        """Test getting cell at invalid position."""
        cell = world.get_cell(Position(-1, -1))
        assert cell is None

    def test_get_neighbors(self, world):
        """Test getting neighboring positions."""
        neighbors = world.get_neighbors(Position(2, 2))

        expected = [
            Position(1, 2),  # Left
            Position(3, 2),  # Right
            Position(2, 1),  # Up
            Position(2, 3),  # Down
        ]

        assert len(neighbors) == 4
        for pos in expected:
            assert pos in neighbors

    def test_get_neighbors_edge(self, world):
        """Test getting neighbors at edge."""
        neighbors = world.get_neighbors(Position(0, 0))

        expected = [Position(1, 0), Position(0, 1)]  # Right  # Down

        assert len(neighbors) == 2
        for pos in expected:
            assert pos in neighbors

    def test_get_neighbors_wrapped(self, world):
        """Test getting neighbors with edge wrapping."""
        world.config.wrap_edges = True
        neighbors = world.get_neighbors(Position(0, 0))

        expected = [
            Position(4, 0),  # Left (wrapped)
            Position(1, 0),  # Right
            Position(0, 4),  # Up (wrapped)
            Position(0, 1),  # Down
        ]

        assert len(neighbors) == 4
        for pos in expected:
            assert pos in neighbors

    def test_get_observation(self, world):
        """Test getting agent observations."""
        agent = Agent("observer", Position(2, 2))
        world.add_agent(agent)

        # Add some features to observe
        world.set_cell(Position(1, 2), CellType.GOAL)
        world.set_cell(Position(3, 2), CellType.WALL)

        observation = world.get_observation("observer", radius=1)

        assert "agent_position" in observation
        assert "local_grid" in observation
        assert "nearby_agents" in observation
        assert observation["agent_position"] == Position(2, 2)

    def test_get_observation_nonexistent_agent(self, world):
        """Test observation for non-existent agent."""
        observation = world.get_observation("ghost")
        assert observation is None

    def test_step(self, world):
        """Test world step."""
        agent = Agent("stepper", Position(1, 1), energy=100.0)
        world.add_agent(agent)

        initial_step = world.step_count
        world.step()

        assert world.step_count == initial_step + 1
        # Agent energy should decrease due to step penalty
        assert agent.energy < 100.0

    def test_step_energy_consumption(self, world):
        """Test energy consumption during step."""
        agent = Agent("tired", Position(1, 1), energy=100.0)
        world.add_agent(agent)
        world.config.step_penalty = -2.0

        world.step()

        assert agent.energy == 98.0

    def test_reset(self, world):
        """Test world reset."""
        # Add agent and set some state
        agent = Agent("resettable", Position(2, 2))
        world.add_agent(agent)
        world.set_cell(Position(1, 1), CellType.WALL)
        world.step_count = 10

        world.reset()

        assert world.step_count == 0
        assert world.agents == {}
        # Grid should be reset to empty
        for x in range(world.width):
            for y in range(world.height):
                assert world.grid[x][y].type == CellType.EMPTY

    def test_get_state(self, world):
        """Test getting world state."""
        agent = Agent("stateful", Position(1, 1))
        world.add_agent(agent)
        world.set_cell(Position(2, 2), CellType.GOAL)

        state = world.get_state()

        assert "agents" in state
        assert "grid" in state
        assert "step_count" in state
        assert "config" in state
        assert len(state["agents"]) == 1

    def test_load_state(self, world):
        """Test loading world state."""
        state = {
            "agents": {
                "loaded_agent": {
                    "id": "loaded_agent",
                    "position": {"x": 3, "y": 3},
                    "energy": 80.0,
                    "resources": {},
                    "properties": {},
                }
            },
            "grid": [[{"type": "empty", "value": 0.0} for _ in range(5)] for _ in range(5)],
            "step_count": 42,
        }

        world.load_state(state)

        assert world.step_count == 42
        assert "loaded_agent" in world.agents
        assert world.agents["loaded_agent"].position == Position(3, 3)
        assert world.agents["loaded_agent"].energy == 80.0

    def test_pathfind(self, world):
        """Test pathfinding."""
        # Add some walls
        world.set_cell(Position(2, 1), CellType.WALL)
        world.set_cell(Position(2, 2), CellType.WALL)
        world.set_cell(Position(2, 3), CellType.WALL)

        start = Position(1, 2)
        goal = Position(3, 2)

        path = world.find_path(start, goal)

        assert path is not None
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal

    def test_pathfind_no_path(self, world):
        """Test pathfinding when no path exists."""
        # Create a wall barrier
        for y in range(5):
            world.set_cell(Position(2, y), CellType.WALL)

        start = Position(1, 2)
        goal = Position(3, 2)

        path = world.find_path(start, goal)
        assert path is None

    def test_get_distance_map(self, world):
        """Test distance map calculation."""
        target = Position(2, 2)
        distance_map = world.get_distance_map(target)

        assert distance_map[2][2] == 0
        assert distance_map[1][2] == 1
        assert distance_map[3][2] == 1
        assert distance_map[2][1] == 1
        assert distance_map[2][3] == 1
        assert distance_map[0][0] == 4  # Manhattan distance

    def test_resource_collection(self, world):
        """Test resource collection mechanics."""
        agent = Agent("collector", Position(2, 2))
        world.add_agent(agent)

        # Place resource
        resource_cell = Cell(
            CellType.RESOURCE,
            Position(2, 2),
            value=5.0,
            properties={"resource_type": "gold", "amount": 10},
        )
        world.grid[2][2] = resource_cell

        collected = world.collect_resource("collector")

        assert collected is True
        assert agent.resources.get("gold", 0) == 10
        assert world.grid[2][2].type == CellType.EMPTY

    def test_resource_collection_no_resource(self, world):
        """Test resource collection with no resource."""
        agent = Agent("empty_handed", Position(1, 1))
        world.add_agent(agent)

        collected = world.collect_resource("empty_handed")

        assert collected is False
        assert agent.resources == {}


class TestGridWorldIntegration:
    """Integration tests for GridWorld."""

    def test_multi_agent_simulation(self):
        """Test multi-agent simulation."""
        config = GridWorldConfig(width=10, height=10, max_agents=5)
        world = GridWorld(config)

        # Add multiple agents
        agents = [Agent(f"agent_{i}", Position(i, 0)) for i in range(5)]

        for agent in agents:
            world.add_agent(agent)

        assert len(world.agents) == 5

        # Simulate several steps
        for _ in range(10):
            world.step()

        assert world.step_count == 10

    def test_goal_seeking_scenario(self):
        """Test goal-seeking scenario."""
        config = GridWorldConfig(width=5, height=5)
        world = GridWorld(config)

        # Add agent and goal
        agent = Agent("seeker", Position(0, 0))
        world.add_agent(agent)
        world.set_cell(Position(4, 4), CellType.GOAL)

        # Agent should be able to reach goal
        path = world.find_path(agent.position, Position(4, 4))
        assert path is not None
        assert len(path) == 9  # Manhattan distance + 1

    def test_maze_navigation(self):
        """Test navigation in maze-like environment."""
        config = GridWorldConfig(width=7, height=7)
        world = GridWorld(config)

        # Create simple maze
        walls = [
            Position(1, 1),
            Position(1, 2),
            Position(1, 3),
            Position(3, 1),
            Position(3, 3),
            Position(3, 4),
            Position(5, 1),
            Position(5, 2),
            Position(5, 3),
        ]

        for wall_pos in walls:
            world.set_cell(wall_pos, CellType.WALL)

        # Test pathfinding through maze
        start = Position(0, 0)
        goal = Position(6, 6)

        path = world.find_path(start, goal)
        assert path is not None

        # Verify path doesn't go through walls
        for pos in path:
            cell = world.get_cell(pos)
            assert cell.type != CellType.WALL

    def test_resource_economy(self):
        """Test resource-based economy simulation."""
        config = GridWorldConfig(width=5, height=5, resource_respawn=True)
        world = GridWorld(config)

        # Add agents
        collector = Agent("collector", Position(0, 0))
        world.add_agent(collector)

        # Place resources
        resource_positions = [Position(2, 2), Position(3, 3), Position(4, 4)]
        for pos in resource_positions:
            world.set_cell(pos, CellType.RESOURCE)
            world.grid[pos.x][pos.y].properties = {
                "resource_type": "energy",
                "amount": 20,
            }

        # Collect resources

        # Move to first resource and collect
        world.move_agent("collector", Position(2, 2))
        world.collect_resource("collector")

        assert collector.resources.get("energy", 0) == 20

    def test_hazard_avoidance(self):
        """Test hazard avoidance behavior."""
        config = GridWorldConfig(width=5, height=5)
        world = GridWorld(config)

        # Add agent and hazards
        agent = Agent("cautious", Position(0, 0), energy=100.0)
        world.add_agent(agent)

        hazard_positions = [Position(1, 0), Position(0, 1), Position(1, 1)]
        for pos in hazard_positions:
            world.set_cell(pos, CellType.HAZARD)
            world.grid[pos.x][pos.y].value = -10.0  # Penalty

        # Agent should avoid hazards when pathfinding
        goal = Position(4, 4)
        path = world.find_path(agent.position, goal)

        # Path should not include hazard positions
        if path:  # Only check if a path was found
            for pos in path:
                assert pos not in hazard_positions
        else:
            # If no path found, that's also acceptable for hazard avoidance
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=world.grid_world"])

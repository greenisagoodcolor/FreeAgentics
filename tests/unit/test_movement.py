import unittest

import numpy as np

from agents.base.data_model import Agent, AgentStatus, Position
from agents.base.movement import (
    CollisionSystem,
    MovementConstraints,
    MovementController,
    MovementMode,
    PathfindingGrid,
    SteeringBehaviors,
    TerrainType,
)
from agents.base.state_manager import AgentStateManager


class TestMovementConstraints(unittest.TestCase):
    """Test movement constraints"""

    def test_default_constraints(self) -> None:
        """Test default movement constraints"""
        constraints = MovementConstraints()
        self.assertEqual(constraints.max_speed, 5.0)
        self.assertEqual(constraints.max_acceleration, 2.0)
        self.assertEqual(constraints.collision_radius, 0.5)
        self.assertEqual(constraints.mode_speeds[MovementMode.WALKING], 1.0)
        self.assertEqual(constraints.mode_speeds[MovementMode.RUNNING], 2.0)
        self.assertEqual(constraints.mode_speeds[MovementMode.SNEAKING], 0.5)

    def test_terrain_speeds(self) -> None:
        """Test terrain speed modifiers"""
        constraints = MovementConstraints()
        self.assertEqual(constraints.terrain_speeds[TerrainType.GROUND], 1.0)
        self.assertEqual(constraints.terrain_speeds[TerrainType.WATER], 0.3)
        self.assertEqual(constraints.terrain_speeds[TerrainType.IMPASSABLE], 0.0)


class TestCollisionSystem(unittest.TestCase):
    """Test collision detection system"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.collision_system = CollisionSystem()

    def test_static_obstacle_collision(self) -> None:
        """Test collision with static obstacles"""
        obstacle_pos = Position(5.0, 5.0, 0.0)
        self.collision_system.add_static_obstacle(obstacle_pos, 1.0)
        test_pos1 = Position(5.5, 5.0, 0.0)
        self.assertTrue(self.collision_system.check_collision(test_pos1, 0.5))
        test_pos2 = Position(7.0, 5.0, 0.0)
        self.assertFalse(self.collision_system.check_collision(test_pos2, 0.5))

    def test_dynamic_obstacle_collision(self) -> None:
        """Test collision with dynamic obstacles"""
        self.collision_system.update_dynamic_obstacle("agent1", Position(3.0, 3.0, 0.0))
        test_pos = Position(3.5, 3.0, 0.0)
        self.assertTrue(self.collision_system.check_collision(test_pos, 0.5))
        self.assertFalse(self.collision_system.check_collision(test_pos, 0.5, exclude_id="agent1"))

    def test_collision_normal(self) -> None:
        """Test collision normal calculation"""
        self.collision_system.add_static_obstacle(Position(0.0, 0.0, 0.0), 1.0)
        test_pos = Position(1.5, 0.0, 0.0)
        normal = self.collision_system.get_collision_normal(test_pos, 0.5)
        self.assertIsNotNone(normal)
        self.assertAlmostEqual(normal[0], 1.0)
        self.assertAlmostEqual(normal[1], 0.0)


class TestPathfindingGrid(unittest.TestCase):
    """Test grid-based pathfinding"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.grid = PathfindingGrid(10, 10, 1.0)

    def test_coordinate_conversion(self) -> None:
        """Test world to grid coordinate conversion"""
        world_pos = Position(2.5, 3.5, 0.0)
        grid_pos = self.grid.world_to_grid(world_pos)
        self.assertEqual(grid_pos, (2, 3))
        world_pos2 = self.grid.grid_to_world((2, 3))
        self.assertAlmostEqual(world_pos2.x, 2.5)
        self.assertAlmostEqual(world_pos2.y, 3.5)

    def test_neighbor_finding(self) -> None:
        """Test finding neighboring cells"""
        neighbors = self.grid.get_neighbors((5, 5))
        self.assertEqual(len(neighbors), 8)
        corner_neighbors = self.grid.get_neighbors((0, 0))
        self.assertEqual(len(corner_neighbors), 3)

    def test_pathfinding_simple(self) -> None:
        """Test simple pathfinding without obstacles"""
        start = Position(1.5, 1.5, 0.0)
        goal = Position(8.5, 8.5, 0.0)
        path = self.grid.find_path(start, goal)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0].x, start.x)
        self.assertEqual(path[0].y, start.y)
        self.assertAlmostEqual(path[-1].x, goal.x, delta=1.0)
        self.assertAlmostEqual(path[-1].y, goal.y, delta=1.0)

    def test_pathfinding_with_obstacles(self) -> None:
        """Test pathfinding around obstacles"""
        for y in range(3, 8):
            self.grid.set_obstacle((5, y))
        start = Position(2.5, 5.5, 0.0)
        goal = Position(7.5, 5.5, 0.0)
        path = self.grid.find_path(start, goal)
        self.assertIsNotNone(path)
        for point in path:
            grid_pos = self.grid.world_to_grid(point)
            self.assertNotIn(grid_pos, self.grid.obstacles)

    def test_no_path_exists(self) -> None:
        """Test when no path exists"""
        goal_grid = (8, 8)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    self.grid.set_obstacle((goal_grid[0] + dx, goal_grid[1] + dy))
        start = Position(1.5, 1.5, 0.0)
        goal = Position(8.5, 8.5, 0.0)
        path = self.grid.find_path(start, goal)
        self.assertIsNone(path)


class TestMovementController(unittest.TestCase):
    """Test the main movement controller"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.state_manager = AgentStateManager()
        self.collision_system = CollisionSystem()
        self.pathfinding_grid = PathfindingGrid(20, 20)
        self.movement_controller = MovementController(
            self.state_manager, self.collision_system, self.pathfinding_grid
        )
        self.agent = Agent(name="TestAgent", position=Position(5.0, 5.0, 0.0))
        self.state_manager.register_agent(self.agent)
        self.movement_controller.register_agent(self.agent)

    def test_agent_registration(self) -> None:
        """Test agent registration with movement controller"""
        self.assertIn(self.agent.agent_id, self.movement_controller.movement_states)
        self.assertIn(self.agent.agent_id, self.movement_controller.movement_constraints)
        self.assertIn(self.agent.agent_id, self.collision_system.dynamic_obstacles)

    def test_set_destination(self) -> None:
        """Test setting movement destination"""
        destination = Position(10.0, 10.0, 0.0)
        success = self.movement_controller.set_destination(self.agent.agent_id, destination)
        self.assertTrue(success)
        state = self.movement_controller.movement_states[self.agent.agent_id]
        self.assertIsNotNone(state.path)
        self.assertEqual(state.destination, destination)
        agent = self.state_manager.get_agent(self.agent.agent_id)
        self.assertEqual(agent.status, AgentStatus.MOVING)

    def test_movement_update(self) -> None:
        """Test movement update along path"""
        destination = Position(7.0, 5.0, 0.0)
        self.movement_controller.set_destination(self.agent.agent_id, destination)
        initial_pos = Position(self.agent.position.x, self.agent.position.y, self.agent.position.z)
        self.movement_controller.update(0.1)
        agent = self.state_manager.get_agent(self.agent.agent_id)
        self.assertNotEqual(agent.position.x, initial_pos.x)
        dx = agent.position.x - initial_pos.x
        self.assertGreater(dx, 0)

    def test_movement_modes(self) -> None:
        """Test different movement modes"""
        self.movement_controller.set_movement_mode(self.agent.agent_id, MovementMode.RUNNING)
        state = self.movement_controller.movement_states[self.agent.agent_id]
        self.assertEqual(state.mode, MovementMode.RUNNING)

    def test_apply_force(self) -> None:
        """Test applying external force"""
        force = np.array([5.0, 0.0, 0.0])
        self.movement_controller.apply_force(self.agent.agent_id, force)
        state = self.movement_controller.movement_states[self.agent.agent_id]
        self.assertGreater(state.velocity[0], 0)
        agent = self.state_manager.get_agent(self.agent.agent_id)
        self.assertEqual(agent.status, AgentStatus.MOVING)

    def test_jump(self) -> None:
        """Test jumping mechanics"""
        state = self.movement_controller.movement_states[self.agent.agent_id]
        state.is_grounded = True
        success = self.movement_controller.jump(self.agent.agent_id)
        self.assertTrue(success)
        self.assertFalse(state.is_grounded)
        self.assertGreater(state.velocity[2], 0)
        self.assertEqual(state.mode, MovementMode.JUMPING)
        success2 = self.movement_controller.jump(self.agent.agent_id)
        self.assertFalse(success2)

    def test_movement_info(self) -> None:
        """Test getting movement information"""
        info = self.movement_controller.get_movement_info(self.agent.agent_id)
        self.assertIsNotNone(info)
        self.assertIn("position", info)
        self.assertIn("velocity", info)
        self.assertIn("speed", info)
        self.assertIn("mode", info)
        self.assertIn("is_grounded", info)


class TestSteeringBehaviors(unittest.TestCase):
    """Test steering behaviors"""

    def test_seek_behavior(self) -> None:
        """Test seek steering behavior"""
        position = np.array([0.0, 0.0, 0.0])
        target = np.array([10.0, 0.0, 0.0])
        max_speed = 5.0
        steering = SteeringBehaviors.seek(position, target, max_speed)
        self.assertAlmostEqual(steering[0], max_speed)
        self.assertAlmostEqual(steering[1], 0.0)

    def test_flee_behavior(self) -> None:
        """Test flee steering behavior"""
        position = np.array([5.0, 5.0, 0.0])
        threat = np.array([10.0, 5.0, 0.0])
        max_speed = 5.0
        steering = SteeringBehaviors.flee(position, threat, max_speed)
        self.assertLess(steering[0], 0)
        self.assertAlmostEqual(steering[1], 0.0)

    def test_arrive_behavior(self) -> None:
        """Test arrive steering behavior"""
        position = np.array([8.0, 0.0, 0.0])
        target = np.array([10.0, 0.0, 0.0])
        max_speed = 5.0
        slowing_radius = 5.0
        steering = SteeringBehaviors.arrive(position, target, max_speed, slowing_radius)
        expected_speed = max_speed * (2.0 / slowing_radius)
        self.assertAlmostEqual(np.linalg.norm(steering), expected_speed)

    def test_wander_behavior(self) -> None:
        """Test wander steering behavior"""
        velocity = np.array([1.0, 0.0, 0.0])
        wander_angle = 0.0
        wander_rate = 0.5
        max_speed = 5.0
        steering, new_angle = SteeringBehaviors.wander(
            velocity, wander_angle, wander_rate, max_speed
        )
        self.assertAlmostEqual(np.linalg.norm(steering), max_speed)
        self.assertNotEqual(new_angle, wander_angle)

    def test_separate_behavior(self) -> None:
        """Test separation steering behavior"""
        position = np.array([5.0, 5.0, 0.0])
        neighbors = [np.array([4.0, 5.0, 0.0]), np.array([6.0, 5.0, 0.0])]
        separation_radius = 2.0
        max_speed = 5.0
        steering = SteeringBehaviors.separate(position, neighbors, separation_radius, max_speed)
        self.assertLess(np.linalg.norm(steering), 1.0)


if __name__ == "__main__":
    unittest.main()

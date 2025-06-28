"""
Module for FreeAgentics Active Inference implementation.
"""

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .data_model import Agent, AgentStatus, Position
from .state_manager import AgentStateManager

"""
Agent Movement Mechanics
This module provides movement capabilities for agents including:
- Pathfinding algorithms (A*, Dijkstra)
- Collision detection and avoidance
- Physics-based movement with velocity and acceleration
- Different movement modes (walking, running, etc.)
- Terrain interaction
"""


class MovementMode(Enum):
    """Different movement modes for agents"""

    WALKING = "walking"
    RUNNING = "running"
    SNEAKING = "sneaking"
    JUMPING = "jumping"
    FLYING = "flying"
    SWIMMING = "swimming"


class TerrainType(Enum):
    """Different terrain types that affect movement"""

    GROUND = "ground"
    WATER = "water"
    AIR = "air"
    ROUGH = "rough"
    IMPASSABLE = "impassable"


@dataclass
class MovementConstraints:
    """Constraints for agent movement"""

    max_speed: float = 5.0
    max_acceleration: float = 2.0
    max_turn_rate: float = math.pi
    collision_radius: float = 0.5
    step_height: float = 0.5
    mode_speeds: Dict[MovementMode, float] = field(
        default_factory=lambda: {
            MovementMode.WALKING: 1.0,
            MovementMode.RUNNING: 2.0,
            MovementMode.SNEAKING: 0.5,
            MovementMode.JUMPING: 1.5,
            MovementMode.FLYING: 3.0,
            MovementMode.SWIMMING: 0.8,
        }
    )
    terrain_speeds: Dict[TerrainType, float] = field(
        default_factory=lambda: {
            TerrainType.GROUND: 1.0,
            TerrainType.WATER: 0.3,
            TerrainType.AIR: 1.0,
            TerrainType.ROUGH: 0.6,
            TerrainType.IMPASSABLE: 0.0,
        }
    )


@dataclass
class MovementState:
    """Current movement state of an agent"""

    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mode: MovementMode = MovementMode.WALKING
    is_grounded: bool = True
    current_terrain: TerrainType = TerrainType.GROUND
    path: Optional[List[Position]] = None
    path_index: int = 0
    destination: Optional[Position] = None


class CollisionSystem:
    """Handles collision detection and response"""

    def __init__(self) -> None:
        self.static_obstacles: List[dict] = []
        self.dynamic_obstacles: Dict[str, Position] = {}

    def add_static_obstacle(self, position: Position, radius: float) -> None:
        """Add a static obstacle to the collision system"""
        self.static_obstacles.append({"position": position, "radius": radius})

    def update_dynamic_obstacle(self, id: str, position: Position) -> None:
        """Update position of a dynamic obstacle (other agents)"""
        self.dynamic_obstacles[id] = position

    def check_collision(
        self, position: Position, radius: float, exclude_id: Optional[str] = None
    ) -> bool:
        """Check if a position would result in collision"""
        for obstacle in self.static_obstacles:
            distance = position.distance_to(obstacle["position"])
            if distance < radius + obstacle["radius"]:
                return True
        for obs_id, obs_pos in self.dynamic_obstacles.items():
            if obs_id != exclude_id:
                distance = position.distance_to(obs_pos)
                if distance < radius * 2:
                    return True
        return False

    def get_collision_normal(self, position: Position, radius: float) -> Optional[np.ndarray]:
        """Get the normal vector of the nearest collision"""
        min_distance = float("inf")
        collision_normal = None
        for obstacle in self.static_obstacles:
            distance = position.distance_to(obstacle["position"])
            penetration = radius + obstacle["radius"] - distance
            if penetration >= 0 and distance < min_distance:
                min_distance = distance
                diff = position.to_array() - obstacle["position"].to_array()
                collision_normal = diff / np.linalg.norm(diff)
        return collision_normal


class PathfindingGrid:
    """Grid-based pathfinding for navigation"""

    def __init__(self, width: int, height: int, cell_size: float = 1.0) -> None:
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.obstacles: Set[Tuple[int, int]] = set()
        self.terrain_costs: Dict[Tuple[int, int], float] = {}

    def world_to_grid(self, position: Position) -> Tuple[int, int]:
        """Convert world position to grid coordinates"""
        x = int(position.x / self.cell_size)
        y = int(position.y / self.cell_size)
        return (x, y)

    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Position:
        """Convert grid coordinates to world position"""
        x = (grid_pos[0] + 0.5) * self.cell_size
        y = (grid_pos[1] + 0.5) * self.cell_size
        return Position(x, y, 0)

    def set_obstacle(self, grid_pos: Tuple[int, int]) -> None:
        """Mark a grid cell as obstacle"""
        self.obstacles.add(grid_pos)

    def set_terrain_cost(self, grid_pos: Tuple[int, int], cost: float) -> None:
        """Set movement cost for a grid cell"""
        self.terrain_costs[grid_pos] = cost

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x = pos[0] + dx
                new_y = pos[1] + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    new_pos = (new_x, new_y)
                    if new_pos not in self.obstacles:
                        neighbors.append(new_pos)
        return neighbors

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """Get cost of moving between two cells"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        base_cost = math.sqrt(dx * dx + dy * dy)
        terrain_mult = self.terrain_costs.get(to_pos, 1.0)
        return base_cost * terrain_mult

    def find_path(self, start: Position, goal: Position) -> Optional[List[Position]]:
        """Find path using A* algorithm"""
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)
        if start_grid in self.obstacles or goal_grid in self.obstacles:
            return None
        open_set: List[Tuple[float, Tuple[int, int]]] = [(0.0, start_grid)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start_grid: 0.0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal_grid:
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(current))
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.get_movement_cost(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None


class MovementController:
    """Main controller for agent movement"""

    def __init__(
        self,
        state_manager: AgentStateManager,
        collision_system: CollisionSystem,
        pathfinding_grid: PathfindingGrid,
    ) -> None:
        self.state_manager = state_manager
        self.collision_system = collision_system
        self.pathfinding_grid = pathfinding_grid
        self.movement_states: Dict[str, MovementState] = {}
        self.movement_constraints: Dict[str, MovementConstraints] = {}

    def register_agent(
        self, agent: Agent, constraints: Optional[MovementConstraints] = None
    ) -> None:
        """Register an agent with the movement controller"""
        self.movement_states[agent.agent_id] = MovementState()
        self.movement_constraints[agent.agent_id] = constraints or MovementConstraints()
        self.collision_system.update_dynamic_obstacle(agent.agent_id, agent.position)

    def set_destination(self, agent_id: str, destination: Position) -> bool:
        """Set movement destination for an agent"""
        agent = self.state_manager.get_agent(agent_id)
        if not agent:
            return False
        path = self.pathfinding_grid.find_path(agent.position, destination)
        if not path:
            return False
        state = self.movement_states[agent_id]
        state.path = path
        state.path_index = 0
        state.destination = destination
        self.state_manager.update_agent_status(agent_id, AgentStatus.MOVING)
        return True

    def set_movement_mode(self, agent_id: str, mode: MovementMode) -> None:
        """Change movement mode for an agent"""
        if agent_id in self.movement_states:
            self.movement_states[agent_id].mode = mode

    def update(self, delta_time: float) -> None:
        """Update all agent movements"""
        for agent_id, state in self.movement_states.items():
            agent = self.state_manager.get_agent(agent_id)
            if not agent or agent.status != AgentStatus.MOVING:
                continue
            if state.path and state.path_index < len(state.path):
                self._update_path_following(agent, state, delta_time)
            else:
                self._update_physics(agent, state, delta_time)

    def _update_path_following(self, agent: Agent, state: MovementState, delta_time: float) -> None:
        """Update agent following a path"""
        if state.path is None:
            return
        constraints = self.movement_constraints[agent.agent_id]
        target = state.path[state.path_index]
        direction = target.to_array() - agent.position.to_array()
        distance = np.linalg.norm(direction)
        if distance < 0.5:
            state.path_index += 1
            if state.path_index >= len(state.path):
                state.path = None
                state.velocity = np.zeros(3)
                self.state_manager.update_agent_status(agent.agent_id, AgentStatus.IDLE)
                return
        if distance > 0:
            direction = direction / distance
            base_speed = constraints.max_speed
            mode_mult = constraints.mode_speeds.get(state.mode, 1.0)
            terrain_mult = constraints.terrain_speeds.get(state.current_terrain, 1.0)
            desired_speed = base_speed * mode_mult * terrain_mult
            desired_velocity = direction * min(desired_speed, float(distance / delta_time))
            velocity_diff = desired_velocity - state.velocity
            max_accel = constraints.max_acceleration * delta_time
            if np.linalg.norm(velocity_diff) > max_accel:
                velocity_diff = velocity_diff / np.linalg.norm(velocity_diff) * max_accel
            state.velocity += velocity_diff
        self._apply_movement(agent, state, delta_time)

    def _update_physics(self, agent: Agent, state: MovementState, delta_time: float) -> None:
        """Update physics-based movement"""
        friction = 0.9
        state.velocity *= friction
        if not state.is_grounded and state.mode != MovementMode.FLYING:
            state.velocity[2] -= 9.8 * delta_time
        self._apply_movement(agent, state, delta_time)
        if np.linalg.norm(state.velocity) < 0.1:
            state.velocity = np.zeros(3)
            self.state_manager.update_agent_status(agent.agent_id, AgentStatus.IDLE)

    def _apply_movement(self, agent: Agent, state: MovementState, delta_time: float) -> None:
        """Apply movement with collision detection"""
        constraints = self.movement_constraints[agent.agent_id]
        movement = state.velocity * delta_time
        new_pos_array = agent.position.to_array() + movement
        new_position = Position(new_pos_array[0], new_pos_array[1], new_pos_array[2])
        if not self.collision_system.check_collision(
            new_position, constraints.collision_radius, agent.agent_id
        ):
            agent.velocity = state.velocity
            self.state_manager.update_agent_position(agent.agent_id, new_position)
            self.collision_system.update_dynamic_obstacle(agent.agent_id, new_position)
        else:
            normal = self.collision_system.get_collision_normal(
                new_position, constraints.collision_radius
            )
            if normal is not None:
                state.velocity = state.velocity - np.dot(state.velocity, normal) * normal
                movement = state.velocity * delta_time
                new_pos_array = agent.position.to_array() + movement
                new_position = Position(new_pos_array[0], new_pos_array[1], new_pos_array[2])
                if not self.collision_system.check_collision(
                    new_position, constraints.collision_radius, agent.agent_id
                ):
                    agent.velocity = state.velocity
                    self.state_manager.update_agent_position(agent.agent_id, new_position)
                    self.collision_system.update_dynamic_obstacle(agent.agent_id, new_position)
            else:
                state.velocity = np.zeros(3)

    def apply_force(self, agent_id: str, force: np.ndarray) -> None:
        """Apply external force to an agent"""
        if agent_id not in self.movement_states:
            return
        state = self.movement_states[agent_id]
        constraints = self.movement_constraints[agent_id]
        acceleration = force
        if np.linalg.norm(acceleration) > constraints.max_acceleration:
            acceleration = (
                acceleration / np.linalg.norm(acceleration) * constraints.max_acceleration
            )
        state.acceleration = acceleration
        state.velocity += acceleration
        agent = self.state_manager.get_agent(agent_id)
        if agent and agent.status == AgentStatus.IDLE:
            self.state_manager.update_agent_status(agent_id, AgentStatus.MOVING)

    def jump(self, agent_id: str, jump_force: float = 5.0) -> bool:
        """Make an agent jump"""
        if agent_id not in self.movement_states:
            return False
        state = self.movement_states[agent_id]
        if not state.is_grounded:
            return False
        state.velocity[2] = jump_force
        state.is_grounded = False
        state.mode = MovementMode.JUMPING
        return True

    def get_movement_info(self, agent_id: str) -> Optional[dict]:
        """Get movement information for an agent"""
        if agent_id not in self.movement_states:
            return None
        state = self.movement_states[agent_id]
        agent = self.state_manager.get_agent(agent_id)
        if not agent:
            return None
        destination_info = None
        if state.destination:
            destination_info = {
                "x": state.destination.x,
                "y": state.destination.y,
                "z": state.destination.z,
            }
        return {
            "position": {"x": agent.position.x, "y": agent.position.y, "z": agent.position.z},
            "velocity": state.velocity.tolist(),
            "speed": float(np.linalg.norm(state.velocity)),
            "mode": state.mode.value,
            "is_grounded": state.is_grounded,
            "has_path": state.path is not None,
            "destination": destination_info,
        }


class SteeringBehaviors:
    """Advanced steering behaviors for more natural movement"""

    @staticmethod
    def seek(position: np.ndarray, target: np.ndarray, max_speed: float) -> np.ndarray:
        """Seek steering behavior - move towards target"""
        desired = target - position
        distance = np.linalg.norm(desired)
        if distance > 0:
            desired = (desired / distance) * max_speed
        return np.array(desired)

    @staticmethod
    def flee(position: np.ndarray, threat: np.ndarray, max_speed: float) -> np.ndarray:
        """Flee steering behavior - move away from threat"""
        desired = position - threat
        distance = np.linalg.norm(desired)
        if distance > 0:
            desired = (desired / distance) * max_speed
        return np.array(desired)

    @staticmethod
    def arrive(
        position: np.ndarray, target: np.ndarray, max_speed: float, slowing_radius: float = 5.0
    ) -> np.ndarray:
        """Arrive steering behavior - slow down when approaching target"""
        desired = target - position
        distance = np.linalg.norm(desired)
        if distance > 0:
            if distance < slowing_radius:
                speed = float(max_speed * (distance / slowing_radius))
            else:
                speed = float(max_speed)
            desired = (desired / distance) * speed
        return np.array(desired)

    @staticmethod
    def wander(
        velocity: np.ndarray, wander_angle: float, wander_rate: float, max_speed: float
    ) -> tuple[np.ndarray, float]:
        """Wander steering behavior - random movement"""
        wander_angle += (np.random.random() - 0.5) * wander_rate
        if np.linalg.norm(velocity) > 0:
            current_angle = np.arctan2(velocity[1], velocity[0])
        else:
            current_angle = 0
        new_angle = current_angle + wander_angle
        wander_force = np.array([np.cos(new_angle) * max_speed, np.sin(new_angle) * max_speed, 0])
        return (wander_force, wander_angle)

    @staticmethod
    def separate(
        position: np.ndarray,
        neighbors: List[np.ndarray],
        separation_radius: float,
        max_speed: float,
    ) -> np.ndarray:
        """Separation steering behavior - avoid crowding"""
        steering = np.zeros(3)
        count = 0
        for neighbor in neighbors:
            distance = np.linalg.norm(position - neighbor)
            if 0 < distance < separation_radius:
                diff = position - neighbor
                diff = diff / distance
                diff = diff / distance
                steering += diff
                count += 1
        if count > 0:
            steering = steering / count
            if np.linalg.norm(steering) > 0:
                steering = steering / np.linalg.norm(steering) * max_speed
        return steering

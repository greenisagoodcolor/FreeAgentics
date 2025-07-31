#!/usr/bin/env python3
"""
FreeAgentics Active Inference Demo
==================================

This demo showcases multiple Active Inference agents operating in a grid world,
demonstrating key concepts like belief updates, free energy minimization, and
emergent coordination.

Run with: python demo_active_inference.py
"""

import logging
import random
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import PyMDP, but work without it
from pymdp import utils  # noqa: E402
from pymdp.agent import Agent as PyMDPAgent  # noqa: E402

PYMDP_AVAILABLE = True
logger.info("PyMDP loaded successfully")

# Try to import visualization dependencies
try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle, Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - using text-based visualization")


class CellType(Enum):
    """Types of cells in the grid world."""

    EMPTY = "empty"
    WALL = "wall"
    RESOURCE = "resource"
    DANGER = "danger"
    GOAL = "goal"


@dataclass
class Position:
    """2D position in the grid."""

    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def distance_to(self, other: "Position") -> float:
        """Manhattan distance to another position."""
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class Observation:
    """What an agent observes at its current position."""

    position: Position
    cell_type: CellType
    nearby_agents: List[str]  # IDs of nearby agents
    nearby_resources: List[Position]
    nearby_dangers: List[Position]
    timestamp: float


class GridWorld:
    """The environment where agents operate."""

    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.grid: Dict[Position, CellType] = {}
        self.resources: List[Position] = []
        self.dangers: List[Position] = []
        self.goals: List[Position] = []
        self.agents: Dict[str, "ActiveInferenceAgent"] = {}

        self._initialize_world()

    def _initialize_world(self):
        """Create an interesting world with various features."""
        # Initialize empty grid
        for x in range(self.width):
            for y in range(self.height):
                self.grid[Position(x, y)] = CellType.EMPTY

        # Add walls to create interesting topology
        self._create_walls()

        # Add resources (agents seek these)
        self._place_features(CellType.RESOURCE, 15, self.resources)

        # Add dangers (agents avoid these)
        self._place_features(CellType.DANGER, 8, self.dangers)

        # Add goals (special high-value resources)
        self._place_features(CellType.GOAL, 3, self.goals)

        logger.info(
            f"Created world: {self.width}x{self.height} with {len(self.resources)} resources, "
            f"{len(self.dangers)} dangers, {len(self.goals)} goals"
        )

    def _create_walls(self):
        """Create interesting wall patterns."""
        # Create a few rooms
        for i in range(3):
            x = random.randint(3, self.width - 8)
            y = random.randint(3, self.height - 8)
            w = random.randint(4, 7)
            h = random.randint(4, 7)

            # Create room walls with doors
            for dx in range(w):
                if dx != w // 2:  # Leave door
                    self.grid[Position(x + dx, y)] = CellType.WALL
                    self.grid[Position(x + dx, y + h - 1)] = CellType.WALL

            for dy in range(h):
                if dy != h // 2:  # Leave door
                    self.grid[Position(x, y + dy)] = CellType.WALL
                    self.grid[Position(x + w - 1, y + dy)] = CellType.WALL

    def _place_features(self, feature_type: CellType, count: int, storage: List[Position]):
        """Place features randomly in empty cells."""
        placed = 0
        attempts = 0

        while placed < count and attempts < count * 10:
            pos = Position(random.randint(0, self.width - 1), random.randint(0, self.height - 1))

            if self.grid[pos] == CellType.EMPTY:
                self.grid[pos] = feature_type
                storage.append(pos)
                placed += 1

            attempts += 1

    def get_observation(self, agent_id: str) -> Observation:
        """Get what an agent observes at its current position."""
        agent = self.agents.get(agent_id)
        if not agent:
            return None

        pos = agent.position

        # Find nearby entities
        nearby_agents = []
        nearby_resources = []
        nearby_dangers = []

        # Check surrounding cells (3x3 area)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                check_pos = Position(pos.x + dx, pos.y + dy)

                # Check bounds
                if 0 <= check_pos.x < self.width and 0 <= check_pos.y < self.height:
                    # Check for other agents
                    for other_id, other_agent in self.agents.items():
                        if other_id != agent_id and other_agent.position == check_pos:
                            nearby_agents.append(other_id)

                    # Check cell type
                    cell = self.grid.get(check_pos, CellType.EMPTY)
                    if cell == CellType.RESOURCE or cell == CellType.GOAL:
                        nearby_resources.append(check_pos)
                    elif cell == CellType.DANGER:
                        nearby_dangers.append(check_pos)

        return Observation(
            position=pos,
            cell_type=self.grid.get(pos, CellType.EMPTY),
            nearby_agents=nearby_agents,
            nearby_resources=nearby_resources,
            nearby_dangers=nearby_dangers,
            timestamp=time.time(),
        )

    def is_valid_position(self, pos: Position) -> bool:
        """Check if a position is valid for movement."""
        if not (0 <= pos.x < self.width and 0 <= pos.y < self.height):
            return False
        return self.grid.get(pos, CellType.EMPTY) != CellType.WALL

    def move_agent(self, agent_id: str, new_pos: Position) -> bool:
        """Move an agent to a new position if valid."""
        if not self.is_valid_position(new_pos):
            return False

        agent = self.agents.get(agent_id)
        if agent:
            agent.position = new_pos

            # Handle resource collection
            if self.grid[new_pos] == CellType.RESOURCE:
                agent.resources_collected += 1
                self.grid[new_pos] = CellType.EMPTY
                self.resources.remove(new_pos)
                logger.info(f"Agent {agent_id} collected resource at {new_pos}")

            # Handle goal reaching
            elif self.grid[new_pos] == CellType.GOAL:
                agent.goals_reached += 1
                logger.info(f"Agent {agent_id} reached GOAL at {new_pos}!")

            return True
        return False


class ActiveInferenceAgent:
    """An agent that uses Active Inference for decision making."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        position: Position,
        world: GridWorld,
        agent_type: str = "explorer",
    ):
        self.agent_id = agent_id
        self.name = name
        self.position = position
        self.world = world
        self.agent_type = agent_type

        # Agent stats
        self.resources_collected = 0
        self.goals_reached = 0
        self.steps_taken = 0
        self.total_free_energy = 0.0

        # Beliefs about the world (probability distributions)
        self.location_beliefs = self._initialize_location_beliefs()
        self.resource_beliefs = self._initialize_resource_beliefs()
        self.danger_beliefs = self._initialize_danger_beliefs()

        # Preferences (what the agent wants to observe)
        self.preferences = self._initialize_preferences()

        # Action history
        self.action_history: List[str] = []
        self.observation_history: List[Observation] = []

        # Visual properties
        self.color = self._get_agent_color()

        # Initialize PyMDP agent if available
        self.pymdp_agent = None
        if PYMDP_AVAILABLE and agent_type == "explorer":
            self._initialize_pymdp()

    def _get_agent_color(self) -> str:
        """Get color based on agent type."""
        colors = {"explorer": "blue", "collector": "green", "analyzer": "purple", "scout": "orange"}
        return colors.get(self.agent_type, "gray")

    def _initialize_location_beliefs(self) -> np.ndarray:
        """Initialize beliefs about own location."""
        # Start with high certainty about current position
        beliefs = np.zeros((self.world.width, self.world.height))
        beliefs[self.position.x, self.position.y] = 1.0
        return beliefs

    def _initialize_resource_beliefs(self) -> np.ndarray:
        """Initialize beliefs about resource locations."""
        # Start with uniform uncertainty
        beliefs = np.ones((self.world.width, self.world.height)) * 0.1
        return beliefs

    def _initialize_danger_beliefs(self) -> np.ndarray:
        """Initialize beliefs about danger locations."""
        # Start with low uniform danger belief
        beliefs = np.ones((self.world.width, self.world.height)) * 0.05
        return beliefs

    def _initialize_preferences(self) -> Dict[str, float]:
        """Initialize what the agent prefers to observe."""
        if self.agent_type == "explorer":
            return {
                "explore_unknown": 0.8,
                "find_resources": 0.6,
                "avoid_danger": 0.9,
                "reach_goal": 1.0,
            }
        elif self.agent_type == "collector":
            return {
                "explore_unknown": 0.4,
                "find_resources": 1.0,
                "avoid_danger": 0.7,
                "reach_goal": 0.8,
            }
        else:
            return {
                "explore_unknown": 0.5,
                "find_resources": 0.5,
                "avoid_danger": 0.8,
                "reach_goal": 0.9,
            }

    def _initialize_pymdp(self):
        """Initialize PyMDP agent for full Active Inference."""
        try:
            # Define the generative model dimensions
            num_states = [self.world.width * self.world.height]  # Location states
            num_obs = [self.world.width * self.world.height]  # Location observations
            num_controls = [5]  # Stay, North, South, East, West

            # Create simple generative model
            # A matrix: P(observation|state) - perfect observation for now
            A = np.eye(num_states[0])

            # B matrix: P(next_state|current_state, action) - transition dynamics
            B = np.zeros((num_states[0], num_states[0], num_controls[0]))

            # Fill in transition probabilities
            for x in range(self.world.width):
                for y in range(self.world.height):
                    state_idx = y * self.world.width + x
                    pos = Position(x, y)

                    # Stay action
                    B[state_idx, state_idx, 0] = 1.0

                    # Movement actions
                    moves = [
                        (0, -1, 1),  # North
                        (0, 1, 2),  # South
                        (1, 0, 3),  # East
                        (-1, 0, 4),  # West
                    ]

                    for dx, dy, action_idx in moves:
                        new_pos = Position(x + dx, y + dy)
                        if self.world.is_valid_position(new_pos):
                            new_state_idx = new_pos.y * self.world.width + new_pos.x
                            B[new_state_idx, state_idx, action_idx] = 0.9  # Some uncertainty
                            B[state_idx, state_idx, action_idx] = 0.1  # Might fail
                        else:
                            B[state_idx, state_idx, action_idx] = 1.0  # Hit wall, stay

            # C matrix: Preferences over observations
            C = np.zeros(num_states[0])

            # Assign preferences based on cell types
            for x in range(self.world.width):
                for y in range(self.world.height):
                    state_idx = y * self.world.width + x
                    pos = Position(x, y)
                    cell_type = self.world.grid.get(pos, CellType.EMPTY)

                    if cell_type == CellType.GOAL:
                        C[state_idx] = 10.0  # High reward
                    elif cell_type == CellType.RESOURCE:
                        C[state_idx] = 5.0  # Medium reward
                    elif cell_type == CellType.DANGER:
                        C[state_idx] = -10.0  # Negative reward
                    else:
                        C[state_idx] = -0.1  # Small cost for empty cells

            # Create PyMDP agent
            self.pymdp_agent = PyMDPAgent(
                A=utils.obj_array_from_numpy(A),
                B=utils.obj_array_from_numpy(B),
                C=utils.obj_array_from_numpy(C),
                action_selection="deterministic",
            )

            logger.info(f"Initialized PyMDP agent for {self.name}")

        except Exception as e:
            logger.warning(f"Failed to initialize PyMDP for {self.name}: {e}")
            self.pymdp_agent = None

    def observe(self, observation: Observation):
        """Process an observation and update beliefs."""
        self.observation_history.append(observation)

        # Update location beliefs (we know exactly where we are)
        self.location_beliefs.fill(0)
        self.location_beliefs[observation.position.x, observation.position.y] = 1.0

        # Update resource beliefs based on observation
        for resource_pos in observation.nearby_resources:
            self.resource_beliefs[resource_pos.x, resource_pos.y] = 0.9

        # Update danger beliefs
        for danger_pos in observation.nearby_dangers:
            self.danger_beliefs[danger_pos.x, danger_pos.y] = 0.9

        # Decay beliefs over time (forgetting)
        self.resource_beliefs *= 0.95
        self.danger_beliefs *= 0.95

        # Update PyMDP agent if available
        if self.pymdp_agent:
            state_idx = observation.position.y * self.world.width + observation.position.x
            self.pymdp_agent.infer_states(obs=[state_idx])

    def calculate_free_energy(self, position: Position) -> float:
        """Calculate expected free energy for a position."""
        # Simplified free energy calculation
        # F = surprise + uncertainty - expected utility

        # Surprise: how unexpected is this position given our beliefs
        surprise = -np.log(self.location_beliefs[position.x, position.y] + 1e-10)

        # Uncertainty: entropy of beliefs at this position
        uncertainty = 0.1  # Simplified

        # Expected utility: based on preferences and what we expect to find
        utility = 0.0
        cell_type = self.world.grid.get(position, CellType.EMPTY)

        if cell_type == CellType.GOAL:
            utility = self.preferences["reach_goal"] * 10
        elif cell_type == CellType.RESOURCE:
            utility = self.preferences["find_resources"] * 5
        elif cell_type == CellType.DANGER:
            utility = -self.preferences["avoid_danger"] * 10

        # Add exploration bonus for unvisited areas
        if self.location_beliefs[position.x, position.y] < 0.1:
            utility += self.preferences["explore_unknown"] * 2

        free_energy = surprise + uncertainty - utility
        return free_energy

    def select_action(self) -> str:
        """Select action using Active Inference."""
        current_pos = self.position

        # If using PyMDP
        if self.pymdp_agent:
            try:
                action_idx = self.pymdp_agent.select_action()
                actions = ["stay", "north", "south", "east", "west"]
                return actions[action_idx[0]]
            except Exception:
                pass  # Fall back to simple method

        # Simple Active Inference action selection
        possible_actions = {
            "stay": Position(current_pos.x, current_pos.y),
            "north": Position(current_pos.x, current_pos.y - 1),
            "south": Position(current_pos.x, current_pos.y + 1),
            "east": Position(current_pos.x + 1, current_pos.y),
            "west": Position(current_pos.x - 1, current_pos.y),
        }

        # Calculate expected free energy for each action
        action_values = {}

        for action, new_pos in possible_actions.items():
            if self.world.is_valid_position(new_pos):
                # Calculate expected free energy
                efe = self.calculate_free_energy(new_pos)

                # Add small random noise for exploration
                efe += random.gauss(0, 0.1)

                action_values[action] = -efe  # Minimize free energy
            else:
                action_values[action] = -1000  # Invalid action

        # Select action with highest value (lowest free energy)
        best_action = max(action_values, key=action_values.get)

        # Update total free energy
        self.total_free_energy += -action_values[best_action]

        return best_action

    def act(self, action: str) -> bool:
        """Execute an action in the world."""
        self.action_history.append(action)
        self.steps_taken += 1

        # Map action to new position
        moves = {"stay": (0, 0), "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}

        dx, dy = moves.get(action, (0, 0))
        new_pos = Position(self.position.x + dx, self.position.y + dy)

        # Try to move
        success = self.world.move_agent(self.agent_id, new_pos)

        if success and action != "stay":
            logger.debug(f"{self.name} moved {action} to {new_pos}")

        return success

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's current state."""
        return {
            "id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "position": asdict(self.position),
            "resources_collected": self.resources_collected,
            "goals_reached": self.goals_reached,
            "steps_taken": self.steps_taken,
            "avg_free_energy": self.total_free_energy / max(1, self.steps_taken),
            "last_action": self.action_history[-1] if self.action_history else None,
            "color": self.color,
        }


class ActiveInferenceDemo:
    """Main demo controller."""

    def __init__(self, world_size: int = 20, num_agents: int = 6):
        self.world = GridWorld(world_size, world_size)
        self.agents: List[ActiveInferenceAgent] = []
        self.tick = 0
        self.running = False

        # Create diverse agents
        self._create_agents(num_agents)

        # Visualization
        self.fig = None
        self.animation = None

    def _create_agents(self, num_agents: int):
        """Create a diverse set of agents."""
        agent_types = ["explorer", "collector", "analyzer", "scout"]
        names = [
            "Alpha Explorer",
            "Beta Collector",
            "Gamma Analyzer",
            "Delta Scout",
            "Epsilon Hunter",
            "Zeta Guardian",
        ]

        for i in range(num_agents):
            # Find a valid starting position
            attempts = 0
            while attempts < 100:
                pos = Position(
                    random.randint(1, self.world.width - 2),
                    random.randint(1, self.world.height - 2),
                )

                if self.world.is_valid_position(pos):
                    agent = ActiveInferenceAgent(
                        agent_id=f"agent_{i}",
                        name=names[i % len(names)],
                        position=pos,
                        world=self.world,
                        agent_type=agent_types[i % len(agent_types)],
                    )

                    self.agents.append(agent)
                    self.world.agents[agent.agent_id] = agent
                    logger.info(f"Created {agent.name} ({agent.agent_type}) at {pos}")
                    break

                attempts += 1

    def step(self):
        """Run one simulation step."""
        self.tick += 1

        # Each agent observes and acts
        for agent in self.agents:
            # Observe
            observation = self.world.get_observation(agent.agent_id)
            agent.observe(observation)

            # Select action using Active Inference
            action = agent.select_action()

            # Execute action
            agent.act(action)

        # Log summary every 10 ticks
        if self.tick % 10 == 0:
            self._log_summary()

    def _log_summary(self):
        """Log summary statistics."""
        total_resources = sum(a.resources_collected for a in self.agents)
        total_goals = sum(a.goals_reached for a in self.agents)
        avg_free_energy = np.mean(
            [a.total_free_energy / max(1, a.steps_taken) for a in self.agents]
        )

        logger.info(
            f"Tick {self.tick}: Resources collected: {total_resources}, "
            f"Goals reached: {total_goals}, Avg free energy: {avg_free_energy:.2f}"
        )

    def run_headless(self, steps: int = 100):
        """Run simulation without visualization."""
        logger.info(f"Running headless simulation for {steps} steps...")

        for _ in range(steps):
            self.step()

        # Final summary
        logger.info("Simulation complete!")
        self._print_final_summary()

    def _print_final_summary(self):
        """Print final summary of the simulation."""
        print("\n" + "=" * 60)
        print("ACTIVE INFERENCE DEMO - FINAL SUMMARY")
        print("=" * 60)

        for agent in self.agents:
            summary = agent.get_state_summary()
            print(f"\n{summary['name']} ({summary['type']}):")
            print(f"  Position: ({summary['position']['x']}, {summary['position']['y']})")
            print(f"  Resources collected: {summary['resources_collected']}")
            print(f"  Goals reached: {summary['goals_reached']}")
            print(f"  Steps taken: {summary['steps_taken']}")
            print(f"  Avg free energy: {summary['avg_free_energy']:.3f}")

        print("\nWorld state:")
        print(f"  Remaining resources: {len(self.world.resources)}")
        print(f"  Remaining goals: {len(self.world.goals)}")

    def visualize(self):
        """Create animated visualization."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - running headless")
            self.run_headless()
            return

        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[3, 1])

        # Main world view
        self.ax_world = self.fig.add_subplot(gs[0, :])
        self.ax_world.set_title("Active Inference Multi-Agent World", fontsize=16)

        # Statistics
        self.ax_stats = self.fig.add_subplot(gs[1, 0])
        self.ax_stats.set_title("Agent Performance", fontsize=12)

        # Free energy
        self.ax_energy = self.fig.add_subplot(gs[1, 1])
        self.ax_energy.set_title("Free Energy Over Time", fontsize=12)

        # Initialize plots
        self._init_visualization()

        # Create animation
        self.animation = animation.FuncAnimation(self.fig, self._animate, interval=200, blit=False)

        plt.tight_layout()
        plt.show()

    def _init_visualization(self):
        """Initialize visualization elements."""
        # World grid
        self.ax_world.set_xlim(-0.5, self.world.width - 0.5)
        self.ax_world.set_ylim(-0.5, self.world.height - 0.5)
        self.ax_world.set_aspect("equal")
        self.ax_world.grid(True, alpha=0.3)

        # Draw grid
        for x in range(self.world.width):
            for y in range(self.world.height):
                pos = Position(x, y)
                cell_type = self.world.grid.get(pos, CellType.EMPTY)

                if cell_type == CellType.WALL:
                    rect = Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="black", edgecolor="gray")
                    self.ax_world.add_patch(rect)
                elif cell_type == CellType.RESOURCE:
                    circle = Circle((x, y), 0.3, color="gold", alpha=0.7)
                    self.ax_world.add_patch(circle)
                elif cell_type == CellType.DANGER:
                    rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, facecolor="red", alpha=0.5)
                    self.ax_world.add_patch(rect)
                elif cell_type == CellType.GOAL:
                    circle = Circle((x, y), 0.4, color="lime", edgecolor="darkgreen", linewidth=2)
                    self.ax_world.add_patch(circle)

        # Agent markers
        self.agent_markers = {}
        for agent in self.agents:
            marker = Circle(
                (agent.position.x, agent.position.y),
                0.25,
                color=agent.color,
                edgecolor="black",
                linewidth=2,
            )
            self.ax_world.add_patch(marker)
            self.agent_markers[agent.agent_id] = marker

        # Initialize stats
        self.energy_history = {agent.agent_id: [] for agent in self.agents}

    def _animate(self, frame):
        """Animation update function."""
        # Run simulation step
        self.step()

        # Update agent positions
        for agent in self.agents:
            marker = self.agent_markers[agent.agent_id]
            marker.center = (agent.position.x, agent.position.y)

            # Update energy history
            avg_energy = agent.total_free_energy / max(1, agent.steps_taken)
            self.energy_history[agent.agent_id].append(avg_energy)

        # Update statistics
        self._update_stats_plot()
        self._update_energy_plot()

        # Update title
        total_resources = sum(a.resources_collected for a in self.agents)
        total_goals = sum(a.goals_reached for a in self.agents)
        self.ax_world.set_title(
            f"Active Inference Demo - Tick {self.tick} | "
            f"Resources: {total_resources} | Goals: {total_goals}",
            fontsize=16,
        )

        return list(self.agent_markers.values())

    def _update_stats_plot(self):
        """Update agent statistics plot."""
        self.ax_stats.clear()

        names = [a.name.split()[0] for a in self.agents]  # First word of name
        resources = [a.resources_collected for a in self.agents]
        goals = [a.goals_reached for a in self.agents]

        x = np.arange(len(names))
        width = 0.35

        self.ax_stats.bar(x - width / 2, resources, width, label="Resources", color="gold")
        self.ax_stats.bar(x + width / 2, goals, width, label="Goals", color="lime")

        self.ax_stats.set_xlabel("Agents")
        self.ax_stats.set_ylabel("Count")
        self.ax_stats.set_xticks(x)
        self.ax_stats.set_xticklabels(names, rotation=45)
        self.ax_stats.legend()
        self.ax_stats.set_title("Agent Performance")

    def _update_energy_plot(self):
        """Update free energy plot."""
        self.ax_energy.clear()

        for agent in self.agents:
            history = self.energy_history[agent.agent_id]
            if history:
                self.ax_energy.plot(
                    history[-50:], label=agent.name.split()[0], color=agent.color, linewidth=2
                )

        self.ax_energy.set_xlabel("Time")
        self.ax_energy.set_ylabel("Average Free Energy")
        self.ax_energy.legend(loc="upper right", fontsize=8)
        self.ax_energy.set_title("Free Energy Over Time")
        self.ax_energy.grid(True, alpha=0.3)


def main():
    """Run the Active Inference demo."""
    print("\n" + "=" * 60)
    print("🧠 FREEAGENTICS ACTIVE INFERENCE DEMO 🤖")
    print("=" * 60)
    print("\nThis demo shows multiple Active Inference agents:")
    print("- Exploring an unknown environment")
    print("- Collecting resources while avoiding dangers")
    print("- Minimizing free energy through active inference")
    print("- Emergent coordination without explicit communication")
    print("\nAgent Types:")
    print("- 🔵 Explorer: High exploration drive")
    print("- 🟢 Collector: Focuses on resources")
    print("- 🟣 Analyzer: Balanced approach")
    print("- 🟠 Scout: Fast movement\n")

    # Create and run demo
    demo = ActiveInferenceDemo(world_size=20, num_agents=6)

    # Check if we can visualize
    if MATPLOTLIB_AVAILABLE:
        print("Starting visual demo...")
        print("Close the window to end the simulation.\n")
        demo.visualize()
    else:
        print("Running headless demo (install matplotlib for visualization)...")
        demo.run_headless(steps=100)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Active Inference Demo with PyMDP.
Demonstrates real Active Inference agents using inferactively-pymdp
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np

from agents.base_agent import BasicExplorerAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ActiveInferenceWorld:
    """A grid world for Active Inference agents."""

    def __init__(self, width: int = 8, height: int = 8):
        """Initialize the Active Inference world."""
        self.width = width
        self.height = height
        self.agents: Dict[str, BasicExplorerAgent] = {}
        self.resources = self._place_resources()
        self.obstacles = self._place_obstacles()
        self.time_step = 0

    def _place_resources(self) -> List[Tuple[int, int]]:
        """Place goal resources in the world."""
        resources = [
            (2, 2),
            (5, 2),
            (2, 5),
            (5, 5),  # Corners of inner square
            (3, 3),
            (4, 4),  # Center area
        ]
        return resources

    def _place_obstacles(self) -> List[Tuple[int, int]]:
        """Place obstacles in the world."""
        obstacles = [
            (1, 3),
            (3, 1),
            (6, 3),
            (3, 6),
            (0, 0),
            (7, 7),
        ]  # Cross pattern  # Corners
        return obstacles

    def add_agent(self, agent: BasicExplorerAgent, start_pos: Tuple[int, int]):
        """Add an Active Inference agent to the world."""
        self.agents[agent.agent_id] = agent
        agent.position = list(start_pos)
        agent.start()
        logger.info(f"Added Active Inference agent {agent.name} at position {start_pos}")

    def get_observation(self, agent: BasicExplorerAgent) -> Dict:
        """Generate observation for the agent based on its position."""
        x, y = agent.position

        # Create 3x3 surroundings matrix
        surroundings = np.zeros((3, 3))

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy

                # Check bounds
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    surroundings[dx + 1, dy + 1] = -2  # Out of bounds
                elif (nx, ny) in self.obstacles:
                    surroundings[dx + 1, dy + 1] = -1  # Obstacle
                elif (nx, ny) in self.resources:
                    surroundings[dx + 1, dy + 1] = 1  # Goal/resource
                else:
                    surroundings[dx + 1, dy + 1] = 0  # Empty

        # Check if another agent is nearby
        for other_id, other_agent in self.agents.items():
            if other_id != agent.agent_id:
                ox, oy = other_agent.position
                if abs(ox - x) <= 1 and abs(oy - y) <= 1:
                    surroundings[ox - x + 1, oy - y + 1] = 2  # Other agent

        return {
            "position": agent.position.copy(),
            "surroundings": surroundings,
            "time_step": self.time_step,
        }

    def execute_action(self, agent: BasicExplorerAgent, action: str) -> bool:
        """Execute the agent's chosen action."""
        x, y = agent.position
        new_x, new_y = x, y

        # Map action to movement
        if action == "up":
            new_y -= 1
        elif action == "down":
            new_y += 1
        elif action == "left":
            new_x -= 1
        elif action == "right":
            new_x += 1
        elif action == "stay":
            return True

        # Check if new position is valid
        if (
            0 <= new_x < self.width
            and 0 <= new_y < self.height
            and (new_x, new_y) not in self.obstacles
        ):
            # Check if another agent is there
            occupied = False
            for other_agent in self.agents.values():
                if other_agent.agent_id != agent.agent_id:
                    if other_agent.position == [new_x, new_y]:
                        occupied = True
                        break

            if not occupied:
                agent.position = [new_x, new_y]
                return True

        return False

    def step(self):
        """Execute one time step of Active Inference."""
        self.time_step += 1
        logger.info(f"\n=== Active Inference Time Step {self.time_step} ===")

        for agent in self.agents.values():
            # Get observation
            observation = self.get_observation(agent)

            # Agent performs Active Inference step
            action = agent.step(observation)

            # Execute action in world
            success = self.execute_action(agent, action)

            # Check for resource collection
            pos_tuple = tuple(agent.position)
            if pos_tuple in self.resources:
                self.resources.remove(pos_tuple)
                logger.info(f"{agent.name} reached goal at {pos_tuple}!")
                # Update agent's preferences (reward signal)
                if hasattr(agent, "known_goals"):
                    agent.known_goals = set()
                else:
                    agent.known_goals = set()
                agent.known_goals.add(pos_tuple)

            # Log agent status with Active Inference metrics
            metrics_str = ""
            if "total_free_energy" in agent.metrics:
                metrics_str = f", F={agent.metrics['total_free_energy']:.3f}"
            if "belief_entropy" in agent.metrics:
                metrics_str += f", H={agent.metrics['belief_entropy']:.3f}"

            logger.info(
                f"{agent.name}: pos={agent.position}, action={action}, "
                f"success={success}{metrics_str}"
            )

    def display(self):
        """Display the world state with Active Inference visualization."""
        print("\n" + "=" * (self.width * 3 + 1))

        # Create display grid
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Place obstacles
        for x, y in self.obstacles:
            grid[y][x] = "#"

        # Place resources
        for x, y in self.resources:
            grid[y][x] = "G"

        # Place agents with their IDs
        for i, agent in enumerate(self.agents.values()):
            x, y = agent.position
            grid[y][x] = str(i + 1)

        # Print grid
        for row in grid:
            print(" ".join(row))

        print("=" * (self.width * 3 + 1))
        print("Legend: 1,2=Agents, G=Goal, #=Obstacle, .=Empty")

        # Print Active Inference metrics
        print("\nActive Inference Metrics:")
        for agent in self.agents.values():
            print(f"{agent.name}:")
            if "total_free_energy" in agent.metrics:
                print(f"  Free Energy: {agent.metrics['total_free_energy']:.3f}")
                print(f"  Accuracy: {agent.metrics.get('accuracy', 0):.3f}")
                print(f"  Complexity: {agent.metrics.get('complexity', 0):.3f}")
            print(f"  Belief Entropy: {agent.metrics.get('belief_entropy', 0):.3f}")


def run_active_inference_demo():
    """Run the Active Inference demonstration."""
    print("\n" + "=" * 60)
    print("FreeAgentics Active Inference Demo (with PyMDP)")
    print("=" * 60)
    print("\nThis demonstrates real Active Inference agents that:")
    print("- Use variational inference to update beliefs")
    print("- Minimize expected free energy for action selection")
    print("- Balance exploration (epistemic value) and exploitation (pragmatic value)")
    print("- Track free energy decomposition (accuracy vs complexity)")

    # Create world
    world = ActiveInferenceWorld(width=8, height=8)

    # Create Active Inference agents
    agent1 = BasicExplorerAgent("ai_agent_1", "AI Explorer Alpha", grid_size=8)
    agent2 = BasicExplorerAgent("ai_agent_2", "AI Explorer Beta", grid_size=8)

    # Configure agents for better demonstration
    agent1.exploration_rate = 0.1  # More goal-directed
    agent2.exploration_rate = 0.3  # More exploratory

    # Add agents to world at different starting positions
    world.add_agent(agent1, (1, 1))
    world.add_agent(agent2, (6, 6))

    # Display initial state
    print("\nInitial World State:")
    world.display()
    time.sleep(2)

    # Run simulation
    num_steps = 15
    print(f"\nRunning Active Inference simulation for {num_steps} steps...")
    print("Watch how agents balance exploration and goal-seeking...\n")

    for i in range(num_steps):
        world.step()

        # Display world every 3 steps
        if (i + 1) % 3 == 0:
            world.display()
            time.sleep(1.5)

    # Final statistics
    print("\n" + "=" * 60)
    print("Active Inference Simulation Complete!")
    print("=" * 60)

    for agent in world.agents.values():
        print(f"\n{agent.name} Final Statistics:")
        print(f"  Position: {agent.position}")
        print(f"  Total Steps: {agent.total_steps}")

        if agent.pymdp_agent:
            print("\n  PyMDP Active Inference Metrics:")
            print(f"    Free Energy: {agent.metrics.get('total_free_energy', 0):.3f}")
            print(f"    Accuracy: {agent.metrics.get('accuracy', 0):.3f}")
            print(f"    Complexity: {agent.metrics.get('complexity', 0):.3f}")
            print(f"    Belief Entropy: {agent.metrics.get('belief_entropy', 0):.3f}")
            print(f"    Expected Free Energy: {agent.metrics.get('expected_free_energy', 0):.3f}")

            # Show belief distribution if available
            if "state_posterior" in agent.beliefs:
                posterior = np.array(agent.beliefs["state_posterior"][0])
                max_belief_idx = np.argmax(posterior)
                max_belief_prob = posterior[max_belief_idx]
                believed_x = max_belief_idx // 8
                believed_y = max_belief_idx % 8
                print(
                    f"    Strongest Belief: Position ({believed_x}, {believed_y}) with P={max_belief_prob:.3f}"
                )

    print(f"\nGoals Remaining: {len(world.resources)}")
    print(f"Goals Collected: {6 - len(world.resources)}")
    print("\n" + "=" * 60)
    print("This demo shows how Active Inference agents use PyMDP to:")
    print("- Maintain probabilistic beliefs about their location")
    print("- Select actions that minimize expected free energy")
    print("- Balance information-seeking and goal-seeking behavior")
    print("=" * 60)


if __name__ == "__main__":
    run_active_inference_demo()

#!/usr/bin/env python3
"""
Simple FreeAgentics Demo
Demonstrates basic functionality without complex dependencies
"""

import logging
import random
import time
from typing import List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleAgent:
    """A simple agent that explores a grid world."""

    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.position = (0, 0)
        self.visited = set()
        self.energy = 100

    def perceive(self, world_state: dict) -> dict:
        """Perceive the world around the agent."""
        x, y = self.position
        observation = {
            "position": self.position,
            "energy": self.energy,
            "nearby_cells": world_state.get("nearby_cells", []),
            "visited_count": len(self.visited),
        }
        return observation

    def decide_action(self, observation: dict) -> str:
        """Decide on an action based on observation."""
        # Simple exploration strategy
        possible_actions = ["north", "south", "east", "west", "rest"]

        # Prefer unvisited directions
        x, y = self.position
        unvisited_directions = []

        if (x, y - 1) not in self.visited:
            unvisited_directions.append("north")
        if (x, y + 1) not in self.visited:
            unvisited_directions.append("south")
        if (x + 1, y) not in self.visited:
            unvisited_directions.append("east")
        if (x - 1, y) not in self.visited:
            unvisited_directions.append("west")

        # Choose action
        if self.energy < 20:
            return "rest"
        elif unvisited_directions:
            return random.choice(unvisited_directions)
        else:
            return random.choice(possible_actions)

    def execute_action(self, action: str, world_bounds: Tuple[int, int]) -> bool:
        """Execute the chosen action."""
        x, y = self.position
        new_x, new_y = x, y

        if action == "north":
            new_y = y - 1
        elif action == "south":
            new_y = y + 1
        elif action == "east":
            new_x = x + 1
        elif action == "west":
            new_x = x - 1
        elif action == "rest":
            self.energy = min(100, self.energy + 20)
            return True

        # Check bounds
        max_x, max_y = world_bounds
        if 0 <= new_x < max_x and 0 <= new_y < max_y:
            self.position = (new_x, new_y)
            self.visited.add(self.position)
            self.energy -= 5
            return True
        return False


class SimpleWorld:
    """A simple grid world for agents to explore."""

    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.agents = {}
        self.resources = self._place_resources()
        self.time_step = 0

    def _place_resources(self) -> List[Tuple[int, int]]:
        """Place resources randomly in the world."""
        num_resources = (self.width * self.height) // 10
        resources = []
        for _ in range(num_resources):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            resources.append((x, y))
        return resources

    def add_agent(self, agent: SimpleAgent):
        """Add an agent to the world."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.name} to the world")

    def get_world_state(self, agent_position: Tuple[int, int]) -> dict:
        """Get the world state from an agent's perspective."""
        x, y = agent_position
        nearby_cells = []

        # Check adjacent cells
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                cell_info = {"position": (nx, ny), "has_resource": (nx, ny) in self.resources}
                nearby_cells.append(cell_info)

        return {"nearby_cells": nearby_cells, "time_step": self.time_step}

    def step(self):
        """Execute one time step of the simulation."""
        self.time_step += 1
        logger.info(f"\n=== Time Step {self.time_step} ===")

        for agent in self.agents.values():
            # Perceive
            world_state = self.get_world_state(agent.position)
            observation = agent.perceive(world_state)

            # Decide
            action = agent.decide_action(observation)

            # Act
            success = agent.execute_action(action, (self.width, self.height))

            # Check for resource collection
            if agent.position in self.resources:
                self.resources.remove(agent.position)
                agent.energy = min(100, agent.energy + 30)
                logger.info(f"{agent.name} collected a resource at {agent.position}!")

            # Log agent status
            logger.info(
                f"{agent.name}: pos={agent.position}, action={action}, "
                f"energy={agent.energy}, visited={len(agent.visited)}"
            )

    def display(self):
        """Display the world state."""
        print("\n" + "=" * (self.width * 2 + 1))
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                # Check what's at this position
                agent_here = False
                for agent in self.agents.values():
                    if agent.position == (x, y):
                        row += "A "
                        agent_here = True
                        break

                if not agent_here:
                    if (x, y) in self.resources:
                        row += "R "
                    else:
                        row += ". "
            print(row)
        print("=" * (self.width * 2 + 1))
        print("Legend: A=Agent, R=Resource, .=Empty")


def run_demo():
    """Run the simple demo."""
    print("\n" + "=" * 60)
    print("FreeAgentics Simple Demo")
    print("=" * 60)

    # Create world
    world = SimpleWorld(width=8, height=8)

    # Create agents
    agent1 = SimpleAgent("agent_1", "Explorer Alpha")
    agent2 = SimpleAgent("agent_2", "Explorer Beta")

    # Add agents to world
    world.add_agent(agent1)
    agent2.position = (7, 7)  # Start Beta at opposite corner
    world.add_agent(agent2)

    # Run simulation
    num_steps = 20
    print(f"\nRunning simulation for {num_steps} steps...")

    for i in range(num_steps):
        world.step()

        # Display world every 5 steps
        if (i + 1) % 5 == 0:
            world.display()
            time.sleep(1)  # Pause for visibility

    # Final statistics
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)

    for agent in world.agents.values():
        coverage = (len(agent.visited) / (world.width * world.height)) * 100
        print(f"\n{agent.name}:")
        print(f"  Final Position: {agent.position}")
        print(f"  Energy: {agent.energy}")
        print(f"  Cells Visited: {len(agent.visited)}")
        print(f"  Coverage: {coverage:.1f}%")

    print(f"\nResources Remaining: {len(world.resources)}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()

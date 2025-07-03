#!/usr/bin/env python3
"""
Basic Active Inference Example using PyMDP

This script demonstrates the core concepts of Active Inference using a simple
grid world environment. An agent uses beliefs to navigate and find rewards
while minimizing free energy using the validated PyMDP library.

Usage:
    python docs/examples/active_inference_basic_example.py
"""

from typing import List

import matplotlib.pyplot as plt
import torch

from inference.engine import (
    DiscreteGenerativeModel,
    InferenceConfig,
    ModelDimensions,
    ModelParameters,
    PolicyConfig,
    VariationalMessagePassing,
)
from inference.engine.pymdp_policy_selector import PyMDPPolicySelector


class SimpleGridWorld:
    """Simple 2D grid world environment for Active Inference demonstration"""

    def __init__(self, size: int = 4) -> None:
        self.size = size
        self.num_states = size * size
        self.num_actions = 4  # up, down, left, right
        self.num_observations = 3  # empty, wall, reward

        # Agent position
        self.agent_pos = 0  # Start at top-left

        # Reward positions (last cell)
        self.reward_pos = self.num_states - 1

        # Wall positions (middle obstacles)
        self.walls = {5, 6, 9, 10}  # Some walls in 4x4 grid

    def get_observation(self) -> int:
        """Get observation at current position"""
        if self.agent_pos in self.walls:
            return 1  # wall
        elif self.agent_pos == self.reward_pos:
            return 2  # reward
        else:
            return 0  # empty

    def step(self, action: int) -> tuple[int, bool]:
        """Take action and return (observation, done)"""
        # Convert action to movement
        row, col = divmod(self.agent_pos, self.size)

        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)

        new_pos = row * self.size + col

        # Don't move into walls
        if new_pos not in self.walls:
            self.agent_pos = new_pos

        observation = self.get_observation()
        done = self.agent_pos == self.reward_pos

        return observation, done

    def reset(self) -> None:
        """Reset environment"""
        self.agent_pos = 0
        return self.get_observation()


def create_grid_world_model(env: SimpleGridWorld) -> DiscreteGenerativeModel:
    """Create Active Inference model for grid world"""

    dims = ModelDimensions(
        num_states=env.num_states,
        num_observations=env.num_observations,
        num_actions=env.num_actions,
    )

    params = ModelParameters(use_gpu=torch.cuda.is_available())
    model = DiscreteGenerativeModel(dims, params)

    # Set up observation model (A matrix)
    # Each state deterministically generates its observation
    A = torch.zeros(env.num_observations, env.num_states)
    for state in range(env.num_states):
        if state in env.walls:
            A[1, state] = 1.0  # wall observation
        elif state == env.reward_pos:
            A[2, state] = 1.0  # reward observation
        else:
            A[0, state] = 1.0  # empty observation

    model.A = A

    # Set up transition model (B tensor)
    # Actions deterministically move the agent
    B = torch.zeros(env.num_states, env.num_states, env.num_actions)

    for state in range(env.num_states):
        row, col = divmod(state, env.size)

        for action in range(env.num_actions):
            new_row, new_col = row, col

            if action == 0:  # up
                new_row = max(0, row - 1)
            elif action == 1:  # down
                new_row = min(env.size - 1, row + 1)
            elif action == 2:  # left
                new_col = max(0, col - 1)
            elif action == 3:  # right
                new_col = min(env.size - 1, col + 1)

            new_state = new_row * env.size + new_col

            # Don't transition into walls
            if new_state in env.walls:
                new_state = state

            B[new_state, state, action] = 1.0

    model.B = B

    # Set up preferences (C matrix)
    # Agent prefers reward observations
    C = torch.zeros(env.num_observations, 1)
    C[2, 0] = 2.0  # High preference for reward
    C[1, 0] = -1.0  # Avoid walls
    C[0, 0] = 0.0  # Neutral about empty spaces

    model.C = C

    # Set up initial state prior (D vector)
    # Agent knows it starts at position 0
    D = torch.zeros(env.num_states)
    D[0] = 1.0

    model.D = D

    return model


def run_active_inference_episode(
    env: SimpleGridWorld, model: DiscreteGenerativeModel, max_steps: int = 20
) -> tuple[list[int], list[torch.Tensor], list[int]]:
    """Run one episode of Active Inference navigation"""

    # Set up inference
    inference_config = InferenceConfig(num_iterations=10, convergence_threshold=1e-4)
    inference = VariationalMessagePassing(inference_config)

    # Set up policy selection
    policy_config = PolicyConfig(precision=2.0, planning_horizon=3, exploration_bonus=0.1)
    policy_selector = PyMDPPolicySelector(policy_config)

    # Initialize
    observation = env.reset()
    belief = model.D.clone()  # Start with initial prior

    observations = []
    beliefs = []
    actions = []

    for step in range(max_steps):
        observations.append(observation)
        beliefs.append(belief.clone())

        # Update beliefs based on observation
        obs_tensor = torch.tensor(observation, dtype=torch.long)
        belief = inference.infer_states(obs_tensor, model, belief)

        # Select action using expected free energy
        action_probs = policy_selector.evaluate_policies(belief, model)
        action = policy_selector.select_action(action_probs)
        actions.append(action)

        print(
            f"Step {step}: Pos={env.agent_pos}, Obs={observation}, "
            f"Action={action}, Belief_entropy={-torch.sum(belief * torch.log(belief + 1e-8)):.3f}"
        )

        # Take action in environment
        observation, done = env.step(action)

        if done:
            print(f"Reached goal in {step + 1} steps!")
            break

    return observations, beliefs, actions


def visualize_beliefs(beliefs: List[torch.Tensor], env: SimpleGridWorld):
    """Visualize belief evolution over time"""

    len(beliefs)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Show beliefs at different time steps
    steps_to_show = [0, 1, 2, len(beliefs) // 2, len(beliefs) - 2, len(beliefs) - 1]

    for i, step in enumerate(steps_to_show):
        if step >= len(beliefs):
            continue

        ax = axes[i]
        belief_grid = beliefs[step].reshape(env.size, env.size)

        im = ax.imshow(belief_grid.numpy(), cmap="hot", interpolation="nearest")
        ax.set_title(f"Belief at Step {step}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Mark walls and reward
        for state in env.walls:
            row, col = divmod(state, env.size)
            ax.plot(col, row, "bx", markersize=10, markeredgewidth=2)

        reward_row, reward_col = divmod(env.reward_pos, env.size)
        ax.plot(reward_col, reward_row, "g*", markersize=15)

    plt.tight_layout()
    plt.suptitle("Active Inference Belief Evolution", y=1.02)
    plt.show()


def plot_free_energy(
    beliefs: List[torch.Tensor], observations: List[int], model: DiscreteGenerativeModel
):
    """Plot free energy over time"""

    inference = VariationalMessagePassing(InferenceConfig())
    free_energies = []

    for belief, obs in zip(beliefs, observations):
        obs_tensor = torch.tensor(obs, dtype=torch.long)
        fe = inference.compute_free_energy(belief, obs_tensor, model)
        free_energies.append(fe.item())

    plt.figure(figsize=(10, 6))
    plt.plot(free_energies, "b-", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Variational Free Energy")
    plt.title("Free Energy Minimization During Navigation")
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    """Main function demonstrating Active Inference"""

    print("=== Active Inference Grid World Example ===\n")

    # Create environment
    env = SimpleGridWorld(size=4)
    print(f"Grid World: {env.size}x{env.size}")
    print("Agent starts at position 0 (top-left)")
    print(f"Goal is at position {env.reward_pos} (bottom-right)")
    print(f"Walls at positions: {env.walls}\n")

    # Create Active Inference model
    model = create_grid_world_model(env)
    print("Created Active Inference model with:")
    print(f"- {model.A.shape[1]} states")
    print(f"- {model.A.shape[0]} observations")
    print(f"- {model.B.shape[2]} actions\n")

    # Run episode
    print("Running Active Inference episode...\n")
    observations, beliefs, actions = run_active_inference_episode(env, model)

    # Analysis
    print(f"\nEpisode completed with {len(actions)} actions taken")
    print(f"Final position: {env.agent_pos}")
    print(f"Success: {'Yes' if env.agent_pos == env.reward_pos else 'No'}")

    # Show action sequence
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    action_sequence = [action_names[a] for a in actions]
    print(f"Action sequence: {' -> '.join(action_sequence)}")

    # Visualizations
    print("\nGenerating visualizations...")
    visualize_beliefs(beliefs, env)
    plot_free_energy(beliefs, observations, model)

    # Compute final metrics
    final_belief = beliefs[-1]
    belief_entropy = -torch.sum(final_belief * torch.log(final_belief + 1e-8))
    print(f"\nFinal belief entropy: {belief_entropy:.3f}")
    print(f"Most likely state: {torch.argmax(final_belief).item()}")
    print(f"Actual state: {env.agent_pos}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
."""
Active Inference with Graph Neural Network Integration Example

This script demonstrates how to integrate Graph Neural Networks (GNNs) with
Active Inference for agents operating in graph-structured environments.
The agent uses GNN features to update beliefs and make decisions.

Usage:
    python docs/examples/active_inference_gnn_example.py
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from inference.engine import (
    DiscreteExpectedFreeEnergy,
    DiscreteGenerativeModel,
    GNNActiveInferenceAdapter,
    GNNIntegrationConfig,
    InferenceConfig,
    ModelDimensions,
    ModelParameters,
    PolicyConfig,
    VariationalMessagePassing,
)
from inference.engine.pymdp_policy_selector import PyMDPPolicySelector


class GraphEnvironment:
    """Graph-based environment where agent navigates between nodes."""

    def __init__(self, num_nodes: int = 10, edge_prob: float = 0.3) -> None:
        """Initialize."""
        self.num_nodes = num_nodes
        self.current_node = 0
        self.goal_node = num_nodes - 1

        # Generate random graph
        self.graph = self._generate_graph(num_nodes, edge_prob)
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).todense()

        # Node features (random for demonstration)
        self.node_features = (
            torch.randn(num_nodes, 4)  # 4-dimensional features)

        # Mark goal node with special feature
        self.node_features[self.goal_node, -1] = 2.0  # Goal marker

    def _generate_graph(self, num_nodes: int, edge_prob: float) -> nx.Graph:
        """Generate a connected random graph."""
        while True:
            G = nx.erdos_renyi_graph(num_nodes, edge_prob)
            if nx.is_connected(G):
                return G

    def get_graph_observation(self) -> Data:
        """Get current graph observation as PyTorch Geometric Data object."""
        # Create edge index
        edges = list(self.graph.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Add self-loops and make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Node features with current position marked
        node_features = self.node_features.clone()
        node_features[self.current_node, 0] += 1.0  # Mark current position

        return Data(x=node_features, edge_index=edge_index)

    def get_neighbors(self) -> List[int]:
        """Get neighboring nodes of current position."""
        return list(self.graph.neighbors(self.current_node))

    def step(self, target_node: int) -> tuple[int, bool]:
        """Move to target node if it's a neighbor."""
        if target_node in self.get_neighbors():
            self.current_node = target_node

        # Observation is current node index
        observation = self.current_node
        done = self.current_node == self.goal_node

        return observation, done

    def reset(self) -> None:
        """Reset environment."""
        self.current_node = 0
        return self.current_node

    def visualize(self, agent_path: Optional[List[int]] = None):
        """Visualize the graph with agent path."""
        plt.figure(figsize=(10, 8))

        pos = nx.spring_layout(self.graph, seed=42)

        # Draw nodes
        node_colors = ["lightblue"] * self.num_nodes
        node_colors[0] = "green"  # Start
        node_colors[self.goal_node] = "red"  # Goal
        if agent_path:
            node_colors[self.current_node] = "orange"  # Current position

        nx.draw(
            self.graph, pos, node_color= (
                node_colors, with_labels=True, node_size=500, font_size=10)
        )

        # Draw agent path if provided
        if agent_path and len(agent_path) > 1:
            path_edges = (
                [(agent_path[i], agent_path[i + 1]) for i in range(len(agent_path) - 1)])
            nx.draw_networkx_edges(
                self.graph, pos, edgelist= (
                    path_edges, edge_color="red", width=3, alpha=0.7)
            )

        plt.title("Graph Environment\n(Green=Start, Red=Goal, Orange=Current)")
        plt.axis("off")
        plt.show()


class GraphGNN(nn.Module):
    """Simple Graph Neural Network for processing graph observations."""

    def __init__(self, input_dim: int = (
        4, hidden_dim: int = 32, output_dim: int = 16) -> None:)
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through GNN."""
        # Node-level processing
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)

        # Global graph representation
        if batch is None:
            # Single graph
            graph_repr = torch.mean(x, dim=0, keepdim=True)
        else:
            # Multiple graphs in batch
            graph_repr = global_mean_pool(x, batch)

        return graph_repr


class GraphActiveInferenceAgent:
    """Active Inference agent that processes graph observations via GNN."""

    def __init__(self, env: GraphEnvironment) -> None:
        self.env = env

        # Set up GNN
        self.gnn = GraphGNN(input_dim=4, hidden_dim=32, output_dim=16)
        self.gnn_config = GNNIntegrationConfig(
            gnn_hidden_dim=32, num_gnn_layers=3, aggregation_method="mean"
        )

        # Set up Active Inference model
        self.model = self._create_model()

        # Set up inference
        self.inference_config = (
            InferenceConfig(num_iterations=10, convergence_threshold=1e-4))
        self.inference = VariationalMessagePassing(self.inference_config)

        # Set up policy selection
        self.policy_config = (
            PolicyConfig(precision=2.0, planning_horizon=2, exploration_bonus=0.2))
        self.policy_selector = PyMDPPolicySelector(self.policy_config)

        # GNN-Active Inference adapter
        self.gnn_adapter = GNNActiveInferenceAdapter(self.gnn, self.gnn_config)

        # Current belief state
        self.belief = self.model.D.clone()

    def _create_model(self) -> DiscreteGenerativeModel:
        """Create generative model for graph navigation."""

        dims = ModelDimensions(
            num_states=self.env.num_nodes,
            num_observations=self.env.num_nodes,  # Observe current node
            num_actions= (
                self.env.num_nodes,  # Move to any node (filtered by environment))
        )

        params = ModelParameters(use_gpu=torch.cuda.is_available())
        model = DiscreteGenerativeModel(dims, params)

        # Observation model: deterministic observation of current state
        A = torch.eye(self.env.num_nodes)
        model.A = A

        # Transition model based on graph connectivity
        B = (
            torch.zeros(self.env.num_nodes, self.env.num_nodes, self.env.num_nodes))

        for state in range(self.env.num_nodes):
            neighbors = list(self.env.graph.neighbors(state))

            for action in range(self.env.num_nodes):
                if action in neighbors:
                    # Can move to neighbor
                    B[action, state, action] = 1.0
                else:
                    # Stay in current state if invalid action
                    B[state, state, action] = 1.0

        model.B = B

        # Preferences: strong preference for goal state
        C = torch.zeros(self.env.num_nodes, 1)
        C[self.env.goal_node, 0] = 3.0  # High preference for goal
        model.C = C

        # Initial state prior
        D = torch.zeros(self.env.num_nodes)
        D[0] = 1.0  # Start at node 0
        model.D = D

        return model

    def perceive(self, graph_obs: Data) -> torch.Tensor:
        """Process graph observation and update beliefs."""

        # Extract GNN features
        with torch.no_grad():
            gnn_features = (
                self.gnn_adapter.process_graph_observation(graph_obs))

        # Get discrete observation (current node)
        current_node = self.env.current_node
        obs_tensor = torch.tensor(current_node, dtype=torch.long)

        # Update beliefs using standard Active Inference
        self.belief = (
            self.inference.infer_states(obs_tensor, self.model, self.belief))

        # Enhance beliefs with GNN features (simple mixing)
        gnn_influence = torch.softmax(gnn_features.squeeze(), dim=0)
        self.belief = 0.8 * self.belief + 0.2 * gnn_influence
        self.belief = self.belief / self.belief.sum()  # Normalize

        return self.belief

    def act(self) -> int:
        """Select action based on current beliefs."""

        # Get valid actions (neighboring nodes)
        valid_actions = self.env.get_neighbors()

        # Evaluate policies
        action_probs = (
            self.policy_selector.evaluate_policies(self.belief, self.model))

        # Filter to valid actions only
        valid_probs = torch.zeros_like(action_probs)
        for action in valid_actions:
            valid_probs[action] = action_probs[action]

        # Renormalize
        if valid_probs.sum() > 0:
            valid_probs = valid_probs / valid_probs.sum()
        else:
            # Fallback: uniform over valid actions
            for action in valid_actions:
                valid_probs[action] = 1.0 / len(valid_actions)

        # Sample action
        action = torch.multinomial(valid_probs, 1).item()

        # Ensure action is valid
        if action not in valid_actions:
            action = np.random.choice(valid_actions)

        return action

    def reset(self) -> None:
        """Reset agent for new episode."""
        self.belief = self.model.D.clone()


def run_gnn_active_inference_episode(
    env: GraphEnvironment, agent: GraphActiveInferenceAgent, max_steps: int = 20
) -> tuple[list[int], list[torch.Tensor]]:
    """Run episode with GNN-enhanced Active Inference."""

    # Reset
    current_node = env.reset()
    agent.reset()

    path = [current_node]
    beliefs = []

    print(f"Starting navigation from node {current_node} to goal node {env.goal_node}")
    print(f"Graph has {env.num_nodes} nodes and {len(env.graph.edges())} edges\n")

    for step in range(max_steps):
        # Get graph observation
        graph_obs = env.get_graph_observation()

        # Agent perceives and updates beliefs
        belief = agent.perceive(graph_obs)
        beliefs.append(belief.clone())

        # Agent acts
        action = agent.act()

        # Environment step
        current_node, done = env.step(action)
        path.append(current_node)

        # Logging
        belief_entropy = -torch.sum(belief * torch.log(belief + 1e-8))
        most_likely_state = torch.argmax(belief).item()

        print(
            f"Step {step}: Node {env.current_node}, Action {action}, "
            f"Belief_entropy= (
                {belief_entropy:.3f}, Most_likely={most_likely_state}")
        )

        if done:
            print(f"\nReached goal in {step + 1} steps!")
            break

    return path, beliefs


def visualize_belief_evolution(beliefs: List[torch.Tensor], env: GraphEnvironment):
    """Visualize how beliefs evolve over time."""

    num_steps = min(len(beliefs), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(num_steps):
        ax = axes[i]
        belief = beliefs[i].numpy()

        # Bar plot of belief distribution
        bars = ax.bar(range(len(belief)), belief, alpha=0.7)

        # Highlight current position and goal
        bars[env.current_node].set_color("orange")
        bars[env.goal_node].set_color("red")

        ax.set_title(f"Belief Distribution - Step {i}")
        ax.set_xlabel("Node")
        ax.set_ylabel("Belief Probability")
        ax.set_ylim(0, 1)

        # Add text annotations
        ax.axvline(x= (
            0, color="green", linestyle="--", alpha=0.5, label="Start"))
        ax.axvline(x= (
            env.goal_node, color="red", linestyle="--", alpha=0.5, label="Goal"))

    plt.tight_layout()
    plt.suptitle("Belief Evolution During GNN-Enhanced Navigation", y=1.02)
    plt.show()


def compare_inference_methods(env: GraphEnvironment) -> dict:
    """Compare standard vs GNN-enhanced Active Inference."""

    results = {
        "standard": {"steps": [], "success": []},
        "gnn_enhanced": {"steps": [], "success": []},
    }

    num_trials = 5

    print("Comparing inference methods over multiple trials...\n")

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")

        # Standard Active Inference
        env.reset()
        standard_agent = GraphActiveInferenceAgent(env)
        # Remove GNN influence for standard comparison
        standard_agent.gnn_influence = 0.0

        path, _ = (
            run_gnn_active_inference_episode(env, standard_agent, max_steps=15))
        success = env.current_node == env.goal_node

        results["standard"]["steps"].append(len(path) - 1)
        results["standard"]["success"].append(success)

        print(f"  Standard: {len(path)-1} steps, Success: {success}")

        # GNN-enhanced Active Inference
        env.reset()
        gnn_agent = GraphActiveInferenceAgent(env)

        path, _ = (
            run_gnn_active_inference_episode(env, gnn_agent, max_steps=15))
        success = env.current_node == env.goal_node

        results["gnn_enhanced"]["steps"].append(len(path) - 1)
        results["gnn_enhanced"]["success"].append(success)

        print(f"  GNN-enhanced: {len(path)-1} steps, Success: {success}")
        print()

    return results


def plot_comparison_results(results: dict):
    """Plot comparison between standard and GNN-enhanced methods."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Success rate comparison
    standard_success = np.mean(results["standard"]["success"])
    gnn_success = np.mean(results["gnn_enhanced"]["success"])

    ax1.bar(
        ["Standard", "GNN-Enhanced"],
        [standard_success, gnn_success],
        color=["blue", "green"],
        alpha=0.7,
    )
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate Comparison")
    ax1.set_ylim(0, 1)

    # Add value labels
    ax1.text(0, standard_success + 0.02, f"{standard_success:.2f}", ha="center",
        va="bottom")
    ax1.text(1, gnn_success + 0.02, f"{gnn_success:.2f}", ha="center",
        va="bottom")

    # Steps comparison (for successful episodes only)
    standard_steps = [
        s
        for s, success in zip(results["standard"]["steps"], results["standard"]["success"])
        if success
    ]
    gnn_steps = [
        s
        for s, success in zip(results["gnn_enhanced"]["steps"],
            results["gnn_enhanced"]["success"])
        if success
    ]

    if standard_steps and gnn_steps:
        ax2.boxplot([standard_steps, gnn_steps], labels=["Standard",
            "GNN-Enhanced"])
        ax2.set_ylabel("Steps to Goal")
        ax2.set_title("Efficiency Comparison\n(Successful Episodes Only)")

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating GNN-Active Inference integration."""

    print("=== Active Inference with GNN Integration Example ===\n")

    # Create graph environment
    env = GraphEnvironment(num_nodes=8, edge_prob=0.4)
    print(f"Created graph environment with {env.num_nodes} nodes")
    print(f"Start: Node 0, Goal: Node {env.goal_node}")
    print(f"Graph connectivity: {len(env.graph.edges())} edges\n")

    # Visualize environment
    print("Graph structure:")
    env.visualize()

    # Create GNN-enhanced agent
    agent = GraphActiveInferenceAgent(env)
    print("Created GNN-enhanced Active Inference agent")
    print(f"GNN architecture: {agent.gnn}")
    print()

    # Run single episode
    print("Running single episode with detailed tracking...\n")
    path, beliefs = run_gnn_active_inference_episode(env, agent)

    # Show results
    print(f"\nEpisode Results:")
    print(f"Path taken: {' -> '.join(map(str, path))}")
    print(f"Path length: {len(path) - 1} steps")
    print(f"Success: {'Yes' if env.current_node == env.goal_node else 'No'}")

    # Visualize results
    print("\nGenerating visualizations...")
    env.visualize(agent_path=path)
    visualize_belief_evolution(beliefs, env)

    # Compare methods
    print("\nRunning comparison between standard and GNN-enhanced methods...")
    comparison_results = compare_inference_methods(env)

    # Show comparison statistics
    standard_avg_steps = np.mean(comparison_results["standard"]["steps"])
    gnn_avg_steps = np.mean(comparison_results["gnn_enhanced"]["steps"])
    standard_success_rate = np.mean(comparison_results["standard"]["success"])
    gnn_success_rate = np.mean(comparison_results["gnn_enhanced"]["success"])

    print(f"\nComparison Results:")
    print(f"Standard Active Inference:")
    print(f"  Average steps: {standard_avg_steps:.1f}")
    print(f"  Success rate: {standard_success_rate:.2f}")
    print(f"GNN-Enhanced Active Inference:")
    print(f"  Average steps: {gnn_avg_steps:.1f}")
    print(f"  Success rate: {gnn_success_rate:.2f}")

    plot_comparison_results(comparison_results)


if __name__ == "__main__":
    main()

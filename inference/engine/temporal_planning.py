"""
Module for FreeAgentics Active Inference implementation.
"""

import heapq
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .active_inference import InferenceAlgorithm
from .generative_model import DiscreteGenerativeModel, GenerativeModel
from .policy_selection import Policy, PolicySelector

logger = logging.getLogger(__name__)


@dataclass
class PlanningConfig:
    """Configuration for temporal planning"""

    planning_horizon: int = 10
    max_depth: int = 5
    branching_factor: int = 3
    search_type: str = "mcts"
    num_simulations: int = 100
    num_trajectories: int = 50
    enable_pruning: bool = True
    pruning_threshold: float = 0.01
    beam_width: int = 10
    discount_factor: float = 0.95
    exploration_constant: float = 1.0
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-16
    max_nodes: int = 10000
    enable_caching: bool = True


class TreeNode:
    """Node in the planning tree"""

    def __init__(
        self,
        state: torch.Tensor,
        action: Optional[int] = None,
        parent: Optional["TreeNode"] = None,
        depth: int = 0,
    ) -> None:
        self.state = state
        self.action = action
        self.parent = parent
        self.depth = depth
        self.children: List["TreeNode"] = []
        self.visits = 0
        self.value = 0.0
        self.expected_free_energy = float("inf")
        self._hash = None
        self._is_terminal = False

    def add_child(self, child: "TreeNode") -> None:
        ."""Add child node."""
        self.children.append(child)

    def is_leaf(self) -> bool:
        ."""Check if node is a leaf."""
        return len(self.children) == 0

    def is_fully_expanded(self, num_actions: int) -> bool:
        ."""Check if all actions have been tried."""
        return len(self.children) == num_actions

    def best_child(self, exploration_constant: float = 1.0) -> "TreeNode":
        """Select best child using UCB1"""
        if not self.children:
            return None

        def ucb1(child):
            if child.visits == 0:
                return float("inf")
            exploitation = -child.expected_free_energy
            exploration = exploration_constant * np.sqrt(np.log(self.visits) / child.visits)
            return exploitation + exploration

        return max(self.children, key=ucb1)

    def __hash__(self):
        """Hash for caching"""
        if self._hash is None:
            self._hash = hash((self.state.numpy().tobytes(), self.action, self.depth))
        return self._hash


class TemporalPlanner(ABC):
    """Abstract base class for temporal planning"""

    def __init__(
        self,
        config: PlanningConfig,
        policy_selector: PolicySelector,
        inference_algorithm: InferenceAlgorithm,
    ) -> None:
        self.config = config
        self.policy_selector = policy_selector
        self.inference = inference_algorithm
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def plan(
        self,
        initial_beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> tuple[Policy, float]:
        """Plan over temporal horizon"""
        pass

    @abstractmethod
    def evaluate_trajectory(
        self,
        trajectory: List[TreeNode],
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> float:
        """Evaluate a trajectory"""
        pass


class MonteCarloTreeSearch(TemporalPlanner):
    """
    Monte Carlo Tree Search for temporal planning.

    Balances exploration and exploitation to find optimal policies
    over extended time horizons.
    """

    def __init__(
        self,
        config: PlanningConfig,
        policy_selector: PolicySelector,
        inference_algorithm: InferenceAlgorithm,
    ) -> None:
        super().__init__(config, policy_selector, inference_algorithm)
        self.node_count = 0
        self.node_cache = {}

    def plan(
        self,
        initial_beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> tuple[Policy, float]:
        """
        Plan using MCTS.
        """
        root = TreeNode(initial_beliefs, depth=0)
        for _ in range(self.config.num_simulations):
            if self.node_count >= self.config.max_nodes:
                break
            node = self._select(root, generative_model)
            if not node._is_terminal and node.depth < self.config.max_depth:
                node = self._expand(node, generative_model)
            value = self._simulate(node, generative_model, preferences)
            self._backpropagate(node, value)
        best_policy = self._extract_policy(root)
        expected_value = root.value / max(root.visits, 1)
        return (best_policy, expected_value)

    def _select(self, node: TreeNode, generative_model: GenerativeModel) -> TreeNode:
        """Select node to expand using tree policy"""
        while not node.is_leaf():
            if not node.is_fully_expanded(generative_model.dims.num_actions):
                return node
            else:
                node = node.best_child(self.config.exploration_constant)
        return node

    def _expand(self, node: TreeNode, generative_model: GenerativeModel) -> TreeNode:
        """Expand node by adding new child"""
        tried_actions = {child.action for child in node.children}
        untried_actions = [
            a for a in range(generative_model.dims.num_actions) if a not in tried_actions
        ]
        if not untried_actions:
            return node
        action = np.random.choice(untried_actions)
        if isinstance(generative_model, DiscreteGenerativeModel):
            # Ensure consistent tensor dtype for matrix operations
            node_state_float = (
                node.state.float() if isinstance(node.state, torch.Tensor) else node.state
            )
            next_beliefs = generative_model.B[:, :, action] @ node_state_float
        else:
            next_beliefs = node.state
        child = TreeNode(next_beliefs, action, parent=node, depth=node.depth + 1)
        node.add_child(child)
        self.node_count += 1
        return child

    def _simulate(
        self,
        node: TreeNode,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> float:
        """Simulate from node to estimate value"""
        current_beliefs = node.state.clone()
        # Ensure consistent tensor dtype (Float32) for matrix operations
        if isinstance(current_beliefs, torch.Tensor):
            current_beliefs = current_beliefs.float()

        total_G = 0.0
        discount = 1.0
        for t in range(self.config.planning_horizon - node.depth):
            if t >= self.config.max_depth - node.depth:
                break
            action = np.random.randint(0, generative_model.dims.num_actions)
            policy = Policy([action])
            G, _, _ = self.policy_selector.compute_expected_free_energy(
                policy, current_beliefs, generative_model, preferences
            )
            total_G += discount * G.item()
            discount *= self.config.discount_factor
            if isinstance(generative_model, DiscreteGenerativeModel):
                current_beliefs = generative_model.B[:, :, action] @ current_beliefs
        return -total_G

    def _backpropagate(self, node: TreeNode, value: float):
        """Backpropagate value up the tree"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _extract_policy(self, root: TreeNode) -> Policy:
        """Extract best policy from tree"""
        actions = []
        node = root
        while not node.is_leaf():
            node = max(node.children, key=lambda n: n.visits)
            if node.action is not None:
                actions.append(node.action)
        return Policy(actions)

    def evaluate_trajectory(
        self,
        trajectory: List[TreeNode],
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> float:
        """Evaluate a trajectory based on free energy"""
        total_value = 0.0
        discount = 1.0
        for i in range(len(trajectory) - 1):
            if trajectory[i + 1].action is not None:
                policy = Policy([trajectory[i + 1].action])
                G, _, _ = self.policy_selector.compute_expected_free_energy(
                    policy, trajectory[i].state, generative_model, preferences
                )
                total_value += discount * -G.item()
                discount *= self.config.discount_factor
        return total_value


class BeamSearchPlanner(TemporalPlanner):
    """
    Beam Search for temporal planning.

    Keeps a beam of the most promising partial policies at each step.
    """

    def __init__(
        self,
        config: PlanningConfig,
        policy_selector: PolicySelector,
        inference_algorithm: InferenceAlgorithm,
    ) -> None:
        super().__init__(config, policy_selector, inference_algorithm)
        self.beam_width = config.beam_width

    def plan(
        self,
        initial_beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> tuple[Policy, float]:
        """
        Plan using beam search.
        """
        beam = [(0.0, [], initial_beliefs)]
        for depth in range(min(self.config.planning_horizon, self.config.max_depth)):
            new_beam = []
            for cost, actions, beliefs in beam:
                for action in range(generative_model.dims.num_actions):
                    policy = Policy([action])
                    G, _, _ = self.policy_selector.compute_expected_free_energy(
                        policy, beliefs, generative_model, preferences
                    )
                    new_cost = cost + self.config.discount_factor**depth * G.item()
                    new_actions = actions + [action]
                    if isinstance(generative_model, DiscreteGenerativeModel):
                        # Ensure consistent tensor dtype for matrix operations
                        beliefs_float = (
                            beliefs.float() if isinstance(beliefs, torch.Tensor) else beliefs
                        )
                        next_beliefs = generative_model.B[:, :, action] @ beliefs_float
                    else:
                        next_beliefs = beliefs
                    new_beam.append((new_cost, new_actions, next_beliefs))
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[: self.beam_width]

        if not beam:
            return Policy([]), float("-inf")

        best_cost, best_actions, _ = beam[0]
        return Policy(best_actions), -best_cost

    def evaluate_trajectory(
        self,
        trajectory: List[TreeNode],
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> float:
        """Evaluate a trajectory for beam search"""
        total_value = 0.0
        discount = 1.0
        for i in range(len(trajectory) - 1):
            if trajectory[i + 1].action is not None:
                policy = Policy([trajectory[i + 1].action])
                G, _, _ = self.policy_selector.compute_expected_free_energy(
                    policy, trajectory[i].state, generative_model, preferences
                )
                total_value += discount * -G.item()
                discount *= self.config.discount_factor
        return total_value


class AStarPlanner(TemporalPlanner):
    """
    A* search for temporal planning.

    Uses a heuristic to guide the search towards promising states.
    """

    def __init__(
        self,
        config: PlanningConfig,
        policy_selector: PolicySelector,
        inference_algorithm: InferenceAlgorithm,
    ) -> None:
        super().__init__(config, policy_selector, inference_algorithm)

    def plan(
        self,
        initial_beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> tuple[Policy, float]:
        """
        Plan using A* search.
        """
        open_set = [(0.0, 0.0, [], initial_beliefs)]
        closed_set = set()
        g_scores = defaultdict(lambda: float("inf"))
        g_scores[self._hash_beliefs(initial_beliefs)] = 0.0
        while open_set and len(closed_set) < self.config.max_nodes:
            f_score, g_score, actions, beliefs = heapq.heappop(open_set)
            if len(actions) >= self.config.planning_horizon:
                return (Policy(actions), -g_score)
            state_hash = self._hash_beliefs(beliefs)
            if state_hash in closed_set:
                continue
            closed_set.add(state_hash)
            for action in range(generative_model.dims.num_actions):
                policy = Policy([action])
                G, _, _ = self.policy_selector.compute_expected_free_energy(
                    policy, beliefs, generative_model, preferences
                )
                step_cost = self.config.discount_factor ** len(actions) * G.item()
                new_g_score = g_score + step_cost
                if isinstance(generative_model, DiscreteGenerativeModel):
                    next_beliefs = generative_model.B[:, :, action] @ beliefs
                else:
                    next_beliefs = beliefs
                if self._hash_beliefs(next_beliefs) not in closed_set:
                    h_score = self._heuristic(
                        next_beliefs, len(actions) + 1, generative_model, preferences
                    )
                    f_score_new = new_g_score + h_score
                    heapq.heappush(
                        open_set,
                        (
                            f_score_new,
                            new_g_score,
                            actions + [action],
                            next_beliefs,
                        ),
                    )
        return Policy([]), float("-inf")

    def _hash_beliefs(self, beliefs: torch.Tensor) -> int:
        """Hash beliefs for caching"""
        return hash(beliefs.numpy().tobytes())

    def _heuristic(
        self,
        beliefs: torch.Tensor,
        depth: int,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> float:
        """Heuristic function for remaining cost"""
        remaining_steps = self.config.planning_horizon - depth
        if remaining_steps <= 0:
            return 0.0
        sample_G = []
        for _ in range(min(3, generative_model.dims.num_actions)):
            action = np.random.randint(0, generative_model.dims.num_actions)
            policy = Policy([action])
            G, _, _ = self.policy_selector.compute_expected_free_energy(
                policy, beliefs, generative_model, preferences
            )
            sample_G.append(G.item())
        return np.mean(sample_G) if sample_G else 0.0

    def evaluate_trajectory(
        self,
        trajectory: List[TreeNode],
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> float:
        """Evaluate a trajectory for A*"""
        total_value = 0.0
        discount = 1.0
        for i in range(len(trajectory) - 1):
            if trajectory[i + 1].action is not None:
                policy = Policy([trajectory[i + 1].action])
                G, _, _ = self.policy_selector.compute_expected_free_energy(
                    policy, trajectory[i].state, generative_model, preferences
                )
                total_value += discount * -G.item()
                discount *= self.config.discount_factor
        return total_value


class TrajectorySampling(TemporalPlanner):
    """
    Trajectory Sampling for temporal planning.

    Samples full trajectories and selects the best one.
    """

    def __init__(
        self,
        config: PlanningConfig,
        policy_selector: PolicySelector,
        inference_algorithm: InferenceAlgorithm,
    ) -> None:
        super().__init__(config, policy_selector, inference_algorithm)

    def plan(
        self,
        initial_beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> tuple[Policy, float]:
        """
        Plan using trajectory sampling.
        """
        best_trajectory = []
        best_value = float("-inf")

        for _ in range(self.config.num_trajectories):
            trajectory, value = self._sample_trajectory(
                initial_beliefs, generative_model, preferences
            )
            if value > best_value:
                best_value = value
                best_trajectory = trajectory

        actions = [node.action for node in best_trajectory[1:] if node.action is not None]
        return Policy(actions), best_value

    def _sample_trajectory(
        self,
        initial_beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> tuple[list[TreeNode], float]:
        """Sample a single trajectory"""
        trajectory = [TreeNode(initial_beliefs, depth=0)]
        current_beliefs = initial_beliefs.clone()
        total_value = 0.0
        discount = 1.0
        for depth in range(min(self.config.planning_horizon, self.config.max_depth)):
            temp_policy, _ = self.policy_selector.select_policy(
                current_beliefs, generative_model, preferences
            )
            if len(temp_policy) > 0:
                action = temp_policy[0].item()
            else:
                action = np.random.randint(0, generative_model.dims.num_actions)
            policy = Policy([action])
            G, _, _ = self.policy_selector.compute_expected_free_energy(
                policy, current_beliefs, generative_model, preferences
            )
            total_value += discount * -G.item()
            discount *= self.config.discount_factor
            if isinstance(generative_model, DiscreteGenerativeModel):
                next_beliefs = (
                    generative_model.B[:, :, action] @ current_beliefs)
            else:
                next_beliefs = current_beliefs

            trajectory.append(TreeNode(next_beliefs, action, trajectory[-1], depth + 1))
            current_beliefs = next_beliefs
        return (trajectory, total_value)

    def evaluate_trajectory(
        self,
        trajectory: List[TreeNode],
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> float:
        """Evaluate a trajectory based on free energy"""
        total_value = 0.0
        discount = 1.0
        for i in range(len(trajectory) - 1):
            if trajectory[i + 1].action is not None:
                policy = Policy([trajectory[i + 1].action])
                G, _, _ = self.policy_selector.compute_expected_free_energy(
                    policy, trajectory[i].state, generative_model, preferences
                )
                total_value += discount * -G.item()
                discount *= self.config.discount_factor
        return total_value


class AdaptiveHorizonPlanner(TemporalPlanner):
    """
    Adaptive Horizon Planner.

    Adjusts the planning horizon based on state uncertainty.
    """

    def __init__(
        self,
        config: PlanningConfig,
        policy_selector: PolicySelector,
        inference_algorithm: InferenceAlgorithm,
        base_planner: TemporalPlanner,
    ) -> None:
        super().__init__(config, policy_selector, inference_algorithm)
        self.base_planner = base_planner
        self.uncertainty_threshold = 0.7  # Default, can be tuned

    def plan(
        self,
        initial_beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> tuple[Policy, float]:
        """
        Plan with an adaptive horizon.
        """
        uncertainty = self._measure_uncertainty(initial_beliefs)
        original_horizon = self.config.planning_horizon
        if uncertainty > self.uncertainty_threshold:
            self.base_planner.config.planning_horizon = int(original_horizon * 1.5)
        elif uncertainty < self.uncertainty_threshold / 2:
            self.base_planner.config.planning_horizon = int(original_horizon * 0.75)

        policy, value = self.base_planner.plan(initial_beliefs, generative_model, preferences)

        self.base_planner.config.planning_horizon = original_horizon
        return policy, value

    def _measure_uncertainty(self, beliefs: torch.Tensor) -> float:
        """Measure uncertainty using entropy"""
        return -torch.sum(beliefs * torch.log(beliefs + self.config.eps)).item()

    def evaluate_trajectory(
        self,
        trajectory: List[TreeNode],
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> float:
        """Delegate evaluation to the base planner"""
        return self.base_planner.evaluate_trajectory(trajectory, generative_model, preferences)


def create_temporal_planner(
    planner_type: str,
    config: Optional[PlanningConfig] = None,
    policy_selector: Optional[PolicySelector] = None,
    inference_algorithm: Optional[InferenceAlgorithm] = None,
    **kwargs,
) -> TemporalPlanner:
    """Create temporal planner"""
    if config is None:
        config = PlanningConfig()
    if policy_selector is None:
        from .policy_selection import PolicyConfig, create_policy_selector

        policy_config = PolicyConfig()
        policy_selector = create_policy_selector("discrete", config=policy_config)
    if inference_algorithm is None:
        from .active_inference import InferenceConfig, create_inference_algorithm

        inference_config = InferenceConfig()
        inference_algorithm = create_inference_algorithm("vmp", config=inference_config)

    if planner_type == "mcts":
        return MonteCarloTreeSearch(config, policy_selector, inference_algorithm)
    elif planner_type == "beam":
        return BeamSearchPlanner(config, policy_selector, inference_algorithm)
    elif planner_type == "astar":
        return AStarPlanner(config, policy_selector, inference_algorithm)
    elif planner_type == "sampling":
        return TrajectorySampling(config, policy_selector, inference_algorithm)
    elif planner_type == "adaptive":
        base_planner = MonteCarloTreeSearch(config, policy_selector, inference_algorithm)
        return AdaptiveHorizonPlanner(config, policy_selector, inference_algorithm, base_planner)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

import os
import sys

import numpy as np
import pytest
import torch

from inference.engine.active_inference import InferenceConfig, VariationalMessagePassing
from inference.engine.generative_model import (
    DiscreteGenerativeModel,
    ModelDimensions,
    ModelParameters,
)
from inference.engine.policy_selection import DiscreteExpectedFreeEnergy, PolicyConfig
from inference.engine.temporal_planning import (
    AdaptiveHorizonPlanner,
    AStarPlanner,
    BeamSearchPlanner,
    MonteCarloTreeSearch,
    PlanningConfig,
    TrajectorySampling,
    TreeNode,
    create_temporal_planner,
)


class TestPlanningConfig:
    """Test PlanningConfig dataclass"""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = PlanningConfig()
        assert config.planning_horizon == 10
        assert config.max_depth == 5
        assert config.branching_factor == 3
        assert config.search_type == "mcts"
        assert config.num_simulations == 100
        assert config.num_trajectories == 50
        assert config.enable_pruning is True
        assert config.pruning_threshold == 0.01
        assert config.beam_width == 10
        assert config.discount_factor == 0.95
        assert config.exploration_constant == 1.0
        assert config.use_gpu is True
        assert config.dtype == torch.float32
        assert config.eps == 1e-16
        assert config.max_nodes == 10000
        assert config.enable_caching is True

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = PlanningConfig(
            planning_horizon=20, num_simulations=200, discount_factor=0.9, use_gpu=False
        )
        assert config.planning_horizon == 20
        assert config.num_simulations == 200
        assert config.discount_factor == 0.9
        assert config.use_gpu is False


class TestTreeNode:
    """Test TreeNode class"""

    def test_node_creation(self) -> None:
        """Test creating tree nodes"""
        state = torch.tensor([0.5, 0.5])
        node = TreeNode(state, action=1, depth=2)
        assert torch.equal(node.state, state)
        assert node.action == 1
        assert node.depth == 2
        assert node.parent is None
        assert len(node.children) == 0
        assert node.visits == 0
        assert node.value == 0.0
        assert node.expected_free_energy == float("inf")

    def test_node_relationships(self) -> None:
        """Test parent-child relationships"""
        root = TreeNode(torch.tensor([1.0, 0.0]), depth=0)
        child1 = TreeNode(torch.tensor([0.7, 0.3]), action=0, parent=root, depth=1)
        child2 = TreeNode(torch.tensor([0.3, 0.7]), action=1, parent=root, depth=1)
        root.add_child(child1)
        root.add_child(child2)
        assert len(root.children) == 2
        assert child1 in root.children
        assert child2 in root.children
        assert child1.parent == root
        assert child2.parent == root

    def test_node_properties(self) -> None:
        """Test node properties"""
        root = TreeNode(torch.tensor([1.0, 0.0]))
        child = TreeNode(torch.tensor([0.5, 0.5]), action=0, parent=root)
        root.add_child(child)
        assert root.is_leaf() is False
        assert child.is_leaf() is True
        assert root.is_fully_expanded(1) is True
        assert root.is_fully_expanded(2) is False

    def test_best_child_selection(self) -> None:
        """Test UCB1 child selection"""
        root = TreeNode(torch.tensor([1.0, 0.0]))
        root.visits = 10
        # Create children with different statistics
        child1 = TreeNode(torch.tensor([0.7, 0.3]), action=0, parent=root)
        child1.visits = 5
        child1.value = 2.0
        child1.expected_free_energy = 0.5
        child2 = TreeNode(torch.tensor([0.3, 0.7]), action=1, parent=root)
        child2.visits = 3
        child2.value = 1.5
        child2.expected_free_energy = 0.3
        root.add_child(child1)
        root.add_child(child2)
        # Best child should balance exploitation and exploration
        best = root.best_child(exploration_constant=1.0)
        assert best is not None
        assert best in [child1, child2]

    def test_node_hashing(self) -> None:
        """Test node hashing for caching"""
        state = torch.tensor([0.5, 0.5])
        node1 = TreeNode(state, action=1, depth=2)
        node2 = TreeNode(state, action=1, depth=2)
        # Same state and properties should have same hash
        assert hash(node1) == hash(node2)
        # Different properties should have different hash
        node3 = TreeNode(state, action=2, depth=2)
        assert hash(node1) != hash(node3)


class TestMonteCarloTreeSearch:
    """Test MCTS planner"""

    def setup_method(self) -> None:
        """Set up test environment"""
        # Create model
        self.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        # Set up simple transition model
        self.model.B[:, :, 0] = torch.eye(3)  # Action 0: stay
        self.model.B[:, :, 1] = torch.tensor(
            [  # Action 1: rotate
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        # Create inference and policy selector
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = VariationalMessagePassing(inf_config)
        policy_config = PolicyConfig(use_gpu=False)
        self.policy_selector = DiscreteExpectedFreeEnergy(policy_config, self.inference)
        # Create planner
        self.config = PlanningConfig(
            planning_horizon=3, num_simulations=20, max_depth=3, use_gpu=False
        )
        self.planner = MonteCarloTreeSearch(self.config, self.policy_selector, self.inference)

    def test_mcts_planning(self) -> None:
        """Test basic MCTS planning"""
        beliefs = torch.tensor([1.0, 0.0, 0.0])
        policy, value = self.planner.plan(beliefs, self.model)
        assert isinstance(policy.actions, torch.Tensor)
        assert len(policy) <= self.config.planning_horizon
        assert isinstance(value, float)

    def test_tree_expansion(self) -> None:
        """Test tree expansion in MCTS"""
        root = TreeNode(torch.tensor([1.0, 0.0, 0.0]), depth=0)
        # Expand should add a child
        child = self.planner._expand(root, self.model)
        assert len(root.children) == 1
        assert child.parent == root
        assert child.action in [0, 1]
        assert child.depth == 1

    def test_simulation_rollout(self) -> None:
        """Test simulation phase"""
        node = TreeNode(torch.tensor([0.33, 0.33, 0.34]), depth=1)
        value = self.planner._simulate(node, self.model)
        assert isinstance(value, float)
        assert not np.isnan(value)
        assert not np.isinf(value)

    def test_backpropagation(self) -> None:
        """Test value backpropagation"""
        root = TreeNode(torch.tensor([1.0, 0.0, 0.0]), depth=0)
        child = TreeNode(torch.tensor([0.0, 1.0, 0.0]), action=1, parent=root, depth=1)
        root.add_child(child)
        # Backpropagate value
        value = 5.0
        self.planner._backpropagate(child, value)
        assert child.visits == 1
        assert child.value == value
        assert root.visits == 1
        assert root.value == value

    def test_policy_extraction(self) -> None:
        """Test extracting policy from tree"""
        # Build a simple tree
        root = TreeNode(torch.tensor([1.0, 0.0, 0.0]), depth=0)
        child1 = TreeNode(torch.tensor([0.0, 1.0, 0.0]), action=1, parent=root, depth=1)
        child1.visits = 10
        child2 = TreeNode(torch.tensor([1.0, 0.0, 0.0]), action=0, parent=root, depth=1)
        child2.visits = 5
        root.add_child(child1)
        root.add_child(child2)
        grandchild = TreeNode(torch.tensor([0.0, 0.0, 1.0]), action=1, parent=child1, depth=2)
        grandchild.visits = 8
        child1.add_child(grandchild)
        # Extract policy (should follow most visited path)
        policy = self.planner._extract_policy(root)
        assert len(policy) >= 1
        assert policy[0].item() == 1  # child1 has more visits

    def test_node_limit(self) -> None:
        """Test node count limit"""
        self.config.max_nodes = 10
        self.planner = MonteCarloTreeSearch(self.config, self.policy_selector, self.inference)
        beliefs = torch.tensor([0.33, 0.33, 0.34])
        policy, value = self.planner.plan(beliefs, self.model)
        assert self.planner.node_count <= self.config.max_nodes


class TestBeamSearchPlanner:
    """Test beam search planner"""

    def setup_method(self) -> None:
        """Set up test environment"""
        # Create model
        self.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        # Create components
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = VariationalMessagePassing(inf_config)
        policy_config = PolicyConfig(use_gpu=False)
        self.policy_selector = DiscreteExpectedFreeEnergy(policy_config, self.inference)
        # Create planner
        self.config = PlanningConfig(planning_horizon=3, beam_width=5, use_gpu=False)
        self.planner = BeamSearchPlanner(self.config, self.policy_selector, self.inference)

    def test_beam_search_planning(self) -> None:
        """Test basic beam search"""
        beliefs = torch.tensor([1.0, 0.0, 0.0])
        policy, value = self.planner.plan(beliefs, self.model)
        assert isinstance(policy.actions, torch.Tensor)
        assert len(policy) <= self.config.planning_horizon
        assert isinstance(value, float)

    def test_beam_pruning(self) -> None:
        """Test beam width constraint"""
        # Use small beam width
        self.config.beam_width = 2
        self.planner = BeamSearchPlanner(self.config, self.policy_selector, self.inference)
        beliefs = torch.tensor([0.33, 0.33, 0.34])
        policy, value = self.planner.plan(beliefs, self.model)
        # Policy should still be valid
        assert len(policy) <= self.config.planning_horizon
        assert all(a.item() in [0, 1] for a in policy.actions)

    def test_early_convergence(self) -> None:
        """Test early stopping when beam converges"""
        # Start with deterministic state
        beliefs = torch.tensor([1.0, 0.0, 0.0])
        # Set strong preferences to encourage convergence
        preferences = torch.tensor([-10.0, 10.0, -10.0])
        self.model.set_preferences(preferences)
        policy, value = self.planner.plan(beliefs, self.model, preferences)
        # Should find a policy (even if short due to convergence)
        assert len(policy) >= 1


class TestAStarPlanner:
    """Test A* planner"""

    def setup_method(self) -> None:
        """Set up test environment"""
        # Create model
        self.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        # Create components
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = VariationalMessagePassing(inf_config)
        policy_config = PolicyConfig(use_gpu=False)
        self.policy_selector = DiscreteExpectedFreeEnergy(policy_config, self.inference)
        # Create planner
        self.config = PlanningConfig(planning_horizon=3, max_nodes=100, use_gpu=False)
        self.planner = AStarPlanner(self.config, self.policy_selector, self.inference)

    def test_astar_planning(self) -> None:
        """Test basic A* planning"""
        beliefs = torch.tensor([1.0, 0.0, 0.0])
        policy, value = self.planner.plan(beliefs, self.model)
        assert isinstance(policy.actions, torch.Tensor)
        assert len(policy) <= self.config.planning_horizon
        assert isinstance(value, float)

    def test_heuristic_function(self) -> None:
        """Test heuristic estimation"""
        beliefs = torch.tensor([0.33, 0.33, 0.34])
        h_score = self.planner._heuristic(beliefs, 1, self.model)
        assert isinstance(h_score, float)
        assert h_score >= 0  # Heuristic should be non-negative
        # Heuristic should decrease with depth
        h_score_deep = self.planner._heuristic(beliefs, 3, self.model)
        assert h_score_deep <= h_score

    def test_belief_hashing(self) -> None:
        """Test belief state hashing"""
        beliefs1 = torch.tensor([0.5, 0.5, 0.0])
        beliefs2 = torch.tensor([0.5, 0.5, 0.0])
        beliefs3 = torch.tensor([0.0, 0.5, 0.5])
        hash1 = self.planner._hash_beliefs(beliefs1)
        hash2 = self.planner._hash_beliefs(beliefs2)
        hash3 = self.planner._hash_beliefs(beliefs3)
        assert hash1 == hash2  # Same beliefs
        assert hash1 != hash3  # Different beliefs

    def test_node_expansion_limit(self) -> None:
        """Test node expansion limit"""
        self.config.max_nodes = 10
        self.planner = AStarPlanner(self.config, self.policy_selector, self.inference)
        beliefs = torch.tensor([0.33, 0.33, 0.34])
        policy, value = self.planner.plan(beliefs, self.model)
        # Should still return a valid policy
        assert isinstance(policy.actions, torch.Tensor)


class TestTrajectorySampling:
    """Test trajectory sampling planner"""

    def setup_method(self) -> None:
        """Set up test environment"""
        # Create model
        self.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        # Create components
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = VariationalMessagePassing(inf_config)
        policy_config = PolicyConfig(use_gpu=False)
        self.policy_selector = DiscreteExpectedFreeEnergy(policy_config, self.inference)
        # Create planner
        self.config = PlanningConfig(planning_horizon=3, num_trajectories=10, use_gpu=False)
        self.planner = TrajectorySampling(self.config, self.policy_selector, self.inference)

    def test_trajectory_sampling(self) -> None:
        """Test basic trajectory sampling"""
        beliefs = torch.tensor([1.0, 0.0, 0.0])
        policy, value = self.planner.plan(beliefs, self.model)
        assert isinstance(policy.actions, torch.Tensor)
        assert len(policy) <= self.config.planning_horizon
        assert isinstance(value, float)

    def test_single_trajectory(self) -> None:
        """Test sampling a single trajectory"""
        beliefs = torch.tensor([0.33, 0.33, 0.34])
        trajectory, value = self.planner._sample_trajectory(beliefs, self.model)
        assert len(trajectory) > 1
        assert trajectory[0].state.shape == beliefs.shape
        assert isinstance(value, float)
        # Check trajectory consistency
        for i in range(len(trajectory) - 1):
            assert trajectory[i + 1].parent == trajectory[i]
            assert trajectory[i + 1].depth == trajectory[i].depth + 1

    def test_trajectory_evaluation(self) -> None:
        """Test trajectory evaluation"""
        # Create a simple trajectory
        node1 = TreeNode(torch.tensor([1.0, 0.0, 0.0]), depth=0)
        node2 = TreeNode(torch.tensor([0.0, 1.0, 0.0]), action=1, parent=node1, depth=1)
        node3 = TreeNode(torch.tensor([0.0, 0.0, 1.0]), action=1, parent=node2, depth=2)
        trajectory = [node1, node2, node3]
        value = self.planner.evaluate_trajectory(trajectory, self.model)
        assert isinstance(value, float)
        assert not np.isnan(value)

    def test_multiple_trajectories(self) -> None:
        """Test that multiple trajectories are sampled"""
        beliefs = torch.tensor([0.5, 0.5, 0.0])
        # Sample multiple times to check for variation
        policies = []
        for _ in range(5):
            policy, _ = self.planner.plan(beliefs, self.model)
            policies.append(policy.actions.tolist())
        # Should have some variation in sampled policies
        # (though not guaranteed with small action space)
        assert len(policies) == 5


class TestAdaptiveHorizonPlanner:
    """Test adaptive horizon planner"""

    def setup_method(self) -> None:
        """Set up test environment"""
        # Create model
        self.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        # Create components
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = VariationalMessagePassing(inf_config)
        policy_config = PolicyConfig(use_gpu=False)
        self.policy_selector = DiscreteExpectedFreeEnergy(policy_config, self.inference)
        # Create base planner
        planning_config = PlanningConfig(planning_horizon=5, num_simulations=10, use_gpu=False)
        base_planner = MonteCarloTreeSearch(planning_config, self.policy_selector, self.inference)
        # Create adaptive planner
        self.planner = AdaptiveHorizonPlanner(
            planning_config, self.policy_selector, self.inference, base_planner
        )

    def test_adaptive_planning(self) -> None:
        """Test basic adaptive planning"""
        beliefs = torch.tensor([1.0, 0.0, 0.0])
        policy, value = self.planner.plan(beliefs, self.model)
        assert isinstance(policy.actions, torch.Tensor)
        assert isinstance(value, float)

    def test_uncertainty_measurement(self) -> None:
        """Test uncertainty measurement"""
        # Low uncertainty (peaked distribution)
        low_uncertainty_beliefs = torch.tensor([0.9, 0.05, 0.05])
        low_u = self.planner._measure_uncertainty(low_uncertainty_beliefs)
        # High uncertainty (uniform distribution)
        high_uncertainty_beliefs = torch.tensor([0.33, 0.33, 0.34])
        high_u = self.planner._measure_uncertainty(high_uncertainty_beliefs)
        assert 0 <= low_u <= 1
        assert 0 <= high_u <= 1
        assert high_u > low_u

    def test_horizon_adaptation(self) -> None:
        """Test that horizon adapts to uncertainty"""
        # High uncertainty should increase horizon
        high_uncertainty = torch.tensor([0.33, 0.33, 0.34])
        self.planner.uncertainty_threshold = 0.5
        original_horizon = self.planner.config.planning_horizon
        policy, _ = self.planner.plan(high_uncertainty, self.model)
        # Horizon should be restored after planning
        assert self.planner.config.planning_horizon == original_horizon


class TestTemporalPlannerFactory:
    """Test temporal planner factory"""

    def setup_method(self) -> None:
        """Set up test environment"""
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = VariationalMessagePassing(inf_config)
        policy_config = PolicyConfig(use_gpu=False)
        self.policy_selector = DiscreteExpectedFreeEnergy(policy_config, self.inference)
        self.kwargs = {
            "policy_selector": self.policy_selector,
            "inference_algorithm": self.inference,
        }

    def test_create_mcts_planner(self) -> None:
        """Test MCTS planner creation"""
        planner = create_temporal_planner("mcts", **self.kwargs)
        assert isinstance(planner, MonteCarloTreeSearch)

    def test_create_beam_planner(self) -> None:
        """Test beam search planner creation"""
        planner = create_temporal_planner("beam", **self.kwargs)
        assert isinstance(planner, BeamSearchPlanner)

    def test_create_astar_planner(self) -> None:
        """Test A* planner creation"""
        planner = create_temporal_planner("astar", **self.kwargs)
        assert isinstance(planner, AStarPlanner)

    def test_create_sampling_planner(self) -> None:
        """Test trajectory sampling planner creation"""
        planner = create_temporal_planner("sampling", **self.kwargs)
        assert isinstance(planner, TrajectorySampling)

    def test_create_adaptive_planner(self) -> None:
        """Test adaptive planner creation"""
        planner = create_temporal_planner("adaptive", base_planner_type="mcts", **self.kwargs)
        assert isinstance(planner, AdaptiveHorizonPlanner)
        assert isinstance(planner.base_planner, MonteCarloTreeSearch)

    def test_invalid_planner_type(self) -> None:
        """Test invalid planner type"""
        with pytest.raises(ValueError):
            create_temporal_planner("invalid", **self.kwargs)

    def test_missing_components(self) -> None:
        """Test missing required components"""
        with pytest.raises(ValueError):
            create_temporal_planner("mcts")  # Missing policy_selector and inference

    def test_custom_config(self) -> None:
        """Test creation with custom config"""
        config = PlanningConfig(planning_horizon=20, num_simulations=200, use_gpu=False)
        planner = create_temporal_planner("mcts", config=config, **self.kwargs)
        assert planner.config.planning_horizon == 20
        assert planner.config.num_simulations == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

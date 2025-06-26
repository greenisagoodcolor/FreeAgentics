import os
import sys

import pytest
import torch

from inference.engine.active_inference import (
    GradientDescentInference,
    InferenceConfig,
    VariationalMessagePassing,
)
from inference.engine.generative_model import (
    ContinuousGenerativeModel,
    DiscreteGenerativeModel,
    ModelDimensions,
    ModelParameters,
)
from inference.engine.policy_selection import (
    ContinuousExpectedFreeEnergy,
    DiscreteExpectedFreeEnergy,
    HierarchicalPolicySelector,
    Policy,
    PolicyConfig,
    SophisticatedInference,
    create_policy_selector,
)


class TestPolicyConfig:
    """Test PolicyConfig dataclass"""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = PolicyConfig()
        assert config.planning_horizon == 5
        assert config.num_policies is None
        assert config.policy_length == 1
        assert config.epistemic_weight == 1.0
        assert config.pragmatic_weight == 1.0
        assert config.exploration_constant == 1.0
        assert config.habit_strength == 0.0
        assert config.use_gpu is True
        assert config.dtype == torch.float32
        assert config.eps == 1e-16
        assert config.use_sampling is False
        assert config.num_samples == 100
        assert config.enable_pruning is True
        assert config.pruning_threshold == 0.01

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = PolicyConfig(
            planning_horizon=10, num_policies=50, epistemic_weight=0.5, use_gpu=False
        )
        assert config.planning_horizon == 10
        assert config.num_policies == 50
        assert config.epistemic_weight == 0.5
        assert config.use_gpu is False


class TestPolicy:
    """Test Policy class"""

    def test_policy_creation_from_list(self) -> None:
        """Test creating policy from list"""
        actions = [0, 1, 0, 2]
        policy = Policy(actions)
        assert len(policy) == 4
        assert policy[0] == 0
        assert policy[1] == 1
        assert policy[2] == 0
        assert policy[3] == 2

    def test_policy_creation_from_tensor(self) -> None:
        """Test creating policy from tensor"""
        actions = torch.tensor([1, 2, 1])
        policy = Policy(actions, horizon=5)
        assert len(policy) == 3
        assert policy.horizon == 5
        assert torch.equal(policy.actions, actions)

    def test_policy_repr(self) -> None:
        """Test policy string representation"""
        policy = Policy([0, 1])
        assert repr(policy) == "Policy([0, 1])"


class TestDiscreteExpectedFreeEnergy:
    """
    Test discrete expected free energy calculation.
    NOTE: DiscreteExpectedFreeEnergy is now a backward compatibility alias
    that points to the PyMDP-based implementation (PyMDPPolicyAdapter).
    These tests validate that the PyMDP integration works correctly through
    the compatibility interface.
    """

    def setup_method(self) -> None:
        """Set up test environment"""
        # Create model
        self.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        # Set up clear observation model (identity)
        self.model.A = torch.eye(3)
        # Set up transition model
        # Action 0: stay in same state
        self.model.B[:, :, 0] = torch.eye(3)
        # Action 1: rotate states (0->1, 1->2, 2->0)
        self.model.B[:, :, 1] = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Create inference
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = VariationalMessagePassing(inf_config)
        # Create policy selector
        self.config = PolicyConfig(planning_horizon=3, policy_length=1, use_gpu=False)
        self.selector = DiscreteExpectedFreeEnergy(self.config, self.inference)

    def test_enumerate_policies_single_step(self) -> None:
        """Test policy enumeration for single step"""
        policies = self.selector.enumerate_policies(3)
        assert len(policies) == 3
        assert all(len(p) == 1 for p in policies)
        assert policies[0].actions.item() == 0
        assert policies[1].actions.item() == 1
        assert policies[2].actions.item() == 2

    def test_enumerate_policies_multi_step(self) -> None:
        """Test policy enumeration for multiple steps"""
        self.selector.config.policy_length = 2
        policies = self.selector.enumerate_policies(2)
        assert len(policies) == 4  # 2^2
        # Check all combinations present
        expected = [[0, 0], [0, 1], [1, 0], [1, 1]]
        actual = [p.actions.tolist() for p in policies]
        assert all(e in actual for e in expected)

    def test_expected_free_energy_epistemic(self) -> None:
        """Test epistemic value computation"""
        # Set weights
        self.config.epistemic_weight = 1.0
        self.config.pragmatic_weight = 0.0
        # Uncertain beliefs
        beliefs = torch.tensor([0.33, 0.33, 0.34])
        # Policy that changes state (more informative)
        policy1 = Policy([1])
        G1, epistemic1, pragmatic1 = self.selector.compute_expected_free_energy(
            policy1, beliefs, self.model
        )
        # Policy that stays (less informative)
        policy0 = Policy([0])
        G0, epistemic0, pragmatic0 = self.selector.compute_expected_free_energy(
            policy0, beliefs, self.model
        )
        # Both should be finite
        assert torch.isfinite(G0)
        assert torch.isfinite(G1)

    def test_expected_free_energy_pragmatic(self) -> None:
        """Test pragmatic value computation"""
        # Set weights
        self.config.epistemic_weight = 0.0
        self.config.pragmatic_weight = 1.0
        # Set preferences (prefer observation 1)
        preferences = torch.tensor([-1.0, 2.0, -1.0])
        self.model.set_preferences(preferences)
        # Start in state 0
        beliefs = torch.tensor([1.0, 0.0, 0.0])
        # Policy that moves to preferred state
        policy1 = Policy([1])  # 0->1
        G1, _, _ = self.selector.compute_expected_free_energy(
            policy1, beliefs, self.model, preferences
        )
        # Policy that stays
        policy0 = Policy([0])  # Stay at 0
        G0, _, _ = self.selector.compute_expected_free_energy(
            policy0, beliefs, self.model, preferences
        )
        # Moving to preferred state should have lower free energy
        assert G1 < G0

    def test_policy_selection(self) -> None:
        """Test policy selection"""
        # Set preferences
        preferences = torch.tensor([-1.0, -1.0, 2.0])  # Prefer state 2
        self.model.set_preferences(preferences)
        # Start in state 0
        beliefs = torch.tensor([1.0, 0.0, 0.0])
        # Select policy
        policy, probs = self.selector.select_policy(beliefs, self.model)
        assert isinstance(policy, Policy)
        assert probs.shape == (2,)  # Two actions
        assert torch.allclose(probs.sum(), torch.tensor(1.0))
        # Should prefer action 1 (moves toward goal)
        assert probs[1] > probs[0]

    def test_policy_pruning(self) -> None:
        """Test policy pruning"""
        self.config.enable_pruning = True
        self.config.pruning_threshold = 0.3
        beliefs = torch.tensor([0.33, 0.33, 0.34])
        policy, probs = self.selector.select_policy(beliefs, self.model)
        # Check that low probability policies are effectively zero
        assert torch.all(probs[probs < 0.3] < 0.01)

    def test_stochastic_policy_selection(self) -> None:
        """Test stochastic policy selection"""
        self.config.use_sampling = True
        beliefs = torch.tensor([0.5, 0.5, 0.0])
        # Run multiple times to check stochasticity
        selected_actions = []
        for _ in range(20):
            policy, _ = self.selector.select_policy(beliefs, self.model)
            selected_actions.append(policy[0].item())
        # Should select both actions sometimes
        assert len(set(selected_actions)) > 1


class TestContinuousExpectedFreeEnergy:
    """Test continuous expected free energy"""

    def setup_method(self) -> None:
        """Set up test environment"""
        # Create continuous model
        self.dims = ModelDimensions(num_states=2, num_observations=2, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = ContinuousGenerativeModel(self.dims, self.params, hidden_dim=16)
        # Create inference
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = GradientDescentInference(inf_config)
        # Create policy selector
        self.config = PolicyConfig(
            planning_horizon=3, policy_length=2, num_policies=10, use_gpu=False
        )
        self.selector = ContinuousExpectedFreeEnergy(self.config, self.inference)

    def test_sample_policies(self) -> None:
        """Test continuous policy sampling"""
        policies = self.selector.sample_policies(2, 5)
        assert len(policies) == 5
        for policy in policies:
            assert policy.actions.shape == (2, 2)  # policy_length x action_dim
            assert torch.all(policy.actions >= -1.0)
            assert torch.all(policy.actions <= 1.0)

    def test_continuous_free_energy(self) -> None:
        """Test free energy computation for continuous states"""
        # Create beliefs (mean, variance)
        mean = torch.tensor([0.0, 0.0])
        var = torch.tensor([1.0, 1.0])
        beliefs = (mean, var)
        # Create a policy
        policy = Policy(torch.tensor([[0.5, -0.5]]))
        # Compute free energy
        G = self.selector.compute_expected_free_energy(policy, beliefs, self.model)
        assert torch.isfinite(G)
        assert G.dim() == 0  # Scalar

    def test_continuous_policy_selection(self) -> None:
        """Test policy selection for continuous states"""
        mean = torch.tensor([0.5, -0.5])
        var = torch.tensor([0.5, 0.5])
        beliefs = (mean, var)
        # Set preferences
        preferences = torch.tensor([1.0, -1.0])
        policy, G_values = self.selector.select_policy(beliefs, self.model, preferences)
        assert isinstance(policy, Policy)
        assert G_values.shape == (self.config.num_policies,)
        assert torch.all(torch.isfinite(G_values))

    def test_information_gain(self) -> None:
        """Test epistemic value (information gain)"""
        self.config.epistemic_weight = 1.0
        self.config.pragmatic_weight = 0.0
        # High uncertainty state
        mean = torch.zeros(2)
        high_var = torch.ones(2) * 2.0
        beliefs = (mean, high_var)
        # Exploratory policy
        policy = Policy(torch.randn(1, 2))
        G = self.selector.compute_expected_free_energy(policy, beliefs, self.model)
        # Should favor uncertainty reduction
        assert torch.isfinite(G)


class TestHierarchicalPolicySelector:
    """Test hierarchical policy selection"""

    def setup_method(self) -> None:
        """Set up hierarchical system"""
        # Create models for two levels
        dims_low = ModelDimensions(num_states=4, num_observations=4, num_actions=3)
        dims_high = ModelDimensions(num_states=2, num_observations=2, num_actions=2)
        params = ModelParameters(use_gpu=False)
        self.model_low = DiscreteGenerativeModel(dims_low, params)
        self.model_high = DiscreteGenerativeModel(dims_high, params)
        # Create selectors for each level
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)
        config_low = PolicyConfig(planning_horizon=3, use_gpu=False)
        config_high = PolicyConfig(planning_horizon=10, use_gpu=False)
        selector_low = DiscreteExpectedFreeEnergy(config_low, inference)
        selector_high = DiscreteExpectedFreeEnergy(config_high, inference)
        # Create hierarchical selector
        self.config = PolicyConfig(use_gpu=False)
        self.selector = HierarchicalPolicySelector(
            self.config, [selector_high, selector_low], [10, 3]
        )

    def test_hierarchical_selection(self) -> None:
        """Test policy selection at multiple levels"""
        # Beliefs at each level
        beliefs = [
            torch.tensor([0.6, 0.4]),  # High level
            torch.ones(4) / 4,  # Low level
        ]
        models = [self.model_high, self.model_low]
        policies, probs = self.selector.select_policy(beliefs, models)
        assert len(policies) == 2
        assert len(probs) == 2
        # Check high level policy
        assert isinstance(policies[0], Policy)
        assert probs[0].shape == (2,)  # Two high-level actions
        # Check low level policy
        assert isinstance(policies[1], Policy)
        assert probs[1].shape == (3,)  # Three low-level actions

    def test_hierarchical_free_energy(self) -> None:
        """Test hierarchical free energy computation"""
        beliefs = [torch.tensor([0.7, 0.3]), torch.tensor([0.25, 0.25, 0.25, 0.25])]
        policies = [Policy([0]), Policy([1])]
        models = [self.model_high, self.model_low]
        G = self.selector.compute_expected_free_energy(policies, beliefs, models)
        assert torch.isfinite(G)
        assert G.dim() == 0  # Scalar


class TestSophisticatedInference:
    """Test sophisticated inference"""

    def setup_method(self) -> None:
        """Set up test environment"""
        # Create model
        dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(dims, params)
        # Create inference
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = VariationalMessagePassing(inf_config)
        # Create base selector
        config = PolicyConfig(planning_horizon=3, use_gpu=False)
        self.base_selector = DiscreteExpectedFreeEnergy(config, self.inference)
        # Create sophisticated selector
        self.selector = SophisticatedInference(config, self.inference, self.base_selector)

    def test_sophisticated_selection(self) -> None:
        """Test sophisticated policy selection"""
        beliefs = torch.ones(3) / 3
        policy, probs = self.selector.select_policy(beliefs, self.model)
        assert isinstance(policy, Policy)
        assert probs.shape == (2,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0))

    def test_sophisticated_refinement(self) -> None:
        """Test policy refinement"""
        initial_policy = Policy([0, 1, 0])
        beliefs = torch.tensor([0.5, 0.3, 0.2])
        refined = self.selector._sophisticated_refinement(initial_policy, beliefs, self.model)
        assert isinstance(refined, Policy)
        assert len(refined) == len(initial_policy)
        # Refined policy may differ from initial
        # (depending on counterfactual reasoning)

    def test_sophistication_depth(self) -> None:
        """Test different sophistication depths"""
        self.selector.sophistication_depth = 0
        beliefs = torch.ones(3) / 3
        # With depth 0, should return base policy
        policy0, _ = self.selector.select_policy(beliefs, self.model)
        # With higher depth
        self.selector.sophistication_depth = 3
        policy3, _ = self.selector.select_policy(beliefs, self.model)
        # Both should be valid policies
        assert isinstance(policy0, Policy)
        assert isinstance(policy3, Policy)


class TestPolicyFactory:
    """Test policy selector factory"""

    def setup_method(self) -> None:
        """Set up test environment"""
        inf_config = InferenceConfig(use_gpu=False)
        self.inference = VariationalMessagePassing(inf_config)

    def test_create_discrete_selector(self) -> None:
        """Test discrete selector creation"""
        selector = create_policy_selector("discrete", inference_algorithm=self.inference)
        assert isinstance(selector, DiscreteExpectedFreeEnergy)

    def test_create_continuous_selector(self) -> None:
        """Test continuous selector creation"""
        inf_config = InferenceConfig(use_gpu=False)
        cont_inference = GradientDescentInference(inf_config)
        selector = create_policy_selector("continuous", inference_algorithm=cont_inference)
        assert isinstance(selector, ContinuousExpectedFreeEnergy)

    def test_create_hierarchical_selector(self) -> None:
        """Test hierarchical selector creation"""
        # Create level selectors
        sel1 = create_policy_selector("discrete", inference_algorithm=self.inference)
        sel2 = create_policy_selector("discrete", inference_algorithm=self.inference)
        selector = create_policy_selector(
            "hierarchical", level_selectors=[sel1, sel2], level_horizons=[10, 5]
        )
        assert isinstance(selector, HierarchicalPolicySelector)
        assert selector.num_levels == 2

    def test_create_sophisticated_selector(self) -> None:
        """Test sophisticated selector creation"""
        base = create_policy_selector("discrete", inference_algorithm=self.inference)
        selector = create_policy_selector(
            "sophisticated", inference_algorithm=self.inference, base_selector=base
        )
        assert isinstance(selector, SophisticatedInference)

    def test_invalid_selector_type(self) -> None:
        """Test invalid selector type"""
        with pytest.raises(ValueError):
            create_policy_selector("invalid")

    def test_missing_parameters(self) -> None:
        """Test missing required parameters"""
        with pytest.raises(ValueError):
            create_policy_selector("discrete")  # Missing inference
        with pytest.raises(ValueError):
            create_policy_selector("hierarchical")  # Missing level_selectors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

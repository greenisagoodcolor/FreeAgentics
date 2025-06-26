import os
import sys

import pytest
import torch

from inference.engine.generative_model import (
    ContinuousGenerativeModel,
    DiscreteGenerativeModel,
    FactorizedGenerativeModel,
    HierarchicalGenerativeModel,
    ModelDimensions,
    ModelParameters,
    create_generative_model,
)


class TestModelDimensions:
    """Test ModelDimensions dataclass"""

    def test_basic_dimensions(self) -> None:
        """Test basic dimension initialization"""
        dims = ModelDimensions(num_states=10, num_observations=8, num_actions=4)
        assert dims.num_states == 10
        assert dims.num_observations == 8
        assert dims.num_actions == 4
        assert dims.num_modalities == 1  # Default
        assert dims.num_factors == 1  # Default
        assert dims.time_horizon == 1  # Default

    def test_full_dimensions(self) -> None:
        """Test full dimension specification"""
        dims = ModelDimensions(
            num_states=20,
            num_observations=15,
            num_actions=5,
            num_modalities=2,
            num_factors=3,
            time_horizon=10,
        )
        assert dims.num_modalities == 2
        assert dims.num_factors == 3
        assert dims.time_horizon == 10


class TestModelParameters:
    """Test ModelParameters dataclass"""

    def test_default_parameters(self) -> None:
        """Test default parameter values"""
        params = ModelParameters()
        assert params.learning_rate == 0.01
        assert params.precision_init == 1.0
        assert params.use_sparse is False
        assert params.use_gpu is True
        assert params.dtype == torch.float32
        assert params.eps == 1e-8
        assert params.temperature == 1.0

    def test_custom_parameters(self) -> None:
        """Test custom parameter values"""
        params = ModelParameters(learning_rate=0.1, use_sparse=True, use_gpu=False, temperature=0.5)
        assert params.learning_rate == 0.1
        assert params.use_sparse is True
        assert params.use_gpu is False
        assert params.temperature == 0.5


class TestDiscreteGenerativeModel:
    """Test discrete generative model"""

    def setup_method(self) -> None:
        """Set up test model"""
        self.dims = ModelDimensions(num_states=5, num_observations=4, num_actions=3, time_horizon=5)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)

    def test_initialization(self) -> None:
        """Test model initialization"""
        # Check dimensions
        assert self.model.A.shape == (4, 5)  # obs x states
        assert self.model.B.shape == (5, 5, 3)  # states x states x actions
        assert self.model.C.shape == (4, 5)  # obs x time
        assert self.model.D.shape == (5,)  # states
        # Check normalization
        assert torch.allclose(self.model.A.sum(dim=0), torch.ones(5))
        assert torch.allclose(self.model.B.sum(dim=0), torch.ones(5, 3))
        assert torch.allclose(self.model.D.sum(), torch.tensor(1.0))

    def test_observation_model(self) -> None:
        """Test observation model computation"""
        # Single state
        state = torch.tensor([0.2, 0.3, 0.1, 0.3, 0.1])
        obs = self.model.observation_model(state)
        assert obs.shape == (4,)
        assert torch.allclose(obs.sum(), torch.tensor(1.0), atol=1e-6)
        # Batch of states
        states = torch.rand(10, 5)
        states = states / states.sum(dim=1, keepdim=True)
        obs_batch = self.model.observation_model(states)
        assert obs_batch.shape == (10, 4)
        assert torch.allclose(obs_batch.sum(dim=1), torch.ones(10), atol=1e-6)

    def test_transition_model(self) -> None:
        """Test transition model computation"""
        # Single state and action
        state = torch.tensor([0.2, 0.3, 0.1, 0.3, 0.1])
        next_state = self.model.transition_model(state, action=0)
        assert next_state.shape == (5,)
        assert torch.allclose(next_state.sum(), torch.tensor(1.0), atol=1e-6)
        # Batch processing
        states = torch.rand(10, 5)
        states = states / states.sum(dim=1, keepdim=True)
        actions = torch.randint(0, 3, (10,))
        next_states = self.model.transition_model(states, actions)
        assert next_states.shape == (10, 5)
        assert torch.allclose(next_states.sum(dim=1), torch.ones(10), atol=1e-6)

    def test_preferences(self) -> None:
        """Test preference setting and retrieval"""
        # Set preferences
        prefs = torch.tensor([0.0, 0.0, 10.0, 0.0])  # Prefer observation 2
        self.model.set_preferences(prefs)
        # Check all timesteps updated
        for t in range(self.dims.time_horizon):
            assert torch.allclose(self.model.get_preferences(t), prefs)
        # Set timestep-specific preference
        prefs_t2 = torch.tensor([5.0, 0.0, 0.0, 5.0])
        self.model.set_preferences(prefs_t2, timestep=2)
        assert torch.allclose(self.model.get_preferences(2), prefs_t2)
        assert torch.allclose(self.model.get_preferences(1), prefs)  # Others unchanged

    def test_model_update(self) -> None:
        """Test model parameter updates"""
        # Generate fake trajectory
        observations = [torch.tensor(i % 4) for i in range(10)]
        states = [torch.rand(5) for _ in range(10)]
        states = [s / s.sum() for s in states]
        actions = [i % 3 for i in range(9)]
        # Store original parameters
        A_orig = self.model.A.clone()
        B_orig = self.model.B.clone()
        # Update model
        self.model.update_model(observations, states, actions)
        # Check parameters changed
        assert not torch.allclose(self.model.A, A_orig)
        assert not torch.allclose(self.model.B, B_orig)
        # Check still normalized
        assert torch.allclose(self.model.A.sum(dim=0), torch.ones(5))
        assert torch.allclose(self.model.B.sum(dim=0), torch.ones(5, 3))


class TestContinuousGenerativeModel:
    """Test continuous generative model"""

    def setup_method(self) -> None:
        """Set up test model"""
        self.dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = ContinuousGenerativeModel(self.dims, self.params, hidden_dim=32)

    def test_initialization(self) -> None:
        """Test model initialization"""
        # Check network components exist
        assert hasattr(self.model, "obs_net")
        assert hasattr(self.model, "trans_net")
        assert hasattr(self.model, "C")
        assert hasattr(self.model, "D_mean")
        assert hasattr(self.model, "D_log_var")
        # Check dimensions
        assert self.model.C.shape == (3, 1)  # Default time horizon
        assert self.model.D_mean.shape == (4,)
        assert self.model.D_log_var.shape == (4,)

    def test_observation_model(self) -> None:
        """Test observation model"""
        # Single state
        state = torch.randn(4)
        obs_mean, obs_var = self.model.observation_model(state)
        assert obs_mean.shape == (3,)
        assert obs_var.shape == (3,)
        assert torch.all(obs_var > 0)  # Positive variance
        # Batch of states
        states = torch.randn(10, 4)
        obs_mean_batch, obs_var_batch = self.model.observation_model(states)
        assert obs_mean_batch.shape == (10, 3)
        assert torch.all(obs_var_batch > 0)

    def test_transition_model(self) -> None:
        """Test transition model"""
        # Single state and action
        state = torch.randn(4)
        action = torch.tensor([0])
        next_mean, next_var = self.model.transition_model(state, action)
        assert next_mean.shape == (4,)
        assert next_var.shape == (4,)
        assert torch.all(next_var > 0)
        # With one-hot action
        action_onehot = torch.tensor([1.0, 0.0])
        next_mean2, next_var2 = self.model.transition_model(state, action_onehot)
        assert next_mean2.shape == (4,)

    def test_forward_pass(self) -> None:
        """Test complete forward pass"""
        states = torch.randn(5, 4)
        actions = torch.randint(0, 2, (5, 1))
        outputs = self.model.forward(states, actions)
        assert "obs_mean" in outputs
        assert "obs_var" in outputs
        assert "next_mean" in outputs
        assert "next_var" in outputs
        assert outputs["obs_mean"].shape == (5, 3)
        assert outputs["next_mean"].shape == (5, 4)


class TestHierarchicalGenerativeModel:
    """Test hierarchical generative model"""

    def setup_method(self) -> None:
        """Set up test model"""
        # Define 3-level hierarchy
        self.dims_list = [
            ModelDimensions(num_states=8, num_observations=6, num_actions=4),  # Level 0
            ModelDimensions(num_states=4, num_observations=3, num_actions=2),  # Level 1
            ModelDimensions(num_states=2, num_observations=2, num_actions=1),  # Level 2
        ]
        self.params = ModelParameters(use_gpu=False)
        self.model = HierarchicalGenerativeModel(self.dims_list, self.params)

    def test_initialization(self) -> None:
        """Test hierarchical model initialization"""
        assert self.model.num_levels == 3
        assert len(self.model.levels) == 3
        # Check each level
        for i, level in enumerate(self.model.levels):
            assert isinstance(level, DiscreteGenerativeModel)
            assert level.dims.num_states == self.dims_list[i].num_states
        # Check E matrices
        assert (0, 1) in self.model.E_matrices
        assert (1, 2) in self.model.E_matrices
        assert self.model.E_matrices[(0, 1)].shape == (4, 8)
        assert self.model.E_matrices[(1, 2)].shape == (2, 4)

    def test_hierarchical_observation_model(self) -> None:
        """Test hierarchical observation computation"""
        # Create state distributions for each level
        states = [torch.ones(8) / 8, torch.ones(4) / 4, torch.ones(2) / 2]
        observations = self.model.hierarchical_observation_model(states)
        assert len(observations) == 3
        assert observations[0].shape == (6,)
        assert observations[1].shape == (3,)
        assert observations[2].shape == (2,)
        # Check normalization
        for obs in observations:
            assert torch.allclose(obs.sum(), torch.tensor(1.0), atol=1e-6)

    def test_hierarchical_transition_model(self) -> None:
        """Test hierarchical state transitions"""
        states = [torch.ones(8) / 8, torch.ones(4) / 4, torch.ones(2) / 2]
        actions = [0, 1, 0]
        next_states = self.model.hierarchical_transition_model(states, actions)
        assert len(next_states) == 3
        for i, next_state in enumerate(next_states):
            assert next_state.shape == states[i].shape
            assert torch.allclose(next_state.sum(), torch.tensor(1.0), atol=1e-6)


class TestFactorizedGenerativeModel:
    """Test factorized generative model"""

    def setup_method(self) -> None:
        """Set up test model"""
        self.factor_dims = [3, 4, 2]  # 3 factors with different dimensions
        self.num_obs = 10
        self.num_actions = 4
        self.params = ModelParameters(use_gpu=False)
        self.model = FactorizedGenerativeModel(
            self.factor_dims, self.num_obs, self.num_actions, self.params
        )

    def test_initialization(self) -> None:
        """Test factorized model initialization"""
        assert self.model.num_factors == 3
        assert self.model.dims.num_states == 24  # 3 * 4 * 2
        assert len(self.model.factor_B) == 3
        # Check factor transition models
        assert self.model.factor_B[0].shape == (3, 3, 4)
        assert self.model.factor_B[1].shape == (4, 4, 4)
        assert self.model.factor_B[2].shape == (2, 2, 4)

    def test_factor_index_conversion(self) -> None:
        """Test conversion between factor and state indices"""
        # Test factor to state
        factor_idx = [1, 2, 0]
        state_idx = self.model.factor_to_state_idx(factor_idx)
        assert state_idx == 1 * 8 + 2 * 2 + 0  # = 12
        # Test state to factor
        factor_idx_back = self.model.state_to_factor_idx(state_idx)
        assert factor_idx_back == factor_idx
        # Test all possible combinations
        for i in range(3):
            for j in range(4):
                for k in range(2):
                    idx = [i, j, k]
                    state = self.model.factor_to_state_idx(idx)
                    idx_back = self.model.state_to_factor_idx(state)
                    assert idx == idx_back

    def test_factorized_transition(self) -> None:
        """Test factorized state transitions"""
        # Create factor states
        factor_states = [torch.ones(3) / 3, torch.ones(4) / 4, torch.ones(2) / 2]
        next_factor_states = self.model.factorized_transition(factor_states, action=0)
        assert len(next_factor_states) == 3
        assert next_factor_states[0].shape == (3,)
        assert next_factor_states[1].shape == (4,)
        assert next_factor_states[2].shape == (2,)
        # Check normalization
        for state in next_factor_states:
            assert torch.allclose(state.sum(), torch.tensor(1.0), atol=1e-6)


class TestModelFactory:
    """Test model factory function"""

    def test_create_discrete_model(self) -> None:
        """Test discrete model creation"""
        dims = ModelDimensions(num_states=10, num_observations=8, num_actions=4)
        model = create_generative_model("discrete", dimensions=dims)
        assert isinstance(model, DiscreteGenerativeModel)
        assert model.dims.num_states == 10

    def test_create_continuous_model(self) -> None:
        """Test continuous model creation"""
        dims = ModelDimensions(num_states=5, num_observations=3, num_actions=2)
        model = create_generative_model("continuous", dimensions=dims, hidden_dim=64)
        assert isinstance(model, ContinuousGenerativeModel)
        assert model.hidden_dim == 64

    def test_create_hierarchical_model(self) -> None:
        """Test hierarchical model creation"""
        dims_list = [
            ModelDimensions(num_states=8, num_observations=6, num_actions=4),
            ModelDimensions(num_states=4, num_observations=3, num_actions=2),
        ]
        model = create_generative_model("hierarchical", dimensions_list=dims_list)
        assert isinstance(model, HierarchicalGenerativeModel)
        assert model.num_levels == 2

    def test_create_factorized_model(self) -> None:
        """Test factorized model creation"""
        model = create_generative_model(
            "factorized", factor_dimensions=[3, 4], num_observations=10, num_actions=3
        )
        assert isinstance(model, FactorizedGenerativeModel)
        assert model.num_factors == 2

    def test_invalid_model_type(self) -> None:
        """Test invalid model type raises error"""
        with pytest.raises(ValueError):
            create_generative_model("invalid_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

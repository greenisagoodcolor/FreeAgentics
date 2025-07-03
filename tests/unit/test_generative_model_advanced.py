"""
Comprehensive test suite for agents.active_inference.generative_model module.
Covers all classes and functions with edge cases and mathematical validation.
"""

import pytest
import torch
import torch.nn as nn

from agents.active_inference.generative_model import (
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

    def test_model_dimensions_creation(self):
        """Test basic model dimensions creation"""
        dims = ModelDimensions(
            num_states=5,
            num_observations=3,
            num_actions=4,
            num_modalities=2,
            num_factors=3,
            time_horizon=10,
        )
        assert dims.num_states == 5
        assert dims.num_observations == 3
        assert dims.num_actions == 4
        assert dims.num_modalities == 2
        assert dims.num_factors == 3
        assert dims.time_horizon == 10

    def test_model_dimensions_defaults(self):
        """Test model dimensions with default values"""
        dims = ModelDimensions(num_states=5, num_observations=3, num_actions=4)
        assert dims.num_states == 5
        assert dims.num_observations == 3
        assert dims.num_actions == 4
        assert dims.num_modalities == 1  # default
        assert dims.num_factors == 1  # default
        assert dims.time_horizon == 1  # default

    def test_model_dimensions_edge_cases(self):
        """Test model dimensions with edge cases"""
        # Minimal dimensions
        dims = ModelDimensions(num_states=1, num_observations=1, num_actions=1)
        assert dims.num_states == 1
        assert dims.num_observations == 1
        assert dims.num_actions == 1

        # Large dimensions
        dims = ModelDimensions(num_states=1000, num_observations=500, num_actions=100)
        assert dims.num_states == 1000
        assert dims.num_observations == 500
        assert dims.num_actions == 100


class TestModelParameters:
    """Test ModelParameters dataclass"""

    def test_model_parameters_creation(self):
        """Test basic model parameters creation"""
        params = ModelParameters(
            learning_rate=0.001,
            precision_init=2.0,
            use_sparse=True,
            use_gpu=False,
            dtype=torch.float64,
            eps=1e-10,
            temperature=0.5,
        )
        assert params.learning_rate == 0.001
        assert params.precision_init == 2.0
        assert params.use_sparse is True
        assert params.use_gpu is False
        assert params.dtype == torch.float64
        assert params.eps == 1e-10
        assert params.temperature == 0.5

    def test_model_parameters_defaults(self):
        """Test model parameters with default values"""
        params = ModelParameters()
        assert params.learning_rate == 0.01
        assert params.precision_init == 1.0
        assert params.use_sparse is False
        assert params.use_gpu is True
        assert params.dtype == torch.float32
        assert params.eps == 1e-8
        assert params.temperature == 1.0

    def test_model_parameters_combinations(self):
        """Test various parameter combinations"""
        # High precision, high learning rate
        params = ModelParameters(learning_rate=0.1, precision_init=10.0)
        assert params.learning_rate == 0.1
        assert params.precision_init == 10.0

        # Low precision, low learning rate
        params = ModelParameters(learning_rate=0.0001, precision_init=0.1)
        assert params.learning_rate == 0.0001
        assert params.precision_init == 0.1


class TestDiscreteGenerativeModel:
    """Test DiscreteGenerativeModel class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2, time_horizon=5)
        self.params = ModelParameters()

    def test_initialization(self):
        """Test model initialization"""
        model = DiscreteGenerativeModel(self.dims, self.params)

        assert model.dims == self.dims
        assert model.params == self.params

        # Check matrix shapes
        assert model.A.shape == (3, 4)  # observations x states
        assert model.B.shape == (4, 4, 2)  # states x states x actions
        assert model.C.shape == (3, 5)  # observations x time_horizon
        assert model.D.shape == (4,)  # states

        # Check normalizations
        assert torch.allclose(model.A.sum(dim=0), torch.ones(4))
        assert torch.allclose(model.B.sum(dim=0), torch.ones(4, 2))
        assert torch.allclose(model.D.sum(), torch.tensor(1.0))

    def test_observation_model_single_state(self):
        """Test observation model with single state"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        state = torch.tensor([0.5, 0.3, 0.2, 0.0])

        obs_probs = model.observation_model(state)

        assert obs_probs.shape == (3,)
        assert torch.all(obs_probs >= 0)
        assert torch.allclose(obs_probs, torch.matmul(model.A, state))

    def test_observation_model_batch_states(self):
        """Test observation model with batch of states"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        states = torch.tensor(
            [[0.5, 0.3, 0.2, 0.0], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]]
        )

        obs_probs = model.observation_model(states)

        assert obs_probs.shape == (3, 3)  # batch_size x observations
        assert torch.all(obs_probs >= 0)
        assert torch.allclose(obs_probs, torch.matmul(states, model.A.T))

    def test_transition_model_single_state_int_action(self):
        """Test transition model with single state and integer action"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        state = torch.tensor([0.4, 0.3, 0.2, 0.1])
        action = 1

        next_state = model.transition_model(state, action)

        assert next_state.shape == (4,)
        assert torch.all(next_state >= 0)
        assert torch.allclose(next_state, torch.matmul(model.B[:, :, action], state))

    def test_transition_model_single_state_tensor_action(self):
        """Test transition model with single state and tensor action"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        state = torch.tensor([0.4, 0.3, 0.2, 0.1])
        action = torch.tensor([0])

        next_state = model.transition_model(state, action)

        assert next_state.shape == (4,)
        assert torch.all(next_state >= 0)

    def test_transition_model_single_state_onehot_action(self):
        """Test transition model with single state and one-hot action"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        state = torch.tensor([0.4, 0.3, 0.2, 0.1])
        action = torch.tensor([0.0, 1.0])  # one-hot for action 1

        next_state = model.transition_model(state, action)

        assert next_state.shape == (4,)
        assert torch.all(next_state >= 0)
        assert torch.allclose(next_state, torch.matmul(model.B[:, :, 1], state))

    def test_transition_model_batch_states(self):
        """Test transition model with batch of states"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        states = torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.25, 0.25, 0.25, 0.25]])
        actions = torch.tensor([0, 1])

        next_states = model.transition_model(states, actions)

        assert next_states.shape == (2, 4)
        assert torch.all(next_states >= 0)

    def test_transition_model_batch_onehot_actions(self):
        """Test transition model with batch of one-hot actions"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        states = torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.25, 0.25, 0.25, 0.25]])
        actions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # action 0  # action 1

        next_states = model.transition_model(states, actions)

        assert next_states.shape == (2, 4)
        assert torch.all(next_states >= 0)

    def test_set_preferences_all_timesteps(self):
        """Test setting preferences for all timesteps"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        preferences = torch.tensor([1.0, -0.5, 0.0])

        model.set_preferences(preferences)

        for t in range(self.dims.time_horizon):
            assert torch.allclose(model.C[:, t], preferences)

    def test_set_preferences_specific_timestep(self):
        """Test setting preferences for specific timestep"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        preferences = torch.tensor([2.0, -1.0, 0.5])
        timestep = 2

        model.set_preferences(preferences, timestep)

        assert torch.allclose(model.C[:, timestep], preferences)
        # Other timesteps should remain zero
        for t in range(self.dims.time_horizon):
            if t != timestep:
                assert torch.allclose(model.C[:, t], torch.zeros(3))

    def test_get_preferences(self):
        """Test getting preferences for timestep"""
        model = DiscreteGenerativeModel(self.dims, self.params)
        preferences = torch.tensor([1.5, -2.0, 1.0])
        timestep = 3

        model.set_preferences(preferences, timestep)
        retrieved_prefs = model.get_preferences(timestep)

        assert torch.allclose(retrieved_prefs, preferences)

    def test_update_model(self):
        """Test model parameter updates"""
        model = DiscreteGenerativeModel(self.dims, self.params)

        # Store original matrices
        A_orig = model.A.clone()
        B_orig = model.B.clone()

        # Mock inputs for update
        observations = torch.randn(5, 3)
        states = torch.randn(5, 4)
        actions = torch.randint(0, 2, (5,))

        model.update_model(observations, states, actions)

        # Matrices should have changed
        assert not torch.allclose(model.A, A_orig)
        assert not torch.allclose(model.B, B_orig)

        # Normalizations should be preserved
        assert torch.allclose(model.A.sum(dim=0), torch.ones(4), atol=1e-6)
        assert torch.allclose(model.B.sum(dim=0), torch.ones(4, 2), atol=1e-6)


class TestContinuousGenerativeModel:
    """Test ContinuousGenerativeModel class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.dims = ModelDimensions(num_states=3, num_observations=2, num_actions=2, time_horizon=4)
        self.params = ModelParameters()

    def test_initialization(self):
        """Test model initialization"""
        model = ContinuousGenerativeModel(self.dims, self.params, hidden_dim=16)

        assert model.dims == self.dims
        assert model.params == self.params
        assert model.hidden_dim == 16

        # Check network architectures
        assert isinstance(model.obs_net, nn.Sequential)
        assert isinstance(model.trans_net, nn.Sequential)

        # Check preference and prior shapes
        assert model.C.shape == (2, 4)  # observations x time_horizon
        assert model.D_mean.shape == (3,)  # states
        assert model.D_log_var.shape == (3,)  # states

    def test_observation_model_single_state(self):
        """Test observation model with single state"""
        model = ContinuousGenerativeModel(self.dims, self.params)
        state = torch.randn(3)

        mean, var = model.observation_model(state)

        assert mean.shape == (2,)
        assert var.shape == (2,)
        assert torch.all(var > 0)  # Variance should be positive

    def test_observation_model_batch_states(self):
        """Test observation model with batch of states"""
        model = ContinuousGenerativeModel(self.dims, self.params)
        states = torch.randn(5, 3)

        mean, var = model.observation_model(states)

        assert mean.shape == (5, 2)
        assert var.shape == (5, 2)
        assert torch.all(var > 0)

    def test_transition_model_single_state_int_action(self):
        """Test transition model with single state and integer action"""
        model = ContinuousGenerativeModel(self.dims, self.params)
        state = torch.randn(3)
        action = 1

        mean, var = model.transition_model(state, action)

        assert mean.shape == (3,)
        assert var.shape == (3,)
        assert torch.all(var > 0)

    def test_transition_model_single_state_tensor_action(self):
        """Test transition model with single state and tensor action"""
        model = ContinuousGenerativeModel(self.dims, self.params)
        state = torch.randn(3)
        action = torch.tensor([0])

        mean, var = model.transition_model(state, action)

        assert mean.shape == (3,)
        assert var.shape == (3,)
        assert torch.all(var > 0)

    def test_transition_model_batch_states_onehot_actions(self):
        """Test transition model with batch states and one-hot actions"""
        model = ContinuousGenerativeModel(self.dims, self.params)
        states = torch.randn(4, 3)
        actions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        mean, var = model.transition_model(states, actions)

        assert mean.shape == (4, 3)
        assert var.shape == (4, 3)
        assert torch.all(var > 0)

    def test_transition_model_batch_states_indices(self):
        """Test transition model with batch states and action indices"""
        model = ContinuousGenerativeModel(self.dims, self.params)
        states = torch.randn(3, 3)
        # Action indices as column vector
        actions = torch.tensor([[0], [1], [0]])

        mean, var = model.transition_model(states, actions)

        assert mean.shape == (3, 3)
        assert var.shape == (3, 3)
        assert torch.all(var > 0)

    def test_transition_model_batch_size_mismatch_broadcast(self):
        """Test transition model with mismatched batch sizes (broadcasting)"""
        model = ContinuousGenerativeModel(self.dims, self.params)
        state = torch.randn(1, 3)  # Single state
        actions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Two actions

        mean, var = model.transition_model(state, actions)

        assert mean.shape == (2, 3)
        assert var.shape == (2, 3)

    def test_transition_model_batch_size_incompatible(self):
        """Test transition model with incompatible batch sizes"""
        model = ContinuousGenerativeModel(self.dims, self.params)
        states = torch.randn(3, 3)  # 3 states
        actions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 2 actions

        with pytest.raises(ValueError, match="Batch sizes.*are incompatible"):
            model.transition_model(states, actions)

    def test_forward_pass(self):
        """Test forward pass"""
        model = ContinuousGenerativeModel(self.dims, self.params)
        states = torch.randn(2, 3)
        actions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        output = model.forward(states, actions)

        assert "obs_mean" in output
        assert "obs_var" in output
        assert "next_mean" in output
        assert "next_var" in output

        assert output["obs_mean"].shape == (2, 2)
        assert output["obs_var"].shape == (2, 2)
        assert output["next_mean"].shape == (2, 3)
        assert output["next_var"].shape == (2, 3)

        assert torch.all(output["obs_var"] > 0)
        assert torch.all(output["next_var"] > 0)


class TestHierarchicalGenerativeModel:
    """Test HierarchicalGenerativeModel class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.dims_list = [
            ModelDimensions(num_states=3, num_observations=2, num_actions=2),
            ModelDimensions(num_states=4, num_observations=3, num_actions=3),
            ModelDimensions(num_states=2, num_observations=2, num_actions=2),
        ]
        self.params = ModelParameters()

    def test_initialization(self):
        """Test hierarchical model initialization"""
        model = HierarchicalGenerativeModel(self.dims_list, self.params)

        assert model.dims_list == self.dims_list
        assert model.params == self.params
        assert model.num_levels == 3
        assert len(model.levels) == 3

        # Check that each level is a DiscreteGenerativeModel
        for level in model.levels:
            assert isinstance(level, DiscreteGenerativeModel)

        # Check E matrices (inter-level connections)
        assert len(model.E_matrices) == 2  # 3 levels = 2 connections
        assert (0, 1) in model.E_matrices
        assert (1, 2) in model.E_matrices

        # Check E matrix shapes
        assert model.E_matrices[(0, 1)].shape == (2, 3)  # lower_actions x upper_states
        assert model.E_matrices[(1, 2)].shape == (3, 4)

    def test_hierarchical_observation_model(self):
        """Test hierarchical observation model"""
        model = HierarchicalGenerativeModel(self.dims_list, self.params)

        states = [
            torch.tensor([0.3, 0.4, 0.3]),
            torch.tensor([0.25, 0.25, 0.25, 0.25]),
            torch.tensor([0.6, 0.4]),
        ]

        observations = model.hierarchical_observation_model(states)

        assert len(observations) == 3
        assert observations[0].shape == (2,)  # level 0: 2 observations
        assert observations[1].shape == (3,)  # level 1: 3 observations
        assert observations[2].shape == (2,)  # level 2: 2 observations

    def test_hierarchical_transition_model(self):
        """Test hierarchical transition model"""
        model = HierarchicalGenerativeModel(self.dims_list, self.params)

        states = [
            torch.tensor([0.3, 0.4, 0.3]),
            torch.tensor([0.25, 0.25, 0.25, 0.25]),
            torch.tensor([0.6, 0.4]),
        ]
        actions = [0, 1, 0]

        next_states = model.hierarchical_transition_model(states, actions)

        assert len(next_states) == 3
        assert next_states[0].shape == (3,)  # level 0: 3 states
        assert next_states[1].shape == (4,)  # level 1: 4 states
        assert next_states[2].shape == (2,)  # level 2: 2 states

    def test_empty_hierarchy(self):
        """Test empty hierarchy"""
        with pytest.raises(IndexError):
            HierarchicalGenerativeModel([], self.params)

    def test_single_level_hierarchy(self):
        """Test single level hierarchy"""
        single_dims = [ModelDimensions(num_states=3, num_observations=2, num_actions=2)]
        model = HierarchicalGenerativeModel(single_dims, self.params)

        assert model.num_levels == 1
        assert len(model.E_matrices) == 0  # No inter-level connections


class TestFactorizedGenerativeModel:
    """Test FactorizedGenerativeModel class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.factor_dims = [3, 4, 2]  # 3 factors with dimensions 3, 4, 2
        self.num_obs = 5
        self.num_actions = 3
        self.params = ModelParameters()

    def test_initialization(self):
        """Test factorized model initialization"""
        model = FactorizedGenerativeModel(
            self.factor_dims, self.num_obs, self.num_actions, self.params
        )

        assert model.factor_dims == self.factor_dims
        assert model.num_factors == 3
        assert model.num_obs == self.num_obs
        assert model.num_actions == self.num_actions
        assert model.params == self.params

        # Check computed dimensions
        total_states = 3 * 4 * 2  # 24
        assert model.dims.num_states == total_states
        assert model.dims.num_observations == self.num_obs
        assert model.dims.num_actions == self.num_actions

        # Check factor-specific B matrices
        assert len(model.factor_B) == 3
        assert model.factor_B[0].shape == (3, 3, 3)  # factor_dim x factor_dim x actions
        assert model.factor_B[1].shape == (4, 4, 3)
        assert model.factor_B[2].shape == (2, 2, 3)

        # Check observation matrix
        assert model.A.shape == (5, 24)  # observations x total_states

        # Check normalizations
        for B_factor in model.factor_B:
            assert torch.allclose(B_factor.sum(dim=0), torch.ones(B_factor.shape[1:]))
        assert torch.allclose(model.A.sum(dim=0), torch.ones(24))

    def test_factor_to_state_idx(self):
        """Test converting factor indices to state index"""
        model = FactorizedGenerativeModel(
            self.factor_dims, self.num_obs, self.num_actions, self.params
        )

        # Test various factor combinations
        assert model.factor_to_state_idx([0, 0, 0]) == 0
        assert model.factor_to_state_idx([1, 0, 0]) == 8  # 1 * (4 * 2) + 0 * 2 + 0
        assert model.factor_to_state_idx([0, 1, 0]) == 2  # 0 * (4 * 2) + 1 * 2 + 0
        assert model.factor_to_state_idx([0, 0, 1]) == 1  # 0 * (4 * 2) + 0 * 2 + 1
        assert model.factor_to_state_idx([2, 3, 1]) == 23  # 2 * 8 + 3 * 2 + 1

    def test_state_to_factor_idx(self):
        """Test converting state index to factor indices"""
        model = FactorizedGenerativeModel(
            self.factor_dims, self.num_obs, self.num_actions, self.params
        )

        # Test various state indices
        assert model.state_to_factor_idx(0) == [0, 0, 0]
        assert model.state_to_factor_idx(1) == [0, 0, 1]
        assert model.state_to_factor_idx(2) == [0, 1, 0]
        assert model.state_to_factor_idx(8) == [1, 0, 0]
        assert model.state_to_factor_idx(23) == [2, 3, 1]

    def test_factor_state_conversion_consistency(self):
        """Test that factor-state conversions are consistent"""
        model = FactorizedGenerativeModel(
            self.factor_dims, self.num_obs, self.num_actions, self.params
        )

        # Test round-trip conversion for all possible states
        total_states = 3 * 4 * 2
        for state_idx in range(total_states):
            factor_indices = model.state_to_factor_idx(state_idx)
            recovered_state_idx = model.factor_to_state_idx(factor_indices)
            assert recovered_state_idx == state_idx

    def test_factorized_transition(self):
        """Test factorized transition computation"""
        model = FactorizedGenerativeModel(
            self.factor_dims, self.num_obs, self.num_actions, self.params
        )

        factor_states = [
            torch.tensor([0.5, 0.3, 0.2]),  # factor 0
            torch.tensor([0.25, 0.25, 0.25, 0.25]),  # factor 1
            torch.tensor([0.7, 0.3]),  # factor 2
        ]
        action = 1

        next_factor_states = model.factorized_transition(factor_states, action)

        assert len(next_factor_states) == 3
        assert next_factor_states[0].shape == (3,)
        assert next_factor_states[1].shape == (4,)
        assert next_factor_states[2].shape == (2,)

        # Check that each factor state is properly normalized (probability
        # distribution)
        for next_state in next_factor_states:
            assert torch.all(next_state >= 0)

    def test_edge_case_single_factor(self):
        """Test factorized model with single factor"""
        model = FactorizedGenerativeModel([5], 3, 2, self.params)

        assert model.num_factors == 1
        assert model.dims.num_states == 5
        assert len(model.factor_B) == 1
        assert model.factor_B[0].shape == (5, 5, 2)

    def test_edge_case_binary_factors(self):
        """Test factorized model with all binary factors"""
        model = FactorizedGenerativeModel([2, 2, 2], 4, 2, self.params)

        assert model.dims.num_states == 8  # 2^3
        assert all(B.shape[0] == 2 and B.shape[1] == 2 for B in model.factor_B)


class TestCreateGenerativeModel:
    """Test create_generative_model factory function"""

    def test_create_discrete_model(self):
        """Test creating discrete generative model"""
        dims = ModelDimensions(num_states=3, num_observations=2, num_actions=2)
        model = create_generative_model("discrete", dims=dims)

        assert isinstance(model, DiscreteGenerativeModel)
        assert model.dims == dims

    def test_create_discrete_model_with_parameters(self):
        """Test creating discrete model with custom parameters"""
        dims = ModelDimensions(num_states=3, num_observations=2, num_actions=2)
        params = ModelParameters(learning_rate=0.001)
        model = create_generative_model("discrete", dims=dims, params=params)

        assert isinstance(model, DiscreteGenerativeModel)
        assert model.params.learning_rate == 0.001

    def test_create_continuous_model(self):
        """Test creating continuous generative model"""
        dims = ModelDimensions(num_states=3, num_observations=2, num_actions=2)
        model = create_generative_model("continuous", dims=dims, hidden_dim=64)

        assert isinstance(model, ContinuousGenerativeModel)
        assert model.hidden_dim == 64

    def test_create_hierarchical_model(self):
        """Test creating hierarchical generative model"""
        dims_list = [
            ModelDimensions(num_states=3, num_observations=2, num_actions=2),
            ModelDimensions(num_states=4, num_observations=3, num_actions=3),
        ]
        model = create_generative_model("hierarchical", dims_list=dims_list)

        assert isinstance(model, HierarchicalGenerativeModel)
        assert len(model.levels) == 2

    def test_create_factorized_model(self):
        """Test creating factorized generative model"""
        model = create_generative_model("factorized", factor_dims=[3, 4], num_obs=5, num_actions=3)

        assert isinstance(model, FactorizedGenerativeModel)
        assert model.factor_dims == [3, 4]

    def test_create_model_alternative_kwargs(self):
        """Test creating model with alternative keyword names"""
        dims = ModelDimensions(num_states=3, num_observations=2, num_actions=2)

        # Test alternative names for dims
        model = create_generative_model("discrete", dimensions=dims)
        assert isinstance(model, DiscreteGenerativeModel)

    def test_create_factorized_model_alternative_kwargs(self):
        """Test creating factorized model with alternative keyword names"""
        model = create_generative_model(
            "factorized", factor_dimensions=[3, 4], num_observations=5, num_actions=3
        )

        assert isinstance(model, FactorizedGenerativeModel)
        assert model.factor_dims == [3, 4]

    def test_create_hierarchical_model_alternative_kwargs(self):
        """Test creating hierarchical model with alternative keyword names"""
        dims_list = [ModelDimensions(num_states=3, num_observations=2, num_actions=2)]
        model = create_generative_model("hierarchical", dimensions_list=dims_list)

        assert isinstance(model, HierarchicalGenerativeModel)

    def test_create_invalid_model_type(self):
        """Test creating model with invalid type"""
        with pytest.raises(ValueError, match="Invalid model type"):
            create_generative_model("invalid_type")

    def test_create_factorized_model_defaults(self):
        """Test creating factorized model with default values"""
        model = create_generative_model(
            "factorized",
            factor_dims=[3, 2],
            num_obs=4,
            # num_actions not specified, should default to 4
        )

        assert model.num_actions == 4


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""

    def test_discrete_model_zero_dimensions(self):
        """Test discrete model with edge case dimensions"""
        dims = ModelDimensions(num_states=1, num_observations=1, num_actions=1)
        model = DiscreteGenerativeModel(dims, ModelParameters())

        assert model.A.shape == (1, 1)
        assert model.B.shape == (1, 1, 1)
        assert model.D.shape == (1,)

    def test_continuous_model_large_dimensions(self):
        """Test continuous model with large dimensions"""
        dims = ModelDimensions(num_states=100, num_observations=50, num_actions=10)
        model = ContinuousGenerativeModel(dims, ModelParameters())

        # Should not raise errors
        state = torch.randn(100)
        mean, var = model.observation_model(state)
        assert mean.shape == (50,)
        assert var.shape == (50,)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        dims = ModelDimensions(num_states=3, num_observations=2, num_actions=2)
        model = DiscreteGenerativeModel(dims, ModelParameters())

        # Test with very small state probabilities
        state = torch.tensor([1e-10, 1e-10, 1.0 - 2e-10])
        obs_probs = model.observation_model(state)
        assert torch.all(torch.isfinite(obs_probs))

        # Test transition with extreme action probabilities
        action = torch.tensor([1.0 - 1e-10, 1e-10])
        next_state = model.transition_model(state, action)
        assert torch.all(torch.isfinite(next_state))

    def test_torch_dtype_consistency(self):
        """Test that models maintain dtype consistency"""
        params = ModelParameters(dtype=torch.float64)
        dims = ModelDimensions(num_states=3, num_observations=2, num_actions=2)

        # Note: The current implementation doesn't enforce dtype consistency
        # This test documents the current behavior and can be updated if dtype
        # enforcement is added
        model = DiscreteGenerativeModel(dims, params)

        # Model matrices will be in default dtype (float32) regardless of params.dtype
        # This is because the implementation doesn't currently use params.dtype
        assert model.A.dtype == torch.float32  # Current behavior

    def test_device_consistency(self):
        """Test model behavior with different devices"""
        dims = ModelDimensions(num_states=3, num_observations=2, num_actions=2)
        model = DiscreteGenerativeModel(dims, ModelParameters())

        # Test with CPU tensors (default)
        state = torch.tensor([0.5, 0.3, 0.2])
        obs_probs = model.observation_model(state)
        assert obs_probs.device == torch.device("cpu")

    def test_model_reproducibility(self):
        """Test that models produce consistent results with same initialization"""
        dims = ModelDimensions(num_states=3, num_observations=2, num_actions=2)

        # Set random seed for reproducibility
        torch.manual_seed(42)
        model1 = DiscreteGenerativeModel(dims, ModelParameters())

        torch.manual_seed(42)
        model2 = DiscreteGenerativeModel(dims, ModelParameters())

        # Models should have identical parameters
        assert torch.allclose(model1.A, model2.A)
        assert torch.allclose(model1.B, model2.B)
        assert torch.allclose(model1.D, model2.D)

    def test_large_factorized_model(self):
        """Test factorized model with many factors"""
        factor_dims = [2] * 10  # 10 binary factors = 1024 total states
        model = FactorizedGenerativeModel(factor_dims, 5, 2, ModelParameters())

        assert model.dims.num_states == 1024
        assert model.num_factors == 10
        assert len(model.factor_B) == 10

    def test_hierarchical_model_different_levels(self):
        """Test hierarchical model with very different level dimensions"""
        dims_list = [
            ModelDimensions(num_states=2, num_observations=1, num_actions=1),
            ModelDimensions(num_states=100, num_observations=50, num_actions=10),
        ]
        model = HierarchicalGenerativeModel(dims_list, ModelParameters())

        assert len(model.levels) == 2
        assert model.E_matrices[(0, 1)].shape == (1, 2)  # actions x states

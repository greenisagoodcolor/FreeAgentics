"""
Comprehensive tests for Generative Model classes.
"""

import pytest
import torch

from inference.engine.generative_model import (
    ContinuousGenerativeModel,
    DiscreteGenerativeModel,
    FactorizedGenerativeModel,
    GenerativeModel,
    HierarchicalGenerativeModel,
    ModelDimensions,
    ModelParameters,
    create_generative_model,
)


class TestModelDimensions:
    """Test ModelDimensions dataclass."""

    def test_default_dimensions(self):
        """Test default dimension values."""
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        assert dims.num_states == 4
        assert dims.num_observations == 3
        assert dims.num_actions == 2
        assert dims.num_modalities == 1
        assert dims.num_factors == 1
        assert dims.time_horizon == 1

    def test_custom_dimensions(self):
        """Test custom dimension values."""
        dims = ModelDimensions(
            num_states=10,
            num_observations=5,
            num_actions=3,
            num_modalities=2,
            num_factors=2,
            time_horizon=10,
        )
        assert dims.num_states == 10
        assert dims.num_observations == 5
        assert dims.num_actions == 3
        assert dims.num_modalities == 2
        assert dims.num_factors == 2
        assert dims.time_horizon == 10

    def test_dimension_edge_cases(self):
        """Test dimension edge cases."""
        # Test with minimum dimensions
        dims = ModelDimensions(num_states=1, num_observations=1, num_actions=1)
        assert dims.num_states == 1
        assert dims.num_observations == 1
        assert dims.num_actions == 1


class TestModelParameters:
    """Test ModelParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = ModelParameters()
        assert params.learning_rate == 0.01
        assert params.precision_init == 1.0
        assert params.use_sparse is False
        assert params.use_gpu is True
        assert params.dtype == torch.float32
        assert params.eps == 1e-8
        assert params.temperature == 1.0

    def test_custom_parameters(self):
        """Test custom parameter values."""
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


class TestGenerativeModel:
    """Test abstract GenerativeModel base class."""

    def test_abstract_enforcement(self):
        """Test that abstract methods must be implemented."""
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters()

        with pytest.raises(TypeError):
            GenerativeModel(dims, params)

    def test_device_selection(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test device selection logic."""
        # CPU model
        model_parameters_fixture.use_gpu = False

        class ConcreteModel(GenerativeModel):
            def observation_model(self, states):
                return torch.zeros(3)

            def transition_model(self, states, actions):
                return torch.zeros(4)

            def get_preferences(self, timestep=None):
                return torch.zeros(3)

            def get_initial_prior(self):
                return torch.ones(4) / 4

        model = ConcreteModel(
            model_dimensions_fixture,
            model_parameters_fixture)
        assert model.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="CUDA not available")
    def test_gpu_device_selection(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test GPU device selection."""
        model_parameters_fixture.use_gpu = True

        class ConcreteModel(GenerativeModel):
            def observation_model(self, states):
                return torch.zeros(3)

            def transition_model(self, states, actions):
                return torch.zeros(4)

            def get_preferences(self, timestep=None):
                return torch.zeros(3)

            def get_initial_prior(self):
                return torch.ones(4) / 4

        model = ConcreteModel(
            model_dimensions_fixture,
            model_parameters_fixture)
        assert model.device.type == "cuda"


class TestDiscreteGenerativeModel:
    """Test DiscreteGenerativeModel class."""

    def test_initialization(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test discrete model initialization."""
        model = DiscreteGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        assert hasattr(model, "A")
        assert hasattr(model, "B")
        assert hasattr(model, "C")
        assert hasattr(model, "D")

        # Check matrix dimensions - aligned with pymdp conventions
        # A matrix: P(obs | states) - (num_observations, num_states)
        assert model.A.shape == (
            model_dimensions_fixture.num_observations,
            model_dimensions_fixture.num_states,
        )

        # B matrix: P(next_state | current_state, action) - (num_states, num_states, num_actions)
        # This matches pymdp convention: B[factor].shape = (num_states[factor],
        # num_states[factor], num_controls[factor])
        assert model.B.shape == (
            model_dimensions_fixture.num_states,
            model_dimensions_fixture.num_states,
            model_dimensions_fixture.num_actions,
        )

        # C matrix: Prior preferences - can be time-varying (extended from basic pymdp)
        # Basic pymdp: C[modality].shape = (num_obs[modality],)
        # Our extension: C.shape = (num_observations, time_horizon) for temporal
        # preferences
        assert model.C.shape == (
            model_dimensions_fixture.num_observations,
            model_dimensions_fixture.time_horizon,
        )

        # D vector: Initial state prior - (num_states,) matches pymdp
        # convention
        assert model.D.shape == (model_dimensions_fixture.num_states,)

    def test_observation_model(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test observation model computation."""
        model = DiscreteGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        # Test with one-hot encoded state (pymdp convention for discrete
        # states)
        state_one_hot = torch.zeros(model_dimensions_fixture.num_states)
        state_one_hot[0] = 1.0  # One-hot encoding for first state
        obs_probs = model.observation_model(state_one_hot)
        assert obs_probs.shape == (model_dimensions_fixture.num_observations,)
        assert torch.allclose(obs_probs.sum(), torch.tensor(1.0))

        # Test with state distribution
        state_dist = (
            torch.ones(
                model_dimensions_fixture.num_states) /
            model_dimensions_fixture.num_states)
        obs_probs = model.observation_model(state_dist)
        assert obs_probs.shape == (model_dimensions_fixture.num_observations,)
        assert torch.allclose(obs_probs.sum(), torch.tensor(1.0))

    def test_transition_model(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test transition model computation."""
        model = DiscreteGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        # Test with one-hot encoded state and action (pymdp convention)
        state_one_hot = torch.zeros(model_dimensions_fixture.num_states)
        state_one_hot[0] = 1.0  # One-hot encoding for first state
        action = torch.tensor(0, dtype=torch.long)
        next_state_probs = model.transition_model(state_one_hot, action)
        assert next_state_probs.shape == (model_dimensions_fixture.num_states,)
        assert torch.allclose(next_state_probs.sum(), torch.tensor(1.0))

        # Test with state distribution
        state_dist = (
            torch.ones(
                model_dimensions_fixture.num_states) /
            model_dimensions_fixture.num_states)
        next_state_probs = model.transition_model(state_dist, action)
        assert next_state_probs.shape == (model_dimensions_fixture.num_states,)
        assert torch.allclose(next_state_probs.sum(), torch.tensor(1.0))

    def test_preferences(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test preference retrieval."""
        model = DiscreteGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        # Default preferences return full time-varying matrix (extended from basic
        # pymdp)
        prefs = model.get_preferences()
        assert prefs.shape == (
            model_dimensions_fixture.num_observations,
            model_dimensions_fixture.time_horizon,
        )

        # Test with specific timestep (returns 1D preferences for that
        # timestep)
        prefs_t = model.get_preferences(timestep=0)
        assert prefs_t.shape == (model_dimensions_fixture.num_observations,)

        # Test with timestep beyond horizon (should return last timestep)
        prefs_beyond = model.get_preferences(
            timestep=model_dimensions_fixture.time_horizon + 5)
        assert prefs_beyond.shape == (
            model_dimensions_fixture.num_observations,)

    def test_initial_prior(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test initial prior."""
        model = DiscreteGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        prior = model.get_initial_prior()
        assert prior.shape == (model_dimensions_fixture.num_states,)
        assert torch.allclose(prior.sum(), torch.tensor(1.0))

    def test_probability_normalization(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test that all probability distributions are normalized."""
        model = DiscreteGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        # Check A matrix columns sum to 1
        for s in range(model_dimensions_fixture.num_states):
            assert torch.allclose(model.A[:, s].sum(), torch.tensor(1.0))

        # Check B matrix columns sum to 1 - aligned with pymdp conventions
        # B[next_state, current_state, action] so B[:, s, a] should sum to 1
        # (transition probabilities)
        for a in range(model_dimensions_fixture.num_actions):
            for s in range(model_dimensions_fixture.num_states):
                assert torch.allclose(
                    model.B[:, s, a].sum(), torch.tensor(1.0))

        # Check D sums to 1
        assert torch.allclose(model.D.sum(), torch.tensor(1.0))


class TestContinuousGenerativeModel:
    """Test ContinuousGenerativeModel class."""

    def test_initialization(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test continuous model initialization."""
        model = ContinuousGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        assert hasattr(model, "obs_net")
        assert hasattr(model, "trans_net")
        assert hasattr(model, "obs_mean")
        assert hasattr(model, "obs_log_var")
        assert hasattr(model, "trans_mean")
        assert hasattr(model, "trans_log_var")
        assert hasattr(model, "C")
        assert hasattr(model, "D_mean")
        assert hasattr(model, "D_log_var")

    def test_observation_model(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test continuous observation model."""
        model = ContinuousGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        # Test with continuous state
        state = torch.randn(model_dimensions_fixture.num_states)
        obs_mean, obs_var = model.observation_model(state)
        assert obs_mean.shape[0] == model_dimensions_fixture.num_observations
        assert obs_var.shape[0] == model_dimensions_fixture.num_observations

    def test_transition_model(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test continuous transition model."""
        model = ContinuousGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        state = torch.randn(model_dimensions_fixture.num_states)
        # Use one-hot encoded action for continuous model (standard approach)
        action = torch.zeros(model_dimensions_fixture.num_actions)
        action[0] = 1.0  # One-hot encoding for first action
        next_mean, next_var = model.transition_model(state, action)
        assert next_mean.shape == state.shape
        assert next_var.shape[0] == model_dimensions_fixture.num_states

    def test_neural_network_structure(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test neural network architectures."""
        model = ContinuousGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        # Check observation network components
        assert isinstance(model.obs_net, torch.nn.Module)
        assert isinstance(model.obs_mean, torch.nn.Module)

        # Check transition network components
        assert isinstance(model.trans_net, torch.nn.Module)
        assert isinstance(model.trans_mean, torch.nn.Module)

        # Check parameters for Gaussian distributions
        assert isinstance(model.obs_log_var, torch.nn.Parameter)
        assert isinstance(model.trans_log_var, torch.nn.Parameter)

        # Check preference and prior parameters
        assert isinstance(model.C, torch.nn.Parameter)
        assert isinstance(model.D_mean, torch.nn.Parameter)
        assert isinstance(model.D_log_var, torch.nn.Parameter)

    def test_gradient_flow(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test gradient flow through continuous model."""
        model = ContinuousGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        state = torch.randn(
            model_dimensions_fixture.num_states,
            requires_grad=True)
        action = torch.zeros(model_dimensions_fixture.num_actions)
        action[0] = 1.0  # One-hot encoded action

        # Forward pass
        next_mean, next_var = model.transition_model(state, action)
        obs_mean, obs_var = model.observation_model(next_mean)

        # Compute loss
        loss = obs_mean.sum() + next_mean.sum()
        loss.backward()

        # Check gradients exist
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()


class TestHierarchicalGenerativeModel:
    """Test HierarchicalGenerativeModel class."""

    def test_initialization(self, model_parameters_fixture):
        """Test hierarchical model initialization."""
        # Define hierarchical dimensions for multiple levels
        dims_list = [
            ModelDimensions(num_states=8, num_observations=4, num_actions=3),
            ModelDimensions(num_states=6, num_observations=3, num_actions=2),
            ModelDimensions(num_states=4, num_observations=2, num_actions=2),
        ]

        model = HierarchicalGenerativeModel(
            dims_list, model_parameters_fixture)

        assert hasattr(model, "num_levels")
        assert model.num_levels == 3
        assert hasattr(model, "dimensions")
        assert len(model.dimensions) == 3

    def test_hierarchical_state_handling(self, model_parameters_fixture):
        """Test handling of hierarchical states."""
        dims_list = [
            ModelDimensions(num_states=8, num_observations=4, num_actions=3),
            ModelDimensions(num_states=6, num_observations=3, num_actions=2),
            ModelDimensions(num_states=4, num_observations=2, num_actions=2),
        ]

        model = HierarchicalGenerativeModel(
            dims_list, model_parameters_fixture)

        # Test state at different levels
        for level in range(model.num_levels):
            level_state = (
                torch.ones(
                    model.dimensions[level].num_states) /
                model.dimensions[level].num_states)
            # Use the level's own observation model
            obs = model.levels[level].observation_model(level_state)
            assert obs.shape[0] == model.dimensions[level].num_observations

    def test_top_down_bottom_up_processing(self, model_parameters_fixture):
        """Test hierarchical message passing."""
        dims_list = [
            ModelDimensions(num_states=8, num_observations=4, num_actions=3),
            ModelDimensions(num_states=6, num_observations=3, num_actions=2),
            ModelDimensions(num_states=4, num_observations=2, num_actions=2),
        ]

        model = HierarchicalGenerativeModel(
            dims_list, model_parameters_fixture)

        # Test hierarchical observation and transition models
        states = [
            torch.ones(
                model.dimensions[0].num_states) /
            model.dimensions[0].num_states,
            torch.ones(
                model.dimensions[1].num_states) /
            model.dimensions[1].num_states,
            torch.ones(
                model.dimensions[2].num_states) /
            model.dimensions[2].num_states,
        ]

        # Test hierarchical observation model
        obs_list = model.hierarchical_observation_model(states)
        assert len(obs_list) == 3
        for i, obs in enumerate(obs_list):
            assert obs.shape[0] == model.dimensions[i].num_observations

        # Test hierarchical transition model
        actions = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        next_states = model.hierarchical_transition_model(states, actions)
        assert len(next_states) == 3
        for i, next_state in enumerate(next_states):
            assert next_state.shape[0] == model.dimensions[i].num_states


class TestFactorizedGenerativeModel:
    """Test FactorizedGenerativeModel class."""

    def test_initialization(self, model_parameters_fixture):
        """Test factorized model initialization."""
        factor_dims = [3, 4, 3]  # States per factor
        model = FactorizedGenerativeModel(
            factor_dimensions=factor_dims,
            num_observations=8,
            num_actions=3,
            parameters=model_parameters_fixture,
        )

        assert hasattr(model, "num_factors")
        assert model.num_factors == 3
        assert hasattr(model, "factor_dims")
        assert model.factor_dims == factor_dims

    def test_factorized_state_representation(self, model_parameters_fixture):
        """Test factorized state handling."""
        factor_dims = [3, 4, 3]
        model = FactorizedGenerativeModel(
            factor_dimensions=factor_dims,
            num_observations=8,
            num_actions=3,
            parameters=model_parameters_fixture,
        )

        # Test individual factor states
        _ = [
            torch.ones(3) / 3,  # Factor 0: 3 states
            torch.ones(4) / 4,  # Factor 1: 4 states
            torch.ones(3) / 3,  # Factor 2: 3 states
        ]

        # Test state index conversion methods
        assert hasattr(model, "factor_to_state_idx")
        assert hasattr(model, "state_to_factor_idx")

        # Test conversion round-trip
        factor_indices = [0, 1, 2]
        state_idx = model.factor_to_state_idx(factor_indices)
        recovered_indices = model.state_to_factor_idx(state_idx)
        assert recovered_indices == factor_indices

    def test_factor_independence(self, model_parameters_fixture):
        """Test factor independence in transitions."""
        factor_dims = [3, 4, 3]
        model = FactorizedGenerativeModel(
            factor_dimensions=factor_dims,
            num_observations=8,
            num_actions=3,
            parameters=model_parameters_fixture,
        )

        # Test factorized transition
        factor_states = [
            torch.ones(3) / 3,  # Factor 0: 3 states
            torch.ones(4) / 4,  # Factor 1: 4 states
            torch.ones(3) / 3,  # Factor 2: 3 states
        ]

        action = 0
        next_factor_states = model.factorized_transition(factor_states, action)

        assert len(next_factor_states) == 3
        for i, next_factor_state in enumerate(next_factor_states):
            assert next_factor_state.shape == factor_states[i].shape
            assert torch.allclose(next_factor_state.sum(), torch.tensor(1.0))


class TestCreateGenerativeModel:
    """Test factory function for creating generative models."""

    def test_create_discrete_model(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test creating discrete model."""
        model = create_generative_model(
            "discrete",
            dimensions=model_dimensions_fixture,
            parameters=model_parameters_fixture)
        assert isinstance(model, DiscreteGenerativeModel)

    def test_create_continuous_model(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test creating continuous model."""
        model = create_generative_model(
            "continuous",
            dimensions=model_dimensions_fixture,
            parameters=model_parameters_fixture)
        assert isinstance(model, ContinuousGenerativeModel)

    def test_create_hierarchical_model(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test creating hierarchical model."""
        # For hierarchical model, need a list of dimensions
        dims_list = [model_dimensions_fixture for _ in range(3)]
        model = create_generative_model(
            "hierarchical",
            dimensions_list=dims_list,
            parameters=model_parameters_fixture)
        assert isinstance(model, HierarchicalGenerativeModel)

    def test_create_factorized_model(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test creating factorized model."""
        model = create_generative_model(
            "factorized",
            factor_dimensions=[4, 3, 2],
            num_observations=model_dimensions_fixture.num_observations,
            num_actions=model_dimensions_fixture.num_actions,
            parameters=model_parameters_fixture,
        )
        assert isinstance(model, FactorizedGenerativeModel)

    def test_invalid_model_type(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError):
            create_generative_model(
                "invalid_type",
                dimensions=model_dimensions_fixture,
                parameters=model_parameters_fixture,
            )


class TestModelIntegration:
    """Integration tests for generative models."""

    def test_discrete_model_inference_loop(
        self, model_dimensions_fixture, model_parameters_fixture
    ):
        """Test complete inference loop with discrete model."""
        model = DiscreteGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        # Initialize state
        state = model.get_initial_prior()

        # Observation sequence
        obs_sequence = [0, 1, 2, 1, 0]

        for obs_idx in obs_sequence:
            # Get observation probabilities
            model.observation_model(state)

            # Simulate action selection
            action = torch.tensor(0, dtype=torch.long)

            # Transition to next state
            state = model.transition_model(state, action)

            assert torch.allclose(state.sum(), torch.tensor(1.0))

    def test_continuous_model_trajectory(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test trajectory generation with continuous model."""
        model = ContinuousGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        # Initial state
        initial = model.get_initial_prior()
        # Handle if it returns a tuple
        if isinstance(initial, tuple):
            state = initial[0]
        else:
            state = initial
        trajectory = [state]

        # Generate trajectory
        for t in range(10):
            action = torch.randn(model_dimensions_fixture.num_actions) * 0.1
            result = model.transition_model(state, action)
            # Handle if transition_model returns (mean, variance)
            if isinstance(result, tuple):
                state = result[0]
            else:
                state = result
            trajectory.append(state)

        # Check trajectory properties
        assert len(trajectory) == 11
        for s in trajectory:
            assert not torch.isnan(s).any()
            assert not torch.isinf(s).any()

    def test_model_serialization(
            self,
            model_dimensions_fixture,
            model_parameters_fixture):
        """Test model save and load."""
        model = DiscreteGenerativeModel(
            model_dimensions_fixture,
            model_parameters_fixture)

        # Save model state
        state_dict = (
            model.state_dict()
            if hasattr(model, "state_dict")
            else {"A": model.A, "B": model.B, "C": model.C, "D": model.D}
        )

        # Create new model and load state
        new_model = DiscreteGenerativeModel(
            model_dimensions_fixture, model_parameters_fixture)
        if hasattr(new_model, "load_state_dict"):
            new_model.load_state_dict(state_dict)
        else:
            new_model.A = state_dict["A"]
            new_model.B = state_dict["B"]
            new_model.C = state_dict["C"]
            new_model.D = state_dict["D"]

        # Verify state is preserved
        assert torch.allclose(model.A, new_model.A)
        assert torch.allclose(model.B, new_model.B)

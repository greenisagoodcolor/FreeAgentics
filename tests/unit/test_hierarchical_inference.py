"""
Module for FreeAgentics Active Inference implementation.
"""

import pytest
import torch
import torch.nn as nn

from inference.engine.active_inference import InferenceConfig, VariationalMessagePassing
from inference.engine.generative_model import (
    DiscreteGenerativeModel,
    ModelDimensions,
    ModelParameters,
)
from inference.engine.hierarchical_inference import (
    HierarchicalConfig,
    HierarchicalInference,
    HierarchicalLevel,
    HierarchicalState,
    TemporalHierarchicalInference,
    create_hierarchical_inference,
)
from inference.engine.precision import GradientPrecisionOptimizer, PrecisionConfig


class TestHierarchicalConfig:
    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = HierarchicalConfig()
        assert config.num_levels == 3
        assert config.level_dims == [8, 16, 32]
        assert config.timescales == [1.0, 4.0, 16.0]
        assert config.bottom_up_weight == 0.3
        assert config.top_down_weight == 0.2
        assert config.lateral_weight == 0.5
        assert config.use_precision_weighting

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = HierarchicalConfig(
            num_levels=4,
            level_dims=[4, 8, 16, 32],
            timescales=[1.0, 2.0, 4.0, 8.0],
            use_gpu=False,
        )
        assert config.num_levels == 4
        assert len(config.level_dims) == 4
        assert len(config.timescales) == 4
        assert not config.use_gpu


class TestHierarchicalState:
    def test_state_creation(self) -> None:
        """Test hierarchical state creation"""
        beliefs = torch.randn(2, 8)
        predictions = torch.randn(2, 8)
        errors = torch.randn(2, 8)
        precision = torch.ones(2, 8)
        state = HierarchicalState(
            beliefs=beliefs, predictions=predictions, errors=errors, precision=precision
        )
        assert torch.equal(state.beliefs, beliefs)
        assert torch.equal(state.predictions, predictions)
        assert torch.equal(state.errors, errors)
        assert torch.equal(state.precision, precision)
        assert isinstance(state.temporal_buffer, list)
        assert len(state.temporal_buffer) == 0


class TestHierarchicalLevel:
    @pytest.fixture
    def setup_level(self) -> None:
        """Setup a hierarchical level"""
        config = HierarchicalConfig(use_gpu=False)
        dims = ModelDimensions(num_states=8, num_observations=4, num_actions=2)
        params = ModelParameters(use_gpu=False)
        gen_model = DiscreteGenerativeModel(dims, params)
        inf_config = InferenceConfig(use_gpu=False)
        inference = VariationalMessagePassing(inf_config)
        prec_config = PrecisionConfig(use_gpu=False)
        precision_opt = GradientPrecisionOptimizer(prec_config)
        level = HierarchicalLevel(
            level_id=1,
            config=config,
            generative_model=gen_model,
            inference_algorithm=inference,
            precision_optimizer=precision_opt,
        )
        return (level, config, gen_model)

    def test_initialization(self, setup_level) -> None:
        """Test level initialization"""
        level, config, gen_model = setup_level
        assert level.level_id == 1
        assert level.config == config
        assert level.generative_model == gen_model
        assert level.state_dim == 8
        assert level.obs_dim == 4
        assert level.timescale == 4.0
        assert level.prediction_horizon == 4
        assert hasattr(level, "bottom_up_net")
        assert hasattr(level, "top_down_net")
        assert isinstance(level.temporal_net, nn.GRUCell)

    def test_initialize_state(self, setup_level) -> None:
        """Test state initialization"""
        level, _, _ = setup_level
        batch_size = 3
        state = level.initialize_state(batch_size)
        assert isinstance(state, HierarchicalState)
        assert state.beliefs.shape == (3, 8)
        assert state.predictions.shape == (3, 8)
        assert state.errors.shape == (3, 8)
        assert state.precision.shape == (3, 8)
        assert torch.allclose(state.beliefs.sum(dim=-1), torch.ones(3), atol=1e-06)

    def test_compute_prediction_error(self, setup_level) -> None:
        """Test prediction error computation"""
        level, _, _ = setup_level
        observations = torch.softmax(torch.randn(2, 8), dim=-1)
        predictions = torch.softmax(torch.randn(2, 8), dim=-1)
        error = level.compute_prediction_error(observations, predictions)
        assert error.shape == (2, 8)
        assert not torch.any(torch.isnan(error))

    def test_update_beliefs(self, setup_level) -> None:
        """Test belief update"""
        level, _, _ = setup_level
        batch_size = 2
        level.initialize_state(batch_size)
        bottom_up_input = torch.randn(2, 8)
        top_down_input = torch.randn(2, 32)
        updated_beliefs = level.update_beliefs(bottom_up_input, top_down_input)
        assert updated_beliefs.shape == (2, 8)
        assert torch.allclose(updated_beliefs.sum(dim=-1), torch.ones(2), atol=1e-06)
        assert torch.equal(updated_beliefs, level.state.beliefs)

    def test_temporal_buffer(self, setup_level) -> None:
        """Test temporal buffer updates"""
        level, _, _ = setup_level
        level.initialize_state(2)
        for _ in range(5):
            level.update_beliefs()
        assert len(level.state.temporal_buffer) == 4
        for beliefs in level.state.temporal_buffer:
            assert beliefs.shape == (2, 8)


class TestHierarchicalInference:
    @pytest.fixture
    def setup_hierarchical_system(self) -> None:
        """Setup hierarchical inference system"""
        config = HierarchicalConfig(num_levels=3, level_dims=[8, 16, 32], use_gpu=False)
        models = []
        algorithms = []
        optimizers = []
        for i in range(3):
            dims = ModelDimensions(
                num_states=config.level_dims[i],
                num_observations=config.level_dims[i] // 2,
                num_actions=2,
            )
            params = ModelParameters(use_gpu=False)
            models.append(DiscreteGenerativeModel(dims, params))
            inf_config = InferenceConfig(use_gpu=False)
            algorithms.append(VariationalMessagePassing(inf_config))
            prec_config = PrecisionConfig(use_gpu=False)
            optimizers.append(GradientPrecisionOptimizer(prec_config))
        system = HierarchicalInference(config, models, algorithms, optimizers)
        return (system, config, models)

    def test_initialization(self, setup_hierarchical_system) -> None:
        """Test system initialization"""
        system, config, models = setup_hierarchical_system
        assert system.num_levels == 3
        assert len(system.levels) == 3
        assert system.timestep == 0
        for i, level in enumerate(system.levels):
            assert level.level_id == i
            assert level.state_dim == config.level_dims[i]

    def test_system_initialize(self, setup_hierarchical_system) -> None:
        """Test initializing all levels"""
        system, _, _ = setup_hierarchical_system
        batch_size = 2
        system.initialize(batch_size)
        for level in system.levels:
            assert level.state is not None
            assert level.state.beliefs.shape[0] == batch_size

    def test_step(self, setup_hierarchical_system) -> None:
        """Test one step of hierarchical inference"""
        system, _, _ = setup_hierarchical_system
        batch_size = 2
        system.initialize(batch_size)
        observations = torch.randn(batch_size, 4)
        beliefs = system.step(observations)
        assert len(beliefs) == 3
        for i, level_beliefs in enumerate(beliefs):
            assert level_beliefs.shape == (batch_size, system.config.level_dims[i])
            assert torch.allclose(level_beliefs.sum(dim=-1), torch.ones(batch_size), atol=1e-06)

    def test_timescale_updates(self, setup_hierarchical_system) -> None:
        """Test that levels update according to their timescales"""
        system, _, _ = setup_hierarchical_system
        batch_size = 2
        system.initialize(batch_size)
        observations = torch.randn(batch_size, 4)
        initial_beliefs = [level.state.beliefs.clone() for level in system.levels]
        for _ in range(4):
            system.step(observations)
        assert not torch.allclose(system.levels[0].state.beliefs, initial_beliefs[0])
        assert not torch.allclose(system.levels[1].state.beliefs, initial_beliefs[1])

    def test_hierarchical_free_energy(self, setup_hierarchical_system) -> None:
        """Test free energy computation"""
        system, _, _ = setup_hierarchical_system
        batch_size = 2
        system.initialize(batch_size)
        free_energies = system.get_hierarchical_free_energy()
        assert len(free_energies) == 3
        for fe in free_energies:
            assert fe.shape == (batch_size,)
            assert not torch.any(torch.isnan(fe))
            assert not torch.any(torch.isinf(fe))

    def test_get_effective_beliefs(self, setup_hierarchical_system) -> None:
        """Test getting effective beliefs"""
        system, _, _ = setup_hierarchical_system
        batch_size = 2
        system.initialize(batch_size)
        effective_beliefs = system.get_effective_beliefs(target_level=0)
        assert effective_beliefs.shape == (batch_size, 8)
        assert torch.allclose(effective_beliefs.sum(dim=-1), torch.ones(batch_size), atol=1e-06)
        with pytest.raises(ValueError):
            system.get_effective_beliefs(target_level=5)


class TestTemporalHierarchicalInference:
    @pytest.fixture
    def setup_temporal_system(self) -> None:
        """Setup temporal hierarchical system"""
        config = HierarchicalConfig(
            num_levels=2,
            level_dims=[8, 16],
            timescales=[1.0, 4.0],
            prediction_horizon=[2, 8],
            use_gpu=False,
        )
        models = []
        algorithms = []
        for i in range(2):
            dims = ModelDimensions(
                num_states=config.level_dims[i],
                num_observations=config.level_dims[i] // 2,
                num_actions=2,
            )
            params = ModelParameters(use_gpu=False)
            models.append(DiscreteGenerativeModel(dims, params))
            inf_config = InferenceConfig(use_gpu=False)
            algorithms.append(VariationalMessagePassing(inf_config))
        system = TemporalHierarchicalInference(config, models, algorithms)
        return (system, config)

    def test_initialization(self, setup_temporal_system) -> None:
        """Test temporal system initialization"""
        system, config = setup_temporal_system
        assert isinstance(system, TemporalHierarchicalInference)
        assert len(system.temporal_predictors) == 2
        for predictor in system.temporal_predictors:
            assert isinstance(predictor, nn.LSTM)

    def test_predict_future_states(self, setup_temporal_system) -> None:
        """Test future state prediction"""
        system, _ = setup_temporal_system
        batch_size = 2
        system.initialize(batch_size)
        current_state = system.levels[0].state.beliefs
        predictions = system.predict_future_states(0, current_state, horizon=3)
        assert len(predictions) == 3
        for pred in predictions:
            assert pred.shape == (batch_size, 8)
            assert torch.allclose(pred.sum(dim=-1), torch.ones(batch_size), atol=1e-06)

    def test_hierarchical_planning(self, setup_temporal_system) -> None:
        """Test hierarchical planning"""
        system, _ = setup_temporal_system
        batch_size = 2
        system.initialize(batch_size)
        trajectories = system.hierarchical_planning(planning_horizon=5)
        assert len(trajectories) == 2
        assert len(trajectories[0]) == min(5, 2)
        assert len(trajectories[1]) == min(5, 8)

    def test_coarse_to_fine_inference(self, setup_temporal_system) -> None:
        """Test coarse-to-fine inference"""
        system, _ = setup_temporal_system
        batch_size = 2
        system.initialize(batch_size)  # Initialize the system before inference
        observations = torch.randn(batch_size, 4)
        beliefs = system.coarse_to_fine_inference(observations, iterations=3)
        assert len(beliefs) == 2
        assert beliefs[0].shape == (batch_size, 8)
        assert beliefs[1].shape == (batch_size, 16)
        for level_beliefs in beliefs:
            assert torch.allclose(level_beliefs.sum(dim=-1), torch.ones(batch_size), atol=1e-06)


class TestFactoryFunction:
    def test_create_standard_hierarchical(self) -> None:
        """Test creating standard hierarchical inference"""
        config = HierarchicalConfig(num_levels=2, use_gpu=False)
        models = []
        algorithms = []
        for i in range(2):
            dims = ModelDimensions(num_states=8, num_observations=4, num_actions=2)
            params = ModelParameters(use_gpu=False)
            models.append(DiscreteGenerativeModel(dims, params))
            inf_config = InferenceConfig(use_gpu=False)
            algorithms.append(VariationalMessagePassing(inf_config))
        system = create_hierarchical_inference(
            "standard",
            config,
            generative_models=models,
            inference_algorithms=algorithms,
        )
        assert isinstance(system, HierarchicalInference)
        assert system.num_levels == 2

    def test_create_temporal_hierarchical(self) -> None:
        """Test creating temporal hierarchical inference"""
        config = HierarchicalConfig(num_levels=2, use_gpu=False)
        models = []
        algorithms = []
        for i in range(2):
            dims = ModelDimensions(num_states=8, num_observations=4, num_actions=2)
            params = ModelParameters(use_gpu=False)
            models.append(DiscreteGenerativeModel(dims, params))
            inf_config = InferenceConfig(use_gpu=False)
            algorithms.append(VariationalMessagePassing(inf_config))
        system = create_hierarchical_inference(
            "temporal",
            config,
            generative_models=models,
            inference_algorithms=algorithms,
        )
        assert isinstance(system, TemporalHierarchicalInference)
        assert hasattr(system, "temporal_predictors")

    def test_invalid_type(self) -> None:
        """Test invalid inference type"""
        config = HierarchicalConfig(use_gpu=False)
        # Create minimal required parameters to reach type validation
        dims = ModelDimensions(num_states=8, num_observations=4, num_actions=2)
        params = ModelParameters(use_gpu=False)
        models = [DiscreteGenerativeModel(dims, params)]
        inf_config = InferenceConfig(use_gpu=False)
        algorithms = [VariationalMessagePassing(inf_config)]
        with pytest.raises(ValueError, match="Unknown inference type"):
            create_hierarchical_inference(
                "invalid", config, generative_models=models, inference_algorithms=algorithms
            )

    def test_missing_required_params(self) -> None:
        """Test missing required parameters"""
        config = HierarchicalConfig(use_gpu=False)
        with pytest.raises(ValueError, match="requires generative_models"):
            create_hierarchical_inference("standard", config)

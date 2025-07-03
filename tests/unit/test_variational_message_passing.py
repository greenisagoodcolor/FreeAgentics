"""
Comprehensive tests for Variational Message Passing algorithm
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from inference.algorithms.variational_message_passing import (
    InferenceConfig,
    VariationalMessagePassing,
)
from inference.engine.active_inference import (
    BatchDiscreteInferenceStrategy,
    DefaultInferenceStrategy,
    InferenceAlgorithm,
    InferenceStrategy,
    SingleDiscreteInferenceStrategy,
    SoftObservationInferenceStrategy,
)
from inference.engine.generative_model import (
    DiscreteGenerativeModel,
    GenerativeModel,
    ModelDimensions,
    ModelParameters,
)


class TestInferenceConfig:
    """Test InferenceConfig dataclass"""

    def test_default_configuration(self):
        """Test default configuration values"""
        config = InferenceConfig()

        assert config.algorithm == "variational_message_passing"
        assert config.num_iterations == 16
        assert config.convergence_threshold == 1e-4
        assert config.learning_rate == 0.1
        assert config.gradient_clip == 1.0
        assert config.use_natural_gradient is True
        assert config.damping_factor == 0.1
        assert config.momentum == 0.9
        assert config.precision_parameter == 1.0
        assert config.use_gpu is True
        assert config.dtype == torch.float32
        assert config.eps == 1e-16
        assert config.use_temporal_processing is True
        assert config.temporal_window == 5
        assert config.gnn_metadata == {}

    def test_custom_configuration(self):
        """Test custom configuration values"""
        metadata = {"model_name": "test", "version": "1.0"}
        config = InferenceConfig(
            algorithm="custom_vmp",
            num_iterations=32,
            convergence_threshold=1e-6,
            learning_rate=0.01,
            gradient_clip=0.5,
            use_natural_gradient=False,
            damping_factor=0.2,
            momentum=0.95,
            precision_parameter=2.0,
            use_gpu=False,
            dtype=torch.float64,
            eps=1e-12,
            use_temporal_processing=False,
            temporal_window=10,
            gnn_metadata=metadata,
        )

        assert config.algorithm == "custom_vmp"
        assert config.num_iterations == 32
        assert config.convergence_threshold == 1e-6
        assert config.learning_rate == 0.01
        assert config.gradient_clip == 0.5
        assert config.use_natural_gradient is False
        assert config.damping_factor == 0.2
        assert config.momentum == 0.95
        assert config.precision_parameter == 2.0
        assert config.use_gpu is False
        assert config.dtype == torch.float64
        assert config.eps == 1e-12
        assert config.use_temporal_processing is False
        assert config.temporal_window == 10
        assert config.gnn_metadata == metadata

    def test_validation_errors(self):
        """Test configuration validation errors"""
        # Test num_iterations validation
        with pytest.raises(ValueError, match="num_iterations must be positive"):
            InferenceConfig(num_iterations=0)

        with pytest.raises(ValueError, match="num_iterations must be positive"):
            InferenceConfig(num_iterations=-1)

        # Test convergence_threshold validation
        with pytest.raises(ValueError, match="convergence_threshold must be non-negative"):
            InferenceConfig(convergence_threshold=-0.1)

        # Test learning_rate validation
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            InferenceConfig(learning_rate=0)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            InferenceConfig(learning_rate=-0.1)

        # Test gradient_clip validation
        with pytest.raises(ValueError, match="gradient_clip must be positive"):
            InferenceConfig(gradient_clip=0)

        # Test damping_factor validation
        with pytest.raises(ValueError, match="damping_factor must be non-negative"):
            InferenceConfig(damping_factor=-0.1)

        # Test momentum validation
        with pytest.raises(ValueError, match="momentum must be between 0 and 1"):
            InferenceConfig(momentum=-0.1)

        with pytest.raises(ValueError, match="momentum must be between 0 and 1"):
            InferenceConfig(momentum=1.1)

        # Test precision_parameter validation
        with pytest.raises(ValueError, match="precision_parameter must be positive"):
            InferenceConfig(precision_parameter=0)

        # Test eps validation
        with pytest.raises(ValueError, match="eps must be positive"):
            InferenceConfig(eps=0)

    def test_edge_case_valid_values(self):
        """Test edge case valid values"""
        # Zero convergence threshold is valid
        config = InferenceConfig(convergence_threshold=0)
        assert config.convergence_threshold == 0

        # Zero damping factor is valid
        config = InferenceConfig(damping_factor=0)
        assert config.damping_factor == 0

        # Momentum at boundaries
        config = InferenceConfig(momentum=0)
        assert config.momentum == 0

        config = InferenceConfig(momentum=1)
        assert config.momentum == 1


class TestInferenceAlgorithm:
    """Test InferenceAlgorithm abstract base class"""

    def test_initialization(self):
        """Test algorithm initialization"""
        config = InferenceConfig()

        # Create a concrete implementation for testing
        class ConcreteAlgorithm(InferenceAlgorithm):
            def infer_states(self, observations, generative_model, prior_beliefs=None):
                return torch.zeros(4)

        algorithm = ConcreteAlgorithm(config)

        assert algorithm.config == config
        assert algorithm.eps == config.eps
        assert algorithm.gnn_metadata == {}
        assert isinstance(algorithm.device, torch.device)

    def test_initialization_with_metadata(self):
        """Test initialization with GNN metadata"""
        metadata = {"model_name": "test_model", "layers": 3}
        config = InferenceConfig(gnn_metadata=metadata)

        class ConcreteAlgorithm(InferenceAlgorithm):
            def infer_states(self, observations, generative_model, prior_beliefs=None):
                return torch.zeros(4)

        algorithm = ConcreteAlgorithm(config)
        assert algorithm.gnn_metadata == metadata

    @patch("torch.cuda.is_available")
    def test_device_selection(self, mock_cuda):
        """Test GPU/CPU device selection"""
        # Test GPU available and requested
        mock_cuda.return_value = True
        config = InferenceConfig(use_gpu=True)

        class ConcreteAlgorithm(InferenceAlgorithm):
            def infer_states(self, observations, generative_model, prior_beliefs=None):
                return torch.zeros(4)

        algorithm = ConcreteAlgorithm(config)
        assert algorithm.device.type == "cuda"

        # Test GPU not available
        mock_cuda.return_value = False
        algorithm = ConcreteAlgorithm(config)
        assert algorithm.device.type == "cpu"

        # Test GPU not requested
        config = InferenceConfig(use_gpu=False)
        algorithm = ConcreteAlgorithm(config)
        assert algorithm.device.type == "cpu"

    def test_validate_pymdp_matrices(self):
        """Test PyMDP matrix validation"""
        config = InferenceConfig()

        class ConcreteAlgorithm(InferenceAlgorithm):
            def infer_states(self, observations, generative_model, prior_beliefs=None):
                return torch.zeros(4)

        algorithm = ConcreteAlgorithm(config)

        # Test with model without matrices
        model = Mock(spec=[])
        assert algorithm.validate_pymdp_matrices(model) is True

        # Test with valid A matrix
        model = Mock()
        model.A = torch.tensor([[0.9, 0.1], [0.1, 0.9]])  # Columns sum to 1
        assert algorithm.validate_pymdp_matrices(model) is True

        # Test with invalid A matrix
        model.A = torch.tensor([[0.5, 0.5], [0.4, 0.4]])  # Columns don't sum to 1
        assert algorithm.validate_pymdp_matrices(model) is False

        # Test with valid B matrix
        model = Mock()
        model.B = torch.zeros(3, 3, 2)
        # Make transitions sum to 1
        for a in range(2):
            for s in range(3):
                model.B[:, s, a] = torch.tensor([0.33, 0.33, 0.34])
        assert algorithm.validate_pymdp_matrices(model) is True

        # Test with invalid B matrix
        model.B = torch.ones(3, 3, 2) * 0.5  # Transitions don't sum to 1
        assert algorithm.validate_pymdp_matrices(model) is False

        # Test with valid D vector
        model = Mock()
        model.D = torch.tensor([0.25, 0.25, 0.25, 0.25])  # Sums to 1
        assert algorithm.validate_pymdp_matrices(model) is True

        # Test with invalid D vector
        model.D = torch.tensor([0.2, 0.3, 0.4, 0.5])  # Doesn't sum to 1
        assert algorithm.validate_pymdp_matrices(model) is False

    def test_get_model_dimensions(self):
        """Test extracting model dimensions"""
        config = InferenceConfig()

        class ConcreteAlgorithm(InferenceAlgorithm):
            def infer_states(self, observations, generative_model, prior_beliefs=None):
                return torch.zeros(4)

        algorithm = ConcreteAlgorithm(config)

        # Test with dims attribute
        model = Mock()
        model.dims = ModelDimensions(num_states=10, num_observations=5, num_actions=3)
        dims = algorithm.get_model_dimensions(model)
        assert dims.num_states == 10
        assert dims.num_observations == 5
        assert dims.num_actions == 3

        # Test inferring from A matrix
        model = Mock(spec=["A"])
        model.A = torch.zeros(8, 12)  # 8 observations, 12 states
        dims = algorithm.get_model_dimensions(model)
        assert dims.num_states == 12
        assert dims.num_observations == 8
        assert dims.num_actions == 2  # Default

        # Test inferring from A and B matrices
        model = Mock(spec=["A", "B"])
        model.A = torch.zeros(8, 12)
        model.B = torch.zeros(12, 12, 4)  # 4 actions
        dims = algorithm.get_model_dimensions(model)
        assert dims.num_states == 12
        assert dims.num_observations == 8
        assert dims.num_actions == 4

        # Test default dimensions
        model = Mock(spec=[])
        dims = algorithm.get_model_dimensions(model)
        assert dims.num_states == 4
        assert dims.num_observations == 3
        assert dims.num_actions == 2


class TestVariationalMessagePassing:
    """Test VariationalMessagePassing implementation"""

    def test_initialization(self):
        """Test VMP initialization"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        assert vmp.config == config
        assert vmp.belief_history == []
        assert isinstance(vmp, InferenceAlgorithm)

    def test_prepare_belief(self):
        """Test belief preparation"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        # Mock generative model
        model = Mock()
        model.dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)

        # Test with prior_beliefs
        prior_beliefs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        belief = vmp._prepare_belief(prior_beliefs, None, model)
        assert torch.allclose(belief, prior_beliefs / prior_beliefs.sum())

        # Test with prior (backward compatibility)
        prior = torch.tensor([0.25, 0.25, 0.25, 0.25])
        belief = vmp._prepare_belief(None, prior, model)
        assert torch.allclose(belief, prior)

        # Test with no priors - uniform
        belief = vmp._prepare_belief(None, None, model)
        assert torch.allclose(belief, torch.ones(4) / 4)

        # Test normalization
        unnormalized = torch.tensor([1.0, 2.0, 3.0, 4.0])
        belief = vmp._prepare_belief(unnormalized, None, model)
        assert torch.allclose(belief.sum(), torch.tensor(1.0))

    def test_get_state_dimension(self):
        """Test getting state dimension from model"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        # Test with dims attribute
        model = Mock()
        model.dims = ModelDimensions(num_states=10, num_observations=5, num_actions=3)
        assert vmp._get_state_dimension(model) == 10

        # Test without dims - default
        model = Mock(spec=[])
        assert vmp._get_state_dimension(model) == 4

    def test_observation_type_detection(self):
        """Test observation type detection methods"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        # Test single discrete observation
        obs = torch.tensor(2)
        assert vmp._is_single_discrete_observation(obs) is True

        obs = torch.tensor([2], dtype=torch.long)
        assert vmp._is_single_discrete_observation(obs) is True

        # Test soft observation distribution
        obs = torch.tensor([[0.1, 0.3, 0.6]])
        assert vmp._is_soft_observation_distribution(obs) is True

        # Test batch discrete observations
        obs = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        assert vmp._is_batch_discrete_observations(obs) is True

        # Test non-matching cases
        obs = torch.tensor([[0.1, 0.3], [0.4, 0.2]])  # 2D but not single row
        assert vmp._is_soft_observation_distribution(obs) is False

        obs = torch.tensor([1.5, 2.5])  # Float values
        assert vmp._is_batch_discrete_observations(obs) is False

    def test_select_inference_strategy(self):
        """Test inference strategy selection"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        # Single discrete observation
        obs = torch.tensor(2)
        strategy = vmp._select_inference_strategy(obs)
        assert isinstance(strategy, SingleDiscreteInferenceStrategy)

        # Soft observation
        obs = torch.tensor([[0.1, 0.3, 0.6]])
        strategy = vmp._select_inference_strategy(obs)
        assert isinstance(strategy, SoftObservationInferenceStrategy)

        # Batch discrete observations
        obs = torch.tensor([1, 2, 3], dtype=torch.long)
        strategy = vmp._select_inference_strategy(obs)
        assert isinstance(strategy, BatchDiscreteInferenceStrategy)

        # Default case
        obs = torch.randn(5, 3)  # Random continuous observations
        strategy = vmp._select_inference_strategy(obs)
        assert isinstance(strategy, DefaultInferenceStrategy)

    def test_infer_states_single_observation(self):
        """Test state inference with single observation"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        # Create mock generative model
        model = Mock()
        model.dims = ModelDimensions(num_states=3, num_observations=4, num_actions=2)
        # A matrix: P(obs|state)
        model.A = torch.tensor(
            [
                [0.9, 0.1, 0.1],  # obs 0
                [0.05, 0.8, 0.1],  # obs 1
                [0.03, 0.05, 0.7],  # obs 2
                [0.02, 0.05, 0.1],  # obs 3
            ]
        )

        # Observe state 1 (should favor state 1)
        obs = torch.tensor(1)
        prior = torch.tensor([0.33, 0.33, 0.34])

        posterior = vmp.infer_states(obs, model, prior_beliefs=prior)

        # State 1 should have highest probability
        assert posterior.argmax() == 1
        assert torch.allclose(posterior.sum(), torch.tensor(1.0))

    def test_infer_states_soft_observation(self):
        """Test state inference with soft observation distribution"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        # Create mock generative model
        model = Mock()
        model.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        model.A = torch.eye(3)  # Identity for simplicity

        # Soft observation favoring obs 2
        obs = torch.tensor([[0.1, 0.2, 0.7]])
        prior = torch.ones(3) / 3

        posterior = vmp.infer_states(obs, model, prior_beliefs=prior)

        # Should favor state 2
        assert posterior.squeeze().argmax() == 2
        assert posterior.shape == (1, 3)

    def test_infer_states_batch_observations(self):
        """Test state inference with batch of observations"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        # Create mock generative model
        model = Mock()
        model.dims = ModelDimensions(num_states=2, num_observations=2, num_actions=2)
        model.A = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

        # Sequence of observations
        obs = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        prior = torch.tensor([0.5, 0.5])

        posterior = vmp.infer_states(obs, model, prior_beliefs=prior)

        # Should return batch of posteriors
        assert posterior.shape == (4, 2)
        # Each should sum to 1
        for i in range(4):
            assert torch.allclose(posterior[i].sum(), torch.tensor(1.0))

    def test_infer_states_no_observation_model(self):
        """Test inference when model has no observation matrix"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        # Model without A matrix
        model = Mock(spec=["dims"])
        model.dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)

        obs = torch.tensor(1)
        prior = torch.tensor([0.1, 0.2, 0.3, 0.4])

        posterior = vmp.infer_states(obs, model, prior_beliefs=prior)

        # Should return normalized prior
        assert torch.allclose(posterior, prior / prior.sum())


class TestInferenceStrategies:
    """Test individual inference strategy implementations"""

    def test_single_discrete_inference_strategy(self):
        """Test SingleDiscreteInferenceStrategy"""
        strategy = SingleDiscreteInferenceStrategy()

        # Mock model
        model = Mock()
        model.A = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

        # Test with scalar observation
        obs = torch.tensor(0)
        belief = torch.tensor([0.5, 0.5])

        posterior = strategy.infer(obs, model, belief)

        # Should favor state 0
        assert posterior[0] > posterior[1]
        assert torch.allclose(posterior.sum(), torch.tensor(1.0))

        # Test with 1D observation
        obs = torch.tensor([1])
        belief = torch.tensor([0.3, 0.7])

        posterior = strategy.infer(obs, model, belief)

        # Should favor state 1
        assert posterior[1] > posterior[0]

    def test_soft_observation_inference_strategy(self):
        """Test SoftObservationInferenceStrategy"""
        strategy = SoftObservationInferenceStrategy()

        # Mock model
        model = Mock()
        model.A = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.7]])

        # Soft observation distribution
        obs = torch.tensor([[0.6, 0.3, 0.1]])  # Favors obs 0
        belief = torch.ones(3) / 3

        posterior = strategy.infer(obs, model, belief)

        # Should favor state 0
        assert posterior.shape == (1, 3)
        assert posterior.squeeze().argmax() == 0
        assert torch.allclose(posterior.sum(), torch.tensor(1.0))

    def test_batch_discrete_inference_strategy(self):
        """Test BatchDiscreteInferenceStrategy"""
        strategy = BatchDiscreteInferenceStrategy()

        # Mock model
        model = Mock()
        model.A = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

        # Batch of observations
        obs = torch.tensor([0, 1, 0], dtype=torch.long)
        belief = torch.tensor([0.5, 0.5])

        posteriors = strategy.infer(obs, model, belief)

        assert posteriors.shape == (3, 2)
        # Check that posteriors are normalized
        for i in range(3):
            assert torch.allclose(posteriors[i].sum(), torch.tensor(1.0))

        # The posteriors should be updated based on observations
        # First observation (0) should increase belief in state 0
        assert posteriors[0, 0] > posteriors[0, 1]

        # After observing 0 then 1, beliefs should be updated accordingly
        # Note: BatchDiscreteInferenceStrategy updates beliefs sequentially
        # so the exact values depend on the cumulative updates

    def test_default_inference_strategy(self):
        """Test DefaultInferenceStrategy"""
        strategy = DefaultInferenceStrategy()

        model = Mock()

        # Test with existing belief
        obs = torch.randn(5, 3)
        belief = torch.tensor([0.2, 0.3, 0.5])

        result = strategy.infer(obs, model, belief)

        # Should expand belief to batch size
        assert result.shape == (5, 3)
        assert torch.allclose(result[0], belief)

        # Test with no belief - uniform
        model.dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        belief = torch.tensor([0.25, 0.25, 0.25, 0.25])
        obs = torch.randn(3)

        result = strategy.infer(obs, model, belief)
        assert torch.allclose(result, belief)


class TestFreeEnergyComputation:
    """Test free energy computation methods"""

    def test_compute_free_energy_basic(self):
        """Test basic free energy computation"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)
        strategy = DefaultInferenceStrategy()

        # Mock model with prior
        model = Mock()
        model.D = torch.tensor([0.25, 0.25, 0.25, 0.25])
        model.A = torch.eye(4)
        # Make sure get_initial_prior is not called or returns tensor if called
        if hasattr(model, "get_initial_prior"):
            delattr(model, "get_initial_prior")

        beliefs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        observations = torch.tensor(2)

        # Note: compute_free_energy is a method of DefaultInferenceStrategy
        # not VariationalMessagePassing in the implementation
        free_energy = strategy.compute_free_energy(beliefs, observations, model)

        assert isinstance(free_energy, torch.Tensor)
        assert free_energy.dim() == 0  # Scalar

    def test_compute_free_energy_with_custom_prior(self):
        """Test free energy with custom prior"""
        strategy = DefaultInferenceStrategy()

        # Mock model with get_initial_prior method
        model = Mock()
        # Make sure get_initial_prior returns a tensor, not a Mock
        prior_tensor = torch.tensor([0.1, 0.1, 0.4, 0.4])
        model.get_initial_prior = Mock(return_value=prior_tensor)
        model.A = torch.eye(4)

        beliefs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        observations = torch.tensor(0)

        free_energy = strategy.compute_free_energy(beliefs, observations, model)

        assert model.get_initial_prior.called
        assert isinstance(free_energy, torch.Tensor)

    def test_normalize_tensor_dimensions(self):
        """Test tensor dimension normalization"""
        strategy = DefaultInferenceStrategy()

        # Test scalar tensors
        beliefs = torch.tensor(0.5)
        observations = torch.tensor(1)

        norm_beliefs, norm_obs = strategy._normalize_tensor_dimensions(beliefs, observations)

        assert norm_beliefs.dim() == 1
        assert norm_obs.dim() == 1

    def test_extract_prior_tensor(self):
        """Test prior extraction from model"""
        strategy = DefaultInferenceStrategy()

        # Test with get_initial_prior method
        model = Mock()
        model.get_initial_prior = Mock(return_value=torch.tensor([0.3, 0.7]))
        beliefs = torch.tensor([0.5, 0.5])

        prior = strategy._extract_prior_tensor(model, beliefs)
        assert torch.allclose(prior, torch.tensor([0.3, 0.7]))

        # Test with D attribute
        model = Mock(spec=["D"])
        model.D = torch.tensor([0.2, 0.3, 0.5])
        beliefs = torch.tensor([0.33, 0.33, 0.34])

        prior = strategy._extract_prior_tensor(model, beliefs)
        assert torch.allclose(prior, model.D)

        # Test fallback to uniform
        model = Mock(spec=[])
        beliefs = torch.tensor([0.2, 0.3, 0.5])

        prior = strategy._extract_prior_tensor(model, beliefs)
        expected = torch.ones(3) / 3
        assert torch.allclose(prior, expected)

    def test_convert_prior_to_tensor(self):
        """Test prior conversion to tensor"""
        strategy = DefaultInferenceStrategy()

        # Test tensor input
        prior = torch.tensor([0.5, 0.5])
        result = strategy._convert_prior_to_tensor(prior)
        assert torch.equal(result, prior)

        # Test tuple input (continuous case)
        prior = (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))
        result = strategy._convert_prior_to_tensor(prior)
        assert torch.equal(result, prior[0])

        # Test non-tensor input
        prior = [0.3, 0.7]
        result = strategy._convert_prior_to_tensor(prior)
        assert torch.allclose(result, torch.tensor([0.3, 0.7]))

    def test_compute_complexity_term(self):
        """Test KL divergence computation"""
        strategy = DefaultInferenceStrategy()

        beliefs = torch.tensor([0.3, 0.7])
        prior = torch.tensor([0.5, 0.5])

        complexity = strategy._compute_complexity_term(beliefs, prior)

        # KL divergence should be non-negative
        assert complexity >= 0

        # Test when beliefs equal prior - KL should be ~0
        beliefs = prior.clone()
        complexity = strategy._compute_complexity_term(beliefs, prior)
        assert torch.allclose(complexity, torch.tensor(0.0), atol=1e-6)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""

    def test_empty_belief_history(self):
        """Test VMP with empty belief history"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        assert len(vmp.belief_history) == 0

        # Should still work with empty history
        model = Mock()
        model.dims = ModelDimensions(num_states=2, num_observations=2, num_actions=2)
        # Make A matrix subscriptable
        model.A = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
        obs = torch.tensor(0)

        result = vmp.infer_states(obs, model)
        assert result is not None
        assert torch.allclose(result.sum(), torch.tensor(1.0))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        config = InferenceConfig(eps=1e-16)
        vmp = VariationalMessagePassing(config)

        # Model with very small probabilities
        model = Mock()
        model.A = torch.tensor([[1e-10, 1 - 1e-10], [1 - 1e-10, 1e-10]])
        model.dims = ModelDimensions(num_states=2, num_observations=2, num_actions=2)

        obs = torch.tensor(0)
        prior = torch.tensor([0.5, 0.5])

        posterior = vmp.infer_states(obs, model, prior_beliefs=prior)

        # Should not contain NaN or Inf
        assert not torch.isnan(posterior).any()
        assert not torch.isinf(posterior).any()
        assert torch.allclose(posterior.sum(), torch.tensor(1.0))

    def test_different_tensor_types(self):
        """Test with different tensor types"""
        config = InferenceConfig(dtype=torch.float64)
        vmp = VariationalMessagePassing(config)

        model = Mock()
        model.A = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)
        model.dims = ModelDimensions(num_states=2, num_observations=2, num_actions=2)

        # Test with different observation dtypes
        for dtype in [torch.int32, torch.int64, torch.long]:
            obs = torch.tensor(1, dtype=dtype)
            prior = torch.tensor([0.5, 0.5], dtype=torch.float64)

            posterior = vmp.infer_states(obs, model, prior_beliefs=prior)
            assert posterior.dtype == torch.float64  # Should maintain float type


class TestIntegration:
    """Integration tests for VMP algorithm"""

    def test_full_inference_pipeline(self):
        """Test complete inference pipeline"""
        # Create configuration
        config = InferenceConfig(num_iterations=10, learning_rate=0.1, use_temporal_processing=True)

        # Create VMP algorithm
        vmp = VariationalMessagePassing(config)

        # Create realistic generative model
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(
            learning_rate=0.01,
            use_sparse=True,  # Use sparse initialization for simple model
            use_gpu=False,
        )

        # Create discrete generative model
        model = DiscreteGenerativeModel(dimensions=dims, parameters=params)

        # Manually set the matrices to known values for testing
        model.A = torch.nn.Parameter(
            torch.tensor(
                [[0.8, 0.1, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05], [0.1, 0.1, 0.9, 0.0]],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        # Set up transition dynamics
        B = torch.zeros(4, 4, 2, dtype=torch.float32)
        # Action 0: stay in place
        for i in range(4):
            B[i, i, 0] = 1.0

        # Action 1: move to next state (cyclic)
        for i in range(4):
            B[(i + 1) % 4, i, 1] = 1.0

        model.B = torch.nn.Parameter(B, requires_grad=False)
        model.D = torch.nn.Parameter(torch.ones(4, dtype=torch.float32) / 4, requires_grad=False)

        # Run inference over sequence
        observation_sequence = torch.tensor([0, 1, 2, 2, 1, 0], dtype=torch.long)

        posteriors = []
        belief = None

        for obs in observation_sequence:
            belief = vmp.infer_states(obs.unsqueeze(0), model, prior_beliefs=belief)
            posteriors.append(belief.squeeze())

        # Check that beliefs evolve over time
        assert len(posteriors) == len(observation_sequence)

        # Beliefs should be different
        for i in range(1, len(posteriors)):
            assert not torch.allclose(posteriors[i], posteriors[i - 1])

    def test_backward_compatibility(self):
        """Test backward compatibility with 'prior' parameter"""
        config = InferenceConfig()
        vmp = VariationalMessagePassing(config)

        model = Mock()
        model.A = torch.eye(3)
        model.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)

        obs = torch.tensor(1)
        prior = torch.tensor([0.2, 0.5, 0.3])

        # Test using 'prior' parameter (old style)
        posterior = vmp.infer_states(obs, model, prior=prior)

        assert posterior is not None
        assert torch.allclose(posterior.sum(), torch.tensor(1.0))

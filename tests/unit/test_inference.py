"""
Module for FreeAgentics Active Inference implementation.
"""

from typing import Optional

import pytest
import torch

from inference.engine.active_inference import (
    BeliefPropagation,
    ExpectationMaximization,
    GradientDescentInference,
    InferenceConfig,
    NaturalGradientInference,
    ParticleFilterInference,
    VariationalMessagePassing,
    create_inference_algorithm,
)
from inference.engine.generative_model import (
    ContinuousGenerativeModel,
    DiscreteGenerativeModel,
    ModelDimensions,
    ModelParameters,
)


class TestInferenceConfig:
    """Test InferenceConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = InferenceConfig()
        assert config.num_iterations == 16
        assert config.convergence_threshold == 1e-4
        assert config.learning_rate == 0.1
        assert config.gradient_clip == 1.0
        assert config.use_natural_gradient is True
        assert config.damping_factor == 0.1
        assert config.momentum == 0.9
        assert config.use_gpu is True
        assert config.dtype == torch.float32

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = InferenceConfig(num_iterations=32, learning_rate=0.01, use_gpu=False)
        assert config.num_iterations == 32
        assert config.learning_rate == 0.01
        assert config.use_gpu is False


class TestVariationalMessagePassing:
    """Test VMP algorithm."""

    def setup_method(self) -> None:
        """Set up test model and algorithm"""
        self.dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        # Set up observation model for testing
        # State 0 -> Obs 0, State 1 -> Obs 1, etc.
        self.model.A = torch.zeros(3, 4)
        self.model.A[0, 0] = 0.9
        self.model.A[1, 0] = 0.05
        self.model.A[2, 0] = 0.05
        self.model.A[0, 1] = 0.05
        self.model.A[1, 1] = 0.9
        self.model.A[2, 1] = 0.05
        self.model.A[0, 2] = 0.05
        self.model.A[1, 2] = 0.05
        self.model.A[2, 2] = 0.9
        self.model.A[:, 3] = 1 / 3  # Uniform for state 3
        self.config = InferenceConfig(use_gpu=False, num_iterations=10)
        self.vmp = VariationalMessagePassing(self.config)

    def test_single_observation_inference(self) -> None:
        """Test inference with single observation"""
        # Observe state 1
        observation = torch.tensor(1)
        beliefs = self.vmp.infer_states(observation, self.model)
        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))
        assert beliefs[1] > 0.65  # Should be confident about state 1 (realistic threshold)

    def test_batch_observation_inference(self) -> None:
        """Test inference with batch of observations"""
        observations = torch.tensor([0, 1, 2])
        beliefs = self.vmp.infer_states(observations, self.model)
        assert beliefs.shape == (3, 4)
        assert torch.allclose(beliefs.sum(dim=1), torch.ones(3))
        assert beliefs[0, 0] > 0.65  # Realistic threshold for batch optimization
        assert beliefs[1, 1] > 0.65
        assert beliefs[2, 2] > 0.65

    def test_observation_distribution_inference(self) -> None:
        """Test inference with observation distributions"""
        # Soft observation - mostly obs 1 with some uncertainty
        obs_dist = torch.tensor([[0.1, 0.8, 0.1]])
        beliefs = self.vmp.infer_states(obs_dist, self.model)
        assert beliefs.shape == (1, 4)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))
        assert beliefs[0, 1] > beliefs[0, 0]  # State 1 should be most likely

    def test_prior_influence(self) -> None:
        """Test influence of prior on inference"""
        observation = torch.tensor(1)
        # Strong prior for state 0
        prior = torch.tensor([0.9, 0.05, 0.03, 0.02])
        beliefs_with_prior = self.vmp.infer_states(observation, self.model, prior)
        # Compare with uniform prior
        beliefs_uniform = self.vmp.infer_states(observation, self.model)
        # Prior should pull beliefs toward state 0
        assert beliefs_with_prior[0] > beliefs_uniform[0]

    def test_free_energy_computation(self) -> None:
        """Test free energy calculation"""
        observation = torch.tensor(1)
        beliefs = self.vmp.infer_states(observation, self.model)
        free_energy = self.vmp.compute_free_energy(beliefs, observation, self.model)
        assert isinstance(free_energy.item(), float)
        assert not torch.isnan(free_energy)
        assert not torch.isinf(free_energy)

    def test_convergence(self) -> None:
        """Test that algorithm converges"""
        observation = torch.tensor(1)
        # Run with different iteration counts
        config_short = InferenceConfig(use_gpu=False, num_iterations=2)
        config_long = InferenceConfig(use_gpu=False, num_iterations=50)
        vmp_short = VariationalMessagePassing(config_short)
        vmp_long = VariationalMessagePassing(config_long)
        beliefs_short = vmp_short.infer_states(observation, self.model)
        beliefs_long = vmp_long.infer_states(observation, self.model)
        # Should converge to similar result
        assert torch.allclose(beliefs_short, beliefs_long, atol=0.01)


class TestBeliefPropagation:
    """Test Belief Propagation algorithm."""

    def setup_method(self) -> None:
        """Set up test environment"""
        self.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        self.config = InferenceConfig(use_gpu=False)
        self.bp = BeliefPropagation(self.config)

    def test_basic_inference(self) -> None:
        """Test basic belief propagation"""
        observation = torch.tensor(1)
        beliefs = self.bp.infer_states(observation, self.model)
        assert beliefs.shape == (3,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))

    def test_temporal_update(self) -> None:
        """Test temporal belief update"""
        observation = torch.tensor(1)
        previous_beliefs = torch.tensor([0.7, 0.2, 0.1])
        action = torch.tensor(0)
        beliefs = self.bp.infer_states(
            observation, self.model, actions=action, previous_states=previous_beliefs
        )
        assert beliefs.shape == (3,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))
        # Should be influenced by both observation and transition
        beliefs_no_temporal = self.bp.infer_states(observation, self.model)
        assert not torch.allclose(beliefs, beliefs_no_temporal)


class TestGradientDescentInference:
    """Test gradient-based inference."""

    def setup_method(self) -> None:
        """Set up continuous model"""
        self.dims = ModelDimensions(num_states=2, num_observations=2, num_actions=1)
        self.params = ModelParameters(use_gpu=False)
        self.model = ContinuousGenerativeModel(self.dims, self.params, hidden_dim=16)
        self.config = InferenceConfig(use_gpu=False, num_iterations=20)
        self.gd = GradientDescentInference(self.config)

    def test_continuous_inference(self) -> None:
        """Test inference for continuous states"""
        observation = torch.randn(2)
        mean, var = self.gd.infer_states(observation, self.model)
        assert mean.shape == (2,)
        assert var.shape == (2,)
        assert torch.all(var > 0)  # Variance should be positive

    def test_prior_initialization(self) -> None:
        """Test initialization from prior"""
        observation = torch.randn(2)
        # Test with tuple prior
        prior_mean = torch.tensor([1.0, -1.0])
        prior_var = torch.tensor([0.5, 0.5])
        prior = (prior_mean, torch.log(prior_var))
        mean, var = self.gd.infer_states(observation, self.model, prior)
        assert mean.shape == (2,)
        assert var.shape == (2,)

    def test_free_energy_continuous(self) -> None:
        """Test free energy for continuous model"""
        observation = torch.randn(2)
        mean, var = self.gd.infer_states(observation, self.model)
        free_energy = self.gd.compute_free_energy((mean, var), observation, self.model)
        assert isinstance(free_energy.item(), float)
        assert not torch.isnan(free_energy)


class TestNaturalGradientInference:
    """Test natural gradient inference."""

    def setup_method(self) -> None:
        """Set up test environment"""
        self.dims = ModelDimensions(num_states=2, num_observations=2, num_actions=1)
        self.params = ModelParameters(use_gpu=False)
        self.model = ContinuousGenerativeModel(self.dims, self.params, hidden_dim=16)
        self.config = InferenceConfig(use_gpu=False, num_iterations=20)
        self.ng = NaturalGradientInference(self.config)

    def test_natural_gradient_step(self) -> None:
        """Test natural gradient computation"""
        grad_mean = torch.tensor([0.1, -0.2])
        grad_log_var = torch.tensor([0.05, -0.05])
        mean = torch.tensor([0.0, 0.0])
        log_var = torch.tensor([0.0, 0.0])  # var = 1
        nat_grad_mean, nat_grad_log_var = self.ng._natural_gradient_step(
            grad_mean, grad_log_var, mean, log_var
        )
        # Natural gradient should be scaled by Fisher information
        assert nat_grad_mean.shape == grad_mean.shape
        assert nat_grad_log_var.shape == grad_log_var.shape

    def test_natural_vs_standard_gradient(self) -> None:
        """Compare natural gradient with standard gradient"""
        observation = torch.randn(2)
        # Natural gradient
        mean_ng, var_ng = self.ng.infer_states(observation, self.model)
        # Standard gradient
        gd = GradientDescentInference(self.config)
        mean_gd, var_gd = gd.infer_states(observation, self.model)
        # Both should converge but potentially to slightly different values
        assert mean_ng.shape == mean_gd.shape
        assert var_ng.shape == var_gd.shape


class TestExpectationMaximization:
    """Test EM algorithm"""

    def setup_method(self) -> None:
        """Set up test environment"""
        self.dims = ModelDimensions(num_states=3, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False, learning_rate=0.1)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        self.config = InferenceConfig(use_gpu=False)
        self.em = ExpectationMaximization(self.config)

    def test_e_step(self) -> None:
        """Test expectation step"""
        observation = torch.tensor(1)
        beliefs = self.em.infer_states(observation, self.model)
        assert beliefs.shape == (3,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))

    def test_em_iteration(self) -> None:
        """Test complete EM iteration"""
        observations = [torch.tensor(i % 3) for i in range(10)]
        actions = [torch.tensor(i % 2) for i in range(9)]
        # Store original parameters
        A_orig = self.model.A.clone()
        # Run EM iteration
        beliefs = self.em.em_iteration(observations, self.model, actions=actions)
        assert len(beliefs) == len(observations)
        # Parameters should have changed
        assert not torch.allclose(self.model.A, A_orig)

    def test_parameter_update(self) -> None:
        """Test M-step parameter updates"""
        # Create synthetic data
        observations = [torch.tensor(i % 3) for i in range(20)]
        beliefs = [torch.ones(3) / 3 for _ in range(20)]
        actions = [torch.tensor(0) for _ in range(19)]
        # Store original
        A_orig = self.model.A.clone()
        B_orig = self.model.B.clone()
        # Update parameters
        self.em.update_parameters(observations, beliefs, self.model, actions)
        # Should have changed
        assert not torch.allclose(self.model.A, A_orig)
        assert not torch.allclose(self.model.B, B_orig)


class TestParticleFilterInference:
    """Test particle filter algorithm"""

    def setup_method(self) -> None:
        """Set up test environment"""
        self.dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(self.dims, self.params)
        self.config = InferenceConfig(use_gpu=False)
        self.pf = ParticleFilterInference(self.config, num_particles=50)

    def test_particle_initialization(self) -> None:
        """Test particle initialization"""
        observation = torch.tensor(1)
        mean, particles, weights = self.pf.infer_states(observation, self.model)
        assert particles.shape == (50,)  # num_particles
        assert weights.shape == (50,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0))
        # Mean should be a probability distribution
        assert mean.shape == (4,)  # num_states
        assert torch.allclose(mean.sum(), torch.tensor(1.0), atol=0.01)

    def test_resampling(self) -> None:
        """Test particle resampling"""
        # Create particles with very uneven weights
        particles = torch.arange(50).float()
        weights = torch.zeros(50)
        weights[0] = 0.9  # One particle has most weight
        weights[1:] = 0.1 / 49
        new_particles, new_weights = self.pf._resample(particles, weights)
        assert new_particles.shape == particles.shape
        assert torch.allclose(new_weights.sum(), torch.tensor(1.0))
        assert torch.allclose(
            new_weights, torch.ones(50) / 50
        )  # Should be uniform after resampling

    def test_continuous_model_particles(self) -> None:
        """Test particle filter with continuous model"""
        cont_dims = ModelDimensions(num_states=2, num_observations=2, num_actions=1)
        cont_params = ModelParameters(use_gpu=False)
        cont_model = ContinuousGenerativeModel(cont_dims, cont_params)
        observation = torch.randn(2)
        mean, particles, weights = self.pf.infer_states(observation, cont_model)
        assert mean.shape == (2,)  # Continuous state dimension
        assert particles.shape == (50, 2)
        assert weights.shape == (50,)

    def test_sequential_update(self) -> None:
        """Test sequential particle updates"""
        observations = [torch.tensor(i % 3) for i in range(5)]
        particles = None
        weights = None
        for obs in observations:
            mean, particles, weights = self.pf.infer_states(
                obs, self.model, particles=particles, weights=weights
            )
            assert particles is not None
            assert weights is not None
            assert torch.allclose(weights.sum(), torch.tensor(1.0))


class TestInferenceFactory:
    """Test inference algorithm factory"""

    def test_create_vmp(self) -> None:
        """Test VMP creation"""
        algo = create_inference_algorithm("vmp")
        assert isinstance(algo, VariationalMessagePassing)

    def test_create_bp(self) -> None:
        """Test BP creation"""
        algo = create_inference_algorithm("bp")
        assert isinstance(algo, BeliefPropagation)

    def test_create_gradient(self) -> None:
        """Test gradient descent creation"""
        algo = create_inference_algorithm("gradient")
        assert isinstance(algo, GradientDescentInference)

    def test_create_natural(self) -> None:
        """Test natural gradient creation"""
        algo = create_inference_algorithm("natural")
        assert isinstance(algo, NaturalGradientInference)

    def test_create_em(self) -> None:
        """Test EM creation"""
        algo = create_inference_algorithm("em")
        assert isinstance(algo, ExpectationMaximization)

    def test_create_particle(self) -> None:
        """Test particle filter creation"""
        algo = create_inference_algorithm("particle", num_particles=100)
        assert isinstance(algo, ParticleFilterInference)
        assert algo.num_particles == 100

    def test_invalid_algorithm(self) -> None:
        """Test invalid algorithm type"""
        with pytest.raises(ValueError):
            create_inference_algorithm("invalid")

    def test_custom_config(self) -> None:
        """Test creation with custom config"""
        config = InferenceConfig(num_iterations=50, use_gpu=False)
        algo = create_inference_algorithm("vmp", config=config)
        assert algo.config.num_iterations == 50
        assert algo.config.use_gpu is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

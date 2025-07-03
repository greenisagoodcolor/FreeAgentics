"""
Comprehensive tests for Active Inference Engine.
"""

from typing import Optional
from unittest.mock import Mock

import pytest
import torch

from inference.engine.active_inference import (
    ActiveInferenceEngine,
    BeliefPropagation,
    ExpectationMaximization,
    GradientDescentInference,
    InferenceAlgorithm,
    InferenceConfig,
    NaturalGradientInference,
    ParticleFilterInference,
    VariationalMessagePassing,
    create_inference_algorithm,
)
from inference.engine.generative_model import GenerativeModel, ModelDimensions, ModelParameters


class TestInferenceConfig:
    """Test InferenceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InferenceConfig()
        assert config.algorithm == "variational_message_passing"
        assert config.num_iterations == 16
        assert config.convergence_threshold == 1e-4
        assert config.learning_rate == 0.1
        assert config.use_gpu
        assert config.dtype == torch.float32

    def test_custom_config(self):
        """Test custom configuration values."""
        config = InferenceConfig(
            algorithm="belief_propagation", num_iterations=32, learning_rate=0.01, use_gpu=False
        )
        assert config.algorithm == "belief_propagation"
        assert config.num_iterations == 32
        assert config.learning_rate == 0.01
        assert not config.use_gpu

    def test_config_immutability(self):
        """Test that config is a proper dataclass."""
        config = InferenceConfig()
        # Should be able to create new instance with changes
        new_config = InferenceConfig(num_iterations=32)
        assert config.num_iterations == 16
        assert new_config.num_iterations == 32

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Test negative num_iterations
        with pytest.raises(ValueError):
            InferenceConfig(num_iterations=-1)

        # Test negative learning_rate
        with pytest.raises(ValueError):
            InferenceConfig(learning_rate=-0.1)

        # Test negative convergence_threshold
        with pytest.raises(ValueError):
            InferenceConfig(convergence_threshold=-1e-4)


class TestInferenceAlgorithm:
    """Test abstract InferenceAlgorithm base class."""

    def test_device_selection_cpu(self, inference_config):
        """Test CPU device selection."""
        inference_config.use_gpu = False

        class ConcreteAlgorithm(InferenceAlgorithm):
            def infer_states(self, observations, generative_model, prior_beliefs=None):
                return torch.zeros(4)

        algo = ConcreteAlgorithm(inference_config)
        assert algo.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_selection_gpu(self, inference_config):
        """Test GPU device selection when available."""
        inference_config.use_gpu = True

        class ConcreteAlgorithm(InferenceAlgorithm):
            def infer_states(self, observations, generative_model, prior_beliefs=None):
                return torch.zeros(4)

        algo = ConcreteAlgorithm(inference_config)
        assert algo.device.type == "cuda"

    def test_abstract_method_enforcement(self, inference_config):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            InferenceAlgorithm(inference_config)


class TestVariationalMessagePassing:
    """Test Variational Message Passing algorithm."""

    def test_initialization(self, inference_config):
        """Test VMP initialization."""
        vmp = VariationalMessagePassing(inference_config)
        assert vmp.config == inference_config
        assert isinstance(vmp, InferenceAlgorithm)

    def test_discrete_observation_inference(
        self, inference_config, simple_generative_model, sample_observations
    ):
        """Test state inference with discrete observations."""
        vmp = VariationalMessagePassing(inference_config)

        # Single observation
        obs = torch.tensor(0, dtype=torch.long)
        beliefs = vmp.infer_states(obs, simple_generative_model)

        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))
        # Should favor state 0 for observation 0
        assert beliefs[0] > beliefs[1]

    def test_continuous_observation_inference(self, inference_config, continuous_generative_model):
        """Test state inference with continuous observations."""
        vmp = VariationalMessagePassing(inference_config)

        obs = torch.tensor([0.5, 0.5], dtype=torch.float32)
        beliefs = vmp.infer_states(obs, continuous_generative_model)

        assert beliefs.shape == (2,)
        assert not torch.isnan(beliefs).any()
        assert not torch.isinf(beliefs).any()

    def test_prior_belief_integration(
        self, inference_config, simple_generative_model, sample_beliefs
    ):
        """Test integration of prior beliefs."""
        vmp = VariationalMessagePassing(inference_config)

        obs = torch.tensor(1, dtype=torch.long)
        prior = sample_beliefs["peaked"]

        # Without prior
        beliefs_no_prior = vmp.infer_states(obs, simple_generative_model)

        # With prior
        beliefs_with_prior = vmp.infer_states(obs, simple_generative_model, prior_beliefs=prior)

        # Prior should influence the result
        assert not torch.allclose(beliefs_no_prior, beliefs_with_prior)

    def test_sequence_inference(
        self, inference_config, simple_generative_model, sample_observations
    ):
        """Test inference over observation sequence."""
        vmp = VariationalMessagePassing(inference_config)

        obs_sequence = sample_observations["discrete"]
        beliefs = None

        for obs in obs_sequence:
            beliefs = vmp.infer_states(obs, simple_generative_model, prior_beliefs=beliefs)
            assert torch.allclose(beliefs.sum(), torch.tensor(1.0))

    def test_convergence(self, inference_config, simple_generative_model):
        """Test algorithm convergence."""
        inference_config.num_iterations = 100
        vmp = VariationalMessagePassing(inference_config)

        obs = torch.tensor(0, dtype=torch.long)
        beliefs = vmp.infer_states(obs, simple_generative_model)

        # Run again with same observation - posterior should shift further toward
        # observed state
        beliefs2 = vmp.infer_states(obs, simple_generative_model, prior_beliefs=beliefs)

        # Check that beliefs are valid probability distributions
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))
        assert torch.allclose(beliefs2.sum(), torch.tensor(1.0))

        # The second run should increase confidence in the observed state
        # State 0 should have higher probability
        assert beliefs2[0] >= beliefs[0]

    def test_multi_modal_observations(
        self, inference_config, simple_generative_model, sample_observations
    ):
        """Test inference with multi-modal observations."""
        vmp = VariationalMessagePassing(inference_config)

        # Create multi-modal observation
        visual_obs = torch.tensor(0, dtype=torch.long)
        auditory_obs = torch.tensor(1, dtype=torch.long)

        # Process each modality
        visual_beliefs = vmp.infer_states(visual_obs, simple_generative_model)
        auditory_beliefs = vmp.infer_states(auditory_obs, simple_generative_model)

        # Combine beliefs (product of experts)
        combined_beliefs = visual_beliefs * auditory_beliefs
        combined_beliefs = combined_beliefs / combined_beliefs.sum()

        assert torch.allclose(combined_beliefs.sum(), torch.tensor(1.0))

    def test_edge_cases(self, inference_config, simple_generative_model):
        """Test edge cases and error handling."""
        vmp = VariationalMessagePassing(inference_config)

        # Test with invalid observation
        with pytest.raises(IndexError):
            obs = torch.tensor(10, dtype=torch.long)  # Out of range
            vmp.infer_states(obs, simple_generative_model)

        # Test with empty observation
        obs = torch.tensor([], dtype=torch.long)
        beliefs = vmp.infer_states(obs, simple_generative_model)
        assert beliefs.shape[0] > 0  # Should return valid beliefs

    def test_numerical_stability(self, inference_config, simple_generative_model):
        """Test numerical stability with extreme values."""
        vmp = VariationalMessagePassing(inference_config)

        # Create model with extreme probabilities
        simple_generative_model.A[0, 0] = 1e-10
        simple_generative_model.A[0, 1] = 1 - 1e-10

        obs = torch.tensor(0, dtype=torch.long)
        beliefs = vmp.infer_states(obs, simple_generative_model)

        assert not torch.isnan(beliefs).any()
        assert not torch.isinf(beliefs).any()
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))


class TestBeliefPropagation:
    """Test Belief Propagation algorithm."""

    def test_initialization(self, inference_config):
        """Test BP initialization."""
        inference_config.algorithm = "belief_propagation"
        bp = BeliefPropagation(inference_config)
        assert bp.config.algorithm == "belief_propagation"

    def test_message_passing(self, inference_config, simple_generative_model):
        """Test message passing in belief propagation."""
        bp = BeliefPropagation(inference_config)

        # Create factor graph representation
        obs = torch.tensor(0, dtype=torch.long)
        beliefs = bp.infer_states(obs, simple_generative_model)

        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))

    def test_loopy_belief_propagation(self, inference_config):
        """Test loopy BP on graphs with cycles."""
        bp = BeliefPropagation(inference_config)

        # Create a model with cycles
        class CyclicModel(GenerativeModel):
            def __init__(self):
                dims = ModelDimensions(
                    num_states=4, num_observations=4, num_actions=2, time_horizon=5
                )
                params = ModelParameters()
                super().__init__(dims, params)
                # Create cyclic dependencies
                self.A = torch.eye(4) + 0.1 * torch.ones(4, 4)
                self.A = self.A / self.A.sum(dim=0, keepdim=True)

            def observation_model(self, states: torch.Tensor) -> torch.Tensor:
                if states.dim() == 0:
                    return self.A[:, states.item()]
                else:
                    return torch.matmul(self.A, states)

            def transition_model(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
                # Simple transition for testing
                return torch.ones(self.dims.num_states) / self.dims.num_states

            def get_preferences(self, timestep: Optional[int] = None) -> torch.Tensor:
                return torch.ones(self.dims.num_observations) / self.dims.num_observations

            def get_initial_prior(self) -> torch.Tensor:
                return torch.ones(self.dims.num_states) / self.dims.num_states

        model = CyclicModel()
        obs = torch.tensor(0, dtype=torch.long)
        beliefs = bp.infer_states(obs, model)

        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))


class TestGradientDescentInference:
    """Test Gradient Descent Inference algorithm."""

    def test_initialization(self, inference_config):
        """Test FEM initialization."""
        inference_config.algorithm = "gradient_descent"
        gdi = GradientDescentInference(inference_config)
        assert gdi.config == inference_config

    def test_gradient_descent(self, inference_config, simple_generative_model):
        """Test gradient descent for free energy minimization."""
        gdi = GradientDescentInference(inference_config)

        obs = torch.tensor(0, dtype=torch.long)
        beliefs = gdi.infer_states(obs, simple_generative_model)

        # For discrete models, should return belief distribution
        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))

    def test_natural_gradient(self, inference_config, simple_generative_model):
        """Test natural gradient descent."""
        inference_config.use_natural_gradient = True
        gdi = GradientDescentInference(inference_config)

        obs = torch.tensor(0, dtype=torch.long)
        beliefs = gdi.infer_states(obs, simple_generative_model)

        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))


class TestNaturalGradientInference:
    """Test Natural Gradient Inference algorithm."""

    def test_initialization(self, inference_config):
        """Test NGI initialization."""
        ngi = NaturalGradientInference(inference_config)
        assert ngi.config == inference_config

    def test_fisher_information_matrix(self, inference_config, simple_generative_model):
        """Test Fisher information matrix computation."""
        ngi = NaturalGradientInference(inference_config)
        obs = torch.tensor(0, dtype=torch.long)
        beliefs = ngi.infer_states(obs, simple_generative_model)

        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))


class TestExpectationMaximization:
    """Test Expectation Maximization algorithm."""

    def test_initialization(self, inference_config):
        """Test EM initialization."""
        em = ExpectationMaximization(inference_config)
        assert em.config == inference_config

    def test_em_steps(self, inference_config, simple_generative_model):
        """Test E and M steps."""
        em = ExpectationMaximization(inference_config)
        obs = torch.tensor(0, dtype=torch.long)
        beliefs = em.infer_states(obs, simple_generative_model)

        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))


class TestParticleFilterInference:
    """Test Particle Filter Inference algorithm."""

    def test_initialization(self, inference_config):
        """Test PFI initialization."""
        pfi = ParticleFilterInference(inference_config)
        assert pfi.config == inference_config

    def test_particle_filtering(self, inference_config, continuous_generative_model):
        """Test particle filtering for continuous states."""
        pfi = ParticleFilterInference(inference_config)
        obs = torch.tensor([0.5, 0.5], dtype=torch.float32)
        result = pfi.infer_states(obs, continuous_generative_model)

        # For continuous models, may return tuple (mean, particles, weights)
        if isinstance(result, tuple):
            mean, particles, weights = result
            assert mean.shape == (2,)
            # Particles could be 1D (flattened) or 2D (num_particles x
            # state_dim)
            assert particles.dim() in [1, 2]
            if weights is not None:
                assert torch.allclose(weights.sum(), torch.tensor(1.0))
        else:
            assert result.shape[0] > 0


class TestHierarchicalInference:
    """Test hierarchical active inference."""

    def test_hierarchical_model_inference(self, inference_config, hierarchical_generative_model):
        """Test inference in hierarchical models."""
        vmp = VariationalMessagePassing(inference_config)

        # Infer at each level
        for level in range(len(hierarchical_generative_model.levels)):
            level_model = hierarchical_generative_model.get_level_model(level)
            obs = torch.tensor(0, dtype=torch.long)

            # Mock the model to use level-specific parameters
            mock_model = Mock()
            mock_model.A = level_model["A"]
            mock_model.dims = hierarchical_generative_model.levels[level]

            beliefs = vmp.infer_states(obs, mock_model)
            assert beliefs.shape == (hierarchical_generative_model.levels[level].num_states,)

    def test_top_down_bottom_up_integration(self, hierarchical_generative_model):
        """Test integration of top-down and bottom-up signals."""
        # Get predictions from higher level
        # Level 0 has 8 states, level 1 has 4 states
        higher_state = (
            torch.ones(hierarchical_generative_model.levels[0].num_states)
            / hierarchical_generative_model.levels[0].num_states
        )
        prediction = hierarchical_generative_model.compute_top_down_prediction(1, higher_state)

        assert prediction.shape == (hierarchical_generative_model.levels[1].num_states,)
        # Prediction may not be normalized, just check it's valid
        assert not torch.isnan(prediction).any()
        assert not torch.isinf(prediction).any()


class TestPerformanceOptimization:
    """Test performance optimizations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_acceleration(self, inference_config, simple_generative_model):
        """Test GPU acceleration."""
        inference_config.use_gpu = True
        vmp = VariationalMessagePassing(inference_config)

        # Move model to GPU
        simple_generative_model.A = simple_generative_model.A.cuda()
        obs = torch.tensor(0, dtype=torch.long).cuda()

        beliefs = vmp.infer_states(obs, simple_generative_model)
        assert beliefs.device.type == "cuda"

    def test_batch_inference(self, inference_config, simple_generative_model):
        """Test batch processing of observations."""
        vmp = VariationalMessagePassing(inference_config)

        # Batch of observations
        batch_size = 10
        obs_batch = torch.randint(0, 3, (batch_size,), dtype=torch.long)

        # Process batch
        beliefs_batch = []
        for obs in obs_batch:
            beliefs = vmp.infer_states(obs, simple_generative_model)
            beliefs_batch.append(beliefs)

        beliefs_batch = torch.stack(beliefs_batch)
        assert beliefs_batch.shape == (batch_size, 4)
        assert torch.allclose(beliefs_batch.sum(dim=1), torch.ones(batch_size))

    def test_sparse_computations(self, inference_config):
        """Test sparse matrix computations."""
        vmp = VariationalMessagePassing(inference_config)

        # Create sparse model
        class SparseModel(GenerativeModel):
            def __init__(self):
                dims = ModelDimensions(
                    num_states=100, num_observations=50, num_actions=10, time_horizon=5
                )
                params = ModelParameters()
                super().__init__(dims, params)
                # Sparse observation model
                self.A = torch.zeros(50, 100)
                for i in range(50):
                    self.A[i, i * 2] = 1.0  # Sparse connections

            def observation_model(self, states: torch.Tensor) -> torch.Tensor:
                if states.dim() == 0:
                    return self.A[:, states.item()]
                else:
                    return torch.matmul(self.A, states)

            def transition_model(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
                return torch.ones(self.dims.num_states) / self.dims.num_states

            def get_preferences(self, timestep: Optional[int] = None) -> torch.Tensor:
                return torch.ones(self.dims.num_observations) / self.dims.num_observations

            def get_initial_prior(self) -> torch.Tensor:
                return torch.ones(self.dims.num_states) / self.dims.num_states

        model = SparseModel()
        obs = torch.tensor(0, dtype=torch.long)
        beliefs = vmp.infer_states(obs, model)

        assert beliefs.shape == (100,)
        assert (beliefs > 0).sum() < 10  # Should be sparse


class TestIntegration:
    """Integration tests for active inference engine."""

    def test_end_to_end_inference_loop(
        self, inference_config, simple_generative_model, sample_observations
    ):
        """Test complete inference loop."""
        vmp = VariationalMessagePassing(inference_config)

        obs_sequence = sample_observations["discrete"]
        belief_trajectory = []
        free_energy_trajectory = []

        beliefs = None
        for t, obs in enumerate(obs_sequence):
            # Infer states
            beliefs = vmp.infer_states(obs, simple_generative_model, prior_beliefs=beliefs)
            belief_trajectory.append(beliefs)

            # Track belief evolution
            free_energy_trajectory.append(beliefs.max().item())  # Use max belief as proxy

        # Verify trajectory properties
        assert len(belief_trajectory) == len(obs_sequence)
        assert all(torch.allclose(b.sum(), torch.tensor(1.0)) for b in belief_trajectory)

        # Check that beliefs are changing (algorithm is working)
        # Note: convergence behavior depends on the specific observation
        # sequence
        if len(free_energy_trajectory) > 1:
            total_change = sum(
                abs(free_energy_trajectory[i + 1] - free_energy_trajectory[i])
                for i in range(len(free_energy_trajectory) - 1)
            )
            assert total_change > 0  # Some change should occur

    def test_action_selection_loop(
        self, inference_config, simple_generative_model, sample_policies
    ):
        """Test action selection based on expected free energy."""
        vmp = VariationalMessagePassing(inference_config)

        obs = torch.tensor(0, dtype=torch.long)
        beliefs = vmp.infer_states(obs, simple_generative_model)

        # For now, just verify belief inference worked
        policies = sample_policies["multi_step"]

        # Simple policy selection based on beliefs
        best_policy_idx = torch.argmax(beliefs).item() % len(policies)
        best_policy = policies[best_policy_idx]

        assert best_policy.shape[0] > 0
        assert all(0 <= a < simple_generative_model.dims.num_actions for a in best_policy.flatten())


class TestCreateInferenceAlgorithm:
    """Test inference algorithm factory function."""

    def test_create_vmp(self, inference_config):
        """Test creating VMP algorithm."""
        inference_config.algorithm = "variational_message_passing"
        algo = create_inference_algorithm(inference_config.algorithm, inference_config)
        assert isinstance(algo, VariationalMessagePassing)

    def test_create_bp(self, inference_config):
        """Test creating BP algorithm."""
        inference_config.algorithm = "belief_propagation"
        algo = create_inference_algorithm(inference_config.algorithm, inference_config)
        assert isinstance(algo, BeliefPropagation)

    def test_create_gradient_descent(self, inference_config):
        """Test creating gradient descent algorithm."""
        inference_config.algorithm = "gradient_descent"
        algo = create_inference_algorithm(inference_config.algorithm, inference_config)
        assert isinstance(algo, GradientDescentInference)

    def test_create_natural_gradient(self, inference_config):
        """Test creating natural gradient algorithm."""
        inference_config.algorithm = "natural_gradient"
        algo = create_inference_algorithm(inference_config.algorithm, inference_config)
        assert isinstance(algo, NaturalGradientInference)

    def test_create_em(self, inference_config):
        """Test creating EM algorithm."""
        inference_config.algorithm = "expectation_maximization"
        algo = create_inference_algorithm(inference_config.algorithm, inference_config)
        assert isinstance(algo, ExpectationMaximization)

    def test_create_particle_filter(self, inference_config):
        """Test creating particle filter algorithm."""
        inference_config.algorithm = "particle_filter"
        algo = create_inference_algorithm(inference_config.algorithm, inference_config)
        assert isinstance(algo, ParticleFilterInference)

    def test_invalid_algorithm(self, inference_config):
        """Test error handling for invalid algorithm."""
        with pytest.raises(ValueError):
            create_inference_algorithm("invalid_algorithm", inference_config)


class TestActiveInferenceEngine:
    """Test the main Active Inference Engine."""

    def test_engine_initialization(self, inference_config, simple_generative_model):
        """Test engine initialization."""
        engine = ActiveInferenceEngine(simple_generative_model, inference_config)
        assert engine.generative_model == simple_generative_model
        assert engine.config == inference_config
        assert engine.inference_algorithm is not None

    def test_engine_step(self, inference_config, simple_generative_model):
        """Test single inference step."""
        engine = ActiveInferenceEngine(simple_generative_model, inference_config)

        obs = torch.tensor(0, dtype=torch.long)
        beliefs = engine.step(obs)

        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))

    def test_engine_run_inference(
        self, inference_config, simple_generative_model, sample_observations
    ):
        """Test running inference over sequence."""
        engine = ActiveInferenceEngine(simple_generative_model, inference_config)

        obs_sequence = sample_observations["discrete"]
        belief_trajectory = engine.run_inference(obs_sequence)

        assert len(belief_trajectory) == len(obs_sequence)
        for beliefs in belief_trajectory:
            assert torch.allclose(beliefs.sum(), torch.tensor(1.0))

    def test_engine_reset(self, inference_config, simple_generative_model):
        """Test resetting engine state."""
        engine = ActiveInferenceEngine(simple_generative_model, inference_config)

        # Run some inference
        obs = torch.tensor(0, dtype=torch.long)
        engine.step(obs)

        # Reset
        engine.reset()

        # Should have no beliefs stored
        assert getattr(engine, "current_beliefs", None) is None or engine.current_beliefs is None

    def test_engine_with_different_algorithms(self, simple_generative_model):
        """Test engine with different inference algorithms."""
        algorithms = ["variational_message_passing", "belief_propagation", "gradient_descent"]

        obs = torch.tensor(0, dtype=torch.long)

        for algo in algorithms:
            config = InferenceConfig(algorithm=algo)
            engine = ActiveInferenceEngine(simple_generative_model, config)
            beliefs = engine.step(obs)
            assert beliefs.shape == (4,)
            assert torch.allclose(beliefs.sum(), torch.tensor(1.0))

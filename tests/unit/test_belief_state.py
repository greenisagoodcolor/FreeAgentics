"""
Module for FreeAgentics Active Inference implementation.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from inference.engine import (
    BeliefState,
    BeliefStateConfig,
    ContinuousBeliefState,
    DiscreteBeliefState,
    create_belief_state,
    create_continuous_belief_state,
    create_discrete_belief_state,
)


class TestBeliefStateConfig:
    """Test BeliefStateConfig dataclass"""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = BeliefStateConfig()
        assert config.use_gpu is True
        assert config.dtype == torch.float32
        assert config.eps == 1e-8
        assert config.normalize_on_update is True
        assert config.entropy_regularization == 0.0
        assert config.compression_enabled is False
        assert config.sparse_threshold == 1e-6
        assert config.max_history_length == 100

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = BeliefStateConfig(use_gpu=False, eps=1e-6, max_history_length=50)
        assert config.use_gpu is False
        assert config.eps == 1e-6
        assert config.max_history_length == 50


class TestDiscreteBeliefState:
    """Test discrete belief state implementation"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return BeliefStateConfig(use_gpu=False)  # Force CPU for testing

    @pytest.fixture
    def belief_state(self, config):
        """Create test belief state"""
        return DiscreteBeliefState(num_states=4, config=config)

    def test_initialization(self, config) -> None:
        """Test belief state initialization"""
        belief_state = DiscreteBeliefState(num_states=4, config=config)
        assert belief_state.num_states == 4
        assert belief_state.beliefs.shape == (4,)
        assert torch.allclose(belief_state.beliefs, torch.ones(4) / 4)  # Uniform prior
        assert belief_state.update_count == 0

    def test_initialization_with_priors(self, config) -> None:
        """Test initialization with custom priors"""
        initial_beliefs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        belief_state = DiscreteBeliefState(
            num_states=4, config=config, initial_beliefs=initial_beliefs
        )
        assert torch.allclose(belief_state.beliefs, initial_beliefs)

    def test_get_set_beliefs(self, belief_state) -> None:
        """Test getting and setting beliefs"""
        # Test get
        beliefs = belief_state.get_beliefs()
        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs, torch.ones(4) / 4)
        # Test set
        new_beliefs = torch.tensor([0.5, 0.3, 0.1, 0.1])
        belief_state.set_beliefs(new_beliefs)
        assert torch.allclose(belief_state.beliefs, new_beliefs)

    def test_bayesian_update(self, belief_state) -> None:
        """Test Bayesian belief update"""
        # Observe state 0
        observation = torch.tensor(0)
        belief_state.update_beliefs(observation, update_method="bayes")
        # Belief in state 0 should increase
        assert belief_state.beliefs[0] > 0.25  # Greater than uniform
        assert belief_state.update_count == 1
        assert belief_state.metadata["last_update_method"] == "bayes"

    def test_linear_update(self, belief_state) -> None:
        """Test linear interpolation update"""
        observation = torch.tensor(1)
        belief_state.update_beliefs(observation, update_method="linear")
        # Should interpolate toward observed state
        assert belief_state.beliefs[1] > belief_state.beliefs[0]
        assert belief_state.metadata["last_update_method"] == "linear"

    def test_momentum_update(self, belief_state) -> None:
        """Test momentum-based update"""
        # First update
        belief_state.update_beliefs(torch.tensor(0), update_method="momentum")
        # Second update with momentum
        belief_state.update_beliefs(torch.tensor(0), update_method="momentum")
        assert len(belief_state.belief_history) == 2
        assert belief_state.metadata["last_update_method"] == "momentum"

    def test_entropy_calculation(self, belief_state) -> None:
        """Test entropy computation"""
        # Uniform distribution should have maximum entropy
        uniform_entropy = belief_state.entropy()
        expected_entropy = torch.log(torch.tensor(4.0))  # log(num_states)
        assert torch.allclose(uniform_entropy, expected_entropy, atol=1e-6)
        # Concentrated distribution should have lower entropy
        belief_state.set_beliefs(torch.tensor([0.9, 0.05, 0.03, 0.02]))
        concentrated_entropy = belief_state.entropy()
        assert concentrated_entropy < uniform_entropy

    def test_most_likely_state(self, belief_state) -> None:
        """Test most likely state identification"""
        belief_state.set_beliefs(torch.tensor([0.1, 0.6, 0.2, 0.1]))
        assert belief_state.most_likely_state() == 1

    def test_top_k_states(self, belief_state) -> None:
        """Test top-k state retrieval"""
        belief_state.set_beliefs(torch.tensor([0.1, 0.4, 0.3, 0.2]))
        indices, probs = belief_state.get_top_k_states(k=2)
        assert len(indices) == 2
        assert len(probs) == 2
        assert indices[0] == 1  # Most likely
        assert indices[1] == 2  # Second most likely

    def test_sampling(self, belief_state) -> None:
        """Test state sampling"""
        # Set deterministic beliefs for testing
        belief_state.set_beliefs(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        # Should always sample state 0
        samples = [belief_state.sample_state() for _ in range(10)]
        assert all(s == 0 for s in samples)

    def test_temperature_sampling(self, belief_state) -> None:
        """Test temperature-scaled sampling"""
        belief_state.set_beliefs(torch.tensor([0.7, 0.2, 0.05, 0.05]))
        # High temperature should be more random
        high_temp_sample = belief_state.sample_state(temperature=2.0)
        assert high_temp_sample in [0, 1, 2, 3]
        # Low temperature should be more deterministic
        low_temp_sample = belief_state.sample_state(temperature=0.1)
        assert low_temp_sample in [0, 1, 2, 3]

    def test_kl_divergence(self, belief_state) -> None:
        """Test KL divergence computation"""
        # Create another belief state
        other_belief = DiscreteBeliefState(
            num_states=4,
            config=belief_state.config,
            initial_beliefs=torch.tensor([0.4, 0.3, 0.2, 0.1]),
        )
        kl_div = belief_state.kl_divergence(other_belief)
        assert kl_div >= 0  # KL divergence is non-negative
        # KL divergence to self should be zero
        self_kl = belief_state.kl_divergence(belief_state)
        assert torch.allclose(self_kl, torch.tensor(0.0), atol=1e-6)

    def test_history_tracking(self, belief_state) -> None:
        """Test belief history tracking"""
        assert len(belief_state.belief_history) == 0
        # Make some updates
        for i in range(3):
            belief_state.update_beliefs(torch.tensor(i % 4))
        assert len(belief_state.belief_history) == 3
        assert len(belief_state.entropy_history) == 3

    def test_reset_to_uniform(self, belief_state) -> None:
        """Test resetting to uniform distribution"""
        # Make some updates
        belief_state.update_beliefs(torch.tensor(0))
        belief_state.update_beliefs(torch.tensor(1))
        assert belief_state.update_count > 0
        assert len(belief_state.belief_history) > 0
        # Reset
        belief_state.reset_to_uniform()
        assert torch.allclose(belief_state.beliefs, torch.ones(4) / 4)
        assert belief_state.update_count == 0
        assert len(belief_state.belief_history) == 0

    def test_compression(self, belief_state) -> None:
        """Test belief compression"""
        # Enable compression
        belief_state.config.compression_enabled = True
        belief_state.config.sparse_threshold = 0.1
        # Set beliefs with small values
        belief_state.set_beliefs(torch.tensor([0.8, 0.15, 0.03, 0.02]))
        # Compress
        belief_state.compress()
        # Small values should be zeroed
        assert belief_state.beliefs[2] == 0.0
        assert belief_state.beliefs[3] == 0.0

    def test_clone(self, belief_state) -> None:
        """Test belief state cloning"""
        # Make some changes
        belief_state.update_beliefs(torch.tensor(0))
        belief_state.metadata["test_key"] = "test_value"
        # Clone
        cloned = belief_state.clone()
        assert cloned.num_states == belief_state.num_states
        assert torch.allclose(cloned.beliefs, belief_state.beliefs)
        assert cloned.update_count == belief_state.update_count
        assert cloned.metadata["test_key"] == "test_value"
        # Modifications to clone shouldn't affect original
        cloned.update_beliefs(torch.tensor(1))
        assert not torch.allclose(cloned.beliefs, belief_state.beliefs)

    def test_serialization(self, belief_state) -> None:
        """Test serialization to dictionary"""
        belief_state.update_beliefs(torch.tensor(0))
        belief_state.metadata["test_key"] = "test_value"
        # Serialize
        data = belief_state.to_dict()
        assert data["type"] == "DiscreteBeliefState"
        assert data["num_states"] == 4
        assert "beliefs" in data
        assert "config" in data
        assert "metadata" in data
        assert data["metadata"]["test_key"] == "test_value"
        # Deserialize
        new_belief = DiscreteBeliefState(num_states=1, config=BeliefStateConfig()).from_dict(data)
        assert new_belief.num_states == 4
        assert torch.allclose(new_belief.beliefs, belief_state.beliefs)
        assert new_belief.metadata["test_key"] == "test_value"

    def test_file_save_load(self, belief_state) -> None:
        """Test saving and loading from files"""
        belief_state.update_beliefs(torch.tensor(2))
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON save/load
            json_path = Path(tmpdir) / "belief_state.json"
            belief_state.save(json_path)
            loaded_belief = DiscreteBeliefState.load(json_path)
            assert torch.allclose(loaded_belief.beliefs, belief_state.beliefs)
            # Test pickle save/load
            pkl_path = Path(tmpdir) / "belief_state.pkl"
            belief_state.save(pkl_path)
            loaded_belief_pkl = DiscreteBeliefState.load(pkl_path)
            assert torch.allclose(loaded_belief_pkl.beliefs, belief_state.beliefs)


class TestContinuousBeliefState:
    """Test continuous belief state implementation"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return BeliefStateConfig(use_gpu=False)

    @pytest.fixture
    def belief_state(self, config):
        """Create test continuous belief state"""
        return ContinuousBeliefState(state_dim=3, config=config)

    def test_initialization(self, config) -> None:
        """Test continuous belief state initialization"""
        belief_state = ContinuousBeliefState(state_dim=3, config=config)
        assert belief_state.state_dim == 3
        assert belief_state.mean.shape == (3,)
        assert belief_state.cov.shape == (3, 3)
        assert torch.allclose(belief_state.mean, torch.zeros(3))
        assert torch.allclose(belief_state.cov, torch.eye(3))

    def test_initialization_with_parameters(self, config) -> None:
        """Test initialization with custom parameters"""
        mean = torch.tensor([1.0, 2.0, 3.0])
        cov = 2.0 * torch.eye(3)
        belief_state = ContinuousBeliefState(
            state_dim=3, config=config, initial_mean=mean, initial_cov=cov
        )
        assert torch.allclose(belief_state.mean, mean)
        assert torch.allclose(belief_state.cov, cov)

    def test_get_set_beliefs(self, belief_state) -> None:
        """Test getting and setting beliefs"""
        beliefs = belief_state.get_beliefs()
        # get_beliefs returns concatenated mean and log_var, check properties instead
        assert belief_state.mean.shape == (3,)
        assert belief_state.cov.shape == (3, 3)
        assert beliefs.shape == (6,)  # 3 for mean + 3 for log_var
        # Set new beliefs
        new_mean = torch.tensor([1.0, 0.0, -1.0])
        new_cov = 0.5 * torch.eye(3)
        belief_state.set_beliefs((new_mean, new_cov))
        assert torch.allclose(belief_state.mean, new_mean)
        assert torch.allclose(belief_state.cov, new_cov)

    def test_gaussian_bayesian_update(self, belief_state) -> None:
        """Test Gaussian Bayesian update"""
        obs = torch.tensor([1.0, 0.0, 0.0])
        obs_cov = 0.1 * torch.eye(3)
        belief_state.update_beliefs((obs, obs_cov), update_method="bayes")
        # Mean should move toward observation
        assert belief_state.mean[0] > 0
        assert belief_state.update_count == 1
        # Covariance should decrease (more certain)
        assert torch.det(belief_state.cov) < 1.0  # Less than initial determinant

    def test_kalman_update(self, belief_state) -> None:
        """Test Kalman filter update"""
        obs = torch.tensor([0.5, 0.5, 0.5])
        obs_cov = 0.2 * torch.eye(3)
        belief_state.update_beliefs((obs, obs_cov), update_method="kalman")
        # Check that update occurred
        assert not torch.allclose(belief_state.mean, torch.zeros(3))
        assert belief_state.update_count == 1

    def test_entropy_calculation(self, belief_state) -> None:
        """Test differential entropy calculation"""
        entropy = belief_state.entropy()
        assert entropy.item() > 0  # Should be positive for Gaussian
        # Smaller covariance should have lower entropy
        belief_state.set_beliefs((belief_state.mean, 0.1 * torch.eye(3)))
        small_entropy = belief_state.entropy()
        assert small_entropy < entropy

    def test_most_likely_state(self, belief_state) -> None:
        """Test most likely state (mean) retrieval"""
        belief_state.set_beliefs((torch.tensor([1.0, 2.0, 3.0]), belief_state.cov))
        most_likely = belief_state.most_likely_state()
        assert torch.allclose(most_likely, torch.tensor([1.0, 2.0, 3.0]))

    def test_sampling(self, belief_state) -> None:
        """Test state sampling from Gaussian"""
        samples = belief_state.sample_state(num_samples=1000)
        assert samples.shape == (1000, 3)
        # Samples should be roughly centered around mean with sufficient samples
        sample_mean = samples.mean(dim=0)
        # With 1000 samples, the sample mean should be much closer to the true mean
        assert torch.allclose(sample_mean, belief_state.mean, atol=0.1)


class TestFactoryFunctions:
    """Test factory functions for belief state creation"""

    def test_create_discrete_belief_state(self) -> None:
        """Test discrete belief state factory"""
        belief_state = create_discrete_belief_state(num_states=5)
        assert isinstance(belief_state, DiscreteBeliefState)
        assert belief_state.num_states == 5

    def test_create_continuous_belief_state(self) -> None:
        """Test continuous belief state factory"""
        belief_state = create_continuous_belief_state(state_dim=4)
        assert isinstance(belief_state, ContinuousBeliefState)
        assert belief_state.state_dim == 4

    def test_create_belief_state_factory(self) -> None:
        """Test general belief state factory"""
        # Discrete
        discrete_belief = create_belief_state("discrete", num_states=3)
        assert isinstance(discrete_belief, DiscreteBeliefState)
        assert discrete_belief.num_states == 3
        # Continuous
        continuous_belief = create_belief_state("continuous", state_dim=2)
        assert isinstance(continuous_belief, ContinuousBeliefState)
        assert continuous_belief.state_dim == 2
        # Invalid type
        with pytest.raises(ValueError):
            create_belief_state("invalid_type")


class TestBeliefStateIntegration:
    """Test integration with other Active Inference components"""

    def test_discrete_belief_with_inference(self) -> None:
        """Test discrete belief state with inference algorithms"""
        from inference.engine import (
            DiscreteGenerativeModel,
            InferenceConfig,
            ModelDimensions,
            ModelParameters,
            VariationalMessagePassing,
        )

        # Create components
        config = BeliefStateConfig(use_gpu=False)
        belief_state = create_discrete_belief_state(num_states=4, config=config)
        # Create simple generative model
        dims = ModelDimensions(num_states=4, num_observations=2, num_actions=2)
        params = ModelParameters(use_gpu=False)
        model = DiscreteGenerativeModel(dims, params)
        # Create inference algorithm
        inference_config = InferenceConfig(num_iterations=5)
        inference = VariationalMessagePassing(inference_config)
        # Test inference with belief state
        observation = torch.tensor(0)
        updated_beliefs = inference.infer_states(observation, model, belief_state.get_beliefs())
        # Update belief state with results
        belief_state.set_beliefs(updated_beliefs)
        assert belief_state.beliefs.shape == (4,)
        assert torch.allclose(belief_state.beliefs.sum(), torch.tensor(1.0), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])

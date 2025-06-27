"""
Module for FreeAgentics Active Inference implementation.
"""

import pytest
import torch

from inference.engine.generative_model import (
    ContinuousGenerativeModel,
    DiscreteGenerativeModel,
    ModelDimensions,
    ModelParameters,
)
from inference.engine.parameter_learning import (
    ContinuousParameterLearner,
    DiscreteParameterLearner,
    Experience,
    ExperienceBuffer,
    LearningConfig,
    OnlineParameterLearner,
    create_parameter_learner,
)


class TestLearningConfig:
    def test_default_config(self) -> None:
        """Test default learning configuration"""
        config = LearningConfig()
        assert config.learning_rate_A == 0.01
        assert config.learning_rate_B == 0.01
        assert config.use_bayesian_learning is True
        assert config.use_experience_replay is True
        assert config.replay_buffer_size == 10000

    def test_custom_config(self) -> None:
        """Test custom learning configuration"""
        config = LearningConfig(learning_rate_A=0.05, use_bayesian_learning=False, use_gpu=False)
        assert config.learning_rate_A == 0.05
        assert config.use_bayesian_learning is False
        assert config.use_gpu is False


class TestExperienceBuffer:
    def test_buffer_initialization(self) -> None:
        """Test buffer initialization"""
        buffer = ExperienceBuffer(max_size=100)
        assert len(buffer) == 0
        assert buffer.max_size == 100

    def test_add_experience(self) -> None:
        """Test adding experiences"""
        buffer = ExperienceBuffer(max_size=2)
        exp1 = Experience(
            state=torch.tensor([1.0, 0.0]),
            action=torch.tensor([0.0, 1.0]),
            observation=torch.tensor([0.5]),
            next_state=torch.tensor([0.0, 1.0]),
        )
        buffer.add(exp1)
        assert len(buffer) == 1
        exp2 = Experience(
            state=torch.tensor([0.0, 1.0]),
            action=torch.tensor([1.0, 0.0]),
            observation=torch.tensor([0.7]),
            next_state=torch.tensor([1.0, 0.0]),
        )
        buffer.add(exp2)
        assert len(buffer) == 2
        exp3 = Experience(
            state=torch.tensor([0.5, 0.5]),
            action=torch.tensor([0.5, 0.5]),
            observation=torch.tensor([0.6]),
            next_state=torch.tensor([0.4, 0.6]),
        )
        buffer.add(exp3)
        assert len(buffer) == 2

    def test_sample_experiences(self) -> None:
        """Test sampling from buffer"""
        buffer = ExperienceBuffer(max_size=10)
        for i in range(5):
            exp = Experience(
                state=torch.tensor([float(i)]),
                action=torch.tensor([float(i)]),
                observation=torch.tensor([float(i)]),
                next_state=torch.tensor([float(i + 1)]),
            )
            buffer.add(exp)
        batch = buffer.sample(batch_size=3)
        assert len(batch) == 3
        assert all(isinstance(exp, Experience) for exp in batch)

    def test_clear_buffer(self) -> None:
        """Test clearing buffer"""
        buffer = ExperienceBuffer(max_size=10)
        for i in range(5):
            exp = Experience(
                state=torch.tensor([float(i)]),
                action=torch.tensor([float(i)]),
                observation=torch.tensor([float(i)]),
                next_state=torch.tensor([float(i + 1)]),
            )
            buffer.add(exp)
        assert len(buffer) == 5
        buffer.clear()
        assert len(buffer) == 0


class TestDiscreteParameterLearner:
    def setup_method(self) -> None:
        """Setup for tests"""
        self.config = LearningConfig(use_gpu=False)
        self.model_dims = {"num_states": 4, "num_observations": 3, "num_actions": 2}
        self.learner = DiscreteParameterLearner(self.config, self.model_dims)
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(dims, params)

    def test_initialization(self) -> None:
        """Test learner initialization"""
        assert self.learner.num_states == 4
        assert self.learner.num_observations == 3
        assert self.learner.num_actions == 2
        assert self.learner.pA.shape == (3, 4)
        assert self.learner.pB.shape == (4, 4, 2)
        assert self.learner.pD.shape == (4,)

    def test_bayesian_update(self) -> None:
        """Test Bayesian parameter update"""
        experiences = []
        for i in range(10):
            exp = Experience(
                state=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                action=torch.tensor([0.0, 1.0]),
                observation=torch.tensor([0.0, 1.0, 0.0]),
                next_state=torch.tensor([0.0, 1.0, 0.0, 0.0]),
                timestamp=i,
            )
            experiences.append(exp)
        metrics = self.learner.update_parameters(experiences, self.model)
        assert "A_update_norm" in metrics
        assert "B_update_norm" in metrics
        assert "A_entropy" in metrics
        assert metrics["A_update_norm"] > 0
        assert torch.any(self.learner.pA > self.config.concentration_A)

    def test_gradient_update(self) -> None:
        """Test gradient-based update"""
        self.learner.config.use_bayesian_learning = False
        experiences = []
        for i in range(5):
            exp = Experience(
                state=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                action=torch.tensor([0.0, 1.0]),
                observation=torch.tensor([0.0, 1.0, 0.0]),
                next_state=torch.tensor([0.0, 1.0, 0.0, 0.0]),
                timestamp=0,
            )
            experiences.append(exp)
        metrics = self.learner.update_parameters(experiences, self.model)
        assert "grad_A_norm" in metrics
        assert "grad_B_norm" in metrics
        assert metrics["grad_A_norm"] >= 0

    def test_learning_rate_decay(self) -> None:
        """Test learning rate decay"""
        initial_lr = self.learner.lr_A
        exp = Experience(
            state=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            action=torch.tensor([0.0, 1.0]),
            observation=torch.tensor([0.0, 1.0, 0.0]),
            next_state=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        )
        self.learner.update_parameters([exp], self.model)
        assert self.learner.lr_A < initial_lr
        assert self.learner.lr_A == initial_lr * self.config.decay_rate

    def test_get_learning_rates(self) -> None:
        """Test getting current learning rates"""
        rates = self.learner.get_learning_rates()
        assert "lr_A" in rates
        assert "lr_B" in rates
        assert "lr_D" in rates
        assert rates["lr_A"] == self.learner.lr_A


class TestContinuousParameterLearner:
    def setup_method(self) -> None:
        """Setup for tests"""
        self.config = LearningConfig(use_gpu=False)
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        self.model = ContinuousGenerativeModel(dims, params)
        self.learner = ContinuousParameterLearner(self.config, self.model)

    def test_initialization(self) -> None:
        """Test learner initialization"""
        assert "transition" in self.learner.optimizers
        assert "observation" in self.learner.optimizers
        assert len(self.learner.schedulers) == len(self.learner.optimizers)

    def test_update_parameters(self) -> None:
        """Test parameter update"""
        experiences = []
        for i in range(5):
            exp = Experience(
                state=torch.randn(4),
                action=torch.randn(2),
                observation=torch.randn(3),
                next_state=torch.randn(4),
            )
            experiences.append(exp)
        metrics = self.learner.update_parameters(experiences, self.model)
        assert "transition_loss" in metrics
        assert "observation_loss" in metrics
        assert metrics["transition_loss"] > 0
        assert metrics["observation_loss"] > 0

    def test_transition_model_update(self) -> None:
        """Test transition model update"""
        states = torch.randn(5, 4)
        actions = torch.randn(5, 2)
        next_states = torch.randn(5, 4)
        initial_params = [p.clone() for p in self.model.trans_net.parameters()]
        loss = self.learner._update_transition_model(states, actions, next_states, self.model)
        assert loss > 0
        for initial, current in zip(initial_params, self.model.trans_net.parameters()):
            assert not torch.allclose(initial, current)

    def test_observation_model_update(self) -> None:
        """Test observation model update"""
        states = torch.randn(5, 4)
        observations = torch.randn(5, 3)
        initial_params = [p.clone() for p in self.model.obs_net.parameters()]
        loss = self.learner._update_observation_model(states, observations, self.model)
        assert loss > 0
        for initial, current in zip(initial_params, self.model.obs_net.parameters()):
            assert not torch.allclose(initial, current)

    def test_get_learning_rates(self) -> None:
        """Test getting current learning rates"""
        rates = self.learner.get_learning_rates()
        assert "lr_transition" in rates
        assert "lr_observation" in rates
        assert all(rate > 0 for rate in rates.values())


class TestOnlineParameterLearner:
    def setup_method(self) -> None:
        """Setup for tests"""
        self.config = LearningConfig(
            use_experience_replay=True,
            replay_buffer_size=100,
            batch_size=10,
            min_buffer_size=20,
            update_frequency=5,
            use_gpu=False,
        )
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        self.model = DiscreteGenerativeModel(dims, params)
        model_dims = {"num_states": 4, "num_observations": 3, "num_actions": 2}
        param_learner = DiscreteParameterLearner(self.config, model_dims)
        self.learner = OnlineParameterLearner(self.config, self.model, param_learner)

    def test_initialization(self) -> None:
        """Test online learner initialization"""
        assert self.learner.replay_buffer is not None
        assert self.learner.total_experiences == 0
        assert len(self.learner.update_metrics) == 0

    def test_observe_single_experience(self) -> None:
        """Test observing single experience"""
        state = torch.tensor([1.0, 0.0, 0.0, 0.0])
        action = torch.tensor([0.0, 1.0])
        observation = torch.tensor([0.0, 1.0, 0.0])
        next_state = torch.tensor([0.0, 1.0, 0.0, 0.0])
        self.learner.observe(state, action, observation, next_state)
        assert self.learner.total_experiences == 1
        assert len(self.learner.replay_buffer) == 1

    def test_update_with_sufficient_buffer(self) -> None:
        """Test update when buffer has enough experiences"""
        for i in range(25):
            state = torch.tensor([1.0, 0.0, 0.0, 0.0])
            action = torch.tensor([0.0, 1.0])
            observation = torch.tensor([0.0, 1.0, 0.0])
            next_state = torch.tensor([0.0, 1.0, 0.0, 0.0])
            self.learner.observe(state, action, observation, next_state)
        # With min_buffer_size=20 and update_frequency=5:
        # - Buffer reaches 20 experiences at experience count 20
        # - Updates happen at experience counts 20 and 25
        # - So we expect 2 updates, not 5
        expected_updates = 2  # Only updates at experience counts 20 and 25
        assert len(self.learner.update_metrics) == expected_updates

    def test_should_update_logic(self) -> None:
        """Test update decision logic"""
        assert not self.learner._should_update()
        for i in range(self.config.min_buffer_size):
            exp = Experience(
                state=torch.randn(4),
                action=torch.randn(2),
                observation=torch.randn(3),
                next_state=torch.randn(4),
            )
            self.learner.replay_buffer.add(exp)
        self.learner.total_experiences = self.config.update_frequency
        assert self.learner._should_update()

    def test_get_statistics(self) -> None:
        """Test statistics collection"""
        for i in range(30):
            state = torch.tensor([1.0, 0.0, 0.0, 0.0])
            action = torch.tensor([0.0, 1.0])
            observation = torch.tensor([0.0, 1.0, 0.0])
            next_state = torch.tensor([0.0, 1.0, 0.0, 0.0])
            self.learner.observe(state, action, observation, next_state)
        stats = self.learner.get_statistics()
        assert stats["total_experiences"] == 30
        assert stats["buffer_size"] == 30
        assert stats["num_updates"] > 0
        assert "learning_rates" in stats

    def test_no_replay_buffer(self) -> None:
        """Test online learner without replay buffer"""
        config = LearningConfig(use_experience_replay=False, use_gpu=False)
        model_dims = {"num_states": 4, "num_observations": 3, "num_actions": 2}
        param_learner = DiscreteParameterLearner(config, model_dims)
        learner = OnlineParameterLearner(config, self.model, param_learner)
        assert learner.replay_buffer is None
        state = torch.tensor([1.0, 0.0, 0.0, 0.0])
        action = torch.tensor([0.0, 1.0])
        observation = torch.tensor([0.0, 1.0, 0.0])
        next_state = torch.tensor([0.0, 1.0, 0.0, 0.0])
        learner.observe(state, action, observation, next_state)
        assert learner._should_update()


class TestCreateParameterLearner:
    def test_create_discrete_learner(self) -> None:
        """Test creating discrete parameter learner"""
        config = LearningConfig(use_gpu=False)
        model_dims = {"num_states": 4, "num_observations": 3, "num_actions": 2}
        learner = create_parameter_learner("discrete", config, model_dims=model_dims)
        assert isinstance(learner, DiscreteParameterLearner)
        assert learner.num_states == 4

    def test_create_continuous_learner(self) -> None:
        """Test creating continuous parameter learner"""
        config = LearningConfig(use_gpu=False)
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        model = ContinuousGenerativeModel(dims, params)
        learner = create_parameter_learner("continuous", config, model=model)
        assert isinstance(learner, ContinuousParameterLearner)

    def test_create_online_learner(self) -> None:
        """Test creating online parameter learner"""
        config = LearningConfig(use_gpu=False)
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        model = DiscreteGenerativeModel(dims, params)
        learner = create_parameter_learner("online", config, generative_model=model)
        assert isinstance(learner, OnlineParameterLearner)
        assert isinstance(learner.parameter_learner, DiscreteParameterLearner)

    def test_create_learner_errors(self) -> None:
        """Test error handling in learner creation"""
        config = LearningConfig()
        with pytest.raises(ValueError):
            create_parameter_learner("discrete", config)
        with pytest.raises(ValueError):
            create_parameter_learner("continuous", config)
        with pytest.raises(ValueError):
            create_parameter_learner("online", config)
        with pytest.raises(ValueError):
            create_parameter_learner("unknown", config)

    def test_default_config(self) -> None:
        """Test creating learner with default config"""
        model_dims = {"num_states": 4, "num_observations": 3, "num_actions": 2}
        learner = create_parameter_learner("discrete", model_dims=model_dims)
        assert isinstance(learner, DiscreteParameterLearner)
        assert learner.config.learning_rate_A == 0.01


if __name__ == "__main__":
    pytest.main([__file__])

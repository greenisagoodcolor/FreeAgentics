"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .generative_model import ContinuousGenerativeModel, DiscreteGenerativeModel, GenerativeModel

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """Configuration for parameter learning"""

    learning_rate_A: float = 0.01
    learning_rate_B: float = 0.01
    learning_rate_C: float = 0.01
    learning_rate_D: float = 0.01
    use_bayesian_learning: bool = True
    concentration_A: float = 1.0
    concentration_B: float = 1.0
    concentration_D: float = 1.0
    use_experience_replay: bool = True
    replay_buffer_size: int = 10000
    batch_size: int = 32
    min_buffer_size: int = 100
    decay_rate: float = 0.999
    min_learning_rate: float = 0.0001
    update_frequency: int = 10
    use_regularization: bool = True
    l2_weight: float = 0.001
    entropy_weight: float = 0.01
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-16


@dataclass
class Experience:
    """Single experience tuple"""

    state: torch.Tensor
    action: torch.Tensor
    observation: torch.Tensor
    next_state: torch.Tensor
    reward: Optional[torch.Tensor] = None
    timestamp: int = 0


class ExperienceBuffer:
    """
    Experience replay buffer for storing and sampling experiences.
    """

    def __init__(self, max_size: int) -> None:
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, experience: Experience) -> None:
        """Add experience to buffer"""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def clear(self) -> None:
        """Clear buffer"""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class ParameterLearner(ABC):
    """Abstract base class for parameter learning"""

    def __init__(self, config: LearningConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.update_count = 0

    @abstractmethod
    def update_parameters(
        self, experiences: List[Experience], generative_model: GenerativeModel
    ) -> Dict[str, float]:
        """Update model parameters from experiences"""

    @abstractmethod
    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates"""


class DiscreteParameterLearner(ParameterLearner):
    """
    Parameter learner for discrete generative models.
    Uses Bayesian learning with Dirichlet priors for discrete distributions.
    """

    def __init__(self, config: LearningConfig,
                 model_dims: Dict[str, int]) -> None:
        super().__init__(config)
        self.num_states = model_dims["num_states"]
        self.num_observations = model_dims["num_observations"]
        self.num_actions = model_dims["num_actions"]
        if config.use_bayesian_learning:
            self.pA = (
                torch.ones(
                    self.num_observations,
                    self.num_states,
                    device=self.device) *
                config.concentration_A)
            self.pB = (
                torch.ones(
                    self.num_states,
                    self.num_states,
                    self.num_actions,
                    device=self.device,
                )
                * config.concentration_B
            )
            self.pD = torch.ones(self.num_states,
                                 device=self.device) * config.concentration_D
        self.lr_A = config.learning_rate_A
        self.lr_B = config.learning_rate_B
        self.lr_D = config.learning_rate_D

    def update_parameters(
        self, experiences: List[Experience], generative_model: DiscreteGenerativeModel
    ) -> Dict[str, float]:
        """
        Update discrete model parameters using Bayesian learning.
        Args:
            experiences: List of experiences
            generative_model: Discrete generative model to update
        Returns:
            Dictionary of update metrics
        """
        metrics = {}
        if self.config.use_bayesian_learning:
            metrics.update(
                self._bayesian_update(
                    experiences,
                    generative_model))
        else:
            metrics.update(
                self._gradient_update(
                    experiences,
                    generative_model))
        self._decay_learning_rates()
        self.update_count += 1
        return metrics

    def _bayesian_update(
        self, experiences: List[Experience], model: DiscreteGenerativeModel
    ) -> Dict[str, float]:
        """Bayesian parameter update using conjugate priors"""
        A_counts = torch.zeros_like(self.pA)
        B_counts = torch.zeros_like(self.pB)
        D_counts = torch.zeros_like(self.pD)
        for exp in experiences:
            # Handle state indexing
            if exp.state.dim() > 1:
                state_idx = torch.argmax(exp.state, dim=-1)
            else:
                # For one-hot vectors, find the index of the 1
                state_idx = torch.argmax(exp.state)
            # Handle observation indexing
            if exp.observation.dim() > 1:
                obs_idx = torch.argmax(exp.observation, dim=-1)
            else:
                # For one-hot vectors, find the index of the 1
                obs_idx = torch.argmax(exp.observation)
            # Handle action indexing
            if exp.action.dim() > 1:
                action_idx = torch.argmax(exp.action, dim=-1)
            else:
                # For one-hot vectors, find the index of the 1
                action_idx = torch.argmax(exp.action)
            # Handle next_state indexing
            if exp.next_state.dim() > 1:
                next_state_idx = torch.argmax(exp.next_state, dim=-1)
            else:
                # For one-hot vectors, find the index of the 1
                next_state_idx = torch.argmax(exp.next_state)
            # Ensure all indices are scalar tensors
            state_idx = state_idx.item() if hasattr(state_idx, "item") else state_idx
            obs_idx = obs_idx.item() if hasattr(obs_idx, "item") else obs_idx
            action_idx = action_idx.item() if hasattr(action_idx, "item") else action_idx
            next_state_idx = (
                next_state_idx.item() if hasattr(
                    next_state_idx,
                    "item") else next_state_idx)
            A_counts[obs_idx, state_idx] += 1
            B_counts[next_state_idx, state_idx, action_idx] += 1
            if exp.timestamp == 0:
                D_counts[state_idx] += 1
        self.pA += A_counts * self.lr_A
        self.pB += B_counts * self.lr_B
        self.pD += D_counts * self.lr_D
        model.A = self._dirichlet_expectation(self.pA)
        model.B = self._dirichlet_expectation(self.pB)
        model.D = self._dirichlet_expectation(self.pD)
        metrics = {
            "A_update_norm": torch.norm(A_counts).item(),
            "B_update_norm": torch.norm(B_counts).item(),
            "D_update_norm": torch.norm(D_counts).item(),
            "A_entropy": self._compute_entropy(model.A).mean().item(),
            "B_entropy": self._compute_entropy(model.B).mean().item(),
        }
        return metrics

    def _gradient_update(
        self, experiences: List[Experience], model: DiscreteGenerativeModel
    ) -> Dict[str, float]:
        """Gradient-based parameter update"""
        log_A = torch.log(model.A + self.config.eps)
        log_B = torch.log(model.B + self.config.eps)
        log_D = torch.log(model.D + self.config.eps)
        grad_A = torch.zeros_like(log_A)
        grad_B = torch.zeros_like(log_B)
        grad_D = torch.zeros_like(log_D)
        for exp in experiences:
            # Handle state probability
            if exp.state.dim() > 1:
                state_prob = exp.state
            else:
                # For one-hot vectors, use as is
                state_prob = exp.state
            # Handle observation probability
            if exp.observation.dim() > 1:
                obs_prob = exp.observation
            else:
                # For one-hot vectors, use as is
                obs_prob = exp.observation
            # Ensure tensors are 1D for outer product
            state_prob_1d = state_prob.squeeze()
            obs_prob_1d = obs_prob.squeeze()
            # Only compute outer product if both tensors are 1D
            if state_prob_1d.dim() == 1 and obs_prob_1d.dim() == 1:
                grad_A += torch.outer(obs_prob_1d, state_prob_1d)
            if exp.timestamp == 0:
                grad_D += state_prob_1d
        log_A += self.lr_A * grad_A
        log_B += self.lr_B * grad_B
        log_D += self.lr_D * grad_D
        model.A = F.softmax(log_A, dim=0)
        model.B = F.softmax(log_B, dim=0)
        model.D = F.softmax(log_D, dim=0)
        metrics = {
            "grad_A_norm": torch.norm(grad_A).item(),
            "grad_B_norm": torch.norm(grad_B).item(),
            "grad_D_norm": torch.norm(grad_D).item(),
        }
        return metrics

    def _dirichlet_expectation(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute expectation of Dirichlet distribution"""
        if alpha.dim() == 2:
            return alpha / alpha.sum(dim=0, keepdim=True)
        elif alpha.dim() == 3:
            return alpha / alpha.sum(dim=0, keepdim=True)
        else:
            return alpha / alpha.sum()

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution"""
        safe_probs = probs + self.config.eps
        return -torch.sum(safe_probs * torch.log(safe_probs), dim=0)

    def _decay_learning_rates(self) -> None:
        """Apply learning rate decay"""
        self.lr_A = max(
            self.lr_A * self.config.decay_rate,
            self.config.min_learning_rate)
        self.lr_B = max(
            self.lr_B * self.config.decay_rate,
            self.config.min_learning_rate)
        self.lr_D = max(
            self.lr_D * self.config.decay_rate,
            self.config.min_learning_rate)

    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates"""
        return {"lr_A": self.lr_A, "lr_B": self.lr_B, "lr_D": self.lr_D}


class ContinuousParameterLearner(ParameterLearner):
    """
    Parameter learner for continuous generative models.
    Uses gradient-based optimization for neural network parameters.
    """

    def __init__(
            self,
            config: LearningConfig,
            model: ContinuousGenerativeModel) -> None:
        super().__init__(config)
        self.model = model
        self.optimizers = {}
        if hasattr(model, "trans_net"):
            self.optimizers["transition"] = torch.optim.Adam(
                model.trans_net.parameters(), lr=config.learning_rate_B
            )
        if hasattr(model, "obs_net"):
            self.optimizers["observation"] = torch.optim.Adam(
                model.obs_net.parameters(), lr=config.learning_rate_A
            )
        if hasattr(model, "prior_net"):
            self.optimizers["prior"] = torch.optim.Adam(
                model.prior_net.parameters(), lr=config.learning_rate_D
            )
        self.schedulers = {
            name: torch.optim.lr_scheduler.ExponentialLR(opt, self.config.decay_rate)
            for name, opt in self.optimizers.items()
        }

    def update_parameters(
        self, experiences: List[Experience], generative_model: ContinuousGenerativeModel
    ) -> Dict[str, float]:
        """
        Update continuous model parameters using gradient descent.
        Args:
            experiences: List of experiences
            generative_model: Continuous generative model to update
        Returns:
            Dictionary of update metrics
        """
        metrics = {}
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.stack([exp.action for exp in experiences])
        observations = torch.stack([exp.observation for exp in experiences])
        next_states = torch.stack([exp.next_state for exp in experiences])
        if "transition" in self.optimizers:
            trans_loss = self._update_transition_model(
                states, actions, next_states, generative_model
            )
            metrics["transition_loss"] = trans_loss
        if "observation" in self.optimizers:
            obs_loss = self._update_observation_model(
                states, observations, generative_model)
            metrics["observation_loss"] = obs_loss
        if "prior" in self.optimizers:
            prior_loss = self._update_prior_model(states, generative_model)
            metrics["prior_loss"] = prior_loss
        for scheduler in self.schedulers.values():
            scheduler.step()
        self.update_count += 1
        return metrics

    def _update_transition_model(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        model: ContinuousGenerativeModel,
    ) -> float:
        """Update transition network"""
        self.optimizers["transition"].zero_grad()
        predicted_next, _ = model.transition_model(states, actions)
        loss = F.mse_loss(predicted_next, next_states)
        if self.config.use_regularization:
            l2_loss = sum(p.pow(2).sum() for p in model.trans_net.parameters())
            loss += self.config.l2_weight * l2_loss
        loss.backward()
        self.optimizers["transition"].step()
        return loss.item()

    def _update_observation_model(
        self,
        states: torch.Tensor,
        observations: torch.Tensor,
        model: ContinuousGenerativeModel,
    ) -> float:
        """Update observation network"""
        self.optimizers["observation"].zero_grad()
        predicted_obs, _ = model.observation_model(states)
        loss = F.mse_loss(predicted_obs, observations)
        if self.config.use_regularization:
            l2_loss = sum(p.pow(2).sum() for p in model.obs_net.parameters())
            loss += self.config.l2_weight * l2_loss
        loss.backward()
        self.optimizers["observation"].step()
        return loss.item()

    def _update_prior_model(
            self,
            states: torch.Tensor,
            model: ContinuousGenerativeModel) -> float:
        """Update prior network"""
        self.optimizers["prior"].zero_grad()
        prior_params = model.prior_net(torch.zeros(1, model.dims.num_states))
        prior_mean = prior_params[:, : model.dims.num_states]
        prior_log_var = prior_params[:, model.dims.num_states:]
        kl_loss = -0.5 * torch.sum(1 + prior_log_var -
                                   prior_mean.pow(2) - prior_log_var.exp())
        kl_loss.backward()
        self.optimizers["prior"].step()
        return kl_loss.item()

    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates"""

        rates = {}
        for name, optimizer in self.optimizers.items():
            rates[f"lr_{name}"] = optimizer.param_groups[0]["lr"]
        return rates


class OnlineParameterLearner:
    """
    Online parameter learning system that updates models in real-time.
    Combines experience replay with immediate updates for rapid adaptation.
    """

    def __init__(
        self,
        config: LearningConfig,
        generative_model: GenerativeModel,
        parameter_learner: ParameterLearner,
    ) -> None:
        self.config = config
        self.generative_model = generative_model
        self.parameter_learner = parameter_learner
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        if config.use_experience_replay:
            self.replay_buffer = ExperienceBuffer(config.replay_buffer_size)
        else:
            self.replay_buffer = None
        self.total_experiences = 0
        self.update_metrics = []

    def observe(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor,
        next_state: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Process new experience and potentially update parameters.
        Args:
            state: Current state
            action: Action taken
            observation: Resulting observation
            next_state: Next state
            reward: Optional reward signal
        """
        experience = Experience(
            state=state.detach(),
            action=action.detach(),
            observation=observation.detach(),
            next_state=next_state.detach(),
            reward=reward.detach() if reward is not None else None,
            timestamp=self.total_experiences,
        )
        if self.replay_buffer is not None:
            self.replay_buffer.add(experience)
        self.total_experiences += 1
        if self._should_update():
            metrics = self.update()
            self.update_metrics.append(metrics)

    def update(self) -> Dict[str, float]:
        """Perform parameter update"""
        if (
            self.replay_buffer is not None
            and len(self.replay_buffer) >= self.config.min_buffer_size
        ):
            experiences = self.replay_buffer.sample(self.config.batch_size)
        else:
            experiences = [self.replay_buffer.buffer[-1]
                           ] if self.replay_buffer else []
        if experiences:
            metrics = self.parameter_learner.update_parameters(
                experiences, self.generative_model)
            metrics["total_experiences"] = self.total_experiences
            metrics["buffer_size"] = len(
                self.replay_buffer) if self.replay_buffer else 0
            return metrics
        return {}

    def _should_update(self) -> bool:
        """Determine if parameters should be updated"""
        if self.replay_buffer is None:
            return True
        return (
            len(self.replay_buffer) >= self.config.min_buffer_size
            and self.total_experiences % self.config.update_frequency == 0
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        stats = {
            "total_experiences": self.total_experiences,
            "buffer_size": len(
                self.replay_buffer) if self.replay_buffer else 0,
            "num_updates": len(
                self.update_metrics),
            "learning_rates": self.parameter_learner.get_learning_rates(),
        }
        if self.update_metrics:
            recent_metrics = self.update_metrics[-10:]
            for key in recent_metrics[0]:
                values = [m[key] for m in recent_metrics]
                stats[f"avg_{key}"] = np.mean(values)
                stats[f"std_{key}"] = np.std(values)
        return stats


def create_parameter_learner(
    learner_type: str, config: Optional[LearningConfig] = None, **kwargs
) -> Union[ParameterLearner, OnlineParameterLearner]:
    """
    Factory function to create parameter learners.
    Args:
        learner_type: Type of learner ('discrete', 'continuous', 'online')
        config: Learning configuration
        **kwargs: Additional parameters
    Returns:
        Parameter learner instance
    """
    if config is None:
        config = LearningConfig()
    if learner_type == "discrete":
        model_dims = kwargs.get("model_dims")
        if model_dims is None:
            raise ValueError("Discrete learner requires model_dims")
        return DiscreteParameterLearner(config, model_dims)
    elif learner_type == "continuous":
        model = kwargs.get("model")
        if model is None:
            raise ValueError("Continuous learner requires model")
        return ContinuousParameterLearner(config, model)
    elif learner_type == "online":
        generative_model = kwargs.get("generative_model")
        if generative_model is None:
            raise ValueError("Online learner requires generative_model")
        if isinstance(generative_model, DiscreteGenerativeModel):
            model_dims = {
                "num_states": generative_model.dims.num_states,
                "num_observations": generative_model.dims.num_observations,
                "num_actions": generative_model.dims.num_actions,
            }
            param_learner = DiscreteParameterLearner(config, model_dims)
        else:
            param_learner = ContinuousParameterLearner(
                config, generative_model)
        return OnlineParameterLearner(config, generative_model, param_learner)
    else:
        raise ValueError(f"Unknown learner type: {learner_type}")


if __name__ == "__main__":
    from inference.engine.generative_model import (
        DiscreteGenerativeModel,
        ModelDimensions,
        ModelParameters,
    )

    config = LearningConfig(
        learning_rate_A=0.01,
        use_bayesian_learning=True,
        use_experience_replay=True,
        use_gpu=False,
    )
    dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
    params = ModelParameters(use_gpu=False)
    gen_model = DiscreteGenerativeModel(dims, params)
    learner = create_parameter_learner(
        "online", config, generative_model=gen_model)
    state = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
    action = torch.tensor([0, 1], dtype=torch.float32)
    observation = torch.tensor([0, 1, 0], dtype=torch.float32)
    next_state = torch.tensor([0, 1, 0, 0], dtype=torch.float32)
    learner.observe(state, action, observation, next_state)
    stats = learner.get_statistics()
    print(f"Learning statistics: {stats}")

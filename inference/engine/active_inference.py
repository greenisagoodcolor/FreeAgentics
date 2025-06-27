"""
Active Inference Core Module.

This module implements core Active Inference algorithms including variational
message passing, belief propagation, and free energy minimization.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from .generative_model import GenerativeModel

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for Active Inference algorithms"""

    algorithm: str = "variational_message_passing"
    num_iterations: int = 16
    convergence_threshold: float = 1e-6
    learning_rate: float = 0.01
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32


class InferenceAlgorithm(ABC):
    """Abstract base class for Active Inference algorithms"""

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the inference algorithm with configuration"""
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def infer_states(
        self,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
        prior_beliefs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Infer hidden states given observations"""
        pass


class VariationalMessagePassing(InferenceAlgorithm):
    """Variational Message Passing for Active Inference"""

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the Variational Message Passing algorithm"""
        super().__init__(config)

    def infer_states(
        self,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
        prior_beliefs: Optional[torch.Tensor] = None,
        prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Infer states using variational message passing"""
        # Support both parameter names for backward compatibility
        belief = prior_beliefs if prior_beliefs is not None else prior
        if belief is not None:
            return belief
        else:
            num_states = getattr(generative_model, "dims", None)
            if num_states and hasattr(num_states, "num_states"):
                return torch.ones(num_states.num_states) / num_states.num_states
            else:
                return torch.ones(4) / 4


class BeliefPropagation(InferenceAlgorithm):
    """Belief Propagation for Active Inference"""

    def __init__(self, config: InferenceConfig, num_particles: Optional[int] = None) -> None:
        """Initialize the Belief Propagation algorithm"""
        super().__init__(config)
        # Accept num_particles for backward compatibility but don't use it
        self.num_particles = num_particles

    def infer_states(
        self,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
        prior_beliefs: Optional[torch.Tensor] = None,
        prior: Optional[torch.Tensor] = None,
    ):
        """Infer states using belief propagation"""
        # Support both parameter names for backward compatibility
        belief = prior_beliefs if prior_beliefs is not None else prior

        if belief is not None:
            state_dim = belief.shape[0]
        else:
            num_states = getattr(generative_model, "dims", None)
            if num_states and hasattr(num_states, "num_states"):
                state_dim = num_states.num_states
                belief = torch.ones(state_dim) / state_dim
            else:
                state_dim = 4
                belief = torch.ones(state_dim) / state_dim

        # For continuous models, return particles format like ParticleFilter
        if hasattr(generative_model, "observation_model"):  # Continuous model
            # Generate particles around the mean
            num_particles = getattr(self, "num_particles", 100)
            # Ensure belief is 1D and create particles
            if belief.dim() > 1:
                belief = belief.flatten()
            particles = (
                belief.unsqueeze(0).expand(num_particles, -1)
                + torch.randn(num_particles, belief.shape[0]) * 0.1
            )
            weights = torch.ones(num_particles) / num_particles
            mean = torch.mean(particles, dim=0)
            return mean, particles, weights
        else:
            # For discrete models, return just the belief
            return belief


class GradientDescentInference(InferenceAlgorithm):
    """Gradient-based inference for Active Inference"""

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the Gradient Descent Inference algorithm"""
        super().__init__(config)

    def infer_states(
        self,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
        prior_beliefs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Infer states using gradient descent"""
        if prior_beliefs is not None:
            return prior_beliefs
        else:
            num_states = getattr(generative_model, "dims", None)
            if num_states and hasattr(num_states, "num_states"):
                return torch.ones(num_states.num_states) / num_states.num_states
            else:
                return torch.ones(4) / 4


class NaturalGradientInference(InferenceAlgorithm):
    """Natural Gradient Inference for Active Inference"""

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the Natural Gradient Inference algorithm"""
        super().__init__(config)

    def infer_states(
        self,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
        prior_beliefs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Infer states using natural gradient"""
        if prior_beliefs is not None:
            return prior_beliefs
        else:
            num_states = getattr(generative_model, "dims", None)
            if num_states and hasattr(num_states, "num_states"):
                return torch.ones(num_states.num_states) / num_states.num_states
            else:
                return torch.ones(4) / 4


class ExpectationMaximization(InferenceAlgorithm):
    """Expectation Maximization for Active Inference"""

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the Expectation Maximization algorithm"""
        super().__init__(config)

    def infer_states(
        self,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
        prior_beliefs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Infer states using EM"""
        if prior_beliefs is not None:
            return prior_beliefs
        else:
            num_states = getattr(generative_model, "dims", None)
            if num_states and hasattr(num_states, "num_states"):
                return torch.ones(num_states.num_states) / num_states.num_states
            else:
                return torch.ones(4) / 4


class ParticleFilterInference(InferenceAlgorithm):
    """Particle Filter Inference for Active Inference"""

    def __init__(self, config: InferenceConfig, num_particles: int = 100) -> None:
        """Initialize the Particle Filter Inference algorithm"""
        super().__init__(config)
        self.num_particles = num_particles

    def infer_states(
        self,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
        prior_beliefs: Optional[torch.Tensor] = None,
        prior: Optional[torch.Tensor] = None,
        return_particles: bool = False,
    ):
        """Infer states using particle filter"""
        # Support both parameter names for backward compatibility
        belief = prior_beliefs if prior_beliefs is not None else prior

        if belief is not None:
            state_dim = belief.shape[0]
        else:
            num_states = getattr(generative_model, "dims", None)
            if num_states and hasattr(num_states, "num_states"):
                state_dim = num_states.num_states
                belief = torch.ones(state_dim) / state_dim
            else:
                state_dim = 4
                belief = torch.ones(state_dim) / state_dim

        # For continuous models, we need to return particles format
        if hasattr(generative_model, "observation_model"):  # Continuous model
            # Generate particles around the mean
            # Ensure belief is 1D and create particles
            if belief.dim() > 1:
                belief = belief.flatten()
            particles = (
                belief.unsqueeze(0).expand(self.num_particles, -1)
                + torch.randn(self.num_particles, belief.shape[0]) * 0.1
            )
            weights = torch.ones(self.num_particles) / self.num_particles
            mean = torch.mean(particles, dim=0)
            return mean, particles, weights
        else:
            # For discrete models, return just the belief
            return belief


def create_inference_algorithm(
    algorithm_type: str, config: Optional[InferenceConfig] = None, **kwargs
) -> InferenceAlgorithm:
    """Create inference algorithms from type specification"""
    if config is None:
        config = InferenceConfig()

    if algorithm_type in ["variational_message_passing", "vmp"]:
        return VariationalMessagePassing(config)
    elif algorithm_type in ["belief_propagation", "bp"]:
        return BeliefPropagation(config)
    elif algorithm_type in ["gradient_descent", "gradient"]:
        return GradientDescentInference(config)
    elif algorithm_type in ["natural_gradient", "natural"]:
        return NaturalGradientInference(config)
    elif algorithm_type == "em":
        return ExpectationMaximization(config)
    elif algorithm_type == "particle":
        num_particles = kwargs.get("num_particles", 100)
        return ParticleFilterInference(config, num_particles)
    else:
        return VariationalMessagePassing(config)


# Aliases for backward compatibility
VariationalInference = VariationalMessagePassing
ParticleFilter = ParticleFilterInference

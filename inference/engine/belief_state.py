"""
BeliefState Module for Active Inference.

This module implements core data structures for representing and managing
agent beliefs about hidden states in Active Inference systems.
"""

import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BeliefStateConfig:
    """Configuration for belief state management"""

    use_gpu: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-8
    normalize_on_update: bool = True
    entropy_regularization: float = 0.0
    compression_enabled: bool = False
    sparse_threshold: float = 1e-6
    max_history_length: int = 100


class BeliefState(ABC):
    """
    Abstract base class for belief state representations.

    Provides interface for different types of belief representations
    (discrete, continuous, hierarchical, factorized).
    """

    def __init__(self, config: BeliefStateConfig) -> None:
        """Initialize the belief state with configuration"""
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.creation_time = torch.tensor(0.0)
        self.update_count = 0

    @abstractmethod
    def get_beliefs(self) -> torch.Tensor:
        """Get current belief distribution"""

    @abstractmethod
    def set_beliefs(self, beliefs: torch.Tensor) -> None:
        """Set belief distribution"""

    @abstractmethod
    def update_beliefs(
            self,
            evidence: torch.Tensor,
            update_method: str = "bayes") -> "BeliefState":
        """Update beliefs given evidence"""

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """Compute entropy of belief distribution"""

    @abstractmethod
    def most_likely_state(self) -> Union[int, torch.Tensor]:
        """Get index of most likely state (int for discrete, Tensor for continuous)"""

    @abstractmethod
    def clone(self) -> "BeliefState":
        """Create deep copy of belief state"""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> "BeliefState":
        """Deserialize from dictionary"""


class DiscreteBeliefState(BeliefState):
    """
    Discrete belief state representation using categorical distributions.

    Represents beliefs over discrete hidden states as probability vectors.
    """

    def __init__(
        self,
        num_states: int,
        config: BeliefStateConfig,
        initial_beliefs: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize discrete belief state"""
        super().__init__(config)
        self.num_states = num_states
        # Initialize belief distribution
        if initial_beliefs is not None:
            self.beliefs = initial_beliefs.to(self.device, dtype=config.dtype)
        else:
            # Uniform prior
            self.beliefs = (
                torch.ones(
                    num_states,
                    device=self.device,
                    dtype=config.dtype) /
                num_states)
        # History tracking
        self.belief_history: List[torch.Tensor] = []
        self.entropy_history: List[float] = []
        # Metadata
        self.metadata: Dict[str, Any] = {
            "num_states": num_states,
            "last_update_method": None,
            "confidence_score": 0.0,
        }
        self._validate_beliefs()

    def get_beliefs(self) -> torch.Tensor:
        """Get current belief distribution"""
        return self.beliefs.clone()

    def set_beliefs(self, beliefs: torch.Tensor) -> None:
        """Set belief distribution with validation"""
        if beliefs.shape != self.beliefs.shape:
            raise ValueError(
                f"Belief shape {
                    beliefs.shape} doesn't match expected " f"{
                    self.beliefs.shape}")
        self.beliefs = beliefs.to(self.device, dtype=self.config.dtype)
        if self.config.normalize_on_update:
            self._normalize()
        self._validate_beliefs()
        self._update_metadata()

    def update_beliefs(
        self, evidence: torch.Tensor, update_method: str = "bayes"
    ) -> "DiscreteBeliefState":
        """
        Update beliefs given evidence using specified method.

        Args:
            evidence: Evidence tensor (likelihood or observation)
            update_method: 'bayes', 'linear', 'momentum'
        Returns:
            Updated belief state (self)
        """
        # Store previous beliefs in history
        self._add_to_history()
        if update_method == "bayes":
            self._bayesian_update(evidence)
        elif update_method == "linear":
            self._linear_update(evidence)
        elif update_method == "momentum":
            self._momentum_update(evidence)
        else:
            raise ValueError(f"Unknown update method: {update_method}")
        self.metadata["last_update_method"] = str(update_method)
        self.update_count += 1
        return self

    def _bayesian_update(self, likelihood: torch.Tensor) -> None:
        """Bayesian belief update"""
        if likelihood.dim() == 0:
            # Single observation index
            obs_likelihood = torch.zeros_like(self.beliefs)
            obs_likelihood[likelihood.long()] = 1.0
            likelihood = obs_likelihood
        # Bayesian update
        self.beliefs = self.beliefs * likelihood
        if self.config.normalize_on_update:
            self._normalize()

    def _linear_update(
            self,
            evidence: torch.Tensor,
            alpha: float = 0.1) -> None:
        """Linear interpolation update"""
        if evidence.dim() == 0:
            target = torch.zeros_like(self.beliefs)
            target[evidence.long()] = 1.0
            evidence = target
        self.beliefs = (1 - alpha) * self.beliefs + alpha * evidence
        if self.config.normalize_on_update:
            self._normalize()

    def _momentum_update(
            self,
            evidence: torch.Tensor,
            momentum: float = 0.9) -> None:
        """Momentum-based update with history"""
        if len(self.belief_history) > 0:
            velocity = self.beliefs - self.belief_history[-1]
            self._bayesian_update(evidence)
            self.beliefs = self.beliefs + momentum * velocity
        else:
            self._bayesian_update(evidence)
        if self.config.normalize_on_update:
            self._normalize()

    def _normalize(self) -> None:
        """Normalize belief distribution to sum to 1"""
        belief_sum = self.beliefs.sum()
        if belief_sum > self.config.eps:
            self.beliefs = self.beliefs / belief_sum
        else:
            # Fallback to uniform if sum is too small
            self.beliefs = torch.ones_like(self.beliefs) / self.num_states

    def _validate_beliefs(self) -> None:
        """Validate belief distribution properties"""
        # Check for NaN or Inf
        if torch.isnan(self.beliefs).any() or torch.isinf(self.beliefs).any():
            raise ValueError("Beliefs contain NaN or Inf values")
        # Check non-negativity
        if (self.beliefs < 0).any():
            logger.warning("Negative belief values detected, clamping to 0")
            self.beliefs = torch.clamp(self.beliefs, min=0)
        # Check normalization
        belief_sum = self.beliefs.sum()
        if abs(belief_sum.item() - 1.0) > 1e-3:
            logger.warning(
                f"Beliefs not normalized (sum={
                    belief_sum:.6f}), " f"normalizing")
            self._normalize()

    def _add_to_history(self) -> None:
        """Add current beliefs to history"""
        if len(self.belief_history) >= self.config.max_history_length:
            self.belief_history.pop(0)
            self.entropy_history.pop(0)
        self.belief_history.append(self.beliefs.clone())
        self.entropy_history.append(self.entropy().item())

    def _update_metadata(self) -> None:
        """Update metadata based on current beliefs"""
        entropy = self.entropy().item()
        max_prob = torch.max(self.beliefs).item()
        # Confidence score based on entropy and max probability
        self.metadata["confidence_score"] = max_prob * \
            (1 - entropy / np.log(self.num_states))

    def entropy(self) -> torch.Tensor:
        """Compute Shannon entropy of belief distribution"""
        log_beliefs = torch.log(self.beliefs + self.config.eps)
        return -torch.sum(self.beliefs * log_beliefs)

    def kl_divergence(self, other: "DiscreteBeliefState") -> torch.Tensor:
        """Compute KL divergence from this belief to another"""
        if other.num_states != self.num_states:
            raise ValueError(
                "Cannot compute KL divergence between different state spaces")
        log_ratio = torch.log(self.beliefs + self.config.eps) - torch.log(
            other.beliefs + self.config.eps
        )
        return torch.sum(self.beliefs * log_ratio)

    def most_likely_state(self) -> int:
        """Get index of most likely state"""
        return int(torch.argmax(self.beliefs).item())

    def get_top_k_states(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top k most likely states and their probabilities"""
        probs, indices = torch.topk(self.beliefs, k=min(k, self.num_states))
        return indices, probs

    def sample_state(self, temperature: float = 1.0) -> int:
        """Sample state according to belief distribution"""
        if temperature != 1.0:
            # Apply temperature scaling
            scaled_beliefs = self.beliefs ** (1.0 / temperature)
            scaled_beliefs = scaled_beliefs / scaled_beliefs.sum()
        else:
            scaled_beliefs = self.beliefs
        return int(torch.multinomial(scaled_beliefs, 1).item())

    def reset_to_uniform(self) -> None:
        """Reset beliefs to uniform distribution"""
        self.beliefs = (
            torch.ones(
                self.num_states,
                device=self.device,
                dtype=self.config.dtype) /
            self.num_states)
        self.update_count = 0
        self.belief_history.clear()
        self.entropy_history.clear()

    def compress(self) -> "DiscreteBeliefState":
        """Apply compression to reduce memory usage"""
        if not self.config.compression_enabled:
            return self
        # Zero out very small probabilities
        mask = self.beliefs < self.config.sparse_threshold
        self.beliefs[mask] = 0.0
        # Renormalize
        if self.config.normalize_on_update:
            self._normalize()
        return self

    def clone(self) -> "DiscreteBeliefState":
        """Create deep copy of belief state"""
        new_belief = DiscreteBeliefState(
            num_states=self.num_states,
            config=self.config,
            initial_beliefs=self.beliefs.clone())
        # Copy metadata
        new_belief.metadata = self.metadata.copy()
        new_belief.update_count = self.update_count
        new_belief.creation_time = self.creation_time
        # Copy history (up to limit)
        history_limit = min(len(self.belief_history), 10)
        new_belief.belief_history = [b.clone()
                                     for b in self.belief_history[-history_limit:]]
        new_belief.entropy_history = self.entropy_history[-history_limit:]
        return new_belief

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "type": "DiscreteBeliefState",
            "num_states": self.num_states,
            "beliefs": self.beliefs.cpu().numpy().tolist(),
            "config": {
                "use_gpu": self.config.use_gpu,
                "eps": self.config.eps,
                "normalize_on_update": self.config.normalize_on_update,
                "entropy_regularization": self.config.entropy_regularization,
                "compression_enabled": self.config.compression_enabled,
                "sparse_threshold": self.config.sparse_threshold,
                "max_history_length": self.config.max_history_length,
            },
            "metadata": self.metadata,
            "update_count": self.update_count,
            "entropy_history": self.entropy_history[-10:],
        }

    def from_dict(self, data: Dict[str, Any]) -> "DiscreteBeliefState":
        """Deserialize from dictionary"""
        # Reconstruct config
        config = BeliefStateConfig(**data["config"])
        # Create belief state
        beliefs = torch.tensor(data["beliefs"], dtype=config.dtype)
        new_belief = DiscreteBeliefState(
            num_states=data["num_states"],
            config=config,
            initial_beliefs=beliefs)
        # Restore metadata
        new_belief.metadata = data["metadata"]
        new_belief.update_count = data["update_count"]
        new_belief.entropy_history = data["entropy_history"]
        return new_belief

    def save(self, filepath: Union[str, Path]) -> None:
        """Save belief state to file"""
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif filepath.suffix == ".pkl":
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DiscreteBeliefState":
        """Load belief state from file"""
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            with open(filepath) as f:
                data = json.load(f)
            dummy_instance = cls(num_states=1, config=BeliefStateConfig())
            return dummy_instance.from_dict(data)
        elif filepath.suffix == ".pkl":
            with open(filepath, "rb") as f:
                loaded_obj = pickle.load(f)
                if not isinstance(loaded_obj, cls):
                    raise TypeError(
                        f"Expected {
                            cls.__name__}, got " f"{
                            type(loaded_obj).__name__}")
                return loaded_obj
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def __repr__(self) -> str:
        """Return string representation of discrete belief state"""
        entropy = self.entropy().item()
        most_likely = self.most_likely_state()
        confidence = self.metadata["confidence_score"]
        return (
            f"DiscreteBeliefState(states={self.num_states}, "
            f"entropy={entropy:.3f}, "
            f"most_likely={most_likely}, "
            f"confidence={confidence:.3f}, "
            f"updates={self.update_count})"
        )


class ContinuousBeliefState(BeliefState):
    """
    Continuous belief state representation using Gaussian distributions.

    Represents beliefs as multivariate Gaussian with mean and covariance.
    """

    def __init__(
        self,
        state_dim: int,
        config: BeliefStateConfig,
        initial_mean: Optional[torch.Tensor] = None,
        initial_cov: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize continuous belief state"""
        super().__init__(config)
        self.state_dim = state_dim
        # Initialize Gaussian parameters
        if initial_mean is not None:
            self.mean = initial_mean.to(self.device, dtype=config.dtype)
        else:
            self.mean = torch.zeros(
                state_dim,
                device=self.device,
                dtype=config.dtype)
        if initial_cov is not None:
            self.cov = initial_cov.to(self.device, dtype=config.dtype)
        else:
            self.cov = torch.eye(
                state_dim,
                device=self.device,
                dtype=config.dtype)
        # For numerical stability
        self.log_var = torch.log(torch.diag(self.cov))
        # History and metadata
        self.mean_history: List[torch.Tensor] = []
        self.cov_history: List[torch.Tensor] = []
        self.metadata = {
            "state_dim": state_dim,
            "determinant": torch.det(self.cov).item(),
            "trace": torch.trace(self.cov).item(),
        }

    def get_beliefs(self) -> torch.Tensor:
        """Get current belief parameters.

        Returns concatenated mean and covariance diagonal for compatibility.
        """
        return torch.cat([self.mean, self.log_var])

    def set_beliefs(self,
                    beliefs: Union[torch.Tensor,
                                   tuple[torch.Tensor,
                                         torch.Tensor]]) -> None:
        """Set belief parameters"""
        if isinstance(beliefs, tuple):
            mean, cov = beliefs
            self.mean = mean.to(self.device, dtype=self.config.dtype)
            self.cov = cov.to(self.device, dtype=self.config.dtype)
        else:
            # Assume beliefs is just the mean, keep current covariance
            self.mean = beliefs.to(self.device, dtype=self.config.dtype)
        self.log_var = torch.log(torch.diag(self.cov) + self.config.eps)
        self._update_metadata()

    def update_beliefs(
        self,
        evidence: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        update_method: str = "bayes",
    ) -> "ContinuousBeliefState":
        """
        Update Gaussian belief with evidence.

        Args:
            evidence: Evidence tensor or tuple of (observation, covariance)
            update_method: 'bayes', 'kalman', 'variational'
        """
        self._add_to_history()
        # Handle both tensor and tuple evidence inputs
        if isinstance(evidence, tuple):
            obs, obs_cov = evidence
        elif evidence.dim() == 1:
            obs = evidence[: self.state_dim]
            obs_cov = torch.eye(self.state_dim, device=self.device) * 0.1
        else:
            obs = (
                evidence[0]
                if evidence.shape[0] >= 1
                else torch.zeros(self.state_dim, device=self.device)
            )
            obs_cov = torch.eye(self.state_dim, device=self.device) * 0.1

        evidence_tuple = (obs, obs_cov)
        if update_method == "bayes":
            self._gaussian_bayesian_update(evidence_tuple)
        elif update_method == "kalman":
            self._kalman_update(evidence_tuple)
        elif update_method == "variational":
            self._variational_update(evidence_tuple)
        else:
            raise ValueError(f"Unknown update method: {update_method}")
        self.update_count += 1
        return self

    def _gaussian_bayesian_update(
            self, evidence: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Perform standard Gaussian Bayesian update"""
        obs, obs_cov = evidence
        # Bayesian update for Gaussian
        # posterior_cov^{-1} = prior_cov^{-1} + obs_cov^{-1}
        cov_inv = torch.inverse(
            self.cov +
            self.config.eps *
            torch.eye(
                self.state_dim,
                device=self.device))
        obs_cov_inv = torch.inverse(
            obs_cov +
            self.config.eps *
            torch.eye(
                self.state_dim,
                device=self.device))
        posterior_cov_inv = cov_inv + obs_cov_inv
        self.cov = torch.inverse(posterior_cov_inv)
        # posterior_mean = posterior_cov *
        # (prior_cov^{-1} * prior_mean + obs_cov^{-1} * obs)
        self.mean = self.cov @ (cov_inv @ self.mean + obs_cov_inv @ obs)
        self.log_var = torch.log(torch.diag(self.cov) + self.config.eps)

    def _kalman_update(self,
                       evidence: tuple[torch.Tensor,
                                       torch.Tensor]) -> None:
        """Kalman filter update"""
        obs, obs_cov = evidence
        # Kalman gain
        S = self.cov + obs_cov  # Innovation covariance
        K = self.cov @ torch.inverse(S)  # Kalman gain
        # Update
        innovation = obs - self.mean
        self.mean = self.mean + K @ innovation
        self.cov = (
            torch.eye(
                self.state_dim,
                device=self.device) -
            K) @ self.cov
        self.log_var = torch.log(torch.diag(self.cov) + self.config.eps)

    def _variational_update(self,
                            evidence: tuple[torch.Tensor,
                                            torch.Tensor]) -> None:
        """Perform variational Bayes update with regularization"""
        obs, obs_cov = evidence
        # Add entropy regularization to covariance
        reg_cov = self.cov + self.config.entropy_regularization * torch.eye(
            self.state_dim, device=self.device
        )
        # Standard Bayesian update with regularized covariance
        cov_inv = torch.inverse(
            reg_cov +
            self.config.eps *
            torch.eye(
                self.state_dim,
                device=self.device))
        obs_cov_inv = torch.inverse(
            obs_cov +
            self.config.eps *
            torch.eye(
                self.state_dim,
                device=self.device))
        posterior_cov_inv = cov_inv + obs_cov_inv
        self.cov = torch.inverse(posterior_cov_inv)
        self.mean = self.cov @ (cov_inv @ self.mean + obs_cov_inv @ obs)
        self.log_var = torch.log(torch.diag(self.cov) + self.config.eps)

    def _add_to_history(self) -> None:
        """Add current state to history"""
        if len(self.mean_history) >= self.config.max_history_length:
            self.mean_history.pop(0)
            self.cov_history.pop(0)
        self.mean_history.append(self.mean.clone())
        self.cov_history.append(self.cov.clone())

    def _update_metadata(self) -> None:
        """Update metadata"""
        self.metadata["determinant"] = torch.det(self.cov).item()
        self.metadata["trace"] = torch.trace(self.cov).item()

    def entropy(self) -> torch.Tensor:
        """Compute differential entropy of Gaussian"""
        # H = 0.5 * log((2π)^d * |Σ|)
        det_cov = torch.det(self.cov)
        entropy_val = 0.5 * (
            self.state_dim * np.log(2 * np.pi) + torch.log(det_cov + self.config.eps)
        )
        return torch.tensor(
            entropy_val,
            device=self.device,
            dtype=self.config.dtype)

    def most_likely_state(self) -> torch.Tensor:
        """Get most likely state (mean for continuous distributions)"""
        return self.mean.clone()

    def sample_state(self, num_samples: int = 1) -> torch.Tensor:
        """Sample states from Gaussian belief"""
        dist = torch.distributions.MultivariateNormal(self.mean, self.cov)
        return dist.sample((num_samples,))

    def clone(self) -> "ContinuousBeliefState":
        """Create deep copy"""
        new_belief = ContinuousBeliefState(
            state_dim=self.state_dim,
            config=self.config,
            initial_mean=self.mean.clone(),
            initial_cov=self.cov.clone(),
        )
        new_belief.metadata = self.metadata.copy()
        new_belief.update_count = self.update_count
        return new_belief

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "type": "ContinuousBeliefState",
            "state_dim": self.state_dim,
            "mean": self.mean.cpu().numpy().tolist(),
            "cov": self.cov.cpu().numpy().tolist(),
            "config": {
                "use_gpu": self.config.use_gpu,
                "eps": self.config.eps,
                "entropy_regularization": self.config.entropy_regularization,
            },
            "metadata": self.metadata,
            "update_count": self.update_count,
        }

    def from_dict(self, data: Dict[str, Any]) -> "ContinuousBeliefState":
        """Deserialize from dictionary"""
        config = BeliefStateConfig(**data["config"])
        mean = torch.tensor(data["mean"], dtype=config.dtype)
        cov = torch.tensor(data["cov"], dtype=config.dtype)
        new_belief = ContinuousBeliefState(
            state_dim=data["state_dim"],
            config=config,
            initial_mean=mean,
            initial_cov=cov)
        new_belief.metadata = data["metadata"]
        new_belief.update_count = data["update_count"]
        return new_belief

    def __repr__(self) -> str:
        """Return string representation of continuous belief state"""
        entropy = self.entropy().item()
        det = self.metadata["determinant"]
        return (
            f"ContinuousBeliefState(dim={self.state_dim}, "
            f"entropy={entropy:.3f}, "
            f"det={det:.3f}, "
            f"updates={self.update_count})"
        )


# Factory functions for easy creation
def create_discrete_belief_state(
    num_states: int,
    config: Optional[BeliefStateConfig] = None,
    initial_beliefs: Optional[torch.Tensor] = None,
) -> DiscreteBeliefState:
    """Create discrete belief state with optional configuration"""
    if config is None:
        config = BeliefStateConfig()
    return DiscreteBeliefState(num_states, config, initial_beliefs)


def create_continuous_belief_state(
    state_dim: int,
    config: Optional[BeliefStateConfig] = None,
    initial_mean: Optional[torch.Tensor] = None,
    initial_cov: Optional[torch.Tensor] = None,
) -> ContinuousBeliefState:
    """Create continuous belief state with optional configuration"""
    if config is None:
        config = BeliefStateConfig()
    return ContinuousBeliefState(state_dim, config, initial_mean, initial_cov)


def create_belief_state(belief_type: str, **kwargs: Any) -> BeliefState:
    """
    Create belief states by type.

    Args:
        belief_type: 'discrete' or 'continuous'
        **kwargs: Arguments for specific belief state type
    Returns:
        BeliefState instance
    """
    if belief_type == "discrete":
        return create_discrete_belief_state(**kwargs)
    elif belief_type == "continuous":
        return create_continuous_belief_state(**kwargs)
    else:
        raise ValueError(f"Unknown belief type: {belief_type}")

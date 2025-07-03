"""
Belief Update Module for Active Inference.

This module implements belief update mechanisms for Active Inference systems,
integrating with GraphNN (Graph Neural Networks) when needed.

Note: GraphNN refers to Graph Neural Networks (machine learning),
distinct from GNN (Generalized Notation Notation) from Active
Inference Institute.
Reference:
https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .generative_model import GenerativeModel

logger = logging.getLogger(__name__)


@dataclass
class BeliefUpdateConfig:
    """Configuration for belief update mechanisms"""

    update_method: str = "variational"
    learning_rate: float = 0.01
    num_iterations: int = 10
    convergence_threshold: float = 1e-6
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32


class BeliefUpdater(ABC):
    """Abstract base class for belief update mechanisms"""

    def __init__(self, config: BeliefUpdateConfig) -> None:
        """Initialize belief updater"""
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def update_beliefs(
        self,
        current_beliefs: torch.Tensor,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
    ) -> torch.Tensor:
        """Update beliefs given observations and generative model"""


class DirectGraphObservationModel:
    """Direct mapping from graph features to observations"""

    def __init__(self, config: BeliefUpdateConfig) -> None:
        """Initialize belief updater"""
        self.config = config

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        """Map graph features to observations"""
        return graph_features


class LearnedGraphObservationModel(nn.Module):
    """Learned mapping from graph features to observations"""

    def __init__(self, config: BeliefUpdateConfig, input_dim: int, output_dim: int) -> None:
        """Initialize learned graph observation model"""
        super().__init__()
        self.config = config
        self.network = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        """Map graph features to observations"""
        result: torch.Tensor = self.network(graph_features)
        return result


class GraphNNBeliefUpdater(BeliefUpdater):
    """Belief updater that integrates Graph Neural Network features"""

    def __init__(self, config: BeliefUpdateConfig) -> None:
        """Initialize belief updater"""
        super().__init__(config)

    def update_beliefs(
        self,
        current_beliefs: torch.Tensor,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
    ) -> torch.Tensor:
        """Update beliefs using GraphNN integration"""
        # Placeholder implementation
        return current_beliefs


class AttentionGraphBeliefUpdater(BeliefUpdater):
    """Belief updater with attention mechanisms for graph features"""

    def __init__(self, config: BeliefUpdateConfig) -> None:
        """Initialize belief updater"""
        super().__init__(config)

    def update_beliefs(
        self,
        current_beliefs: torch.Tensor,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
    ) -> torch.Tensor:
        """Update beliefs using attention over graph features"""
        # Placeholder implementation
        return current_beliefs


class HierarchicalBeliefUpdater(BeliefUpdater):
    """Hierarchical belief updater for multi-level systems"""

    def __init__(self, config: BeliefUpdateConfig) -> None:
        """Initialize belief updater"""
        super().__init__(config)

    def update_beliefs(
        self,
        current_beliefs: torch.Tensor,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
    ) -> torch.Tensor:
        """Update beliefs in hierarchical manner"""
        # Placeholder implementation
        return current_beliefs


class DirectBeliefUpdater(BeliefUpdater):
    """Direct belief updater with proper normalization (pymdp-aligned)"""

    def __init__(self, config: BeliefUpdateConfig) -> None:
        """Initialize direct belief updater"""
        super().__init__(config)

    def update_beliefs(
        self,
        current_beliefs: torch.Tensor,
        observations: torch.Tensor,
        generative_model: GenerativeModel,
    ) -> torch.Tensor:
        """Update beliefs with proper normalization following pymdp conventions"""
        # Handle invalid beliefs by normalizing them to valid probability
        # distributions

        # Convert numpy arrays to torch tensors if needed (pymdp compatibility)
        if isinstance(current_beliefs, np.ndarray):
            beliefs = torch.from_numpy(current_beliefs.astype(np.float32))
        else:
            beliefs = current_beliefs.clone()

        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations.astype(np.float32))

        # Replace NaN values with small positive numbers
        beliefs = torch.nan_to_num(beliefs, nan=1e-8)

        # Ensure all values are positive
        beliefs = torch.clamp(beliefs, min=1e-8)

        # Normalize to sum to 1.0 (proper probability distribution)
        beliefs = beliefs / beliefs.sum()

        # Perform Bayesian update: P(s|o) ∝ P(o|s)P(s) (pymdp convention)
        if hasattr(generative_model, "observation_model") and hasattr(generative_model, "A"):
            try:
                # Get observation matrix A (observations x states) - pymdp
                # convention
                A_matrix = generative_model.A
                obs_idx = (
                    observations.long() if observations.numel() == 1 else observations[0].long()
                )

                # Extract likelihood P(o|s) for the observed outcome
                if obs_idx < A_matrix.shape[0] and A_matrix.shape[1] == len(beliefs):
                    likelihood = A_matrix[obs_idx, :]  # P(o=obs_idx | s)

                    # Bayesian update: P(s|o) ∝ P(o|s)P(s)
                    posterior = beliefs * likelihood
                    # Normalize posterior
                    posterior = posterior / (posterior.sum() + 1e-8)
                    return posterior
                else:
                    # Dimension mismatch, return normalized beliefs
                    return beliefs
            except Exception:
                # Fallback to normalized beliefs if any error occurs
                return beliefs
        else:
            # No generative model available, just return normalized beliefs
            return beliefs


def create_belief_updater(updater_type: str, config: BeliefUpdateConfig) -> BeliefUpdater:
    """Create belief updaters"""
    if updater_type == "direct":
        return DirectBeliefUpdater(config)
    elif updater_type == "graphnn":
        return GraphNNBeliefUpdater(config)
    elif updater_type == "attention":
        return AttentionGraphBeliefUpdater(config)
    elif updater_type == "hierarchical":
        return HierarchicalBeliefUpdater(config)
    else:
        # Default to direct with proper normalization
        return DirectBeliefUpdater(config)

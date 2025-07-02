"""
Generative Model Architecture for Active Inference aligned with PyMDP conventions.

This module implements the core generative model components following PyMDP's
mathematical formulations and matrix conventions. Supports LLM-generated models
through Generalized Notation Notation (GNN) integration (avoiding confusion with
Graph Neural Networks, sometimes referred to as GMN in this codebase).

PyMDP Matrix Conventions:
- A matrix: P(obs|states) shape (num_obs, num_states) - columns sum to 1
- B matrix: P(next_state|current_state, action) shape (num_states, num_states, num_actions)
- C matrix: Prior preferences (log probabilities) shape (num_obs, time_horizon)
- D vector: Initial state prior shape (num_states,) - sums to 1
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ModelDimensions:
    """Dimensions for the generative model components following PyMDP conventions."""

    num_states: int
    num_observations: int
    num_actions: int
    num_modalities: int = 1
    num_factors: int = 1
    time_horizon: int = 1


@dataclass
class ModelParameters:
    """Hyperparameters for the generative model."""

    learning_rate: float = 0.01
    precision_init: float = 1.0
    use_sparse: bool = False
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-8
    temperature: float = 1.0

    # GNN/GMN notation support
    gnn_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class GenerativeModel(ABC):
    """Abstract base class for Active Inference generative models following PyMDP.

    Defines the interface for models that specify:
    - p(o|s): Observation model (A matrix/function)
    - p(s'|s, a): Transition model (B tensor/function)
    - p(o|C): Prior preferences (C matrix/function)
    - p(s): Initial state prior (D vector/function)

    Compatible with LLM-generated models through GNN/GMN notation.
    """

    def __init__(
        self,
        dimensions: ModelDimensions,
        parameters: ModelParameters,
        gnn_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize generative model with dimensions, parameters, and optional GNN metadata."""
        self.dims = dimensions
        self.params = parameters
        self.device = torch.device(
            "cuda" if parameters.use_gpu and torch.cuda.is_available() else "cpu")

        # GNN/GMN notation support for LLM integration
        self.gnn_metadata = gnn_metadata or parameters.gnn_metadata or {}

    @abstractmethod
    def observation_model(
        self, states: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute p(o|s) following PyMDP A matrix conventions."""

    @abstractmethod
    def transition_model(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute p(s'|s, a) following PyMDP B matrix conventions."""

    @abstractmethod
    def get_preferences(self, timestep: Optional[int] = None) -> torch.Tensor:
        """Get prior preferences p(o|C) following PyMDP C matrix conventions."""

    @abstractmethod
    def get_initial_prior(self) -> Union[torch.Tensor,
                                         Tuple[torch.Tensor, torch.Tensor]]:
        """Get initial state prior p(s) following PyMDP D vector conventions."""

    def set_preferences(self, preferences: torch.Tensor) -> None:
        """Set prior preferences (C matrix) - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement set_preferences")

    def validate_pymdp_matrices(self) -> bool:
        """Validate that matrices follow PyMDP conventions."""
        if (
            not hasattr(self, "A")
            or not hasattr(self, "B")
            or not hasattr(self, "C")
            or not hasattr(self, "D")
        ):
            return False

        # Check A matrix shape and normalization
        if self.A.shape != (self.dims.num_observations, self.dims.num_states):
            return False

        # Check A matrix columns sum to 1 (stochastic)
        for s in range(self.dims.num_states):
            if not torch.allclose(
                    self.A[:, s].sum(), torch.tensor(1.0), atol=1e-6):
                return False

        # Check B matrix shape and normalization
        if self.B.shape != (
                self.dims.num_states,
                self.dims.num_states,
                self.dims.num_actions):
            return False

        # Check B matrix transitions sum to 1
        for a in range(self.dims.num_actions):
            for s in range(self.dims.num_states):
                if not torch.allclose(self.B[:, s, a].sum(),
                                      torch.tensor(1.0), atol=1e-6):
                    return False

        # Check D vector sums to 1
        if not torch.allclose(self.D.sum(), torch.tensor(1.0), atol=1e-6):
            return False

        return True


class DiscreteGenerativeModel(GenerativeModel):
    """Discrete state-space generative model using categorical distributions.

    Follows PyMDP matrix conventions exactly:
    - A matrix: P(obs|states) shape (num_obs, num_states) - columns sum to 1
    - B matrix: P(next_state|current_state, action) shape (num_states, num_states, num_actions)
    - C matrix: Prior preferences (log probabilities) shape (num_obs, time_horizon)
    - D vector: Initial state prior shape (num_states,) - sums to 1

    Supports LLM-generated models through GNN/GMN notation.
    """

    def __init__(
        self,
        dimensions: ModelDimensions,
        parameters: ModelParameters,
        gnn_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize discrete generative model with PyMDP matrix conventions."""
        super().__init__(dimensions, parameters, gnn_metadata)

        # Initialize PyMDP matrices as Parameters for gradient computation
        self.A = nn.Parameter(self._initialize_A(), requires_grad=False)
        self.B = nn.Parameter(self._initialize_B(), requires_grad=False)
        self.C = nn.Parameter(self._initialize_C(), requires_grad=False)
        self.D = nn.Parameter(self._initialize_D(), requires_grad=False)

        # Move to device
        self.to_device(parameters.use_gpu)

        # Validate PyMDP conventions
        if not self.validate_pymdp_matrices():
            logger.warning("Model matrices do not follow PyMDP conventions!")

        logger.info(
            f"Initialized PyMDP-compatible discrete generative model with "
            f"{dimensions.num_states} states, "
            f"{dimensions.num_observations} observations, "
            f"{dimensions.num_actions} actions"
        )

        if self.gnn_metadata:
            logger.info(
                f"Model includes GNN/GMN metadata: {list(self.gnn_metadata.keys())}")

    def _initialize_A(self) -> torch.Tensor:
        """Initialize observation model following PyMDP A matrix conventions.

        A matrix shape: (num_observations, num_states)
        Each column A[:, s] is P(o|s=s) and must sum to 1.
        """
        # Defensive: check for any dynamic isinstance() usage
        # (No such call is present, but add a guard for future-proofing)
        if hasattr(torch.distributions, "Dirichlet") and not isinstance(
            torch.distributions.Dirichlet, type
        ):
            raise TypeError(
                "torch.distributions.Dirichlet is not a type. "
                "Check for dynamic isinstance() usage."
            )
        if self.params.use_sparse:
            # For sparse initialization, assume mostly diagonal observation
            A = torch.eye(
                min(self.dims.num_observations, self.dims.num_states),
                dtype=self.params.dtype,
            )

            # Expand to full size if needed
            if self.dims.num_observations != self.dims.num_states:
                A_full = torch.zeros(
                    self.dims.num_observations,
                    self.dims.num_states,
                    dtype=self.params.dtype,
                )
                min_dim = min(self.dims.num_observations, self.dims.num_states)
                A_full[:min_dim, :min_dim] = A[:min_dim, :min_dim]
                A = A_full

            # Add small random noise
            A += 0.1 * torch.rand_like(A)
        else:
            # Random initialization with Dirichlet (proper categorical
            # distribution)
            A = torch.zeros(
                self.dims.num_observations,
                self.dims.num_states,
                dtype=self.params.dtype,
            )
            for s in range(self.dims.num_states):
                # Sample from Dirichlet for each state (PyMDP convention)
                alpha = torch.ones(
                    self.dims.num_observations,
                    dtype=self.params.dtype,
                    device=self.device)
                # Ensure alpha is positive and properly formatted for Dirichlet
                alpha = torch.clamp(alpha, min=1e-8)
                try:
                    dirichlet_dist = torch.distributions.Dirichlet(alpha)
                    A[:, s] = dirichlet_dist.sample()
                except (TypeError, ValueError):
                    # Fallback to manual normalization if Dirichlet fails
                    random_vals = torch.rand(
                        self.dims.num_observations,
                        dtype=self.params.dtype,
                        device=self.device)
                    A[:, s] = random_vals / random_vals.sum()

        # Ensure columns sum to 1 (PyMDP requirement)
        return self._normalize_columns(A)

    def _initialize_B(self) -> torch.Tensor:
        """Initialize transition model following PyMDP B matrix conventions.

        B matrix shape: (num_states, num_states, num_actions)
        Each B[:, s, a] is P(s'|s=s, a=a) and must sum to 1.
        """
        B = torch.zeros(
            self.dims.num_states,
            self.dims.num_states,
            self.dims.num_actions,
            dtype=self.params.dtype,
        )

        for a in range(self.dims.num_actions):
            if self.params.use_sparse:
                # Sparse transitions - mostly stay in same state
                B[:, :, a] = 0.8 * torch.eye(self.dims.num_states) + 0.2 * torch.rand(
                    self.dims.num_states, self.dims.num_states)
            else:
                # Random transitions using Dirichlet
                for s in range(self.dims.num_states):
                    alpha = torch.ones(
                        self.dims.num_states,
                        dtype=self.params.dtype,
                        device=self.device)
                    # Ensure alpha is positive and properly formatted
                    alpha = torch.clamp(alpha, min=1e-8)
                    try:
                        dirichlet_dist = torch.distributions.Dirichlet(alpha)
                        B[:, s, a] = dirichlet_dist.sample()
                    except (TypeError, ValueError):
                        # Fallback to manual normalization if Dirichlet fails
                        random_vals = torch.rand(
                            self.dims.num_states,
                            dtype=self.params.dtype,
                            device=self.device)
                        B[:, s, a] = random_vals / random_vals.sum()

        # Ensure proper normalization (PyMDP requirement)
        return self._normalize_transitions(B)

    def _initialize_C(self) -> torch.Tensor:
        """Initialize preferences following PyMDP C matrix conventions.

        C matrix shape: (num_observations, time_horizon)
        Contains log preferences - higher values indicate stronger preference.
        """
        # Initialize with neutral preferences (log probabilities = 0)
        C = torch.zeros(
            self.dims.num_observations,
            self.dims.time_horizon,
            dtype=self.params.dtype)
        return C

    def _initialize_D(self) -> torch.Tensor:
        """Initialize uniform prior following PyMDP D vector conventions.

        D vector shape: (num_states,)
        Must sum to 1 (probability distribution).
        """
        D = torch.ones(
            self.dims.num_states,
            dtype=self.params.dtype) / self.dims.num_states
        return D

    def _normalize_columns(self, matrix: torch.Tensor) -> torch.Tensor:
        """Normalize matrix columns to sum to 1 (PyMDP A matrix requirement)."""
        return matrix / (matrix.sum(dim=0, keepdim=True) + self.params.eps)

    def _normalize_transitions(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize transition tensor (PyMDP B matrix requirement)."""
        # Normalize each B[:, s, a] to sum to 1
        for a in range(tensor.shape[2]):
            for s in range(tensor.shape[1]):
                column_sum = tensor[:, s, a].sum()
                if column_sum > self.params.eps:
                    tensor[:, s, a] = tensor[:, s, a] / column_sum
        return tensor

    def to_device(self, use_gpu: bool) -> None:
        """Move all PyMDP matrices to specified device."""
        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.A.data = self.A.data.to(device)
        self.B.data = self.B.data.to(device)
        self.C.data = self.C.data.to(device)
        self.D.data = self.D.data.to(device)

    def observation_model(self, states: torch.Tensor) -> torch.Tensor:
        """Compute observation probabilities following PyMDP A matrix conventions.

        Args:
            states: State distribution [num_states] or state index (scalar)
        Returns:
            observations: Observation probabilities [num_observations]
        """
        if states.dim() == 0:
            # Single state index
            return self.A[:, states]
        else:
            # State distribution: A @ states
            return torch.matmul(self.A, states)

    def transition_model(
        self, states: torch.Tensor, actions: Union[torch.Tensor, int]
    ) -> torch.Tensor:
        """Compute next state probabilities following PyMDP B matrix conventions.

        Args:
            states: Current state distribution [num_states] or state index (scalar)
            actions: Action index (scalar or tensor)
        Returns:
            next_states: Next state probabilities [num_states]
        """
        # Convert actions to tensor if needed
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device)
        elif not actions.is_cuda and self.device.type == "cuda":
            actions = actions.to(self.device)

        # Extract action index
        if actions.dim() > 0:
            action_idx = int(actions.item())
        else:
            action_idx = int(actions)

        if states.dim() == 0:
            # Single state index
            return self.B[:, states, action_idx]
        else:
            # State distribution: B[:, :, action] @ states
            return torch.matmul(self.B[:, :, action_idx], states)

    def get_preferences(self, timestep: Optional[int] = None) -> torch.Tensor:
        """Get prior preferences following PyMDP C matrix conventions."""
        if timestep is None:
            return self.C
        else:
            if timestep >= self.dims.time_horizon:
                # Return last timestep preferences for beyond horizon
                return self.C[:, -1]
            return self.C[:, timestep]

    def get_initial_prior(self) -> torch.Tensor:
        """Get initial state prior following PyMDP D vector conventions."""
        return self.D.clone()

    def set_preferences(
        self,
        preferences: Union[torch.Tensor, np.ndarray],
        timestep: Optional[int] = None,
    ) -> None:
        """Set prior preferences (C matrix) following PyMDP conventions.

        Args:
            preferences: Preference values (log probabilities)
            timestep: Optional specific timestep to set
        """
        if isinstance(preferences, np.ndarray):
            preferences = torch.tensor(
                preferences,
                dtype=self.params.dtype,
                device=self.device)

        if timestep is None:
            if preferences.dim() == 1:
                # Set same preferences for all timesteps
                self.C.data = preferences.unsqueeze(
                    1).repeat(1, self.dims.time_horizon)
            else:
                self.C.data = preferences.to(self.device)
        else:
            self.C[:, timestep] = preferences.to(self.device)

    def update_model_pymdp(
        self,
        observations: List[torch.Tensor],
        states: List[torch.Tensor],
        actions: List[int],
    ) -> None:
        """Update model parameters using PyMDP Bayesian learning.

        Args:
            observations: List of observation indices or distributions
            states: List of state distributions
            actions: List of actions taken
        """
        # Accumulate sufficient statistics (Dirichlet counts)
        A_counts = torch.ones_like(self.A)  # Start with prior counts
        B_counts = torch.ones_like(self.B)

        # Update A matrix statistics (observation model)
        for obs, state in zip(observations, states):
            if obs.dim() == 0:  # Single observation index
                A_counts[obs, :] += state
            else:  # Observation distribution
                A_counts += torch.outer(obs, state)

        # Update B tensor statistics (transition model)
        for t in range(len(states) - 1):
            state_curr = states[t]
            state_next = states[t + 1]
            action = actions[t]
            B_counts[:, :, action] += torch.outer(state_next, state_curr)

        # Apply Bayesian updates with learning rate
        self.A.data = self._normalize_columns(
            (1 - self.params.learning_rate) * self.A.data
            + self.params.learning_rate * self._normalize_columns(A_counts)
        )

        for a in range(self.dims.num_actions):
            B_slice = self.B.data[:, :, a]
            B_counts_slice = B_counts[:, :, a]
            updated_B = (1 - self.params.learning_rate) * B_slice + \
                self.params.learning_rate * self._normalize_columns(B_counts_slice)
            self.B.data[:, :, a] = self._normalize_columns(updated_B)


class ContinuousGenerativeModel(GenerativeModel, nn.Module):
    """Continuous state-space generative model using Gaussian distributions.

    Uses neural networks to parameterize the observation and transition models
    while maintaining compatibility with PyMDP-style inference algorithms.
    """

    def __init__(
        self,
        dimensions: ModelDimensions,
        parameters: ModelParameters,
        hidden_dim: int = 128,
        gnn_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize continuous generative model with PyMDP compatibility."""
        GenerativeModel.__init__(self, dimensions, parameters, gnn_metadata)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim

        # Initialize neural network components
        self._build_networks()

        # Initialize preferences and prior (PyMDP-style)
        self.C = nn.Parameter(
            torch.zeros(
                dimensions.num_observations,
                dimensions.time_horizon,
                dtype=parameters.dtype,
            )
        )
        self.D_mean = nn.Parameter(
            torch.zeros(
                dimensions.num_states,
                dtype=parameters.dtype))
        self.D_log_var = nn.Parameter(
            torch.zeros(
                dimensions.num_states,
                dtype=parameters.dtype))

        # Move to device
        self.to(self.device)

        logger.info(
            f"Initialized continuous generative model with "
            f"{dimensions.num_states}D states, "
            f"{dimensions.num_observations}D observations"
        )

    def _build_networks(self) -> None:
        """Build neural network components for continuous dynamics."""
        # Observation model: p(o|s)
        self.obs_net = nn.Sequential(
            nn.Linear(self.dims.num_states, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.obs_mean = nn.Linear(self.hidden_dim, self.dims.num_observations)
        self.obs_log_var = nn.Parameter(
            torch.zeros(self.dims.num_observations))

        # Transition model: p(s'|s, a)
        self.trans_net = nn.Sequential(
            nn.Linear(
                self.dims.num_states +
                self.dims.num_actions,
                self.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim),
            nn.ReLU(),
        )
        self.trans_mean = nn.Linear(self.hidden_dim, self.dims.num_states)
        self.trans_log_var = nn.Parameter(torch.zeros(self.dims.num_states))

    def observation_model(
            self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute observation distribution parameters given states.

        Args:
            states: State values [batch_size x num_states] or [num_states]
        Returns:
            obs_mean: Mean of observation distribution
            obs_var: Variance of observation distribution
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)

        h = self.obs_net(states)
        mean = self.obs_mean(h)
        var = torch.exp(self.obs_log_var)

        return mean.squeeze(0) if states.shape[0] == 1 else mean, var

    def transition_model(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute state transition distribution with PyMDP compatibility."""
        # Track if inputs were originally single tensors
        states_was_1d = states.dim() == 1
        actions_was_1d = (
            actions.dim() == 1 and (
                actions.shape[0] == 1 or actions.shape[0] == self.dims.num_actions)) or (
            actions.dim() == 2 and actions.shape[0] == 1)

        if states.dim() == 1:
            states = states.unsqueeze(0)

        if actions.dim() == 2 and actions.shape[-1] == 1:
            # Actions are in shape [batch_size, 1] - convert to indices
            actions = actions.squeeze(-1)

        # Handle different action representations
        if (
            actions.dim() == 1
            and actions.shape[0] == self.dims.num_actions
            and actions.dtype.is_floating_point
            and torch.allclose(actions.sum(), torch.tensor(1.0))
        ):
            # Actions are one-hot encoded
            actions_onehot = actions.float().unsqueeze(0)
        elif actions.dim() > 1 and actions.shape[-1] == self.dims.num_actions:
            # Actions are already one-hot encoded
            actions_onehot = actions.float()
        elif actions.dim() <= 1 or (actions.dim() == 2 and actions.shape[-1] == 1):
            # Actions are indices - convert to one-hot
            actions_flat = actions.flatten() if actions.dim() > 1 else actions
            actions_onehot = F.one_hot(
                actions_flat.long(), num_classes=self.dims.num_actions
            ).float()
        else:
            raise ValueError(f"Invalid action shape: {actions.shape}")

        if actions_onehot.dim() == 1:
            actions_onehot = actions_onehot.unsqueeze(0)

        # Match batch sizes
        if states.shape[0] != actions_onehot.shape[0]:
            if states.shape[0] == 1:
                states = states.repeat(actions_onehot.shape[0], 1)
            else:
                raise ValueError(
                    f"Batch size mismatch: states ({states.shape[0]}) and "
                    f"actions ({actions_onehot.shape[0]})"
                )

        # Forward pass
        state_action = torch.cat([states, actions_onehot], dim=-1)
        h = self.trans_net(state_action)
        mean = self.trans_mean(h)
        var = torch.exp(self.trans_log_var)

        # Preserve original tensor dimensions
        if states_was_1d and actions_was_1d:
            mean = mean.squeeze(0)

        return mean, var

    def get_preferences(self, timestep: Optional[int] = None) -> torch.Tensor:
        """Get prior preferences following PyMDP conventions."""
        if timestep is None:
            return self.C
        else:
            if timestep >= self.dims.time_horizon:
                return self.C[:, -1]
            return self.C[:, timestep]

    def get_initial_prior(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial state prior parameters (mean, variance)."""
        return self.D_mean, torch.exp(self.D_log_var)

    def set_preferences(self, preferences: torch.Tensor) -> None:
        """Set prior preferences (C matrix)."""
        if preferences.dim() == 1:
            self.C.data = preferences.unsqueeze(
                1).repeat(1, self.dims.time_horizon)
        else:
            self.C.data = preferences.to(self.device)

    def forward(
        self, states: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the generative model.

        Args:
            states: Current states
            actions: Optional actions for transition
        Returns:
            Dictionary with model outputs
        """
        outputs = {}

        # Observation model
        obs_mean, obs_var = self.observation_model(states)
        outputs["obs_mean"] = obs_mean
        outputs["obs_var"] = obs_var

        # Transition model if actions provided
        if actions is not None:
            next_mean, next_var = self.transition_model(states, actions)
            outputs["next_mean"] = next_mean
            outputs["next_var"] = next_var

        return outputs


class HierarchicalGenerativeModel(DiscreteGenerativeModel):
    """Hierarchical generative model with multiple levels following PyMDP conventions.

    Higher levels operate at slower timescales and provide context to lower levels.
    Each level maintains PyMDP matrix conventions.
    """

    def __init__(
        self,
        dimensions_list: List[ModelDimensions],
        parameters: ModelParameters,
        level_connections: Optional[Dict[int, List[int]]] = None,
        gnn_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize hierarchical model with PyMDP conventions.

        Args:
            dimensions_list: List of dimensions for each level
            parameters: Model parameters
            level_connections: Dict mapping level index to connected lower levels
            gnn_metadata: Optional GNN/GMN metadata for LLM compatibility
        """
        # Initialize base level (level 0)
        super().__init__(dimensions_list[0], parameters, gnn_metadata)

        self.num_levels = len(dimensions_list)
        self.dimensions = dimensions_list
        self.level_connections = level_connections or {
            i: [i + 1] for i in range(self.num_levels - 1)
        }

        # Initialize higher levels as separate DiscreteGenerativeModel
        # instances
        self.levels: List[GenerativeModel] = [
            self]  # Level 0 is the base model
        for i in range(1, self.num_levels):
            level_model = DiscreteGenerativeModel(
                dimensions_list[i], parameters)
            self.levels.append(level_model)

        # Initialize inter-level connections (E matrices)
        self.E_matrices = {}
        for upper_idx, lower_indices in self.level_connections.items():
            for lower_idx in lower_indices:
                if lower_idx < self.num_levels:
                    # E[lower_state, upper_state]: how upper level influences lower
                    # level
                    E = torch.rand(
                        dimensions_list[lower_idx].num_states,
                        dimensions_list[upper_idx].num_states,
                        dtype=parameters.dtype,
                    )
                    E = self._normalize_columns(E)
                    self.E_matrices[(upper_idx, lower_idx)] = E.to(self.device)

        logger.info(
            f"Initialized hierarchical model with {
                self.num_levels} levels")

    def get_level(self, level: int) -> GenerativeModel:
        """Get model at specific level."""
        return self.levels[level]

    def hierarchical_observation_model(
            self, states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute observations for all levels considering hierarchical influence.

        Args:
            states: List of state distributions for each level
        Returns:
            observations: List of observation distributions for each level
        """
        observations = []

        for level in range(self.num_levels):
            # Get base observation from level's own model
            level_obs_result = self.levels[level].observation_model(
                states[level])

            # Handle both tensor and tuple returns
            if not isinstance(level_obs_result, torch.Tensor):
                # Use mean for continuous models (tuple case)
                level_obs = level_obs_result[0]
            else:
                level_obs = level_obs_result

            # Add influence from higher levels
            for upper_level in range(level):
                if (upper_level, level) in self.E_matrices:
                    # Modulate observation based on higher level state
                    E = self.E_matrices[(upper_level, level)]
                    influence = torch.matmul(E, states[upper_level])

                    # Ensure compatible shapes for multiplication
                    if level_obs.dim() == 1:
                        level_obs = level_obs.unsqueeze(0)
                    if influence.dim() == 1:
                        influence = influence.unsqueeze(0)

                    # Match dimensions by broadcasting or reshaping
                    if level_obs.shape[-1] != influence.shape[-1]:
                        # If dimensions don't match, truncate or pad to match
                        min_dim = min(level_obs.shape[-1], influence.shape[-1])
                        level_obs = level_obs[..., :min_dim]
                        influence = influence[..., :min_dim]

                    # Combine with level observation (multiplicative
                    # modulation)
                    level_obs = level_obs * influence
                    level_obs = level_obs / \
                        (level_obs.sum(dim=-1, keepdim=True) + self.params.eps)

            # Squeeze single-element batch dimensions for consistency
            if level_obs.dim() == 2 and level_obs.shape[0] == 1:
                level_obs = level_obs.squeeze(0)

            observations.append(level_obs)
        return observations

    def hierarchical_transition_model(
        self, states: List[torch.Tensor], actions: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Compute state transitions considering hierarchical influence.

        Args:
            states: List of current state distributions
            actions: List of actions for each level
        Returns:
            next_states: List of next state distributions
        """
        next_states = []

        for level in range(self.num_levels):
            # Get base transition
            next_state_result = self.levels[level].transition_model(
                states[level], actions[level])

            # Handle both tensor and tuple returns
            if not isinstance(next_state_result, torch.Tensor):
                # Use mean for continuous models (tuple case)
                next_state = next_state_result[0]
            else:
                next_state = next_state_result

            # Add influence from higher levels
            for upper_level in range(level):
                if (upper_level, level) in self.E_matrices:
                    E = self.E_matrices[(upper_level, level)]
                    influence = torch.matmul(E, states[upper_level])

                    # Modulate transition
                    next_state = next_state * influence
                    next_state = next_state / (
                        next_state.sum(dim=-1, keepdim=True) + self.params.eps
                    )

            next_states.append(next_state)
        return next_states


class FactorizedGenerativeModel(DiscreteGenerativeModel):
    """Factorized generative model where states decompose into independent factors.

    Follows PyMDP conventions while allowing for more efficient inference
    and learning in high-dimensional spaces through factorization.
    """

    def __init__(
        self,
        factor_dimensions: List[int],
        num_observations: int,
        num_actions: int,
        parameters: ModelParameters,
        gnn_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize factorized model with PyMDP conventions.

        Args:
            factor_dimensions: Number of states for each factor
            num_observations: Total number of observations
            num_actions: Number of actions
            parameters: Model parameters
            gnn_metadata: Optional GNN/GMN metadata
        """
        self.num_factors = len(factor_dimensions)
        self.factor_dims = factor_dimensions

        # Total states is product of factor dimensions
        total_states = int(np.prod(factor_dimensions))
        dims = ModelDimensions(
            num_states=total_states,
            num_observations=num_observations,
            num_actions=num_actions,
            num_factors=self.num_factors,
        )

        super().__init__(dims, parameters, gnn_metadata)

        # Initialize factor-specific components
        self._initialize_factors()

        logger.info(
            f"Initialized factorized model with {
                self.num_factors} factors: {factor_dimensions}")

    def _initialize_factors(self) -> None:
        """Initialize factor-specific transition models following PyMDP conventions."""
        self.factor_B = []
        for _, dim in enumerate(self.factor_dims):
            # Each factor has its own transition model
            B_f = torch.zeros(
                dim,
                dim,
                self.dims.num_actions,
                dtype=self.params.dtype)
            for a in range(self.dims.num_actions):
                # Initialize with mostly independent transitions
                B_f[:, :, a] = 0.9 * \
                    torch.eye(dim) + 0.1 * torch.rand(dim, dim)
                B_f[:, :, a] = self._normalize_columns(B_f[:, :, a])
            self.factor_B.append(B_f.to(self.device))

    def factor_to_state_idx(self, factor_indices: List[int]) -> int:
        """Convert factor indices to global state index."""
        idx = 0
        multiplier = 1
        for f in reversed(range(self.num_factors)):
            idx += factor_indices[f] * multiplier
            multiplier *= self.factor_dims[f]
        return idx

    def state_to_factor_idx(self, state_idx: int) -> List[int]:
        """Convert global state index to factor indices."""
        indices = []
        for f in reversed(range(self.num_factors)):
            indices.append(state_idx % self.factor_dims[f])
            state_idx //= self.factor_dims[f]
        return list(reversed(indices))

    def factorized_transition(
        self, factor_states: List[torch.Tensor], action: int
    ) -> List[torch.Tensor]:
        """Compute transitions for each factor independently following PyMDP.

        Args:
            factor_states: List of state distributions for each factor
            action: Action to take
        Returns:
            next_factor_states: List of next state distributions
        """
        next_states = []
        for f in range(self.num_factors):
            # B_f[:, :, action] @ factor_states[f] following PyMDP convention
            next_state = torch.matmul(
                self.factor_B[f][:, :, action], factor_states[f])
            next_states.append(next_state)
        return next_states


def create_generative_model(model_type: str, **kwargs: Any) -> GenerativeModel:
    """Factory function to create PyMDP-compatible generative models.

    Args:
        model_type: Type of model ('discrete', 'continuous', 'hierarchical', 'factorized')
        **kwargs: Model-specific parameters
    Returns:
        Generative model instance compatible with PyMDP conventions
    """
    if model_type == "discrete":
        dims = kwargs.get("dimensions")
        if dims is None:
            raise ValueError(
                "DiscreteGenerativeModel requires 'dimensions' parameter")
        params = kwargs.get("parameters", ModelParameters())
        gnn_metadata = kwargs.get("gnn_metadata")
        return DiscreteGenerativeModel(dims, params, gnn_metadata)

    elif model_type == "continuous":
        dims = kwargs.get("dimensions")
        if dims is None:
            raise ValueError(
                "ContinuousGenerativeModel requires 'dimensions' parameter")
        params = kwargs.get("parameters", ModelParameters())
        hidden_dim = kwargs.get("hidden_dim", 128)
        gnn_metadata = kwargs.get("gnn_metadata")
        return ContinuousGenerativeModel(
            dims, params, hidden_dim, gnn_metadata)

    elif model_type == "hierarchical":
        dims_list = kwargs.get("dimensions_list")
        if dims_list is None:
            raise ValueError(
                "HierarchicalGenerativeModel requires 'dimensions_list' parameter")
        params = kwargs.get("parameters", ModelParameters())
        connections = kwargs.get("level_connections")
        gnn_metadata = kwargs.get("gnn_metadata")
        return HierarchicalGenerativeModel(
            dims_list, params, connections, gnn_metadata)

    elif model_type == "factorized":
        factor_dims = kwargs.get("factor_dimensions")
        num_obs = kwargs.get("num_observations")
        num_actions = kwargs.get("num_actions")
        if None in [factor_dims, num_obs, num_actions]:
            raise ValueError(
                "FactorizedGenerativeModel requires 'factor_dimensions', 'num_observations', and 'num_actions' parameters"
            )
        # Type assertions after None check
        assert factor_dims is not None
        assert num_obs is not None
        assert num_actions is not None
        params = kwargs.get("parameters", ModelParameters())
        gnn_metadata = kwargs.get("gnn_metadata")
        return FactorizedGenerativeModel(
            factor_dims, num_obs, num_actions, params, gnn_metadata)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# GNN/GMN Integration Functions for LLM Model Generation
def create_gnn_compatible_model(gnn_spec: Dict[str, Any]) -> GenerativeModel:
    """Create generative model from GNN specification for LLM integration.

    Args:
        gnn_spec: GNN specification dictionary with semantic model description

    Returns:
        GenerativeModel configured according to GNN specification
    """
    # Extract model type and dimensions from GNN spec
    model_type = gnn_spec.get("model_type", "discrete_generative_model")

    # Extract dimensions
    state_space = gnn_spec.get("state_space", {})
    obs_space = gnn_spec.get("observation_space", {})
    action_space = gnn_spec.get("action_space", {})
    time_settings = gnn_spec.get("time_settings", {})

    dims = ModelDimensions(
        num_states=state_space.get("size", 4),
        num_observations=obs_space.get("size", 3),
        num_actions=action_space.get("size", 2),
        time_horizon=time_settings.get("horizon", 1),
    )

    # Create parameters with GNN metadata
    # Default for LLM compatibility
    params = ModelParameters(use_gpu=False, gnn_metadata=gnn_spec)

    # Create model based on type
    model: GenerativeModel
    if "discrete" in model_type:
        model = DiscreteGenerativeModel(dims, params, gnn_spec)
    elif "continuous" in model_type:
        model = ContinuousGenerativeModel(dims, params, gnn_metadata=gnn_spec)
    else:
        # Default to discrete
        model = DiscreteGenerativeModel(dims, params, gnn_spec)

    # Apply semantic initialization if specified
    if "initial_parameterization" in gnn_spec:
        _apply_gnn_initialization(model, gnn_spec)

    return model


def _apply_gnn_initialization(
        model: GenerativeModel, gnn_spec: Dict[str, Any]) -> None:
    """Apply GNN-specified initialization to model matrices."""
    init_spec = gnn_spec.get("initial_parameterization", {})

    if isinstance(model, DiscreteGenerativeModel):
        # Apply A matrix initialization
        if init_spec.get("A_matrix") == "identity_with_noise":
            model.A.data = torch.eye(
                model.dims.num_observations, model.dims.num_states
            ) + 0.1 * torch.rand_like(model.A.data)
            model.A.data = model._normalize_columns(model.A.data)

        # Apply B matrix initialization
        if init_spec.get("B_matrix") == "linear_progression":
            for a in range(model.dims.num_actions):
                if a == 0:  # Stay action
                    model.B.data[:, :, a] = torch.eye(model.dims.num_states)
                else:  # Move action
                    B_move = torch.zeros(
                        model.dims.num_states, model.dims.num_states)
                    for s in range(model.dims.num_states - 1):
                        B_move[s + 1, s] = 1.0
                    B_move[-1, -1] = 1.0  # Absorbing state
                    model.B.data[:, :, a] = B_move

        # Apply C vector initialization
        if init_spec.get("C_vector") == "goal_seeking":
            # Avoid, neutral, prefer
            preferences = torch.tensor([-1.0, 0.0, 2.0])
            if len(preferences) == model.dims.num_observations:
                model.set_preferences(preferences)


def validate_gnn_model_compatibility(
        model: GenerativeModel, gnn_spec: Dict[str, Any]) -> bool:
    """Validate that model is compatible with GNN specification.

    Args:
        model: Generative model to validate
        gnn_spec: GNN specification

    Returns:
        True if compatible, False otherwise
    """
    # Check dimensional compatibility
    state_space = gnn_spec.get("state_space", {})
    if state_space.get("size", model.dims.num_states) != model.dims.num_states:
        return False

    obs_space = gnn_spec.get("observation_space", {})
    if obs_space.get(
        "size",
            model.dims.num_observations) != model.dims.num_observations:
        return False

    action_space = gnn_spec.get("action_space", {})
    if action_space.get(
        "size",
            model.dims.num_actions) != model.dims.num_actions:
        return False

    # Check PyMDP matrix conventions
    if isinstance(model, DiscreteGenerativeModel):
        if not model.validate_pymdp_matrices():
            return False

    return True


# Example usage and demonstration
if __name__ == "__main__":
    # Create PyMDP-compatible discrete model
    dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
    params = ModelParameters(learning_rate=0.01, use_gpu=False)
    model = DiscreteGenerativeModel(dims, params)

    # Validate PyMDP conventions
    assert model.validate_pymdp_matrices(), "Model should follow PyMDP conventions"

    # Test observation model
    state = torch.ones(4) / 4  # Uniform state distribution
    obs_result = model.observation_model(state)
    try:
        obs = obs_result[0] if len(obs_result) == 2 else obs_result
    except (TypeError, IndexError):
        obs = obs_result
    print(f"Observation distribution: {obs}")
    assert torch.allclose(obs.sum(), torch.tensor(
        1.0)), "Observations should sum to 1"

    # Test transition model
    action_tensor = torch.tensor(0, dtype=torch.long)
    next_state_result = model.transition_model(state, action_tensor)
    try:
        next_state = next_state_result[0] if len(
            next_state_result) == 2 else next_state_result
    except (TypeError, IndexError):
        next_state = next_state_result
    print(f"Next state distribution: {next_state}")
    assert torch.allclose(next_state.sum(), torch.tensor(1.0)
                          ), "Next states should sum to 1"

    # Set preferences following PyMDP conventions
    # Avoid obs 0, neutral obs 1, prefer obs 2
    preferences = torch.tensor([-1.0, 0.0, 2.0])
    model.set_preferences(preferences)
    print(f"Preferences set: {model.get_preferences(0)}")

    # Test GNN integration
    gnn_spec = {
        "model_type": "discrete_generative_model", "state_space": {
            "size": 4, "semantic_labels": [
                "start", "middle", "goal", "trap"]}, "observation_space": {
            "size": 3, "semantic_labels": [
                "wall", "open", "goal_reached"]}, "action_space": {
            "size": 2, "semantic_labels": [
                "wait", "move"]}, "llm_generated": True, }

    gnn_model = create_gnn_compatible_model(gnn_spec)
    assert validate_gnn_model_compatibility(
        gnn_model, gnn_spec), "GNN model should be compatible"
    print("✓ PyMDP-compatible model with GNN/GMN notation successfully created")

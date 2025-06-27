"""
Generative Model Architecture for Active Inference.

This module implements the core generative model components that define
the probabilistic relationships between hidden states, observations, and
actions.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ModelDimensions:
    """Dimensions for the generative model components"""

    num_states: int
    num_observations: int
    num_actions: int
    num_modalities: int = 1
    num_factors: int = 1
    time_horizon: int = 1


@dataclass
class ModelParameters:
    """Hyperparameters for the generative model"""

    learning_rate: float = 0.01
    precision_init: float = 1.0
    use_sparse: bool = False
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-8
    temperature: float = 1.0


class GenerativeModel(ABC):
    """Abstract base class for Active Inference generative models.


    Defines the interface for models that specify:
    - p(o|s): Observation model (A matrix/function)
    - p(s'|s, a): Transition model (B tensor/function)
    - p(o|C): Prior preferences (C matrix/function)
    - p(s): Initial state prior (D vector/function)
    """

    def __init__(self, dimensions: ModelDimensions, parameters: ModelParameters) -> None:
        """Initialize generative model with dimensions and parameters"""
        self.dims = dimensions
        self.params = parameters
        self.device = torch.device(
            "cuda" if parameters.use_gpu and torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def observation_model(self, states: torch.Tensor) -> torch.Tensor:
        """Compute p(o|s)"""
        pass

    @abstractmethod
    def transition_model(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute p(s'|s, a)"""
        pass

    @abstractmethod
    def get_preferences(self, timestep: Optional[int] = None) -> torch.Tensor:
        """Get prior preferences p(o|C)"""
        pass

    @abstractmethod
    def get_initial_prior(self) -> torch.Tensor:
        """Get initial state prior p(s)"""
        pass


class DiscreteGenerativeModel(GenerativeModel):
    """Discrete state-space generative model using categorical distributions.


    Components:
    - A: Observation model matrix [num_obs x num_states]
    - B: Transition model tensor [num_states x num_states x num_actions]
    - C: Preference matrix [num_obs x time_horizon]
    - D: Initial state prior [num_states]
    """

    def __init__(self, dimensions: ModelDimensions, parameters: ModelParameters) -> None:
        """Initialize discrete generative model"""
        super().__init__(dimensions, parameters)
        # Initialize model components
        self.A = nn.Parameter(self._initialize_A(), requires_grad=False)
        self.B = nn.Parameter(self._initialize_B(), requires_grad=False)
        self.C = nn.Parameter(self._initialize_C(), requires_grad=False)
        self.D = nn.Parameter(self._initialize_D(), requires_grad=False)
        # Move to device
        self.to_device(parameters.use_gpu)
        logger.info(
            f"Initialized discrete generative model with "
            f"{dimensions.num_states} states, "
            f"{dimensions.num_observations} observations, "
            f"{dimensions.num_actions} actions"
        )

    def _initialize_A(self) -> torch.Tensor:
        """Initialize observation model with random categorical distributions"""
        if self.params.use_sparse:
            # For sparse initialization, assume mostly diagonal
            A = torch.eye(
                self.dims.num_observations,
                self.dims.num_states,
                dtype=self.params.dtype,
            )
            # Add small random noise
            A += 0.1 * torch.rand_like(A)
        else:
            # Random initialization with Dirichlet
            A = torch.zeros(
                self.dims.num_observations,
                self.dims.num_states,
                dtype=self.params.dtype,
            )
            for s in range(self.dims.num_states):
                # Sample from Dirichlet for each state
                alpha = torch.ones(self.dims.num_observations)
                A[:, s] = torch.distributions.Dirichlet(alpha).sample()
        return self._normalize(A, dim=0)

    def _initialize_B(self) -> torch.Tensor:
        """Initialize transition model"""
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
                    self.dims.num_states, self.dims.num_states
                )
            else:
                # Random transitions
                for s in range(self.dims.num_states):
                    alpha = torch.ones(self.dims.num_states)
                    B[:, s, a] = torch.distributions.Dirichlet(alpha).sample()
        return self._normalize(B, dim=0)

    def _initialize_C(self) -> torch.Tensor:
        """Initialize preferences (log probabilities)"""
        # Uniform preferences by default
        C = torch.zeros(self.dims.num_observations, self.dims.time_horizon, dtype=self.params.dtype)
        return C

    def _initialize_D(self) -> torch.Tensor:
        """Initialize uniform prior over states"""
        D = torch.ones(self.dims.num_states, dtype=self.params.dtype) / self.dims.num_states
        return D

    def _normalize(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Normalize tensor along specified dimension"""
        return tensor / (tensor.sum(dim=dim, keepdim=True) + self.params.eps)

    def to_device(self, use_gpu: bool) -> None:
        """Move all components to specified device"""
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.A.data = self.A.data.to(device)
        self.B.data = self.B.data.to(device)
        self.C.data = self.C.data.to(device)
        self.D.data = self.D.data.to(device)

    def observation_model(self, states: torch.Tensor) -> torch.Tensor:
        """Compute observation probabilities given states.


        Args:
            states: State distribution [batch_size x num_states] or
                [num_states]
        Returns:
            observations: Observation probabilities [batch_size x num_obs] or
                [num_obs]
        """
        if states.dim() == 1:
            return self.A @ states
        else:
            return states @ self.A.T

    def transition_model(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute next state probabilities.


        Args:
            states: Current state distribution [batch_size x num_states] or
                [num_states]
            actions: Actions [batch_size] or scalar
        Returns:
            next_states: Next state probabilities
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor([actions], device=self.device)
        elif not actions.is_cuda and self.device.type == "cuda":
            actions = actions.to(self.device)
        if states.dim() == 1:
            # Single state distribution
            if actions.numel() == 1:
                return self.B[:, :, int(actions.item())] @ states
            else:
                raise ValueError("Multiple actions for single state not supported")
        else:
            # Batch processing
            batch_size = states.shape[0]
            next_states = torch.zeros_like(states)
            for i in range(batch_size):
                action = int(actions[i].item() if actions.dim() > 0 else actions.item())
                next_states[i] = self.B[:, :, action] @ states[i]
            return next_states

    def get_preferences(self, timestep: Optional[int] = None) -> torch.Tensor:
        """Get prior preferences (C matrix)"""
        if timestep is None:
            return self.C
        else:
            if timestep >= self.dims.time_horizon:
                # Return last timestep preferences for beyond horizon
                return self.C[:, -1]
            return self.C[:, timestep]

    def get_initial_prior(self) -> torch.Tensor:
        """Get initial state prior"""
        return self.D.clone()

    def set_preferences(
        self,
        preferences: Union[torch.Tensor, np.ndarray],
        timestep: Optional[int] = None,
    ) -> None:
        """Set prior preferences.

        Args:
            preferences: Preference values (log probabilities)
            timestep: Optional specific timestep to set
        """
        if isinstance(preferences, np.ndarray):
            preferences = torch.tensor(preferences, dtype=self.params.dtype, device=self.device)
        if timestep is None:
            if preferences.dim() == 1:
                # Set same preferences for all timesteps
                self.C.data = preferences.unsqueeze(1).repeat(1, self.dims.time_horizon)
            else:
                self.C.data = preferences.to(self.device)
        else:
            self.C[:, timestep] = preferences.to(self.device)

    def update_model(
        self,
        observations: List[torch.Tensor],
        states: List[torch.Tensor],
        actions: List[int],
    ) -> None:
        """Update model parameters from experience using Bayesian learning.


        Args:
            observations: List of observation indices or distributions
            states: List of state distributions
            actions: List of actions taken
        """
        # Accumulate sufficient statistics
        A_counts = torch.zeros_like(self.A)
        B_counts = torch.zeros_like(self.B)
        # Update A matrix statistics
        for obs, state in zip(observations, states):
            if obs.dim() == 0:  # Single observation index
                A_counts[obs, :] += state
            else:  # Observation distribution
                A_counts += torch.outer(obs, state)
        # Update B tensor statistics
        for t in range(len(states) - 1):
            state_curr = states[t]
            state_next = states[t + 1]
            action = actions[t]
            B_counts[:, :, action] += torch.outer(state_next, state_curr)
        # Apply updates with learning rate
        self.A.data = (
            1 - self.params.learning_rate
        ) * self.A.data + self.params.learning_rate * self._normalize(A_counts + 1.0, dim=0)
        for a in range(self.dims.num_actions):
            self.B.data[:, :, a] = (1 - self.params.learning_rate) * self.B.data[
                :, :, a
            ] + self.params.learning_rate * self._normalize(B_counts[:, :, a] + 1.0, dim=0)


class ContinuousGenerativeModel(GenerativeModel, nn.Module):
    """Continuous state-space generative model using Gaussian distributions.


    Uses neural networks to parameterize the observation and transition models.
    """

    def __init__(
        self,
        dimensions: ModelDimensions,
        parameters: ModelParameters,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize continuous generative model"""
        GenerativeModel.__init__(self, dimensions, parameters)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        # Initialize neural network components
        self._build_networks()
        # Initialize preferences and prior
        self.C = nn.Parameter(
            torch.zeros(
                dimensions.num_observations,
                dimensions.time_horizon,
                dtype=parameters.dtype,
            )
        )
        self.D_mean = nn.Parameter(torch.zeros(dimensions.num_states, dtype=parameters.dtype))
        self.D_log_var = nn.Parameter(torch.zeros(dimensions.num_states, dtype=parameters.dtype))
        # Move to device
        self.to(self.device)
        logger.info(
            f"Initialized continuous generative model with "
            f"{dimensions.num_states}D states, "
            f"{dimensions.num_observations}D observations"
        )

    def _build_networks(self) -> None:
        """Build neural network components"""
        # Observation model: p(o|s)
        self.obs_net = nn.Sequential(
            nn.Linear(self.dims.num_states, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.obs_mean = nn.Linear(self.hidden_dim, self.dims.num_observations)
        self.obs_log_var = nn.Parameter(torch.zeros(self.dims.num_observations))
        # Transition model: p(s'|s, a)
        self.trans_net = nn.Sequential(
            nn.Linear(self.dims.num_states + self.dims.num_actions, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.trans_mean = nn.Linear(self.hidden_dim, self.dims.num_states)
        self.trans_log_var = nn.Parameter(torch.zeros(self.dims.num_states))

    def observation_model(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
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
    ) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Compute state transition distribution"""
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() > 1 and actions.shape[-1] == self.dims.num_actions:
            actions_onehot = actions.float()
        elif actions.dim() <= 1:
            actions_onehot = F.one_hot(actions.long(), num_classes=self.dims.num_actions).float()
        else:
            raise ValueError(f"Invalid action shape: {actions.shape}")
        if actions_onehot.dim() == 1:
            actions_onehot = actions_onehot.unsqueeze(0)
        if states.shape[0] != actions_onehot.shape[0]:
            if states.shape[0] == 1:
                states = states.repeat(actions_onehot.shape[0], 1)
            else:
                raise ValueError(
                    f"Batch size mismatch: states ({states.shape[0]}) and "
                    f"actions ({actions_onehot.shape[0]})"
                )
        state_action = torch.cat([states, actions_onehot], dim=-1)
        h = self.trans_net(state_action)
        mean = self.trans_mean(h)
        var = torch.exp(self.trans_log_var)
        return mean, var

    def get_preferences(self, timestep: Optional[int] = None) -> torch.Tensor:
        """Get prior preferences"""
        if timestep is None:
            return self.C
        else:
            if timestep >= self.dims.time_horizon:
                return self.C[:, -1]
            return self.C[:, timestep]

    def get_initial_prior(self) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Get initial state prior parameters"""
        return self.D_mean, torch.exp(self.D_log_var)

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
    """Hierarchical generative model with multiple levels of abstraction.


    Higher levels operate at slower timescales and provide context to lower levels.
    """

    def __init__(
        self,
        dimensions: List[ModelDimensions],
        parameters: ModelParameters,
        level_connections: Optional[Dict[int, List[int]]] = None,
    ) -> None:
        """Initialize hierarchical model.

        Args:
            dimensions: List of dimensions for each level
            parameters: Model parameters
            level_connections: Dict mapping level index to connected lower levels
        """
        # Initialize base level
        super().__init__(dimensions[0], parameters)
        self.num_levels = len(dimensions)
        self.dimensions = dimensions
        self.level_connections = level_connections or {
            i: [i + 1] for i in range(self.num_levels - 1)
        }
        # Initialize higher levels
        self.levels = [self]  # Level 0 is the base model
        for i in range(1, self.num_levels):
            level_model = DiscreteGenerativeModel(dimensions[i], parameters)
            self.levels.append(level_model)
        # Initialize inter-level connections (E matrices)
        self.E_matrices = {}
        for upper_idx, lower_indices in self.level_connections.items():
            for lower_idx in lower_indices:
                if lower_idx < self.num_levels:
                    # E[lower_state, upper_state]: how upper level
                    # influences lower level
                    E = torch.rand(
                        dimensions[lower_idx].num_states,
                        dimensions[upper_idx].num_states,
                        dtype=parameters.dtype,
                    )
                    E = self._normalize(E, dim=0)
                    self.E_matrices[(upper_idx, lower_idx)] = E.to(self.device)
        logger.info(f"Initialized hierarchical model with {self.num_levels} levels")

    def get_level(self, level: int) -> DiscreteGenerativeModel:
        """Get model at specific level"""
        return self.levels[level]

    def hierarchical_observation_model(self, states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute observations for all levels considering hierarchical influence.


        Args:
            states: List of state distributions for each level
        Returns:
            observations: List of observation distributions for each level
        """
        observations = []
        for level in range(self.num_levels):
            # Get base observation from level's own model
            level_obs = self.levels[level].observation_model(states[level])
            # Add influence from higher levels
            for upper_level in range(level):
                if (upper_level, level) in self.E_matrices:
                    # Modulate observation based on higher level state
                    E = self.E_matrices[(upper_level, level)]
                    influence = E @ states[upper_level]
                    # Combine with level observation
                    # (multiplicative modulation)
                    level_obs = level_obs * influence.unsqueeze(0)
                    level_obs = self._normalize(level_obs, dim=-1)
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
            next_state = self.levels[level].transition_model(states[level], actions[level])
            # Add influence from higher levels
            for upper_level in range(level):
                if (upper_level, level) in self.E_matrices:
                    E = self.E_matrices[(upper_level, level)]
                    influence = E @ states[upper_level]
                    # Modulate transition
                    next_state = next_state * influence
                    next_state = self._normalize(next_state, dim=-1)
            next_states.append(next_state)
        return next_states


class FactorizedGenerativeModel(DiscreteGenerativeModel):
    """Factorized generative model where states decompose into independent factors.

    This allows for more efficient inference and learning in high-dimensional spaces.
    """

    def __init__(
        self,
        factor_dimensions: List[int],
        num_observations: int,
        num_actions: int,
        parameters: ModelParameters,
    ) -> None:
        """Initialize factorized model.


        Args:
            factor_dimensions: Number of states for each factor
            num_observations: Total number of observations
            num_actions: Number of actions
            parameters: Model parameters
        """
        self.num_factors = len(factor_dimensions)
        self.factor_dims = factor_dimensions
        # Total states is product of factor dimensions
        total_states = np.prod(factor_dimensions)
        dims = ModelDimensions(
            num_states=total_states,
            num_observations=num_observations,
            num_actions=num_actions,
            num_factors=self.num_factors,
        )
        super().__init__(dims, parameters)
        # Initialize factor-specific components
        self._initialize_factors()
        logger.info(
            f"Initialized factorized model with {self.num_factors} factors: " f"{factor_dimensions}"
        )

    def _initialize_factors(self):
        """Initialize factor-specific transition models"""
        self.factor_B = []
        for _, dim in enumerate(self.factor_dims):
            # Each factor has its own transition model
            B_f = torch.zeros(dim, dim, self.dims.num_actions, dtype=self.params.dtype)
            for a in range(self.dims.num_actions):
                # Initialize with mostly independent transitions
                B_f[:, :, a] = 0.9 * torch.eye(dim) + 0.1 * torch.rand(dim, dim)
                B_f[:, :, a] = self._normalize(B_f[:, :, a], dim=0)
            self.factor_B.append(B_f.to(self.device))

    def factor_to_state_idx(self, factor_indices: List[int]) -> int:
        """Convert factor indices to global state index"""
        idx = 0
        multiplier = 1
        for f in reversed(range(self.num_factors)):
            idx += factor_indices[f] * multiplier
            multiplier *= self.factor_dims[f]
        return idx

    def state_to_factor_idx(self, state_idx: int) -> List[int]:
        """Convert global state index to factor indices"""
        indices = []
        for f in reversed(range(self.num_factors)):
            indices.append(state_idx % self.factor_dims[f])
            state_idx //= self.factor_dims[f]
        return list(reversed(indices))

    def factorized_transition(
        self, factor_states: List[torch.Tensor], action: int
    ) -> List[torch.Tensor]:
        """
        Compute transitions for each factor independently.
        Args:
            factor_states: List of state distributions for each factor
            action: Action to take
        Returns:
            next_factor_states: List of next state distributions
        """
        next_states = []
        for f in range(self.num_factors):
            next_state = self.factor_B[f][:, :, action] @ factor_states[f]
            next_states.append(next_state)
        return next_states


def create_generative_model(model_type: str, **kwargs) -> GenerativeModel:
    """
    Factory function to create generative models.
    Args:
        model_type: Type of model ('discrete', 'continuous', 'hierarchical',
            'factorized')
        **kwargs: Model-specific parameters
    Returns:
        Generative model instance
    """
    if model_type == "discrete":
        dims = kwargs.get("dimensions")
        params = kwargs.get("parameters", ModelParameters())
        return DiscreteGenerativeModel(dims, params)
    elif model_type == "continuous":
        dims = kwargs.get("dimensions")
        params = kwargs.get("parameters", ModelParameters())
        hidden_dim = kwargs.get("hidden_dim", 128)
        return ContinuousGenerativeModel(dims, params, hidden_dim)
    elif model_type == "hierarchical":
        dims_list = kwargs.get("dimensions_list")
        params = kwargs.get("parameters", ModelParameters())
        connections = kwargs.get("level_connections")
        return HierarchicalGenerativeModel(dims_list, params, connections)
    elif model_type == "factorized":
        factor_dims = kwargs.get("factor_dimensions")
        num_obs = kwargs.get("num_observations")
        num_actions = kwargs.get("num_actions")
        params = kwargs.get("parameters", ModelParameters())
        return FactorizedGenerativeModel(factor_dims, num_obs, num_actions, params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Create discrete model
    dims = ModelDimensions(num_states=10, num_observations=8, num_actions=4)
    params = ModelParameters(learning_rate=0.01, use_gpu=False)
    model = DiscreteGenerativeModel(dims, params)
    # Test observation model
    state = torch.ones(10) / 10  # Uniform state
    obs = model.observation_model(state)
    print(f"Observation distribution: {obs}")
    # Test transition model
    next_state = model.transition_model(state, action=0)
    print(f"Next state distribution: {next_state}")
    # Set preferences
    preferences = torch.tensor([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])  # Prefer observation 3
    model.set_preferences(preferences)
    print(f"Preferences set: {model.get_preferences(0)}")

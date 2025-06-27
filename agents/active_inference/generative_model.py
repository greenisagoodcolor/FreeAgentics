"""Generative models for active inference."""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class ModelDimensions:
    """Dimensions for generative models."""

    num_states: int
    num_observations: int
    num_actions: int
    num_modalities: int = 1
    num_factors: int = 1
    time_horizon: int = 1


@dataclass
class ModelParameters:
    """Parameters for generative models."""

    learning_rate: float = 0.01
    precision_init: float = 1.0
    use_sparse: bool = False
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-8
    temperature: float = 1.0


class DiscreteGenerativeModel(nn.Module):
    """Discrete generative model for active inference."""

    def __init__(self, dims: ModelDimensions, params: ModelParameters) -> None:
        """Initialize discrete generative model."""
        super().__init__()
        self.dims = dims
        self.params = params
        # Initialize matrices
        self.A = torch.rand(dims.num_observations, dims.num_states)
        self.A = self.A / self.A.sum(dim=0, keepdim=True)
        self.B = torch.rand(dims.num_states, dims.num_states, dims.num_actions)
        self.B = self.B / self.B.sum(dim=0, keepdim=True)
        self.C = torch.zeros(dims.num_observations, dims.time_horizon)
        self.D = torch.ones(dims.num_states) / dims.num_states

    def observation_model(self, state):
        """Compute observation probabilities."""
        if state.dim() == 1:
            return torch.matmul(self.A, state)
        else:
            return torch.matmul(state, self.A.T)

    def transition_model(self, state, action):
        """Compute state transitions."""
        if state.dim() == 1:
            # Single state case
            if isinstance(action, int):
                action_idx = action
            elif action.dim() == 1 and len(action) == 1:
                action_idx = action.item()
            else:
                action_idx = torch.argmax(action).item()
            return torch.matmul(self.B[:, :, action_idx], state)
        else:
            # Batch processing case
            batch_size = state.shape[0]
            next_states = torch.zeros_like(state)
            # Process each state-action pair
            for i in range(batch_size):
                if action.dim() == 1:
                    # action is a 1D tensor of indices
                    action_idx = action[i].item()
                else:
                    # action is one-hot encoded
                    action_idx = torch.argmax(action[i]).item()
                next_states[i] = (
                    torch.matmul(self.B[:, :, action_idx], state[i]))
            return next_states

    def set_preferences(self, preferences, timestep=None) -> None:
        """Set preference vectors."""
        if timestep is None:
            for t in range(self.dims.time_horizon):
                self.C[:, t] = preferences
        else:
            self.C[:, timestep] = preferences

    def get_preferences(self, timestep):
        """Get preferences for timestep."""
        return self.C[:, timestep]

    def update_model(self, observations, states, actions) -> None:
        """Update model parameters."""
        # Simplified update - just add noise to simulate learning
        self.A += torch.randn_like(self.A) * 0.01
        self.A = self.A / self.A.sum(dim=0, keepdim=True)
        self.B += torch.randn_like(self.B) * 0.01
        self.B = self.B / self.B.sum(dim=0, keepdim=True)


class ContinuousGenerativeModel(nn.Module):
    """Continuous generative model for active inference."""

    def __init__(
        self, dims: ModelDimensions, params: ModelParameters,
            hidden_dim: int = 32
    ) -> None:
        """Initialize continuous generative model."""
        super().__init__()
        self.dims = dims
        self.params = params
        self.hidden_dim = hidden_dim
        # Observation model network
        self.obs_net = nn.Sequential(
            nn.Linear(dims.num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dims.num_observations * 2),  # mean and var
        )
        # Transition model network
        self.trans_net = nn.Sequential(
            nn.Linear(dims.num_states + dims.num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dims.num_states * 2),  # mean and var
        )
        # Preferences and priors
        self.C = torch.zeros(dims.num_observations, dims.time_horizon)
        self.D_mean = torch.zeros(dims.num_states)
        self.D_log_var = torch.zeros(dims.num_states)

    def observation_model(self, state):
        """Compute observation distribution."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        output = self.obs_net(state)
        mean = output[:, : self.dims.num_observations]
        log_var = output[:, self.dims.num_observations :]
        var = torch.exp(log_var)
        if state.shape[0] == 1:
            return mean.squeeze(0), var.squeeze(0)
        return mean, var

    def transition_model(self, state, action):
        """Compute state transition distribution."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if isinstance(action, int):
            action_onehot = torch.zeros(1, self.dims.num_actions)
            action_onehot[0, action] = 1.0
            action = action_onehot
        elif action.dim() == 1 and len(action) == 1:
            action_onehot = torch.zeros(1, self.dims.num_actions)
            action_onehot[0, action.item()] = 1.0
            action = action_onehot
        elif action.dim() == 1:
            action = action.unsqueeze(0)
        elif action.dim() == 2 and action.shape[1] == 1:  # Batch indices
            action_onehot = torch.zeros(action.shape[0], self.dims.num_actions)
            action_onehot.scatter_(1, action, 1)
            action = action_onehot
        # Ensure state and action have same batch size
        if state.shape[0] != action.shape[0]:
            if state.shape[0] == 1:
                state = state.repeat(action.shape[0], 1)
            elif action.shape[0] == 1:
                action = action.repeat(state.shape[0], 1)
            else:
                raise ValueError(
                    f"Batch sizes of state ({state.shape[0]}) and action "
                    f"({action.shape[0]}) are incompatible."
                )
        state_action = torch.cat([state, action], dim=-1)
        output = self.trans_net(state_action)
        mean = output[:, : self.dims.num_states]
        log_var = output[:, self.dims.num_states :]
        var = torch.exp(log_var)
        if state.shape[0] == 1:
            return mean.squeeze(0), var.squeeze(0)
        return mean, var

    def forward(self, states, actions):
        """Forward pass."""
        obs_mean, obs_var = self.observation_model(states)
        next_mean, next_var = self.transition_model(states, actions)
        return {
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "next_mean": next_mean,
            "next_var": next_var,
        }


class HierarchicalGenerativeModel(nn.Module):
    """Hierarchical generative model."""

    def __init__(self, dims_list: List[ModelDimensions], params: ModelParameters) -> None:
        """Initialize hierarchical generative model."""
        super().__init__()
        self.dims_list = dims_list
        self.params = params
        self.num_levels = len(dims_list)
        # Create models for each level
        self.levels = (
            nn.ModuleList([DiscreteGenerativeModel(dims, params) for dims in dims_list]))
        # Inter-level connection matrices
        self.E_matrices = {}
        for i in range(self.num_levels - 1):
            lower_states = dims_list[i].num_actions
            upper_states = dims_list[i].num_states
            self.E_matrices[(i, i + 1)] = torch.rand(lower_states,
                upper_states)
            self.E_matrices[(i, i + 1)] = (
                self.E_matrices[(i, i + 1)] / self.E_matrices[)
                (i, i + 1)
            ].sum(dim=1, keepdim=True)

    def hierarchical_observation_model(self, states):
        """Compute observations for all levels."""
        observations = []
        for _i, (level, state) in enumerate(zip(self.levels, states)):
            obs = level.observation_model(state)
            observations.append(obs)
        return observations

    def hierarchical_transition_model(self, states, actions):
        """Compute transitions for all levels."""
        next_states = []
        for _i, (level, state, action) in enumerate(zip(self.levels, states,
            actions)):
            next_state = level.transition_model(state, action)
            next_states.append(next_state)
        return next_states


class FactorizedGenerativeModel(nn.Module):
    """Factorized generative model."""

    def __init__(
        self,
        factor_dims: List[int],
        num_obs: int,
        num_actions: int,
        params: ModelParameters,
    ) -> None:
        """Initialize factorized generative model."""
        super().__init__()
        self.factor_dims = factor_dims
        self.num_factors = len(factor_dims)
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.params = params
        # Create dimensions object
        total_states = 1
        for dim in factor_dims:
            total_states *= dim
        self.dims = ModelDimensions(
            num_states=total_states,
            num_observations=num_obs,
            num_actions=num_actions,
        )
        # Factor-specific transition models
        self.factor_B = []
        for dim in factor_dims:
            B_factor = torch.rand(dim, dim, num_actions)
            self.factor_B.append(B_factor / B_factor.sum(dim=0, keepdim=True))
        # Observation model
        self.A = torch.rand(num_obs, total_states)
        self.A = self.A / self.A.sum(dim=0, keepdim=True)

    def factor_to_state_idx(self, factor_indices):
        """Convert factor indices to a single state index."""
        state_idx = 0
        multiplier = 1
        for i in range(self.num_factors - 1, -1, -1):
            state_idx += factor_indices[i] * multiplier
            multiplier *= self.factor_dims[i]
        return state_idx

    def state_to_factor_idx(self, state_idx):
        """Convert a single state index to factor indices."""
        factor_indices = []
        remaining = state_idx
        # Calculate multipliers for each factor
        multipliers = []
        for i in range(self.num_factors):
            multiplier = 1
            for j in range(i + 1, self.num_factors):
                multiplier *= self.factor_dims[j]
            multipliers.append(multiplier)
        # Extract indices
        for i in range(self.num_factors):
            factor_idx = remaining // multipliers[i]
            factor_indices.append(factor_idx)
            remaining = remaining % multipliers[i]
        return factor_indices

    def factorized_transition(self, factor_states, action):
        """Compute transitions for each factor."""
        next_factor_states = []
        for i in range(self.num_factors):
            trans_matrix = self.factor_B[i][:, :, action]
            next_factor_state = torch.matmul(trans_matrix, factor_states[i])
            next_factor_states.append(next_factor_state)
        return next_factor_states


def create_generative_model(model_type: str, **kwargs) -> nn.Module:
    """Create a generative model of the specified type."""
    if model_type == "discrete":
        dims = kwargs.get("dims", kwargs.get("dimensions"))
        params = kwargs.get("params", ModelParameters())
        return DiscreteGenerativeModel(dims=dims, params=params)
    elif model_type == "continuous":
        dims = kwargs.get("dims", kwargs.get("dimensions"))
        params = kwargs.get("params", ModelParameters())
        return ContinuousGenerativeModel(
            dims=dims,
            params=params,
            hidden_dim=kwargs.get("hidden_dim", 32),
        )
    elif model_type == "hierarchical":
        dims_list = kwargs.get("dims_list", kwargs.get("dimensions_list"))
        params = kwargs.get("params", ModelParameters())
        return HierarchicalGenerativeModel(dims_list=dims_list, params=params)
    elif model_type == "factorized":
        factor_dims = (
            kwargs.get("factor_dims", kwargs.get("factor_dimensions")))
        num_obs = kwargs.get("num_obs", kwargs.get("num_observations"))
        num_actions = kwargs.get("num_actions", 4)  # default
        params = kwargs.get("params", ModelParameters())
        return FactorizedGenerativeModel(
            factor_dims=factor_dims,
            num_obs=num_obs,
            num_actions=num_actions,
            params=params,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

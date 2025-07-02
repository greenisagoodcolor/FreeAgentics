"""
Module for FreeAgentics Active Inference implementation.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

"""
Hierarchical Inference for Active Inference Engine
This module implements hierarchical active inference with multiple levels
of abstraction and temporal prediction.
"""


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical inference systems"""

    num_levels: int = 3
    level_dims: List[int] = None
    timescales: List[float] = None
    bottom_up_weight: float = 0.3
    top_down_weight: float = 0.2
    lateral_weight: float = 0.5
    use_precision_weighting: bool = True
    prediction_horizon: List[int] = None
    use_gpu: bool = True

    def __post_init__(self):
        if self.level_dims is None:
            self.level_dims = [8, 16, 32][: self.num_levels]
        if self.timescales is None:
            self.timescales = [1.0, 4.0, 16.0][: self.num_levels]
        if self.prediction_horizon is None:
            self.prediction_horizon = [int(ts) for ts in self.timescales]


@dataclass
class HierarchicalState:
    """State representation for hierarchical inference"""

    beliefs: torch.Tensor
    predictions: torch.Tensor
    errors: torch.Tensor
    precision: torch.Tensor
    temporal_buffer: List[torch.Tensor] = None

    def __post_init__(self):
        if self.temporal_buffer is None:
            self.temporal_buffer = []


class HierarchicalLevel(nn.Module):
    """A single level in the hierarchical inference system"""

    def __init__(
        self,
        level_id: int,
        config: HierarchicalConfig,
        generative_model,
        inference_algorithm,
        precision_optimizer,
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.level_id = level_id
        self.config = config
        self.generative_model = generative_model
        self.inference_algorithm = inference_algorithm
        self.precision_optimizer = precision_optimizer
        # Get dimensions from generative model
        self.state_dim = generative_model.dims.num_states
        self.obs_dim = generative_model.dims.num_observations
        # Set timescale for this level
        self.timescale = config.timescales[level_id] if level_id < len(
            config.timescales) else 1.0
        self.prediction_horizon = (
            config.prediction_horizon[level_id] if level_id < len(
                config.prediction_horizon) else 1)
        # Determine input dimension for this level
        if input_dim is None:
            input_dim = self.state_dim if level_id > 0 else self.obs_dim
        self.input_dim = input_dim
        # Input projection for dimension matching
        if self.input_dim != self.state_dim:
            self.input_projection = nn.Linear(self.input_dim, self.state_dim)
        else:
            self.input_projection = nn.Identity()
        # Neural networks for hierarchical connections
        self.bottom_up_net = nn.Linear(self.state_dim, self.state_dim)
        # Top-down network - input size depends on higher level
        higher_dim = (
            config.level_dims[level_id + 1]
            if level_id + 1 < len(config.level_dims)
            else self.state_dim
        )
        self.top_down_net = nn.Linear(higher_dim, self.state_dim)
        # Temporal prediction network
        self.temporal_net = nn.GRUCell(self.state_dim, self.state_dim)
        # State
        self.state = None

    def initialize_state(self, batch_size: int) -> HierarchicalState:
        """Initialize the hierarchical state for this level"""
        device = next(self.parameters()).device
        # Initialize beliefs as uniform distributions
        beliefs = torch.ones(batch_size, self.state_dim, device=device)
        beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True)
        predictions = torch.zeros(batch_size, self.state_dim, device=device)
        errors = torch.zeros(batch_size, self.state_dim, device=device)
        precision = torch.ones(batch_size, self.state_dim, device=device)
        self.state = HierarchicalState(
            beliefs=beliefs,
            predictions=predictions,
            errors=errors,
            precision=precision,
            temporal_buffer=[],
        )
        return self.state

    def compute_prediction_error(
        self, observations: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction error between observations and predictions"""
        # Simple L2 error for now
        error = (observations - predictions).pow(2)
        return error

    def update_beliefs(
        self,
        bottom_up_input: Optional[torch.Tensor] = None,
        top_down_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update beliefs based on bottom-up and top-down inputs"""
        if self.state is None:
            raise ValueError(
                "State not initialized. Call initialize_state first.")
        current_beliefs = self.state.beliefs
        # Process inputs
        if bottom_up_input is not None:
            projected_input = self.input_projection(bottom_up_input)
            bottom_up_processed = torch.sigmoid(
                self.bottom_up_net(projected_input))
        else:
            bottom_up_processed = torch.zeros_like(current_beliefs)
        if top_down_input is not None:
            top_down_processed = torch.sigmoid(
                self.top_down_net(top_down_input))
        else:
            top_down_processed = torch.zeros_like(current_beliefs)
        # Weighted combination
        updated_beliefs = (
            self.config.bottom_up_weight * bottom_up_processed
            + self.config.top_down_weight * top_down_processed
            + self.config.lateral_weight * current_beliefs
        )
        # Normalize to ensure valid probability distribution
        updated_beliefs = torch.softmax(updated_beliefs, dim=-1)
        # Update state
        self.state.beliefs = updated_beliefs
        # Update temporal buffer
        self.state.temporal_buffer.append(updated_beliefs.clone())
        if len(self.state.temporal_buffer) > self.prediction_horizon:
            self.state.temporal_buffer.pop(0)
        return updated_beliefs


class HierarchicalInference(nn.Module):
    """Main hierarchical inference system"""

    def __init__(
        self,
        config: HierarchicalConfig,
        generative_models: list,
        inference_algorithms: list,
        precision_optimizers: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_levels = config.num_levels
        self.timestep = 0
        # Create levels with correct input_dim for each
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            precision_opt = precision_optimizers[i] if precision_optimizers else None
            if i == 0:
                input_dim = generative_models[0].dims.num_observations
            else:
                input_dim = generative_models[i - 1].dims.num_states
            level = HierarchicalLevel(
                level_id=i,
                config=config,
                generative_model=generative_models[i],
                inference_algorithm=inference_algorithms[i],
                precision_optimizer=precision_opt,
                input_dim=input_dim,
            )
            self.levels.append(level)

    def initialize(self, batch_size: int) -> None:
        """Initialize all levels"""
        for level in self.levels:
            level.initialize_state(batch_size)

    def step(self, observations: torch.Tensor) -> List[torch.Tensor]:
        """Perform one step of hierarchical inference"""
        self.timestep += 1
        beliefs = []
        # Bottom-up pass
        current_input = observations
        for i, level in enumerate(self.levels):
            # Get top-down input from higher level
            top_down_input = None
            if i + \
                    1 < len(self.levels) and self.levels[i + 1].state is not None:
                top_down_input = self.levels[i + 1].state.beliefs
            # Update beliefs
            level_beliefs = level.update_beliefs(current_input, top_down_input)
            beliefs.append(level_beliefs)
            # Pass beliefs to next level
            current_input = level_beliefs
        return beliefs

    def get_hierarchical_free_energy(self) -> List[torch.Tensor]:
        """Compute free energy at each level"""
        free_energies = []
        for level in self.levels:
            if level.state is not None:
                # Simple free energy approximation
                beliefs = level.state.beliefs
                entropy = -(beliefs * torch.log(beliefs + 1e-8)).sum(dim=-1)
                free_energy = -entropy  # Simplified
                free_energies.append(free_energy)
            else:
                free_energies.append(torch.tensor(0.0))
        return free_energies

    def get_effective_beliefs(self, target_level: int) -> torch.Tensor:
        """Get effective beliefs at a target level"""
        if target_level >= self.num_levels:
            raise ValueError(
                f"Target level {target_level} exceeds number of levels {
                    self.num_levels}")
        return self.levels[target_level].state.beliefs


class TemporalHierarchicalInference(HierarchicalInference):
    """Hierarchical inference with temporal prediction capabilities"""

    def __init__(
        self,
        config: HierarchicalConfig,
        generative_models: list,
        inference_algorithms: list,
        precision_optimizers: Optional[list] = None,
    ) -> None:
        super().__init__(
            config,
            generative_models,
            inference_algorithms,
            precision_optimizers)
        # Add temporal predictors for each level
        self.temporal_predictors = nn.ModuleList()
        for i in range(self.num_levels):
            state_dim = config.level_dims[i]
            predictor = nn.LSTM(state_dim, state_dim, batch_first=True)
            self.temporal_predictors.append(predictor)

    def predict_future_states(
        self, level_id: int, current_state: torch.Tensor, horizon: int
    ) -> List[torch.Tensor]:
        """Predict future states at a given level"""
        predictor = self.temporal_predictors[level_id]
        predictions = []
        hidden = None
        input_state = current_state.unsqueeze(1)  # Add sequence dimension
        for _ in range(horizon):
            output, hidden = predictor(input_state, hidden)
            predicted_state = torch.softmax(output.squeeze(1), dim=-1)
            predictions.append(predicted_state)
            input_state = predicted_state.unsqueeze(1)
        return predictions

    def hierarchical_planning(
            self, planning_horizon: int) -> List[List[torch.Tensor]]:
        """Perform hierarchical planning across all levels"""
        trajectories = []
        for i, level in enumerate(self.levels):
            if level.state is not None:
                horizon = min(planning_horizon, level.prediction_horizon)
                trajectory = self.predict_future_states(
                    i, level.state.beliefs, horizon)
                trajectories.append(trajectory)
            else:
                trajectories.append([])
        return trajectories

    def coarse_to_fine_inference(
        self, observations: torch.Tensor, iterations: int = 3
    ) -> List[torch.Tensor]:
        """Perform coarse-to-fine inference"""
        # Start with standard hierarchical step
        beliefs = self.step(observations)
        # Refine with multiple iterations
        for _ in range(iterations - 1):
            beliefs = self.step(observations)
        return beliefs


def create_hierarchical_inference(
    inference_type: str, config: HierarchicalConfig, **kwargs
) -> HierarchicalInference:
    """Factory function for creating hierarchical inference systems"""
    generative_models = kwargs.get("generative_models")
    inference_algorithms = kwargs.get("inference_algorithms")
    precision_optimizers = kwargs.get("precision_optimizers")
    if not generative_models:
        raise ValueError(
            "create_hierarchical_inference requires generative_models")
    if not inference_algorithms:
        raise ValueError(
            "create_hierarchical_inference requires inference_algorithms")
    if inference_type == "standard":
        return HierarchicalInference(
            config,
            generative_models,
            inference_algorithms,
            precision_optimizers)
    elif inference_type == "temporal":
        return TemporalHierarchicalInference(
            config,
            generative_models,
            inference_algorithms,
            precision_optimizers)
    else:
        raise ValueError(f"Unknown inference type: {inference_type}")

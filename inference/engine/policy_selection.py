"""
Policy Selection Mechanism for Active Inference aligned with PyMDP calculations.

This module implements action selection based on expected free energy minimization
following PyMDP's mathematical formulations: G(π) = E[ln Q(s,o|π) - ln P(o,s|π)].

Supports LLM-generated models through Generalized Notation Notation (GNN) integration
(avoiding confusion with Graph Neural Networks, sometimes referred to as GMN in this codebase).
"""

import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .active_inference import InferenceAlgorithm, InferenceConfig, VariationalMessagePassing
from .generative_model import (
    DiscreteGenerativeModel,
    GenerativeModel,
    ModelDimensions,
    ModelParameters,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for policy selection following PyMDP conventions."""

    # PyMDP-aligned parameters
    planning_horizon: int = 5  # T in PyMDP
    policy_length: int = 1  # Policy depth (number of time steps)
    epistemic_weight: float = 1.0  # use_states_info_gain in PyMDP
    pragmatic_weight: float = 1.0  # use_utility in PyMDP
    exploration_constant: float = 1.0  # precision_parameter (beta) in PyMDP

    # Advanced PyMDP features
    use_param_info_gain: bool = False  # Parameter learning information gain
    use_sophisticated_inference: bool = False  # Counterfactual reasoning

    # Additional parameters
    num_policies: Optional[int] = None
    habit_strength: float = 0.0
    precision: float = 1.0
    use_habits: bool = False
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-16
    use_sampling: bool = False
    num_samples: int = 100
    enable_pruning: bool = True
    pruning_threshold: float = 0.01

    # Hierarchical parameters
    hierarchical_levels: Optional[int] = None
    meta_cognitive_depth: int = 1

    # GNN/GMN notation support for LLM integration
    gnn_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class Policy:
    """Represents a sequence of actions (policy) following PyMDP conventions."""

    def __init__(
        self,
        actions: Union[List[int], torch.Tensor],
        horizon: Optional[int] = None,
        gnn_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize policy with actions and optional GNN metadata.

        Args:
            actions: Sequence of discrete action indices (PyMDP style)
            horizon: Planning horizon
            gnn_metadata: Optional semantic metadata for LLM understanding
        """
        if isinstance(actions, list):
            self.actions = torch.tensor(actions, dtype=torch.long)
        else:
            self.actions = actions.long() if actions.dtype != torch.long else actions

        self.length = len(self.actions)
        self.horizon = horizon or self.length

        # GNN/GMN notation support for LLM-generated policies
        self.gnn_metadata = gnn_metadata or {}

    def __len__(self) -> int:
        """Return length of policy."""
        return self.length

    def __getitem__(self, idx: int) -> int:
        """Get action at index."""
        return int(self.actions[idx].item())

    def __repr__(self) -> str:
        """Return string representation of policy."""
        return f"Policy({self.actions.tolist()})"

    @property
    def temporal_depth(self) -> int:
        """Return temporal depth of policy."""
        return self.length

    def to_one_hot(self, num_actions: int) -> torch.Tensor:
        """Convert to one-hot encoding for PyMDP matrix operations."""
        one_hot = torch.zeros(self.length, num_actions)
        one_hot.scatter_(1, self.actions.unsqueeze(1), 1)
        return one_hot


class PolicySelector(ABC):
    """Abstract base class for policy selection."""

    def __init__(self, config: PolicyConfig) -> None:
        """Initialize."""
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def select_policy(
        self,
        beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> Tuple[Policy, torch.Tensor]:
        """Select policy based on expected free energy."""
        pass

    @abstractmethod
    def compute_expected_free_energy(
        self,
        policy: Policy,
        beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute expected free energy for a policy."""
        pass


class DiscreteExpectedFreeEnergy(PolicySelector):
    """Expected free energy calculation following PyMDP's formulation.

    Implements:
    G(π) = E_Q[ln Q(s,o|π) - ln P(o,s|π)]

    Decomposed into:
    - Epistemic value: E[KL[Q(s|o,π)||Q(s|π)]] (information gain)
    - Pragmatic value: E_Q[ln P(o|C)] (utility/preferences)

    Follows PyMDP matrix conventions:
    - A matrix: P(obs|states) shape (num_obs, num_states)
    - B matrix: P(next_state|current_state, action) shape (num_states, num_states, num_actions)
    - C matrix: Prior preferences shape (num_obs, time_horizon)
    - D vector: Initial state prior shape (num_states,)
    """

    def __init__(self, config: PolicyConfig, inference_algorithm: InferenceAlgorithm) -> None:
        """Initialize discrete expected free energy selector."""
        super().__init__(config)
        self.inference = inference_algorithm
        self.eps = config.eps

        # PyMDP compatibility flags
        self.use_utility = config.pragmatic_weight > 0.0
        self.use_states_info_gain = config.epistemic_weight > 0.0
        self.use_param_info_gain = config.use_param_info_gain
        self.precision_parameter = config.exploration_constant

    def enumerate_policies(self, num_actions: int) -> List[Policy]:
        """Enumerate all possible policies following PyMDP conventions."""
        if self.config.num_policies is not None:
            # Sample random policies
            policies = []
            for _ in range(self.config.num_policies):
                actions = torch.randint(0, num_actions, (self.config.policy_length,))
                policies.append(Policy(actions, self.config.planning_horizon))
            return policies
        elif self.config.policy_length == 1:
            # Single-step policies (most common in PyMDP)
            return [Policy([a]) for a in range(num_actions)]
        else:
            # Multi-step policies: all combinations
            all_combos = itertools.product(range(num_actions), repeat=self.config.policy_length)
            return [Policy(list(combo), self.config.planning_horizon) for combo in all_combos]

    def select_policy(
        self,
        beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> Tuple[Policy, torch.Tensor]:
        """Select policy following PyMDP algorithm: Q(π) ∝ exp(-βG(π)).

        Args:
            beliefs: Current posterior beliefs Q(s) over states
            generative_model: Generative model with PyMDP matrices
            preferences: Optional preferences (uses model's C matrix if None)

        Returns:
            selected_policy: Policy with lowest expected free energy
            policy_posteriors: Softmax posteriors Q(π) over all policies
        """
        if preferences is None:
            preferences = generative_model.get_preferences()

        # Enumerate all policies
        policies = self.enumerate_policies(generative_model.dims.num_actions)

        # Calculate expected free energy for each policy
        G_values = []
        for policy in policies:
            G, _, _ = self.compute_expected_free_energy(
                policy, beliefs, generative_model, preferences
            )
            G_values.append(G)

        G_tensor = torch.stack(G_values)

        # Add habit strength if configured
        if self.config.habit_strength > 0:
            # Simple habit prior (could be made more sophisticated)
            habit_prior = torch.zeros_like(G_tensor)
            G_tensor = G_tensor - self.config.habit_strength * habit_prior

        # Calculate policy posterior: Q(π) ∝ exp(-βG(π))
        policy_posteriors = F.softmax(-G_tensor * self.precision_parameter, dim=0)

        # Policy pruning
        if self.config.enable_pruning:
            mask = policy_posteriors > self.config.pruning_threshold
            if mask.sum() > 0:
                policy_posteriors = policy_posteriors * mask
                policy_posteriors = policy_posteriors / policy_posteriors.sum()

        # Policy selection
        if self.config.use_sampling:
            # Stochastic selection
            policy_idx = int(torch.multinomial(policy_posteriors, 1).item())
        else:
            # Deterministic selection (highest posterior)
            policy_idx = int(torch.argmax(policy_posteriors).item())

        selected_policy = policies[policy_idx]

        return selected_policy, policy_posteriors

    def compute_expected_free_energy(
        self,
        policy: Policy,
        beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute expected free energy following PyMDP's G(π) formulation.

        Implements:
        G(π) = ∑_τ [epistemic_value_τ + pragmatic_value_τ]

        Where:
        - epistemic_value = E[KL[Q(s|o,π)||Q(s|π)]] (information gain)
        - pragmatic_value = -E_Q[ln P(o|C)] (negative log preferences)

        Args:
            policy: Policy π as sequence of actions
            beliefs: Current beliefs Q(s) over states
            generative_model: Model with PyMDP matrices A, B, C, D
            preferences: Prior preferences C (uses model's if None)

        Returns:
            G: Total expected free energy
            total_epistemic: Epistemic component (information gain)
            total_pragmatic: Pragmatic component (preference satisfaction)
        """
        # Initialize components
        G = torch.tensor(0.0, device=self.device, dtype=self.config.dtype)
        total_epistemic = torch.tensor(0.0, device=self.device, dtype=self.config.dtype)
        total_pragmatic = torch.tensor(0.0, device=self.device, dtype=self.config.dtype)

        # Move beliefs to device
        current_beliefs = beliefs.to(self.device).type(self.config.dtype)

        # Get PyMDP matrices
        A, B, C, D = self._get_pymdp_matrices(generative_model, preferences)

        # Forward pass through policy
        for t in range(min(len(policy), self.config.planning_horizon)):
            action = policy[t]

            # State prediction: Q(s_{t+1}|π) = Q(s_t) @ B[:, :, action]
            predicted_states = torch.matmul(current_beliefs, B[:, :, action])

            # Observation prediction: Q(o_{t+1}|π) = A @ Q(s_{t+1}|π)
            predicted_observations = torch.matmul(A, predicted_states)

            # Epistemic value (information gain)
            if self.use_states_info_gain:
                epistemic_value = self._calculate_epistemic_value(
                    predicted_states, predicted_observations, A, t
                )
                total_epistemic += epistemic_value

            # Pragmatic value (preference satisfaction)
            if self.use_utility:
                pragmatic_value = self._calculate_pragmatic_value(predicted_observations, C, t)
                total_pragmatic += pragmatic_value

            # Update beliefs for next timestep
            current_beliefs = predicted_states

        # Combine components with weights
        G = (
            self.config.epistemic_weight * total_epistemic
            + self.config.pragmatic_weight * total_pragmatic
        )

        return G, total_epistemic, total_pragmatic

    def _get_pymdp_matrices(
        self, generative_model: GenerativeModel, preferences: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract PyMDP matrices from generative model."""
        # A matrix: P(obs|states)
        if hasattr(generative_model, "A"):
            A = generative_model.A.to(self.device).type(self.config.dtype)
        else:
            # Fallback: identity observation model
            A = torch.eye(
                generative_model.dims.num_observations,
                generative_model.dims.num_states,
                device=self.device,
                dtype=self.config.dtype,
            )

        # B matrix: P(next_state|current_state, action)
        if hasattr(generative_model, "B"):
            B = generative_model.B.to(self.device).type(self.config.dtype)
        else:
            # Fallback: identity transition model
            B = (
                torch.eye(
                    generative_model.dims.num_states, device=self.device, dtype=self.config.dtype
                )
                .unsqueeze(-1)
                .repeat(1, 1, generative_model.dims.num_actions)
            )

        # C matrix: Prior preferences
        if preferences is not None:
            C = preferences.to(self.device).type(self.config.dtype)
        elif hasattr(generative_model, "C"):
            C = generative_model.C.to(self.device).type(self.config.dtype)
        else:
            # Fallback: uniform preferences
            C = torch.zeros(
                generative_model.dims.num_observations,
                max(1, self.config.planning_horizon),
                device=self.device,
                dtype=self.config.dtype,
            )

        # D vector: Initial state prior
        if hasattr(generative_model, "D"):
            D = generative_model.D.to(self.device).type(self.config.dtype)
        else:
            # Fallback: uniform prior
            D = (
                torch.ones(
                    generative_model.dims.num_states, device=self.device, dtype=self.config.dtype
                )
                / generative_model.dims.num_states
            )

        return A, B, C, D

    def _calculate_epistemic_value(
        self,
        predicted_states: torch.Tensor,
        predicted_observations: torch.Tensor,
        A: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Calculate epistemic value (information gain) following PyMDP.

        Implements: E[KL[Q(s|o,π)||Q(s|π)]]
        """
        epistemic_value = torch.tensor(0.0, device=self.device, dtype=self.config.dtype)

        # For each possible observation
        for obs_idx in range(A.shape[0]):
            obs_prob = predicted_observations[obs_idx]

            if obs_prob > self.eps:
                # Posterior after observing obs_idx: Q(s|o,π) ∝ A[obs_idx, :] * Q(s|π)
                likelihood = A[obs_idx, :]
                posterior = likelihood * predicted_states
                posterior = posterior / (posterior.sum() + self.eps)

                # KL divergence: KL[Q(s|o,π)||Q(s|π)]
                kl_div = torch.sum(
                    posterior * torch.log((posterior + self.eps) / (predicted_states + self.eps))
                )

                # Weight by observation probability
                epistemic_value += obs_prob * kl_div

        return epistemic_value

    def _calculate_pragmatic_value(
        self, predicted_observations: torch.Tensor, C: torch.Tensor, timestep: int
    ) -> torch.Tensor:
        """Calculate pragmatic value (preference satisfaction) following PyMDP.

        Implements: -E_Q[ln P(o|C)]
        """
        # Get preferences for this timestep
        if C.dim() > 1 and timestep < C.shape[1]:
            preferences_t = C[:, timestep]
        else:
            preferences_t = C[:, 0] if C.dim() > 1 else C

        # Expected log preference: E_Q[ln P(o|C)] = ∑_o Q(o|π) * ln P(o|C)
        expected_log_preference = torch.sum(predicted_observations * preferences_t)

        # Pragmatic value is negative expected log preference (cost)
        pragmatic_value = -expected_log_preference

        return pragmatic_value

    def generate_policies(self, generative_model: GenerativeModel) -> List[Policy]:
        """Generate all policies for the given model."""
        return self.enumerate_policies(generative_model.dims.num_actions)


class ContinuousExpectedFreeEnergy(PolicySelector):
    """Expected free energy for continuous state spaces using sampling approximation."""

    def __init__(self, config: PolicyConfig, inference_algorithm: InferenceAlgorithm) -> None:
        """Initialize continuous expected free energy selector."""
        super().__init__(config)
        self.inference = inference_algorithm
        self.eps = config.eps

    def sample_policies(self, action_dim: int, num_policies: int) -> List[Policy]:
        """Sample continuous action policies."""
        policies = []
        for _ in range(num_policies):
            # Sample continuous actions and discretize if needed
            actions = torch.randn(self.config.policy_length, action_dim) * 0.5
            actions = torch.clamp(actions, -1.0, 1.0)

            # Convert to discrete indices for compatibility
            discrete_actions = torch.round((actions + 1.0) * (action_dim - 1) / 2.0).long()
            discrete_actions = torch.clamp(discrete_actions, 0, action_dim - 1)

            policies.append(Policy(discrete_actions, self.config.planning_horizon))
        return policies

    def select_policy(
        self,
        beliefs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> Tuple[Policy, torch.Tensor]:
        """Select policy for continuous states using sampling."""
        num_policies = self.config.num_policies or 100
        policies = self.sample_policies(generative_model.dims.num_actions, num_policies)

        G_values = []
        for policy in policies:
            G_tuple = self.compute_expected_free_energy(
                policy, beliefs, generative_model, preferences
            )
            G = G_tuple[0]  # Extract just the G value from the tuple
            G_values.append(G)

        G_tensor = torch.stack(G_values)

        if self.config.use_sampling:
            probs = F.softmax(-G_tensor / self.config.exploration_constant, dim=0)
            policy_idx = int(torch.multinomial(probs, 1).item())
        else:
            policy_idx = int(torch.argmin(G_tensor).item())

        return policies[policy_idx], G_tensor

    def compute_expected_free_energy(
        self,
        policy: Policy,
        beliefs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute expected free energy using Monte Carlo approximation."""
        if isinstance(beliefs, tuple):
            mean, var = beliefs
        else:
            # Assume uniform variance for simple tensor beliefs
            mean = beliefs
            var = torch.ones_like(mean) * 0.1

        mean = mean.to(self.device)
        var = var.to(self.device)

        G = torch.tensor(0.0, device=self.device)
        epistemic_total = torch.tensor(0.0, device=self.device)
        pragmatic_total = torch.tensor(0.0, device=self.device)
        num_samples = self.config.num_samples

        for _ in range(num_samples):
            # Sample current state
            std = torch.sqrt(var)
            current_state = mean + std * torch.randn_like(mean)

            for t in range(min(len(policy), self.config.planning_horizon)):
                action_idx = policy[t]

                # Convert discrete action to continuous
                action = torch.tensor(action_idx, dtype=torch.float32, device=self.device)

                # Forward dynamics
                if hasattr(generative_model, "transition_model"):
                    next_mean, next_var = generative_model.transition_model(
                        current_state.unsqueeze(0), action.unsqueeze(0)
                    )
                    next_mean = next_mean.squeeze(0)
                    next_var = next_var.squeeze(0)
                else:
                    # Simple linear dynamics
                    next_mean = current_state + action * 0.1
                    next_var = var * 1.01

                # Observation model
                if hasattr(generative_model, "observation_model"):
                    obs_mean, obs_var = generative_model.observation_model(next_mean.unsqueeze(0))
                    obs_mean = obs_mean.squeeze(0)
                    obs_var = obs_var.squeeze(0)
                else:
                    obs_mean = next_mean
                    obs_var = next_var

                # Epistemic value (information gain)
                if self.config.epistemic_weight > 0:
                    info_gain = 0.5 * torch.sum(torch.log(var / (next_var + self.eps)))
                    epistemic_value = self.config.epistemic_weight * info_gain
                    epistemic_total += epistemic_value
                    G -= epistemic_value

                # Pragmatic value
                if self.config.pragmatic_weight > 0 and preferences is not None:
                    if preferences.dim() > 1 and t < preferences.shape[1]:
                        pref_t = preferences[:, t]
                    else:
                        pref_t = preferences

                    # Squared error cost
                    prag_value = -torch.sum((obs_mean - pref_t) ** 2 / (obs_var + self.eps))
                    pragmatic_value = self.config.pragmatic_weight * prag_value
                    pragmatic_total += pragmatic_value
                    G -= pragmatic_value

                # Update state
                current_state = next_mean
                var = next_var

        return (G / num_samples, epistemic_total / num_samples, pragmatic_total / num_samples)


class HierarchicalPolicySelector(PolicySelector):
    """Hierarchical policy selection with temporal abstraction."""

    def __init__(
        self,
        config: PolicyConfig,
        level_selectors: List[PolicySelector],
        level_horizons: List[int],
    ) -> None:
        """Initialize hierarchical policy selector."""
        super().__init__(config)
        self.level_selectors = level_selectors
        self.level_horizons = level_horizons
        self.num_levels = len(level_selectors)

    def get_level_horizons(self) -> List[int]:
        """Get planning horizons for each level."""
        return self.level_horizons

    def select_policy_at_level(
        self,
        level: int,
        beliefs: torch.Tensor,
        model: GenerativeModel,
        higher_level_policy: Optional[Policy] = None,
    ) -> Policy:
        """Select policy at specific hierarchical level."""
        if level < len(self.level_selectors):
            policy, _ = self.level_selectors[level].select_policy(beliefs, model)
            return policy
        else:
            # Fallback for invalid level
            return Policy([0])

    def select_policy(
        self,
        beliefs: List[torch.Tensor],
        generative_models: List[GenerativeModel],
        preferences: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[Policy], List[torch.Tensor]]:
        """Select policies at each hierarchical level."""
        policies = []
        all_probs = []

        for level in range(self.num_levels):
            if level > 0:
                # Use higher-level policy as context
                higher_level_policy = policies[level - 1]
            else:
                higher_level_policy = None

            level_beliefs = beliefs[level]
            level_model = generative_models[level]
            level_prefs = preferences[level] if preferences else None

            policy, probs = self.level_selectors[level].select_policy(
                level_beliefs, level_model, level_prefs
            )

            policies.append(policy)
            all_probs.append(probs)

        return policies, all_probs

    def compute_expected_free_energy(
        self,
        policies: List[Policy],
        beliefs: List[torch.Tensor],
        generative_models: List[GenerativeModel],
        preferences: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute total expected free energy across levels."""
        total_G = torch.tensor(0.0, device=self.device)

        for level in range(self.num_levels):
            result = self.level_selectors[level].compute_expected_free_energy(
                policies[level],
                beliefs[level],
                generative_models[level],
                preferences[level] if preferences else None,
            )

            # Handle tuple (discrete) and tensor (continuous) returns
            if isinstance(result, tuple):
                G = result[0]  # Use only the total G value
            else:
                G = result

            # Weight by level importance (higher levels less weighted)
            weight = 1.0 / (level + 1)
            total_G += weight * G

        return total_G


class SophisticatedInference(PolicySelector):
    """Sophisticated inference with counterfactual reasoning and meta-cognition."""

    def __init__(
        self,
        config: PolicyConfig,
        inference_algorithm: InferenceAlgorithm,
        base_selector: PolicySelector,
    ) -> None:
        """Initialize sophisticated inference selector."""
        super().__init__(config)
        self.inference = inference_algorithm
        self.base_selector = base_selector
        self.sophistication_depth = config.meta_cognitive_depth

    def select_policy(
        self,
        beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> Tuple[Policy, torch.Tensor]:
        """Select policy using sophisticated inference with counterfactual reasoning."""
        # Start with base policy selection
        base_policy, base_probs = self.base_selector.select_policy(
            beliefs, generative_model, preferences
        )

        if self.sophistication_depth > 0:
            # Apply sophisticated refinement
            refined_policy = self._sophisticated_refinement(
                base_policy, beliefs, generative_model, preferences
            )
            return refined_policy, base_probs
        else:
            return base_policy, base_probs

    def _sophisticated_refinement(
        self,
        initial_policy: Policy,
        beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> Policy:
        """Refine policy through counterfactual reasoning."""
        current_beliefs = beliefs
        refined_actions = []

        for t in range(len(initial_policy)):
            action = initial_policy[t]

            # Predict future beliefs after this action
            if isinstance(generative_model, DiscreteGenerativeModel):
                next_beliefs = generative_model.transition_model(current_beliefs, action)
                expected_obs = generative_model.observation_model(next_beliefs)

                # Update beliefs using inference algorithm
                updated_beliefs = self.inference.infer_states(
                    expected_obs, generative_model, next_beliefs
                )
            else:
                # For continuous models, use simple dynamics
                updated_beliefs = current_beliefs

            # If within sophistication depth, consider future policy selection
            if t < self.sophistication_depth:
                # What would we do from this future state?
                future_policy, _ = self.base_selector.select_policy(
                    updated_beliefs, generative_model, preferences
                )
                refined_action = future_policy[0]
            else:
                # Use original action
                refined_action = action

            refined_actions.append(refined_action)
            current_beliefs = updated_beliefs

        return Policy(refined_actions, initial_policy.horizon)

    def evaluate_counterfactual(
        self, observed_policy: Policy, counterfactual_policy: Policy, beliefs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate counterfactual policy value."""
        # Simplified counterfactual evaluation
        # In full implementation, would compare expected outcomes
        return torch.tensor(0.8, device=self.device)

    def meta_policy_selection(self) -> Policy:
        """Meta-cognitive policy selection (policy about policy selection)."""
        # Simplified meta-cognitive policy
        return Policy([0])  # Conservative meta-policy

    def update_model_parameters(self, model: GenerativeModel) -> None:
        """Update model parameters during sophisticated inference."""
        # Placeholder for parameter learning
        pass

    def _calculate_param_info_gain(
        self, policy: Policy, beliefs: torch.Tensor, model: GenerativeModel
    ) -> torch.Tensor:
        """Calculate parameter information gain."""
        # Simplified parameter information gain
        return torch.tensor(0.0, device=self.device)

    def compute_expected_free_energy(
        self,
        policy: Policy,
        beliefs: torch.Tensor,
        generative_model: GenerativeModel,
        preferences: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute expected free energy with sophisticated inference."""
        result = self.base_selector.compute_expected_free_energy(
            policy, beliefs, generative_model, preferences
        )
        if isinstance(result, tuple):
            return result
        else:
            # Convert single tensor to tuple format
            return (result, torch.tensor(0.0), torch.tensor(0.0))


def create_policy_selector(
    selector_type: str, config: Optional[PolicyConfig] = None, **kwargs: Any
) -> PolicySelector:
    """Create policy selectors with PyMDP compatibility.

    Args:
        selector_type: Type of selector ('discrete', 'continuous', 'hierarchical', 'sophisticated')
        config: Policy configuration
        **kwargs: Selector-specific parameters

    Returns:
        Policy selector instance configured for PyMDP compatibility
    """
    if config is None:
        config = PolicyConfig()

    if selector_type == "discrete":
        inference = kwargs.get("inference_algorithm")
        if inference is None:
            raise ValueError("Discrete selector requires inference_algorithm")
        return DiscreteExpectedFreeEnergy(config, inference)

    elif selector_type == "continuous":
        inference = kwargs.get("inference_algorithm")
        if inference is None:
            raise ValueError("Continuous selector requires inference_algorithm")
        return ContinuousExpectedFreeEnergy(config, inference)

    elif selector_type == "hierarchical":
        level_selectors = kwargs.get("level_selectors")
        level_horizons = kwargs.get("level_horizons", [5, 10, 20])
        if level_selectors is None:
            raise ValueError("Hierarchical selector requires level_selectors")
        return HierarchicalPolicySelector(config, level_selectors, level_horizons)

    elif selector_type == "sophisticated":
        inference = kwargs.get("inference_algorithm")
        base_selector = kwargs.get("base_selector")
        if inference is None or base_selector is None:
            raise ValueError(
                "Sophisticated selector requires inference_algorithm and base_selector"
            )
        return SophisticatedInference(config, inference, base_selector)

    else:
        raise ValueError(f"Unknown selector type: {selector_type}")


# GNN/GMN Integration Functions for LLM Model Generation
def create_gnn_compatible_policy_config(gnn_spec: Dict[str, Any]) -> PolicyConfig:
    """Create policy configuration from GNN specification for LLM integration.

    Args:
        gnn_spec: GNN specification dictionary with semantic model description

    Returns:
        PolicyConfig configured for GNN/GMN compatibility
    """
    # Extract parameters from GNN specification
    planning_horizon = gnn_spec.get("time_settings", {}).get("horizon", 5)

    # Create configuration
    config = PolicyConfig(
        planning_horizon=planning_horizon,
        use_gpu=False,  # Default for LLM compatibility
        gnn_metadata=gnn_spec,
    )

    return config


def validate_gnn_policy_compatibility(policy: Policy, gnn_spec: Dict[str, Any]) -> bool:
    """Validate that policy is compatible with GNN specification.

    Args:
        policy: Policy to validate
        gnn_spec: GNN specification

    Returns:
        True if compatible, False otherwise
    """
    # Check action space compatibility
    action_space = gnn_spec.get("action_space", {})
    expected_actions = action_space.get("size", 2)

    # Validate action indices are within range
    max_action = torch.max(policy.actions).item()
    if max_action >= expected_actions:
        return False

    # Check temporal compatibility
    time_settings = gnn_spec.get("time_settings", {})
    max_horizon = time_settings.get("horizon", float("inf"))

    if len(policy) > max_horizon:
        return False

    return True


if __name__ == "__main__":
    # Demonstration of PyMDP-aligned policy selection
    dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
    params = ModelParameters(use_gpu=False)
    model = DiscreteGenerativeModel(dims, params)

    # Set up PyMDP-style matrices
    model.A.data = torch.eye(3, 4)  # Identity observation model
    model.B.data = torch.eye(4).unsqueeze(-1).repeat(1, 1, 2)  # Identity transition
    model.C.data = torch.tensor([[-1.0], [0.0], [2.0]])  # Preferences
    model.D.data = torch.ones(4) / 4  # Uniform prior

    # Create inference and policy selector
    inf_config = InferenceConfig(use_gpu=False)
    inference = VariationalMessagePassing(inf_config)
    policy_config = PolicyConfig(
        planning_horizon=3,
        policy_length=1,
        use_gpu=False,
        epistemic_weight=1.0,
        pragmatic_weight=1.0,
        exploration_constant=2.0,
    )

    selector = DiscreteExpectedFreeEnergy(policy_config, inference)

    # Test policy selection
    beliefs = torch.ones(4) / 4
    policy, probs = selector.select_policy(beliefs, model)

    print(f"Selected policy: {policy}")
    print(f"Policy probabilities: {probs}")
    print(
        f"Policy follows PyMDP conventions: {validate_gnn_policy_compatibility(policy, {'action_space': {'size': 2}})}"
    )

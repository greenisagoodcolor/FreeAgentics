"""Precision optimization for active inference"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


@dataclass
class PrecisionConfig:
    """Configuration for precision optimization"""

    learning_rate: float = 0.01
    meta_learning_rate: float = 0.001
    momentum: float = 0.9
    gradient_clip: float = 1.0
    min_precision: float = 0.1
    max_precision: float = 100.0
    init_precision: float = 1.0
    volatility_window: int = 10
    volatility_threshold: float = 0.5
    adaptation_rate: float = 0.1
    num_levels: int = 1
    level_coupling: float = 0.5
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32


class GradientPrecisionOptimizer:
    """Gradient-based precision optimization"""

    def __init__(self, config: PrecisionConfig, num_modalities: int) -> None:
        """Initialize"""
        self.config = config
        self.num_modalities = num_modalities
        self.log_precision = torch.log(torch.ones(num_modalities) * config.init_precision)
        self.error_history = []

    def optimize_precision(self, errors, context=None):
        """Optimize precision based on errors"""
        # Ensure errors are 1D for this optimizer
        if errors.dim() == 2:
            errors = errors.squeeze(0)

        # Simple optimization - adjust precision based on error magnitude
        if errors.dim() == 1:
            error_magnitude = errors.abs()  # Keep individual values for 1D
        else:
            error_magnitude = errors.abs().mean(dim=0)  # Average batch for 2D

        # Update log precision
        learning_rate = self.config.learning_rate
        self.log_precision += learning_rate * (error_magnitude - 1.0)

        # Clamp to bounds
        precision = torch.exp(self.log_precision)
        precision = torch.clamp(precision, self.config.min_precision, self.config.max_precision)
        self.log_precision = torch.log(precision)

        # Store error history
        self.error_history.append(errors.detach().clone())
        if len(self.error_history) > self.config.volatility_window:
            self.error_history.pop(0)

        return precision

    def estimate_volatility(self):
        """Estimate volatility from error history"""
        if len(self.error_history) < 2:
            return torch.zeros(self.num_modalities)

        errors = torch.stack(self.error_history)
        return errors.std(dim=0)

    def adapt_to_volatility(self):
        """Adapt precision based on volatility"""
        volatility = self.estimate_volatility()
        adaptation = self.config.adaptation_rate * volatility
        self.log_precision += adaptation


class HierarchicalPrecisionOptimizer:
    """Hierarchical precision optimization"""

    def __init__(self, config: PrecisionConfig, level_dims: List[int]) -> None:
        self.config = config
        self.level_dims = level_dims
        self.num_levels = len(level_dims)

        # Initialize precision for each level
        self.level_precisions = []
        for dim in level_dims:
            precision = torch.ones(dim) * config.init_precision
            self.level_precisions.append(torch.log(precision))

        # Coupling weights between levels
        self.coupling_weights = []
        for i in range(self.num_levels - 1):
            weight = torch.ones(1) * config.level_coupling
            self.coupling_weights.append(weight)

    def optimize_precision(self, errors_list, context=None):
        """Optimize precision across all levels"""
        precisions = []

        for i, (errors, log_precision) in enumerate(zip(errors_list, self.level_precisions)):
            if errors.dim() == 1:
                errors = errors.unsqueeze(0)

            # Update precision for this level
            error_magnitude = errors.abs().mean(dim=0)
            learning_rate = self.config.learning_rate
            log_precision += learning_rate * (error_magnitude - 1.0)

            # Apply coupling from other levels
            if i > 0:
                coupling_effect = self.coupling_weights[i - 1] * self.level_precisions[i - 1].mean()
                log_precision += coupling_effect * 0.1

            # Clamp to bounds
            precision = torch.exp(log_precision)
            precision = torch.clamp(precision, self.config.min_precision, self.config.max_precision)
            self.level_precisions[i] = torch.log(precision)
            precisions.append(precision)

        return precisions

    def estimate_volatility(self, error_history_list):
        """Estimate volatility for each level"""
        volatilities = []
        for i, error_history in enumerate(error_history_list):
            if len(error_history) < 2:
                volatilities.append(torch.zeros(self.level_dims[i]))
            else:
                errors = pad_sequence([e[0] for e in error_history], batch_first=True)
                log_abs_diff = torch.log(torch.abs(errors[1:] - errors[:-1]) + 1e-6)
                volatility = torch.exp(torch.mean(log_abs_diff, dim=0))
                # Ensure volatility has the correct shape (squeeze extra
                # dimensions)
                volatility = volatility.squeeze()
                # Handle scalar case - if all dimensions collapsed to scalar, expand to
                # expected shape
                if volatility.dim() == 0:
                    volatility = volatility.expand(self.level_dims[i])
                volatilities.append(volatility)
        return volatilities


class MetaLearningPrecisionOptimizer:
    """Meta-learning precision optimization"""

    def __init__(
        self,
        config: PrecisionConfig,
        input_dim: int,
        hidden_dim: int,
        num_modalities: int,
    ) -> None:
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        # Meta-learning network
        # extract_features produces: num_modalities * 3 error features
        # (mean + std + max for each modality) + context features
        # Use a larger input dimension to accommodate variable context size
        # error features + max context size
        max_input_dim = num_modalities * 3 + input_dim
        self.meta_network = nn.Sequential(
            nn.Linear(max_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
        )

        # Base precision
        self.base_precision = torch.ones(num_modalities) * config.init_precision

        # Context buffer
        self.context_buffer = []
        self.max_context_size = 100

    def extract_features(self, errors, context=None):
        """Extract features for meta-learning"""
        if errors.dim() == 1:
            errors = errors.unsqueeze(0)

        # Basic statistics
        error_mean = errors.mean(dim=0)
        error_std = errors.std(dim=0)
        error_max = errors.max(dim=0)[0]

        features = torch.cat([error_mean, error_std, error_max])

        if context is not None:
            features = torch.cat([features, context])

        # Pad features to match network input dimension
        expected_dim = self.meta_network[0].in_features
        if features.shape[0] < expected_dim:
            padding = torch.zeros(expected_dim - features.shape[0])
            features = torch.cat([features, padding])
        elif features.shape[0] > expected_dim:
            features = features[:expected_dim]

        return features

    def optimize_precision(self, errors, context=None):
        """Optimize precision using meta-learning"""
        # Ensure errors are 2D for consistency
        if errors.dim() == 1:
            errors = errors.unsqueeze(0)

        features = self.extract_features(errors, context)

        # Get precision adjustment from meta-network
        adjustment = self.meta_network(features)

        # Clamp adjustment to prevent overflow/underflow in exp()
        adjustment = torch.clamp(adjustment, -10.0, 10.0)

        # Apply adjustment to base precision
        precision = self.base_precision * torch.exp(adjustment)

        # Handle NaN/inf values
        precision = torch.where(torch.isfinite(precision), precision, self.base_precision)

        precision = torch.clamp(precision, self.config.min_precision, self.config.max_precision)

        # Update context buffer (store the 2D version)
        self.context_buffer.append((errors.detach().clone(), context))
        if len(self.context_buffer) > self.max_context_size:
            self.context_buffer.pop(0)

        return precision

    def meta_update(self, num_steps: int = 10):
        """Perform meta-learning update"""
        if len(self.context_buffer) < num_steps:
            return

        optimizer = torch.optim.Adam(
            self.meta_network.parameters(), lr=self.config.meta_learning_rate
        )

        for _ in range(num_steps):
            # Sample from context buffer
            batch_idx = torch.randint(
                0, len(self.context_buffer), (min(32, len(self.context_buffer)),)
            )

            total_loss = 0
            for idx in batch_idx:
                errors, context = self.context_buffer[idx]
                features = self.extract_features(errors, context)

                # Compute loss (simplified)
                adjustment = self.meta_network(features)
                loss = adjustment.pow(2).mean()  # Regularization loss
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    def estimate_volatility(self):
        """Estimate volatility from context buffer"""
        if len(self.context_buffer) < 2:
            return torch.zeros(self.num_modalities)

        errors = torch.stack([item[0].mean(dim=0) for item in self.context_buffer])
        return errors.std(dim=0)


class AdaptivePrecisionController:
    """Adaptive precision controller that switches between strategies"""

    def __init__(self, config: PrecisionConfig, num_modalities: int, context_dim: int = 0) -> None:
        self.config = config
        self.num_modalities = num_modalities
        self.context_dim = context_dim

        # Initialize different optimizers
        self.gradient_optimizer = GradientPrecisionOptimizer(config, num_modalities)
        self.meta_optimizer = MetaLearningPrecisionOptimizer(
            config, context_dim + num_modalities * 3, 64, num_modalities
        )

        # Strategy selection
        self.strategy = "gradient"  # "gradient", "meta", "hybrid"
        self.performance_history = {"gradient": [], "meta": [], "hybrid": []}
        self.strategy_performance = {"gradient": 0.0, "meta": 0.0, "hybrid": 0.0}

    def optimize(self, errors, context=None):
        """Optimize precision using selected strategy"""
        if self.strategy == "gradient":
            precision = self.gradient_optimizer.optimize_precision(errors, context)
        elif self.strategy == "meta":
            precision = self.meta_optimizer.optimize_precision(errors, context)
        elif self.strategy == "hybrid":
            # Use both and average
            grad_precision = self.gradient_optimizer.optimize_precision(errors, context)
            meta_precision = self.meta_optimizer.optimize_precision(errors, context)
            precision = (grad_precision + meta_precision) / 2
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Track performance (lower error = better)
        error_metric = errors.abs().mean().item()
        self.performance_history[self.strategy].append(error_metric)

        return precision

    def evaluate_strategy(self):
        """Evaluate and potentially switch strategy"""
        # Update strategy performance metrics
        for strategy in ["gradient", "meta", "hybrid"]:
            if len(self.performance_history[strategy]) > 0:
                # Use average of recent performance
                recent = self.performance_history[strategy][-10:]
                self.strategy_performance[strategy] = sum(recent) / len(recent)
            else:
                self.strategy_performance[strategy] = float("inf")

        # Simple strategy evaluation based on recent performance
        if len(self.performance_history[self.strategy]) > 10:
            recent_performance = self.strategy_performance[self.strategy]

            # Switch if performance is poor
            if recent_performance > 1.0:  # High error
                strategies = ["gradient", "meta", "hybrid"]
                strategies.remove(self.strategy)
                self.strategy = strategies[torch.randint(0, len(strategies), (1,)).item()]

    def get_volatility_estimate(self):
        """Get volatility estimate from current optimizer"""
        if self.strategy == "gradient":
            return self.gradient_optimizer.estimate_volatility()
        elif self.strategy == "meta":
            return self.meta_optimizer.estimate_volatility()
        else:
            # Average both
            grad_vol = self.gradient_optimizer.estimate_volatility()
            meta_vol = self.meta_optimizer.estimate_volatility()
            return (grad_vol + meta_vol) / 2

    def get_precision_stats(self):
        """Get precision statistics"""
        # Return dummy stats for now
        current_precision = torch.ones(self.num_modalities)
        return {
            "mean": current_precision,
            "std": torch.zeros(self.num_modalities),
            "min": current_precision,
            "max": current_precision,
            "current": current_precision,
        }


def create_precision_optimizer(
    optimizer_type: str, config: Optional[PrecisionConfig] = None, **kwargs
):
    """Factory function for creating precision optimizers"""
    if config is None:
        config = PrecisionConfig()

    if optimizer_type == "gradient":
        num_modalities = kwargs.get("num_modalities", 1)
        return GradientPrecisionOptimizer(config, num_modalities)

    elif optimizer_type == "hierarchical":
        level_dims = kwargs.get("level_dims", [1])
        return HierarchicalPrecisionOptimizer(config, level_dims)

    elif optimizer_type == "meta":
        input_dim = kwargs.get("input_dim", 10)
        hidden_dim = kwargs.get("hidden_dim", 64)
        num_modalities = kwargs.get("num_modalities", 1)
        return MetaLearningPrecisionOptimizer(config, input_dim, hidden_dim, num_modalities)

    elif optimizer_type == "adaptive":
        num_modalities = kwargs.get("num_modalities", 1)
        context_dim = kwargs.get("context_dim", 0)
        return AdaptivePrecisionController(config, num_modalities, context_dim)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

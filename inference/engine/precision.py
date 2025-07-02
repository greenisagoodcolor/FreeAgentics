"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

"""\nPrecision Optimization for Active Inference
This module implements mechanisms to optimize precision parameters that control
the balance between prior beliefs and sensory evidence in Active Inference.
"""
logger = logging.getLogger(__name__)


@dataclass
class PrecisionConfig:
    """Configuration for precision optimization"""

    # Learning parameters
    learning_rate: float = 0.01
    meta_learning_rate: float = 0.001
    momentum: float = 0.9
    gradient_clip: float = 1.0
    # Precision bounds
    min_precision: float = 0.1
    max_precision: float = 100.0
    init_precision: float = 1.0
    # Adaptive parameters
    volatility_window: int = 10
    volatility_threshold: float = 0.5
    adaptation_rate: float = 0.1
    # Hierarchical parameters
    num_levels: int = 1
    level_coupling: float = 0.5
    # Optimization settings
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32


class PrecisionOptimizer(ABC):
    """Abstract base class for precision optimization"""

    def __init__(self, config: PrecisionConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def optimize_precision(
        self, prediction_errors: torch.Tensor, current_precision: torch.Tensor
    ) -> torch.Tensor:
        """Optimize precision given prediction errors."""

    @abstractmethod
    def estimate_volatility(self,
                            error_history: List[torch.Tensor]) -> torch.Tensor:
        """Estimate environmental volatility"""


class GradientPrecisionOptimizer(PrecisionOptimizer):
    """
    Gradient-based precision optimization.
    Optimizes precision parameters by minimizing variational free energy
    with respect to precision weights.
    """

    def __init__(self, config: PrecisionConfig,
                 num_modalities: int = 1) -> None:
        super().__init__(config)
        self.num_modalities = num_modalities
        # Initialize precision parameters
        self.log_precision = nn.Parameter(
            torch.log(
                torch.ones(
                    num_modalities,
                    device=self.device) *
                config.init_precision))
        # Optimizer for precision parameters
        self.optimizer = torch.optim.Adam(
            [self.log_precision], lr=config.learning_rate)
        # Maintain error buffer for volatility estimation
        self.error_buffer: List[torch.Tensor] = []
        self.volatility_estimate = torch.ones(
            num_modalities, device=self.device)

    def optimize_precision(
        self,
        prediction_errors: torch.Tensor,
        current_precision: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Optimize precision using gradient descent on free energy.
        Args:
            prediction_errors: Prediction errors [batch_size x num_modalities] or [num_modalities]
            current_precision: Optional current precision values
        Returns:
            optimized_precision: Updated precision values
        """
        if prediction_errors.dim() == 1:
            prediction_errors = prediction_errors.unsqueeze(0)
        # Get current precision
        precision = torch.exp(self.log_precision)
        # Compute free energy gradient w.r.t. log precision
        # F = 0.5 * (precision * error^2 - log(precision))
        # dF/d(log_precision) = 0.5 * precision * (error^2 - 1/precision)
        squared_errors = prediction_errors**2
        mean_squared_errors = squared_errors.mean(dim=0)
        # Free energy gradient
        grad = 0.5 * precision * (mean_squared_errors - 1.0 / precision)
        # Apply gradient
        self.optimizer.zero_grad()
        self.log_precision.grad = grad
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            [self.log_precision], self.config.gradient_clip)
        # Update
        self.optimizer.step()
        # Clamp precision to valid range
        self.log_precision.data = torch.clamp(
            self.log_precision.data,
            torch.log(torch.tensor(self.config.min_precision)),
            torch.log(torch.tensor(self.config.max_precision)),
        )
        # Update error buffer for volatility estimation
        self.error_buffer.append(mean_squared_errors.detach())
        if len(self.error_buffer) > self.config.volatility_window:
            self.error_buffer.pop(0)
        return torch.exp(self.log_precision).detach()

    def estimate_volatility(
        self, error_history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Estimate environmental volatility from error history.
        Args:
            error_history: Optional external error history
        Returns:
            volatility: Estimated volatility per modality
        """
        if error_history is None:
            error_history = self.error_buffer
        if len(error_history) < 2:
            return self.volatility_estimate
        # Stack errors
        errors = torch.stack(error_history)
        # Compute volatility as variance of errors over time
        volatility = torch.var(errors, dim=0)
        # Smooth volatility estimate
        self.volatility_estimate = (
            1 - self.config.adaptation_rate
        ) * self.volatility_estimate + self.config.adaptation_rate * volatility
        return self.volatility_estimate

    def adapt_to_volatility(self) -> None:
        """Adapt precision based on estimated volatility"""
        volatility = self.estimate_volatility()
        # High volatility -> lower precision (more flexible beliefs)
        # Low volatility -> higher precision (more confident beliefs)
        volatility_factor = torch.exp(-volatility /
                                      self.config.volatility_threshold)
        # Adjust precision
        with torch.no_grad():
            self.log_precision.data += self.config.adaptation_rate * (
                torch.log(volatility_factor) - self.log_precision.data
            )
            # Clamp to valid range
            self.log_precision.data = torch.clamp(
                self.log_precision.data,
                torch.log(torch.tensor(self.config.min_precision)),
                torch.log(torch.tensor(self.config.max_precision)),
            )


class HierarchicalPrecisionOptimizer(PrecisionOptimizer):
    """
    Hierarchical precision optimization with multiple levels.
    Higher levels modulate precision at lower levels, allowing for
    context-dependent precision adjustment.
    """

    def __init__(self, config: PrecisionConfig, level_dims: List[int]) -> None:
        super().__init__(config)
        self.num_levels = len(level_dims)
        self.level_dims = level_dims
        # Initialize precision for each level
        self.level_precisions = nn.ParameterList([nn.Parameter(torch.log(torch.ones(
            dim, device=self.device) * config.init_precision)) for dim in level_dims])
        # Inter-level coupling weights
        self.coupling_weights = nn.ModuleList(
            [
                nn.Linear(level_dims[i], level_dims[i + 1], bias=False)
                for i in range(self.num_levels - 1)
            ]
        )
        # Initialize coupling weights
        for w in self.coupling_weights:
            nn.init.xavier_uniform_(w.weight, gain=config.level_coupling)
        # Optimizers for each level
        self.optimizers = [
            torch.optim.Adam([p], lr=config.learning_rate * (0.5**i))
            for i, p in enumerate(self.level_precisions)
        ]
        # Volatility estimates per level
        self.level_volatilities = [torch.ones(
            dim, device=self.device) for dim in level_dims]

    def _optimize_precision_impl(
        self,
        prediction_errors: List[torch.Tensor],
        current_precision: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Optimize precision hierarchically.
        Args:
            prediction_errors: List of errors per level
            current_precision: Optional current precisions
        Returns:
            optimized_precisions: List of updated precisions per level
        """
        optimized_precisions: List[torch.Tensor] = []
        # Process from highest to lowest level
        for level in range(self.num_levels):
            errors = prediction_errors[level]
            if errors.dim() == 1:
                errors = errors.unsqueeze(0)
            # Get base precision for this level
            base_precision = torch.exp(self.level_precisions[level])
            # Apply modulation from higher levels
            if level > 0:
                # Higher level precision modulates lower level
                higher_precision = optimized_precisions[level - 1]
                modulation = torch.sigmoid(
                    self.coupling_weights[level - 1](higher_precision))
                effective_precision = base_precision * modulation
            else:
                effective_precision = base_precision
            # Compute gradient
            squared_errors = errors**2
            mean_squared_errors = squared_errors.mean(dim=0)
            # Free energy gradient
            grad = 0.5 * effective_precision * \
                (mean_squared_errors - 1.0 / effective_precision)
            # Update this level's precision
            self.optimizers[level].zero_grad()
            self.level_precisions[level].grad = grad
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                [self.level_precisions[level]], self.config.gradient_clip
            )
            # Update
            self.optimizers[level].step()
            # Clamp to valid range
            self.level_precisions[level].data = torch.clamp(
                self.level_precisions[level].data,
                torch.log(torch.tensor(self.config.min_precision)),
                torch.log(torch.tensor(self.config.max_precision)),
            )
            optimized_precisions.append(effective_precision.detach())
        return optimized_precisions

    def _estimate_volatility_impl(
        self, error_history: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Estimate volatility at each level"""
        volatilities = []
        for level in range(self.num_levels):
            if len(error_history) < 2:
                volatilities.append(self.level_volatilities[level])
                continue
            # Extract errors for this level
            level_errors = [hist[level] for hist in error_history]
            errors = torch.stack(level_errors)
            # Compute volatility
            volatility = torch.var(errors, dim=0)
            # Update estimate
            self.level_volatilities[level] = (
                1 - self.config.adaptation_rate
            ) * self.level_volatilities[level] + self.config.adaptation_rate * volatility
            volatilities.append(self.level_volatilities[level])
        return volatilities

    def optimize_precision(
        self, prediction_errors: torch.Tensor, current_precision: torch.Tensor
    ) -> torch.Tensor:
        """Optimize precision for single tensor - converts to hierarchical format"""
        # Convert to list format for hierarchical processing
        errors_list = [prediction_errors]
        result = self._optimize_precision_hierarchical(errors_list, None)
        return result[0] if result else prediction_errors.new_zeros(
            prediction_errors.shape[-1])

    def estimate_volatility(self,
                            error_history: List[torch.Tensor]) -> torch.Tensor:
        """Estimate volatility from error history - converts to hierarchical format"""
        # Convert to hierarchical format
        hierarchical_history = [[err] for err in error_history]
        result = self._estimate_volatility_hierarchical(hierarchical_history)
        return result[0] if result else torch.ones(1, device=self.device)

    def _optimize_precision_hierarchical(
        self,
        prediction_errors: List[torch.Tensor],
        current_precision: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Original hierarchical optimization implementation"""
        return self._optimize_precision_impl(
            prediction_errors, current_precision)

    def _estimate_volatility_hierarchical(
        self, error_history: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Original hierarchical volatility estimation"""

        return self._estimate_volatility_impl(error_history)


class MetaLearningPrecisionOptimizer(PrecisionOptimizer):
    """
    Meta-learning approach to precision optimization.
    Learns to predict optimal precision values based on task context
    and error patterns using a neural network.
    """

    def __init__(
        self,
        config: PrecisionConfig,
        input_dim: int,
        hidden_dim: int = 64,
        num_modalities: int = 1,
    ) -> None:
        super().__init__(config)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        # Meta-learning network - input dimension is error features (6) + context features
        # Error features: mean(2) + std(2) + max(2) = 6
        # Context features: varies based on context tensor
        error_feature_dim = 6  # Fixed: 2 mean + 2 std + 2 max
        context_feature_dim = input_dim  # Use input_dim as context dimension
        total_input_dim = error_feature_dim + context_feature_dim
        self.meta_network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
        ).to(self.device)
        # Base precision values
        self.base_precision = nn.Parameter(
            torch.log(
                torch.ones(
                    num_modalities,
                    device=self.device) *
                config.init_precision))
        # Optimizers
        self.meta_optimizer = torch.optim.Adam(
            self.meta_network.parameters(), lr=config.meta_learning_rate
        )
        self.base_optimizer = torch.optim.Adam(
            [self.base_precision], lr=config.learning_rate)
        # Context buffer
        self.context_buffer: List[Dict[str, torch.Tensor]] = []
        self.max_context_size = 100

    def extract_features(
        self, prediction_errors: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract features for meta-learning"""
        # Basic error statistics
        error_mean = prediction_errors.mean(dim=0)
        error_std = prediction_errors.std(dim=0)
        error_max = prediction_errors.max(dim=0)[0]
        # Combine features
        features = torch.cat([error_mean, error_std, error_max])
        # Add context if provided
        if context is not None:
            features = torch.cat([features, context.flatten()])
        # Pad if necessary
        if features.shape[0] < self.input_dim + self.num_modalities:
            padding = torch.zeros(
                self.input_dim + self.num_modalities - features.shape[0],
                device=self.device,
            )
            features = torch.cat([features, padding])
        return features[: self.input_dim + self.num_modalities]

    def optimize_precision(
        self, prediction_errors: torch.Tensor, current_precision: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimize precision using meta-learning.
        Args:
            prediction_errors: Prediction errors
            current_precision: Current precision values (used as context)
        Returns:
            optimized_precision: Updated precision values
        """
        if prediction_errors.dim() == 1:
            prediction_errors = prediction_errors.unsqueeze(0)
        # Use current_precision as context
        context = current_precision if current_precision is not None else None
        # Extract features
        features = self.extract_features(prediction_errors, context)
        # Predict precision adjustment
        with torch.no_grad():
            precision_adjustment = self.meta_network(features)
        # Apply adjustment to base precision
        log_precision = self.base_precision + precision_adjustment
        precision = torch.exp(log_precision)
        # Clamp to valid range
        precision = torch.clamp(
            precision,
            self.config.min_precision,
            self.config.max_precision)
        # Store context for meta-learning
        self.context_buffer.append(
            {
                "features": features.detach(),
                "errors": prediction_errors.detach(),
                "precision": precision.detach(),
            }
        )
        if len(self.context_buffer) > self.max_context_size:
            self.context_buffer.pop(0)
        return precision

    def meta_update(self, num_steps: int = 1) -> None:
        """Perform meta-learning update"""
        if len(self.context_buffer) < 10:
            return
        for _ in range(num_steps):
            # Sample batch from context buffer
            batch_size = min(32, len(self.context_buffer))
            indices = torch.randperm(len(self.context_buffer))[:batch_size]
            total_loss = torch.zeros(1, requires_grad=True, device=self.device)
            for idx in indices:
                context = self.context_buffer[idx]
                features = context["features"]
                errors = context["errors"]
                # Predict precision
                predicted_adjustment = self.meta_network(features)
                predicted_precision = torch.exp(
                    self.base_precision + predicted_adjustment)
                # Compute loss (free energy)
                squared_errors = errors**2
                mean_squared_errors = squared_errors.mean(dim=0)
                free_energy = 0.5 * torch.sum(
                    predicted_precision * mean_squared_errors - torch.log(predicted_precision)
                )
                total_loss += free_energy
            # Update meta-network
            self.meta_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.meta_network.parameters(), self.config.gradient_clip
            )
            self.meta_optimizer.step()
            # Update base precision
            self.base_optimizer.zero_grad()
            base_loss = -torch.sum(self.base_precision)  # Regularization
            base_loss.backward()
            self.base_optimizer.step()
            # Clamp base precision
            self.base_precision.data = torch.clamp(
                self.base_precision.data,
                torch.log(torch.tensor(self.config.min_precision)),
                torch.log(torch.tensor(self.config.max_precision)),
            )

    def estimate_volatility(
        self, error_history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Estimate volatility from context buffer"""
        if not self.context_buffer:
            return torch.ones(self.num_modalities, device=self.device)
        # Extract recent errors
        recent_errors = [ctx["errors"]
                         for ctx in self.context_buffer[-self.config.volatility_window:]]
        if len(recent_errors) < 2:
            return torch.ones(self.num_modalities, device=self.device)
        # Stack and compute variance
        errors = torch.stack([e.mean(dim=0) for e in recent_errors])
        volatility = torch.var(errors, dim=0)
        return volatility


class AdaptivePrecisionController:
    """
    High-level controller for adaptive precision optimization.
    Combines multiple precision optimization strategies and adapts
    to different environmental conditions.
    """

    def __init__(
        self,
        config: PrecisionConfig,
        num_modalities: int = 1,
        context_dim: Optional[int] = None,
    ) -> None:
        self.config = config
        self.num_modalities = num_modalities
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        # Type annotation for optional meta optimizer
        self.meta_optimizer: Optional[MetaLearningPrecisionOptimizer]

        # Initialize different optimizers
        self.gradient_optimizer = GradientPrecisionOptimizer(
            config, num_modalities)
        if context_dim is not None:
            self.meta_optimizer = MetaLearningPrecisionOptimizer(
                config, context_dim, num_modalities=num_modalities
            )
        else:
            self.meta_optimizer = None
        # Strategy selection
        self.strategy = "gradient"  # 'gradient', 'meta', 'hybrid'
        self.strategy_performance = {"gradient": 0.0, "meta": 0.0}
        # Performance tracking
        self.error_history: List[torch.Tensor] = []
        self.precision_history: List[torch.Tensor] = []

    def optimize(self, prediction_errors: torch.Tensor,
                 context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimize precision using adaptive strategy selection.
        Args:
            prediction_errors: Current prediction errors
            context: Optional context information
        Returns:
            precision: Optimized precision values
        """
        prediction_errors = self._ensure_tensor(prediction_errors)
        context = self._ensure_tensor(context) if context is not None else None

        self._update_error_history(prediction_errors)
        precision = self._compute_precision_with_strategy(
            prediction_errors, context)
        self._update_precision_history(precision)
        self._perform_periodic_updates()

        return precision

    def _ensure_tensor(self, data: torch.Tensor) -> torch.Tensor:
        """Ensure data is a PyTorch tensor on the correct device"""
        if hasattr(data, "numpy") and callable(getattr(data, "numpy")):
            return data
        elif hasattr(data, "shape"):
            return torch.from_numpy(data).float().to(self.device)
        return data

    def _update_error_history(self, prediction_errors: torch.Tensor) -> None:
        """Update error history with size limit"""
        self.error_history.append(prediction_errors.detach())
        if len(self.error_history) > 100:
            self.error_history.pop(0)

    def _compute_precision_with_strategy(
        self, prediction_errors: torch.Tensor, context: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute precision based on selected strategy"""
        if self.strategy == "gradient":
            return self.gradient_optimizer.optimize_precision(
                prediction_errors)
        elif self.strategy == "meta" and self.meta_optimizer is not None:
            return self._compute_meta_precision(prediction_errors, context)
        elif self.strategy == "hybrid" and self.meta_optimizer is not None:
            return self._compute_hybrid_precision(prediction_errors, context)
        else:
            return self.gradient_optimizer.optimize_precision(
                prediction_errors)

    def _compute_meta_precision(
        self, prediction_errors: torch.Tensor, context: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute precision using meta-learning strategy"""
        if context is not None:
            return self.meta_optimizer.optimize_precision(
                prediction_errors, context)
        else:
            return self.meta_optimizer.optimize_precision(
                prediction_errors, torch.zeros_like(prediction_errors)
            )

    def _compute_hybrid_precision(
        self, prediction_errors: torch.Tensor, context: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute precision using hybrid strategy"""
        grad_precision = self.gradient_optimizer.optimize_precision(
            prediction_errors)
        meta_precision = self._compute_meta_precision(
            prediction_errors, context)
        return 0.5 * (grad_precision + meta_precision)

    def _update_precision_history(self, precision: torch.Tensor) -> None:
        """Update precision history with size limit"""
        self.precision_history.append(precision.detach())
        if len(self.precision_history) > 100:
            self.precision_history.pop(0)

    def _perform_periodic_updates(self) -> None:
        """Perform periodic updates and evaluations"""
        self.gradient_optimizer.adapt_to_volatility()

        if self.meta_optimizer is not None and len(
                self.error_history) % 10 == 0:
            self.meta_optimizer.meta_update()

        if len(self.error_history) % 20 == 0:
            self._evaluate_strategies()

    def _evaluate_strategies(self) -> None:
        """Evaluate performance of different strategies"""
        if len(self.error_history) < 20:
            return
        # Compute recent performance (lower error = better)
        recent_errors = torch.stack(self.error_history[-20:])
        mean_error = recent_errors.mean().item()
        # Update performance tracking
        self.strategy_performance[self.strategy] = 0.9 * \
            self.strategy_performance[self.strategy] + 0.1 * (-mean_error)
        # Consider switching strategies
        if self.meta_optimizer is not None:
            if self.strategy_performance["meta"] > self.strategy_performance["gradient"] + 0.1:
                self.strategy = "meta"
            elif self.strategy_performance["gradient"] > self.strategy_performance["meta"] + 0.1:
                self.strategy = "gradient"
            else:
                self.strategy = "hybrid"
        logger.info(
            f"Precision strategy: {
                self.strategy}, " f"Performance: {
                self.strategy_performance}")

    def get_volatility_estimate(self) -> torch.Tensor:
        """Get current volatility estimate"""
        return self.gradient_optimizer.estimate_volatility()

    def get_precision_stats(self) -> Dict[str, torch.Tensor]:
        """Get statistics about precision optimization"""
        if not self.precision_history:
            return {}
        precision_tensor = torch.stack(self.precision_history)
        return {
            "mean": precision_tensor.mean(dim=0),
            "std": precision_tensor.std(dim=0),
            "min": precision_tensor.min(dim=0)[0],
            "max": precision_tensor.max(dim=0)[0],
            "current": self.precision_history[-1],
        }


def create_precision_optimizer(
        optimizer_type: str,
        config: Optional[PrecisionConfig] = None,
        **kwargs: Any) -> PrecisionOptimizer:
    """
    Factory function to create precision optimizers.
    Args:
        optimizer_type: Type of optimizer ('gradient', 'hierarchical', 'meta', 'adaptive')
        config: Precision configuration
        **kwargs: Optimizer-specific parameters
    Returns:
        Precision optimizer instance
    """
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
        return MetaLearningPrecisionOptimizer(
            config, input_dim, hidden_dim, num_modalities)
    elif optimizer_type == "adaptive":
        num_modalities = kwargs.get("num_modalities", 1)
        context_dim = kwargs.get("context_dim", None)
        controller = AdaptivePrecisionController(
            config, num_modalities, context_dim)
        return controller  # type: ignore[return-value]
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = PrecisionConfig(learning_rate=0.01, use_gpu=False)
    # Create gradient-based optimizer
    optimizer = GradientPrecisionOptimizer(config, num_modalities=2)
    # Simulate prediction errors
    for t in range(50):
        # Generate random errors (higher variance in first modality)
        errors = torch.randn(10, 2) * torch.tensor([2.0, 0.5])
        # Optimize precision
        precision = optimizer.optimize_precision(errors)
        # Adapt to volatility every 10 steps
        if t % 10 == 0:
            optimizer.adapt_to_volatility()
            volatility = optimizer.estimate_volatility()
            print(
                f"Step {t}: Precision={
                    precision.numpy()}, Volatility={
                    volatility.numpy()}")
    print("\nFinal precision:", precision.numpy())

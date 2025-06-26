import os
import sys

import pytest
import torch

from inference.engine.precision import (
    AdaptivePrecisionController,
    GradientPrecisionOptimizer,
    HierarchicalPrecisionOptimizer,
    MetaLearningPrecisionOptimizer,
    PrecisionConfig,
    create_precision_optimizer,
)


class TestPrecisionConfig:
    """Test PrecisionConfig dataclass"""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = PrecisionConfig()
        assert config.learning_rate == 0.01
        assert config.meta_learning_rate == 0.001
        assert config.momentum == 0.9
        assert config.gradient_clip == 1.0
        assert config.min_precision == 0.1
        assert config.max_precision == 100.0
        assert config.init_precision == 1.0
        assert config.volatility_window == 10
        assert config.volatility_threshold == 0.5
        assert config.adaptation_rate == 0.1
        assert config.num_levels == 1
        assert config.level_coupling == 0.5
        assert config.use_gpu is True
        assert config.dtype == torch.float32

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = PrecisionConfig(
            learning_rate=0.1,
            min_precision=0.01,
            max_precision=1000.0,
            use_gpu=False,
        )
        assert config.learning_rate == 0.1
        assert config.min_precision == 0.01
        assert config.max_precision == 1000.0
        assert config.use_gpu is False


class TestGradientPrecisionOptimizer:
    """Test gradient-based precision optimization"""

    def setup_method(self) -> None:
        """Set up test environment"""
        self.config = PrecisionConfig(use_gpu=False, init_precision=1.0)
        self.optimizer = GradientPrecisionOptimizer(self.config, num_modalities=2)

    def test_initialization(self) -> None:
        """Test optimizer initialization"""
        assert self.optimizer.num_modalities == 2
        assert self.optimizer.log_precision.shape == (2,)
        # Check initial precision
        precision = torch.exp(self.optimizer.log_precision)
        assert torch.allclose(precision, torch.tensor([1.0, 1.0]))

    def test_precision_optimization(self) -> None:
        """Test basic precision optimization"""
        # High errors should increase precision
        high_errors = torch.randn(10, 2) * 5.0
        precision1 = self.optimizer.optimize_precision(high_errors)
        # Low errors should decrease precision
        low_errors = torch.randn(10, 2) * 0.1
        precision2 = self.optimizer.optimize_precision(low_errors)
        assert precision1.shape == (2,)
        assert precision2.shape == (2,)
        # Precision should be within bounds
        assert torch.all(precision1 >= self.config.min_precision)
        assert torch.all(precision1 <= self.config.max_precision)

    def test_single_observation_optimization(self) -> None:
        """Test optimization with single observation"""
        error = torch.tensor([1.0, 0.5])
        precision = self.optimizer.optimize_precision(error)
        assert precision.shape == (2,)
        assert torch.all(precision > 0)

    def test_volatility_estimation(self) -> None:
        """Test volatility estimation"""
        # Add consistent errors
        for _ in range(5):
            errors = torch.ones(1, 2)
            self.optimizer._optimize_precision_hierarchical(errors)
        volatility1 = self.optimizer.estimate_volatility()
        # Add variable errors
        for i in range(5):
            errors = torch.ones(1, 2) * (i % 2)
            self.optimizer._optimize_precision_hierarchical(errors)
        volatility2 = self.optimizer.estimate_volatility()
        # Variable errors should have higher volatility
        assert torch.all(volatility2 > volatility1)

    def test_volatility_adaptation(self) -> None:
        """Test adaptation to volatility"""
        # Create high volatility scenario
        for i in range(20):
            errors = torch.randn(1, 2) * (5.0 if i % 2 == 0 else 0.1)
            self.optimizer._optimize_precision_hierarchical(errors)
        # Get precision before adaptation
        precision_before = torch.exp(self.optimizer.log_precision.data.clone())
        # Adapt to volatility
        self.optimizer.adapt_to_volatility()
        # Get precision after adaptation
        precision_after = torch.exp(self.optimizer.log_precision.data)
        # Precision should have changed
        assert not torch.allclose(precision_before, precision_after)

    def test_gradient_clipping(self) -> None:
        """Test gradient clipping"""
        # Create extreme errors
        extreme_errors = torch.ones(1, 2) * 1000.0
        # Optimize (should not explode due to clipping)
        precision = self.optimizer.optimize_precision(extreme_errors)
        assert torch.all(torch.isfinite(precision))
        assert torch.all(precision <= self.config.max_precision)


class TestHierarchicalPrecisionOptimizer:
    """Test hierarchical precision optimization"""

    def setup_method(self) -> None:
        """Set up test environment"""
        self.config = PrecisionConfig(use_gpu=False)
        self.level_dims = [3, 2, 1]
        self.optimizer = HierarchicalPrecisionOptimizer(self.config, self.level_dims)

    def test_initialization(self) -> None:
        """Test hierarchical initialization"""
        assert self.optimizer.num_levels == 3
        assert len(self.optimizer.level_precisions) == 3
        assert len(self.optimizer.coupling_weights) == 2
        # Check dimensions
        for i, dim in enumerate(self.level_dims):
            assert self.optimizer.level_precisions[i].shape == (dim,)

    def test_hierarchical_optimization(self) -> None:
        """Test optimization across levels"""
        # Create errors for each level
        errors = [
            torch.randn(5, 3),  # Level 0
            torch.randn(5, 2),  # Level 1
            torch.randn(5, 1),  # Level 2
        ]
        precisions = self.optimizer._optimize_precision_hierarchical(errors)
        assert len(precisions) == 3
        for i, p in enumerate(precisions):
            assert p.shape == (self.level_dims[i],)
            assert torch.all(p > 0)

    def test_level_coupling(self) -> None:
        """Test inter-level coupling"""
        # High errors at top level
        errors = [
            torch.randn(5, 3) * 0.1,  # Low errors at level 0
            torch.randn(5, 2) * 0.1,  # Low errors at level 1
            torch.randn(5, 1) * 5.0,  # High errors at level 2
        ]
        precisions = self.optimizer._optimize_precision_hierarchical(errors)
        # All levels should be affected due to coupling
        assert all(torch.all(p > 0) for p in precisions)

    def test_volatility_per_level(self) -> None:
        """Test volatility estimation per level"""
        error_history = []
        # Generate variable error patterns
        for t in range(15):
            errors = [
                torch.randn(1, 3) * (1.0 if t % 2 == 0 else 0.1),
                torch.randn(1, 2) * 0.5,  # Constant
                torch.randn(1, 1) * (t * 0.1),  # Increasing
            ]
            error_history.append(errors)
            self.optimizer._optimize_precision_hierarchical(errors)
        # Transpose the error history
        error_history_per_level = list(zip(*error_history))
        volatilities = self.optimizer._estimate_volatility_hierarchical(error_history_per_level)
        assert len(volatilities) == 3
        assert volatilities[0].shape == (3,)
        # Level 0 should have high volatility (alternating)
        # Level 1 should have low volatility (constant)
        # Level 2 should have moderate volatility (trend)
        assert volatilities[0].mean() > volatilities[1].mean()


class TestMetaLearningPrecisionOptimizer:
    """Test meta-learning precision optimization"""

    def setup_method(self) -> None:
        """Set up test environment"""
        self.config = PrecisionConfig(use_gpu=False)
        self.optimizer = MetaLearningPrecisionOptimizer(
            self.config, input_dim=3, hidden_dim=32, num_modalities=2
        )

    def test_initialization(self) -> None:
        """Test meta-learning initialization"""
        assert self.optimizer.input_dim == 3
        assert self.optimizer.hidden_dim == 32
        assert self.optimizer.num_modalities == 2
        assert hasattr(self.optimizer, "meta_network")
        assert hasattr(self.optimizer, "base_precision")

    def test_feature_extraction(self) -> None:
        """Test feature extraction"""
        errors = torch.randn(10, 2)
        context = torch.randn(3)
        features = self.optimizer.extract_features(errors, context)
        # Should have correct dimension
        assert features.shape[0] == (6 + self.optimizer.input_dim)  # 6 error stats + context

    def test_precision_optimization_with_context(self) -> None:
        """Test optimization with context"""
        errors = torch.randn(10, 2)
        context = torch.tensor([1.0, 0.0, 0.5])
        precision = self.optimizer.optimize_precision(errors, context)
        assert precision.shape == (2,)
        assert torch.all(precision > 0)

    def test_context_buffer(self) -> None:
        """Test context buffer management"""
        # Fill buffer
        for i in range(150):
            errors = torch.randn(5, 2)
            self.optimizer._optimize_precision_hierarchical(errors)
        # Check buffer size
        assert len(self.optimizer.context_buffer) == self.optimizer.max_context_size

    def test_meta_update(self) -> None:
        """Test meta-learning update"""
        # Generate some context
        for _ in range(20):
            errors = torch.randn(5, 2)
            self.optimizer._optimize_precision_hierarchical(errors)
        # Store network weights before update
        weights_before = self.optimizer.meta_network[0].weight.data.clone()
        # Perform meta update
        self.optimizer.meta_update(num_steps=1)
        # Check weights have changed
        assert not torch.allclose(weights_before, self.optimizer.meta_network[0].weight.data)

    def test_volatility_from_buffer(self) -> None:
        """Test volatility estimation from context buffer"""
        # Add variable errors
        for i in range(20):
            errors = torch.randn(5, 2) * (1.0 if i % 3 == 0 else 0.1)
            self.optimizer._optimize_precision_hierarchical(errors)
        volatility = self.optimizer.estimate_volatility()
        assert volatility.shape == (2,)
        assert torch.all(volatility > 0)


class TestAdaptivePrecisionController:
    """Test adaptive precision controller"""

    def setup_method(self) -> None:
        """Set up test environment"""
        self.config = PrecisionConfig(use_gpu=False)
        self.controller = AdaptivePrecisionController(self.config, num_modalities=2, context_dim=4)

    def test_initialization(self) -> None:
        """Test controller initialization"""
        assert self.controller.num_modalities == 2
        assert self.controller.context_dim == 4
        assert hasattr(self.controller, "gradient_optimizer")
        assert hasattr(self.controller, "meta_optimizer")

    def test_optimization_with_strategy(self) -> None:
        """Test optimization with different strategies"""
        errors = torch.randn(10, 2)
        context = torch.randn(4)
        # Test gradient strategy
        self.controller.strategy = "gradient"
        precision1 = self.controller.optimize(errors, context)
        # Test meta strategy
        self.controller.strategy = "meta"
        precision2 = self.controller.optimize(errors, context)
        assert precision1.shape == (2,)
        assert precision2.shape == (2,)
        assert not torch.allclose(precision1, precision2)

    def test_strategy_evaluation(self) -> None:
        """Test strategy evaluation"""
        # Generate some data
        for _ in range(20):
            errors = torch.randn(5, 2)
            context = torch.randn(4)
            self.controller.optimize(errors, context)
        # Evaluate strategies
        self.controller.evaluate_strategy()
        # Check performance stats updated
        assert self.controller.strategy_performance["gradient"] > 0
        assert self.controller.strategy_performance["meta"] > 0

    def test_volatility_estimate(self) -> None:
        """Test volatility estimation"""
        for _ in range(15):
            errors = torch.randn(5, 2)
            self.controller.optimize(errors)
        volatility = self.controller.get_volatility_estimate()
        assert volatility.shape == (2,)

    def test_precision_statistics(self) -> None:
        """Test precision statistics"""
        for _ in range(10):
            self.controller.optimize(torch.randn(5, 2))
        stats = self.controller.get_precision_stats()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats


class TestPrecisionFactory:
    """Test precision optimizer factory"""

    def test_create_gradient_optimizer(self) -> None:
        """Test creating gradient optimizer"""
        optimizer = create_precision_optimizer("gradient", num_modalities=2)
        assert isinstance(optimizer, GradientPrecisionOptimizer)

    def test_create_hierarchical_optimizer(self) -> None:
        """Test creating hierarchical optimizer"""
        optimizer = create_precision_optimizer("hierarchical", level_dims=[2, 2])
        assert isinstance(optimizer, HierarchicalPrecisionOptimizer)

    def test_create_meta_optimizer(self) -> None:
        """Test creating meta-learning optimizer"""
        optimizer = create_precision_optimizer("meta", input_dim=4, hidden_dim=16, num_modalities=2)
        assert isinstance(optimizer, MetaLearningPrecisionOptimizer)

    def test_create_adaptive_controller(self) -> None:
        """Test creating adaptive controller"""
        controller = create_precision_optimizer("adaptive", num_modalities=2, context_dim=4)
        assert isinstance(controller, AdaptivePrecisionController)

    def test_invalid_optimizer_type(self) -> None:
        """Test invalid optimizer type"""
        with pytest.raises(ValueError):
            create_precision_optimizer("invalid_type")

    def test_custom_config(self) -> None:
        """Test creation with custom config"""
        config = PrecisionConfig(learning_rate=0.5)
        optimizer = create_precision_optimizer("gradient", config=config, num_modalities=2)
        assert optimizer.config.learning_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

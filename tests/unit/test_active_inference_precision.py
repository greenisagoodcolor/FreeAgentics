"""
Comprehensive tests for Active Inference Precision Optimization.

Tests the precision optimization mechanisms for active inference including
gradient-based, hierarchical, meta-learning, and adaptive precision control.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from agents.active_inference.precision import (
    AdaptivePrecisionController,
    GradientPrecisionOptimizer,
    HierarchicalPrecisionOptimizer,
    MetaLearningPrecisionOptimizer,
    PrecisionConfig,
    create_precision_optimizer,
)


class TestPrecisionConfig:
    """Test PrecisionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
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

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PrecisionConfig(
            learning_rate=0.05,
            min_precision=0.01,
            max_precision=1000.0,
            init_precision=10.0,
            use_gpu=False,
        )

        assert config.learning_rate == 0.05
        assert config.min_precision == 0.01
        assert config.max_precision == 1000.0
        assert config.init_precision == 10.0
        assert config.use_gpu is False


class TestGradientPrecisionOptimizer:
    """Test GradientPrecisionOptimizer class."""

    def setup_method(self):
        """Set up test optimizer."""
        self.config = PrecisionConfig(
            learning_rate=0.1,
            min_precision=0.1,
            max_precision=10.0,
            init_precision=1.0)
        self.num_modalities = 3
        self.optimizer = GradientPrecisionOptimizer(
            self.config, self.num_modalities)

    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.config == self.config
        assert self.optimizer.num_modalities == 3
        assert self.optimizer.log_precision.shape == (3,)
        assert torch.allclose(
            torch.exp(
                self.optimizer.log_precision),
            torch.ones(3))
        assert self.optimizer.error_history == []

    def test_optimize_precision_1d_errors(self):
        """Test optimizing precision with 1D error vector."""
        errors = torch.tensor([0.5, 1.0, 1.5])

        precision = self.optimizer.optimize_precision(errors)

        # Check shape
        assert precision.shape == (3,)

        # Check precision increased where errors were larger
        assert precision[2] > precision[0]  # Higher error -> higher precision

        # Check bounds
        assert (precision >= self.config.min_precision).all()
        assert (precision <= self.config.max_precision).all()

        # Check error history
        assert len(self.optimizer.error_history) == 1
        assert self.optimizer.error_history[0].shape == (
            3,
        )  # GradientPrecisionOptimizer stores 1D errors

    def test_optimize_precision_2d_errors(self):
        """Test optimizing precision with 2D error matrix."""
        errors = torch.tensor([[0.5, 1.0, 1.5], [0.8, 1.2, 2.0]])

        precision = self.optimizer.optimize_precision(errors)

        assert precision.shape == (3,)
        # Mean errors: [0.65, 1.1, 1.75]
        # Precision should increase most for modality 2
        assert precision[2] > precision[1] > precision[0]

    def test_optimize_precision_with_context(self):
        """Test that context parameter is accepted (but not used)."""
        errors = torch.ones(3) * 0.5
        context = torch.randn(5)

        precision = self.optimizer.optimize_precision(errors, context=context)

        assert precision.shape == (3,)

    def test_error_history_window(self):
        """Test error history respects window size."""
        # Set small window for testing
        self.optimizer.config.volatility_window = 3

        for i in range(5):
            errors = torch.ones(3) * (i + 1)
            self.optimizer.optimize_precision(errors)

        # Should only keep last 3
        assert len(self.optimizer.error_history) == 3
        assert torch.allclose(
            self.optimizer.error_history[0],
            torch.ones(
                1,
                3) * 3)
        assert torch.allclose(
            self.optimizer.error_history[-1], torch.ones(1, 3) * 5)

    def test_estimate_volatility_insufficient_history(self):
        """Test volatility estimation with insufficient history."""
        volatility = self.optimizer.estimate_volatility()
        assert volatility.shape == (3,)
        assert torch.allclose(volatility, torch.zeros(3))

        # Add one error
        self.optimizer.optimize_precision(torch.ones(3))
        volatility = self.optimizer.estimate_volatility()
        assert torch.allclose(volatility, torch.zeros(3))

    def test_estimate_volatility_with_history(self):
        """Test volatility estimation with sufficient history."""
        # Add varying errors
        errors_list = [
            torch.tensor([0.5, 1.0, 1.5]),
            torch.tensor([0.8, 0.9, 1.6]),
            torch.tensor([0.3, 1.1, 1.4]),
        ]

        for errors in errors_list:
            self.optimizer.optimize_precision(errors)

        volatility = self.optimizer.estimate_volatility()
        assert volatility.shape == (3,)
        # Should have non-zero volatility due to variation
        assert (volatility > 0).all()

        # Modality 0 has highest variation [0.5, 0.8, 0.3]
        # Modality 1 has low variation [1.0, 0.9, 1.1]
        assert volatility[0] > volatility[1]

    def test_adapt_to_volatility(self):
        """Test precision adaptation based on volatility."""
        # Create history with high volatility
        for i in range(3):
            errors = torch.randn(3) * (i + 1)
            self.optimizer.optimize_precision(errors)

        initial_log_precision = self.optimizer.log_precision.clone()
        self.optimizer.adapt_to_volatility()

        # Precision should have changed
        assert not torch.allclose(
            self.optimizer.log_precision,
            initial_log_precision)

    def test_precision_bounds(self):
        """Test precision respects min/max bounds."""
        # Large positive errors should increase precision
        large_errors = torch.ones(3) * 100.0
        precision = self.optimizer.optimize_precision(large_errors)
        assert (precision <= self.config.max_precision).all()

        # Reset and test minimum
        self.optimizer.log_precision = torch.log(torch.ones(3) * 0.01)
        small_errors = torch.ones(3) * 0.001
        precision = self.optimizer.optimize_precision(small_errors)
        assert (precision >= self.config.min_precision).all()


class TestHierarchicalPrecisionOptimizer:
    """Test HierarchicalPrecisionOptimizer class."""

    def setup_method(self):
        """Set up test optimizer."""
        self.config = PrecisionConfig(
            learning_rate=0.1,
            min_precision=0.1,
            max_precision=10.0,
            init_precision=1.0,
            level_coupling=0.3,
        )
        self.level_dims = [2, 3, 4]  # 3 levels with different dimensions
        self.optimizer = HierarchicalPrecisionOptimizer(
            self.config, self.level_dims)

    def test_initialization(self):
        """Test hierarchical optimizer initialization."""
        assert self.optimizer.config == self.config
        assert self.optimizer.level_dims == [2, 3, 4]
        assert self.optimizer.num_levels == 3

        # Check precision initialization for each level
        assert len(self.optimizer.level_precisions) == 3
        assert self.optimizer.level_precisions[0].shape == (2,)
        assert self.optimizer.level_precisions[1].shape == (3,)
        assert self.optimizer.level_precisions[2].shape == (4,)

        # Check all initialized to init_precision
        for level_precision in self.optimizer.level_precisions:
            assert torch.allclose(
                torch.exp(level_precision),
                torch.ones_like(level_precision))

        # Check coupling weights
        assert len(self.optimizer.coupling_weights) == 2  # num_levels - 1
        for weight in self.optimizer.coupling_weights:
            assert torch.allclose(weight, torch.tensor([0.3]))

    def test_optimize_precision_single_level(self):
        """Test optimizing precision for single level."""
        errors_list = [
            torch.tensor([0.5, 0.8]),
            torch.tensor([1.0, 1.2, 0.9]),
            torch.tensor([1.5, 2.0, 1.8, 1.6]),
        ]

        precisions = self.optimizer.optimize_precision(errors_list)

        assert len(precisions) == 3
        assert precisions[0].shape == (2,)
        assert precisions[1].shape == (3,)
        assert precisions[2].shape == (4,)

        # Check bounds
        for precision in precisions:
            assert (precision >= self.config.min_precision).all()
            assert (precision <= self.config.max_precision).all()

    def test_optimize_precision_with_coupling(self):
        """Test that coupling affects precision optimization."""
        errors_list = [
            torch.ones(2) * 2.0,  # High error at level 0
            torch.ones(3) * 0.5,  # Low error at level 1
            torch.ones(4) * 0.5,  # Low error at level 2
        ]

        # Run multiple times to see coupling effect
        for _ in range(5):
            precisions = self.optimizer.optimize_precision(errors_list)

        # Level 0 has high error, should have higher precision
        assert precisions[0].mean() > precisions[1].mean()

        # Level 1 should be influenced by level 0 through coupling
        # (though effect may be small with single update)

    def test_optimize_precision_1d_errors(self):
        """Test handling 1D error inputs."""
        errors_list = [
            torch.tensor([0.5, 0.8]),  # 1D
            torch.tensor([1.0, 1.2, 0.9]),  # 1D
            torch.tensor([1.5, 2.0, 1.8, 1.6]),  # 1D
        ]

        precisions = self.optimizer.optimize_precision(errors_list)

        # Should handle 1D inputs by unsqueezing
        assert len(precisions) == 3
        for i, precision in enumerate(precisions):
            assert precision.shape == (self.level_dims[i],)

    def test_estimate_volatility_empty_history(self):
        """Test volatility estimation with empty history."""
        error_history_list = [[], [], []]

        volatilities = self.optimizer.estimate_volatility(error_history_list)

        assert len(volatilities) == 3
        for i, volatility in enumerate(volatilities):
            assert volatility.shape == (self.level_dims[i],)
            assert torch.allclose(volatility, torch.zeros(self.level_dims[i]))

    def test_estimate_volatility_with_history(self):
        """Test volatility estimation with history."""
        # Create history with varying errors
        error_history_list = [
            [(torch.randn(1, 2),) for _ in range(5)],
            [(torch.randn(1, 3),) for _ in range(5)],
            [(torch.randn(1, 4),) for _ in range(5)],
        ]

        volatilities = self.optimizer.estimate_volatility(error_history_list)

        assert len(volatilities) == 3
        for i, volatility in enumerate(volatilities):
            assert volatility.shape == (self.level_dims[i],)
            # Should have non-zero volatility
            assert (volatility > 0).any()


class TestMetaLearningPrecisionOptimizer:
    """Test MetaLearningPrecisionOptimizer class."""

    def setup_method(self):
        """Set up test optimizer."""
        self.config = PrecisionConfig(
            init_precision=2.0,
            min_precision=0.1,
            max_precision=10.0,
            meta_learning_rate=0.01)
        self.input_dim = 5
        self.hidden_dim = 32
        self.num_modalities = 3
        self.optimizer = MetaLearningPrecisionOptimizer(
            self.config, self.input_dim, self.hidden_dim, self.num_modalities
        )

    def test_initialization(self):
        """Test meta-learning optimizer initialization."""
        assert self.optimizer.config == self.config
        assert self.optimizer.input_dim == 5
        assert self.optimizer.hidden_dim == 32
        assert self.optimizer.num_modalities == 3

        # Check meta network structure
        assert isinstance(self.optimizer.meta_network, nn.Sequential)
        assert len(self.optimizer.meta_network) == 5  # 3 Linear + 2 ReLU

        # Check network dimensions
        max_input_dim = (
            self.num_modalities * 3 + self.input_dim
        )  # error features (mean+std+max per modality) + context
        assert self.optimizer.meta_network[0].in_features == max_input_dim
        assert self.optimizer.meta_network[0].out_features == self.hidden_dim
        assert self.optimizer.meta_network[-1].out_features == self.num_modalities

        # Check base precision
        assert self.optimizer.base_precision.shape == (3,)
        assert torch.allclose(
            self.optimizer.base_precision,
            torch.ones(3) * 2.0)

        # Check context buffer
        assert self.optimizer.context_buffer == []
        assert self.optimizer.max_context_size == 100

    def test_extract_features_1d_errors_no_context(self):
        """Test feature extraction with 1D errors, no context."""
        errors = torch.tensor([0.5, 1.0, 1.5])

        features = self.optimizer.extract_features(errors)

        # Should have padded to match network input
        expected_dim = self.optimizer.meta_network[0].in_features
        assert features.shape == (expected_dim,)

        # First 9 features should be: mean(3) + std(3) + max(3)
        # Check they're not all zeros (padding comes after)
        assert not torch.allclose(features[:9], torch.zeros(9))

    def test_extract_features_2d_errors_with_context(self):
        """Test feature extraction with 2D errors and context."""
        errors = torch.tensor([[0.5, 1.0, 1.5], [0.8, 1.2, 2.0]])
        context = torch.randn(3)

        features = self.optimizer.extract_features(errors, context)

        expected_dim = self.optimizer.meta_network[0].in_features
        assert features.shape == (expected_dim,)

        # Should include context features
        # First 9 are error stats, next 3 are context
        assert not torch.allclose(features[:12], torch.zeros(12))

    def test_extract_features_truncation(self):
        """Test feature truncation when too many features."""
        errors = torch.ones(3)
        # Create context larger than expected
        large_context = torch.randn(100)

        features = self.optimizer.extract_features(errors, large_context)

        expected_dim = self.optimizer.meta_network[0].in_features
        assert features.shape == (expected_dim,)

    def test_optimize_precision_basic(self):
        """Test basic precision optimization."""
        errors = torch.tensor([0.5, 1.0, 1.5])

        precision = self.optimizer.optimize_precision(errors)

        assert precision.shape == (3,)
        assert (precision >= self.config.min_precision).all()
        assert (precision <= self.config.max_precision).all()

        # Check context buffer updated
        assert len(self.optimizer.context_buffer) == 1
        assert self.optimizer.context_buffer[0][0].shape == (1, 3)
        assert self.optimizer.context_buffer[0][1] is None

    def test_optimize_precision_with_context(self):
        """Test precision optimization with context."""
        errors = torch.ones(3) * 0.8
        context = torch.randn(5)

        precision = self.optimizer.optimize_precision(errors, context)

        assert precision.shape == (3,)

        # Check context stored
        assert len(self.optimizer.context_buffer) == 1
        assert torch.allclose(self.optimizer.context_buffer[0][1], context)

    def test_context_buffer_limit(self):
        """Test context buffer respects size limit."""
        self.optimizer.max_context_size = 5

        for i in range(10):
            errors = torch.ones(3) * i
            self.optimizer.optimize_precision(errors)

        assert len(self.optimizer.context_buffer) == 5
        # Should have kept last 5
        assert torch.allclose(
            self.optimizer.context_buffer[0][0],
            torch.ones(
                1,
                3) * 5)
        assert torch.allclose(
            self.optimizer.context_buffer[-1][0], torch.ones(1, 3) * 9)

    def test_meta_update_insufficient_data(self):
        """Test meta update with insufficient data."""
        # Add only a few samples
        for i in range(3):
            errors = torch.ones(3) * i
            self.optimizer.optimize_precision(errors)

        # Should return without error
        self.optimizer.meta_update(num_steps=10)

    @patch("torch.optim.Adam")
    def test_meta_update_with_data(self, mock_adam_class):
        """Test meta update with sufficient data."""
        mock_optimizer = Mock()
        mock_adam_class.return_value = mock_optimizer

        # Add sufficient samples
        for i in range(15):
            errors = torch.randn(3)
            context = torch.randn(2)
            self.optimizer.optimize_precision(errors, context)

        self.optimizer.meta_update(num_steps=5)

        # Check optimizer was created and used
        mock_adam_class.assert_called_once()
        assert mock_optimizer.zero_grad.call_count == 5
        assert mock_optimizer.step.call_count == 5

    def test_estimate_volatility_insufficient_buffer(self):
        """Test volatility estimation with insufficient buffer."""
        volatility = self.optimizer.estimate_volatility()
        assert volatility.shape == (3,)
        assert torch.allclose(volatility, torch.zeros(3))

    def test_estimate_volatility_with_buffer(self):
        """Test volatility estimation with sufficient buffer."""
        # Add varying errors
        for i in range(5):
            errors = torch.randn(2, 3) * (i + 1)
            self.optimizer.optimize_precision(errors)

        volatility = self.optimizer.estimate_volatility()
        assert volatility.shape == (3,)
        assert (volatility > 0).all()


class TestAdaptivePrecisionController:
    """Test AdaptivePrecisionController class."""

    def setup_method(self):
        """Set up test controller."""
        self.config = PrecisionConfig()
        self.num_modalities = 3
        self.context_dim = 5
        self.controller = AdaptivePrecisionController(
            self.config, self.num_modalities, self.context_dim
        )

    def test_initialization(self):
        """Test adaptive controller initialization."""
        assert self.controller.config == self.config
        assert self.controller.num_modalities == 3
        assert self.controller.context_dim == 5

        # Check optimizers
        assert isinstance(
            self.controller.gradient_optimizer,
            GradientPrecisionOptimizer)
        assert isinstance(
            self.controller.meta_optimizer,
            MetaLearningPrecisionOptimizer)

        # Check strategy
        assert self.controller.strategy == "gradient"
        assert self.controller.performance_history == {
            "gradient": [], "meta": [], "hybrid": []}
        assert self.controller.strategy_performance == {
            "gradient": 0.0, "meta": 0.0, "hybrid": 0.0}

    def test_optimize_gradient_strategy(self):
        """Test optimization with gradient strategy."""
        errors = torch.tensor([0.5, 1.0, 1.5])

        precision = self.controller.optimize(errors)

        assert precision.shape == (3,)
        assert len(self.controller.performance_history["gradient"]) == 1
        assert self.controller.performance_history["gradient"][0] == errors.abs(
        ).mean().item()

    def test_optimize_meta_strategy(self):
        """Test optimization with meta strategy."""
        self.controller.strategy = "meta"
        errors = torch.ones(3) * 0.8
        context = torch.randn(5)

        precision = self.controller.optimize(errors, context)

        assert precision.shape == (3,)
        assert len(self.controller.performance_history["meta"]) == 1

    def test_optimize_hybrid_strategy(self):
        """Test optimization with hybrid strategy."""
        self.controller.strategy = "hybrid"
        errors = torch.ones(3)

        precision = self.controller.optimize(errors)

        assert precision.shape == (3,)
        assert len(self.controller.performance_history["hybrid"]) == 1

        # Should be average of gradient and meta
        # (though exact values depend on optimizer internals)

    def test_optimize_invalid_strategy(self):
        """Test optimization with invalid strategy."""
        self.controller.strategy = "invalid"
        errors = torch.ones(3)

        with pytest.raises(ValueError) as excinfo:
            self.controller.optimize(errors)

        assert "Unknown strategy: invalid" in str(excinfo.value)

    def test_evaluate_strategy_no_history(self):
        """Test strategy evaluation with no history."""
        self.controller.evaluate_strategy()

        # Should set all performances to inf
        assert self.controller.strategy_performance["gradient"] == float("inf")
        assert self.controller.strategy_performance["meta"] == float("inf")
        assert self.controller.strategy_performance["hybrid"] == float("inf")

    def test_evaluate_strategy_with_history(self):
        """Test strategy evaluation with performance history."""
        # Add performance history
        self.controller.performance_history["gradient"] = [
            0.5, 0.4, 0.3, 0.2, 0.1]
        self.controller.performance_history["meta"] = [0.8, 0.7, 0.6]
        self.controller.performance_history["hybrid"] = []

        self.controller.evaluate_strategy()

        # Should compute averages
        assert self.controller.strategy_performance["gradient"] == pytest.approx(
            0.3)
        assert self.controller.strategy_performance["meta"] == pytest.approx(
            0.7)
        assert self.controller.strategy_performance["hybrid"] == float("inf")

    @patch("torch.randint")
    def test_evaluate_strategy_switch(self, mock_randint):
        """Test strategy switching based on poor performance."""
        mock_randint.return_value.item.return_value = 0

        # Set up poor performance for current strategy
        self.controller.strategy = "gradient"
        self.controller.performance_history["gradient"] = [
            2.0] * 15  # High error

        self.controller.evaluate_strategy()

        # Should have switched strategy
        assert self.controller.strategy in ["meta", "hybrid"]

    def test_get_volatility_estimate_gradient(self):
        """Test volatility estimate with gradient strategy."""
        # Add some errors to gradient optimizer
        for i in range(3):
            self.controller.gradient_optimizer.optimize_precision(
                torch.randn(3))

        volatility = self.controller.get_volatility_estimate()
        assert volatility.shape == (3,)

    def test_get_volatility_estimate_meta(self):
        """Test volatility estimate with meta strategy."""
        self.controller.strategy = "meta"

        # Add some errors to meta optimizer
        for i in range(3):
            self.controller.meta_optimizer.optimize_precision(torch.randn(3))

        volatility = self.controller.get_volatility_estimate()
        assert volatility.shape == (3,)

    def test_get_volatility_estimate_hybrid(self):
        """Test volatility estimate with hybrid strategy."""
        self.controller.strategy = "hybrid"

        volatility = self.controller.get_volatility_estimate()
        assert volatility.shape == (3,)
        # Should be average of both

    def test_get_precision_stats(self):
        """Test getting precision statistics."""
        stats = self.controller.get_precision_stats()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "current" in stats

        assert stats["mean"].shape == (3,)
        assert stats["std"].shape == (3,)
        assert torch.allclose(stats["mean"], torch.ones(3))
        assert torch.allclose(stats["std"], torch.zeros(3))


class TestCreatePrecisionOptimizer:
    """Test create_precision_optimizer factory function."""

    def test_create_gradient_optimizer_default(self):
        """Test creating gradient optimizer with defaults."""
        optimizer = create_precision_optimizer("gradient", num_modalities=4)

        assert isinstance(optimizer, GradientPrecisionOptimizer)
        assert optimizer.num_modalities == 4
        assert optimizer.config.learning_rate == 0.01  # Default

    def test_create_gradient_optimizer_custom_config(self):
        """Test creating gradient optimizer with custom config."""
        config = PrecisionConfig(learning_rate=0.05)
        optimizer = create_precision_optimizer(
            "gradient", config, num_modalities=2)

        assert isinstance(optimizer, GradientPrecisionOptimizer)
        assert optimizer.config.learning_rate == 0.05

    def test_create_hierarchical_optimizer(self):
        """Test creating hierarchical optimizer."""
        optimizer = create_precision_optimizer(
            "hierarchical", level_dims=[2, 3, 4])

        assert isinstance(optimizer, HierarchicalPrecisionOptimizer)
        assert optimizer.level_dims == [2, 3, 4]

    def test_create_meta_optimizer(self):
        """Test creating meta-learning optimizer."""
        optimizer = create_precision_optimizer(
            "meta", input_dim=10, hidden_dim=64, num_modalities=5
        )

        assert isinstance(optimizer, MetaLearningPrecisionOptimizer)
        assert optimizer.input_dim == 10
        assert optimizer.hidden_dim == 64
        assert optimizer.num_modalities == 5

    def test_create_adaptive_controller(self):
        """Test creating adaptive controller."""
        optimizer = create_precision_optimizer(
            "adaptive", num_modalities=3, context_dim=8)

        assert isinstance(optimizer, AdaptivePrecisionController)
        assert optimizer.num_modalities == 3
        assert optimizer.context_dim == 8

    def test_create_unknown_type(self):
        """Test creating with unknown type."""
        with pytest.raises(ValueError) as excinfo:
            create_precision_optimizer("unknown_type")

        assert "Unknown optimizer type: unknown_type" in str(excinfo.value)

    def test_create_with_defaults(self):
        """Test creating optimizers with minimal parameters."""
        # Should use default values
        gradient = create_precision_optimizer("gradient")
        assert gradient.num_modalities == 1

        hierarchical = create_precision_optimizer("hierarchical")
        assert hierarchical.level_dims == [1]

        meta = create_precision_optimizer("meta")
        assert meta.input_dim == 10
        assert meta.hidden_dim == 64
        assert meta.num_modalities == 1

        adaptive = create_precision_optimizer("adaptive")
        assert adaptive.num_modalities == 1
        assert adaptive.context_dim == 0

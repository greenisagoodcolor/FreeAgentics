"""Test precision optimizer implementations"""

import torch

from inference.engine.precision import GradientPrecisionOptimizer, PrecisionConfig


class TestGradientPrecisionOptimizer:
    """Test GradientPrecisionOptimizer functionality"""

    def test_initialization(self):
        """Test proper initialization of precision optimizer"""
        config = PrecisionConfig(
            init_precision=2.0,
            learning_rate=0.01,
            min_precision=0.1,
            max_precision=10.0)

        num_modalities = 3
        optimizer = GradientPrecisionOptimizer(config, num_modalities)

        # Check initial precision values
        initial_precision = torch.exp(optimizer.log_precision).detach()
        assert initial_precision.shape == (num_modalities,)
        assert torch.allclose(
            initial_precision,
            torch.ones(num_modalities) *
            config.init_precision)

    def test_optimize_precision(self):
        """Test precision optimization with various error patterns"""
        config = PrecisionConfig(
            init_precision=1.0,
            learning_rate=0.1,
            min_precision=0.1,
            max_precision=5.0)

        optimizer = GradientPrecisionOptimizer(config, num_modalities=2)

        # Test with small errors - precision should increase
        small_errors = torch.tensor([[0.1, 0.2]])
        initial = torch.exp(optimizer.log_precision).detach().clone()

        for _ in range(5):
            optimizer.optimize_precision(small_errors)

        # Precision should have increased for small errors
        final = torch.exp(optimizer.log_precision).detach()
        assert (final > initial).any()

        # Test with large errors - precision should decrease
        large_errors = torch.tensor([[2.0, 3.0]])
        initial = torch.exp(optimizer.log_precision).detach().clone()

        for _ in range(5):
            optimizer.optimize_precision(large_errors)

        final = torch.exp(optimizer.log_precision).detach()
        # Precision should have decreased for large errors
        assert (final < initial).any()

    def test_bounds_enforcement(self):
        """Test that precision stays within configured bounds"""
        config = PrecisionConfig(
            init_precision=1.0,
            learning_rate=1.0,  # High learning rate to test bounds
            min_precision=0.5,
            max_precision=2.0,
        )

        optimizer = GradientPrecisionOptimizer(config, num_modalities=1)

        # Apply many updates with large errors to push towards lower bound
        large_errors = torch.tensor([[10.0]])
        for _ in range(20):
            precision = optimizer.optimize_precision(large_errors)
            assert precision >= config.min_precision
            assert precision <= config.max_precision

        # Apply many updates with tiny errors to push towards upper bound
        tiny_errors = torch.tensor([[0.01]])
        for _ in range(20):
            precision = optimizer.optimize_precision(tiny_errors)
            assert precision >= config.min_precision
            assert precision <= config.max_precision

    def test_volatility_estimation(self):
        """Test volatility estimation functionality"""
        config = PrecisionConfig()
        optimizer = GradientPrecisionOptimizer(config, num_modalities=2)

        # Create error history with increasing variance
        error_history = []
        for i in range(15):
            variance = 0.1 + i * 0.1
            errors = torch.randn(2) * variance
            error_history.append(errors.abs())

        volatility = optimizer.estimate_volatility(error_history)

        # Check shape and positivity
        assert volatility.shape == (2,)
        assert (volatility > 0).all()

        # Test with constant errors - low volatility
        constant_history = [torch.ones(2) * 0.5 for _ in range(15)]
        const_volatility = optimizer.estimate_volatility(constant_history)

        # Volatility should be lower for constant errors
        assert (const_volatility < volatility).any()

    def test_batch_processing(self):
        """Test handling of batched prediction errors"""
        config = PrecisionConfig()
        optimizer = GradientPrecisionOptimizer(config, num_modalities=3)

        # Test various batch sizes
        for batch_size in [1, 5, 10]:
            batch_errors = torch.randn(batch_size, 3).abs()
            precision = optimizer.optimize_precision(batch_errors)

            # Should return precision for each modality
            assert precision.shape == (3,)
            assert (precision > 0).all()

    def test_gradient_clipping(self):
        """Test that gradient clipping prevents instability"""
        config = PrecisionConfig(
            learning_rate=1.0,
            gradient_clip=0.1)  # Small clip value

        optimizer = GradientPrecisionOptimizer(config, num_modalities=1)

        # Apply extreme errors
        extreme_errors = torch.tensor([[1000.0]])
        initial = torch.exp(optimizer.log_precision).detach().clone()

        # Should not explode due to gradient clipping
        optimizer.optimize_precision(extreme_errors)
        final = torch.exp(optimizer.log_precision).detach()

        # Change should be limited by gradient clipping
        change = torch.abs(final - initial)
        assert change < 1.0  # Should be a reasonable change despite extreme error

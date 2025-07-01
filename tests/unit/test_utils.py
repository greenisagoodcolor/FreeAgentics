"""
Tests for inference engine utility functions.
"""

import numpy as np
import pytest
import torch

from inference.engine.utils import normalize_beliefs, to_one_hot


class TestToOneHot:
    """Test one-hot encoding utility function."""

    def test_basic_one_hot_encoding(self):
        """Test basic one-hot encoding functionality."""
        # Single index
        indices = torch.tensor([0, 1, 2])
        num_classes = 3
        one_hot = to_one_hot(indices, num_classes)

        expected = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        assert torch.equal(one_hot, expected)
        assert one_hot.shape == (3, 3)

    def test_single_element_encoding(self):
        """Test encoding single element."""
        index = torch.tensor([1])
        num_classes = 4
        one_hot = to_one_hot(index, num_classes)

        expected = torch.tensor([[0, 1, 0, 0]])
        assert torch.equal(one_hot, expected)
        assert one_hot.shape == (1, 4)

    def test_scalar_encoding(self):
        """Test encoding scalar tensor."""
        index = torch.tensor(2)
        num_classes = 5
        one_hot = to_one_hot(index, num_classes)

        expected = torch.tensor([0, 0, 1, 0, 0])
        assert torch.equal(one_hot, expected)
        assert one_hot.shape == (5,)

    def test_batch_encoding(self):
        """Test batch one-hot encoding."""
        indices = torch.tensor([[0, 1], [2, 0]])
        num_classes = 3
        one_hot = to_one_hot(indices, num_classes)

        expected = torch.tensor([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]])

        assert torch.equal(one_hot, expected)
        assert one_hot.shape == (2, 2, 3)

    def test_edge_cases(self):
        """Test edge cases for one-hot encoding."""
        # Empty tensor
        empty_indices = torch.tensor([], dtype=torch.long)
        num_classes = 3
        one_hot = to_one_hot(empty_indices, num_classes)
        assert one_hot.shape == (0, 3)

        # Single class
        indices = torch.tensor([0, 0, 0])
        num_classes = 1
        one_hot = to_one_hot(indices, num_classes)
        expected = torch.tensor([[1], [1], [1]])
        assert torch.equal(one_hot, expected)

    def test_different_dtypes(self):
        """Test one-hot encoding with different tensor dtypes."""
        # Long tensor (most common)
        indices_long = torch.tensor([0, 1, 2], dtype=torch.long)
        one_hot_long = to_one_hot(indices_long, 3)

        # Int tensor
        indices_int = torch.tensor([0, 1, 2], dtype=torch.int)
        one_hot_int = to_one_hot(indices_int, 3)

        # Results should be identical
        assert torch.equal(one_hot_long, one_hot_int)


class TestNormalizeBeliefs:
    """Test belief normalization utility function."""

    def test_basic_normalization(self):
        """Test basic belief normalization."""
        beliefs = torch.tensor([1.0, 2.0, 3.0])
        normalized = normalize_beliefs(beliefs)

        # Should sum to 1
        assert torch.allclose(normalized.sum(), torch.tensor(1.0))

        # Should preserve relative proportions
        expected = torch.tensor([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
        assert torch.allclose(normalized, expected)

    def test_batch_normalization(self):
        """Test batch belief normalization."""
        beliefs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.1, 0.2, 0.7]])
        normalized = normalize_beliefs(beliefs)

        # Each row should sum to 1
        row_sums = normalized.sum(dim=-1)
        expected_sums = torch.ones(3)
        assert torch.allclose(row_sums, expected_sums)

        # Check specific values
        assert torch.allclose(normalized[0], torch.tensor([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]))
        assert torch.allclose(normalized[1], torch.tensor([4.0 / 15.0, 5.0 / 15.0, 6.0 / 15.0]))
        assert torch.allclose(normalized[2], torch.tensor([0.1, 0.2, 0.7]))  # Already normalized

    def test_zero_beliefs(self):
        """Test normalization of zero beliefs."""
        beliefs = torch.tensor([0.0, 0.0, 0.0])
        normalized = normalize_beliefs(beliefs)

        # Should handle division by zero gracefully
        assert torch.isnan(normalized).all()

    def test_single_element_beliefs(self):
        """Test normalization of single element."""
        beliefs = torch.tensor([5.0])
        normalized = normalize_beliefs(beliefs)

        assert torch.allclose(normalized, torch.tensor([1.0]))

    def test_uniform_beliefs(self):
        """Test normalization of uniform beliefs."""
        beliefs = torch.tensor([1.0, 1.0, 1.0, 1.0])
        normalized = normalize_beliefs(beliefs)

        expected = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert torch.allclose(normalized, expected)

    def test_negative_beliefs(self):
        """Test normalization with negative values."""
        beliefs = torch.tensor([-1.0, 2.0, 3.0])
        normalized = normalize_beliefs(beliefs)

        # Should still sum to 1
        assert torch.allclose(normalized.sum(), torch.tensor(1.0))

        # Check proportions
        total = -1.0 + 2.0 + 3.0  # = 4.0
        expected = torch.tensor([-1.0 / 4.0, 2.0 / 4.0, 3.0 / 4.0])
        assert torch.allclose(normalized, expected)

    def test_multidimensional_normalization(self):
        """Test normalization of multidimensional tensors."""
        # 3D tensor
        beliefs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        normalized = normalize_beliefs(beliefs)

        # Last dimension should sum to 1
        last_dim_sums = normalized.sum(dim=-1)
        expected_sums = torch.ones(2, 2)
        assert torch.allclose(last_dim_sums, expected_sums)

    def test_very_small_beliefs(self):
        """Test normalization with very small values."""
        beliefs = torch.tensor([1e-10, 2e-10, 3e-10])
        normalized = normalize_beliefs(beliefs)

        # Should still sum to 1
        assert torch.allclose(normalized.sum(), torch.tensor(1.0))

        # Should preserve relative proportions
        expected = torch.tensor([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
        assert torch.allclose(normalized, expected, rtol=1e-5)

    def test_very_large_beliefs(self):
        """Test normalization with very large values."""
        beliefs = torch.tensor([1e10, 2e10, 3e10])
        normalized = normalize_beliefs(beliefs)

        # Should still sum to 1
        assert torch.allclose(normalized.sum(), torch.tensor(1.0))

        # Should preserve relative proportions
        expected = torch.tensor([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
        assert torch.allclose(normalized, expected, rtol=1e-5)

    def test_preserve_dtype(self):
        """Test that normalization preserves tensor dtype."""
        # Float32
        beliefs_f32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        normalized_f32 = normalize_beliefs(beliefs_f32)
        assert normalized_f32.dtype == torch.float32

        # Float64
        beliefs_f64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        normalized_f64 = normalize_beliefs(beliefs_f64)
        assert normalized_f64.dtype == torch.float64

    def test_preserve_device(self):
        """Test that normalization preserves tensor device."""
        beliefs = torch.tensor([1.0, 2.0, 3.0])
        device = beliefs.device
        normalized = normalize_beliefs(beliefs)
        assert normalized.device == device


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_one_hot_then_normalize(self):
        """Test combining one-hot encoding with normalization."""
        indices = torch.tensor([0, 1, 2])
        one_hot = to_one_hot(indices, 3)

        # One-hot vectors are already normalized
        normalized = normalize_beliefs(one_hot.float())

        # Should be identical (within floating point precision)
        assert torch.allclose(normalized, one_hot.float())

        # Each row should sum to 1
        assert torch.allclose(normalized.sum(dim=-1), torch.ones(3))

    def test_normalize_then_one_hot_argmax(self):
        """Test normalize then convert back via argmax."""
        beliefs = torch.tensor([[0.1, 0.8, 0.1], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])

        normalized = normalize_beliefs(beliefs)
        argmax_indices = torch.argmax(normalized, dim=-1)
        reconstructed_one_hot = to_one_hot(argmax_indices, 3)

        # Argmax should select the highest probability class
        expected_indices = torch.tensor([1, 0, 2])
        assert torch.equal(argmax_indices, expected_indices)

        expected_one_hot = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        assert torch.equal(reconstructed_one_hot, expected_one_hot)

    def test_utilities_with_active_inference_data(self):
        """Test utilities with realistic Active Inference data."""
        # Simulate belief distribution over states
        num_states = 4
        num_timesteps = 3

        # Raw beliefs (unnormalized)
        raw_beliefs = torch.tensor(
            [
                [2.0, 1.0, 0.5, 0.1],  # Strong belief in state 0
                [1.0, 3.0, 1.0, 0.2],  # Strong belief in state 1
                [0.5, 1.0, 2.0, 1.5],  # Mixed beliefs
            ]
        )

        # Normalize beliefs
        beliefs = normalize_beliefs(raw_beliefs)

        # Check properties
        assert beliefs.shape == (num_timesteps, num_states)
        assert torch.allclose(beliefs.sum(dim=-1), torch.ones(num_timesteps))

        # Convert most likely states to one-hot
        most_likely_states = torch.argmax(beliefs, dim=-1)
        one_hot_states = to_one_hot(most_likely_states, num_states)

        # Check that argmax produces expected results
        expected_states = torch.tensor([0, 1, 2])  # Highest probability states
        assert torch.equal(most_likely_states, expected_states)

        # One-hot should be proper categorical distributions
        assert torch.equal(one_hot_states.sum(dim=-1), torch.ones(num_timesteps, dtype=torch.long))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

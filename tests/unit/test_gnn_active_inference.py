"""
Tests for GNN Active Inference Adapter.

Comprehensive test suite for the GNN-based Active Inference adapter module,
testing belief updates, policy selection, and integration with generative models.
"""

from unittest.mock import Mock

import pytest
import torch

from inference.gnn.active_inference import GNNActiveInferenceAdapter


class TestGNNActiveInferenceAdapter:
    """Test GNN Active Inference Adapter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the config
        self.mock_config = Mock()
        self.mock_config.hidden_dim = 64
        self.mock_config.num_layers = 3

        # Mock the generative model
        self.mock_generative_model = Mock()
        self.mock_generative_model.state_dim = 4
        self.mock_generative_model.obs_dim = 3

        # Mock the belief updater
        self.mock_belief_updater = Mock()

        # Mock the policy selector
        self.mock_policy_selector = Mock()

        # Create adapter
        self.adapter = GNNActiveInferenceAdapter(
            config=self.mock_config,
            generative_model=self.mock_generative_model,
            belief_updater=self.mock_belief_updater,
            policy_selector=self.mock_policy_selector,
        )

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.config == self.mock_config
        assert self.adapter.generative_model == self.mock_generative_model
        assert self.adapter.belief_updater == self.mock_belief_updater
        assert self.adapter.policy_selector == self.mock_policy_selector

    def test_update_beliefs_basic(self):
        """Test basic belief updating."""
        # Setup mock returns
        current_beliefs = torch.randn(4)
        observation = torch.randn(3)
        updated_beliefs = torch.randn(4)
        mock_policy = Mock()
        g_values = torch.randn(2)

        self.mock_belief_updater.update_beliefs.return_value = updated_beliefs
        self.mock_policy_selector.select_policy.return_value = (mock_policy, g_values)

        # Call update_beliefs
        result_beliefs, result_policy, result_g_values = self.adapter.update_beliefs(
            current_beliefs, observation
        )

        # Verify calls
        self.mock_belief_updater.update_beliefs.assert_called_once_with(
            current_beliefs, observation, self.mock_generative_model
        )
        self.mock_policy_selector.select_policy.assert_called_once_with(
            updated_beliefs, self.mock_generative_model, None
        )

        # Verify returns
        assert torch.equal(result_beliefs, updated_beliefs)
        assert result_policy == mock_policy
        assert torch.equal(result_g_values, g_values)

    def test_update_beliefs_with_preferences(self):
        """Test belief updating with preferences."""
        # Setup mock returns
        current_beliefs = torch.randn(4)
        observation = torch.randn(3)
        preferences = torch.randn(2)
        updated_beliefs = torch.randn(4)
        mock_policy = Mock()
        g_values = torch.randn(2)

        self.mock_belief_updater.update_beliefs.return_value = updated_beliefs
        self.mock_policy_selector.select_policy.return_value = (mock_policy, g_values)

        # Call update_beliefs with preferences
        result_beliefs, result_policy, result_g_values = self.adapter.update_beliefs(
            current_beliefs, observation, preferences
        )

        # Verify policy selector was called with preferences
        self.mock_policy_selector.select_policy.assert_called_once_with(
            updated_beliefs, self.mock_generative_model, preferences
        )

        # Verify returns
        assert torch.equal(result_beliefs, updated_beliefs)
        assert result_policy == mock_policy
        assert torch.equal(result_g_values, g_values)

    def test_update_beliefs_tensor_shapes(self):
        """Test belief updating with different tensor shapes."""
        # Test with batch dimension
        batch_size = 8
        current_beliefs = torch.randn(batch_size, 4)
        observation = torch.randn(batch_size, 3)
        updated_beliefs = torch.randn(batch_size, 4)
        mock_policy = Mock()
        g_values = torch.randn(batch_size, 2)

        self.mock_belief_updater.update_beliefs.return_value = updated_beliefs
        self.mock_policy_selector.select_policy.return_value = (mock_policy, g_values)

        # Call update_beliefs
        result_beliefs, result_policy, result_g_values = self.adapter.update_beliefs(
            current_beliefs, observation
        )

        # Verify tensor shapes are preserved
        assert result_beliefs.shape == (batch_size, 4)
        assert result_g_values.shape == (batch_size, 2)
        assert torch.equal(result_beliefs, updated_beliefs)
        assert torch.equal(result_g_values, g_values)

    def test_update_beliefs_error_handling(self):
        """Test error handling in belief updating."""
        current_beliefs = torch.randn(4)
        observation = torch.randn(3)

        # Mock belief updater to raise exception
        self.mock_belief_updater.update_beliefs.side_effect = RuntimeError("Belief update failed")

        # Should propagate the error
        with pytest.raises(RuntimeError, match="Belief update failed"):
            self.adapter.update_beliefs(current_beliefs, observation)

    def test_update_beliefs_policy_selector_error(self):
        """Test error handling in policy selection."""
        current_beliefs = torch.randn(4)
        observation = torch.randn(3)
        updated_beliefs = torch.randn(4)

        # Mock successful belief update but failed policy selection
        self.mock_belief_updater.update_beliefs.return_value = updated_beliefs
        self.mock_policy_selector.select_policy.side_effect = ValueError("Policy selection failed")

        # Should propagate the error
        with pytest.raises(ValueError, match="Policy selection failed"):
            self.adapter.update_beliefs(current_beliefs, observation)

    def test_update_beliefs_none_preferences(self):
        """Test belief updating with None preferences."""
        current_beliefs = torch.randn(4)
        observation = torch.randn(3)
        updated_beliefs = torch.randn(4)
        mock_policy = Mock()
        g_values = torch.randn(2)

        self.mock_belief_updater.update_beliefs.return_value = updated_beliefs
        self.mock_policy_selector.select_policy.return_value = (mock_policy, g_values)

        # Call with explicit None preferences
        result_beliefs, result_policy, result_g_values = self.adapter.update_beliefs(
            current_beliefs, observation, preferences=None
        )

        # Verify policy selector was called with None
        self.mock_policy_selector.select_policy.assert_called_once_with(
            updated_beliefs, self.mock_generative_model, None
        )

    def test_update_beliefs_return_type_annotation(self):
        """Test that return type matches type annotation."""
        current_beliefs = torch.randn(4)
        observation = torch.randn(3)
        updated_beliefs = torch.randn(4)
        mock_policy = Mock()
        g_values = torch.randn(2)

        self.mock_belief_updater.update_beliefs.return_value = updated_beliefs
        self.mock_policy_selector.select_policy.return_value = (mock_policy, g_values)

        result = self.adapter.update_beliefs(current_beliefs, observation)

        # Should return a tuple with 3 elements
        assert isinstance(result, tuple)
        assert len(result) == 3

        result_beliefs, result_policy, result_g_values = result
        assert isinstance(result_beliefs, torch.Tensor)
        assert result_policy is not None  # Policy mock object
        assert isinstance(result_g_values, torch.Tensor)


class TestGNNActiveInferenceAdapterIntegration:
    """Integration tests for GNN Active Inference Adapter."""

    def test_adapter_with_realistic_components(self):
        """Test adapter with more realistic mock components."""
        # Create more realistic mocks
        config = Mock()
        config.hidden_dim = 64
        config.num_layers = 2
        config.dropout = 0.1

        generative_model = Mock()
        generative_model.state_dim = 5
        generative_model.obs_dim = 4
        generative_model.action_dim = 3

        belief_updater = Mock()
        policy_selector = Mock()

        # Create adapter
        adapter = GNNActiveInferenceAdapter(
            config=config,
            generative_model=generative_model,
            belief_updater=belief_updater,
            policy_selector=policy_selector,
        )

        # Test full workflow
        current_beliefs = torch.softmax(torch.randn(5), dim=0)  # Proper belief distribution
        observation = torch.randn(4)
        preferences = torch.randn(3)

        # Mock realistic returns
        updated_beliefs = torch.softmax(torch.randn(5), dim=0)
        policy_mock = Mock()
        policy_mock.action_probabilities = torch.softmax(torch.randn(3), dim=0)
        g_values = torch.randn(3)

        belief_updater.update_beliefs.return_value = updated_beliefs
        policy_selector.select_policy.return_value = (policy_mock, g_values)

        # Execute
        result_beliefs, result_policy, result_g_values = adapter.update_beliefs(
            current_beliefs, observation, preferences
        )

        # Verify components were called with correct arguments
        belief_updater.update_beliefs.assert_called_once_with(
            current_beliefs, observation, generative_model
        )
        policy_selector.select_policy.assert_called_once_with(
            updated_beliefs, generative_model, preferences
        )

        # Verify results have expected properties
        assert torch.allclose(
            result_beliefs.sum(), torch.tensor(1.0), atol=1e-6
        )  # Belief sums to 1
        assert hasattr(result_policy, "action_probabilities")
        assert result_g_values.shape == (3,)

    def test_adapter_workflow_multiple_steps(self):
        """Test adapter through multiple belief update steps."""
        # Setup
        config = Mock()
        generative_model = Mock()
        belief_updater = Mock()
        policy_selector = Mock()

        adapter = GNNActiveInferenceAdapter(
            config, generative_model, belief_updater, policy_selector
        )

        # Initial beliefs
        beliefs = torch.softmax(torch.randn(4), dim=0)

        # Simulate multiple time steps
        for t in range(5):
            observation = torch.randn(3)

            # Mock updated beliefs that gradually concentrate
            concentration = 1.0 + t * 0.5
            updated_beliefs = torch.softmax(torch.randn(4) * concentration, dim=0)

            mock_policy = Mock()
            mock_policy.timestep = t
            g_values = torch.randn(2)

            belief_updater.update_beliefs.return_value = updated_beliefs
            policy_selector.select_policy.return_value = (mock_policy, g_values)

            # Update beliefs
            beliefs, policy, g_vals = adapter.update_beliefs(beliefs, observation)

            # Verify beliefs are valid distributions
            assert torch.allclose(beliefs.sum(), torch.tensor(1.0), atol=1e-6)
            assert torch.all(beliefs >= 0)

            # Verify policy is returned
            assert policy.timestep == t

        # Verify belief updater was called 5 times
        assert belief_updater.update_beliefs.call_count == 5
        assert policy_selector.select_policy.call_count == 5

    def test_adapter_memory_efficiency(self):
        """Test that adapter doesn't accumulate unnecessary tensors."""
        config = Mock()
        generative_model = Mock()
        belief_updater = Mock()
        policy_selector = Mock()

        adapter = GNNActiveInferenceAdapter(
            config, generative_model, belief_updater, policy_selector
        )

        # Setup mocks to return new tensors each time
        def make_new_beliefs(*args):
            return torch.randn(4)

        def make_new_policy(*args):
            return Mock(), torch.randn(2)

        belief_updater.update_beliefs.side_effect = make_new_beliefs
        policy_selector.select_policy.side_effect = make_new_policy

        # Run multiple updates
        beliefs = torch.randn(4)
        observation = torch.randn(3)

        for _ in range(10):
            beliefs, policy, g_values = adapter.update_beliefs(beliefs, observation)

            # Each call should return new tensor instances
            # (In real implementation, this helps prevent memory leaks)
            assert isinstance(beliefs, torch.Tensor)
            assert isinstance(g_values, torch.Tensor)


class TestGNNActiveInferenceAdapterEdgeCases:
    """Test edge cases and error conditions."""

    def test_adapter_with_zero_dimensional_tensors(self):
        """Test adapter with scalar tensors."""
        config = Mock()
        generative_model = Mock()
        belief_updater = Mock()
        policy_selector = Mock()

        adapter = GNNActiveInferenceAdapter(
            config, generative_model, belief_updater, policy_selector
        )

        # Test with scalar inputs
        current_beliefs = torch.tensor(1.0)  # Scalar
        observation = torch.tensor(0.5)  # Scalar

        belief_updater.update_beliefs.return_value = torch.tensor(0.8)
        policy_selector.select_policy.return_value = (Mock(), torch.tensor(0.3))

        result_beliefs, result_policy, result_g_values = adapter.update_beliefs(
            current_beliefs, observation
        )

        assert result_beliefs.dim() == 0  # Scalar
        assert result_g_values.dim() == 0  # Scalar

    def test_adapter_with_empty_tensors(self):
        """Test adapter behavior with empty tensors."""
        config = Mock()
        generative_model = Mock()
        belief_updater = Mock()
        policy_selector = Mock()

        adapter = GNNActiveInferenceAdapter(
            config, generative_model, belief_updater, policy_selector
        )

        # Test with empty tensors
        current_beliefs = torch.empty(0)
        observation = torch.empty(0)

        belief_updater.update_beliefs.return_value = torch.empty(0)
        policy_selector.select_policy.return_value = (Mock(), torch.empty(0))

        result_beliefs, result_policy, result_g_values = adapter.update_beliefs(
            current_beliefs, observation
        )

        assert result_beliefs.numel() == 0
        assert result_g_values.numel() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Comprehensive tests for GNN Integration with Active Inference.

Tests the sophisticated graph neural network integration system that bridges
GNN layers with Active Inference, providing adapters and mappers for translating
between graph representations and AI states.
"""

# Mock the problematic GNN layers to avoid import issues
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn


# Mock the GNN layers that may cause import issues
class MockGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)


class MockGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)


class MockSAGELayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)


# Mock the gnn.layers module
gnn_layers_mock = Mock()
gnn_layers_mock.GCNLayer = MockGCNLayer
gnn_layers_mock.GATLayer = MockGATLayer
gnn_layers_mock.SAGELayer = MockSAGELayer
sys.modules["inference.gnn.layers"] = gnn_layers_mock

from inference.engine.active_inference import InferenceConfig
from inference.engine.generative_model import (
    DiscreteGenerativeModel,
    ModelDimensions,
    ModelParameters,
)
from inference.engine.gnn_integration import (
    DirectGraphMapper,
    GNNActiveInferenceAdapter,
    GNNIntegrationConfig,
    GraphFeatureAggregator,
    GraphToStateMapper,
    LearnedGraphMapper,
)


class TestGNNIntegrationConfig:
    """Test GNNIntegrationConfig configuration class."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = GNNIntegrationConfig()

        assert config.gnn_type == "gcn"
        assert config.num_layers == 3
        assert config.hidden_dim == 64
        assert config.output_dim == 32
        assert config.dropout == 0.1
        assert config.aggregation_method == "mean"
        assert config.use_edge_features is True
        assert config.use_global_features is True
        assert config.state_mapping == "direct"
        assert config.observation_mapping == "learned"
        assert config.use_gpu is True
        assert config.dtype == torch.float32
        assert config.eps == 1e-16

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = GNNIntegrationConfig(
            gnn_type="gat",
            num_layers=5,
            hidden_dim=128,
            output_dim=64,
            dropout=0.2,
            aggregation_method="attention",
            use_edge_features=False,
            use_global_features=False,
            state_mapping="learned",
            observation_mapping="direct",
            use_gpu=False,
            dtype=torch.float64,
            eps=1e-8,
        )

        assert config.gnn_type == "gat"
        assert config.num_layers == 5
        assert config.hidden_dim == 128
        assert config.output_dim == 64
        assert config.dropout == 0.2
        assert config.aggregation_method == "attention"
        assert config.use_edge_features is False
        assert config.use_global_features is False
        assert config.state_mapping == "learned"
        assert config.observation_mapping == "direct"
        assert config.use_gpu is False
        assert config.dtype == torch.float64
        assert config.eps == 1e-8


class TestDirectGraphMapper:
    """Test DirectGraphMapper functionality."""

    def test_direct_mapper_creation_no_projection(self):
        """Test creating DirectGraphMapper without projection layers."""
        config = GNNIntegrationConfig(output_dim=32, use_gpu=False)
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=32)

        assert mapper.state_dim == 32
        assert mapper.observation_dim == 32
        assert mapper.state_projection is None
        assert mapper.obs_projection is None

    def test_direct_mapper_creation_with_projection(self):
        """Test creating DirectGraphMapper with projection layers."""
        config = GNNIntegrationConfig(output_dim=32, use_gpu=False)
        mapper = DirectGraphMapper(config, state_dim=16, observation_dim=24)

        assert mapper.state_dim == 16
        assert mapper.observation_dim == 24
        assert mapper.state_projection is not None
        assert mapper.obs_projection is not None
        assert mapper.state_projection.in_features == 32
        assert mapper.state_projection.out_features == 16
        assert mapper.obs_projection.in_features == 32
        assert mapper.obs_projection.out_features == 24

    def test_direct_mapper_map_to_states(self):
        """Test mapping graph features to states."""
        config = GNNIntegrationConfig(output_dim=32, use_gpu=False, state_mapping="direct")
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=32)

        graph_features = torch.randn(5, 32)
        states = mapper.map_to_states(graph_features)

        assert states.shape == (5, 32)
        # Check softmax normalization for direct mapping
        assert torch.allclose(states.sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_direct_mapper_map_to_states_with_indices(self):
        """Test mapping graph features to states with node indices."""
        config = GNNIntegrationConfig(output_dim=32, use_gpu=False)
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=32)

        graph_features = torch.randn(10, 32)
        node_indices = torch.tensor([0, 2, 4])
        states = mapper.map_to_states(graph_features, node_indices)

        assert states.shape == (3, 32)

    def test_direct_mapper_map_to_observations(self):
        """Test mapping graph features to observations."""
        config = GNNIntegrationConfig(output_dim=32, use_gpu=False)
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=32)

        graph_features = torch.randn(5, 32)
        observations = mapper.map_to_observations(graph_features)

        assert observations.shape == (5, 32)

    def test_direct_mapper_map_to_observations_with_projection(self):
        """Test mapping graph features to observations with projection."""
        config = GNNIntegrationConfig(output_dim=32, use_gpu=False)
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=16)

        graph_features = torch.randn(5, 32)
        observations = mapper.map_to_observations(graph_features)

        assert observations.shape == (5, 16)


class TestLearnedGraphMapper:
    """Test LearnedGraphMapper functionality."""

    def test_learned_mapper_creation(self):
        """Test creating LearnedGraphMapper."""
        config = GNNIntegrationConfig(output_dim=32, hidden_dim=64, dropout=0.1, use_gpu=False)
        mapper = LearnedGraphMapper(config, state_dim=16, observation_dim=24)

        assert mapper.state_dim == 16
        assert mapper.observation_dim == 24
        assert hasattr(mapper, "state_mapper")
        assert hasattr(mapper, "obs_mapper")
        assert isinstance(mapper.state_mapper, nn.Sequential)
        assert isinstance(mapper.obs_mapper, nn.Sequential)

    def test_learned_mapper_map_to_states(self):
        """Test mapping graph features to states using learned mapping."""
        config = GNNIntegrationConfig(output_dim=32, hidden_dim=64, use_gpu=False)
        mapper = LearnedGraphMapper(config, state_dim=16, observation_dim=24)

        graph_features = torch.randn(5, 32)
        states = mapper.map_to_states(graph_features)

        assert states.shape == (5, 16)
        # Check softmax normalization
        assert torch.allclose(states.sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_learned_mapper_map_to_observations(self):
        """Test mapping graph features to observations using learned mapping."""
        config = GNNIntegrationConfig(output_dim=32, hidden_dim=64, use_gpu=False)
        mapper = LearnedGraphMapper(config, state_dim=16, observation_dim=24)

        graph_features = torch.randn(5, 32)
        observations = mapper.map_to_observations(graph_features)

        assert observations.shape == (5, 24)

    def test_learned_mapper_with_node_indices(self):
        """Test learned mapper with node indices."""
        config = GNNIntegrationConfig(output_dim=32, hidden_dim=64, use_gpu=False)
        mapper = LearnedGraphMapper(config, state_dim=16, observation_dim=24)

        graph_features = torch.randn(10, 32)
        node_indices = torch.tensor([1, 3, 5, 7])

        states = mapper.map_to_states(graph_features, node_indices)
        observations = mapper.map_to_observations(graph_features, node_indices)

        assert states.shape == (4, 16)
        assert observations.shape == (4, 24)


class TestGraphFeatureAggregator:
    """Test GraphFeatureAggregator functionality."""

    def test_aggregator_creation_mean(self):
        """Test creating aggregator with mean aggregation."""
        config = GNNIntegrationConfig(aggregation_method="mean", output_dim=32, use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        assert aggregator.config.aggregation_method == "mean"

    def test_aggregator_creation_attention(self):
        """Test creating aggregator with attention aggregation."""
        config = GNNIntegrationConfig(
            aggregation_method="attention", output_dim=32, hidden_dim=64, use_gpu=False
        )
        aggregator = GraphFeatureAggregator(config)

        assert aggregator.config.aggregation_method == "attention"
        assert hasattr(aggregator, "attention")

    def test_aggregate_mean(self):
        """Test mean aggregation."""
        config = GNNIntegrationConfig(aggregation_method="mean", use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(6, 32)
        batch = torch.tensor([0, 0, 1, 1, 1, 2])  # 3 graphs with 2, 3, 1 nodes

        result = aggregator.aggregate(node_features, batch)

        assert result.shape == (3, 32)
        # Check that mean is computed correctly for first graph
        expected_mean = node_features[:2].mean(dim=0)
        assert torch.allclose(result[0], expected_mean, atol=1e-6)

    def test_aggregate_max(self):
        """Test max aggregation."""
        config = GNNIntegrationConfig(aggregation_method="max", use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(4, 32)
        batch = torch.tensor([0, 0, 1, 1])  # 2 graphs with 2 nodes each

        result = aggregator.aggregate(node_features, batch)

        assert result.shape == (2, 32)
        # Check that max is computed correctly
        expected_max = node_features[:2].max(dim=0)[0]
        assert torch.allclose(result[0], expected_max, atol=1e-6)

    def test_aggregate_sum(self):
        """Test sum aggregation."""
        config = GNNIntegrationConfig(aggregation_method="sum", use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(4, 32)
        batch = torch.tensor([0, 0, 1, 1])  # 2 graphs with 2 nodes each

        result = aggregator.aggregate(node_features, batch)

        assert result.shape == (2, 32)
        # Check that sum is computed correctly
        expected_sum = node_features[:2].sum(dim=0)
        assert torch.allclose(result[0], expected_sum, atol=1e-6)

    def test_aggregate_attention(self):
        """Test attention aggregation."""
        config = GNNIntegrationConfig(
            aggregation_method="attention", output_dim=32, hidden_dim=16, use_gpu=False
        )
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(4, 32)
        batch = torch.tensor([0, 0, 1, 1])  # 2 graphs with 2 nodes each

        result = aggregator.aggregate(node_features, batch)

        assert result.shape == (2, 32)

    def test_aggregate_single_mean(self):
        """Test single graph mean aggregation."""
        config = GNNIntegrationConfig(aggregation_method="mean", use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(5, 32)
        result = aggregator.aggregate_single(node_features)

        assert result.shape == (1, 32)
        expected_mean = node_features.mean(dim=0, keepdim=True)
        assert torch.allclose(result, expected_mean, atol=1e-6)

    def test_aggregate_single_max(self):
        """Test single graph max aggregation."""
        config = GNNIntegrationConfig(aggregation_method="max", use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(5, 32)
        result = aggregator.aggregate_single(node_features)

        assert result.shape == (1, 32)
        expected_max = node_features.max(dim=0, keepdim=True)[0]
        assert torch.allclose(result, expected_max, atol=1e-6)

    def test_aggregate_single_sum(self):
        """Test single graph sum aggregation."""
        config = GNNIntegrationConfig(aggregation_method="sum", use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(5, 32)
        result = aggregator.aggregate_single(node_features)

        assert result.shape == (1, 32)
        expected_sum = node_features.sum(dim=0, keepdim=True)
        assert torch.allclose(result, expected_sum, atol=1e-6)


class DummyGNN(nn.Module):
    """Dummy GNN for testing GNNActiveInferenceAdapter."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.linear(x)


class TestGNNActiveInferenceAdapter:
    """Test GNNActiveInferenceAdapter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = GNNIntegrationConfig(
            output_dim=32, hidden_dim=64, use_gpu=False, state_mapping="direct"
        )
        self.gnn_model = DummyGNN(10, 32)

        # Create mock generative model and inference algorithm
        self.dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        self.params = ModelParameters(use_gpu=False)
        self.gen_model = DiscreteGenerativeModel(self.dims, self.params)

        # Mock inference algorithm
        self.inference = Mock()
        self.inference.infer_states = Mock(return_value=torch.randn(4))

    def test_adapter_creation(self):
        """Test creating GNN-Active Inference adapter."""
        adapter = GNNActiveInferenceAdapter(
            self.config, self.gnn_model, self.gen_model, self.inference
        )

        assert adapter.config == self.config
        assert adapter.gnn_model == self.gnn_model
        assert adapter.generative_model == self.gen_model
        assert adapter.inference == self.inference
        assert hasattr(adapter, "mapper")
        assert hasattr(adapter, "aggregator")

    def test_adapter_process_graph(self):
        """Test processing graph data."""
        adapter = GNNActiveInferenceAdapter(
            self.config, self.gnn_model, self.gen_model, self.inference
        )

        node_features = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])

        graph_data = adapter.process_graph(node_features, edge_index)

        assert "node_features" in graph_data
        assert "edge_index" in graph_data
        assert "graph_features" in graph_data
        assert graph_data["node_features"].shape == (5, 32)
        assert torch.equal(graph_data["edge_index"], edge_index)

    def test_adapter_process_graph_with_edge_features(self):
        """Test processing graph data with edge features."""
        adapter = GNNActiveInferenceAdapter(
            self.config, self.gnn_model, self.gen_model, self.inference
        )

        node_features = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        edge_features = torch.randn(5, 8)

        graph_data = adapter.process_graph(node_features, edge_index, edge_features)

        # Edge features are passed to GNN but not returned in graph_data dict
        assert "node_features" in graph_data
        assert "graph_features" in graph_data
        assert "edge_index" in graph_data

    def test_adapter_graph_to_beliefs(self):
        """Test converting graph data to beliefs."""
        adapter = GNNActiveInferenceAdapter(
            self.config, self.gnn_model, self.gen_model, self.inference
        )

        graph_data = {
            "node_features": torch.randn(5, 32),
            "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
            "graph_features": torch.randn(1, 32),
        }

        beliefs = adapter.graph_to_beliefs(graph_data)

        assert beliefs.shape == (1, 4)  # (1, num_states) from graph aggregation

    def test_adapter_graph_to_observations(self):
        """Test converting graph data to observations."""
        adapter = GNNActiveInferenceAdapter(
            self.config, self.gnn_model, self.gen_model, self.inference
        )

        graph_data = {
            "node_features": torch.randn(5, 32),
            "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
            "graph_features": torch.randn(1, 32),
        }

        observations = adapter.graph_to_observations(graph_data)

        assert observations.shape == (1, 3)  # (1, num_observations) from graph aggregation

    def test_adapter_update_beliefs_with_graph(self):
        """Test updating beliefs with graph data."""
        adapter = GNNActiveInferenceAdapter(
            self.config, self.gnn_model, self.gen_model, self.inference
        )

        current_beliefs = torch.randn(4)
        graph_data = {
            "node_features": torch.randn(5, 32),
            "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
            "graph_features": torch.randn(1, 32),
        }

        updated_beliefs = adapter.update_beliefs_with_graph(current_beliefs, graph_data)

        assert updated_beliefs.shape == (4,)

        # Verify inference was called
        self.inference.infer_states.assert_called_once()

    def test_adapter_compute_expected_free_energy_with_graph(self):
        """Test computing expected free energy with graph data."""
        adapter = GNNActiveInferenceAdapter(
            self.config, self.gnn_model, self.gen_model, self.inference
        )

        policy = Mock()
        graph_data = {
            "node_features": torch.randn(5, 32),
            "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
            "graph_features": torch.randn(1, 32),
        }

        efe = adapter.compute_expected_free_energy_with_graph(policy, graph_data)

        assert isinstance(efe, torch.Tensor)
        assert efe.dim() == 0  # scalar


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_end_to_end_gcn_integration(self):
        """Test end-to-end GCN integration scenario."""
        # Setup
        config = GNNIntegrationConfig(
            gnn_type="gcn", output_dim=16, use_gpu=False, state_mapping="direct"
        )
        gnn_model = DummyGNN(8, 16)
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        gen_model = DiscreteGenerativeModel(dims, params)

        # Mock inference
        inference = Mock()
        inference.infer_states = Mock(return_value=torch.randn(4))

        # Create adapter
        adapter = GNNActiveInferenceAdapter(config, gnn_model, gen_model, inference)

        # Process graph
        node_features = torch.randn(6, 8)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])

        graph_data = adapter.process_graph(node_features, edge_index)
        beliefs = adapter.graph_to_beliefs(graph_data)

        assert beliefs.shape == (1, 4)  # (1, num_states) from graph aggregation
        assert torch.allclose(
            beliefs.sum(dim=-1), torch.tensor(1.0), atol=1e-6
        )  # Softmax normalized

    def test_end_to_end_learned_mapping(self):
        """Test end-to-end scenario with learned mapping."""
        config = GNNIntegrationConfig(
            state_mapping="learned",
            observation_mapping="learned",
            output_dim=24,
            hidden_dim=48,
            use_gpu=False,
        )
        gnn_model = DummyGNN(12, 24)
        dims = ModelDimensions(num_states=6, num_observations=4, num_actions=3)
        params = ModelParameters(use_gpu=False)
        gen_model = DiscreteGenerativeModel(dims, params)

        # Mock inference
        inference = Mock()
        inference.infer_states = Mock(return_value=torch.randn(6))

        adapter = GNNActiveInferenceAdapter(config, gnn_model, gen_model, inference)

        # Test complete workflow
        node_features = torch.randn(8, 12)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])

        graph_data = adapter.process_graph(node_features, edge_index)
        beliefs = adapter.graph_to_beliefs(graph_data)

        assert beliefs.shape == (1, 6)  # (1, num_states) from graph aggregation

    def test_belief_update_workflow(self):
        """Test belief update workflow with graph integration."""
        config = GNNIntegrationConfig(use_gpu=False, output_dim=16)
        gnn_model = DummyGNN(10, 16)
        dims = ModelDimensions(num_states=5, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        gen_model = DiscreteGenerativeModel(dims, params)

        # Mock inference
        inference = Mock()
        inference.infer_states = Mock(return_value=torch.randn(5))

        adapter = GNNActiveInferenceAdapter(config, gnn_model, gen_model, inference)

        # Initial beliefs
        current_beliefs = torch.softmax(torch.randn(5), dim=0)

        # Graph observation
        node_features = torch.randn(4, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        graph_data = adapter.process_graph(node_features, edge_index)

        # Update beliefs
        updated_beliefs = adapter.update_beliefs_with_graph(current_beliefs, graph_data)

        assert updated_beliefs.shape == (5,)
        # Verify inference was called with observations
        assert inference.infer_states.called

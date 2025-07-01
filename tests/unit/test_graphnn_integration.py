"""
Simplified tests for GraphNN Integration with Active Inference.

Tests the core graph neural network integration functionality including
configuration, layers, and basic mapping functionality.
"""

from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn

from inference.engine.graphnn_integration import (
    DirectGraphMapper,
    GATLayer,
    GCNLayer,
    GraphFeatureAggregator,
    GraphNNIntegrationConfig,
    GraphSAGELayer,
    LearnedGraphMapper,
)


class TestGraphNNIntegrationConfig:
    """Test GraphNNIntegrationConfig configuration class."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = GraphNNIntegrationConfig()

        assert config.graphnn_type == "gcn"
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
        config = GraphNNIntegrationConfig(
            graphnn_type="gat",
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

        assert config.graphnn_type == "gat"
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


class TestGraphNNLayers:
    """Test stub GraphNN layer implementations."""

    def test_gcn_layer_creation(self):
        """Test GCN layer creation."""
        layer = GCNLayer(10, 20)

        assert hasattr(layer, "linear")
        assert layer.linear.in_features == 10
        assert layer.linear.out_features == 20

    def test_gcn_layer_forward(self):
        """Test GCN layer forward pass."""
        layer = GCNLayer(10, 20)
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

        output = layer.forward(x, edge_index)

        assert output.shape == (5, 20)
        assert isinstance(output, torch.Tensor)

    def test_gat_layer_creation(self):
        """Test GAT layer creation."""
        layer = GATLayer(15, 25)

        assert hasattr(layer, "linear")
        assert layer.linear.in_features == 15
        assert layer.linear.out_features == 25

    def test_gat_layer_forward(self):
        """Test GAT layer forward pass."""
        layer = GATLayer(15, 25)
        x = torch.randn(3, 15)
        edge_index = torch.tensor([[0, 1], [1, 2]])

        output = layer.forward(x, edge_index)

        assert output.shape == (3, 25)
        assert isinstance(output, torch.Tensor)

    def test_graphsage_layer_creation(self):
        """Test GraphSAGE layer creation."""
        layer = GraphSAGELayer(8, 16)

        assert hasattr(layer, "linear")
        assert layer.linear.in_features == 8
        assert layer.linear.out_features == 16

    def test_graphsage_layer_forward(self):
        """Test GraphSAGE layer forward pass."""
        layer = GraphSAGELayer(8, 16)
        x = torch.randn(4, 8)

        output = layer.forward(x)

        assert output.shape == (4, 16)
        assert isinstance(output, torch.Tensor)


class TestDirectGraphMapper:
    """Test DirectGraphMapper functionality."""

    def test_direct_mapper_creation_no_projection(self):
        """Test creating DirectGraphMapper without projection layers."""
        config = GraphNNIntegrationConfig(output_dim=32, use_gpu=False)
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=32)

        assert mapper.state_dim == 32
        assert mapper.observation_dim == 32
        assert mapper.state_projection is None
        assert mapper.obs_projection is None

    def test_direct_mapper_creation_with_projection(self):
        """Test creating DirectGraphMapper with projection layers."""
        config = GraphNNIntegrationConfig(output_dim=32, use_gpu=False)
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
        config = GraphNNIntegrationConfig(output_dim=32, use_gpu=False, state_mapping="direct")
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=32)

        graph_features = torch.randn(5, 32)
        states = mapper.map_to_states(graph_features)

        assert states.shape == (5, 32)
        # Check softmax normalization for direct mapping
        assert torch.allclose(states.sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_direct_mapper_map_to_states_with_indices(self):
        """Test mapping graph features to states with node indices."""
        config = GraphNNIntegrationConfig(output_dim=32, use_gpu=False)
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=32)

        graph_features = torch.randn(10, 32)
        node_indices = torch.tensor([0, 2, 4])
        states = mapper.map_to_states(graph_features, node_indices)

        assert states.shape == (3, 32)

    def test_direct_mapper_map_to_observations(self):
        """Test mapping graph features to observations."""
        config = GraphNNIntegrationConfig(output_dim=32, use_gpu=False)
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=32)

        graph_features = torch.randn(5, 32)
        observations = mapper.map_to_observations(graph_features)

        assert observations.shape == (5, 32)

    def test_direct_mapper_map_to_observations_with_projection(self):
        """Test mapping graph features to observations with projection."""
        config = GraphNNIntegrationConfig(output_dim=32, use_gpu=False)
        mapper = DirectGraphMapper(config, state_dim=32, observation_dim=16)

        graph_features = torch.randn(5, 32)
        observations = mapper.map_to_observations(graph_features)

        assert observations.shape == (5, 16)


class TestLearnedGraphMapper:
    """Test LearnedGraphMapper functionality."""

    def test_learned_mapper_creation(self):
        """Test creating LearnedGraphMapper."""
        config = GraphNNIntegrationConfig(output_dim=32, hidden_dim=64, dropout=0.1, use_gpu=False)
        mapper = LearnedGraphMapper(config, state_dim=16, observation_dim=24)

        assert mapper.state_dim == 16
        assert mapper.observation_dim == 24
        assert hasattr(mapper, "state_mapper")
        assert hasattr(mapper, "obs_mapper")
        assert isinstance(mapper.state_mapper, nn.Sequential)
        assert isinstance(mapper.obs_mapper, nn.Sequential)

    def test_learned_mapper_map_to_states(self):
        """Test mapping graph features to states using learned mapping."""
        config = GraphNNIntegrationConfig(output_dim=32, hidden_dim=64, use_gpu=False)
        mapper = LearnedGraphMapper(config, state_dim=16, observation_dim=24)

        graph_features = torch.randn(5, 32)
        states = mapper.map_to_states(graph_features)

        assert states.shape == (5, 16)
        # Check softmax normalization
        assert torch.allclose(states.sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_learned_mapper_map_to_observations(self):
        """Test mapping graph features to observations using learned mapping."""
        config = GraphNNIntegrationConfig(output_dim=32, hidden_dim=64, use_gpu=False)
        mapper = LearnedGraphMapper(config, state_dim=16, observation_dim=24)

        graph_features = torch.randn(5, 32)
        observations = mapper.map_to_observations(graph_features)

        assert observations.shape == (5, 24)

    def test_learned_mapper_with_node_indices(self):
        """Test learned mapper with node indices."""
        config = GraphNNIntegrationConfig(output_dim=32, hidden_dim=64, use_gpu=False)
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
        config = GraphNNIntegrationConfig(aggregation_method="mean", output_dim=32, use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        assert aggregator.config.aggregation_method == "mean"

    def test_aggregator_creation_attention(self):
        """Test creating aggregator with attention aggregation."""
        config = GraphNNIntegrationConfig(
            aggregation_method="attention", output_dim=32, hidden_dim=64, use_gpu=False
        )
        aggregator = GraphFeatureAggregator(config)

        assert aggregator.config.aggregation_method == "attention"
        assert hasattr(aggregator, "attention")

    def test_aggregate_mean(self):
        """Test mean aggregation."""
        config = GraphNNIntegrationConfig(aggregation_method="mean", use_gpu=False)
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
        config = GraphNNIntegrationConfig(aggregation_method="max", use_gpu=False)
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
        config = GraphNNIntegrationConfig(aggregation_method="sum", use_gpu=False)
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
        config = GraphNNIntegrationConfig(
            aggregation_method="attention", output_dim=32, hidden_dim=16, use_gpu=False
        )
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(4, 32)
        batch = torch.tensor([0, 0, 1, 1])  # 2 graphs with 2 nodes each

        result = aggregator.aggregate(node_features, batch)

        assert result.shape == (2, 32)

    def test_aggregate_single_mean(self):
        """Test single graph mean aggregation."""
        config = GraphNNIntegrationConfig(aggregation_method="mean", use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(5, 32)
        result = aggregator.aggregate_single(node_features)

        assert result.shape == (1, 32)
        expected_mean = node_features.mean(dim=0, keepdim=True)
        assert torch.allclose(result, expected_mean, atol=1e-6)

    def test_aggregate_single_max(self):
        """Test single graph max aggregation."""
        config = GraphNNIntegrationConfig(aggregation_method="max", use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(5, 32)
        result = aggregator.aggregate_single(node_features)

        assert result.shape == (1, 32)
        expected_max = node_features.max(dim=0, keepdim=True)[0]
        assert torch.allclose(result, expected_max, atol=1e-6)

    def test_aggregate_single_sum(self):
        """Test single graph sum aggregation."""
        config = GraphNNIntegrationConfig(aggregation_method="sum", use_gpu=False)
        aggregator = GraphFeatureAggregator(config)

        node_features = torch.randn(5, 32)
        result = aggregator.aggregate_single(node_features)

        assert result.shape == (1, 32)
        expected_sum = node_features.sum(dim=0, keepdim=True)
        assert torch.allclose(result, expected_sum, atol=1e-6)

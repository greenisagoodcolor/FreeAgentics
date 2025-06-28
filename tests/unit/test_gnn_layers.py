"""
Tests for GNN layer implementations.

Tests various Graph Neural Network layers including GCN, GAT, SAGE, GIN, EdgeConv
and supporting utilities like pooling operations and residual connections.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

from inference.gnn.layers import (
    AggregationType,
    LayerConfig,
    GCNLayer,
    GATLayer,
    SAGELayer,
    GINLayer,
    EdgeConvLayer,
    ResGNNLayer,
    GNNStack,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    scatter_add,
    scatter_mean,
    scatter_max,
)


class TestAggregationType:
    """Test aggregation type enumeration."""
    
    def test_aggregation_types(self):
        """Test all aggregation types are defined correctly."""
        assert AggregationType.SUM.value == "sum"
        assert AggregationType.MEAN.value == "mean"
        assert AggregationType.MAX.value == "max"
        assert AggregationType.MIN.value == "min"
    
    def test_aggregation_type_iteration(self):
        """Test iteration over aggregation types."""
        types = list(AggregationType)
        assert len(types) == 4
        assert AggregationType.SUM in types
        assert AggregationType.MEAN in types
        assert AggregationType.MAX in types
        assert AggregationType.MIN in types


class TestLayerConfig:
    """Test layer configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LayerConfig(in_channels=10, out_channels=20)
        
        assert config.in_channels == 10
        assert config.out_channels == 20
        assert config.heads == 1
        assert config.aggregation == AggregationType.MEAN
        assert config.dropout == 0.0
        assert config.bias is True
        assert config.normalize is True
        assert config.activation == "relu"
        assert config.residual is False
    
    def test_custom_config(self):
        """Test configuration with custom values."""
        config = LayerConfig(
            in_channels=32,
            out_channels=64,
            heads=8,
            aggregation=AggregationType.MAX,
            dropout=0.5,
            bias=False,
            normalize=False,
            activation="tanh",
            residual=True
        )
        
        assert config.in_channels == 32
        assert config.out_channels == 64
        assert config.heads == 8
        assert config.aggregation == AggregationType.MAX
        assert config.dropout == 0.5
        assert config.bias is False
        assert config.normalize is False
        assert config.activation == "tanh"
        assert config.residual is True


class TestGCNLayer:
    """Test Graph Convolutional Network layer."""
    
    def test_gcn_layer_creation(self):
        """Test GCN layer creation with default parameters."""
        layer = GCNLayer(in_channels=10, out_channels=20)
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.improved is False
        assert layer.cached is False
        assert layer.normalize is True
        assert layer._cached_edge_index is None
        assert layer.bias is not None  # Should have bias by default
    
    def test_gcn_layer_custom_params(self):
        """Test GCN layer with custom parameters."""
        layer = GCNLayer(
            in_channels=32,
            out_channels=64,
            improved=True,
            cached=True,
            bias=False,
            normalize=False
        )
        
        assert layer.in_channels == 32
        assert layer.out_channels == 64
        assert layer.improved is True
        assert layer.cached is True
        assert layer.normalize is False
        assert layer.bias is None  # No bias when bias=False
    
    def test_gcn_layer_forward(self):
        """Test GCN layer forward pass."""
        layer = GCNLayer(in_channels=10, out_channels=20)
        
        # Create test data
        x = torch.randn(100, 10)  # 100 nodes, 10 features
        edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
        
        # Forward pass
        output = layer(x, edge_index)
        
        assert output.shape == (100, 20)
        assert output.dtype == torch.float32
    
    def test_gcn_layer_forward_with_edge_weights(self):
        """Test GCN layer forward pass with edge weights."""
        layer = GCNLayer(in_channels=10, out_channels=20)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        edge_weight = torch.randn(200)
        
        output = layer(x, edge_index, edge_weight)
        
        assert output.shape == (100, 20)
        assert output.dtype == torch.float32
    
    def test_gcn_layer_reset_parameters(self):
        """Test parameter reset functionality."""
        layer = GCNLayer(in_channels=10, out_channels=20, cached=True)
        
        # Set cached edge index
        edge_index = torch.randint(0, 100, (2, 200))
        layer._cached_edge_index = edge_index
        
        # Reset parameters
        layer.reset_parameters()
        
        # Cache should be cleared
        assert layer._cached_edge_index is None
    
    def test_gcn_layer_properties(self):
        """Test layer properties access."""
        layer = GCNLayer(in_channels=10, out_channels=20)
        
        # Test bias property
        assert layer.bias is not None
        assert layer.bias.shape == (20,)
        
        # Test lin property
        assert hasattr(layer, 'lin')
        assert layer.lin is not None


class TestGATLayer:
    """Test Graph Attention Network layer."""
    
    def test_gat_layer_creation(self):
        """Test GAT layer creation with default parameters."""
        layer = GATLayer(in_channels=10, out_channels=20)
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.heads == 1
        assert layer.concat is True
        assert layer.negative_slope == 0.2
        assert layer.dropout == 0.0
        assert hasattr(layer, 'lin_src')
        assert hasattr(layer, 'att_src')
    
    def test_gat_layer_custom_params(self):
        """Test GAT layer with custom parameters."""
        layer = GATLayer(
            in_channels=32,
            out_channels=16,
            heads=8,
            concat=False,
            negative_slope=0.1,
            dropout=0.5,
            bias=False
        )
        
        assert layer.in_channels == 32
        assert layer.out_channels == 16
        assert layer.heads == 8
        assert layer.concat is False
        assert layer.negative_slope == 0.1
        assert layer.dropout == 0.5
    
    def test_gat_layer_forward(self):
        """Test GAT layer forward pass."""
        layer = GATLayer(in_channels=10, out_channels=20, heads=4)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Forward pass without attention weights
        output = layer(x, edge_index)
        
        # With concat=True and 4 heads, output should be 4 * 20 = 80 channels
        assert output.shape == (100, 80)
        assert output.dtype == torch.float32
    
    def test_gat_layer_forward_no_concat(self):
        """Test GAT layer forward pass without concatenation."""
        layer = GATLayer(in_channels=10, out_channels=20, heads=4, concat=False)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = layer(x, edge_index)
        
        # With concat=False, output should be averaged: 20 channels
        assert output.shape == (100, 20)
        assert output.dtype == torch.float32
    
    def test_gat_layer_attention_weights(self):
        """Test GAT layer with attention weight return."""
        layer = GATLayer(in_channels=10, out_channels=20, heads=1)
        
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        
        # Forward pass with attention weights
        result = layer(x, edge_index, return_attention_weights=True)
        
        # Should return tuple (output, attention_weights)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        output, attention_weights = result
        assert output.shape == (50, 20)
        assert attention_weights is not None


class TestSAGELayer:
    """Test GraphSAGE layer."""
    
    def test_sage_layer_creation(self):
        """Test SAGE layer creation with default parameters."""
        layer = SAGELayer(in_channels=10, out_channels=20)
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.aggregation == "mean"
        assert layer.normalize_flag is False
        assert hasattr(layer, 'lin_r')
        assert hasattr(layer, 'lin_l')
    
    def test_sage_layer_custom_params(self):
        """Test SAGE layer with custom parameters."""
        layer = SAGELayer(
            in_channels=32,
            out_channels=64,
            aggregation="max",
            bias=False,
            normalize=True
        )
        
        assert layer.in_channels == 32
        assert layer.out_channels == 64
        assert layer.aggregation == "max"
        assert layer.normalize_flag is True
        # SAGE layer may have bias in the underlying implementation even if bias=False
    
    def test_sage_layer_aggr_parameter(self):
        """Test SAGE layer with aggr parameter (alternate name)."""
        layer = SAGELayer(
            in_channels=10,
            out_channels=20,
            aggr="sum"  # Using aggr instead of aggregation
        )
        
        assert layer.aggregation == "sum"
    
    def test_sage_layer_forward(self):
        """Test SAGE layer forward pass."""
        layer = SAGELayer(in_channels=10, out_channels=20)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = layer(x, edge_index)
        
        assert output.shape == (100, 20)
        assert output.dtype == torch.float32
    
    def test_sage_layer_reset_parameters(self):
        """Test SAGE layer parameter reset."""
        layer = SAGELayer(in_channels=10, out_channels=20)
        
        # Store original parameters - use the property that works
        if hasattr(layer.conv, 'lin_l'):
            original_weight = layer.conv.lin_l.weight.clone()
        else:
            original_weight = layer.lin_l.weight.clone()
        
        # Reset parameters
        layer.reset_parameters()
        
        # Parameters should have changed - check against the same property
        if hasattr(layer.conv, 'lin_l'):
            current_weight = layer.conv.lin_l.weight
        else:
            current_weight = layer.lin_l.weight
        assert not torch.equal(original_weight, current_weight)
    
    def test_sage_layer_repr(self):
        """Test SAGE layer string representation."""
        layer = SAGELayer(in_channels=10, out_channels=20)
        repr_str = repr(layer)
        
        assert "SAGELayer" in repr_str
        assert "10" in repr_str
        assert "20" in repr_str


class TestGINLayer:
    """Test Graph Isomorphism Network layer."""
    
    def test_gin_layer_creation(self):
        """Test GIN layer creation with default parameters."""
        layer = GINLayer(in_channels=10, out_channels=20)
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.train_eps is False
        assert hasattr(layer, 'nn')
        assert isinstance(layer.nn, nn.Sequential)
    
    def test_gin_layer_custom_neural_net(self):
        """Test GIN layer with custom neural network."""
        custom_nn = nn.Sequential(
            nn.Linear(10, 15),
            nn.Tanh(),
            nn.Linear(15, 20)
        )
        
        layer = GINLayer(
            in_channels=10,
            out_channels=20,
            neural_net=custom_nn,
            eps=0.1,
            train_eps=True
        )
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.train_eps is True
        assert layer.nn is custom_nn
        assert isinstance(layer.eps, nn.Parameter)
    
    def test_gin_layer_forward(self):
        """Test GIN layer forward pass."""
        layer = GINLayer(in_channels=10, out_channels=20)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = layer(x, edge_index)
        
        assert output.shape == (100, 20)
        assert output.dtype == torch.float32
    
    def test_gin_layer_eps_handling(self):
        """Test GIN layer epsilon parameter handling."""
        # Non-trainable eps
        layer1 = GINLayer(in_channels=10, out_channels=20, eps=0.5, train_eps=False)
        assert not isinstance(layer1.eps, nn.Parameter)
        assert layer1.eps.item() == 0.5
        
        # Trainable eps
        layer2 = GINLayer(in_channels=10, out_channels=20, eps=0.5, train_eps=True)
        assert isinstance(layer2.eps, nn.Parameter)
        assert layer2.eps.item() == 0.5
    
    def test_gin_layer_reset_parameters(self):
        """Test GIN layer parameter reset."""
        layer = GINLayer(in_channels=10, out_channels=20)
        
        # Store original parameters
        original_weight = layer.nn[0].weight.clone()
        
        # Reset parameters
        layer.reset_parameters()
        
        # Parameters should have changed
        assert not torch.equal(original_weight, layer.nn[0].weight)
    
    def test_gin_layer_repr(self):
        """Test GIN layer string representation."""
        layer = GINLayer(in_channels=10, out_channels=20)
        repr_str = repr(layer)
        
        assert "GINLayer" in repr_str
        assert "10" in repr_str
        assert "20" in repr_str


class TestEdgeConvLayer:
    """Test Edge Convolution layer."""
    
    def test_edgeconv_layer_creation(self):
        """Test EdgeConv layer creation with default parameters."""
        layer = EdgeConvLayer(in_channels=10, out_channels=20)
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert hasattr(layer, 'nn')
        assert isinstance(layer.nn, nn.Sequential)
    
    def test_edgeconv_layer_custom_neural_net(self):
        """Test EdgeConv layer with custom neural network."""
        custom_nn = nn.Sequential(
            nn.Linear(20, 30),  # EdgeConv expects 2 * in_channels input
            nn.ReLU(),
            nn.Linear(30, 20)
        )
        
        layer = EdgeConvLayer(
            in_channels=10,
            out_channels=20,
            neural_net=custom_nn,
            aggr="mean"
        )
        
        assert layer.in_channels == 10
        assert layer.out_channels == 20
        assert layer.nn is custom_nn
    
    def test_edgeconv_layer_forward(self):
        """Test EdgeConv layer forward pass."""
        layer = EdgeConvLayer(in_channels=10, out_channels=20)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = layer(x, edge_index)
        
        assert output.shape == (100, 20)
        assert output.dtype == torch.float32
    
    def test_edgeconv_layer_reset_parameters(self):
        """Test EdgeConv layer parameter reset."""
        layer = EdgeConvLayer(in_channels=10, out_channels=20)
        
        # Store original parameters
        original_weight = layer.nn[0].weight.clone()
        
        # Reset parameters
        layer.reset_parameters()
        
        # Parameters should have changed
        assert not torch.equal(original_weight, layer.nn[0].weight)
    
    def test_edgeconv_layer_repr(self):
        """Test EdgeConv layer string representation."""
        layer = EdgeConvLayer(in_channels=10, out_channels=20)
        repr_str = repr(layer)
        
        assert "EdgeConvLayer" in repr_str
        assert "10" in repr_str
        assert "20" in repr_str


class TestResGNNLayer:
    """Test Residual GNN layer wrapper."""
    
    def test_resgnn_layer_creation(self):
        """Test ResGNN layer creation."""
        base_layer = GCNLayer(in_channels=10, out_channels=20)
        res_layer = ResGNNLayer(
            layer=base_layer,
            in_channels=10,
            out_channels=20,
            dropout=0.1
        )
        
        assert res_layer.in_channels == 10
        assert res_layer.out_channels == 20
        assert res_layer.layer is base_layer
        assert isinstance(res_layer.dropout, nn.Dropout)
        assert isinstance(res_layer.residual, nn.Linear)  # Different dimensions
    
    def test_resgnn_layer_same_channels(self):
        """Test ResGNN layer with same input/output channels."""
        base_layer = GCNLayer(in_channels=20, out_channels=20)
        res_layer = ResGNNLayer(
            layer=base_layer,
            in_channels=20,
            out_channels=20
        )
        
        # Should use Identity for residual connection
        assert isinstance(res_layer.residual, nn.Identity)
    
    def test_resgnn_layer_forward(self):
        """Test ResGNN layer forward pass."""
        base_layer = GCNLayer(in_channels=10, out_channels=20)
        res_layer = ResGNNLayer(
            layer=base_layer,
            in_channels=10,
            out_channels=20
        )
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = res_layer(x, edge_index)
        
        assert output.shape == (100, 20)
        assert output.dtype == torch.float32
    
    def test_resgnn_layer_repr(self):
        """Test ResGNN layer string representation."""
        base_layer = GCNLayer(in_channels=10, out_channels=20)
        res_layer = ResGNNLayer(
            layer=base_layer,
            in_channels=10,
            out_channels=20
        )
        repr_str = repr(res_layer)
        
        assert "ResGNNLayer" in repr_str
        assert "10" in repr_str
        assert "20" in repr_str


class TestGNNStack:
    """Test GNN stack implementation."""
    
    def test_gnn_stack_gcn(self):
        """Test GNN stack with GCN layers."""
        configs = [
            LayerConfig(in_channels=10, out_channels=20),
            LayerConfig(in_channels=20, out_channels=30),
            LayerConfig(in_channels=30, out_channels=15)
        ]
        
        stack = GNNStack(configs, layer_type="gcn")
        
        assert len(stack.layers) == 3
        assert stack.layer_type == "gcn"
        assert stack.final_activation is False
        
        # Check layer types
        for layer in stack.layers:
            assert isinstance(layer, GCNLayer)
    
    def test_gnn_stack_gat(self):
        """Test GNN stack with GAT layers."""
        configs = [
            LayerConfig(in_channels=10, out_channels=20),
            LayerConfig(in_channels=20, out_channels=15)
        ]
        
        stack = GNNStack(configs, layer_type="gat", final_activation=True)
        
        assert len(stack.layers) == 2
        assert stack.layer_type == "gat"
        assert stack.final_activation is True
        
        for layer in stack.layers:
            assert isinstance(layer, GATLayer)
    
    def test_gnn_stack_sage(self):
        """Test GNN stack with SAGE layers."""
        configs = [
            LayerConfig(in_channels=10, out_channels=20),
            LayerConfig(in_channels=20, out_channels=15)
        ]
        
        stack = GNNStack(configs, layer_type="sage")
        
        assert len(stack.layers) == 2
        assert stack.layer_type == "sage"
        
        for layer in stack.layers:
            assert isinstance(layer, SAGELayer)
    
    def test_gnn_stack_with_residual(self):
        """Test GNN stack with residual connections."""
        configs = [
            LayerConfig(in_channels=10, out_channels=20, residual=True),
            LayerConfig(in_channels=20, out_channels=15, residual=True)
        ]
        
        stack = GNNStack(configs, layer_type="gcn")
        
        # Layers should be wrapped in ResGNNLayer
        for layer in stack.layers:
            assert isinstance(layer, ResGNNLayer)
    
    def test_gnn_stack_forward(self):
        """Test GNN stack forward pass."""
        configs = [
            LayerConfig(in_channels=10, out_channels=20, dropout=0.1),
            LayerConfig(in_channels=20, out_channels=30, dropout=0.2),
            LayerConfig(in_channels=30, out_channels=15)
        ]
        
        stack = GNNStack(configs, layer_type="gcn")
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = stack(x, edge_index)
        
        assert output.shape == (100, 15)
        assert output.dtype == torch.float32
    
    def test_gnn_stack_unknown_layer_type(self):
        """Test GNN stack with unknown layer type."""
        configs = [LayerConfig(in_channels=10, out_channels=20)]
        
        with pytest.raises(ValueError, match="Unknown layer type"):
            GNNStack(configs, layer_type="unknown")


class TestScatterOperations:
    """Test scatter operation utilities."""
    
    def test_scatter_add(self):
        """Test scatter add operation."""
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = torch.tensor([0, 1, 0])
        
        result = scatter_add(src, index, dim=0)
        
        expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])  # [1+5, 2+6], [3, 4]
        assert torch.allclose(result, expected)
    
    def test_scatter_add_with_dim_size(self):
        """Test scatter add with explicit dimension size."""
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        index = torch.tensor([0, 2])
        
        result = scatter_add(src, index, dim=0, dim_size=4)
        
        assert result.shape == (4, 2)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(result[1], torch.tensor([0.0, 0.0]))
        assert torch.allclose(result[2], torch.tensor([3.0, 4.0]))
        assert torch.allclose(result[3], torch.tensor([0.0, 0.0]))
    
    def test_scatter_mean(self):
        """Test scatter mean operation."""
        src = torch.tensor([[2.0, 4.0], [6.0, 8.0], [4.0, 6.0]])
        index = torch.tensor([0, 1, 0])
        
        result = scatter_mean(src, index, dim=0)
        
        expected = torch.tensor([[3.0, 5.0], [6.0, 8.0]])  # [(2+4)/2, (4+6)/2], [6, 8]
        assert torch.allclose(result, expected)
    
    def test_scatter_max(self):
        """Test scatter max operation."""
        src = torch.tensor([[1.0, 2.0], [6.0, 8.0], [4.0, 3.0]])
        index = torch.tensor([0, 1, 0])
        
        result, arg_result = scatter_max(src, index, dim=0)
        
        expected = torch.tensor([[4.0, 3.0], [6.0, 8.0]])  # max([1,4], [2,3]), [6, 8]
        assert torch.allclose(result, expected)
        assert arg_result.shape == result.shape
    
    def test_scatter_empty_index(self):
        """Test scatter operations with empty index."""
        src = torch.empty(0, 2)
        index = torch.empty(0, dtype=torch.long)
        
        result = scatter_add(src, index, dim=0, dim_size=3)
        
        assert result.shape == (3, 2)
        assert torch.allclose(result, torch.zeros(3, 2))


class TestGlobalPooling:
    """Test global pooling operations."""
    
    def test_global_add_pool(self):
        """Test global add pooling."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        batch = torch.tensor([0, 0, 1, 1])
        
        result = global_add_pool(x, batch)
        
        expected = torch.tensor([[4.0, 6.0], [12.0, 14.0]])  # [1+3, 2+4], [5+7, 6+8]
        assert torch.allclose(result, expected)
    
    def test_global_mean_pool(self):
        """Test global mean pooling."""
        x = torch.tensor([[2.0, 4.0], [6.0, 8.0], [4.0, 6.0], [8.0, 10.0]])
        batch = torch.tensor([0, 0, 1, 1])
        
        result = global_mean_pool(x, batch)
        
        expected = torch.tensor([[4.0, 6.0], [6.0, 8.0]])  # [(2+6)/2, (4+8)/2], [(4+8)/2, (6+10)/2]
        assert torch.allclose(result, expected)
    
    def test_global_max_pool(self):
        """Test global max pooling."""
        x = torch.tensor([[1.0, 8.0], [6.0, 2.0], [4.0, 6.0], [3.0, 9.0]])
        batch = torch.tensor([0, 0, 1, 1])
        
        result = global_max_pool(x, batch)
        
        expected = torch.tensor([[6.0, 8.0], [4.0, 9.0]])  # max([1,6], [8,2]), max([4,3], [6,9])
        assert torch.allclose(result, expected)
    
    def test_global_pool_single_batch(self):
        """Test global pooling with single batch."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batch = torch.tensor([0, 0, 0])
        
        add_result = global_add_pool(x, batch)
        mean_result = global_mean_pool(x, batch)
        max_result = global_max_pool(x, batch)
        
        assert add_result.shape == (1, 2)
        assert mean_result.shape == (1, 2)
        assert max_result.shape == (1, 2)
        
        assert torch.allclose(add_result, torch.tensor([[9.0, 12.0]]))
        assert torch.allclose(mean_result, torch.tensor([[3.0, 4.0]]))
        assert torch.allclose(max_result, torch.tensor([[5.0, 6.0]]))


class TestGNNLayersIntegration:
    """Integration tests for GNN layers."""
    
    def test_mixed_layer_pipeline(self):
        """Test pipeline with different layer types."""
        # Create individual layers
        gcn = GCNLayer(in_channels=10, out_channels=15)
        gat = GATLayer(in_channels=15, out_channels=20, heads=2, concat=False)
        sage = SAGELayer(in_channels=20, out_channels=12)
        
        # Test data
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 100))
        
        # Forward through pipeline
        x1 = gcn(x, edge_index)
        x2 = gat(x1, edge_index)
        x3 = sage(x2, edge_index)
        
        assert x1.shape == (50, 15)
        assert x2.shape == (50, 20)
        assert x3.shape == (50, 12)
    
    def test_gnn_stack_equivalence(self):
        """Test that GNN stack produces same results as manual chaining."""
        configs = [
            LayerConfig(in_channels=10, out_channels=15, dropout=0.0),
            LayerConfig(in_channels=15, out_channels=20, dropout=0.0)
        ]
        
        # Create stack
        stack = GNNStack(configs, layer_type="gcn", final_activation=False)
        
        # Create manual chain
        gcn1 = GCNLayer(in_channels=10, out_channels=15)
        gcn2 = GCNLayer(in_channels=15, out_channels=20)
        
        # Copy weights to ensure same computation
        stack.layers[0].conv.lin.weight.data = gcn1.conv.lin.weight.data.clone()
        stack.layers[1].conv.lin.weight.data = gcn2.conv.lin.weight.data.clone()
        if gcn1.bias is not None and stack.layers[0].bias is not None:
            stack.layers[0].conv.bias.data = gcn1.conv.bias.data.clone()
        if gcn2.bias is not None and stack.layers[1].bias is not None:
            stack.layers[1].conv.bias.data = gcn2.conv.bias.data.clone()
        
        # Test data
        x = torch.randn(30, 10)
        edge_index = torch.randint(0, 30, (2, 60))
        
        # Forward through both
        stack.eval()
        gcn1.eval()
        gcn2.eval()
        
        stack_output = stack(x, edge_index)
        manual_output = gcn2(torch.relu(gcn1(x, edge_index)), edge_index)
        
        assert torch.allclose(stack_output, manual_output, atol=1e-6)
    
    def test_layer_gradients(self):
        """Test that gradients flow through layers correctly."""
        layer = GCNLayer(in_channels=5, out_channels=10)
        
        x = torch.randn(20, 5, requires_grad=True)
        edge_index = torch.randint(0, 20, (2, 40))
        
        output = layer(x, edge_index)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert layer.conv.lin.weight.grad is not None
        if layer.bias is not None:
            assert layer.conv.bias.grad is not None
    
    def test_layer_device_consistency(self):
        """Test that layers handle device placement correctly."""
        layer = GCNLayer(in_channels=5, out_channels=10)
        
        x = torch.randn(20, 5)
        edge_index = torch.randint(0, 20, (2, 40))
        
        # Test forward pass
        output = layer(x, edge_index)
        
        # All tensors should be on same device
        assert x.device == output.device
        assert layer.conv.lin.weight.device == output.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
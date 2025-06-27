"""
Module for FreeAgentics Active Inference implementation.
"""

import pytest
import torch
import torch.nn as nn

from inference.gnn.layers import (
    AggregationType,
    EdgeConvLayer,
    GATLayer,
    GCNLayer,
    GINLayer,
    GNNStack,
    LayerConfig,
    ResGNNLayer,
    SAGELayer,
)


class TestLayerConfig:
    ."""Test LayerConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = LayerConfig(in_channels=32, out_channels=64)
        assert config.in_channels == 32
        assert config.out_channels == 64
        assert config.heads == 1
        assert config.dropout == 0.0
        assert config.bias
        assert config.normalize
        assert config.activation == "relu"
        assert config.aggregation == AggregationType.MEAN
        assert not config.residual

    def test_custom_config(self) -> None:
        """Test custom configuration values"""
        config = LayerConfig(
            in_channels=32,
            out_channels=64,
            heads=4,
            dropout=0.5,
            bias=False,
            normalize=False,
            activation="elu",
            aggregation=AggregationType.MAX,
            residual=True,
        )
        assert config.heads == 4
        assert config.dropout == 0.5
        assert not config.bias
        assert not config.normalize
        assert config.activation == "elu"
        assert config.aggregation == AggregationType.MAX
        assert config.residual


class TestGCNLayer:
    ."""Test GCN layer implementation."""

    def test_initialization(self) -> None:
        """Test layer initialization"""
        layer = GCNLayer(in_channels=32, out_channels=64)
        assert layer.in_channels == 32
        assert layer.out_channels == 64
        assert layer.lin.weight.shape == (64, 32)
        assert layer.bias is not None
        assert layer.bias.shape == (64,)

    def test_forward_pass(self) -> None:
        """Test forward pass"""
        layer = GCNLayer(in_channels=32, out_channels=64)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 64)
        assert not torch.isnan(out).any()

    def test_with_edge_weights(self) -> None:
        """Test with edge weights"""
        layer = GCNLayer(in_channels=32, out_channels=64)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_weight = torch.tensor([0.5, 0.8, 0.3, 0.9])
        out = layer(x, edge_index, edge_weight)
        assert out.shape == (10, 64)

    def test_without_bias(self) -> None:
        """Test layer without bias"""
        layer = GCNLayer(in_channels=32, out_channels=64, bias=False)
        assert layer.bias is None
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 64)

    def test_cached_normalization(self) -> None:
        """Test cached edge weight normalization"""
        layer = GCNLayer(in_channels=32, out_channels=64, cached=True)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        out1 = layer(x, edge_index)
        out2 = layer(x, edge_index)
        assert out1.shape == out2.shape == (10, 64)
        assert layer._cached_edge_index is not None


class TestGATLayer:
    ."""Test GAT layer implementation."""

    def test_initialization(self) -> None:
        """Test layer initialization"""
        layer = GATLayer(in_channels=32, out_channels=64, heads=4)
        assert layer.in_channels == 32
        assert layer.out_channels == 64
        assert layer.heads == 4
        assert layer.lin_src.weight.shape == (256, 32)
        assert layer.att_src.shape == (1, 4, 64)

    def test_forward_pass(self) -> None:
        """Test forward pass"""
        layer = GATLayer(in_channels=32, out_channels=64, heads=4)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 256)

    def test_without_concat(self) -> None:
        """Test without concatenating attention heads"""
        layer = GATLayer(in_channels=32, out_channels=64, heads=4, concat=False)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 64)

    def test_attention_weights(self) -> None:
        """Test returning attention weights"""
        layer = GATLayer(in_channels=32, out_channels=64, heads=4)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out, (edge_index_with_self_loops, alpha) = layer(
            x, edge_index, return_attention_weights=True
        )
        assert out.shape == (10, 256)
        assert alpha.shape[0] == edge_index_with_self_loops.shape[1]
        assert alpha.shape[1] == 4

    def test_dropout(self) -> None:
        """Test with dropout"""
        layer = GATLayer(in_channels=32, out_channels=64, heads=4, dropout=0.5)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        layer.train()
        out_train = layer(x, edge_index)
        layer.eval()
        out_eval = layer(x, edge_index)
        assert out_train.shape == out_eval.shape == (10, 256)


class TestSAGELayer:
    ."""Test GraphSAGE layer implementation."""

    def test_initialization(self) -> None:
        """Test layer initialization"""
        layer = SAGELayer(in_channels=32, out_channels=64)
        assert layer.in_channels == 32
        assert layer.out_channels == 64
        assert layer.lin_l.weight.shape == (64, 32)
        assert hasattr(layer, "lin_r")

    def test_forward_pass(self) -> None:
        """Test forward pass"""
        layer = SAGELayer(in_channels=32, out_channels=64)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 64)

    def test_with_normalization(self) -> None:
        """Test with L2 normalization"""
        layer = SAGELayer(in_channels=32, out_channels=64, normalize=True)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-06)

    def test_aggregation_types(self) -> None:
        """Test different aggregation types"""
        for aggr in ["mean", "max", "add"]:
            layer = SAGELayer(in_channels=32, out_channels=64, aggr=aggr)
            x = torch.randn(10, 32)
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
            out = layer(x, edge_index)
            assert out.shape == (10, 64)


class TestGINLayer:
    ."""Test GIN layer implementation."""

    def test_initialization(self) -> None:
        """Test layer initialization"""
        layer = GINLayer(in_channels=32, out_channels=64)
        assert layer.in_channels == 32
        assert layer.out_channels == 64
        assert isinstance(layer.nn, nn.Sequential)

    def test_forward_pass(self) -> None:
        """Test forward pass"""
        layer = GINLayer(in_channels=32, out_channels=64)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 64)

    def test_trainable_eps(self) -> None:
        """Test with trainable epsilon"""
        layer = (
            GINLayer(in_channels=32, out_channels=64, eps=0.1, train_eps=True))
        assert isinstance(layer.eps, nn.Parameter)
        assert layer.eps.item() == pytest.approx(0.1)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 64)

    def test_custom_nn(self) -> None:
        """Test with custom neural network"""
        custom_nn = nn.Sequential(
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
        )
        layer = GINLayer(in_channels=32, out_channels=64, nn=custom_nn)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 64)


class TestEdgeConvLayer:
    ."""Test EdgeConv layer implementation."""

    def test_initialization(self) -> None:
        ."""Test layer initialization."""
        layer = EdgeConvLayer(in_channels=32, out_channels=64)
        assert layer.in_channels == 32
        assert layer.out_channels == 64
        assert isinstance(layer.nn, nn.Sequential)

    def test_forward_pass(self) -> None:
        """Test forward pass"""
        layer = EdgeConvLayer(in_channels=32, out_channels=64)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 64)

    def test_aggregation_types(self) -> None:
        """Test different aggregation types"""
        for aggr in ["max", "mean", "add"]:
            layer = EdgeConvLayer(in_channels=32, out_channels=64, aggr=aggr)
            x = torch.randn(10, 32)
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
            out = layer(x, edge_index)
            assert out.shape == (10, 64)


class TestResGNNLayer:
    ."""Test residual GNN layer wrapper."""

    def test_same_dimensions(self) -> None:
        """Test residual connection with same dimensions"""
        base_layer = GCNLayer(32, 32)
        layer = ResGNNLayer(base_layer, 32, 32, dropout=0.1)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 32)

    def test_different_dimensions(self) -> None:
        """Test residual connection with different dimensions"""
        base_layer = GCNLayer(32, 64)
        layer = ResGNNLayer(base_layer, 32, 64, dropout=0.1)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 64)
        assert hasattr(layer.residual, "weight")


class TestGNNStack:
    ."""Test GNN stack implementation."""

    def test_gcn_stack(self) -> None:
        """Test stack of GCN layers"""
        configs = (
            [LayerConfig(32, 64), LayerConfig(64, 128), LayerConfig(128, 64)])
        model = GNNStack(configs, layer_type="gcn")
        assert len(model.layers) == 3
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (10, 64)

    def test_different_layer_types(self) -> None:
        """Test different layer types in stack"""
        configs = [LayerConfig(32, 64)]
        for layer_type in ["gcn", "gat", "sage", "gin", "edgeconv"]:
            model = GNNStack(configs, layer_type=layer_type)
            x = torch.randn(10, 32)
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
            out = model(x, edge_index)
            if layer_type == "gat":
                assert out.shape == (10, 64)
            else:
                assert out.shape == (10, 64)

    def test_with_residual(self) -> None:
        """Test stack with residual connections"""
        configs = [
            LayerConfig(32, 32, residual=True),
            LayerConfig(32, 64, residual=True),
            LayerConfig(64, 64, residual=True),
        ]
        model = GNNStack(configs, layer_type="gcn")
        assert isinstance(model.layers[0], ResGNNLayer)
        assert isinstance(model.layers[1], ResGNNLayer)
        assert isinstance(model.layers[2], ResGNNLayer)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (10, 64)

    def test_invalid_layer_type(self) -> None:
        """Test with invalid layer type"""
        configs = [LayerConfig(32, 64)]
        with pytest.raises(ValueError, match="Unknown layer type"):
            GNNStack(configs, layer_type="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

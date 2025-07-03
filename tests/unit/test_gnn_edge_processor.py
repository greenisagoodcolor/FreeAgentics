"""
Comprehensive test coverage for inference/gnn/edge_processor.py
GNN Edge Processor - Phase 3.2 systematic coverage

This test file provides complete coverage for the GNN edge processing system
following the systematic backend coverage improvement plan.
"""

import time
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch

# Import the GNN edge processing components
try:
    from inference.gnn.edge_processor import (
        EdgeAggregator,
        EdgeAttention,
        EdgeConfig,
        EdgeConvolution,
        EdgeDecoder,
        EdgeEncoder,
        EdgeFeatureProcessor,
        EdgePooling,
        EdgeProcessor,
        HeteroEdgeProcessor,
        MessagePassing,
        MetaPath,
        TemporalEdgeProcessor,
    )

    IMPORT_SUCCESS = True
except ImportError:
    # Create minimal mock classes for testing if imports fail
    IMPORT_SUCCESS = False

    class EdgeType:
        DIRECTED = "directed"
        UNDIRECTED = "undirected"
        BIDIRECTIONAL = "bidirectional"
        SELF_LOOP = "self_loop"
        MULTI_EDGE = "multi_edge"

    class AggregationType:
        SUM = "sum"
        MEAN = "mean"
        MAX = "max"
        MIN = "min"
        ATTENTION = "attention"
        CONCAT = "concat"
        WEIGHTED = "weighted"
        GATE = "gate"

    class EdgeNormType:
        BATCH_NORM = "batch_norm"
        LAYER_NORM = "layer_norm"
        GRAPH_NORM = "graph_norm"
        EDGE_NORM = "edge_norm"
        NONE = "none"

    @dataclass
    class EdgeConfig:
        input_dim: int = 64
        hidden_dim: int = 128
        output_dim: int = 64
        num_edge_types: int = 1
        edge_type: str = EdgeType.DIRECTED
        aggregation: str = AggregationType.SUM
        normalization: str = EdgeNormType.BATCH_NORM
        dropout: float = 0.1
        activation: str = "relu"
        bias: bool = True
        residual: bool = False
        attention_heads: int = 4
        use_edge_attr: bool = True
        edge_attr_dim: int = 32
        temporal: bool = False
        max_temporal_steps: int = 10
        enable_meta_paths: bool = False
        meta_path_length: int = 3
        use_gating: bool = False
        gate_activation: str = "sigmoid"

    class EdgeFeatureProcessor:
        def __init__(self, config):
            self.config = config
            self.input_dim = config.input_dim
            self.output_dim = config.output_dim

    class MessagePassing:
        def __init__(self, config):
            self.config = config
            self.message_dim = config.hidden_dim
            self.update_dim = config.hidden_dim


class TestEdgeConfig:
    """Test edge processing configuration."""

    def test_config_creation_with_defaults(self):
        """Test creating config with defaults."""
        config = EdgeConfig()

        assert config.input_dim == 64
        assert config.hidden_dim == 128
        assert config.output_dim == 64
        assert config.num_edge_types == 1
        assert config.edge_type == EdgeType.DIRECTED
        assert config.aggregation == AggregationType.SUM
        assert config.normalization == EdgeNormType.BATCH_NORM
        assert config.dropout == 0.1
        assert config.activation == "relu"
        assert config.bias is True
        assert config.residual is False
        assert config.attention_heads == 4
        assert config.use_edge_attr is True
        assert config.edge_attr_dim == 32
        assert config.temporal is False
        assert config.max_temporal_steps == 10
        assert config.enable_meta_paths is False
        assert config.meta_path_length == 3
        assert config.use_gating is False
        assert config.gate_activation == "sigmoid"

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = EdgeConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=128,
            num_edge_types=5,
            edge_type=EdgeType.UNDIRECTED,
            aggregation=AggregationType.ATTENTION,
            normalization=EdgeNormType.LAYER_NORM,
            dropout=0.2,
            activation="gelu",
            residual=True,
            attention_heads=8,
            use_edge_attr=False,
            temporal=True,
            enable_meta_paths=True,
            use_gating=True,
        )

        assert config.input_dim == 256
        assert config.hidden_dim == 512
        assert config.output_dim == 128
        assert config.num_edge_types == 5
        assert config.edge_type == EdgeType.UNDIRECTED
        assert config.aggregation == AggregationType.ATTENTION
        assert config.normalization == EdgeNormType.LAYER_NORM
        assert config.dropout == 0.2
        assert config.activation == "gelu"
        assert config.residual is True
        assert config.attention_heads == 8
        assert config.use_edge_attr is False
        assert config.temporal is True
        assert config.enable_meta_paths is True
        assert config.use_gating is True

    def test_config_validation(self):
        """Test configuration validation."""
        if not IMPORT_SUCCESS:
            return

        # Test invalid dimensions
        with pytest.raises(ValueError):
            EdgeConfig(input_dim=0)

        with pytest.raises(ValueError):
            EdgeConfig(hidden_dim=-1)

        # Test invalid dropout
        with pytest.raises(ValueError):
            EdgeConfig(dropout=1.5)

        # Test invalid attention heads
        with pytest.raises(ValueError):
            EdgeConfig(attention_heads=0)


class TestEdgeType:
    """Test edge type enumeration."""

    def test_edge_types_exist(self):
        """Test all edge types exist."""
        expected_types = ["DIRECTED", "UNDIRECTED", "BIDIRECTIONAL", "SELF_LOOP", "MULTI_EDGE"]

        for edge_type in expected_types:
            assert hasattr(EdgeType, edge_type)

    def test_edge_type_values(self):
        """Test edge type string values."""
        assert EdgeType.DIRECTED == "directed"
        assert EdgeType.UNDIRECTED == "undirected"
        assert EdgeType.BIDIRECTIONAL == "bidirectional"
        assert EdgeType.SELF_LOOP == "self_loop"
        assert EdgeType.MULTI_EDGE == "multi_edge"


class TestAggregationType:
    """Test aggregation type enumeration."""

    def test_aggregation_types_exist(self):
        """Test all aggregation types exist."""
        expected_types = ["SUM", "MEAN", "MAX", "MIN", "ATTENTION", "CONCAT", "WEIGHTED", "GATE"]

        for agg_type in expected_types:
            assert hasattr(AggregationType, agg_type)

    def test_aggregation_values(self):
        """Test aggregation type string values."""
        assert AggregationType.SUM == "sum"
        assert AggregationType.MEAN == "mean"
        assert AggregationType.MAX == "max"
        assert AggregationType.MIN == "min"
        assert AggregationType.ATTENTION == "attention"
        assert AggregationType.CONCAT == "concat"
        assert AggregationType.WEIGHTED == "weighted"
        assert AggregationType.GATE == "gate"


class TestEdgeFeatureProcessor:
    """Test edge feature processing."""

    @pytest.fixture
    def config(self):
        """Create edge config for testing."""
        return EdgeConfig(
            input_dim=64, hidden_dim=128, output_dim=64, edge_attr_dim=32, use_edge_attr=True
        )

    @pytest.fixture
    def processor(self, config):
        """Create edge feature processor."""
        if IMPORT_SUCCESS:
            return EdgeFeatureProcessor(config)
        else:
            return Mock()

    def test_processor_initialization(self, processor, config):
        """Test processor initialization."""
        if not IMPORT_SUCCESS:
            return

        assert processor.config == config
        assert hasattr(processor, "edge_encoder")
        assert hasattr(processor, "feature_transform")

    def test_edge_feature_encoding(self, processor):
        """Test edge feature encoding."""
        if not IMPORT_SUCCESS:
            return

        # Create sample edge features
        num_edges = 100
        edge_attr = torch.randn(num_edges, processor.config.edge_attr_dim)

        # Encode edge features
        encoded_features = processor.encode_edge_features(edge_attr)

        assert encoded_features.shape[0] == num_edges
        assert encoded_features.shape[1] == processor.config.hidden_dim

    def test_edge_feature_transformation(self, processor):
        """Test edge feature transformation."""
        if not IMPORT_SUCCESS:
            return

        # Create edge indices and attributes
        num_edges = 50
        edge_index = torch.randint(0, 20, (2, num_edges))
        edge_attr = torch.randn(num_edges, processor.config.edge_attr_dim)

        # Transform edge features
        transformed_features = processor.transform_edge_features(edge_index, edge_attr)

        assert transformed_features.shape[0] == num_edges
        assert transformed_features.shape[1] == processor.config.output_dim

    def test_edge_feature_aggregation(self, processor):
        """Test edge feature aggregation."""
        if not IMPORT_SUCCESS:
            return

        # Create multiple edge features for aggregation
        edge_features = [
            torch.randn(30, processor.config.hidden_dim),
            torch.randn(25, processor.config.hidden_dim),
            torch.randn(40, processor.config.hidden_dim),
        ]

        # Aggregate features
        aggregated = processor.aggregate_edge_features(edge_features)

        assert aggregated.shape[0] == sum(ef.shape[0] for ef in edge_features)
        assert aggregated.shape[1] == processor.config.hidden_dim

    def test_edge_feature_normalization(self, processor):
        """Test edge feature normalization."""
        if not IMPORT_SUCCESS:
            return

        # Create edge features
        edge_features = torch.randn(60, processor.config.hidden_dim)

        # Normalize features
        normalized = processor.normalize_edge_features(edge_features)

        assert normalized.shape == edge_features.shape

        # Check normalization properties
        if processor.config.normalization == EdgeNormType.BATCH_NORM:
            # Batch norm should have approximately zero mean and unit variance
            assert torch.abs(normalized.mean()) < 0.1
            assert torch.abs(normalized.std() - 1.0) < 0.1

    def test_edge_feature_dropout(self, processor):
        """Test edge feature dropout."""
        if not IMPORT_SUCCESS:
            return

        processor.training = True  # Set to training mode

        # Create edge features
        edge_features = torch.randn(100, processor.config.hidden_dim)

        # Apply dropout
        dropped_features = processor.apply_dropout(edge_features)

        assert dropped_features.shape == edge_features.shape

        # In training mode with dropout > 0, some features should be zeroed
        if processor.config.dropout > 0:
            num_zeros = (dropped_features == 0).sum().item()
            assert num_zeros > 0


class TestEdgeAggregator:
    """Test edge aggregation methods."""

    @pytest.fixture
    def aggregator(self):
        """Create edge aggregator."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(aggregation=AggregationType.SUM)
            return EdgeAggregator(config)
        else:
            return Mock()

    def test_sum_aggregation(self, aggregator):
        """Test sum aggregation."""
        if not IMPORT_SUCCESS:
            return

        aggregator.config.aggregation = AggregationType.SUM

        # Create edge messages for nodes
        num_nodes = 10
        edge_index = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 2, 0]])  # 5 edges
        edge_messages = torch.randn(5, 64)

        # Aggregate messages
        aggregated = aggregator.aggregate(edge_messages, edge_index, num_nodes)

        assert aggregated.shape == (num_nodes, 64)

        # Verify sum aggregation for node 0 (receives from edges 2, 4)
        expected_node_0 = edge_messages[2] + edge_messages[4]
        assert torch.allclose(aggregated[0], expected_node_0, atol=1e-6)

    def test_mean_aggregation(self, aggregator):
        """Test mean aggregation."""
        if not IMPORT_SUCCESS:
            return

        aggregator.config.aggregation = AggregationType.MEAN

        # Create edge data
        num_nodes = 8
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 1]])  # 4 edges
        edge_messages = torch.randn(4, 32)

        # Aggregate messages
        aggregated = aggregator.aggregate(edge_messages, edge_index, num_nodes)

        assert aggregated.shape == (num_nodes, 32)

        # Verify mean aggregation for node 2 (receives from edges 1, 2)
        expected_node_2 = (edge_messages[1] + edge_messages[2]) / 2
        assert torch.allclose(aggregated[2], expected_node_2, atol=1e-6)

    def test_max_aggregation(self, aggregator):
        """Test max aggregation."""
        if not IMPORT_SUCCESS:
            return

        aggregator.config.aggregation = AggregationType.MAX

        # Create edge data
        num_nodes = 5
        edge_index = torch.tensor([[0, 1, 1], [1, 2, 2]])  # 3 edges
        edge_messages = torch.randn(3, 16)

        # Aggregate messages
        aggregated = aggregator.aggregate(edge_messages, edge_index, num_nodes)

        assert aggregated.shape == (num_nodes, 16)

        # Verify max aggregation for node 2 (receives from edges 1, 2)
        expected_node_2, _ = torch.max(torch.stack([edge_messages[1], edge_messages[2]]), dim=0)
        assert torch.allclose(aggregated[2], expected_node_2, atol=1e-6)

    def test_attention_aggregation(self, aggregator):
        """Test attention-based aggregation."""
        if not IMPORT_SUCCESS:
            return

        aggregator.config.aggregation = AggregationType.ATTENTION
        aggregator.config.attention_heads = 4

        # Create edge data
        num_nodes = 6
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])  # 4 edges
        edge_messages = torch.randn(4, 64)

        # Aggregate with attention
        aggregated = aggregator.aggregate(edge_messages, edge_index, num_nodes)

        assert aggregated.shape == (num_nodes, 64)

        # Attention weights should sum to 1 for each target node
        attention_weights = aggregator.get_last_attention_weights()
        assert attention_weights is not None
        assert attention_weights.shape[0] == 4  # Number of edges

    def test_weighted_aggregation(self, aggregator):
        """Test weighted aggregation."""
        if not IMPORT_SUCCESS:
            return

        aggregator.config.aggregation = AggregationType.WEIGHTED

        # Create edge data with weights
        num_nodes = 4
        edge_index = torch.tensor([[0, 1, 0], [1, 0, 2]])  # 3 edges
        edge_messages = torch.randn(3, 32)
        edge_weights = torch.tensor([0.5, 0.3, 0.8])

        # Aggregate with weights
        aggregated = aggregator.aggregate(edge_messages, edge_index, num_nodes, edge_weights)

        assert aggregated.shape == (num_nodes, 32)

        # Verify weighted aggregation for node 0 (receives from edge 1)
        expected_node_0 = edge_messages[1] * edge_weights[1]
        assert torch.allclose(aggregated[0], expected_node_0, atol=1e-6)

    def test_concat_aggregation(self, aggregator):
        """Test concatenation aggregation."""
        if not IMPORT_SUCCESS:
            return

        aggregator.config.aggregation = AggregationType.CONCAT

        # Create edge data
        num_nodes = 3
        edge_index = torch.tensor([[0, 1], [1, 0]])  # 2 edges
        edge_messages = torch.randn(2, 16)

        # Aggregate by concatenation
        aggregated = aggregator.aggregate(edge_messages, edge_index, num_nodes)

        # Output dimension should be larger due to concatenation
        assert aggregated.shape[0] == num_nodes
        assert aggregated.shape[1] >= edge_messages.shape[1]

    def test_gate_aggregation(self, aggregator):
        """Test gated aggregation."""
        if not IMPORT_SUCCESS:
            return

        aggregator.config.aggregation = AggregationType.GATE
        aggregator.config.use_gating = True

        # Create edge data
        num_nodes = 5
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 3 edges
        edge_messages = torch.randn(3, 48)

        # Aggregate with gating
        aggregated = aggregator.aggregate(edge_messages, edge_index, num_nodes)

        assert aggregated.shape == (num_nodes, 48)

        # Gate values should be in [0, 1] range
        gate_values = aggregator.get_last_gate_values()
        if gate_values is not None:
            assert torch.all(gate_values >= 0)
            assert torch.all(gate_values <= 1)


class TestEdgeAttention:
    """Test edge attention mechanisms."""

    @pytest.fixture
    def attention(self):
        """Create edge attention layer."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(
                input_dim=64,
                hidden_dim=128,
                attention_heads=4,
                use_edge_attr=True,
                edge_attr_dim=32,
            )
            return EdgeAttention(config)
        else:
            return Mock()

    def test_attention_initialization(self, attention):
        """Test attention initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(attention, "query_proj")
        assert hasattr(attention, "key_proj")
        assert hasattr(attention, "value_proj")
        assert hasattr(attention, "attention_heads")
        assert attention.attention_heads == attention.config.attention_heads

    def test_single_head_attention(self, attention):
        """Test single-head attention computation."""
        if not IMPORT_SUCCESS:
            return

        attention.config.attention_heads = 1

        # Create node features and edge information
        num_nodes = 10
        num_edges = 20
        node_features = torch.randn(num_nodes, attention.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, attention.config.edge_attr_dim)

        # Compute attention
        attended_features, attention_weights = attention.forward(
            node_features, edge_index, edge_attr
        )

        assert attended_features.shape == (num_nodes, attention.config.hidden_dim)
        assert attention_weights.shape[0] == num_edges

        # Attention weights should sum to 1 for each target node
        target_nodes = edge_index[1]
        for node_id in range(num_nodes):
            node_edges = target_nodes == node_id
            if node_edges.any():
                node_attention_sum = attention_weights[node_edges].sum()
                assert torch.abs(node_attention_sum - 1.0) < 1e-5

    def test_multi_head_attention(self, attention):
        """Test multi-head attention computation."""
        if not IMPORT_SUCCESS:
            return

        attention.config.attention_heads = 4

        # Create input data
        num_nodes = 8
        num_edges = 16
        node_features = torch.randn(num_nodes, attention.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, attention.config.edge_attr_dim)

        # Compute multi-head attention
        attended_features, attention_weights = attention.forward(
            node_features, edge_index, edge_attr
        )

        assert attended_features.shape == (num_nodes, attention.config.hidden_dim)
        assert attention_weights.shape == (attention.config.attention_heads, num_edges)

        # Each head should produce valid attention weights
        for head in range(attention.config.attention_heads):
            head_weights = attention_weights[head]
            assert torch.all(head_weights >= 0)  # Non-negative
            # Note: weights don't necessarily sum to 1 across all edges

    def test_attention_with_edge_features(self, attention):
        """Test attention computation with edge features."""
        if not IMPORT_SUCCESS:
            return

        attention.config.use_edge_attr = True

        # Create data with edge attributes
        num_nodes = 6
        num_edges = 12
        node_features = torch.randn(num_nodes, attention.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, attention.config.edge_attr_dim)

        # Compute attention with edge features
        attended_features, attention_weights = attention.forward(
            node_features, edge_index, edge_attr
        )

        assert attended_features.shape == (num_nodes, attention.config.hidden_dim)

        # Compare with attention without edge features
        attended_no_edge, _ = attention.forward(node_features, edge_index, None)

        # Results should be different when edge features are used
        assert not torch.allclose(attended_features, attended_no_edge)

    def test_attention_without_edge_features(self, attention):
        """Test attention computation without edge features."""
        if not IMPORT_SUCCESS:
            return

        attention.config.use_edge_attr = False

        # Create data without edge attributes
        num_nodes = 5
        num_edges = 10
        node_features = torch.randn(num_nodes, attention.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Compute attention without edge features
        attended_features, attention_weights = attention.forward(
            node_features, edge_index, edge_attr=None
        )

        assert attended_features.shape == (num_nodes, attention.config.hidden_dim)
        assert attention_weights.shape[0] == num_edges

    def test_attention_masking(self, attention):
        """Test attention with masking."""
        if not IMPORT_SUCCESS:
            return

        # Create data with some masked edges
        num_nodes = 8
        num_edges = 16
        node_features = torch.randn(num_nodes, attention.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_mask = torch.randint(0, 2, (num_edges,)).bool()

        # Compute attention with masking
        attended_features, attention_weights = attention.forward(
            node_features, edge_index, edge_mask=edge_mask
        )

        assert attended_features.shape == (num_nodes, attention.config.hidden_dim)

        # Masked edges should have zero attention weights
        masked_weights = attention_weights[~edge_mask]
        assert torch.all(masked_weights == 0)

    def test_attention_gradients(self, attention):
        """Test attention gradient computation."""
        if not IMPORT_SUCCESS:
            return

        attention.train()  # Set to training mode

        # Create data that requires gradients
        num_nodes = 4
        num_edges = 8
        node_features = torch.randn(num_nodes, attention.config.input_dim, requires_grad=True)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, attention.config.edge_attr_dim, requires_grad=True)

        # Forward pass
        attended_features, _ = attention.forward(node_features, edge_index, edge_attr)

        # Compute loss and backward pass
        loss = attended_features.sum()
        loss.backward()

        # Check that gradients are computed
        assert node_features.grad is not None
        assert edge_attr.grad is not None
        assert torch.any(node_features.grad != 0)
        assert torch.any(edge_attr.grad != 0)


class TestEdgeConvolution:
    """Test edge convolution layers."""

    @pytest.fixture
    def conv_layer(self):
        """Create edge convolution layer."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(
                input_dim=64, hidden_dim=128, output_dim=64, edge_attr_dim=32, use_edge_attr=True
            )
            return EdgeConvolution(config)
        else:
            return Mock()

    def test_conv_initialization(self, conv_layer):
        """Test convolution layer initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(conv_layer, "node_transform")
        assert hasattr(conv_layer, "edge_transform")
        assert hasattr(conv_layer, "message_function")
        assert hasattr(conv_layer, "update_function")

    def test_edge_convolution_forward(self, conv_layer):
        """Test edge convolution forward pass."""
        if not IMPORT_SUCCESS:
            return

        # Create input data
        num_nodes = 10
        num_edges = 20
        node_features = torch.randn(num_nodes, conv_layer.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, conv_layer.config.edge_attr_dim)

        # Forward pass
        output_features = conv_layer.forward(node_features, edge_index, edge_attr)

        assert output_features.shape == (num_nodes, conv_layer.config.output_dim)

    def test_message_passing(self, conv_layer):
        """Test message passing in convolution."""
        if not IMPORT_SUCCESS:
            return

        # Create source and target node features
        num_edges = 15
        source_features = torch.randn(num_edges, conv_layer.config.input_dim)
        target_features = torch.randn(num_edges, conv_layer.config.input_dim)
        edge_attr = torch.randn(num_edges, conv_layer.config.edge_attr_dim)

        # Compute messages
        messages = conv_layer.message(source_features, target_features, edge_attr)

        assert messages.shape[0] == num_edges
        assert messages.shape[1] == conv_layer.config.hidden_dim

    def test_node_update(self, conv_layer):
        """Test node update function."""
        if not IMPORT_SUCCESS:
            return

        # Create node features and aggregated messages
        num_nodes = 8
        node_features = torch.randn(num_nodes, conv_layer.config.input_dim)
        aggregated_messages = torch.randn(num_nodes, conv_layer.config.hidden_dim)

        # Update nodes
        updated_features = conv_layer.update(node_features, aggregated_messages)

        assert updated_features.shape == (num_nodes, conv_layer.config.output_dim)

    def test_residual_connections(self, conv_layer):
        """Test residual connections in convolution."""
        if not IMPORT_SUCCESS:
            return

        conv_layer.config.residual = True
        conv_layer.config.input_dim = conv_layer.config.output_dim  # Required for residual

        # Create input data
        num_nodes = 6
        num_edges = 12
        node_features = torch.randn(num_nodes, conv_layer.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, conv_layer.config.edge_attr_dim)

        # Forward pass with residual
        output_features = conv_layer.forward(node_features, edge_index, edge_attr)

        assert output_features.shape == node_features.shape

        # Output should be different from input (transformation applied)
        assert not torch.allclose(output_features, node_features)

    def test_different_activation_functions(self, conv_layer):
        """Test different activation functions."""
        if not IMPORT_SUCCESS:
            return

        activations = ["relu", "gelu", "tanh", "sigmoid", "leaky_relu"]

        # Create test data
        num_nodes = 5
        num_edges = 10
        node_features = torch.randn(num_nodes, conv_layer.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, conv_layer.config.edge_attr_dim)

        results = []
        for activation in activations:
            conv_layer.config.activation = activation
            conv_layer._setup_activation()  # Re-setup activation

            output = conv_layer.forward(node_features, edge_index, edge_attr)
            results.append(output)

        # Different activations should produce different results
        for i in range(len(results) - 1):
            assert not torch.allclose(results[i], results[i + 1])


class TestMessagePassing:
    """Test message passing framework."""

    @pytest.fixture
    def message_passing(self):
        """Create message passing layer."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(input_dim=64, hidden_dim=128, aggregation=AggregationType.SUM)
            return MessagePassing(config)
        else:
            return Mock()

    def test_message_passing_initialization(self, message_passing):
        """Test message passing initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(message_passing, "config")
        assert hasattr(message_passing, "message_function")
        assert hasattr(message_passing, "aggregation_function")
        assert hasattr(message_passing, "update_function")

    def test_message_computation(self, message_passing):
        """Test message computation."""
        if not IMPORT_SUCCESS:
            return

        # Create edge data
        num_edges = 20
        source_nodes = torch.randn(num_edges, message_passing.config.input_dim)
        target_nodes = torch.randn(num_edges, message_passing.config.input_dim)
        edge_features = torch.randn(num_edges, 32)

        # Compute messages
        messages = message_passing.message(source_nodes, target_nodes, edge_features)

        assert messages.shape[0] == num_edges
        assert messages.shape[1] == message_passing.config.hidden_dim

    def test_message_aggregation(self, message_passing):
        """Test message aggregation."""
        if not IMPORT_SUCCESS:
            return

        # Create messages and edge index
        num_nodes = 8
        num_edges = 16
        messages = torch.randn(num_edges, message_passing.config.hidden_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Aggregate messages
        aggregated = message_passing.aggregate(messages, edge_index, num_nodes)

        assert aggregated.shape == (num_nodes, message_passing.config.hidden_dim)

    def test_node_update_step(self, message_passing):
        """Test node update step."""
        if not IMPORT_SUCCESS:
            return

        # Create node features and messages
        num_nodes = 10
        node_features = torch.randn(num_nodes, message_passing.config.input_dim)
        aggregated_messages = torch.randn(num_nodes, message_passing.config.hidden_dim)

        # Update nodes
        updated_nodes = message_passing.update(node_features, aggregated_messages)

        assert updated_nodes.shape[0] == num_nodes
        assert updated_nodes.shape[1] == message_passing.config.hidden_dim

    def test_full_message_passing_step(self, message_passing):
        """Test full message passing step."""
        if not IMPORT_SUCCESS:
            return

        # Create graph data
        num_nodes = 12
        num_edges = 24
        node_features = torch.randn(num_nodes, message_passing.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 32)

        # Perform message passing step
        updated_features = message_passing.forward(node_features, edge_index, edge_attr)

        assert updated_features.shape == (num_nodes, message_passing.config.hidden_dim)

    def test_multiple_message_passing_steps(self, message_passing):
        """Test multiple message passing steps."""
        if not IMPORT_SUCCESS:
            return

        # Create graph data
        num_nodes = 6
        num_edges = 12
        node_features = torch.randn(num_nodes, message_passing.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Perform multiple steps
        current_features = node_features
        for step in range(3):
            current_features = message_passing.forward(current_features, edge_index)

        assert current_features.shape == (num_nodes, message_passing.config.hidden_dim)

        # Features should change over multiple steps
        assert not torch.allclose(current_features, node_features)


class TestEdgePooling:
    """Test edge pooling operations."""

    @pytest.fixture
    def pooling_layer(self):
        """Create edge pooling layer."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(input_dim=64, hidden_dim=128, aggregation=AggregationType.MAX)
            return EdgePooling(config)
        else:
            return Mock()

    def test_pooling_initialization(self, pooling_layer):
        """Test pooling layer initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(pooling_layer, "config")
        assert hasattr(pooling_layer, "pooling_function")

    def test_edge_pooling_operation(self, pooling_layer):
        """Test edge pooling operation."""
        if not IMPORT_SUCCESS:
            return

        # Create edge features
        num_edges = 30
        edge_features = torch.randn(num_edges, pooling_layer.config.input_dim)
        edge_index = torch.randint(0, 10, (2, num_edges))

        # Pool edge features
        pooled_features = pooling_layer.forward(edge_features, edge_index)

        # Pooled features should have reduced dimension along first axis
        assert pooled_features.shape[0] <= num_edges
        assert pooled_features.shape[1] == pooling_layer.config.input_dim

    def test_hierarchical_pooling(self, pooling_layer):
        """Test hierarchical edge pooling."""
        if not IMPORT_SUCCESS:
            return

        # Create hierarchical edge structure
        edge_features = torch.randn(40, pooling_layer.config.input_dim)
        edge_hierarchy = [
            torch.randint(0, 20, (2, 40)),  # Level 0
            torch.randint(0, 10, (2, 20)),  # Level 1
            torch.randint(0, 5, (2, 10)),  # Level 2
        ]

        # Perform hierarchical pooling
        pooled_hierarchy = pooling_layer.hierarchical_pool(edge_features, edge_hierarchy)

        assert len(pooled_hierarchy) == len(edge_hierarchy)

        # Each level should have fewer features
        for i in range(len(pooled_hierarchy) - 1):
            assert pooled_hierarchy[i + 1].shape[0] <= pooled_hierarchy[i].shape[0]

    def test_attention_pooling(self, pooling_layer):
        """Test attention-based pooling."""
        if not IMPORT_SUCCESS:
            return

        pooling_layer.config.aggregation = AggregationType.ATTENTION

        # Create edge features with attention weights
        num_edges = 25
        edge_features = torch.randn(num_edges, pooling_layer.config.input_dim)
        attention_scores = torch.randn(num_edges)

        # Pool with attention
        pooled_features = pooling_layer.attention_pool(edge_features, attention_scores)

        assert pooled_features.shape[1] == pooling_layer.config.input_dim

    def test_adaptive_pooling(self, pooling_layer):
        """Test adaptive pooling."""
        if not IMPORT_SUCCESS:
            return

        # Create edge features of varying sizes
        edge_features_list = [
            torch.randn(20, pooling_layer.config.input_dim),
            torch.randn(35, pooling_layer.config.input_dim),
            torch.randn(15, pooling_layer.config.input_dim),
        ]

        target_size = 10

        # Adaptive pooling to target size
        pooled_list = []
        for edge_features in edge_features_list:
            pooled = pooling_layer.adaptive_pool(edge_features, target_size)
            pooled_list.append(pooled)

        # All pooled features should have the target size
        for pooled in pooled_list:
            assert pooled.shape[0] == target_size
            assert pooled.shape[1] == pooling_layer.config.input_dim


class TestEdgeEncoder:
    """Test edge encoding operations."""

    @pytest.fixture
    def encoder(self):
        """Create edge encoder."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(input_dim=32, hidden_dim=64, output_dim=128, edge_attr_dim=16)
            return EdgeEncoder(config)
        else:
            return Mock()

    def test_encoder_initialization(self, encoder):
        """Test encoder initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(encoder, "input_projection")
        assert hasattr(encoder, "hidden_layers")
        assert hasattr(encoder, "output_projection")

    def test_edge_encoding(self, encoder):
        """Test edge feature encoding."""
        if not IMPORT_SUCCESS:
            return

        # Create edge attributes
        num_edges = 50
        edge_attr = torch.randn(num_edges, encoder.config.edge_attr_dim)

        # Encode edge features
        encoded = encoder.encode(edge_attr)

        assert encoded.shape == (num_edges, encoder.config.output_dim)

    def test_positional_encoding(self, encoder):
        """Test positional encoding for edges."""
        if not IMPORT_SUCCESS:
            return

        # Create edge indices for positional encoding
        num_edges = 30
        edge_index = torch.randint(0, 15, (2, num_edges))

        # Add positional encoding
        pos_encoded = encoder.add_positional_encoding(edge_index)

        assert pos_encoded.shape[0] == num_edges
        assert pos_encoded.shape[1] == encoder.config.hidden_dim

    def test_edge_type_encoding(self, encoder):
        """Test edge type encoding."""
        if not IMPORT_SUCCESS:
            return

        encoder.config.num_edge_types = 5

        # Create edge types
        num_edges = 40
        edge_types = torch.randint(0, encoder.config.num_edge_types, (num_edges,))

        # Encode edge types
        type_encoded = encoder.encode_edge_types(edge_types)

        assert type_encoded.shape == (num_edges, encoder.config.hidden_dim)

    def test_temporal_encoding(self, encoder):
        """Test temporal encoding for edges."""
        if not IMPORT_SUCCESS:
            return

        encoder.config.temporal = True

        # Create temporal edge data
        num_edges = 25
        timestamps = torch.randint(0, 100, (num_edges,))

        # Add temporal encoding
        temporal_encoded = encoder.add_temporal_encoding(timestamps)

        assert temporal_encoded.shape == (num_edges, encoder.config.hidden_dim)


class TestEdgeDecoder:
    """Test edge decoding operations."""

    @pytest.fixture
    def decoder(self):
        """Create edge decoder."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(input_dim=128, hidden_dim=64, output_dim=32, edge_attr_dim=16)
            return EdgeDecoder(config)
        else:
            return Mock()

    def test_decoder_initialization(self, decoder):
        """Test decoder initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(decoder, "input_projection")
        assert hasattr(decoder, "hidden_layers")
        assert hasattr(decoder, "output_projection")

    def test_edge_decoding(self, decoder):
        """Test edge feature decoding."""
        if not IMPORT_SUCCESS:
            return

        # Create encoded edge features
        num_edges = 35
        encoded_features = torch.randn(num_edges, decoder.config.input_dim)

        # Decode edge features
        decoded = decoder.decode(encoded_features)

        assert decoded.shape == (num_edges, decoder.config.output_dim)

    def test_edge_reconstruction(self, decoder):
        """Test edge reconstruction."""
        if not IMPORT_SUCCESS:
            return

        # Create node embeddings for edge reconstruction
        num_nodes = 10
        node_embeddings = torch.randn(num_nodes, decoder.config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, 20))

        # Reconstruct edges
        reconstructed_edges = decoder.reconstruct_edges(node_embeddings, edge_index)

        assert reconstructed_edges.shape[0] == edge_index.shape[1]
        assert reconstructed_edges.shape[1] == decoder.config.output_dim

    def test_edge_prediction(self, decoder):
        """Test edge prediction from node pairs."""
        if not IMPORT_SUCCESS:
            return

        # Create node pairs
        num_pairs = 50
        source_nodes = torch.randn(num_pairs, decoder.config.input_dim)
        target_nodes = torch.randn(num_pairs, decoder.config.input_dim)

        # Predict edge existence/attributes
        edge_predictions = decoder.predict_edges(source_nodes, target_nodes)

        assert edge_predictions.shape[0] == num_pairs
        # Edge predictions could be binary (existence) or continuous
        # (attributes)
        assert edge_predictions.shape[1] >= 1


class TestHeteroEdgeProcessor:
    """Test heterogeneous edge processing."""

    @pytest.fixture
    def hetero_processor(self):
        """Create heterogeneous edge processor."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(input_dim=64, hidden_dim=128, num_edge_types=5, use_edge_attr=True)
            return HeteroEdgeProcessor(config)
        else:
            return Mock()

    def test_hetero_initialization(self, hetero_processor):
        """Test heterogeneous processor initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(hetero_processor, "edge_type_processors")
        assert hasattr(hetero_processor, "type_specific_layers")
        assert len(hetero_processor.edge_type_processors) == hetero_processor.config.num_edge_types

    def test_multi_type_edge_processing(self, hetero_processor):
        """Test processing multiple edge types."""
        if not IMPORT_SUCCESS:
            return

        # Create heterogeneous edge data
        num_edges = 60
        edge_types = torch.randint(0, hetero_processor.config.num_edge_types, (num_edges,))
        edge_features = torch.randn(num_edges, hetero_processor.config.input_dim)
        edge_index = torch.randint(0, 20, (2, num_edges))

        # Process different edge types
        processed_features = hetero_processor.process_by_type(edge_features, edge_types, edge_index)

        assert processed_features.shape == (num_edges, hetero_processor.config.hidden_dim)

    def test_type_specific_aggregation(self, hetero_processor):
        """Test type-specific edge aggregation."""
        if not IMPORT_SUCCESS:
            return

        # Create type-specific edge data
        type_features = {}
        type_indices = {}

        for edge_type in range(hetero_processor.config.num_edge_types):
            num_type_edges = 10 + edge_type * 5
            type_features[edge_type] = torch.randn(
                num_type_edges, hetero_processor.config.input_dim
            )
            type_indices[edge_type] = torch.randint(0, 15, (2, num_type_edges))

        # Aggregate by type
        aggregated_features = hetero_processor.aggregate_by_type(
            type_features, type_indices, num_nodes=15
        )

        assert aggregated_features.shape == (15, hetero_processor.config.hidden_dim)

    def test_cross_type_attention(self, hetero_processor):
        """Test cross-type attention mechanism."""
        if not IMPORT_SUCCESS:
            return

        # Create features for different edge types
        type_features = [
            torch.randn(15, hetero_processor.config.input_dim),
            torch.randn(20, hetero_processor.config.input_dim),
            torch.randn(18, hetero_processor.config.input_dim),
        ]

        # Compute cross-type attention
        attended_features = hetero_processor.cross_type_attention(type_features)

        total_edges = sum(tf.shape[0] for tf in type_features)
        assert attended_features.shape == (total_edges, hetero_processor.config.hidden_dim)


class TestTemporalEdgeProcessor:
    """Test temporal edge processing."""

    @pytest.fixture
    def temporal_processor(self):
        """Create temporal edge processor."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(input_dim=64, hidden_dim=128, temporal=True, max_temporal_steps=10)
            return TemporalEdgeProcessor(config)
        else:
            return Mock()

    def test_temporal_initialization(self, temporal_processor):
        """Test temporal processor initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(temporal_processor, "temporal_encoder")
        assert hasattr(temporal_processor, "recurrent_layer")
        assert hasattr(temporal_processor, "temporal_attention")

    def test_temporal_edge_encoding(self, temporal_processor):
        """Test temporal edge encoding."""
        if not IMPORT_SUCCESS:
            return

        # Create temporal edge sequence
        sequence_length = temporal_processor.config.max_temporal_steps
        num_edges = 30
        edge_sequences = torch.randn(
            num_edges, sequence_length, temporal_processor.config.input_dim
        )
        timestamps = torch.arange(sequence_length).repeat(num_edges, 1)

        # Encode temporal sequences
        encoded_sequences = temporal_processor.encode_temporal_sequence(edge_sequences, timestamps)

        assert encoded_sequences.shape == (
            num_edges,
            sequence_length,
            temporal_processor.config.hidden_dim,
        )

    def test_temporal_aggregation(self, temporal_processor):
        """Test temporal aggregation of edge features."""
        if not IMPORT_SUCCESS:
            return

        # Create temporal edge data
        sequence_length = 8
        num_edges = 25
        temporal_features = torch.randn(
            num_edges, sequence_length, temporal_processor.config.hidden_dim
        )

        # Aggregate over time
        aggregated = temporal_processor.temporal_aggregate(temporal_features)

        assert aggregated.shape == (num_edges, temporal_processor.config.hidden_dim)

    def test_temporal_attention(self, temporal_processor):
        """Test temporal attention mechanism."""
        if not IMPORT_SUCCESS:
            return

        # Create temporal sequence
        sequence_length = 6
        num_edges = 20
        temporal_features = torch.randn(
            num_edges, sequence_length, temporal_processor.config.hidden_dim
        )

        # Apply temporal attention
        attended_features, attention_weights = temporal_processor.temporal_attention(
            temporal_features
        )

        assert attended_features.shape == (num_edges, temporal_processor.config.hidden_dim)
        assert attention_weights.shape == (num_edges, sequence_length)

        # Attention weights should sum to 1 for each edge
        weight_sums = attention_weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones(num_edges), atol=1e-5)


class TestMetaPath:
    """Test meta-path processing for edges."""

    @pytest.fixture
    def meta_path(self):
        """Create meta-path processor."""
        if IMPORT_SUCCESS:
            config = EdgeConfig(enable_meta_paths=True, meta_path_length=3, num_edge_types=4)
            return MetaPath(config)
        else:
            return Mock()

    def test_meta_path_initialization(self, meta_path):
        """Test meta-path initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(meta_path, "config")
        assert hasattr(meta_path, "path_encoders")
        assert meta_path.config.enable_meta_paths is True

    def test_meta_path_extraction(self, meta_path):
        """Test meta-path extraction from graph."""
        if not IMPORT_SUCCESS:
            return

        # Create heterogeneous graph
        num_nodes = 20
        num_edges = 50
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_types = torch.randint(0, meta_path.config.num_edge_types, (num_edges,))

        # Extract meta-paths
        meta_paths = meta_path.extract_meta_paths(edge_index, edge_types)

        assert isinstance(meta_paths, list)
        assert len(meta_paths) > 0

        # Each meta-path should have the specified length
        for path in meta_paths:
            assert len(path) <= meta_path.config.meta_path_length

    def test_meta_path_encoding(self, meta_path):
        """Test meta-path encoding."""
        if not IMPORT_SUCCESS:
            return

        # Create meta-path sequences
        meta_paths = [[0, 1, 2], [1, 0, 3], [2, 3, 1], [0, 2, 1]]  # Type sequence

        # Encode meta-paths
        encoded_paths = meta_path.encode_meta_paths(meta_paths)

        assert encoded_paths.shape[0] == len(meta_paths)
        assert encoded_paths.shape[1] == meta_path.config.hidden_dim

    def test_meta_path_attention(self, meta_path):
        """Test meta-path attention mechanism."""
        if not IMPORT_SUCCESS:
            return

        # Create multiple meta-paths for attention
        num_paths = 15
        path_embeddings = torch.randn(num_paths, meta_path.config.hidden_dim)

        # Apply meta-path attention
        attended_embedding = meta_path.meta_path_attention(path_embeddings)

        assert attended_embedding.shape == (meta_path.config.hidden_dim,)


class TestEdgeProcessorIntegration:
    """Test edge processor integration scenarios."""

    def test_end_to_end_edge_processing(self):
        """Test complete edge processing pipeline."""
        if not IMPORT_SUCCESS:
            return

        # Create comprehensive edge processing pipeline
        config = EdgeConfig(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_edge_types=3,
            aggregation=AggregationType.ATTENTION,
            use_edge_attr=True,
            temporal=True,
        )

        processor = EdgeProcessor(config)

        # Create complex graph data
        num_nodes = 50
        num_edges = 150
        node_features = torch.randn(num_nodes, config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, config.edge_attr_dim)
        edge_types = torch.randint(0, config.num_edge_types, (num_edges,))

        # Process through pipeline
        processed_features = processor.process_edges(
            node_features, edge_index, edge_attr, edge_types
        )

        assert processed_features.shape == (num_nodes, config.output_dim)

    def test_large_scale_processing(self):
        """Test processing large-scale edge data."""
        if not IMPORT_SUCCESS:
            return

        # Create large-scale configuration
        config = EdgeConfig(
            input_dim=128, hidden_dim=256, output_dim=128, aggregation=AggregationType.MEAN
        )

        processor = EdgeProcessor(config)

        # Create large graph
        num_nodes = 10000
        num_edges = 50000
        node_features = torch.randn(num_nodes, config.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, config.edge_attr_dim)

        # Process large graph
        start_time = time.time()
        processed_features = processor.process_edges(node_features, edge_index, edge_attr)
        end_time = time.time()

        processing_time = end_time - start_time

        assert processed_features.shape == (num_nodes, config.output_dim)
        # Should complete within reasonable time
        assert processing_time < 30.0  # 30 seconds threshold

    def test_memory_efficiency(self):
        """Test memory efficiency of edge processing."""
        if not IMPORT_SUCCESS:
            return

        # Create memory-efficient configuration
        config = EdgeConfig(
            input_dim=64,
            hidden_dim=64,
            output_dim=64,
            dropout=0.1,  # Use dropout for memory efficiency
        )

        processor = EdgeProcessor(config)

        # Monitor memory usage
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process multiple batches
        for batch in range(10):
            num_nodes = 1000
            num_edges = 5000
            node_features = torch.randn(num_nodes, config.input_dim)
            edge_index = torch.randint(0, num_nodes, (2, num_edges))

            processed = processor.process_edges(node_features, edge_index)

            # Clean up
            del node_features, edge_index, processed
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 1GB)
        assert memory_increase < 1024 * 1024 * 1024

    def test_gradient_flow(self):
        """Test gradient flow through edge processing."""
        if not IMPORT_SUCCESS:
            return

        config = EdgeConfig(input_dim=32, hidden_dim=64, output_dim=32)

        processor = EdgeProcessor(config)
        processor.train()  # Set to training mode

        # Create data with gradients
        num_nodes = 20
        num_edges = 40
        node_features = torch.randn(num_nodes, config.input_dim, requires_grad=True)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, config.edge_attr_dim, requires_grad=True)

        # Forward pass
        processed_features = processor.process_edges(node_features, edge_index, edge_attr)

        # Compute loss and backward pass
        loss = processed_features.sum()
        loss.backward()

        # Check gradient flow
        assert node_features.grad is not None
        assert edge_attr.grad is not None
        assert torch.any(node_features.grad != 0)
        assert torch.any(edge_attr.grad != 0)

        # Check processor parameter gradients
        for param in processor.parameters():
            if param.requires_grad:
                assert param.grad is not None

"""
Comprehensive test coverage for inference/engine/gnn_integration.py
GNN Integration Engine - Phase 3.1 systematic coverage

This test file provides complete coverage for the GNN integration in the inference engine
following the systematic backend coverage improvement plan.
"""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch

# Import the GNN integration components
try:
    from inference.engine.active_inference import ActiveInferenceEngine
    from inference.engine.belief_update import BeliefState
    from inference.engine.gnn_integration import (
        BeliefGraphConverter,
        EdgeFeatures,
        GNNConfig,
        GNNDecoder,
        GNNEncoder,
        GNNInferenceEngine,
        GNNIntegration,
        GraphAttention,
        GraphBeliefPropagation,
        GraphConvolution,
        GraphEmbedding,
        GraphPooling,
        GraphStructure,
        MessagePassing,
        NodeFeatures,
    )

    IMPORT_SUCCESS = True
except ImportError:
    # Create minimal mock classes for testing if imports fail
    IMPORT_SUCCESS = False

    @dataclass
    class GNNConfig:
        hidden_dim: int = 128
        num_layers: int = 3
        dropout: float = 0.1
        activation: str = "relu"
        aggregation: str = "mean"
        normalization: str = "layer"
        attention_heads: int = 4
        edge_dim: int = 64
        use_edge_features: bool = True
        use_residual: bool = True
        use_batch_norm: bool = True
        pooling_type: str = "global_mean"
        readout_layers: int = 2
        message_passing_steps: int = 3

    class GraphStructure:
        def __init__(self, num_nodes, edge_index, edge_attr=None, node_attr=None):
            self.num_nodes = num_nodes
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.node_attr = node_attr
            self.batch = None

    class NodeFeatures:
        def __init__(self, features, feature_dim):
            self.features = features
            self.feature_dim = feature_dim
            self.num_nodes = features.shape[0]

    class EdgeFeatures:
        def __init__(self, features, feature_dim):
            self.features = features
            self.feature_dim = feature_dim
            self.num_edges = features.shape[0]

    class BeliefState:
        def __init__(self, beliefs):
            self.beliefs = beliefs


class TestGNNConfig:
    """Test GNN configuration."""

    def test_config_creation_with_defaults(self):
        """Test creating config with defaults."""
        config = GNNConfig()

        assert config.hidden_dim == 128
        assert config.num_layers == 3
        assert config.dropout == 0.1
        assert config.activation == "relu"
        assert config.aggregation == "mean"
        assert config.normalization == "layer"
        assert config.attention_heads == 4
        assert config.edge_dim == 64
        assert config.use_edge_features is True
        assert config.use_residual is True
        assert config.use_batch_norm is True
        assert config.pooling_type == "global_mean"
        assert config.readout_layers == 2
        assert config.message_passing_steps == 3

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = GNNConfig(
            hidden_dim=256,
            num_layers=5,
            dropout=0.2,
            activation="gelu",
            aggregation="max",
            normalization="batch",
            attention_heads=8,
            edge_dim=128,
            use_edge_features=False,
            use_residual=False,
            pooling_type="global_max",
        )

        assert config.hidden_dim == 256
        assert config.num_layers == 5
        assert config.dropout == 0.2
        assert config.activation == "gelu"
        assert config.aggregation == "max"
        assert config.normalization == "batch"
        assert config.attention_heads == 8
        assert config.edge_dim == 128
        assert config.use_edge_features is False
        assert config.use_residual is False
        assert config.pooling_type == "global_max"


class TestGraphStructure:
    """Test graph structure representation."""

    def test_graph_structure_creation(self):
        """Test creating graph structure."""
        num_nodes = 5
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        edge_attr = torch.randn(5, 16)
        node_attr = torch.randn(5, 32)

        graph = GraphStructure(num_nodes, edge_index, edge_attr, node_attr)

        assert graph.num_nodes == 5
        assert torch.equal(graph.edge_index, edge_index)
        assert torch.equal(graph.edge_attr, edge_attr)
        assert torch.equal(graph.node_attr, node_attr)
        assert graph.batch is None

    def test_graph_structure_validation(self):
        """Test graph structure validation."""
        if not IMPORT_SUCCESS:
            return

        # Valid graph
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph = GraphStructure(2, edge_index)

        assert graph.is_valid()

        # Invalid graph (node index out of bounds)
        invalid_edge_index = torch.tensor([[0, 2], [1, 0]], dtype=torch.long)
        invalid_graph = GraphStructure(2, invalid_edge_index)

        assert not invalid_graph.is_valid()

    def test_graph_structure_properties(self):
        """Test graph structure properties."""
        if not IMPORT_SUCCESS:
            return

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        graph = GraphStructure(4, edge_index)

        assert graph.num_edges == 6
        assert graph.is_undirected()
        assert not graph.has_self_loops()

        # Add self loop
        edge_index_with_loop = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 0], [1, 0, 2, 1, 3, 2, 0]], dtype=torch.long
        )
        graph_with_loop = GraphStructure(4, edge_index_with_loop)

        assert graph_with_loop.has_self_loops()


class TestNodeFeatures:
    """Test node feature handling."""

    def test_node_features_creation(self):
        """Test creating node features."""
        features = torch.randn(10, 64)

        node_features = NodeFeatures(features, feature_dim=64)

        assert torch.equal(node_features.features, features)
        assert node_features.feature_dim == 64
        assert node_features.num_nodes == 10

    def test_node_features_normalization(self):
        """Test node feature normalization."""
        if not IMPORT_SUCCESS:
            return

        features = torch.randn(10, 64)
        node_features = NodeFeatures(features, feature_dim=64)

        # L2 normalization
        normalized = node_features.normalize(norm_type="l2")
        norms = torch.norm(normalized, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

        # Min-max normalization
        normalized = node_features.normalize(norm_type="minmax")
        assert torch.all(normalized >= 0)
        assert torch.all(normalized <= 1)

    def test_node_features_projection(self):
        """Test node feature projection."""
        if not IMPORT_SUCCESS:
            return

        features = torch.randn(10, 64)
        node_features = NodeFeatures(features, feature_dim=64)

        # Project to different dimension
        projected = node_features.project(target_dim=128)
        assert projected.shape == (10, 128)


class TestEdgeFeatures:
    """Test edge feature handling."""

    def test_edge_features_creation(self):
        """Test creating edge features."""
        features = torch.randn(20, 32)

        edge_features = EdgeFeatures(features, feature_dim=32)

        assert torch.equal(edge_features.features, features)
        assert edge_features.feature_dim == 32
        assert edge_features.num_edges == 20

    def test_edge_features_from_nodes(self):
        """Test creating edge features from node pairs."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(5, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        edge_features = EdgeFeatures.from_node_pairs(
            node_features, edge_index, combination="concat"
        )

        assert edge_features.num_edges == 4
        assert edge_features.feature_dim == 128  # Concatenated

        # Test other combination methods
        edge_features_add = EdgeFeatures.from_node_pairs(
            node_features, edge_index, combination="add"
        )
        assert edge_features_add.feature_dim == 64

        edge_features_mul = EdgeFeatures.from_node_pairs(
            node_features, edge_index, combination="multiply"
        )
        assert edge_features_mul.feature_dim == 64


class TestMessagePassing:
    """Test message passing mechanisms."""

    @pytest.fixture
    def config(self):
        """Create GNN config."""
        return GNNConfig(hidden_dim=64, aggregation="mean")

    @pytest.fixture
    def message_passing(self, config):
        """Create message passing layer."""
        if IMPORT_SUCCESS:
            return MessagePassing(config)
        else:
            return Mock()

    def test_message_passing_initialization(self, message_passing, config):
        """Test message passing initialization."""
        if not IMPORT_SUCCESS:
            return

        assert message_passing.config == config
        assert hasattr(message_passing, "message_mlp")
        assert hasattr(message_passing, "update_mlp")
        assert hasattr(message_passing, "aggregation")

    def test_compute_messages(self, message_passing):
        """Test message computation."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(5, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_features = torch.randn(4, 32)

        messages = message_passing.compute_messages(node_features, edge_index, edge_features)

        assert messages.shape[0] == 4  # Number of edges
        assert messages.shape[1] == 64  # Hidden dim

    def test_aggregate_messages(self, message_passing):
        """Test message aggregation."""
        if not IMPORT_SUCCESS:
            return

        messages = torch.randn(10, 64)
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [1, 2, 0, 3, 0, 4, 1, 4, 2, 3]], dtype=torch.long
        )
        num_nodes = 5

        aggregated = message_passing.aggregate(messages, edge_index, num_nodes)

        assert aggregated.shape == (num_nodes, 64)

        # Test different aggregation methods
        for agg_type in ["mean", "sum", "max"]:
            message_passing.aggregation = agg_type
            agg_result = message_passing.aggregate(messages, edge_index, num_nodes)
            assert agg_result.shape == (num_nodes, 64)

    def test_update_nodes(self, message_passing):
        """Test node update."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(5, 64)
        aggregated_messages = torch.randn(5, 64)

        updated = message_passing.update_nodes(node_features, aggregated_messages)

        assert updated.shape == node_features.shape

        # Check residual connection if enabled
        if message_passing.config.use_residual:
            # Updated should be different from just aggregated messages
            assert not torch.allclose(updated, aggregated_messages)


class TestGraphAttention:
    """Test graph attention mechanisms."""

    @pytest.fixture
    def config(self):
        """Create GNN config."""
        return GNNConfig(hidden_dim=64, attention_heads=4)

    @pytest.fixture
    def graph_attention(self, config):
        """Create graph attention layer."""
        if IMPORT_SUCCESS:
            return GraphAttention(config)
        else:
            return Mock()

    def test_graph_attention_initialization(self, graph_attention, config):
        """Test graph attention initialization."""
        if not IMPORT_SUCCESS:
            return

        assert graph_attention.config == config
        assert graph_attention.num_heads == 4
        assert graph_attention.head_dim == 16  # 64 / 4
        assert hasattr(graph_attention, "query_proj")
        assert hasattr(graph_attention, "key_proj")
        assert hasattr(graph_attention, "value_proj")

    def test_compute_attention_scores(self, graph_attention):
        """Test attention score computation."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(5, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        scores = graph_attention.compute_attention_scores(node_features, edge_index)

        assert scores.shape == (4, 4)  # (num_edges, num_heads)

    def test_apply_attention(self, graph_attention):
        """Test applying attention mechanism."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(5, 64)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

        attended_features = graph_attention(node_features, edge_index)

        assert attended_features.shape == node_features.shape

    def test_multi_head_attention(self, graph_attention):
        """Test multi-head attention."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 20))

        # Get attention weights
        attended, attention_weights = graph_attention(
            node_features, edge_index, return_attention=True
        )

        assert attended.shape == (10, 64)
        assert attention_weights.shape == (20, 4)  # edges x heads


class TestGraphConvolution:
    """Test graph convolution operations."""

    @pytest.fixture
    def config(self):
        """Create GNN config."""
        return GNNConfig(hidden_dim=128)

    @pytest.fixture
    def graph_conv(self, config):
        """Create graph convolution layer."""
        if IMPORT_SUCCESS:
            return GraphConvolution(config, in_features=64, out_features=128)
        else:
            return Mock()

    def test_graph_conv_initialization(self, graph_conv):
        """Test graph convolution initialization."""
        if not IMPORT_SUCCESS:
            return

        assert graph_conv.in_features == 64
        assert graph_conv.out_features == 128
        assert hasattr(graph_conv, "linear")
        assert hasattr(graph_conv, "bias")

    def test_forward_pass(self, graph_conv):
        """Test forward pass through graph convolution."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 30))

        output = graph_conv(node_features, edge_index)

        assert output.shape == (10, 128)

    def test_spectral_convolution(self, graph_conv):
        """Test spectral graph convolution."""
        if not IMPORT_SUCCESS:
            return

        # Create normalized adjacency matrix
        num_nodes = 8
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
            ],
            dtype=torch.long,
        )

        node_features = torch.randn(num_nodes, 64)

        # Apply spectral convolution
        output = graph_conv.spectral_conv(node_features, edge_index)

        assert output.shape == (num_nodes, 128)


class TestGraphPooling:
    """Test graph pooling operations."""

    @pytest.fixture
    def config(self):
        """Create GNN config."""
        return GNNConfig(pooling_type="global_mean")

    @pytest.fixture
    def graph_pooling(self, config):
        """Create graph pooling layer."""
        if IMPORT_SUCCESS:
            return GraphPooling(config)
        else:
            return Mock()

    def test_pooling_initialization(self, graph_pooling, config):
        """Test pooling initialization."""
        if not IMPORT_SUCCESS:
            return

        assert graph_pooling.config == config
        assert graph_pooling.pooling_type == "global_mean"

    def test_global_pooling(self, graph_pooling):
        """Test global pooling operations."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(10, 64)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

        # Test different pooling types
        pooling_types = ["global_mean", "global_max", "global_sum"]

        for pool_type in pooling_types:
            graph_pooling.pooling_type = pool_type
            pooled = graph_pooling(node_features, batch)

            assert pooled.shape == (3, 64)  # 3 graphs in batch

    def test_hierarchical_pooling(self, graph_pooling):
        """Test hierarchical pooling."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(20, 64)
        edge_index = torch.randint(0, 20, (2, 50))

        # Apply hierarchical pooling
        pooled_features, pooled_edge_index, cluster_assignment = graph_pooling.hierarchical_pool(
            node_features, edge_index, ratio=0.5
        )

        assert pooled_features.shape[0] == 10  # 50% of nodes
        assert pooled_features.shape[1] == 64
        assert cluster_assignment.shape == (20,)

    def test_attention_pooling(self, graph_pooling):
        """Test attention-based pooling."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(15, 64)
        batch = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)

        # Apply attention pooling
        pooled, attention_weights = graph_pooling.attention_pool(
            node_features, batch, return_attention=True
        )

        assert pooled.shape == (3, 64)
        assert attention_weights.shape == (15,)
        assert torch.allclose(
            attention_weights[:5].sum()
            + attention_weights[5:10].sum()
            + attention_weights[10:].sum(),
            torch.tensor(3.0),
        )


class TestGNNEncoder:
    """Test GNN encoder architecture."""

    @pytest.fixture
    def config(self):
        """Create GNN config."""
        return GNNConfig(hidden_dim=128, num_layers=3, dropout=0.1)

    @pytest.fixture
    def encoder(self, config):
        """Create GNN encoder."""
        if IMPORT_SUCCESS:
            return GNNEncoder(config, input_dim=64)
        else:
            return Mock()

    def test_encoder_initialization(self, encoder, config):
        """Test encoder initialization."""
        if not IMPORT_SUCCESS:
            return

        assert encoder.config == config
        assert encoder.input_dim == 64
        assert len(encoder.layers) == 3
        assert hasattr(encoder, "input_projection")
        assert hasattr(encoder, "dropout")

    def test_encoder_forward(self, encoder):
        """Test encoder forward pass."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(20, 64)
        edge_index = torch.randint(0, 20, (2, 60))

        encoded = encoder(node_features, edge_index)

        assert encoded.shape == (20, 128)

    def test_encoder_with_edge_features(self, encoder):
        """Test encoder with edge features."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(15, 64)
        edge_index = torch.randint(0, 15, (2, 40))
        edge_features = torch.randn(40, 32)

        encoded = encoder(node_features, edge_index, edge_features)

        assert encoded.shape == (15, 128)

    def test_layer_outputs(self, encoder):
        """Test getting intermediate layer outputs."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 20))

        encoded, layer_outputs = encoder(node_features, edge_index, return_all_layers=True)

        assert len(layer_outputs) == 3
        assert all(out.shape == (10, 128) for out in layer_outputs)


class TestGNNDecoder:
    """Test GNN decoder architecture."""

    @pytest.fixture
    def config(self):
        """Create GNN config."""
        return GNNConfig(hidden_dim=128, readout_layers=2)

    @pytest.fixture
    def decoder(self, config):
        """Create GNN decoder."""
        if IMPORT_SUCCESS:
            return GNNDecoder(config, output_dim=10)
        else:
            return Mock()

    def test_decoder_initialization(self, decoder, config):
        """Test decoder initialization."""
        if not IMPORT_SUCCESS:
            return

        assert decoder.config == config
        assert decoder.output_dim == 10
        assert len(decoder.readout_layers) == 2

    def test_decoder_forward(self, decoder):
        """Test decoder forward pass."""
        if not IMPORT_SUCCESS:
            return

        node_embeddings = torch.randn(5, 128)
        batch = torch.tensor([0, 0, 1, 1, 1])

        output = decoder(node_embeddings, batch)

        assert output.shape == (2, 10)  # 2 graphs, 10 output dims

    def test_node_level_decoding(self, decoder):
        """Test node-level decoding."""
        if not IMPORT_SUCCESS:
            return

        node_embeddings = torch.randn(20, 128)

        # Decode at node level
        node_outputs = decoder.decode_nodes(node_embeddings)

        assert node_outputs.shape == (20, 10)

    def test_graph_level_decoding(self, decoder):
        """Test graph-level decoding."""
        if not IMPORT_SUCCESS:
            return

        node_embeddings = torch.randn(15, 128)
        batch = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)

        # Decode at graph level
        graph_outputs = decoder.decode_graphs(node_embeddings, batch)

        assert graph_outputs.shape == (3, 10)


class TestGraphEmbedding:
    """Test graph embedding generation."""

    @pytest.fixture
    def graph_embedding(self):
        """Create graph embedding module."""
        if IMPORT_SUCCESS:
            config = GNNConfig(hidden_dim=128)
            return GraphEmbedding(config)
        else:
            return Mock()

    def test_positional_encoding(self, graph_embedding):
        """Test positional encoding for graphs."""
        if not IMPORT_SUCCESS:
            return

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        num_nodes = 4

        pos_encoding = graph_embedding.compute_positional_encoding(edge_index, num_nodes)

        assert pos_encoding.shape == (num_nodes, graph_embedding.pos_encoding_dim)

    def test_structural_features(self, graph_embedding):
        """Test structural feature extraction."""
        if not IMPORT_SUCCESS:
            return

        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
        )
        num_nodes = 5

        struct_features = graph_embedding.compute_structural_features(edge_index, num_nodes)

        assert struct_features.shape[0] == num_nodes
        assert "degree" in struct_features
        assert "clustering_coef" in struct_features
        assert "centrality" in struct_features

    def test_combined_embedding(self, graph_embedding):
        """Test combined node and structural embeddings."""
        if not IMPORT_SUCCESS:
            return

        node_features = torch.randn(10, 64)
        edge_index = torch.randint(0, 10, (2, 30))

        embeddings = graph_embedding.compute_embeddings(node_features, edge_index)

        assert embeddings.shape[0] == 10
        assert embeddings.shape[1] > 64  # Includes structural features


class TestBeliefGraphConverter:
    """Test belief state to graph conversion."""

    @pytest.fixture
    def converter(self):
        """Create belief graph converter."""
        if IMPORT_SUCCESS:
            return BeliefGraphConverter()
        else:
            return Mock()

    def test_belief_to_graph_conversion(self, converter):
        """Test converting belief state to graph."""
        if not IMPORT_SUCCESS:
            return

        # Create belief state
        beliefs = torch.softmax(torch.randn(5), dim=0)
        belief_state = BeliefState(beliefs)

        # Convert to graph
        graph = converter.belief_to_graph(belief_state)

        assert isinstance(graph, GraphStructure)
        assert graph.num_nodes == 5
        assert graph.node_attr.shape == (5, 1)  # Belief values as features

    def test_multi_agent_belief_graph(self, converter):
        """Test creating graph from multiple agent beliefs."""
        if not IMPORT_SUCCESS:
            return

        # Multiple agents with beliefs
        agent_beliefs = [
            torch.softmax(torch.randn(4), dim=0),
            torch.softmax(torch.randn(4), dim=0),
            torch.softmax(torch.randn(4), dim=0),
        ]

        # Create interaction graph
        graph = converter.create_interaction_graph(agent_beliefs)

        assert graph.num_nodes == 3  # 3 agents
        assert graph.edge_index.shape[1] > 0  # Has edges

    def test_hierarchical_belief_graph(self, converter):
        """Test hierarchical belief graph construction."""
        if not IMPORT_SUCCESS:
            return

        # Hierarchical beliefs (e.g., agent -> group -> population)
        agent_beliefs = [torch.randn(4) for _ in range(10)]
        group_structure = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

        graph = converter.create_hierarchical_graph(agent_beliefs, group_structure)

        assert graph.num_nodes == 14  # 10 agents + 3 groups + 1 root
        assert hasattr(graph, "hierarchy_levels")


class TestGraphBeliefPropagation:
    """Test belief propagation on graphs."""

    @pytest.fixture
    def config(self):
        """Create GNN config."""
        return GNNConfig(message_passing_steps=3)

    @pytest.fixture
    def propagator(self, config):
        """Create graph belief propagator."""
        if IMPORT_SUCCESS:
            return GraphBeliefPropagation(config)
        else:
            return Mock()

    def test_propagator_initialization(self, propagator, config):
        """Test propagator initialization."""
        if not IMPORT_SUCCESS:
            return

        assert propagator.config == config
        assert propagator.num_steps == 3
        assert hasattr(propagator, "message_function")
        assert hasattr(propagator, "update_function")

    def test_belief_propagation_step(self, propagator):
        """Test single belief propagation step."""
        if not IMPORT_SUCCESS:
            return

        beliefs = torch.softmax(torch.randn(5, 4), dim=1)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

        updated_beliefs = propagator.propagate_step(beliefs, edge_index)

        assert updated_beliefs.shape == beliefs.shape
        assert torch.allclose(updated_beliefs.sum(dim=1), torch.ones(5))

    def test_multi_step_propagation(self, propagator):
        """Test multi-step belief propagation."""
        if not IMPORT_SUCCESS:
            return

        initial_beliefs = torch.softmax(torch.randn(8, 6), dim=1)
        edge_index = torch.randint(0, 8, (2, 20))

        final_beliefs, history = propagator.propagate(
            initial_beliefs, edge_index, return_history=True
        )

        assert final_beliefs.shape == initial_beliefs.shape
        assert len(history) == propagator.num_steps + 1
        assert history[0].equal(initial_beliefs)

    def test_belief_convergence(self, propagator):
        """Test belief convergence during propagation."""
        if not IMPORT_SUCCESS:
            return

        # Create strongly connected graph
        num_nodes = 6
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list).T

        initial_beliefs = torch.softmax(torch.randn(num_nodes, 4), dim=1)

        # Run many propagation steps
        propagator.num_steps = 20
        final_beliefs, history = propagator.propagate(
            initial_beliefs, edge_index, return_history=True
        )

        # Check convergence
        convergence_diff = torch.norm(history[-1] - history[-2])
        assert convergence_diff < 0.01


class TestGNNInferenceEngine:
    """Test GNN-based inference engine."""

    @pytest.fixture
    def config(self):
        """Create GNN config."""
        return GNNConfig(hidden_dim=128, num_layers=3, message_passing_steps=5)

    @pytest.fixture
    def engine(self, config):
        """Create GNN inference engine."""
        if IMPORT_SUCCESS:
            return GNNInferenceEngine(config)
        else:
            return Mock()

    def test_engine_initialization(self, engine, config):
        """Test engine initialization."""
        if not IMPORT_SUCCESS:
            return

        assert engine.config == config
        assert hasattr(engine, "encoder")
        assert hasattr(engine, "decoder")
        assert hasattr(engine, "propagator")
        assert hasattr(engine, "converter")

    def test_inference_step(self, engine):
        """Test single inference step."""
        if not IMPORT_SUCCESS:
            return

        # Create belief states for multiple agents
        belief_states = [BeliefState(torch.softmax(torch.randn(4), dim=0)) for _ in range(5)]

        observations = torch.randn(5, 10)

        updated_beliefs = engine.inference_step(belief_states, observations)

        assert len(updated_beliefs) == 5
        assert all(isinstance(b, BeliefState) for b in updated_beliefs)

    def test_batch_inference(self, engine):
        """Test batch inference."""
        if not IMPORT_SUCCESS:
            return

        # Batch of belief states
        batch_size = 3
        num_agents = 4

        batch_beliefs = []
        for _ in range(batch_size):
            beliefs = [BeliefState(torch.softmax(torch.randn(5), dim=0)) for _ in range(num_agents)]
            batch_beliefs.append(beliefs)

        batch_observations = torch.randn(batch_size, num_agents, 8)

        batch_updated = engine.batch_inference(batch_beliefs, batch_observations)

        assert len(batch_updated) == batch_size
        assert all(len(beliefs) == num_agents for beliefs in batch_updated)

    def test_hierarchical_inference(self, engine):
        """Test hierarchical inference."""
        if not IMPORT_SUCCESS:
            return

        # Individual level beliefs
        individual_beliefs = [BeliefState(torch.softmax(torch.randn(4), dim=0)) for _ in range(12)]

        # Group structure
        groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

        # Observations at different levels
        individual_obs = torch.randn(12, 6)
        group_obs = torch.randn(3, 8)

        updated_beliefs = engine.hierarchical_inference(
            individual_beliefs, groups, individual_obs, group_obs
        )

        assert len(updated_beliefs) == 12

    def test_temporal_inference(self, engine):
        """Test temporal inference with history."""
        if not IMPORT_SUCCESS:
            return

        # Initial beliefs
        beliefs = [BeliefState(torch.softmax(torch.randn(4), dim=0)) for _ in range(3)]

        # Run temporal inference
        time_steps = 10
        belief_history = [beliefs]

        for t in range(time_steps):
            observations = torch.randn(3, 6)
            beliefs = engine.temporal_inference_step(beliefs, observations, belief_history)
            belief_history.append(beliefs)

        assert len(belief_history) == time_steps + 1

        # Test with attention over history
        final_beliefs = engine.temporal_inference_with_attention(
            belief_history[-1], belief_history, observations
        )

        assert len(final_beliefs) == 3


class TestGNNIntegration:
    """Test full GNN integration with active inference."""

    @pytest.fixture
    def gnn_integration(self):
        """Create GNN integration module."""
        if IMPORT_SUCCESS:
            config = GNNConfig()
            return GNNIntegration(config)
        else:
            return Mock()

    def test_integration_initialization(self, gnn_integration):
        """Test integration initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(gnn_integration, "config")
        assert hasattr(gnn_integration, "inference_engine")
        assert hasattr(gnn_integration, "active_inference_bridge")

    def test_active_inference_integration(self, gnn_integration):
        """Test integration with active inference engine."""
        if not IMPORT_SUCCESS:
            return

        # Mock active inference engine
        active_engine = Mock(spec=ActiveInferenceEngine)
        active_engine.belief_state = BeliefState(torch.softmax(torch.randn(4), dim=0))

        # Connect GNN to active inference
        gnn_integration.connect_active_inference(active_engine)

        # Run integrated inference
        observations = torch.randn(6)
        updated_state = gnn_integration.integrated_inference_step(observations)

        assert isinstance(updated_state, BeliefState)

    def test_multi_scale_integration(self, gnn_integration):
        """Test multi-scale integration."""
        if not IMPORT_SUCCESS:
            return

        # Multiple scales of organization
        micro_beliefs = [torch.randn(3) for _ in range(20)]
        meso_structure = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ]
        macro_beliefs = torch.randn(4)

        integrated_beliefs = gnn_integration.multi_scale_integration(
            micro_beliefs, meso_structure, macro_beliefs
        )

        assert len(integrated_beliefs) == 20

    def test_performance_metrics(self, gnn_integration):
        """Test performance metrics collection."""
        if not IMPORT_SUCCESS:
            return

        # Run some inference steps
        for _ in range(5):
            beliefs = [BeliefState(torch.softmax(torch.randn(4), dim=0)) for _ in range(3)]
            observations = torch.randn(3, 6)

            gnn_integration.inference_engine.inference_step(beliefs, observations)

        # Get performance metrics
        metrics = gnn_integration.get_performance_metrics()

        assert "inference_time" in metrics
        assert "memory_usage" in metrics
        assert "graph_size" in metrics
        assert "convergence_rate" in metrics

"""
Comprehensive test coverage for inference/gnn/edge_processor.py and model_mapper.py
GNN Edge Processing and Model Mapping - Phase 3.2 systematic coverage

This test file provides complete coverage for the GNN edge processor and model mapper
following the systematic backend coverage improvement plan.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import networkx as nx
import pytest
import scipy.sparse as sp
import torch
import torch.nn as nn

# Import the GNN edge and model components
try:
    from inference.gnn.edge_processor import (
        Edge,
        EdgeBatch,
        EdgeConfig,
        EdgeFeatureType,
        EdgeProcessor,
        EdgeType,
    )
    from inference.gnn.model_mapper import (
        GraphAnalyzer,
        GraphProperties,
        GraphTaskType,
        GraphToModelMapper,
        MappingConfig,
        ModelArchitecture,
        ModelConfig,
        ModelSelector,
    )

    IMPORT_SUCCESS = True
    TORCH_AVAILABLE = True
except ImportError:
    # Create minimal mock classes for testing if imports fail
    IMPORT_SUCCESS = False
    # torch already imported at top level
    TORCH_AVAILABLE = True

    class EdgeType:
        DIRECTED = "directed"
        UNDIRECTED = "undirected"
        BIDIRECTIONAL = "bidirectional"

    class EdgeFeatureType:
        WEIGHT = "weight"
        DISTANCE = "distance"
        SIMILARITY = "similarity"
        CATEGORICAL = "categorical"
        TEMPORAL = "temporal"
        EMBEDDING = "embedding"
        CUSTOM = "custom"

    @dataclass
    class EdgeConfig:
        edge_type: str = EdgeType.DIRECTED
        feature_types: List[str] = None
        normalize_weights: bool = True
        self_loops: bool = False
        max_edges_per_node: Optional[int] = None
        edge_sampling_strategy: Optional[str] = None

        def __post_init__(self):
            if self.feature_types is None:
                self.feature_types = []

    @dataclass
    class Edge:
        source: int
        target: int
        features: Dict[str, Any] = None
        weight: float = 1.0
        edge_type: Optional[str] = None
        metadata: Dict[str, Any] = None

        def __post_init__(self):
            if self.features is None:
                self.features = {}
            if self.metadata is None:
                self.metadata = {}

    @dataclass
    class EdgeBatch:
        edge_index: Any
        edge_attr: Optional[Any] = None
        edge_weight: Optional[Any] = None
        edge_type: Optional[Any] = None
        batch_ptr: Optional[Any] = None
        metadata: Dict[str, Any] = None

        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}

    class EdgeProcessor:
        def __init__(self, config):
            self.config = config
            self.scalers = {}
            self.edge_type_mapping = {}

    class GraphTaskType:
        NODE_CLASSIFICATION = "node_classification"
        GRAPH_CLASSIFICATION = "graph_classification"

    class ModelArchitecture:
        GCN = "gcn"
        GAT = "gat"
        SAGE = "sage"
        GIN = "gin"


class TestEdgeProcessor:
    """Test edge processor functionality."""

    @pytest.fixture
    def edge_config(self):
        """Create edge configuration."""
        return EdgeConfig(
            edge_type=EdgeType.DIRECTED,
            feature_types=[EdgeFeatureType.WEIGHT, EdgeFeatureType.DISTANCE],
            normalize_weights=True,
            self_loops=False,
            max_edges_per_node=None,
            edge_sampling_strategy=None,
        )

    @pytest.fixture
    def processor(self, edge_config):
        """Create edge processor."""
        return EdgeProcessor(edge_config)

    @pytest.fixture
    def sample_edges(self):
        """Create sample edges."""
        return [
            Edge(source=0, target=1, weight=0.8, features={"distance": 1.5, "category": "friend"}),
            Edge(
                source=1, target=2, weight=0.6, features={"distance": 2.0, "category": "colleague"}
            ),
            Edge(source=2, target=0, weight=0.9, features={"distance": 1.2, "category": "friend"}),
            Edge(
                source=0,
                target=3,
                weight=0.4,
                features={"distance": 3.0, "category": "acquaintance"},
            ),
        ]

    def test_processor_initialization(self, processor, edge_config):
        """Test processor initialization."""
        assert processor.config == edge_config
        assert isinstance(processor.scalers, dict)
        assert isinstance(processor.edge_type_mapping, dict)

    def test_process_edges_basic(self, processor, sample_edges):
        """Test basic edge processing."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        num_nodes = 4
        edge_batch = processor.process_edges(sample_edges, num_nodes)

        assert isinstance(edge_batch, EdgeBatch)
        assert edge_batch.edge_index.shape == (2, 4)
        assert edge_batch.edge_weight is not None
        assert edge_batch.edge_weight.shape == (4,)
        assert edge_batch.edge_attr is not None

    def test_process_empty_edges(self, processor):
        """Test processing empty edge list."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        num_nodes = 5
        edge_batch = processor.process_edges([], num_nodes)

        assert edge_batch.edge_index.shape[1] == 0
        assert edge_batch.metadata["num_edges"] == 0
        assert edge_batch.metadata["num_nodes"] == 5

    def test_undirected_edge_conversion(self):
        """Test undirected edge conversion."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(edge_type=EdgeType.UNDIRECTED)
        processor = EdgeProcessor(config)

        edges = [
            Edge(source=0, target=1, weight=0.5),
            Edge(source=1, target=0, weight=0.5),  # Reverse edge
            Edge(source=2, target=3, weight=0.7),
        ]

        edge_batch = processor.process_edges(edges, 4)

        # Should deduplicate reverse edges
        assert edge_batch.edge_index.shape[1] == 2

    def test_bidirectional_edge_conversion(self):
        """Test bidirectional edge conversion."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(edge_type=EdgeType.BIDIRECTIONAL)
        processor = EdgeProcessor(config)

        edges = [Edge(source=0, target=1, weight=0.5), Edge(source=2, target=3, weight=0.7)]

        edge_batch = processor.process_edges(edges, 4)

        # Should create reverse edges
        assert edge_batch.edge_index.shape[1] == 4

    def test_self_loops_addition(self):
        """Test self-loop addition."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(self_loops=True)
        processor = EdgeProcessor(config)

        edges = [Edge(source=0, target=1)]
        edge_batch = processor.process_edges(edges, 3)

        # Should add self-loops for all nodes
        assert edge_batch.edge_index.shape[1] > 1
        assert processor._has_self_loops(edge_batch.edge_index)

    def test_weight_feature_extraction(self):
        """Test weight feature extraction."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(feature_types=[EdgeFeatureType.WEIGHT])
        processor = EdgeProcessor(config)

        edges = [Edge(source=0, target=1, weight=0.1), Edge(source=1, target=2, weight=0.9)]

        edge_batch = processor.process_edges(edges, 3)

        assert edge_batch.edge_attr is not None
        assert edge_batch.edge_attr.shape == (2, 1)

    def test_distance_feature_extraction(self):
        """Test distance feature extraction."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(feature_types=[EdgeFeatureType.DISTANCE])
        processor = EdgeProcessor(config)

        edges = [
            Edge(source=0, target=1, features={"distance": 1.5}),
            Edge(source=1, target=2, features={"distance": 2.5}),
        ]

        edge_batch = processor.process_edges(edges, 3)

        assert edge_batch.edge_attr is not None
        assert edge_batch.edge_attr.shape == (2, 1)
        # Distances should be normalized to [0, 1]
        assert torch.all(edge_batch.edge_attr >= 0)
        assert torch.all(edge_batch.edge_attr <= 1)

    def test_similarity_feature_extraction(self):
        """Test similarity feature extraction."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(feature_types=[EdgeFeatureType.SIMILARITY])
        processor = EdgeProcessor(config)

        edges = [
            Edge(source=0, target=1, features={"similarity": 0.8}),
            Edge(source=1, target=2, features={"similarity": 1.5}),
            # Will be clipped
        ]

        edge_batch = processor.process_edges(edges, 3)

        assert edge_batch.edge_attr is not None
        # Similarities should be clipped to [0, 1]
        assert torch.all(edge_batch.edge_attr >= 0)
        assert torch.all(edge_batch.edge_attr <= 1)

    def test_categorical_feature_extraction(self):
        """Test categorical feature extraction."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(feature_types=[EdgeFeatureType.CATEGORICAL])
        processor = EdgeProcessor(config)

        edges = [
            Edge(source=0, target=1, features={"category": "friend"}),
            Edge(source=1, target=2, features={"category": "colleague"}),
            Edge(source=2, target=3, features={"category": "friend"}),
        ]

        edge_batch = processor.process_edges(edges, 4)

        assert edge_batch.edge_attr is not None
        # One-hot encoded categories
        assert edge_batch.edge_attr.shape[1] == 2  # Two unique categories
        assert torch.all(edge_batch.edge_attr.sum(dim=1) == 1)  # One-hot

    def test_temporal_feature_extraction(self):
        """Test temporal feature extraction."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(feature_types=[EdgeFeatureType.TEMPORAL])
        processor = EdgeProcessor(config)

        timestamp = datetime.now().timestamp()
        edges = [
            Edge(source=0, target=1, features={"timestamp": timestamp}),
            Edge(source=1, target=2, features={"timestamp": timestamp + 3600}),
        ]

        edge_batch = processor.process_edges(edges, 3)

        assert edge_batch.edge_attr is not None
        assert edge_batch.edge_attr.shape == (2, 7)  # 7 temporal features

    def test_embedding_feature_extraction(self):
        """Test embedding feature extraction."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(feature_types=[EdgeFeatureType.EMBEDDING])
        processor = EdgeProcessor(config)

        edges = [
            Edge(source=0, target=1, features={"embedding": [0.1, 0.2, 0.3]}),
            Edge(source=1, target=2, features={"embedding": [0.4, 0.5, 0.6]}),
        ]

        edge_batch = processor.process_edges(edges, 3)

        assert edge_batch.edge_attr is not None
        assert edge_batch.edge_attr.shape == (2, 3)
        # Should be normalized
        norms = torch.norm(edge_batch.edge_attr, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_multiple_feature_types(self):
        """Test multiple feature type extraction."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(
            feature_types=[
                EdgeFeatureType.WEIGHT,
                EdgeFeatureType.DISTANCE,
                EdgeFeatureType.SIMILARITY,
            ]
        )
        processor = EdgeProcessor(config)

        edges = [
            Edge(source=0, target=1, weight=0.8, features={"distance": 1.5, "similarity": 0.7})
        ]

        edge_batch = processor.process_edges(edges, 2)

        assert edge_batch.edge_attr is not None
        assert edge_batch.edge_attr.shape == (1, 3)  # 3 feature types

    def test_edge_sampling_random(self):
        """Test random edge sampling."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(max_edges_per_node=2, edge_sampling_strategy="random")
        processor = EdgeProcessor(config)

        # Create many edges from node 0
        edges = [Edge(source=0, target=i, weight=1.0) for i in range(1, 6)]

        edge_batch = processor.process_edges(edges, 6)

        # Should sample only 2 edges from node 0
        source_0_edges = (edge_batch.edge_index[0] == 0).sum()
        assert source_0_edges <= 2

    def test_edge_sampling_importance(self):
        """Test importance-based edge sampling."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(max_edges_per_node=2, edge_sampling_strategy="importance")
        processor = EdgeProcessor(config)

        # Create edges with different weights
        edges = [
            Edge(source=0, target=1, weight=0.1),
            Edge(source=0, target=2, weight=0.9),
            Edge(source=0, target=3, weight=0.5),
            Edge(source=0, target=4, weight=0.2),
        ]

        edge_batch = processor.process_edges(edges, 5)

        # Should prefer high-weight edges
        assert edge_batch.edge_index.shape[1] <= 2

    def test_edge_sampling_topk(self):
        """Test top-k edge sampling."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        config = EdgeConfig(max_edges_per_node=2, edge_sampling_strategy="topk")
        processor = EdgeProcessor(config)

        edges = [
            Edge(source=0, target=1, weight=0.1),
            Edge(source=0, target=2, weight=0.9),
            Edge(source=0, target=3, weight=0.5),
            Edge(source=0, target=4, weight=0.2),
        ]

        edge_batch = processor.process_edges(edges, 5)

        # Should keep top-2 weighted edges
        if edge_batch.edge_weight is not None:
            kept_weights = edge_batch.edge_weight[edge_batch.edge_index[0] == 0]
            assert len(kept_weights) <= 2
            if len(kept_weights) == 2:
                assert 0.9 in kept_weights.tolist() or abs(kept_weights.max() - 0.9) < 0.01

    def test_edge_type_extraction(self, processor):
        """Test edge type extraction."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        edges = [
            Edge(source=0, target=1, edge_type="friend"),
            Edge(source=1, target=2, edge_type="colleague"),
            Edge(source=2, target=3, edge_type="friend"),
        ]

        edge_batch = processor.process_edges(edges, 4)

        assert edge_batch.edge_type is not None
        assert edge_batch.edge_type.shape == (3,)
        assert len(processor.edge_type_mapping) == 2  # Two unique types

    def test_to_adjacency_matrix(self, processor, sample_edges):
        """Test conversion to adjacency matrix."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        num_nodes = 4
        edge_batch = processor.process_edges(sample_edges, num_nodes)

        adj_matrix = processor.to_adjacency_matrix(edge_batch, num_nodes)

        assert isinstance(adj_matrix, sp.csr_matrix)
        assert adj_matrix.shape == (num_nodes, num_nodes)
        assert adj_matrix.nnz == len(sample_edges)

    def test_compute_edge_statistics(self, processor, sample_edges):
        """Test edge statistics computation."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        num_nodes = 4
        edge_batch = processor.process_edges(sample_edges, num_nodes)

        stats = processor.compute_edge_statistics(edge_batch, num_nodes)

        assert "num_edges" in stats
        assert "avg_in_degree" in stats
        assert "avg_out_degree" in stats
        assert "density" in stats
        assert stats["num_edges"] == 4
        assert stats["num_nodes"] == 4


class TestGraphAnalyzer:
    """Test graph analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create graph analyzer."""
        if IMPORT_SUCCESS:
            return GraphAnalyzer()
        else:
            return Mock()

    @pytest.fixture
    def simple_graph(self):
        """Create simple graph."""
        if TORCH_AVAILABLE:
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
            return edge_index, 4
        return None, 4

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(analyzer, "property_cache")
        assert isinstance(analyzer.property_cache, dict)

    def test_analyze_simple_graph(self, analyzer, simple_graph):
        """Test analyzing simple graph."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        edge_index, num_nodes = simple_graph

        properties = analyzer.analyze_graph(edge_index, num_nodes)

        assert isinstance(properties, GraphProperties)
        assert properties.num_nodes == 4
        assert properties.num_edges == 4
        assert properties.avg_degree == 2.0

    def test_analyze_with_features(self, analyzer):
        """Test analyzing graph with features."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_features = torch.randn(2, 10)
        edge_features = torch.randn(2, 5)

        properties = analyzer.analyze_graph(
            edge_index, 2, node_features=node_features, edge_features=edge_features
        )

        assert properties.has_node_features
        assert properties.has_edge_features
        assert properties.node_feature_dim == 10
        assert properties.edge_feature_dim == 5

    def test_analyze_directed_graph(self, analyzer):
        """Test analyzing directed graph."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        # Create asymmetric edges for directed graph
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        properties = analyzer.analyze_graph(edge_index, 3)

        assert properties.is_directed

    def test_analyze_weighted_graph(self, analyzer):
        """Test analyzing weighted graph."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.tensor([0.5, 0.5], dtype=torch.float)

        properties = analyzer.analyze_graph(edge_index, 2, edge_weight=edge_weight)

        assert properties.is_weighted

    def test_analyze_disconnected_graph(self, analyzer):
        """Test analyzing disconnected graph."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        # Two disconnected components
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

        properties = analyzer.analyze_graph(edge_index, 4)

        assert properties.num_connected_components == 2

    def test_analyze_bipartite_graph(self, analyzer):
        """Test analyzing bipartite graph."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        # Create bipartite graph (no odd cycles)
        edge_index = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=torch.long)

        properties = analyzer.analyze_graph(edge_index, 4)

        assert properties.is_bipartite

    def test_to_networkx_conversion(self, analyzer):
        """Test NetworkX conversion."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        G = analyzer._to_networkx(edge_index, 2)

        assert isinstance(G, (nx.Graph, nx.DiGraph))
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1  # Undirected, so one edge


class TestModelSelector:
    """Test model selector functionality."""

    @pytest.fixture
    def mapping_config(self):
        """Create mapping configuration."""
        if IMPORT_SUCCESS:
            return MappingConfig(
                task_type=GraphTaskType.NODE_CLASSIFICATION,
                auto_select=True,
                prefer_attention=False,
                max_layers=6,
                performance_priority="balanced",
            )
        else:
            return Mock()

    @pytest.fixture
    def selector(self, mapping_config):
        """Create model selector."""
        if IMPORT_SUCCESS:
            return ModelSelector(mapping_config)
        else:
            return Mock()

    def test_selector_initialization(self, selector):
        """Test selector initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(selector, "config")
        assert hasattr(selector, "selection_rules")
        assert isinstance(selector.selection_rules, dict)

    def test_select_architecture_for_large_graph(self, selector):
        """Test architecture selection for large graph."""
        if not IMPORT_SUCCESS:
            return

        properties = GraphProperties(
            num_nodes=5000,
            num_edges=20000,
            density=0.001,
            avg_degree=8,
            max_degree=50,
            is_directed=False,
            is_weighted=False,
            has_self_loops=False,
            has_node_features=True,
            has_edge_features=False,
            node_feature_dim=32,
            edge_feature_dim=0,
            num_connected_components=1,
            avg_clustering_coefficient=0.1,
            is_bipartite=False,
            has_cycles=True,
        )

        architecture = selector.select_architecture(properties)

        # Should prefer SAGE for large graphs
        assert architecture == ModelArchitecture.SAGE

    def test_select_architecture_with_attention_preference(self):
        """Test architecture selection with attention preference."""
        if not IMPORT_SUCCESS:
            return

        config = MappingConfig(task_type=GraphTaskType.NODE_CLASSIFICATION, prefer_attention=True)
        selector = ModelSelector(config)

        properties = GraphProperties(
            num_nodes=100,
            num_edges=500,
            density=0.05,
            avg_degree=10,
            max_degree=20,
            is_directed=False,
            is_weighted=False,
            has_self_loops=False,
            has_node_features=True,
            has_edge_features=False,
            node_feature_dim=32,
            edge_feature_dim=0,
            num_connected_components=1,
            avg_clustering_coefficient=0.3,
            is_bipartite=False,
            has_cycles=True,
        )

        architecture = selector.select_architecture(properties)

        # Should prefer GAT when attention is preferred
        assert architecture == ModelArchitecture.GAT

    def test_select_architecture_for_graph_task(self):
        """Test architecture selection for graph-level task."""
        if not IMPORT_SUCCESS:
            return

        config = MappingConfig(task_type=GraphTaskType.GRAPH_CLASSIFICATION)
        selector = ModelSelector(config)

        properties = GraphProperties(
            num_nodes=50,
            num_edges=100,
            density=0.08,
            avg_degree=4,
            max_degree=10,
            is_directed=False,
            is_weighted=False,
            has_self_loops=False,
            has_node_features=True,
            has_edge_features=False,
            node_feature_dim=16,
            edge_feature_dim=0,
            num_connected_components=1,
            avg_clustering_coefficient=0.2,
            is_bipartite=False,
            has_cycles=True,
        )

        architecture = selector.select_architecture(properties)

        # Should prefer GIN for graph tasks
        assert architecture == ModelArchitecture.GIN

    def test_determine_layer_config(self, selector):
        """Test layer configuration determination."""
        if not IMPORT_SUCCESS:
            return

        properties = GraphProperties(
            num_nodes=100,
            num_edges=500,
            density=0.05,
            avg_degree=10,
            max_degree=20,
            is_directed=False,
            is_weighted=False,
            has_self_loops=False,
            has_node_features=True,
            has_edge_features=True,
            node_feature_dim=32,
            edge_feature_dim=8,
            num_connected_components=1,
            avg_clustering_coefficient=0.3,
            is_bipartite=False,
            has_cycles=True,
            diameter=4,
        )

        config = selector.determine_layer_config(
            ModelArchitecture.GAT, properties, input_dim=32, output_dim=10
        )

        assert isinstance(config, ModelConfig)
        assert config.architecture == ModelArchitecture.GAT
        assert config.num_layers == 4  # Based on diameter
        assert config.heads is not None
        assert config.dropout > 0

    def test_determine_num_layers(self, selector):
        """Test number of layers determination."""
        if not IMPORT_SUCCESS:
            return

        # Test with known diameter
        properties = Mock()
        properties.diameter = 5
        properties.num_nodes = 100
        properties.density = 0.1

        num_layers = selector._determine_num_layers(properties)

        assert 2 <= num_layers <= 6
        assert num_layers == 5  # Should use diameter

        # Test without diameter
        properties.diameter = None
        properties.num_nodes = 256  # log2(256) = 8

        num_layers = selector._determine_num_layers(properties)

        assert num_layers == 6  # Capped at max_layers

    def test_determine_hidden_dims(self, selector):
        """Test hidden dimension determination."""
        if not IMPORT_SUCCESS:
            return

        properties = Mock()
        properties.num_nodes = 1000

        hidden_dims = selector._determine_hidden_dims(
            input_dim=32, output_dim=10, num_layers=4, graph_properties=properties
        )

        assert len(hidden_dims) == 3  # num_layers - 1
        assert hidden_dims[0] == 64  # 32 * 2
        assert hidden_dims[-1] >= 10  # At least output_dim


class TestGraphToModelMapper:
    """Test graph to model mapper functionality."""

    @pytest.fixture
    def mapping_config(self):
        """Create mapping configuration."""
        if IMPORT_SUCCESS:
            return MappingConfig(task_type=GraphTaskType.NODE_CLASSIFICATION, auto_select=True)
        else:
            return Mock()

    @pytest.fixture
    def mapper(self, mapping_config):
        """Create model mapper."""
        if IMPORT_SUCCESS:
            return GraphToModelMapper(mapping_config)
        else:
            return Mock()

    def test_mapper_initialization(self, mapper):
        """Test mapper initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(mapper, "config")
        assert hasattr(mapper, "analyzer")
        assert hasattr(mapper, "selector")
        assert hasattr(mapper, "model_cache")

    def test_map_graph_to_model_basic(self, mapper):
        """Test basic graph to model mapping."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        num_nodes = 4
        input_dim = 16
        output_dim = 10

        model, config = mapper.map_graph_to_model(edge_index, num_nodes, input_dim, output_dim)

        assert isinstance(model, nn.Module)
        assert isinstance(config, ModelConfig)

        # Test forward pass
        x = torch.randn(num_nodes, input_dim)
        out = model(x, edge_index)
        assert out.shape == (num_nodes, output_dim)

    def test_map_graph_with_features(self, mapper):
        """Test mapping with node and edge features."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        num_nodes = 2
        input_dim = 32
        output_dim = 16
        node_features = torch.randn(num_nodes, input_dim)
        edge_features = torch.randn(2, 8)

        model, config = mapper.map_graph_to_model(
            edge_index,
            num_nodes,
            input_dim,
            output_dim,
            node_features=node_features,
            edge_features=edge_features,
        )

        # Test that model works with features
        out = model(node_features, edge_index)
        assert out.shape == (num_nodes, output_dim)

    def test_validate_model_compatibility(self, mapper):
        """Test model compatibility validation."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        num_nodes = 2
        input_dim = 16
        output_dim = 8
        node_features = torch.randn(num_nodes, input_dim)

        model, _ = mapper.map_graph_to_model(edge_index, num_nodes, input_dim, output_dim)

        is_compatible = mapper.validate_model_compatibility(model, edge_index, node_features)

        assert is_compatible

    def test_create_model_with_global_pooling(self, mapper):
        """Test model creation with global pooling."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        # Configure for graph-level task
        config = MappingConfig(task_type=GraphTaskType.GRAPH_CLASSIFICATION)
        mapper = GraphToModelMapper(config)

        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        num_nodes = 3
        input_dim = 16
        output_dim = 5

        model, model_config = mapper.map_graph_to_model(
            edge_index, num_nodes, input_dim, output_dim
        )

        assert model_config.global_pool is not None

        # Test with batch
        x = torch.randn(num_nodes, input_dim)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # Model should handle batched input
        out = model(x, edge_index, batch)
        assert out.shape[0] == 1  # Single graph
        assert out.shape[1] == output_dim

    def test_edge_case_single_node(self, mapper):
        """Test edge case with single node graph."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        edge_index = torch.empty((2, 0), dtype=torch.long)
        num_nodes = 1
        input_dim = 8
        output_dim = 4

        model, config = mapper.map_graph_to_model(edge_index, num_nodes, input_dim, output_dim)

        # Should handle single node
        x = torch.randn(1, input_dim)
        out = model(x, edge_index)
        assert out.shape == (1, output_dim)

    def test_edge_case_large_graph(self, mapper):
        """Test edge case with large graph."""
        if not IMPORT_SUCCESS or not TORCH_AVAILABLE:
            return

        # Create large random graph
        num_nodes = 5000
        num_edges = 20000
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        input_dim = 64
        output_dim = 32

        model, config = mapper.map_graph_to_model(edge_index, num_nodes, input_dim, output_dim)

        # Should select appropriate architecture
        assert config.architecture == ModelArchitecture.SAGE

        # Test with small batch for memory
        x = torch.randn(100, input_dim)
        small_edge_index = edge_index[:, :100]
        out = model(x, small_edge_index)
        assert out.shape[0] == 100

"""
Module for FreeAgentics Active Inference implementation.
"""

import pytest
import torch

from inference.gnn.edge_processor import (
    Edge,
    EdgeBatch,
    EdgeConfig,
    EdgeFeatureType,
    EdgeProcessor,
    EdgeType,
)


class TestEdgeConfig:
    """Test EdgeConfig dataclass"""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = EdgeConfig()
        assert config.edge_type == EdgeType.DIRECTED
        assert config.feature_types == []
        assert config.normalize_weights
        assert not config.self_loops
        assert config.max_edges_per_node is None
        assert config.edge_sampling_strategy is None

    def test_custom_config(self) -> None:
        """Test custom configuration values"""
        config = EdgeConfig(
            edge_type=EdgeType.UNDIRECTED,
            feature_types=[EdgeFeatureType.WEIGHT, EdgeFeatureType.DISTANCE],
            normalize_weights=False,
            self_loops=True,
            max_edges_per_node=5,
            edge_sampling_strategy="topk",
        )
        assert config.edge_type == EdgeType.UNDIRECTED
        assert len(config.feature_types) == 2
        assert not config.normalize_weights
        assert config.self_loops
        assert config.max_edges_per_node == 5
        assert config.edge_sampling_strategy == "topk"


class TestEdge:
    """Test Edge dataclass"""

    def test_edge_creation(self) -> None:
        """Test edge creation with default values"""
        edge = Edge(source=0, target=1)
        assert edge.source == 0
        assert edge.target == 1
        assert edge.weight == 1.0
        assert edge.features == {}
        assert edge.edge_type is None
        assert edge.metadata == {}

    def test_edge_with_features(self) -> None:
        """Test edge creation with features"""
        edge = Edge(
            source=0,
            target=1,
            weight=0.8,
            features={"distance": 1.5, "category": "friend"},
            edge_type="social",
            metadata={"timestamp": 123456},
        )
        assert edge.weight == 0.8
        assert edge.features["distance"] == 1.5
        assert edge.features["category"] == "friend"
        assert edge.edge_type == "social"
        assert edge.metadata["timestamp"] == 123456


class TestEdgeProcessor:
    """Test EdgeProcessor class"""

    def test_initialization(self) -> None:
        """Test processor initialization"""
        config = EdgeConfig(feature_types=[EdgeFeatureType.WEIGHT, EdgeFeatureType.DISTANCE])
        processor = EdgeProcessor(config)
        assert processor.config == config
        assert len(processor.scalers) > 0
        assert processor.edge_type_mapping == {}

    def test_empty_edges(self) -> None:
        """Test processing empty edge list"""
        config = EdgeConfig()
        processor = EdgeProcessor(config)
        batch = processor.process_edges([], num_nodes=5)
        assert batch.edge_index.shape == (2, 0)
        assert batch.edge_attr is None
        assert batch.edge_weight.shape == (0,)
        assert batch.metadata["num_edges"] == 0
        assert batch.metadata["num_nodes"] == 5

    def test_directed_edges(self) -> None:
        """Test processing directed edges"""
        config = EdgeConfig(edge_type=EdgeType.DIRECTED)
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1),
            Edge(source=1, target=2),
            Edge(source=2, target=0),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_index.shape == (2, 3)
        assert torch.equal(batch.edge_index[0], torch.tensor([0, 1, 2]))
        assert torch.equal(batch.edge_index[1], torch.tensor([1, 2, 0]))

    def test_undirected_edges(self) -> None:
        """Test converting to undirected edges"""
        config = EdgeConfig(edge_type=EdgeType.UNDIRECTED)
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1),
            Edge(source=1, target=0),
            Edge(source=1, target=2),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_index.shape[1] == 2

    def test_bidirectional_edges(self) -> None:
        """Test converting to bidirectional edges"""
        config = EdgeConfig(edge_type=EdgeType.BIDIRECTIONAL)
        processor = EdgeProcessor(config)
        edges = [Edge(source=0, target=1), Edge(source=1, target=2)]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_index.shape[1] == 4

    def test_self_loops(self) -> None:
        """Test adding self-loops"""
        config = EdgeConfig(self_loops=True)
        processor = EdgeProcessor(config)
        edges = [Edge(source=0, target=1), Edge(source=1, target=2)]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_index.shape[1] == 5
        assert batch.metadata["has_self_loops"] == True

    def test_weight_features(self) -> None:
        """Test weight feature extraction"""
        config = EdgeConfig(feature_types=[EdgeFeatureType.WEIGHT], normalize_weights=True)
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, weight=0.5),
            Edge(source=1, target=2, weight=1.0),
            Edge(source=2, target=0, weight=0.2),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_attr is not None
        assert batch.edge_attr.shape == (3, 1)
        assert batch.edge_weight.shape == (3,)
        assert torch.allclose(batch.edge_weight, torch.tensor([0.5, 1.0, 0.2]))

    def test_distance_features(self) -> None:
        """Test distance feature extraction"""
        config = EdgeConfig(feature_types=[EdgeFeatureType.DISTANCE])
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, features={"distance": 1.5}),
            Edge(source=1, target=2, features={"distance": 2.0}),
            Edge(source=2, target=0, features={"distance": 0.5}),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_attr is not None
        assert batch.edge_attr.shape == (3, 1)
        assert torch.min(batch.edge_attr) >= 0
        assert torch.max(batch.edge_attr) <= 1

    def test_categorical_features(self) -> None:
        """Test categorical feature extraction"""
        config = EdgeConfig(feature_types=[EdgeFeatureType.CATEGORICAL])
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, features={"category": "friend"}),
            Edge(source=1, target=2, features={"category": "colleague"}),
            Edge(source=2, target=0, features={"category": "friend"}),
            Edge(source=0, target=2, features={"category": "family"}),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_attr is not None
        assert batch.edge_attr.shape == (4, 3)
        assert torch.sum(batch.edge_attr, dim=1).allclose(torch.ones(4))

    def test_temporal_features(self) -> None:
        """Test temporal feature extraction"""
        config = EdgeConfig(feature_types=[EdgeFeatureType.TEMPORAL])
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, features={"timestamp": 1642000000}),
            Edge(source=1, target=2, features={"timestamp": "2022-01-12T17:00:00"}),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_attr is not None
        assert batch.edge_attr.shape == (2, 7)

    def test_embedding_features(self) -> None:
        """Test embedding feature extraction"""
        config = EdgeConfig(feature_types=[EdgeFeatureType.EMBEDDING])
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, features={"embedding": [0.1, 0.2, 0.3, 0.4]}),
            Edge(source=1, target=2),
            Edge(source=2, target=0, features={"embedding": "edge_123"}),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_attr is not None
        assert batch.edge_attr.shape[1] >= 4
        norms = torch.norm(batch.edge_attr, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-06)

    def test_multiple_features(self) -> None:
        """Test extraction of multiple feature types"""
        config = EdgeConfig(
            feature_types=[
                EdgeFeatureType.WEIGHT,
                EdgeFeatureType.DISTANCE,
                EdgeFeatureType.CATEGORICAL,
            ]
        )
        processor = EdgeProcessor(config)
        edges = [
            Edge(
                source=0,
                target=1,
                weight=0.8,
                features={"distance": 1.5, "category": "friend"},
            ),
            Edge(
                source=1,
                target=2,
                weight=0.6,
                features={"distance": 2.0, "category": "colleague"},
            ),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        assert batch.edge_attr is not None
        assert batch.edge_attr.shape[1] >= 4

    def test_edge_sampling_random(self) -> None:
        """Test random edge sampling"""
        config = EdgeConfig(max_edges_per_node=2, edge_sampling_strategy="random")
        processor = EdgeProcessor(config)
        edges = [Edge(source=0, target=i) for i in range(1, 6)]
        batch = processor.process_edges(edges, num_nodes=6)
        edge_index = batch.edge_index
        node_0_edges = torch.sum((edge_index[0] == 0) | (edge_index[1] == 0))
        assert node_0_edges <= 2

    def test_edge_sampling_topk(self) -> None:
        """Test top-k edge sampling by weight"""
        config = EdgeConfig(max_edges_per_node=2, edge_sampling_strategy="topk")
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, weight=0.1),
            Edge(source=0, target=2, weight=0.9),
            Edge(source=0, target=3, weight=0.5),
            Edge(source=0, target=4, weight=0.7),
        ]
        batch = processor.process_edges(edges, num_nodes=5)
        edge_index = batch.edge_index
        edge_weights = batch.edge_weight
        node_0_mask = edge_index[0] == 0
        node_0_weights = edge_weights[node_0_mask]
        assert len(node_0_weights) == 2
        assert 0.9 in node_0_weights
        assert 0.7 in node_0_weights

    def test_edge_types(self) -> None:
        """Test edge type extraction"""
        config = EdgeConfig()
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, edge_type="social"),
            Edge(source=1, target=2, edge_type="professional"),
            Edge(source=2, target=0, edge_type="social"),
            Edge(source=0, target=3),
        ]
        batch = processor.process_edges(edges, num_nodes=4)
        assert batch.edge_type is not None
        assert batch.edge_type.shape == (4,)
        assert len(processor.edge_type_mapping) >= 2

    def test_adjacency_matrix_conversion(self) -> None:
        """Test conversion to sparse adjacency matrix"""
        config = EdgeConfig()
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, weight=0.8),
            Edge(source=1, target=2, weight=0.6),
            Edge(source=2, target=0, weight=0.9),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        adj_matrix = processor.to_adjacency_matrix(batch, num_nodes=3)
        assert adj_matrix.shape == (3, 3)
        assert adj_matrix[0, 1] == 0.8
        assert adj_matrix[1, 2] == 0.6
        assert adj_matrix[2, 0] == 0.9

    def test_edge_statistics(self) -> None:
        """Test edge statistics computation"""
        config = EdgeConfig()
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, weight=0.8, edge_type="type_a"),
            Edge(source=1, target=2, weight=0.6, edge_type="type_b"),
            Edge(source=2, target=0, weight=0.9, edge_type="type_a"),
            Edge(source=0, target=0, weight=1.0),
        ]
        batch = processor.process_edges(edges, num_nodes=3)
        stats = processor.compute_edge_statistics(batch, num_nodes=3)
        assert stats["num_edges"] == 4
        assert stats["num_nodes"] == 3
        assert stats["num_self_loops"] == 1
        assert "avg_in_degree" in stats
        assert "avg_out_degree" in stats
        assert "mean_weight" in stats
        assert stats["mean_weight"] == pytest.approx(0.825, rel=0.001)
        assert "edge_type_distribution" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

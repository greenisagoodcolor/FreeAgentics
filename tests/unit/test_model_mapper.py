"""
Module for FreeAgentics Active Inference implementation.
"""

import pytest
import torch
import torch.nn as nn

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


class TestGraphProperties:
    """Test GraphProperties dataclass"""

    def test_graph_properties_creation(self) -> None:
        """Test creating graph properties"""
        props = GraphProperties(
            num_nodes=100,
            num_edges=500,
            density=0.05,
            avg_degree=10.0,
            max_degree=25,
            is_directed=False,
            is_weighted=True,
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
        assert props.num_nodes == 100
        assert props.num_edges == 500
        assert props.density == 0.05
        assert props.is_weighted
        assert props.has_node_features
        assert props.node_feature_dim == 32


class TestGraphAnalyzer:
    """Test GraphAnalyzer class"""

    def test_analyze_small_graph(self) -> None:
        """Test analyzing a small graph"""
        analyzer = GraphAnalyzer()
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        props = analyzer.analyze_graph(edge_index, num_nodes=4)
        assert props.num_nodes == 4
        assert props.num_edges == 4
        assert props.avg_degree == 2.0
        assert props.max_degree == 2
        assert props.has_cycles
        assert props.num_connected_components == 1

    def test_analyze_directed_graph(self) -> None:
        """Test analyzing a directed graph"""
        analyzer = GraphAnalyzer()
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        props = analyzer.analyze_graph(edge_index, num_nodes=3)
        assert props.is_directed
        assert props.num_edges == 3

    def test_analyze_weighted_graph(self) -> None:
        """Test analyzing a weighted graph"""
        analyzer = GraphAnalyzer()
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_weight = torch.tensor([0.5, 0.8, 0.3])
        props = analyzer.analyze_graph(edge_index, num_nodes=3, edge_weight=edge_weight)
        assert props.is_weighted

    def test_analyze_with_features(self) -> None:
        """Test analyzing graph with node and edge features"""
        analyzer = GraphAnalyzer()
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_features = torch.randn(2, 16)
        edge_features = torch.randn(2, 8)
        props = analyzer.analyze_graph(
            edge_index,
            num_nodes=2,
            node_features=node_features,
            edge_features=edge_features,
        )
        assert props.has_node_features
        assert props.has_edge_features
        assert props.node_feature_dim == 16
        assert props.edge_feature_dim == 8

    def test_analyze_disconnected_graph(self) -> None:
        """Test analyzing a disconnected graph"""
        analyzer = GraphAnalyzer()
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        props = analyzer.analyze_graph(edge_index, num_nodes=4)
        assert props.num_connected_components == 2

    def test_analyze_bipartite_graph(self) -> None:
        """Test analyzing a bipartite graph"""
        analyzer = GraphAnalyzer()
        edge_index = torch.tensor([[0, 0, 1, 1], [2, 3, 2, 3]], dtype=torch.long)
        props = analyzer.analyze_graph(edge_index, num_nodes=4)
        assert props.is_bipartite


class TestModelSelector:
    """Test ModelSelector class"""

    def test_gcn_selection(self) -> None:
        """Test GCN architecture selection"""
        config = MappingConfig(task_type=GraphTaskType.NODE_CLASSIFICATION, auto_select=True)
        selector = ModelSelector(config)
        props = GraphProperties(
            num_nodes=100,
            num_edges=1000,
            density=0.2,
            avg_degree=20,
            max_degree=30,
            is_directed=False,
            is_weighted=False,
            has_self_loops=False,
            has_node_features=True,
            has_edge_features=False,
            node_feature_dim=32,
            edge_feature_dim=0,
            num_connected_components=1,
            avg_clustering_coefficient=0.2,
            is_bipartite=False,
            has_cycles=True,
        )
        arch = selector.select_architecture(props)
        assert arch in [ModelArchitecture.GCN, ModelArchitecture.GAT]

    def test_gat_selection_with_attention_preference(self) -> None:
        """Test GAT selection with attention preference"""
        config = MappingConfig(
            task_type=GraphTaskType.NODE_CLASSIFICATION,
            auto_select=True,
            prefer_attention=True,
        )
        selector = ModelSelector(config)
        props = GraphProperties(
            num_nodes=100,
            num_edges=500,
            density=0.1,
            avg_degree=10,
            max_degree=50,
            is_directed=False,
            is_weighted=False,
            has_self_loops=False,
            has_node_features=True,
            has_edge_features=False,
            node_feature_dim=64,
            edge_feature_dim=0,
            num_connected_components=1,
            avg_clustering_coefficient=0.3,
            is_bipartite=False,
            has_cycles=True,
        )
        arch = selector.select_architecture(props)
        assert arch == ModelArchitecture.GAT

    def test_sage_selection_for_large_graphs(self) -> None:
        """Test SAGE selection for large sparse graphs"""
        config = MappingConfig(task_type=GraphTaskType.NODE_CLASSIFICATION, auto_select=True)
        selector = ModelSelector(config)
        props = GraphProperties(
            num_nodes=10000,
            num_edges=50000,
            density=0.001,
            avg_degree=10,
            max_degree=30,
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
        arch = selector.select_architecture(props)
        assert arch == ModelArchitecture.SAGE

    def test_gin_selection_for_graph_classification(self) -> None:
        """Test GIN selection for graph classification"""
        config = MappingConfig(task_type=GraphTaskType.GRAPH_CLASSIFICATION, auto_select=True)
        selector = ModelSelector(config)
        props = GraphProperties(
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
            node_feature_dim=32,
            edge_feature_dim=0,
            num_connected_components=1,
            avg_clustering_coefficient=0.4,
            is_bipartite=False,
            has_cycles=True,
        )
        arch = selector.select_architecture(props)
        assert arch == ModelArchitecture.GIN

    def test_manual_architecture_override(self) -> None:
        """Test manual architecture override"""
        config = MappingConfig(
            task_type=GraphTaskType.NODE_CLASSIFICATION,
            auto_select=False,
            manual_overrides={"architecture": "edgeconv"},
        )
        selector = ModelSelector(config)
        props = GraphProperties(
            num_nodes=100,
            num_edges=200,
            density=0.04,
            avg_degree=4,
            max_degree=10,
            is_directed=False,
            is_weighted=False,
            has_self_loops=False,
            has_node_features=True,
            has_edge_features=False,
            node_feature_dim=32,
            edge_feature_dim=0,
            num_connected_components=1,
            avg_clustering_coefficient=0.2,
            is_bipartite=False,
            has_cycles=True,
        )
        arch = selector.select_architecture(props)
        assert arch == ModelArchitecture.EDGECONV

    def test_layer_config_determination(self) -> None:
        """Test layer configuration determination"""
        config = MappingConfig(
            task_type=GraphTaskType.NODE_CLASSIFICATION, max_layers=8, min_layers=2
        )
        selector = ModelSelector(config)
        props = GraphProperties(
            num_nodes=100,
            num_edges=500,
            density=0.1,
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
            diameter=4,
        )
        model_config = selector.determine_layer_config(
            ModelArchitecture.GCN, props, input_dim=32, output_dim=10
        )
        assert model_config.architecture == ModelArchitecture.GCN
        assert model_config.num_layers >= 2
        assert model_config.num_layers <= 8
        assert len(model_config.hidden_channels) == model_config.num_layers - 1
        assert model_config.output_channels == 10

    def test_gat_specific_config(self) -> None:
        """Test GAT-specific configuration"""
        config = MappingConfig(task_type=GraphTaskType.NODE_CLASSIFICATION)
        selector = ModelSelector(config)
        props = GraphProperties(
            num_nodes=100,
            num_edges=1000,
            density=0.2,
            avg_degree=20,
            max_degree=40,
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
        model_config = selector.determine_layer_config(
            ModelArchitecture.GAT, props, input_dim=32, output_dim=10
        )
        assert model_config.heads is not None
        assert model_config.heads >= 1
        assert model_config.heads <= 8


class TestGraphToModelMapper:
    """Test GraphToModelMapper class"""

    def test_map_simple_graph(self) -> None:
        """Test mapping a simple graph to model"""
        config = MappingConfig(task_type=GraphTaskType.NODE_CLASSIFICATION, auto_select=True)
        mapper = GraphToModelMapper(config)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        num_nodes = 5
        input_dim = 16
        output_dim = 3
        node_features = torch.randn(num_nodes, input_dim)
        model, model_config = mapper.map_graph_to_model(
            edge_index, num_nodes, input_dim, output_dim, node_features=node_features
        )
        assert isinstance(model, nn.Module)
        assert model_config.output_channels == output_dim
        output = model(node_features, edge_index)
        assert output.shape == (num_nodes, output_dim)

    def test_map_large_graph(self) -> None:
        """Test mapping a large graph"""
        config = MappingConfig(
            task_type=GraphTaskType.NODE_CLASSIFICATION, auto_select=True, max_layers=5
        )
        mapper = GraphToModelMapper(config)
        num_nodes = 1000
        num_edges = 5000
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        input_dim = 32
        output_dim = 10
        node_features = torch.randn(num_nodes, input_dim)
        model, model_config = mapper.map_graph_to_model(
            edge_index, num_nodes, input_dim, output_dim, node_features=node_features
        )
        assert model_config.architecture in [
            ModelArchitecture.SAGE,
            ModelArchitecture.GCN,
        ]
        assert model_config.num_layers <= 5

    def test_graph_level_task(self) -> None:
        """Test mapping for graph-level tasks"""
        config = MappingConfig(task_type=GraphTaskType.GRAPH_CLASSIFICATION, auto_select=True)
        mapper = GraphToModelMapper(config)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        num_nodes = 4
        input_dim = 8
        output_dim = 2
        node_features = torch.randn(num_nodes, input_dim)
        model, model_config = mapper.map_graph_to_model(
            edge_index, num_nodes, input_dim, output_dim, node_features=node_features
        )
        assert model_config.global_pool is not None
        batch = torch.zeros(num_nodes, dtype=torch.long)
        output = model(node_features, edge_index, batch)
        assert output.shape == (1, output_dim)

    def test_model_validation(self) -> None:
        """Test model validation"""
        config = MappingConfig(task_type=GraphTaskType.NODE_CLASSIFICATION)
        mapper = GraphToModelMapper(config)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        num_nodes = 2
        input_dim = 16
        output_dim = 4
        node_features = torch.randn(num_nodes, input_dim)
        model, _ = mapper.map_graph_to_model(
            edge_index, num_nodes, input_dim, output_dim, node_features=node_features
        )
        is_valid = mapper.validate_model_compatibility(model, edge_index, node_features)
        assert is_valid
        wrong_features = torch.randn(num_nodes, input_dim + 5)
        is_valid = mapper.validate_model_compatibility(model, edge_index, wrong_features)
        assert not is_valid

    def test_performance_priority(self) -> None:
        """Test different performance priorities"""
        config_speed = MappingConfig(
            task_type=GraphTaskType.NODE_CLASSIFICATION, performance_priority="speed"
        )
        mapper_speed = GraphToModelMapper(config_speed)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_features = torch.randn(2, 16)
        _, config_result = mapper_speed.map_graph_to_model(
            edge_index, 2, 16, 4, node_features=node_features
        )
        assert config_result.architecture in [
            ModelArchitecture.GCN,
            ModelArchitecture.SAGE,
        ]
        config_accuracy = MappingConfig(
            task_type=GraphTaskType.GRAPH_CLASSIFICATION,
            performance_priority="accuracy",
        )
        mapper_accuracy = GraphToModelMapper(config_accuracy)
        _, config_result = mapper_accuracy.map_graph_to_model(
            edge_index, 2, 16, 4, node_features=node_features
        )
        assert config_result.architecture in [
            ModelArchitecture.GIN,
            ModelArchitecture.GAT,
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

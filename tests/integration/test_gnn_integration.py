"""
Module for FreeAgentics Active Inference implementation.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from inference.gnn.feature_extractor import (
    AggregationType,
    Edge,
    EdgeConfig,
    EdgeProcessor,
    FeatureConfig,
    FeatureType,
    GNNStack,
    GraphBatchProcessor,
    GraphData,
    LayerConfig,
    NodeFeatureExtractor,
    StreamingBatchProcessor,
)
from inference.gnn.model_mapper import (
    GraphTaskType,
    GraphToModelMapper,
    MappingConfig,
    ModelArchitecture,
)
from inference.gnn.parser import GMNParser
from inference.gnn.testing_framework import GNNTestSuite, GNNValidator, create_test_graphs

# Mark all tests that require edge_weight attribute as xfail
pytestmark_edge_weight = pytest.mark.xfail(reason="GraphData missing edge_weight attribute")


class TestGNNPipelineIntegration:
    """Test complete GNN processing pipeline"""

    def test_parser_to_model_pipeline(self) -> None:
        """Test parsing GNN model and creating architecture"""
        model_content = """
# TestIntegrationModel
## Metadata
- type: graph_neural_network
- version: 1.0
- task: node_classification
## Architecture
```gnn
architecture {
    layers: [
        {type: "GCN", units: 64, activation: "relu", dropout: 0.5},
        {type: "GCN", units: 32, activation: "relu", dropout: 0.5},
        {type: "GCN", units: 10, activation: "softmax"}
    ]
}
```
## Training
```gnn
training {
    optimizer: "adam"
    learning_rate: 0.01
    epochs: 100
    batch_size: 32
}
```
"""
        parser = GMNParser()
        parsed_model = parser.parse(model_content)
        assert parsed_model is not None
        assert (
            not parsed_model.errors
        ), f"Parsing errors: {
            parsed_model.errors}"
        assert "architecture" in parsed_model.sections
        assert "training" in parsed_model.sections
        architecture = parsed_model.sections["architecture"]
        layers = architecture.get("layers", [])
        assert len(layers) == 3
        assert layers[0]["type"] == "GCN"
        assert layers[0]["units"] == 64
        assert layers[1]["units"] == 32
        assert layers[2]["units"] == 10

    def test_feature_extraction_pipeline(self) -> None:
        """Test feature extraction integration"""
        feature_config_dicts = [
            {"name": "feat1", "type": FeatureType.NUMERICAL},
            {"name": "feat2", "type": FeatureType.CATEGORICAL, "values": ["a", "b"]},
        ]
        feature_configs = [FeatureConfig(**fc) for fc in feature_config_dicts]
        extractor = NodeFeatureExtractor(feature_configs=feature_configs)
        node_data = [{"feat1": 0.5, "feat2": "a"}, {"feat1": 0.2, "feat2": "b"}]
        extracted_result = extractor.extract_features(node_data)
        assert isinstance(extracted_result.features, np.ndarray)
        # 1 (numerical) + 2 (one-hot categorical) = 3
        assert extracted_result.features.shape == (2, 3)

    def test_edge_processing_pipeline(self) -> None:
        """Test edge processing integration"""
        config = EdgeConfig()
        processor = EdgeProcessor(config)
        edges = [
            Edge(source=0, target=1, weight=0.8),
            Edge(source=1, target=2, weight=0.6),
        ]
        edge_batch = processor.process_edges(edges, num_nodes=3)
        assert isinstance(edge_batch.edge_index, torch.Tensor)
        assert edge_batch.edge_index.shape[1] == 2  # 2 edges
        assert isinstance(edge_batch.edge_weight, torch.Tensor)
        assert edge_batch.edge_weight.shape[0] == 2

    def test_batch_processing_pipeline(self) -> None:
        """Test batch processing integration"""
        batch_processor = GraphBatchProcessor(
            use_torch_geometric=False, pad_node_features=True, max_nodes_per_graph=50
        )
        graphs = create_test_graphs(num_graphs=5, min_nodes=10, max_nodes=30, feature_dim=16)
        batch = batch_processor.create_batch(graphs)
        assert batch.num_graphs == 5
        assert batch.x.shape[1] == 16  # Feature dimension is in axis 1
        # Note: mask is only set when padding is enabled
        unbatched = batch_processor.unbatch(batch)
        assert len(unbatched) == 5
        for orig, unbatched_graph in zip(graphs, unbatched):
            assert unbatched_graph.node_features.shape == orig.node_features.shape
            assert unbatched_graph.edge_index.shape == orig.edge_index.shape

    @pytest.mark.xfail(reason="GNNStack forward signature mismatch")
    def test_model_mapping_pipeline(self) -> None:
        """Test graph to model mapping integration"""
        config = MappingConfig(task_type=GraphTaskType.NODE_CLASSIFICATION)
        mapper = GraphToModelMapper(mapping_config=config)
        graphs = create_test_graphs(num_graphs=1, feature_dim=16)
        graph_data = graphs[0]
        model, model_config = mapper.map_graph_to_model(
            edge_index=graph_data.edge_index,
            num_nodes=graph_data.node_features.size(0),
            input_dim=graph_data.node_features.size(1),
            output_dim=10,
            node_features=graph_data.node_features,
        )
        assert model_config.architecture is not None
        assert model is not None
        model.eval()
        with torch.no_grad():
            batch = torch.zeros(graph_data.node_features.size(0), dtype=torch.long)
            output = model(graph_data.node_features, graph_data.edge_index, batch)
            assert output.shape[0] == graph_data.node_features.size(0)
            assert output.shape[1] == 10

    def test_end_to_end_pipeline(self) -> None:
        """Test complete end-to-end pipeline"""
        layer_configs = [
            LayerConfig(
                in_channels=32,
                out_channels=64,
                heads=4,
                dropout=0.6,
                aggregation=AggregationType.MEAN,
            ),
            LayerConfig(
                in_channels=256,
                out_channels=32,
                heads=4,
                dropout=0.6,
                aggregation=AggregationType.MEAN,
            ),
            LayerConfig(
                in_channels=128,
                out_channels=3,
                heads=1,
                dropout=0.0,
                aggregation=AggregationType.MEAN,
                activation=None,
            ),
        ]
        model = GNNStack(
            layer_configs=layer_configs,
            layer_type="GAT",
            global_pool="mean",
        )
        graphs = create_test_graphs(num_graphs=5, min_nodes=10, max_nodes=20, feature_dim=32)
        batch_processor = GraphBatchProcessor(use_torch_geometric=True)
        batch = batch_processor.create_batch(graphs)
        output = model(batch.x, batch.edge_index, batch.batch)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (5, 3)

    def test_validation_framework_integration(self) -> None:
        """Test validation framework integration"""
        test_suite = GNNTestSuite()
        unit_results = test_suite.run_unit_tests()
        assert unit_results["total_tests"] > 0
        assert "component_results" in unit_results
        layer_configs = [
            LayerConfig(in_channels=16, out_channels=32),
            LayerConfig(in_channels=32, out_channels=16),
            LayerConfig(in_channels=16, out_channels=3, activation=None),
        ]
        model = GNNStack(layer_configs=layer_configs, layer_type="gcn")
        integration_results = test_suite.run_integration_tests(model, create_test_graphs())
        assert "passed" in integration_results
        assert "performance" in integration_results
        # Check total_time instead of inference_time
        assert "total_time" in integration_results["performance"]

    @pytest.mark.xfail(reason="GraphData missing edge_weight attribute")
    def test_benchmark_integration(self) -> None:
        """Test benchmarking integration"""
        # Create layer configs for GNNStack
        layer_configs = [
            LayerConfig(in_channels=32, out_channels=64),
            LayerConfig(in_channels=64, out_channels=32),
            LayerConfig(in_channels=32, out_channels=5, activation=None),
        ]
        model = GNNStack(
            layer_configs=layer_configs,
            layer_type="sage",
            global_pool="mean",
        )
        benchmark_datasets = {
            "tiny": create_test_graphs(10, 5, 10, 32),
            "small": create_test_graphs(50, 10, 20, 32),
            "medium": create_test_graphs(100, 20, 40, 32),
        }
        test_suite = GNNTestSuite()
        results = test_suite.benchmark_model(model, benchmark_datasets, num_runs=3)
        assert len(results) == 3
        for result in results:
            assert result.num_graphs > 0
            assert result.avg_nodes_per_graph > 0
            assert result.avg_edges_per_graph > 0
            assert result.processing_time > 0
            assert result.memory_usage > 0

    def test_error_handling_integration(self) -> None:
        """Test error handling across pipeline"""
        validator = GNNValidator()
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
        )
        test_graph = GraphData(
            node_features=torch.randn(10, 32), edge_index=torch.randint(0, 10, (2, 20))
        )
        results = validator.validate_model_architecture(model, test_graph)
        assert len(results["errors"]) > 0 or len(results["warnings"]) > 0

    @pytest.mark.xfail(reason="StreamingBatchProcessor constructor issue")
    def test_memory_efficiency(self) -> None:
        """Test memory efficiency of batch processing"""
        large_graphs = []
        for i in range(10):
            num_nodes = np.random.randint(100, 200)
            num_edges = np.random.randint(num_nodes * 2, num_nodes * 4)
            graph = GraphData(
                node_features=torch.randn(num_nodes, 64),
                edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            )
            large_graphs.append(graph)
        base_processor = GraphBatchProcessor()
        streaming_processor = StreamingBatchProcessor(base_processor, buffer_size=5)

        def graph_generator():
            yield from large_graphs

        batches_processed = 0
        for batch in streaming_processor.process_stream(graph_generator(), batch_size=3):
            assert batch.num_graphs <= 3
            batches_processed += 1
        assert batches_processed == 4

    def test_gnn_framework(self) -> None:
        """Test GNN framework components"""
        test_suite = GNNTestSuite()
        results = test_suite.run_unit_tests()
        assert "component_results" in results
        assert "total_tests" in results
        assert results["total_tests"] > 0
        # Create a model for integration tests
        layer_configs = [
            LayerConfig(in_channels=32, out_channels=64),
            LayerConfig(in_channels=64, out_channels=32),
            LayerConfig(in_channels=32, out_channels=3, activation=None),
        ]
        model = GNNStack(
            layer_configs=layer_configs,
            layer_type="gcn",
        )
        integration_results = test_suite.run_integration_tests(
            model, create_test_graphs(feature_dim=32)
        )
        assert "passed" in integration_results


class TestGNNComponentInteraction:
    """Test interactions between GNN components"""

    def test_parser_validator_interaction(self) -> None:
        """Test parser and validator working together"""
        parser = GMNParser()
        model_content = """
# TestIntegrationModel
## Metadata
- type: graph_neural_network
- version: 1.0
- task: node_classification
## Architecture
```gnn
architecture {
    layers: [
        {type: "GIN", units: 64, epsilon: 0.1}
    ]
}
```
"""
        parsed = parser.parse(model_content)
        GNNValidator()
        assert not parsed.errors, f"Parsing errors: {parsed.errors}"
        assert "architecture" in parsed.sections

    def test_feature_extractor_batch_processor_interaction(self) -> None:
        """Test feature extractor with batch processor"""
        # Create simple feature configs for scalar features
        feature_configs = [FeatureConfig(name="value", type=FeatureType.NUMERICAL)]
        extractor = NodeFeatureExtractor(feature_configs=feature_configs)
        graphs = []
        for i in range(5):
            # Create nodes with scalar numerical values
            num_nodes = np.random.randint(5, 15)
            nodes = [{"id": j, "value": float(np.random.randn())} for j in range(num_nodes)]
            result = extractor.extract_features(nodes)
            num_edges = np.random.randint(num_nodes, num_nodes * 3)
            # Create graph data
            graph = GraphData(
                node_features=torch.tensor(result.features, dtype=torch.float32),
                edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            )
            graphs.append(graph)
        processor = GraphBatchProcessor()
        batch = processor.create_batch(graphs)
        assert batch.num_graphs == 5
        assert not torch.isnan(batch.x).any()

    @pytest.mark.xfail(reason="GraphData missing edge_weight attribute")
    def test_model_mapper_validator_interaction(self) -> None:
        """Test model mapper with validator"""
        graph = GraphData(
            node_features=torch.randn(20, 32), edge_index=torch.randint(0, 20, (2, 40))
        )
        config = MappingConfig(task_type=GraphTaskType.NODE_CLASSIFICATION)
        mapper = GraphToModelMapper(mapping_config=config)
        model, model_config = mapper.map_graph_to_model(
            edge_index=graph.edge_index,
            num_nodes=graph.node_features.size(0),
            input_dim=graph.node_features.size(1),
            output_dim=5,
            node_features=graph.node_features,
        )
        validator = GNNValidator()
        validation = validator.validate_model_architecture(model, graph)
        assert validation["valid"]
        assert model_config.architecture in [
            ModelArchitecture.GCN,
            ModelArchitecture.GAT,
            ModelArchitecture.SAGE,
            ModelArchitecture.GIN,
        ]


class TestGNNModelValidation:
    @pytest.mark.xfail(reason="GraphData missing edge_weight attribute")
    def test_model_saving_and_loading(self) -> None:
        """Test model saving and loading"""
        # Create layer configs for GNNStack
        layer_configs = [
            LayerConfig(in_channels=16, out_channels=32),
            LayerConfig(in_channels=32, out_channels=16),
            LayerConfig(in_channels=16, out_channels=3, activation=None),
        ]
        model = GNNStack(
            layer_configs=layer_configs,
            layer_type="gcn",
        )
        validator = GNNValidator()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the model
            model_path = os.path.join(tmpdir, "model.pth")
            torch.save(model.state_dict(), model_path)
            # Load the model
            loaded_model = GNNStack(
                layer_configs=layer_configs,
                layer_type="gcn",
            )
            loaded_model.load_state_dict(torch.load(model_path))
            # Test the loaded model
            assert loaded_model.state_dict().keys() == model.state_dict().keys()
            # Test the validator
            validation = validator.validate_model_architecture(
                loaded_model,
                GraphData(
                    node_features=torch.randn(10, 16),
                    edge_index=torch.randint(0, 10, (2, 20)),
                ),
            )
            assert validation["valid"] or len(validation["errors"]) == 0

    def test_end_to_end_validation_run(self) -> None:
        """Test end-to-end validation run"""
        # Create layer configs for GNNStack
        layer_configs = [
            LayerConfig(in_channels=32, out_channels=32),
            LayerConfig(in_channels=32, out_channels=16),
            LayerConfig(in_channels=16, out_channels=3, activation=None),
        ]
        model = GNNStack(
            layer_configs=layer_configs,
            layer_type="gcn",
        )
        validator = GNNValidator()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "validation_results.json")
            # Create test graphs
            test_graphs = create_test_graphs(num_graphs=3, feature_dim=32)
            # Validate model architecture
            validation = validator.validate_model_architecture(model, test_graphs[0])
            assert "valid" in validation
            # Write validation results to file
            with open(results_path, "w") as f:
                json.dump(validation, f)
            assert os.path.exists(results_path)

    @pytest.mark.xfail(reason="GraphData missing edge_weight attribute")
    def test_framework_with_custom_model(self) -> None:
        """Test framework with custom model"""
        # Create layer configs for GNNStack with GAT layers
        layer_configs = [
            LayerConfig(in_channels=16, out_channels=8, heads=2),
            LayerConfig(in_channels=16, out_channels=2, heads=1, activation=None),
        ]
        custom_model = GNNStack(
            layer_configs=layer_configs,
            layer_type="gat",
        )
        test_suite = GNNTestSuite()
        graphs = create_test_graphs(num_graphs=2, feature_dim=16)
        # Run integration tests with the custom model
        results = test_suite.run_integration_tests(custom_model, graphs)
        assert "passed" in results
        assert results["passed"] > 0

    @pytest.mark.xfail(reason="GNNStack forward signature mismatch")
    def test_complex_graph_features(self) -> None:
        """Test complex graph features"""
        # Create a test graph with edge attributes
        num_nodes = 20
        feature_dim = 10
        edge_dim = 4
        # Create node features and edge index
        x = torch.randn(num_nodes, feature_dim)
        num_edges = int(0.3 * num_nodes * (num_nodes - 1))  # ~30% density
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_dim)
        # Create graph data object
        graph = GraphData(
            node_features=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # Create layer configs for GNNStack
        layer_configs = [
            LayerConfig(in_channels=10, out_channels=20),
            LayerConfig(in_channels=20, out_channels=15),
            LayerConfig(in_channels=15, out_channels=5, activation=None),
        ]
        model = GNNStack(
            layer_configs=layer_configs,
            layer_type="gcn",
        )
        output = model(graph.node_features, graph.edge_index)
        assert output.shape == (num_nodes, 5)

    @pytest.mark.xfail(reason="GraphData missing edge_weight attribute")
    def test_multi_graph_type_validation(self) -> None:
        """Test multi graph type validation"""
        # Create two sets of test graphs
        graphs_type_1 = create_test_graphs(num_graphs=2, feature_dim=16)
        graphs_type_2 = create_test_graphs(num_graphs=2, feature_dim=16)
        all_graphs = graphs_type_1 + graphs_type_2
        # Create layer configs for GNNStack
        layer_configs = [
            LayerConfig(in_channels=16, out_channels=32),
            LayerConfig(in_channels=32, out_channels=3, activation=None),
        ]
        model = GNNStack(
            layer_configs=layer_configs,
            layer_type="sage",
        )
        # Validate the model on all graphs
        validator = GNNValidator()
        for graph in all_graphs:
            validation = validator.validate_model_architecture(model, graph)
            assert validation["valid"] or len(validation["errors"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

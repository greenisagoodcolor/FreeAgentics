import numpy as np
import pytest
import torch

from inference.gnn.batch_processor import (
    BatchedGraphData,
    DynamicBatchSampler,
    GraphBatchProcessor,
    GraphData,
    StreamingBatchProcessor,
    create_mini_batches,
)

"""
Unit tests for Batch Processing Module
Tests efficient batch processing of multiple graphs including padding,
masking, and memory optimization.
"""


class TestGraphData:
    """Test GraphData dataclass"""

    def test_graph_data_creation(self) -> None:
        """Test creating GraphData"""
        node_features = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        graph = GraphData(node_features=node_features, edge_index=edge_index)
        assert graph.node_features.shape == (10, 32)
        assert graph.edge_index.shape == (2, 3)
        assert graph.edge_attr is None
        assert graph.edge_weight is None

    def test_graph_data_with_attributes(self) -> None:
        """Test GraphData with all attributes"""
        graph = GraphData(
            node_features=torch.randn(5, 16),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 8),
            edge_weight=torch.tensor([0.5, 0.8]),
            graph_attr=torch.randn(4),
            target=torch.tensor([1]),
            mask=torch.ones(5, dtype=torch.bool),
            metadata={"name": "test_graph"},
        )
        assert graph.edge_attr.shape == (2, 8)
        assert graph.edge_weight.shape == (2,)
        assert graph.graph_attr.shape == (4,)
        assert graph.target.shape == (1,)
        assert graph.mask.shape == (5,)
        assert graph.metadata["name"] == "test_graph"


class TestGraphBatchProcessor:
    """Test GraphBatchProcessor class"""

    def test_initialization(self) -> None:
        """Test processor initialization"""
        processor = GraphBatchProcessor(pad_node_features=True, max_nodes_per_graph=50)
        assert processor.pad_node_features
        assert processor.max_nodes_per_graph == 50

    def test_create_empty_batch(self) -> None:
        """Test creating empty batch"""
        processor = GraphBatchProcessor()
        batch = processor.create_batch([])
        assert batch.num_graphs == 0
        assert batch.x.shape == (0, 0)
        assert batch.edge_index.shape == (2, 0)

    def test_create_batch_simple(self) -> None:
        """Test creating batch from simple graphs"""
        processor = GraphBatchProcessor(use_torch_geometric=False)
        graphs = [
            GraphData(
                node_features=torch.randn(3, 16),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            ),
            GraphData(
                node_features=torch.randn(4, 16),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            ),
        ]
        batch = processor.create_batch(graphs)
        assert batch.num_graphs == 2
        assert batch.x.shape == (7, 16)
        assert batch.edge_index.shape == (2, 5)
        assert batch.num_nodes_per_graph == [3, 4]
        assert batch.num_edges_per_graph == [2, 3]

    def test_batch_with_edge_attributes(self) -> None:
        """Test batching with edge attributes"""
        processor = GraphBatchProcessor(use_torch_geometric=False)
        graphs = [
            GraphData(
                node_features=torch.randn(3, 16),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.randn(2, 4),
                edge_weight=torch.tensor([0.5, 0.8]),
            ),
            GraphData(
                node_features=torch.randn(2, 16),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.randn(1, 4),
                edge_weight=torch.tensor([0.9]),
            ),
        ]
        batch = processor.create_batch(graphs)
        assert batch.edge_attr is not None
        assert batch.edge_attr.shape == (3, 4)
        assert batch.edge_weight is not None
        assert batch.edge_weight.shape == (3,)

    def test_batch_with_padding(self) -> None:
        """Test batching with node feature padding"""
        processor = GraphBatchProcessor(
            pad_node_features=True, max_nodes_per_graph=5, use_torch_geometric=False
        )
        graphs = [
            GraphData(
                node_features=torch.randn(3, 16),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            ),
            GraphData(
                node_features=torch.randn(2, 16),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            ),
        ]
        batch = processor.create_batch(graphs)
        assert batch.x.shape == (2, 5, 16)
        assert batch.mask is not None
        assert batch.mask.shape == (2, 5)
        assert batch.mask[0, :3].all() == True
        assert batch.mask[0, 3:].all() == False
        assert batch.mask[1, :2].all() == True
        assert batch.mask[1, 2:].all() == False

    def test_batch_with_targets(self) -> None:
        """Test batching with targets"""
        processor = GraphBatchProcessor(use_torch_geometric=False)
        graphs = [
            GraphData(
                node_features=torch.randn(3, 16),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                target=torch.tensor([0]),
            ),
            GraphData(
                node_features=torch.randn(4, 16),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                target=torch.tensor([1]),
            ),
        ]
        batch = processor.create_batch(graphs)
        assert batch.target is not None
        assert batch.target.shape == (2, 1)
        assert batch.target[0].item() == 0
        assert batch.target[1].item() == 1

    def test_batch_with_graph_attributes(self) -> None:
        """Test batching with graph-level attributes"""
        processor = GraphBatchProcessor(pad_graph_features=True, use_torch_geometric=False)
        graphs = [
            GraphData(
                node_features=torch.randn(3, 16),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                graph_attr=torch.randn(10),
            ),
            GraphData(
                node_features=torch.randn(2, 16),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                graph_attr=torch.randn(8),
            ),
        ]
        batch = processor.create_batch(graphs)
        assert batch.graph_attr is not None
        assert batch.graph_attr.shape == (2, 10)

    def test_unbatch(self) -> None:
        """Test unbatching back to individual graphs"""
        processor = GraphBatchProcessor(use_torch_geometric=False)
        original_graphs = [
            GraphData(
                node_features=torch.randn(3, 16),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
                edge_attr=torch.randn(3, 4),
                target=torch.tensor([0]),
            ),
            GraphData(
                node_features=torch.randn(4, 16),
                edge_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
                edge_attr=torch.randn(2, 4),
                target=torch.tensor([1]),
            ),
        ]
        batch = processor.create_batch(original_graphs)
        unbatched_graphs = processor.unbatch(batch)
        assert len(unbatched_graphs) == 2
        assert unbatched_graphs[0].node_features.shape == (3, 16)
        assert unbatched_graphs[0].edge_index.shape == (2, 3)
        assert unbatched_graphs[0].edge_attr.shape == (3, 4)
        assert unbatched_graphs[0].target.item() == 0
        assert unbatched_graphs[1].node_features.shape == (4, 16)
        assert unbatched_graphs[1].edge_index.shape == (2, 2)
        assert unbatched_graphs[1].edge_attr.shape == (2, 4)
        assert unbatched_graphs[1].target.item() == 1

    def test_collate_fn(self) -> None:
        """Test collate function for DataLoader"""
        processor = GraphBatchProcessor()
        graphs = [
            GraphData(node_features=torch.randn(5, 32), edge_index=torch.randint(0, 5, (2, 10)))
            for _ in range(4)
        ]
        batch = processor.collate_fn(graphs)
        assert batch.num_graphs == 4
        assert batch.x.shape[0] == 20


class TestDynamicBatchSampler:
    """Test DynamicBatchSampler class"""

    def test_initialization(self) -> None:
        """Test sampler initialization"""
        graph_sizes = [(10, 20), (12, 25), (50, 100), (55, 110)]
        sampler = DynamicBatchSampler(graph_sizes=graph_sizes, batch_size=2, size_threshold=0.2)
        assert sampler.batch_size == 2
        assert len(sampler.size_groups) > 0

    def test_size_grouping(self) -> None:
        """Test grouping by similar sizes"""
        graph_sizes = [(10, 20), (11, 22), (50, 100), (52, 105), (100, 200), (98, 195)]
        sampler = DynamicBatchSampler(
            graph_sizes=graph_sizes, batch_size=2, size_threshold=0.2, shuffle=False
        )
        assert len(sampler.size_groups) == 3
        for group in sampler.size_groups:
            assert len(group) == 2

    def test_batch_generation(self) -> None:
        """Test batch generation"""
        graph_sizes = [(10, 20) for _ in range(10)]
        sampler = DynamicBatchSampler(graph_sizes=graph_sizes, batch_size=3, shuffle=False)
        batches = list(sampler)
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_sampler_length(self) -> None:
        """Test sampler length calculation"""
        graph_sizes = [(10, 20) for _ in range(10)]
        sampler = DynamicBatchSampler(graph_sizes=graph_sizes, batch_size=3)
        assert len(sampler) == 4


class TestStreamingBatchProcessor:
    """Test StreamingBatchProcessor class"""

    def test_initialization(self) -> None:
        """Test streaming processor initialization"""
        base_processor = GraphBatchProcessor()
        streaming = StreamingBatchProcessor(batch_processor=base_processor, buffer_size=100)
        assert streaming.buffer_size == 100
        assert streaming.buffer == []

    def test_process_stream(self) -> None:
        """Test processing graph stream"""
        base_processor = GraphBatchProcessor()
        streaming = StreamingBatchProcessor(base_processor)

        def graph_generator():
            for i in range(10):
                yield GraphData(
                    node_features=torch.randn(5, 16),
                    edge_index=torch.randint(0, 5, (2, 10)),
                    target=torch.tensor([i % 3]),
                )

        batches = list(streaming.process_stream(graph_generator(), batch_size=3))
        assert len(batches) == 4
        assert batches[0].num_graphs == 3
        assert batches[1].num_graphs == 3
        assert batches[2].num_graphs == 3
        assert batches[3].num_graphs == 1

    def test_process_stream_with_function(self) -> None:
        """Test processing with custom function"""
        base_processor = GraphBatchProcessor()
        streaming = StreamingBatchProcessor(base_processor)

        def graph_generator():
            for i in range(6):
                yield GraphData(
                    node_features=torch.randn(3, 8),
                    edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                )

        def process_fn(batch):
            return batch.x.shape[0]

        results = list(
            streaming.process_stream(graph_generator(), batch_size=2, process_fn=process_fn)
        )
        assert len(results) == 3
        assert results[0] == 6
        assert results[1] == 6
        assert results[2] == 6


class TestUtilityFunctions:
    """Test utility functions"""

    def test_create_mini_batches(self) -> None:
        """Test mini-batch creation"""
        graphs = [
            GraphData(node_features=torch.randn(5, 16), edge_index=torch.randint(0, 5, (2, 10)))
            for _ in range(10)
        ]
        batches = create_mini_batches(graphs, batch_size=3, shuffle=False, drop_last=False)
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_create_mini_batches_drop_last(self) -> None:
        """Test mini-batch creation with drop_last"""
        graphs = [
            GraphData(node_features=torch.randn(5, 16), edge_index=torch.randint(0, 5, (2, 10)))
            for _ in range(10)
        ]
        batches = create_mini_batches(graphs, batch_size=3, shuffle=False, drop_last=True)
        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

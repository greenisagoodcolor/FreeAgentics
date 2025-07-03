"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data

"""
Batch Processing Module for Graphs
This module implements efficient batch processing mechanisms for handling
multiple graphs or subgraphs simultaneously, including padding, masking,
and memory optimization.
"""
logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """Container for single graph data"""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    edge_weight: Optional[torch.Tensor] = None
    graph_attr: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchedGraphData:
    """Container for batched graph data"""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    edge_weight: Optional[torch.Tensor] = None
    batch: torch.Tensor = None
    ptr: torch.Tensor = None
    graph_attr: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    num_graphs: int = 0
    num_nodes_per_graph: List[int] = field(default_factory=list)
    num_edges_per_graph: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphBatchProcessor:
    """
    Handles efficient batch processing of multiple graphs.
    Features:
    - Dynamic batching with padding and masking
    - Memory-efficient graph concatenation
    - Support for variable-sized graphs
    - Optimized for GPU processing
    """

    def __init__(
        self,
        pad_node_features: bool = True,
        pad_graph_features: bool = True,
        max_nodes_per_graph: Optional[int] = None,
        max_edges_per_graph: Optional[int] = None,
        use_torch_geometric: bool = True,
    ) -> None:
        """
        Initialize batch processor.
        Args:
            pad_node_features: Whether to pad node features
            pad_graph_features: Whether to pad graph features
            max_nodes_per_graph: Maximum nodes per graph (for padding)
            max_edges_per_graph: Maximum edges per graph (for filtering)
            use_torch_geometric: Use PyTorch Geometric's batching
        """
        self.pad_node_features = pad_node_features
        self.pad_graph_features = pad_graph_features
        self.max_nodes_per_graph = max_nodes_per_graph
        self.max_edges_per_graph = max_edges_per_graph
        self.use_torch_geometric = use_torch_geometric

    def create_batch(
        self, graphs: List[GraphData], follow_batch: Optional[List[str]] = None
    ) -> BatchedGraphData:
        """
        Create a batch from list of graphs.
        Args:
            graphs: List of GraphData objects
            follow_batch: Attributes to track in batch
        Returns:
            BatchedGraphData with all graphs combined
        """
        if not graphs:
            return self._create_empty_batch()
        if self.use_torch_geometric:
            return self._create_batch_torch_geometric(graphs, follow_batch)
        else:
            return self._create_batch_manual(graphs)

    def _create_batch_torch_geometric(
        self, graphs: List[GraphData], follow_batch: Optional[List[str]] = None
    ) -> BatchedGraphData:
        """Create batch using PyTorch Geometric"""
        data_list = []
        for graph in graphs:
            data = Data(
                x=graph.node_features,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                edge_weight=graph.edge_weight,
                y=graph.target,
                mask=graph.mask,
            )
            if graph.graph_attr is not None:
                data.graph_attr = graph.graph_attr
            data_list.append(data)
        batch = Batch.from_data_list(data_list, follow_batch=follow_batch)
        return BatchedGraphData(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr if hasattr(batch, "edge_attr") else None,
            edge_weight=batch.edge_weight if hasattr(batch, "edge_weight") else None,
            batch=batch.batch,
            ptr=batch.ptr,
            graph_attr=batch.graph_attr if hasattr(batch, "graph_attr") else None,
            target=batch.y if hasattr(batch, "y") else None,
            mask=batch.mask if hasattr(batch, "mask") else None,
            num_graphs=batch.num_graphs,
            num_nodes_per_graph=[g.node_features.size(0) for g in graphs],
            num_edges_per_graph=[g.edge_index.size(1) for g in graphs],
        )

    def _create_batch_manual(self, graphs: List[GraphData]) -> BatchedGraphData:
        """Manually create batch without PyTorch Geometric"""
        num_graphs = len(graphs)
        num_nodes_per_graph = [g.node_features.size(0) for g in graphs]
        num_edges_per_graph = [g.edge_index.size(1) for g in graphs]
        node_offsets = torch.cumsum(torch.tensor([0] + num_nodes_per_graph[:-1]), dim=0)
        if self.pad_node_features and self.max_nodes_per_graph:
            x = self._pad_node_features(graphs)
            mask = self._create_node_masks(graphs)
        else:
            x = torch.cat([g.node_features for g in graphs], dim=0)
            mask = None
        edge_indices = []
        for i, graph in enumerate(graphs):
            offset_edge_index = graph.edge_index + node_offsets[i]
            edge_indices.append(offset_edge_index)
        edge_index = torch.cat(edge_indices, dim=1)
        if any(g.edge_attr is not None for g in graphs):
            edge_attr = torch.cat(
                [
                    (
                        g.edge_attr
                        if g.edge_attr is not None
                        else torch.zeros(g.edge_index.size(1), 1)
                    )
                    for g in graphs
                ],
                dim=0,
            )
        else:
            edge_attr = None
        if any(g.edge_weight is not None for g in graphs):
            edge_weight = torch.cat(
                [
                    (
                        g.edge_weight
                        if g.edge_weight is not None
                        else torch.ones(g.edge_index.size(1))
                    )
                    for g in graphs
                ],
                dim=0,
            )
        else:
            edge_weight = None
        batch = torch.cat(
            [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes_per_graph)]
        )
        ptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_nodes_per_graph), dim=0)))
        if any(g.graph_attr is not None for g in graphs):
            graph_attr = self._batch_graph_attributes(graphs)
        else:
            graph_attr = None
        if any(g.target is not None for g in graphs):
            targets = [g.target if g.target is not None else torch.zeros(0) for g in graphs]
            target = pad_sequence(targets, batch_first=True, padding_value=-1)
        else:
            target = None
        return BatchedGraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            batch=batch,
            ptr=ptr,
            graph_attr=graph_attr,
            target=target,
            mask=mask,
            num_graphs=num_graphs,
            num_nodes_per_graph=num_nodes_per_graph,
            num_edges_per_graph=num_edges_per_graph,
        )

    def _pad_node_features(self, graphs: List[GraphData]) -> torch.Tensor:
        """Pad node features to maximum size"""
        max_nodes = self.max_nodes_per_graph or max(g.node_features.size(0) for g in graphs)
        feature_dim = graphs[0].node_features.size(1)
        padded_features = []
        for graph in graphs:
            num_nodes = graph.node_features.size(0)
            if num_nodes < max_nodes:
                padding = torch.zeros(max_nodes - num_nodes, feature_dim)
                padded = torch.cat([graph.node_features, padding], dim=0)
            else:
                padded = graph.node_features[:max_nodes]
            padded_features.append(padded)
        return torch.stack(padded_features)

    def _create_node_masks(self, graphs: List[GraphData]) -> torch.Tensor:
        """Create masks for padded nodes"""
        max_nodes = self.max_nodes_per_graph or max(g.node_features.size(0) for g in graphs)
        masks = []
        for graph in graphs:
            num_nodes = min(graph.node_features.size(0), max_nodes)
            mask = torch.zeros(max_nodes, dtype=torch.bool)
            mask[:num_nodes] = True
            masks.append(mask)
        return torch.stack(masks)

    def _batch_graph_attributes(self, graphs: List[GraphData]) -> torch.Tensor:
        """Batch graph-level attributes"""
        if self.pad_graph_features:
            max_dim = max(g.graph_attr.size(-1) if g.graph_attr is not None else 0 for g in graphs)
            attrs = []
            for g in graphs:
                if g.graph_attr is not None:
                    if g.graph_attr.size(-1) < max_dim:
                        padding = torch.zeros(max_dim - g.graph_attr.size(-1))
                        attr = torch.cat([g.graph_attr, padding])
                    else:
                        attr = g.graph_attr
                else:
                    attr = torch.zeros(max_dim)
                attrs.append(attr)
            return torch.stack(attrs)
        else:
            return torch.cat(
                [g.graph_attr if g.graph_attr is not None else torch.zeros(1) for g in graphs]
            )

    def unbatch(self, batched_data: BatchedGraphData) -> List[GraphData]:
        """
        Unbatch data back to individual graphs.
        Args:
            batched_data: Batched graph data
        Returns:
            List of individual GraphData objects
        """
        graphs = []
        for i in range(batched_data.num_graphs):
            if batched_data.batch is not None:
                node_mask = batched_data.batch == i
                node_indices = torch.where(node_mask)[0]
            else:
                start_idx = batched_data.ptr[i]
                end_idx = batched_data.ptr[i + 1]
                node_indices = torch.arange(start_idx, end_idx)
            if batched_data.x.dim() == 3:
                node_features = batched_data.x[i]
                if batched_data.mask is not None:
                    node_features = node_features[batched_data.mask[i]]
            else:
                node_features = batched_data.x[node_indices]
            edge_mask = (
                (batched_data.edge_index[0] >= node_indices[0])
                & (batched_data.edge_index[0] < node_indices[0] + len(node_indices))
                & (batched_data.edge_index[1] >= node_indices[0])
                & (batched_data.edge_index[1] < node_indices[0] + len(node_indices))
            )
            edge_index = batched_data.edge_index[:, edge_mask] - node_indices[0]
            edge_attr = None
            if batched_data.edge_attr is not None:
                edge_attr = batched_data.edge_attr[edge_mask]
            edge_weight = None
            if batched_data.edge_weight is not None:
                edge_weight = batched_data.edge_weight[edge_mask]
            graph_attr = None
            if batched_data.graph_attr is not None:
                graph_attr = batched_data.graph_attr[i]
            target = None
            if batched_data.target is not None:
                target = batched_data.target[i]
            graphs.append(
                GraphData(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    edge_weight=edge_weight,
                    graph_attr=graph_attr,
                    target=target,
                )
            )
        return graphs

    def _create_empty_batch(self) -> BatchedGraphData:
        """Create an empty batch"""
        return BatchedGraphData(
            x=torch.empty(0, 0),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            batch=torch.empty(0, dtype=torch.long),
            ptr=torch.tensor([0], dtype=torch.long),
            num_graphs=0,
        )

    def collate_fn(self, batch: List[GraphData]) -> BatchedGraphData:
        """
        Collate function for DataLoader.
        Args:
            batch: List of GraphData objects
        Returns:
            BatchedGraphData
        """

        return self.create_batch(batch)


class DynamicBatchSampler:
    """
    Dynamic batch sampler that groups graphs by size for efficiency.
    Creates batches where graphs have similar numbers of nodes/edges
    to minimize padding overhead.
    """

    def __init__(
        self,
        graph_sizes: List[tuple[int, int]],
        batch_size: int,
        size_threshold: float = 0.2,
        shuffle: bool = True,
    ) -> None:
        """
        Initialize dynamic batch sampler.
        Args:
            graph_sizes: List of (num_nodes, num_edges) for each graph
            batch_size: Maximum batch size
            size_threshold: Relative size difference threshold
            shuffle: Whether to shuffle within size groups
        """
        self.graph_sizes = graph_sizes
        self.batch_size = batch_size
        self.size_threshold = size_threshold
        self.shuffle = shuffle
        self.size_groups = self._group_by_size()

    def _group_by_size(self) -> List[List[int]]:
        """Group graph indices by similar sizes"""
        groups = []
        assigned = [False] * len(self.graph_sizes)
        for i in range(len(self.graph_sizes)):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            base_nodes, base_edges = self.graph_sizes[i]
            for j in range(i + 1, len(self.graph_sizes)):
                if assigned[j]:
                    continue
                nodes, edges = self.graph_sizes[j]
                node_ratio = abs(nodes - base_nodes) / (base_nodes + 1e-06)
                edge_ratio = abs(edges - base_edges) / (base_edges + 1e-06)
                if node_ratio <= self.size_threshold and edge_ratio <= self.size_threshold:
                    group.append(j)
                    assigned[j] = True
            groups.append(group)
        return groups

    def __iter__(self):
        """Iterate over batches"""
        groups = self.size_groups.copy()
        if self.shuffle:
            np.random.shuffle(groups)
        for group in groups:
            indices = group.copy()
            if self.shuffle:
                np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                yield batch

    def __len__(self):
        """Total number of batches"""

        return sum(
            (len(group) + self.batch_size - 1) // self.batch_size for group in self.size_groups
        )


class StreamingBatchProcessor:
    """

    Processes graphs in a streaming fashion for memory efficiency.
    Useful for very large datasets that don't fit in memory.
    """

    def __init__(
        self,
        batch_processor: GraphBatchProcessor,
        buffer_size: int = 100,
        prefetch_factor: int = 2,
    ) -> None:
        """
        Initialize streaming processor.
        Args:
            batch_processor: Base batch processor
            buffer_size: Size of internal buffer
            prefetch_factor: Number of batches to prefetch
        """
        self.batch_processor = batch_processor
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        self.buffer = []

    def process_stream(self, graph_iterator, batch_size: int, process_fn=None):
        """
        Process graphs from an iterator in batches.
        Args:
            graph_iterator: Iterator yielding GraphData objects
            batch_size: Batch size
            process_fn: Optional processing function for each batch
        Yields:
            Processed batch results
        """
        for graph in graph_iterator:
            self.buffer.append(graph)
            if len(self.buffer) >= batch_size:
                batch_data = self.buffer[:batch_size]
                self.buffer = self.buffer[batch_size:]
                batched = self.batch_processor.create_batch(batch_data)
                if process_fn:
                    result = process_fn(batched)
                    yield result
                else:
                    yield batched
        if self.buffer:
            batched = self.batch_processor.create_batch(self.buffer)
            if process_fn:
                result = process_fn(batched)
                yield result
            else:
                yield batched
            self.buffer = []


def create_mini_batches(
    graphs: List[GraphData],
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> List[List[GraphData]]:
    """
    Create mini-batches from a list of graphs.
    Args:
        graphs: List of graphs
        batch_size: Batch size
        shuffle: Whether to shuffle
        drop_last: Drop last incomplete batch
    Returns:
        List of mini-batches
    """
    indices = list(range(len(graphs)))
    if shuffle:
        np.random.shuffle(indices)
    batches = []
    for i in range(0, len(indices), batch_size):
        if i + batch_size <= len(indices) or not drop_last:
            batch_indices = indices[i : i + batch_size]
            batch = [graphs[idx] for idx in batch_indices]
            batches.append(batch)
    return batches


if __name__ == "__main__":
    graphs = []
    for i in range(10):
        num_nodes = np.random.randint(5, 20)
        num_edges = np.random.randint(num_nodes, num_nodes * 3)
        node_features = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        graph = GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=torch.randn(num_edges, 8),
            target=torch.tensor([i % 3]),
        )
        graphs.append(graph)
    processor = GraphBatchProcessor(pad_node_features=True, max_nodes_per_graph=25)
    batched = processor.create_batch(graphs)
    print(f"Batched {batched.num_graphs} graphs")
    print(f"Total nodes: {batched.x.size(0)}")
    print(f"Total edges: {batched.edge_index.size(1)}")
    print(f"Node features shape: {batched.x.shape}")
    print(f"Batch tensor shape: {batched.batch.shape}")
    unbatched = processor.unbatch(batched)
    print(f"\nUnbatched back to {len(unbatched)} graphs")

"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]
import torch
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

# Configure logging
logger = logging.getLogger(__name__)

"""
This module implements edge processing for GNN feature extraction and normalization.
It handles various edge relationships, directed/undirected processing, and weighted attributes.
"""


class EdgeType(Enum):
    """Types of edges in the graph"""

    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    BIDIRECTIONAL = "bidirectional"


class EdgeFeatureType(Enum):
    """Types of edge features"""

    WEIGHT = "weight"
    DISTANCE = "distance"
    SIMILARITY = "similarity"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


@dataclass
class EdgeConfig:
    """Configuration for edge processing"""

    edge_type: EdgeType = EdgeType.DIRECTED
    feature_types: List[EdgeFeatureType] = field(default_factory=list)
    normalize_weights: bool = True
    self_loops: bool = False
    max_edges_per_node: Optional[int] = None
    edge_sampling_strategy: Optional[str] = None  # "random", "importance", "topk"


@dataclass
class Edge:
    """Represents an edge in the graph"""

    source: int
    target: int
    features: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    edge_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeBatch:
    """Batch of edges for efficient processing"""

    edge_index: torch.Tensor  # Shape: [2, num_edges]
    edge_attr: Optional[torch.Tensor] = None  # Shape: [num_edges, num_features]
    edge_weight: Optional[torch.Tensor] = None  # Shape: [num_edges]
    edge_type: Optional[torch.Tensor] = None  # Shape: [num_edges]
    batch_ptr: Optional[torch.Tensor] = None  # For batched graphs
    metadata: Dict[str, Any] = field(default_factory=dict)


class EdgeProcessor:
    """
    Processes edge features and relationships for GNN computation.
    Handles:
    - Edge feature extraction and normalization
    - Directed/undirected edge conversion
    - Weighted relationship processing
    - Efficient edge representation
    - Edge sampling and filtering
    """

    def __init__(self, config: EdgeConfig) -> None:
        """
        Initialize the edge processor.
        Args:
            config: Edge processing configuration
        """
        self.config = config
        self.scalers: Dict[str, Any] = {}
        self.edge_type_mapping: Dict[str, int] = {}
        self._initialize_processors()

    def _initialize_processors(self) -> None:
        """Initialize feature processors"""
        for feature_type in self.config.feature_types:
            if feature_type in [
                EdgeFeatureType.WEIGHT,
                EdgeFeatureType.SIMILARITY,
            ]:
                self.scalers[feature_type.value] = StandardScaler()
            elif feature_type == EdgeFeatureType.DISTANCE:
                # Use MinMaxScaler for distances to ensure non-negative values
                self.scalers[feature_type.value] = MinMaxScaler()

    def process_edges(self, edges: List[Edge], num_nodes: int) -> EdgeBatch:
        """
        Process a list of edges into an efficient batch representation.
        Args:
            edges: List of Edge objects
            num_nodes: Total number of nodes in the graph
        Returns:
            EdgeBatch with processed edges
        """
        if not edges:
            return self._create_empty_batch(num_nodes)
        # Convert edges based on edge type
        processed_edges = self._convert_edge_type(edges)
        # Extract edge indices
        edge_index = self._extract_edge_index(processed_edges)
        # Extract edge features
        edge_attr = self._extract_edge_features(processed_edges)
        # Extract edge weights
        edge_weight = self._extract_edge_weights(processed_edges)
        # Handle edge sampling if configured
        if self.config.max_edges_per_node:
            edge_index, edge_attr, edge_weight = self._sample_edges(
                edge_index, edge_attr, edge_weight, num_nodes
            )
        # Create edge type tensor if needed
        edge_type = self._extract_edge_types(processed_edges) if edges else None
        return EdgeBatch(
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            edge_type=edge_type,
            metadata={
                "num_edges": edge_index.shape[1],
                "num_nodes": num_nodes,
                "has_self_loops": self._has_self_loops(edge_index),
            },
        )

    def _convert_edge_type(self, edges: List[Edge]) -> List[Edge]:
        """Convert edges based on configured edge type"""
        if self.config.edge_type == EdgeType.UNDIRECTED:
            return self._make_undirected(edges)
        elif self.config.edge_type == EdgeType.BIDIRECTIONAL:
            return self._make_bidirectional(edges)
        return edges

    def _make_undirected(self, edges: List[Edge]) -> List[Edge]:
        """Convert directed edges to undirected"""
        undirected_edges = []
        seen_pairs = set()
        for edge in edges:
            pair = (min(edge.source, edge.target), max(edge.source, edge.target))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                # Create undirected edge
                new_edge = Edge(
                    source=pair[0],
                    target=pair[1],
                    features=edge.features.copy(),
                    weight=edge.weight,
                    edge_type=edge.edge_type,
                    metadata=edge.metadata.copy(),
                )
                undirected_edges.append(new_edge)
        return undirected_edges

    def _make_bidirectional(self, edges: List[Edge]) -> List[Edge]:
        """Convert edges to bidirectional (add reverse edges)"""
        bidirectional_edges = []
        for edge in edges:
            # Add original edge
            bidirectional_edges.append(edge)
            # Add reverse edge if not self-loop
            if edge.source != edge.target:
                reverse_edge = Edge(
                    source=edge.target,
                    target=edge.source,
                    features=edge.features.copy(),
                    weight=edge.weight,
                    edge_type=edge.edge_type,
                    metadata=edge.metadata.copy(),
                )
                bidirectional_edges.append(reverse_edge)
        return bidirectional_edges

    def _extract_edge_index(self, edges: List[Edge]) -> torch.Tensor:
        """Extract edge indices as tensor"""
        sources = [edge.source for edge in edges]
        targets = [edge.target for edge in edges]
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        # Add self-loops if configured
        if self.config.self_loops:
            num_nodes = max(max(sources), max(targets)) + 1
            self_loop_index = torch.arange(num_nodes, dtype=torch.long)
            self_loops = torch.stack([self_loop_index, self_loop_index])
            edge_index = torch.cat([edge_index, self_loops], dim=1)
        return edge_index

    def _extract_edge_features(self, edges: List[Edge]) -> Optional[torch.Tensor]:
        """Extract and normalize edge features"""
        if not self.config.feature_types:
            return None
        feature_arrays = []
        for feature_type in self.config.feature_types:
            features: Optional[torch.Tensor] = None
            if feature_type == EdgeFeatureType.WEIGHT:
                features = self._extract_weight_features(edges)
            elif feature_type == EdgeFeatureType.DISTANCE:
                features = self._extract_distance_features(edges)
            elif feature_type == EdgeFeatureType.SIMILARITY:
                features = self._extract_similarity_features(edges)
            elif feature_type == EdgeFeatureType.CATEGORICAL:
                features = self._extract_categorical_features(edges)
            elif feature_type == EdgeFeatureType.TEMPORAL:
                features = self._extract_temporal_features(edges)
            elif feature_type == EdgeFeatureType.EMBEDDING:
                features = self._extract_embedding_features(edges)
            else:
                features = self._extract_custom_features(edges, feature_type.value)
            if features is not None:
                feature_arrays.append(features)
        if not feature_arrays:
            return None
        # Concatenate all features
        edge_attr = torch.cat(feature_arrays, dim=1)
        # Add features for self-loops if needed
        if self.config.self_loops:
            num_self_loops = edge_attr.shape[0] - len(edges)
            if num_self_loops > 0:
                self_loop_features = torch.zeros(num_self_loops, edge_attr.shape[1])
                edge_attr = torch.cat([edge_attr, self_loop_features], dim=0)
        return edge_attr

    def _extract_weight_features(self, edges: List[Edge]) -> torch.Tensor:
        """Extract edge weight features"""
        weights = np.array([edge.weight for edge in edges]).reshape(-1, 1)
        if self.config.normalize_weights:
            if "weight" not in self.scalers:
                self.scalers["weight"] = StandardScaler()
            weights = self.scalers["weight"].fit_transform(weights)
        return torch.tensor(weights, dtype=torch.float32)

    def _extract_distance_features(self, edges: List[Edge]) -> Optional[torch.Tensor]:
        """Extract distance-based features"""
        distances: List[float] = []
        for edge in edges:
            if "distance" in edge.features:
                distances.append(edge.features["distance"])
            elif "positions" in edge.metadata:
                # Calculate distance from positions
                pos_source = edge.metadata["positions"][edge.source]
                pos_target = edge.metadata["positions"][edge.target]
                distance = np.linalg.norm(np.array(pos_source) - np.array(pos_target))
                distances.append(distance)
            else:
                distances.append(0.0)
        distances_array = np.array(distances).reshape(-1, 1)
        # Normalize distances to [0, 1] to ensure non-negative values
        if "distance" not in self.scalers:
            self.scalers["distance"] = MinMaxScaler()
        distances_normalized = self.scalers["distance"].fit_transform(distances_array)
        return torch.tensor(distances_normalized, dtype=torch.float32)

    def _extract_similarity_features(self, edges: List[Edge]) -> Optional[torch.Tensor]:
        """Extract similarity-based features"""
        similarities: List[float] = []
        for edge in edges:
            if "similarity" in edge.features:
                similarities.append(edge.features["similarity"])
            else:
                # Default similarity based on weight
                similarities.append(edge.weight)
        similarities_array = np.array(similarities).reshape(-1, 1)
        # Ensure similarities are in [0, 1]
        similarities_clipped = np.clip(similarities_array, 0, 1)
        return torch.tensor(similarities_clipped, dtype=torch.float32)

    def _extract_categorical_features(self, edges: List[Edge]) -> Optional[torch.Tensor]:
        """Extract categorical edge features"""
        if not any("category" in edge.features for edge in edges):
            return None
        # Collect all categories
        categories = []
        for edge in edges:
            category = edge.features.get("category", "unknown")
            categories.append(str(category))
        # Create category mapping
        unique_categories = sorted(set(categories))
        category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        # One-hot encode
        num_categories = len(unique_categories)
        one_hot = np.zeros((len(edges), num_categories))
        for i, category in enumerate(categories):
            one_hot[i, category_to_idx[category]] = 1.0
        return torch.tensor(one_hot, dtype=torch.float32)

    def _extract_temporal_features(self, edges: List[Edge]) -> Optional[torch.Tensor]:
        """Extract temporal edge features"""
        temporal_features = []
        for edge in edges:
            if "timestamp" in edge.features:
                # Extract temporal features similar to node feature extractor
                timestamp = edge.features["timestamp"]
                features = self._decompose_timestamp(timestamp)
                temporal_features.append(features)
            else:
                temporal_features.append([0.0] * 7)  # Default temporal features
        return torch.tensor(temporal_features, dtype=torch.float32)

    def _decompose_timestamp(self, timestamp: Union[int, float, str]) -> List[float]:
        """Decompose timestamp into multiple features"""
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp)
            timestamp = dt.timestamp()
        else:
            dt = datetime.fromtimestamp(timestamp)
        return [
            dt.hour / 24.0,
            dt.weekday() / 6.0,
            dt.day / 31.0,
            dt.month / 12.0,
            (dt.year - 2000) / 100.0,
            float(dt.weekday() >= 5),
            timestamp / 1e10,
        ]

    def _extract_embedding_features(self, edges: List[Edge]) -> Optional[torch.Tensor]:
        """Extract embedding features from edges"""
        embeddings: List[np.ndarray] = []
        embedding_dim = None
        for edge in edges:
            if "embedding" in edge.features:
                embedding = edge.features["embedding"]
                if isinstance(embedding, (list, np.ndarray)):
                    if embedding_dim is None:
                        embedding_dim = len(embedding)
                    embeddings.append(
                        np.array(embedding) if isinstance(embedding, list) else embedding
                    )
                else:
                    # Generate embedding from edge properties
                    if embedding_dim is None:
                        embedding_dim = 8  # Default dimension
                    hash_val = hash((edge.source, edge.target, edge.edge_type))
                    np.random.seed(abs(hash_val) % (2**32))
                    embedding = np.random.randn(embedding_dim) * 0.1
                    np.random.seed()
                    embeddings.append(embedding)
            else:
                if embedding_dim is None:
                    embedding_dim = 8
                embeddings.append(np.zeros(embedding_dim))
        embeddings_array = np.array(embeddings)
        # Normalize embeddings, handling zero vectors properly
        embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32)
        # Handle zero vectors by replacing them with small random vectors
        norms = torch.norm(embeddings_tensor, p=2, dim=1, keepdim=True)
        zero_mask = norms.squeeze() < 1e-8
        if zero_mask.any():
            embeddings_tensor[zero_mask] = torch.randn_like(embeddings_tensor[zero_mask]) * 0.1
        # Now normalize to unit vectors
        embeddings_normalized = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
        return embeddings_normalized

    def _extract_custom_features(
        self, edges: List[Edge], feature_name: str
    ) -> Optional[torch.Tensor]:
        """Extract custom features from edges"""
        features = []
        for edge in edges:
            if feature_name in edge.features:
                feature = edge.features[feature_name]
                if isinstance(feature, (list, np.ndarray)):
                    features.append(feature)
                else:
                    features.append([float(feature)])
            else:
                features.append([0.0])
        return torch.tensor(features, dtype=torch.float32)

    def _extract_edge_weights(self, edges: List[Edge]) -> torch.Tensor:
        """Extract edge weights"""
        weights = [edge.weight for edge in edges]
        # Add weights for self-loops if needed
        if self.config.self_loops:
            num_self_loops = max(max(e.source for e in edges), max(e.target for e in edges)) + 1
            weights.extend([1.0] * num_self_loops)
        return torch.tensor(weights, dtype=torch.float32)

    def _extract_edge_types(self, edges: List[Edge]) -> Optional[torch.Tensor]:
        """Extract edge types if available"""
        if not any(edge.edge_type for edge in edges):
            return None
        # Create edge type mapping
        edge_types = []
        for edge in edges:
            edge_type = edge.edge_type or "default"
            if edge_type not in self.edge_type_mapping:
                self.edge_type_mapping[edge_type] = len(self.edge_type_mapping)
            edge_types.append(self.edge_type_mapping[edge_type])
        # Add types for self-loops if needed
        if self.config.self_loops:
            default_type = self.edge_type_mapping.get("self_loop", len(self.edge_type_mapping))
            if "self_loop" not in self.edge_type_mapping:
                self.edge_type_mapping["self_loop"] = default_type
            num_self_loops = len(edges) - len(edge_types)
            edge_types.extend([default_type] * num_self_loops)
        return torch.tensor(edge_types, dtype=torch.long)

    def _sample_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Sample edges based on configured strategy"""
        if self.config.edge_sampling_strategy == "random":
            return self._random_sample_edges(edge_index, edge_attr, edge_weight, num_nodes)
        elif self.config.edge_sampling_strategy == "importance":
            return self._importance_sample_edges(edge_index, edge_attr, edge_weight, num_nodes)
        elif self.config.edge_sampling_strategy == "topk":
            return self._topk_sample_edges(edge_index, edge_attr, edge_weight, num_nodes)
        else:
            return edge_index, edge_attr, edge_weight

    def _random_sample_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Randomly sample edges per node"""
        sampled_indices = set()
        for node in range(num_nodes):
            # Find outgoing edges from this node
            outgoing_edges = torch.where(edge_index[0] == node)[0]
            if (
                self.config.max_edges_per_node is not None
                and len(outgoing_edges) > self.config.max_edges_per_node
            ):
                # Randomly sample outgoing edges
                perm = torch.randperm(len(outgoing_edges))[: self.config.max_edges_per_node]
                sampled_indices.update(outgoing_edges[perm].tolist())
            else:
                sampled_indices.update(outgoing_edges.tolist())
        # Convert to sorted list for consistent ordering
        sampled_indices_list = sorted(list(sampled_indices))
        sampled_indices_tensor = torch.tensor(sampled_indices_list, dtype=torch.long)
        # Extract sampled edges
        edge_index = edge_index[:, sampled_indices_tensor]
        edge_weight = edge_weight[sampled_indices_tensor]
        if edge_attr is not None:
            edge_attr = edge_attr[sampled_indices_tensor]
        return edge_index, edge_attr, edge_weight

    def _importance_sample_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Sample edges based on importance (weight)"""
        sampled_indices = set()
        for node in range(num_nodes):
            # Find outgoing edges from this node
            outgoing_edges = torch.where(edge_index[0] == node)[0]
            if (
                self.config.max_edges_per_node is not None
                and len(outgoing_edges) > self.config.max_edges_per_node
            ):
                # Sample based on weights
                node_weights = edge_weight[outgoing_edges]
                # Normalize weights to probabilities
                probs = node_weights / node_weights.sum()
                # Sample without replacement
                sampled = torch.multinomial(
                    probs,
                    min(self.config.max_edges_per_node, len(outgoing_edges)),
                    replacement=False,
                )
                sampled_indices.update(outgoing_edges[sampled].tolist())
            else:
                sampled_indices.update(outgoing_edges.tolist())
        # Convert to sorted list for consistent ordering
        sampled_indices_list = sorted(list(sampled_indices))
        sampled_indices_tensor = torch.tensor(sampled_indices_list, dtype=torch.long)
        # Extract sampled edges
        edge_index = edge_index[:, sampled_indices_tensor]
        edge_weight = edge_weight[sampled_indices_tensor]
        if edge_attr is not None:
            edge_attr = edge_attr[sampled_indices_tensor]
        return edge_index, edge_attr, edge_weight

    def _topk_sample_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Sample top-k edges by weight per node"""
        sampled_indices = set()
        for node in range(num_nodes):
            # Find outgoing edges from this node
            outgoing_edges = torch.where(edge_index[0] == node)[0]
            if (
                self.config.max_edges_per_node is not None
                and len(outgoing_edges) > self.config.max_edges_per_node
            ):
                # Get top-k by weight
                node_weights = edge_weight[outgoing_edges]
                topk_values, topk_indices = torch.topk(node_weights, self.config.max_edges_per_node)
                sampled_indices.update(outgoing_edges[topk_indices].tolist())
            else:
                sampled_indices.update(outgoing_edges.tolist())
        # Convert to sorted list for consistent ordering
        sampled_indices_list = sorted(list(sampled_indices))
        sampled_indices_tensor = torch.tensor(sampled_indices_list, dtype=torch.long)
        # Extract sampled edges
        edge_index = edge_index[:, sampled_indices_tensor]
        edge_weight = edge_weight[sampled_indices_tensor]
        if edge_attr is not None:
            edge_attr = edge_attr[sampled_indices_tensor]
        return edge_index, edge_attr, edge_weight

    def _has_self_loops(self, edge_index: torch.Tensor) -> bool:
        """Check if edge index contains self-loops"""
        return bool(torch.any(edge_index[0] == edge_index[1]).item())

    def _create_empty_batch(self, num_nodes: int) -> EdgeBatch:
        """Create an empty edge batch"""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        # Add self-loops if configured
        if self.config.self_loops:
            self_loop_index = torch.arange(num_nodes, dtype=torch.long)
            edge_index = torch.stack([self_loop_index, self_loop_index])
            edge_weight = torch.ones(num_nodes, dtype=torch.float32)
        else:
            edge_weight = torch.zeros(0, dtype=torch.float32)
        return EdgeBatch(
            edge_index=edge_index,
            edge_attr=None,
            edge_weight=edge_weight,
            edge_type=None,
            metadata={"num_edges": edge_index.shape[1], "num_nodes": num_nodes},
        )

    def to_adjacency_matrix(self, edge_batch: EdgeBatch, num_nodes: int) -> sp.csr_matrix:
        """Convert edge batch to sparse adjacency matrix"""
        edge_index = edge_batch.edge_index.numpy()
        weights = (
            edge_batch.edge_weight.numpy()
            if edge_batch.edge_weight is not None
            else np.ones(edge_index.shape[1])
        )
        # Create sparse matrix
        adj_matrix = sp.csr_matrix(
            (weights, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)
        )
        return adj_matrix

    def compute_edge_statistics(self, edge_batch: EdgeBatch, num_nodes: int) -> Dict[str, Any]:
        """Compute statistics about the edge batch"""
        # Compute statistics about the edge batch
        edge_index = edge_batch.edge_index
        # Degree statistics
        in_degree = torch.bincount(edge_index[1], minlength=num_nodes).float()
        out_degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
        # Edge weight statistics
        if edge_batch.edge_weight is not None:
            weight_stats = {
                "mean_weight": edge_batch.edge_weight.mean().item(),
                "std_weight": edge_batch.edge_weight.std().item(),
                "min_weight": edge_batch.edge_weight.min().item(),
                "max_weight": edge_batch.edge_weight.max().item(),
            }
        else:
            weight_stats = {}
        # Edge type distribution
        if edge_batch.edge_type is not None:
            edge_type_counts = torch.bincount(edge_batch.edge_type)
            edge_type_dist = {f"type_{i}": count.item() for i, count in enumerate(edge_type_counts)}
        else:
            edge_type_dist = {}
        return {
            "num_edges": edge_index.shape[1],
            "num_nodes": num_nodes,
            "avg_in_degree": in_degree.mean().item(),
            "avg_out_degree": out_degree.mean().item(),
            "max_in_degree": in_degree.max().item(),
            "max_out_degree": out_degree.max().item(),
            "num_self_loops": torch.sum(edge_index[0] == edge_index[1]).item(),
            "density": edge_index.shape[1] / (num_nodes * num_nodes),
            **weight_stats,
            "edge_type_distribution": edge_type_dist,
        }


# Example usage
if __name__ == "__main__":
    # Configure edge processing
    config = EdgeConfig(
        edge_type=EdgeType.DIRECTED,
        feature_types=[
            EdgeFeatureType.WEIGHT,
            EdgeFeatureType.DISTANCE,
            EdgeFeatureType.CATEGORICAL,
        ],
        normalize_weights=True,
        self_loops=True,
        max_edges_per_node=10,
        edge_sampling_strategy="importance",
    )
    # Create edge processor
    processor = EdgeProcessor(config)
    # Example edges
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
        Edge(
            source=2,
            target=0,
            weight=0.9,
            features={"distance": 1.2, "category": "friend"},
        ),
        Edge(
            source=0,
            target=3,
            weight=0.4,
            features={"distance": 3.0, "category": "acquaintance"},
        ),
    ]
    # Process edges
    edge_batch = processor.process_edges(edges, num_nodes=4)
    print(f"Edge index shape: {edge_batch.edge_index.shape}")
    print(
        f"Edge attributes shape: {edge_batch.edge_attr.shape if edge_batch.edge_attr is not None else None}"
    )
    print(
        f"Edge weights shape: {edge_batch.edge_weight.shape if edge_batch.edge_weight is not None else 'None'}"
    )
    # Compute statistics
    stats = processor.compute_edge_statistics(edge_batch, num_nodes=4)
    print(f"Edge statistics: {stats}")

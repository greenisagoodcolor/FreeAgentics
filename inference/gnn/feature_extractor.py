"""Feature extractor for GNN models."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features that can be extracted."""

    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    GRAPH_STRUCTURAL = "graph_structural"


class NormalizationStrategy(Enum):
    """Normalization strategies for features."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    feature_types: List[FeatureType] = field(
        default_factory=lambda: [FeatureType.NUMERICAL]
    )
    normalization_strategy: NormalizationStrategy = NormalizationStrategy.STANDARD
    handle_missing: bool = True
    temporal_window_size: int = 10
    spatial_resolution: float = 7  # H3 resolution
    cache_features: bool = True


class NodeFeatureExtractor:
    """Extract features from nodes for GNN processing."""

    def __init__(self, config: FeatureConfig):
        """Initialize the feature extractor."""
        self.config = config
        self.normalizers: Dict[str, Any] = {}
        self.feature_cache: Dict[str, Tensor] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self._custom_extractors: Dict[str, Callable] = {}
        self._pca_components: Optional[Tensor] = None

    def extract_features(
        self,
        nodes: List[Dict[str, Any]],
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> Tensor:
        """Extract features from nodes."""
        if not nodes:
            return torch.empty(0, 0)

        # Check cache
        cache_key = self._get_cache_key(nodes)
        if self.config.cache_features and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Extract features by type
        feature_lists = []

        for feature_type in self.config.feature_types:
            if feature_type == FeatureType.SPATIAL:
                features = self._extract_spatial_features(nodes)
            elif feature_type == FeatureType.TEMPORAL:
                features = self._extract_temporal_features(nodes)
            elif feature_type == FeatureType.CATEGORICAL:
                features = self._extract_categorical_features(nodes)
            elif feature_type == FeatureType.NUMERICAL:
                features = self._extract_numerical_features(nodes)
            elif feature_type == FeatureType.GRAPH_STRUCTURAL:
                features = self._extract_graph_structural_features(nodes, edges)
            else:
                continue

            if features is not None:
                feature_lists.append(features)

        # Apply custom extractors
        for name, extractor in self._custom_extractors.items():
            try:
                custom_features = extractor(nodes)
                if isinstance(custom_features, Tensor) and custom_features.numel() > 0:
                    feature_lists.append(custom_features)
            except Exception as e:
                logger.warning(f"Custom extractor {name} failed: {e}")

        # Combine features
        if not feature_lists:
            features = torch.zeros(len(nodes), 1)
        else:
            features = torch.cat(feature_lists, dim=1)

        # Apply normalization
        if self.config.normalization_strategy != NormalizationStrategy.NONE:
            features = self._normalize_features(features)

        # Track feature statistics
        self._update_feature_stats(features)

        # Cache results
        if self.config.cache_features:
            self.feature_cache[cache_key] = features

        return features

    def _extract_spatial_features(
        self, nodes: List[Dict[str, Any]]
    ) -> Optional[Tensor]:
        """Extract spatial features from nodes using H3 hexagonal indexing.

        This is a critical component of the PyMDP+GMN+GNN+H3+LLM innovation stack.
        H3 provides hierarchical spatial indexing for multi-resolution analysis.
        """
        try:
            import h3

            h3_available = True
        except ImportError:
            h3_available = False

        spatial_data = []
        h3_indices = []

        for node in nodes:
            if "position" in node:
                pos = node["position"]

                # Handle different position formats
                if isinstance(pos, dict) and "lat" in pos and "lon" in pos:
                    # Geographic coordinates - use H3 if available
                    try:
                        lat, lon = float(pos["lat"]), float(pos["lon"])
                    except (ValueError, TypeError):
                        # Invalid lat/lon values, use default
                        lat, lon = 0.0, 0.0

                    if h3_available:
                        try:
                            # Convert to H3 index for hierarchical spatial analysis
                            h3_index = h3.latlng_to_cell(
                                lat, lon, int(self.config.spatial_resolution)
                            )
                            h3_indices.append(h3_index)

                            # Get H3 cell center for consistent positioning
                            center_lat, center_lon = h3.cell_to_latlng(h3_index)
                            spatial_data.append([center_lat, center_lon])
                        except Exception:
                            # Fallback to raw coordinates
                            spatial_data.append([lat, lon])
                            h3_indices.append(None)
                    else:
                        spatial_data.append([lat, lon])
                        h3_indices.append(None)

                elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    # Cartesian coordinates - ensure they are floats
                    try:
                        coords = [float(pos[0]), float(pos[1])]
                        spatial_data.append(coords)
                    except (ValueError, TypeError):
                        # Invalid coordinate values, use default
                        spatial_data.append([0.0, 0.0])
                    h3_indices.append(None)
                else:
                    # Handle malformed position data (e.g., strings, invalid dicts)
                    spatial_data.append([0.0, 0.0])
                    h3_indices.append(None)
            else:
                if self.config.handle_missing:
                    spatial_data.append([0.0, 0.0])
                    h3_indices.append(None)
                else:
                    return None

        features = torch.tensor(spatial_data, dtype=torch.float32)

        # Apply spatial resolution scaling for all coordinates when resolution != 1.0
        if self.config.spatial_resolution != 1.0:
            features = features / self.config.spatial_resolution

        # Store H3 indices for advanced spatial operations
        if hasattr(self, "last_h3_indices"):
            self.last_h3_indices = h3_indices

        return features

    def _extract_temporal_features(
        self, nodes: List[Dict[str, Any]]
    ) -> Optional[Tensor]:
        """Extract temporal features from nodes."""
        import datetime

        temporal_data = []

        for node in nodes:
            # Try to extract temporal features from timestamp or time_series
            temporal_row = []

            if "timestamp" in node:
                timestamp = node["timestamp"]
                if isinstance(timestamp, datetime.datetime):
                    # Extract temporal features: hour, day_of_week, day_of_month, month, is_weekend
                    temporal_row = [
                        timestamp.hour / 23.0,  # Normalized hour (0-1)
                        timestamp.weekday() / 6.0,  # Normalized day of week (0-1)
                        timestamp.day / 31.0,  # Normalized day of month (0-1)
                        timestamp.month / 12.0,  # Normalized month (0-1)
                        1.0 if timestamp.weekday() >= 5 else 0.0,  # Is weekend
                    ]
                else:
                    temporal_row = [0.0, 0.0, 0.0, 0.0, 0.0]
            elif "time_series" in node:
                ts = node["time_series"]
                if isinstance(ts, list) and len(ts) >= self.config.temporal_window_size:
                    # Use last temporal_window_size values
                    window = ts[-self.config.temporal_window_size :]
                    temporal_row = window
                else:
                    # Pad with zeros if needed
                    window = [0.0] * self.config.temporal_window_size
                    if isinstance(ts, list):
                        window[: len(ts)] = ts[: self.config.temporal_window_size]
                    temporal_row = window
            else:
                if self.config.handle_missing:
                    # Default temporal features: hour, day_of_week, day_of_month, month, is_weekend
                    temporal_row = [0.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    return None

            temporal_data.append(temporal_row)

        features = torch.tensor(temporal_data, dtype=torch.float32)

        # Apply temporal windowing if we have time series data (window size > 5)
        if features.shape[1] > 5:
            features = self._apply_temporal_windowing(features)

        return features

    def _extract_categorical_features(
        self, nodes: List[Dict[str, Any]]
    ) -> Optional[Tensor]:
        """Extract categorical features from nodes."""
        # Collect all categories
        all_categories = set()
        for node in nodes:
            if "category" in node:
                all_categories.add(node["category"])

        if not all_categories:
            return None

        # Create category to index mapping
        category_to_idx = {cat: idx for idx, cat in enumerate(sorted(all_categories))}
        num_categories = len(all_categories)

        # One-hot encode
        categorical_data = []
        for node in nodes:
            one_hot = [0.0] * num_categories
            if "category" in node and node["category"] in category_to_idx:
                one_hot[category_to_idx[node["category"]]] = 1.0
            categorical_data.append(one_hot)

        return torch.tensor(categorical_data, dtype=torch.float32)

    def _extract_numerical_features(
        self, nodes: List[Dict[str, Any]]
    ) -> Optional[Tensor]:
        """Extract numerical features from nodes."""
        # Determine numerical fields from both node level and attributes
        numerical_fields = set()
        for node in nodes:
            # Check direct node properties
            for key, value in node.items():
                if isinstance(value, (int, float)) and key not in [
                    "id",
                    "index",
                ]:
                    numerical_fields.add(key)

            # Check attributes nested structure
            if "attributes" in node and isinstance(node["attributes"], dict):
                for key, value in node["attributes"].items():
                    if isinstance(value, (int, float)):
                        numerical_fields.add(f"attributes.{key}")

        if not numerical_fields:
            return None

        numerical_fields_list = sorted(numerical_fields)

        # Extract values
        numerical_data = []
        for node in nodes:
            values = []
            for field_name in numerical_fields_list:
                if field_name.startswith("attributes."):
                    # Extract from attributes
                    attr_name = field_name[len("attributes.") :]
                    if "attributes" in node and attr_name in node["attributes"]:
                        value = node["attributes"][attr_name]
                        if value is None:
                            values.append(0.0 if self.config.handle_missing else np.nan)
                        else:
                            try:
                                values.append(float(value))
                            except (ValueError, TypeError):
                                values.append(
                                    0.0 if self.config.handle_missing else np.nan
                                )
                    else:
                        values.append(0.0 if self.config.handle_missing else np.nan)
                else:
                    # Extract from node directly
                    if field_name in node:
                        value = node[field_name]
                        if value is None:
                            values.append(0.0 if self.config.handle_missing else np.nan)
                        else:
                            try:
                                values.append(float(value))
                            except (ValueError, TypeError):
                                values.append(
                                    0.0 if self.config.handle_missing else np.nan
                                )
                    else:
                        values.append(0.0 if self.config.handle_missing else np.nan)
            numerical_data.append(values)

        features = torch.tensor(numerical_data, dtype=torch.float32)

        # Handle NaN and infinite values
        if self.config.handle_missing:
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        return features

    def _extract_graph_structural_features(
        self,
        nodes: List[Dict[str, Any]],
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[Tensor]:
        """Extract graph structural features."""
        # If edges is None but nodes contain edge information, extract edges from nodes
        if edges is None:
            edges = self._extract_edges_from_nodes(nodes)
            if edges is None:
                return None

        num_nodes = len(nodes)

        # Calculate degree features
        in_degree = torch.zeros(num_nodes)
        out_degree = torch.zeros(num_nodes)

        for src, dst in edges:
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                out_degree[src] += 1
                in_degree[dst] += 1

        # Calculate clustering coefficient (simplified)
        clustering = torch.zeros(num_nodes)

        # Create adjacency lists
        adj_list: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
        for src, dst in edges:
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                adj_list[src].add(dst)
                adj_list[dst].add(src)  # Treat as undirected for clustering

        for i in range(num_nodes):
            neighbors = adj_list[i]
            if len(neighbors) > 1:
                # Count edges between neighbors
                edge_count = 0
                neighbors_list = list(neighbors)
                for j in range(len(neighbors_list)):
                    for k in range(j + 1, len(neighbors_list)):
                        if neighbors_list[k] in adj_list[neighbors_list[j]]:
                            edge_count += 1

                max_edges = len(neighbors) * (len(neighbors) - 1) / 2
                clustering[i] = edge_count / max_edges if max_edges > 0 else 0

        # Combine structural features
        features = torch.stack([in_degree, out_degree, clustering], dim=1)

        return features

    def _extract_edges_from_nodes(
        self, nodes: List[Dict[str, Any]]
    ) -> Optional[List[Tuple[int, int]]]:
        """Extract edge list from nodes that contain edge information."""
        # Create node ID to index mapping
        node_id_to_idx = {node.get("id", str(i)): i for i, node in enumerate(nodes)}
        edges = []

        for i, node in enumerate(nodes):
            if "edges" in node and isinstance(node["edges"], list):
                for neighbor_id in node["edges"]:
                    if neighbor_id in node_id_to_idx:
                        neighbor_idx = node_id_to_idx[neighbor_id]
                        if neighbor_idx != i:  # Avoid self-loops
                            edges.append((i, neighbor_idx))

        return edges if edges else None

    def _apply_temporal_windowing(self, features: Union[Tensor, List]) -> Tensor:
        """Apply temporal windowing to extract statistics."""
        # Handle both tensor input and direct node list input for backward compatibility
        if isinstance(features, list):
            # This is a list of nodes, extract temporal features first
            temporal_features = self._extract_temporal_features(features)
            if temporal_features is None:
                return torch.zeros(len(features), 4)  # Default stats
            features = temporal_features

        # If features has only 5 dimensions (temporal extracted features), return as is
        if features.shape[1] <= 5:
            return features

        # Compute statistics over temporal dimension (for time series data)
        mean_features = features.mean(dim=1, keepdim=True)
        std_features = features.std(dim=1, keepdim=True)
        min_features = features.min(dim=1, keepdim=True)[0]
        max_features = features.max(dim=1, keepdim=True)[0]

        # Concatenate statistics
        windowed_features = torch.cat(
            [mean_features, std_features, min_features, max_features], dim=1
        )

        return windowed_features

    def _normalize_features(self, features: Tensor) -> Tensor:
        """Normalize features based on strategy."""
        if self.config.normalization_strategy == NormalizationStrategy.STANDARD:
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(
                dim=0, keepdim=True, unbiased=False
            )  # Use population std to avoid NaN with 1 sample
            std = torch.where((std == 0) | torch.isnan(std), torch.ones_like(std), std)
            return (features - mean) / std

        elif self.config.normalization_strategy == NormalizationStrategy.MINMAX:
            min_val = features.min(dim=0, keepdim=True)[0]
            max_val = features.max(dim=0, keepdim=True)[0]
            range_val = max_val - min_val
            range_val = torch.where(
                range_val == 0, torch.ones_like(range_val), range_val
            )
            return (features - min_val) / range_val

        elif self.config.normalization_strategy == NormalizationStrategy.ROBUST:
            # Use median and MAD for robust normalization
            median = features.median(dim=0, keepdim=True)[0]
            mad = (features - median).abs().median(dim=0, keepdim=True)[0]
            mad = torch.where(mad == 0, torch.ones_like(mad), mad)
            return (features - median) / (
                1.4826 * mad
            )  # 1.4826 is consistency constant

        return features

    def _compute_features(self, data: Any) -> Tensor:
        """Compute features from raw data (placeholder for custom extractors)."""
        if isinstance(data, dict):
            # Try custom extractors
            for name, extractor in self._custom_extractors.items():
                try:
                    result = extractor(data)
                    if isinstance(result, Tensor):
                        return result
                except Exception as e:
                    logger.warning(f"Custom extractor {name} failed: {e}")

        # Default: return zero vector
        return torch.zeros(1, 1)

    def _update_feature_stats(self, features: Tensor) -> None:
        """Update feature statistics for monitoring."""
        if features.numel() == 0:
            return

        # Compute statistics per feature dimension
        for i in range(features.shape[1]):
            feature_col = features[:, i]
            stats = {
                "mean": feature_col.mean().item(),
                "std": feature_col.std().item(),
                "min": feature_col.min().item(),
                "max": feature_col.max().item(),
            }
            self.feature_stats[f"feature_{i}"] = stats

    def _get_cache_key(self, nodes: List[Dict[str, Any]]) -> str:
        """Generate cache key for nodes."""
        # Hash based on node IDs, feature types, and actual node content
        node_ids = [str(node.get("id", i)) for i, node in enumerate(nodes)]
        feature_types = [ft.value for ft in self.config.feature_types]

        # Include content hash to detect changes in node data
        import json

        node_content = []
        for node in nodes:
            # Create a normalized representation of the node
            content = {
                "id": node.get("id"),
                "position": node.get("position"),
                "timestamp": str(node.get("timestamp", "")),
                "category": node.get("category"),
                "attributes": node.get("attributes", {}),
                "edges": node.get("edges", []),
            }
            node_content.append(json.dumps(content, sort_keys=True))

        key_str = (
            f"{','.join(node_ids)}:{','.join(feature_types)}:{','.join(node_content)}"
        )
        return str(hash(key_str))

    def compute_feature_importance(
        self, features: Tensor, targets: Optional[Tensor] = None
    ) -> List[float]:
        """Compute feature importance scores."""
        importance_scores = []

        if targets is not None:
            # Compute correlation-based importance
            for i in range(features.shape[1]):
                correlation = torch.corrcoef(torch.stack([features[:, i], targets]))[
                    0, 1
                ]
                importance_scores.append(abs(correlation.item()))
        else:
            # Use variance as proxy for importance
            variances = features.var(dim=0)
            for i in range(features.shape[1]):
                importance_scores.append(variances[i].item())

        # Store in feature stats as dictionary for backward compatibility
        importance_dict = {
            f"feature_{i}": score for i, score in enumerate(importance_scores)
        }
        self.feature_stats["importance"] = importance_dict

        return importance_scores

    def select_features(
        self,
        features: Tensor,
        n_features: Optional[int] = None,
        k: Optional[int] = None,
        method: str = "variance",
    ) -> Tensor:
        """Select top k features based on importance."""
        # Handle parameter compatibility - accept both n_features and k
        num_features = n_features or k
        if num_features is None:
            num_features = min(5, features.shape[1])

        if "importance" not in self.feature_stats:
            self.compute_feature_importance(features)

        importance_scores = self.feature_stats["importance"]

        # Sort features by importance
        sorted_indices = sorted(
            range(len(importance_scores)),
            key=lambda i: importance_scores[f"feature_{i}"],
            reverse=True,
        )[:num_features]

        # Select features
        selected_features = features[:, sorted_indices]

        return selected_features

    def polynomial_features(self, features: Tensor, degree: int = 2) -> Tensor:
        """Generate polynomial features."""
        if degree < 1:
            return features

        poly_features = [features]

        for d in range(2, degree + 1):
            # Generate degree d terms
            degree_features = []

            # For simplicity, only generate powers of individual features
            # Full polynomial expansion would be more complex
            degree_features.append(features**d)

            if degree_features:
                poly_features.extend(degree_features)

        return torch.cat(poly_features, dim=1)

    def pca_transform(self, features: Tensor, n_components: int) -> Tensor:
        """Apply PCA transformation to features."""
        if self._pca_components is None:
            # Compute PCA components
            centered = features - features.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / (features.shape[0] - 1)

            eigenvalues, eigenvectors = torch.linalg.eigh(cov)

            # Sort by eigenvalues (descending)
            indices = eigenvalues.argsort(descending=True)
            self._pca_components = eigenvectors[:, indices[:n_components]]

        # Transform features
        centered = features - features.mean(dim=0, keepdim=True)
        transformed = centered @ self._pca_components

        return transformed

    def add_custom_extractor(
        self, name: str, extractor: Callable[[Dict[str, Any]], Tensor]
    ):
        """Add a custom feature extractor."""
        self._custom_extractors[name] = extractor
        logger.info(f"Added custom feature extractor: {name}")

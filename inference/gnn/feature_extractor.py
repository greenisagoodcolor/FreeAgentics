"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (  # type: ignore[import-untyped]
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

try:
    import h3
except ImportError:
    h3 = None

# Import from layers.py for consistency
from .layers import AggregationType

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """Represents an edge in a graph with source, target, and optional weight"""

    source: int
    target: int
    weight: float = 1.0
    edge_type: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeConfig:
    """Configuration for edge processing"""

    directed: bool = True
    self_loops: bool = False
    weight_threshold: float = 0.0
    max_edges_per_node: Optional[int] = None
    edge_features: List[str] = field(default_factory=list)


@dataclass
class EdgeBatch:
    """Batch of processed edges"""

    edge_index: torch.Tensor  # Shape: [2, num_edges]
    edge_weight: torch.Tensor  # Shape: [num_edges]
    # Shape: [num_edges, num_features]
    edge_attr: Optional[torch.Tensor] = None
    num_edges: int = 0


class EdgeProcessor:
    """Processes edges for GNN training"""

    def __init__(self, config: EdgeConfig) -> None:
        """Initialize edge processor with configuration"""
        self.config = config

    def process_edges(self, edges: List[Edge], num_nodes: int) -> EdgeBatch:
        """
        Process a list of edges into a batch format.

        Args:
            edges: List of Edge objects
            num_nodes: Total number of nodes in the graph

        Returns:
            EdgeBatch with processed edge data
        """
        if not edges:
            return EdgeBatch(
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_weight=torch.zeros(0),
                num_edges=0,
            )

        # Filter edges based on configuration
        filtered_edges = []
        for edge in edges:
            # Check weight threshold
            if edge.weight < self.config.weight_threshold:
                continue

            # Check node bounds
            if edge.source >= num_nodes or edge.target >= num_nodes:
                continue

            # Skip self-loops if not allowed
            if not self.config.self_loops and edge.source == edge.target:
                continue

            filtered_edges.append(edge)

        if not filtered_edges:
            return EdgeBatch(
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_weight=torch.zeros(0),
                num_edges=0,
            )

        # Extract edge indices and weights
        sources = [edge.source for edge in filtered_edges]
        targets = [edge.target for edge in filtered_edges]
        weights = [edge.weight for edge in filtered_edges]

        # Create edge index tensor
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)

        # Add reverse edges if undirected
        if not self.config.directed:
            reverse_edge_index = torch.tensor([targets, sources], dtype=torch.long)
            edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

        return EdgeBatch(
            edge_index=edge_index, edge_weight=edge_weight, num_edges=edge_index.shape[1]
        )


class FeatureType(Enum):
    """Types of node features"""

    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    EMBEDDING = "embedding"
    TEXT = "text"
    GRAPH_STRUCTURAL = "graph_structural"


class NormalizationType(Enum):
    """Types of normalization methods"""

    STANDARD = "standard"  # Zero mean, unit variance
    MINMAX = "minmax"  # Scale to [0, 1]
    ROBUST = "robust"  # Robust to outliers
    LOG = "log"  # Log transformation
    NONE = "none"  # No normalization


@dataclass
class FeatureConfig:
    """Configuration for a feature"""

    name: str
    type: FeatureType
    dimension: Optional[int] = None
    normalization: NormalizationType = NormalizationType.STANDARD
    default_value: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    preprocessing_fn: Optional[Callable] = None
    values: Optional[List[str]] = None  # Added for categorical features


@dataclass
class ExtractionResult:
    """Result of feature extraction"""

    features: np.ndarray
    feature_names: List[str]
    feature_dims: Dict[str, tuple[int, int]]  # name -> (start_idx, end_idx)
    metadata: Dict[str, Any] = field(default_factory=dict)
    missing_mask: Optional[np.ndarray] = None


class NodeFeatureExtractor:
    """
    Extracts and normalizes node features for GNN processing.
    Handles:
    - Multiple feature types (spatial, temporal, categorical, numerical)
    - Missing data imputation
    - Feature normalization and scaling
    - Embedding generation
    - Graph structural features
    """

    def __init__(self, feature_configs: List[FeatureConfig]) -> None:
        """
        Initialize the feature extractor.
        Args:
            feature_configs: List of feature configurations
        """
        self.feature_configs = {config.name: config for config in feature_configs}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self._initialize_processors()

    def _initialize_processors(self) -> None:
        """Initialize feature processors based on configurations"""
        for name, config in self.feature_configs.items():
            if config.type == FeatureType.NUMERICAL:
                if config.normalization == NormalizationType.STANDARD:
                    self.scalers[name] = StandardScaler()
                elif config.normalization == NormalizationType.MINMAX:
                    self.scalers[name] = MinMaxScaler()
                elif config.normalization == NormalizationType.ROBUST:
                    self.scalers[name] = RobustScaler()
            elif config.type == FeatureType.CATEGORICAL:
                self.encoders[name] = LabelEncoder()
            elif config.type == FeatureType.TEXT:
                self.vectorizers[name] = TfidfVectorizer(max_features=config.dimension or 100)

    def extract_features(
        self, nodes: List[Dict[str, Any]], graph: Optional[Any] = None
    ) -> ExtractionResult:
        """
        Extract features from a list of nodes using Template Method pattern.
        Args:
            nodes: List of node dictionaries with feature values
            graph: Optional graph structure for structural features
        Returns:
            ExtractionResult with extracted and normalized features
        """
        if not nodes:
            return self._create_empty_result()

        extraction_context = self._initialize_extraction_context(nodes)
        self._extract_all_feature_types(nodes, graph, extraction_context)
        all_features = self._concatenate_features(extraction_context["feature_arrays"])

        return self._create_extraction_result(nodes, all_features, extraction_context)

    def _create_empty_result(self) -> ExtractionResult:
        """Create empty extraction result for no nodes"""
        return ExtractionResult(features=np.array([]), feature_names=[], feature_dims={})

    def _initialize_extraction_context(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize context for feature extraction"""
        return {
            "feature_arrays": [],
            "feature_names": [],
            "feature_dims": {},
            "current_idx": 0,
            "missing_mask": np.zeros((len(nodes), len(self.feature_configs)), dtype=bool),
        }

    def _extract_all_feature_types(
        self, nodes: List[Dict[str, Any]], graph: Optional[Any], extraction_context: Dict[str, Any]
    ) -> None:
        """Extract features for all configured feature types"""
        for feature_idx, (name, config) in enumerate(self.feature_configs.items()):
            logger.debug(f"Extracting feature: {name}")

            features, names = self._extract_single_feature_type(nodes, config, graph)
            self._update_missing_mask(nodes, name, feature_idx, extraction_context)
            self._record_feature_info(name, features, names, extraction_context)

    def _extract_single_feature_type(
        self, nodes: List[Dict[str, Any]], config: FeatureConfig, graph: Optional[Any]
    ) -> tuple[np.ndarray, List[str]]:
        """Extract features for a single feature type using Strategy pattern"""
        feature_extractors = {
            FeatureType.SPATIAL: lambda: self._extract_spatial_features(nodes, config),
            FeatureType.TEMPORAL: lambda: self._extract_temporal_features(nodes, config),
            FeatureType.CATEGORICAL: lambda: self._extract_categorical_features(nodes, config),
            FeatureType.NUMERICAL: lambda: self._extract_numerical_features(nodes, config),
            FeatureType.EMBEDDING: lambda: self._extract_embedding_features(nodes, config),
            FeatureType.TEXT: lambda: self._extract_text_features(nodes, config),
            FeatureType.GRAPH_STRUCTURAL: lambda: self._extract_structural_features(
                nodes, config, graph
            ),
        }

        extractor = feature_extractors.get(config.type)
        if extractor:
            return extractor()
        else:
            logger.warning(f"Unknown feature type: {config.type}")
            return (np.zeros((len(nodes), 1), dtype=np.float32), [f"{config.name}_unknown"])

    def _update_missing_mask(
        self,
        nodes: List[Dict[str, Any]],
        name: str,
        feature_idx: int,
        extraction_context: Dict[str, Any],
    ) -> None:
        """Update missing value mask for current feature"""
        for i, node in enumerate(nodes):
            if name not in node or node[name] is None:
                extraction_context["missing_mask"][i, feature_idx] = True

    def _record_feature_info(
        self, name: str, features: np.ndarray, names: List[str], extraction_context: Dict[str, Any]
    ) -> None:
        """Record feature information in extraction context"""
        # Record feature dimensions
        current_idx = extraction_context["current_idx"]
        extraction_context["feature_dims"][name] = (current_idx, current_idx + features.shape[1])
        extraction_context["current_idx"] += features.shape[1]

        # Add to arrays
        extraction_context["feature_arrays"].append(features)
        extraction_context["feature_names"].extend(names)

    def _concatenate_features(self, feature_arrays: List[np.ndarray]) -> np.ndarray:
        """Concatenate all feature arrays"""
        if feature_arrays:
            return np.concatenate(feature_arrays, axis=1)
        else:
            return np.zeros((0, 0))

    def _create_extraction_result(
        self,
        nodes: List[Dict[str, Any]],
        all_features: np.ndarray,
        extraction_context: Dict[str, Any],
    ) -> ExtractionResult:
        """Create final extraction result"""
        return ExtractionResult(
            features=all_features,
            feature_names=extraction_context["feature_names"],
            feature_dims=extraction_context["feature_dims"],
            missing_mask=extraction_context["missing_mask"],
            metadata={
                "num_nodes": len(nodes),
                "num_features": len(extraction_context["feature_names"]),
                "extraction_time": datetime.now().isoformat(),
            },
        )

    def _extract_spatial_features(
        self, nodes: List[Dict[str, Any]], config: FeatureConfig
    ) -> tuple[np.ndarray, list[str]]:
        """Extract spatial features like coordinates, H3 cells, regions"""
        feature_values = []
        for node in nodes:
            value = node.get(config.name, config.default_value)
            if value is None:
                # Handle missing spatial data
                if config.name in ["x", "y", "z"]:
                    feature_values.append([0.0] * (3 if config.name == "z" else 2))
                elif "h3" in config.name.lower():
                    feature_values.append([0.0] * 7)  # H3 cell features
                else:
                    feature_values.append([0.0])
            else:
                if isinstance(value, (list, tuple)):
                    feature_values.append(list(value))
                elif "h3" in config.name.lower() and isinstance(value, str):
                    # Extract H3 cell features
                    h3_features = self._extract_h3_features(value)
                    feature_values.append(h3_features)
                else:
                    feature_values.append([float(value)])
        features = np.array(feature_values, dtype=np.float32)
        # Apply normalization
        if config.normalization != NormalizationType.NONE:
            features = self._normalize_features(features, config.name, config.normalization)
        # Generate feature names
        if features.shape[1] == 1:
            names = [config.name]
        else:
            names = [f"{config.name}_{i}" for i in range(features.shape[1])]
        return features, names

    def _extract_h3_features(self, h3_cell: str) -> List[float]:
        """Extract features from H3 cell identifier"""
        try:
            # Get H3 resolution
            resolution = h3.get_resolution(h3_cell)
            # Get center coordinates
            lat, lng = h3.cell_to_latlng(h3_cell)
            # Get parent cells at different resolutions
            parent_cells = []
            for res in range(max(0, resolution - 2), resolution):
                parent = h3.cell_to_parent(h3_cell, res)
                parent_lat, parent_lng = h3.cell_to_latlng(parent)
                parent_cells.extend([parent_lat, parent_lng])
            features = [resolution, lat, lng] + parent_cells
            # Pad to fixed size
            while len(features) < 7:
                features.append(0.0)
            return features[:7]
        except Exception:
            logger.warning(f"Invalid H3 cell: {h3_cell}")
            return [0.0] * 7

    def _extract_temporal_features(
        self, nodes: List[Dict[str, Any]], config: FeatureConfig
    ) -> tuple[np.ndarray, list[str]]:
        """Extract temporal features like timestamps, durations, ages"""
        feature_values = []
        for node in nodes:
            value = node.get(config.name, config.default_value)
            if value is None:
                feature_values.append([0.0] * 7)  # Default temporal features
            else:
                if isinstance(value, (int, float)):
                    # Assume Unix timestamp
                    temporal_features = self._extract_timestamp_features(value)
                elif isinstance(value, str):
                    # Parse datetime string
                    try:
                        dt = datetime.fromisoformat(value)
                        timestamp = dt.timestamp()
                        temporal_features = self._extract_timestamp_features(timestamp)
                    except Exception:
                        temporal_features = [0.0] * 7
                elif isinstance(value, datetime):
                    timestamp = value.timestamp()
                    temporal_features = self._extract_timestamp_features(timestamp)
                else:
                    temporal_features = [0.0] * 7
                feature_values.append(temporal_features)
        features = np.array(feature_values, dtype=np.float32)
        # Apply normalization
        if config.normalization != NormalizationType.NONE:
            features = self._normalize_features(features, config.name, config.normalization)
        # Generate feature names
        names = [
            f"{config.name}_hour",
            f"{config.name}_day_of_week",
            f"{config.name}_day_of_month",
            f"{config.name}_month",
            f"{config.name}_year",
            f"{config.name}_is_weekend",
            f"{config.name}_timestamp_normalized",
        ]
        return features, names[: features.shape[1]]

    def _extract_timestamp_features(self, timestamp: float) -> List[float]:
        """Extract multiple features from a timestamp"""
        dt = datetime.fromtimestamp(timestamp)
        features = [
            dt.hour / 24.0,  # Normalized hour
            dt.weekday() / 6.0,  # Normalized day of week
            dt.day / 31.0,  # Normalized day of month
            dt.month / 12.0,  # Normalized month
            (dt.year - 2000) / 100.0,  # Normalized year
            float(dt.weekday() >= 5),  # Is weekend
            timestamp / 1e10,  # Normalized timestamp
        ]
        return features

    def _extract_categorical_features(
        self, nodes: List[Dict[str, Any]], config: FeatureConfig
    ) -> tuple[np.ndarray, list[str]]:
        """Extract categorical features with one-hot or label encoding"""
        values = []
        for node in nodes:
            value = node.get(config.name, config.default_value)
            if value is None:
                value = "unknown"
            values.append(str(value))
        # Fit encoder if not already fitted
        if config.name not in self.encoders:
            self.encoders[config.name] = LabelEncoder()
        try:
            # Try to transform with existing encoder
            encoded = self.encoders[config.name].transform(values)
        except ValueError:
            # Refit if new categories found
            self.encoders[config.name].fit(values)
            encoded = self.encoders[config.name].transform(values)
        # One-hot encode
        num_classes = len(self.encoders[config.name].classes_)
        one_hot = np.zeros((len(nodes), num_classes), dtype=np.float32)
        one_hot[np.arange(len(nodes)), encoded] = 1.0
        # Generate feature names
        names = [f"{config.name}_{cls}" for cls in self.encoders[config.name].classes_]
        return one_hot, names

    def _extract_numerical_features(
        self, nodes: List[Dict[str, Any]], config: FeatureConfig
    ) -> tuple[np.ndarray, list[str]]:
        """Extract numerical features with appropriate scaling"""
        values = []
        for node in nodes:
            value = node.get(config.name, config.default_value)
            if value is None:
                value = 0.0 if config.default_value is None else config.default_value
            # Handle different numerical formats
            if isinstance(value, (list, tuple)):
                values.append(list(value))
            else:
                values.append([float(value)])
        features = np.array(values, dtype=np.float32)
        # Apply constraints if specified
        if "min" in config.constraints:
            features = np.maximum(features, config.constraints["min"])
        if "max" in config.constraints:
            features = np.minimum(features, config.constraints["max"])
        # Apply normalization
        if config.normalization != NormalizationType.NONE:
            features = self._normalize_features(features, config.name, config.normalization)
        # Generate feature names
        if features.shape[1] == 1:
            names = [config.name]
        else:
            names = [f"{config.name}_{i}" for i in range(features.shape[1])]
        return features, names

    def _extract_embedding_features(
        self, nodes: List[Dict[str, Any]], config: FeatureConfig
    ) -> tuple[np.ndarray, list[str]]:
        """Extract pre-computed embeddings or generate new ones"""
        embeddings = []
        embedding_dim = config.dimension or 16
        for node in nodes:
            value = node.get(config.name, None)
            if value is None:
                # Generate random embedding or use zero embedding
                if config.default_value == "random":
                    embedding = np.random.randn(embedding_dim) * 0.1
                else:
                    embedding = np.zeros(embedding_dim)
            elif isinstance(value, (list, np.ndarray)):
                embedding = np.array(value)
                if len(embedding) != embedding_dim:
                    # Resize embedding if needed
                    if len(embedding) > embedding_dim:
                        embedding = embedding[:embedding_dim]
                    else:
                        padding = np.zeros(embedding_dim - len(embedding))
                        embedding = np.concatenate([embedding, padding])
            else:
                # Generate embedding from value (e.g., using hash)
                embedding = self._generate_embedding_from_value(value, embedding_dim)
            embeddings.append(embedding)
        features = np.array(embeddings, dtype=np.float32)
        # Normalize embeddings
        if config.normalization != NormalizationType.NONE:
            features = F.normalize(torch.from_numpy(features), p=2, dim=1).numpy()
        # Generate feature names
        names = [f"{config.name}_{i}" for i in range(embedding_dim)]
        return features, names

    def _generate_embedding_from_value(self, value: Any, dim: int) -> np.ndarray:
        """Generate embedding from arbitrary value using hashing"""
        # Convert value to string and hash
        str_value = str(value)
        hash_value = hash(str_value)
        # Use hash as seed for reproducible random embedding
        np.random.seed(abs(hash_value) % (2**32))
        embedding = np.random.randn(dim) * 0.1
        np.random.seed()  # Reset seed
        return embedding

    def _extract_text_features(
        self, nodes: List[Dict[str, Any]], config: FeatureConfig
    ) -> tuple[np.ndarray, list[str]]:
        """Extract features from text using TF-IDF or other methods"""
        texts = []
        for node in nodes:
            value = node.get(config.name, config.default_value)
            if value is None:
                texts.append("")
            else:
                texts.append(str(value))
        # Get or create vectorizer
        if config.name not in self.vectorizers:
            self.vectorizers[config.name] = TfidfVectorizer(
                max_features=config.dimension or 100, stop_words="english"
            )
        try:
            # Transform texts
            features = self.vectorizers[config.name].fit_transform(texts).toarray()
        except Exception:
            # Fallback to zero features
            features = np.zeros((len(nodes), config.dimension or 100))
        # Generate feature names
        try:
            vocab = self.vectorizers[config.name].get_feature_names_out()
            names = [f"{config.name}_{word}" for word in vocab]
        except Exception:
            names = [f"{config.name}_{i}" for i in range(features.shape[1])]
        return features.astype(np.float32), names

    def _extract_structural_features(
        self, nodes: List[Dict[str, Any]], config: FeatureConfig, graph: Optional[Any]
    ) -> tuple[np.ndarray, list[str]]:
        """Extract graph structural features like degree, centrality, etc"""
        if graph is None:
            # Return default features if no graph provided
            num_features = 5  # degree, in_degree, out_degree, clustering, pagerank
            features = np.zeros((len(nodes), num_features), dtype=np.float32)
            names = [
                f"{config.name}_degree",
                f"{config.name}_in_degree",
                f"{config.name}_out_degree",
                f"{config.name}_clustering",
                f"{config.name}_pagerank",
            ]
            return features, names
        # Extract structural features from graph
        # This is a placeholder - actual implementation would depend on graph
        # library
        structural_features: List[List[float]] = []
        for i, node in enumerate(nodes):
            node_id = node.get("id", i)
            # Example structural features
            structural_feats: List[float] = [
                float(graph.degree(node_id) if hasattr(graph, "degree") else 0),
                float(graph.in_degree(node_id) if hasattr(graph, "in_degree") else 0),
                float(graph.out_degree(node_id) if hasattr(graph, "out_degree") else 0),
                0.0,  # Clustering coefficient placeholder
                0.0,  # PageRank placeholder
            ]
            structural_features.append(structural_feats)
        features = np.array(structural_features, dtype=np.float32)
        # Normalize
        if config.normalization != NormalizationType.NONE:
            features = self._normalize_features(features, config.name, config.normalization)
        names = [
            f"{config.name}_degree",
            f"{config.name}_in_degree",
            f"{config.name}_out_degree",
            f"{config.name}_clustering",
            f"{config.name}_pagerank",
        ]
        return features, names

    def _normalize_features(
        self, features: np.ndarray, name: str, normalization: NormalizationType
    ) -> np.ndarray:
        """Apply normalization to features using Strategy pattern"""
        if normalization == NormalizationType.NONE:
            return features

        original_shape = features.shape
        features, is_single_feature = self._prepare_features_for_normalization(features)
        features = self._apply_normalization_strategy(features, name, normalization)
        features = self._restore_feature_shape(features, is_single_feature, original_shape)

        return features

    def _prepare_features_for_normalization(self, features: np.ndarray) -> tuple[np.ndarray, bool]:
        """Prepare features for normalization by handling dimensionality"""
        if features.ndim == 1:
            return features.reshape(-1, 1), True
        else:
            return features, False

    def _apply_normalization_strategy(
        self, features: np.ndarray, name: str, normalization: NormalizationType
    ) -> np.ndarray:
        """Apply specific normalization strategy"""
        normalization_strategies = {
            NormalizationType.LOG: self._apply_log_normalization,
            NormalizationType.STANDARD: lambda f, n: self._apply_scaler_normalization(
                f, n, StandardScaler
            ),
            NormalizationType.MINMAX: lambda f, n: self._apply_scaler_normalization(
                f, n, MinMaxScaler
            ),
            NormalizationType.ROBUST: lambda f, n: self._apply_scaler_normalization(
                f, n, RobustScaler
            ),
        }

        strategy = normalization_strategies.get(normalization)
        if strategy:
            return strategy(features, name)
        else:
            return features

    def _apply_log_normalization(self, features: np.ndarray, name: str) -> np.ndarray:
        """Apply log transformation normalization"""
        return np.log1p(np.abs(features))

    def _apply_scaler_normalization(
        self, features: np.ndarray, name: str, scaler_class
    ) -> np.ndarray:
        """Apply scikit-learn scaler normalization"""
        if name not in self.scalers:
            self.scalers[name] = scaler_class()

        try:
            return self.scalers[name].fit_transform(features)
        except Exception:
            logger.warning(f"Failed to normalize {name}, using raw values")
            return features

    def _restore_feature_shape(
        self, features: np.ndarray, is_single_feature: bool, original_shape: tuple
    ) -> np.ndarray:
        """Restore original feature shape after normalization"""
        if is_single_feature:
            return features.ravel()
        else:
            return features

    def handle_missing_data(
        self, features: np.ndarray, missing_mask: np.ndarray, strategy: str = "mean"
    ) -> np.ndarray:
        """
        Handle missing data in features.
        Args:
            features: Feature array with missing values
            missing_mask: Boolean mask indicating missing values
            strategy: Imputation strategy ("mean", "median", "zero", "forward_fill")
        Returns:
            Features with missing values imputed
        """
        if not np.any(missing_mask):
            return features
        features_imputed = features.copy()
        for feature_idx in range(features.shape[1]):
            missing_in_feature = missing_mask[:, feature_idx]
            if not np.any(missing_in_feature):
                continue
            if strategy == "mean":
                fill_value = np.mean(features[~missing_in_feature, feature_idx])
            elif strategy == "median":
                fill_value = np.median(features[~missing_in_feature, feature_idx])
            elif strategy == "zero":
                fill_value = 0.0
            elif strategy == "forward_fill":
                # Forward fill missing values
                last_valid = 0.0
                for i in range(len(features)):
                    if missing_in_feature[i]:
                        features_imputed[i, feature_idx] = last_valid
                    else:
                        last_valid = features_imputed[i, feature_idx]
                continue
            else:
                fill_value = 0.0
            features_imputed[missing_in_feature, feature_idx] = fill_value
        return features_imputed


# Example usage
if __name__ == "__main__":
    # Define feature configurations
    feature_configs = [
        FeatureConfig(
            name="position",
            type=FeatureType.SPATIAL,
            dimension=2,
            normalization=NormalizationType.MINMAX,
        ),
        FeatureConfig(
            name="timestamp",
            type=FeatureType.TEMPORAL,
            normalization=NormalizationType.STANDARD,
        ),
        FeatureConfig(name="status", type=FeatureType.CATEGORICAL),
        FeatureConfig(
            name="energy",
            type=FeatureType.NUMERICAL,
            normalization=NormalizationType.MINMAX,
            constraints={"min": 0, "max": 1},
        ),
        FeatureConfig(
            name="embedding",
            type=FeatureType.EMBEDDING,
            dimension=16,
            normalization=NormalizationType.NONE,
        ),
    ]
    # Create extractor
    extractor = NodeFeatureExtractor(feature_configs)
    # Example nodes
    nodes = [
        {
            "id": 1,
            "position": [0.5, 0.3],
            "timestamp": 1642000000,
            "status": "active",
            "energy": 0.8,
        },
        {
            "id": 2,
            "position": [0.7, 0.9],
            "timestamp": 1642001000,
            "status": "idle",
            "energy": 0.3,
        },
    ]
    # Extract features
    result = extractor.extract_features(nodes)
    print(f"Features shape: {result.features.shape}")
    print(f"Feature names: {result.feature_names[:10]}...")  # First 10
    print(f"Feature dimensions: {result.feature_dims}")


@dataclass
class LayerConfig:
    """Configuration for individual GNN layers"""

    in_channels: int
    out_channels: int
    heads: int = 1
    dropout: float = 0.0
    aggregation: AggregationType = AggregationType.MEAN
    activation: Optional[str] = "relu"
    layer_type: str = "GCN"


@dataclass
class GraphData:
    """Represents a single graph with node features and edge information"""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    num_nodes: int = 0
    num_edges: int = 0

    def __post_init__(self) -> None:
        if self.num_nodes == 0:
            self.num_nodes = self.node_features.size(0)
        if self.num_edges == 0:
            self.num_edges = self.edge_index.size(1)


@dataclass
class GraphBatch:
    """Batched graph data for efficient processing"""

    x: torch.Tensor  # Node features
    edge_index: torch.Tensor  # Edge indices
    batch: torch.Tensor  # Graph assignment for each node
    edge_attr: Optional[torch.Tensor] = None
    num_graphs: int = 0
    mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.num_graphs == 0:
            self.num_graphs = int(self.batch.max().item()) + 1


class GraphBatchProcessor:
    """Processes graphs into batches for training"""

    def __init__(
        self,
        use_torch_geometric: bool = True,
        pad_node_features: bool = False,
        max_nodes_per_graph: int = 100,
    ) -> None:
        """Initialize batch processor"""
        self.use_torch_geometric = use_torch_geometric
        self.pad_node_features = pad_node_features
        self.max_nodes_per_graph = max_nodes_per_graph

    def create_batch(self, graphs: List[GraphData]) -> GraphBatch:
        """Create a batch from list of graphs"""
        if not graphs:
            return GraphBatch(
                x=torch.empty(0, 0),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                batch=torch.empty(0, dtype=torch.long),
            )

        # Collect all node features and edge indices
        all_x = []
        all_edge_index = []
        batch_indices = []
        node_offset = 0

        for i, graph in enumerate(graphs):
            all_x.append(graph.node_features)

            # Adjust edge indices by node offset
            adjusted_edges = graph.edge_index + node_offset
            all_edge_index.append(adjusted_edges)

            # Create batch indices for this graph
            graph_batch = torch.full((graph.num_nodes,), i, dtype=torch.long)
            batch_indices.append(graph_batch)

            node_offset += graph.num_nodes

        # Concatenate all data
        x = torch.cat(all_x, dim=0)
        edge_index = torch.cat(all_edge_index, dim=1)
        batch = torch.cat(batch_indices, dim=0)

        return GraphBatch(x=x, edge_index=edge_index, batch=batch, num_graphs=len(graphs))

    def unbatch(self, batch: GraphBatch) -> List[GraphData]:
        """Unbatch a GraphBatch back into individual graphs"""
        graphs = []

        for i in range(batch.num_graphs):
            # Get nodes for this graph
            node_mask = batch.batch == i
            graph_x = batch.x[node_mask]

            # Get edges for this graph
            node_indices = torch.where(node_mask)[0]
            edge_mask = torch.isin(batch.edge_index[0], node_indices)
            graph_edge_index = batch.edge_index[:, edge_mask]

            # Reindex edges to start from 0
            node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
            reindexed_edges = torch.tensor(
                [
                    [node_mapping[edge[0].item()], node_mapping[edge[1].item()]]
                    for edge in graph_edge_index.t()
                ]
            ).t()

            graphs.append(
                GraphData(
                    node_features=graph_x,
                    edge_index=reindexed_edges,
                    num_nodes=graph_x.size(0),
                    num_edges=reindexed_edges.size(1),
                )
            )

        return graphs


class StreamingBatchProcessor(GraphBatchProcessor):
    """Streaming version of batch processor for large datasets"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.buffer: List[GraphData] = []
        self.buffer_size = kwargs.get("buffer_size", 32)

    def add_graph(self, graph: GraphData) -> Optional[GraphBatch]:
        """Add graph to buffer and return batch if buffer is full"""
        self.buffer.append(graph)

        if len(self.buffer) >= self.buffer_size:
            batch = self.create_batch(self.buffer)
            self.buffer.clear()
            return batch

        return None

    def flush(self) -> Optional[GraphBatch]:
        """Flush remaining graphs in buffer"""
        if self.buffer:
            batch = self.create_batch(self.buffer)
            self.buffer.clear()
            return batch
        return None


class GNNStack(torch.nn.Module):
    """Stack of GNN layers for graph neural network architectures"""

    def __init__(
        self, layer_configs: List[LayerConfig], layer_type: str = "GCN", global_pool: str = "mean"
    ) -> None:
        """
        Initialize GNN stack.

        Args:
            layer_configs: List of layer configurations
            layer_type: Type of GNN layer to use
            global_pool: Global pooling method
        """
        super().__init__()

        self.layer_configs = layer_configs
        self.layer_type = layer_type.upper()
        self.global_pool = global_pool

        # Build layers
        self.layers = torch.nn.ModuleList()
        for config in layer_configs:
            layer = self._create_layer(config)
            self.layers.append(layer)

        # Activation functions
        self.activations = torch.nn.ModuleList()
        for config in layer_configs:
            if config.activation == "relu":
                self.activations.append(torch.nn.ReLU())
            elif config.activation == "tanh":
                self.activations.append(torch.nn.Tanh())
            elif config.activation == "sigmoid":
                self.activations.append(torch.nn.Sigmoid())
            else:
                self.activations.append(torch.nn.Identity())

        # Dropout layers
        self.dropouts = torch.nn.ModuleList(
            [torch.nn.Dropout(config.dropout) for config in layer_configs]
        )

    def _create_layer(self, config: LayerConfig) -> torch.nn.Module:
        """Create a GNN layer based on configuration"""
        if self.layer_type == "GCN":
            return torch.nn.Linear(config.in_channels, config.out_channels)
        elif self.layer_type == "GAT":
            # For GAT, out_channels is usually heads * head_dim
            return torch.nn.Linear(config.in_channels, config.out_channels * config.heads)
        else:
            # Default to linear layer
            return torch.nn.Linear(config.in_channels, config.out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GNN stack.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment for nodes

        Returns:
            Graph-level representations
        """
        # Apply layers sequentially
        for i, (layer, activation, dropout) in enumerate(
            zip(self.layers, self.activations, self.dropouts)
        ):
            x = layer(x)
            x = activation(x)
            x = dropout(x)

        # Global pooling
        return self._global_pool(x, batch)

    def _global_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Apply global pooling to get graph-level representations"""
        num_graphs = int(batch.max().item()) + 1
        graph_embeddings = []

        for i in range(num_graphs):
            mask = batch == i
            graph_nodes = x[mask]

            if self.global_pool == "mean":
                graph_emb = graph_nodes.mean(dim=0)
            elif self.global_pool == "max":
                graph_emb = graph_nodes.max(dim=0)[0]
            elif self.global_pool == "sum":
                graph_emb = graph_nodes.sum(dim=0)
            else:
                # Default to mean
                graph_emb = graph_nodes.mean(dim=0)

            graph_embeddings.append(graph_emb)

        return torch.stack(graph_embeddings)

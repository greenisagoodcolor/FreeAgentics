"""H3 spatial integration for GNN feature extraction.

This module implements the H3 component of the critical innovation stack:
PyMDP + GMN + GNN + H3 + LLM

H3 provides hierarchical hexagonal spatial indexing for multi-resolution
spatial analysis and spatially-aware Active Inference.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    import h3

    H3_AVAILABLE = True
except ImportError:
    logger.warning("H3 library not available - spatial features will use fallback")
    H3_AVAILABLE = False


class H3SpatialProcessor:
    """H3-based spatial processing for GNN features."""

    def __init__(self, default_resolution: int = 7, max_resolution: int = 15):
        """Initialize the H3 spatial processor."""
        self.default_resolution = default_resolution
        self.max_resolution = max_resolution
        self.h3_cache: Dict[str, Optional[str]] = {}

    def latlng_to_h3(self, lat: float, lon: float,
        resolution: int = None) -> Optional[str]:
        """Convert lat/lng to H3 index with caching."""
        if not H3_AVAILABLE:
            return None

        resolution = resolution or self.default_resolution
        cache_key = f"{lat:.6f}_{lon:.6f}_{resolution}"

        if cache_key not in self.h3_cache:
            try:
                h3_index = h3.latlng_to_cell(lat, lon, resolution)
                self.h3_cache[cache_key] = h3_index
            except Exception as e:
                logger.warning(f"H3 conversion failed for {lat}, {lon}: {e}")
                return None

        return self.h3_cache[cache_key]

    def h3_to_latlng(self, h3_index: str) -> Optional[Tuple[float, float]]:
        """Convert H3 index to lat/lng center."""
        if not H3_AVAILABLE or not h3_index:
            return None

        try:
            result = h3.cell_to_latlng(h3_index)
            return tuple(result) if result else None
        except Exception as e:
            logger.warning(f"H3 to lat/lng conversion failed for {h3_index}: {e}")
            return None

    def get_h3_neighbors(self, h3_index: str, k: int = 1) -> List[str]:
        """Get k-ring neighbors of H3 cell."""
        if not H3_AVAILABLE or not h3_index:
            return []

        try:
            return list(h3.grid_disk(h3_index, k))
        except Exception as e:
            logger.warning(f"H3 neighbors failed for {h3_index}: {e}")
            return []

    def get_h3_distance(self, h3_index1: str, h3_index2: str) -> Optional[int]:
        """Get H3 grid distance between two cells."""
        if not H3_AVAILABLE or not h3_index1 or not h3_index2:
            return None

        try:
            distance = h3.grid_distance(h3_index1, h3_index2)
            return int(distance) if distance is not None else None
        except Exception as e:
            logger.warning(f"H3 distance failed for {h3_index1}, {h3_index2}: {e}")
            return None

    def adaptive_resolution(self, agent_density: float,
        observation_scale: float) -> int:
        """Calculate adaptive H3 resolution based on agent density and scale."""
        # Base resolution
        resolution = self.default_resolution

        # Adjust for agent density (more agents = higher resolution)
        if agent_density > 0.1:  # High density
            resolution = min(resolution + 2, self.max_resolution)
        elif agent_density > 0.01:  # Medium density
            resolution = min(resolution + 1, self.max_resolution)

        # Adjust for observation scale (smaller scale = higher resolution)
        if observation_scale < 0.1:  # Fine-grained observations
            resolution = min(resolution + 1, self.max_resolution)
        elif observation_scale > 1.0:  # Coarse observations
            resolution = max(resolution - 1, 1)

        return resolution

    def create_h3_spatial_graph(self, h3_indices: List[str],
        k: int = 1) -> Tuple[Tensor, Tensor]:
        """Create spatial graph edges based on H3 adjacency."""
        if not H3_AVAILABLE or not h3_indices:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0,
                dtype=torch.float32)

        # Remove None values and create index mapping
        valid_indices = [
            (i,
            h3_idx) for i,
            h3_idx in enumerate(h3_indices) if h3_idx is not None
        ]
        if not valid_indices:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0,
                dtype=torch.float32)

        edges = []
        edge_weights = []

        # Create adjacency matrix based on H3 proximity
        for i, (node_i, h3_i) in enumerate(valid_indices):
            for j, (node_j, h3_j) in enumerate(valid_indices):
                if i != j:
                    distance = self.get_h3_distance(h3_i, h3_j)
                    if distance is not None and distance <= k:
                        edges.append([node_i, node_j])
                        # Weight by inverse distance (closer = stronger connection)
                        weight = 1.0 / (distance + 1.0)
                        edge_weights.append(weight)

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0, dtype=torch.float32)

        return edge_index, edge_weights


class H3MultiResolutionAnalyzer:
    """Multi-resolution spatial analysis using H3."""

    def __init__(self, resolutions: List[int] = None):
        """Initialize the H3 multi-resolution analyzer."""
        self.resolutions = resolutions or [5, 7, 9, 11]  # Multiple scales
        self.processor = H3SpatialProcessor()

    def extract_multi_resolution_features(
        self, positions: List[Tuple[float, float]]
    ) -> Dict[str, Tensor]:
        """Extract features at multiple H3 resolutions."""
        if not H3_AVAILABLE:
            return {"fallback_features": torch.zeros(len(positions), 2)}

        features = {}

        for resolution in self.resolutions:
            res_features = []
            h3_indices = []

            for lat, lon in positions:
                h3_index = self.processor.latlng_to_h3(lat, lon, resolution)
                h3_indices.append(h3_index)

                if h3_index:
                    # H3 index as numerical feature (hash to fixed-size vector)
                    h3_hash = hash(h3_index) % 1000000  # Normalize to reasonable range
                    center_lat, center_lon = self.processor.h3_to_latlng(h3_index)
                    res_features.append([center_lat, center_lon, h3_hash / 1000000.0])
                else:
                    res_features.append([lat, lon, 0.0])

            features[f"resolution_{resolution}"] = torch.tensor(res_features,
                dtype=torch.float32)
            features[f"h3_indices_{resolution}"] = h3_indices

        return features

    def compute_spatial_relationships(self, h3_indices: List[str]) -> Dict[str, Any]:
        """Compute spatial relationships between H3 cells."""
        if not H3_AVAILABLE or not h3_indices:
            return {}

        relationships: Dict[str, Any] = {
            "adjacency_matrix": [],
            "distance_matrix": [],
            "neighbor_counts": [],
            "cluster_info": {},
        }

        valid_indices = [idx for idx in h3_indices if idx is not None]
        n = len(valid_indices)

        if n == 0:
            return relationships

        # Compute distance matrix
        distance_matrix = np.zeros((n, n))
        adjacency_matrix = np.zeros((n, n))

        for i, h3_i in enumerate(valid_indices):
            neighbors = self.processor.get_h3_neighbors(h3_i, k=1)
            neighbor_count = len(neighbors)
            relationships["neighbor_counts"].append(neighbor_count)

            for j, h3_j in enumerate(valid_indices):
                if i != j:
                    distance = self.processor.get_h3_distance(h3_i, h3_j)
                    if distance is not None:
                        distance_matrix[i, j] = distance
                        if distance == 1:  # Adjacent cells
                            adjacency_matrix[i, j] = 1

        relationships["adjacency_matrix"] = adjacency_matrix
        relationships["distance_matrix"] = distance_matrix

        # Identify spatial clusters
        relationships["cluster_info"] = self._identify_h3_clusters(valid_indices)

        return relationships

    def _identify_h3_clusters(self, h3_indices: List[str]) -> Dict[str, Any]:
        """Identify spatial clusters using H3 adjacency."""
        if not h3_indices:
            return {}

        clusters = []
        visited = set()

        def dfs_cluster(h3_index: str, current_cluster: Set[str]):
            if h3_index in visited or h3_index not in h3_indices:
                return

            visited.add(h3_index)
            current_cluster.add(h3_index)

            # Add adjacent cells to cluster
            neighbors = self.processor.get_h3_neighbors(h3_index, k=1)
            for neighbor in neighbors:
                if neighbor in h3_indices and neighbor not in visited:
                    dfs_cluster(neighbor, current_cluster)

        # Find all clusters
        for h3_index in h3_indices:
            if h3_index not in visited:
                cluster: Set[str] = set()
                dfs_cluster(h3_index, cluster)
                if cluster:
                    clusters.append(list(cluster))

        return {
            "num_clusters": len(clusters),
            "clusters": clusters,
            "largest_cluster_size": max(len(c) for c in clusters) if clusters else 0,
            "average_cluster_size": (
                sum(len(c) for c in clusters) / len(clusters) if clusters else 0
            ),
        }


class GNNSpatialIntegration:
    """Integration layer between GNN and H3 spatial processing."""

    def __init__(self, default_resolution: int = 7):
        """Initialize the GNN spatial integration."""
        self.h3_processor = H3SpatialProcessor(default_resolution)
        self.multi_res_analyzer = H3MultiResolutionAnalyzer()

    def create_spatial_aware_features(self, nodes: List[Dict[str, Any]]) -> Dict[str,
        Tensor]:
        """Create spatially-aware features for GNN processing."""
        # Extract positions
        positions = []
        for node in nodes:
            if "position" in node:
                pos = node["position"]
                if isinstance(pos, dict) and "lat" in pos and "lon" in pos:
                    positions.append((pos["lat"], pos["lon"]))
                elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    # Assume list coordinates are lat/lon
                    positions.append((pos[0], pos[1]))
                else:
                    positions.append((0.0, 0.0))
            else:
                positions.append((0.0, 0.0))

        if not positions:
            return {"empty_features": torch.zeros(0, 2)}

        # Multi-resolution H3 features
        multi_res_features = self.multi_res_analyzer.extract_multi_resolution_features(positions)

        # Spatial graph construction
        if H3_AVAILABLE and "h3_indices_7" in multi_res_features:
            h3_indices = multi_res_features["h3_indices_7"]
            (
                edge_index,
                edge_weights,
            ) = self.h3_processor.create_h3_spatial_graph(h3_indices)
            multi_res_features["spatial_edge_index"] = edge_index
            multi_res_features["spatial_edge_weights"] = edge_weights

        # Spatial relationships
        if H3_AVAILABLE and "h3_indices_7" in multi_res_features:
            relationships = self.multi_res_analyzer.compute_spatial_relationships(
                multi_res_features["h3_indices_7"]
            )
            multi_res_features.update(relationships)

        return multi_res_features

    def adaptive_spatial_resolution(self, agents: List[Any],
        observation_scale: float) -> int:
        """Calculate adaptive spatial resolution for agent processing."""
        if not agents:
            return self.h3_processor.default_resolution

        # Estimate agent density (simplified)
        agent_density = len(agents) / 1000.0  # Normalized estimate

        return self.h3_processor.adaptive_resolution(agent_density, observation_scale)


# Global H3 integration instance
h3_spatial_integration = GNNSpatialIntegration()


def integrate_h3_with_active_inference(
    agent, spatial_features: Dict[str, Tensor]
) -> Dict[str, Any]:
    """Integrate H3 spatial features with Active Inference belief updates."""
    if not hasattr(agent, "pymdp_agent") or agent.pymdp_agent is None:
        return {}

    integration_results = {}

    # Use H3 spatial structure to inform PyMDP state space
    if (
        "spatial_edge_index" in spatial_features
        and spatial_features["spatial_edge_index"].numel() > 0
    ):
        edge_index = spatial_features["spatial_edge_index"]
        edge_weights = spatial_features.get("spatial_edge_weights",
            torch.ones(edge_index.size(1)))

        # Create spatial adjacency matrix for PyMDP
        num_nodes = max(edge_index.max().item() + 1, 1)
        spatial_adjacency = torch.zeros(num_nodes, num_nodes)

        for i, (src, dst) in enumerate(edge_index.t()):
            spatial_adjacency[src, dst] = edge_weights[i]

        integration_results["spatial_adjacency"] = spatial_adjacency
        integration_results["spatial_connectivity"] = torch.sum(spatial_adjacency > 0).item()

    # Multi-resolution spatial context for belief priors
    if "resolution_7" in spatial_features:
        spatial_context = spatial_features["resolution_7"]
        if spatial_context.numel() > 0:
            # Compute spatial variance as uncertainty measure
            spatial_variance = torch.var(spatial_context, dim=0)
            integration_results["spatial_uncertainty"] = spatial_variance.mean().item()

    return integration_results

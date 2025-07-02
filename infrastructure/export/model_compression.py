"""
Model Compression Utilities

Implements various compression techniques to reduce model size for edge deployment.
"""

import json
import logging
import struct
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression level presets"""

    NONE = "none"
    LIGHT = "light"  # ~20% reduction
    MEDIUM = "medium"  # ~40% reduction
    AGGRESSIVE = "aggressive"  # ~60% reduction
    EXTREME = "extreme"  # ~80% reduction, may impact performance


@dataclass
class CompressionStats:
    """Statistics about compression results"""

    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    parameters_quantized: int
    nodes_pruned: int
    edges_pruned: int
    precision_loss: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_ratio": self.compression_ratio,
            "reduction_percent": (1 - 1 / self.compression_ratio) * 100,
            "parameters_quantized": self.parameters_quantized,
            "nodes_pruned": self.nodes_pruned,
            "edges_pruned": self.edges_pruned,
            "precision_loss": self.precision_loss,
        }


class ModelCompressor:
    """
    Compresses GNN models for edge deployment.

    Techniques:
    - Quantization (float32 -> int8/int16)
    - Pruning (remove low-importance nodes/edges)
    - Weight sharing
    - Sparse representation
    """

    def __init__(self) -> None:
        """Initialize compressor"""
        self.compression_configs = {
            CompressionLevel.NONE: {
                "quantize_bits": 32,
                "prune_threshold": 0.0,
                "weight_sharing": False,
                "sparse_threshold": 0.0,
            },
            CompressionLevel.LIGHT: {
                "quantize_bits": 16,
                "prune_threshold": 0.05,
                "weight_sharing": False,
                "sparse_threshold": 0.1,
            },
            CompressionLevel.MEDIUM: {
                "quantize_bits": 8,
                "prune_threshold": 0.1,
                "weight_sharing": True,
                "sparse_threshold": 0.2,
            },
            CompressionLevel.AGGRESSIVE: {
                "quantize_bits": 8,
                "prune_threshold": 0.2,
                "weight_sharing": True,
                "sparse_threshold": 0.3,
            },
            CompressionLevel.EXTREME: {
                "quantize_bits": 4,
                "prune_threshold": 0.3,
                "weight_sharing": True,
                "sparse_threshold": 0.4,
            },
        }

    def compress_model(
        self,
        model_data: Dict[str, Any],
        level: CompressionLevel = CompressionLevel.MEDIUM,
    ) -> tuple[dict[str, Any], CompressionStats]:
        """
        Compress GNN model based on compression level.

        Args:
            model_data: Original model data
            level: Compression level to apply

        Returns:
            Tuple of (compressed_model, stats)
        """
        logger.info(f"Compressing model with level: {level.value}")

        # Get original size
        original_size = len(json.dumps(model_data).encode())

        # Get compression config
        config = self.compression_configs[level]

        # Deep copy model to avoid modifying original
        compressed = json.loads(json.dumps(model_data))

        # Track statistics
        params_quantized = 0
        nodes_pruned = 0
        edges_pruned = 0
        precision_losses = []

        # Apply compression techniques
        if "gnn_model" in compressed:
            gnn = compressed["gnn_model"]

            # 1. Quantize parameters
            if config["quantize_bits"] < 32:
                quant_stats = self._quantize_parameters(gnn, config["quantize_bits"])
                params_quantized = quant_stats["count"]
                precision_losses.append(quant_stats["precision_loss"])

            # 2. Prune low-importance nodes/edges
            if config["prune_threshold"] > 0:
                prune_stats = self._prune_graph(gnn, config["prune_threshold"])
                nodes_pruned = prune_stats["nodes_pruned"]
                edges_pruned = prune_stats["edges_pruned"]

            # 3. Apply weight sharing
            if config["weight_sharing"]:
                self._apply_weight_sharing(gnn)

            # 4. Convert to sparse representation
            if config["sparse_threshold"] > 0:
                self._convert_to_sparse(gnn, config["sparse_threshold"])

        # Calculate compressed size
        compressed_size = len(json.dumps(compressed).encode())

        # Create statistics
        stats = CompressionStats(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=original_size / compressed_size,
            parameters_quantized=params_quantized,
            nodes_pruned=nodes_pruned,
            edges_pruned=edges_pruned,
            precision_loss=(np.mean(precision_losses) if precision_losses else 0.0,),
        )

        logger.info(f"Compression complete: {stats.compression_ratio:.2f}x reduction")

        return compressed, stats

    def _quantize_parameters(self, gnn: Dict[str, Any], bits: int) -> Dict[str, Any]:
        """Quantize floating point parameters to reduce precision"""
        stats = {"count": 0, "precision_loss": 0.0}
        losses = []

        # Quantize node parameters
        if "nodes" in gnn:
            for node in gnn["nodes"]:
                if "parameters" in node:
                    for key, value in node["parameters"].items():
                        if isinstance(value, float):
                            quantized = self._quantize_value(value, bits)
                            loss = abs(value - quantized)
                            losses.append(loss)
                            node["parameters"][key] = quantized
                            stats["count"] += 1

        # Quantize edge parameters
        if "edges" in gnn:
            for edge in gnn["edges"]:
                if "weight" in edge:
                    original = edge["weight"]
                    edge["weight"] = self._quantize_value(original, bits)
                    losses.append(abs(original - edge["weight"]))
                    stats["count"] += 1

        stats["precision_loss"] = np.mean(losses) if losses else 0.0
        return stats

    def _quantize_value(self, value: float, bits: int) -> float:
        """Quantize a single value to specified bit precision"""
        if bits == 32:
            return value
        elif bits == 16:
            # Convert to half precision
            return float(np.float16(value))
        elif bits == 8:
            # Quantize to 8-bit range [-1, 1]
            scale = 127.0
            quantized = round(value * scale) / scale
            return max(-1.0, min(1.0, quantized))
        elif bits == 4:
            # Quantize to 4-bit range (16 levels)
            scale = 7.0
            quantized = round(value * scale) / scale
            return max(-1.0, min(1.0, quantized))
        else:
            return value

    def _prune_graph(self, gnn: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Prune low-importance nodes and edges"""
        stats = {"nodes_pruned": 0, "edges_pruned": 0}

        # Calculate importance scores for nodes
        node_importance = {}
        if "nodes" in gnn:
            for node in gnn["nodes"]:
                # Simple importance based on parameter magnitude
                importance = 0.0
                if "parameters" in node:
                    values = [abs(v) for v in node["parameters"].values()
                              if isinstance(v, (int, float))]
                    importance = np.mean(values) if values else 0.0
                node_importance[node["id"]] = importance

            # Prune nodes below threshold
            original_count = len(gnn["nodes"])
            gnn["nodes"] = [
                node for node in gnn["nodes"] if node_importance.get(
                    node["id"], 0) >= threshold]
            stats["nodes_pruned"] = original_count - len(gnn["nodes"])

            # Get remaining node IDs
            remaining_nodes = {node["id"] for node in gnn["nodes"]}

        # Prune edges with low weights or disconnected nodes
        if "edges" in gnn:
            original_count = len(gnn["edges"])
            gnn["edges"] = [
                edge
                for edge in gnn["edges"]
                if (
                    abs(edge.get("weight", 1.0)) >= threshold
                    and edge.get("source") in remaining_nodes
                    and edge.get("target") in remaining_nodes
                )
            ]
            stats["edges_pruned"] = original_count - len(gnn["edges"])

        return stats

    def _apply_weight_sharing(self, gnn: Dict[str, Any]):
        """Apply weight sharing to reduce unique parameter values"""
        # Collect all weight values
        all_weights = []

        if "edges" in gnn:
            for edge in gnn["edges"]:
                if "weight" in edge:
                    all_weights.append(edge["weight"])

        if not all_weights:
            return

        # Cluster weights into buckets
        num_buckets = min(16, len(set(all_weights)))
        if num_buckets < 2:
            return

        # Use k-means style clustering
        weights_array = np.array(all_weights)
        min_weight = weights_array.min()
        max_weight = weights_array.max()

        # Create buckets
        buckets = np.linspace(min_weight, max_weight, num_buckets)

        # Create shared weight table
        shared_weights = {}
        for i, bucket_center in enumerate(buckets):
            shared_weights[i] = float(bucket_center)

        # Map weights to nearest bucket
        if "edges" in gnn:
            for edge in gnn["edges"]:
                if "weight" in edge:
                    # Find nearest bucket
                    distances = [abs(edge["weight"] - b) for b in buckets]
                    nearest_idx = np.argmin(distances)
                    edge["weight_idx"] = nearest_idx
                    edge["weight"] = shared_weights[nearest_idx]

        # Store shared weight table
        gnn["shared_weights"] = shared_weights

    def _convert_to_sparse(self, gnn: Dict[str, Any], threshold: float):
        """Convert to sparse representation for values near zero"""
        # For edges, store only non-zero weights
        if "edges" in gnn:
            for edge in gnn["edges"]:
                if "weight" in edge and abs(edge["weight"]) < threshold:
                    edge["weight"] = 0.0

        # For node parameters, use sparse encoding
        if "nodes" in gnn:
            for node in gnn["nodes"]:
                if "parameters" in node:
                    sparse_params = {}
                    for key, value in node["parameters"].items():
                        if isinstance(value, (int, float)) and abs(value) >= threshold:
                            sparse_params[key] = value
                    node["parameters"] = sparse_params

    def compress_binary(self, model_data: Dict[str, Any]) -> bytes:
        """
        Compress model to binary format for maximum space savings.

        Args:
            model_data: Model dictionary

        Returns:
            Compressed binary data
        """
        # Convert to JSON and compress with zlib
        json_str = json.dumps(model_data, separators=(",", ":"))
        compressed = zlib.compress(json_str.encode("utf-8"), level=9)

        # Add header with version and compression info
        header = struct.pack("!HH", 1, len(json_str))  # version, original_size

        return header + compressed

    def decompress_binary(self, data: bytes) -> Dict[str, Any]:
        """
        Decompress binary model data.

        Args:
            data: Compressed binary data

        Returns:
            Model dictionary
        """
        # Read header
        version, original_size = struct.unpack("!HH", data[:4])

        if version != 1:
            raise ValueError(f"Unsupported binary format version: {version}")

        # Decompress
        decompressed = zlib.decompress(data[4:])

        # Parse JSON
        return json.loads(decompressed.decode("utf-8"))

    def estimate_runtime_memory(self, model_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate runtime memory requirements.

        Args:
            model_data: Model dictionary

        Returns:
            Memory estimates in MB
        """
        estimates = {
            "model_base": 0.0,
            "inference_buffer": 0.0,
            "knowledge_graph": 0.0,
            "total": 0.0,
        }

        # Estimate model size
        if "gnn_model" in model_data:
            gnn = model_data["gnn_model"]

            # Count parameters
            num_params = 0
            if "nodes" in gnn:
                for node in gnn["nodes"]:
                    if "parameters" in node:
                        num_params += len(node["parameters"])

            if "edges" in gnn:
                num_params += len(gnn["edges"])

            # Assume 4 bytes per parameter in memory
            estimates["model_base"] = (num_params * 4) / (1024 * 1024)

        # Inference buffer (typically 2-3x model size)
        estimates["inference_buffer"] = estimates["model_base"] * 2.5

        # Knowledge graph estimate
        if "knowledge_size" in model_data:
            estimates["knowledge_graph"] = model_data["knowledge_size"] / (1024 * 1024)
        else:
            estimates["knowledge_graph"] = 50.0  # Default 50MB

        # Total
        estimates["total"] = sum(estimates.values())

        return estimates

    def validate_compressed_model(
        self, original: Dict[str, Any], compressed: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate compressed model maintains essential structure.

        Args:
            original: Original model
            compressed: Compressed model

        Returns:
            Validation results
        """
        results = {
            "structure_intact": True,
            "critical_nodes_present": True,
            "edge_connectivity": True,
            "parameters_valid": True,
        }

        try:
            # Check basic structure
            if "gnn_model" not in compressed:
                results["structure_intact"] = False
                return results

            orig_gnn = original.get("gnn_model", {})
            comp_gnn = compressed["gnn_model"]

            # Check critical nodes (e.g., input/output nodes)
            critical_node_types = {"input", "output", "memory"}
            orig_critical = {
                n["id"] for n in orig_gnn.get(
                    "nodes",
                    []) if n.get("type") in critical_node_types}
            comp_critical = {
                n["id"] for n in comp_gnn.get(
                    "nodes",
                    []) if n.get("type") in critical_node_types}

            if orig_critical != comp_critical:
                results["critical_nodes_present"] = False

            # Check edge connectivity
            comp_nodes = {n["id"] for n in comp_gnn.get("nodes", [])}
            for edge in comp_gnn.get("edges", []):
                if edge.get("source") not in comp_nodes or edge.get(
                        "target") not in comp_nodes:
                    results["edge_connectivity"] = False
                    break

            # Validate parameters
            for node in comp_gnn.get("nodes", []):
                if "parameters" in node:
                    for key, value in node["parameters"].items():
                        if not isinstance(value, (int, float, str, bool, type(None))):
                            results["parameters_valid"] = False
                            break

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {k: False for k in results}

        return results

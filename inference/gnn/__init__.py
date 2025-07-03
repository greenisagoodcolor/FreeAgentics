"""
GNN (Graph Neural Network) inference module for FreeAgentics.

This module provides graph neural network implementations for agent
belief updating, coalition analysis, and spatial reasoning.
"""

from .executor import GMNExecutor
from .feature_extractor import (
    ExtractionResult,
    FeatureConfig,
    FeatureType,
    NodeFeatureExtractor,
    NormalizationType,
)
from .layers import AggregationType, GATLayer, GCNLayer, LayerConfig
from .model import GMNModel, GNNModel
from .parser import GMNParser
from .validator import GMNValidator

__all__ = [
    "AggregationType",
    "GCNLayer",
    "GATLayer",
    "LayerConfig",
    "FeatureType",
    "NormalizationType",
    "FeatureConfig",
    "ExtractionResult",
    "NodeFeatureExtractor",
    "GMNExecutor",
    "GMNModel",
    "GNNModel",
    "GMNParser",
    "GMNValidator",
]

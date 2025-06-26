"""
Spatial Computing Module for FreeAgentics
This module provides comprehensive spatial operations using H3 hexagonal grid system.
Includes pathfinding, resource distribution, visibility calculations, and world modeling.
"""

from .spatial_api import (
    ObservationModel,
    ResourceDistribution,
    ResourceType,
    SpatialAPI,
    SpatialCoordinate,
)

__all__ = [
    "SpatialAPI",
    "SpatialCoordinate",
    "ResourceType",
    "ResourceDistribution",
    "ObservationModel",
]

__version__ = "1.0.0"

"""Coalition formation module for FreeAgentics.

This module implements coalition formation algorithms for multi-agent systems
using Active Inference principles.
"""

from coalitions.coalition import Coalition
from coalitions.coalition_manager import CoalitionManager
from coalitions.formation_strategies import (
    GreedyFormation,
    HierarchicalFormation,
    OptimalFormation,
)

__all__ = [
    "CoalitionManager",
    "Coalition",
    "GreedyFormation",
    "OptimalFormation",
    "HierarchicalFormation",
]

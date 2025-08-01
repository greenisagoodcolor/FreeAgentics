"""Active Inference components for FreeAgentics.

This package implements PyMDP-based active inference system for agent belief
state management and policy selection, as specified in Task 44.
"""

from inference.active.belief_manager import (
    BeliefState,
    BeliefStateManager,
    BeliefStateRepository,
    BeliefUpdateResult,
    InMemoryBeliefRepository,
)
from inference.active.config import ActiveInferenceConfig, PyMDPSetup
from inference.active.gmn_parser import GMNParser, parse_gmn_spec

__all__ = [
    "GMNParser",
    "parse_gmn_spec",
    "ActiveInferenceConfig",
    "PyMDPSetup",
    "BeliefState",
    "BeliefStateManager",
    "BeliefUpdateResult",
    "BeliefStateRepository",
    "InMemoryBeliefRepository",
]

"""
Test fixtures and factories for FreeAgentics test suite.

This module provides centralized access to all test data factories and fixtures.
"""

from .mock_factory import MockFactory
from .test_data_factory import (
    DataFactory,
    create_active_inference_state,
    create_agent,
    create_agent_batch,
    create_coalition,
    create_gnn_graph_data,
    create_test_scenario,
    create_world_cell,
)

# Export all factories
__all__ = [
    "MockFactory",
    "DataFactory",
    "create_agent",
    "create_agent_batch",
    "create_coalition",
    "create_world_cell",
    "create_active_inference_state",
    "create_gnn_graph_data",
    "create_test_scenario",
]

# Create global instances
mock_factory = MockFactory()
test_data_factory = DataFactory()

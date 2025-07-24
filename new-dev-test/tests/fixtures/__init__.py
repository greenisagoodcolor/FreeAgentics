"""Test Data Management and Fixtures System.

This module provides comprehensive test data management with:
- Factory functions with Builder pattern
- Type-safe data generation
- Schema validation with Pydantic
- Reusable pytest fixtures
- Performance test data generators
"""

from .builders import AgentBuilder, CoalitionBuilder, KnowledgeEdgeBuilder, KnowledgeNodeBuilder
from .factories import AgentFactory, CoalitionFactory, KnowledgeGraphFactory, PerformanceDataFactory
from .fixtures import (
    agent_fixture,
    clean_database,
    coalition_fixture,
    db_session,
    knowledge_graph_fixture,
)
from .generators import (
    generate_agent_batch,
    generate_coalition_scenario,
    generate_knowledge_graph,
    generate_performance_dataset,
)
from .schemas import AgentSchema, CoalitionSchema, KnowledgeEdgeSchema, KnowledgeNodeSchema

__all__ = [
    # Builders
    "AgentBuilder",
    "CoalitionBuilder",
    "KnowledgeNodeBuilder",
    "KnowledgeEdgeBuilder",
    # Factories
    "AgentFactory",
    "CoalitionFactory",
    "KnowledgeGraphFactory",
    "PerformanceDataFactory",
    # Fixtures
    "agent_fixture",
    "coalition_fixture",
    "knowledge_graph_fixture",
    "db_session",
    "clean_database",
    # Schemas
    "AgentSchema",
    "CoalitionSchema",
    "KnowledgeNodeSchema",
    "KnowledgeEdgeSchema",
    # Generators
    "generate_agent_batch",
    "generate_coalition_scenario",
    "generate_knowledge_graph",
    "generate_performance_dataset",
]

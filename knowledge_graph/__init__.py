"""Knowledge Graph Evolution system for FreeAgentics.

This module provides a dynamic knowledge graph system that allows agents
to build, query, and evolve their understanding of the world.
"""

from knowledge_graph.evolution import EvolutionEngine, MutationOperator
from knowledge_graph.graph_engine import (
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
)
from knowledge_graph.query import GraphQuery, QueryResult

__all__ = [
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "EvolutionEngine",
    "MutationOperator",
    "GraphQuery",
    "QueryResult",
]

"""Knowledge Graph Evolution system for FreeAgentics.

This module provides a dynamic knowledge graph system that allows agents
to build, query, and evolve their understanding of the world.
"""

from knowledge_graph.evolution import EvolutionEngine, MutationOperator
from knowledge_graph.extraction import (
    ConversationExtractor,
    ConversationMessage,
    ContextAwareExtractor,
    CoReferenceResolver,
    EntityExtractionStrategy,
    ExtractionContext,
    ExtractionPipeline,
    ExtractionResult,
    LLMFallbackExtractor,
    PatternRelationExtractor,
    RelationExtractionStrategy,
    SpacyEntityExtractor,
)
from knowledge_graph.graph_engine import KnowledgeEdge, KnowledgeGraph, KnowledgeNode
from knowledge_graph.query import GraphQuery, QueryResult
from knowledge_graph.schema import (
    ConflictResolutionStrategy,
    ConversationEntity,
    ConversationOntology,
    ConversationRelation,
    EntityType,
    KnowledgeGraphSchema,
    Provenance,
    RelationType,
    SchemaEvolutionManager,
    SchemaValidator,
    TemporalMetadata,
)

__all__ = [
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "EvolutionEngine",
    "MutationOperator",
    "GraphQuery",
    "QueryResult",
    "ConversationEntity",
    "ConversationOntology",
    "ConversationRelation",
    "ConflictResolutionStrategy",
    "EntityType",
    "KnowledgeGraphSchema",
    "Provenance",
    "RelationType",
    "SchemaEvolutionManager",
    "SchemaValidator",
    "TemporalMetadata",
    # Extraction pipeline components
    "ConversationExtractor",
    "ConversationMessage",
    "ContextAwareExtractor",
    "CoReferenceResolver",
    "EntityExtractionStrategy",
    "ExtractionContext",
    "ExtractionPipeline",
    "ExtractionResult",
    "LLMFallbackExtractor",
    "PatternRelationExtractor",
    "RelationExtractionStrategy",
    "SpacyEntityExtractor",
]

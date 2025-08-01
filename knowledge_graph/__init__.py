"""Knowledge Graph Evolution system for FreeAgentics.

This module provides a dynamic knowledge graph system that allows agents
to build, query, and evolve their understanding of the world.
"""

from knowledge_graph.evolution import EvolutionEngine, MutationOperator
from knowledge_graph.extraction import (
    ContextAwareExtractor,
    ConversationExtractor,
    ConversationMessage,
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

# Task 34.5 - Graph Query API
from knowledge_graph.query_api import (
    GraphQueryEngine,
    InMemoryQueryCache,
    QueryComplexityAnalyzer,
    QueryFilter,
    QueryOptions,
)
from knowledge_graph.query_api import QueryResult as QueryAPIResult
from knowledge_graph.query_api import QueryType, RedisQueryCache

# Task 34.4 - Real-time Graph Updates
from knowledge_graph.realtime_updater import (
    DefaultConflictResolver,
    GraphUpdateEvent,
    RealtimeGraphUpdater,
    UpdateEventType,
    WebSocketEventStreamer,
)
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
    # Core graph components
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "EvolutionEngine",
    "MutationOperator",
    "GraphQuery",
    "QueryResult",
    # Schema and ontology
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
    # Task 34.4 - Real-time updates
    "DefaultConflictResolver",
    "GraphUpdateEvent",
    "RealtimeGraphUpdater",
    "UpdateEventType",
    "WebSocketEventStreamer",
    # Task 34.5 - Query API
    "GraphQueryEngine",
    "InMemoryQueryCache",
    "QueryComplexityAnalyzer",
    "QueryFilter",
    "QueryOptions",
    "QueryAPIResult",
    "QueryType",
    "RedisQueryCache",
]

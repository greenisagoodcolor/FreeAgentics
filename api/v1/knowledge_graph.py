"""Production Knowledge Graph API (Task 34.5).

This module provides both REST and GraphQL interfaces for querying knowledge graphs
with semantic search, path-finding, caching, and comprehensive monitoring.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from auth.security_implementation import TokenData, get_current_user
from knowledge_graph.graph_engine import KnowledgeGraph
from knowledge_graph.query_api import (
    GraphQueryEngine,
    InMemoryQueryCache,
    QueryFilter,
    QueryOptions,
    QueryType,
    SortOrder,
)
from knowledge_graph.schema import EntityType, RelationType
from observability.prometheus_metrics import PrometheusMetricsCollector as PrometheusMetrics

logger = logging.getLogger(__name__)

router = APIRouter()

# Global query engine instance (in production, this would be dependency-injected)
_query_engine: Optional[GraphQueryEngine] = None
_metrics = PrometheusMetrics()


def get_query_engine() -> GraphQueryEngine:
    """Get or create query engine instance."""
    global _query_engine
    if _query_engine is None:
        # In production, this would use the actual knowledge graph
        knowledge_graph = KnowledgeGraph("production")
        cache = InMemoryQueryCache(max_size=1000)
        _query_engine = GraphQueryEngine(knowledge_graph, cache)
    return _query_engine


class KnowledgeNode(BaseModel):
    """Knowledge graph node for API responses."""

    id: str
    label: str
    type: str = "concept"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: Dict[str, Any] = {}
    metadata: Optional[Dict[str, Any]] = None
    similarity_score: Optional[float] = None
    distance_from_center: Optional[int] = None


class KnowledgeEdge(BaseModel):
    """Knowledge graph edge for API responses."""

    id: str
    source_id: str = Field(..., alias="source")
    target_id: str = Field(..., alias="target")
    type: str = "relates_to"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: Dict[str, Any] = {}
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeGraphResponse(BaseModel):
    """Comprehensive knowledge graph response."""

    nodes: List[KnowledgeNode]
    edges: List[KnowledgeEdge]
    paths: List[List[str]] = []
    metadata: Dict[str, Any] = {}
    query_info: Dict[str, Any] = {}

    class Config:
        """Pydantic configuration for field aliasing."""

        allow_population_by_field_name = True


class QueryResultResponse(BaseModel):
    """Response for query operations."""

    success: bool = True
    data: KnowledgeGraphResponse
    execution_time_ms: float
    cache_hit: bool = False
    total_count: int = 0
    has_more: bool = False
    query_id: str


# REST API Endpoints


@router.get("/nodes", response_model=QueryResultResponse)
async def get_nodes(
    node_ids: Optional[str] = Query(None, description="Comma-separated node IDs"),
    labels: Optional[str] = Query(None, description="Comma-separated node labels"),
    entity_types: Optional[str] = Query(None, description="Comma-separated entity types"),
    limit: int = Query(100, ge=1, le=10000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    sort_by: Optional[str] = Query(None, description="Sort field (label, confidence, created_at)"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="Sort order"),
    include_metadata: bool = Query(True, description="Include node metadata"),
    conversation_ids: Optional[str] = Query(None, description="Filter by conversation IDs"),
    confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    current_user: TokenData = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> QueryResultResponse:
    """Query nodes in the knowledge graph.

    Supports filtering by IDs, labels, entity types, and various metadata fields.
    Results are cached for performance and include comprehensive metadata.
    """
    try:
        engine = get_query_engine()

        # Parse query parameters
        options = QueryOptions(
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
            include_metadata=include_metadata,
        )

        # Build filters
        filters = QueryFilter()
        if entity_types:
            type_names = [t.strip() for t in entity_types.split(",")]
            try:
                filters.entity_types = [EntityType(name) for name in type_names]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid entity type: {e}")

        if conversation_ids:
            filters.conversation_ids = [c.strip() for c in conversation_ids.split(",")]

        if confidence_threshold is not None:
            filters.confidence_threshold = confidence_threshold

        # Parse node IDs and labels
        kwargs = {}
        if node_ids:
            kwargs["node_ids"] = [id.strip() for id in node_ids.split(",")]
        if labels:
            kwargs["labels"] = [label.strip() for label in labels.split(",")]

        # Execute query
        result = await engine.execute_query(QueryType.NODE_LOOKUP, options, filters, **kwargs)

        # Convert to API response format
        response_data = _convert_query_result_to_response(result)

        # Record metrics
        _metrics.increment_counter("kg_api_requests_total", {"endpoint": "nodes"})

        return QueryResultResponse(
            data=response_data,
            execution_time_ms=result.execution_time_ms,
            cache_hit=result.cache_hit,
            total_count=result.total_count,
            has_more=result.has_more,
            query_id=result.query_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Node query failed: {e}", exc_info=True)
        _metrics.increment_counter("kg_api_errors_total", {"endpoint": "nodes"})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/paths", response_model=QueryResultResponse)
async def find_paths(
    start_node_id: str = Query(..., description="Starting node ID"),
    end_node_id: str = Query(..., description="Target node ID"),
    max_paths: int = Query(10, ge=1, le=100, description="Maximum paths to find"),
    max_depth: int = Query(6, ge=1, le=20, description="Maximum path length"),
    relation_types: Optional[str] = Query(
        None, description="Comma-separated relation types to follow"
    ),
    current_user: TokenData = Depends(get_current_user),
) -> QueryResultResponse:
    """Find paths between two nodes in the knowledge graph.

    Uses optimized graph traversal algorithms to find shortest paths
    with configurable constraints on path length and relation types.
    """
    try:
        engine = get_query_engine()

        options = QueryOptions(max_depth=max_depth)
        filters = QueryFilter()

        # Parse relation types
        if relation_types:
            type_names = [t.strip() for t in relation_types.split(",")]
            try:
                filters.relation_types = [RelationType(name) for name in type_names]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid relation type: {e}")

        # Execute path finding
        result = await engine.execute_query(
            QueryType.PATH_FINDING,
            options,
            filters,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            max_paths=max_paths,
        )

        response_data = _convert_query_result_to_response(result)

        _metrics.increment_counter("kg_api_requests_total", {"endpoint": "paths"})

        return QueryResultResponse(
            data=response_data,
            execution_time_ms=result.execution_time_ms,
            cache_hit=result.cache_hit,
            total_count=result.total_count,
            query_id=result.query_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Path finding failed: {e}", exc_info=True)
        _metrics.increment_counter("kg_api_errors_total", {"endpoint": "paths"})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/search", response_model=QueryResultResponse)
async def semantic_search(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query text"),
    similarity_threshold: float = Query(
        0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    ),
    entity_types: Optional[str] = Query(None, description="Comma-separated entity types to search"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    current_user: TokenData = Depends(get_current_user),
) -> QueryResultResponse:
    """Perform semantic search on the knowledge graph.

    Uses advanced NLP techniques to find nodes semantically similar to the query text.
    Results are ranked by similarity score and can be filtered by entity types.
    """
    try:
        engine = get_query_engine()

        options = QueryOptions(limit=limit, offset=offset)
        filters = QueryFilter()

        if entity_types:
            type_names = [t.strip() for t in entity_types.split(",")]
            try:
                filters.entity_types = [EntityType(name) for name in type_names]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid entity type: {e}")

        # Execute semantic search
        result = await engine.execute_query(
            QueryType.SEMANTIC_SEARCH,
            options,
            filters,
            query_text=q,
            similarity_threshold=similarity_threshold,
        )

        response_data = _convert_query_result_to_response(result)

        _metrics.increment_counter("kg_api_requests_total", {"endpoint": "search"})

        return QueryResultResponse(
            data=response_data,
            execution_time_ms=result.execution_time_ms,
            cache_hit=result.cache_hit,
            total_count=result.total_count,
            has_more=result.has_more,
            query_id=result.query_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic search failed: {e}", exc_info=True)
        _metrics.increment_counter("kg_api_errors_total", {"endpoint": "search"})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/neighborhood", response_model=QueryResultResponse)
async def get_neighborhood(
    center_node_id: str = Query(..., description="Center node ID"),
    radius: int = Query(2, ge=1, le=10, description="Neighborhood radius"),
    entity_types: Optional[str] = Query(None, description="Filter by entity types"),
    limit: int = Query(200, ge=1, le=5000, description="Maximum nodes to return"),
    current_user: TokenData = Depends(get_current_user),
) -> QueryResultResponse:
    """Get the neighborhood around a specific node.

    Returns all nodes within a specified radius (number of hops) from the center node,
    along with all edges connecting them. Useful for exploring local graph structure.
    """
    try:
        engine = get_query_engine()

        options = QueryOptions(limit=limit)
        filters = QueryFilter()

        if entity_types:
            type_names = [t.strip() for t in entity_types.split(",")]
            try:
                filters.entity_types = [EntityType(name) for name in type_names]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid entity type: {e}")

        # Execute neighborhood query
        result = await engine.execute_query(
            QueryType.NEIGHBORHOOD,
            options,
            filters,
            center_node_id=center_node_id,
            radius=radius,
        )

        response_data = _convert_query_result_to_response(result)

        _metrics.increment_counter("kg_api_requests_total", {"endpoint": "neighborhood"})

        return QueryResultResponse(
            data=response_data,
            execution_time_ms=result.execution_time_ms,
            cache_hit=result.cache_hit,
            total_count=result.total_count,
            query_id=result.query_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Neighborhood query failed: {e}", exc_info=True)
        _metrics.increment_counter("kg_api_errors_total", {"endpoint": "neighborhood"})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=Dict[str, Any])
async def get_graph_statistics(
    entity_types: Optional[str] = Query(None, description="Filter statistics by entity types"),
    aggregations: str = Query(
        "count,type_distribution,avg_confidence", description="Comma-separated aggregations"
    ),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get statistical information about the knowledge graph.

    Provides various aggregated metrics including node counts, type distributions,
    confidence statistics, and connectivity metrics.
    """
    try:
        engine = get_query_engine()

        options = QueryOptions(limit=1)  # We only need metadata
        filters = QueryFilter()

        if entity_types:
            type_names = [t.strip() for t in entity_types.split(",")]
            try:
                filters.entity_types = [EntityType(name) for name in type_names]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid entity type: {e}")

        agg_list = [agg.strip() for agg in aggregations.split(",")]

        # Execute aggregation query
        result = await engine.execute_query(
            QueryType.AGGREGATION,
            options,
            filters,
            aggregations=agg_list,
        )

        # Get engine statistics
        engine_stats = await engine.get_query_statistics()

        response = {
            "graph_statistics": result.metadata,
            "engine_statistics": engine_stats,
            "query_info": {
                "execution_time_ms": result.execution_time_ms,
                "cache_hit": result.cache_hit,
                "query_id": result.query_id,
            },
        }

        _metrics.increment_counter("kg_api_requests_total", {"endpoint": "stats"})

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistics query failed: {e}", exc_info=True)
        _metrics.increment_counter("kg_api_errors_total", {"endpoint": "stats"})
        raise HTTPException(status_code=500, detail="Internal server error")


# Backward compatibility endpoints


@router.get("/knowledge-graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    limit: int = Query(100, ge=1, le=1000, description="Maximum nodes to return"),
    current_user: TokenData = Depends(get_current_user),
) -> KnowledgeGraphResponse:
    """Get knowledge graph data for visualization (backward compatibility).

    Returns a subset of the knowledge graph suitable for visualization.
    For more advanced querying, use the specific endpoints (/nodes, /search, etc.).
    """
    try:
        engine = get_query_engine()

        options = QueryOptions(limit=limit, include_metadata=True)

        # Get nodes
        node_result = await engine.execute_query(QueryType.NODE_LOOKUP, options)

        # Get edges for the returned nodes if we have any
        edges_data = []
        if node_result.nodes:
            node_ids = [node["id"] for node in node_result.nodes]
            edge_result = await engine.execute_query(QueryType.SUBGRAPH, options, node_ids=node_ids)
            edges_data = edge_result.edges

        response_data = KnowledgeGraphResponse(
            nodes=[KnowledgeNode(**node) for node in node_result.nodes],
            edges=[KnowledgeEdge(**edge) for edge in edges_data],
            metadata={
                "total_nodes": node_result.total_count,
                "total_edges": len(edges_data),
                "query_time_ms": node_result.execution_time_ms,
                "cache_hit": node_result.cache_hit,
            },
            query_info={
                "query_id": node_result.query_id,
                "has_more": node_result.has_more,
            },
        )

        _metrics.increment_counter("kg_api_requests_total", {"endpoint": "knowledge_graph"})

        return response_data

    except Exception as e:
        logger.error(f"Knowledge graph query failed: {e}", exc_info=True)
        _metrics.increment_counter("kg_api_errors_total", {"endpoint": "knowledge_graph"})

        # Fallback to demo data for backward compatibility
        logger.warning("Falling back to demo data due to query failure")
        return _get_demo_knowledge_graph()


@router.get("/", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph_alias(
    current_user: TokenData = Depends(get_current_user),
) -> KnowledgeGraphResponse:
    """Alias for /knowledge-graph endpoint."""
    return await get_knowledge_graph(current_user=current_user)


# Utility functions


def _convert_query_result_to_response(result: Any) -> KnowledgeGraphResponse:
    """Convert query engine result to API response format."""
    nodes = []
    for node_data in result.nodes:
        node = KnowledgeNode(
            id=node_data["id"],
            label=node_data["label"],
            type=node_data["type"],
            confidence=node_data.get("confidence", 1.0),
            properties=node_data.get("properties", {}),
            metadata=node_data.get("metadata"),
            similarity_score=node_data.get("similarity_score"),
            distance_from_center=node_data.get("distance_from_center"),
        )
        nodes.append(node)

    edges = []
    for edge_data in result.edges:
        edge = KnowledgeEdge(
            id=edge_data["id"],
            source=edge_data["source_id"],
            target=edge_data["target_id"],
            type=edge_data["type"],
            confidence=edge_data.get("confidence", 1.0),
            properties=edge_data.get("properties", {}),
            metadata=edge_data.get("metadata"),
        )
        edges.append(edge)

    return KnowledgeGraphResponse(
        nodes=nodes,
        edges=edges,
        paths=result.paths,
        metadata=result.metadata,
        query_info={
            "query_id": result.query_id,
            "query_type": result.query_type.value,
        },
    )


def _get_demo_knowledge_graph() -> KnowledgeGraphResponse:
    """Return demo knowledge graph for fallback scenarios."""
    nodes = [
        KnowledgeNode(
            id="agent-1",
            label="Main Agent",
            type="agent",
            confidence=0.95,
            properties={"status": "active"},
        ),
        KnowledgeNode(
            id="belief-1",
            label="Environment State",
            type="belief",
            confidence=0.8,
            properties={"confidence": 0.8},
        ),
        KnowledgeNode(
            id="goal-1",
            label="Explore Area",
            type="goal",
            confidence=0.9,
            properties={"priority": "high"},
        ),
        KnowledgeNode(
            id="action-1",
            label="Move Forward",
            type="action",
            confidence=0.85,
            properties={"cost": 1.0},
        ),
        KnowledgeNode(
            id="observation-1",
            label="Sensor Data",
            type="observation",
            confidence=0.75,
            properties={"timestamp": "2025-07-29T12:00:00Z"},
        ),
    ]

    edges = [
        KnowledgeEdge(
            id="edge-1",
            source="agent-1",
            target="belief-1",
            type="has_belief",
            confidence=0.9,
        ),
        KnowledgeEdge(
            id="edge-2", source="agent-1", target="goal-1", type="pursues", confidence=0.95
        ),
        KnowledgeEdge(
            id="edge-3", source="goal-1", target="action-1", type="requires", confidence=0.85
        ),
    ]

    return KnowledgeGraphResponse(
        nodes=nodes,
        edges=edges,
        metadata={
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "last_updated": "2025-07-29T12:00:00Z",
            "mode": "demo",
        },
    )

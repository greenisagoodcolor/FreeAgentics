"""Knowledge Graph API endpoints.

This module provides REST API endpoints for managing and querying
knowledge graphs in the FreeAgentics system.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from knowledge_graph.evolution import EvolutionEngine
from knowledge_graph.graph_engine import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
)
from knowledge_graph.query import GraphQuery, QueryEngine, QueryType
from knowledge_graph.storage import FileStorageBackend, StorageManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize storage manager
storage_manager = StorageManager(FileStorageBackend("./knowledge_graphs"))

# Cache for active graphs
active_graphs: Dict[str, KnowledgeGraph] = {}
query_engines: Dict[str, QueryEngine] = {}
evolution_engine = EvolutionEngine()


class NodeCreateRequest(BaseModel):
    """Request to create a knowledge node."""

    type: str = Field(..., description="Node type")
    label: str = Field(..., description="Node label")
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: Optional[str] = Field(None, description="Source of knowledge")


class EdgeCreateRequest(BaseModel):
    """Request to create a knowledge edge."""

    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    type: str = Field(..., description="Edge type")
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class GraphQueryRequest(BaseModel):
    """Request to query a knowledge graph."""

    query_type: str = Field(..., description="Type of query")
    node_ids: Optional[List[str]] = None
    node_types: Optional[List[str]] = None
    node_labels: Optional[List[str]] = None
    node_properties: Optional[Dict[str, Any]] = None
    edge_types: Optional[List[str]] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    center_id: Optional[str] = None
    radius: int = Field(1, ge=1)
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0)
    limit: Optional[int] = Field(None, ge=1)
    order_by: Optional[str] = None
    descending: bool = False


class EvolutionRequest(BaseModel):
    """Request to evolve a knowledge graph."""

    context: Dict[str, Any] = Field(..., description="Evolution context")


class GraphResponse(BaseModel):
    """Response containing graph information."""

    graph_id: str
    version: int
    created_at: str
    updated_at: str
    node_count: int
    edge_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeResponse(BaseModel):
    """Response containing node information."""

    id: str
    type: str
    label: str
    properties: Dict[str, Any]
    created_at: str
    updated_at: str
    version: int
    confidence: float
    source: Optional[str]


class EdgeResponse(BaseModel):
    """Response containing edge information."""

    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    created_at: str
    confidence: float


class QueryResultResponse(BaseModel):
    """Response containing query results."""

    query_type: str
    nodes: List[NodeResponse]
    edges: List[EdgeResponse]
    paths: List[List[str]]
    aggregates: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any]


def get_active_graph(graph_id: str) -> KnowledgeGraph:
    """Get active graph or load from storage."""
    if graph_id not in active_graphs:
        graph = storage_manager.load(graph_id)
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Graph {graph_id} not found",
            )
        active_graphs[graph_id] = graph

    return active_graphs[graph_id]


def get_query_engine(graph_id: str) -> QueryEngine:
    """Get query engine for graph."""
    if graph_id not in query_engines:
        graph = get_active_graph(graph_id)
        query_engines[graph_id] = QueryEngine(graph)

    return query_engines[graph_id]


@router.post("/graphs", response_model=GraphResponse)
async def create_graph() -> GraphResponse:
    """Create a new knowledge graph."""
    graph = KnowledgeGraph()

    # Save to storage
    if not storage_manager.save(graph):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save graph",
        )

    # Cache it
    active_graphs[graph.graph_id] = graph

    logger.info(f"Created knowledge graph {graph.graph_id}")

    return GraphResponse(
        graph_id=graph.graph_id,
        version=graph.version,
        created_at=graph.created_at.isoformat(),
        updated_at=graph.updated_at.isoformat(),
        node_count=0,
        edge_count=0,
    )


@router.get("/graphs", response_model=List[GraphResponse])
async def list_graphs() -> List[GraphResponse]:
    """List all knowledge graphs."""
    graphs = storage_manager.list()

    responses = []
    for graph_info in graphs:
        responses.append(
            GraphResponse(
                graph_id=graph_info["graph_id"],
                version=graph_info["version"],
                created_at=graph_info["created_at"],
                updated_at=graph_info["updated_at"],
                node_count=graph_info.get("node_count", 0),
                edge_count=graph_info.get("edge_count", 0),
            )
        )

    return responses


@router.get("/graphs/{graph_id}", response_model=GraphResponse)
async def get_graph(graph_id: str) -> GraphResponse:
    """Get information about a specific graph."""
    graph = get_active_graph(graph_id)

    return GraphResponse(
        graph_id=graph.graph_id,
        version=graph.version,
        created_at=graph.created_at.isoformat(),
        updated_at=graph.updated_at.isoformat(),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        metadata={
            "node_types": {
                nt.value: len(graph.type_index.get(nt, [])) for nt in NodeType
            }
        },
    )


@router.delete("/graphs/{graph_id}")
async def delete_graph(graph_id: str):
    """Delete a knowledge graph."""
    if not storage_manager.exists(graph_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph {graph_id} not found",
        )

    # Remove from cache
    if graph_id in active_graphs:
        del active_graphs[graph_id]
    if graph_id in query_engines:
        del query_engines[graph_id]

    # Delete from storage
    if not storage_manager.delete(graph_id):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete graph",
        )

    logger.info(f"Deleted knowledge graph {graph_id}")

    return {"message": f"Graph {graph_id} deleted successfully"}


@router.post("/graphs/{graph_id}/nodes", response_model=NodeResponse)
async def create_node(
    graph_id: str, request: NodeCreateRequest
) -> NodeResponse:
    """Create a node in the knowledge graph."""
    graph = get_active_graph(graph_id)

    # Create node
    try:
        node_type = NodeType(request.type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid node type: {request.type}",
        )

    node = KnowledgeNode(
        type=node_type,
        label=request.label,
        properties=request.properties,
        confidence=request.confidence,
        source=request.source,
    )

    if not graph.add_node(node):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add node",
        )

    # Save graph
    storage_manager.save(graph)

    # Broadcast via WebSocket
    try:
        import asyncio

        from api.v1.websocket import broadcast_system_event

        asyncio.create_task(
            broadcast_system_event(
                "knowledge_graph_updated",
                {
                    "graph_id": graph_id,
                    "action": "node_added",
                    "node_id": node.id,
                },
            )
        )
    except Exception as e:
        logger.error(f"Failed to broadcast event: {e}")

    return NodeResponse(
        id=node.id,
        type=node.type.value,
        label=node.label,
        properties=node.properties,
        created_at=node.created_at.isoformat(),
        updated_at=node.updated_at.isoformat(),
        version=node.version,
        confidence=node.confidence,
        source=node.source,
    )


@router.get("/graphs/{graph_id}/nodes", response_model=List[NodeResponse])
async def list_nodes(
    graph_id: str,
    type: Optional[str] = Query(None, description="Filter by node type"),
    label: Optional[str] = Query(None, description="Filter by label"),
    limit: Optional[int] = Query(100, ge=1, le=1000),
) -> List[NodeResponse]:
    """List nodes in the knowledge graph."""
    graph = get_active_graph(graph_id)

    nodes = list(graph.nodes.values())

    # Apply filters
    if type:
        try:
            node_type = NodeType(type)
            nodes = [n for n in nodes if n.type == node_type]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid node type: {type}",
            )

    if label:
        nodes = [n for n in nodes if n.label == label]

    # Apply limit
    nodes = nodes[:limit]

    # Convert to response
    responses = []
    for node in nodes:
        responses.append(
            NodeResponse(
                id=node.id,
                type=node.type.value,
                label=node.label,
                properties=node.properties,
                created_at=node.created_at.isoformat(),
                updated_at=node.updated_at.isoformat(),
                version=node.version,
                confidence=node.confidence,
                source=node.source,
            )
        )

    return responses


@router.get("/graphs/{graph_id}/nodes/{node_id}", response_model=NodeResponse)
async def get_node(graph_id: str, node_id: str) -> NodeResponse:
    """Get a specific node."""
    graph = get_active_graph(graph_id)

    node = graph.get_node(node_id)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node {node_id} not found",
        )

    return NodeResponse(
        id=node.id,
        type=node.type.value,
        label=node.label,
        properties=node.properties,
        created_at=node.created_at.isoformat(),
        updated_at=node.updated_at.isoformat(),
        version=node.version,
        confidence=node.confidence,
        source=node.source,
    )


@router.put("/graphs/{graph_id}/nodes/{node_id}")
async def update_node(graph_id: str, node_id: str, properties: Dict[str, Any]):
    """Update node properties."""
    graph = get_active_graph(graph_id)

    if not graph.update_node(node_id, properties):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node {node_id} not found",
        )

    # Save graph
    storage_manager.save(graph)

    return {"message": f"Node {node_id} updated successfully"}


@router.delete("/graphs/{graph_id}/nodes/{node_id}")
async def delete_node(graph_id: str, node_id: str):
    """Delete a node from the graph."""
    graph = get_active_graph(graph_id)

    if not graph.remove_node(node_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node {node_id} not found",
        )

    # Save graph
    storage_manager.save(graph)

    return {"message": f"Node {node_id} deleted successfully"}


@router.post("/graphs/{graph_id}/edges", response_model=EdgeResponse)
async def create_edge(
    graph_id: str, request: EdgeCreateRequest
) -> EdgeResponse:
    """Create an edge in the knowledge graph."""
    graph = get_active_graph(graph_id)

    # Validate edge type
    try:
        edge_type = EdgeType(request.type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid edge type: {request.type}",
        )

    # Create edge
    edge = KnowledgeEdge(
        source_id=request.source_id,
        target_id=request.target_id,
        type=edge_type,
        properties=request.properties,
        confidence=request.confidence,
    )

    if not graph.add_edge(edge):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add edge (check if nodes exist)",
        )

    # Save graph
    storage_manager.save(graph)

    return EdgeResponse(
        id=edge.id,
        source_id=edge.source_id,
        target_id=edge.target_id,
        type=edge.type.value,
        properties=edge.properties,
        created_at=edge.created_at.isoformat(),
        confidence=edge.confidence,
    )


@router.get("/graphs/{graph_id}/edges", response_model=List[EdgeResponse])
async def list_edges(
    graph_id: str,
    type: Optional[str] = Query(None, description="Filter by edge type"),
    source_id: Optional[str] = Query(
        None, description="Filter by source node"
    ),
    target_id: Optional[str] = Query(
        None, description="Filter by target node"
    ),
    limit: Optional[int] = Query(100, ge=1, le=1000),
) -> List[EdgeResponse]:
    """List edges in the knowledge graph."""
    graph = get_active_graph(graph_id)

    edges = list(graph.edges.values())

    # Apply filters
    if type:
        try:
            edge_type = EdgeType(type)
            edges = [e for e in edges if e.type == edge_type]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid edge type: {type}",
            )

    if source_id:
        edges = [e for e in edges if e.source_id == source_id]

    if target_id:
        edges = [e for e in edges if e.target_id == target_id]

    # Apply limit
    edges = edges[:limit]

    # Convert to response
    responses = []
    for edge in edges:
        responses.append(
            EdgeResponse(
                id=edge.id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                type=edge.type.value,
                properties=edge.properties,
                created_at=edge.created_at.isoformat(),
                confidence=edge.confidence,
            )
        )

    return responses


@router.post("/graphs/{graph_id}/query", response_model=QueryResultResponse)
async def query_graph(
    graph_id: str, request: GraphQueryRequest
) -> QueryResultResponse:
    """Query the knowledge graph."""
    engine = get_query_engine(graph_id)

    # Build query
    try:
        query_type = QueryType(request.query_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query type: {request.query_type}",
        )

    # Convert string types to enums
    node_types = None
    if request.node_types:
        try:
            node_types = [NodeType(t) for t in request.node_types]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid node type: {e}",
            )

    edge_types = None
    if request.edge_types:
        try:
            edge_types = [EdgeType(t) for t in request.edge_types]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid edge type: {e}",
            )

    query = GraphQuery(
        query_type=query_type,
        node_ids=request.node_ids,
        node_types=node_types,
        node_labels=request.node_labels,
        node_properties=request.node_properties,
        edge_types=edge_types,
        source_id=request.source_id,
        target_id=request.target_id,
        center_id=request.center_id,
        radius=request.radius,
        confidence_threshold=request.confidence_threshold,
        limit=request.limit,
        order_by=request.order_by,
        descending=request.descending,
    )

    # Execute query
    result = engine.execute(query)

    # Convert to response
    node_responses = []
    for node in result.nodes:
        node_responses.append(
            NodeResponse(
                id=node.id,
                type=node.type.value,
                label=node.label,
                properties=node.properties,
                created_at=node.created_at.isoformat(),
                updated_at=node.updated_at.isoformat(),
                version=node.version,
                confidence=node.confidence,
                source=node.source,
            )
        )

    edge_responses = []
    for edge in result.edges:
        edge_responses.append(
            EdgeResponse(
                id=edge.id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                type=edge.type.value,
                properties=edge.properties,
                created_at=edge.created_at.isoformat(),
                confidence=edge.confidence,
            )
        )

    return QueryResultResponse(
        query_type=result.query_type.value,
        nodes=node_responses,
        edges=edge_responses,
        paths=result.paths,
        aggregates=result.aggregates,
        execution_time=result.execution_time,
        metadata=result.metadata,
    )


@router.post("/graphs/{graph_id}/evolve")
async def evolve_graph(graph_id: str, request: EvolutionRequest):
    """Evolve the knowledge graph based on context."""
    graph = get_active_graph(graph_id)

    # Clear query cache since graph will change
    if graph_id in query_engines:
        query_engines[graph_id].clear_cache()

    # Evolve graph
    metrics = evolution_engine.evolve(graph, request.context)

    # Save evolved graph
    storage_manager.save(graph)

    # Broadcast evolution event
    try:
        import asyncio

        from api.v1.websocket import broadcast_system_event

        asyncio.create_task(
            broadcast_system_event(
                "knowledge_graph_evolved",
                {
                    "graph_id": graph_id,
                    "metrics": {
                        "nodes_added": metrics.nodes_added,
                        "nodes_removed": metrics.nodes_removed,
                        "nodes_updated": metrics.nodes_updated,
                        "edges_added": metrics.edges_added,
                        "edges_removed": metrics.edges_removed,
                        "confidence_changes": metrics.confidence_changes,
                        "contradictions_resolved": metrics.contradictions_resolved,
                    },
                },
            )
        )
    except Exception as e:
        logger.error(f"Failed to broadcast evolution event: {e}")

    return {
        "message": "Graph evolved successfully",
        "metrics": {
            "nodes_added": metrics.nodes_added,
            "nodes_removed": metrics.nodes_removed,
            "nodes_updated": metrics.nodes_updated,
            "edges_added": metrics.edges_added,
            "edges_removed": metrics.edges_removed,
            "confidence_changes": metrics.confidence_changes,
            "contradictions_resolved": metrics.contradictions_resolved,
        },
    }


@router.get("/graphs/{graph_id}/neighbors/{node_id}")
async def get_neighbors(
    graph_id: str,
    node_id: str,
    edge_type: Optional[str] = Query(None, description="Filter by edge type"),
) -> List[NodeResponse]:
    """Get neighboring nodes."""
    graph = get_active_graph(graph_id)

    # Get neighbors
    edge_type_enum = None
    if edge_type:
        try:
            edge_type_enum = EdgeType(edge_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid edge type: {edge_type}",
            )

    neighbor_ids = graph.get_neighbors(node_id, edge_type_enum)

    # Get neighbor nodes
    responses = []
    for neighbor_id in neighbor_ids:
        node = graph.get_node(neighbor_id)
        if node:
            responses.append(
                NodeResponse(
                    id=node.id,
                    type=node.type.value,
                    label=node.label,
                    properties=node.properties,
                    created_at=node.created_at.isoformat(),
                    updated_at=node.updated_at.isoformat(),
                    version=node.version,
                    confidence=node.confidence,
                    source=node.source,
                )
            )

    return responses


@router.get("/graphs/{graph_id}/importance")
async def get_node_importance(graph_id: str) -> Dict[str, float]:
    """Calculate importance scores for all nodes."""
    graph = get_active_graph(graph_id)

    importance = graph.calculate_node_importance()

    return importance


@router.get("/graphs/{graph_id}/communities")
async def detect_communities(graph_id: str) -> List[List[str]]:
    """Detect communities in the graph."""
    graph = get_active_graph(graph_id)

    communities = graph.detect_communities()

    # Convert sets to lists for JSON serialization
    return [list(community) for community in communities]

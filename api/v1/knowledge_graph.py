"""Knowledge graph endpoint for UI compatibility."""

import logging
from typing import Dict, List, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.security_implementation import TokenData, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


class KnowledgeNode(BaseModel):
    """Knowledge graph node."""

    id: str
    label: str
    type: str = "concept"
    properties: Dict[str, Any] = {}


class KnowledgeEdge(BaseModel):
    """Knowledge graph edge."""

    source: str
    target: str
    type: str = "relates_to"
    weight: float = 1.0


class KnowledgeGraphResponse(BaseModel):
    """Knowledge graph response."""

    nodes: List[KnowledgeNode]
    edges: List[KnowledgeEdge]
    metadata: Dict[str, Any] = {}


@router.get("/knowledge-graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    current_user: TokenData = Depends(get_current_user),
) -> KnowledgeGraphResponse:
    """Get knowledge graph data for visualization.

    In demo mode, returns a mock knowledge graph.
    In production, would query the actual knowledge base.
    """
    # For now, return a mock knowledge graph
    nodes = [
        KnowledgeNode(
            id="agent-1", label="Main Agent", type="agent", properties={"status": "active"}
        ),
        KnowledgeNode(
            id="belief-1", label="Environment State", type="belief", properties={"confidence": 0.8}
        ),
        KnowledgeNode(
            id="goal-1", label="Explore Area", type="goal", properties={"priority": "high"}
        ),
        KnowledgeNode(id="action-1", label="Move Forward", type="action", properties={"cost": 1.0}),
    ]

    edges = [
        KnowledgeEdge(source="agent-1", target="belief-1", type="has_belief"),
        KnowledgeEdge(source="agent-1", target="goal-1", type="pursues"),
        KnowledgeEdge(source="goal-1", target="action-1", type="requires"),
    ]

    return KnowledgeGraphResponse(
        nodes=nodes,
        edges=edges,
        metadata={
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "last_updated": "2025-07-29T12:00:00Z",
        },
    )


@router.get("/", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph_alias(
    current_user: TokenData = Depends(get_current_user),
) -> KnowledgeGraphResponse:
    """Alias for /knowledge-graph endpoint."""
    return await get_knowledge_graph(current_user)

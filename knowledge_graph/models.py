# knowledge_graph/models.py
from typing import Any, Dict, List

from pydantic import BaseModel


class NodeModel(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = {}


class EdgeModel(BaseModel):
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any] = {}


class KnowledgeGraphResponse(BaseModel):
    nodes: List[NodeModel] = []
    edges: List[EdgeModel] = []
    metadata: Dict[str, Any] = {}

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to ensure consistent dict serialization"""
        return {
            "nodes": [node.model_dump() for node in self.nodes],
            "edges": [edge.model_dump() for edge in self.edges],
            "metadata": self.metadata,
        }

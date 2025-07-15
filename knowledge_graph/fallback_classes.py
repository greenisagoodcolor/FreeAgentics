"""Fallback classes for knowledge graph dependencies to reduce complexity."""

import logging
from typing import Any, Dict, Optional


class StorageManager:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def save(self, graph: Any) -> None:
        return True

    def delete(self, graph_id: str) -> None:
        return True


class FileStorageBackend:
    def __init__(self, path):
        pass


class EventDispatcher:
    async def dispatch_event(self, event: Dict[str, Any]) -> None:
        pass


def get_event_dispatcher() -> Optional[EventDispatcher]:
    return EventDispatcher()


class ConversationEventIntegration:
    pass


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph_id = "unknown"
        self.node_count = 0
        self.edge_count = 0

    def add_node(self, node: Any) -> None:
        return True

    def add_edge(self, edge: Any) -> None:
        return True

    def save(self) -> None:
        pass


class KnowledgeNode:
    def __init__(self, **kwargs: Any) -> None:
        self.id = kwargs.get("id", "unknown")
        self.type = kwargs.get("type")
        self.label = kwargs.get("label", "")
        self.properties = kwargs.get("properties", {})
        self.confidence = kwargs.get("confidence", 1.0)
        self.source = kwargs.get("source")


class KnowledgeEdge:
    def __init__(self, **kwargs: Any) -> None:
        self.id = kwargs.get("id", "unknown")
        self.source_id = kwargs.get("source_id")
        self.target_id = kwargs.get("target_id")
        self.type = kwargs.get("type")
        self.properties = kwargs.get("properties", {})
        self.confidence = kwargs.get("confidence", 1.0)


class QueryEngine:
    def execute(self, query: Any) -> Dict[str, Any]:
        return {}


class GraphSearchEngine:
    pass


class EvolutionEngine:
    pass


class GraphQuery:
    def __init__(self, **kwargs: Any) -> None:
        self.query_type = kwargs.get("query_type")
        self.node_ids = kwargs.get("node_ids")
        self.node_types = kwargs.get("node_types")
        self.node_labels = kwargs.get("node_labels")
        self.node_properties = kwargs.get("node_properties")
        self.edge_types = kwargs.get("edge_types")
        self.source_id = kwargs.get("source_id")
        self.target_id = kwargs.get("target_id")
        self.center_id = kwargs.get("center_id")
        self.radius = kwargs.get("radius", 1)
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.0)
        self.limit = kwargs.get("limit")
        self.order_by = kwargs.get("order_by")
        self.descending = kwargs.get("descending", False)


class NodeType:
    def __init__(self, name: str) -> None:
        self.name = name
        self.value = name


class EdgeType:
    def __init__(self, name: str) -> None:
        self.name = name
        self.value = name


def create_node_event(**kwargs: Any) -> Dict[str, Any]:
    return {}


class EventType:
    NODE_CREATED = "node_created"


class EventSource:
    API_REQUEST = "api_request"


def create_all_fallback_classes():
    """Create and return all fallback classes."""
    logger = logging.getLogger(__name__)

    return {
        "logger": logger,
        "ConversationEventIntegration": ConversationEventIntegration,
        "EventSource": EventSource,
        "EventType": EventType,
        "create_node_event": create_node_event,
        "get_event_dispatcher": get_event_dispatcher,
        "EvolutionEngine": EvolutionEngine,
        "KnowledgeEdge": KnowledgeEdge,
        "KnowledgeGraph": KnowledgeGraph,
        "KnowledgeNode": KnowledgeNode,
        "EdgeType": EdgeType,
        "GraphQuery": GraphQuery,
        "NodeType": NodeType,
        "QueryEngine": QueryEngine,
        "GraphSearchEngine": GraphSearchEngine,
        "FileStorageBackend": FileStorageBackend,
        "StorageManager": StorageManager,
    }

"""Core knowledge graph engine for agent knowledge representation.

This module implements a temporal knowledge graph with versioning,
allowing agents to maintain and evolve their understanding over time.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import networkx as nx

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""

    ENTITY = "entity"  # Real-world entities (agents, objects, locations)
    CONCEPT = "concept"  # Abstract concepts and categories
    PROPERTY = "property"  # Properties and attributes
    EVENT = "event"  # Events and actions
    BELIEF = "belief"  # Agent beliefs and predictions
    GOAL = "goal"  # Agent goals and objectives
    OBSERVATION = "observation"  # Sensory observations


class EdgeType(Enum):
    """Types of edges in the knowledge graph."""

    IS_A = "is_a"  # Inheritance/type relationship
    HAS_PROPERTY = "has_property"  # Entity has property
    RELATED_TO = "related_to"  # General relation
    CAUSES = "causes"  # Causal relationship
    PRECEDES = "precedes"  # Temporal ordering
    BELIEVES = "believes"  # Agent believes fact
    OBSERVES = "observes"  # Agent observes entity
    DESIRES = "desires"  # Agent desires state
    LOCATED_AT = "located_at"  # Spatial relationship
    PART_OF = "part_of"  # Compositional relationship


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    id: str = field(default_factory=lambda: str(uuid4()))
    type: NodeType = NodeType.ENTITY
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    confidence: float = 1.0  # Confidence in this knowledge
    source: Optional[str] = None  # Source of this knowledge (agent_id, sensor, etc.)

    def update(self, properties: Dict[str, Any]):
        """Update node properties and increment version."""
        self.properties.update(properties)
        self.updated_at = datetime.now()
        self.version += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph."""

    id: str = field(default_factory=lambda: str(uuid4()))
    source_id: str = ""
    target_id: str = ""
    type: EdgeType = EdgeType.RELATED_TO
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "confidence": self.confidence,
        }


class KnowledgeGraph:
    """Temporal knowledge graph with versioning support."""

    def __init__(self, graph_id: Optional[str] = None):
        """Initialize knowledge graph.

        Args:
            graph_id: Unique identifier for this graph
        """
        self.graph_id = graph_id or str(uuid4())
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.version = 1
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # History tracking
        self.node_history: Dict[str, List[KnowledgeNode]] = {}
        self.edge_history: Dict[str, List[KnowledgeEdge]] = {}

        # Indexes for efficient querying
        self.type_index: Dict[NodeType, Set[str]] = {}
        self.label_index: Dict[str, Set[str]] = {}
        self.property_index: Dict[str, Dict[Any, Set[str]]] = {}

        logger.info(f"Created knowledge graph {self.graph_id}")

    def add_node(self, node: KnowledgeNode) -> bool:
        """Add a node to the graph.

        Args:
            node: Node to add

        Returns:
            True if node was added successfully
        """
        if node.id in self.nodes:
            logger.warning(f"Node {node.id} already exists")
            return False

        # Add to graph
        self.graph.add_node(node.id, data=node)
        self.nodes[node.id] = node

        # Update indexes
        if node.type not in self.type_index:
            self.type_index[node.type] = set()
        self.type_index[node.type].add(node.id)

        if node.label:
            if node.label not in self.label_index:
                self.label_index[node.label] = set()
            self.label_index[node.label].add(node.id)

        # Initialize history
        self.node_history[node.id] = [node]

        self._update_version()
        logger.debug(f"Added node {node.id} ({node.label})")
        return True

    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """Add an edge to the graph.

        Args:
            edge: Edge to add

        Returns:
            True if edge was added successfully
        """
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            logger.error("Source or target node not found")
            return False

        if edge.id in self.edges:
            logger.warning(f"Edge {edge.id} already exists")
            return False

        # Add to graph
        self.graph.add_edge(edge.source_id, edge.target_id, key=edge.id, data=edge)
        self.edges[edge.id] = edge

        # Initialize history
        self.edge_history[edge.id] = [edge]

        self._update_version()
        logger.debug(f"Added edge {edge.id} ({edge.type.value})")
        return True

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update a node's properties.

        Args:
            node_id: ID of node to update
            properties: Properties to update

        Returns:
            True if node was updated successfully
        """
        if node_id not in self.nodes:
            logger.error(f"Node {node_id} not found")
            return False

        node = self.nodes[node_id]

        # Create a copy for history
        import copy

        historical_node = copy.deepcopy(node)

        # Update node
        node.update(properties)

        # Update history
        self.node_history[node_id].append(historical_node)

        self._update_version()
        logger.debug(f"Updated node {node_id}")
        return True

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the graph.

        Args:
            node_id: ID of node to remove

        Returns:
            True if node was removed successfully
        """
        if node_id not in self.nodes:
            logger.error(f"Node {node_id} not found")
            return False

        node = self.nodes[node_id]

        # Remove from indexes
        self.type_index[node.type].discard(node_id)
        if node.label:
            self.label_index[node.label].discard(node_id)

        # Remove associated edges
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            del self.edges[edge_id]

        # Remove from graph
        self.graph.remove_node(node_id)
        del self.nodes[node_id]

        self._update_version()
        logger.debug(f"Removed node {node_id}")
        return True

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node or None if not found
        """
        return self.nodes.get(node_id)

    def get_neighbors(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[str]:
        """Get neighboring nodes.

        Args:
            node_id: Source node ID
            edge_type: Filter by edge type

        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.nodes:
            return []

        neighbors = []
        for _, target, key, data in self.graph.out_edges(node_id, keys=True, data=True):
            edge = data["data"]
            if edge_type is None or edge.type == edge_type:
                neighbors.append(target)

        return neighbors

    def find_nodes_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """Find all nodes of a given type.

        Args:
            node_type: Type to search for

        Returns:
            List of matching nodes
        """
        node_ids = self.type_index.get(node_type, set())
        return [self.nodes[node_id] for node_id in node_ids]

    def find_nodes_by_label(self, label: str) -> List[KnowledgeNode]:
        """Find all nodes with a given label.

        Args:
            label: Label to search for

        Returns:
            List of matching nodes
        """
        node_ids = self.label_index.get(label, set())
        return [self.nodes[node_id] for node_id in node_ids]

    def find_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest path between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            List of node IDs forming path, or None if no path exists
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_subgraph(
        self, node_ids: List[str], include_edges: bool = True
    ) -> "KnowledgeGraph":
        """Extract a subgraph containing specified nodes.

        Args:
            node_ids: List of node IDs to include
            include_edges: Whether to include edges between nodes

        Returns:
            New KnowledgeGraph containing subgraph
        """
        subgraph = KnowledgeGraph()

        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                import copy

                node_copy = copy.deepcopy(self.nodes[node_id])
                subgraph.add_node(node_copy)

        # Add edges if requested
        if include_edges:
            for edge in self.edges.values():
                if edge.source_id in node_ids and edge.target_id in node_ids:
                    import copy

                    edge_copy = copy.deepcopy(edge)
                    subgraph.add_edge(edge_copy)

        return subgraph

    def merge(self, other: "KnowledgeGraph", conflict_resolution: str = "newer"):
        """Merge another knowledge graph into this one.

        Args:
            other: Graph to merge
            conflict_resolution: How to resolve conflicts ("newer", "higher_confidence", "keep_both")
        """
        # Merge nodes
        for node in other.nodes.values():
            if node.id in self.nodes:
                # Handle conflict
                if conflict_resolution == "newer":
                    if node.updated_at > self.nodes[node.id].updated_at:
                        self.update_node(node.id, node.properties)
                elif conflict_resolution == "higher_confidence":
                    if node.confidence > self.nodes[node.id].confidence:
                        self.update_node(node.id, node.properties)
                elif conflict_resolution == "keep_both":
                    # Create new node with different ID
                    import copy

                    new_node = copy.deepcopy(node)
                    new_node.id = str(uuid4())
                    self.add_node(new_node)
            else:
                import copy

                self.add_node(copy.deepcopy(node))

        # Merge edges
        for edge in other.edges.values():
            if edge.id not in self.edges:
                # Check if nodes exist
                if edge.source_id in self.nodes and edge.target_id in self.nodes:
                    import copy

                    self.add_edge(copy.deepcopy(edge))

        logger.info(f"Merged graph {other.graph_id} into {self.graph_id}")

    def calculate_node_importance(self) -> Dict[str, float]:
        """Calculate importance scores for all nodes using PageRank.

        Returns:
            Dictionary mapping node IDs to importance scores
        """
        if not self.nodes:
            return {}

        try:
            scores = nx.pagerank(self.graph)
            return scores
        except Exception:
            # Fallback to degree centrality
            scores = nx.degree_centrality(self.graph)
            return scores

    def detect_communities(self) -> List[Set[str]]:
        """Detect communities in the graph.

        Returns:
            List of sets, each containing node IDs in a community
        """
        if not self.nodes:
            return []

        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()

        try:
            # Use Louvain community detection
            import community

            partition = community.best_partition(undirected)

            # Group nodes by community
            communities: Dict[int, Set[str]] = {}
            for node_id, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = set()
                communities[comm_id].add(node_id)

            return list(communities.values())
        except Exception:
            # Fallback to connected components
            return [set(comp) for comp in nx.connected_components(undirected)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "graph_id": self.graph_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "statistics": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "node_types": {
                    nt.value: len(self.type_index.get(nt, [])) for nt in NodeType
                },
            },
        }

    def save_to_file(self, filepath: str):
        """Save graph to JSON file.

        Args:
            filepath: Path to save file
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved graph to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> "KnowledgeGraph":
        """Load graph from JSON file.

        Args:
            filepath: Path to load from

        Returns:
            Loaded KnowledgeGraph
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        graph = cls(graph_id=data["graph_id"])

        # Load nodes
        for node_data in data["nodes"]:
            node = KnowledgeNode(
                id=node_data["id"],
                type=NodeType(node_data["type"]),
                label=node_data["label"],
                properties=node_data["properties"],
                confidence=node_data["confidence"],
                source=node_data.get("source"),
            )
            graph.add_node(node)

        # Load edges
        for edge_data in data["edges"]:
            edge = KnowledgeEdge(
                id=edge_data["id"],
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                type=EdgeType(edge_data["type"]),
                properties=edge_data["properties"],
                confidence=edge_data["confidence"],
            )
            graph.add_edge(edge)

        logger.info(f"Loaded graph from {filepath}")
        return graph

    def _update_version(self):
        """Update graph version and timestamp."""
        self.version += 1
        self.updated_at = datetime.now()

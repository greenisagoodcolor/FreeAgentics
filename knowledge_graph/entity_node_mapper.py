"""
Entity to Knowledge Graph Node Mapper.
Maps extracted entities to knowledge graph nodes using various strategies
"""

import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional

from knowledge_graph.graph_engine import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
)
from knowledge_graph.nlp_entity_extractor import (
    Entity,
    EntityType,
    Relationship,
)

logger = logging.getLogger(__name__)


class MappingStrategy(Enum):
    """Strategies for mapping entities to nodes."""

    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_MATCH = "semantic_match"
    CREATE_NEW = "create_new"


@dataclass
class Node:
    """Simplified node representation for compatibility."""

    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Simplified edge representation for compatibility."""

    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeMapping:
    """Represents a mapping from entity to knowledge graph node."""

    entity: Entity
    node: Node
    confidence: float
    strategy: MappingStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MappingResult:
    """Result of bulk entity mapping."""

    mappings: List[NodeMapping]
    total_entities: int
    successful_mappings: int
    failed_mappings: int
    execution_time: float


class GraphEngine:
    """Simplified GraphEngine interface for compatibility."""

    def __init__(self):
        self.graph = KnowledgeGraph()

    async def find_nodes_by_name(self, name: str) -> List[Node]:
        """Find nodes by name/label."""
        matching_nodes = []
        for node_id, node in self.graph.nodes.items():
            if (
                node.label.lower() == name.lower()
                or node.properties.get("name", "").lower() == name.lower()
            ):
                matching_nodes.append(
                    Node(
                        id=node.id,
                        type=node.type.value,
                        properties=node.properties,
                    )
                )
        return matching_nodes

    async def search_similar_nodes(
        self, name: str, context: Optional[Dict] = None
    ) -> List[Node]:
        """Search for similar nodes using fuzzy matching."""
        similar_nodes = []
        threshold = 0.7

        for node_id, node in self.graph.nodes.items():
            # Check name similarity
            name_similarity = SequenceMatcher(
                None, name.lower(), node.label.lower()
            ).ratio()
            prop_name_similarity = SequenceMatcher(
                None, name.lower(), node.properties.get("name", "").lower()
            ).ratio()

            max_similarity = max(name_similarity, prop_name_similarity)

            # Check aliases
            aliases = node.properties.get("aliases", [])
            for alias in aliases:
                alias_similarity = SequenceMatcher(
                    None, name.lower(), alias.lower()
                ).ratio()
                max_similarity = max(max_similarity, alias_similarity)

            if max_similarity >= threshold:
                similar_node = Node(
                    id=node.id,
                    type=node.type.value,
                    properties={
                        **node.properties,
                        "similarity": max_similarity,
                    },
                )
                similar_nodes.append(similar_node)

        # Sort by similarity
        similar_nodes.sort(
            key=lambda n: n.properties.get("similarity", 0), reverse=True
        )
        return similar_nodes

    async def create_node(self, type: str, properties: Dict[str, Any]) -> Node:
        """Create a new node."""
        # Map string type to NodeType
        node_type_mapping = {
            "Technology": NodeType.ENTITY,
            "Person": NodeType.ENTITY,
            "Organization": NodeType.ENTITY,
            "Concept": NodeType.CONCEPT,
            "Location": NodeType.ENTITY,
            "Date": NodeType.ENTITY,
            "Event": NodeType.EVENT,
            "Product": NodeType.ENTITY,
        }

        node_type = node_type_mapping.get(type, NodeType.ENTITY)

        knowledge_node = KnowledgeNode(
            type=node_type,
            label=properties.get("name", ""),
            properties=properties,
        )

        self.graph.add_node(knowledge_node)

        return Node(id=knowledge_node.id, type=type, properties=properties)

    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Dict[str, Any],
    ) -> Edge:
        """Create a new edge."""
        # Map string type to EdgeType
        edge_type_mapping = {
            "used_for": EdgeType.RELATED_TO,
            "implemented_by": EdgeType.RELATED_TO,
            "works_at": EdgeType.RELATED_TO,
            "employs": EdgeType.RELATED_TO,
            "uses": EdgeType.RELATED_TO,
            "related_to": EdgeType.RELATED_TO,
        }

        mapped_edge_type = edge_type_mapping.get(edge_type, EdgeType.RELATED_TO)

        knowledge_edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            type=mapped_edge_type,
            properties=properties,
        )

        self.graph.add_edge(knowledge_edge)

        return Edge(
            id=knowledge_edge.id,
            source_id=source_id,
            target_id=target_id,
            type=edge_type,
            properties=properties,
        )

    async def merge_nodes(self, nodes: List[Node]) -> Node:
        """Merge multiple nodes into one."""
        if not nodes:
            raise ValueError("Cannot merge empty node list")

        # Use the first node as base
        base_node = nodes[0]
        merged_properties = dict(base_node.properties)

        # Merge properties from other nodes
        for node in nodes[1:]:
            for key, value in node.properties.items():
                if key not in merged_properties:
                    merged_properties[key] = value
                elif key == "aliases":
                    # Merge aliases
                    existing_aliases = merged_properties.get("aliases", [])
                    new_aliases = value if isinstance(value, list) else [value]
                    merged_properties["aliases"] = list(
                        set(existing_aliases + new_aliases)
                    )

        merged_properties["merged"] = True
        merged_properties["merged_from"] = [node.id for node in nodes]

        # Create new merged node
        return Node(
            id=f"merged_{base_node.id}",
            type=base_node.type,
            properties=merged_properties,
        )

    async def update_from_conversation(self, conversation, message):
        """Update graph from conversation - placeholder implementation."""
        logger.debug(f"Updating graph from conversation {conversation.id}")


class EntityNodeMapper:
    """Maps entities to knowledge graph nodes."""

    def __init__(self, graph_engine: GraphEngine):
        self.graph_engine = graph_engine
        self.mapping_cache: Dict[str, NodeMapping] = {}
        self.similarity_threshold = 0.8
        self.enable_deduplication = False

    async def map_entity(
        self, entity: Entity, context: Optional[Dict[str, Any]] = None
    ) -> Optional[NodeMapping]:
        """Map a single entity to a knowledge graph node."""
        try:
            # Check cache first
            cache_key = f"{entity.text}_{entity.type.value}"
            if cache_key in self.mapping_cache:
                return self.mapping_cache[cache_key]

            # Try exact match first
            nodes = await self.graph_engine.find_nodes_by_name(entity.text)
            if nodes:
                mapping = NodeMapping(
                    entity=entity,
                    node=nodes[0],
                    confidence=0.95,
                    strategy=MappingStrategy.EXACT_MATCH,
                )
                self.mapping_cache[cache_key] = mapping
                return mapping

            # Try fuzzy/semantic match
            similar_nodes = await self.graph_engine.search_similar_nodes(
                entity.text, context
            )
            if similar_nodes:
                # Check if similarity is above threshold
                best_node = similar_nodes[0]
                similarity = best_node.properties.get("similarity", 0)

                if similarity >= self.similarity_threshold:
                    strategy = (
                        MappingStrategy.SEMANTIC_MATCH
                        if context
                        else MappingStrategy.FUZZY_MATCH
                    )
                    confidence = similarity * 0.9  # Reduce confidence for fuzzy matches

                    mapping = NodeMapping(
                        entity=entity,
                        node=best_node,
                        confidence=confidence,
                        strategy=strategy,
                    )
                    self.mapping_cache[cache_key] = mapping
                    return mapping

            # Create new node
            node_type = self._entity_type_to_node_type(entity.type)
            properties = {
                "name": entity.text,
                "entity_type": entity.type.value,
                "confidence": entity.confidence,
                "created_from": "entity_extraction",
            }

            new_node = await self.graph_engine.create_node(node_type, properties)

            mapping = NodeMapping(
                entity=entity,
                node=new_node,
                confidence=entity.confidence,
                strategy=MappingStrategy.CREATE_NEW,
            )
            self.mapping_cache[cache_key] = mapping
            return mapping

        except Exception as e:
            logger.error(f"Error mapping entity {entity.text}: {e}")
            return None

    async def map_entities(self, entities: List[Entity]) -> List[NodeMapping]:
        """Map multiple entities to nodes."""
        mappings = []
        for entity in entities:
            mapping = await self.map_entity(entity)
            if mapping:
                mappings.append(mapping)
        return mappings

    async def map_entities_bulk(self, entities: List[Entity]) -> MappingResult:
        """Map entities in bulk with performance metrics."""
        start_time = time.time()

        mappings = await self.map_entities(entities)

        execution_time = time.time() - start_time
        successful_mappings = len(mappings)
        failed_mappings = len(entities) - successful_mappings

        return MappingResult(
            mappings=mappings,
            total_entities=len(entities),
            successful_mappings=successful_mappings,
            failed_mappings=failed_mappings,
            execution_time=execution_time,
        )

    async def map_relationship(
        self, relationship: Relationship, entity_mappings: List[NodeMapping]
    ) -> Optional[Edge]:
        """Map a relationship to a knowledge graph edge."""
        # Find the mappings for source and target entities
        source_mapping = None
        target_mapping = None

        for mapping in entity_mappings:
            if mapping.entity == relationship.source:
                source_mapping = mapping
            elif mapping.entity == relationship.target:
                target_mapping = mapping

        if not source_mapping or not target_mapping:
            logger.warning("Could not find mappings for relationship entities")
            return None

        # Create edge
        edge = await self.graph_engine.create_edge(
            source_id=source_mapping.node.id,
            target_id=target_mapping.node.id,
            edge_type=relationship.type,
            properties={"confidence": relationship.confidence},
        )

        return edge

    def _entity_type_to_node_type(self, entity_type: EntityType) -> str:
        """Map EntityType to node type string."""
        mapping = {
            EntityType.PERSON: "Person",
            EntityType.ORGANIZATION: "Organization",
            EntityType.TECHNOLOGY: "Technology",
            EntityType.CONCEPT: "Concept",
            EntityType.LOCATION: "Location",
            EntityType.DATE: "Date",
            EntityType.EVENT: "Event",
            EntityType.PRODUCT: "Product",
        }
        return mapping.get(entity_type, "Entity")

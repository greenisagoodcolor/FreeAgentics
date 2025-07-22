"""Evolution engine for knowledge graph mutation and adaptation.

This module implements algorithms for evolving knowledge graphs based on
new observations, agent interactions, and learning.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from knowledge_graph.graph_engine import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
)

logger = logging.getLogger(__name__)


@dataclass
class EvolutionMetrics:
    """Metrics for tracking graph evolution."""

    nodes_added: int = 0
    nodes_removed: int = 0
    nodes_updated: int = 0
    edges_added: int = 0
    edges_removed: int = 0
    confidence_changes: int = 0
    communities_merged: int = 0
    contradictions_resolved: int = 0


class MutationOperator(ABC):
    """Abstract base class for graph mutation operators."""

    @abstractmethod
    def apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> EvolutionMetrics:
        """Apply mutation to the graph.

        Args:
            graph: Knowledge graph to mutate
            context: Context information for mutation

        Returns:
            Metrics about changes made
        """
        pass

    @abstractmethod
    def can_apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> bool:
        """Check if this mutation can be applied.

        Args:
            graph: Knowledge graph
            context: Context information

        Returns:
            True if mutation is applicable
        """
        pass


class ObservationIntegrator(MutationOperator):
    """Integrate new observations into the knowledge graph."""

    def can_apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> bool:
        """Check if observations are available to integrate."""
        return "observations" in context and len(context["observations"]) > 0

    def apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> EvolutionMetrics:
        """Integrate observations into the graph."""
        metrics = EvolutionMetrics()
        observations = context.get("observations", [])
        observer_id = context.get("observer_id", "unknown")

        for obs in observations:
            # Create observation node
            obs_node = KnowledgeNode(
                type=NodeType.OBSERVATION,
                label=f"obs_{observer_id}_{datetime.now().timestamp()}",
                properties={
                    "data": obs.get("data"),
                    "timestamp": obs.get("timestamp", datetime.now()),
                    "location": obs.get("location"),
                    "confidence": obs.get("confidence", 0.8),
                },
                source=observer_id,
                confidence=obs.get("confidence", 0.8),
            )

            if graph.add_node(obs_node):
                metrics.nodes_added += 1

            # Link observation to observed entities
            entity_id = obs.get("entity_id")
            if entity_id and entity_id in graph.nodes:
                edge = KnowledgeEdge(
                    source_id=obs_node.id,  # Changed from observer_id to obs_node.id
                    target_id=entity_id,
                    type=EdgeType.OBSERVES,
                    properties={"observation_id": obs_node.id},
                )
                if graph.add_edge(edge):
                    metrics.edges_added += 1

            # Update entity properties based on observation
            if entity_id and "properties" in obs:
                graph.update_node(entity_id, obs["properties"])
                metrics.nodes_updated += 1

        logger.info(f"Integrated {len(observations)} observations")
        return metrics


class BeliefUpdater(MutationOperator):
    """Update agent beliefs based on new evidence."""

    def can_apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> bool:
        """Check if belief updates are needed."""
        return "agent_id" in context and "evidence" in context

    def apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> EvolutionMetrics:
        """Update beliefs based on evidence."""
        metrics = EvolutionMetrics()
        agent_id = context["agent_id"]
        evidence = context["evidence"]

        # Find existing beliefs
        belief_nodes = [
            n for n in graph.find_nodes_by_type(NodeType.BELIEF) if n.source == agent_id
        ]

        for belief in belief_nodes:
            # Check if evidence contradicts belief
            if self._contradicts(belief, evidence):
                # Reduce confidence
                old_confidence = belief.confidence
                belief.confidence *= 0.7  # Reduce by 30%

                if belief.confidence < 0.2:
                    # Remove low-confidence beliefs
                    graph.remove_node(belief.id)
                    metrics.nodes_removed += 1
                else:
                    metrics.confidence_changes += 1

                logger.debug(
                    f"Reduced belief confidence: {old_confidence:.2f} -> {belief.confidence:.2f}"
                )

            elif self._supports(belief, evidence):
                # Increase confidence
                old_confidence = belief.confidence
                belief.confidence = min(1.0, belief.confidence * 1.2)
                metrics.confidence_changes += 1

                logger.debug(
                    f"Increased belief confidence: {old_confidence:.2f} -> {belief.confidence:.2f}"
                )

        # Add new beliefs from evidence
        for fact in evidence.get("facts", []):
            belief_node = KnowledgeNode(
                type=NodeType.BELIEF,
                label=fact.get("label", "belie"),
                properties=fact.get("properties", {}),
                source=agent_id,
                confidence=fact.get("confidence", 0.6),
            )

            if graph.add_node(belief_node):
                metrics.nodes_added += 1

                # Link to related entities
                for entity_id in fact.get("entities", []):
                    if entity_id in graph.nodes:
                        edge = KnowledgeEdge(
                            source_id=agent_id,
                            target_id=belief_node.id,
                            type=EdgeType.BELIEVES,
                        )
                        if graph.add_edge(edge):
                            metrics.edges_added += 1

        return metrics

    def _contradicts(self, belief: KnowledgeNode, evidence: Dict[str, Any]) -> bool:
        """Check if evidence contradicts a belief."""
        # Simplified contradiction check
        contradictions = evidence.get("contradictions", [])
        for contradiction in contradictions:
            if belief.label == contradiction.get("belief_label"):
                return True
            if any(
                belief.properties.get(k) != v
                for k, v in contradiction.get("properties", {}).items()
            ):
                return True
        return False

    def _supports(self, belief: KnowledgeNode, evidence: Dict[str, Any]) -> bool:
        """Check if evidence supports a belief."""
        supports = evidence.get("supports", [])
        for support in supports:
            if belief.label == support.get("belief_label"):
                return True
            if all(belief.properties.get(k) == v for k, v in support.get("properties", {}).items()):
                return True
        return False


class ConceptGeneralizer(MutationOperator):
    """Generalize from specific instances to abstract concepts."""

    def can_apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> bool:
        """Check if there are enough instances to generalize."""
        min_instances = context.get("min_instances", 3)
        entity_nodes = graph.find_nodes_by_type(NodeType.ENTITY)

        # Group by similar properties
        property_groups: Dict[frozenset, List[Any]] = {}
        for node in entity_nodes:
            key = frozenset(node.properties.keys())
            if key not in property_groups:
                property_groups[key] = []
            property_groups[key].append(node)

        return any(len(group) >= min_instances for group in property_groups.values())

    def apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> EvolutionMetrics:
        """Create concept nodes from similar entities."""
        metrics = EvolutionMetrics()
        min_instances = context.get("min_instances", 3)
        similarity_threshold = context.get("similarity_threshold", 0.7)

        entity_nodes = graph.find_nodes_by_type(NodeType.ENTITY)

        # Find clusters of similar entities
        clusters = self._cluster_entities(entity_nodes, similarity_threshold)

        for cluster in clusters:
            if len(cluster) >= min_instances:
                # Create concept node
                concept_properties = self._extract_common_properties(cluster)
                concept_node = KnowledgeNode(
                    type=NodeType.CONCEPT,
                    label=f"concept_{concept_properties.get('type', 'unknown')}",
                    properties=concept_properties,
                    confidence=0.8,
                )

                if graph.add_node(concept_node):
                    metrics.nodes_added += 1

                    # Link instances to concept
                    for entity in cluster:
                        edge = KnowledgeEdge(
                            source_id=entity.id,
                            target_id=concept_node.id,
                            type=EdgeType.IS_A,
                            confidence=0.9,
                        )
                        if graph.add_edge(edge):
                            metrics.edges_added += 1

        return metrics

    def _cluster_entities(
        self, entities: List[KnowledgeNode], threshold: float
    ) -> List[List[KnowledgeNode]]:
        """Cluster entities by similarity."""
        clusters = []
        used = set()

        for i, entity1 in enumerate(entities):
            if entity1.id in used:
                continue

            cluster = [entity1]
            used.add(entity1.id)

            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                if entity2.id not in used:
                    similarity = self._calculate_similarity(entity1, entity2)
                    if similarity >= threshold:
                        cluster.append(entity2)
                        used.add(entity2.id)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _calculate_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        """Calculate similarity between two nodes."""
        # Property key overlap
        keys1 = set(node1.properties.keys())
        keys2 = set(node2.properties.keys())
        key_similarity = len(keys1 & keys2) / max(len(keys1 | keys2), 1)

        # Property value similarity
        common_keys = keys1 & keys2
        if common_keys:
            value_matches = sum(
                1 for k in common_keys if node1.properties[k] == node2.properties[k]
            )
            value_similarity = value_matches / len(common_keys)
        else:
            value_similarity = 0

        # Label similarity (simple string matching)
        label_similarity = 1.0 if node1.label == node2.label else 0.0

        # Weighted average
        return 0.3 * key_similarity + 0.5 * value_similarity + 0.2 * label_similarity

    def _extract_common_properties(self, nodes: List[KnowledgeNode]) -> Dict[str, Any]:
        """Extract common properties from a cluster of nodes."""
        if not nodes:
            return {}

        # Find properties present in all nodes
        common_keys = set(nodes[0].properties.keys())
        for node in nodes[1:]:
            common_keys &= set(node.properties.keys())

        # Extract common values
        common_properties = {}
        for key in common_keys:
            values = [node.properties[key] for node in nodes]
            # Use most common value
            unique_values = list(set(values))
            if len(unique_values) == 1:
                common_properties[key] = unique_values[0]
            else:
                # Use most frequent
                from collections import Counter

                common_properties[key] = Counter(values).most_common(1)[0][0]

        return common_properties


class CausalLearner(MutationOperator):
    """Learn causal relationships from temporal patterns."""

    def can_apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> bool:
        """Check if there are temporal patterns to analyze."""
        return "temporal_events" in context and len(context["temporal_events"]) > 1

    def apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> EvolutionMetrics:
        """Identify and add causal relationships."""
        metrics = EvolutionMetrics()
        events = context.get("temporal_events", [])
        confidence_threshold = context.get("causality_threshold", 0.7)

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.get("timestamp", 0))

        # Look for patterns
        for i in range(len(sorted_events) - 1):
            event1 = sorted_events[i]
            event2 = sorted_events[i + 1]

            # Check temporal proximity
            time_diff = event2["timestamp"] - event1["timestamp"]
            if time_diff < context.get("max_causal_delay", 10):
                # Calculate causality confidence
                confidence = self._calculate_causality_confidence(event1, event2, sorted_events)

                if confidence >= confidence_threshold:
                    # Create event nodes if needed
                    node1_id = self._ensure_event_node(graph, event1)
                    node2_id = self._ensure_event_node(graph, event2)

                    if node1_id and node2_id:
                        # Add causal edge
                        edge = KnowledgeEdge(
                            source_id=node1_id,
                            target_id=node2_id,
                            type=EdgeType.CAUSES,
                            properties={
                                "time_delay": time_diff,
                                "confidence": confidence,
                            },
                            confidence=confidence,
                        )

                        if graph.add_edge(edge):
                            metrics.edges_added += 1

        return metrics

    def _calculate_causality_confidence(
        self,
        event1: Dict[str, Any],
        event2: Dict[str, Any],
        all_events: List[Dict[str, Any]],
    ) -> float:
        """Calculate confidence in causal relationship."""
        # Count co-occurrences
        co_occurrences = 0
        event1_occurrences = 0

        for i in range(len(all_events) - 1):
            if self._events_match(all_events[i], event1):
                event1_occurrences += 1
                # Check if event2 follows
                for j in range(i + 1, min(i + 5, len(all_events))):
                    if self._events_match(all_events[j], event2):
                        co_occurrences += 1
                        break

        if event1_occurrences == 0:
            return 0.0

        # Basic conditional probability
        return co_occurrences / event1_occurrences

    def _events_match(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
        """Check if two events match (same type and key properties)."""
        if event1.get("type") != event2.get("type"):
            return False

        # Check key properties
        key_props = ["action", "entity_id", "location"]
        for prop in key_props:
            if prop in event1 and prop in event2:
                if event1[prop] != event2[prop]:
                    return False

        return True

    def _ensure_event_node(self, graph: KnowledgeGraph, event: Dict[str, Any]) -> Optional[str]:
        """Ensure event exists as node in graph."""
        # Check if event node already exists
        event_id_raw = event.get("id")
        if isinstance(event_id_raw, str) and event_id_raw in graph.nodes:
            return event_id_raw

        # Create new event node
        node = KnowledgeNode(
            type=NodeType.EVENT,
            label=event.get("type", "event"),
            properties={
                "timestamp": event.get("timestamp"),
                "action": event.get("action"),
                "entity_id": event.get("entity_id"),
                "location": event.get("location"),
                "data": event.get("data", {}),
            },
        )

        if graph.add_node(node):
            return node.id

        return None


class ContradictionResolver(MutationOperator):
    """Resolve contradictions in the knowledge graph."""

    def can_apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> bool:
        """Check if there are contradictions to resolve."""
        # Simple check: look for nodes with conflicting properties
        nodes_by_subject: Dict[Any, List[Any]] = {}
        for node in graph.nodes.values():
            subject = node.properties.get("subject")
            if subject:
                if subject not in nodes_by_subject:
                    nodes_by_subject[subject] = []
                nodes_by_subject[subject].append(node)

        # Check for conflicts
        for nodes in nodes_by_subject.values():
            if len(nodes) > 1 and self._has_conflicts(nodes):
                return True

        return False

    def apply(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> EvolutionMetrics:
        """Resolve contradictions in the graph."""
        metrics = EvolutionMetrics()
        resolution_strategy = context.get("resolution_strategy", "highest_confidence")

        # Find contradictory nodes
        contradictions = self._find_contradictions(graph)

        for contradiction_set in contradictions:
            if resolution_strategy == "highest_confidence":
                # Keep node with highest confidence
                sorted_nodes = sorted(contradiction_set, key=lambda n: n.confidence, reverse=True)
                keeper = sorted_nodes[0]

                # Remove others
                for node in sorted_nodes[1:]:
                    graph.remove_node(node.id)
                    metrics.nodes_removed += 1

            elif resolution_strategy == "merge":
                # Merge properties
                merged_properties = {}
                total_confidence = 0.0

                for node in contradiction_set:
                    for key, value in node.properties.items():
                        if key not in merged_properties:
                            merged_properties[key] = value
                        # For conflicts, use weighted average or most confident
                        elif isinstance(value, (int, float)):
                            # Weighted average for numeric properties
                            if key not in merged_properties:
                                merged_properties[key] = 0
                            merged_properties[key] += value * node.confidence
                            total_confidence += node.confidence

                # Normalize numeric properties
                if total_confidence > 0:
                    for key, value in merged_properties.items():
                        if isinstance(value, (int, float)):
                            merged_properties[key] = value / total_confidence

                # Update first node, remove others
                keeper = contradiction_set[0]
                graph.update_node(keeper.id, merged_properties)
                metrics.nodes_updated += 1

                for node in contradiction_set[1:]:
                    # Transfer edges
                    for edge in graph.edges.values():
                        if edge.source_id == node.id:
                            edge.source_id = keeper.id
                        elif edge.target_id == node.id:
                            edge.target_id = keeper.id

                    graph.remove_node(node.id)
                    metrics.nodes_removed += 1

            metrics.contradictions_resolved += 1

        return metrics

    def _find_contradictions(self, graph: KnowledgeGraph) -> List[List[KnowledgeNode]]:
        """Find sets of contradictory nodes."""
        contradictions = []
        processed = set()

        for node1 in graph.nodes.values():
            if node1.id in processed:
                continue

            contradiction_set = [node1]
            processed.add(node1.id)

            for node2 in graph.nodes.values():
                if node2.id != node1.id and node2.id not in processed:
                    if self._are_contradictory(node1, node2):
                        contradiction_set.append(node2)
                        processed.add(node2.id)

            if len(contradiction_set) > 1:
                contradictions.append(contradiction_set)

        return contradictions

    def _has_conflicts(self, nodes: List[KnowledgeNode]) -> bool:
        """Check if a set of nodes has conflicts."""
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self._are_contradictory(nodes[i], nodes[j]):
                    return True
        return False

    def _are_contradictory(self, node1: KnowledgeNode, node2: KnowledgeNode) -> bool:
        """Check if two nodes are contradictory."""
        # Same subject but conflicting properties
        subject1 = node1.properties.get("subject")
        subject2 = node2.properties.get("subject")

        if subject1 and subject1 == subject2:
            # Check for conflicting boolean properties
            for key in set(node1.properties.keys()) & set(node2.properties.keys()):
                val1 = node1.properties[key]
                val2 = node2.properties[key]

                if isinstance(val1, bool) and isinstance(val2, bool):
                    if val1 != val2:
                        return True

                # Check for mutually exclusive categories
                if key in ["state", "status", "type"]:
                    if val1 != val2:
                        return True

        return False


class EvolutionEngine:
    """Engine for evolving knowledge graphs."""

    def __init__(self):
        """Initialize evolution engine with default operators."""
        self.operators: List[MutationOperator] = [
            ObservationIntegrator(),
            BeliefUpdater(),
            ConceptGeneralizer(),
            CausalLearner(),
            ContradictionResolver(),
        ]

        self.evolution_history: List[Dict[str, Any]] = []

    def add_operator(self, operator: MutationOperator):
        """Add a custom mutation operator.

        Args:
            operator: Mutation operator to add
        """
        self.operators.append(operator)

    def evolve(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> EvolutionMetrics:
        """Evolve the knowledge graph based on context.

        Args:
            graph: Knowledge graph to evolve
            context: Context information for evolution

        Returns:
            Aggregated evolution metrics
        """
        total_metrics = EvolutionMetrics()
        applied_operators = []

        # Apply each applicable operator
        for operator in self.operators:
            if operator.can_apply(graph, context):
                try:
                    metrics = operator.apply(graph, context)

                    # Aggregate metrics
                    total_metrics.nodes_added += metrics.nodes_added
                    total_metrics.nodes_removed += metrics.nodes_removed
                    total_metrics.nodes_updated += metrics.nodes_updated
                    total_metrics.edges_added += metrics.edges_added
                    total_metrics.edges_removed += metrics.edges_removed
                    total_metrics.confidence_changes += metrics.confidence_changes
                    total_metrics.communities_merged += metrics.communities_merged
                    total_metrics.contradictions_resolved += metrics.contradictions_resolved

                    applied_operators.append(operator.__class__.__name__)

                except Exception as e:
                    logger.error(f"Error applying {operator.__class__.__name__}: {e}")

        # Record evolution history
        self.evolution_history.append(
            {
                "timestamp": datetime.now(),
                "graph_version": graph.version,
                "context": context,
                "applied_operators": applied_operators,
                "metrics": total_metrics,
            }
        )

        logger.info(f"Evolution complete. Applied {len(applied_operators)} operators")
        return total_metrics

    def suggest_evolution(self, graph: KnowledgeGraph, context: Dict[str, Any]) -> List[str]:
        """Suggest which operators would be beneficial.

        Args:
            graph: Knowledge graph
            context: Current context

        Returns:
            List of suggested operator names
        """
        suggestions = []

        for operator in self.operators:
            if operator.can_apply(graph, context):
                suggestions.append(operator.__class__.__name__)

        return suggestions

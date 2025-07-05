"""
Test suite for Knowledge Graph Evolution system.

Tests the graph engine, evolution operators, query system, and storage.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest

from knowledge_graph.evolution import (
    BeliefUpdater,
    CausalLearner,
    ConceptGeneralizer,
    ContradictionResolver,
    EvolutionEngine,
    ObservationIntegrator,
)
from knowledge_graph.graph_engine import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
)
from knowledge_graph.query import GraphQuery, QueryEngine, QueryType
from knowledge_graph.storage import (
    DatabaseStorageBackend,
    FileStorageBackend,
    PickleStorageBackend,
    StorageManager,
)
from tests.db_infrastructure.fixtures import db_session

# Import test infrastructure
from tests.db_infrastructure.test_config import DatabaseTestCase


class TestKnowledgeNode:
    """Test KnowledgeNode functionality."""

    def test_node_creation(self):
        """Test creating a knowledge node."""
        node = KnowledgeNode(
            type=NodeType.ENTITY,
            label="test_entity",
            properties={"name": "Test", "value": 42},
            confidence=0.8,
            source="agent_1",
        )

        assert node.id is not None
        assert node.type == NodeType.ENTITY
        assert node.label == "test_entity"
        assert node.properties["name"] == "Test"
        assert node.properties["value"] == 42
        assert node.confidence == 0.8
        assert node.source == "agent_1"
        assert node.version == 1

    def test_node_update(self):
        """Test updating node properties."""
        node = KnowledgeNode(type=NodeType.ENTITY, label="test")
        original_updated_at = node.updated_at
        original_version = node.version

        node.update({"new_prop": "value"})

        assert node.properties["new_prop"] == "value"
        assert node.version == original_version + 1
        assert node.updated_at > original_updated_at

    def test_node_to_dict(self):
        """Test converting node to dictionary."""
        node = KnowledgeNode(
            type=NodeType.CONCEPT, label="test_concept", properties={"abstract": True}
        )

        node_dict = node.to_dict()

        assert node_dict["id"] == node.id
        assert node_dict["type"] == "concept"
        assert node_dict["label"] == "test_concept"
        assert node_dict["properties"]["abstract"] is True
        assert "created_at" in node_dict
        assert "updated_at" in node_dict


class TestKnowledgeEdge:
    """Test KnowledgeEdge functionality."""

    def test_edge_creation(self):
        """Test creating a knowledge edge."""
        edge = KnowledgeEdge(
            source_id="node1",
            target_id="node2",
            type=EdgeType.RELATED_TO,
            properties={"strength": 0.7},
            confidence=0.9,
        )

        assert edge.id is not None
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.type == EdgeType.RELATED_TO
        assert edge.properties["strength"] == 0.7
        assert edge.confidence == 0.9

    def test_edge_to_dict(self):
        """Test converting edge to dictionary."""
        edge = KnowledgeEdge(source_id="a", target_id="b", type=EdgeType.CAUSES)

        edge_dict = edge.to_dict()

        assert edge_dict["id"] == edge.id
        assert edge_dict["source_id"] == "a"
        assert edge_dict["target_id"] == "b"
        assert edge_dict["type"] == "causes"


class TestKnowledgeGraph:
    """Test KnowledgeGraph functionality."""

    def test_graph_creation(self):
        """Test creating a knowledge graph."""
        graph = KnowledgeGraph()

        assert graph.graph_id is not None
        assert graph.version == 1
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = KnowledgeGraph()

        node1 = KnowledgeNode(type=NodeType.ENTITY, label="entity1")
        node2 = KnowledgeNode(type=NodeType.ENTITY, label="entity2")

        assert graph.add_node(node1) is True
        assert graph.add_node(node2) is True
        assert len(graph.nodes) == 2

        # Test duplicate node
        assert graph.add_node(node1) is False
        assert len(graph.nodes) == 2

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = KnowledgeGraph()

        node1 = KnowledgeNode(type=NodeType.ENTITY, label="entity1")
        node2 = KnowledgeNode(type=NodeType.ENTITY, label="entity2")

        graph.add_node(node1)
        graph.add_node(node2)

        edge = KnowledgeEdge(source_id=node1.id, target_id=node2.id, type=EdgeType.RELATED_TO)

        assert graph.add_edge(edge) is True
        assert len(graph.edges) == 1

        # Test invalid edge (missing nodes)
        bad_edge = KnowledgeEdge(
            source_id="missing", target_id="also_missing", type=EdgeType.RELATED_TO
        )
        assert graph.add_edge(bad_edge) is False

    def test_update_node(self):
        """Test updating nodes in graph."""
        graph = KnowledgeGraph()

        node = KnowledgeNode(type=NodeType.ENTITY, label="test")
        graph.add_node(node)

        original_version = graph.version

        assert graph.update_node(node.id, {"new_prop": "value"}) is True

        updated_node = graph.get_node(node.id)
        assert updated_node.properties["new_prop"] == "value"
        assert graph.version > original_version

    def test_remove_node(self):
        """Test removing nodes from graph."""
        graph = KnowledgeGraph()

        node1 = KnowledgeNode(type=NodeType.ENTITY, label="entity1")
        node2 = KnowledgeNode(type=NodeType.ENTITY, label="entity2")

        graph.add_node(node1)
        graph.add_node(node2)

        edge = KnowledgeEdge(source_id=node1.id, target_id=node2.id, type=EdgeType.RELATED_TO)
        graph.add_edge(edge)

        # Remove node should also remove connected edges
        assert graph.remove_node(node1.id) is True
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0

    def test_get_neighbors(self):
        """Test getting neighboring nodes."""
        graph = KnowledgeGraph()

        # Create a simple graph: n1 -> n2 -> n3
        nodes = []
        for i in range(3):
            node = KnowledgeNode(type=NodeType.ENTITY, label=f"entity{i}")
            graph.add_node(node)
            nodes.append(node)

        graph.add_edge(
            KnowledgeEdge(source_id=nodes[0].id, target_id=nodes[1].id, type=EdgeType.RELATED_TO)
        )

        graph.add_edge(
            KnowledgeEdge(source_id=nodes[1].id, target_id=nodes[2].id, type=EdgeType.CAUSES)
        )

        # Test neighbors
        neighbors = graph.get_neighbors(nodes[0].id)
        assert len(neighbors) == 1
        assert nodes[1].id in neighbors

        # Test filtered neighbors
        neighbors = graph.get_neighbors(nodes[1].id, edge_type=EdgeType.CAUSES)
        assert len(neighbors) == 1
        assert nodes[2].id in neighbors

    def test_find_path(self):
        """Test path finding in graph."""
        graph = KnowledgeGraph()

        # Create a graph with path: n1 -> n2 -> n3
        nodes = []
        for i in range(3):
            node = KnowledgeNode(type=NodeType.ENTITY, label=f"entity{i}")
            graph.add_node(node)
            nodes.append(node)

        graph.add_edge(
            KnowledgeEdge(source_id=nodes[0].id, target_id=nodes[1].id, type=EdgeType.RELATED_TO)
        )

        graph.add_edge(
            KnowledgeEdge(source_id=nodes[1].id, target_id=nodes[2].id, type=EdgeType.RELATED_TO)
        )

        # Find path
        path = graph.find_path(nodes[0].id, nodes[2].id)
        assert path is not None
        assert len(path) == 3
        assert path == [nodes[0].id, nodes[1].id, nodes[2].id]

        # No path case
        isolated_node = KnowledgeNode(type=NodeType.ENTITY, label="isolated")
        graph.add_node(isolated_node)

        path = graph.find_path(nodes[0].id, isolated_node.id)
        assert path is None

    def test_node_importance(self):
        """Test calculating node importance."""
        graph = KnowledgeGraph()

        # Create hub-and-spoke pattern
        hub = KnowledgeNode(type=NodeType.ENTITY, label="hub")
        graph.add_node(hub)

        spokes = []
        for i in range(4):
            spoke = KnowledgeNode(type=NodeType.ENTITY, label=f"spoke{i}")
            graph.add_node(spoke)
            spokes.append(spoke)

            graph.add_edge(
                KnowledgeEdge(source_id=hub.id, target_id=spoke.id, type=EdgeType.RELATED_TO)
            )

        # Calculate importance
        importance = graph.calculate_node_importance()

        # Hub should have highest importance
        assert importance[hub.id] > importance[spokes[0].id]

    def test_merge_graphs(self):
        """Test merging two knowledge graphs."""
        graph1 = KnowledgeGraph()
        graph2 = KnowledgeGraph()

        # Add nodes to graph1
        node1 = KnowledgeNode(type=NodeType.ENTITY, label="shared", confidence=0.7)
        node2 = KnowledgeNode(type=NodeType.ENTITY, label="unique1")
        graph1.add_node(node1)
        graph1.add_node(node2)

        # Add nodes to graph2
        node3 = KnowledgeNode(
            id=node1.id,  # Same ID as node1
            type=NodeType.ENTITY,
            label="shared",
            confidence=0.9,  # Higher confidence
        )
        node4 = KnowledgeNode(type=NodeType.ENTITY, label="unique2")
        graph2.add_node(node3)
        graph2.add_node(node4)

        # Merge with higher confidence resolution
        original_count = len(graph1.nodes)
        graph1.merge(graph2, conflict_resolution="higher_confidence")

        # Should have 3 nodes total
        assert len(graph1.nodes) == 3

        # Shared node should have higher confidence
        merged_node = graph1.get_node(node1.id)
        assert merged_node.confidence == 0.9


class TestEvolutionOperators:
    """Test knowledge graph evolution operators."""

    def test_observation_integrator(self):
        """Test integrating observations into graph."""
        graph = KnowledgeGraph()
        integrator = ObservationIntegrator()

        # Add an entity to observe
        entity = KnowledgeNode(type=NodeType.ENTITY, label="target_entity")
        graph.add_node(entity)

        context = {
            "observer_id": "sensor_1",
            "observations": [
                {
                    "entity_id": entity.id,
                    "data": {"temperature": 25.5},
                    "timestamp": datetime.now(),
                    "confidence": 0.95,
                    "properties": {"state": "active"},
                }
            ],
        }

        assert integrator.can_apply(graph, context) is True

        metrics = integrator.apply(graph, context)

        assert metrics.nodes_added > 0  # Observation node added
        assert metrics.edges_added > 0  # Observation edge added
        assert metrics.nodes_updated > 0  # Entity updated

        # Check entity was updated
        updated_entity = graph.get_node(entity.id)
        assert updated_entity.properties["state"] == "active"

    def test_belief_updater(self):
        """Test updating agent beliefs."""
        graph = KnowledgeGraph()
        updater = BeliefUpdater()

        # Add existing belief
        belief = KnowledgeNode(
            type=NodeType.BELIEF,
            label="weather_sunny",
            properties={"weather": "sunny"},
            source="agent_1",
            confidence=0.8,
        )
        graph.add_node(belief)

        # Evidence that contradicts belief
        context = {
            "agent_id": "agent_1",
            "evidence": {
                "contradictions": [
                    {"belief_label": "weather_sunny", "properties": {"weather": "rainy"}}
                ]
            },
        }

        assert updater.can_apply(graph, context) is True

        metrics = updater.apply(graph, context)

        assert metrics.confidence_changes > 0

        # Check belief confidence was reduced
        updated_belief = graph.get_node(belief.id)
        assert updated_belief.confidence < 0.8

    def test_concept_generalizer(self):
        """Test generalizing concepts from entities."""
        graph = KnowledgeGraph()
        generalizer = ConceptGeneralizer()

        # Add similar entities
        for i in range(5):
            entity = KnowledgeNode(
                type=NodeType.ENTITY,
                label=f"car_{i}",
                properties={
                    "type": "vehicle",
                    "wheels": 4,
                    "color": "red" if i % 2 == 0 else "blue",
                },
            )
            graph.add_node(entity)

        context = {"min_instances": 3, "similarity_threshold": 0.6}

        assert generalizer.can_apply(graph, context) is True

        metrics = generalizer.apply(graph, context)

        assert metrics.nodes_added > 0  # Concept node added
        assert metrics.edges_added > 0  # IS_A edges added

        # Check concept was created
        concepts = graph.find_nodes_by_type(NodeType.CONCEPT)
        assert len(concepts) > 0

        # Check concept has common properties
        concept = concepts[0]
        assert concept.properties.get("type") == "vehicle"
        assert concept.properties.get("wheels") == 4

    def test_causal_learner(self):
        """Test learning causal relationships."""
        graph = KnowledgeGraph()
        learner = CausalLearner()

        # Create temporal events
        base_time = datetime.now()
        events = []

        for i in range(5):
            # Pattern: button_press always followed by light_on
            events.append(
                {
                    "id": f"event_{i*2}",
                    "type": "button_press",
                    "timestamp": base_time.timestamp() + i * 10,
                    "action": "press",
                }
            )

            events.append(
                {
                    "id": f"event_{i*2+1}",
                    "type": "light_on",
                    "timestamp": base_time.timestamp() + i * 10 + 2,
                    "action": "activate",
                }
            )

        context = {"temporal_events": events, "causality_threshold": 0.7, "max_causal_delay": 5}

        assert learner.can_apply(graph, context) is True

        metrics = learner.apply(graph, context)

        assert metrics.edges_added > 0  # Causal edges added

        # Check causal edges exist
        causal_edges = [e for e in graph.edges.values() if e.type == EdgeType.CAUSES]
        assert len(causal_edges) > 0

    def test_contradiction_resolver(self):
        """Test resolving contradictions."""
        graph = KnowledgeGraph()
        resolver = ContradictionResolver()

        # Add contradictory beliefs about same subject
        belief1 = KnowledgeNode(
            type=NodeType.BELIEF,
            label="door_state",
            properties={"subject": "front_door", "state": "open"},
            confidence=0.6,
        )

        belief2 = KnowledgeNode(
            type=NodeType.BELIEF,
            label="door_state",
            properties={"subject": "front_door", "state": "closed"},
            confidence=0.9,
        )

        graph.add_node(belief1)
        graph.add_node(belief2)

        context = {"resolution_strategy": "highest_confidence"}

        assert resolver.can_apply(graph, context) is True

        metrics = resolver.apply(graph, context)

        assert metrics.contradictions_resolved > 0
        assert metrics.nodes_removed > 0

        # Only high confidence belief should remain
        assert len(graph.nodes) == 1
        remaining = list(graph.nodes.values())[0]
        assert remaining.properties["state"] == "closed"


class TestEvolutionEngine:
    """Test the evolution engine."""

    def test_evolution_engine(self):
        """Test complete evolution process."""
        graph = KnowledgeGraph()
        engine = EvolutionEngine()

        # Add some initial nodes
        entity = KnowledgeNode(type=NodeType.ENTITY, label="test_entity")
        graph.add_node(entity)

        # Complex context with multiple evolution triggers
        context = {
            "observer_id": "agent_1",
            "agent_id": "agent_1",
            "observations": [
                {"entity_id": entity.id, "data": {"status": "active"}, "confidence": 0.9}
            ],
            "evidence": {
                "facts": [
                    {
                        "label": "entity_active",
                        "properties": {"entity": entity.id, "active": True},
                        "entities": [entity.id],
                    }
                ]
            },
        }

        # Evolve graph
        metrics = engine.evolve(graph, context)

        # Should have applied multiple operators
        assert metrics.nodes_added > 0
        assert len(engine.evolution_history) > 0

        # Check suggestions
        suggestions = engine.suggest_evolution(graph, context)
        assert isinstance(suggestions, list)


class TestQueryEngine:
    """Test the query engine."""

    def test_node_lookup_query(self):
        """Test basic node lookup."""
        graph = KnowledgeGraph()

        # Add test nodes
        for i in range(5):
            node = KnowledgeNode(
                type=NodeType.ENTITY if i < 3 else NodeType.CONCEPT,
                label=f"node_{i}",
                properties={"value": i},
                confidence=0.5 + i * 0.1,
            )
            graph.add_node(node)

        engine = QueryEngine(graph)

        # Query by type
        query = GraphQuery(query_type=QueryType.NODE_LOOKUP, node_types=[NodeType.ENTITY])

        result = engine.execute(query)

        assert result.node_count() == 3
        assert all(n.type == NodeType.ENTITY for n in result.nodes)

        # Query with confidence threshold
        query = GraphQuery(query_type=QueryType.NODE_LOOKUP, confidence_threshold=0.7)

        result = engine.execute(query)
        assert all(n.confidence >= 0.7 for n in result.nodes)

    def test_path_query(self):
        """Test path finding queries."""
        graph = KnowledgeGraph()

        # Create chain: n1 -> n2 -> n3
        nodes = []
        for i in range(3):
            node = KnowledgeNode(type=NodeType.ENTITY, label=f"node_{i}")
            graph.add_node(node)
            nodes.append(node)

        for i in range(2):
            edge = KnowledgeEdge(
                source_id=nodes[i].id, target_id=nodes[i + 1].id, type=EdgeType.RELATED_TO
            )
            graph.add_edge(edge)

        engine = QueryEngine(graph)

        query = GraphQuery(
            query_type=QueryType.PATH_QUERY, source_id=nodes[0].id, target_id=nodes[2].id
        )

        result = engine.execute(query)

        assert len(result.paths) > 0
        assert result.paths[0] == [nodes[0].id, nodes[1].id, nodes[2].id]

    def test_neighborhood_query(self):
        """Test neighborhood exploration."""
        graph = KnowledgeGraph()

        # Create star pattern
        center = KnowledgeNode(type=NodeType.ENTITY, label="center")
        graph.add_node(center)

        neighbors = []
        for i in range(4):
            neighbor = KnowledgeNode(type=NodeType.ENTITY, label=f"neighbor_{i}")
            graph.add_node(neighbor)
            neighbors.append(neighbor)

            edge = KnowledgeEdge(
                source_id=center.id, target_id=neighbor.id, type=EdgeType.RELATED_TO
            )
            graph.add_edge(edge)

        engine = QueryEngine(graph)

        query = GraphQuery(query_type=QueryType.NEIGHBORHOOD, center_id=center.id, radius=1)

        result = engine.execute(query)

        # Should include center and all neighbors
        assert result.node_count() == 5
        assert len(result.edges) == 4

    def test_aggregate_query(self):
        """Test aggregation queries."""
        graph = KnowledgeGraph()

        # Add nodes with numeric properties
        for i in range(10):
            node = KnowledgeNode(
                type=NodeType.ENTITY,
                label=f"node_{i}",
                properties={"value": i * 10, "group": i % 2},
                confidence=0.5 + (i % 3) * 0.2,
            )
            graph.add_node(node)

        engine = QueryEngine(graph)

        query = GraphQuery(query_type=QueryType.AGGREGATE)

        result = engine.execute(query)

        assert result.aggregates["count"] == 10
        assert "avg_confidence" in result.aggregates
        assert "value_avg" in result.aggregates
        assert "value_std" in result.aggregates
        assert "type_distribution" in result.aggregates


class TestStorage:
    """Test storage backends."""

    def test_file_storage(self):
        """Test file-based storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStorageBackend(tmpdir)

            # Create and save graph
            graph = KnowledgeGraph()
            node = KnowledgeNode(type=NodeType.ENTITY, label="test")
            graph.add_node(node)

            assert storage.save_graph(graph) is True
            assert storage.graph_exists(graph.graph_id) is True

            # Load graph
            loaded = storage.load_graph(graph.graph_id)
            assert loaded is not None
            assert len(loaded.nodes) == 1
            assert loaded.nodes[node.id].label == "test"

            # List graphs
            graphs = storage.list_graphs()
            assert len(graphs) == 1
            assert graphs[0]["graph_id"] == graph.graph_id

            # Delete graph
            assert storage.delete_graph(graph.graph_id) is True
            assert storage.graph_exists(graph.graph_id) is False

    def test_database_storage(self, db_session):
        """Test database storage with real PostgreSQL."""
        # Use test database URL
        from tests.db_infrastructure.test_config import TEST_DATABASE_URL

        storage = DatabaseStorageBackend(TEST_DATABASE_URL)

        # Create and save graph
        graph = KnowledgeGraph()

        # Add nodes and edges
        node1 = KnowledgeNode(type=NodeType.ENTITY, label="node1")
        node2 = KnowledgeNode(type=NodeType.CONCEPT, label="node2")
        graph.add_node(node1)
        graph.add_node(node2)

        edge = KnowledgeEdge(source_id=node1.id, target_id=node2.id, type=EdgeType.IS_A)
        graph.add_edge(edge)

        assert storage.save_graph(graph) is True

        # Load graph
        loaded = storage.load_graph(graph.graph_id)
        assert loaded is not None
        assert len(loaded.nodes) == 2
        assert len(loaded.edges) == 1

        # Check relationships preserved
        assert loaded.get_neighbors(node1.id) == [node2.id]

        # Clean up - delete the graph
        storage.delete_graph(graph.graph_id)

    def test_pickle_storage(self):
        """Test pickle storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PickleStorageBackend(tmpdir)

            # Create complex graph
            graph = KnowledgeGraph()

            for i in range(5):
                node = KnowledgeNode(
                    type=NodeType.ENTITY,
                    label=f"node_{i}",
                    properties={"complex": {"nested": {"value": i}}},
                )
                graph.add_node(node)

            assert storage.save_graph(graph) is True

            # Load and verify
            loaded = storage.load_graph(graph.graph_id)
            assert loaded is not None
            assert len(loaded.nodes) == 5

            # Check complex properties preserved
            for node in loaded.nodes.values():
                assert "complex" in node.properties
                assert "nested" in node.properties["complex"]

    def test_storage_manager(self):
        """Test storage manager abstraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorageBackend(tmpdir)
            manager = StorageManager(backend)

            # Create and save graph
            graph = KnowledgeGraph()
            node = KnowledgeNode(type=NodeType.ENTITY, label="managed")
            graph.add_node(node)

            assert manager.save(graph) is True
            assert manager.exists(graph.graph_id) is True

            # Load graph
            loaded = manager.load(graph.graph_id)
            assert loaded is not None

            # List graphs
            graphs = manager.list()
            assert len(graphs) == 1

            # Delete graph
            assert manager.delete(graph.graph_id) is True
            assert manager.exists(graph.graph_id) is False

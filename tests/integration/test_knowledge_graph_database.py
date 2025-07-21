"""
Integration tests for Knowledge Graph with real PostgreSQL database.

This demonstrates migrating from SQLite/mocked storage to real PostgreSQL
for knowledge graph persistence and querying.
"""

from datetime import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from knowledge_graph.graph_engine import (
    EdgeType,
    KnowledgeGraph,
    NodeType,
)
from knowledge_graph.storage import DatabaseStorageBackend
from tests.db_infrastructure.factories import AgentFactory
from tests.db_infrastructure.fixtures import DatabaseTestCase
from tests.db_infrastructure.test_config import TEST_DATABASE_URL


class TestKnowledgeGraphDatabase(DatabaseTestCase):
    """Test knowledge graph with real PostgreSQL database."""

    @pytest.fixture
    def storage_backend(self):
        """Create database storage backend for tests."""
        return DatabaseStorageBackend(TEST_DATABASE_URL)

    def test_knowledge_graph_persistence(self, db_session: Session, storage_backend):
        """Test saving and loading knowledge graphs from PostgreSQL."""
        # Create a knowledge graph
        graph = KnowledgeGraph()

        # Add various node types
        nodes = []
        for i, node_type in enumerate(
            [NodeType.ENTITY, NodeType.CONCEPT, NodeType.BELIEF]
        ):
            node = graph.create_node(
                type=node_type,
                label=f"test_{node_type.value}_{i}",
                properties={
                    "description": f"Test {node_type.value}",
                    "confidence": 0.8 + i * 0.05,
                    "metadata": {"version": 1, "test": True},
                },
            )
            nodes.append(node)

        # Add edges between nodes
        edges = []
        edges.append(
            graph.create_edge(
                source_id=nodes[0].id,
                target_id=nodes[1].id,
                type=EdgeType.RELATED_TO,
                properties={"strength": 0.9},
            )
        )
        edges.append(
            graph.create_edge(
                source_id=nodes[1].id,
                target_id=nodes[2].id,
                type=EdgeType.CAUSES,
                properties={"probability": 0.75},
            )
        )

        # Save to database
        assert storage_backend.save_graph(graph)

        # Verify it's in PostgreSQL
        result = db_session.execute(
            text("SELECT COUNT(*) FROM knowledge_graphs WHERE graph_id = :id"),
            {"id": str(graph.graph_id)},
        ).scalar()
        assert result == 1

        # Load the graph back
        loaded_graph = storage_backend.load_graph(graph.graph_id)
        assert loaded_graph is not None
        assert len(loaded_graph.nodes) == 3
        assert len(loaded_graph.edges) == 2

        # Verify node properties are preserved
        for original, loaded in zip(nodes, loaded_graph.nodes.values()):
            assert loaded.label == original.label
            assert loaded.properties["confidence"] == original.properties["confidence"]
            assert loaded.properties["metadata"]["test"] is True

    def test_agent_knowledge_graph_integration(
        self, db_session: Session, storage_backend
    ):
        """Test integration between agents and their knowledge graphs."""
        # Create agents
        explorer = AgentFactory(name="Explorer Agent", template="explorer")
        analyst = AgentFactory(name="Analyst Agent", template="analyst")
        db_session.add_all([explorer, analyst])
        db_session.commit()

        # Create knowledge graph for explorer's discoveries
        explorer_graph = KnowledgeGraph()

        # Explorer discovers locations
        locations = []
        for i in range(3):
            location = explorer_graph.create_node(
                type=NodeType.ENTITY,
                label=f"location_{i}",
                properties={
                    "coordinates": [i * 10, i * 5],
                    "terrain": ["plains", "forest", "mountain"][i],
                    "discovered_by": str(explorer.id),
                    "discovery_time": datetime.utcnow().isoformat(),
                },
            )
            locations.append(location)

        # Add relationships between locations
        for i in range(len(locations) - 1):
            explorer_graph.create_edge(
                source_id=locations[i].id,
                target_id=locations[i + 1].id,
                type=EdgeType.CONNECTED_TO,
                properties={"distance": 15.0, "traversable": True},
            )

        # Save explorer's graph
        storage_backend.save_graph(explorer_graph)

        # Create knowledge graph for analyst's insights
        analyst_graph = KnowledgeGraph()

        # Analyst creates concepts based on explorer's discoveries
        pattern_concept = analyst_graph.create_node(
            type=NodeType.CONCEPT,
            label="terrain_pattern",
            properties={
                "pattern": "elevation_gradient",
                "confidence": 0.85,
                "derived_from": [str(loc.id) for loc in locations],
                "analyst": str(analyst.id),
            },
        )

        # Analyst creates beliefs
        belief = analyst_graph.create_node(
            type=NodeType.BELIEF,
            label="optimal_path_exists",
            properties={
                "belie": "A direct path through forests minimizes travel time",
                "confidence": 0.92,
                "evidence": ["terrain_analysis", "distance_calculations"],
                "created_by": str(analyst.id),
            },
        )

        # Link concept to belief
        analyst_graph.create_edge(
            source_id=pattern_concept.id,
            target_id=belief.id,
            type=EdgeType.SUPPORTS,
            properties={"support_strength": 0.8},
        )

        # Save analyst's graph
        storage_backend.save_graph(analyst_graph)

        # Create a merged graph combining both agents' knowledge
        merged_graph = KnowledgeGraph()

        # Load and merge both graphs
        loaded_explorer = storage_backend.load_graph(explorer_graph.graph_id)
        loaded_analyst = storage_backend.load_graph(analyst_graph.graph_id)

        merged_graph.merge(loaded_explorer)
        merged_graph.merge(loaded_analyst)

        # Add cross-agent relationships
        # Link location to concept
        location_id = list(loaded_explorer.nodes.keys())[0]
        concept_id = pattern_concept.id

        merged_graph.create_edge(
            source_id=location_id,
            target_id=concept_id,
            type=EdgeType.OBSERVED_IN,
            properties={"observation_strength": 0.7},
        )

        # Save merged graph
        storage_backend.save_graph(merged_graph)

        # Verify all graphs are in database
        graph_count = db_session.execute(
            text("SELECT COUNT(*) FROM knowledge_graphs")
        ).scalar()
        assert graph_count >= 3

    def test_knowledge_graph_complex_queries(
        self, db_session: Session, storage_backend
    ):
        """Test complex queries on knowledge graphs stored in PostgreSQL."""
        # Create a rich knowledge graph
        graph = KnowledgeGraph()

        # Create a network of entities, concepts, and beliefs
        entities = []
        for i in range(5):
            entity = graph.create_node(
                type=NodeType.ENTITY,
                label=f"entity_{i}",
                properties={
                    "value": i * 10,
                    "category": "A" if i % 2 == 0 else "B",
                    "active": True,
                },
            )
            entities.append(entity)

        concepts = []
        for i in range(3):
            concept = graph.create_node(
                type=NodeType.CONCEPT,
                label=f"concept_{i}",
                properties={
                    "abstraction_level": i + 1,
                    "domain": ["physics", "biology", "chemistry"][i],
                },
            )
            concepts.append(concept)

        # Create relationships
        # Entities to concepts
        for i, entity in enumerate(entities):
            if i < len(concepts):
                graph.create_edge(
                    source_id=entity.id,
                    target_id=concepts[i % len(concepts)].id,
                    type=EdgeType.INSTANCE_OF,
                    properties={"confidence": 0.8 + i * 0.02},
                )

        # Concepts to concepts (hierarchy)
        for i in range(len(concepts) - 1):
            graph.create_edge(
                source_id=concepts[i].id,
                target_id=concepts[i + 1].id,
                type=EdgeType.SPECIALIZES,
                properties={"specialization_degree": 0.7},
            )

        # Save graph
        storage_backend.save_graph(graph)

        # Now test complex queries using direct SQL
        # 1. Find all entities of category A
        category_a_query = text(
            """
            SELECT node_data->>'label' as label, node_data->'properties'->>'value' as value
            FROM knowledge_nodes
            WHERE graph_id = :graph_id
                AND node_data->>'type' = 'entity'
                AND node_data->'properties'->>'category' = 'A'
            ORDER BY node_data->'properties'->>'value'
        """
        )

        results = db_session.execute(
            category_a_query, {"graph_id": str(graph.graph_id)}
        ).all()

        assert len(results) == 3  # entities 0, 2, 4
        assert results[0].label == "entity_0"

        # 2. Find all relationships with high confidence
        high_confidence_edges = text(
            """
            SELECT
                edge_data->>'type' as edge_type,
                edge_data->'properties'->>'confidence' as confidence
            FROM knowledge_edges
            WHERE graph_id = :graph_id
                AND (edge_data->'properties'->>'confidence')::float > 0.8
        """
        )

        edge_results = db_session.execute(
            high_confidence_edges, {"graph_id": str(graph.graph_id)}
        ).all()

        assert len(edge_results) > 0

        # 3. Find concept hierarchy depth
        hierarchy_query = text(
            """
            WITH RECURSIVE concept_hierarchy AS (
                -- Base case: top-level concepts
                SELECT
                    node_id,
                    node_data->>'label' as label,
                    0 as depth
                FROM knowledge_nodes
                WHERE graph_id = :graph_id
                    AND node_data->>'type' = 'concept'
                    AND node_id NOT IN (
                        SELECT edge_data->>'target_id'
                        FROM knowledge_edges
                        WHERE graph_id = :graph_id
                            AND edge_data->>'type' = 'specializes'
                    )

                UNION ALL

                -- Recursive case
                SELECT
                    kn.node_id,
                    kn.node_data->>'label' as label,
                    ch.depth + 1
                FROM knowledge_nodes kn
                JOIN knowledge_edges ke ON ke.edge_data->>'target_id' = kn.node_id::text
                JOIN concept_hierarchy ch ON ch.node_id::text = ke.edge_data->>'source_id'
                WHERE kn.graph_id = :graph_id
                    AND ke.edge_data->>'type' = 'specializes'
            )
            SELECT label, depth
            FROM concept_hierarchy
            ORDER BY depth, label
        """
        )

        hierarchy_results = db_session.execute(
            hierarchy_query, {"graph_id": str(graph.graph_id)}
        ).all()

        assert len(hierarchy_results) > 0
        max_depth = max(r.depth for r in hierarchy_results)
        assert max_depth >= 1  # We have at least 2 levels

    def test_knowledge_graph_performance_with_large_dataset(
        self, db_session: Session, storage_backend
    ):
        """Test performance with larger knowledge graphs."""
        # Create a large knowledge graph
        graph = KnowledgeGraph()

        # Add many nodes (simulating a real-world scenario)
        node_count = 100
        edge_count = 200

        nodes = []
        # Batch create nodes for better performance
        for i in range(node_count):
            node_type = [NodeType.ENTITY, NodeType.CONCEPT, NodeType.BELIEF][i % 3]
            node = graph.create_node(
                type=node_type,
                label=f"{node_type.value}_{i}",
                properties={
                    "index": i,
                    "batch": i // 10,
                    "active": i % 2 == 0,
                    "score": i * 0.1,
                    "tags": [f"tag_{j}" for j in range(i % 5)],
                },
            )
            nodes.append(node)

        # Create edges with various relationship types
        import random

        edge_types = list(EdgeType)

        for i in range(edge_count):
            source = random.choice(nodes)
            target = random.choice(nodes)
            if source.id != target.id:  # No self-loops
                edge_type = random.choice(edge_types)
                graph.create_edge(
                    source_id=source.id,
                    target_id=target.id,
                    type=edge_type,
                    properties={
                        "weight": random.random(),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

        # Time the save operation
        import time

        start_time = time.time()
        storage_backend.save_graph(graph)
        save_time = time.time() - start_time

        print(
            f"Saved {node_count} nodes and {len(graph.edges)} edges in {save_time:.2f} seconds"
        )

        # Time the load operation
        start_time = time.time()
        loaded_graph = storage_backend.load_graph(graph.graph_id)
        load_time = time.time() - start_time

        print(f"Loaded graph in {load_time:.2f} seconds")

        assert loaded_graph is not None
        assert len(loaded_graph.nodes) == node_count

        # Test batch operations
        # Update all nodes in a batch
        batch_3_nodes = [
            n for n in loaded_graph.nodes.values() if n.properties.get("batch") == 3
        ]

        for node in batch_3_nodes:
            node.properties["batch_updated"] = True
            node.confidence = 0.95

        # Save updated graph
        start_time = time.time()
        storage_backend.save_graph(loaded_graph)
        update_time = time.time() - start_time

        print(f"Updated and saved graph in {update_time:.2f} seconds")

        # Performance assertions
        assert save_time < 5.0  # Should save in under 5 seconds
        assert load_time < 3.0  # Should load in under 3 seconds
        assert update_time < 5.0  # Should update in under 5 seconds

    def test_knowledge_graph_concurrent_access(
        self, db_session: Session, storage_backend
    ):
        """Test concurrent access to knowledge graphs."""
        import threading
        import time

        # Create initial graph
        graph = KnowledgeGraph()
        _initial_node = graph.create_node(
            type=NodeType.ENTITY,
            label="shared_resource",
            properties={"counter": 0},
        )
        storage_backend.save_graph(graph)

        results = []
        errors = []

        def update_graph(thread_id: int):
            """Simulate concurrent graph updates."""
            try:
                # Load graph
                local_graph = storage_backend.load_graph(graph.graph_id)

                # Add thread-specific node
                _thread_node = local_graph.create_node(
                    type=NodeType.ENTITY,
                    label=f"thread_{thread_id}_node",
                    properties={
                        "thread_id": thread_id,
                        "timestamp": time.time(),
                    },
                )

                # Update shared resource
                shared_node = list(local_graph.nodes.values())[0]
                shared_node.properties["last_thread"] = thread_id

                # Save graph
                storage_backend.save_graph(local_graph)

                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {str(e)}")

        # Launch concurrent threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_graph, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 5

        # Verify final state
        final_graph = storage_backend.load_graph(graph.graph_id)
        assert len(final_graph.nodes) >= 6  # Initial + 5 thread nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test Entity to Node Mapping for Knowledge Graph
Following TDD principles - write tests first, then implementation
"""

from unittest.mock import AsyncMock, Mock

import pytest

from knowledge_graph.entity_node_mapper import (
    Edge,
    EntityNodeMapper,
    GraphEngine,
    MappingResult,
    MappingStrategy,
    Node,
    NodeMapping,
)
from knowledge_graph.nlp_entity_extractor import (
    Entity,
    EntityType,
    Relationship,
)


class TestEntityToNodeMapping:
    """Test suite for entity to knowledge graph node mapping"""

    def test_node_mapping_data_structure(self):
        """Test NodeMapping data structure"""
        entity = Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.9)
        node = Node(id="node_123", type="Technology", properties={"name": "Python"})

        mapping = NodeMapping(
            entity=entity,
            node=node,
            confidence=0.85,
            strategy=MappingStrategy.EXACT_MATCH,
        )

        assert mapping.entity == entity
        assert mapping.node == node
        assert mapping.confidence == 0.85
        assert mapping.strategy == MappingStrategy.EXACT_MATCH

    def test_mapping_strategies(self):
        """Test different mapping strategies"""
        assert MappingStrategy.EXACT_MATCH.value == "exact_match"
        assert MappingStrategy.FUZZY_MATCH.value == "fuzzy_match"
        assert MappingStrategy.SEMANTIC_MATCH.value == "semantic_match"
        assert MappingStrategy.CREATE_NEW.value == "create_new"

    @pytest.mark.asyncio
    async def test_mapper_initialization(self):
        """Test EntityNodeMapper initialization"""
        graph_engine = Mock(spec=GraphEngine)
        mapper = EntityNodeMapper(graph_engine=graph_engine)

        assert mapper.graph_engine == graph_engine
        assert mapper.mapping_cache == {}
        assert mapper.similarity_threshold == 0.8

    @pytest.mark.asyncio
    async def test_map_single_entity_exact_match(self):
        """Test mapping a single entity with exact match"""
        # Setup mock graph engine
        graph_engine = Mock(spec=GraphEngine)
        existing_node = Node(
            id="node_python",
            type="Technology",
            properties={"name": "Python", "category": "programming_language"},
        )
        graph_engine.find_nodes_by_name = AsyncMock(return_value=[existing_node])

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        # Create test entity
        entity = Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.95)

        # Map entity to node
        result = await mapper.map_entity(entity)

        assert isinstance(result, NodeMapping)
        assert result.entity == entity
        assert result.node == existing_node
        assert result.strategy == MappingStrategy.EXACT_MATCH
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_map_entity_fuzzy_match(self):
        """Test mapping with fuzzy matching"""
        graph_engine = Mock(spec=GraphEngine)
        existing_node = Node(
            id="node_js",
            type="Technology",
            properties={"name": "JavaScript", "aliases": ["JS", "ECMAScript"]},
        )
        graph_engine.find_nodes_by_name = AsyncMock(return_value=[])
        graph_engine.search_similar_nodes = AsyncMock(return_value=[existing_node])

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        # Entity with slightly different name
        entity = Entity("JS", EntityType.TECHNOLOGY, 0, 2, 0.9)

        result = await mapper.map_entity(entity)

        assert result.strategy == MappingStrategy.FUZZY_MATCH
        assert result.node == existing_node
        assert 0.7 <= result.confidence <= 0.9

    @pytest.mark.asyncio
    async def test_map_entity_create_new_node(self):
        """Test creating new node when no match found"""
        graph_engine = Mock(spec=GraphEngine)
        graph_engine.find_nodes_by_name = AsyncMock(return_value=[])
        graph_engine.search_similar_nodes = AsyncMock(return_value=[])

        new_node = Node(
            id="node_rust",
            type="Technology",
            properties={"name": "Rust", "created_from": "entity_extraction"},
        )
        graph_engine.create_node = AsyncMock(return_value=new_node)

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        entity = Entity("Rust", EntityType.TECHNOLOGY, 0, 4, 0.95)

        result = await mapper.map_entity(entity)

        assert result.strategy == MappingStrategy.CREATE_NEW
        assert result.node.properties["name"] == "Rust"
        graph_engine.create_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_map_multiple_entities(self):
        """Test mapping multiple entities in batch"""
        graph_engine = Mock(spec=GraphEngine)

        # Setup different scenarios
        python_node = Node("node_1", "Technology", {"name": "Python"})
        graph_engine.find_nodes_by_name = AsyncMock(
            side_effect=lambda name: [python_node] if name == "Python" else []
        )
        graph_engine.search_similar_nodes = AsyncMock(return_value=[])
        graph_engine.create_node = AsyncMock(
            side_effect=lambda **kwargs: Node(
                f"node_{kwargs['properties']['name'].lower()}",
                kwargs["type"],
                kwargs["properties"],
            )
        )

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        entities = [
            Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.95),
            Entity("Django", EntityType.TECHNOLOGY, 10, 16, 0.9),
            Entity("Guido van Rossum", EntityType.PERSON, 20, 36, 0.85),
        ]

        results = await mapper.map_entities(entities)

        assert len(results) == 3

        # Check Python was matched
        python_result = next(r for r in results if r.entity.text == "Python")
        assert python_result.strategy == MappingStrategy.EXACT_MATCH

        # Check Django was created
        django_result = next(r for r in results if r.entity.text == "Django")
        assert django_result.strategy == MappingStrategy.CREATE_NEW

        # Check person was created with correct type
        person_result = next(r for r in results if r.entity.text == "Guido van Rossum")
        assert person_result.node.type == "Person"

    @pytest.mark.asyncio
    async def test_map_relationships(self):
        """Test mapping relationships between entities"""
        graph_engine = Mock(spec=GraphEngine)

        # Create nodes
        python_node = Node("node_python", "Technology", {"name": "Python"})
        django_node = Node("node_django", "Technology", {"name": "Django"})

        # Mock node lookups
        graph_engine.find_nodes_by_name = AsyncMock(
            side_effect=lambda name: {
                "Python": [python_node],
                "Django": [django_node],
            }.get(name, [])
        )

        # Mock edge creation
        created_edge = Edge(
            id="edge_1",
            source_id="node_python",
            target_id="node_django",
            type="used_for",
            properties={"confidence": 0.8},
        )
        graph_engine.create_edge = AsyncMock(return_value=created_edge)

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        # Create entities and relationship
        python_entity = Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.95)
        django_entity = Entity("Django", EntityType.TECHNOLOGY, 10, 16, 0.9)
        relationship = Relationship(
            source=python_entity,
            target=django_entity,
            type="used_for",
            confidence=0.8,
        )

        # Map entities first
        entity_mappings = await mapper.map_entities([python_entity, django_entity])

        # Map relationship
        edge_result = await mapper.map_relationship(relationship, entity_mappings)

        assert edge_result is not None
        assert edge_result.source_id == python_node.id
        assert edge_result.target_id == django_node.id
        assert edge_result.type == "used_for"

    @pytest.mark.asyncio
    async def test_entity_type_to_node_type_mapping(self):
        """Test correct mapping of entity types to node types"""
        graph_engine = Mock(spec=GraphEngine)
        graph_engine.find_nodes_by_name = AsyncMock(return_value=[])
        graph_engine.search_similar_nodes = AsyncMock(return_value=[])

        created_nodes = {}

        def create_node_mock(type, properties):
            node = Node(f"node_{len(created_nodes)}", type, properties)
            created_nodes[properties["name"]] = node
            return node

        graph_engine.create_node = AsyncMock(side_effect=create_node_mock)

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        # Test different entity types
        test_cases = [
            (Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.9), "Technology"),
            (Entity("John Doe", EntityType.PERSON, 0, 8, 0.9), "Person"),
            (
                Entity("Google", EntityType.ORGANIZATION, 0, 6, 0.9),
                "Organization",
            ),
            (Entity("AI", EntityType.CONCEPT, 0, 2, 0.9), "Concept"),
            (Entity("New York", EntityType.LOCATION, 0, 8, 0.9), "Location"),
        ]

        for entity, expected_node_type in test_cases:
            result = await mapper.map_entity(entity)
            assert result.node.type == expected_node_type

    @pytest.mark.asyncio
    async def test_mapping_with_context(self):
        """Test mapping with conversation context"""
        graph_engine = Mock(spec=GraphEngine)

        # Existing nodes
        ml_node = Node("node_ml", "Concept", {"name": "Machine Learning"})
        python_node = Node("node_python", "Technology", {"name": "Python"})

        graph_engine.find_nodes_by_name = AsyncMock(
            side_effect=lambda name: {"ML": [], "Python": [python_node]}.get(name, [])
        )

        # ML should match to Machine Learning with context
        graph_engine.search_similar_nodes = AsyncMock(
            side_effect=lambda name, context: [ml_node]
            if name == "ML" and context
            else []
        )

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        # Entity with ambiguous abbreviation
        ml_entity = Entity("ML", EntityType.CONCEPT, 0, 2, 0.8)

        # Map with context
        context = {
            "previous_entities": ["Machine Learning", "Python"],
            "domain": "AI",
        }

        result = await mapper.map_entity(ml_entity, context=context)

        assert result.node == ml_node
        assert result.strategy == MappingStrategy.SEMANTIC_MATCH

    @pytest.mark.asyncio
    async def test_caching_mechanism(self):
        """Test that mapper caches results for efficiency"""
        graph_engine = Mock(spec=GraphEngine)
        node = Node("node_1", "Technology", {"name": "Python"})
        graph_engine.find_nodes_by_name = AsyncMock(return_value=[node])

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        entity = Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.95)

        # First mapping
        result1 = await mapper.map_entity(entity)

        # Second mapping of same entity
        result2 = await mapper.map_entity(entity)

        # Should use cache, not call graph engine again
        assert graph_engine.find_nodes_by_name.call_count == 1
        assert result1.node == result2.node

    @pytest.mark.asyncio
    async def test_bulk_mapping_result(self):
        """Test bulk mapping result structure"""
        graph_engine = Mock(spec=GraphEngine)
        graph_engine.find_nodes_by_name = AsyncMock(return_value=[])
        graph_engine.search_similar_nodes = AsyncMock(return_value=[])
        graph_engine.create_node = AsyncMock(
            side_effect=lambda **kwargs: Node(
                f"node_{kwargs['properties']['name']}",
                kwargs["type"],
                kwargs["properties"],
            )
        )

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        entities = [
            Entity("React", EntityType.TECHNOLOGY, 0, 5, 0.9),
            Entity("Facebook", EntityType.ORGANIZATION, 10, 18, 0.85),
        ]

        result = await mapper.map_entities_bulk(entities)

        assert isinstance(result, MappingResult)
        assert result.total_entities == 2
        assert result.successful_mappings == 2
        assert result.failed_mappings == 0
        assert len(result.mappings) == 2
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_mapping(self):
        """Test error handling during mapping process"""
        graph_engine = Mock(spec=GraphEngine)
        graph_engine.find_nodes_by_name = AsyncMock(
            side_effect=Exception("Database connection error")
        )

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        entity = Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.95)

        # Should handle error gracefully
        result = await mapper.map_entity(entity)

        assert result is None or result.node is None

    @pytest.mark.asyncio
    async def test_merge_duplicate_nodes(self):
        """Test merging duplicate nodes during mapping"""
        graph_engine = Mock(spec=GraphEngine)

        # Multiple nodes for same entity
        nodes = [
            Node(
                "node_1",
                "Technology",
                {"name": "JavaScript", "aliases": ["JS"]},
            ),
            Node(
                "node_2",
                "Technology",
                {"name": "JS", "full_name": "JavaScript"},
            ),
        ]

        graph_engine.find_nodes_by_name = AsyncMock(return_value=nodes)
        graph_engine.merge_nodes = AsyncMock(
            return_value=Node(
                "node_merged",
                "Technology",
                {"name": "JavaScript", "aliases": ["JS"], "merged": True},
            )
        )

        mapper = EntityNodeMapper(graph_engine=graph_engine)
        mapper.enable_deduplication = True

        entity = Entity("JavaScript", EntityType.TECHNOLOGY, 0, 10, 0.95)

        result = await mapper.map_entity(entity)

        assert result.node.properties.get("merged") is True
        graph_engine.merge_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_confidence_propagation(self):
        """Test that entity confidence affects mapping confidence"""
        graph_engine = Mock(spec=GraphEngine)
        node = Node("node_1", "Technology", {"name": "Python"})
        graph_engine.find_nodes_by_name = AsyncMock(return_value=[node])

        mapper = EntityNodeMapper(graph_engine=graph_engine)

        # High confidence entity
        high_conf_entity = Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.95)
        high_result = await mapper.map_entity(high_conf_entity)

        # Low confidence entity
        low_conf_entity = Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.5)
        low_result = await mapper.map_entity(low_conf_entity)

        assert high_result.confidence > low_result.confidence

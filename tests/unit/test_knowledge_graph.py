"""
Comprehensive tests for Knowledge Graph implementation.

Tests the graph-based knowledge representation system that allows
agents to store, query, and update their beliefs about the world,
including pattern extraction and relationship analysis.
"""

from datetime import datetime

from knowledge.knowledge_graph import BeliefNode, KnowledgeEdge, KnowledgeGraph, PatternExtractor


class TestBeliefNode:
    """Test BeliefNode dataclass."""

    def test_belief_node_creation(self):
        """Test creating belief node with all fields."""
        created_time = datetime.utcnow()
        updated_time = datetime.utcnow()
        supporting = ["evidence1", "evidence2"]
        contradicting = ["counter1"]
        metadata = {"source": "test", "priority": "high"}

        node = BeliefNode(
            id="test_id",
            statement="The sky is blue",
            confidence=0.8,
            supporting_patterns=supporting,
            contradicting_patterns=contradicting,
            created_at=created_time,
            updated_at=updated_time,
            metadata=metadata,
        )

        assert node.id == "test_id"
        assert node.statement == "The sky is blue"
        assert node.confidence == 0.8
        assert node.supporting_patterns == supporting
        assert node.contradicting_patterns == contradicting
        assert node.created_at == created_time
        assert node.updated_at == updated_time
        assert node.metadata == metadata

    def test_belief_node_defaults(self):
        """Test default values for optional fields."""
        node = BeliefNode(id="test_id", statement="Test statement", confidence=0.5)

        assert node.supporting_patterns == []
        assert node.contradicting_patterns == []
        assert isinstance(node.created_at, datetime)
        assert isinstance(node.updated_at, datetime)
        assert node.metadata == {}


class TestKnowledgeEdge:
    """Test KnowledgeEdge dataclass."""

    def test_knowledge_edge_creation(self):
        """Test creating knowledge edge with all fields."""
        metadata = {"created_by": "test", "context": "belief_update"}

        edge = KnowledgeEdge(
            id="edge_id",
            source_id="node1",
            target_id="node2",
            relationship_type="supports",
            strength=0.9,
            metadata=metadata,
        )

        assert edge.id == "edge_id"
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.relationship_type == "supports"
        assert edge.strength == 0.9
        assert edge.metadata == metadata

    def test_knowledge_edge_defaults(self):
        """Test default values for optional fields."""
        edge = KnowledgeEdge(
            id="edge_id",
            source_id="node1",
            target_id="node2",
            relationship_type="contradicts",
            strength=0.7,
        )

        assert edge.metadata == {}


class TestKnowledgeGraph:
    """Test KnowledgeGraph class."""

    def setup_method(self):
        """Set up test knowledge graph."""
        self.graph = KnowledgeGraph("test_agent")

    def test_knowledge_graph_initialization(self):
        """Test knowledge graph initialization."""
        assert self.graph.agent_id == "test_agent"
        assert isinstance(self.graph.nodes, dict)
        assert len(self.graph.nodes) == 0
        assert isinstance(self.graph.edges, dict)
        assert len(self.graph.edges) == 0
        assert isinstance(self.graph.node_relationships, dict)
        assert len(self.graph.node_relationships) == 0
        assert isinstance(self.graph.created_at, datetime)

    def test_add_belief_basic(self):
        """Test adding basic belief to graph."""
        node = self.graph.add_belief(statement="The weather is sunny", confidence=0.8)

        assert isinstance(node, BeliefNode)
        assert node.statement == "The weather is sunny"
        assert node.confidence == 0.8
        assert node.supporting_patterns == []
        assert node.contradicting_patterns == []
        assert node.metadata == {}
        assert isinstance(node.id, str)

        # Check node is stored in graph
        assert node.id in self.graph.nodes
        assert self.graph.nodes[node.id] == node
        assert node.id in self.graph.node_relationships
        assert self.graph.node_relationships[node.id] == set()

    def test_add_belief_with_all_fields(self):
        """Test adding belief with all optional fields."""
        supporting = ["observation1", "measurement2"]
        contradicting = ["counter_evidence"]
        metadata = {"source": "sensor", "location": "outdoor"}

        node = self.graph.add_belief(
            statement="Temperature is 25°C",
            confidence=0.95,
            supporting_patterns=supporting,
            contradicting_patterns=contradicting,
            metadata=metadata,
        )

        assert node.supporting_patterns == supporting
        assert node.contradicting_patterns == contradicting
        assert node.metadata == metadata
        assert node.id in self.graph.nodes

    def test_update_belief_success(self):
        """Test successfully updating existing belief."""
        # Add initial belief
        node = self.graph.add_belief("Initial statement", 0.5)
        original_id = node.id
        original_created_at = node.created_at

        # Update belief
        new_supporting = ["new_evidence"]
        new_contradicting = ["new_counter"]
        new_metadata = {"updated": "true"}

        success = self.graph.update_belief(
            node_id=original_id,
            confidence=0.8,
            supporting_patterns=new_supporting,
            contradicting_patterns=new_contradicting,
            metadata=new_metadata,
        )

        assert success is True

        # Check updates
        updated_node = self.graph.nodes[original_id]
        assert updated_node.confidence == 0.8
        assert updated_node.supporting_patterns == new_supporting
        assert updated_node.contradicting_patterns == new_contradicting
        assert updated_node.metadata == new_metadata
        assert updated_node.created_at == original_created_at  # Should not change
        assert updated_node.updated_at > original_created_at  # Should be updated

    def test_update_belief_partial(self):
        """Test partial update of belief (only some fields)."""
        # Add initial belief
        node = self.graph.add_belief(
            "Test statement",
            0.6,
            supporting_patterns=["old_evidence"],
            metadata={"original": "value"},
        )
        original_id = node.id

        # Update only confidence and metadata
        success = self.graph.update_belief(
            node_id=original_id, confidence=0.9, metadata={"new_field": "new_value"}
        )

        assert success is True

        updated_node = self.graph.nodes[original_id]
        assert updated_node.confidence == 0.9
        assert updated_node.supporting_patterns == ["old_evidence"]  # Unchanged
        # Metadata should be merged
        assert updated_node.metadata["original"] == "value"
        assert updated_node.metadata["new_field"] == "new_value"

    def test_update_belief_nonexistent(self):
        """Test updating non-existent belief."""
        success = self.graph.update_belief(node_id="nonexistent_id", confidence=0.5)

        assert success is False

    def test_add_relationship_success(self):
        """Test successfully adding relationship between nodes."""
        # Add two nodes
        node1 = self.graph.add_belief("Statement 1", 0.7)
        node2 = self.graph.add_belief("Statement 2", 0.8)

        # Add relationship
        metadata = {"context": "logical_inference"}
        edge = self.graph.add_relationship(
            source_id=node1.id,
            target_id=node2.id,
            relationship_type="supports",
            strength=0.9,
            metadata=metadata,
        )

        assert isinstance(edge, KnowledgeEdge)
        assert edge.source_id == node1.id
        assert edge.target_id == node2.id
        assert edge.relationship_type == "supports"
        assert edge.strength == 0.9
        assert edge.metadata == metadata
        assert isinstance(edge.id, str)

        # Check edge is stored
        assert edge.id in self.graph.edges
        assert self.graph.edges[edge.id] == edge

        # Check node relationships are updated
        assert node2.id in self.graph.node_relationships[node1.id]
        assert node1.id in self.graph.node_relationships[node2.id]

    def test_add_relationship_missing_source(self):
        """Test adding relationship with missing source node."""
        node = self.graph.add_belief("Statement", 0.5)

        edge = self.graph.add_relationship(
            source_id="nonexistent", target_id=node.id, relationship_type="supports"
        )

        assert edge is None
        assert len(self.graph.edges) == 0

    def test_add_relationship_missing_target(self):
        """Test adding relationship with missing target node."""
        node = self.graph.add_belief("Statement", 0.5)

        edge = self.graph.add_relationship(
            source_id=node.id, target_id="nonexistent", relationship_type="contradicts"
        )

        assert edge is None
        assert len(self.graph.edges) == 0

    def test_add_relationship_default_strength(self):
        """Test adding relationship with default strength."""
        node1 = self.graph.add_belief("Statement 1", 0.6)
        node2 = self.graph.add_belief("Statement 2", 0.7)

        edge = self.graph.add_relationship(
            source_id=node1.id, target_id=node2.id, relationship_type="relates_to"
        )

        assert edge is not None
        assert edge.strength == 1.0  # Default strength
        assert edge.metadata == {}  # Default metadata

    def test_query_beliefs_no_filters(self):
        """Test querying beliefs without filters."""
        # Add multiple beliefs
        self.graph.add_belief("Weather is sunny", 0.8)
        self.graph.add_belief("Temperature is warm", 0.9)
        self.graph.add_belief("Rain is expected", 0.3)

        results = self.graph.query_beliefs()

        assert len(results) == 3
        # Should be sorted by confidence (descending)
        assert results[0].confidence == 0.9  # Temperature
        assert results[1].confidence == 0.8  # Weather
        assert results[2].confidence == 0.3  # Rain

    def test_query_beliefs_with_pattern(self):
        """Test querying beliefs with pattern matching."""
        self.graph.add_belief("Weather is sunny", 0.8)
        self.graph.add_belief("Weather is cloudy", 0.6)
        self.graph.add_belief("Temperature is warm", 0.9)

        results = self.graph.query_beliefs(pattern="weather")

        assert len(results) == 2
        # Check both weather-related beliefs are returned
        statements = [node.statement for node in results]
        assert "Weather is sunny" in statements
        assert "Weather is cloudy" in statements
        assert "Temperature is warm" not in statements

    def test_query_beliefs_with_confidence_filter(self):
        """Test querying beliefs with confidence threshold."""
        self.graph.add_belief("High confidence", 0.9)
        self.graph.add_belief("Medium confidence", 0.6)
        self.graph.add_belief("Low confidence", 0.2)

        results = self.graph.query_beliefs(min_confidence=0.7)

        assert len(results) == 1
        assert results[0].statement == "High confidence"
        assert results[0].confidence == 0.9

    def test_query_beliefs_with_max_results(self):
        """Test querying beliefs with result limit."""
        for i in range(5):
            self.graph.add_belief(f"Statement {i}", 0.5 + i * 0.1)

        results = self.graph.query_beliefs(max_results=3)

        assert len(results) == 3
        # Should get top 3 by confidence
        assert results[0].confidence == 0.9
        assert results[1].confidence == 0.8
        assert results[2].confidence == 0.7

    def test_query_beliefs_combined_filters(self):
        """Test querying beliefs with multiple filters."""
        self.graph.add_belief("Weather is sunny", 0.9)
        self.graph.add_belief("Weather is cloudy", 0.5)
        self.graph.add_belief("Weather forecast reliable", 0.8)
        self.graph.add_belief("Temperature reading", 0.4)

        results = self.graph.query_beliefs(pattern="weather", min_confidence=0.7, max_results=2)

        assert len(results) == 2
        statements = [node.statement for node in results]
        assert "Weather is sunny" in statements
        assert "Weather forecast reliable" in statements
        # "Weather is cloudy" excluded by confidence filter
        # "Temperature reading" excluded by pattern filter

    def test_get_related_beliefs_with_relationships(self):
        """Test getting related beliefs for connected nodes."""
        # Add nodes
        node1 = self.graph.add_belief("Central belief", 0.8)
        node2 = self.graph.add_belief("Supporting belief", 0.7)
        node3 = self.graph.add_belief("Related belief", 0.6)
        self.graph.add_belief("Unrelated belief", 0.5)

        # Add relationships
        self.graph.add_relationship(node1.id, node2.id, "supports")
        self.graph.add_relationship(node1.id, node3.id, "relates_to")
        # node4 is not connected

        related = self.graph.get_related_beliefs(node1.id)

        assert len(related) == 2
        related_statements = [node.statement for node in related]
        assert "Supporting belief" in related_statements
        assert "Related belief" in related_statements
        assert "Unrelated belief" not in related_statements

    def test_get_related_beliefs_no_relationships(self):
        """Test getting related beliefs for isolated node."""
        node = self.graph.add_belief("Isolated belief", 0.8)

        related = self.graph.get_related_beliefs(node.id)

        assert len(related) == 0

    def test_get_related_beliefs_nonexistent_node(self):
        """Test getting related beliefs for non-existent node."""
        related = self.graph.get_related_beliefs("nonexistent_id")

        assert len(related) == 0

    def test_update_from_message_new_sender(self):
        """Test updating knowledge from message with new sender."""
        message = "The sky is blue today"
        sender = "weather_agent"

        updated_nodes = self.graph.update_from_message(message, sender)

        assert len(updated_nodes) == 1
        node = updated_nodes[0]
        assert isinstance(node, BeliefNode)
        assert f"Agent {sender} communicated: {message}" in node.statement
        assert node.confidence == 0.8
        assert node.metadata["source"] == "communication"
        assert node.metadata["sender"] == sender

        # Check node is stored in graph
        assert node.id in self.graph.nodes

    def test_update_from_message_existing_sender(self):
        """Test updating knowledge from message with existing sender."""
        sender = "weather_agent"

        # First message
        message1 = "The sky is blue"
        updated_nodes1 = self.graph.update_from_message(message1, sender)
        original_node = updated_nodes1[0]
        original_confidence = original_node.confidence

        # Second message from same sender
        message2 = "The weather is sunny"
        updated_nodes2 = self.graph.update_from_message(message2, sender)

        assert len(updated_nodes2) == 1
        updated_node = updated_nodes2[0]

        # Should update existing node
        assert updated_node.id == original_node.id
        assert updated_node.confidence > original_confidence
        assert f"Agent {sender} communicated: {message2}" in updated_node.statement
        assert updated_node.updated_at > original_node.created_at

    def test_get_statistics_empty_graph(self):
        """Test getting statistics for empty graph."""
        stats = self.graph.get_statistics()

        assert stats["num_nodes"] == 0
        assert stats["num_edges"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["agent_id"] == "test_agent"
        assert isinstance(stats["created_at"], str)  # ISO format

    def test_get_statistics_populated_graph(self):
        """Test getting statistics for populated graph."""
        # Add nodes
        self.graph.add_belief("Belief 1", 0.8)
        self.graph.add_belief("Belief 2", 0.6)
        node3 = self.graph.add_belief("Belief 3", 0.9)
        node4 = self.graph.add_belief("Belief 4", 0.7)

        # Add edge
        self.graph.add_relationship(node3.id, node4.id, "supports")

        stats = self.graph.get_statistics()

        assert stats["num_nodes"] == 4
        assert stats["num_edges"] == 1
        assert stats["avg_confidence"] == (0.8 + 0.6 + 0.9 + 0.7) / 4
        assert stats["agent_id"] == "test_agent"

    def test_clear_graph(self):
        """Test clearing all data from graph."""
        # Add data
        node1 = self.graph.add_belief("Belief 1", 0.8)
        node2 = self.graph.add_belief("Belief 2", 0.6)
        self.graph.add_relationship(node1.id, node2.id, "supports")

        # Verify data exists
        assert len(self.graph.nodes) == 2
        assert len(self.graph.edges) == 1
        assert len(self.graph.node_relationships) == 2

        # Clear graph
        self.graph.clear()

        # Verify all data is cleared
        assert len(self.graph.nodes) == 0
        assert len(self.graph.edges) == 0
        assert len(self.graph.node_relationships) == 0

    def test_to_dict_empty_graph(self):
        """Test exporting empty graph to dictionary."""
        graph_dict = self.graph.to_dict()

        assert graph_dict["agent_id"] == "test_agent"
        assert isinstance(graph_dict["created_at"], str)
        assert graph_dict["nodes"] == {}
        assert graph_dict["edges"] == {}

    def test_to_dict_populated_graph(self):
        """Test exporting populated graph to dictionary."""
        # Add nodes
        node1 = self.graph.add_belief(
            "Belief 1", 0.8, supporting_patterns=["evidence1"], metadata={"source": "test"}
        )
        node2 = self.graph.add_belief("Belief 2", 0.6)

        # Add edge
        edge = self.graph.add_relationship(
            node1.id, node2.id, "supports", 0.9, metadata={"context": "test"}
        )

        graph_dict = self.graph.to_dict()

        # Check structure
        assert graph_dict["agent_id"] == "test_agent"
        assert len(graph_dict["nodes"]) == 2
        assert len(graph_dict["edges"]) == 1

        # Check node data
        node1_dict = graph_dict["nodes"][node1.id]
        assert node1_dict["statement"] == "Belief 1"
        assert node1_dict["confidence"] == 0.8
        assert node1_dict["supporting_patterns"] == ["evidence1"]
        assert node1_dict["metadata"] == {"source": "test"}
        assert isinstance(node1_dict["created_at"], str)
        assert isinstance(node1_dict["updated_at"], str)

        # Check edge data
        edge_dict = graph_dict["edges"][edge.id]
        assert edge_dict["source_id"] == node1.id
        assert edge_dict["target_id"] == node2.id
        assert edge_dict["relationship_type"] == "supports"
        assert edge_dict["strength"] == 0.9
        assert edge_dict["metadata"] == {"context": "test"}


class TestPatternExtractor:
    """Test PatternExtractor class."""

    def setup_method(self):
        """Set up test pattern extractor."""
        self.extractor = PatternExtractor(min_support=0.3, min_confidence=0.5)

    def test_pattern_extractor_initialization(self):
        """Test pattern extractor initialization."""
        assert self.extractor.min_support == 0.3
        assert self.extractor.min_confidence == 0.5
        assert isinstance(self.extractor.patterns, list)
        assert len(self.extractor.patterns) == 0
        assert isinstance(self.extractor.relationships, list)
        assert len(self.extractor.relationships) == 0

    def test_extract_patterns_empty_data(self):
        """Test pattern extraction with empty data."""
        patterns = self.extractor.extract_patterns([])

        assert len(patterns) == 0
        assert self.extractor.patterns == []

    def test_extract_patterns_single_record(self):
        """Test pattern extraction with single record."""
        data = [{"color": "red", "size": "large", "type": "fruit"}]

        patterns = self.extractor.extract_patterns(data)

        # With single record, all values have support = 1.0
        assert len(patterns) == 3  # All attributes meet min_support=0.3

        # Check pattern structure
        color_pattern = next(p for p in patterns if p["attribute"] == "color")
        assert color_pattern["value"] == "red"
        assert color_pattern["support"] == 1.0
        assert color_pattern["count"] == 1
        assert color_pattern["type"] == "frequent_value"

    def test_extract_patterns_multiple_records(self):
        """Test pattern extraction with multiple records."""
        data = [
            {"color": "red", "size": "large"},
            {"color": "red", "size": "small"},
            {"color": "blue", "size": "large"},
            {"color": "red", "size": "large"},
        ]

        patterns = self.extractor.extract_patterns(data)

        # Check frequent patterns (support >= 0.3)
        color_patterns = [p for p in patterns if p["attribute"] == "color"]
        red_pattern = next(p for p in color_patterns if p["value"] == "red")
        assert red_pattern["support"] == 0.75  # 3/4
        assert red_pattern["count"] == 3

        # Blue should also be included (support = 0.25 < 0.3, so not included)
        blue_patterns = [p for p in color_patterns if p["value"] == "blue"]
        assert len(blue_patterns) == 0  # Below min_support

        # Large size appears three times (support = 0.75 >= 0.3)
        size_patterns = [p for p in patterns if p["attribute"] == "size"]
        large_pattern = next(p for p in size_patterns if p["value"] == "large")
        assert abs(large_pattern["support"] - 0.75) < 1e-10
        assert large_pattern["count"] == 3

    def test_extract_patterns_different_support_threshold(self):
        """Test pattern extraction with different support threshold."""
        extractor = PatternExtractor(min_support=0.6)  # Higher threshold

        data = [
            {"type": "A", "category": "X"},
            {"type": "A", "category": "Y"},
            {"type": "B", "category": "X"},
        ]

        patterns = extractor.extract_patterns(data)

        # Only patterns with support >= 0.6 should be included
        # type "A" has support 2/3 = 0.67 >= 0.6 ✓
        # category "X" has support 2/3 = 0.67 >= 0.6 ✓
        # type "B" has support 1/3 = 0.33 < 0.6 ✗
        # category "Y" has support 1/3 = 0.33 < 0.6 ✗

        assert len(patterns) == 2
        pattern_descriptions = [(p["attribute"], p["value"]) for p in patterns]
        assert ("type", "A") in pattern_descriptions
        assert ("category", "X") in pattern_descriptions

    def test_extract_relationships_insufficient_data(self):
        """Test relationship extraction with insufficient data."""
        data = [{"attr1": "value1"}]  # Single record

        relationships = self.extractor.extract_relationships(data)

        assert len(relationships) == 0

    def test_extract_relationships_basic(self):
        """Test basic relationship extraction."""
        data = [
            {"weather": "sunny", "activity": "outdoor"},
            {"weather": "sunny", "activity": "outdoor"},
            {"weather": "rainy", "activity": "indoor"},
        ]

        relationships = self.extractor.extract_relationships(data)

        # sunny-outdoor appears twice (confidence = 2/3 = 0.67 >= 0.5)
        sunny_outdoor = next(
            r for r in relationships if r["value1"] == "sunny" and r["value2"] == "outdoor"
        )
        assert sunny_outdoor["confidence"] == 2 / 3
        assert sunny_outdoor["support"] == 2 / 3
        assert sunny_outdoor["count"] == 2
        assert sunny_outdoor["type"] == "co_occurrence"
        assert sunny_outdoor["attribute1"] == "weather"
        assert sunny_outdoor["attribute2"] == "activity"

        # rainy-indoor appears once (confidence = 1/3 = 0.33 < 0.5, not
        # included)
        rainy_indoor_relationships = [
            r for r in relationships if r["value1"] == "rainy" and r["value2"] == "indoor"
        ]
        assert len(rainy_indoor_relationships) == 0

    def test_extract_relationships_multiple_attributes(self):
        """Test relationship extraction with multiple attributes."""
        data = [
            {"A": "1", "B": "X", "C": "alpha"},
            {"A": "1", "B": "X", "C": "beta"},
            {"A": "1", "B": "Y", "C": "alpha"},
            {"A": "2", "B": "X", "C": "alpha"},
        ]

        relationships = self.extractor.extract_relationships(data)

        # Check for strong relationships (confidence >= 0.5)
        # A=1, B=X appears twice (2/4 = 0.5)
        # A=1, C=alpha appears twice (2/4 = 0.5)
        # B=X, C=alpha appears twice (2/4 = 0.5)

        # Find A=1, B=X relationship
        a1_bx = next(
            (
                r
                for r in relationships
                if r["attribute1"] == "A"
                and r["value1"] == "1"
                and r["attribute2"] == "B"
                and r["value2"] == "X"
            ),
            None,
        )
        assert a1_bx is not None
        assert a1_bx["confidence"] == 0.5

    def test_extract_relationships_different_confidence_threshold(self):
        """Test relationship extraction with different confidence threshold."""
        extractor = PatternExtractor(min_confidence=0.8)  # Higher threshold

        data = [
            {"type": "A", "result": "success"},
            {"type": "A", "result": "success"},
            {"type": "A", "result": "failure"},
            {"type": "B", "result": "success"},
        ]

        relationships = extractor.extract_relationships(data)

        # type=A, result=success appears twice (2/4 = 0.5 < 0.8, not included)
        # No relationships should meet the 0.8 threshold
        assert len(relationships) == 0

    def test_get_pattern_summary_empty(self):
        """Test pattern summary with no extracted patterns."""
        summary = self.extractor.get_pattern_summary()

        assert summary["total_patterns"] == 0
        assert summary["pattern_types"] == []
        assert summary["total_relationships"] == 0
        assert summary["relationship_types"] == []
        assert summary["min_support"] == 0.3
        assert summary["min_confidence"] == 0.5

    def test_get_pattern_summary_with_data(self):
        """Test pattern summary with extracted patterns and relationships."""
        data = [
            {"color": "red", "size": "large"},
            {"color": "red", "size": "large"},
            {"color": "blue", "size": "small"},
        ]

        patterns = self.extractor.extract_patterns(data)
        relationships = self.extractor.extract_relationships(data)

        summary = self.extractor.get_pattern_summary()

        assert summary["total_patterns"] == len(patterns)
        assert "frequent_value" in summary["pattern_types"]
        assert summary["total_relationships"] == len(relationships)
        if relationships:
            assert "co_occurrence" in summary["relationship_types"]
        assert summary["min_support"] == 0.3
        assert summary["min_confidence"] == 0.5

    def test_filter_patterns_by_type(self):
        """Test filtering patterns by type."""
        data = [{"attr1": "value1", "attr2": "value2"}] * 2

        self.extractor.extract_patterns(data)

        # Filter by type
        filtered = self.extractor.filter_patterns(pattern_type="frequent_value")

        assert len(filtered) == len(self.extractor.patterns)
        assert all(p["type"] == "frequent_value" for p in filtered)

        # Filter by non-existent type
        filtered_empty = self.extractor.filter_patterns(pattern_type="nonexistent")
        assert len(filtered_empty) == 0

    def test_filter_patterns_by_support(self):
        """Test filtering patterns by support threshold."""
        data = [
            # freq appears 3 times, rare appears 1 time
            {"freq": "high", "rare": "A"},
            {"freq": "high", "rare": "B"},
            {"freq": "high", "rare": "C"},
        ]

        self.extractor.extract_patterns(data)

        # Filter by higher support threshold
        filtered = self.extractor.filter_patterns(min_support=0.8)

        # Only "freq": "high" should meet the 0.8 threshold (support = 1.0)
        assert len(filtered) == 1
        assert filtered[0]["attribute"] == "freq"
        assert filtered[0]["value"] == "high"
        assert filtered[0]["support"] == 1.0

    def test_filter_patterns_combined(self):
        """Test filtering patterns by both type and support."""
        data = [{"attr": "common"}] * 3 + [{"attr": "rare"}]

        self.extractor.extract_patterns(data)

        # Filter by type and support
        filtered = self.extractor.filter_patterns(pattern_type="frequent_value", min_support=0.6)

        # Only "attr": "common" should meet both criteria
        assert len(filtered) == 1
        assert filtered[0]["value"] == "common"
        assert filtered[0]["support"] == 0.75


class TestIntegrationScenarios:
    """Test integrated scenarios combining knowledge graph and pattern extraction."""

    def setup_method(self):
        """Set up integration test components."""
        self.graph = KnowledgeGraph("integration_agent")
        self.extractor = PatternExtractor()

    def test_belief_network_construction(self):
        """Test building a network of interconnected beliefs."""
        # Add core beliefs
        weather_node = self.graph.add_belief("Weather is sunny", 0.9)
        activity_node = self.graph.add_belief("Outdoor activities are preferable", 0.8)
        mood_node = self.graph.add_belief("People are in good mood", 0.7)
        traffic_node = self.graph.add_belief("Traffic is lighter", 0.6)

        # Create relationships
        self.graph.add_relationship(weather_node.id, activity_node.id, "enables", 0.8)
        self.graph.add_relationship(weather_node.id, mood_node.id, "influences", 0.7)
        self.graph.add_relationship(weather_node.id, traffic_node.id, "correlates_with", 0.5)
        self.graph.add_relationship(activity_node.id, mood_node.id, "supports", 0.6)

        # Test network properties
        assert len(self.graph.nodes) == 4
        assert len(self.graph.edges) == 4

        # Test weather node connections
        weather_related = self.graph.get_related_beliefs(weather_node.id)
        assert len(weather_related) == 3

        # Test query capabilities
        high_confidence_beliefs = self.graph.query_beliefs(min_confidence=0.8)
        assert len(high_confidence_beliefs) == 2  # weather and activity

        # Test statistics
        stats = self.graph.get_statistics()
        assert stats["num_nodes"] == 4
        assert stats["num_edges"] == 4
        assert abs(stats["avg_confidence"] - (0.9 + 0.8 + 0.7 + 0.6) / 4) < 1e-10

    def test_pattern_extraction_from_beliefs(self):
        """Test extracting patterns from belief data."""
        # Add beliefs with metadata that can be analyzed for patterns
        beliefs_data = [
            {"source": "sensor", "confidence_level": "high", "domain": "weather"},
            {"source": "sensor", "confidence_level": "high", "domain": "temperature"},
            {"source": "communication", "confidence_level": "medium", "domain": "weather"},
            {"source": "inference", "confidence_level": "high", "domain": "behavior"},
            {"source": "sensor", "confidence_level": "low", "domain": "noise"},
        ]

        # Add corresponding beliefs to graph
        for i, data in enumerate(beliefs_data):
            self.graph.add_belief(f"Belief {i}", 0.5 + i * 0.1, metadata=data)

        # Extract patterns from the metadata
        patterns = self.extractor.extract_patterns(beliefs_data)
        relationships = self.extractor.extract_relationships(beliefs_data)

        # Verify pattern extraction
        assert len(patterns) > 0

        # Check for expected patterns (sensor source appears 3/5 times = 0.6)
        sensor_pattern = next(
            (p for p in patterns if p["attribute"] == "source" and p["value"] == "sensor"), None
        )
        assert sensor_pattern is not None
        assert sensor_pattern["support"] == 0.6

        # Check relationships
        if relationships:
            # Should find correlations between source and confidence_level
            source_confidence_rels = [
                r
                for r in relationships
                if (r["attribute1"] == "source" and r["attribute2"] == "confidence_level")
                or (r["attribute1"] == "confidence_level" and r["attribute2"] == "source")
            ]
            assert len(source_confidence_rels) > 0

    def test_dynamic_belief_evolution(self):
        """Test how beliefs evolve over time with new information."""
        # Initial belief about weather
        _ = self.graph.add_belief("Weather forecast: sunny", 0.6)

        # Simulate receiving new information
        messages = [
            ("weather_service", "Forecast updated: partly cloudy"),
            ("local_observer", "I see clouds forming"),
            ("weather_service", "Rain expected in afternoon"),
        ]

        # Process messages and track belief evolution
        for sender, message in messages:
            self.graph.update_from_message(message, sender)

        # Check that communication beliefs were created
        communication_beliefs = self.graph.query_beliefs(pattern="communicated")
        # weather_service messages get merged
        assert len(communication_beliefs) == 2

        # Weather service should have higher confidence due to multiple
        # messages
        weather_service_belief = next(
            b for b in communication_beliefs if "weather_service" in b.statement
        )
        assert weather_service_belief.confidence > 0.8  # Increased from base 0.8

        # Test querying by sender context
        weather_service_beliefs = self.graph.query_beliefs(pattern="weather_service")
        assert len(weather_service_beliefs) == 1

    def test_belief_contradiction_detection(self):
        """Test detection and handling of contradictory beliefs."""
        # Add contradictory beliefs
        sunny_belief = self.graph.add_belief("Weather is sunny", 0.8)
        rainy_belief = self.graph.add_belief("Weather is rainy", 0.7)

        # Explicitly mark contradiction
        contradiction_edge = self.graph.add_relationship(
            sunny_belief.id, rainy_belief.id, "contradicts", 0.9
        )

        assert contradiction_edge is not None
        assert contradiction_edge.relationship_type == "contradicts"

        # Test finding contradictory beliefs
        sunny_related = self.graph.get_related_beliefs(sunny_belief.id)
        assert len(sunny_related) == 1
        assert sunny_related[0].id == rainy_belief.id

        # Test querying with contradiction context
        weather_beliefs = self.graph.query_beliefs(pattern="weather")
        assert len(weather_beliefs) == 2

        # Could implement contradiction resolution logic here
        # For now, just verify the contradictory relationship exists
        contradiction_edges = [
            edge for edge in self.graph.edges.values() if edge.relationship_type == "contradicts"
        ]
        assert len(contradiction_edges) == 1

    def test_knowledge_graph_export_import_cycle(self):
        """Test exporting and reconstructing knowledge graph."""
        # Build a complex graph
        beliefs = []
        for i in range(3):
            belief = self.graph.add_belief(
                f"Complex belief {i}",
                0.5 + i * 0.2,
                supporting_patterns=[f"evidence_{i}"],
                metadata={"category": "test", "index": i},
            )
            beliefs.append(belief)

        # Add relationships
        for i in range(len(beliefs) - 1):
            self.graph.add_relationship(beliefs[i].id, beliefs[i + 1].id, "leads_to", 0.8)

        # Export to dictionary
        exported_dict = self.graph.to_dict()

        # Verify export structure
        assert len(exported_dict["nodes"]) == 3
        assert len(exported_dict["edges"]) == 2
        assert exported_dict["agent_id"] == "integration_agent"

        # Verify node data preservation
        for node_id, node_data in exported_dict["nodes"].items():
            original_node = self.graph.nodes[node_id]
            assert node_data["statement"] == original_node.statement
            assert node_data["confidence"] == original_node.confidence
            assert node_data["supporting_patterns"] == original_node.supporting_patterns
            assert node_data["metadata"] == original_node.metadata

        # Verify edge data preservation
        for edge_id, edge_data in exported_dict["edges"].items():
            original_edge = self.graph.edges[edge_id]
            assert edge_data["source_id"] == original_edge.source_id
            assert edge_data["target_id"] == original_edge.target_id
            assert edge_data["relationship_type"] == original_edge.relationship_type
            assert edge_data["strength"] == original_edge.strength

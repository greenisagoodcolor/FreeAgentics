"""Comprehensive tests for gmn_parser.py to boost coverage."""

import pytest

from inference.active.gmn_parser import (
    GMNEdge,
    GMNGraph,
    GMNNode,
    GMNParser,
    parse_gmn_spec,
)


class TestGMNNode:
    """Test GMNNode class."""

    def test_initialization(self):
        """Test node initialization."""
        node = GMNNode(id="test_node", type="state", properties={"num_states": 4})
        assert node.id == "test_node"
        assert node.type == "state"
        assert node.properties == {"num_states": 4}

    def test_initialization_without_properties(self):
        """Test node initialization without properties."""
        node = GMNNode(id="test", type="observation")
        assert node.id == "test"
        assert node.type == "observation"
        assert node.properties == {}


class TestGMNEdge:
    """Test GMNEdge class."""

    def test_initialization(self):
        """Test edge initialization."""
        edge = GMNEdge(
            source="node1",
            target="node2",
            type="depends_on",
            properties={"weight": 0.5},
        )
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.type == "depends_on"
        assert edge.properties == {"weight": 0.5}

    def test_initialization_without_properties(self):
        """Test edge initialization without properties."""
        edge = GMNEdge(source="a", target="b", type="generates")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.type == "generates"
        assert edge.properties == {}


class TestGMNGraph:
    """Test GMNGraph class."""

    def test_initialization(self):
        """Test graph initialization."""
        graph = GMNGraph()
        assert graph.nodes == {}
        assert graph.edges == []
        assert graph.metadata == {}

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = GMNGraph()
        node = GMNNode("test", "state")
        graph.nodes["test"] = node
        assert "test" in graph.nodes
        assert graph.nodes["test"] == node

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = GMNGraph()
        edge = GMNEdge("a", "b", "depends_on")
        graph.edges.append(edge)
        assert len(graph.edges) == 1
        assert graph.edges[0] == edge

    def test_metadata(self):
        """Test graph metadata."""
        graph = GMNGraph()
        graph.metadata = {"version": "1.0", "author": "test"}
        assert graph.metadata["version"] == "1.0"
        assert graph.metadata["author"] == "test"


class TestGMNParser:
    """Test GMNParser class."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return GMNParser()

    def test_initialization(self, parser):
        """Test parser initialization."""
        assert parser.validation_errors == []

    def test_parse_valid_dict_spec(self, parser):
        """Test parsing valid dictionary specification."""
        spec = {
            "nodes": [
                {
                    "id": "state1",
                    "type": "state",
                    "properties": {"num_states": 4},
                },
                {
                    "id": "obs1",
                    "type": "observation",
                    "properties": {"num_observations": 3},
                },
            ],
            "edges": [{"source": "state1", "target": "obs1", "type": "generates"}],
        }

        graph = parser.parse(spec)

        assert len(graph.nodes) == 2
        assert "state1" in graph.nodes
        assert "obs1" in graph.nodes
        assert graph.nodes["state1"].type == "state"
        assert graph.nodes["obs1"].type == "observation"
        assert len(graph.edges) == 1
        assert graph.edges[0].source == "state1"
        assert graph.edges[0].target == "obs1"

    def test_parse_string_spec(self, parser):
        """Test parsing string specification."""
        spec = """
        node state1 state {num_states: 4}
        node obs1 observation {num_observations: 3}
        edge state1 -> obs1 generates
        """

        graph = parser.parse(spec)

        assert len(graph.nodes) == 2
        assert "state1" in graph.nodes
        assert "obs1" in graph.nodes
        assert len(graph.edges) == 1

    def test_parse_node_from_dict(self, parser):
        """Test parsing node from dictionary."""
        node_spec = {
            "id": "test_node",
            "type": "action",
            "properties": {"num_actions": 5},
        }

        node = parser._parse_node(node_spec)

        assert node.id == "test_node"
        assert node.type == "action"
        assert node.properties["num_actions"] == 5

    def test_parse_edge_from_dict(self, parser):
        """Test parsing edge from dictionary."""
        edge_spec = {
            "source": "node1",
            "target": "node2",
            "type": "depends_on",
            "properties": {"weight": 0.8},
        }

        edge = parser._parse_edge(edge_spec)

        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.type == "depends_on"
        assert edge.properties["weight"] == 0.8

    def test_parse_string_spec_with_metadata(self, parser):
        """Test parsing string spec with metadata."""
        spec = """
        metadata {version: "1.0", model: "test_model"}
        node s1 state
        node a1 action
        edge s1 -> a1 depends_on
        """

        graph = parser.parse(spec)

        assert graph.metadata["version"] == "1.0"
        assert graph.metadata["model"] == "test_model"
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_parse_complex_properties(self, parser):
        """Test parsing complex node properties."""
        spec = """
        node belief state {
            num_states: 10,
            initial_values: [0.1, 0.2, 0.3, 0.4],
            modality: "visual"
        }
        """

        graph = parser.parse(spec)

        assert "belief" in graph.nodes
        node = graph.nodes["belief"]
        assert node.properties["num_states"] == 10
        assert node.properties["initial_values"] == [0.1, 0.2, 0.3, 0.4]
        assert node.properties["modality"] == "visual"

    def test_parse_multiple_edges(self, parser):
        """Test parsing multiple edges."""
        spec = {
            "nodes": [
                {"id": "s1", "type": "state"},
                {"id": "s2", "type": "state"},
                {"id": "a1", "type": "action"},
                {"id": "o1", "type": "observation"},
            ],
            "edges": [
                {"source": "s1", "target": "a1", "type": "depends_on"},
                {"source": "s2", "target": "a1", "type": "depends_on"},
                {"source": "a1", "target": "o1", "type": "generates"},
            ],
        }

        graph = parser.parse(spec)

        assert len(graph.edges) == 3
        edge_types = [e.type for e in graph.edges]
        assert edge_types.count("depends_on") == 2
        assert edge_types.count("generates") == 1

    def test_validation_missing_node_id(self, parser):
        """Test validation catches missing node ID."""
        spec = {"nodes": [{"type": "state"}]}  # Missing ID

        with pytest.raises(ValueError, match="GMN validation errors"):
            parser.parse(spec)

    def test_validation_missing_edge_source(self, parser):
        """Test validation catches missing edge source."""
        spec = {
            "nodes": [{"id": "n1", "type": "state"}],
            "edges": [{"target": "n1", "type": "depends_on"}],  # Missing source
        }

        with pytest.raises(ValueError, match="GMN validation errors"):
            parser.parse(spec)

    def test_validation_invalid_edge_reference(self, parser):
        """Test validation catches invalid edge references."""
        spec = {
            "nodes": [{"id": "n1", "type": "state"}],
            "edges": [
                {
                    "source": "n1",
                    "target": "n2",
                    "type": "depends_on",
                }  # n2 doesn't exist
            ],
        }

        with pytest.raises(ValueError, match="GMN validation errors"):
            parser.parse(spec)

    def test_validation_unknown_node_type(self, parser):
        """Test validation handles unknown node types."""
        spec = {"nodes": [{"id": "n1", "type": "unknown_type"}]}

        with pytest.raises(ValueError, match="Unknown node type"):
            parser.parse(spec)

    def test_validate_graph_structure(self, parser):
        """Test graph structure validation."""
        graph = GMNGraph()

        # Add nodes
        graph.nodes["s1"] = GMNNode("s1", "state")
        graph.nodes["a1"] = GMNNode("a1", "action")

        # Add valid edge
        graph.edges.append(GMNEdge("s1", "a1", "depends_on"))

        # Should not raise
        parser._validate_graph(graph)
        assert parser.validation_errors == []

    def test_empty_spec(self, parser):
        """Test parsing empty specification."""
        spec = {}
        graph = parser.parse(spec)

        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.metadata == {}

    def test_parse_gmn_format_variations(self, parser):
        """Test parsing various GMN format variations."""
        # Test with different whitespace
        spec1 = "node   s1   state"
        graph1 = parser.parse(spec1)
        assert "s1" in graph1.nodes

        # Test with tabs
        spec2 = "node\ts1\tstate"
        graph2 = parser.parse(spec2)
        assert "s1" in graph2.nodes

        # Test with properties on same line
        spec3 = "node s1 state {num_states: 5}"
        graph3 = parser.parse(spec3)
        assert graph3.nodes["s1"].properties["num_states"] == 5

    def test_parse_quoted_strings(self, parser):
        """Test parsing quoted string values."""
        spec = """
        node agent state {
            name: "test agent",
            description: "This is a test"
        }
        """

        graph = parser.parse(spec)

        assert graph.nodes["agent"].properties["name"] == "test agent"
        assert graph.nodes["agent"].properties["description"] == "This is a test"

    def test_edge_properties_parsing(self, parser):
        """Test parsing edge properties."""
        spec = """
        node s1 state
        node s2 state
        edge s1 -> s2 transition {probability: 0.7, condition: "active"}
        """

        graph = parser.parse(spec)

        edge = graph.edges[0]
        assert edge.properties["probability"] == 0.7
        assert edge.properties["condition"] == "active"


class TestParseGMNSpec:
    """Test the parse_gmn_spec convenience function."""

    def test_parse_dict_spec(self):
        """Test parsing dictionary spec via convenience function."""
        spec = {"nodes": [{"id": "n1", "type": "state"}], "edges": []}

        graph = parse_gmn_spec(spec)

        assert "n1" in graph.nodes
        assert graph.nodes["n1"].type == "state"

    def test_parse_string_spec(self):
        """Test parsing string spec via convenience function."""
        spec = "node n1 state"
        graph = parse_gmn_spec(spec)

        assert "n1" in graph.nodes
        assert graph.nodes["n1"].type == "state"

    def test_parse_with_validation_error(self):
        """Test that validation errors are properly raised."""
        spec = {"nodes": [{"type": "state"}]}  # Missing ID

        with pytest.raises(ValueError, match="GMN validation errors"):
            parse_gmn_spec(spec)


class TestAdvancedScenarios:
    """Test advanced parsing scenarios."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return GMNParser()

    def test_hierarchical_graph(self, parser):
        """Test parsing hierarchical graph structures."""
        spec = {
            "nodes": [
                {"id": "parent", "type": "state", "properties": {"level": 0}},
                {
                    "id": "child1",
                    "type": "state",
                    "properties": {"level": 1, "parent": "parent"},
                },
                {
                    "id": "child2",
                    "type": "state",
                    "properties": {"level": 1, "parent": "parent"},
                },
            ],
            "edges": [
                {"source": "parent", "target": "child1", "type": "hierarchy"},
                {"source": "parent", "target": "child2", "type": "hierarchy"},
            ],
        }

        graph = parser.parse(spec)

        assert len(graph.nodes) == 3
        assert graph.nodes["child1"].properties["parent"] == "parent"
        assert graph.nodes["child2"].properties["parent"] == "parent"

    def test_cyclic_dependencies(self, parser):
        """Test parsing graphs with cycles."""
        spec = {
            "nodes": [
                {"id": "a", "type": "state"},
                {"id": "b", "type": "state"},
                {"id": "c", "type": "state"},
            ],
            "edges": [
                {"source": "a", "target": "b", "type": "depends_on"},
                {"source": "b", "target": "c", "type": "depends_on"},
                {"source": "c", "target": "a", "type": "depends_on"},
            ],
        }

        graph = parser.parse(spec)

        assert len(graph.edges) == 3
        # Verify cycle exists
        sources = [e.source for e in graph.edges]
        targets = [e.target for e in graph.edges]
        assert set(sources) == set(targets) == {"a", "b", "c"}

    def test_multi_modal_observations(self, parser):
        """Test parsing multi-modal observation nodes."""
        spec = {
            "nodes": [
                {
                    "id": "visual_obs",
                    "type": "observation",
                    "properties": {
                        "modality": "visual",
                        "num_observations": 100,
                        "shape": [10, 10],
                    },
                },
                {
                    "id": "audio_obs",
                    "type": "observation",
                    "properties": {
                        "modality": "audio",
                        "num_observations": 50,
                        "sample_rate": 16000,
                    },
                },
            ]
        }

        graph = parser.parse(spec)

        assert graph.nodes["visual_obs"].properties["modality"] == "visual"
        assert graph.nodes["visual_obs"].properties["shape"] == [10, 10]
        assert graph.nodes["audio_obs"].properties["sample_rate"] == 16000

    def test_temporal_edges(self, parser):
        """Test parsing temporal edges."""
        spec = """
        node s_t state
        node s_t1 state
        edge s_t -> s_t1 temporal {delay: 1, type: "next_state"}
        """

        graph = parser.parse(spec)

        edge = graph.edges[0]
        assert edge.type == "temporal"
        assert edge.properties["delay"] == 1
        assert edge.properties["type"] == "next_state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

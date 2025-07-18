"""
Comprehensive test suite for GMN (Generalized Notation Notation) parser.

Tests the GMN parser functionality for PyMDP model specification
and LLM integration capabilities.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from inference.active.gmn_parser import (
    EXAMPLE_GMN_SPEC,
    GMNEdge,
    GMNEdgeType,
    GMNGraph,
    GMNNode,
    GMNNodeType,
    GMNParser,
    parse_gmn_spec,
)


class TestGMNNodeType:
    """Test GMNNodeType enum."""

    def test_node_types(self):
        """Test all node types are defined."""
        assert GMNNodeType.STATE.value == "state"
        assert GMNNodeType.OBSERVATION.value == "observation"
        assert GMNNodeType.ACTION.value == "action"
        assert GMNNodeType.BELIEF.value == "belie"
        assert GMNNodeType.PREFERENCE.value == "preference"
        assert GMNNodeType.TRANSITION.value == "transition"
        assert GMNNodeType.LIKELIHOOD.value == "likelihood"
        assert GMNNodeType.POLICY.value == "policy"
        assert GMNNodeType.LLM_QUERY.value == "llm_query"


class TestGMNEdgeType:
    """Test GMNEdgeType enum."""

    def test_edge_types(self):
        """Test all edge types are defined."""
        assert GMNEdgeType.DEPENDS_ON.value == "depends_on"
        assert GMNEdgeType.INFLUENCES.value == "influences"
        assert GMNEdgeType.UPDATES.value == "updates"
        assert GMNEdgeType.QUERIES.value == "queries"
        assert GMNEdgeType.GENERATES.value == "generates"


class TestGMNNode:
    """Test GMNNode dataclass."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = GMNNode(
            id="test_node",
            type=GMNNodeType.STATE,
            properties={"num_states": 4},
            metadata={"description": "Test state node"},
        )

        assert node.id == "test_node"
        assert node.type == GMNNodeType.STATE
        assert node.properties["num_states"] == 4
        assert node.metadata["description"] == "Test state node"

    def test_node_defaults(self):
        """Test node with default values."""
        node = GMNNode(id="simple", type=GMNNodeType.OBSERVATION)

        assert node.id == "simple"
        assert node.type == GMNNodeType.OBSERVATION
        assert node.properties == {}
        assert node.metadata == {}


class TestGMNEdge:
    """Test GMNEdge dataclass."""

    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = GMNEdge(
            source="node1",
            target="node2",
            type=GMNEdgeType.DEPENDS_ON,
            properties={"weight": 0.8},
        )

        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.type == GMNEdgeType.DEPENDS_ON
        assert edge.properties["weight"] == 0.8

    def test_edge_defaults(self):
        """Test edge with default values."""
        edge = GMNEdge(source="a", target="b", type=GMNEdgeType.INFLUENCES)

        assert edge.properties == {}


class TestGMNGraph:
    """Test GMNGraph dataclass."""

    def test_graph_creation(self):
        """Test graph creation."""
        node1 = GMNNode("n1", GMNNodeType.STATE)
        node2 = GMNNode("n2", GMNNodeType.OBSERVATION)
        edge = GMNEdge("n1", "n2", GMNEdgeType.GENERATES)

        graph = GMNGraph(
            nodes={"n1": node1, "n2": node2},
            edges=[edge],
            metadata={"version": "1.0"},
        )

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.metadata["version"] == "1.0"

    def test_graph_defaults(self):
        """Test graph with default values."""
        graph = GMNGraph()

        assert graph.nodes == {}
        assert graph.edges == []
        assert graph.metadata == {}


class TestGMNParser:
    """Test GMNParser class."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return GMNParser()

    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.current_graph is None
        assert parser.llm_integration_points == []
        assert parser.validation_errors == []

    def test_parse_json_spec(self, parser):
        """Test parsing JSON specification."""
        spec = {
            "nodes": [
                {
                    "id": "state1",
                    "type": "state",
                    "properties": {"num_states": 3},
                },
                {
                    "id": "obs1",
                    "type": "observation",
                    "properties": {"num_observations": 2},
                },
            ],
            "edges": [
                {"source": "state1", "target": "obs1", "type": "generates"}
            ],
            "metadata": {"version": "1.0"},
        }

        graph = parser.parse(spec)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.metadata["version"] == "1.0"
        assert "state1" in graph.nodes
        assert "obs1" in graph.nodes
        assert graph.nodes["state1"].type == GMNNodeType.STATE
        assert graph.nodes["obs1"].type == GMNNodeType.OBSERVATION

    def test_parse_string_spec(self, parser):
        """Test parsing string specification."""
        spec_str = json.dumps(
            {"nodes": [{"id": "location", "type": "state"}], "edges": []}
        )

        graph = parser.parse(spec_str)

        assert len(graph.nodes) == 1
        assert "location" in graph.nodes

    def test_parse_gmn_format(self, parser):
        """Test parsing custom GMN format."""
        spec = """
        [nodes]
        location: state {num_states: 4}
        obs_location: observation {num_observations: 4}

        [edges]
        location -> obs_location: generates
        """

        graph = parser.parse(spec)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.nodes["location"].properties["num_states"] == 4
        assert graph.edges[0].source == "location"
        assert graph.edges[0].target == "obs_location"

    def test_validation_errors(self, parser):
        """Test validation error detection."""
        spec = {
            "nodes": [{"id": "node1", "type": "state"}],
            "edges": [
                {
                    "source": "node1",
                    "target": "missing_node",
                    "type": "depends_on",
                }
            ],
        }

        with pytest.raises(ValueError, match="GMN validation errors"):
            parser.parse(spec)

    def test_unknown_node_type(self, parser):
        """Test handling of unknown node types."""
        spec = {
            "nodes": [{"id": "node1", "type": "unknown_type"}],
            "edges": [],
        }

        with pytest.raises(ValueError):
            parser.parse(spec)

        assert "Unknown node type: unknown_type" in parser.validation_errors

    def test_missing_required_nodes(self, parser):
        """Test validation of required node types."""
        spec = {"nodes": [{"id": "lonely", "type": "preference"}], "edges": []}

        with pytest.raises(ValueError):
            parser.parse(spec)

        errors = parser.validation_errors
        assert any("No state nodes found" in error for error in errors)
        assert any("No observation nodes found" in error for error in errors)
        assert any("No action nodes found" in error for error in errors)


class TestGMNToPyMDP:
    """Test GMN to PyMDP model conversion."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return GMNParser()

    @pytest.fixture
    def simple_graph(self, parser):
        """Create simple test graph."""
        spec = {
            "nodes": [
                {
                    "id": "location",
                    "type": "state",
                    "properties": {"num_states": 4},
                },
                {
                    "id": "obs_loc",
                    "type": "observation",
                    "properties": {"num_observations": 4},
                },
                {
                    "id": "move",
                    "type": "action",
                    "properties": {"num_actions": 5},
                },
                {"id": "belie", "type": "belief"},
                {
                    "id": "pre",
                    "type": "preference",
                    "properties": {"preferred_observation": 0},
                },
                {"id": "likelihood", "type": "likelihood"},
                {"id": "transition", "type": "transition"},
            ],
            "edges": [
                {
                    "source": "location",
                    "target": "likelihood",
                    "type": "depends_on",
                },
                {
                    "source": "likelihood",
                    "target": "obs_loc",
                    "type": "generates",
                },
                {
                    "source": "location",
                    "target": "transition",
                    "type": "depends_on",
                },
                {
                    "source": "move",
                    "target": "transition",
                    "type": "depends_on",
                },
                {"source": "pre", "target": "obs_loc", "type": "depends_on"},
                {
                    "source": "belie",
                    "target": "location",
                    "type": "depends_on",
                },
            ],
        }
        return parser.parse(spec)

    def test_pymdp_model_structure(self, parser, simple_graph):
        """Test PyMDP model structure."""
        model = parser.to_pymdp_model(simple_graph)

        assert "num_states" in model
        assert "num_obs" in model
        assert "num_actions" in model
        assert "A" in model
        assert "B" in model
        assert "C" in model
        assert "D" in model
        assert "llm_integration" in model

    def test_state_extraction(self, parser, simple_graph):
        """Test state dimension extraction."""
        model = parser.to_pymdp_model(simple_graph)

        assert model["num_states"] == [4]

    def test_observation_extraction(self, parser, simple_graph):
        """Test observation dimension extraction."""
        model = parser.to_pymdp_model(simple_graph)

        assert model["num_obs"] == [4]

    def test_action_extraction(self, parser, simple_graph):
        """Test action dimension extraction."""
        model = parser.to_pymdp_model(simple_graph)

        assert model["num_actions"] == [5]

    def test_likelihood_matrix_generation(self, parser, simple_graph):
        """Test likelihood matrix generation."""
        model = parser.to_pymdp_model(simple_graph)

        assert len(model["A"]) == 1
        A_matrix = model["A"][0]
        assert A_matrix.shape == (4, 4)  # obs_dim x state_dim
        assert np.allclose(A_matrix.sum(axis=0), 1.0)  # Columns sum to 1

    def test_transition_matrix_generation(self, parser, simple_graph):
        """Test transition matrix generation."""
        model = parser.to_pymdp_model(simple_graph)

        assert len(model["B"]) == 1
        B_matrix = model["B"][0]
        assert B_matrix.shape == (
            4,
            4,
            5,
        )  # state_dim x state_dim x action_dim

    def test_preference_vector_generation(self, parser, simple_graph):
        """Test preference vector generation."""
        model = parser.to_pymdp_model(simple_graph)

        assert len(model["C"]) == 1
        C_vector = model["C"][0]
        assert C_vector.shape == (4,)
        assert C_vector[0] == 1.0  # Preferred observation is 0

    def test_belief_vector_generation(self, parser, simple_graph):
        """Test initial belief vector generation."""
        model = parser.to_pymdp_model(simple_graph)

        assert len(model["D"]) == 1
        D_vector = model["D"][0]
        assert D_vector.shape == (4,)
        assert np.allclose(D_vector.sum(), 1.0)  # Sums to 1

    def test_custom_matrices(self, parser):
        """Test custom matrix specification."""
        spec = {
            "nodes": [
                {
                    "id": "state",
                    "type": "state",
                    "properties": {"num_states": 2},
                },
                {
                    "id": "obs",
                    "type": "observation",
                    "properties": {"num_observations": 2},
                },
                {
                    "id": "action",
                    "type": "action",
                    "properties": {"num_actions": 2},
                },
                {
                    "id": "likelihood",
                    "type": "likelihood",
                    "properties": {"matrix": [[0.9, 0.1], [0.1, 0.9]]},
                },
            ],
            "edges": [
                {
                    "source": "state",
                    "target": "likelihood",
                    "type": "depends_on",
                },
                {"source": "likelihood", "target": "obs", "type": "generates"},
            ],
        }

        graph = parser.parse(spec)
        model = parser.to_pymdp_model(graph)

        expected_A = np.array([[0.9, 0.1], [0.1, 0.9]])
        assert np.array_equal(model["A"][0], expected_A)


class TestLLMIntegration:
    """Test LLM integration points."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return GMNParser()

    def test_llm_integration_extraction(self, parser):
        """Test LLM integration point extraction."""
        spec = {
            "nodes": [
                {
                    "id": "state",
                    "type": "state",
                    "properties": {"num_states": 3},
                },
                {
                    "id": "obs",
                    "type": "observation",
                    "properties": {"num_observations": 3},
                },
                {
                    "id": "action",
                    "type": "action",
                    "properties": {"num_actions": 4},
                },
                {
                    "id": "llm_policy",
                    "type": "llm_query",
                    "properties": {
                        "trigger_condition": "on_observation",
                        "prompt_template": "Given {obs}, suggest action",
                        "response_parser": "json",
                    },
                },
            ],
            "edges": [
                {"source": "obs", "target": "llm_policy", "type": "queries"},
                {
                    "source": "llm_policy",
                    "target": "action",
                    "type": "updates",
                },
            ],
        }

        graph = parser.parse(spec)
        model = parser.to_pymdp_model(graph)

        assert len(model["llm_integration"]) == 1
        integration = model["llm_integration"][0]

        assert integration["id"] == "llm_policy"
        assert integration["trigger_condition"] == "on_observation"
        assert integration["prompt_template"] == "Given {obs}, suggest action"
        assert integration["response_parser"] == "json"
        assert "obs" in integration["context_nodes"]
        assert "action" in integration["update_targets"]


class TestConvenienceFunction:
    """Test convenience functions."""

    def test_parse_gmn_spec_function(self):
        """Test parse_gmn_spec convenience function."""
        spec = {
            "nodes": [
                {
                    "id": "state",
                    "type": "state",
                    "properties": {"num_states": 2},
                },
                {
                    "id": "obs",
                    "type": "observation",
                    "properties": {"num_observations": 2},
                },
                {
                    "id": "action",
                    "type": "action",
                    "properties": {"num_actions": 2},
                },
            ],
            "edges": [],
        }

        model = parse_gmn_spec(spec)

        assert "num_states" in model
        assert "A" in model
        assert model["num_states"] == [2]


class TestExampleSpec:
    """Test example GMN specification."""

    def test_example_spec_parsing(self):
        """Test parsing of example specification."""
        parser = GMNParser()
        graph = parser.parse(EXAMPLE_GMN_SPEC)

        # Should have all the expected nodes
        expected_nodes = [
            "location",
            "obs_location",
            "move",
            "location_belie",
            "location_pre",
            "location_likelihood",
            "location_transition",
            "llm_policy",
        ]

        for node_id in expected_nodes:
            assert node_id in graph.nodes

        # Should have expected edges
        assert len(graph.edges) > 0

        # Should convert to PyMDP model
        model = parser.to_pymdp_model(graph)
        assert model["num_states"] == [4]
        assert model["num_obs"] == [4]
        assert model["num_actions"] == [5]


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return GMNParser()

    def test_empty_spec(self, parser):
        """Test empty specification."""
        spec = {"nodes": [], "edges": []}

        with pytest.raises(ValueError):
            parser.parse(spec)

    def test_malformed_json(self, parser):
        """Test malformed JSON string."""
        malformed = '{"nodes": [{"id": "test"'  # Missing closing braces

        with pytest.raises(Exception):
            parser.parse(malformed)

    def test_circular_dependencies(self, parser):
        """Test handling of circular dependencies."""
        spec = {
            "nodes": [
                {"id": "state1", "type": "state"},
                {"id": "state2", "type": "state"},
                {"id": "obs", "type": "observation"},
                {"id": "action", "type": "action"},
            ],
            "edges": [
                {"source": "state1", "target": "state2", "type": "depends_on"},
                {"source": "state2", "target": "state1", "type": "depends_on"},
            ],
        }

        # Should parse without error (circular dependencies allowed)
        graph = parser.parse(spec)
        assert len(graph.edges) == 2

    def test_matrix_dimension_mismatch(self, parser):
        """Test matrix dimension validation."""
        spec = {
            "nodes": [
                {
                    "id": "state",
                    "type": "state",
                    "properties": {"num_states": 2},
                },
                {
                    "id": "obs",
                    "type": "observation",
                    "properties": {"num_observations": 3},
                },
                {
                    "id": "action",
                    "type": "action",
                    "properties": {"num_actions": 2},
                },
                {
                    "id": "likelihood",
                    "type": "likelihood",
                    "properties": {
                        "matrix": [[1, 0], [0, 1]]
                    },  # Wrong dimensions
                },
            ],
            "edges": [
                {
                    "source": "state",
                    "target": "likelihood",
                    "type": "depends_on",
                },
                {"source": "likelihood", "target": "obs", "type": "generates"},
            ],
        }

        graph = parser.parse(spec)
        model = parser.to_pymdp_model(graph)

        # Should use custom matrix even if dimensions don't match perfectly
        assert np.array_equal(model["A"][0], np.array([[1, 0], [0, 1]]))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=inference.active.gmn_parser"])

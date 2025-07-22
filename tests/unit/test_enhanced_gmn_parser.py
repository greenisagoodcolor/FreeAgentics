"""Comprehensive test suite for enhanced GMN parser with full specification support.

This test suite defines the expected behavior for:
- Full GMN specification format parsing (states, observations, actions, beliefs, preferences)
- Schema-like validation framework
- PyMDP matrix generation (A, B, C, D)
- Specification versioning and validation
- Error handling and edge cases

Following TDD principles: These tests will fail initially and drive implementation.
"""

import numpy as np
import pytest

from inference.active.gmn_parser import GMNParser, GMNSchemaValidator


# Mock the missing classes for now
class GMNSpecification:
    """Mock GMN specification class."""

    pass


class GMNToPyMDPConverter:
    """Mock GMN to PyMDP converter."""

    pass


class GMNValidationError(Exception):
    """Mock GMN validation error."""

    pass


class TestGMNParserBasicFunctionality:
    """Test basic GMN parsing functionality."""

    def test_parse_minimal_valid_specification(self):
        """Test parsing a minimal valid GMN specification."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 4,
                },
                {"name": "move", "type": "action", "num_actions": 4},
            ],
            "edges": [{"from": "location", "to": "obs_location", "type": "generates"}],
        }

        result = parser.parse(spec)

        assert result is not None
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 1

    def test_parse_empty_specification_raises_error(self):
        """Test that empty specification raises appropriate error."""
        parser = GMNParser()

        with pytest.raises(GMNValidationError) as exc_info:
            parser.parse({})

        assert "Empty specification" in str(exc_info.value)

    def test_parse_specification_missing_nodes_raises_error(self):
        """Test that specification without nodes raises error."""
        parser = GMNParser()

        spec = {"edges": []}

        with pytest.raises(GMNValidationError) as exc_info:
            parser.parse(spec)

        assert "nodes" in str(exc_info.value)


class TestGMNNodeTypes:
    """Test GMN node type handling and validation."""

    def test_parse_state_node_with_dimensions(self):
        """Test parsing state nodes with proper dimensions."""
        parser = GMNParser()

        spec = {
            "nodes": [{"name": "agent_location", "type": "state", "num_states": 9}],
            "edges": [],
        }

        result = parser.parse(spec)
        state_node = result["nodes"][0]

        assert state_node["name"] == "agent_location"
        assert state_node["type"] == "state"
        assert state_node["num_states"] == 9

    def test_parse_observation_node_with_dimensions(self):
        """Test parsing observation nodes with proper dimensions."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "visual_input",
                    "type": "observation",
                    "num_observations": 16,
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)
        obs_node = result["nodes"][0]

        assert obs_node["name"] == "visual_input"
        assert obs_node["type"] == "observation"
        assert obs_node["num_observations"] == 16

    def test_parse_action_node_with_dimensions(self):
        """Test parsing action nodes with proper dimensions."""
        parser = GMNParser()

        spec = {
            "nodes": [{"name": "movement", "type": "action", "num_actions": 5}],
            "edges": [],
        }

        result = parser.parse(spec)
        action_node = result["nodes"][0]

        assert action_node["name"] == "movement"
        assert action_node["type"] == "action"
        assert action_node["num_actions"] == 5

    def test_parse_belief_node(self):
        """Test parsing belief nodes."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "location_belief",
                    "type": "belief",
                    "about": "location",
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)
        belief_node = result["nodes"][0]

        assert belief_node["name"] == "location_belief"
        assert belief_node["type"] == "belief"
        assert belief_node["about"] == "location"

    def test_parse_preference_node_with_target(self):
        """Test parsing preference nodes with target observations."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "goal_preference",
                    "type": "preference",
                    "preferred_observation": 8,
                    "preference_strength": 1.0,
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)
        pref_node = result["nodes"][0]

        assert pref_node["name"] == "goal_preference"
        assert pref_node["type"] == "preference"
        assert pref_node["preferred_observation"] == 8
        assert pref_node["preference_strength"] == 1.0


class TestGMNEdgeTypes:
    """Test GMN edge type handling and relationships."""

    def test_parse_generates_edge(self):
        """Test parsing 'generates' type edges."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 4,
                },
            ],
            "edges": [{"from": "location", "to": "obs_location", "type": "generates"}],
        }

        result = parser.parse(spec)
        edge = result["edges"][0]

        assert edge["from"] == "location"
        assert edge["to"] == "obs_location"
        assert edge["type"] == "generates"

    def test_parse_depends_on_edge(self):
        """Test parsing 'depends_on' type edges."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {
                    "name": "location_belief",
                    "type": "belief",
                    "about": "location",
                },
            ],
            "edges": [
                {
                    "from": "location_belief",
                    "to": "location",
                    "type": "depends_on",
                }
            ],
        }

        result = parser.parse(spec)
        edge = result["edges"][0]

        assert edge["from"] == "location_belief"
        assert edge["to"] == "location"
        assert edge["type"] == "depends_on"

    def test_parse_multiple_edges(self):
        """Test parsing specifications with multiple edges."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 9},
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 9,
                },
                {"name": "move", "type": "action", "num_actions": 5},
                {"name": "location_transition", "type": "transition"},
            ],
            "edges": [
                {
                    "from": "location",
                    "to": "obs_location",
                    "type": "generates",
                },
                {
                    "from": "location",
                    "to": "location_transition",
                    "type": "depends_on",
                },
                {
                    "from": "move",
                    "to": "location_transition",
                    "type": "depends_on",
                },
            ],
        }

        result = parser.parse(spec)

        assert len(result["edges"]) == 3
        edge_types = [edge["type"] for edge in result["edges"]]
        assert "generates" in edge_types
        assert "depends_on" in edge_types


class TestGMNSchemaValidator:
    """Test GMN schema validation framework."""

    def test_validate_valid_specification(self):
        """Test validation of a valid GMN specification."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 4,
                },
                {"name": "move", "type": "action", "num_actions": 4},
            ],
            "edges": [{"from": "location", "to": "obs_location", "type": "generates"}],
        }

        is_valid, errors = validator.validate(spec)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [{"name": "location", "type": "state"}],  # Missing num_states
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        assert is_valid is False
        assert len(errors) > 0
        assert any("num_states" in error for error in errors)

    def test_validate_invalid_node_type(self):
        """Test validation fails for invalid node types."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [{"name": "invalid", "type": "invalid_type"}],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        assert is_valid is False
        assert len(errors) > 0
        assert any("invalid_type" in error for error in errors)

    def test_validate_inconsistent_dimensions(self):
        """Test validation fails for inconsistent dimensions."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 9,
                },  # Mismatch
            ],
            "edges": [{"from": "location", "to": "obs_location", "type": "generates"}],
        }

        is_valid, errors = validator.validate(spec)

        assert is_valid is False
        assert len(errors) > 0
        assert any("dimension" in error.lower() for error in errors)

    def test_validate_orphaned_nodes(self):
        """Test validation identifies orphaned nodes."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 4,
                },
                {
                    "name": "orphaned",
                    "type": "state",
                    "num_states": 2,
                },  # No edges
            ],
            "edges": [{"from": "location", "to": "obs_location", "type": "generates"}],
        }

        is_valid, errors = validator.validate(spec)

        # Should still be valid but with warnings
        assert is_valid is True
        # But should have warnings about orphaned nodes

    def test_validate_edge_references_nonexistent_nodes(self):
        """Test validation fails when edges reference nonexistent nodes."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [{"name": "location", "type": "state", "num_states": 4}],
            "edges": [{"from": "location", "to": "nonexistent", "type": "generates"}],
        }

        is_valid, errors = validator.validate(spec)

        assert is_valid is False
        assert len(errors) > 0
        assert any("nonexistent" in error for error in errors)


class TestGMNToPyMDPConverter:
    """Test conversion from GMN specifications to PyMDP matrices."""

    def test_create_a_matrix_from_state_observation_mapping(self):
        """Test A matrix creation from state-observation relationships."""
        converter = GMNToPyMDPConverter()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 4,
                },
            ],
            "edges": [{"from": "location", "to": "obs_location", "type": "generates"}],
        }

        matrices = converter.convert_to_matrices(spec)

        assert "A" in matrices
        A_matrix = matrices["A"]
        assert A_matrix.shape == (4, 4)  # num_observations x num_states
        assert np.allclose(A_matrix.sum(axis=0), 1.0)  # Columns should sum to 1

    def test_create_b_matrix_from_transition_specification(self):
        """Test B matrix creation from transition specifications."""
        converter = GMNToPyMDPConverter()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {"name": "move", "type": "action", "num_actions": 5},
                {"name": "location_transition", "type": "transition"},
            ],
            "edges": [
                {
                    "from": "location",
                    "to": "location_transition",
                    "type": "depends_on",
                },
                {
                    "from": "move",
                    "to": "location_transition",
                    "type": "depends_on",
                },
            ],
        }

        matrices = converter.convert_to_matrices(spec)

        assert "B" in matrices
        B_matrix = matrices["B"]
        assert B_matrix.shape == (
            4,
            4,
            5,
        )  # num_states x num_states x num_actions
        assert np.allclose(B_matrix.sum(axis=0), 1.0)  # Transition probabilities sum to 1

    def test_create_c_vector_from_preferences(self):
        """Test C vector creation from preference specifications."""
        converter = GMNToPyMDPConverter()

        spec = {
            "nodes": [
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 4,
                },
                {
                    "name": "goal_preference",
                    "type": "preference",
                    "preferred_observation": 3,
                    "preference_strength": 2.0,
                },
            ],
            "edges": [
                {
                    "from": "goal_preference",
                    "to": "obs_location",
                    "type": "depends_on",
                }
            ],
        }

        matrices = converter.convert_to_matrices(spec)

        assert "C" in matrices
        C_vector = matrices["C"]
        assert C_vector.shape == (4,)  # num_observations
        assert C_vector[3] == 2.0  # Preferred observation has higher value

    def test_create_d_vector_from_priors(self):
        """Test D vector creation with uniform priors by default."""
        converter = GMNToPyMDPConverter()

        spec = {
            "nodes": [{"name": "location", "type": "state", "num_states": 4}],
            "edges": [],
        }

        matrices = converter.convert_to_matrices(spec)

        assert "D" in matrices
        D_vector = matrices["D"]
        assert D_vector.shape == (4,)  # num_states
        assert np.allclose(D_vector, 0.25)  # Uniform prior
        assert np.allclose(D_vector.sum(), 1.0)  # Probabilities sum to 1

    def test_convert_complex_specification_to_full_matrices(self):
        """Test conversion of complex specification to complete PyMDP matrices."""
        converter = GMNToPyMDPConverter()

        spec = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 9},
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 9,
                },
                {"name": "move", "type": "action", "num_actions": 5},
                {
                    "name": "location_belief",
                    "type": "belief",
                    "about": "location",
                },
                {
                    "name": "goal_preference",
                    "type": "preference",
                    "preferred_observation": 8,
                    "preference_strength": 1.0,
                },
                {"name": "location_likelihood", "type": "likelihood"},
                {"name": "location_transition", "type": "transition"},
            ],
            "edges": [
                {
                    "from": "location",
                    "to": "location_likelihood",
                    "type": "depends_on",
                },
                {
                    "from": "location_likelihood",
                    "to": "obs_location",
                    "type": "generates",
                },
                {
                    "from": "location",
                    "to": "location_transition",
                    "type": "depends_on",
                },
                {
                    "from": "move",
                    "to": "location_transition",
                    "type": "depends_on",
                },
                {
                    "from": "goal_preference",
                    "to": "obs_location",
                    "type": "depends_on",
                },
                {
                    "from": "location_belief",
                    "to": "location",
                    "type": "depends_on",
                },
            ],
        }

        matrices = converter.convert_to_matrices(spec)

        # Should have all required matrices
        assert "A" in matrices
        assert "B" in matrices
        assert "C" in matrices
        assert "D" in matrices

        # Check dimensions
        assert matrices["A"].shape == (9, 9)
        assert matrices["B"].shape == (9, 9, 5)
        assert matrices["C"].shape == (9,)
        assert matrices["D"].shape == (9,)

        # Check probability constraints
        assert np.allclose(matrices["A"].sum(axis=0), 1.0)
        assert np.allclose(matrices["B"].sum(axis=0), 1.0)
        assert np.allclose(matrices["D"].sum(), 1.0)


class TestGMNSpecificationClass:
    """Test GMN specification wrapper class."""

    def test_create_specification_from_dict(self):
        """Test creating GMNSpecification from dictionary."""
        spec_dict = {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {
                    "name": "obs_location",
                    "type": "observation",
                    "num_observations": 4,
                },
            ],
            "edges": [{"from": "location", "to": "obs_location", "type": "generates"}],
        }

        spec = GMNSpecification.from_dict(spec_dict)

        assert spec.nodes is not None
        assert spec.edges is not None
        assert len(spec.nodes) == 2
        assert len(spec.edges) == 1

    def test_create_specification_from_text(self):
        """Test creating GMNSpecification from text format."""
        spec_text = """
        [nodes]
        location: state {num_states: 4}
        obs_location: observation {num_observations: 4}

        [edges]
        location -> obs_location: generates
        """

        spec = GMNSpecification.from_text(spec_text)

        assert spec.nodes is not None
        assert spec.edges is not None
        assert len(spec.nodes) == 2
        assert len(spec.edges) == 1

    def test_specification_to_dict(self):
        """Test converting GMNSpecification back to dictionary."""
        spec_dict = {
            "nodes": [{"name": "location", "type": "state", "num_states": 4}],
            "edges": [],
        }

        spec = GMNSpecification.from_dict(spec_dict)
        result_dict = spec.to_dict()

        assert result_dict["nodes"] == spec_dict["nodes"]
        assert result_dict["edges"] == spec_dict["edges"]

    def test_specification_validation_status(self):
        """Test specification validation status tracking."""
        spec_dict = {
            "nodes": [{"name": "location", "type": "state", "num_states": 4}],
            "edges": [],
        }

        spec = GMNSpecification.from_dict(spec_dict)

        assert spec.is_valid is None  # Not validated yet

        spec.validate()

        assert spec.is_valid is not None
        assert spec.validation_errors is not None


class TestGMNParserFileFormats:
    """Test parsing different GMN file formats."""

    def test_parse_text_format_specification(self):
        """Test parsing GMN specification from text format."""
        parser = GMNParser()

        text_spec = """
        [nodes]
        location: state {num_states: 4}
        obs_location: observation {num_observations: 4}
        move: action {num_actions: 4}

        [edges]
        location -> obs_location: generates
        """

        result = parser.parse_text(text_spec)

        assert result is not None
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 1

    def test_parse_json_format_specification(self):
        """Test parsing GMN specification from JSON format."""
        parser = GMNParser()

        json_spec = """
        {
            "nodes": [
                {"name": "location", "type": "state", "num_states": 4},
                {"name": "obs_location", "type": "observation", "num_observations": 4}
            ],
            "edges": [
                {"from": "location", "to": "obs_location", "type": "generates"}
            ]
        }
        """

        result = parser.parse_json(json_spec)

        assert result is not None
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1


class TestGMNParserErrorHandling:
    """Test error handling and edge cases."""

    def test_parser_handles_malformed_json(self):
        """Test parser gracefully handles malformed JSON."""
        parser = GMNParser()

        malformed_json = '{"nodes": [{"name": "test"'  # Missing closing braces

        with pytest.raises(GMNValidationError) as exc_info:
            parser.parse_json(malformed_json)

        assert "JSON" in str(exc_info.value) or "parse" in str(exc_info.value)

    def test_parser_handles_circular_dependencies(self):
        """Test parser detects and handles circular dependencies."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {"name": "node_a", "type": "state", "num_states": 2},
                {"name": "node_b", "type": "state", "num_states": 2},
            ],
            "edges": [
                {"from": "node_a", "to": "node_b", "type": "depends_on"},
                {
                    "from": "node_b",
                    "to": "node_a",
                    "type": "depends_on",
                },  # Circular
            ],
        }

        with pytest.raises(GMNValidationError) as exc_info:
            parser.parse(spec)

        assert "circular" in str(exc_info.value).lower()

    def test_parser_handles_negative_dimensions(self):
        """Test parser rejects negative dimensions."""
        parser = GMNParser()

        spec = {
            "nodes": [{"name": "invalid", "type": "state", "num_states": -1}],
            "edges": [],
        }

        with pytest.raises(GMNValidationError) as exc_info:
            parser.parse(spec)

        assert (
            "negative" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower()
        )


class TestGMNVersioning:
    """Test GMN specification versioning capabilities."""

    def test_specification_version_tracking(self):
        """Test that specifications track version information."""
        spec_dict = {
            "version": "1.0",
            "nodes": [{"name": "location", "type": "state", "num_states": 4}],
            "edges": [],
        }

        spec = GMNSpecification.from_dict(spec_dict)

        assert spec.version == "1.0"

    def test_specification_upgrade_version(self):
        """Test upgrading specification version."""
        spec_dict = {
            "version": "1.0",
            "nodes": [{"name": "location", "type": "state", "num_states": 4}],
            "edges": [],
        }

        spec = GMNSpecification.from_dict(spec_dict)
        upgraded_spec = spec.upgrade_version("1.1")

        assert upgraded_spec.version == "1.1"
        assert spec.version == "1.0"  # Original unchanged

    def test_specification_compatibility_check(self):
        """Test checking compatibility between specification versions."""
        v1_spec = GMNSpecification.from_dict(
            {
                "version": "1.0",
                "nodes": [{"name": "location", "type": "state", "num_states": 4}],
                "edges": [],
            }
        )

        v2_spec = GMNSpecification.from_dict(
            {
                "version": "2.0",
                "nodes": [{"name": "location", "type": "state", "num_states": 4}],
                "edges": [],
            }
        )

        # Same structure should be compatible
        assert v1_spec.is_compatible_with(v2_spec)

        v3_spec = GMNSpecification.from_dict(
            {
                "version": "2.0",
                "nodes": [{"name": "different", "type": "state", "num_states": 8}],
                "edges": [],
            }
        )

        # Different structure should not be compatible
        assert not v1_spec.is_compatible_with(v3_spec)


# Fixtures for common test data
@pytest.fixture
def minimal_gmn_spec():
    """Fixture providing minimal valid GMN specification."""
    return {
        "nodes": [
            {"name": "location", "type": "state", "num_states": 4},
            {
                "name": "obs_location",
                "type": "observation",
                "num_observations": 4,
            },
            {"name": "move", "type": "action", "num_actions": 4},
        ],
        "edges": [{"from": "location", "to": "obs_location", "type": "generates"}],
    }


@pytest.fixture
def complex_gmn_spec():
    """Fixture providing complex GMN specification."""
    return {
        "nodes": [
            {"name": "location", "type": "state", "num_states": 9},
            {
                "name": "obs_location",
                "type": "observation",
                "num_observations": 9,
            },
            {"name": "move", "type": "action", "num_actions": 5},
            {"name": "location_belief", "type": "belief", "about": "location"},
            {
                "name": "goal_preference",
                "type": "preference",
                "preferred_observation": 8,
                "preference_strength": 1.0,
            },
            {"name": "location_likelihood", "type": "likelihood"},
            {"name": "location_transition", "type": "transition"},
        ],
        "edges": [
            {
                "from": "location",
                "to": "location_likelihood",
                "type": "depends_on",
            },
            {
                "from": "location_likelihood",
                "to": "obs_location",
                "type": "generates",
            },
            {
                "from": "location",
                "to": "location_transition",
                "type": "depends_on",
            },
            {
                "from": "move",
                "to": "location_transition",
                "type": "depends_on",
            },
            {
                "from": "goal_preference",
                "to": "obs_location",
                "type": "depends_on",
            },
            {
                "from": "location_belief",
                "to": "location",
                "type": "depends_on",
            },
        ],
    }

"""Test suite for GMN Schema and Validation Models.

Following TDD principles - tests first, then implementation.
Focus on probability distribution validation and schema compliance.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import the models we're about to create
from inference.active.gmn_schema import (
    GMNEdge,
    GMNEdgeType,
    GMNNode,
    GMNNodeType,
    GMNSchemaValidator,
    GMNSpecification,
    GMNValidationError,
    ProbabilityDistribution,
)


class TestProbabilityDistribution:
    """Test probability distribution validation."""

    def test_valid_probability_distribution_creation(self):
        """Test creating valid probability distributions."""
        # Valid uniform distribution
        prob_dist = ProbabilityDistribution([0.25, 0.25, 0.25, 0.25])
        assert prob_dist.values == [0.25, 0.25, 0.25, 0.25]
        assert prob_dist.is_valid()

        # Valid non-uniform distribution
        prob_dist = ProbabilityDistribution([0.1, 0.2, 0.3, 0.4])
        assert prob_dist.is_valid()

    def test_probability_distribution_normalization(self):
        """Test automatic normalization of probability distributions."""
        # Should auto-normalize
        prob_dist = ProbabilityDistribution([1, 2, 3, 4], auto_normalize=True)
        expected = [0.1, 0.2, 0.3, 0.4]
        assert prob_dist.values == expected
        assert prob_dist.is_valid()

    def test_invalid_probability_distribution_raises_error(self):
        """Test that invalid probability distributions raise validation errors."""
        # Sum > 1
        with pytest.raises(GMNValidationError, match="Probabilities must sum to 1.0"):
            ProbabilityDistribution([0.5, 0.5, 0.5])

        # Negative values
        with pytest.raises(GMNValidationError, match="All probabilities must be non-negative"):
            ProbabilityDistribution([-0.1, 0.6, 0.5])

        # Empty distribution (now caught by Pydantic field validation)
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProbabilityDistribution([])

    def test_probability_distribution_numpy_integration(self):
        """Test integration with numpy arrays."""
        np_array = np.array([0.2, 0.3, 0.5])
        prob_dist = ProbabilityDistribution(np_array)
        assert prob_dist.is_valid()
        assert prob_dist.to_numpy().tolist() == [0.2, 0.3, 0.5]

    def test_probability_distribution_edge_cases(self):
        """Test edge cases for probability distributions."""
        # Single element (valid)
        prob_dist = ProbabilityDistribution([1.0])
        assert prob_dist.is_valid()

        # Very small values (numerical precision)
        prob_dist = ProbabilityDistribution([0.333333333, 0.333333333, 0.333333334])
        assert prob_dist.is_valid()

        # Zero values (valid)
        prob_dist = ProbabilityDistribution([0.0, 0.5, 0.5])
        assert prob_dist.is_valid()


class TestGMNNode:
    """Test GMN node creation and validation."""

    def test_create_state_node(self):
        """Test creating a state node."""
        node = GMNNode(
            id="location",
            type=GMNNodeType.STATE,
            properties={"num_states": 4, "description": "Agent location"},
        )
        assert node.id == "location"
        assert node.type == GMNNodeType.STATE
        assert node.properties["num_states"] == 4

    def test_create_observation_node(self):
        """Test creating an observation node."""
        node = GMNNode(
            id="obs_location", type=GMNNodeType.OBSERVATION, properties={"num_observations": 4}
        )
        assert node.id == "obs_location"
        assert node.type == GMNNodeType.OBSERVATION

    def test_create_action_node(self):
        """Test creating an action node."""
        node = GMNNode(
            id="move",
            type=GMNNodeType.ACTION,
            properties={"num_actions": 5, "actions": ["stay", "north", "south", "east", "west"]},
        )
        assert node.id == "move"
        assert node.type == GMNNodeType.ACTION
        assert len(node.properties["actions"]) == 5

    def test_node_with_probability_distribution(self):
        """Test node with embedded probability distribution."""
        initial_belief = ProbabilityDistribution([0.25, 0.25, 0.25, 0.25])
        node = GMNNode(
            id="initial_state", type=GMNNodeType.BELIEF, properties={"distribution": initial_belief}
        )
        assert node.properties["distribution"].is_valid()

    def test_invalid_node_id_raises_error(self):
        """Test that invalid node IDs raise validation errors."""
        with pytest.raises(GMNValidationError, match="Node ID cannot be empty"):
            GMNNode(id="", type=GMNNodeType.STATE)

        # Test invalid characters in ID
        with pytest.raises(GMNValidationError, match="Node ID contains invalid characters"):
            GMNNode(id="invalid-node!", type=GMNNodeType.STATE)


class TestGMNEdge:
    """Test GMN edge creation and validation."""

    def test_create_dependency_edge(self):
        """Test creating a dependency edge."""
        edge = GMNEdge(
            source="location",
            target="location_likelihood",
            type=GMNEdgeType.DEPENDS_ON,
            properties={"weight": 1.0},
        )
        assert edge.source == "location"
        assert edge.target == "location_likelihood"
        assert edge.type == GMNEdgeType.DEPENDS_ON

    def test_create_generation_edge(self):
        """Test creating a generation edge."""
        edge = GMNEdge(
            source="location_likelihood", target="obs_location", type=GMNEdgeType.GENERATES
        )
        assert edge.source == "location_likelihood"
        assert edge.target == "obs_location"
        assert edge.type == GMNEdgeType.GENERATES

    def test_invalid_edge_source_raises_error(self):
        """Test that invalid edge sources raise validation errors."""
        with pytest.raises(GMNValidationError, match="Edge source cannot be empty"):
            GMNEdge(source="", target="target", type=GMNEdgeType.DEPENDS_ON)

    def test_invalid_edge_target_raises_error(self):
        """Test that invalid edge targets raise validation errors."""
        with pytest.raises(GMNValidationError, match="Edge target cannot be empty"):
            GMNEdge(source="source", target="", type=GMNEdgeType.DEPENDS_ON)


class TestGMNSpecification:
    """Test complete GMN specification validation."""

    def test_create_minimal_valid_gmn_specification(self):
        """Test creating a minimal valid GMN specification."""
        nodes = [
            GMNNode(id="state1", type=GMNNodeType.STATE, properties={"num_states": 2}),
            GMNNode(id="obs1", type=GMNNodeType.OBSERVATION, properties={"num_observations": 2}),
            GMNNode(id="action1", type=GMNNodeType.ACTION, properties={"num_actions": 2}),
        ]

        edges = [GMNEdge(source="state1", target="obs1", type=GMNEdgeType.GENERATES)]

        spec = GMNSpecification(
            name="minimal_agent", description="Minimal test agent", nodes=nodes, edges=edges
        )

        assert spec.name == "minimal_agent"
        assert len(spec.nodes) == 3
        assert len(spec.edges) == 1

    def test_gmn_specification_validation_missing_required_nodes(self):
        """Test that missing required nodes raise validation errors."""
        # Missing state node
        nodes = [
            GMNNode(id="obs1", type=GMNNodeType.OBSERVATION, properties={"num_observations": 2}),
            GMNNode(id="action1", type=GMNNodeType.ACTION, properties={"num_actions": 2}),
        ]

        with pytest.raises(
            GMNValidationError, match="GMN specification must contain at least one STATE node"
        ):
            GMNSpecification(name="invalid", nodes=nodes, edges=[])

    def test_gmn_specification_validation_undefined_edge_references(self):
        """Test that undefined edge references raise validation errors."""
        nodes = [
            GMNNode(id="state1", type=GMNNodeType.STATE, properties={"num_states": 2}),
            GMNNode(id="obs1", type=GMNNodeType.OBSERVATION, properties={"num_observations": 2}),
            GMNNode(id="action1", type=GMNNodeType.ACTION, properties={"num_actions": 2}),
        ]

        edges = [GMNEdge(source="undefined_node", target="obs1", type=GMNEdgeType.GENERATES)]

        with pytest.raises(GMNValidationError, match="Edge references undefined node"):
            GMNSpecification(name="invalid", nodes=nodes, edges=edges)

    def test_gmn_specification_to_dict(self):
        """Test converting GMN specification to dictionary."""
        nodes = [
            GMNNode(id="state1", type=GMNNodeType.STATE, properties={"num_states": 2}),
            GMNNode(id="obs1", type=GMNNodeType.OBSERVATION, properties={"num_observations": 2}),
        ]

        spec = GMNSpecification(name="test", nodes=nodes, edges=[])
        spec_dict = spec.to_dict()

        assert spec_dict["name"] == "test"
        assert len(spec_dict["nodes"]) == 2
        assert spec_dict["nodes"][0]["id"] == "state1"
        assert spec_dict["nodes"][0]["type"] == "state"

    def test_gmn_specification_from_dict(self):
        """Test creating GMN specification from dictionary."""
        spec_dict = {
            "name": "test_agent",
            "description": "Test agent description",
            "nodes": [
                {"id": "state1", "type": "state", "properties": {"num_states": 2}},
                {"id": "obs1", "type": "observation", "properties": {"num_observations": 2}},
                {"id": "action1", "type": "action", "properties": {"num_actions": 2}},
            ],
            "edges": [{"source": "state1", "target": "obs1", "type": "generates"}],
        }

        spec = GMNSpecification.from_dict(spec_dict)
        assert spec.name == "test_agent"
        assert len(spec.nodes) == 3
        assert len(spec.edges) == 1


class TestGMNSchemaValidator:
    """Test GMN schema validation functionality."""

    def test_validate_probability_matrices(self):
        """Test validation of probability matrices in GMN."""
        # Valid A matrix (likelihood) - columns should sum to 1
        A_matrix = np.array(
            [
                [0.8, 0.3],  # obs=0 given states [state0, state1]
                [0.2, 0.7],  # obs=1 given states [state0, state1]
            ]
        )

        validator = GMNSchemaValidator()
        is_valid, errors = validator.validate_probability_matrix(A_matrix, "A")
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_probability_matrices(self):
        """Test validation of invalid probability matrices."""
        # Invalid A matrix (doesn't sum to 1)
        A_matrix = np.array(
            [
                [[0.5, 0.2], [0.3, 0.7]],  # First column doesn't sum to 1
                [[0.2, 0.8], [0.7, 0.3]],
            ]
        )

        validator = GMNSchemaValidator()
        is_valid, errors = validator.validate_probability_matrix(A_matrix, "A")
        assert not is_valid
        assert len(errors) > 0
        assert "sum to 1.0" in errors[0]

    def test_validate_complete_gmn_specification(self):
        """Test validation of complete GMN specification."""
        nodes = [
            GMNNode(id="location", type=GMNNodeType.STATE, properties={"num_states": 4}),
            GMNNode(
                id="obs_location", type=GMNNodeType.OBSERVATION, properties={"num_observations": 4}
            ),
            GMNNode(id="move", type=GMNNodeType.ACTION, properties={"num_actions": 5}),
        ]

        edges = [GMNEdge(source="location", target="obs_location", type=GMNEdgeType.GENERATES)]

        spec = GMNSpecification(name="explorer", nodes=nodes, edges=edges)

        validator = GMNSchemaValidator()
        is_valid, errors = validator.validate_specification(spec)
        assert is_valid
        assert len(errors) == 0

    def test_validate_gmn_mathematical_consistency(self):
        """Test mathematical consistency validation."""
        # Create GMN with inconsistent dimensions
        nodes = [
            GMNNode(id="location", type=GMNNodeType.STATE, properties={"num_states": 4}),
            GMNNode(
                id="obs_location", type=GMNNodeType.OBSERVATION, properties={"num_observations": 3}
            ),  # Mismatch
            GMNNode(
                id="likelihood",
                type=GMNNodeType.LIKELIHOOD,
                properties={
                    "matrix": [[0.8, 0.2], [0.2, 0.8]]  # Wrong dimensions
                },
            ),
        ]

        edges = [
            GMNEdge(source="location", target="likelihood", type=GMNEdgeType.DEPENDS_ON),
            GMNEdge(source="likelihood", target="obs_location", type=GMNEdgeType.GENERATES),
        ]

        spec = GMNSpecification(name="inconsistent", nodes=nodes, edges=edges)

        validator = GMNSchemaValidator()
        is_valid, errors = validator.validate_specification(spec)
        assert not is_valid
        assert any("dimension mismatch" in error.lower() for error in errors)


class TestGMNIntegrationWithExistingParser:
    """Test integration with existing GMN parser functionality."""

    def test_schema_validation_with_existing_parser_output(self):
        """Test that schema validation works with existing parser output."""
        # Mock the existing parser
        from inference.active.gmn_parser import GMNParser

        parser = GMNParser()

        # Create a simple GMN specification
        gmn_text = """
        [nodes]
        location: state {num_states: 4}
        obs_location: observation {num_observations: 4}
        move: action {num_actions: 5}

        [edges]
        location -> obs_location: generates
        """

        # Parse with existing parser
        parsed_graph = parser.parse(gmn_text)

        # Convert to our new schema format (this is what we need to implement)
        validator = GMNSchemaValidator()

        # Validate compatibility
        is_compatible = validator.validate_parser_compatibility(parsed_graph)
        assert is_compatible

    @patch("inference.active.gmn_parser.GMNParser")
    def test_backward_compatibility_with_existing_code(self, mock_parser):
        """Test that new schema models are backward compatible."""
        # Mock existing parser behavior
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance

        # Test that we can still use existing parser alongside new schema
        validator = GMNSchemaValidator()

        # Should not break existing functionality
        assert hasattr(validator, "validate_specification")
        assert hasattr(validator, "validate_probability_matrix")


class TestPerformanceAndErrorBoundaries:
    """Test performance and error boundary conditions."""

    def test_large_gmn_specification_validation_performance(self):
        """Test performance with large GMN specifications."""
        import time

        # Create large GMN specification
        nodes = []
        edges = []

        # Add required nodes
        nodes.append(GMNNode(id="state_0", type=GMNNodeType.STATE, properties={"num_states": 10}))
        nodes.append(
            GMNNode(id="obs_0", type=GMNNodeType.OBSERVATION, properties={"num_observations": 10})
        )

        for i in range(1, 100):  # 98 more state nodes
            nodes.append(
                GMNNode(id=f"node_{i}", type=GMNNodeType.STATE, properties={"num_states": 10})
            )

        for i in range(99):  # 99 edges
            edges.append(
                GMNEdge(
                    source=f"node_{i}" if i > 0 else "state_0",
                    target=f"node_{i+1}" if i < 98 else "obs_0",
                    type=GMNEdgeType.DEPENDS_ON,
                )
            )

        spec = GMNSpecification(name="large_test", nodes=nodes, edges=edges)

        validator = GMNSchemaValidator()

        start_time = time.time()
        is_valid, errors = validator.validate_specification(spec)
        end_time = time.time()

        # Validation should complete within performance budget (50ms from requirements)
        assert (end_time - start_time) < 0.05  # 50ms

    def test_validation_with_malformed_data(self):
        """Test validation with malformed or edge case data."""
        validator = GMNSchemaValidator()

        # Test with None input
        is_valid, errors = validator.validate_specification(None)
        assert not is_valid
        assert "Specification cannot be None" in errors[0]

        # Test with circular references
        nodes = [
            GMNNode(id="node1", type=GMNNodeType.STATE, properties={"num_states": 2}),
            GMNNode(id="node2", type=GMNNodeType.STATE, properties={"num_states": 2}),
            GMNNode(id="obs1", type=GMNNodeType.OBSERVATION, properties={"num_observations": 2}),
        ]

        edges = [
            GMNEdge(source="node1", target="node2", type=GMNEdgeType.DEPENDS_ON),
            GMNEdge(source="node2", target="node1", type=GMNEdgeType.DEPENDS_ON),  # Circular
        ]

        spec = GMNSpecification(name="circular", nodes=nodes, edges=edges)
        is_valid, errors = validator.validate_specification(spec)
        # Should detect circular dependencies
        assert any("circular" in error.lower() for error in errors)


if __name__ == "__main__":
    pytest.main([__file__])

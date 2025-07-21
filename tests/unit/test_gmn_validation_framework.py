"""Unit tests for GMN validation framework with comprehensive validation rules.

This test suite follows TDD principles and implements comprehensive validation
for GMN specifications including syntax, semantic, mathematical, type, and
constraint validation. All tests are designed to fail initially and drive
the implementation of robust validators with hard failures on any violation.
"""

import pytest

# Import mock for missing classes
from tests.unit.gmn_mocks import GMNValidationError


# Mock the validator classes until they are implemented
class GMNSyntaxValidator:
    """Mock syntax validator."""

    def validate(self, spec):
        if not spec:
            raise GMNValidationError("Empty specification")
        return True


class GMNSemanticValidator:
    """Mock semantic validator."""

    def validate(self, spec):
        return True


class GMNMathematicalValidator:
    """Mock mathematical validator."""

    def validate(self, spec):
        return True


class GMNTypeValidator:
    """Mock type validator."""

    def validate(self, spec):
        return True


class GMNConstraintValidator:
    """Mock constraint validator."""

    def validate(self, spec):
        return True


class GMNComprehensiveValidator:
    """Mock comprehensive validator."""

    def validate(self, spec):
        return True


class TestGMNSyntaxValidation:
    """Test suite for GMN syntax validation with hard failures."""

    def test_empty_specification_fails_validation(self):
        """Test that empty GMN specification fails validation with hard failure."""
        validator = GMNSyntaxValidator()

        with pytest.raises(GMNValidationError, match="Empty specification"):
            validator.validate({})

    def test_non_dict_specification_fails_validation(self):
        """Test that non-dictionary specification fails validation."""
        validator = GMNSyntaxValidator()

        with pytest.raises(
            GMNValidationError, match="Specification must be a dictionary"
        ):
            validator.validate("invalid")

        with pytest.raises(
            GMNValidationError, match="Specification must be a dictionary"
        ):
            validator.validate([])

        with pytest.raises(
            GMNValidationError, match="Specification must be a dictionary"
        ):
            validator.validate(None)

    def test_missing_required_top_level_fields_fails_validation(self):
        """Test that missing required fields fail validation."""
        validator = GMNSyntaxValidator()

        # Missing nodes field
        with pytest.raises(GMNValidationError, match="Missing required field: nodes"):
            validator.validate({"edges": []})

        # Invalid nodes type
        with pytest.raises(GMNValidationError, match="Field 'nodes' must be a list"):
            validator.validate({"nodes": "invalid"})

    def test_invalid_node_structure_fails_validation(self):
        """Test that invalid node structures fail validation."""
        validator = GMNSyntaxValidator()

        # Non-dict node
        spec = {"nodes": ["invalid_node"]}
        with pytest.raises(GMNValidationError, match="Node must be a dictionary"):
            validator.validate(spec)

        # Missing required fields
        spec = {"nodes": [{}]}
        with pytest.raises(GMNValidationError, match="Missing required field: name"):
            validator.validate(spec)

        spec = {"nodes": [{"name": "test"}]}
        with pytest.raises(GMNValidationError, match="Missing required field: type"):
            validator.validate(spec)

    def test_invalid_node_names_fail_validation(self):
        """Test that invalid node names fail validation."""
        validator = GMNSyntaxValidator()

        # Empty name
        spec = {"nodes": [{"name": "", "type": "state"}]}
        with pytest.raises(GMNValidationError, match="Node name cannot be empty"):
            validator.validate(spec)

        # Invalid characters in name
        spec = {"nodes": [{"name": "node-with-dashes", "type": "state"}]}
        with pytest.raises(
            GMNValidationError, match="Node name contains invalid characters"
        ):
            validator.validate(spec)

        # Duplicate names
        spec = {
            "nodes": [
                {"name": "duplicate", "type": "state"},
                {"name": "duplicate", "type": "observation"},
            ]
        }
        with pytest.raises(GMNValidationError, match="Duplicate node name: duplicate"):
            validator.validate(spec)

    def test_invalid_edge_structure_fails_validation(self):
        """Test that invalid edge structures fail validation."""
        validator = GMNSyntaxValidator()

        # Non-dict edge
        spec = {
            "nodes": [{"name": "node1", "type": "state"}],
            "edges": ["invalid_edge"],
        }
        with pytest.raises(GMNValidationError, match="Edge must be a dictionary"):
            validator.validate(spec)

        # Missing required edge fields
        spec = {"nodes": [{"name": "node1", "type": "state"}], "edges": [{}]}
        with pytest.raises(GMNValidationError, match="Missing required field: from"):
            validator.validate(spec)

    def test_malformed_gmn_text_format_fails_validation(self):
        """Test that malformed GMN text format fails validation."""
        validator = GMNSyntaxValidator()

        # Invalid section headers
        malformed_text = """
        [invalid section]
        node1: state
        """
        with pytest.raises(GMNValidationError, match="Invalid section header"):
            validator.validate_text(malformed_text)

        # Invalid node syntax
        malformed_text = """
        [nodes]
        invalid node syntax
        """
        with pytest.raises(GMNValidationError, match="Invalid node syntax"):
            validator.validate_text(malformed_text)


class TestGMNSemanticValidation:
    """Test suite for GMN semantic validation with logical consistency checks."""

    def test_unreferenced_nodes_fail_validation(self):
        """Test that unreferenced nodes fail semantic validation."""
        validator = GMNSemanticValidator()

        spec = {
            "nodes": [
                {"name": "orphan_node", "type": "state"},
                {"name": "connected_node", "type": "observation"},
            ],
            "edges": [],
        }
        with pytest.raises(GMNValidationError, match="Unreferenced node: orphan_node"):
            validator.validate(spec)

    def test_circular_dependencies_fail_validation(self):
        """Test that circular dependencies fail validation."""
        validator = GMNSemanticValidator()

        spec = {
            "nodes": [
                {"name": "node1", "type": "state"},
                {"name": "node2", "type": "belief"},
                {"name": "node3", "type": "transition"},
            ],
            "edges": [
                {"from": "node1", "to": "node2", "type": "depends_on"},
                {"from": "node2", "to": "node3", "type": "depends_on"},
                {"from": "node3", "to": "node1", "type": "depends_on"},
            ],
        }
        with pytest.raises(GMNValidationError, match="Circular dependency detected"):
            validator.validate(spec)

    def test_invalid_edge_relationships_fail_validation(self):
        """Test that invalid edge relationships fail validation."""
        validator = GMNSemanticValidator()

        # State cannot depend on observation
        spec = {
            "nodes": [
                {"name": "state1", "type": "state"},
                {"name": "obs1", "type": "observation"},
            ],
            "edges": [{"from": "state1", "to": "obs1", "type": "depends_on"}],
        }
        with pytest.raises(
            GMNValidationError,
            match="Invalid dependency: state cannot depend on observation",
        ):
            validator.validate(spec)

    def test_missing_required_connections_fail_validation(self):
        """Test that missing required connections fail validation."""
        validator = GMNSemanticValidator()

        # Belief node must be connected to a state
        spec = {
            "nodes": [
                {"name": "belief1", "type": "belief"},
                {"name": "state1", "type": "state"},
            ],
            "edges": [],
        }
        with pytest.raises(
            GMNValidationError,
            match="Belief node 'belief1' must be connected to a state",
        ):
            validator.validate(spec)

    def test_invalid_node_references_in_edges_fail_validation(self):
        """Test that edges referencing non-existent nodes fail validation."""
        validator = GMNSemanticValidator()

        spec = {
            "nodes": [{"name": "existing_node", "type": "state"}],
            "edges": [
                {
                    "from": "existing_node",
                    "to": "non_existent_node",
                    "type": "generates",
                }
            ],
        }
        with pytest.raises(
            GMNValidationError,
            match="Edge references non-existent node: non_existent_node",
        ):
            validator.validate(spec)


class TestGMNMathematicalValidation:
    """Test suite for GMN mathematical validation with probability constraints."""

    def test_invalid_probability_distributions_fail_validation(self):
        """Test that invalid probability distributions fail validation."""
        validator = GMNMathematicalValidator()

        # Probabilities don't sum to 1
        spec = {
            "nodes": [
                {
                    "name": "belief1",
                    "type": "belief",
                    "initial_distribution": [0.3, 0.3, 0.3],  # Sum = 0.9
                }
            ]
        }
        with pytest.raises(
            GMNValidationError,
            match="Probability distribution does not sum to 1",
        ):
            validator.validate(spec)

        # Negative probabilities
        spec = {
            "nodes": [
                {
                    "name": "belief1",
                    "type": "belief",
                    "initial_distribution": [0.5, -0.2, 0.7],
                }
            ]
        }
        with pytest.raises(
            GMNValidationError,
            match="Probability distribution contains negative values",
        ):
            validator.validate(spec)

    def test_dimension_mismatches_fail_validation(self):
        """Test that dimension mismatches fail validation."""
        validator = GMNMathematicalValidator()

        spec = {
            "nodes": [
                {"name": "state1", "type": "state", "num_states": 4},
                {"name": "obs1", "type": "observation", "num_observations": 3},
            ],
            "edges": [{"from": "state1", "to": "obs1", "type": "generates"}],
        }
        with pytest.raises(
            GMNValidationError,
            match="Dimension mismatch: state has 4 dimensions but observation has 3",
        ):
            validator.validate(spec)

    def test_invalid_numerical_ranges_fail_validation(self):
        """Test that invalid numerical ranges fail validation."""
        validator = GMNMathematicalValidator()

        # Zero or negative dimensions
        spec = {"nodes": [{"name": "state1", "type": "state", "num_states": 0}]}
        with pytest.raises(
            GMNValidationError, match="Number of states must be positive"
        ):
            validator.validate(spec)

        # Invalid precision values
        spec = {
            "nodes": [
                {
                    "name": "belief1",
                    "type": "belief",
                    "constraints": {"precision": -1.0},
                }
            ]
        }
        with pytest.raises(GMNValidationError, match="Precision must be positive"):
            validator.validate(spec)

    def test_matrix_constraint_violations_fail_validation(self):
        """Test that matrix constraint violations fail validation."""
        validator = GMNMathematicalValidator()

        # Transition matrix with invalid probabilities
        spec = {
            "nodes": [
                {
                    "name": "transition1",
                    "type": "transition",
                    "matrix": [
                        [0.6, 0.3],
                        [0.4, 0.8],
                    ],  # Columns don't sum to 1
                }
            ]
        }
        with pytest.raises(
            GMNValidationError, match="Transition matrix columns must sum to 1"
        ):
            validator.validate(spec)


class TestGMNTypeValidation:
    """Test suite for GMN type validation with comprehensive type checking."""

    def test_invalid_node_types_fail_validation(self):
        """Test that invalid node types fail validation."""
        validator = GMNTypeValidator()

        spec = {"nodes": [{"name": "invalid", "type": "invalid_type"}]}
        with pytest.raises(GMNValidationError, match="Invalid node type: invalid_type"):
            validator.validate(spec)

    def test_invalid_edge_types_fail_validation(self):
        """Test that invalid edge types fail validation."""
        validator = GMNTypeValidator()

        spec = {
            "nodes": [
                {"name": "node1", "type": "state"},
                {"name": "node2", "type": "observation"},
            ],
            "edges": [{"from": "node1", "to": "node2", "type": "invalid_edge_type"}],
        }
        with pytest.raises(
            GMNValidationError, match="Invalid edge type: invalid_edge_type"
        ):
            validator.validate(spec)

    def test_missing_required_attributes_for_node_types_fail_validation(self):
        """Test that missing required attributes for specific node types fail validation."""
        validator = GMNTypeValidator()

        # State node missing num_states
        spec = {"nodes": [{"name": "state1", "type": "state"}]}  # Missing num_states
        with pytest.raises(
            GMNValidationError,
            match="State node 'state1' missing required attribute: num_states",
        ):
            validator.validate(spec)

        # Belief node missing about
        spec = {"nodes": [{"name": "belief1", "type": "belief"}]}  # Missing about
        with pytest.raises(
            GMNValidationError,
            match="Belief node 'belief1' missing required attribute: about",
        ):
            validator.validate(spec)

    def test_incorrect_attribute_types_fail_validation(self):
        """Test that incorrect attribute types fail validation."""
        validator = GMNTypeValidator()

        # num_states should be integer
        spec = {"nodes": [{"name": "state1", "type": "state", "num_states": "invalid"}]}
        with pytest.raises(GMNValidationError, match="num_states must be an integer"):
            validator.validate(spec)

        # initial_distribution should be list
        spec = {
            "nodes": [
                {
                    "name": "belief1",
                    "type": "belief",
                    "initial_distribution": "invalid",
                }
            ]
        }
        with pytest.raises(
            GMNValidationError, match="initial_distribution must be a list"
        ):
            validator.validate(spec)


class TestGMNConstraintValidation:
    """Test suite for GMN constraint validation with business rule enforcement."""

    def test_invalid_preference_constraints_fail_validation(self):
        """Test that invalid preference constraints fail validation."""
        validator = GMNConstraintValidator()

        # Preferred observation out of range
        spec = {
            "nodes": [
                {"name": "obs1", "type": "observation", "num_observations": 3},
                {
                    "name": "pref1",
                    "type": "preference",
                    "preferred_observation": 5,
                },
            ]
        }
        with pytest.raises(
            GMNValidationError,
            match="Preferred observation index 5 out of range",
        ):
            validator.validate(spec)

    def test_constraint_consistency_violations_fail_validation(self):
        """Test that constraint consistency violations fail validation."""
        validator = GMNConstraintValidator()

        # Conflicting constraints
        spec = {
            "nodes": [
                {
                    "name": "belief1",
                    "type": "belief",
                    "constraints": {
                        "min_entropy": 2.0,
                        "max_entropy": 1.0,
                    },  # min > max
                }
            ]
        }
        with pytest.raises(GMNValidationError, match="Conflicting entropy constraints"):
            validator.validate(spec)

    def test_invalid_business_rules_fail_validation(self):
        """Test that violations of business rules fail validation."""
        validator = GMNConstraintValidator()

        # Action space too large
        spec = {
            "nodes": [{"name": "action1", "type": "action", "num_actions": 1000000}]
        }
        with pytest.raises(GMNValidationError, match="Action space too large"):
            validator.validate(spec)


class TestGMNValidationFrameworkIntegration:
    """Test suite for integrated validation framework functionality."""

    def test_comprehensive_validation_pipeline_fails_on_any_violation(self):
        """Test that comprehensive validation fails if any validator fails."""
        framework = GMNValidationFramework()

        # Specification with multiple violations
        spec = {
            "nodes": [
                # Syntax violation: missing type
                {"name": "invalid1"},
                # Type violation: invalid type
                {"name": "invalid2", "type": "invalid_type"},
                # Mathematical violation: negative dimension
                {"name": "invalid3", "type": "state", "num_states": -1},
            ]
        }

        with pytest.raises(GMNValidationError) as exc_info:
            framework.validate(spec)

        # Should contain all validation errors
        error_message = str(exc_info.value)
        assert "Missing required field: type" in error_message
        assert "Invalid node type: invalid_type" in error_message
        assert "Number of states must be positive" in error_message

    def test_validation_with_reality_checkpoints(self):
        """Test validation with reality checkpoints to catch real issues."""
        framework = GMNValidationFramework()

        # Test against known problematic patterns
        problematic_spec = {
            "nodes": [
                {
                    "name": "state1",
                    "type": "state",
                    "num_states": 1,
                },  # Trivial state space
                {
                    "name": "obs1",
                    "type": "observation",
                    "num_observations": 100,
                },  # Huge observation space
            ],
            "edges": [{"from": "state1", "to": "obs1", "type": "generates"}],
        }

        with pytest.raises(GMNValidationError, match="Suspicious dimension ratio"):
            framework.validate_with_reality_checks(problematic_spec)

    def test_valid_specifications_pass_all_validation(self):
        """Test that valid specifications pass all validation layers."""
        framework = GMNValidationFramework()

        valid_spec = {
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

        # Should not raise any exceptions
        result = framework.validate(valid_spec)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validation_error_messages_are_comprehensive(self):
        """Test that validation error messages provide comprehensive information."""
        framework = GMNValidationFramework()

        invalid_spec = {
            "nodes": [{"name": "test", "type": "invalid_type", "num_states": -1}]
        }

        with pytest.raises(GMNValidationError) as exc_info:
            framework.validate(invalid_spec)

        error_message = str(exc_info.value)
        # Should include validator type, specific error, and helpful context
        assert "Type Validation Error" in error_message
        assert "Invalid node type: invalid_type" in error_message
        assert "Node: test" in error_message


# Import the actual validator implementations
from inference.active.gmn_validation import (
    GMNConstraintValidator,
    GMNMathematicalValidator,
    GMNSemanticValidator,
    GMNSyntaxValidator,
    GMNTypeValidator,
    GMNValidationFramework,
)

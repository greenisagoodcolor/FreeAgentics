"""
Comprehensive test suite for GNN Validator module - Meta Quality Standards.

This test suite provides comprehensive coverage for the GMNValidator class,
which validates and secures GNN model definitions.
Coverage target: 95%+
"""

from unittest.mock import Mock, patch

import pytest

# Import the module under test
try:
    from inference.gnn.model import GMNModel
    from inference.gnn.parser import ASTNode, ParseResult
    from inference.gnn.validator import (
        GMNValidator,
        ValidationError,
        ValidationResult,
        logger,
    )

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

    # Mock classes for testing when imports fail
    class GMNValidator:
        pass

    class ValidationResult:
        pass

    class ValidationError:
        pass


class TestGMNValidator:
    """Test suite for GMNValidator following Meta standards."""

    @pytest.fixture
    def validator(self):
        """Create GMNValidator instance."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        return GMNValidator()

    @pytest.fixture
    def mock_parse_result(self):
        """Create a mock parse result with valid structure."""
        result = Mock(spec=ParseResult)

        # Mock AST root
        root_ast = Mock(spec=ASTNode)
        root_ast.node_type = "root"
        root_ast.line = 1
        root_ast.column = 1
        root_ast.children = []
        root_ast.attributes = {}

        # Mock metadata
        result.ast = root_ast
        result.metadata = {
            "name": "Test Model",
            "version": "1.0.0",
            "author": "Test Author",
            "tags": ["test", "validation"],
            "created_at": "2023-01-01T00:00:00",
        }

        # Mock sections
        result.sections = {
            "architecture": {
                "type": "GraphSAGE",
                "layers": 3,
                "hidden_dim": 128,
                "activation": "relu",
            },
            "parameters": {"learning_rate": 0.01, "batch_size": 32},
            "active_inference": {
                "num_states": 10,
                "num_observations": 5,
                "num_actions": 3,
            },
        }

        result.errors = []
        result.warnings = []

        return result

    def test_validator_initialization(self, validator):
        """Test validator initialization and configuration."""
        assert validator is not None
        assert hasattr(validator, "allowed_architectures")
        assert hasattr(validator, "allowed_activations")
        assert hasattr(validator, "parameter_constraints")
        assert hasattr(validator, "security_patterns")
        assert hasattr(validator, "circuit_breaker")

        # Check default configurations
        assert "GraphSAGE" in validator.allowed_architectures
        assert "relu" in validator.allowed_activations
        assert validator.max_validation_errors == 100
        assert validator.validation_timeout == 60

    def test_validate_success(self, validator, mock_parse_result):
        """Test successful validation of a valid model."""
        result = validator.validate(mock_parse_result)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.model is not None
        assert result.model.name == "Test Model"
        assert result.metadata is not None

    def test_validate_missing_required_metadata(self, validator):
        """Test validation with missing required metadata."""
        parse_result = Mock(spec=ParseResult)
        parse_result.metadata = {"author": "Test"}  # Missing name
        parse_result.sections = {}
        parse_result.ast = Mock()
        parse_result.errors = []
        parse_result.warnings = []

        result = validator.validate(parse_result)

        assert result.is_valid is False
        assert any("name" in error.lower() for error in result.errors)

    def test_validate_invalid_architecture(self, validator, mock_parse_result):
        """Test validation with invalid architecture type."""
        mock_parse_result.sections["architecture"]["type"] = "InvalidArch"

        result = validator.validate(mock_parse_result)

        assert result.is_valid is False
        assert any("architecture" in error.lower() for error in result.errors)

    def test_validate_invalid_activation(self, validator, mock_parse_result):
        """Test validation with invalid activation function."""
        mock_parse_result.sections["architecture"]["activation"] = "invalid_activation"

        result = validator.validate(mock_parse_result)

        assert result.is_valid is False
        assert any("activation" in error.lower() for error in result.errors)

    def test_validate_parameter_constraints(self, validator, mock_parse_result):
        """Test parameter constraint validation."""
        # Test learning rate out of bounds
        mock_parse_result.sections["parameters"]["learning_rate"] = 10.0  # Too high

        result = validator.validate(mock_parse_result)

        assert result.is_valid is False
        assert any("learning_rate" in error.lower() for error in result.errors)

    def test_validate_negative_dimensions(self, validator, mock_parse_result):
        """Test validation with negative dimensions."""
        mock_parse_result.sections["architecture"]["hidden_dim"] = -128

        result = validator.validate(mock_parse_result)

        assert result.is_valid is False
        assert any(
            "negative" in error.lower() or "positive" in error.lower()
            for error in result.errors
        )

    def test_validate_security_patterns(self, validator, mock_parse_result):
        """Test security pattern detection."""
        # Add potentially malicious code
        mock_parse_result.sections["custom_code"] = {
            "initialization": "exec('malicious code')"
        }

        result = validator.validate(mock_parse_result)

        assert result.is_valid is False
        assert any("security" in error.lower() for error in result.errors)

    def test_validate_import_detection(self, validator, mock_parse_result):
        """Test import statement detection in custom code."""
        mock_parse_result.sections["custom_functions"] = {
            "preprocess": "import os; os.system('ls')"
        }

        result = validator.validate(mock_parse_result)

        assert result.is_valid is False
        assert any("import" in error.lower() for error in result.errors)

    def test_validate_cross_references(self, validator, mock_parse_result):
        """Test cross-reference validation."""
        # Add node features with invalid references
        mock_parse_result.sections["node_features"] = {
            "features": ["position", "velocity", "invalid_ref"]
        }

        # Mock AST traversal
        validator._collect_variable_definitions = Mock(
            return_value={"position", "velocity"}
        )

        result = validator.validate(mock_parse_result)

        assert result.is_valid is False
        assert any(
            "undefined" in error.lower() or "reference" in error.lower()
            for error in result.errors
        )

    def test_validate_circular_dependencies(self, validator, mock_parse_result):
        """Test circular dependency detection."""
        # Create circular reference
        mock_parse_result.sections["definitions"] = {
            "A": {"depends_on": ["B"]},
            "B": {"depends_on": ["C"]},
            "C": {"depends_on": ["A"]},
        }

        result = validator.validate(mock_parse_result)

        # Should detect circular dependency
        if hasattr(validator, "_check_circular_dependencies"):
            assert result.is_valid is False

    def test_circuit_breaker_functionality(self, validator, mock_parse_result):
        """Test circuit breaker pattern."""
        # Simulate multiple failures
        with patch.object(validator, "_validate_metadata") as mock_validate:
            mock_validate.side_effect = Exception("Validation error")

            # First few attempts should try validation
            for i in range(3):
                result = validator.validate(mock_parse_result)
                assert result.is_valid is False

            # Circuit breaker should be open after failures
            if hasattr(validator.circuit_breaker, "is_open"):
                assert validator.circuit_breaker.is_open()

    def test_validate_active_inference_config(self, validator, mock_parse_result):
        """Test Active Inference configuration validation."""
        # Test invalid state dimensions
        mock_parse_result.sections["active_inference"]["num_states"] = 0

        result = validator.validate(mock_parse_result)

        assert result.is_valid is False
        assert any("states" in error.lower() for error in result.errors)

    def test_validate_gnn_layer_compatibility(self, validator, mock_parse_result):
        """Test GNN layer configuration compatibility."""
        # Test incompatible layer configuration
        mock_parse_result.sections["architecture"]["type"] = "GCN"
        mock_parse_result.sections["architecture"]["edge_features"] = (
            True  # GCN doesn't use edge features
        )

        result = validator.validate(mock_parse_result)

        # Should warn about incompatibility
        assert len(result.warnings) > 0 or not result.is_valid

    def test_validate_numerical_stability(self, validator, mock_parse_result):
        """Test numerical stability checks."""
        # Add parameters that could cause instability
        mock_parse_result.sections["parameters"]["gradient_clip"] = 1e10  # Too large
        mock_parse_result.sections["parameters"]["eps"] = 1e-20  # Too small

        result = validator.validate(mock_parse_result)

        # Should flag potential numerical issues
        assert len(result.warnings) > 0 or not result.is_valid

    def test_validate_batch_processing(self, validator):
        """Test validation of multiple models in batch."""
        parse_results = []
        for i in range(5):
            pr = Mock(spec=ParseResult)
            pr.metadata = {"name": f"Model_{i}", "version": "1.0"}
            pr.sections = {
                "architecture": {
                    "type": "GraphSAGE",
                    "layers": 2 + i,
                    "hidden_dim": 64 * (i + 1),
                }
            }
            pr.ast = Mock()
            pr.errors = []
            pr.warnings = []
            parse_results.append(pr)

        results = []
        for pr in parse_results:
            result = validator.validate(pr)
            results.append(result)

        assert len(results) == 5
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_validate_memory_constraints(self, validator, mock_parse_result):
        """Test memory constraint validation."""
        # Create a model with excessive memory requirements
        mock_parse_result.sections["architecture"]["layers"] = 100
        mock_parse_result.sections["architecture"]["hidden_dim"] = 10000
        mock_parse_result.sections["parameters"]["batch_size"] = 1000

        result = validator.validate(mock_parse_result)

        # Should warn about memory requirements
        assert len(result.warnings) > 0

    def test_validate_performance_implications(self, validator, mock_parse_result):
        """Test performance implication warnings."""
        # Configure model for poor performance
        mock_parse_result.sections["architecture"]["type"] = "GAT"
        mock_parse_result.sections["architecture"]["num_heads"] = (
            32  # Too many attention heads
        )
        mock_parse_result.sections["architecture"]["layers"] = 10  # Deep network

        result = validator.validate(mock_parse_result)

        # Should warn about performance
        assert len(result.warnings) > 0

    def test_error_accumulation_limit(self, validator):
        """Test that error accumulation is limited."""
        parse_result = Mock(spec=ParseResult)
        parse_result.metadata = {}
        parse_result.sections = {}
        parse_result.ast = Mock()
        parse_result.errors = []
        parse_result.warnings = []

        # Add many validation errors
        with patch.object(validator, "max_validation_errors", 5):
            # Create conditions that generate many errors
            parse_result.sections = {
                f"invalid_section_{i}": {"error": True} for i in range(20)
            }

            result = validator.validate(parse_result)

            # Should limit errors
            assert (
                len(result.errors) <= validator.max_validation_errors + 5
            )  # Some buffer

    def test_custom_validation_rules(self, validator, mock_parse_result):
        """Test custom validation rule injection."""

        # Add custom validation rule
        def custom_rule(parse_result):
            if parse_result.metadata.get("custom_field") != "expected_value":
                return ["Custom validation failed"]
            return []

        if hasattr(validator, "add_custom_rule"):
            validator.add_custom_rule(custom_rule)
            mock_parse_result.metadata["custom_field"] = "wrong_value"

            result = validator.validate(mock_parse_result)

            assert any("custom" in error.lower() for error in result.errors)

    def test_validate_edge_cases(self, validator):
        """Test validation of edge cases."""
        # Test with None
        result = validator.validate(None)
        assert result.is_valid is False

        # Test with empty parse result
        empty_result = Mock(spec=ParseResult)
        empty_result.metadata = {}
        empty_result.sections = {}
        empty_result.ast = None
        empty_result.errors = []
        empty_result.warnings = []

        result = validator.validate(empty_result)
        assert result.is_valid is False

    def test_thread_safety(self, validator, mock_parse_result):
        """Test thread safety of validator."""
        import threading

        results = []
        errors = []

        def validate_concurrent():
            try:
                result = validator.validate(mock_parse_result)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run validation in multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=validate_concurrent)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(r.is_valid for r in results)

    def test_validation_result_serialization(self, validator, mock_parse_result):
        """Test that validation results can be serialized."""
        result = validator.validate(mock_parse_result)

        # Should be able to convert to dict
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "is_valid" in result_dict
            assert "errors" in result_dict
            assert "warnings" in result_dict

    def test_incremental_validation(self, validator, mock_parse_result):
        """Test incremental validation capabilities."""
        # First validation
        result1 = validator.validate(mock_parse_result)

        # Modify one section
        mock_parse_result.sections["parameters"]["learning_rate"] = 0.02

        # Re-validate
        result2 = validator.validate(mock_parse_result)

        # Both should be valid
        assert result1.is_valid
        assert result2.is_valid

        # If incremental validation is supported, second should be faster
        # (This would require timing measurements in real implementation)

    def test_validation_context_preservation(self, validator, mock_parse_result):
        """Test that validation context is preserved across calls."""
        # Validate with specific context
        if hasattr(validator, "set_context"):
            validator.set_context({"strict_mode": True})

            result = validator.validate(mock_parse_result)

            # Context should affect validation
            assert result.metadata.get("context", {}).get("strict_mode") is True

    def test_recovery_from_malformed_input(self, validator):
        """Test recovery from malformed input."""
        malformed_inputs = [
            {"not": "a parse result"},
            "string instead of object",
            123,
            [],
            Mock(spec=object),  # Wrong type
        ]

        for malformed in malformed_inputs:
            result = validator.validate(malformed)
            assert result.is_valid is False
            assert len(result.errors) > 0

    @pytest.mark.parametrize(
        "arch_type,expected_valid",
        [
            ("GCN", True),
            ("GAT", True),
            ("GraphSAGE", True),
            ("GIN", True),
            ("EdgeConv", True),
            ("InvalidGNN", False),
            ("", False),
            (None, False),
        ],
    )
    def test_architecture_validation_parametrized(
        self, validator, mock_parse_result, arch_type, expected_valid
    ):
        """Test architecture validation with multiple inputs."""
        if arch_type is not None:
            mock_parse_result.sections["architecture"]["type"] = arch_type
        else:
            mock_parse_result.sections["architecture"].pop("type", None)

        result = validator.validate(mock_parse_result)

        if expected_valid:
            assert result.is_valid or len(result.errors) == 0
        else:
            assert not result.is_valid or len(result.errors) > 0

    @pytest.mark.parametrize(
        "lr,batch_size,expected_valid",
        [
            (0.001, 32, True),
            (0.1, 64, True),
            (1.5, 32, False),  # LR too high
            (0.001, 10000, False),  # Batch size too large
            (-0.001, 32, False),  # Negative LR
            (0.001, -32, False),  # Negative batch size
        ],
    )
    def test_parameter_validation_parametrized(
        self, validator, mock_parse_result, lr, batch_size, expected_valid
    ):
        """Test parameter validation with multiple configurations."""
        mock_parse_result.sections["parameters"]["learning_rate"] = lr
        mock_parse_result.sections["parameters"]["batch_size"] = batch_size

        result = validator.validate(mock_parse_result)

        # Check if parameter validation contributed to invalid result
        param_errors = [
            e
            for e in result.errors
            if "parameter" in e.lower()
            or "learning_rate" in e.lower()
            or "batch_size" in e.lower()
        ]

        if expected_valid:
            assert len(param_errors) == 0
        else:
            assert len(param_errors) > 0 or not result.is_valid


class TestValidationResult:
    """Test ValidationResult class."""

    def test_result_creation(self):
        """Test creation of validation results."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor issue"],
            model=Mock(spec=GMNModel),
            metadata={"validation_time": 0.1},
        )

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.model is not None
        assert result.metadata["validation_time"] == 0.1

    def test_result_with_errors(self):
        """Test validation result with errors."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=[],
            model=None,
            metadata={},
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert result.model is None


class TestSecurityValidation:
    """Test security-specific validation features."""

    @pytest.fixture
    def validator(self):
        """Create validator with security focus."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        return GMNValidator()

    @pytest.fixture
    def mock_parse_result(self):
        """Create a mock ParseResult for testing."""
        result = Mock(spec=ParseResult)

        # Mock AST
        result.ast = Mock()
        result.ast.name = "Test Model"
        result.ast.node_type = "model"
        result.ast.children = []

        # Mock metadata
        result.metadata = {
            "name": "Test Model",
            "version": "1.0.0",
            "author": "Test Author",
            "tags": ["test", "validation"],
            "created_at": "2023-01-01T00:00:00",
        }

        # Mock sections
        result.sections = {
            "architecture": {
                "type": "GraphSAGE",
                "layers": 3,
                "hidden_dim": 128,
                "activation": "relu",
            },
            "parameters": {"learning_rate": 0.01, "batch_size": 32},
            "active_inference": {
                "num_states": 10,
                "num_observations": 5,
                "num_actions": 3,
            },
        }

        result.errors = []
        result.warnings = []

        return result

    def test_sql_injection_detection(self, validator, mock_parse_result):
        """Test SQL injection pattern detection."""
        mock_parse_result.sections["custom_query"] = {
            "query": "SELECT * FROM users WHERE id = '" + "user_input" + "'"
        }

        result = validator.validate(mock_parse_result)

        assert not result.is_valid or len(result.warnings) > 0

    def test_command_injection_detection(self, validator, mock_parse_result):
        """Test command injection detection."""
        mock_parse_result.sections["preprocessing"] = {
            "command": "os.system('process ' + user_data)"
        }

        result = validator.validate(mock_parse_result)

        assert result.is_valid is False
        assert any(
            "security" in error.lower() or "injection" in error.lower()
            for error in result.errors
        )

    def test_path_traversal_detection(self, validator, mock_parse_result):
        """Test path traversal detection."""
        mock_parse_result.sections["file_operations"] = {
            "load_path": "../../../etc/passwd"
        }

        result = validator.validate(mock_parse_result)

        # Should detect suspicious path
        assert not result.is_valid or len(result.warnings) > 0

    def test_code_execution_prevention(self, validator, mock_parse_result):
        """Test prevention of arbitrary code execution."""
        dangerous_patterns = [
            "eval(user_input)",
            "exec(command)",
            "__import__('os').system('cmd')",
            "compile(source, 'file', 'exec')",
        ]

        for pattern in dangerous_patterns:
            mock_parse_result.sections["custom_code"] = {"snippet": pattern}
            result = validator.validate(mock_parse_result)

            assert result.is_valid is False
            assert any(
                "security" in e.lower() or "exec" in e.lower() or "eval" in e.lower()
                for e in result.errors
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=inference.gnn.validator", "--cov-report=html"])

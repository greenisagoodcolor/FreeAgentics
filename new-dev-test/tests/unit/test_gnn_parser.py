"""
Test suite for GNN Parser module - ConfigParser class.

This test suite provides comprehensive coverage for the ConfigParser class,
which parses GNN model configurations into AST structures.
Coverage target: 95%+
"""

from unittest.mock import patch

import pytest

# Import the module under test
try:
    from inference.gnn.parser import ASTNode, ConfigParser, ParseResult

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

    # Mock classes for testing when imports fail
    class ConfigParser:
        pass

    class ASTNode:
        pass

    class ParseResult:
        pass


class TestASTNode:
    """Test suite for ASTNode class."""

    def test_ast_node_creation(self):
        """Test ASTNode creation with basic parameters."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        node = ASTNode(node_type="test_type", line=1, column=5)

        assert node.node_type == "test_type"
        assert node.line == 1
        assert node.column == 5
        assert node.children == []
        assert node.attributes == {}

    def test_ast_node_with_children_and_attributes(self):
        """Test ASTNode with children and attributes."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        child1 = ASTNode("child1", 2, 1)
        child2 = ASTNode("child2", 3, 1)

        parent = ASTNode(
            node_type="parent",
            line=1,
            column=1,
            children=[child1, child2],
            attributes={"key1": "value1", "key2": "value2"},
        )

        assert len(parent.children) == 2
        assert parent.children[0] == child1
        assert parent.children[1] == child2
        assert parent.attributes["key1"] == "value1"
        assert parent.attributes["key2"] == "value2"

    def test_ast_node_modification(self):
        """Test modifying ASTNode after creation."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        node = ASTNode("test", 1, 1)

        # Add children
        child = ASTNode("child", 2, 1)
        node.children.append(child)

        # Add attributes
        node.attributes["new_key"] = "new_value"

        assert len(node.children) == 1
        assert node.children[0] == child
        assert node.attributes["new_key"] == "new_value"


class TestParseResult:
    """Test suite for ParseResult class."""

    def test_parse_result_creation(self):
        """Test ParseResult creation with default values."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        result = ParseResult()

        assert result.ast is None
        assert result.metadata == {}
        assert result.sections == {}
        assert result.errors == []
        assert result.warnings == []

    def test_parse_result_with_data(self):
        """Test ParseResult with actual data."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        ast_node = ASTNode("root", 1, 1)
        result = ParseResult(
            ast=ast_node,
            metadata={"name": "test"},
            sections={"architecture": "GraphSAGE"},
            errors=["error1"],
            warnings=["warning1"],
        )

        assert result.ast == ast_node
        assert result.metadata["name"] == "test"
        assert result.sections["architecture"] == "GraphSAGE"
        assert "error1" in result.errors
        assert "warning1" in result.warnings


class TestConfigParser:
    """Test suite for ConfigParser class."""

    @pytest.fixture
    def parser(self):
        """Create ConfigParser instance."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        return ConfigParser()

    @pytest.fixture
    def valid_config(self):
        """Valid configuration for testing."""
        return {
            "version": "2.0",
            "architecture": "GraphSAGE",
            "layers": [
                {"type": "conv", "input_dim": 10, "output_dim": 64},
                {"type": "conv", "input_dim": 64, "output_dim": 32},
            ],
            "hyperparameters": {
                "learning_rate": 0.01,
                "dropout": 0.1,
                "batch_size": 32,
            },
            "metadata": {
                "name": "test_model",
                "author": "test_user",
                "created_at": "2023-01-01",
            },
        }

    def test_parser_initialization(self, parser):
        """Test ConfigParser initialization."""
        assert parser.current_line == 0
        assert parser.current_column == 0

    def test_parse_valid_config(self, parser, valid_config):
        """Test parsing a valid configuration."""
        result = parser.parse(valid_config)

        assert isinstance(result, ParseResult)
        assert result.ast is not None
        assert result.ast.node_type == "root"
        assert result.ast.attributes["version"] == "2.0"
        assert len(result.errors) == 0

    def test_parse_config_sections(self, parser, valid_config):
        """Test that all sections are parsed correctly."""
        result = parser.parse(valid_config)

        # Check sections
        assert "architecture" in result.sections
        assert "layers" in result.sections
        assert "hyperparameters" in result.sections
        assert result.metadata == valid_config["metadata"]

        # Check section values
        assert result.sections["architecture"] == "GraphSAGE"
        assert len(result.sections["layers"]) == 2
        assert result.sections["hyperparameters"]["learning_rate"] == 0.01

    def test_parse_architecture_section(self, parser):
        """Test parsing architecture section specifically."""
        config = {"architecture": "GAT"}
        result = parser.parse(config)

        # Find architecture node in AST
        arch_nodes = [child for child in result.ast.children if child.node_type == "architecture"]
        assert len(arch_nodes) == 1
        assert arch_nodes[0].attributes["name"] == "GAT"

    def test_parse_layers_section(self, parser):
        """Test parsing layers section specifically."""
        layers_config = [
            {"type": "conv", "input_dim": 10, "output_dim": 64},
            {"type": "attention", "num_heads": 8},
            {"type": "pooling", "pool_type": "mean"},
        ]
        config = {"layers": layers_config}
        result = parser.parse(config)

        # Find layers node in AST
        layer_nodes = [child for child in result.ast.children if child.node_type == "layers"]
        assert len(layer_nodes) == 1

        layers_node = layer_nodes[0]
        assert len(layers_node.children) == 3

        # Check individual layer nodes
        for i, layer_child in enumerate(layers_node.children):
            assert layer_child.node_type == "layer"
            assert layer_child.attributes["index"] == i
            assert layer_child.attributes["type"] == layers_config[i]["type"]
            assert layer_child.attributes["config"] == layers_config[i]

    def test_parse_hyperparameters_section(self, parser):
        """Test parsing hyperparameters section."""
        hyperparams = {
            "learning_rate": 0.001,
            "dropout": 0.2,
            "weight_decay": 1e-5,
        }
        config = {"hyperparameters": hyperparams}
        result = parser.parse(config)

        # Find hyperparameters node
        hyper_nodes = [
            child for child in result.ast.children if child.node_type == "hyperparameters"
        ]
        assert len(hyper_nodes) == 1
        assert hyper_nodes[0].attributes == hyperparams

    def test_parse_metadata_section(self, parser):
        """Test parsing metadata section."""
        metadata = {
            "name": "test_model",
            "version": "1.0",
            "author": "test_user",
            "tags": ["gnn", "test"],
        }
        config = {"metadata": metadata}
        result = parser.parse(config)

        # Find metadata node
        meta_nodes = [child for child in result.ast.children if child.node_type == "metadata"]
        assert len(meta_nodes) == 1
        assert meta_nodes[0].attributes == metadata
        assert result.metadata == metadata

    def test_parse_empty_config(self, parser):
        """Test parsing empty configuration."""
        result = parser.parse({})

        assert result.ast is not None
        assert result.ast.node_type == "root"
        assert result.ast.attributes["version"] == "1.0"  # Default
        assert len(result.ast.children) == 0
        assert len(result.errors) == 0

    def test_parse_partial_config(self, parser):
        """Test parsing configuration with only some sections."""
        config = {"architecture": "GCN", "metadata": {"name": "partial_model"}}
        result = parser.parse(config)

        assert len(result.ast.children) == 2  # architecture + metadata
        assert "architecture" in result.sections
        assert result.metadata["name"] == "partial_model"
        assert "layers" not in result.sections
        assert "hyperparameters" not in result.sections

    def test_parse_with_custom_version(self, parser):
        """Test parsing with custom version in config."""
        config = {"version": "3.0", "architecture": "Custom"}
        result = parser.parse(config)

        assert result.ast.attributes["version"] == "3.0"

    def test_parse_error_handling(self, parser):
        """Test error handling during parsing."""
        # Create a config that will cause an exception
        with patch.object(parser, "_parse_architecture") as mock_parse:
            mock_parse.side_effect = Exception("Parse error")

            config = {"architecture": "test"}
            result = parser.parse(config)

            assert len(result.errors) > 0
            assert "Parse error" in result.errors[0]

    def test_parse_architecture_internal(self, parser):
        """Test internal _parse_architecture method."""
        arch_node = parser._parse_architecture("GraphSAGE")

        assert arch_node.node_type == "architecture"
        assert arch_node.attributes["name"] == "GraphSAGE"
        assert arch_node.line == parser.current_line
        assert arch_node.column == parser.current_column

    def test_parse_layers_internal(self, parser):
        """Test internal _parse_layers method."""
        layers = [
            {"type": "conv", "input_dim": 10},
            {"type": "attention", "num_heads": 4},
        ]
        layers_node = parser._parse_layers(layers)

        assert layers_node.node_type == "layers"
        assert len(layers_node.children) == 2

        for i, child in enumerate(layers_node.children):
            assert child.node_type == "layer"
            assert child.attributes["index"] == i
            assert child.attributes["config"] == layers[i]

    def test_parse_hyperparameters_internal(self, parser):
        """Test internal _parse_hyperparameters method."""
        hyperparams = {"lr": 0.01, "dropout": 0.1}
        hyper_node = parser._parse_hyperparameters(hyperparams)

        assert hyper_node.node_type == "hyperparameters"
        assert hyper_node.attributes == hyperparams

    def test_parse_metadata_internal(self, parser):
        """Test internal _parse_metadata method."""
        metadata = {"name": "test", "version": "1.0"}
        meta_node = parser._parse_metadata(metadata)

        assert meta_node.node_type == "metadata"
        assert meta_node.attributes == metadata

    def test_parser_state_tracking(self, parser):
        """Test that parser tracks line and column state."""
        initial_line = parser.current_line
        initial_column = parser.current_column

        # Parse something
        parser.parse({"architecture": "test"})

        # State should be preserved (in this simple implementation)
        assert parser.current_line == initial_line
        assert parser.current_column == initial_column

    def test_complex_nested_config(self, parser):
        """Test parsing complex nested configuration."""
        config = {
            "version": "2.1",
            "architecture": "GAT",
            "layers": [
                {
                    "type": "attention",
                    "input_dim": 128,
                    "output_dim": 64,
                    "num_heads": 8,
                    "dropout": 0.1,
                    "edge_features": True,
                },
                {
                    "type": "conv",
                    "input_dim": 64,
                    "output_dim": 32,
                    "activation": "relu",
                    "normalization": "batch",
                },
            ],
            "hyperparameters": {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "weight_decay": 1e-4,
                "scheduler": {"type": "step", "step_size": 10, "gamma": 0.1},
            },
            "metadata": {
                "name": "complex_gat_model",
                "version": "1.2.3",
                "author": "researcher",
                "institution": "university",
                "tags": ["attention", "graph", "neural", "network"],
                "description": "Complex GAT model for graph classification",
            },
        }

        result = parser.parse(config)

        assert result.ast is not None
        assert len(result.errors) == 0
        assert len(result.ast.children) == 4  # arch, layers, hyperparams, metadata

        # Verify all sections were parsed
        assert "architecture" in result.sections
        assert "layers" in result.sections
        assert "hyperparameters" in result.sections
        assert result.metadata["name"] == "complex_gat_model"

    @pytest.mark.parametrize(
        "architecture",
        ["GCN", "GAT", "GraphSAGE", "GIN", "EdgeConv", "Custom"],
    )
    def test_parse_different_architectures(self, parser, architecture):
        """Test parsing different architecture types."""
        config = {"architecture": architecture}
        result = parser.parse(config)

        assert result.sections["architecture"] == architecture

    def test_parse_thread_safety(self, parser, valid_config):
        """Test that parsing is thread-safe."""
        import threading

        results = []
        errors = []

        def parse_config():
            try:
                result = parser.parse(valid_config.copy())
                results.append(result.sections["architecture"])
            except Exception as e:
                errors.append(e)

        # Parse concurrently
        threads = []
        for _ in range(10):
            t = threading.Thread(target=parse_config)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(arch == "GraphSAGE" for arch in results)

    def test_parse_with_logging(self, parser):
        """Test that parsing errors are logged."""
        with patch("inference.gnn.parser.logger") as mock_logger:
            with patch.object(parser, "_parse_architecture") as mock_parse:
                mock_parse.side_effect = Exception("Test error")

                config = {"architecture": "test"}
                parser.parse(config)

                mock_logger.error.assert_called_once()
                assert "Test error" in str(mock_logger.error.call_args)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=inference.gnn.parser", "--cov-report=html"])

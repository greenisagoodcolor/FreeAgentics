from datetime import datetime
from pathlib import Path

import pytest

from inference.gnn.parser import GMNBlockParser, GMNLexer, GMNParser, GMNSyntaxError, Token


class TestGMNLexer:
    """Test the GNN lexical analyzer"""

    def test_tokenize_simple_object(self) -> None:
        """Test tokenizing a simple GNN object"""
        lexer = GMNLexer('{ type: "GraphSAGE", layers: 3 }')
        tokens = lexer.tokenize()
        assert len(tokens) == 9
        assert tokens[0].type == "LBRACE"
        assert tokens[1].type == "IDENTIFIER"
        assert tokens[1].value == "type"
        assert tokens[3].type == "STRING"
        assert tokens[3].value == '"GraphSAGE"'
        assert tokens[7].type == "NUMBER"
        assert tokens[7].value == "3"
        assert tokens[8].type == "RBRACE"

    def test_tokenize_nested_object(self) -> None:
        """Test tokenizing nested objects"""
        lexer = GMNLexer(
            '\n        architecture {\n            type: "GAT"\n            attention: {\n                heads: 8\n                dropout: 0.2\n            }\n        }\n        '
        )
        tokens = lexer.tokenize()
        identifier_values = [t.value for t in tokens if t.type == "IDENTIFIER"]
        assert "architecture" in identifier_values
        assert "attention" in identifier_values
        assert "heads" in identifier_values
        assert "dropout" in identifier_values

    def test_tokenize_array(self) -> None:
        """Test tokenizing arrays"""
        lexer = GMNLexer('features: ["x", "y", "z"]')
        tokens = lexer.tokenize()
        lbracket_idx = next((i for i, t in enumerate(tokens) if t.type == "LBRACKET"))
        rbracket_idx = next((i for i, t in enumerate(tokens) if t.type == "RBRACKET"))
        array_tokens = tokens[lbracket_idx + 1 : rbracket_idx]
        string_values = [t.value[1:-1] for t in array_tokens if t.type == "STRING"]
        assert string_values == ["x", "y", "z"]

    def test_tokenize_comments(self) -> None:
        """Test that comments are properly ignored"""
        lexer = GMNLexer(
            '\n        type: "GCN"  // This is a comment\n        /* Multi-line\n           comment */\n        layers: 3\n        '
        )
        tokens = lexer.tokenize()
        token_types = [t.type for t in tokens]
        assert "COMMENT" not in token_types
        assert "MULTILINE_COMMENT" not in token_types

    def test_invalid_character(self) -> None:
        """Test error on invalid character"""
        lexer = GMNLexer('type: "test" @ invalid')
        with pytest.raises(GMNSyntaxError) as exc_info:
            lexer.tokenize()
        assert "Unexpected character: @" in str(exc_info.value)


class TestGMNBlockParser:
    """Test the GNN block parser"""

    def test_parse_simple_object(self) -> None:
        """Test parsing a simple object"""
        tokens = [
            Token("LBRACE", "{", 1, 1),
            Token("IDENTIFIER", "type", 1, 3),
            Token("COLON", ":", 1, 7),
            Token("STRING", '"GraphSAGE"', 1, 9),
            Token("RBRACE", "}", 1, 21),
        ]
        parser = GMNBlockParser(tokens)
        result = parser.parse()
        assert result == {"type": "GraphSAGE"}

    def test_parse_numbers(self) -> None:
        """Test parsing different number formats"""
        tokens = [
            Token("IDENTIFIER", "integer", 1, 1),
            Token("COLON", ":", 1, 8),
            Token("NUMBER", "42", 1, 10),
            Token("IDENTIFIER", "float", 2, 1),
            Token("COLON", ":", 2, 6),
            Token("NUMBER", "3.14", 2, 8),
            Token("IDENTIFIER", "negative", 3, 1),
            Token("COLON", ":", 3, 9),
            Token("NUMBER", "-10", 3, 11),
        ]
        parser = GMNBlockParser(tokens)
        result = parser.parse()
        assert result["integer"] == 42
        assert result["float"] == 3.14
        assert result["negative"] == -10

    def test_parse_booleans(self) -> None:
        """Test parsing boolean values"""
        tokens = [
            Token("IDENTIFIER", "flag1", 1, 1),
            Token("COLON", ":", 1, 6),
            Token("IDENTIFIER", "true", 1, 8),
            Token("IDENTIFIER", "flag2", 2, 1),
            Token("COLON", ":", 2, 6),
            Token("IDENTIFIER", "false", 2, 8),
        ]
        parser = GMNBlockParser(tokens)
        result = parser.parse()
        assert result["flag1"] is True
        assert result["flag2"] is False

    def test_parse_nested_objects(self) -> None:
        """Test parsing nested objects"""
        lexer = GMNLexer(
            '\n        {\n            beliefs: {\n                initial: "uniform"\n                update_rule: "bayesian"\n            }\n            preferences: {\n                exploration: 0.7\n                exploitation: 0.3\n            }\n        }\n        '
        )
        tokens = lexer.tokenize()
        parser = GMNBlockParser(tokens)
        result = parser.parse()
        assert result["beliefs"]["initial"] == "uniform"
        assert result["beliefs"]["update_rule"] == "bayesian"
        assert result["preferences"]["exploration"] == 0.7
        assert result["preferences"]["exploitation"] == 0.3

    def test_parse_arrays(self) -> None:
        """Test parsing arrays"""
        lexer = GMNLexer('{ tags: ["explorer", "cautious"], values: [1, 2, 3] }')
        tokens = lexer.tokenize()
        parser = GMNBlockParser(tokens)
        result = parser.parse()
        assert result["tags"] == ["explorer", "cautious"]
        assert result["values"] == [1, 2, 3]


class TestGMNParser:
    """Test the main GNN parser"""

    def test_parse_minimal_gnn(self) -> None:
        """Test parsing a minimal valid GNN file"""
        content = '\n# Test Model\n\n## Metadata\n- Version: 1.0.0\n\n## Architecture\n```gnn\narchitecture {\n  type: "GCN"\n  layers: 2\n}\n```\n'
        parser = GMNParser()
        result = parser.parse(content)
        assert result.metadata["name"] == "Test Model"
        assert result.metadata["version"] == "1.0.0"
        assert result.sections["architecture"]["type"] == "GCN"
        assert result.sections["architecture"]["layers"] == 2
        assert len(result.errors) == 0

    def test_parse_complete_gnn(self) -> None:
        """Test parsing a complete GNN file with all sections"""
        content = '\n# Explorer Cautious Model\n\n## Metadata\n- Version: 1.0.0\n- Author: FreeAgentics Team\n- Created: 2024-01-15T10:00:00Z\n- Modified: 2024-01-15T10:00:00Z\n- Tags: [explorer, cautious, efficient]\n\n## Description\nThis model implements a cautious explorer agent.\n\n## Architecture\n```gnn\narchitecture {\n  type: "GraphSAGE"\n  layers: 3\n  hidden_dim: 128\n  activation: "relu"\n  dropout: 0.2\n}\n```\n\n## Parameters\n```gnn\nparameters {\n  learning_rate: 0.001\n  optimizer: "adam"\n  batch_size: 32\n  epochs: 100\n}\n```\n\n## Active Inference Mapping\n```gnn\nactive_inference {\n  beliefs {\n    initial: "gaussian"\n    update_rule: "variational"\n  }\n  preferences {\n    exploration: 0.3\n    exploitation: 0.7\n  }\n}\n```\n\n## Node Features\n```gnn\nnode_features {\n  spatial: ["x", "y", "z"]\n  numerical: {\n    energy: { range: [0, 1], default: 1.0 }\n  }\n}\n```\n'
        parser = GMNParser()
        result = parser.parse(content)
        assert result.metadata["name"] == "Explorer Cautious Model"
        assert result.metadata["version"] == "1.0.0"
        assert result.metadata["author"] == "FreeAgentics Team"
        assert result.metadata["tags"] == ["explorer", "cautious", "efficient"]
        assert "description" in result.sections
        assert "architecture" in result.sections
        assert "parameters" in result.sections
        assert "active_inference_mapping" in result.sections
        assert "node_features" in result.sections
        arch = result.sections["architecture"]
        assert arch["type"] == "GraphSAGE"
        assert arch["layers"] == 3
        assert arch["hidden_dim"] == 128
        ai = result.sections["active_inference_mapping"]
        assert ai["beliefs"]["initial"] == "gaussian"
        assert ai["preferences"]["exploration"] == 0.3
        assert len(result.errors) == 0

    def test_parse_metadata_variations(self) -> None:
        """Test parsing different metadata formats"""
        content = '\n# Model Name\n\n## Metadata\n- Version: 2.1.0\n- Created: 2024-01-15T10:00:00Z\n- Tags: [tag1, tag2, tag3]\n- Custom-Field: custom value\n\n## Architecture\n```gnn\narchitecture { type: "GCN" }\n```\n'
        parser = GMNParser()
        result = parser.parse(content)
        assert result.metadata["version"] == "2.1.0"
        assert isinstance(result.metadata["created"], datetime)
        assert result.metadata["tags"] == ["tag1", "tag2", "tag3"]
        assert result.metadata["custom-field"] == "custom value"

    def test_missing_required_sections(self) -> None:
        """Test error on missing required sections"""
        content = "\n# Model Name\n\n## Description\nSome description\n"
        parser = GMNParser()
        result = parser.parse(content)
        assert len(result.errors) > 0
        assert any(("architecture" in error for error in result.errors))

    def test_validate_syntax(self) -> None:
        """Test syntax validation without full parsing"""
        valid_content = '\n# Model\n\n## Metadata\n- Version: 1.0.0\n\n## Architecture\n```gnn\ntype: "GCN"\n```\n'
        parser = GMNParser()
        errors = parser.validate_syntax(valid_content)
        assert len(errors) == 0
        invalid_content1 = "\n## Metadata\n- Version: 1.0.0\n"
        errors = parser.validate_syntax(invalid_content1)
        assert any(("# " in error for error in errors))
        invalid_content2 = '\n# Model\n\n## Architecture\n```gnn\ntype: "GCN"\n'
        errors = parser.validate_syntax(invalid_content2)
        assert any(("Unclosed GNN block" in error for error in errors))

    def test_parse_file(self, tmp_path) -> None:
        """Test parsing from file"""
        gnn_file = tmp_path / "test.gnn.md"
        gnn_file.write_text(
            '\n# Test Model\n\n## Metadata\n- Version: 1.0.0\n\n## Architecture\n```gnn\narchitecture {\n  type: "GAT"\n  attention_heads: 4\n}\n```\n'
        )
        parser = GMNParser()
        result = parser.parse_file(gnn_file)
        assert result.metadata["name"] == "Test Model"
        assert result.sections["architecture"]["type"] == "GAT"
        assert result.sections["architecture"]["attention_heads"] == 4

    def test_parse_file_not_found(self) -> None:
        """Test error on file not found"""
        parser = GMNParser()
        with pytest.raises(GMNSyntaxError) as exc_info:
            parser.parse_file(Path("nonexistent.gnn.md"))
        assert "File not found" in str(exc_info.value)

    def test_ast_structure(self) -> None:
        """Test AST structure generation"""
        content = '\n# Model\n\n## Metadata\n- Version: 1.0.0\n\n## Architecture\n```gnn\narchitecture { type: "GCN" }\n```\n'
        parser = GMNParser()
        result = parser.parse(content)
        assert result.ast.node_type == "root"
        assert len(result.ast.children) >= 2
        metadata_node = next((n for n in result.ast.children if n.node_type == "metadata"))
        assert metadata_node.attributes["version"] == "1.0.0"
        arch_node = next((n for n in result.ast.children if n.node_type == "architecture"))
        assert arch_node.attributes["type"] == "GCN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

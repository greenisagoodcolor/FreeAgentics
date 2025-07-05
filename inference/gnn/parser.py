"""Parser for GNN model configurations."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ASTNode:
    """Abstract Syntax Tree node for parsed configurations."""

    node_type: str
    line: int
    column: int
    children: List["ASTNode"] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing a configuration."""

    ast: Optional[ASTNode] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigParser:
    """Parser for GNN model configurations."""

    def __init__(self) -> None:
        """Initialize the parser."""
        self.current_line = 0
        self.current_column = 0

    def parse(self, config: Dict[str, Any]) -> ParseResult:
        """Parse a configuration dictionary into an AST."""
        result = ParseResult()

        try:
            # Create root AST node
            root = ASTNode(
                node_type="root",
                line=0,
                column=0,
                attributes={"version": config.get("version", "1.0")},
            )

            # Parse main sections
            if "architecture" in config:
                arch_node = self._parse_architecture(config["architecture"])
                root.children.append(arch_node)
                result.sections["architecture"] = config["architecture"]

            if "layers" in config:
                layers_node = self._parse_layers(config["layers"])
                root.children.append(layers_node)
                result.sections["layers"] = config["layers"]

            if "hyperparameters" in config:
                hyper_node = self._parse_hyperparameters(config["hyperparameters"])
                root.children.append(hyper_node)
                result.sections["hyperparameters"] = config["hyperparameters"]

            if "metadata" in config:
                meta_node = self._parse_metadata(config["metadata"])
                root.children.append(meta_node)
                result.metadata = config["metadata"]

            result.ast = root

        except Exception as e:
            result.errors.append(f"Parse error: {str(e)}")
            logger.error(f"Failed to parse configuration: {e}")

        return result

    def _parse_architecture(self, architecture: str) -> ASTNode:
        """Parse architecture specification."""
        return ASTNode(
            node_type="architecture",
            line=self.current_line,
            column=self.current_column,
            attributes={"name": architecture},
        )

    def _parse_layers(self, layers: List[Dict[str, Any]]) -> ASTNode:
        """Parse layers configuration."""
        layers_node = ASTNode(
            node_type="layers", line=self.current_line, column=self.current_column
        )

        for i, layer in enumerate(layers):
            layer_node = ASTNode(
                node_type="layer",
                line=self.current_line + i,
                column=self.current_column,
                attributes={"index": i, "type": layer.get("type", "unknown"), "config": layer},
            )
            layers_node.children.append(layer_node)

        return layers_node

    def _parse_hyperparameters(self, hyperparams: Dict[str, Any]) -> ASTNode:
        """Parse hyperparameters."""
        return ASTNode(
            node_type="hyperparameters",
            line=self.current_line,
            column=self.current_column,
            attributes=hyperparams,
        )

    def _parse_metadata(self, metadata: Dict[str, Any]) -> ASTNode:
        """Parse metadata."""
        return ASTNode(
            node_type="metadata",
            line=self.current_line,
            column=self.current_column,
            attributes=metadata,
        )

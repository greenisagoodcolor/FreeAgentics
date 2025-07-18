"""Comprehensive GMN Validation Framework.

This module implements a robust validation framework for GMN specifications
with strict validation rules and hard failures on any violation. The framework
ensures that GMN specifications meet all requirements before processing,
making it critical for VC demo reliability.

The validation framework includes:
- Syntax validation (correct GMN format)
- Semantic validation (logical consistency)
- Mathematical validation (probability constraints)
- Type validation (correct data types)
- Constraint validation (business rules)

All validators implement hard failures with no graceful degradation.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from inference.active.gmn_parser import (
    GMNEdgeType,
    GMNNodeType,
    GMNValidationError,
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """Detailed validation error information."""

    validator: str
    level: ValidationLevel
    message: str
    node_name: Optional[str] = None
    edge_index: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Comprehensive validation result."""

    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    info: List[ValidationError]

    def add_error(self, validator: str, message: str, **kwargs):
        """Add an error to the validation result."""
        error = ValidationError(
            validator=validator,
            level=ValidationLevel.ERROR,
            message=message,
            **kwargs,
        )
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, validator: str, message: str, **kwargs):
        """Add a warning to the validation result."""
        warning = ValidationError(
            validator=validator,
            level=ValidationLevel.WARNING,
            message=message,
            **kwargs,
        )
        self.warnings.append(warning)


class GMNSyntaxValidator:
    """Syntax validator for GMN specifications with hard failures."""

    def __init__(self):
        """Initialize the syntax validator."""
        self.valid_node_name_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        self.required_top_level_fields = {"nodes"}
        self.required_node_fields = {"name", "type"}
        self.required_edge_fields = {"from", "to", "type"}

    def validate(self, spec: Dict[str, Any]) -> ValidationResult:
        """Validate GMN specification syntax with hard failures."""
        if not spec:
            raise GMNValidationError("Empty specification")

        if not isinstance(spec, dict):
            raise GMNValidationError("Specification must be a dictionary")

        result = ValidationResult(
            is_valid=True, errors=[], warnings=[], info=[]
        )

        # Validate top-level structure
        self._validate_top_level_structure(spec, result)

        # Validate nodes
        if "nodes" in spec:
            self._validate_nodes_structure(spec["nodes"], result)

        # Validate edges
        if "edges" in spec:
            self._validate_edges_structure(spec["edges"], result)

        # Hard failure on any errors
        if not result.is_valid:
            error_messages = [
                f"{error.validator}: {error.message}"
                for error in result.errors
            ]
            raise GMNValidationError(
                f"Syntax validation failed: {'; '.join(error_messages)}"
            )

        return result

    def validate_text(self, text: str) -> ValidationResult:
        """Validate GMN text format syntax."""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=[], info=[]
        )

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        current_section = None
        valid_sections = {"nodes", "edges"}

        for line_num, line in enumerate(lines, 1):
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                if section not in valid_sections:
                    result.add_error(
                        "SyntaxValidator",
                        f"Invalid section header: [{section}]",
                        context={"line": line_num, "content": line},
                    )
                current_section = section
            elif current_section == "nodes":
                if not self._validate_node_line_syntax(line):
                    result.add_error(
                        "SyntaxValidator",
                        f"Invalid node syntax: {line}",
                        context={"line": line_num, "section": "nodes"},
                    )
            elif current_section == "edges":
                if not self._validate_edge_line_syntax(line):
                    result.add_error(
                        "SyntaxValidator",
                        f"Invalid edge syntax: {line}",
                        context={"line": line_num, "section": "edges"},
                    )

        if not result.is_valid:
            error_messages = [
                f"Line {error.context.get('line', '?')}: {error.message}"
                for error in result.errors
            ]
            raise GMNValidationError(
                f"Text format validation failed: {'; '.join(error_messages)}"
            )

        return result

    def _validate_top_level_structure(
        self, spec: Dict[str, Any], result: ValidationResult
    ):
        """Validate top-level specification structure."""
        # Check required fields
        for field in self.required_top_level_fields:
            if field not in spec:
                result.add_error(
                    "SyntaxValidator", f"Missing required field: {field}"
                )

        # Validate nodes field type
        if "nodes" in spec and not isinstance(spec["nodes"], list):
            result.add_error("SyntaxValidator", "Field 'nodes' must be a list")

        # Validate edges field type
        if "edges" in spec and not isinstance(spec["edges"], list):
            result.add_error("SyntaxValidator", "Field 'edges' must be a list")

    def _validate_nodes_structure(
        self, nodes: List[Any], result: ValidationResult
    ):
        """Validate nodes structure."""
        if not isinstance(nodes, list):
            result.add_error("SyntaxValidator", "Nodes must be a list")
            return

        node_names = set()

        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                result.add_error(
                    "SyntaxValidator", f"Node {i} must be a dictionary"
                )
                continue

            # Check required fields
            for field in self.required_node_fields:
                if field not in node:
                    result.add_error(
                        "SyntaxValidator",
                        f"Node {i} missing required field: {field}",
                    )

            # Validate node name
            if "name" in node:
                name = node["name"]
                if not name or not isinstance(name, str):
                    result.add_error(
                        "SyntaxValidator",
                        f"Node {i} name cannot be empty",
                        node_name=name,
                    )
                elif not self.valid_node_name_pattern.match(name):
                    result.add_error(
                        "SyntaxValidator",
                        f"Node {i} name contains invalid characters: {name}",
                        node_name=name,
                    )
                elif name in node_names:
                    result.add_error(
                        "SyntaxValidator",
                        f"Duplicate node name: {name}",
                        node_name=name,
                    )
                else:
                    node_names.add(name)

    def _validate_edges_structure(
        self, edges: List[Any], result: ValidationResult
    ):
        """Validate edges structure."""
        if not isinstance(edges, list):
            result.add_error("SyntaxValidator", "Edges must be a list")
            return

        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                result.add_error(
                    "SyntaxValidator", f"Edge {i} must be a dictionary"
                )
                continue

            # Check required fields
            for field in self.required_edge_fields:
                if field not in edge:
                    result.add_error(
                        "SyntaxValidator",
                        f"Edge {i} missing required field: {field}",
                        edge_index=i,
                    )

    def _validate_node_line_syntax(self, line: str) -> bool:
        """Validate node line syntax in text format."""
        # Format: name: type {params}
        return bool(re.match(r"^\w+:\s*\w+(?:\s*\{[^}]*\})?$", line))

    def _validate_edge_line_syntax(self, line: str) -> bool:
        """Validate edge line syntax in text format."""
        # Format: from -> to: type
        return bool(re.match(r"^\w+\s*->\s*\w+:\s*\w+$", line))


class GMNSemanticValidator:
    """Semantic validator for GMN specifications with logical consistency checks."""

    def __init__(self):
        """Initialize the semantic validator."""
        self.valid_dependencies = {
            # Node type -> allowed dependency types
            "belief": ["state"],
            "transition": ["state", "action"],
            "likelihood": ["state"],
            "preference": ["observation"],
        }

    def validate(self, spec: Dict[str, Any]) -> ValidationResult:
        """Validate GMN specification semantics with hard failures."""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=[], info=[]
        )

        nodes_by_name = self._build_node_lookup(spec.get("nodes", []))
        edges = spec.get("edges", [])

        # Validate node references in edges
        self._validate_node_references(edges, nodes_by_name, result)

        # Validate edge relationships
        self._validate_edge_relationships(edges, nodes_by_name, result)

        # Check for circular dependencies
        self._validate_circular_dependencies(edges, result)

        # Validate required connections
        self._validate_required_connections(
            spec.get("nodes", []), edges, result
        )

        # Check for unreferenced nodes
        self._validate_node_usage(spec.get("nodes", []), edges, result)

        # Hard failure on any errors
        if not result.is_valid:
            error_messages = [
                f"{error.validator}: {error.message}"
                for error in result.errors
            ]
            raise GMNValidationError(
                f"Semantic validation failed: {'; '.join(error_messages)}"
            )

        return result

    def _build_node_lookup(
        self, nodes: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Build lookup table for nodes by name."""
        return {node.get("name"): node for node in nodes if "name" in node}

    def _validate_node_references(
        self,
        edges: List[Dict[str, Any]],
        nodes_by_name: Dict[str, Dict[str, Any]],
        result: ValidationResult,
    ):
        """Validate that edges reference existing nodes."""
        for i, edge in enumerate(edges):
            from_node = edge.get("from")
            to_node = edge.get("to")

            if from_node and from_node not in nodes_by_name:
                result.add_error(
                    "SemanticValidator",
                    f"Edge {i} references non-existent node: {from_node}",
                    edge_index=i,
                )

            if to_node and to_node not in nodes_by_name:
                result.add_error(
                    "SemanticValidator",
                    f"Edge {i} references non-existent node: {to_node}",
                    edge_index=i,
                )

    def _validate_edge_relationships(
        self,
        edges: List[Dict[str, Any]],
        nodes_by_name: Dict[str, Dict[str, Any]],
        result: ValidationResult,
    ):
        """Validate logical consistency of edge relationships."""
        for i, edge in enumerate(edges):
            from_node_name = edge.get("from")
            to_node_name = edge.get("to")
            edge_type = edge.get("type")

            if not all([from_node_name, to_node_name, edge_type]):
                continue

            from_node = nodes_by_name.get(from_node_name)
            to_node = nodes_by_name.get(to_node_name)

            if not from_node or not to_node:
                continue

            from_type = from_node.get("type")
            to_type = to_node.get("type")

            # Validate dependency relationships
            if edge_type == "depends_on":
                if from_type == "state" and to_type == "observation":
                    result.add_error(
                        "SemanticValidator",
                        f"Invalid dependency: state cannot depend on observation",
                        edge_index=i,
                    )
                elif from_type == "action" and to_type in [
                    "state",
                    "observation",
                ]:
                    result.add_error(
                        "SemanticValidator",
                        f"Invalid dependency: action cannot depend on {to_type}",
                        edge_index=i,
                    )

            # Validate generation relationships
            elif edge_type == "generates":
                if from_type != "state" and from_type != "likelihood":
                    result.add_error(
                        "SemanticValidator",
                        f"Invalid generation: only states and likelihood nodes can generate",
                        edge_index=i,
                    )

    def _validate_circular_dependencies(
        self, edges: List[Dict[str, Any]], result: ValidationResult
    ):
        """Check for circular dependencies in the dependency graph."""
        # Build dependency graph
        graph = {}
        for edge in edges:
            if edge.get("type") == "depends_on":
                from_node = edge.get("from")
                to_node = edge.get("to")
                if from_node and to_node:
                    if from_node not in graph:
                        graph[from_node] = []
                    graph[from_node].append(to_node)

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    result.add_error(
                        "SemanticValidator",
                        f"Circular dependency detected involving node: {node}",
                    )
                    break

    def _validate_required_connections(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        result: ValidationResult,
    ):
        """Validate that required connections exist."""
        # Build edge lookup
        dependencies = {}
        for edge in edges:
            if edge.get("type") == "depends_on":
                from_node = edge.get("from")
                to_node = edge.get("to")
                if from_node:
                    if from_node not in dependencies:
                        dependencies[from_node] = []
                    dependencies[from_node].append(to_node)

        # Check required connections
        for node in nodes:
            node_name = node.get("name")
            node_type = node.get("type")

            if node_type == "belief":
                # Belief nodes must be connected to a state
                connected_states = [
                    dep
                    for dep in dependencies.get(node_name, [])
                    if any(
                        n.get("name") == dep and n.get("type") == "state"
                        for n in nodes
                    )
                ]
                if not connected_states:
                    result.add_error(
                        "SemanticValidator",
                        f"Belief node '{node_name}' must be connected to a state",
                        node_name=node_name,
                    )

    def _validate_node_usage(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        result: ValidationResult,
    ):
        """Validate that all nodes are properly used."""
        referenced_nodes = set()

        # Collect all referenced nodes
        for edge in edges:
            if "from" in edge:
                referenced_nodes.add(edge["from"])
            if "to" in edge:
                referenced_nodes.add(edge["to"])

        # Check for unreferenced nodes (but allow some node types to be standalone)
        standalone_allowed = {"preference"}  # Preferences can be standalone

        for node in nodes:
            node_name = node.get("name")
            node_type = node.get("type")

            if (
                node_name
                and node_name not in referenced_nodes
                and node_type not in standalone_allowed
            ):
                # This is a warning, not an error, as some nodes might be valid but unreferenced
                result.add_warning(
                    "SemanticValidator",
                    f"Unreferenced node: {node_name}",
                    node_name=node_name,
                )


class GMNMathematicalValidator:
    """Mathematical validator for GMN specifications with probability constraints."""

    def __init__(self):
        """Initialize the mathematical validator."""
        self.tolerance = 1e-10
        self.max_dimension = 1000000  # Reasonable limit for dimensions

    def validate(self, spec: Dict[str, Any]) -> ValidationResult:
        """Validate GMN specification mathematics with hard failures."""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=[], info=[]
        )

        nodes = spec.get("nodes", [])
        edges = spec.get("edges", [])

        # Validate probability distributions
        self._validate_probability_distributions(nodes, result)

        # Validate dimensions
        self._validate_dimensions(nodes, edges, result)

        # Validate numerical ranges
        self._validate_numerical_ranges(nodes, result)

        # Validate matrix constraints
        self._validate_matrix_constraints(nodes, result)

        # Hard failure on any errors
        if not result.is_valid:
            error_messages = [
                f"{error.validator}: {error.message}"
                for error in result.errors
            ]
            raise GMNValidationError(
                f"Mathematical validation failed: {'; '.join(error_messages)}"
            )

        return result

    def _validate_probability_distributions(
        self, nodes: List[Dict[str, Any]], result: ValidationResult
    ):
        """Validate probability distributions sum to 1 and are non-negative."""
        for node in nodes:
            node_name = node.get("name", "unnamed")

            # Check initial_distribution
            if "initial_distribution" in node:
                dist = node["initial_distribution"]
                if not isinstance(dist, list):
                    result.add_error(
                        "MathematicalValidator",
                        f"initial_distribution must be a list",
                        node_name=node_name,
                    )
                    continue

                try:
                    values = np.array(dist, dtype=float)

                    # Check for negative values
                    if np.any(values < 0):
                        result.add_error(
                            "MathematicalValidator",
                            f"Probability distribution contains negative values: {values}",
                            node_name=node_name,
                        )

                    # Check sum
                    total = np.sum(values)
                    if abs(total - 1.0) > self.tolerance:
                        result.add_error(
                            "MathematicalValidator",
                            f"Probability distribution does not sum to 1 (sum={total})",
                            node_name=node_name,
                        )

                except (ValueError, TypeError) as e:
                    result.add_error(
                        "MathematicalValidator",
                        f"Invalid probability distribution format: {e}",
                        node_name=node_name,
                    )

            # Check initial_distributions (factorized)
            if "initial_distributions" in node:
                dists = node["initial_distributions"]
                if not isinstance(dists, dict):
                    result.add_error(
                        "MathematicalValidator",
                        f"initial_distributions must be a dictionary",
                        node_name=node_name,
                    )
                    continue

                for factor_name, dist in dists.items():
                    if not isinstance(dist, list):
                        result.add_error(
                            "MathematicalValidator",
                            f"Distribution for factor '{factor_name}' must be a list",
                            node_name=node_name,
                        )
                        continue

                    try:
                        values = np.array(dist, dtype=float)

                        if np.any(values < 0):
                            result.add_error(
                                "MathematicalValidator",
                                f"Factor '{factor_name}' contains negative values",
                                node_name=node_name,
                            )

                        total = np.sum(values)
                        if abs(total - 1.0) > self.tolerance:
                            result.add_error(
                                "MathematicalValidator",
                                f"Factor '{factor_name}' does not sum to 1 (sum={total})",
                                node_name=node_name,
                            )

                    except (ValueError, TypeError) as e:
                        result.add_error(
                            "MathematicalValidator",
                            f"Invalid format for factor '{factor_name}': {e}",
                            node_name=node_name,
                        )

    def _validate_dimensions(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        result: ValidationResult,
    ):
        """Validate dimension consistency between connected nodes."""
        # Build node lookup
        nodes_by_name = {
            node.get("name"): node for node in nodes if "name" in node
        }

        for edge in edges:
            if edge.get("type") == "generates":
                from_node_name = edge.get("from")
                to_node_name = edge.get("to")

                if (
                    from_node_name in nodes_by_name
                    and to_node_name in nodes_by_name
                ):
                    from_node = nodes_by_name[from_node_name]
                    to_node = nodes_by_name[to_node_name]

                    # Check state -> observation mappings
                    if (
                        from_node.get("type") == "state"
                        and to_node.get("type") == "observation"
                    ):
                        from_dim = from_node.get("num_states")
                        to_dim = to_node.get("num_observations")

                        if from_dim and to_dim and from_dim != to_dim:
                            result.add_error(
                                "MathematicalValidator",
                                f"Dimension mismatch: state has {from_dim} dimensions but observation has {to_dim}",
                                context={
                                    "from_node": from_node_name,
                                    "to_node": to_node_name,
                                },
                            )

    def _validate_numerical_ranges(
        self, nodes: List[Dict[str, Any]], result: ValidationResult
    ):
        """Validate numerical ranges for all numeric fields."""
        for node in nodes:
            node_name = node.get("name", "unnamed")

            # Validate dimension fields
            dimension_fields = [
                "num_states",
                "num_observations",
                "num_actions",
            ]
            for field in dimension_fields:
                if field in node:
                    value = node[field]
                    if not isinstance(value, int) or value <= 0:
                        result.add_error(
                            "MathematicalValidator",
                            f"{field} must be a positive integer, got: {value}",
                            node_name=node_name,
                        )
                    elif value > self.max_dimension:
                        result.add_error(
                            "MathematicalValidator",
                            f"{field} exceeds maximum allowed dimension ({self.max_dimension}): {value}",
                            node_name=node_name,
                        )

            # Validate constraint fields
            constraints = node.get("constraints", {})
            if isinstance(constraints, dict):
                # Precision must be positive
                if "precision" in constraints:
                    precision = constraints["precision"]
                    if (
                        not isinstance(precision, (int, float))
                        or precision <= 0
                    ):
                        result.add_error(
                            "MathematicalValidator",
                            f"Precision must be positive, got: {precision}",
                            node_name=node_name,
                        )

                # Entropy constraints
                min_entropy = constraints.get("min_entropy")
                max_entropy = constraints.get("max_entropy")
                if min_entropy is not None and max_entropy is not None:
                    if min_entropy > max_entropy:
                        result.add_error(
                            "MathematicalValidator",
                            f"Conflicting entropy constraints: min_entropy ({min_entropy}) > max_entropy ({max_entropy})",
                            node_name=node_name,
                        )

    def _validate_matrix_constraints(
        self, nodes: List[Dict[str, Any]], result: ValidationResult
    ):
        """Validate matrix constraints for transition and likelihood matrices."""
        for node in nodes:
            node_name = node.get("name", "unnamed")

            # Validate transition matrices
            if node.get("type") == "transition" and "matrix" in node:
                matrix = node["matrix"]
                try:
                    mat = np.array(matrix, dtype=float)

                    # Check dimensions
                    if len(mat.shape) != 2:
                        result.add_error(
                            "MathematicalValidator",
                            f"Transition matrix must be 2-dimensional",
                            node_name=node_name,
                        )
                        continue

                    # Check that columns sum to 1 (stochastic matrix)
                    col_sums = np.sum(mat, axis=0)
                    if not np.allclose(col_sums, 1.0, atol=self.tolerance):
                        result.add_error(
                            "MathematicalValidator",
                            f"Transition matrix columns must sum to 1, got sums: {col_sums}",
                            node_name=node_name,
                        )

                    # Check non-negative values
                    if np.any(mat < 0):
                        result.add_error(
                            "MathematicalValidator",
                            f"Transition matrix contains negative values",
                            node_name=node_name,
                        )

                except (ValueError, TypeError) as e:
                    result.add_error(
                        "MathematicalValidator",
                        f"Invalid transition matrix format: {e}",
                        node_name=node_name,
                    )


class GMNTypeValidator:
    """Type validator for GMN specifications with comprehensive type checking."""

    def __init__(self):
        """Initialize the type validator."""
        self.valid_node_types = {nt.value for nt in GMNNodeType}
        self.valid_edge_types = {et.value for et in GMNEdgeType}

        self.required_node_attributes = {
            "state": ["num_states"],
            "observation": ["num_observations"],
            "action": ["num_actions"],
            "belief": ["about"],
            "preference": [],
            "transition": [],
            "likelihood": [],
        }

    def validate(self, spec: Dict[str, Any]) -> ValidationResult:
        """Validate GMN specification types with hard failures."""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=[], info=[]
        )

        nodes = spec.get("nodes", [])
        edges = spec.get("edges", [])

        # Validate node types
        self._validate_node_types(nodes, result)

        # Validate edge types
        self._validate_edge_types(edges, result)

        # Validate attribute types
        self._validate_attribute_types(nodes, result)

        # Hard failure on any errors
        if not result.is_valid:
            error_messages = [
                f"{error.validator}: {error.message}"
                for error in result.errors
            ]
            raise GMNValidationError(
                f"Type validation failed: {'; '.join(error_messages)}"
            )

        return result

    def _validate_node_types(
        self, nodes: List[Dict[str, Any]], result: ValidationResult
    ):
        """Validate node types and required attributes."""
        for node in nodes:
            node_name = node.get("name", "unnamed")
            node_type = node.get("type")

            # Check valid node type
            if node_type and node_type not in self.valid_node_types:
                result.add_error(
                    "TypeValidator",
                    f"Invalid node type: {node_type}",
                    node_name=node_name,
                )
                continue

            # Check required attributes for this node type
            if node_type in self.required_node_attributes:
                required_attrs = self.required_node_attributes[node_type]
                for attr in required_attrs:
                    if attr not in node:
                        result.add_error(
                            "TypeValidator",
                            f"{node_type.title()} node '{node_name}' missing required attribute: {attr}",
                            node_name=node_name,
                        )

    def _validate_edge_types(
        self, edges: List[Dict[str, Any]], result: ValidationResult
    ):
        """Validate edge types."""
        for i, edge in enumerate(edges):
            edge_type = edge.get("type")

            if edge_type and edge_type not in self.valid_edge_types:
                result.add_error(
                    "TypeValidator",
                    f"Invalid edge type: {edge_type}",
                    edge_index=i,
                )

    def _validate_attribute_types(
        self, nodes: List[Dict[str, Any]], result: ValidationResult
    ):
        """Validate attribute data types."""
        for node in nodes:
            node_name = node.get("name", "unnamed")

            # Validate integer fields
            int_fields = [
                "num_states",
                "num_observations",
                "num_actions",
                "preferred_observation",
            ]
            for field in int_fields:
                if field in node and not isinstance(node[field], int):
                    result.add_error(
                        "TypeValidator",
                        f"{field} must be an integer, got {type(node[field]).__name__}: {node[field]}",
                        node_name=node_name,
                    )

            # Validate float fields
            float_fields = ["preference_strength"]
            for field in float_fields:
                if field in node and not isinstance(node[field], (int, float)):
                    result.add_error(
                        "TypeValidator",
                        f"{field} must be a number, got {type(node[field]).__name__}: {node[field]}",
                        node_name=node_name,
                    )

            # Validate list fields
            list_fields = ["initial_distribution"]
            for field in list_fields:
                if field in node and not isinstance(node[field], list):
                    result.add_error(
                        "TypeValidator",
                        f"{field} must be a list, got {type(node[field]).__name__}",
                        node_name=node_name,
                    )

            # Validate string fields
            string_fields = ["about"]
            for field in string_fields:
                if field in node and not isinstance(node[field], str):
                    result.add_error(
                        "TypeValidator",
                        f"{field} must be a string, got {type(node[field]).__name__}: {node[field]}",
                        node_name=node_name,
                    )


class GMNConstraintValidator:
    """Constraint validator for GMN specifications with business rule enforcement."""

    def __init__(self):
        """Initialize the constraint validator."""
        self.max_action_space = (
            100000  # Business rule: reasonable action space limit
        )
        self.max_state_space = (
            100000  # Business rule: reasonable state space limit
        )
        self.max_observation_space = (
            100000  # Business rule: reasonable observation space limit
        )

    def validate(self, spec: Dict[str, Any]) -> ValidationResult:
        """Validate GMN specification constraints with hard failures."""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=[], info=[]
        )

        nodes = spec.get("nodes", [])

        # Validate business rules
        self._validate_business_rules(nodes, result)

        # Validate preference constraints
        self._validate_preference_constraints(nodes, result)

        # Validate constraint consistency
        self._validate_constraint_consistency(nodes, result)

        # Hard failure on any errors
        if not result.is_valid:
            error_messages = [
                f"{error.validator}: {error.message}"
                for error in result.errors
            ]
            raise GMNValidationError(
                f"Constraint validation failed: {'; '.join(error_messages)}"
            )

        return result

    def _validate_business_rules(
        self, nodes: List[Dict[str, Any]], result: ValidationResult
    ):
        """Validate business rules for space sizes."""
        for node in nodes:
            node_name = node.get("name", "unnamed")

            # Action space limits
            if node.get("type") == "action" and "num_actions" in node:
                num_actions = node["num_actions"]
                if num_actions > self.max_action_space:
                    result.add_error(
                        "ConstraintValidator",
                        f"Action space too large: {num_actions} exceeds maximum {self.max_action_space}",
                        node_name=node_name,
                    )

            # State space limits
            if node.get("type") == "state" and "num_states" in node:
                num_states = node["num_states"]
                if num_states > self.max_state_space:
                    result.add_error(
                        "ConstraintValidator",
                        f"State space too large: {num_states} exceeds maximum {self.max_state_space}",
                        node_name=node_name,
                    )

            # Observation space limits
            if (
                node.get("type") == "observation"
                and "num_observations" in node
            ):
                num_observations = node["num_observations"]
                if num_observations > self.max_observation_space:
                    result.add_error(
                        "ConstraintValidator",
                        f"Observation space too large: {num_observations} exceeds maximum {self.max_observation_space}",
                        node_name=node_name,
                    )

    def _validate_preference_constraints(
        self, nodes: List[Dict[str, Any]], result: ValidationResult
    ):
        """Validate preference constraints."""
        # Build observation space lookup
        obs_spaces = {}
        for node in nodes:
            if (
                node.get("type") == "observation"
                and "name" in node
                and "num_observations" in node
            ):
                obs_spaces[node["name"]] = node["num_observations"]

        for node in nodes:
            node_name = node.get("name", "unnamed")

            if (
                node.get("type") == "preference"
                and "preferred_observation" in node
            ):
                preferred_obs = node["preferred_observation"]

                # Find associated observation space
                # This is a simplified check - in practice we'd need to trace the graph
                max_obs = max(obs_spaces.values()) if obs_spaces else 0

                if isinstance(preferred_obs, int) and preferred_obs >= max_obs:
                    result.add_error(
                        "ConstraintValidator",
                        f"Preferred observation index {preferred_obs} out of range (max: {max_obs - 1})",
                        node_name=node_name,
                    )

    def _validate_constraint_consistency(
        self, nodes: List[Dict[str, Any]], result: ValidationResult
    ):
        """Validate internal constraint consistency."""
        for node in nodes:
            node_name = node.get("name", "unnamed")
            constraints = node.get("constraints", {})

            if isinstance(constraints, dict):
                # Check entropy constraints
                min_entropy = constraints.get("min_entropy")
                max_entropy = constraints.get("max_entropy")

                if min_entropy is not None and max_entropy is not None:
                    if min_entropy > max_entropy:
                        result.add_error(
                            "ConstraintValidator",
                            f"Conflicting entropy constraints: min_entropy ({min_entropy}) > max_entropy ({max_entropy})",
                            node_name=node_name,
                        )


class GMNValidationFramework:
    """Comprehensive validation framework integrating all validators."""

    def __init__(self):
        """Initialize the validation framework."""
        self.syntax_validator = GMNSyntaxValidator()
        self.semantic_validator = GMNSemanticValidator()
        self.mathematical_validator = GMNMathematicalValidator()
        self.type_validator = GMNTypeValidator()
        self.constraint_validator = GMNConstraintValidator()

    def validate(self, spec: Dict[str, Any]) -> ValidationResult:
        """Run comprehensive validation with hard failures on any violation."""
        # Collect all errors from all validators
        all_errors = []
        all_warnings = []
        all_info = []

        validators = [
            ("Syntax", self.syntax_validator),
            ("Semantic", self.semantic_validator),
            ("Mathematical", self.mathematical_validator),
            ("Type", self.type_validator),
            ("Constraint", self.constraint_validator),
        ]

        for validator_name, validator in validators:
            try:
                result = validator.validate(spec)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                all_info.extend(result.info)
            except GMNValidationError as e:
                # Convert validator exceptions to errors
                error = ValidationError(
                    validator=f"{validator_name}Validator",
                    level=ValidationLevel.ERROR,
                    message=str(e),
                )
                all_errors.append(error)

        # Create comprehensive result
        result = ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            info=all_info,
        )

        return result

    def validate_with_reality_checks(
        self, spec: Dict[str, Any]
    ) -> ValidationResult:
        """Validate with additional reality checks for suspicious patterns."""
        # First run standard validation
        result = self.validate(spec)

        # Add reality checks
        nodes = spec.get("nodes", [])
        self._add_reality_checks(nodes, result)

        return result

    def _add_reality_checks(
        self, nodes: List[Dict[str, Any]], result: ValidationResult
    ):
        """Add reality checks to catch suspicious patterns."""
        state_dims = []
        obs_dims = []

        for node in nodes:
            if node.get("type") == "state" and "num_states" in node:
                state_dims.append(node["num_states"])
            elif (
                node.get("type") == "observation"
                and "num_observations" in node
            ):
                obs_dims.append(node["num_observations"])

        # Check for suspicious dimension ratios
        if state_dims and obs_dims:
            min_state = min(state_dims)
            max_obs = max(obs_dims)

            if (
                max_obs / min_state > 100
            ):  # Observation space 100x larger than state space
                result.add_error(
                    "RealityCheckValidator",
                    f"Suspicious dimension ratio: observation space ({max_obs}) >> state space ({min_state})",
                )

        # Check for trivial state spaces
        if any(dim == 1 for dim in state_dims):
            result.add_warning(
                "RealityCheckValidator",
                "Trivial state space detected (dimension = 1)",
            )

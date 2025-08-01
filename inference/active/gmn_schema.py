"""GMN Schema and Validation Models.

This module provides comprehensive Pydantic models for GMN (Generative Model Network)
specifications with built-in validation, probability distribution checking, and
mathematical consistency verification.

Following Clean Architecture principles:
- Domain models with embedded business rules
- Pure functions for mathematical validation
- Clear separation of concerns
- Type safety with Pydantic
"""

from __future__ import annotations

import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from inference.active.gmn_parser import GMNGraph
from inference.active.gmn_parser import GMNNodeType as ParserGMNNodeType


class GMNValidationError(Exception):
    """Raised when GMN validation fails."""

    pass


class GMNNodeType(str, Enum):
    """Types of nodes in GMN specification."""

    STATE = "state"
    OBSERVATION = "observation"
    ACTION = "action"
    BELIEF = "belief"
    PREFERENCE = "preference"
    TRANSITION = "transition"
    LIKELIHOOD = "likelihood"
    POLICY = "policy"
    LLM_QUERY = "llm_query"


class GMNEdgeType(str, Enum):
    """Types of edges in GMN specification."""

    DEPENDS_ON = "depends_on"
    INFLUENCES = "influences"
    UPDATES = "updates"
    QUERIES = "queries"
    GENERATES = "generates"


class ProbabilityDistribution(BaseModel):
    """A probability distribution with validation.

    Ensures mathematical consistency:
    - All values are non-negative
    - Values sum to 1.0 (within numerical tolerance)
    - Non-empty distribution
    """

    values: List[float] = Field(..., min_length=1)
    tolerance: float = Field(default=1e-6, description="Numerical tolerance for sum validation")
    auto_normalize: bool = Field(default=False, description="Automatically normalize if sum != 1.0")

    def __init__(self, values: Union[List[float], np.ndarray], **kwargs):
        """Initialize probability distribution.

        Args:
            values: Probability values as list or numpy array
            **kwargs: Additional configuration options
        """
        if isinstance(values, np.ndarray):
            values = values.tolist()

        # Auto-normalize if requested
        if kwargs.get("auto_normalize", False):
            total = sum(values)
            if total > 0:
                values = [v / total for v in values]

        super().__init__(values=values, **kwargs)

    @field_validator("values")
    @classmethod
    def validate_values(cls, v):
        """Validate probability values."""
        if not v:
            raise GMNValidationError("Probability distribution cannot be empty")

        # Check for non-negative values
        if any(val < 0 for val in v):
            raise GMNValidationError("All probabilities must be non-negative")

        return v

    @model_validator(mode="after")
    def validate_sum(self):
        """Validate that probabilities sum to 1.0."""
        if not self.values:
            return self

        total = sum(self.values)
        if abs(total - 1.0) > self.tolerance:
            raise GMNValidationError(
                f"Probabilities must sum to 1.0 (got {total:.6f}, tolerance={self.tolerance})"
            )

        return self

    def is_valid(self) -> bool:
        """Check if the distribution is valid."""
        try:
            total = sum(self.values)
            return (
                len(self.values) > 0
                and all(v >= 0 for v in self.values)
                and abs(total - 1.0) <= self.tolerance
            )
        except Exception:
            return False

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.values)

    def normalize(self) -> ProbabilityDistribution:
        """Return normalized version of this distribution."""
        total = sum(self.values)
        if total == 0:
            # Uniform distribution
            normalized_values = [1.0 / len(self.values)] * len(self.values)
        else:
            normalized_values = [v / total for v in self.values]

        return ProbabilityDistribution(
            normalized_values, tolerance=self.tolerance, auto_normalize=False
        )


class GMNNode(BaseModel):
    """A node in a GMN specification."""

    id: str = Field(..., description="Unique identifier for the node")
    type: GMNNodeType = Field(..., description="Type of the node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        """Validate node ID."""
        if not v or not v.strip():
            raise GMNValidationError("Node ID cannot be empty")

        # Check for valid identifier characters (alphanumeric + underscore)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise GMNValidationError(
                f"Node ID contains invalid characters. Use alphanumeric and underscore only: {v}"
            )

        return v.strip()

    @model_validator(mode="after")
    def validate_node_properties(self):
        """Validate node properties based on type."""
        if self.type == GMNNodeType.STATE:
            if "num_states" not in self.properties:
                self.properties["num_states"] = 1
            elif self.properties["num_states"] < 1:
                raise GMNValidationError("State nodes must have num_states >= 1")

        elif self.type == GMNNodeType.OBSERVATION:
            if "num_observations" not in self.properties:
                self.properties["num_observations"] = 1
            elif self.properties["num_observations"] < 1:
                raise GMNValidationError("Observation nodes must have num_observations >= 1")

        elif self.type == GMNNodeType.ACTION:
            if "num_actions" not in self.properties:
                self.properties["num_actions"] = 1
            elif self.properties["num_actions"] < 1:
                raise GMNValidationError("Action nodes must have num_actions >= 1")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties,
            "metadata": self.metadata,
        }


class GMNEdge(BaseModel):
    """An edge in a GMN specification."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: GMNEdgeType = Field(..., description="Type of the edge")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge properties")

    @field_validator("source")
    @classmethod
    def validate_source(cls, v):
        """Validate edge source."""
        if not v or not v.strip():
            raise GMNValidationError("Edge source cannot be empty")
        return v.strip()

    @field_validator("target")
    @classmethod
    def validate_target(cls, v):
        """Validate edge target."""
        if not v or not v.strip():
            raise GMNValidationError("Edge target cannot be empty")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "properties": self.properties,
        }


class GMNSpecification(BaseModel):
    """Complete GMN specification with validation."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique specification ID"
    )
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Optional description")
    version: str = Field(default="1.0.0", description="Specification version")

    nodes: List[GMNNode] = Field(..., min_length=1, description="List of nodes")
    edges: List[GMNEdge] = Field(default_factory=list, description="List of edges")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate specification name."""
        if not v or not v.strip():
            raise GMNValidationError("Specification name cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_specification_structure(self):
        """Validate overall GMN specification structure."""
        if not self.nodes:
            raise GMNValidationError("GMN specification must contain at least one node")

        # Check for required node types
        node_types = {node.type for node in self.nodes}

        if GMNNodeType.STATE not in node_types:
            raise GMNValidationError("GMN specification must contain at least one STATE node")

        if GMNNodeType.OBSERVATION not in node_types and GMNNodeType.ACTION not in node_types:
            raise GMNValidationError(
                "GMN specification must contain at least one OBSERVATION or ACTION node"
            )

        # Validate edge references
        node_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                raise GMNValidationError(f"Edge references undefined node: {edge.source}")
            if edge.target not in node_ids:
                raise GMNValidationError(f"Edge references undefined node: {edge.target}")

        return self

    def get_node_by_id(self, node_id: str) -> Optional[GMNNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_nodes_by_type(self, node_type: GMNNodeType) -> List[GMNNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes if node.type == node_type]

    def get_edges_from_node(self, node_id: str) -> List[GMNEdge]:
        """Get all edges originating from a node."""
        return [edge for edge in self.edges if edge.source == node_id]

    def get_edges_to_node(self, node_id: str) -> List[GMNEdge]:
        """Get all edges targeting a node."""
        return [edge for edge in self.edges if edge.target == node_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GMNSpecification:
        """Create GMN specification from dictionary."""
        # Convert node dictionaries to GMNNode objects
        nodes = []
        for node_data in data.get("nodes", []):
            node = GMNNode(
                id=node_data["id"],
                type=GMNNodeType(node_data["type"]),
                properties=node_data.get("properties", {}),
                metadata=node_data.get("metadata", {}),
            )
            nodes.append(node)

        # Convert edge dictionaries to GMNEdge objects
        edges = []
        for edge_data in data.get("edges", []):
            edge = GMNEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                type=GMNEdgeType(edge_data["type"]),
                properties=edge_data.get("properties", {}),
            )
            edges.append(edge)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description"),
            version=data.get("version", "1.0.0"),
            nodes=nodes,
            edges=edges,
            metadata=data.get("metadata", {}),
        )


class GMNSchemaValidator:
    """Validator for GMN specifications and probability matrices."""

    def __init__(self, tolerance: float = 1e-6):
        """Initialize validator.

        Args:
            tolerance: Numerical tolerance for probability validation
        """
        self.tolerance = tolerance

    def validate_specification(self, spec: Optional[GMNSpecification]) -> Tuple[bool, List[str]]:
        """Validate complete GMN specification.

        Args:
            spec: GMN specification to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if spec is None:
            return False, ["Specification cannot be None"]

        errors = []

        try:
            # Basic structure validation (already done by Pydantic)

            # Check for circular dependencies
            circular_errors = self._check_circular_dependencies(spec)
            errors.extend(circular_errors)

            # Validate mathematical consistency
            math_errors = self._validate_mathematical_consistency(spec)
            errors.extend(math_errors)

            # Validate probability distributions in node properties
            prob_errors = self._validate_node_probability_distributions(spec)
            errors.extend(prob_errors)

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return len(errors) == 0, errors

    def validate_probability_matrix(
        self, matrix: np.ndarray, matrix_name: str
    ) -> Tuple[bool, List[str]]:
        """Validate probability matrix (A, B, C, D matrices).

        Args:
            matrix: Numpy array representing probability matrix
            matrix_name: Name of the matrix for error reporting

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if matrix is None:
            return False, [f"{matrix_name} matrix cannot be None"]

        try:
            # Check for non-negative values
            if np.any(matrix < 0):
                errors.append(f"{matrix_name} matrix contains negative values")

            # Check for NaN or infinite values
            if np.any(np.isnan(matrix)):
                errors.append(f"{matrix_name} matrix contains NaN values")
            if np.any(np.isinf(matrix)):
                errors.append(f"{matrix_name} matrix contains infinite values")

            # Check probability constraints based on matrix type
            if matrix_name.upper() in ["A", "LIKELIHOOD"]:
                # A matrices: columns should sum to 1 (for each state)
                if len(matrix.shape) >= 2:
                    for dim_idx in range(matrix.shape[1]):
                        if len(matrix.shape) == 2:
                            col_sum = np.sum(matrix[:, dim_idx])
                        else:
                            # Multi-dimensional case
                            col_sum = np.sum(matrix[:, dim_idx, ...])

                        if abs(col_sum - 1.0) > self.tolerance:
                            errors.append(
                                f"{matrix_name} matrix column {dim_idx} does not sum to 1.0 "
                                f"(sum={col_sum:.6f})"
                            )

            elif matrix_name.upper() in ["B", "TRANSITION"]:
                # B matrices: for each action, columns should sum to 1
                if len(matrix.shape) >= 3:
                    for action_idx in range(matrix.shape[2]):
                        for state_idx in range(matrix.shape[1]):
                            col_sum = np.sum(matrix[:, state_idx, action_idx])
                            if abs(col_sum - 1.0) > self.tolerance:
                                errors.append(
                                    f"{matrix_name} matrix action {action_idx}, state {state_idx} "
                                    f"does not sum to 1.0 (sum={col_sum:.6f})"
                                )

            elif matrix_name.upper() in ["D", "BELIEF", "INITIAL"]:
                # D vectors: should sum to 1
                total_sum = np.sum(matrix)
                if abs(total_sum - 1.0) > self.tolerance:
                    errors.append(f"{matrix_name} vector does not sum to 1.0 (sum={total_sum:.6f})")

        except Exception as e:
            errors.append(f"Error validating {matrix_name} matrix: {str(e)}")

        return len(errors) == 0, errors

    def validate_parser_compatibility(self, parser_graph: GMNGraph) -> bool:
        """Validate compatibility with existing GMN parser output.

        Args:
            parser_graph: GMN graph from existing parser

        Returns:
            True if compatible, False otherwise
        """
        try:
            # Check if we can convert parser graph to our schema format
            nodes = []
            for node_id, parser_node in parser_graph.nodes.items():
                # Convert parser node type to our node type
                node_type = self._convert_parser_node_type(parser_node.type)
                if node_type:
                    nodes.append(
                        GMNNode(
                            id=node_id,
                            type=node_type,
                            properties=parser_node.properties,
                            metadata=parser_node.metadata,
                        )
                    )

            edges = []
            for parser_edge in parser_graph.edges:
                # Convert parser edge type to our edge type
                edge_type = self._convert_parser_edge_type(parser_edge.type)
                if edge_type:
                    edges.append(
                        GMNEdge(
                            source=parser_edge.source,
                            target=parser_edge.target,
                            type=edge_type,
                            properties=parser_edge.properties,
                        )
                    )

            # Try to create specification
            spec = GMNSpecification(name="compatibility_test", nodes=nodes, edges=edges)

            # Validate the converted specification
            is_valid, _ = self.validate_specification(spec)
            return is_valid

        except Exception:
            return False

    def _check_circular_dependencies(self, spec: GMNSpecification) -> List[str]:
        """Check for circular dependencies in the specification."""
        errors = []

        try:
            # Build adjacency list
            graph = {}
            for edge in spec.edges:
                if edge.source not in graph:
                    graph[edge.source] = []
                graph[edge.source].append(edge.target)

            # DFS to detect cycles
            visited = set()
            rec_stack = set()

            def has_cycle(node):
                visited.add(node)
                rec_stack.add(node)

                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True

                rec_stack.remove(node)
                return False

            # Check each unvisited node
            for node in graph:
                if node not in visited:
                    if has_cycle(node):
                        errors.append(f"Circular dependency detected involving node: {node}")
                        break

        except Exception as e:
            errors.append(f"Error checking circular dependencies: {str(e)}")

        return errors

    def _validate_mathematical_consistency(self, spec: GMNSpecification) -> List[str]:
        """Validate mathematical consistency between connected nodes."""
        errors = []

        try:
            # Check dimension consistency between connected nodes
            for edge in spec.edges:
                source_node = spec.get_node_by_id(edge.source)
                target_node = spec.get_node_by_id(edge.target)

                if not source_node or not target_node:
                    continue

                # Check specific relationships
                if (
                    source_node.type == GMNNodeType.STATE
                    and target_node.type == GMNNodeType.LIKELIHOOD
                    and edge.type == GMNEdgeType.DEPENDS_ON
                ):
                    # Find observation node connected to likelihood
                    obs_edges = spec.get_edges_from_node(target_node.id)
                    for obs_edge in obs_edges:
                        if obs_edge.type == GMNEdgeType.GENERATES:
                            obs_node = spec.get_node_by_id(obs_edge.target)
                            if obs_node and obs_node.type == GMNNodeType.OBSERVATION:
                                # Check A matrix dimensions
                                if "matrix" in target_node.properties:
                                    matrix = np.array(target_node.properties["matrix"])
                                    expected_obs = obs_node.properties.get("num_observations", 1)
                                    expected_states = source_node.properties.get("num_states", 1)

                                    if matrix.shape[0] != expected_obs:
                                        errors.append(
                                            f"Likelihood matrix dimension mismatch: "
                                            f"expected {expected_obs} observations, got {matrix.shape[0]}"
                                        )

        except Exception as e:
            errors.append(f"Error validating mathematical consistency: {str(e)}")

        return errors

    def _validate_node_probability_distributions(self, spec: GMNSpecification) -> List[str]:
        """Validate probability distributions embedded in node properties."""
        errors = []

        for node in spec.nodes:
            try:
                # Check for distribution properties
                if "distribution" in node.properties:
                    dist_data = node.properties["distribution"]
                    if isinstance(dist_data, (list, np.ndarray)):
                        try:
                            dist = ProbabilityDistribution(dist_data)
                            if not dist.is_valid():
                                errors.append(f"Invalid probability distribution in node {node.id}")
                        except GMNValidationError as e:
                            errors.append(f"Node {node.id} distribution error: {str(e)}")

                # Check matrix properties
                if "matrix" in node.properties:
                    matrix = np.array(node.properties["matrix"])
                    matrix_valid, matrix_errors = self.validate_probability_matrix(
                        matrix, f"node_{node.id}"
                    )
                    if not matrix_valid:
                        errors.extend(matrix_errors)

            except Exception as e:
                errors.append(f"Error validating node {node.id}: {str(e)}")

        return errors

    def _convert_parser_node_type(self, parser_type: ParserGMNNodeType) -> Optional[GMNNodeType]:
        """Convert parser node type to schema node type."""
        mapping = {
            ParserGMNNodeType.STATE: GMNNodeType.STATE,
            ParserGMNNodeType.OBSERVATION: GMNNodeType.OBSERVATION,
            ParserGMNNodeType.ACTION: GMNNodeType.ACTION,
            ParserGMNNodeType.BELIEF: GMNNodeType.BELIEF,
            ParserGMNNodeType.PREFERENCE: GMNNodeType.PREFERENCE,
            ParserGMNNodeType.TRANSITION: GMNNodeType.TRANSITION,
            ParserGMNNodeType.LIKELIHOOD: GMNNodeType.LIKELIHOOD,
            ParserGMNNodeType.POLICY: GMNNodeType.POLICY,
            ParserGMNNodeType.LLM_QUERY: GMNNodeType.LLM_QUERY,
        }
        return mapping.get(parser_type)

    def _convert_parser_edge_type(self, parser_type) -> Optional[GMNEdgeType]:
        """Convert parser edge type to schema edge type."""
        # Import here to avoid circular imports
        from inference.active.gmn_parser import GMNEdgeType as ParserGMNEdgeType

        mapping = {
            ParserGMNEdgeType.DEPENDS_ON: GMNEdgeType.DEPENDS_ON,
            ParserGMNEdgeType.INFLUENCES: GMNEdgeType.INFLUENCES,
            ParserGMNEdgeType.UPDATES: GMNEdgeType.UPDATES,
            ParserGMNEdgeType.QUERIES: GMNEdgeType.QUERIES,
            ParserGMNEdgeType.GENERATES: GMNEdgeType.GENERATES,
        }
        return mapping.get(parser_type)


# Utility functions for working with GMN specifications


def create_minimal_gmn(
    name: str, num_states: int = 2, num_observations: int = 2, num_actions: int = 2
) -> GMNSpecification:
    """Create a minimal valid GMN specification for testing."""
    nodes = [
        GMNNode(id="state", type=GMNNodeType.STATE, properties={"num_states": num_states}),
        GMNNode(
            id="observation",
            type=GMNNodeType.OBSERVATION,
            properties={"num_observations": num_observations},
        ),
        GMNNode(id="action", type=GMNNodeType.ACTION, properties={"num_actions": num_actions}),
    ]

    edges = [GMNEdge(source="state", target="observation", type=GMNEdgeType.GENERATES)]

    return GMNSpecification(
        name=name, description="Minimal GMN specification for testing", nodes=nodes, edges=edges
    )


def validate_gmn_specification(spec_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Convenience function to validate GMN specification from dictionary.

    Args:
        spec_dict: Dictionary representation of GMN specification

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    try:
        spec = GMNSpecification.from_dict(spec_dict)
        validator = GMNSchemaValidator()
        return validator.validate_specification(spec)
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]

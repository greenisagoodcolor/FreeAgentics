"""
GMN (Generalized Notation Notation) parser for PyMDP model specification.

This parser enables specification of Active Inference models using GMN notation
and integrates with LLM calls for dynamic model generation and updates.
"""

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class GMNValidationError(Exception):
    """Raised when GMN validation fails."""


class GMNNodeType(Enum):
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


class GMNEdgeType(Enum):
    """Types of edges in GMN specification."""

    DEPENDS_ON = "depends_on"
    INFLUENCES = "influences"
    UPDATES = "updates"
    QUERIES = "queries"
    GENERATES = "generates"


@dataclass
class GMNNode:
    """Node in GMN specification."""

    id: str
    type: GMNNodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GMNEdge:
    """Edge in GMN specification."""

    source: str
    target: str
    type: GMNEdgeType
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GMNGraph:
    """Complete GMN specification graph."""

    nodes: Dict[str, GMNNode] = field(default_factory=dict)
    edges: List[GMNEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GMNSchemaValidator:
    """Validates GMN schema for correctness."""

    @staticmethod
    def validate(gmn_dict: Dict) -> Tuple[bool, List[str]]:
        """Validate GMN schema.

        Args:
            gmn_dict: Dictionary containing GMN specification

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        if "nodes" not in gmn_dict:
            errors.append("Missing required field: nodes")
        if "edges" not in gmn_dict:
            errors.append("Missing required field: edges")

        # Validate nodes
        if "nodes" in gmn_dict:
            if not isinstance(gmn_dict["nodes"], list):
                errors.append("'nodes' must be a list")
            else:
                for i, node in enumerate(gmn_dict["nodes"]):
                    if not isinstance(node, dict):
                        errors.append(f"Node {i} must be a dictionary")
                    elif "id" not in node:
                        errors.append(f"Node {i} missing required field: id")

        # Validate edges
        if "edges" in gmn_dict:
            if not isinstance(gmn_dict["edges"], list):
                errors.append("'edges' must be a list")
            else:
                for i, edge in enumerate(gmn_dict["edges"]):
                    if not isinstance(edge, dict):
                        errors.append(f"Edge {i} must be a dictionary")
                    else:
                        if "source" not in edge:
                            errors.append(f"Edge {i} missing required field: source")
                        if "target" not in edge:
                            errors.append(f"Edge {i} missing required field: target")

        return (len(errors) == 0, errors)


class GMNParser:
    """Parser for GMN specifications to PyMDP models."""

    def __init__(self):
        """Initialize the GMN parser."""
        self.current_graph: Optional[GMNGraph] = None
        self.llm_integration_points: List[Dict[str, Any]] = []
        self.validation_errors: List[str] = []

    def parse(self, gmn_spec: Union[str, Dict[str, Any]]) -> GMNGraph:
        """
        Parse GMN specification into graph structure.

        Args:
            gmn_spec: GMN specification as string or dict

        Returns:
            Parsed GMN graph
        """
        self.validation_errors = []

        # Parse specification
        if isinstance(gmn_spec, str):
            spec_dict = self._parse_string_spec(gmn_spec)
        else:
            spec_dict = gmn_spec

        # Create graph
        graph = GMNGraph()

        # Parse nodes
        if "nodes" in spec_dict:
            for node_spec in spec_dict["nodes"]:
                node = self._parse_node(node_spec)
                graph.nodes[node.id] = node

        # Parse edges
        if "edges" in spec_dict:
            for edge_spec in spec_dict["edges"]:
                edge = self._parse_edge(edge_spec)
                graph.edges.append(edge)

        # Parse metadata
        if "metadata" in spec_dict:
            graph.metadata = spec_dict["metadata"]

        # Validate graph
        self._validate_graph(graph)

        if self.validation_errors:
            error_msg = "GMN validation errors:\n" + "\n".join(self.validation_errors)
            raise ValueError(error_msg)

        self.current_graph = graph
        return graph

    def to_pymdp_model(self, graph: GMNGraph) -> Dict[str, Any]:
        """
        Convert GMN graph to PyMDP model specification.

        Args:
            graph: GMN graph

        Returns:
            PyMDP model specification
        """
        model_spec: Dict[str, Any] = {
            "num_states": [],
            "num_obs": [],
            "num_actions": [],
            "A": [],  # Observation model
            "B": [],  # Transition model
            "C": [],  # Preferences
            "D": [],  # Initial beliefs
            "llm_integration": [],
        }

        # Extract states
        state_nodes = [n for n in graph.nodes.values() if n.type == GMNNodeType.STATE]
        for state in state_nodes:
            num_states = state.properties.get("num_states", 1)
            model_spec["num_states"].append(num_states)

        # Extract observations
        obs_nodes = [n for n in graph.nodes.values() if n.type == GMNNodeType.OBSERVATION]
        for obs in obs_nodes:
            num_obs = obs.properties.get("num_observations", 1)
            model_spec["num_obs"].append(num_obs)

        # Extract actions
        action_nodes = [n for n in graph.nodes.values() if n.type == GMNNodeType.ACTION]
        for action in action_nodes:
            num_actions = action.properties.get("num_actions", 1)
            model_spec["num_actions"].append(num_actions)

        # Build likelihood model (A)
        likelihood_nodes = [n for n in graph.nodes.values() if n.type == GMNNodeType.LIKELIHOOD]
        for likelihood in likelihood_nodes:
            A_matrix = self._build_likelihood_matrix(likelihood, graph)
            model_spec["A"].append(A_matrix)

        # Build transition model (B)
        transition_nodes = [n for n in graph.nodes.values() if n.type == GMNNodeType.TRANSITION]
        for transition in transition_nodes:
            B_matrix = self._build_transition_matrix(transition, graph)
            model_spec["B"].append(B_matrix)

        # Build preference model (C)
        pref_nodes = [n for n in graph.nodes.values() if n.type == GMNNodeType.PREFERENCE]
        for pref in pref_nodes:
            C_vector = self._build_preference_vector(pref, graph)
            model_spec["C"].append(C_vector)

        # Build initial beliefs (D)
        belief_nodes = [n for n in graph.nodes.values() if n.type == GMNNodeType.BELIEF]
        for belief in belief_nodes:
            D_vector = self._build_belief_vector(belief, graph)
            model_spec["D"].append(D_vector)

        # Extract LLM integration points
        llm_nodes = [n for n in graph.nodes.values() if n.type == GMNNodeType.LLM_QUERY]
        for llm_node in llm_nodes:
            integration = self._build_llm_integration(llm_node, graph)
            model_spec["llm_integration"].append(integration)

        return model_spec

    def _parse_string_spec(self, spec_str: str) -> Dict[str, Any]:
        """Parse string specification to dictionary."""
        # Try JSON first
        try:
            return json.loads(spec_str)
        except json.JSONDecodeError:
            pass

        # Try custom GMN format
        return self._parse_gmn_format(spec_str)

    def _parse_gmn_format(self, spec_str: str) -> Dict[str, Any]:
        """Parse custom GMN format."""
        spec_dict: Dict[str, Any] = {"nodes": [], "edges": []}

        lines = spec_str.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Section headers
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].lower()
                continue

            # Parse based on section
            if current_section == "nodes":
                node_match = re.match(r"(\w+)\s*:\s*(\w+)\s*(\{.*\})?", line)
                if node_match:
                    node_id, node_type, props_str = node_match.groups()
                    node_spec = {"id": node_id, "type": node_type}
                    if props_str:
                        # Parse properties (safer parsing)
                        try:
                            # Try ast.literal_eval first
                            node_spec["properties"] = ast.literal_eval(props_str)
                        except (ValueError, SyntaxError):
                            # Fallback: simple key-value parsing
                            node_spec["properties"] = self._parse_simple_properties(props_str)
                    spec_dict["nodes"].append(node_spec)

            elif current_section == "edges":
                edge_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*(\w+)\s*(\{.*\})?", line)
                if edge_match:
                    source, target, edge_type, props_str = edge_match.groups()
                    edge_spec = {
                        "source": source,
                        "target": target,
                        "type": edge_type,
                    }
                    if props_str:
                        try:
                            edge_spec["properties"] = ast.literal_eval(props_str)
                        except (ValueError, SyntaxError):
                            edge_spec["properties"] = self._parse_simple_properties(props_str)
                    spec_dict["edges"].append(edge_spec)

        return spec_dict

    def _parse_simple_properties(self, props_str: str) -> Dict[str, Any]:
        """Parse simple key-value properties from string like {key: value, key2: "string"}."""
        properties: Dict[str, Any] = {}

        # Remove braces
        props_str = props_str.strip("{}")

        # Split by commas (simple approach)
        for item in props_str.split(","):
            if ":" in item:
                key, value = item.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Try to parse value
                if value.startswith('"') and value.endswith('"'):
                    # String value
                    properties[key] = value[1:-1]
                elif value.isdigit():
                    # Integer value
                    properties[key] = int(value)
                elif value.replace(".", "").isdigit():
                    # Float value
                    properties[key] = float(value)
                else:
                    # Keep as string
                    properties[key] = value

        return properties

    def _parse_node(self, node_spec: Dict[str, Any]) -> GMNNode:
        """Parse node specification."""
        node_id = node_spec.get("id", "")
        node_type_str = node_spec.get("type", "state")

        # Map string to enum
        try:
            node_type = GMNNodeType(node_type_str.lower())
        except ValueError:
            node_type = GMNNodeType.STATE
            self.validation_errors.append(f"Unknown node type: {node_type_str}")

        properties = node_spec.get("properties", {})
        metadata = node_spec.get("metadata", {})

        return GMNNode(
            id=node_id,
            type=node_type,
            properties=properties,
            metadata=metadata,
        )

    def _parse_edge(self, edge_spec: Dict[str, Any]) -> GMNEdge:
        """Parse edge specification."""
        source = edge_spec.get("source", "")
        target = edge_spec.get("target", "")
        edge_type_str = edge_spec.get("type", "depends_on")

        # Map string to enum
        try:
            edge_type = GMNEdgeType(edge_type_str.lower())
        except ValueError:
            edge_type = GMNEdgeType.DEPENDS_ON
            self.validation_errors.append(f"Unknown edge type: {edge_type_str}")

        properties = edge_spec.get("properties", {})

        return GMNEdge(source=source, target=target, type=edge_type, properties=properties)

    def _validate_graph(self, graph: GMNGraph) -> None:
        """Validate GMN graph structure."""
        # Check node references in edges
        node_ids = set(graph.nodes.keys())
        for edge in graph.edges:
            if edge.source not in node_ids:
                self.validation_errors.append(f"Edge source '{edge.source}' not found in nodes")
            if edge.target not in node_ids:
                self.validation_errors.append(f"Edge target '{edge.target}' not found in nodes")

        # Check required nodes
        has_state = any(n.type == GMNNodeType.STATE for n in graph.nodes.values())
        has_obs = any(n.type == GMNNodeType.OBSERVATION for n in graph.nodes.values())
        has_action = any(n.type == GMNNodeType.ACTION for n in graph.nodes.values())

        if not has_state:
            self.validation_errors.append("No state nodes found")
        if not has_obs:
            self.validation_errors.append("No observation nodes found")
        if not has_action:
            self.validation_errors.append("No action nodes found")

    def _build_likelihood_matrix(self, likelihood_node: GMNNode, graph: GMNGraph) -> np.ndarray:
        """Build likelihood matrix from node specification."""
        # Find connected state and observation nodes
        state_dims = []
        obs_dim = 1

        for edge in graph.edges:
            if edge.target == likelihood_node.id and edge.type == GMNEdgeType.DEPENDS_ON:
                source_node = graph.nodes.get(edge.source)
                if source_node and source_node.type == GMNNodeType.STATE:
                    state_dims.append(source_node.properties.get("num_states", 1))
            elif edge.source == likelihood_node.id and edge.type == GMNEdgeType.GENERATES:
                target_node = graph.nodes.get(edge.target)
                if target_node and target_node.type == GMNNodeType.OBSERVATION:
                    obs_dim = target_node.properties.get("num_observations", 1)

        # Build matrix
        if not state_dims:
            state_dims = [1]

        shape = [obs_dim] + state_dims

        # Check for custom matrix
        if "matrix" in likelihood_node.properties:
            return np.array(likelihood_node.properties["matrix"])

        # Generate default uniform likelihood
        A = np.ones(shape) / obs_dim
        return A

    def _build_transition_matrix(self, transition_node: GMNNode, graph: GMNGraph) -> np.ndarray:
        """Build transition matrix from node specification."""
        # Find connected state and action nodes
        state_dim = 1
        action_dim = 1

        for edge in graph.edges:
            if edge.source == transition_node.id or edge.target == transition_node.id:
                connected_node = graph.nodes.get(
                    edge.source if edge.target == transition_node.id else edge.target
                )
                if connected_node:
                    if connected_node.type == GMNNodeType.STATE:
                        state_dim = connected_node.properties.get("num_states", 1)
                    elif connected_node.type == GMNNodeType.ACTION:
                        action_dim = connected_node.properties.get("num_actions", 1)

        # Build matrix
        shape = [state_dim, state_dim, action_dim]

        # Check for custom matrix
        if "matrix" in transition_node.properties:
            return np.array(transition_node.properties["matrix"])

        # Generate default identity transitions
        B = np.zeros(shape)
        for a in range(action_dim):
            B[:, :, a] = np.eye(state_dim)
        return B

    def _build_preference_vector(self, pref_node: GMNNode, graph: GMNGraph) -> np.ndarray:
        """Build preference vector from node specification."""
        # Find connected observation node
        obs_dim = 1

        for edge in graph.edges:
            if edge.source == pref_node.id and edge.type == GMNEdgeType.DEPENDS_ON:
                target_node = graph.nodes.get(edge.target)
                if target_node and target_node.type == GMNNodeType.OBSERVATION:
                    obs_dim = target_node.properties.get("num_observations", 1)

        # Check for custom vector
        if "vector" in pref_node.properties:
            return np.array(pref_node.properties["vector"])

        # Generate default preferences
        C = np.zeros(obs_dim)
        preferred_obs = pref_node.properties.get("preferred_observation", 0)
        if 0 <= preferred_obs < obs_dim:
            C[preferred_obs] = 1.0
        return C

    def _build_belief_vector(self, belief_node: GMNNode, graph: GMNGraph) -> np.ndarray:
        """Build belief vector from node specification."""
        # Find connected state node
        state_dim = 1

        for edge in graph.edges:
            if edge.source == belief_node.id and edge.type == GMNEdgeType.DEPENDS_ON:
                target_node = graph.nodes.get(edge.target)
                if target_node and target_node.type == GMNNodeType.STATE:
                    state_dim = target_node.properties.get("num_states", 1)

        # Check for custom vector
        if "vector" in belief_node.properties:
            return np.array(belief_node.properties["vector"])

        # Generate default uniform belief
        D = np.ones(state_dim) / state_dim
        return D

    def _build_llm_integration(self, llm_node: GMNNode, graph: GMNGraph) -> Dict[str, Any]:
        """Build LLM integration specification."""
        integration = {
            "id": llm_node.id,
            "trigger_condition": llm_node.properties.get("trigger_condition", "on_observation"),
            "prompt_template": llm_node.properties.get("prompt_template", ""),
            "response_parser": llm_node.properties.get("response_parser", "json"),
            "update_targets": [],
            "context_nodes": [],
        }

        # Find update targets
        for edge in graph.edges:
            if edge.source == llm_node.id and edge.type == GMNEdgeType.UPDATES:
                integration["update_targets"].append(edge.target)

        # Find context nodes
        for edge in graph.edges:
            if edge.target == llm_node.id and edge.type == GMNEdgeType.QUERIES:
                integration["context_nodes"].append(edge.source)

        return integration


def parse_gmn_spec(spec: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to parse GMN specification to PyMDP model.

    Args:
        spec: GMN specification

    Returns:
        PyMDP model specification
    """
    parser = GMNParser()
    graph = parser.parse(spec)
    return parser.to_pymdp_model(graph)


# Example GMN specification format
EXAMPLE_GMN_SPEC = """
[nodes]
location: state {num_states: 4}
obs_location: observation {num_observations: 4}
move: action {num_actions: 5}
location_belief: belief
location_pref: preference {preferred_observation: 0}
location_likelihood: likelihood
location_transition: transition
llm_policy: llm_query {trigger_condition: "on_observation", prompt_template: "Given current location {obs}, suggest next action"}

[edges]
location -> location_likelihood: depends_on
location_likelihood -> obs_location: generates
location -> location_transition: depends_on
move -> location_transition: depends_on
location_pref -> obs_location: depends_on
location_belief -> location: depends_on
obs_location -> llm_policy: queries
llm_policy -> move: updates
"""

"""GMN Template Library for Agent Types.

This module provides a comprehensive template system for generating GMN (Generative
Model Network) specifications for different agent types. Templates provide structured,
parameterizable GMN generation with built-in validation and caching.

Templates supported:
- ExplorerTemplate: Spatial navigation and exploration agents
- AnalystTemplate: Data analysis and decision-making agents
- CreativeTemplate: Creative generation and exploration agents

Following Clean Architecture principles:
- Templates are pure domain objects with no external dependencies
- Template builders follow Template Method pattern with agent-specific overrides
- Factory pattern for template creation and registration
- Comprehensive caching for performance optimization
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inference.active.gmn_schema import GMNEdge, GMNEdgeType, GMNNode, GMNNodeType, GMNSpecification


class GMNTemplateError(Exception):
    """Raised when template generation or validation fails."""

    pass


class AgentType(str, Enum):
    """Supported agent types for template generation."""

    EXPLORER = "explorer"
    ANALYST = "analyst"
    CREATIVE = "creative"


class GMNTemplateBuilder(ABC):
    """Abstract base class for GMN template builders.

    Implements Template Method pattern where subclasses override specific
    node creation and configuration methods while sharing common structure.
    """

    def __init__(self) -> None:
        """Initialize template builder."""
        self._parameters: Dict[str, Any] = {}
        self._cache_key: Optional[str] = None

    def with_parameters(self, parameters: Dict[str, Any]) -> GMNTemplateBuilder:
        """Set parameters for template generation.

        Args:
            parameters: Dictionary of template parameters

        Returns:
            Self for fluent interface chaining

        Raises:
            GMNTemplateError: If parameters are invalid
        """
        # Validate parameters
        is_valid, errors = self.validate_parameters(parameters)
        if not is_valid:
            raise GMNTemplateError(f"Invalid parameters: {'; '.join(errors)}")

        # Create new instance to maintain immutability
        new_instance = self.__class__()
        new_instance._parameters = {**self.get_default_parameters(), **parameters}
        return new_instance

    @abstractmethod
    def build(self) -> GMNSpecification:
        """Build GMN specification from template.

        Returns:
            Complete GMN specification

        Raises:
            GMNTemplateError: If template generation fails
        """
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this template type.

        Returns:
            Dictionary of default parameter values
        """
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate template parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        pass

    def _get_parameters(self) -> Dict[str, Any]:
        """Get current parameters, merging with defaults."""
        defaults = self.get_default_parameters()
        return {**defaults, **self._parameters}

    def _create_minimal_spec(self, name: str) -> GMNSpecification:
        """Create minimal valid GMN specification.

        Args:
            name: Name for the specification

        Returns:
            Minimal GMN specification with required nodes
        """
        nodes = [
            GMNNode(id="state", type=GMNNodeType.STATE, properties={"num_states": 2}),
            GMNNode(
                id="observation", type=GMNNodeType.OBSERVATION, properties={"num_observations": 2}
            ),
            GMNNode(id="action", type=GMNNodeType.ACTION, properties={"num_actions": 2}),
        ]

        edges = [GMNEdge(source="state", target="observation", type=GMNEdgeType.GENERATES)]

        return GMNSpecification(
            name=name, description=f"{name} agent template", nodes=nodes, edges=edges
        )

    def _create_uniform_probability_matrix(
        self, shape: Tuple[int, ...], normalize_axis: int = 0
    ) -> np.ndarray:
        """Create uniform probability matrix with proper normalization.

        Args:
            shape: Shape of the matrix
            normalize_axis: Axis along which to normalize probabilities

        Returns:
            Normalized probability matrix
        """
        matrix = np.ones(shape)
        # Normalize along specified axis
        matrix = matrix / np.sum(matrix, axis=normalize_axis, keepdims=True)
        return matrix

    def _create_exploration_biased_matrix(
        self, shape: Tuple[int, ...], exploration_factor: float = 0.1, normalize_axis: int = 0
    ) -> np.ndarray:
        """Create exploration-biased probability matrix.

        Args:
            shape: Shape of the matrix
            exploration_factor: Factor controlling exploration vs exploitation (0-1)
            normalize_axis: Axis along which to normalize

        Returns:
            Exploration-biased probability matrix
        """
        # Start with uniform distribution
        matrix = np.ones(shape)

        # Add exploration bias - more uniform = more exploration
        if exploration_factor > 0:
            uniform_component = np.ones(shape)
            # Bias toward uniform distribution based on exploration factor
            matrix = (1 - exploration_factor) * matrix + exploration_factor * uniform_component

        # Normalize
        matrix = matrix / np.sum(matrix, axis=normalize_axis, keepdims=True)
        return matrix

    def _validate_positive_int(self, value: Any, name: str) -> int:
        """Validate that value is a positive integer.

        Args:
            value: Value to validate
            name: Parameter name for error messages

        Returns:
            Validated integer value

        Raises:
            GMNTemplateError: If validation fails
        """
        try:
            int_value = int(value)
            if int_value < 1:
                raise GMNTemplateError(f"{name} must be >= 1, got {int_value}")
            return int_value
        except (TypeError, ValueError):
            raise GMNTemplateError(f"{name} must be a positive integer, got {type(value).__name__}")

    def _validate_float_range(
        self, value: Any, name: str, min_val: float = 0.0, max_val: float = 1.0
    ) -> float:
        """Validate that value is a float within specified range.

        Args:
            value: Value to validate
            name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated float value

        Raises:
            GMNTemplateError: If validation fails
        """
        try:
            float_value = float(value)
            if not (min_val <= float_value <= max_val):
                raise GMNTemplateError(
                    f"{name} must be between {min_val} and {max_val}, got {float_value}"
                )
            return float_value
        except (TypeError, ValueError):
            raise GMNTemplateError(f"{name} must be a float, got {type(value).__name__}")

    def get_cache_key(self) -> str:
        """Generate cache key for current template configuration.

        Returns:
            Unique cache key string
        """
        if self._cache_key is None:
            # Create deterministic hash of class name and parameters
            key_data = {
                "template_class": self.__class__.__name__,
                "parameters": self._get_parameters(),
            }
            key_str = json.dumps(key_data, sort_keys=True)
            self._cache_key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return self._cache_key


class ExplorerTemplate(GMNTemplateBuilder):
    """Template for spatial navigation and exploration agents.

    Creates agents suitable for:
    - Grid world navigation
    - Spatial exploration tasks
    - Path planning and obstacle avoidance
    - Location-based decision making

    Parameters:
    - num_locations: Number of spatial locations (default: 4)
    - num_actions: Number of movement actions (default: 5)
    - location_names: Optional list of location names
    - action_names: Optional list of action names
    - world_type: Type of world ('grid' or 'graph', default: 'graph')
    - grid_width: Width for grid worlds (default: 2)
    - grid_height: Height for grid worlds (default: 2)
    """

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for explorer template."""
        return {
            "num_locations": 4,
            "num_actions": 5,
            "location_names": None,
            "action_names": None,
            "world_type": "graph",
            "grid_width": 2,
            "grid_height": 2,
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate explorer template parameters."""
        errors = []

        try:
            # Validate numeric parameters
            if "num_locations" in parameters:
                self._validate_positive_int(parameters["num_locations"], "num_locations")

            if "num_actions" in parameters:
                self._validate_positive_int(parameters["num_actions"], "num_actions")

            if "grid_width" in parameters:
                self._validate_positive_int(parameters["grid_width"], "grid_width")

            if "grid_height" in parameters:
                self._validate_positive_int(parameters["grid_height"], "grid_height")

            # Validate location names if provided
            if "location_names" in parameters and parameters["location_names"] is not None:
                names = parameters["location_names"]
                if not isinstance(names, list):
                    errors.append("location_names must be a list")
                elif "num_locations" in parameters and len(names) != parameters["num_locations"]:
                    errors.append(
                        f"location_names length ({len(names)}) must match num_locations ({parameters['num_locations']})"
                    )

            # Validate action names if provided
            if "action_names" in parameters and parameters["action_names"] is not None:
                names = parameters["action_names"]
                if not isinstance(names, list):
                    errors.append("action_names must be a list")
                elif "num_actions" in parameters and len(names) != parameters["num_actions"]:
                    errors.append(
                        f"action_names length ({len(names)}) must match num_actions ({parameters['num_actions']})"
                    )

            # Validate world type
            if "world_type" in parameters:
                if parameters["world_type"] not in ["grid", "graph"]:
                    errors.append("world_type must be 'grid' or 'graph'")

        except GMNTemplateError as e:
            errors.append(str(e))

        return len(errors) == 0, errors

    def build(self) -> GMNSpecification:
        """Build explorer agent GMN specification."""
        params = self._get_parameters()

        # Handle grid world vs graph world
        if params["world_type"] == "grid":
            num_locations = params["grid_width"] * params["grid_height"]
        else:
            num_locations = params["num_locations"]

        num_actions = params["num_actions"]

        # Create nodes
        nodes = []

        # Location state node
        location_node = GMNNode(
            id="location_state",
            type=GMNNodeType.STATE,
            properties={
                "num_states": num_locations,
                "description": "Agent's spatial location",
                "world_type": params["world_type"],
            },
            metadata={"template_type": "explorer", "node_role": "spatial_state"},
        )
        nodes.append(location_node)

        # Location observation node
        location_obs_node = GMNNode(
            id="location_observation",
            type=GMNNodeType.OBSERVATION,
            properties={
                "num_observations": num_locations,
                "description": "Observed location information",
            },
            metadata={"template_type": "explorer", "node_role": "spatial_observation"},
        )
        nodes.append(location_obs_node)

        # Movement action node
        movement_node = GMNNode(
            id="movement_action",
            type=GMNNodeType.ACTION,
            properties={
                "num_actions": num_actions,
                "description": "Available movement actions",
                "action_names": params.get("action_names"),
            },
            metadata={"template_type": "explorer", "node_role": "movement"},
        )
        nodes.append(movement_node)

        # Likelihood node (observation model)
        likelihood_node = GMNNode(
            id="location_likelihood",
            type=GMNNodeType.LIKELIHOOD,
            properties={
                "description": "Probability of observing location given true location",
                "matrix": self._create_location_observation_matrix(num_locations).tolist(),
            },
            metadata={"template_type": "explorer", "node_role": "observation_model"},
        )
        nodes.append(likelihood_node)

        # Transition node (dynamics model)
        transition_node = GMNNode(
            id="location_transition",
            type=GMNNodeType.TRANSITION,
            properties={
                "description": "Location transition probabilities given actions",
                "matrix": self._create_location_transition_matrix(
                    num_locations,
                    num_actions,
                    params["world_type"],
                    params.get("grid_width", 2),
                    params.get("grid_height", 2),
                ).tolist(),
            },
            metadata={"template_type": "explorer", "node_role": "dynamics_model"},
        )
        nodes.append(transition_node)

        # Initial belief node
        belief_node = GMNNode(
            id="initial_location_belief",
            type=GMNNodeType.BELIEF,
            properties={
                "description": "Initial belief over locations",
                "distribution": self._create_uniform_probability_matrix((num_locations,)).tolist(),
            },
            metadata={"template_type": "explorer", "node_role": "initial_belief"},
        )
        nodes.append(belief_node)

        # Goal preference node
        preference_node = GMNNode(
            id="location_preference",
            type=GMNNodeType.PREFERENCE,
            properties={
                "description": "Preference over locations (goal locations)",
                "vector": self._create_goal_preference_vector(num_locations).tolist(),
            },
            metadata={"template_type": "explorer", "node_role": "goal_preference"},
        )
        nodes.append(preference_node)

        # Create edges
        edges = []

        # State -> Likelihood -> Observation
        edges.append(
            GMNEdge(
                source="location_state",
                target="location_likelihood",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Likelihood depends on true location"},
            )
        )
        edges.append(
            GMNEdge(
                source="location_likelihood",
                target="location_observation",
                type=GMNEdgeType.GENERATES,
                properties={"description": "Likelihood generates observations"},
            )
        )

        # State + Action -> Transition -> Next State
        edges.append(
            GMNEdge(
                source="location_state",
                target="location_transition",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Transition depends on current location"},
            )
        )
        edges.append(
            GMNEdge(
                source="movement_action",
                target="location_transition",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Transition depends on chosen action"},
            )
        )

        # Preference -> Observation (goal specification)
        edges.append(
            GMNEdge(
                source="location_preference",
                target="location_observation",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Preferences defined over observed locations"},
            )
        )

        # Belief -> State (initial state distribution)
        edges.append(
            GMNEdge(
                source="initial_location_belief",
                target="location_state",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Initial belief over starting locations"},
            )
        )

        return GMNSpecification(
            name="Explorer Agent",
            description="Spatial navigation and exploration agent for grid/graph worlds",
            nodes=nodes,
            edges=edges,
            metadata={
                "template_type": "explorer",
                "num_locations": num_locations,
                "num_actions": num_actions,
                "world_type": params["world_type"],
                "generated_at": time.time(),
            },
        )

    def _create_location_observation_matrix(self, num_locations: int) -> np.ndarray:
        """Create observation matrix for location observations.

        Args:
            num_locations: Number of locations

        Returns:
            A matrix [num_locations, num_locations] where A[obs,state] = P(obs|state)
        """
        # Mostly accurate observations with small noise
        A = (
            np.eye(num_locations) * 0.9
            + np.ones((num_locations, num_locations)) * 0.1 / num_locations
        )
        # Normalize columns (each column sums to 1)
        A = A / np.sum(A, axis=0, keepdims=True)
        return A

    def _create_location_transition_matrix(
        self,
        num_locations: int,
        num_actions: int,
        world_type: str,
        grid_width: int = 2,
        grid_height: int = 2,
    ) -> np.ndarray:
        """Create transition matrix for location dynamics.

        Args:
            num_locations: Number of locations
            num_actions: Number of actions
            world_type: Type of world ('grid' or 'graph')
            grid_width: Width of grid world
            grid_height: Height of grid world

        Returns:
            B matrix [num_locations, num_locations, num_actions] where B[s',s,a] = P(s'|s,a)
        """
        B = np.zeros((num_locations, num_locations, num_actions))

        if world_type == "grid":
            # Create grid world transitions
            for action in range(num_actions):
                for state in range(num_locations):
                    # Convert linear index to grid coordinates
                    row, col = divmod(state, grid_width)

                    if action == 0:  # North
                        next_row = max(0, row - 1)
                        next_state = next_row * grid_width + col
                    elif action == 1:  # South
                        next_row = min(grid_height - 1, row + 1)
                        next_state = next_row * grid_width + col
                    elif action == 2:  # East
                        next_col = min(grid_width - 1, col + 1)
                        next_state = row * grid_width + next_col
                    elif action == 3:  # West
                        next_col = max(0, col - 1)
                        next_state = row * grid_width + next_col
                    else:  # Stay/Other actions
                        next_state = state

                    # Deterministic transition with small noise
                    B[next_state, state, action] = 0.9
                    # Add small probability of staying in place
                    B[state, state, action] += 0.1
        else:
            # Graph world - more connected transitions
            for action in range(num_actions):
                for state in range(num_locations):
                    # Each action connects to different subset of states
                    if action == 0:  # Forward connections
                        next_state = (state + 1) % num_locations
                    elif action == 1:  # Backward connections
                        next_state = (state - 1) % num_locations
                    elif action == 2:  # Jump connections
                        next_state = (state + num_locations // 2) % num_locations
                    elif action == 3:  # Random connections
                        next_state = (state * 2) % num_locations
                    else:  # Stay
                        next_state = state

                    B[next_state, state, action] = 0.8
                    B[state, state, action] += 0.2

        # Ensure columns sum to 1
        for action in range(num_actions):
            for state in range(num_locations):
                col_sum = np.sum(B[:, state, action])
                if col_sum > 0:
                    B[:, state, action] /= col_sum

        return B

    def _create_goal_preference_vector(self, num_locations: int) -> np.ndarray:
        """Create preference vector with goal location preference.

        Args:
            num_locations: Number of locations

        Returns:
            Preference vector favoring goal locations
        """
        # Prefer the last location as goal
        C = np.zeros(num_locations)
        C[-1] = 2.0  # High preference for goal
        C[0] = -1.0  # Negative preference for start
        return C


class AnalystTemplate(GMNTemplateBuilder):
    """Template for data analysis and decision-making agents.

    Creates agents suitable for:
    - Data processing and analysis
    - Pattern recognition
    - Decision making under uncertainty
    - Multi-criteria optimization

    Parameters:
    - num_data_sources: Number of data sources to analyze (default: 3)
    - num_analysis_actions: Number of analysis actions (default: 4)
    - data_types: List of data type names (optional)
    - analysis_methods: List of analysis method names (optional)
    - confidence_threshold: Confidence threshold for decisions (default: 0.7)
    """

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for analyst template."""
        return {
            "num_data_sources": 3,
            "num_analysis_actions": 4,
            "data_types": None,
            "analysis_methods": None,
            "confidence_threshold": 0.7,
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate analyst template parameters."""
        errors = []

        try:
            # Validate numeric parameters
            if "num_data_sources" in parameters:
                self._validate_positive_int(parameters["num_data_sources"], "num_data_sources")

            if "num_analysis_actions" in parameters:
                self._validate_positive_int(
                    parameters["num_analysis_actions"], "num_analysis_actions"
                )

            if "confidence_threshold" in parameters:
                self._validate_float_range(
                    parameters["confidence_threshold"], "confidence_threshold", 0.0, 1.0
                )

            # Validate lists if provided
            if "data_types" in parameters and parameters["data_types"] is not None:
                if not isinstance(parameters["data_types"], list):
                    errors.append("data_types must be a list")

            if "analysis_methods" in parameters and parameters["analysis_methods"] is not None:
                if not isinstance(parameters["analysis_methods"], list):
                    errors.append("analysis_methods must be a list")

        except GMNTemplateError as e:
            errors.append(str(e))

        return len(errors) == 0, errors

    def build(self) -> GMNSpecification:
        """Build analyst agent GMN specification."""
        params = self._get_parameters()

        num_data_sources = params["num_data_sources"]
        num_analysis_actions = params["num_analysis_actions"]
        confidence_threshold = params["confidence_threshold"]

        # Create nodes
        nodes = []

        # Data state node (current data being analyzed)
        data_state_node = GMNNode(
            id="data_state",
            type=GMNNodeType.STATE,
            properties={
                "num_states": num_data_sources,
                "description": "Current data source being analyzed",
                "data_types": params.get("data_types"),
            },
            metadata={"template_type": "analyst", "node_role": "data_state"},
        )
        nodes.append(data_state_node)

        # Analysis confidence state
        confidence_node = GMNNode(
            id="confidence_state",
            type=GMNNodeType.STATE,
            properties={
                "num_states": 3,  # Low, Medium, High confidence
                "description": "Current analysis confidence level",
                "confidence_levels": ["low", "medium", "high"],
            },
            metadata={"template_type": "analyst", "node_role": "confidence_state"},
        )
        nodes.append(confidence_node)

        # Data observations
        data_obs_node = GMNNode(
            id="data_observation",
            type=GMNNodeType.OBSERVATION,
            properties={
                "num_observations": num_data_sources * 2,  # Raw + processed
                "description": "Observed data characteristics",
            },
            metadata={"template_type": "analyst", "node_role": "data_observation"},
        )
        nodes.append(data_obs_node)

        # Analysis actions
        analysis_action_node = GMNNode(
            id="analysis_action",
            type=GMNNodeType.ACTION,
            properties={
                "num_actions": num_analysis_actions,
                "description": "Available analysis methods",
                "analysis_methods": params.get("analysis_methods"),
            },
            metadata={"template_type": "analyst", "node_role": "analysis_action"},
        )
        nodes.append(analysis_action_node)

        # Data likelihood model
        data_likelihood_node = GMNNode(
            id="data_likelihood",
            type=GMNNodeType.LIKELIHOOD,
            properties={
                "description": "Probability of observing data given true data source",
                "matrix": self._create_data_observation_matrix(num_data_sources).tolist(),
            },
            metadata={"template_type": "analyst", "node_role": "data_likelihood"},
        )
        nodes.append(data_likelihood_node)

        # Analysis transition model
        analysis_transition_node = GMNNode(
            id="analysis_transition",
            type=GMNNodeType.TRANSITION,
            properties={
                "description": "Data state transitions based on analysis actions",
                "matrix": self._create_analysis_transition_matrix(
                    num_data_sources, num_analysis_actions
                ).tolist(),
            },
            metadata={"template_type": "analyst", "node_role": "analysis_transition"},
        )
        nodes.append(analysis_transition_node)

        # Decision preferences
        decision_preference_node = GMNNode(
            id="decision_preference",
            type=GMNNodeType.PREFERENCE,
            properties={
                "description": "Preferences for analysis outcomes",
                "vector": self._create_decision_preference_vector(num_data_sources * 2).tolist(),
                "confidence_threshold": confidence_threshold,
            },
            metadata={"template_type": "analyst", "node_role": "decision_preference"},
        )
        nodes.append(decision_preference_node)

        # Initial belief about data
        data_belief_node = GMNNode(
            id="initial_data_belief",
            type=GMNNodeType.BELIEF,
            properties={
                "description": "Initial belief about data sources",
                "distribution": self._create_uniform_probability_matrix(
                    (num_data_sources,)
                ).tolist(),
            },
            metadata={"template_type": "analyst", "node_role": "initial_belief"},
        )
        nodes.append(data_belief_node)

        # Create edges
        edges = []

        # Data state -> likelihood -> observation
        edges.append(
            GMNEdge(
                source="data_state",
                target="data_likelihood",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Data observations depend on true data state"},
            )
        )
        edges.append(
            GMNEdge(
                source="data_likelihood",
                target="data_observation",
                type=GMNEdgeType.GENERATES,
                properties={"description": "Likelihood generates data observations"},
            )
        )

        # Analysis transitions
        edges.append(
            GMNEdge(
                source="data_state",
                target="analysis_transition",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Transitions depend on current data state"},
            )
        )
        edges.append(
            GMNEdge(
                source="analysis_action",
                target="analysis_transition",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Transitions depend on analysis action"},
            )
        )

        # Decision preferences
        edges.append(
            GMNEdge(
                source="decision_preference",
                target="data_observation",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Preferences defined over data observations"},
            )
        )

        # Initial beliefs
        edges.append(
            GMNEdge(
                source="initial_data_belief",
                target="data_state",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Initial belief over data sources"},
            )
        )

        return GMNSpecification(
            name="Analyst Agent",
            description="Data analysis and decision-making agent for multi-source analysis",
            nodes=nodes,
            edges=edges,
            metadata={
                "template_type": "analyst",
                "num_data_sources": num_data_sources,
                "num_analysis_actions": num_analysis_actions,
                "confidence_threshold": confidence_threshold,
                "generated_at": time.time(),
            },
        )

    def _create_data_observation_matrix(self, num_data_sources: int) -> np.ndarray:
        """Create observation matrix for data analysis.

        Args:
            num_data_sources: Number of data sources

        Returns:
            A matrix for data observations
        """
        num_obs = num_data_sources * 2  # Raw + processed observations
        A = np.zeros((num_obs, num_data_sources))

        # Each data source has higher probability of generating its own observations
        for src in range(num_data_sources):
            # Raw observation
            A[src * 2, src] = 0.8
            # Processed observation
            A[src * 2 + 1, src] = 0.7
            # Cross-contamination from other sources
            for other_src in range(num_data_sources):
                if other_src != src:
                    A[src * 2, other_src] = 0.2 / (num_data_sources - 1)
                    A[src * 2 + 1, other_src] = 0.3 / (num_data_sources - 1)

        # Normalize columns
        A = A / np.sum(A, axis=0, keepdims=True)
        return A

    def _create_analysis_transition_matrix(
        self, num_data_sources: int, num_actions: int
    ) -> np.ndarray:
        """Create transition matrix for analysis actions.

        Args:
            num_data_sources: Number of data sources
            num_actions: Number of analysis actions

        Returns:
            B matrix for analysis transitions
        """
        B = np.zeros((num_data_sources, num_data_sources, num_actions))

        for action in range(num_actions):
            for state in range(num_data_sources):
                if action == 0:  # Drill down - stay on same source
                    B[state, state, action] = 0.9
                    # Small chance of discovering related source
                    next_state = (state + 1) % num_data_sources
                    B[next_state, state, action] = 0.1
                elif action == 1:  # Explore - move to related source
                    next_state = (state + 1) % num_data_sources
                    B[next_state, state, action] = 0.7
                    B[state, state, action] = 0.3
                elif action == 2:  # Compare - systematic traversal
                    next_state = (state + num_data_sources // 2) % num_data_sources
                    B[next_state, state, action] = 0.6
                    B[state, state, action] = 0.4
                else:  # Synthesize - random exploration
                    # Uniform transition to encourage broad exploration
                    B[:, state, action] = 1.0 / num_data_sources

        # Ensure normalization
        for action in range(num_actions):
            for state in range(num_data_sources):
                col_sum = np.sum(B[:, state, action])
                if col_sum > 0:
                    B[:, state, action] /= col_sum

        return B

    def _create_decision_preference_vector(self, num_observations: int) -> np.ndarray:
        """Create preference vector for decision making.

        Args:
            num_observations: Number of observation types

        Returns:
            Preference vector for analysis outcomes
        """
        C = np.zeros(num_observations)
        # Prefer processed observations over raw
        for i in range(0, num_observations, 2):
            C[i] = 0.0  # Raw observation
            if i + 1 < num_observations:
                C[i + 1] = 1.0  # Processed observation (preferred)
        return C


class CreativeTemplate(GMNTemplateBuilder):
    """Template for creative generation and exploration agents.

    Creates agents suitable for:
    - Creative content generation
    - Artistic exploration
    - Novel solution finding
    - Divergent thinking tasks

    Parameters:
    - num_creative_states: Number of creative states (default: 5)
    - num_generation_actions: Number of generation actions (default: 6)
    - creativity_level: Level of creativity/exploration (0-1, default: 0.5)
    - exploration_factor: Factor controlling exploration vs exploitation (default: 0.3)
    - creative_domains: List of creative domains (optional)
    """

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for creative template."""
        return {
            "num_creative_states": 5,
            "num_generation_actions": 6,
            "creativity_level": 0.5,
            "exploration_factor": 0.3,
            "creative_domains": None,
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate creative template parameters."""
        errors = []

        try:
            # Validate numeric parameters
            if "num_creative_states" in parameters:
                self._validate_positive_int(
                    parameters["num_creative_states"], "num_creative_states"
                )

            if "num_generation_actions" in parameters:
                self._validate_positive_int(
                    parameters["num_generation_actions"], "num_generation_actions"
                )

            if "creativity_level" in parameters:
                self._validate_float_range(
                    parameters["creativity_level"], "creativity_level", 0.0, 1.0
                )

            if "exploration_factor" in parameters:
                self._validate_float_range(
                    parameters["exploration_factor"], "exploration_factor", 0.0, 1.0
                )

            # Validate lists if provided
            if "creative_domains" in parameters and parameters["creative_domains"] is not None:
                if not isinstance(parameters["creative_domains"], list):
                    errors.append("creative_domains must be a list")

        except GMNTemplateError as e:
            errors.append(str(e))

        return len(errors) == 0, errors

    def build(self) -> GMNSpecification:
        """Build creative agent GMN specification."""
        params = self._get_parameters()

        num_creative_states = params["num_creative_states"]
        num_generation_actions = params["num_generation_actions"]
        creativity_level = params["creativity_level"]
        exploration_factor = params["exploration_factor"]

        # Create nodes
        nodes = []

        # Creative state node
        creative_state_node = GMNNode(
            id="creative_state",
            type=GMNNodeType.STATE,
            properties={
                "num_states": num_creative_states,
                "description": "Current creative/inspirational state",
                "creative_domains": params.get("creative_domains"),
            },
            metadata={"template_type": "creative", "node_role": "creative_state"},
        )
        nodes.append(creative_state_node)

        # Inspiration level state
        inspiration_node = GMNNode(
            id="inspiration_state",
            type=GMNNodeType.STATE,
            properties={
                "num_states": 4,  # Low, Medium, High, Peak inspiration
                "description": "Current inspiration/flow state",
                "inspiration_levels": ["low", "medium", "high", "peak"],
            },
            metadata={"template_type": "creative", "node_role": "inspiration_state"},
        )
        nodes.append(inspiration_node)

        # Creative observations
        creative_obs_node = GMNNode(
            id="creative_observation",
            type=GMNNodeType.OBSERVATION,
            properties={
                "num_observations": num_creative_states + 2,  # States + quality metrics
                "description": "Observed creative outputs and feedback",
            },
            metadata={"template_type": "creative", "node_role": "creative_observation"},
        )
        nodes.append(creative_obs_node)

        # Generation actions
        generation_action_node = GMNNode(
            id="generation_action",
            type=GMNNodeType.ACTION,
            properties={
                "num_actions": num_generation_actions,
                "description": "Available creative generation actions",
                "creativity_level": creativity_level,
            },
            metadata={"template_type": "creative", "node_role": "generation_action"},
        )
        nodes.append(generation_action_node)

        # Creative likelihood model
        creative_likelihood_node = GMNNode(
            id="creative_likelihood",
            type=GMNNodeType.LIKELIHOOD,
            properties={
                "description": "Probability of creative observations given state",
                "matrix": self._create_creative_observation_matrix(num_creative_states).tolist(),
            },
            metadata={"template_type": "creative", "node_role": "creative_likelihood"},
        )
        nodes.append(creative_likelihood_node)

        # Creative transition model (with exploration bias)
        creative_transition_node = GMNNode(
            id="creative_transition",
            type=GMNNodeType.TRANSITION,
            properties={
                "description": "Creative state transitions with exploration bias",
                "matrix": self._create_creative_transition_matrix(
                    num_creative_states, num_generation_actions, exploration_factor
                ).tolist(),
                "exploration_factor": exploration_factor,
            },
            metadata={"template_type": "creative", "node_role": "creative_transition"},
        )
        nodes.append(creative_transition_node)

        # Novelty preferences
        novelty_preference_node = GMNNode(
            id="novelty_preference",
            type=GMNNodeType.PREFERENCE,
            properties={
                "description": "Preferences for novel and creative outputs",
                "vector": self._create_novelty_preference_vector(num_creative_states + 2).tolist(),
                "creativity_level": creativity_level,
            },
            metadata={"template_type": "creative", "node_role": "novelty_preference"},
        )
        nodes.append(novelty_preference_node)

        # Initial creative belief
        creative_belief_node = GMNNode(
            id="initial_creative_belief",
            type=GMNNodeType.BELIEF,
            properties={
                "description": "Initial belief about creative potential",
                "distribution": self._create_exploration_biased_matrix(
                    (num_creative_states,), exploration_factor
                ).tolist(),
            },
            metadata={"template_type": "creative", "node_role": "initial_belief"},
        )
        nodes.append(creative_belief_node)

        # Create edges
        edges = []

        # Creative state -> likelihood -> observation
        edges.append(
            GMNEdge(
                source="creative_state",
                target="creative_likelihood",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Creative observations depend on creative state"},
            )
        )
        edges.append(
            GMNEdge(
                source="creative_likelihood",
                target="creative_observation",
                type=GMNEdgeType.GENERATES,
                properties={"description": "Likelihood generates creative observations"},
            )
        )

        # Creative transitions
        edges.append(
            GMNEdge(
                source="creative_state",
                target="creative_transition",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Transitions depend on current creative state"},
            )
        )
        edges.append(
            GMNEdge(
                source="generation_action",
                target="creative_transition",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Transitions depend on generation action"},
            )
        )

        # Novelty preferences
        edges.append(
            GMNEdge(
                source="novelty_preference",
                target="creative_observation",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Preferences defined over creative observations"},
            )
        )

        # Initial beliefs
        edges.append(
            GMNEdge(
                source="initial_creative_belief",
                target="creative_state",
                type=GMNEdgeType.DEPENDS_ON,
                properties={"description": "Initial belief over creative states"},
            )
        )

        # Inspiration influences creativity
        edges.append(
            GMNEdge(
                source="inspiration_state",
                target="creative_state",
                type=GMNEdgeType.INFLUENCES,
                properties={"description": "Inspiration level influences creative state"},
            )
        )

        return GMNSpecification(
            name="Creative Agent",
            description="Creative generation and exploration agent for divergent thinking tasks",
            nodes=nodes,
            edges=edges,
            metadata={
                "template_type": "creative",
                "num_creative_states": num_creative_states,
                "num_generation_actions": num_generation_actions,
                "creativity_level": creativity_level,
                "exploration_factor": exploration_factor,
                "generated_at": time.time(),
            },
        )

    def _create_creative_observation_matrix(self, num_creative_states: int) -> np.ndarray:
        """Create observation matrix for creative outputs.

        Args:
            num_creative_states: Number of creative states

        Returns:
            A matrix for creative observations
        """
        num_obs = num_creative_states + 2  # States + quality metrics
        A = np.zeros((num_obs, num_creative_states))

        # Each creative state generates its own observations plus quality signals
        for state in range(num_creative_states):
            # Direct state observation
            A[state, state] = 0.7
            # Quality observations (novelty, coherence)
            A[num_creative_states, state] = 0.2  # Novelty signal
            A[num_creative_states + 1, state] = 0.1  # Coherence signal

            # Cross-influence from other states (creative connections)
            for other_state in range(num_creative_states):
                if other_state != state:
                    A[other_state, state] = 0.3 / (num_creative_states - 1)

        # Normalize columns
        A = A / np.sum(A, axis=0, keepdims=True)
        return A

    def _create_creative_transition_matrix(
        self, num_states: int, num_actions: int, exploration_factor: float
    ) -> np.ndarray:
        """Create transition matrix for creative exploration.

        Args:
            num_states: Number of creative states
            num_actions: Number of generation actions
            exploration_factor: Exploration bias factor

        Returns:
            B matrix with exploration bias
        """
        B = np.zeros((num_states, num_states, num_actions))

        for action in range(num_actions):
            for state in range(num_states):
                if action == 0:  # Refine - stay in similar state
                    B[state, state, action] = 0.6
                    # Small jumps to adjacent states
                    next_state = (state + 1) % num_states
                    prev_state = (state - 1) % num_states
                    B[next_state, state, action] = 0.2
                    B[prev_state, state, action] = 0.2
                elif action == 1:  # Explore - jump to distant states
                    # Prefer states that are far from current
                    far_state = (state + num_states // 2) % num_states
                    B[far_state, state, action] = 0.5
                    B[state, state, action] = 0.2
                    # Distribute remaining probability
                    remaining = 0.3 / (num_states - 2)
                    for s in range(num_states):
                        if s != state and s != far_state:
                            B[s, state, action] = remaining
                elif action == 2:  # Combine - blend states
                    # More uniform distribution
                    B[:, state, action] = 1.0 / num_states
                elif action == 3:  # Iterate - systematic progression
                    next_state = (state + 1) % num_states
                    B[next_state, state, action] = 0.8
                    B[state, state, action] = 0.2
                else:  # Random/experimental actions
                    # High exploration bias
                    B[:, state, action] = self._create_exploration_biased_matrix(
                        (num_states,), exploration_factor
                    )

        # Ensure normalization
        for action in range(num_actions):
            for state in range(num_states):
                col_sum = np.sum(B[:, state, action])
                if col_sum > 0:
                    B[:, state, action] /= col_sum

        return B

    def _create_novelty_preference_vector(self, num_observations: int) -> np.ndarray:
        """Create preference vector favoring novel outcomes.

        Args:
            num_observations: Number of observation types

        Returns:
            Preference vector emphasizing novelty
        """
        C = np.zeros(num_observations)
        # Higher preference for diverse creative states
        for i in range(num_observations - 2):
            C[i] = 1.0  # All creative states equally valued
        # Strong preference for novelty signal
        C[-2] = 2.0  # Novelty observation
        C[-1] = 0.5  # Coherence observation (less important for creativity)
        return C


class TemplateCache:
    """LRU cache for template generation results.

    Provides performance optimization by caching generated GMN specifications
    based on template class and parameters.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize template cache.

        Args:
            max_size: Maximum number of cached templates
        """
        self.max_size = max_size
        self._cache: Dict[str, GMNSpecification] = {}
        self._access_order: List[str] = []

    def get_or_create(
        self, template: GMNTemplateBuilder, parameters: Dict[str, Any]
    ) -> GMNSpecification:
        """Get cached template or create new one.

        Args:
            template: Template builder instance
            parameters: Template parameters

        Returns:
            Generated GMN specification
        """
        # Create template with parameters for cache key generation
        parameterized_template = template.with_parameters(parameters)
        cache_key = parameterized_template.get_cache_key()

        # Check cache
        if cache_key in self._cache:
            # Move to end (most recently used)
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]

        # Generate new specification
        spec = parameterized_template.build()

        # Add to cache
        self._cache[cache_key] = spec
        self._access_order.append(cache_key)

        # Evict oldest if over capacity
        if len(self._cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        return spec

    def clear(self) -> None:
        """Clear all cached templates."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class GMNTemplateFactory:
    """Factory for creating GMN templates.

    Provides centralized template creation and registration system
    following Factory pattern.
    """

    def __init__(self):
        """Initialize template factory."""
        self._templates: Dict[str, type] = {
            "explorer": ExplorerTemplate,
            "analyst": AnalystTemplate,
            "creative": CreativeTemplate,
        }
        self._cache = TemplateCache()

    def create_template(self, agent_type: AgentType) -> GMNTemplateBuilder:
        """Create template by agent type enum.

        Args:
            agent_type: Agent type enum

        Returns:
            Template builder instance

        Raises:
            GMNTemplateError: If agent type is not supported
        """
        return self.create_template_by_name(agent_type.value)

    def create_template_by_name(self, template_name: str) -> GMNTemplateBuilder:
        """Create template by string name.

        Args:
            template_name: Name of template type

        Returns:
            Template builder instance

        Raises:
            GMNTemplateError: If template name is not registered
        """
        template_name = template_name.lower().strip()
        if template_name not in self._templates:
            raise GMNTemplateError(
                f"Unknown agent type: {template_name}. Available: {list(self._templates.keys())}"
            )

        template_class = self._templates[template_name]
        return template_class()

    def register_template(self, name: str, template_class: type) -> None:
        """Register custom template class.

        Args:
            name: Template name
            template_class: Template class (must inherit from GMNTemplateBuilder)

        Raises:
            GMNTemplateError: If template class is invalid
        """
        if not issubclass(template_class, GMNTemplateBuilder):
            raise GMNTemplateError("Template class must inherit from GMNTemplateBuilder")

        self._templates[name.lower().strip()] = template_class

    def list_available_templates(self) -> List[str]:
        """List all available template names.

        Returns:
            List of registered template names
        """
        return list(self._templates.keys())

    def create_with_cache(self, template_name: str, parameters: Dict[str, Any]) -> GMNSpecification:
        """Create template with caching.

        Args:
            template_name: Name of template type
            parameters: Template parameters

        Returns:
            Generated GMN specification
        """
        template = self.create_template_by_name(template_name)
        return self._cache.get_or_create(template, parameters)

    def clear_cache(self) -> None:
        """Clear template cache."""
        self._cache.clear()


# Convenience functions


def create_explorer_agent(
    num_locations: int = 4, num_actions: int = 5, world_type: str = "graph", **kwargs
) -> GMNSpecification:
    """Create explorer agent with specified parameters.

    Args:
        num_locations: Number of spatial locations
        num_actions: Number of movement actions
        world_type: Type of world ('grid' or 'graph')
        **kwargs: Additional template parameters

    Returns:
        Generated explorer GMN specification
    """
    template = ExplorerTemplate()
    params = {
        "num_locations": num_locations,
        "num_actions": num_actions,
        "world_type": world_type,
        **kwargs,
    }
    return template.with_parameters(params).build()


def create_analyst_agent(
    num_data_sources: int = 3,
    num_analysis_actions: int = 4,
    confidence_threshold: float = 0.7,
    **kwargs,
) -> GMNSpecification:
    """Create analyst agent with specified parameters.

    Args:
        num_data_sources: Number of data sources
        num_analysis_actions: Number of analysis actions
        confidence_threshold: Decision confidence threshold
        **kwargs: Additional template parameters

    Returns:
        Generated analyst GMN specification
    """
    template = AnalystTemplate()
    params = {
        "num_data_sources": num_data_sources,
        "num_analysis_actions": num_analysis_actions,
        "confidence_threshold": confidence_threshold,
        **kwargs,
    }
    return template.with_parameters(params).build()


def create_creative_agent(
    num_creative_states: int = 5,
    num_generation_actions: int = 6,
    creativity_level: float = 0.5,
    exploration_factor: float = 0.3,
    **kwargs,
) -> GMNSpecification:
    """Create creative agent with specified parameters.

    Args:
        num_creative_states: Number of creative states
        num_generation_actions: Number of generation actions
        creativity_level: Level of creativity (0-1)
        exploration_factor: Exploration bias factor (0-1)
        **kwargs: Additional template parameters

    Returns:
        Generated creative GMN specification
    """
    template = CreativeTemplate()
    params = {
        "num_creative_states": num_creative_states,
        "num_generation_actions": num_generation_actions,
        "creativity_level": creativity_level,
        "exploration_factor": exploration_factor,
        **kwargs,
    }
    return template.with_parameters(params).build()

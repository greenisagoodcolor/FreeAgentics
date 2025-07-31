"""Agent factory service for creating PyMDP agents from GMN models.

This service handles the conversion of parsed GMN models into fully functional
PyMDP active inference agents.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# PyMDP is a required dependency - no fallbacks allowed
from pymdp.agent import Agent
from pymdp import utils


logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating PyMDP agents from GMN-parsed models."""

    def __init__(self):
        """Initialize the agent factory."""
        self.default_planning_horizon = 3
        self.default_inference_algo = "fpi"  # Fixed point iteration
        self.default_policy_len = 1

    async def validate_model(self, model: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a PyMDP model before agent creation.

        Args:
            model: PyMDP model dictionary from GMN parser

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        required_fields = ["num_states", "num_obs", "num_controls"]
        for field in required_fields:
            if field not in model:
                errors.append(f"Missing required field: {field}")

        # Validate dimensions
        if "num_states" in model:
            num_states = model["num_states"]
            if not isinstance(num_states, list) or not all(
                isinstance(n, int) and n > 0 for n in num_states
            ):
                errors.append("num_states must be a list of positive integers")

        if "num_obs" in model:
            num_obs = model["num_obs"]
            if not isinstance(num_obs, list) or not all(
                isinstance(n, int) and n > 0 for n in num_obs
            ):
                errors.append("num_obs must be a list of positive integers")

        if "num_controls" in model:
            num_controls = model["num_controls"]
            if not isinstance(num_controls, list) or not all(
                isinstance(n, int) and n > 0 for n in num_controls
            ):
                errors.append("num_controls must be a list of positive integers")

        # Validate matrices if provided
        if "A" in model and len(errors) == 0:
            A_errors = self._validate_A_matrix(
                model["A"], model.get("num_obs"), model.get("num_states")
            )
            errors.extend(A_errors)

        if "B" in model and len(errors) == 0:
            B_errors = self._validate_B_matrix(
                model["B"], model.get("num_states"), model.get("num_controls")
            )
            errors.extend(B_errors)

        if "C" in model:
            C_errors = self._validate_C_matrix(model["C"], model.get("num_obs"))
            errors.extend(C_errors)

        if "D" in model:
            D_errors = self._validate_D_matrix(model["D"], model.get("num_states"))
            errors.extend(D_errors)

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Model validation successful")
        else:
            logger.warning(f"Model validation failed with {len(errors)} errors")

        return is_valid, errors

    async def create_from_gmn_model(
        self,
        model: Dict[str, Any],
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """Create a PyMDP agent from a GMN-parsed model.

        Args:
            model: PyMDP model dictionary from GMN parser
            agent_id: Unique identifier for the agent
            metadata: Optional metadata about agent creation

        Returns:
            Configured PyMDP Agent instance

        Raises:
            ValueError: If model validation fails
            RuntimeError: If agent creation fails
        """
        logger.info(f"Creating PyMDP agent {agent_id} from GMN model")

        # Validate model first
        is_valid, errors = await self.validate_model(model)
        if not is_valid:
            raise ValueError(f"Model validation failed: {', '.join(errors)}")

        try:
            # Extract dimensions
            num_states = model["num_states"]
            num_obs = model["num_obs"]
            num_controls = model["num_controls"]

            # Create or use provided matrices
            A = self._create_A_matrix(model.get("A"), num_obs, num_states)
            B = self._create_B_matrix(model.get("B"), num_states, num_controls)
            C = self._create_C_matrix(model.get("C"), num_obs)
            D = self._create_D_matrix(model.get("D"), num_states)

            # Extract agent parameters
            planning_horizon = model.get("planning_horizon", self.default_planning_horizon)
            inference_algo = model.get("inference_algo", self.default_inference_algo)
            policy_len = model.get("policy_len", self.default_policy_len)

            # Create agent with parameters
            agent = Agent(
                A=A,
                B=B,
                C=C,
                D=D,
                num_states=num_states,
                num_obs=num_obs,
                num_controls=num_controls,
                planning_horizon=planning_horizon,
                inference_algo=inference_algo,
                policy_len=policy_len,
                use_utility=True,
                use_states_info_gain=True,
                use_param_info_gain=False,
                action_precision=1.0,
                inference_horizon=planning_horizon,
            )

            # Store metadata
            agent.id = agent_id
            agent.metadata = metadata or {}
            agent.gmn_model = model

            logger.info(f"Successfully created agent {agent_id}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            raise RuntimeError(f"Agent creation failed: {str(e)}")

    def _validate_A_matrix(self, A: Any, num_obs: List[int], num_states: List[int]) -> List[str]:
        """Validate observation model matrix."""
        errors = []

        if not isinstance(A, (list, np.ndarray)):
            errors.append("A matrix must be a list or numpy array")
            return errors

        # Convert to numpy for easier validation
        try:
            A_array = np.array(A) if isinstance(A, list) else A

            # Check dimensions
            len(num_states)
            num_modalities = len(num_obs)

            if isinstance(A_array, np.ndarray) and A_array.ndim == 2:
                # Single modality, single factor case
                if A_array.shape != (num_obs[0], num_states[0]):
                    errors.append(
                        f"A matrix shape {A_array.shape} doesn't match expected ({num_obs[0]}, {num_states[0]})"
                    )
            else:
                # Multi-modality or multi-factor case
                if not isinstance(A_array, list) or len(A_array) != num_modalities:
                    errors.append(f"A must have {num_modalities} modalities")

            # Check normalization
            if isinstance(A_array, np.ndarray) and not np.allclose(A_array.sum(axis=0), 1.0):
                errors.append("A matrix columns must sum to 1 (normalized probabilities)")

        except Exception as e:
            errors.append(f"Error validating A matrix: {str(e)}")

        return errors

    def _validate_B_matrix(
        self, B: Any, num_states: List[int], num_controls: List[int]
    ) -> List[str]:
        """Validate transition model matrix."""
        errors = []

        if not isinstance(B, (list, np.ndarray)):
            errors.append("B matrix must be a list or numpy array")
            return errors

        try:
            B_array = np.array(B) if isinstance(B, list) else B

            # Check dimensions for single factor case
            if len(num_states) == 1 and isinstance(B_array, np.ndarray):
                expected_shape = (
                    num_states[0],
                    num_states[0],
                    num_controls[0],
                )
                if B_array.shape != expected_shape:
                    errors.append(
                        f"B matrix shape {B_array.shape} doesn't match expected {expected_shape}"
                    )

                # Check normalization
                if not np.allclose(B_array.sum(axis=0), 1.0):
                    errors.append("B matrix columns must sum to 1")

        except Exception as e:
            errors.append(f"Error validating B matrix: {str(e)}")

        return errors

    def _validate_C_matrix(self, C: Any, num_obs: List[int]) -> List[str]:
        """Validate preference matrix."""
        errors = []

        if C is None:
            return errors  # C is optional

        if not isinstance(C, (list, np.ndarray)):
            errors.append("C matrix must be a list or numpy array")
            return errors

        try:
            C_array = np.array(C) if isinstance(C, list) else C

            # Check dimensions
            if len(num_obs) == 1 and isinstance(C_array, np.ndarray):
                if C_array.shape != (num_obs[0],) and C_array.shape != (
                    num_obs[0],
                    1,
                ):
                    errors.append(
                        f"C matrix shape {C_array.shape} doesn't match expected ({num_obs[0]},)"
                    )

        except Exception as e:
            errors.append(f"Error validating C matrix: {str(e)}")

        return errors

    def _validate_D_matrix(self, D: Any, num_states: List[int]) -> List[str]:
        """Validate initial state distribution."""
        errors = []

        if D is None:
            return errors  # D is optional

        if not isinstance(D, (list, np.ndarray)):
            errors.append("D matrix must be a list or numpy array")
            return errors

        try:
            D_array = np.array(D) if isinstance(D, list) else D

            # Check dimensions and normalization
            if len(num_states) == 1 and isinstance(D_array, np.ndarray):
                if D_array.shape != (num_states[0],) and D_array.shape != (
                    num_states[0],
                    1,
                ):
                    errors.append(
                        f"D matrix shape {D_array.shape} doesn't match expected ({num_states[0]},)"
                    )

                if not np.allclose(D_array.sum(), 1.0):
                    errors.append("D matrix must sum to 1 (probability distribution)")

        except Exception as e:
            errors.append(f"Error validating D matrix: {str(e)}")

        return errors

    def _create_A_matrix(
        self, A_spec: Any, num_obs: List[int], num_states: List[int]
    ) -> List[np.ndarray]:
        """Create observation model matrix."""
        if A_spec is not None:
            # Use provided matrix
            if isinstance(A_spec, list) and all(isinstance(a, np.ndarray) for a in A_spec):
                return A_spec
            elif isinstance(A_spec, np.ndarray):
                return [A_spec]
            else:
                # Convert to numpy
                return [np.array(A_spec)]

        # Generate default A matrix
        A_matrices = []
        for g, no in enumerate(num_obs):
            # Create identity-like observation model
            if len(num_states) == 1:
                ns = num_states[0]
                if no == ns:
                    # Identity mapping
                    A = np.eye(no)
                else:
                    # Random but structured
                    A = utils.random_A_matrix(no, ns)
                    # Add some structure - create identity-like matrix with correct shape
                    identity_like = np.zeros((no, ns))
                    min_dim = min(no, ns)
                    identity_like[:min_dim, :min_dim] = np.eye(min_dim)
                    A = A + 2 * identity_like
                    A = A / A.sum(axis=0, keepdims=True)
            else:
                # Multi-factor case
                A = utils.random_A_matrix(no, np.prod(num_states))

            A_matrices.append(A)

        return A_matrices

    def _create_B_matrix(
        self, B_spec: Any, num_states: List[int], num_controls: List[int]
    ) -> List[np.ndarray]:
        """Create transition model matrix."""
        if B_spec is not None:
            # Use provided matrix
            if isinstance(B_spec, list) and all(isinstance(b, np.ndarray) for b in B_spec):
                return B_spec
            elif isinstance(B_spec, np.ndarray):
                return [B_spec]
            else:
                # Convert to numpy
                return [np.array(B_spec)]

        # Generate default B matrix
        B_matrices = []
        for f, ns in enumerate(num_states):
            nc = num_controls[f] if f < len(num_controls) else num_controls[0]

            # Create structured transition model
            B = np.zeros((ns, ns, nc))

            # Default transitions based on control
            for a in range(nc):
                if nc == 4:  # Assume cardinal directions
                    # Up, Down, Left, Right
                    if a == 0:  # Up - move to lower index
                        # Each state i transitions to state i-1
                        for i in range(1, ns):
                            B[i - 1, i, a] = 1  # Move up
                        B[0, 0, a] = 1  # Stay at top boundary
                    elif a == 1:  # Down - move to higher index
                        # Each state i transitions to state i+1
                        for i in range(ns - 1):
                            B[i + 1, i, a] = 1  # Move down
                        B[ns - 1, ns - 1, a] = 1  # Stay at bottom boundary
                    elif a == 2:  # Left (cyclic)
                        B[:, :, a] = np.roll(np.eye(ns), 1, axis=1)
                    elif a == 3:  # Right (cyclic)
                        B[:, :, a] = np.roll(np.eye(ns), -1, axis=1)
                else:
                    # Generic transitions
                    B[:, :, a] = utils.random_B_matrix(ns, 1)[:, :, 0]

            B_matrices.append(B)

        return B_matrices

    def _create_C_matrix(self, C_spec: Any, num_obs: List[int]) -> List[np.ndarray]:
        """Create preference matrix."""
        if C_spec is not None:
            # Use provided matrix
            if isinstance(C_spec, list) and all(isinstance(c, np.ndarray) for c in C_spec):
                return C_spec
            elif isinstance(C_spec, np.ndarray):
                return [C_spec]
            else:
                # Convert to numpy
                return [np.array(C_spec)]

        # Generate neutral preferences
        C_matrices = []
        for no in num_obs:
            C = np.zeros(no)  # Neutral preferences
            C_matrices.append(C)

        return C_matrices

    def _create_D_matrix(self, D_spec: Any, num_states: List[int]) -> List[np.ndarray]:
        """Create initial state distribution."""
        if D_spec is not None:
            # Use provided matrix
            if isinstance(D_spec, list) and all(isinstance(d, np.ndarray) for d in D_spec):
                return D_spec
            elif isinstance(D_spec, np.ndarray):
                return [D_spec]
            else:
                # Convert to numpy
                return [np.array(D_spec)]

        # Generate uniform initial distribution
        D_matrices = []
        for ns in num_states:
            D = np.ones(ns) / ns  # Uniform distribution
            D_matrices.append(D)

        return D_matrices

"""PyMDP API compatibility adapter with strict type checking.

This module provides a thin adapter layer that translates between PyMDP's actual
API behavior and the expected API behavior, with zero fallbacks and
    strict type checking.

Based on Task 1.3: Create API compatibility adapter with strict type checking
"""

import logging
from typing import Any, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Import PyMDP types for strict checking
from pymdp.agent import Agent as PyMDPAgent

logger = logging.getLogger(__name__)


class PyMDPCompatibilityAdapter:
    """API compatibility adapter for PyMDP with strict type checking.

    This adapter ensures exact API signatures and return types with NO fallbacks.
    Operations must work or raise exceptions - no graceful degradation allowed.
    """

    def __init__(self):
        """Initialize the compatibility adapter."""
        logger.info(
            "Initializing PyMDP compatibility adapter with strict type checking"
        )

    def sample_action(self, pymdp_agent: PyMDPAgent) -> int:
        """Convert PyMDP action result to strict int type.

        PyMDP's sample_action() returns numpy.ndarray[float64] with shape (1,)
        This adapter converts it to exactly int type with no graceful fallbacks.

        Args:
            pymdp_agent: The PyMDP agent instance

        Returns:
            int: The sampled action index as exact int type

        Raises:
            TypeError: If pymdp_agent is not a PyMDPAgent
            RuntimeError: If sample_action() returns unexpected type/format
            ValueError: If conversion to int fails
        """
        # Strict type checking - NO graceful fallbacks
        if not isinstance(pymdp_agent, PyMDPAgent):
            raise TypeError(f"Expected PyMDPAgent, got {type(pymdp_agent)}")

        # Call PyMDP's sample_action
        action_result = pymdp_agent.sample_action()

        # Strict return type validation and conversion
        if not isinstance(action_result, np.ndarray):
            raise RuntimeError(
                f"PyMDP sample_action() returned {type(action_result)}, "
                f"expected numpy.ndarray"
            )

        if action_result.dtype not in [
            np.float64,
            np.float32,
            np.int64,
            np.int32,
        ]:
            raise RuntimeError(
                f"PyMDP sample_action() returned unexpected dtype {action_result.dtype}"
            )

        if action_result.shape != (1,):
            raise RuntimeError(
                f"PyMDP sample_action() returned shape {action_result.shape}, "
                f"expected (1,)"
            )

        # Convert to exact int type - no fallbacks on failure
        try:
            action_int = int(action_result.item())
        except (ValueError, TypeError, OverflowError) as e:
            raise ValueError(f"Failed to convert {action_result} to int: {e}")

        # Validate result is non-negative (action indices should be >= 0)
        if action_int < 0:
            raise ValueError(f"Action index {action_int} is negative")

        logger.debug(f"Converted PyMDP action {action_result} to int {action_int}")
        return action_int

    def infer_policies(
        self, pymdp_agent: PyMDPAgent
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Validate PyMDP policy inference return types with strict checking.

        PyMDP's infer_policies() returns (q_pi, G) where both are numpy arrays.
        This adapter validates the exact return format with no fallbacks.

        Args:
            pymdp_agent: The PyMDP agent instance

        Returns:
            Tuple[NDArray[np.floating], NDArray[np.floating]]: (q_pi, G) arrays

        Raises:
            TypeError: If pymdp_agent is not a PyMDPAgent
            RuntimeError: If infer_policies() returns unexpected type/format
        """
        # Strict type checking
        if not isinstance(pymdp_agent, PyMDPAgent):
            raise TypeError(f"Expected PyMDPAgent, got {type(pymdp_agent)}")

        # Call PyMDP's infer_policies
        policies_result = pymdp_agent.infer_policies()

        # Strict return type validation
        if not isinstance(policies_result, tuple):
            raise RuntimeError(
                f"PyMDP infer_policies() returned {type(policies_result)}, "
                f"expected tuple"
            )

        if len(policies_result) != 2:
            raise RuntimeError(
                f"PyMDP infer_policies() returned tuple of length {len(policies_result)}, expected 2"
            )

        q_pi, G = policies_result

        # Validate q_pi
        if not isinstance(q_pi, np.ndarray):
            raise RuntimeError(f"q_pi is {type(q_pi)}, expected numpy.ndarray")

        if not np.issubdtype(q_pi.dtype, np.floating):
            raise RuntimeError(f"q_pi has dtype {q_pi.dtype}, expected floating point")

        # Validate G
        if not isinstance(G, np.ndarray):
            raise RuntimeError(f"G is {type(G)}, expected numpy.ndarray")

        if not np.issubdtype(G.dtype, np.floating):
            raise RuntimeError(f"G has dtype {G.dtype}, expected floating point")

        logger.debug(
            f"Validated infer_policies return: q_pi shape {q_pi.shape}, G"
            f" shape {G.shape}"
        )
        return q_pi, G

    def _validate_observation_format(
        self, observation: Union[int, List[int], NDArray[Any]]
    ) -> List[int]:
        """Validate and format observation for PyMDP."""
        if isinstance(observation, int):
            return [observation]
        elif isinstance(observation, list):
            return observation
        elif isinstance(observation, np.ndarray):
            return self._handle_numpy_observation(observation)
        else:
            raise TypeError(f"Observation type {type(observation)} not supported")

    def _handle_numpy_observation(self, observation: NDArray[Any]) -> List[int]:
        """Handle different numpy array observation formats."""
        if observation.ndim == 0:
            # 0-dimensional array (scalar)
            return [int(observation.item())]
        elif observation.ndim == 1:
            # 1-dimensional array
            obs_list = observation.astype(int).tolist()
            return obs_list if isinstance(obs_list, list) else [obs_list]
        else:
            # Multi-dimensional arrays not supported
            raise TypeError(
                f"Multi-dimensional observation arrays not supported: shape {observation.shape}"
            )

    def _process_beliefs_result(self, beliefs_result) -> List[NDArray[np.floating]]:
        """Process PyMDP infer_states result into standard format."""
        if isinstance(beliefs_result, list):
            return beliefs_result
        elif isinstance(beliefs_result, np.ndarray):
            return self._handle_numpy_beliefs(beliefs_result)
        else:
            raise RuntimeError(
                f"infer_states returned {type(beliefs_result)}, expected list or numpy.ndarray"
            )

    def _handle_numpy_beliefs(
        self, beliefs_result: NDArray[Any]
    ) -> List[NDArray[np.floating]]:
        """Handle numpy array beliefs result."""
        if beliefs_result.dtype == np.object_:
            return self._handle_object_array_beliefs(beliefs_result)
        else:
            # Regular array - each element is a belief
            return [beliefs_result]

    def _handle_object_array_beliefs(
        self, beliefs_result: NDArray[Any]
    ) -> List[NDArray[np.floating]]:
        """Handle object dtype array beliefs."""
        if beliefs_result.shape == (1,):
            # Extract content from single-element object array
            content = beliefs_result.item()
            if isinstance(content, list):
                return content
            elif isinstance(content, np.ndarray):
                # Single belief array wrapped in object array
                return [content]
            else:
                raise RuntimeError(
                    f"infer_states object array contains {type(content)}, "
                    f"expected list or ndarray"
                )
        else:
            # Multi-element object array - convert to list
            result_list = beliefs_result.tolist()
            # Ensure we return a list of ndarrays
            return [
                (
                    np.array(item, dtype=np.float64)
                    if not isinstance(item, np.ndarray)
                    else item
                )
                for item in result_list
            ]

    def _validate_beliefs_format(
        self, beliefs_list: List[NDArray[np.floating]]
    ) -> None:
        """Validate each belief array format."""
        for i, belief in enumerate(beliefs_list):
            if not isinstance(belief, np.ndarray):
                raise RuntimeError(
                    f"Belief {i} is {type(belief)}, expected numpy.ndarray"
                )

            if not np.issubdtype(belief.dtype, np.floating):
                raise RuntimeError(
                    f"Belief {i} has dtype {belief.dtype}, expected floating point"
                )

    def infer_states(
        self,
        pymdp_agent: PyMDPAgent,
        observation: Union[int, List[int], NDArray[Any]],
    ) -> List[NDArray[np.floating]]:
        """Validate PyMDP state inference with strict input/output validation.

        Args:
            pymdp_agent: The PyMDP agent instance
            observation: Observation in PyMDP format

        Returns:
            List[NDArray[np.floating]]: Posterior beliefs over states

        Raises:
            TypeError: If types are incorrect
            RuntimeError: If operation fails
        """
        # Strict type checking
        if not isinstance(pymdp_agent, PyMDPAgent):
            raise TypeError(f"Expected PyMDPAgent, got {type(pymdp_agent)}")

        # Validate and format observation
        obs_formatted = self._validate_observation_format(observation)

        # Call PyMDP's infer_states
        beliefs_result = pymdp_agent.infer_states(obs_formatted)

        # Process and validate results
        beliefs_list = self._process_beliefs_result(beliefs_result)
        self._validate_beliefs_format(beliefs_list)

        logger.debug(
            f"Validated infer_states return: {len(beliefs_list)} belief arrays"
        )
        return beliefs_list

    def validate_agent_state(self, pymdp_agent: PyMDPAgent) -> bool:
        """Validate that PyMDP agent is in proper state for operations.

        Args:
            pymdp_agent: The PyMDP agent instance

        Returns:
            bool: True if agent is properly initialized

        Raises:
            TypeError: If pymdp_agent is not correct type
            RuntimeError: If agent is not properly initialized
        """
        # Strict type checking
        if not isinstance(pymdp_agent, PyMDPAgent):
            raise TypeError(f"Expected PyMDPAgent, got {type(pymdp_agent)}")

        # Check required attributes exist - only A and B are truly required
        # C and D are optional and have defaults
        required_attrs = ["A", "B"]
        for attr in required_attrs:
            if not hasattr(pymdp_agent, attr):
                raise RuntimeError(f"PyMDP agent missing required attribute: {attr}")

        # Check if agent can sample actions (requires q_pi to be set)
        try:
            # This will fail if q_pi is not initialized
            if hasattr(pymdp_agent, "q_pi") and pymdp_agent.q_pi is not None:
                logger.debug("Agent q_pi is initialized")
            else:
                logger.debug(
                    "Agent q_pi not initialized - may need infer_policies() call"
                )
        except AttributeError:
            logger.debug("Agent does not have q_pi attribute")

        return True

    def safe_array_conversion(
        self, value: Any, target_type: type = int
    ) -> Union[int, float]:
        """Strict array to scalar conversion with no fallbacks.

        Args:
            value: Value to convert (numpy array, scalar, etc.)
            target_type: Target type (int or float)

        Returns:
            Union[int, float]: Converted value matching target_type

        Raises:
            TypeError: If value cannot be converted
            ValueError: If conversion results in invalid value
        """
        if target_type not in [int, float]:
            raise TypeError(f"target_type must be int or float, got {target_type}")

        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError("Cannot convert empty array")
            elif value.size == 1:
                scalar_value = value.item()
            else:
                raise ValueError(
                    f"Cannot convert multi-element array of size {value.size} to scalar"
                )
        # Handle numpy scalars
        elif hasattr(value, "item"):
            scalar_value = value.item()
        # Handle regular scalars
        elif isinstance(value, (int, float, np.integer, np.floating)):
            scalar_value = value
        else:
            raise TypeError(f"Cannot convert {type(value)} to {target_type}")

        # Convert to target type with proper type handling
        try:
            if target_type is int:
                converted_result: int = int(scalar_value)
                return converted_result
            elif target_type is float:
                converted_result_float: float = float(scalar_value)
                return converted_result_float
            else:
                raise TypeError(f"Unsupported target_type: {target_type}")
        except (ValueError, TypeError, OverflowError) as e:
            raise ValueError(f"Failed to convert {scalar_value} to {target_type}: {e}")

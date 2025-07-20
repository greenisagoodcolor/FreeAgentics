"""Hard failure implementations that replace fallback handlers with assertion-based validation.

This module converts all graceful degradation patterns to immediate hard failures
for use in production environments where no performance theater is acceptable.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


class HardFailureError(Exception):
    """Exception raised when a hard failure condition is met."""


class ErrorHandlerHardFailure:
    """Hard failure error handler - raises exceptions instead of graceful degradation."""

    def __init__(self, agent_id: str) -> None:
        """Initialize hard failure error handler.

        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id
        self.error_count: int = 0

    def handle_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle an error by raising it immediately - no recovery information."""
        self.error_count += 1
        logging.error(
            f"Agent {self.agent_id} hard failure in {operation}: {error}"
        )

        # HARD FAILURE: Raise the original error immediately, no graceful degradation
        raise HardFailureError(
            f"Hard failure in {operation} for agent {self.agent_id}: {error}"
        ) from error

    def reset_strategy(self, strategy_name: str) -> None:
        """Hard failure mode - no recovery strategies exist."""
        raise NotImplementedError(
            "Hard failure mode does not support recovery strategies"
        )

    def reset_all_strategies(self) -> None:
        """Hard failure mode - no recovery strategies exist."""
        raise NotImplementedError(
            "Hard failure mode does not support recovery strategies"
        )

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics - minimal information since we fail fast."""
        return {
            "total_errors": self.error_count,
            "mode": "hard_failure",
            "message": "Errors result in immediate failures",
        }


class PyMDPErrorHandlerHardFailure:
    """Hard failure PyMDP error handler - no fallback execution."""

    def __init__(self, agent_id: str, max_recovery_attempts: int = 0) -> None:
        """Initialize hard failure PyMDP error handler.

        Args:
            agent_id: Unique identifier for the agent
            max_recovery_attempts: Ignored in hard failure mode (always 0)
        """
        self.agent_id = agent_id
        # Hard failure mode ignores recovery attempts
        self.max_recovery_attempts = 0
        self.error_count = 0

    def safe_execute(
        self,
        operation_name: str,
        operation_func: Callable[[], Any],
        fallback_func: Optional[Callable[[], Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Any, Optional[Any]]:
        """Execute operation with immediate failure on error - no safe execution."""
        self.error_count += 1

        # HARD FAILURE VALIDATION: Verify operation is callable
        if not callable(operation_func):
            raise ValueError(f"Operation {operation_name} must be callable")

        # HARD FAILURE: Execute operation, let any exceptions propagate immediately
        try:
            result = operation_func()
            return True, result, None
        except Exception as e:
            logging.error(
                f"PyMDP operation {operation_name} failed for {self.agent_id}: {e}"
            )

            # HARD FAILURE: No fallback execution, raise immediately
            raise HardFailureError(
                f"PyMDP operation {operation_name} failed for agent {self.agent_id}: {e}"
            ) from e

    def get_error_report(self) -> Dict[str, Any]:
        """Get error report - minimal since we fail fast."""
        return {
            "agent_id": self.agent_id,
            "total_operations": self.error_count,
            "mode": "hard_failure",
            "message": "All errors result in immediate failures",
        }

    def reset_stats(self) -> None:
        """Reset statistics counter."""
        self.error_count = 0


# Hard failure exception classes
class ActionSelectionErrorHardFailure(Exception):
    """Hard failure for action selection errors."""


class InferenceErrorHardFailure(Exception):
    """Hard failure for inference errors."""


def safe_array_index_hard_failure(
    array: np.ndarray, index: int, default: Any = None
) -> Any:
    """Array indexing with assertion-based bounds checking - no graceful fallbacks."""
    # HARD FAILURE VALIDATION: Array must be valid numpy array
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(array)}")
    if array.size == 0:
        raise ValueError("Array cannot be empty")

    # HARD FAILURE VALIDATION: Index must be within bounds
    if not (0 <= index < len(array)):
        raise IndexError(
            f"Index {index} out of bounds for array of length {len(array)}"
        )

    # Direct access with no fallback
    return array[index]


def safe_pymdp_operation_hard_failure(
    operation_name: str, default_value: Optional[Any] = None
):
    """Enforce hard failures for PyMDP operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # HARD FAILURE VALIDATION: Function must be executable
            if not callable(func):
                raise TypeError(
                    f"PyMDP operation {operation_name} must be callable"
                )

            # HARD FAILURE: Execute with no try/catch, let exceptions propagate
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"PyMDP operation {operation_name} failed: {e}")
                # HARD FAILURE: Raise immediately, no default values
                raise HardFailureError(
                    f"PyMDP operation {operation_name} failed: {e}"
                ) from e

        return wrapper

    return decorator


def validate_observation_hard_failure(observation: Any) -> Any:
    """Observation validation with assertion-based checks."""
    # HARD FAILURE VALIDATION: Observation must exist
    if observation is None:
        raise ValueError("Observation cannot be None")

    # HARD FAILURE VALIDATION: If numpy array, must be valid
    if isinstance(observation, np.ndarray):
        if observation.size == 0:
            raise ValueError("Observation array cannot be empty")
        if np.any(np.isnan(observation)):
            raise ValueError("Observation contains NaN values")
        if np.any(np.isinf(observation)):
            raise ValueError("Observation contains infinite values")

    return observation


def validate_pymdp_matrices_hard_failure(
    A: Any, B: Any, C: Any, D: Any
) -> Tuple[bool, str]:
    """Validate PyMDP matrices with assertion-based checks."""
    # HARD FAILURE VALIDATION: All matrices must be provided
    if A is None:
        raise ValueError("A matrix cannot be None")
    if B is None:
        raise ValueError("B matrix cannot be None")
    if C is None:
        raise ValueError("C matrix cannot be None")
    if D is None:
        raise ValueError("D matrix cannot be None")

    # HARD FAILURE VALIDATION: Must be numpy arrays
    if not isinstance(A, (list, np.ndarray)):
        raise TypeError(f"A matrix must be array-like, got {type(A)}")
    if not isinstance(B, (list, np.ndarray)):
        raise TypeError(f"B matrix must be array-like, got {type(B)}")
    if not isinstance(C, (list, np.ndarray)):
        raise TypeError(f"C matrix must be array-like, got {type(C)}")
    if not isinstance(D, (list, np.ndarray)):
        raise TypeError(f"D matrix must be array-like, got {type(D)}")

    # If validation passes, return success (no graceful degradation)
    return True, "All matrices validated successfully"


def with_error_handling_hard_failure(
    operation_name: str, fallback_result: Optional[Any] = None
):
    """Enforce hard failures instead of fallbacks for error handling."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # HARD FAILURE VALIDATION: Function must be callable
            if not callable(func):
                raise TypeError(f"Operation {operation_name} must be callable")

            # HARD FAILURE: Execute with no try/catch for graceful degradation
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Operation {operation_name} failed: {e}")
                # HARD FAILURE: Raise immediately, ignore fallback_result
                raise HardFailureError(
                    f"Operation {operation_name} failed: {e}"
                ) from e

        return wrapper

    return decorator


# Module-level hard failure assignments (replaces fallback handlers)
ErrorHandler = ErrorHandlerHardFailure
PyMDPErrorHandler = PyMDPErrorHandlerHardFailure
ActionSelectionError = ActionSelectionErrorHardFailure
InferenceError = InferenceErrorHardFailure
safe_array_index = safe_array_index_hard_failure
safe_pymdp_operation = safe_pymdp_operation_hard_failure
validate_observation = validate_observation_hard_failure
validate_pymdp_matrices = validate_pymdp_matrices_hard_failure
with_error_handling = with_error_handling_hard_failure

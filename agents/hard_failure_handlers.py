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
        self.agent_id = agent_id
        self.error_count: int = 0

    def handle_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle an error by raising it immediately - no recovery information."""
        self.error_count += 1
        logging.error(f"Agent {self.agent_id} hard failure in {operation}: {error}")

        # HARD FAILURE: Raise the original error immediately, no graceful degradation
        raise HardFailureError(
            f"Hard failure in {operation} for agent {self.agent_id}: {error}"
        ) from error

    def reset_strategy(self, strategy_name: str) -> None:
        """Hard failure mode - no recovery strategies exist."""
        raise NotImplementedError("Hard failure mode does not support recovery strategies")

    def reset_all_strategies(self) -> None:
        """Hard failure mode - no recovery strategies exist."""
        raise NotImplementedError("Hard failure mode does not support recovery strategies")

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

        # ASSERTION-BASED VALIDATION: Verify operation is callable
        assert callable(operation_func), f"Operation {operation_name} must be callable"

        # HARD FAILURE: Execute operation, let any exceptions propagate immediately
        try:
            result = operation_func()
            return True, result, None
        except Exception as e:
            logging.error(f"PyMDP operation {operation_name} failed for {self.agent_id}: {e}")

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


def safe_array_index_hard_failure(array: np.ndarray, index: int, default: Any = None) -> Any:
    """Array indexing with assertion-based bounds checking - no graceful fallbacks."""
    # ASSERTION-BASED VALIDATION: Array must be valid numpy array
    assert isinstance(array, np.ndarray), f"Expected numpy array, got {type(array)}"
    assert array.size > 0, "Array cannot be empty"

    # ASSERTION-BASED VALIDATION: Index must be within bounds
    assert 0 <= index < len(array), f"Index {index} out of bounds for array of length {len(array)}"

    # Direct access with no fallback
    return array[index]


def safe_pymdp_operation_hard_failure(operation_name: str, default_value: Optional[Any] = None):
    """Decorator for PyMDP operations that enforces hard failures."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # ASSERTION-BASED VALIDATION: Function must be executable
            assert callable(func), f"PyMDP operation {operation_name} must be callable"

            # HARD FAILURE: Execute with no try/catch, let exceptions propagate
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"PyMDP operation {operation_name} failed: {e}")
                # HARD FAILURE: Raise immediately, no default values
                raise HardFailureError(f"PyMDP operation {operation_name} failed: {e}") from e

        return wrapper

    return decorator


def validate_observation_hard_failure(observation: Any) -> Any:
    """Observation validation with assertion-based checks."""
    # ASSERTION-BASED VALIDATION: Observation must exist
    assert observation is not None, "Observation cannot be None"

    # ASSERTION-BASED VALIDATION: If numpy array, must be valid
    if isinstance(observation, np.ndarray):
        assert observation.size > 0, "Observation array cannot be empty"
        assert not np.any(np.isnan(observation)), "Observation contains NaN values"
        assert not np.any(np.isinf(observation)), "Observation contains infinite values"

    return observation


def validate_pymdp_matrices_hard_failure(A: Any, B: Any, C: Any, D: Any) -> Tuple[bool, str]:
    """PyMDP matrix validation with assertion-based checks."""
    # ASSERTION-BASED VALIDATION: All matrices must be provided
    assert A is not None, "A matrix cannot be None"
    assert B is not None, "B matrix cannot be None"
    assert C is not None, "C matrix cannot be None"
    assert D is not None, "D matrix cannot be None"

    # ASSERTION-BASED VALIDATION: Must be numpy arrays
    assert isinstance(A, (list, np.ndarray)), f"A matrix must be array-like, got {type(A)}"
    assert isinstance(B, (list, np.ndarray)), f"B matrix must be array-like, got {type(B)}"
    assert isinstance(C, (list, np.ndarray)), f"C matrix must be array-like, got {type(C)}"
    assert isinstance(D, (list, np.ndarray)), f"D matrix must be array-like, got {type(D)}"

    # If validation passes, return success (no graceful degradation)
    return True, "All matrices validated successfully"


def with_error_handling_hard_failure(operation_name: str, fallback_result: Optional[Any] = None):
    """Decorator for error handling that enforces hard failures instead of fallbacks."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # ASSERTION-BASED VALIDATION: Function must be callable
            assert callable(func), f"Operation {operation_name} must be callable"

            # HARD FAILURE: Execute with no try/catch for graceful degradation
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Operation {operation_name} failed: {e}")
                # HARD FAILURE: Raise immediately, ignore fallback_result
                raise HardFailureError(f"Operation {operation_name} failed: {e}") from e

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

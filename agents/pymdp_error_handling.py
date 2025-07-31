"""Comprehensive PyMDP error handling and recovery system.

This module provides robust error handling for PyMDP operations that can cause
production failures. It implements graceful degradation and recovery strategies
for common PyMDP edge cases.
"""

import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PyMDPErrorType(Enum):
    """Classification of PyMDP error types for targeted handling."""

    NUMPY_CONVERSION = "numpy_conversion"  # unhashable numpy array errors
    MATRIX_DIMENSION = "matrix_dimension"  # dimension/shape mismatches
    INFERENCE_CONVERGENCE = "inference_convergence"  # convergence failures
    POLICY_SAMPLING = "policy_sampling"  # policy sampling errors
    BELIEF_UPDATE = "belief_update"  # belief update failures
    INDEX_ERROR = "index_error"  # array/dict indexing errors
    NUMERICAL_INSTABILITY = "numerical_instability"  # NaN/inf values
    UNKNOWN = "unknown"  # unclassified errors


class PyMDPError(Exception):
    """Custom exception for PyMDP-related errors with recovery context."""

    def __init__(
        self,
        error_type: PyMDPErrorType,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize PyMDP error with type classification and context.

        Args:
            error_type: Type of PyMDP error
            original_error: The original exception that was caught
            context: Additional context for debugging
        """
        self.error_type = error_type
        self.original_error = original_error
        self.context = context or {}
        super().__init__(f"PyMDP Error [{error_type.value}]: {str(original_error)}")


class PyMDPErrorHandler:
    """Production-grade error handling for PyMDP operations."""

    def __init__(self, agent_id: str, max_recovery_attempts: int = 3):
        """Initialize PyMDP error handler with recovery capabilities.

        Args:
            agent_id: Unique identifier for the agent
            max_recovery_attempts: Maximum attempts before giving up
        """
        self.agent_id = agent_id
        self.max_recovery_attempts = max_recovery_attempts
        self.error_count = 0
        self.operation_failures: Dict[str, int] = {}
        self.recovery_stats: Dict[str, Any] = {}

    def execute_with_error_context(
        self,
        operation_name: str,
        operation_func: Callable,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute PyMDP operation with error context but no fallbacks.

        Args:
            operation_name: Name of the operation for logging/tracking
            operation_func: Primary operation to execute
            context: Additional context for error analysis

        Returns:
            Operation result - raises exception on failure

        Raises:
            PyMDPError: If operation fails with detailed context
        """
        context = context or {}
        context["agent_id"] = self.agent_id

        # Track operation attempts
        attempt_count = self.operation_failures.get(operation_name, 0) + 1
        self.operation_failures[operation_name] = attempt_count

        try:
            # Execute operation
            result = operation_func()

            # Success - reset failure count for this operation
            if operation_name in self.operation_failures:
                self.operation_failures[operation_name] = 0

            return result

        except Exception as e:
            self.error_count += 1
            error_type = self._classify_error(e)

            # Create detailed error context
            error_context = {
                **context,
                "operation": operation_name,
                "attempt_number": attempt_count,
                "total_errors": self.error_count,
                "error_type": error_type.value,
            }

            pymdp_error = PyMDPError(error_type, e, error_context)

            logger.error(
                f"PyMDP operation '{operation_name}' failed (attempt {attempt_count}): {pymdp_error}"
            )

            # Hard failure - raise exception with context
            raise pymdp_error

    def _classify_error(self, error: Exception) -> PyMDPErrorType:
        """Classify PyMDP errors for appropriate handling strategies."""
        error_msg = str(error).lower()
        error_type_name = type(error).__name__.lower()

        # Numpy conversion errors (most common production issue)
        if "unhashable" in error_msg and ("numpy" in error_msg or "array" in error_msg):
            return PyMDPErrorType.NUMPY_CONVERSION

        # Matrix dimension mismatches
        if any(
            keyword in error_msg for keyword in ["dimension", "shape", "broadcasting", "matmul"]
        ):
            return PyMDPErrorType.MATRIX_DIMENSION

        # Convergence issues
        if any(
            keyword in error_msg
            for keyword in [
                "convergence",
                "iteration",
                "max_iter",
                "tolerance",
            ]
        ):
            return PyMDPErrorType.INFERENCE_CONVERGENCE

        # Policy sampling errors
        if any(
            keyword in error_msg
            for keyword in [
                "policy",
                "sampling",
                "sample_action",
                "infer_policies",
            ]
        ):
            return PyMDPErrorType.POLICY_SAMPLING

        # Belief update errors
        if any(
            keyword in error_msg for keyword in ["belie", "update", "infer_states", "posterior"]
        ):
            return PyMDPErrorType.BELIEF_UPDATE

        # Indexing errors
        if (
            any(keyword in error_type_name for keyword in ["indexerror", "keyerror"])
            or "index" in error_msg
        ):
            return PyMDPErrorType.INDEX_ERROR

        # Numerical instability
        if any(
            keyword in error_msg
            for keyword in [
                "nan",
                "in",
                "overflow",
                "underflow",
                "divide by zero",
            ]
        ):
            return PyMDPErrorType.NUMERICAL_INSTABILITY

        return PyMDPErrorType.UNKNOWN

    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report for monitoring."""
        return {
            "agent_id": self.agent_id,
            "total_errors": self.error_count,
            "operation_failures": dict(self.operation_failures),
            "successful_recoveries": dict(self.recovery_stats),
            "error_rate": self.error_count / max(sum(self.operation_failures.values()), 1),
            "recovery_rate": sum(self.recovery_stats.values()) / max(self.error_count, 1),
        }

    def reset_stats(self) -> None:
        """Reset error tracking statistics."""
        self.error_count = 0
        self.operation_failures.clear()
        self.recovery_stats.clear()


def strict_numpy_conversion(value: Any, target_type: type = int) -> Any:
    """Strictly convert numpy arrays/scalars to Python primitives with no fallbacks.

    This function provides strict conversion for PyMDP return values that might be 
    numpy arrays instead of scalars. Failures raise exceptions rather than returning defaults.

    Args:
        value: Value to convert (numpy array, scalar, or other)
        target_type: Target Python type (int, float, str, bool)

    Returns:
        Converted value

    Raises:
        ValueError: If conversion fails or value format is invalid
    """
    try:
        # Handle numpy scalars and 0-d arrays
        if hasattr(value, "item"):
            return target_type(value.item())

        # Handle numpy arrays and array-like objects
        elif hasattr(value, "__getitem__") and hasattr(value, "__len__"):
            if len(value) == 0:
                raise ValueError(f"Empty array cannot be converted to {target_type.__name__}")
            elif len(value) == 1:
                # Single element - extract properly
                if hasattr(value, "item"):
                    return target_type(value.item())
                else:
                    return target_type(value[0])
            else:
                # Multi-element array - this is likely an error in PyMDP usage
                raise ValueError(
                    f"Multi-element array {value} cannot be converted to scalar {target_type.__name__}. "
                    f"This may indicate incorrect PyMDP API usage."
                )

        # Handle regular Python types
        elif isinstance(value, (int, float, str, bool, np.number)):
            return target_type(value)

        # Handle None - not allowed
        elif value is None:
            raise ValueError(f"None value cannot be converted to {target_type.__name__}")

        # Unknown type - attempt conversion with strict error handling
        else:
            return target_type(value)

    except (ValueError, TypeError, IndexError, OverflowError) as e:
        raise ValueError(
            f"Failed to convert {type(value)} value {value} to {target_type.__name__}: {e}"
        )


def strict_array_index(array: Any, index: Any) -> Any:
    """Strictly index into arrays with no fallbacks.

    Args:
        array: Array-like object to index
        index: Index (may be numpy array)

    Returns:
        Indexed value

    Raises:
        ValueError: If indexing fails or array is not indexable
        IndexError: If index is out of bounds
    """
    try:
        # Convert index to Python int if it's a numpy type
        safe_index = strict_numpy_conversion(index, int)

        # Perform indexing
        if hasattr(array, "__getitem__"):
            return array[safe_index]
        else:
            raise ValueError(f"Object of type {type(array)} is not indexable")

    except (IndexError, KeyError, TypeError) as e:
        raise IndexError(f"Failed to index array {type(array)} with index {index}: {e}")


def validate_pymdp_matrices(A: Any, B: Any, C: Any, D: Any) -> Tuple[bool, str]:
    """Validate PyMDP matrix dimensions and properties.

    Args:
        A: Observation model
        B: Transition model
        C: Preference vector
        D: Prior beliefs

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    try:
        # Convert to numpy arrays if not already
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)

        # Check for NaN or inf values
        for name, matrix in [("A", A), ("B", B), ("C", C), ("D", D)]:
            if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
                return False, f"Matrix {name} contains NaN or inf values"

        # Check basic dimension requirements
        if A.ndim != 2:
            return False, f"A matrix must be 2D, got {A.ndim}D"
        if B.ndim != 3:
            return False, f"B matrix must be 3D, got {B.ndim}D"
        if C.ndim != 1:
            return False, f"C vector must be 1D, got {C.ndim}D"
        if D.ndim != 1:
            return False, f"D vector must be 1D, got {D.ndim}D"

        # Check dimension consistency
        num_obs, num_states = A.shape
        if B.shape[0] != num_states or B.shape[1] != num_states:
            return (
                False,
                f"B matrix dimensions {B.shape} inconsistent with A matrix {A.shape}",
            )
        if C.shape[0] != num_obs:
            return (
                False,
                f"C vector length {C.shape[0]} inconsistent with A matrix observations {num_obs}",
            )
        if D.shape[0] != num_states:
            return (
                False,
                f"D vector length {D.shape[0]} inconsistent with A matrix states {num_states}",
            )

        # Check probability constraints
        if not np.allclose(A.sum(axis=0), 1.0, rtol=1e-3):
            return (
                False,
                "A matrix columns must sum to 1 (observation probabilities)",
            )
        if not np.allclose(B.sum(axis=0), 1.0, rtol=1e-3):
            return (
                False,
                "B matrix must have transition probabilities summing to 1",
            )
        if not np.allclose(D.sum(), 1.0, rtol=1e-3):
            return False, "D vector must sum to 1 (prior probabilities)"

        return True, "Matrices are valid"

    except Exception as e:
        return False, f"Matrix validation failed: {e}"


# Strict conversion function for backward compatibility
def strict_array_to_int(value: Any) -> int:
    """Convert array/scalar to int with no fallbacks."""
    return strict_numpy_conversion(value, int)

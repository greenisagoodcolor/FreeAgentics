"""Fallback implementations for error handlers when main modules fail to import."""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


class ErrorHandlerFallback:
    """Fallback error handler implementation."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.error_count: int = 0

    def handle_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle an error and return recovery information."""
        self.error_count += 1
        logging.error(f"Agent {self.agent_id} error in {operation}: {error}")

        # Return minimal recovery response
        return {
            "can_retry": False,
            "fallback_action": "stay",
            "severity": "medium",
            "strategy": "general_failure",
            "retry_count": 0,
            "max_retries": 0,
        }

    def reset_strategy(self, strategy_name: str) -> None:
        """Reset a specific recovery strategy (no-op in fallback)."""

    def reset_all_strategies(self) -> None:
        """Reset all recovery strategies (no-op in fallback)."""

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": self.error_count,
            "errors_by_type": {},
            "errors_by_operation": {},
            "recent_errors": [],
        }


class PyMDPErrorHandlerFallback:
    """Fallback PyMDP error handler implementation."""

    def __init__(self, agent_id: str, max_recovery_attempts: int = 3) -> None:
        self.agent_id = agent_id
        self.max_recovery_attempts = max_recovery_attempts
        self.error_count: int = 0

    def safe_execute(
        self,
        operation_name: str,
        operation_func: Callable[[], Any],
        fallback_func: Optional[Callable[[], Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Any, Optional[Any]]:
        """Execute operation with fallback handling."""
        try:
            result = operation_func()
            return True, result, None
        except Exception as e:
            self.error_count += 1
            logging.error(f"PyMDP operation {operation_name} failed for {self.agent_id}: {e}")

            if fallback_func:
                try:
                    fallback_result = fallback_func()
                    return False, fallback_result, e
                except Exception as fallback_error:
                    logging.error(f"Fallback also failed: {fallback_error}")
                    return False, None, e
            else:
                return False, None, e

    def get_error_report(self) -> Dict[str, Any]:
        """Get error report."""
        return {
            "agent_id": self.agent_id,
            "total_errors": self.error_count,
            "operation_failures": {},
            "recovery_stats": {},
        }

    def reset_stats(self) -> None:
        """Reset error statistics."""
        self.error_count = 0

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics (for compatibility)."""
        return {"pymdp_errors": self.error_count}


class ActionSelectionErrorFallback(Exception):
    """Fallback for action selection errors."""


class InferenceErrorFallback(Exception):
    """Fallback for inference errors."""


def safe_array_index_fallback(array: np.ndarray, index: int, default: Any = 0) -> Any:
    """Safe array indexing with bounds checking."""
    try:
        if 0 <= index < len(array):
            return array[index]
        return default
    except Exception:
        return default


def safe_pymdp_operation_fallback(operation_name: str, default_value: Optional[Any] = None):
    """Decorator for safe PyMDP operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"PyMDP operation {operation_name} failed: {e}")
                return default_value

        return wrapper

    return decorator


def validate_observation_fallback(observation: Any) -> Any:
    """Basic observation validation."""
    return observation


def validate_pymdp_matrices_fallback(A: Any, B: Any, C: Any, D: Any) -> Tuple[bool, str]:
    """Basic matrix validation."""
    try:
        # Basic validation - just check if they exist
        if A is None or B is None or C is None or D is None:
            return False, "One or more matrices is None"
        return True, "Validation passed"
    except Exception as e:
        return False, f"Validation error: {e}"


def with_error_handling_fallback(operation_name: str, fallback_result: Optional[Any] = None):
    """Decorator for error handling with fallback result."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Operation {operation_name} failed: {e}")
                return fallback_result

        return wrapper

    return decorator


# Module-level fallback assignments
ErrorHandler = ErrorHandlerFallback
PyMDPErrorHandler = PyMDPErrorHandlerFallback
ActionSelectionError = ActionSelectionErrorFallback
InferenceError = InferenceErrorFallback
safe_array_index = safe_array_index_fallback
safe_pymdp_operation = safe_pymdp_operation_fallback
validate_observation = validate_observation_fallback
validate_pymdp_matrices = validate_pymdp_matrices_fallback
with_error_handling = with_error_handling_fallback

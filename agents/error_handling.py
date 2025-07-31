"""Error handling utilities for PyMDP operations and agent failures."""

import logging
import traceback
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for agent errors."""

    LOW = "low"  # Minor issues, agent can continue
    MEDIUM = "medium"  # Moderate issues, degraded performance
    HIGH = "high"  # Serious issues, fallback required
    CRITICAL = "critical"  # Agent must stop, manual intervention needed


class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize agent error with severity and context.

        Args:
            message: Error message
            severity: Error severity level
            context: Additional error context
        """
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()


class PyMDPError(AgentError):
    """Specific error for PyMDP-related failures."""


class InferenceError(AgentError):
    """Error during inference operations."""


class ActionSelectionError(AgentError):
    """Error during action selection."""


class ErrorRecoveryStrategy:
    """Strategy for recovering from different types of errors."""

    def __init__(
        self,
        name: str,
        fallback_action: str = "stay",
        max_retries: int = 3,
        cooldown_seconds: int = 5,
    ):
        """Initialize error recovery strategy.

        Args:
            name: Strategy name
            fallback_action: Default action when recovery fails
            max_retries: Maximum retry attempts
            cooldown_seconds: Seconds to wait between retries
        """
        self.name = name
        self.fallback_action = fallback_action
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.retry_count = 0
        self.last_error_time = None

    def can_retry(self) -> bool:
        """Check if we can retry based on retry limits and cooldown."""
        if self.retry_count >= self.max_retries:
            return False

        if self.last_error_time:
            time_since_error = (datetime.now() - self.last_error_time).total_seconds()
            if time_since_error < self.cooldown_seconds:
                return False

        return True

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.retry_count += 1
        self.last_error_time = datetime.now()

    def reset(self) -> None:
        """Reset retry counter after successful operation."""
        self.retry_count = 0
        self.last_error_time = None


class ErrorHandler:
    """Centralized error handling for agent operations."""

    def __init__(self, agent_id: str):
        """Initialize error handler for an agent.

        Args:
            agent_id: ID of the agent to handle errors for
        """
        self.agent_id = agent_id
        self.error_history = []
        self.recovery_strategies = {
            "pymdp_failure": ErrorRecoveryStrategy(
                name="PyMDP Failure",
                fallback_action="stay",
                max_retries=3,
                cooldown_seconds=10,
            ),
            "inference_failure": ErrorRecoveryStrategy(
                name="Inference Failure",
                fallback_action="stay",
                max_retries=5,
                cooldown_seconds=5,
            ),
            "action_selection_failure": ErrorRecoveryStrategy(
                name="Action Selection Failure",
                fallback_action="stay",
                max_retries=2,
                cooldown_seconds=2,
            ),
            "general_failure": ErrorRecoveryStrategy(
                name="General Failure",
                fallback_action="stay",
                max_retries=1,
                cooldown_seconds=1,
            ),
        }

    def handle_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle an error and determine recovery strategy.

        Args:
            error: The exception that occurred
            operation: Name of the operation that failed

        Returns:
            Recovery information including fallback action
        """
        # Classify error type
        if "pymdp" in str(error).lower() or isinstance(error, PyMDPError):
            strategy_key = "pymdp_failure"
            severity = ErrorSeverity.HIGH
        elif isinstance(error, InferenceError):
            strategy_key = "inference_failure"
            severity = ErrorSeverity.MEDIUM
        elif isinstance(error, ActionSelectionError):
            strategy_key = "action_selection_failure"
            severity = ErrorSeverity.MEDIUM
        else:
            strategy_key = "general_failure"
            severity = ErrorSeverity.LOW

        strategy = self.recovery_strategies[strategy_key]

        # Record error in history
        error_record = {
            "timestamp": datetime.now(),
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity,
            "strategy": strategy_key,
            "traceback": traceback.format_exc(),
        }
        self.error_history.append(error_record)

        # Log error
        logger.error(
            f"Agent {self.agent_id} error in {operation}: {error} "
            f"(severity: {severity.value}, strategy: {strategy_key})"
        )

        # Determine recovery action
        recovery_info = {
            "can_retry": strategy.can_retry(),
            "fallback_action": strategy.fallback_action,
            "severity": severity,
            "strategy_name": strategy.name,
            "retry_count": strategy.retry_count,
            "error_record": error_record,
        }

        strategy.record_error()

        return recovery_info

    def record_success(self, operation: str) -> None:
        """Record successful operation to reset retry counters."""
        # Reset relevant strategies
        for strategy in self.recovery_strategies.values():
            strategy.reset()

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_history:
            return {"total_errors": 0, "recent_errors": []}

        recent_errors = self.error_history[-10:]  # Last 10 errors

        # Count by type
        error_counts = {}
        for error in recent_errors:
            error_type = error["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "recent_errors": recent_errors,
            "error_counts": error_counts,
            "last_error": self.error_history[-1] if self.error_history else None,
        }


def with_error_handling(operation_name: str, fallback_result: Any = None) -> Callable:
    """Add error handling to agent methods.

    Args:
        operation_name: Name of the operation for logging
        fallback_result: Value to return if all recovery fails
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Ensure agent has error handler
            if not hasattr(self, "error_handler"):
                self.error_handler = ErrorHandler(self.agent_id)

            try:
                result = func(self, *args, **kwargs)
                # Record success
                self.error_handler.record_success(operation_name)
                return result

            except Exception as e:
                # Handle error
                recovery_info = self.error_handler.handle_error(e, operation_name)

                # Check if we can retry
                if recovery_info["can_retry"]:
                    try:
                        # Attempt retry with simplified parameters
                        logger.info(f"Retrying {operation_name} for agent {self.agent_id}")
                        if hasattr(self, f"_fallback_{func.__name__}"):
                            # Use fallback method if available
                            fallback_method = getattr(self, f"_fallback_{func.__name__}")
                            return fallback_method(*args, **kwargs)
                        else:
                            # Use default fallback result
                            return fallback_result
                    except Exception as retry_error:
                        logger.error(f"Retry failed for {operation_name}: {retry_error}")

                # Return fallback result
                return fallback_result

        return wrapper

    return decorator


def safe_pymdp_operation(operation_name: str, default_value: Any = None) -> Callable:
    """Add PyMDP-specific error handling to methods.

    Args:
        operation_name: Name of the PyMDP operation
        default_value: Default value to return on failure
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "error_handler"):
                self.error_handler = ErrorHandler(self.agent_id)

            try:
                # Check if PyMDP is available and agent is initialized
                if not hasattr(self, "pymdp_agent") or self.pymdp_agent is None:
                    raise PyMDPError("PyMDP agent not initialized")

                result = func(self, *args, **kwargs)
                self.error_handler.record_success(operation_name)
                return result

            except Exception as e:
                # Convert to PyMDPError for proper handling
                if not isinstance(e, PyMDPError):
                    e = PyMDPError(f"PyMDP operation failed: {str(e)}")

                recovery_info = self.error_handler.handle_error(e, operation_name)
                logger.debug(f"Recovery info for {operation_name}: {recovery_info}")

                # Try fallback method first
                fallback_method_name = f"_fallback_{func.__name__}"
                if hasattr(self, fallback_method_name):
                    try:
                        logger.info(
                            f"Using fallback method {fallback_method_name} for {operation_name}"
                        )
                        fallback_method = getattr(self, fallback_method_name)
                        return fallback_method(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback method {fallback_method_name} failed: {fallback_error}"
                        )

                # Log specific PyMDP error details
                logger.warning(
                    f"PyMDP operation '{operation_name}' failed for agent {self.agent_id}: {e}\n"
                    f"Falling back to default value: {default_value}"
                )

                return default_value

        return wrapper

    return decorator


# Utility functions for common error patterns


def validate_observation(observation: Any) -> Dict[str, Any]:
    """Validate and sanitize observation data."""
    if observation is None:
        return {"position": [0, 0], "valid": False}

    if isinstance(observation, dict):
        # Ensure required fields exist
        sanitized = {
            "position": observation.get("position", [0, 0]),
            "valid": True,
        }

        # Copy other valid fields
        for key, value in observation.items():
            if key not in ["position"] and isinstance(value, (int, float, str, list, dict)):
                sanitized[key] = value

        return sanitized
    else:
        # Convert non-dict observations
        return {"observation": observation, "position": [0, 0], "valid": True}


def validate_action(action: Any, valid_actions: list) -> str:
    """Validate and sanitize action output."""
    if action is None:
        return "stay"

    if isinstance(action, str) and action in valid_actions:
        return action
    elif isinstance(action, (int, float)):
        # Convert numeric action to string
        action_idx = int(action) % len(valid_actions)
        return valid_actions[action_idx]
    else:
        # Invalid action, return safe default
        return "stay"


def handle_agent_error(error: AgentError, context: Optional[Dict[str, Any]] = None) -> None:
    """Handle agent errors with logging and optional context.

    Args:
        error: The AgentError to handle
        context: Optional context dict for additional error information
    """
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
    }

    logger.error(f"Agent error handled: {error}", extra=error_info)

    # For compatibility with tests, return None
    return None

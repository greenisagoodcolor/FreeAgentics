"""Orchestration Error Types.

Standardized error hierarchy for conversation orchestration with proper
categorization, context preservation, and actionable error messages.
"""

import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ErrorContext:
    """Context information for orchestration errors."""
    
    trace_id: str
    conversation_id: Optional[str] = None
    step_name: Optional[str] = None
    component: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "step_name": self.step_name,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
        }


class OrchestrationError(Exception):
    """Base exception for orchestration-related errors.
    
    Provides structured error context with tracing information,
    component identification, and actionable error messages.
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = False,
        suggested_action: Optional[str] = None,
    ):
        super().__init__(message)
        self.context = context
        self.cause = cause
        self.recoverable = recoverable
        self.suggested_action = suggested_action
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "recoverable": self.recoverable,
            "suggested_action": self.suggested_action,
            "context": self.context.to_dict() if self.context else None,
            "cause": str(self.cause) if self.cause else None,
        }


class ComponentTimeoutError(OrchestrationError):
    """Raised when a component operation times out."""
    
    def __init__(
        self,
        component: str,
        timeout_ms: float,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"{component} operation timed out after {timeout_ms:.1f}ms"
        suggested_action = f"Consider increasing timeout or checking {component} health"
        super().__init__(
            message=message,
            context=context,
            cause=cause,
            recoverable=True,  # Timeouts are often recoverable with retry
            suggested_action=suggested_action,
        )
        self.component = component
        self.timeout_ms = timeout_ms


class ValidationError(OrchestrationError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        field: str,
        value: Any,
        validation_rule: str,
        context: Optional[ErrorContext] = None,
    ):
        message = f"Validation failed for {field}: {validation_rule}. Got: {value}"
        suggested_action = f"Correct the {field} value and retry"
        super().__init__(
            message=message,
            context=context,
            recoverable=False,  # Input validation errors are not recoverable without changes
            suggested_action=suggested_action,
        )
        self.field = field
        self.value = value
        self.validation_rule = validation_rule


class PipelineExecutionError(OrchestrationError):
    """Raised when pipeline execution fails at a specific step."""
    
    def __init__(
        self,
        step_name: str,
        step_index: int,
        total_steps: int,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Pipeline failed at step {step_index + 1}/{total_steps}: {step_name}"
        suggested_action = f"Check {step_name} configuration and dependencies"
        super().__init__(
            message=message,
            context=context,
            cause=cause,
            recoverable=True,  # Pipeline errors may be recoverable with retry/fallback
            suggested_action=suggested_action,
        )
        self.step_name = step_name
        self.step_index = step_index
        self.total_steps = total_steps


class FallbackError(OrchestrationError):
    """Raised when all fallback options have been exhausted."""
    
    def __init__(
        self,
        primary_error: Exception,
        fallback_errors: List[Exception],
        context: Optional[ErrorContext] = None,
    ):
        fallback_count = len(fallback_errors)
        message = f"Primary operation failed and all {fallback_count} fallback options exhausted"
        suggested_action = "Check system health and component availability"
        super().__init__(
            message=message,
            context=context,
            cause=primary_error,
            recoverable=False,  # All options exhausted
            suggested_action=suggested_action,
        )
        self.primary_error = primary_error
        self.fallback_errors = fallback_errors


class CircuitBreakerOpenError(OrchestrationError):
    """Raised when circuit breaker is open and requests are being rejected."""
    
    def __init__(
        self,
        component: str,
        failure_count: int,
        failure_threshold: int,
        recovery_time_seconds: float,
        context: Optional[ErrorContext] = None,
    ):
        message = (
            f"Circuit breaker for {component} is OPEN "
            f"({failure_count}/{failure_threshold} failures). "
            f"Recovery in {recovery_time_seconds:.1f}s"
        )
        suggested_action = f"Wait for circuit breaker recovery or check {component} health"
        super().__init__(
            message=message,
            context=context,
            recoverable=True,  # Circuit breakers auto-recover
            suggested_action=suggested_action,
        )
        self.component = component
        self.failure_count = failure_count
        self.failure_threshold = failure_threshold
        self.recovery_time_seconds = recovery_time_seconds


class ResourceExhaustionError(OrchestrationError):
    """Raised when system resources are exhausted."""
    
    def __init__(
        self,
        resource_type: str,
        current_usage: Union[int, float],
        limit: Union[int, float],
        unit: str = "",
        context: Optional[ErrorContext] = None,
    ):
        message = (
            f"{resource_type} exhausted: {current_usage}{unit}/{limit}{unit} "
            f"({current_usage/limit*100:.1f}%)"
        )
        suggested_action = f"Reduce load or increase {resource_type} capacity"
        super().__init__(
            message=message,
            context=context,
            recoverable=True,  # May recover when load decreases
            suggested_action=suggested_action,
        )
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        self.unit = unit


def create_error_context(
    trace_id: str,
    conversation_id: Optional[str] = None,
    step_name: Optional[str] = None,
    component: Optional[str] = None,
    start_time: Optional[float] = None,
    **metadata
) -> ErrorContext:
    """Create error context with timing information."""
    execution_time_ms = None
    if start_time is not None:
        execution_time_ms = (time.time() - start_time) * 1000
    
    return ErrorContext(
        trace_id=trace_id,
        conversation_id=conversation_id,
        step_name=step_name,
        component=component,
        execution_time_ms=execution_time_ms,
        metadata=metadata,
    )


def categorize_error(error: Exception) -> str:
    """Categorize an error for monitoring and alerting."""
    if isinstance(error, ComponentTimeoutError):
        return "timeout"
    elif isinstance(error, ValidationError):
        return "validation"
    elif isinstance(error, CircuitBreakerOpenError):
        return "circuit_breaker"
    elif isinstance(error, ResourceExhaustionError):
        return "resource_exhaustion"
    elif isinstance(error, FallbackError):
        return "fallback_exhausted"
    elif isinstance(error, PipelineExecutionError):
        return "pipeline_execution"
    elif isinstance(error, OrchestrationError):
        return "orchestration"
    else:
        return "unknown"


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    if isinstance(error, OrchestrationError):
        return error.recoverable
    
    # Check for specific retryable error patterns
    error_str = str(error).lower()
    retryable_patterns = [
        "timeout", "connection", "rate limit", "503", "502", "500",
        "temporary", "transient", "unavailable"
    ]
    
    return any(pattern in error_str for pattern in retryable_patterns)


def get_retry_delay(error: Exception, attempt: int) -> float:
    """Get retry delay based on error type and attempt number."""
    if isinstance(error, ComponentTimeoutError):
        # Longer delays for timeout errors
        return min(2.0 ** attempt, 60.0)
    elif isinstance(error, ResourceExhaustionError):
        # Progressive backoff for resource errors
        return min(1.5 ** attempt, 30.0)
    else:
        # Standard exponential backoff
        return min(1.0 * (2 ** attempt), 45.0)
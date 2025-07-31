"""Structured exception handling for Agent Conversation API.

Provides consistent error codes and responses across all conversation endpoints
as recommended by the Nemesis Committee for production readiness.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Standardized error codes for conversation API."""

    # Conversation errors
    CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"
    CONVERSATION_ALREADY_ACTIVE = "CONVERSATION_ALREADY_ACTIVE"
    CONVERSATION_NOT_ACTIVE = "CONVERSATION_NOT_ACTIVE"
    CONVERSATION_CREATION_FAILED = "CONVERSATION_CREATION_FAILED"
    CONVERSATION_INVALID_STATE = "CONVERSATION_INVALID_STATE"

    # Agent errors
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_CREATION_FAILED = "AGENT_CREATION_FAILED"
    AGENT_INITIALIZATION_FAILED = "AGENT_INITIALIZATION_FAILED"
    AGENT_INVALID_CONFIGURATION = "AGENT_INVALID_CONFIGURATION"

    # Message errors
    MESSAGE_CREATION_FAILED = "MESSAGE_CREATION_FAILED"
    MESSAGE_INVALID_ORDER = "MESSAGE_INVALID_ORDER"
    MESSAGE_TOO_LONG = "MESSAGE_TOO_LONG"

    # LLM Service errors
    LLM_SERVICE_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE"
    LLM_QUOTA_EXCEEDED = "LLM_QUOTA_EXCEEDED"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
    GMN_GENERATION_FAILED = "GMN_GENERATION_FAILED"

    # Database errors
    DATABASE_CONNECTION_FAILED = "DATABASE_CONNECTION_FAILED"
    DATABASE_TRANSACTION_FAILED = "DATABASE_TRANSACTION_FAILED"
    DATABASE_CONSTRAINT_VIOLATION = "DATABASE_CONSTRAINT_VIOLATION"

    # Validation errors
    INVALID_REQUEST_FORMAT = "INVALID_REQUEST_FORMAT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FIELD_VALUE = "INVALID_FIELD_VALUE"

    # Authentication/Authorization errors
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"

    # Rate limiting errors
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # General errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class ConversationError(Exception):
    """Base exception for conversation-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id
        self.timestamp = datetime.now()


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation cannot be found."""

    def __init__(self, conversation_id: str, correlation_id: Optional[str] = None):
        super().__init__(
            message=f"Conversation with ID {conversation_id} not found",
            error_code=ErrorCode.CONVERSATION_NOT_FOUND,
            details={"conversation_id": conversation_id},
            correlation_id=correlation_id,
        )


class ConversationStateError(ConversationError):
    """Raised when conversation is in invalid state for operation."""

    def __init__(
        self,
        conversation_id: str,
        current_state: str,
        required_state: str,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Conversation {conversation_id} is in state '{current_state}', but '{required_state}' is required",
            error_code=ErrorCode.CONVERSATION_INVALID_STATE,
            details={
                "conversation_id": conversation_id,
                "current_state": current_state,
                "required_state": required_state,
            },
            correlation_id=correlation_id,
        )


class AgentCreationError(ConversationError):
    """Raised when agent creation fails."""

    def __init__(
        self,
        reason: str,
        agent_config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Failed to create agent: {reason}",
            error_code=ErrorCode.AGENT_CREATION_FAILED,
            details={"reason": reason, "agent_config": agent_config},
            correlation_id=correlation_id,
        )


class LLMServiceError(ConversationError):
    """Raised when LLM service encounters an error."""

    def __init__(
        self,
        service_name: str,
        reason: str,
        retry_possible: bool = True,
        correlation_id: Optional[str] = None,
    ):
        error_code = ErrorCode.LLM_SERVICE_UNAVAILABLE
        if "quota" in reason.lower():
            error_code = ErrorCode.LLM_QUOTA_EXCEEDED
        elif "invalid" in reason.lower():
            error_code = ErrorCode.LLM_INVALID_RESPONSE

        super().__init__(
            message=f"LLM service '{service_name}' error: {reason}",
            error_code=error_code,
            details={
                "service_name": service_name,
                "reason": reason,
                "retry_possible": retry_possible,
            },
            correlation_id=correlation_id,
        )


class DatabaseError(ConversationError):
    """Raised when database operations fail."""

    def __init__(self, operation: str, reason: str, correlation_id: Optional[str] = None):
        error_code = ErrorCode.DATABASE_CONNECTION_FAILED
        if "constraint" in reason.lower():
            error_code = ErrorCode.DATABASE_CONSTRAINT_VIOLATION
        elif "transaction" in reason.lower():
            error_code = ErrorCode.DATABASE_TRANSACTION_FAILED

        super().__init__(
            message=f"Database operation '{operation}' failed: {reason}",
            error_code=error_code,
            details={"operation": operation, "reason": reason},
            correlation_id=correlation_id,
        )


class ValidationError(ConversationError):
    """Raised when request validation fails."""

    def __init__(self, field: str, value: Any, reason: str, correlation_id: Optional[str] = None):
        super().__init__(
            message=f"Validation failed for field '{field}': {reason}",
            error_code=ErrorCode.INVALID_FIELD_VALUE,
            details={"field": field, "value": str(value), "reason": reason},
            correlation_id=correlation_id,
        )


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    correlation_id: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


def create_http_exception(
    error: ConversationError, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
) -> HTTPException:
    """Convert ConversationError to FastAPI HTTPException with structured response."""

    # Map error codes to appropriate HTTP status codes
    status_code_mapping = {
        ErrorCode.CONVERSATION_NOT_FOUND: status.HTTP_404_NOT_FOUND,
        ErrorCode.AGENT_NOT_FOUND: status.HTTP_404_NOT_FOUND,
        ErrorCode.CONVERSATION_ALREADY_ACTIVE: status.HTTP_409_CONFLICT,
        ErrorCode.CONVERSATION_INVALID_STATE: status.HTTP_409_CONFLICT,
        ErrorCode.INVALID_REQUEST_FORMAT: status.HTTP_400_BAD_REQUEST,
        ErrorCode.MISSING_REQUIRED_FIELD: status.HTTP_400_BAD_REQUEST,
        ErrorCode.INVALID_FIELD_VALUE: status.HTTP_400_BAD_REQUEST,
        ErrorCode.UNAUTHORIZED_ACCESS: status.HTTP_401_UNAUTHORIZED,
        ErrorCode.INSUFFICIENT_PERMISSIONS: status.HTTP_403_FORBIDDEN,
        ErrorCode.RATE_LIMIT_EXCEEDED: status.HTTP_429_TOO_MANY_REQUESTS,
        ErrorCode.LLM_QUOTA_EXCEEDED: status.HTTP_429_TOO_MANY_REQUESTS,
        ErrorCode.SERVICE_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCode.LLM_SERVICE_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    }

    http_status = status_code_mapping.get(error.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    error_response = ErrorResponse(
        error_code=error.error_code.value,
        message=error.message,
        details=error.details,
        timestamp=error.timestamp,
        correlation_id=error.correlation_id,
    )

    return HTTPException(status_code=http_status, detail=error_response.dict())


def handle_database_exception(
    e: Exception, operation: str, correlation_id: Optional[str] = None
) -> ConversationError:
    """Convert database exceptions to structured ConversationError."""
    reason = str(e)

    # Handle specific database exceptions
    if "constraint" in reason.lower():
        return DatabaseError(operation, f"Constraint violation: {reason}", correlation_id)
    elif "connection" in reason.lower():
        return DatabaseError(operation, f"Connection failed: {reason}", correlation_id)
    elif "timeout" in reason.lower():
        return DatabaseError(operation, f"Operation timeout: {reason}", correlation_id)
    else:
        return DatabaseError(operation, reason, correlation_id)


def handle_llm_exception(
    e: Exception, service_name: str, correlation_id: Optional[str] = None
) -> ConversationError:
    """Convert LLM service exceptions to structured ConversationError."""
    reason = str(e)

    # Determine if retry is possible
    retry_possible = True
    if any(keyword in reason.lower() for keyword in ["quota", "rate limit", "billing"]):
        retry_possible = False

    return LLMServiceError(service_name, reason, retry_possible, correlation_id)


def handle_validation_exception(
    e: Exception, correlation_id: Optional[str] = None
) -> ConversationError:
    """Convert validation exceptions to structured ConversationError."""
    # Handle Pydantic validation errors
    if hasattr(e, "errors"):
        errors = e.errors()
        if errors:
            first_error = errors[0]
            field = ".".join(str(loc) for loc in first_error.get("loc", []))
            reason = first_error.get("msg", str(e))
            value = first_error.get("input", "")
            return ValidationError(field, value, reason, correlation_id)

    return ConversationError(
        message=f"Validation error: {str(e)}",
        error_code=ErrorCode.INVALID_REQUEST_FORMAT,
        correlation_id=correlation_id,
    )


# Exception handler decorators for common patterns
def handle_conversation_exceptions(correlation_id: Optional[str] = None):
    """Decorator to handle common conversation exceptions."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ConversationError:
                raise  # Re-raise ConversationErrors as-is
            except Exception as e:
                # Convert unexpected exceptions to structured errors
                if "database" in str(e).lower() or "sql" in str(e).lower():
                    raise handle_database_exception(e, func.__name__, correlation_id)
                elif "llm" in str(e).lower() or "openai" in str(e).lower():
                    raise handle_llm_exception(e, "unknown", correlation_id)
                else:
                    raise ConversationError(
                        message=f"Unexpected error in {func.__name__}: {str(e)}",
                        error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                        details={"function": func.__name__, "error_type": type(e).__name__},
                        correlation_id=correlation_id,
                    )

        return wrapper

    return decorator

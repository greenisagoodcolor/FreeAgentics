"""Structured logging configuration for Agent Conversation API.

Provides OpenTelemetry-compatible logging with correlation IDs and structured
fields as recommended by Jessica Kerr for production observability.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class ConversationJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for conversation API logs."""

    def add_fields(
        self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]
    ) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add service identification
        log_record["service"] = "freeagentics-conversation-api"
        log_record["version"] = "1.0.0"

        # Add correlation ID from context
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_record["correlation_id"] = correlation_id

        # Add trace context if available (OpenTelemetry integration)
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span.is_recording():
                span_context = span.get_span_context()
                log_record["trace_id"] = f"{span_context.trace_id:032x}"
                log_record["span_id"] = f"{span_context.span_id:016x}"
        except ImportError:
            # OpenTelemetry not available
            pass

        # Ensure level is present
        if "level" not in log_record:
            log_record["level"] = record.levelname

        # Add logger name for better filtering
        log_record["logger"] = record.name


def configure_logging(log_level: str = "INFO", enable_json: bool = True) -> None:
    """Configure structured logging for the conversation API."""

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure handler
    handler = logging.StreamHandler(sys.stdout)

    if enable_json:
        # Use structured JSON logging
        formatter = ConversationJsonFormatter(
            fmt="%(timestamp)s %(level)s %(name)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        # Use simple text logging for development
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)

    # Set up root logger
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(handler)

    # Configure specific loggers
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)  # Reduce SQL noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # Reduce HTTP noise
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)  # Keep access logs


def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent configuration."""
    return logging.getLogger(name)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for current context."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def log_conversation_event(
    logger: logging.Logger,
    event_type: str,
    conversation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    message_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs,
) -> None:
    """Log a structured conversation event."""

    event_data = {
        "event_type": event_type,
        "conversation_id": conversation_id,
        "agent_id": agent_id,
        "message_id": message_id,
        "user_id": user_id,
        **kwargs,
    }

    # Remove None values
    event_data = {k: v for k, v in event_data.items() if v is not None}

    logger.info(f"Conversation event: {event_type}", extra=event_data)


def log_performance_metric(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    conversation_id: Optional[str] = None,
    success: bool = True,
    **kwargs,
) -> None:
    """Log performance metrics for conversation operations."""

    metric_data = {
        "metric_type": "performance",
        "operation": operation,
        "duration_ms": duration_ms,
        "conversation_id": conversation_id,
        "success": success,
        **kwargs,
    }

    # Remove None values
    metric_data = {k: v for k, v in metric_data.items() if v is not None}

    # Log as warning if operation is slow
    if duration_ms > 1000:  # > 1 second
        logger.warning(f"Slow operation: {operation} took {duration_ms}ms", extra=metric_data)
    else:
        logger.info(f"Operation completed: {operation} in {duration_ms}ms", extra=metric_data)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    operation: str,
    conversation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **kwargs,
) -> None:
    """Log error with full context information."""

    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "operation": operation,
        "conversation_id": conversation_id,
        "agent_id": agent_id,
        **kwargs,
    }

    # Remove None values
    error_data = {k: v for k, v in error_data.items() if v is not None}

    # Add stack trace for debugging
    import traceback

    error_data["stack_trace"] = traceback.format_exc()

    logger.error(f"Error in {operation}: {str(error)}", extra=error_data)


class ConversationLogger:
    """Convenience class for conversation-specific logging."""

    def __init__(self, logger_name: str):
        self.logger = get_logger(logger_name)

    def conversation_created(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        agent_count: int = 0,
        prompt_length: int = 0,
    ) -> None:
        """Log conversation creation event."""
        log_conversation_event(
            self.logger,
            "conversation_created",
            conversation_id=conversation_id,
            user_id=user_id,
            agent_count=agent_count,
            prompt_length=prompt_length,
        )

    def conversation_started(self, conversation_id: str, duration_ms: float) -> None:
        """Log conversation start event."""
        log_conversation_event(self.logger, "conversation_started", conversation_id=conversation_id)
        log_performance_metric(
            self.logger, "conversation_start", duration_ms, conversation_id=conversation_id
        )

    def message_added(
        self,
        conversation_id: str,
        message_id: str,
        agent_id: str,
        message_length: int,
        processing_time_ms: Optional[float] = None,
    ) -> None:
        """Log message creation event."""
        log_conversation_event(
            self.logger,
            "message_added",
            conversation_id=conversation_id,
            message_id=message_id,
            agent_id=agent_id,
            message_length=message_length,
            processing_time_ms=processing_time_ms,
        )

    def conversation_completed(
        self, conversation_id: str, total_messages: int, total_duration_ms: float
    ) -> None:
        """Log conversation completion event."""
        log_conversation_event(
            self.logger,
            "conversation_completed",
            conversation_id=conversation_id,
            total_messages=total_messages,
            total_duration_ms=total_duration_ms,
        )

    def llm_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        duration_ms: float = 0,
        success: bool = True,
    ) -> None:
        """Log LLM service call."""
        log_conversation_event(
            self.logger,
            "llm_call",
            llm_provider=provider,
            llm_model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            success=success,
        )

    def database_operation(
        self,
        operation: str,
        table: str,
        duration_ms: float,
        success: bool = True,
        affected_rows: Optional[int] = None,
    ) -> None:
        """Log database operation."""
        log_performance_metric(
            self.logger,
            f"db_{operation}",
            duration_ms,
            success=success,
            table=table,
            affected_rows=affected_rows,
        )

    def websocket_event(
        self,
        event_type: str,
        conversation_id: Optional[str] = None,
        client_id: Optional[str] = None,
        connection_count: Optional[int] = None,
    ) -> None:
        """Log WebSocket events."""
        log_conversation_event(
            self.logger,
            f"websocket_{event_type}",
            conversation_id=conversation_id,
            client_id=client_id,
            connection_count=connection_count,
        )

    def error(
        self,
        error: Exception,
        operation: str,
        conversation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log error with context."""
        log_error_with_context(
            self.logger,
            error,
            operation,
            conversation_id=conversation_id,
            agent_id=agent_id,
            **kwargs,
        )


# Global logger instance for conversation API
conversation_logger = ConversationLogger("conversation_api")


# Middleware for request correlation IDs
class CorrelationIdMiddleware:
    """FastAPI middleware to add correlation IDs to requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Generate or extract correlation ID
            headers = dict(scope.get("headers", []))
            correlation_id = headers.get(b"x-correlation-id", b"").decode()

            if not correlation_id:
                correlation_id = str(uuid.uuid4())

            # Set in context
            set_correlation_id(correlation_id)

            # Add to response headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    headers[b"x-correlation-id"] = correlation_id.encode()
                    message["headers"] = list(headers.items())
                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

"""
Honeycomb Distributed Tracing Integration for Authentication Flow.

This module provides Honeycomb-specific instrumentation for the authentication
system, tracking all auth operations with detailed observability.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import Optional

from fastapi import Request

from observability.distributed_tracing import TraceSpan, get_distributed_tracer

logger = logging.getLogger(__name__)

# Honeycomb configuration
HONEYCOMB_API_KEY = os.getenv("HONEYCOMB_API_KEY", "")
HONEYCOMB_DATASET = os.getenv("HONEYCOMB_DATASET", "freeagentics-auth")
HONEYCOMB_ENABLED = bool(HONEYCOMB_API_KEY)


class HoneycombAuthTracer:
    """Honeycomb-specific tracer for authentication operations."""

    def __init__(self):
        """Initialize Honeycomb auth tracer."""
        self.tracer = get_distributed_tracer()
        self.enabled = HONEYCOMB_ENABLED

        if self.enabled:
            logger.info(f"Honeycomb tracing enabled for dataset: {HONEYCOMB_DATASET}")
        else:
            logger.warning("Honeycomb API key not configured - tracing disabled")

    @asynccontextmanager
    async def trace_auth_operation(
        self,
        operation: str,
        request: Optional[Request] = None,
        user_id: Optional[str] = None,
        **extra_tags,
    ):
        """Trace an authentication operation with Honeycomb-specific fields."""
        if not self.enabled:
            yield None
            return

        # Create span with Honeycomb fields
        span = self.tracer.start_span(
            operation_name=f"auth.{operation}", service_name="auth-service"
        )

        # Add standard auth tags
        span.add_tag("auth.operation", operation)
        span.add_tag("honeycomb.dataset", HONEYCOMB_DATASET)

        if user_id:
            span.add_tag("user.id", user_id)

        if request:
            # Add request context
            span.add_tag("http.method", request.method)
            span.add_tag("http.url", str(request.url))
            span.add_tag("http.path", request.url.path)
            span.add_tag("http.client_ip", request.client.host)

            # Add headers (excluding sensitive data)
            span.add_tag("http.user_agent", request.headers.get("User-Agent", ""))
            span.add_tag("http.content_type", request.headers.get("Content-Type", ""))

        # Add extra tags
        for key, value in extra_tags.items():
            span.add_tag(f"auth.{key}", value)

        try:
            yield span
            span.finish(status="ok")
        except Exception as e:
            span.add_tag("error", True)
            span.add_tag("error.message", str(e))
            span.add_tag("error.type", type(e).__name__)
            span.finish(status="error", error=str(e))
            raise
        finally:
            # Send to Honeycomb (if configured)
            await self._send_to_honeycomb(span)

    async def _send_to_honeycomb(self, span: TraceSpan):
        """Send span data to Honeycomb."""
        if not self.enabled:
            return

        # Convert span to Honeycomb event format
        event = {
            "timestamp": span.start_time,
            "duration_ms": span.duration_ms or 0,
            "trace.trace_id": span.trace_id,
            "trace.span_id": span.span_id,
            "trace.parent_id": span.parent_span_id,
            "name": span.operation_name,
            "service_name": span.service_name,
            "status": span.status,
        }

        # Add all tags as fields
        event.update(span.tags)

        # Add logs as annotations
        if span.logs:
            event["annotations"] = json.dumps(span.logs)

        # In production, this would send to Honeycomb API
        # For now, log the event
        logger.debug(f"Honeycomb event: {json.dumps(event, indent=2)}")

    def trace_login(self, func):
        """Decorator to trace login operations."""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            username = (
                kwargs.get("login_data", {}).username
                if "login_data" in kwargs
                else None
            )

            async with self.trace_auth_operation(
                "login", request=request, username=username, auth_type="password"
            ) as span:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    if span:
                        span.add_tag("auth.success", True)
                        span.add_tag(
                            "auth.duration_ms", (time.time() - start_time) * 1000
                        )
                    return result
                except Exception as e:
                    if span:
                        span.add_tag("auth.success", False)
                        span.add_tag("auth.failure_reason", str(e))
                    raise

        return wrapper

    def trace_token_validation(self, func):
        """Decorator to trace token validation operations."""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            token_type = kwargs.get("token_type", "access")

            async with self.trace_auth_operation(
                "token_validation", token_type=token_type
            ) as span:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    if span:
                        span.add_tag("auth.token_valid", True)
                        span.add_tag(
                            "auth.validation_duration_ms",
                            (time.time() - start_time) * 1000,
                        )
                    return result
                except Exception as e:
                    if span:
                        span.add_tag("auth.token_valid", False)
                        span.add_tag("auth.validation_error", str(e))
                    raise

        return wrapper

    def trace_mfa(self, func):
        """Decorator to trace MFA operations."""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            mfa_type = kwargs.get("mfa_type", "totp")
            user_id = kwargs.get("user_id")

            async with self.trace_auth_operation(
                "mfa_verification", user_id=user_id, mfa_type=mfa_type
            ) as span:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    if span:
                        span.add_tag("auth.mfa_success", True)
                        span.add_tag(
                            "auth.mfa_duration_ms", (time.time() - start_time) * 1000
                        )
                    return result
                except Exception as e:
                    if span:
                        span.add_tag("auth.mfa_success", False)
                        span.add_tag("auth.mfa_failure_reason", str(e))
                    raise

        return wrapper

    def trace_permission_check(self, func):
        """Decorator to trace permission/RBAC checks."""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")
            resource = kwargs.get("resource")
            action = kwargs.get("action")

            async with self.trace_auth_operation(
                "permission_check", user_id=user_id, resource=resource, action=action
            ) as span:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    if span:
                        span.add_tag("auth.permission_granted", bool(result))
                        span.add_tag(
                            "auth.check_duration_ms", (time.time() - start_time) * 1000
                        )
                    return result
                except Exception as e:
                    if span:
                        span.add_tag("auth.permission_error", str(e))
                    raise

        return wrapper


# Global instance
honeycomb_auth_tracer = HoneycombAuthTracer()


# Export decorators for easy use
trace_login = honeycomb_auth_tracer.trace_login
trace_token_validation = honeycomb_auth_tracer.trace_token_validation
trace_mfa = honeycomb_auth_tracer.trace_mfa
trace_permission_check = honeycomb_auth_tracer.trace_permission_check

"""Security monitoring middleware for FastAPI.

Integrates security audit logging with API requests and responses.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from auth.security_logging import (
    SecurityEventSeverity,
    SecurityEventType,
    log_api_access,
    log_suspicious_activity,
    security_auditor,
)


class SecurityMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring and logging security events."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security monitoring."""
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.headers.__dict__["_list"].append((b"x-request-id", request_id.encode()))

        # Track request timing
        start_time = time.time()

        # Get user info if authenticated
        user_id = None
        username = None
        if hasattr(request.state, "user"):
            user_id = getattr(request.state.user, "user_id", None)
            username = getattr(request.state.user, "username", None)

        try:
            # Process request
            response = await call_next(request)

            # Calculate response time
            response_time = time.time() - start_time

            # Log API access for monitoring
            log_api_access(
                request=request,
                response_status=response.status_code,
                response_time=response_time,
                user_id=user_id,
                username=username,
            )

            # Add security headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time:.3f}"

            return response

        except Exception as e:
            # Log error
            response_time = time.time() - start_time

            security_auditor.log_event(
                SecurityEventType.API_ERROR,
                SecurityEventSeverity.ERROR,
                f"Unhandled exception in {request.url.path}: {str(e)}",
                request=request,
                user_id=user_id,
                username=username,
                status_code=500,
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "response_time": response_time,
                },
            )

            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id,
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Response-Time": f"{response_time:.3f}",
                },
            )


class SecurityHeadersMiddleware:
    """Middleware to add security headers to all responses."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """Add security headers to response."""
        if scope["type"] == "http":

            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))

                    # Add security headers
                    security_headers = [
                        (b"X-Content-Type-Options", b"nosniff"),
                        (b"X-Frame-Options", b"DENY"),
                        (b"X-XSS-Protection", b"1; mode=block"),
                        (b"Referrer-Policy", b"strict-origin-when-cross-origin"),
                        (b"Permissions-Policy", b"geolocation=(), microphone=(), camera=()"),
                    ]

                    # Add HSTS in production
                    import os

                    if os.getenv("PRODUCTION", "false").lower() == "true":
                        security_headers.append(
                            (b"Strict-Transport-Security", b"max-age=31536000; includeSubDomains")
                        )

                    # Add CSP if configured
                    csp = os.getenv("CSP_DIRECTIVES")
                    if csp:
                        security_headers.append((b"Content-Security-Policy", csp.encode()))

                    # Merge with existing headers
                    message["headers"] = list(headers.items()) + security_headers

                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

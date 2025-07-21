"""Prometheus metrics middleware for HTTP request tracking.

This middleware collects metrics for every HTTP request following
Charity Majors' observability principles.
"""

import time
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from observability.prometheus_metrics import record_http_request

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""

    def __init__(self, app: ASGIApp):
        """Initialize metrics middleware.
        
        Args:
            app: The ASGI application
        """
        super().__init__(app)
        logger.info("Prometheus metrics middleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics.
        
        Args:
            request: The incoming request
            call_next: The next middleware/handler
            
        Returns:
            The response from the handler
        """
        # Start timing
        start_time = time.time()
        
        # Get request details
        method = request.method
        path = request.url.path
        
        # Process request
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception as e:
            # Log error but don't interfere with error handling
            logger.error(f"Request failed: {e}")
            status = "500"
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            try:
                record_http_request(
                    method=method,
                    endpoint=path,
                    status=status,
                    duration=duration
                )
            except Exception as e:
                # Never let metrics collection break the app
                logger.error(f"Failed to record metrics: {e}")
        
        return response

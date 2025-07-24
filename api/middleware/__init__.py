"""API Middleware Package.

This package contains middleware components for the FreeAgentics API including:
- DDoS protection and rate limiting
- Security monitoring and logging
- Request/response tracking
- Authentication and authorization
- Performance monitoring
"""

from auth.security_headers import SecurityHeadersMiddleware

from .ddos_protection import (
    DDoSProtectionMiddleware,
    EndpointRateLimits,
    RateLimitConfig,
    RateLimiter,
    WebSocketRateLimiter,
)
from .metrics import MetricsMiddleware
from .security_monitoring import SecurityMonitoringMiddleware

__all__ = [
    "DDoSProtectionMiddleware",
    "EndpointRateLimits",
    "MetricsMiddleware",
    "RateLimitConfig",
    "RateLimiter",
    "WebSocketRateLimiter",
    "SecurityMonitoringMiddleware",
    "SecurityHeadersMiddleware",
]

"""API Middleware Package.

This package contains middleware components for the FreeAgentics API including:
- DDoS protection and rate limiting
- Security monitoring and logging
- Request/response tracking
- Authentication and authorization
- Performance monitoring
"""

from .ddos_protection import (
    DDoSProtectionMiddleware,
    EndpointRateLimits,
    RateLimitConfig,
    RateLimiter,
    WebSocketRateLimiter,
)
from .security_monitoring import (
    SecurityHeadersMiddleware,
    SecurityMonitoringMiddleware,
)

__all__ = [
    "DDoSProtectionMiddleware",
    "EndpointRateLimits",
    "RateLimitConfig",
    "RateLimiter",
    "WebSocketRateLimiter",
    "SecurityMonitoringMiddleware",
    "SecurityHeadersMiddleware",
]

"""Security headers middleware using enhanced security headers implementation."""

import time
from typing import Callable

from fastapi import Request
from fastapi.responses import Response

from auth.security_headers import SecurityHeadersManager, SecurityPolicy

# Create global security manager with production settings
_security_manager = SecurityHeadersManager(
    SecurityPolicy(
        enable_hsts=True,
        hsts_max_age=31536000,
        hsts_include_subdomains=True,
        hsts_preload=True,
        enable_expect_ct=True,
        expect_ct_enforce=True,
        enable_certificate_pinning=True,
        production_mode=True,
    )
)


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Add comprehensive security headers to all responses."""
    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception:
        # Create error response
        from fastapi.responses import JSONResponse

        response = JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # Apply comprehensive security headers
    security_headers = _security_manager.get_security_headers(request, response)

    for header_name, header_value in security_headers.items():
        if header_value:  # Only set non-empty headers
            response.headers[header_name] = header_value

    # Remove server header for security
    response.headers.pop("server", None)
    response.headers.pop("Server", None)

    # Add timing header
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}"

    return response

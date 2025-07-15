import time
from typing import Callable

from fastapi import Request
from fastapi.responses import Response


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Add security headers to all responses."""
    start_time = time.time()
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self' wss: https:;"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Remove server header
    response.headers.pop("server", None)

    # Add custom headers
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response

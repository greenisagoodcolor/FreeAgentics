import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors without exposing internal details."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid input data provided"},
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions securely."""
    logger.error(f"HTTP exception: {exc.detail}")

    # Don't expose internal error details in production
    if exc.status_code >= 500:
        return JSONResponse(
            status_code=exc.status_code, content={"detail": "An internal error occurred"}
        )

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions without exposing details."""
    logger.exception("Unexpected error occurred")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"},
    )

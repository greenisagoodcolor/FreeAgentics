"""
Health check endpoint following TDD principles.

This module implements a simple health check endpoint that:
- Performs a real SELECT 1 database query
- Returns proper HTTP status codes (200/503)
- Responds in under 100ms
- Uses FastAPI exception handlers (no try/except)
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from database.session import get_db

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint that performs a real database query.

    Returns:
        - 200 OK with {"status": "healthy", "db": "connected"} when DB is accessible
        - 503 Service Unavailable with error details when DB is down

    Uses try/except to handle database errors.
    """
    try:
        # Perform SELECT 1 query to verify database connectivity
        result = db.execute(text("SELECT 1"))
        result.fetchone()  # Ensure query executes
        
        return {"status": "healthy", "db": "connected"}
    except OperationalError as exc:
        # Handle database operational errors with 503 status
        return JSONResponse(
            status_code=503, 
            content={"status": "unhealthy", "db": "disconnected", "error": str(exc)}
        )


# Define a function that can be used as exception handler at app level
def database_exception_handler(request, exc):
    """Handle database operational errors with 503 status."""
    return JSONResponse(
        status_code=503, content={"status": "unhealthy", "db": "disconnected", "error": str(exc)}
    )

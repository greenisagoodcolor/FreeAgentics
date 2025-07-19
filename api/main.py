"""Main FastAPI application module for FreeAgentics API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.security_monitoring import (
    SecurityHeadersMiddleware,
    SecurityMonitoringMiddleware,
)
from api.v1 import (
    agents,
    auth,
    inference,
    mfa,
    monitoring,
    security,
    system,
    websocket,
)
from api.v1.graphql_schema import graphql_app
from auth.security_implementation import SecurityMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events.

    Handles startup and shutdown operations for the FastAPI application.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded back to FastAPI during application runtime.
    """
    # Startup
    logger.info("Starting FreeAgentics API...")

    # Initialize database if in development mode
    from database.session import DATABASE_URL, init_db

    logger.info(f"Database URL: {DATABASE_URL}")

    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(
            f"Database initialization skipped (may already exist): {e}"
        )

    yield
    # Shutdown
    logger.info("Shutting down FreeAgentics API...")


# Create FastAPI app
app = FastAPI(
    title="FreeAgentics API",
    description="Multi-Agent AI Platform API with Active Inference",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(SecurityMonitoringMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Include routers
app.include_router(
    auth.router, prefix="/api/v1", tags=["auth"]
)  # Auth must be first
app.include_router(mfa.router, tags=["mfa"])  # MFA router has its own prefix
app.include_router(agents.router, prefix="/api/v1", tags=["agents"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(system.router, prefix="/api/v1", tags=["system"])
app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])
app.include_router(security.router, prefix="/api/v1", tags=["security"])

# Include GraphQL router
app.include_router(graphql_app, prefix="/api/v1/graphql", tags=["graphql"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to FreeAgentics API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "FreeAgentics API",
        "version": "0.1.0",
    }

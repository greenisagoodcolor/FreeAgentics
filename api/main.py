"""Main FastAPI application module for FreeAgentics API."""

import logging

# Configure logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from api import ui_compatibility
from api.middleware.security_monitoring import SecurityMonitoringMiddleware
from api.v1 import (
    agents,
    auth,
    dev_config,
    health,
    health_extended,
    inference,
    knowledge,
    knowledge_graph,
    mfa,
    monitoring,
    prompts,
    security,
    settings,
    system,
    websocket,
)
from api.v1.graphql_schema import graphql_app
from auth.security_headers import SecurityHeadersManager, SecurityHeadersMiddleware, SecurityPolicy
from observability.performance_metrics import start_performance_tracking
from observability.prometheus_metrics import (
    get_prometheus_content_type,
    get_prometheus_metrics,
    start_prometheus_metrics_collection,
)

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
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

    # Start Prometheus metrics collection
    try:
        await start_prometheus_metrics_collection()
        logger.info("📊 Prometheus metrics collection started")
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics: {e}")

    # Start performance tracking
    try:
        await start_performance_tracking()
        logger.info("✅ Performance tracking started")
    except Exception as e:
        logger.error(f"Failed to start performance tracking: {e}")

    # Initialize providers
    from core.providers import get_database, init_providers

    try:
        init_providers()
        logger.info("✅ Providers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize providers: {e}")

    # Initialize database schema
    try:
        db_provider = get_database()
        db_provider.init_db()
        logger.info("✅ Database schema initialized")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")

    yield
    # Shutdown
    logger.info("Shutting down FreeAgentics API...")

    # Stop Prometheus metrics collection
    try:
        from observability.prometheus_metrics import stop_prometheus_metrics_collection

        await stop_prometheus_metrics_collection()
        logger.info("📊 Prometheus metrics collection stopped")
    except Exception as e:
        logger.warning(f"⚠️ Failed to stop Prometheus metrics collection: {e}")

    # Stop performance tracking
    try:
        from observability.performance_metrics import stop_performance_tracking

        await stop_performance_tracking()
        logger.info("✅ Performance tracking stopped")
    except Exception as e:
        logger.warning(f"⚠️ Failed to stop performance tracking: {e}")


# Create FastAPI app
app = FastAPI(
    title="FreeAgentics API",
    description="Multi-Agent AI Platform API with Active Inference",
    version="0.1.0",
    lifespan=lifespan,
)

# In dev mode, inject auth middleware (must be before other middleware)
if os.getenv("PRODUCTION", "false").lower() != "true" and not os.getenv("DATABASE_URL"):
    from auth.dev_auth import inject_dev_auth_middleware

    inject_dev_auth_middleware(app)
    logger.info("🔑 Dev auth middleware enabled")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://freeagentics.com",
        "https://www.freeagentics.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "accept",
        "authorization",
        "content-type",
        "origin",
        "x-requested-with",
        "x-api-key",
        "x-client-version",
    ],
    expose_headers=["x-total-count", "x-rate-limit-remaining", "x-rate-limit-reset"],
    max_age=86400,
)

# Add security middleware
# app.add_middleware(SecurityMiddleware)  # Temporarily disabled due to conflict
app.add_middleware(SecurityMonitoringMiddleware)

# Create security headers manager with production policy
security_manager = SecurityHeadersManager(
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
app.add_middleware(SecurityHeadersMiddleware, security_manager=security_manager)

# Include routers
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])  # Auth must be first
app.include_router(mfa.router, tags=["mfa"])  # MFA router has its own prefix
app.include_router(dev_config.router, prefix="/api/v1", tags=["dev"])  # Dev config
app.include_router(agents.router, prefix="/api/v1", tags=["agents"])
app.include_router(prompts.router, prefix="/api/v1", tags=["prompts"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(knowledge_graph.router, prefix="/api/v1", tags=["knowledge-graph"])
app.include_router(system.router, prefix="/api/v1", tags=["system"])
app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])
app.include_router(security.router, prefix="/api/v1", tags=["security"])
app.include_router(settings.router, prefix="/api/v1", tags=["settings"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(health_extended.router, prefix="/api/v1", tags=["health"])

# Include UI compatibility router (no prefix - direct /api/agents)
app.include_router(ui_compatibility.router, prefix="/api", tags=["ui-compatibility"])

# Include GraphQL router
app.include_router(graphql_app, prefix="/api/v1/graphql", tags=["graphql"])

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    # Directory might not exist in some environments
    pass


@app.get("/demo")
async def demo():
    """Serve the Active Inference demo."""
    try:
        return FileResponse("static/demo.html", media_type="text/html")
    except FileNotFoundError:
        return {"message": "Demo not available - static files not found"}


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


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    metrics_data = get_prometheus_metrics()
    return Response(
        content=metrics_data,
        media_type=get_prometheus_content_type(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

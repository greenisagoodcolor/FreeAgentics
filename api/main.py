import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from api.middleware.security_monitoring import (
    SecurityHeadersMiddleware,
    SecurityMonitoringMiddleware,
)
from api.ui_compatibility import router as ui_router
from api.v1 import (
    agents,
    auth,
    health,
    health_extended,
    inference,
    mfa,
    monitoring,
    prompts,
    security,
    system,
    websocket,
)
from api.v1.graphql_schema import graphql_app
from auth.security_implementation import SecurityMiddleware
from observability.prometheus_metrics import (
    get_prometheus_content_type,
    get_prometheus_metrics,
    start_prometheus_metrics_collection,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FreeAgentics API...")
    # Start Prometheus metrics collection
    await start_prometheus_metrics_collection()
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
app.include_router(prompts.router, prefix="/api/v1", tags=["prompts"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(system.router, prefix="/api/v1", tags=["system"])
app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])
app.include_router(security.router, prefix="/api/v1", tags=["security"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(health_extended.router, prefix="/api/v1", tags=["health"])

# Include UI compatibility router (no prefix - direct /api/agents)
app.include_router(ui_router, prefix="/api", tags=["ui-compatibility"])

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


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    metrics_data = get_prometheus_metrics()
    return Response(
        content=metrics_data,
        media_type=get_prometheus_content_type(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

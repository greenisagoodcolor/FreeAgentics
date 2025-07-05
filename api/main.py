import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import routers
from api.v1 import agents, inference, monitoring, system, websocket
from api.v1.graphql_schema import graphql_app


# Lifespan manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FreeAgentics API...")
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

# Include routers
app.include_router(agents.router, prefix="/api/v1", tags=["agents"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(system.router, prefix="/api/v1", tags=["system"])
app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])

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

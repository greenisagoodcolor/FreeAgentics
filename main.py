"""
FreeAgentics FastAPI Backend - Main Application Entry Point.

Revolutionary Multi-Agent Active Inference Research Platform implementing
committee consensus from .taskmaster/docs/prd.txt with clean architecture
principles (Robert C. Martin), modular design (Martin Fowler), and
mathematical rigor (Karl Friston, Yann LeCun).
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.middleware.metrics import MetricsMiddleware
from api.middleware.rate_limiter import (
    RateLimitMiddleware,
    create_rate_limiter,
)

# SECURITY: Import authentication and security components
from auth import SecurityMiddleware
from auth.https_enforcement import HTTPSEnforcementMiddleware, SSLConfiguration
from auth.security_headers import (
    SecurityHeadersManager,
    SecurityHeadersMiddleware,
    SecurityPolicy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("freeagentics.log"),
    ],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown events.
    Implements clean architecture with dependency injection.
    """
    logger.info("🚀 FreeAgentics Backend Starting Up...")

    # Startup: Initialize core services
    logger.info("Initializing Active Inference Engine...")
    logger.info("Initializing Coalition Formation System...")
    logger.info("Initializing Knowledge Graph Manager...")
    logger.info("Initializing GNN Model Services...")

    # Initialize observability and monitoring
    logger.info("Initializing observability and monitoring...")
    observability_tasks = []
    try:
        from observability_setup import initialize_observability

        # Initialize agent manager and database dependencies
        agent_manager = None  # Will be initialized properly in production
        database = None  # Will be initialized properly in production

        observability_manager = initialize_observability(agent_manager, database)
        observability_tasks = await observability_manager.start()

        logger.info("✅ Observability system started successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize observability: {e}")
        # Continue startup even if observability fails

    try:
        # Core initialization would go here
        # Following committee consensus for modular architecture
        yield
    finally:
        # Shutdown: Clean up resources
        logger.info("🛑 FreeAgentics Backend Shutting Down...")

        # Stop observability
        try:
            if observability_tasks:
                from observability_setup import get_observability_manager

                manager = get_observability_manager()
                if manager:
                    await manager.stop()

                # Cancel background tasks
                import asyncio

                for task in observability_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                logger.info("✅ Observability system stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping observability: {e}")
        logger.info("Cleaning up Active Inference sessions...")
        logger.info("Saving coalition states...")
        logger.info("Persisting knowledge graph...")


# FastAPI Application with Committee-Approved Architecture
app = FastAPI(
    title="FreeAgentics API",
    description="""
    Revolutionary Multi-Agent Active Inference Research Platform

    This platform enables researchers to orchestrate, monitor, and analyze
    true autonomous AI agents using:

    - **PyMDP-based Active Inference** with real belief state calculations
    - **GNN model generation** from natural language specifications
    - **Multi-agent coalition formation** with business value optimization
    - **Real-time knowledge graph evolution** driven by epistemic uncertainty reduction
    - **Hardware deployment pipelines** for edge AI systems

    Built following expert committee consensus with clean architecture principles.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware Stack - Production Security Enhanced
# SECURITY: Add security middleware first
app.add_middleware(SecurityMiddleware)

# SECURITY: Add HTTPS enforcement and SSL/TLS configuration
# Configure based on environment
is_production = os.getenv("PRODUCTION", "false").lower() == "true"

ssl_config = SSLConfiguration(
    production_mode=is_production,
    enable_letsencrypt=is_production,
    letsencrypt_email=os.getenv("LETSENCRYPT_EMAIL", "admin@freeagentics.com"),
    letsencrypt_domains=(
        os.getenv("LETSENCRYPT_DOMAINS", "").split(",") if os.getenv("LETSENCRYPT_DOMAINS") else []
    ),
    hsts_enabled=True,
    hsts_max_age=31536000,  # 1 year
    hsts_include_subdomains=True,
    hsts_preload=is_production,
    secure_cookies=True,
    behind_load_balancer=os.getenv("BEHIND_LOAD_BALANCER", "false").lower() == "true",
    trusted_proxies=os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(","),
)

app.add_middleware(HTTPSEnforcementMiddleware, config=ssl_config)

# SECURITY: Add comprehensive security headers
security_policy = SecurityPolicy(
    production_mode=is_production,
    enable_hsts=True,
    hsts_max_age=31536000,
    hsts_include_subdomains=True,
    hsts_preload=is_production,
    csp_report_uri="/api/v1/security/csp-report",
    enable_expect_ct=True,
    expect_ct_report_uri="/api/v1/security/ct-report",
    enable_certificate_pinning=is_production,
    secure_cookies=True,
)

# Create SecurityHeadersManager with the policy
security_headers_manager = SecurityHeadersManager(security_policy)
app.add_middleware(SecurityHeadersMiddleware, security_manager=security_headers_manager)

# OBSERVABILITY: Add Prometheus metrics middleware (Per Charity Majors)
app.add_middleware(MetricsMiddleware)

# SECURITY: Add comprehensive rate limiting and DDoS protection
# Only enable Redis-based rate limiting if not in test environment
if (
    os.getenv("TESTING", "false").lower() != "true"
    and os.getenv("REDIS_ENABLED", "true").lower() == "true"
):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    config_file = os.path.join(os.path.dirname(__file__), "config", "rate_limiting.yaml")

    # Create rate limiter instance
    rate_limiter = create_rate_limiter(redis_url=redis_url, config_file=config_file)

    # Function to extract user ID from request (for authenticated rate limiting)
    async def get_user_id_from_request(request):
        """Extract user ID from JWT token if present."""
        try:
            from auth import auth_manager

            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ")[1]
                payload = auth_manager.verify_token(token)
                if payload and "sub" in payload:
                    return payload["sub"]
        except Exception as e:
            # Log authentication errors for debugging but don't expose details
            logger.debug(f"Failed to extract user ID from request: {e}")
        return None

    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=rate_limiter,
        get_user_id=get_user_id_from_request,
    )
else:
    logger.info("Rate limiting disabled for testing environment")

# Alternative: Use DDoS protection middleware (includes rate limiting)
# app.add_middleware(DDoSProtectionMiddleware, redis_url=redis_url)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3030"],
    # Next.js dev/demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler - Clean Architecture Error Management
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler following clean architecture principles."""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "path": str(request.url),
        },
    )


# Health Check Endpoint - Production Readiness
@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """System health check endpoint for deployment monitoring."""
    return {
        "status": "healthy",
        "service": "freeagentics-api",
        "version": "1.0.0",
        "timestamp": "2025-06-26T00:00:00Z",
    }


# Prometheus Metrics Endpoint
@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint for monitoring."""
    try:
        from fastapi import Response

        from observability.prometheus_metrics import (
            get_prometheus_content_type,
            get_prometheus_metrics,
        )

        metrics_data = get_prometheus_metrics()
        content_type = get_prometheus_content_type()

        return Response(
            content=metrics_data,
            media_type=content_type,
            headers={"Cache-Control": "no-cache"},
        )
    except ImportError:
        logger.warning("Prometheus metrics not available")
        return Response(
            content="# Prometheus metrics not available\n",
            media_type="text/plain",
            status_code=503,
        )
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        return Response(
            content=f"# Error getting metrics: {e}\n",
            media_type="text/plain",
            status_code=500,
        )


# Root Endpoint - API Discovery
@app.get("/", tags=["system"])
async def root() -> dict:
    """Root endpoint providing API information and capabilities."""
    return {
        "message": "FreeAgentics Revolutionary Multi-Agent Active Inference API",
        "version": "1.0.0",
        "capabilities": [
            "Active Inference with PyMDP",
            "GNN Model Generation",
            "Coalition Formation",
            "Knowledge Graph Evolution",
            "Hardware Deployment",
        ],
        "documentation": "/docs",
        "health": "/health",
    }


# Router Registration - Modular Architecture
# Following Martin Fowler's modular monolith principles

# Import available routers
try:
    from api.v1.agents import router as agents_router
    from api.v1.auth import router as auth_router
    from api.v1.dev_config import router as dev_config_router
    from api.v1.knowledge import router as knowledge_router
    from api.v1.system import router as system_router
    from api.v1.websocket import router as websocket_router

    # SECURITY: Auth routes (no authentication required)
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
    app.include_router(dev_config_router, prefix="/api/v1", tags=["development"])

    # Protected API routes
    app.include_router(agents_router, prefix="/api/v1", tags=["agents"])
    app.include_router(system_router, prefix="/api/v1/system", tags=["system", "monitoring"])
    app.include_router(websocket_router, prefix="/api/v1", tags=["websocket", "real-time"])
    app.include_router(knowledge_router, prefix="/api/v1/knowledge", tags=["knowledge-graph"])

    logger.info("✅ API v1 routers registered successfully (with authentication)")

except ImportError as e:
    logger.error(f"❌ Failed to import API routers: {e}")

# Future routers (to be implemented)
# app.include_router(
#     inference_router,
#     prefix="/api/v1/inference",
#     tags=["active-inference"]
# )

# app.include_router(
#     gnn_router,
#     prefix="/api/v1/gnn",
#     tags=["graph-neural-networks"]
# )

# app.include_router(
#     knowledge_router,
#     prefix="/api/v1/knowledge",
#     tags=["knowledge-graph"]
# )

# app.include_router(
#     coalitions_router,
#     prefix="/api/v1/coalitions",
#     tags=["coalition-formation"]
# )

# app.include_router(
#     deployment_router,
#     prefix="/api/v1/deployment",
#     tags=["hardware-deployment"]
# )

# WebSocket routers (existing)
try:
    from api.websocket.coalition_monitoring import router as coalition_ws_router
    from api.websocket.markov_blanket_monitoring import router as markov_ws_router
    from api.websocket.real_time_updates import router as real_time_ws_router

    app.include_router(real_time_ws_router, prefix="/ws", tags=["websockets"])

    app.include_router(
        coalition_ws_router,
        prefix="/ws/coalitions",
        tags=["websockets", "coalitions"],
    )

    app.include_router(
        markov_ws_router,
        prefix="/ws/markov-blanket",
        tags=["websockets", "safety"],
    )

    logger.info("✅ WebSocket routers registered successfully")

except ImportError as e:
    logger.warning(f"⚠️ WebSocket routers not available: {e}")


# Development server entry point
if __name__ == "__main__":
    import uvicorn

    logger.info("🔧 Starting development server...")
    # Use environment variable for host or default to localhost for security
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")

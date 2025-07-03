"""
FreeAgentics API Main Entry Point

This module serves as the main FastAPI application entry point for the FreeAgentics
multi-agent system with Active Inference capabilities.
"""

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI application
app = FastAPI(
    title="FreeAgentics API",
    description="Multi-Agent Active Inference System API",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response


@app.get("/")
async def root():
    """Root endpoint providing API status"""
    return {
        "name": "FreeAgentics API",
        "version": "2.1.0",
        "status": "operational",
        "description": "Multi-Agent Active Inference System",
        "endpoints": {"docs": "/docs", "redoc": "/redoc", "health": "/health"},
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FreeAgentics API", "version": "2.1.0"}


@app.get("/api/status")
async def api_status():
    """API status endpoint for frontend integration"""
    return {
        "api_status": "online",
        "agents": {
            "total": 0,
            "active": 0,
            "templates": ["Explorer", "Merchant", "Scholar", "Guardian", "Generalist"],
        },
        "system": {
            "active_inference": "ready",
            "knowledge_graph": "ready",
            "coalition_formation": "ready",
        },
    }


if __name__ == "__main__":
    # This allows running the app directly with python api/main.py
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

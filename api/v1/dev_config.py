"""Development configuration endpoint.

Provides configuration and authentication details for development mode.
Only available when not in production.
"""

import os
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status

from auth.dev_auth import get_dev_token
from core.providers import ProviderMode

router = APIRouter()


@router.get("/dev-config")
async def get_dev_config() -> Dict[str, Any]:
    """Get development configuration including auth token.

    This endpoint is only available in development/demo mode.
    Frontend can call this to get necessary configuration for local development.
    """
    # Ensure not in production
    if os.getenv("PRODUCTION", "false").lower() == "true":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This endpoint is not available in production",
        )

    mode = ProviderMode.get_mode()

    # Get dev token if in dev mode
    token_info = get_dev_token() if mode == "dev" else None

    config = {
        "mode": mode,
        "features": {
            "database": mode != "dev",
            "redis": bool(os.getenv("REDIS_URL")),
            "real_llm": bool(os.getenv("OPENAI_API_KEY")),
            "websocket": True,
            "auth_required": mode != "dev",
        },
        "endpoints": {
            "api": "/api",
            "websocket": "/api/v1/ws" if mode != "dev" else "/api/v1/ws/dev",
            "graphql": "/graphql",
            "docs": "/docs",
        },
    }

    # Add token info if available
    if token_info:
        config["auth"] = {
            "token": token_info["access_token"],
            "type": token_info["token_type"],
            "expires_in": token_info["expires_in"],
            "user": token_info.get("user", {}),
            "note": "Auto-generated dev token. In production, use /api/v1/auth/login",
        }

    # Add helpful messages
    if mode == "dev":
        config["message"] = (
            "🎯 Running in dev mode. Database and auth are mocked. "
            "To use real services, set DATABASE_URL environment variable."
        )
    elif mode == "development":
        config["message"] = (
            "🔧 Running in development mode with real database. "
            "Auth is required. Use /api/v1/auth/login to get tokens."
        )

    return config

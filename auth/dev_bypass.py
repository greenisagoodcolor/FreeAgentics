"""Development mode authentication bypass.

This module provides a unified authentication bypass for development mode,
handling both REST endpoints and WebSocket connections.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from auth.security_implementation import TokenData, UserRole, ROLE_PERMISSIONS
from core.environment import environment

logger = logging.getLogger(__name__)

# Development user constant - used for all dev mode requests
_DEV_USER = TokenData(
    user_id="dev_user",
    username="dev_user",
    role=UserRole.ADMIN,
    permissions=[p.value for p in ROLE_PERMISSIONS[UserRole.ADMIN]],
    fingerprint="dev_fingerprint",
    exp=int((datetime.utcnow() + timedelta(days=365)).timestamp())
)

async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False))
) -> TokenData:
    """Get current user with dev mode bypass.
    
    In development mode when auth is not required, this returns a dev user.
    Otherwise, it delegates to the real authentication system.
    """
    if environment.is_development and not environment.config.auth_required:
        logger.debug("Dev mode: bypassing auth, returning dev user")
        return _DEV_USER
    
    # In production or when auth is required, use real authentication
    from auth.security_implementation import get_current_user
    return await get_current_user(request, credentials)

def get_dev_user() -> TokenData:
    """Get the development user directly.
    
    This is used by WebSocket endpoints that need synchronous auth.
    """
    if environment.is_development and not environment.config.auth_required:
        return _DEV_USER
    
    raise RuntimeError("Dev user only available in development mode without auth")

def is_dev_token(token: str) -> bool:
    """Check if the provided token is the special 'dev' token."""
    return token == "dev" and environment.is_development and not environment.config.auth_required
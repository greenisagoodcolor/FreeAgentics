"""Development mode authentication bypass.

Provides conditional authentication that skips validation in dev mode.
"""

import logging
from typing import Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from auth.security_implementation import TokenData, UserRole, Permission, ROLE_PERMISSIONS, get_current_user as prod_get_current_user
from auth.dev_auth import dev_auth_manager
from core.environment import environment

logger = logging.getLogger(__name__)


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> TokenData:
    """Get current user with dev mode bypass.
    
    In development mode with auth_required=False, returns a default dev user.
    In production or when auth is required, delegates to normal auth flow.
    """
    # Check if we're in dev mode with auth disabled
    if environment.is_development and not environment.config.auth_required:
        # Return default dev user
        logger.debug("Auth bypassed in dev mode")
        return TokenData(
            user_id="dev_user",
            username="dev_user", 
            role=UserRole.ADMIN,
            permissions=[p.value for p in ROLE_PERMISSIONS[UserRole.ADMIN]],
            fingerprint="dev_fingerprint"
        )
    
    # Otherwise use normal auth if credentials provided
    if credentials:
        try:
            return await prod_get_current_user(request, credentials)
        except Exception as e:
            logger.debug(f"Auth validation failed: {e}")
    
    # In dev mode, still return dev user even if auth fails
    if environment.is_development and not environment.config.auth_required:
        return TokenData(
            user_id="dev_user",
            username="dev_user",
            role=UserRole.ADMIN, 
            permissions=[p.value for p in Permission],
            fingerprint="dev_fingerprint"
        )
    
    # In production, raise the exception
    raise


# Alias for compatibility
get_current_user_dev = get_current_user_optional
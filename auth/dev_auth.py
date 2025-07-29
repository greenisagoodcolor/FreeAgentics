"""Development authentication helpers.

This module provides automatic JWT generation for development environments
to enable quick onboarding without manual token creation.
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from auth.security_implementation import (
    Permission,
    TokenData,
    UserRole,
    ROLE_PERMISSIONS,
    create_access_token,
)

logger = logging.getLogger(__name__)


class DevAuthManager:
    """Manages development authentication tokens."""

    def __init__(self):
        self._dev_token: Optional[str] = None
        self._token_data: Optional[TokenData] = None
        self._generated_at: Optional[datetime] = None

    def is_dev_mode(self) -> bool:
        """Check if we're in development mode."""
        return os.getenv("PRODUCTION", "false").lower() != "true" and not os.getenv("DATABASE_URL")

    def get_or_create_dev_token(self) -> Dict[str, str]:
        """Get existing or create new dev token."""
        if not self.is_dev_mode():
            raise RuntimeError("Dev tokens only available in development mode")

        # Reuse token if still valid (within 12 hours)
        if self._dev_token and self._generated_at:
            age = datetime.now(timezone.utc) - self._generated_at
            if age < timedelta(hours=12):
                return {
                    "access_token": self._dev_token,
                    "token_type": "bearer",
                    "expires_in": int((timedelta(hours=12) - age).total_seconds()),
                    "info": "Reusing existing dev token",
                }

        # Generate new token
        user_id = f"dev_{secrets.token_urlsafe(8)}"
        username = "dev_user"
        role = UserRole.ADMIN  # Give admin access for easy development

        # Create token data with all permissions
        token_data = {
            "sub": user_id,
            "username": username,
            "role": role.value,
            "permissions": [p.value for p in ROLE_PERMISSIONS[role]],
            "fingerprint": "dev_fingerprint",  # Static fingerprint for dev
        }

        # Create JWT token (15 minutes for security, but we'll cache it)
        self._dev_token = create_access_token(data=token_data, expires_delta=timedelta(minutes=15))

        self._generated_at = datetime.now(timezone.utc)

        # Log for visibility
        logger.info("🔑 Generated new development JWT token")
        logger.info(f"   User: {username} (role: {role.value})")
        logger.info(f"   Permissions: {', '.join(token_data['permissions'])}")

        return {
            "access_token": self._dev_token,
            "token_type": "bearer",
            "expires_in": 900,  # 15 minutes
            "info": "New dev token generated",
            "user": {
                "id": user_id,
                "username": username,
                "role": role.value,
                "permissions": token_data["permissions"],
            },
        }

    def validate_dev_token(self, token: str) -> bool:
        """Check if token is our dev token."""
        return self.is_dev_mode() and token == self._dev_token


# Global instance
dev_auth_manager = DevAuthManager()


def get_dev_token() -> Optional[Dict[str, str]]:
    """Get development token if in dev mode."""
    if dev_auth_manager.is_dev_mode():
        return dev_auth_manager.get_or_create_dev_token()
    return None


def inject_dev_auth_middleware(app):
    """Inject middleware to auto-add dev token headers in dev mode."""
    from fastapi import Request

    @app.middleware("http")
    async def dev_auth_middleware(request: Request, call_next):
        # Only in dev mode and for API requests without auth
        if (
            dev_auth_manager.is_dev_mode()
            and request.url.path.startswith("/api/")
            and "authorization" not in request.headers
        ):
            # Skip for auth endpoints and health checks
            skip_paths = ["/api/auth/", "/api/health", "/api/dev-config"]
            if not any(request.url.path.startswith(p) for p in skip_paths):
                # Get or create dev token
                token_info = dev_auth_manager.get_or_create_dev_token()

                # Inject authorization header
                request.headers.__dict__["_list"].append(
                    (b"authorization", f"Bearer {token_info['access_token']}".encode())
                )

                # Add fingerprint for token validation
                request.headers.__dict__["_list"].append((b"x-fingerprint", b"dev_fingerprint"))

        response = await call_next(request)
        return response

    logger.info("✅ Dev auth middleware injected")

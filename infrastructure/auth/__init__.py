"""
FreeAgentics Authentication Module
Enterprise-grade authentication with JWT and RBAC
"""

from .auth_middleware import (
    AuthenticationMiddleware,
    require_auth,
    require_permissions,
)
from .auth_models import (
    Permission,
    User,
    UserRole,
    UserSession,
)
from .auth_service import (
    AuthenticationError,
    AuthorizationError,
    AuthService,
)
from .jwt_handler import (
    JWTHandler,
    JWTSecurityError,
    get_jwt_handler,
)

__all__ = [
    "AuthenticationMiddleware",
    "require_auth", 
    "require_permissions",
    "Permission",
    "User",
    "UserRole", 
    "UserSession",
    "AuthenticationError",
    "AuthorizationError",
    "AuthService",
    "JWTHandler",
    "JWTSecurityError",
    "get_jwt_handler",
]
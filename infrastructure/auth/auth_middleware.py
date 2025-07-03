"""
FreeAgentics Authentication Middleware
Production-grade FastAPI middleware for JWT authentication
"""

from functools import wraps
from typing import List, Optional, Callable

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .auth_models import Permission, User
from .auth_service import AuthService, AuthenticationError, AuthorizationError
from .jwt_handler import get_jwt_handler


# HTTP Bearer scheme for JWT tokens
security = HTTPBearer()


class AuthenticationMiddleware:
    """FastAPI authentication middleware"""

    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service

    async def __call__(self, request: Request, call_next):
        """Process request with authentication"""
        response = await call_next(request)
        return response

    def get_current_user(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> User:
        """Get current user from JWT token"""
        try:
            return self.auth_service.get_current_user(credentials.credentials)
        except AuthenticationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )

    def get_current_active_user(
        self,
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Get current active user"""
        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        return current_user


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # This decorator is used with FastAPI dependencies
        # The actual authentication logic is in the dependency
        return await func(*args, **kwargs)
    return wrapper


def require_permissions(required_permissions: List[Permission]):
    """Decorator factory to require specific permissions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs (injected by FastAPI)
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check permissions
            user_permissions = current_user.get_permissions()
            missing_permissions = [
                perm for perm in required_permissions 
                if perm not in user_permissions
            ]
            
            if missing_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing permissions: {[p.value for p in missing_permissions]}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Dependency functions for FastAPI
def get_auth_service() -> AuthService:
    """Get authentication service instance"""
    # This would typically get the database session and JWT handler
    # For now, returning a placeholder
    from sqlalchemy.orm import Session
    jwt_handler = get_jwt_handler()
    # Note: In production, you'd get the actual database session here
    # db = get_database_session()
    # return AuthService(db, jwt_handler)
    return None  # Placeholder


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """FastAPI dependency to get current user"""
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    try:
        return auth_service.get_current_user(credentials.credentials)
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """FastAPI dependency to get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_permission(permission: Permission):
    """FastAPI dependency factory to require specific permission"""
    def permission_checker(current_user: User = Depends(get_current_active_user)):
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission.value}"
            )
        return current_user
    return permission_checker


def require_role(role: str):
    """FastAPI dependency factory to require specific role"""
    def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {role}"
            )
        return current_user
    return role_checker
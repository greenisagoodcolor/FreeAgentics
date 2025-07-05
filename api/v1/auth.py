"""Authentication API endpoints."""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr

from auth import Permission, TokenData, User, UserRole, auth_manager, get_current_user, rate_limit
from auth.security_logging import (
    SecurityEventSeverity,
    SecurityEventType,
    log_login_failure,
    log_login_success,
    security_auditor,
)

router = APIRouter()


class UserRegistration(BaseModel):
    """User registration request."""

    username: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.OBSERVER


class UserLogin(BaseModel):
    """User login request."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


@router.post("/register", response_model=TokenResponse)
@rate_limit(max_requests=5, window_minutes=10)  # Strict rate limit for registration
async def register_user(request: Request, user_data: UserRegistration):
    """Register a new user."""
    try:
        user = auth_manager.register_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role=user_data.role,
        )

        access_token = auth_manager.create_access_token(user)
        refresh_token = auth_manager.create_refresh_token(user)

        # Log successful user registration
        security_auditor.log_event(
            SecurityEventType.USER_CREATED,
            SecurityEventSeverity.INFO,
            f"New user registered: {user.username}",
            request=request,
            user_id=user.user_id,
            username=user.username,
            details={"role": user.role, "email": user.email},
        )

        return TokenResponse(
            access_token=access_token, refresh_token=refresh_token, user=user.dict()
        )
    except Exception as e:
        # Log failed registration attempt
        security_auditor.log_event(
            SecurityEventType.USER_CREATED,
            SecurityEventSeverity.WARNING,
            f"Failed user registration attempt for {user_data.username}: {str(e)}",
            request=request,
            details={"username": user_data.username, "error": str(e)},
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenResponse)
@rate_limit(max_requests=10, window_minutes=5)  # Rate limit login attempts
async def login_user(request: Request, login_data: UserLogin):
    """Authenticate user and return tokens."""
    user = auth_manager.authenticate_user(login_data.username, login_data.password)

    if not user:
        # Log failed login attempt
        log_login_failure(login_data.username, request, "Invalid credentials")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if not user.is_active:
        # Log failed login due to disabled account
        log_login_failure(login_data.username, request, "Account is disabled")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is disabled")

    access_token = auth_manager.create_access_token(user)
    refresh_token = auth_manager.create_refresh_token(user)

    # Log successful login
    log_login_success(user.username, user.user_id, request)

    return TokenResponse(access_token=access_token, refresh_token=refresh_token, user=user.dict())


@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information."""
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "role": current_user.role,
        "permissions": current_user.permissions,
        "exp": current_user.exp.isoformat(),
    }


@router.post("/logout")
async def logout_user(request: Request, current_user: TokenData = Depends(get_current_user)):
    """Logout user (invalidate refresh token)."""
    # Remove refresh token
    if current_user.user_id in auth_manager.refresh_tokens:
        del auth_manager.refresh_tokens[current_user.user_id]

    # Log logout event
    security_auditor.log_event(
        SecurityEventType.LOGOUT,
        SecurityEventSeverity.INFO,
        f"User {current_user.username} logged out",
        request=request,
        user_id=current_user.user_id,
        username=current_user.username,
    )

    return {"message": "Successfully logged out"}


@router.get("/permissions")
async def get_user_permissions(current_user: TokenData = Depends(get_current_user)):
    """Get user permissions for UI role-based rendering."""
    return {
        "permissions": current_user.permissions,
        "role": current_user.role,
        "can_create_agents": Permission.CREATE_AGENT in current_user.permissions,
        "can_delete_agents": Permission.DELETE_AGENT in current_user.permissions,
        "can_view_metrics": Permission.VIEW_METRICS in current_user.permissions,
        "can_admin_system": Permission.ADMIN_SYSTEM in current_user.permissions,
    }

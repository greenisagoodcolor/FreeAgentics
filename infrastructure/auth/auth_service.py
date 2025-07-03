"""
FreeAgentics Authentication Service
Production-grade authentication with security best practices
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .auth_models import (
    User,
    UserSession,
    UserCreate,
    UserLogin,
    UserUpdate,
    UserResponse,
    TokenResponse,
    UserRole,
    Permission,
    pwd_context,
)
from .jwt_handler import JWTHandler, JWTSecurityError


class AuthenticationError(Exception):
    """Authentication related errors"""
    pass


class AuthorizationError(Exception):
    """Authorization related errors"""
    pass


class AuthService:
    """Production authentication service with security best practices"""

    def __init__(self, db: Session, jwt_handler: JWTHandler):
        self.db = db
        self.jwt_handler = jwt_handler

    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create new user with validation"""
        try:
            # Check if user already exists
            existing_user = self.db.query(User).filter(
                (User.email == user_data.email) | (User.username == user_data.username)
            ).first()
            
            if existing_user:
                if existing_user.email == user_data.email:
                    raise AuthenticationError("Email already registered")
                else:
                    raise AuthenticationError("Username already taken")

            # Create user
            db_user = User(
                email=user_data.email,
                username=user_data.username,
                full_name=user_data.full_name,
                role=user_data.role.value,
                is_active=user_data.is_active,
            )
            db_user.set_password(user_data.password)
            
            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)
            
            return UserResponse.from_orm(db_user)
            
        except IntegrityError:
            self.db.rollback()
            raise AuthenticationError("User creation failed due to data integrity")

    def authenticate_user(self, login_data: UserLogin) -> Tuple[User, str]:
        """Authenticate user and return user object and session token"""
        # Get user by username or email
        user = self.db.query(User).filter(
            (User.username == login_data.username_or_email) |
            (User.email == login_data.username_or_email)
        ).first()
        
        if not user:
            raise AuthenticationError("Invalid credentials")
        
        if not user.is_active:
            raise AuthenticationError("Account is disabled")
        
        if not user.verify_password(login_data.password):
            raise AuthenticationError("Invalid credentials")
        
        # Update last login
        user.last_login = datetime.now(timezone.utc)
        
        # Create session
        session_token = self._create_session(user)
        
        self.db.commit()
        
        return user, session_token

    def create_tokens(self, user: User) -> TokenResponse:
        """Create JWT tokens for authenticated user"""
        # Generate access token
        access_token_data = {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "permissions": [p.value for p in user.get_permissions()],
        }
        
        access_token = self.jwt_handler.create_access_token(access_token_data)
        refresh_token = self.jwt_handler.create_refresh_token({"sub": str(user.id)})
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.jwt_handler.access_token_expire_minutes * 60,
            user=UserResponse.from_orm(user)
        )

    def refresh_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token"""
        try:
            payload = self.jwt_handler.verify_refresh_token(refresh_token)
            user_id = payload.get("sub")
            
            if not user_id:
                raise AuthenticationError("Invalid refresh token")
            
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            return self.create_tokens(user)
            
        except JWTSecurityError as e:
            raise AuthenticationError(f"Token refresh failed: {str(e)}")

    def verify_token(self, token: str) -> Dict[str, any]:
        """Verify JWT token and return payload"""
        try:
            return self.jwt_handler.verify_access_token(token)
        except JWTSecurityError as e:
            raise AuthenticationError(f"Token verification failed: {str(e)}")

    def get_current_user(self, token: str) -> User:
        """Get current user from JWT token"""
        payload = self.verify_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise AuthenticationError("Invalid token payload")
        
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise AuthenticationError("User not found")
        
        if not user.is_active:
            raise AuthenticationError("User account is disabled")
        
        return user

    def check_permissions(self, user: User, required_permissions: List[Permission]) -> bool:
        """Check if user has required permissions"""
        user_permissions = user.get_permissions()
        return all(perm in user_permissions for perm in required_permissions)

    def update_user(self, user_id: uuid.UUID, user_data: UserUpdate) -> UserResponse:
        """Update user information"""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise AuthenticationError("User not found")
        
        # Update fields
        if user_data.email is not None:
            # Check if email is already taken
            existing = self.db.query(User).filter(
                User.email == user_data.email,
                User.id != user_id
            ).first()
            if existing:
                raise AuthenticationError("Email already in use")
            user.email = user_data.email
        
        if user_data.full_name is not None:
            user.full_name = user_data.full_name
        
        if user_data.role is not None:
            user.role = user_data.role.value
        
        if user_data.is_active is not None:
            user.is_active = user_data.is_active
        
        user.updated_at = datetime.now(timezone.utc)
        
        self.db.commit()
        self.db.refresh(user)
        
        return UserResponse.from_orm(user)

    def delete_user(self, user_id: uuid.UUID) -> bool:
        """Delete user and associated sessions"""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        
        # Delete user (cascades to sessions)
        self.db.delete(user)
        self.db.commit()
        
        return True

    def logout_user(self, session_token: str) -> bool:
        """Logout user by invalidating session"""
        session = self.db.query(UserSession).filter(
            UserSession.session_token == session_token
        ).first()
        
        if session:
            session.is_active = False
            self.db.commit()
            return True
        
        return False

    def _create_session(self, user: User, expires_minutes: int = 30) -> str:
        """Create user session"""
        session_token = str(uuid.uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
        
        session = UserSession(
            user_id=user.id,
            session_token=session_token,
            expires_at=expires_at,
        )
        
        self.db.add(session)
        return session_token

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        count = self.db.query(UserSession).filter(
            UserSession.expires_at < datetime.now(timezone.utc)
        ).delete()
        
        self.db.commit()
        return count
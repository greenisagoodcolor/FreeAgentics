"""
FreeAgentics Authentication Models
Enterprise-grade JWT-based authentication with RBAC
"""

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# Password hashing context
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    DEVELOPER = "developer"
    OPERATOR = "operator"


class Permission(str, Enum):
    """Permission enumeration for RBAC"""
    READ_AGENTS = "read:agents"
    WRITE_AGENTS = "write:agents"
    DELETE_AGENTS = "delete:agents"
    READ_COALITIONS = "read:coalitions"
    WRITE_COALITIONS = "write:coalitions"
    DELETE_COALITIONS = "delete:coalitions"
    READ_EXPERIMENTS = "read:experiments"
    WRITE_EXPERIMENTS = "write:experiments"
    DELETE_EXPERIMENTS = "delete:experiments"
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [permission for permission in Permission],
    UserRole.DEVELOPER: [
        Permission.READ_AGENTS,
        Permission.WRITE_AGENTS,
        Permission.READ_COALITIONS,
        Permission.WRITE_COALITIONS,
        Permission.READ_EXPERIMENTS,
        Permission.WRITE_EXPERIMENTS,
    ],
    UserRole.OPERATOR: [
        Permission.READ_AGENTS,
        Permission.READ_COALITIONS,
        Permission.READ_EXPERIMENTS,
    ],
    UserRole.USER: [
        Permission.READ_AGENTS,
        Permission.READ_COALITIONS,
        Permission.READ_EXPERIMENTS,
    ],
    UserRole.GUEST: [
        Permission.READ_AGENTS,
    ],
}


class User(Base):
    """User database model"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), default=UserRole.USER.value, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime(timezone=True))
    user_metadata = Column(JSON, default=dict)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")

    def verify_password(self, password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(password, self.hashed_password)

    def set_password(self, password: str) -> None:
        """Set password hash"""
        self.hashed_password = pwd_context.hash(password)

    def get_permissions(self) -> List[Permission]:
        """Get user permissions based on role"""
        return ROLE_PERMISSIONS.get(UserRole(self.role), [])

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in self.get_permissions()


class UserSession(Base):
    """User session database model"""
    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, index=True, nullable=False)
    refresh_token = Column(String(255), unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_accessed = Column(DateTime(timezone=True), default=datetime.utcnow)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True, nullable=False)
    session_metadata = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="sessions")

    @property
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.now(timezone.utc) > self.expires_at

    def extend_session(self, minutes: int = 30) -> None:
        """Extend session expiration"""
        self.expires_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        self.last_accessed = datetime.now(timezone.utc)


# Pydantic models for API
class UserBase(BaseModel):
    """Base user model"""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True

    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'Username must be alphanumeric'
        assert len(v) >= 3, 'Username must be at least 3 characters'
        assert len(v) <= 50, 'Username must be at most 50 characters'
        return v


class UserCreate(UserBase):
    """User creation model"""
    password: str

    @validator('password')
    def password_strength(cls, v):
        assert len(v) >= 8, 'Password must be at least 8 characters'
        assert any(c.isupper() for c in v), 'Password must contain uppercase letter'
        assert any(c.islower() for c in v), 'Password must contain lowercase letter'
        assert any(c.isdigit() for c in v), 'Password must contain digit'
        return v


class UserUpdate(BaseModel):
    """User update model"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """User response model"""
    id: uuid.UUID
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """User login model"""
    username_or_email: str
    password: str


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenRefresh(BaseModel):
    """Token refresh model"""
    refresh_token: str
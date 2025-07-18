"""Security implementation for production deployment.

Addresses the CRITICAL PRODUCTION BLOCKER: No authentication/authorization.
Implements JWT authentication, RBAC authorization, and input validation.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import jwt
import sqlalchemy
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, validator
from sqlalchemy.exc import SQLAlchemyError

from .jwt_handler import jwt_handler  # Import secure JWT handler

logger = logging.getLogger(__name__)

# Security configuration - Use environment variables in production
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_2025_not_for_production")
JWT_SECRET = os.getenv("JWT_SECRET", "dev_jwt_secret_2025_not_for_production")
ALGORITHM = "RS256"  # Updated to use RS256
ACCESS_TOKEN_EXPIRE_MINUTES = (
    15  # Reduced from 30 to 15 per security requirements
)
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Validate that production keys are not using development defaults
if os.getenv("PRODUCTION", "false").lower() == "true":
    if SECRET_KEY == "dev_secret_key_2025_not_for_production":
        raise ValueError("Production environment requires proper SECRET_KEY")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handler
security = HTTPBearer()

# CSRF configuration
CSRF_TOKEN_LENGTH = 32
CSRF_HEADER_NAME = "X-CSRF-Token"
CSRF_COOKIE_NAME = "csrf_token"


class UserRole(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    RESEARCHER = "researcher"
    OBSERVER = "observer"
    AGENT_MANAGER = "agent_manager"


class Permission(str, Enum):
    """Permissions for fine-grained access control."""

    CREATE_AGENT = "create_agent"
    DELETE_AGENT = "delete_agent"
    VIEW_AGENTS = "view_agents"
    MODIFY_AGENT = "modify_agent"
    CREATE_COALITION = "create_coalition"
    VIEW_METRICS = "view_metrics"
    ADMIN_SYSTEM = "admin_system"


# Role-permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.CREATE_AGENT,
        Permission.DELETE_AGENT,
        Permission.VIEW_AGENTS,
        Permission.MODIFY_AGENT,
        Permission.CREATE_COALITION,
        Permission.VIEW_METRICS,
        Permission.ADMIN_SYSTEM,
    ],
    UserRole.RESEARCHER: [
        Permission.CREATE_AGENT,
        Permission.VIEW_AGENTS,
        Permission.MODIFY_AGENT,
        Permission.CREATE_COALITION,
        Permission.VIEW_METRICS,
    ],
    UserRole.AGENT_MANAGER: [
        Permission.CREATE_AGENT,
        Permission.VIEW_AGENTS,
        Permission.MODIFY_AGENT,
        Permission.VIEW_METRICS,
    ],
    UserRole.OBSERVER: [Permission.VIEW_AGENTS, Permission.VIEW_METRICS],
}


class User(BaseModel):
    """User model for authentication."""

    model_config = {"arbitrary_types_allowed": True}

    user_id: str
    username: str
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class TokenData(BaseModel):
    """JWT token data."""

    model_config = {"arbitrary_types_allowed": True}

    user_id: str
    username: str
    role: UserRole
    permissions: List[Permission]
    exp: datetime


class SecurityValidator:
    """Input validation and sanitization."""

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(\bdrop\b.*\btable\b)",
        r"(\balter\b.*\btable\b)",
        r"(;|\b(exec|execute)\b)",
        r"(\bor\b.*=.*\bor\b)",
        r"(\band\b.*=.*\band\b)",
        r"('.*'.*=.*'.*')",
        r"(--|\#|\/\*|\*\/)",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"(;|\||&|`|\$\(|\$\{)",
        r"(\bnc\b|\bnetcat\b)",
        r"(\bwget\b|\bcurl\b)",
        r"(\bssh\b|\btelnet\b)",
        r"(\bftp\b|\bsftp\b)",
        r"(\brm\b|\bmv\b|\bcp\b)",
        r"(\bchmod\b|\bchown\b)",
        r"(\bsu\b|\bsudo\b)",
    ]

    @classmethod
    def validate_sql_input(cls, input_str: str) -> bool:
        """Validate input for SQL injection attempts."""
        if not isinstance(input_str, str):
            return True

        input_lower = input_str.lower()
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                logger.warning(f"SQL injection attempt detected: {pattern}")
                return False
        return True

    @classmethod
    def validate_xss_input(cls, input_str: str) -> bool:
        """Validate input for XSS attempts."""
        if not isinstance(input_str, str):
            return True

        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, input_str, re.IGNORECASE):
                logger.warning(f"XSS attempt detected: {pattern}")
                return False
        return True

    @classmethod
    def validate_command_injection(cls, input_str: str) -> bool:
        """Validate input for command injection attempts."""
        if not isinstance(input_str, str):
            return True

        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, input_str, re.IGNORECASE):
                logger.warning(
                    f"Command injection attempt detected: {pattern}"
                )
                return False
        return True

    @classmethod
    def sanitize_gmn_spec(cls, gmn_spec: str) -> str:
        """Sanitize GMN specification input."""
        if not isinstance(gmn_spec, str):
            raise ValueError("GMN spec must be a string")

        # Validate against injection attacks
        if not cls.validate_sql_input(gmn_spec):
            raise ValueError("Invalid GMN spec: SQL injection detected")

        if not cls.validate_xss_input(gmn_spec):
            raise ValueError("Invalid GMN spec: XSS attempt detected")

        if not cls.validate_command_injection(gmn_spec):
            raise ValueError("Invalid GMN spec: Command injection detected")

        # Size limit
        if len(gmn_spec) > 100000:  # 100KB limit
            raise ValueError("GMN spec too large (max 100KB)")

        # Basic structure validation
        try:
            # Attempt to parse as JSON to check basic structure
            if gmn_spec.strip().startswith("{"):
                json.loads(gmn_spec)
        except json.JSONDecodeError:
            # If not JSON, validate basic GMN syntax
            if not re.match(r'^[a-zA-Z0-9\s\[\]{}(),.:"_-]+$', gmn_spec):
                raise ValueError(
                    "Invalid GMN spec: Contains illegal characters"
                )

        return gmn_spec

    @classmethod
    def sanitize_observation_data(
        cls, observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sanitize observation data."""
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a dictionary")

        sanitized = {}

        for key, value in observation.items():
            # Validate key
            if not isinstance(key, str) or len(key) > 100:
                raise ValueError(f"Invalid observation key: {key}")

            if not cls.validate_sql_input(key):
                raise ValueError(
                    f"Invalid observation key (SQL injection): {key}"
                )

            # Sanitize value based on type
            if isinstance(value, str):
                if not cls.validate_sql_input(value):
                    raise ValueError(
                        f"Invalid observation value (SQL injection): {value}"
                    )
                if not cls.validate_xss_input(value):
                    raise ValueError(
                        f"Invalid observation value (XSS): {value}"
                    )
                if len(value) > 10000:  # 10KB limit per string
                    raise ValueError(f"Observation value too large: {key}")
                sanitized[key] = value
            elif isinstance(value, (int, float)):
                # Validate numeric ranges
                if abs(value) > 1e15:  # Reasonable limit
                    raise ValueError(f"Observation value out of range: {key}")
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                if len(value) > 1000:  # Limit array size
                    raise ValueError(f"Observation array too large: {key}")
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_observation_data(
                    value
                )  # Recursive
            else:
                # Allow None, bool
                if value is not None and not isinstance(value, bool):
                    raise ValueError(f"Invalid observation value type: {key}")
                sanitized[key] = value

        return sanitized


class CSRFProtection:
    """CSRF protection utilities."""

    def __init__(self):
        self._token_store = {}  # In production, use Redis or similar

    def generate_csrf_token(self, session_id: str) -> str:
        """Generate a new CSRF token for the session."""
        token = secrets.token_urlsafe(CSRF_TOKEN_LENGTH)
        self._token_store[session_id] = token
        return token

    def verify_csrf_token(self, session_id: str, token: str) -> bool:
        """Verify CSRF token matches the session."""
        stored_token = self._token_store.get(session_id)
        if not stored_token:
            return False
        return hmac.compare_digest(stored_token, token)

    def invalidate_csrf_token(self, session_id: str):
        """Invalidate CSRF token for session."""
        if session_id in self._token_store:
            del self._token_store[session_id]


class AuthenticationManager:
    """JWT-based authentication manager with enhanced security."""

    def __init__(self):
        self.users = {}  # In production, use database
        self.csrf_protection = CSRFProtection()
        self._fingerprint_store = {}  # Store token fingerprints

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return pwd_context.hash(password)

    def verify_password(
        self, plain_password: str, hashed_password: str
    ) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def create_access_token(
        self, user: User, client_fingerprint: Optional[str] = None
    ) -> str:
        """Create JWT access token using secure handler."""
        permissions = ROLE_PERMISSIONS.get(user.role, [])

        # Generate fingerprint if not provided
        if client_fingerprint is None:
            client_fingerprint = jwt_handler.generate_fingerprint()

        # Store fingerprint for this user
        self._fingerprint_store[user.user_id] = client_fingerprint

        return jwt_handler.create_access_token(
            user_id=user.user_id,
            username=user.username,
            role=user.role.value,
            permissions=[p.value for p in permissions],
            fingerprint=client_fingerprint,
        )

    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token using secure handler."""
        token, family_id = jwt_handler.create_refresh_token(user.user_id)
        return token

    def verify_token(
        self, token: str, client_fingerprint: Optional[str] = None
    ) -> TokenData:
        """Verify and decode JWT token using secure handler."""
        payload = jwt_handler.verify_access_token(
            token, fingerprint=client_fingerprint
        )

        return TokenData(
            user_id=payload["user_id"],
            username=payload["username"],
            role=UserRole(payload["role"]),
            permissions=[Permission(p) for p in payload["permissions"]],
            exp=datetime.fromtimestamp(payload["exp"]),
        )

    def register_user(
        self, username: str, email: str, password: str, role: UserRole
    ) -> User:
        """Register new user."""
        if username in self.users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )

        user_id = secrets.token_urlsafe(16)
        hashed_password = self.hash_password(password)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            created_at=datetime.utcnow(),
        )

        self.users[username] = {"user": user, "password_hash": hashed_password}

        return user

    def authenticate_user(
        self, username: str, password: str
    ) -> Optional[User]:
        """Authenticate user with username/password."""
        user_data = self.users.get(username)
        if not user_data:
            return None

        if not self.verify_password(password, user_data["password_hash"]):
            return None

        user = user_data["user"]
        user.last_login = datetime.utcnow()
        return user

    def refresh_access_token(self, refresh_token: str) -> Tuple[str, str]:
        """Refresh access token using refresh token with rotation."""
        # Use secure JWT handler for refresh token rotation
        user_id = self._extract_user_id_from_refresh_token(refresh_token)

        # Find user by user_id
        user_data = None
        for username, data in self.users.items():
            if data["user"].user_id == user_id:
                user_data = data["user"]
                break

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Rotate refresh token
        (
            new_access_token,
            new_refresh_token,
            family_id,
        ) = jwt_handler.rotate_refresh_token(refresh_token, user_id)

        return new_access_token, new_refresh_token

    def _extract_user_id_from_refresh_token(self, refresh_token: str) -> str:
        """Extract user ID from refresh token for lookup."""
        try:
            # Decode without full verification just to get user_id
            unverified = jwt.decode(
                refresh_token, options={"verify_signature": False}
            )
            return unverified.get("user_id", "")
        except:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

    def logout(self, token: str, user_id: Optional[str] = None):
        """Logout user by revoking tokens."""
        # Revoke the current access token
        jwt_handler.revoke_token(token)

        # Revoke all user tokens if user_id provided
        if user_id:
            jwt_handler.revoke_user_tokens(user_id)

        # Invalidate CSRF token for the session
        if user_id:
            self.csrf_protection.invalidate_csrf_token(user_id)

    def set_token_cookie(
        self, response: Response, token: str, secure: bool = True
    ):
        """Set JWT token in secure httpOnly cookie."""
        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=secure,  # HTTPS only in production
            samesite="strict",  # CSRF protection
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            path="/",
        )

    def set_csrf_cookie(
        self, response: Response, csrf_token: str, secure: bool = True
    ):
        """Set CSRF token in cookie (readable by JS)."""
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=csrf_token,
            httponly=False,  # Needs to be readable by JavaScript
            secure=secure,
            samesite="strict",
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            path="/",
        )

    def cleanup_blacklist(self):
        """Cleanup expired tokens from blacklist."""
        jwt_handler.blacklist._cleanup()


class RateLimiter:
    """Request rate limiting to prevent resource exhaustion."""

    def __init__(self):
        self.requests = {}  # IP -> list of request times
        self.user_requests = {}  # user_id -> list of request times

    def is_rate_limited(
        self, identifier: str, max_requests: int = 100, window_minutes: int = 1
    ) -> bool:
        """Check if identifier is rate limited."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=window_minutes)

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Remove old requests
        self.requests[identifier] = [
            req_time
            for req_time in self.requests[identifier]
            if req_time > window_start
        ]

        # Check rate limit
        if len(self.requests[identifier]) >= max_requests:
            return True

        # Record current request
        self.requests[identifier].append(now)
        return False

    def clear_old_requests(self):
        """Cleanup old request records."""
        cutoff = datetime.utcnow() - timedelta(hours=1)

        for identifier in list(self.requests.keys()):
            self.requests[identifier] = [
                req_time
                for req_time in self.requests[identifier]
                if req_time > cutoff
            ]

            if not self.requests[identifier]:
                del self.requests[identifier]


# Global instances
auth_manager = AuthenticationManager()
rate_limiter = RateLimiter()
security_validator = SecurityValidator()


def get_client_ip(request: Request) -> str:
    """Extract client IP address."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(max_requests: int = 100, window_minutes: int = 1):
    """Rate limiting decorator."""

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = get_client_ip(request)

            if rate_limiter.is_rate_limited(
                client_ip, max_requests, window_minutes
            ):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """Get current authenticated user with fingerprint validation."""
    # Extract fingerprint from cookie or header
    fingerprint = request.cookies.get("__Secure-Fgp", None)
    if not fingerprint:
        fingerprint = request.headers.get("X-Fingerprint", None)

    return auth_manager.verify_token(
        credentials.credentials, client_fingerprint=fingerprint
    )


async def validate_csrf_token(request: Request) -> str:
    """Validate CSRF token from request and return the token."""
    # Get session ID from authorization header
    session_id = None
    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            unverified = jwt.decode(token, options={"verify_signature": False})
            session_id = unverified.get("user_id")
        except:
            pass

    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session required for CSRF validation",
        )

    # Get CSRF token from header or form
    csrf_token = request.headers.get(CSRF_HEADER_NAME)
    if not csrf_token and hasattr(request, "form"):
        form_data = await request.form()
        csrf_token = form_data.get("csrf_token")

    if not csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token required"
        )

    # Validate CSRF token
    if not auth_manager.csrf_protection.verify_csrf_token(
        session_id, csrf_token
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token"
        )

    return csrf_token


def require_permission(permission: Permission):
    """Require specific permission for endpoint access."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = None
            for arg in args:
                if isinstance(arg, TokenData):
                    current_user = arg
                    break

            if not current_user:
                for value in kwargs.values():
                    if isinstance(value, TokenData):
                        current_user = value
                        break

            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if permission not in current_user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(role: UserRole):
    """Require specific role for endpoint access."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = None
            for arg in args:
                if isinstance(arg, TokenData):
                    current_user = arg
                    break

            if not current_user:
                for value in kwargs.values():
                    if isinstance(value, TokenData):
                        current_user = value
                        break

            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if (
                current_user.role != role
                and current_user.role != UserRole.ADMIN
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {role}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def secure_database_query(query_func):
    """Secure database query wrapper with SQL injection protection."""

    @wraps(query_func)
    def wrapper(*args, **kwargs):
        try:
            # Validate all string arguments
            for arg in args:
                if isinstance(arg, str):
                    if not security_validator.validate_sql_input(arg):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid input detected",
                        )

            for key, value in kwargs.items():
                if isinstance(value, str):
                    if not security_validator.validate_sql_input(value):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid input detected",
                        )

            return query_func(*args, **kwargs)

        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred",
            )

    return wrapper


def require_csrf_token(func):
    """Require valid CSRF token for state-changing operations."""

    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        # Get session ID (from user token or session)
        session_id = None

        # Try to get user from request
        for arg in args:
            if isinstance(arg, TokenData):
                session_id = arg.user_id
                break

        if not session_id:
            for value in kwargs.values():
                if isinstance(value, TokenData):
                    session_id = value.user_id
                    break

        if not session_id:
            # Try to extract from authorization header
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                try:
                    token = auth_header.split(" ")[1]
                    unverified = jwt.decode(
                        token, options={"verify_signature": False}
                    )
                    session_id = unverified.get("user_id")
                except:
                    pass

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session required for CSRF validation",
            )

        # Get CSRF token from header or form
        csrf_token = request.headers.get(CSRF_HEADER_NAME)
        if not csrf_token and hasattr(request, "form"):
            form_data = await request.form()
            csrf_token = form_data.get("csrf_token")

        if not csrf_token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token required",
            )

        # Validate CSRF token
        if not auth_manager.csrf_protection.verify_csrf_token(
            session_id, csrf_token
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid CSRF token",
            )

        return await func(request, *args, **kwargs)

    return wrapper


# Security middleware for FastAPI
class SecurityMiddleware:
    """Security middleware for request validation."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Basic security headers
            headers = dict(scope.get("headers", []))

            # Check for security headers
            response_headers = [
                (b"X-Content-Type-Options", b"nosniff"),
                (b"X-Frame-Options", b"DENY"),
                (b"X-XSS-Protection", b"1; mode=block"),
                (
                    b"Strict-Transport-Security",
                    b"max-age=31536000; includeSubDomains",
                ),
                (
                    b"Content-Security-Policy",
                    b"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
                ),
                (b"Referrer-Policy", b"strict-origin-when-cross-origin"),
                (
                    b"Permissions-Policy",
                    b"geolocation=(), microphone=(), camera=()",
                ),
            ]

            # Modify response to include security headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    message.setdefault("headers", [])
                    message["headers"].extend(response_headers)
                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

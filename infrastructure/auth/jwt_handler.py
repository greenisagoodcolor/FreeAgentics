"""
FreeAgentics JWT Token Handler
Production-grade JWT implementation with security best practices
"""

import base64
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class JWTSecurityError(Exception):
    """JWT security related errors"""
    pass


class JWTHandler:
    """Enterprise-grade JWT token handler with security best practices"""

    def __init__(self):
        # Load configuration from environment
        self.secret_key = self._get_secret_key()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(
            os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15")
        )
        self.refresh_token_expire_days = int(
            os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30")
        )
        self.issuer = os.getenv("JWT_ISSUER", "freeagentics-api")
        self.audience = os.getenv("JWT_AUDIENCE", "freeagentics-clients")

        # Security settings
        self.max_token_age_days = 90  # Maximum token age before forced refresh
        self.token_blacklist = set()  # In-memory blacklist (use Redis in production)

    def _get_secret_key(self) -> str:
        """Get JWT secret key with validation"""
        key = os.getenv("JWT_SECRET_KEY")
        if not key:
            # For testing environments, generate a temporary key
            if os.getenv("TESTING") == "1" or "pytest" in os.getenv("_", ""):
                key = "test_jwt_secret_key_32_characters_long_development_only"
            else:
                raise JWTSecurityError("JWT_SECRET_KEY environment variable is required")

        if len(key) < 32:
            raise JWTSecurityError("JWT_SECRET_KEY must be at least 32 characters long")

        # Derive a secure key using PBKDF2
        salt = os.getenv("JWT_SALT", "freeagentics_jwt_salt_2024").encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        return derived_key.decode()

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token with security claims"""
        to_encode = data.copy()
        
        # Add standard claims
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": now,
            "nbf": now,
            "iss": self.issuer,
            "aud": self.audience,
            "jti": str(uuid.uuid4()),  # JWT ID for tracking
            "type": "access"
        })

        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            raise JWTSecurityError(f"Token creation failed: {str(e)}")

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        
        # Add standard claims
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": now,
            "nbf": now,
            "iss": self.issuer,
            "aud": self.audience,
            "jti": str(uuid.uuid4()),
            "type": "refresh"
        })

        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            raise JWTSecurityError(f"Refresh token creation failed: {str(e)}")

    def verify_access_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode access token"""
        try:
            # Check if token is blacklisted
            if token in self.token_blacklist:
                raise JWTSecurityError("Token has been revoked")

            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "nbf", "iss", "aud", "jti"]}
            )

            # Verify token type
            if payload.get("type") != "access":
                raise JWTSecurityError("Invalid token type")

            return payload

        except jwt.ExpiredSignatureError:
            raise JWTSecurityError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise JWTSecurityError(f"Token validation failed: {str(e)}")
        except Exception as e:
            raise JWTSecurityError(f"Token verification error: {str(e)}")

    def verify_refresh_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode refresh token"""
        try:
            # Check if token is blacklisted
            if token in self.token_blacklist:
                raise JWTSecurityError("Refresh token has been revoked")

            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "nbf", "iss", "aud", "jti"]}
            )

            # Verify token type
            if payload.get("type") != "refresh":
                raise JWTSecurityError("Invalid refresh token type")

            return payload

        except jwt.ExpiredSignatureError:
            raise JWTSecurityError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise JWTSecurityError(f"Refresh token validation failed: {str(e)}")
        except Exception as e:
            raise JWTSecurityError(f"Refresh token verification error: {str(e)}")

    def decode_token_without_verification(self, token: str) -> Dict[str, Any]:
        """Decode token without verification (for debugging)"""
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception as e:
            raise JWTSecurityError(f"Token decoding failed: {str(e)}")

    def revoke_token(self, token: str) -> bool:
        """Add token to blacklist"""
        try:
            # In production, store in Redis with expiration
            self.token_blacklist.add(token)
            return True
        except Exception:
            return False

    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        return token in self.token_blacklist

    def get_token_claims(self, token: str) -> Dict[str, Any]:
        """Get token claims without full verification (for inspection)"""
        try:
            # Get header and payload without verification
            unverified_header = jwt.get_unverified_header(token)
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            
            info = {
                "header": unverified_header,
                "payload": unverified_payload,
            }

            # Add readable expiration info
            if "exp" in unverified_payload:
                exp_timestamp = unverified_payload["exp"]
                expires_at = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
                info["expires_at"] = expires_at.isoformat()
                info["is_expired"] = datetime.now(timezone.utc) > expires_at

            return info

        except Exception as e:
            return {"error": str(e)}


# Global JWT handler instance (lazy initialization)
_jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """Get JWT handler instance with lazy initialization"""
    global _jwt_handler
    if _jwt_handler is None:
        _jwt_handler = JWTHandler()
    return _jwt_handler
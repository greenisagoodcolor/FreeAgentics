"""Secure JWT Handler with OWASP Best Practices.

This module implements production-grade JWT handling with:
- RS256 algorithm (RSA public/private key)
- Token fingerprinting to prevent theft
- Refresh token rotation
- Token revocation/blacklist
- Proper expiration times
- JWT ID (jti) tracking
"""

import hashlib
import logging
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
)
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # OWASP recommendation: 15 minutes
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days for refresh tokens
TOKEN_ISSUER = "freeagentics-auth"  # nosec B105
TOKEN_AUDIENCE = "freeagentics-api"  # nosec B105

# Key paths
PRIVATE_KEY_PATH = os.path.join(os.path.dirname(__file__), "keys", "jwt_private.pem")
PUBLIC_KEY_PATH = os.path.join(os.path.dirname(__file__), "keys", "jwt_public.pem")

# Key rotation configuration
KEY_ROTATION_DAYS = 90  # Rotate keys every 90 days
KEY_ROTATION_WARNING_DAYS = 7  # Warn 7 days before rotation


class TokenBlacklist:
    """In-memory token blacklist for revocation.

    In production, use Redis or similar persistent storage.
    """

    def __init__(self) -> None:
        """Initialize token blacklist with cleanup configuration."""
        self._blacklist: Dict[str, float] = {}  # jti -> expiration time
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # Clean up expired entries every hour

    def add(self, jti: str, exp: datetime) -> None:
        """Add token to blacklist."""
        self._blacklist[jti] = exp.timestamp()
        self._cleanup()

    def is_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted."""
        self._cleanup()
        return jti in self._blacklist

    def _cleanup(self) -> None:
        """Remove expired entries from blacklist."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        # Remove expired tokens
        expired_tokens = [jti for jti, exp in self._blacklist.items() if exp < current_time]
        for jti in expired_tokens:
            del self._blacklist[jti]

        self._last_cleanup = current_time

        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens from blacklist")


class RefreshTokenStore:
    """Secure refresh token storage with rotation.

    In production, use encrypted database storage.
    """

    def __init__(self) -> None:
        """Initialize refresh token store with token family tracking."""
        self._tokens: Dict[str, Dict[str, Any]] = (
            {}
        )  # user_id -> {token_hash, family_id, created_at}
        self._token_families: Dict[str, List[str]] = {}  # family_id -> [token_hashes]

    def store(self, user_id: str, token: str, family_id: Optional[str] = None) -> str:
        """Store refresh token and return family ID."""
        token_hash = self._hash_token(token)

        if family_id is None:
            family_id = secrets.token_urlsafe(32)

        # Store token
        self._tokens[user_id] = {
            "token_hash": token_hash,
            "family_id": family_id,
            "created_at": datetime.utcnow(),
        }

        # Track token family
        if family_id not in self._token_families:
            self._token_families[family_id] = []
        self._token_families[family_id].append(token_hash)

        return family_id

    def verify_and_rotate(self, user_id: str, token: str) -> Optional[str]:
        """Verify token and return family ID for rotation."""
        token_hash = self._hash_token(token)

        if user_id not in self._tokens:
            return None

        stored = self._tokens[user_id]
        if stored["token_hash"] != token_hash:
            # Token mismatch - possible theft, invalidate entire family
            family_id = stored.get("family_id")
            if family_id:
                self._invalidate_family(family_id)
            return None

        family_id = stored.get("family_id")
        return family_id if isinstance(family_id, str) else None

    def invalidate(self, user_id: str) -> None:
        """Invalidate user's refresh token."""
        if user_id in self._tokens:
            family_id = self._tokens[user_id].get("family_id")
            if family_id:
                self._invalidate_family(family_id)
            # Only delete if still exists (may have been removed by _invalidate_family)
            if user_id in self._tokens:
                del self._tokens[user_id]

    def _invalidate_family(self, family_id: str) -> None:
        """Invalidate entire token family (theft detection)."""
        if family_id in self._token_families:
            # Find and remove all tokens in family
            for user_id, data in list(self._tokens.items()):
                if data.get("family_id") == family_id:
                    del self._tokens[user_id]
            del self._token_families[family_id]
            logger.warning(f"Invalidated token family {family_id} due to possible theft")

    def _hash_token(self, token: str) -> str:
        """Hash token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()


class JWTHandler:
    """Secure JWT handler with OWASP best practices."""

    def __init__(self) -> None:
        """Initialize JWT handler with RSA keys, blacklist, and refresh token store."""
        self.private_key: RSAPrivateKey
        self.public_key: RSAPublicKey
        self._load_keys()
        self.blacklist = TokenBlacklist()
        self.refresh_store = RefreshTokenStore()
        self._check_key_rotation()

    def _load_keys(self) -> None:
        """Load RSA keys from files."""
        try:
            with open(PRIVATE_KEY_PATH, "rb") as f:
                self.private_key = cast(
                    RSAPrivateKey,
                    serialization.load_pem_private_key(f.read(), password=None),
                )

            with open(PUBLIC_KEY_PATH, "rb") as f:
                self.public_key = cast(RSAPublicKey, serialization.load_pem_public_key(f.read()))

        except FileNotFoundError:
            logger.error("JWT keys not found. Generating new keys...")
            self._generate_keys()
        except Exception as e:
            logger.error(f"Error loading JWT keys: {e}")
            raise

    def _generate_keys(self) -> None:
        """Generate new RSA key pair."""
        # Generate private key
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,  # Strong key size
        )

        # Get public key
        self.public_key = self.private_key.public_key()

        # Save keys
        os.makedirs(os.path.dirname(PRIVATE_KEY_PATH), exist_ok=True)

        # Save private key
        with open(PRIVATE_KEY_PATH, "wb") as f:
            f.write(
                self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Save public key
        with open(PUBLIC_KEY_PATH, "wb") as f:
            f.write(
                self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )

        # Set appropriate permissions
        os.chmod(PRIVATE_KEY_PATH, 0o600)  # Read/write for owner only
        os.chmod(PUBLIC_KEY_PATH, 0o644)  # Read for all, write for owner

        logger.info("Generated new RSA key pair")

    def _check_key_rotation(self) -> None:
        """Check if keys need rotation."""
        try:
            key_stat = os.stat(PRIVATE_KEY_PATH)
            key_age_days = (time.time() - key_stat.st_mtime) / 86400

            if key_age_days > KEY_ROTATION_DAYS:
                logger.error(f"JWT keys are {key_age_days:.0f} days old - rotation required!")
            elif key_age_days > (KEY_ROTATION_DAYS - KEY_ROTATION_WARNING_DAYS):
                logger.warning(
                    f"JWT keys are {key_age_days:.0f} days old - rotation recommended soon"
                )
        except Exception as e:
            # Log failed key age check
            logger.debug(f"Failed to check JWT key age: {e}")

    def create_access_token(
        self,
        user_id: str,
        username: str,
        role: str,
        permissions: list,
        fingerprint: Optional[str] = None,
    ) -> str:
        """Create secure access token with fingerprinting."""
        now = datetime.now(timezone.utc)
        jti = secrets.token_urlsafe(32)

        # Token payload
        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "permissions": permissions,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            "nbf": now,  # Not before
            "iss": TOKEN_ISSUER,
            "aud": TOKEN_AUDIENCE,
            "jti": jti,  # Unique token ID
        }

        # Add fingerprint hash if provided
        if fingerprint:
            payload["fingerprint"] = self._hash_fingerprint(fingerprint)

        # Create token
        token = jwt.encode(payload, self.private_key, algorithm=JWT_ALGORITHM)

        return token

    def create_refresh_token(
        self, user_id: str, family_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """Create refresh token with rotation support."""
        now = datetime.now(timezone.utc)
        jti = secrets.token_urlsafe(32)

        payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "nbf": now,
            "iss": TOKEN_ISSUER,
            "aud": TOKEN_AUDIENCE,
            "jti": jti,
        }

        # Create token
        token = jwt.encode(payload, self.private_key, algorithm=JWT_ALGORITHM)

        # Store for rotation tracking
        family_id = self.refresh_store.store(user_id, token, family_id)

        return token, family_id

    def verify_access_token(self, token: str, fingerprint: Optional[str] = None) -> Dict[str, Any]:
        """Verify access token with fingerprint checking."""
        try:
            # Decode and verify
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[JWT_ALGORITHM],
                issuer=TOKEN_ISSUER,
                audience=TOKEN_AUDIENCE,
                options={"require": ["exp", "iat", "nbf", "jti"]},
            )

            # Check token type
            if payload.get("type") != "access":
                raise jwt.InvalidTokenError("Invalid token type")

            # Check blacklist
            jti = payload.get("jti")
            if jti and self.blacklist.is_blacklisted(jti):
                raise jwt.InvalidTokenError("Token has been revoked")

            # Verify fingerprint if provided
            if fingerprint:
                stored_fingerprint = payload.get("fingerprint")
                if not stored_fingerprint or stored_fingerprint != self._hash_fingerprint(
                    fingerprint
                ):
                    raise jwt.InvalidTokenError("Invalid token fingerprint")

            return cast(Dict[str, Any], payload)

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Access token expired",
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid access token: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed",
            )

    def verify_refresh_token(self, token: str, user_id: str) -> Dict[str, Any]:
        """Verify refresh token and check for reuse."""
        try:
            # Decode and verify
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[JWT_ALGORITHM],
                issuer=TOKEN_ISSUER,
                audience=TOKEN_AUDIENCE,
                options={"require": ["exp", "iat", "nbf", "jti"]},
            )

            # Check token type
            if payload.get("type") != "refresh":
                raise jwt.InvalidTokenError("Invalid token type")

            # Check user ID matches
            if payload.get("user_id") != user_id:
                raise jwt.InvalidTokenError("Token user mismatch")

            # Check blacklist
            jti = payload.get("jti")
            if jti and self.blacklist.is_blacklisted(jti):
                raise jwt.InvalidTokenError("Token has been revoked")

            # Verify with refresh store (detects reuse)
            family_id = self.refresh_store.verify_and_rotate(user_id, token)
            if family_id is None:
                raise jwt.InvalidTokenError("Invalid or reused refresh token")

            payload["family_id"] = family_id
            return cast(Dict[str, Any], payload)

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token expired",
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid refresh token: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Refresh token verification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed",
            )

    def rotate_refresh_token(self, old_token: str, user_id: str) -> Tuple[str, str, str]:
        """Rotate refresh token and create new access token."""
        # Verify old token
        payload = self.verify_refresh_token(old_token, user_id)
        family_id = payload.get("family_id")

        # Revoke old token
        if payload.get("jti"):
            self.blacklist.add(payload["jti"], datetime.fromtimestamp(payload["exp"]))

        # Create new tokens
        # Note: In real implementation, fetch user details from database
        new_access_token = self.create_access_token(
            user_id=user_id,
            username=payload.get("username", ""),
            role=payload.get("role", ""),
            permissions=payload.get("permissions", []),
        )

        new_refresh_token, _ = self.create_refresh_token(user_id, family_id)

        return new_access_token, new_refresh_token, family_id or ""

    def revoke_token(self, token: str) -> None:
        """Revoke a token by adding to blacklist."""
        try:
            # Decode without verification to get jti and exp
            # Security Note: This is intentionally unverified because we need to extract
            # claims from potentially invalid/expired tokens for blacklisting purposes.
            # The token is already considered untrusted at this point.
            unverified = jwt.decode(token, options={"verify_signature": False})  # nosec B608

            jti = unverified.get("jti")
            exp = unverified.get("exp")

            if jti and exp:
                self.blacklist.add(jti, datetime.fromtimestamp(exp))
                logger.info(f"Revoked token {jti}")
        except Exception as e:
            logger.error(f"Error revoking token: {e}")

    def revoke_user_tokens(self, user_id: str) -> None:
        """Revoke all tokens for a user."""
        # Invalidate refresh tokens
        self.refresh_store.invalidate(user_id)

        # Note: Access tokens can't be fully revoked without
        # maintaining a user->tokens mapping or using short expiration
        logger.info(f"Revoked all refresh tokens for user {user_id}")

    def _hash_fingerprint(self, fingerprint: str) -> str:
        """Hash fingerprint for secure storage."""
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def generate_fingerprint(self) -> str:
        """Generate a random fingerprint for token binding."""
        return secrets.token_urlsafe(32)

    def get_key_info(self) -> Dict[str, Any]:
        """Get information about current keys."""
        try:
            key_stat = os.stat(PRIVATE_KEY_PATH)
            key_age_days = (time.time() - key_stat.st_mtime) / 86400

            return {
                "algorithm": JWT_ALGORITHM,
                "key_size": self.private_key.key_size,
                "key_age_days": round(key_age_days, 1),
                "rotation_required": key_age_days > KEY_ROTATION_DAYS,
                "rotation_warning": key_age_days > (KEY_ROTATION_DAYS - KEY_ROTATION_WARNING_DAYS),
            }
        except Exception as e:
            logger.error(f"Error getting key info: {e}")
            return {"algorithm": JWT_ALGORITHM, "error": str(e)}


# Global instance
jwt_handler = JWTHandler()

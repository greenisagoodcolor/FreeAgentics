"""Authentication helpers for tests."""

from datetime import datetime, timedelta
from typing import Dict, List

import jwt

# Import the JWT handler for RS256 token creation
try:
    from auth.jwt_handler import jwt_handler
except ImportError:
    # Fallback for environments where auth module isn't available
    jwt_handler = None


def create_test_token(
    user_id: str = "test-user-123",
    username: str = "testuser",
    role: str = "admin",
    permissions: List[str] = None,
    expires_in: int = 30,
) -> str:
    """Create a test JWT token for authentication in tests.

    Args:
        user_id: User ID for the token
        username: Username for the token
        role: User role
        permissions: List of permissions (defaults to all if admin)
        expires_in: Token expiration in minutes

    Returns:
        JWT token string
    """
    if permissions is None and role == "admin":
        # Grant all permissions for admin in tests (match Permission enum values)
        permissions = [
            "create_agent",
            "delete_agent",
            "view_agents",
            "modify_agent",
            "create_coalition",
            "view_metrics",
            "admin_system",
        ]
    elif permissions is None:
        permissions = ["view_agents"]

    # Use the actual JWT handler if available (for RS256)
    if jwt_handler:
        return jwt_handler.create_access_token(
            user_id=user_id, username=username, role=role, permissions=permissions
        )

    # Fallback to RS256 with mock keys for environments without the handler
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Generate a test RSA key for tests
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048  # Smaller key for faster test execution
    )

    # Create token payload compatible with the app
    now = datetime.utcnow()
    payload = {
        "user_id": user_id,  # App uses user_id, not sub
        "username": username,
        "role": role,
        "permissions": permissions,
        "type": "access",
        "exp": now + timedelta(minutes=expires_in),
        "iat": now,
        "nbf": now,
        "iss": "freeagentics-auth",
        "aud": "freeagentics-api",
        "jti": f"test-{user_id}-{now.timestamp()}",
    }

    return jwt.encode(payload, private_key, algorithm="RS256")


def get_auth_headers(token: str = None, **kwargs) -> Dict[str, str]:
    """Get authorization headers for API requests.

    Args:
        token: JWT token (if not provided, creates one)
        **kwargs: Arguments to pass to create_test_token

    Returns:
        Dictionary with Authorization header
    """
    if token is None:
        token = create_test_token(**kwargs)

    return {"Authorization": f"Bearer {token}"}

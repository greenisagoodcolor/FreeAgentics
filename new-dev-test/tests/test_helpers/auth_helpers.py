"""Authentication helpers for testing."""

from auth.security_implementation import TokenData


def mock_auth_dependency():
    """Mock authentication dependency for testing."""
    return TokenData(
        user_id="test_user_123",
        username="test_user",
        email="test@example.com",
        roles=["admin", "user"],
        permissions=["read", "write", "delete"],
        exp=1735689600,  # Far future expiration
        fingerprint=None,
    )


def mock_user_auth_dependency():
    """Mock regular user authentication dependency for testing."""
    return TokenData(
        user_id="regular_user_456",
        username="regular_user",
        email="user@example.com",
        roles=["user"],
        permissions=["read"],
        exp=1735689600,
        fingerprint=None,
    )


def mock_no_permissions_auth_dependency():
    """Mock authentication with no permissions for testing."""
    return TokenData(
        user_id="no_perm_user_789",
        username="no_perm_user",
        email="noperm@example.com",
        roles=[],
        permissions=[],
        exp=1735689600,
        fingerprint=None,
    )

"""Simple dev auth bypass for UI compatibility."""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import Request
from auth.security_implementation import TokenData, UserRole, ROLE_PERMISSIONS

# Dev user constant with required fields
DEV_USER = TokenData(
    user_id="dev_user",
    username="dev_user",
    role=UserRole.ADMIN,
    permissions=[p.value for p in ROLE_PERMISSIONS[UserRole.ADMIN]],
    fingerprint="dev_fingerprint",
    exp=int((datetime.utcnow() + timedelta(days=365)).timestamp())  # 1 year expiry
)

async def get_dev_user(request: Request) -> TokenData:
    """Always return dev user in dev mode."""
    return DEV_USER
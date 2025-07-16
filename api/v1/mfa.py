"""
Multi-Factor Authentication API endpoints for FreeAgentics.

This module provides REST API endpoints for MFA operations including:
- User enrollment in MFA
- MFA token verification
- MFA status and management
- Backup code management
- Hardware key support
"""

import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.middleware.rate_limiter import rate_limit
from auth.mfa_service import MFAEnrollmentRequest, MFAResponse, MFAService, MFAVerificationRequest
from auth.security_implementation import get_current_user, validate_csrf_token
from database.session import get_db
from observability.security_monitoring import SecurityMonitoringSystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mfa", tags=["Multi-Factor Authentication"])

# Rate limiting configuration for MFA endpoints
MFA_RATE_LIMITS = {
    "enroll": "5/5m",  # 5 enrollments per 5 minutes
    "verify": "10/5m",  # 10 verifications per 5 minutes
    "status": "30/1m",  # 30 status checks per minute
    "disable": "3/1h",  # 3 disable operations per hour
    "regenerate": "5/1h",  # 5 backup code regenerations per hour
}


def get_mfa_service(
    db: Session = Depends(get_db),
    security_monitor: SecurityMonitoringSystem = Depends(lambda: SecurityMonitoringSystem()),
) -> MFAService:
    """Dependency to get MFA service instance."""
    return MFAService(db, security_monitor)


@router.post("/enroll", response_model=MFAResponse)
@rate_limit(MFA_RATE_LIMITS["enroll"])
async def enroll_mfa(
    request: MFAEnrollmentRequest,
    current_user: Dict = Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
    csrf_token: str = Depends(validate_csrf_token),
) -> MFAResponse:
    """
    Enroll user in Multi-Factor Authentication.

    This endpoint allows users to enroll in MFA using various methods:
    - TOTP (Time-based One-Time Password) - most common
    - Hardware security keys (FIDO2/WebAuthn)
    - SMS/Email (future implementation)

    Args:
        request: MFA enrollment request
        current_user: Current authenticated user
        mfa_service: MFA service instance
        csrf_token: CSRF token for protection

    Returns:
        MFA enrollment response with QR code and backup codes

    Raises:
        HTTPException: If enrollment fails or user is not authorized
    """
    try:
        # Ensure user is enrolling for themselves
        if request.user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only enroll MFA for your own account",
            )

        # Set user email from current user if not provided
        if not request.user_email:
            request.user_email = current_user.get(
                "email", f"user_{request.user_id}@freeagentics.com"
            )

        # Enroll user in MFA
        result = await mfa_service.enroll_user(request)

        if result.success:
            logger.info(f"MFA enrollment successful for user {request.user_id}")
            return result
        else:
            logger.warning(f"MFA enrollment failed for user {request.user_id}: {result.message}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA enrollment error for user {request.user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MFA enrollment failed"
        )


@router.post("/verify", response_model=MFAResponse)
@rate_limit(MFA_RATE_LIMITS["verify"])
async def verify_mfa(
    request: MFAVerificationRequest,
    current_user: Dict = Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
    csrf_token: str = Depends(validate_csrf_token),
) -> MFAResponse:
    """
    Verify MFA token.

    This endpoint verifies MFA tokens including:
    - TOTP codes from authenticator apps
    - Backup codes (single-use)
    - Hardware security key responses

    Args:
        request: MFA verification request
        current_user: Current authenticated user
        mfa_service: MFA service instance
        csrf_token: CSRF token for protection

    Returns:
        MFA verification response

    Raises:
        HTTPException: If verification fails or user is not authorized
    """
    try:
        # Ensure user is verifying for themselves
        if request.user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only verify MFA for your own account",
            )

        # Verify MFA token
        result = await mfa_service.verify_mfa(request)

        if result.success:
            logger.info(f"MFA verification successful for user {request.user_id}")
            return result
        else:
            logger.warning(f"MFA verification failed for user {request.user_id}: {result.message}")

            # Return appropriate HTTP status based on failure reason
            if "too many failed attempts" in result.message.lower():
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=result.message
                )
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA verification error for user {request.user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MFA verification failed"
        )


@router.get("/status")
@rate_limit(MFA_RATE_LIMITS["status"])
async def get_mfa_status(
    current_user: Dict = Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
) -> Dict:
    """
    Get MFA status for current user.

    Returns comprehensive MFA status including:
    - Whether MFA is enabled
    - Available MFA methods
    - Remaining backup codes
    - Lockout status

    Args:
        current_user: Current authenticated user
        mfa_service: MFA service instance

    Returns:
        MFA status dictionary

    Raises:
        HTTPException: If status retrieval fails
    """
    try:
        user_id = current_user["user_id"]
        status_info = await mfa_service.get_mfa_status(user_id)

        logger.info(f"MFA status retrieved for user {user_id}")
        return status_info

    except Exception as e:
        logger.error(f"MFA status retrieval error for user {current_user['user_id']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve MFA status",
        )


@router.post("/disable", response_model=MFAResponse)
@rate_limit(MFA_RATE_LIMITS["disable"])
async def disable_mfa(
    current_user: Dict = Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
    csrf_token: str = Depends(validate_csrf_token),
) -> MFAResponse:
    """
    Disable MFA for current user.

    This endpoint completely disables MFA for the user, removing:
    - TOTP secrets
    - Backup codes
    - Hardware key registrations
    - All MFA-related settings

    Args:
        current_user: Current authenticated user
        mfa_service: MFA service instance
        csrf_token: CSRF token for protection

    Returns:
        MFA disable response

    Raises:
        HTTPException: If disable operation fails
    """
    try:
        user_id = current_user["user_id"]
        result = await mfa_service.disable_mfa(user_id)

        if result.success:
            logger.info(f"MFA disabled for user {user_id}")
            return result
        else:
            logger.warning(f"MFA disable failed for user {user_id}: {result.message}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA disable error for user {current_user['user_id']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to disable MFA"
        )


@router.post("/regenerate-backup-codes", response_model=MFAResponse)
@rate_limit(MFA_RATE_LIMITS["regenerate"])
async def regenerate_backup_codes(
    current_user: Dict = Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
    csrf_token: str = Depends(validate_csrf_token),
) -> MFAResponse:
    """
    Regenerate backup codes for current user.

    This endpoint generates new backup codes, invalidating all existing ones.
    Backup codes are single-use and provide recovery access if the primary
    MFA device is unavailable.

    Args:
        current_user: Current authenticated user
        mfa_service: MFA service instance
        csrf_token: CSRF token for protection

    Returns:
        MFA response with new backup codes

    Raises:
        HTTPException: If regeneration fails
    """
    try:
        user_id = current_user["user_id"]
        result = await mfa_service.regenerate_backup_codes(user_id)

        if result.success:
            logger.info(f"Backup codes regenerated for user {user_id}")
            return result
        else:
            logger.warning(f"Backup code regeneration failed for user {user_id}: {result.message}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backup code regeneration error for user {current_user['user_id']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate backup codes",
        )


@router.get("/methods")
@rate_limit(MFA_RATE_LIMITS["status"])
async def get_available_methods() -> Dict:
    """
    Get available MFA methods.

    Returns information about supported MFA methods and their capabilities.

    Returns:
        Dictionary of available MFA methods
    """
    return {
        "methods": [
            {
                "name": "totp",
                "display_name": "Time-based One-Time Password",
                "description": "Use an authenticator app like Google Authenticator or Authy",
                "enabled": True,
                "setup_required": True,
            },
            {
                "name": "backup_code",
                "display_name": "Backup Codes",
                "description": "Single-use recovery codes",
                "enabled": True,
                "setup_required": False,
            },
            {
                "name": "hardware_key",
                "display_name": "Hardware Security Key",
                "description": "FIDO2/WebAuthn compatible security keys",
                "enabled": False,  # Not yet implemented
                "setup_required": True,
            },
            {
                "name": "sms",
                "display_name": "SMS Verification",
                "description": "Receive codes via SMS",
                "enabled": False,  # Not yet implemented
                "setup_required": True,
            },
            {
                "name": "email",
                "display_name": "Email Verification",
                "description": "Receive codes via email",
                "enabled": False,  # Not yet implemented
                "setup_required": True,
            },
        ],
        "recommended_method": "totp",
        "backup_method": "backup_code",
    }


@router.post("/test-token")
@rate_limit("3/1m")  # Very restrictive for testing
async def test_mfa_token(
    token: str,
    current_user: Dict = Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
) -> Dict:
    """
    Test an MFA token without completing enrollment.

    This endpoint allows users to test their TOTP setup during enrollment
    without actually enabling MFA. This helps ensure the authenticator app
    is configured correctly.

    Args:
        token: TOTP token to test
        current_user: Current authenticated user
        mfa_service: MFA service instance

    Returns:
        Test result dictionary

    Raises:
        HTTPException: If test fails
    """
    try:
        user_id = current_user["user_id"]

        # Create test verification request
        test_request = MFAVerificationRequest(user_id=user_id, token=token, method="totp")

        # Get MFA settings (should exist but not be enabled yet)
        mfa_status = await mfa_service.get_mfa_status(user_id)

        if mfa_status["enabled"]:
            return {
                "success": False,
                "message": "MFA is already enabled. Use the verify endpoint instead.",
            }

        # Test the token
        result = await mfa_service.verify_mfa(test_request)

        return {
            "success": result.success,
            "message": "Token is valid" if result.success else "Token is invalid",
            "ready_for_enrollment": result.success,
        }

    except Exception as e:
        logger.error(f"MFA token test error for user {current_user['user_id']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to test MFA token"
        )


# Health check endpoint for MFA service
@router.get("/health")
async def mfa_health_check(mfa_service: MFAService = Depends(get_mfa_service)) -> Dict:
    """
    Health check for MFA service.

    Returns:
        Health status of MFA service
    """
    try:
        # Basic health check - ensure encryption key is available
        test_secret = "test_secret_for_health_check"
        encrypted = mfa_service._encrypt_secret(test_secret)
        decrypted = mfa_service._decrypt_secret(encrypted)

        encryption_healthy = decrypted == test_secret

        return {
            "status": "healthy" if encryption_healthy else "unhealthy",
            "encryption": "ok" if encryption_healthy else "failed",
            "timestamp": "2025-01-16T12:00:00Z",
        }

    except Exception as e:
        logger.error(f"MFA health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e), "timestamp": "2025-01-16T12:00:00Z"}

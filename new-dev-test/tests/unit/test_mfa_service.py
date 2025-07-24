"""
Unit tests for MFA Service.

Tests cover:
- TOTP enrollment and verification
- Backup code generation and validation
- Rate limiting and security features
- Error handling and edge cases
- Security event logging
"""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from auth.mfa_service import MFAEnrollmentRequest, MFAService, MFASettings, MFAVerificationRequest
from auth.security_logging import SecurityEventType


class TestMFAService:
    """Test suite for MFA Service."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return Mock()

    @pytest.fixture
    def mock_security_monitor(self):
        """Mock security monitor."""
        return Mock()

    @pytest.fixture
    def mfa_service(self, mock_db, mock_security_monitor):
        """Create MFA service instance."""
        with patch("auth.mfa_service.MFAService._get_encryption_key") as mock_key:
            # Generate a proper 32-byte base64 encoded key for Fernet
            from cryptography.fernet import Fernet

            mock_key.return_value = Fernet.generate_key()
            return MFAService(mock_db, mock_security_monitor)

    @pytest.fixture
    def sample_mfa_settings(self):
        """Sample MFA settings for testing."""
        return MFASettings(
            id=1,
            user_id="test_user",
            totp_secret="encrypted_secret",
            backup_codes="encrypted_backup_codes",
            is_enabled=True,
            enrollment_date=datetime.utcnow(),
            failed_attempts=0,
            locked_until=None,
        )

    @pytest.mark.asyncio
    async def test_enroll_user_totp_success(self, mfa_service, mock_db):
        """Test successful TOTP enrollment."""
        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = None

        # Create enrollment request
        request = MFAEnrollmentRequest(
            user_id="test_user", user_email="test@example.com", method="totp"
        )

        # Test enrollment
        result = await mfa_service.enroll_user(request)

        # Verify results
        assert result.success
        assert result.qr_code_url is not None
        assert result.backup_codes is not None
        assert len(result.backup_codes) == 10
        assert result.qr_code_url.startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_enroll_user_already_enabled(self, mfa_service, mock_db, sample_mfa_settings):
        """Test enrollment when MFA is already enabled."""
        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = sample_mfa_settings

        # Create enrollment request
        request = MFAEnrollmentRequest(
            user_id="test_user", user_email="test@example.com", method="totp"
        )

        # Test enrollment
        result = await mfa_service.enroll_user(request)

        # Verify results
        assert not result.success
        assert "already enabled" in result.message

    @pytest.mark.asyncio
    async def test_verify_totp_success(self, mfa_service, mock_db):
        """Test successful TOTP verification."""
        # Create mock MFA settings with known secret
        with patch("pyotp.random_base32") as mock_random:
            mock_random.return_value = "JBSWY3DPEHPK3PXP"

            # Create encrypted secret
            secret = "JBSWY3DPEHPK3PXP"
            encrypted_secret = mfa_service._encrypt_secret(secret)

            mfa_settings = MFASettings(
                user_id="test_user",
                totp_secret=encrypted_secret,
                is_enabled=True,
                failed_attempts=0,
                locked_until=None,
            )

            # Mock database query
            mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

            # Generate valid TOTP token
            import pyotp

            totp = pyotp.TOTP(secret)
            valid_token = totp.now()

            # Create verification request
            request = MFAVerificationRequest(user_id="test_user", token=valid_token, method="totp")

            # Test verification
            result = await mfa_service.verify_mfa(request)

            # Verify results
            assert result.success
            assert result.message == "MFA verification successful"

    @pytest.mark.asyncio
    async def test_verify_totp_invalid_token(self, mfa_service, mock_db):
        """Test TOTP verification with invalid token."""
        # Create mock MFA settings with known secret
        secret = "JBSWY3DPEHPK3PXP"
        encrypted_secret = mfa_service._encrypt_secret(secret)

        mfa_settings = MFASettings(
            user_id="test_user",
            totp_secret=encrypted_secret,
            is_enabled=True,
            failed_attempts=0,
            locked_until=None,
        )

        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Create verification request with invalid token
        request = MFAVerificationRequest(
            user_id="test_user",
            token="000000",
            method="totp",  # Invalid token
        )

        # Test verification
        result = await mfa_service.verify_mfa(request)

        # Verify results
        assert not result.success
        assert result.message == "Invalid MFA token"

    @pytest.mark.asyncio
    async def test_verify_backup_code_success(self, mfa_service, mock_db):
        """Test successful backup code verification."""
        # Create backup codes
        backup_codes = ["ABCD1234", "EFGH5678", "IJKL9012"]
        hashed_codes = mfa_service._hash_backup_codes(backup_codes)
        encrypted_codes = mfa_service._encrypt_secret(json.dumps(hashed_codes))

        mfa_settings = MFASettings(
            user_id="test_user",
            backup_codes=encrypted_codes,
            is_enabled=True,
            failed_attempts=0,
            locked_until=None,
        )

        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Create verification request with valid backup code
        request = MFAVerificationRequest(
            user_id="test_user",
            token="",
            method="backup_code",
            backup_code="ABCD1234",
        )

        # Test verification
        result = await mfa_service.verify_mfa(request)

        # Verify results
        assert result.success
        assert result.message == "MFA verification successful"

    @pytest.mark.asyncio
    async def test_verify_backup_code_single_use(self, mfa_service, mock_db):
        """Test that backup codes are single-use."""
        # Create backup codes
        backup_codes = ["ABCD1234", "EFGH5678"]
        hashed_codes = mfa_service._hash_backup_codes(backup_codes)
        encrypted_codes = mfa_service._encrypt_secret(json.dumps(hashed_codes))

        mfa_settings = MFASettings(
            user_id="test_user",
            backup_codes=encrypted_codes,
            is_enabled=True,
            failed_attempts=0,
            locked_until=None,
        )

        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Create verification request
        request = MFAVerificationRequest(
            user_id="test_user",
            token="",
            method="backup_code",
            backup_code="ABCD1234",
        )

        # First use should succeed
        result1 = await mfa_service.verify_mfa(request)
        assert result1.success

        # Second use should fail
        result2 = await mfa_service.verify_mfa(request)
        assert not result2.success

    @pytest.mark.asyncio
    async def test_rate_limiting_lockout(self, mfa_service, mock_db):
        """Test rate limiting and lockout functionality."""
        # Create MFA settings with max failed attempts
        mfa_settings = MFASettings(
            user_id="test_user",
            totp_secret="encrypted_secret",
            is_enabled=True,
            failed_attempts=5,  # At max attempts
            locked_until=None,
        )

        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Create verification request
        request = MFAVerificationRequest(
            user_id="test_user",
            token="000000",
            method="totp",  # Invalid token
        )

        # Test verification (should trigger lockout)
        result = await mfa_service.verify_mfa(request)

        # Verify user is locked out
        assert not result.success
        assert "too many failed attempts" in result.message.lower()

    @pytest.mark.asyncio
    async def test_rate_limiting_reset_after_success(self, mfa_service, mock_db):
        """Test that failed attempts reset after successful verification."""
        # Create mock MFA settings with some failed attempts
        secret = "JBSWY3DPEHPK3PXP"
        encrypted_secret = mfa_service._encrypt_secret(secret)

        mfa_settings = MFASettings(
            user_id="test_user",
            totp_secret=encrypted_secret,
            is_enabled=True,
            failed_attempts=3,  # Some failed attempts
            locked_until=None,
        )

        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Generate valid TOTP token
        import pyotp

        totp = pyotp.TOTP(secret)
        valid_token = totp.now()

        # Create verification request
        request = MFAVerificationRequest(user_id="test_user", token=valid_token, method="totp")

        # Test verification
        result = await mfa_service.verify_mfa(request)

        # Verify results and that failed attempts were reset
        assert result.success
        assert mfa_settings.failed_attempts == 0

    @pytest.mark.asyncio
    async def test_disable_mfa_success(self, mfa_service, mock_db, sample_mfa_settings):
        """Test successful MFA disable."""
        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = sample_mfa_settings

        # Test disable
        result = await mfa_service.disable_mfa("test_user")

        # Verify results
        assert result.success
        assert result.message == "MFA has been disabled"
        assert not sample_mfa_settings.is_enabled
        assert sample_mfa_settings.totp_secret is None
        assert sample_mfa_settings.backup_codes is None

    @pytest.mark.asyncio
    async def test_disable_mfa_not_configured(self, mfa_service, mock_db):
        """Test MFA disable when not configured."""
        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = None

        # Test disable
        result = await mfa_service.disable_mfa("test_user")

        # Verify results
        assert not result.success
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_get_mfa_status_enabled(self, mfa_service, mock_db):
        """Test MFA status for enabled user."""
        # Create backup codes
        backup_codes = ["ABCD1234", "EFGH5678", "IJKL9012"]
        hashed_codes = mfa_service._hash_backup_codes(backup_codes)
        encrypted_codes = mfa_service._encrypt_secret(json.dumps(hashed_codes))

        mfa_settings = MFASettings(
            user_id="test_user",
            totp_secret="encrypted_secret",
            backup_codes=encrypted_codes,
            is_enabled=True,
            failed_attempts=0,
            locked_until=None,
            last_used=datetime.utcnow(),
        )

        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Test status
        status = await mfa_service.get_mfa_status("test_user")

        # Verify results
        assert status["enabled"]
        assert "totp" in status["methods"]
        assert status["backup_codes_remaining"] == 3
        assert status["failed_attempts"] == 0
        assert status["locked_until"] is None

    @pytest.mark.asyncio
    async def test_get_mfa_status_not_configured(self, mfa_service, mock_db):
        """Test MFA status for user without MFA."""
        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = None

        # Test status
        status = await mfa_service.get_mfa_status("test_user")

        # Verify results
        assert not status["enabled"]
        assert status["methods"] == []
        assert status["backup_codes_remaining"] == 0
        assert status["locked_until"] is None

    @pytest.mark.asyncio
    async def test_regenerate_backup_codes_success(self, mfa_service, mock_db, sample_mfa_settings):
        """Test successful backup code regeneration."""
        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = sample_mfa_settings

        # Test regeneration
        result = await mfa_service.regenerate_backup_codes("test_user")

        # Verify results
        assert result.success
        assert result.backup_codes is not None
        assert len(result.backup_codes) == 10
        assert result.message == "Backup codes regenerated successfully"

    @pytest.mark.asyncio
    async def test_regenerate_backup_codes_not_enabled(self, mfa_service, mock_db):
        """Test backup code regeneration when MFA is not enabled."""
        # Create disabled MFA settings
        mfa_settings = MFASettings(user_id="test_user", is_enabled=False)

        # Mock database query
        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Test regeneration
        result = await mfa_service.regenerate_backup_codes("test_user")

        # Verify results
        assert not result.success
        assert "not enabled" in result.message

    def test_encrypt_decrypt_secret(self, mfa_service):
        """Test secret encryption and decryption."""
        secret = "test_secret_123"

        # Test encryption
        encrypted = mfa_service._encrypt_secret(secret)
        assert encrypted != secret
        assert len(encrypted) > len(secret)

        # Test decryption
        decrypted = mfa_service._decrypt_secret(encrypted)
        assert decrypted == secret

    def test_generate_backup_codes(self, mfa_service):
        """Test backup code generation."""
        codes = mfa_service._generate_backup_codes()

        # Verify code format
        assert len(codes) == 10
        for code in codes:
            assert len(code) == 8
            assert code.isalnum()
            assert code.isupper()

    def test_hash_backup_codes(self, mfa_service):
        """Test backup code hashing."""
        codes = ["ABCD1234", "EFGH5678"]
        hashed = mfa_service._hash_backup_codes(codes)

        # Verify hash format
        assert len(hashed) == 2
        for hash_val in hashed:
            assert len(hash_val) == 64  # SHA256 hash length
            assert hash_val.isalnum()

    def test_check_rate_limit_not_locked(self, mfa_service, mock_db):
        """Test rate limit check for non-locked user."""
        mfa_settings = MFASettings(user_id="test_user", failed_attempts=3, locked_until=None)

        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Test rate limit check
        result = mfa_service._check_rate_limit("test_user")
        assert result  # Should not be locked

    def test_check_rate_limit_locked(self, mfa_service, mock_db):
        """Test rate limit check for locked user."""
        mfa_settings = MFASettings(
            user_id="test_user",
            failed_attempts=5,
            locked_until=datetime.utcnow() + timedelta(minutes=10),
        )

        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Test rate limit check
        result = mfa_service._check_rate_limit("test_user")
        assert not result  # Should be locked

    def test_check_rate_limit_expired_lockout(self, mfa_service, mock_db):
        """Test rate limit check with expired lockout."""
        mfa_settings = MFASettings(
            user_id="test_user",
            failed_attempts=5,
            locked_until=datetime.utcnow() - timedelta(minutes=10),  # Expired
        )

        mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

        # Test rate limit check
        result = mfa_service._check_rate_limit("test_user")
        assert result  # Should not be locked (expired)
        assert mfa_settings.failed_attempts == 0  # Should be reset

    @pytest.mark.asyncio
    async def test_security_event_logging(self, mfa_service, mock_db):
        """Test that security events are logged correctly."""
        with patch("auth.mfa_service.security_auditor") as mock_auditor:
            # Mock MFA settings
            mfa_settings = MFASettings(
                user_id="test_user",
                totp_secret="encrypted_secret",
                is_enabled=True,
                failed_attempts=0,
                locked_until=None,
            )

            mock_db.query.return_value.filter_by.return_value.first.return_value = mfa_settings

            # Create verification request with invalid token
            request = MFAVerificationRequest(
                user_id="test_user",
                token="000000",
                method="totp",  # Invalid token
            )

            # Test verification
            await mfa_service.verify_mfa(request)

            # Verify security event was logged
            mock_auditor.log_security_event.assert_called()
            call_args = mock_auditor.log_security_event.call_args
            assert call_args[1]["event_type"] == SecurityEventType.MFA_FAILED

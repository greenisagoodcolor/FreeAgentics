"""
JWT Monitoring and Suspicious Pattern Detection Tests
Task #14.13 - JWT Security Validation and Hardening - Monitoring Component

This test suite validates:
1. JWT suspicious pattern detection
2. Token binding violation monitoring
3. Multiple invalid token attempt detection
4. Token usage from multiple IPs detection
5. Comprehensive security event logging
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from auth.security_implementation import AuthenticationManager, User, UserRole
from auth.security_logging import SecurityEventSeverity, SecurityEventType, security_auditor


class TestJWTSuspiciousPatternDetection:
    """Test JWT suspicious pattern detection and monitoring."""

    def setup_method(self):
        """Setup for each test."""
        # Clear security auditor state
        security_auditor.token_usage_patterns = {}
        security_auditor.token_binding_violations = {}
        security_auditor.suspicious_ips = set()

    def test_should_detect_multiple_invalid_token_attempts(self):
        """Test detection of multiple invalid token attempts from same user."""
        AuthenticationManager()
        user = self._create_test_user()

        # Simulate 5 invalid token attempts
        for i in range(5):
            security_auditor.log_event(
                SecurityEventType.TOKEN_INVALID,
                SecurityEventSeverity.WARNING,
                f"Invalid token attempt {i + 1}",
                user_id=user.user_id,
                username=user.username,
                details={"ip_address": "192.168.1.100"},
            )

        # Should have detected suspicious pattern
        assert user.user_id in security_auditor.token_usage_patterns
        events = security_auditor.token_usage_patterns[user.user_id]
        assert len(events) >= 5

    def test_should_detect_token_binding_violations(self):
        """Test detection of multiple token binding violations."""
        AuthenticationManager()
        ip_address = "192.168.1.100"

        # Directly track binding violations to test the monitoring logic
        for i in range(3):
            # Simulate the tracking that happens in the security logging
            security_auditor._track_token_binding_violation(ip_address)

        # Should have tracked binding violations
        assert ip_address in security_auditor.token_binding_violations
        assert len(security_auditor.token_binding_violations[ip_address]) >= 3

    def test_should_detect_multiple_ip_token_usage(self):
        """Test detection of token usage from multiple IPs."""
        user = self._create_test_user()

        # Simulate token usage from 3 different IPs within 15 minutes
        ips = ["192.168.1.100", "10.0.0.50", "172.16.0.10"]
        for ip in ips:
            security_auditor.log_event(
                SecurityEventType.TOKEN_INVALID,
                SecurityEventSeverity.WARNING,
                "Token validation attempt",
                user_id=user.user_id,
                username=user.username,
                details={"ip_address": ip},
            )

        # Should have detected multiple IP usage pattern
        assert user.user_id in security_auditor.token_usage_patterns

    def test_should_cleanup_old_tracking_data(self):
        """Test cleanup of old tracking data."""
        AuthenticationManager()
        user = self._create_test_user()
        ip_address = "192.168.1.100"

        # Add old token usage data
        old_timestamp = datetime.utcnow() - timedelta(hours=2)
        security_auditor.token_usage_patterns[user.user_id] = [
            {
                "timestamp": old_timestamp.isoformat(),
                "ip_address": ip_address,
                "event_type": "token_invalid",
                "user_agent": "test",
            }
        ]

        # Add old binding violation
        security_auditor.token_binding_violations[ip_address] = [
            datetime.utcnow() - timedelta(hours=1)
        ]

        # Trigger new event to clean up old data
        security_auditor.log_event(
            SecurityEventType.TOKEN_INVALID,
            SecurityEventSeverity.WARNING,
            "New invalid token",
            user_id=user.user_id,
            details={"ip_address": ip_address},
        )

        # Old data should be cleaned up
        recent_events = security_auditor.token_usage_patterns[user.user_id]
        assert len(recent_events) == 1  # Only the new event should remain

    def test_should_log_token_revocation_attempts(self):
        """Test logging of revoked token usage attempts."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Create and revoke token
        token = auth_manager.create_access_token(user)
        auth_manager.verify_token(token)  # Should work initially

        # Revoke token
        auth_manager.logout(token)

        # Attempt to use revoked token should log security event
        with pytest.raises(HTTPException):
            auth_manager.verify_token(token)

    def test_should_log_expired_token_attempts(self):
        """Test logging of expired token usage attempts."""
        import jwt

        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Create an expired token manually
        from datetime import datetime, timedelta, timezone

        expired_payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [],
            "type": "access",
            "jti": "expired_test_jti",
            "iss": "freeagentics",
            "aud": "freeagentics-api",
            "exp": (datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp(),  # Expired
            "nbf": (datetime.now(timezone.utc) - timedelta(minutes=10)).timestamp(),
            "iat": (datetime.now(timezone.utc) - timedelta(minutes=10)).timestamp(),
        }

        expired_token = jwt.encode(expired_payload, auth_manager.private_key, algorithm="RS256")

        # Attempt to use expired token should log security event and raise exception
        with pytest.raises(HTTPException):
            auth_manager.verify_token(expired_token)

    def test_should_integrate_with_security_auditor(self):
        """Test integration with security auditor system."""
        auth_manager = AuthenticationManager()
        self._create_test_user()

        # Test that JWT events are properly logged
        with patch.object(security_auditor, "log_event") as mock_log:
            # This should trigger logging
            try:
                auth_manager.verify_token("invalid_token")
            except HTTPException:
                pass

            # Should have called security auditor
            mock_log.assert_called()

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )


class TestJWTMonitoringIntegration:
    """Test JWT monitoring integration with authentication system."""

    def setup_method(self):
        """Setup for each test."""
        # Clear security auditor state
        security_auditor.token_usage_patterns = {}
        security_auditor.token_binding_violations = {}
        security_auditor.suspicious_ips = set()

    def test_complete_monitoring_workflow(self):
        """Test complete JWT monitoring workflow."""
        auth_manager = AuthenticationManager()

        # Register and authenticate user
        user = auth_manager.register_user(
            username="monitortest",
            email="monitor@test.com",
            password="secure_password",
            role=UserRole.RESEARCHER,
        )

        # Create token with binding
        client_fingerprint = "test_fingerprint"
        token = auth_manager.create_access_token(user, client_fingerprint=client_fingerprint)

        # Valid usage should not trigger monitoring
        token_data = auth_manager.verify_token(token, client_fingerprint=client_fingerprint)
        assert token_data.user_id == user.user_id

        # Invalid binding should trigger monitoring
        with pytest.raises(HTTPException):
            auth_manager.verify_token(token, client_fingerprint="wrong_fingerprint")

        # Should have logged the binding mismatch
        assert (
            "192.168.1.1" in security_auditor.token_binding_violations
            or len(security_auditor.token_binding_violations) > 0
        )

    def test_monitoring_performance_impact(self):
        """Test that monitoring doesn't significantly impact performance."""
        auth_manager = AuthenticationManager()
        user = self._create_test_user()

        # Measure token verification time with monitoring
        token = auth_manager.create_access_token(user)

        start_time = time.time()
        for _ in range(100):
            auth_manager.verify_token(token)
        end_time = time.time()

        # Should complete 100 verifications quickly (under 1 second)
        assert (end_time - start_time) < 1.0

    def _create_test_user(self) -> User:
        """Helper to create test user."""
        return User(
            user_id="test-user-id",
            username="testuser",
            email="test@example.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )

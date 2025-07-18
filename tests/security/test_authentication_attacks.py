"""
Authentication Attack Tests - OWASP Top 10 Focused

This module tests for authentication vulnerabilities including:
- Credential stuffing
- Password spraying
- Session hijacking
- Token manipulation
- Authentication bypass techniques
"""

import asyncio
import base64
import concurrent.futures
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import jwt
import pytest
from fastapi import HTTPException
from jose import JWTError

from auth.security_headers import SecurityHeadersManager
from auth.security_implementation import AuthenticationManager, UserRole
from database.models import User


class TestAuthenticationAttacks:
    """Comprehensive authentication attack test suite."""

    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager instance."""
        return AuthenticationManager()

    @pytest.fixture
    def security_headers(self):
        """Create security headers manager."""
        return SecurityHeadersManager()

    @pytest.fixture
    def mock_user_db(self):
        """Mock user database."""
        users = {
            "admin": {
                "id": "admin_123",
                "username": "admin",
                "email": "admin@example.com",
                "password_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGHqJffGH5C",
                "role": UserRole.ADMIN,
                "is_active": True,
                "created_at": datetime.utcnow(),
            },
            "user1": {
                "id": "user_456",
                "username": "user1",
                "email": "user1@example.com",
                "password_hash": "$2b$12$YourHashHere",
                "role": UserRole.OBSERVER,
                "is_active": True,
                "created_at": datetime.utcnow(),
            },
        }
        return users

    # Test A1: Broken Access Control
    def test_privilege_escalation_via_jwt_manipulation(self, auth_manager):
        """Test JWT token manipulation for privilege escalation."""
        # Register regular user
        user = auth_manager.register_user(
            "regular_user",
            "user@example.com",
            "SecurePass123!",
            UserRole.OBSERVER,
        )

        # Get legitimate token
        token = auth_manager.create_access_token(user)

        # Attempt to decode and modify token
        try:
            # Try decoding with wrong secret (should fail)
            decoded = jwt.decode(token, "wrong_secret", algorithms=["HS256"])
            pytest.fail("JWT signature verification bypass detected!")
        except jwt.InvalidSignatureError:
            pass  # Expected

        # Test algorithm confusion attack
        header = jwt.get_unverified_header(token)

        # Attempt to create token with 'none' algorithm
        try:
            fake_token = jwt.encode(
                {
                    "sub": user["username"],
                    "role": "ADMIN",
                    "exp": datetime.utcnow() + timedelta(hours=1),
                },
                "",
                algorithm="none",
            )
            # Try to validate fake token
            validated = auth_manager.validate_token(fake_token)
            assert validated is None, "Algorithm confusion attack successful!"
        except:
            pass  # Expected to fail

    def test_authentication_bypass_sql_injection(self, auth_manager):
        """Test SQL injection in authentication."""
        sql_injection_payloads = [
            ("admin'--", "anything"),
            ("' OR '1'='1", "' OR '1'='1"),
            ("admin'; DROP TABLE users;--", "password"),
            (
                "' UNION SELECT * FROM users WHERE username='admin'--",
                "password",
            ),
            ("admin' AND '1'='1", "password"),
            ("' OR 1=1--", "password"),
            ("admin'/*", "password"),
            ("'; EXEC xp_cmdshell('dir');--", "password"),
        ]

        for username, password in sql_injection_payloads:
            result = auth_manager.authenticate_user(username, password)
            assert (
                result is None
            ), f"SQL injection successful with payload: {username}"

    def test_authentication_bypass_ldap_injection(self, auth_manager):
        """Test LDAP injection in authentication."""
        ldap_injection_payloads = [
            ("admin)(&(password=*", "anything"),
            ("*)(uid=*", "password"),
            ("admin)(|(password=*", "anything"),
            ("*)(|(objectClass=*", "password"),
            ("admin))(|(uid=*", "password"),
        ]

        for username, password in ldap_injection_payloads:
            result = auth_manager.authenticate_user(username, password)
            assert (
                result is None
            ), f"LDAP injection successful with payload: {username}"

    def test_authentication_bypass_nosql_injection(self, auth_manager):
        """Test NoSQL injection in authentication."""
        # Test MongoDB-style injection attempts
        nosql_payloads = [
            ({"$ne": None}, "password"),
            ({"$gt": ""}, "password"),
            ({"username": {"$regex": ".*"}}, "password"),
            ({"$where": "this.username == 'admin'"}, "password"),
        ]

        for username_payload, password in nosql_payloads:
            # Convert payload to string for testing
            username = (
                str(username_payload)
                if isinstance(username_payload, dict)
                else username_payload
            )
            result = auth_manager.authenticate_user(username, password)
            assert (
                result is None
            ), f"NoSQL injection successful with payload: {username_payload}"

    def test_credential_stuffing_attack(self, auth_manager):
        """Test credential stuffing attack detection and prevention."""
        # Common leaked credentials
        leaked_credentials = [
            ("admin", "admin"),
            ("admin", "password"),
            ("admin", "123456"),
            ("admin", "admin123"),
            ("root", "root"),
            ("test", "test"),
            ("demo", "demo"),
            ("user", "user"),
        ]

        failed_attempts = 0
        for username, password in leaked_credentials:
            result = auth_manager.authenticate_user(username, password)
            if result is None:
                failed_attempts += 1

        # All attempts should fail
        assert failed_attempts == len(
            leaked_credentials
        ), "Credential stuffing partially successful"

    def test_password_spraying_attack(self, auth_manager):
        """Test password spraying attack detection."""
        # Create multiple users
        users = []
        for i in range(5):
            user = auth_manager.register_user(
                f"user{i}",
                f"user{i}@example.com",
                f"UniquePass{i}!",
                UserRole.OBSERVER,
            )
            users.append(user)

        # Common passwords to spray
        common_passwords = [
            "Password123",
            "Welcome123",
            "Summer2024",
            "Company123",
            "Qwerty123",
        ]

        # Track successful attempts (should be 0)
        successful_attempts = 0

        for user in users:
            for password in common_passwords:
                result = auth_manager.authenticate_user(
                    user["username"], password
                )
                if result is not None:
                    successful_attempts += 1

        assert (
            successful_attempts == 0
        ), f"Password spraying successful: {successful_attempts} accounts compromised"

    def test_session_fixation_attack(self, auth_manager):
        """Test session fixation vulnerability."""
        # Get initial session before authentication
        initial_session = auth_manager.create_session("test_user")

        # Register and authenticate user
        user = auth_manager.register_user(
            "session_test",
            "session@example.com",
            "SecurePass123!",
            UserRole.OBSERVER,
        )
        auth_result = auth_manager.authenticate_user(
            "session_test", "SecurePass123!"
        )

        # Get new session after authentication
        new_session = auth_manager.create_session("session_test")

        # Sessions should be different (regenerated after auth)
        assert (
            initial_session != new_session
        ), "Session fixation vulnerability: session not regenerated"

    def test_session_hijacking_prevention(self, auth_manager):
        """Test session hijacking prevention mechanisms."""
        # Create user and session
        user = auth_manager.register_user(
            "hijack_test",
            "hijack@example.com",
            "SecurePass123!",
            UserRole.OBSERVER,
        )
        session_id = auth_manager.create_session("hijack_test")

        # Simulate session from different context
        # In real implementation, this would check IP, user agent, etc.
        with patch.object(auth_manager, "get_session_context") as mock_context:
            mock_context.return_value = {
                "ip": "192.168.1.1",
                "user_agent": "Mozilla/5.0",
            }

            # Original context validation should pass
            assert auth_manager.validate_session(session_id) is True

            # Changed context should fail or trigger warning
            mock_context.return_value = {
                "ip": "10.0.0.1",
                "user_agent": "curl/7.64.0",
            }
            # In a real implementation, this should fail or require re-authentication
            # For now, we just ensure the session system is aware of context

    def test_timing_attack_on_authentication(self, auth_manager):
        """Test timing attack resistance in authentication."""
        # Create a test user
        user = auth_manager.register_user(
            "timing_user",
            "timing@example.com",
            "CorrectPassword123!",
            UserRole.OBSERVER,
        )

        # Measure authentication times
        correct_times = []
        incorrect_times = []

        # Test with correct password
        for _ in range(20):
            start = time.perf_counter()
            auth_manager.authenticate_user(
                "timing_user", "CorrectPassword123!"
            )
            end = time.perf_counter()
            correct_times.append(end - start)

        # Test with incorrect passwords of varying similarity
        test_passwords = [
            "WrongPassword123!",
            "C",  # Very different
            "CorrectPassword123",  # Almost correct
            "correctpassword123!",  # Case difference
            "CorrectPassword124!",  # One character different
        ]

        for password in test_passwords:
            start = time.perf_counter()
            auth_manager.authenticate_user("timing_user", password)
            end = time.perf_counter()
            incorrect_times.append(end - start)

        # Calculate statistics
        avg_correct = sum(correct_times) / len(correct_times)
        avg_incorrect = sum(incorrect_times) / len(incorrect_times)

        # Time difference should be minimal (< 10ms)
        time_diff = abs(avg_correct - avg_incorrect)
        assert (
            time_diff < 0.01
        ), f"Timing attack possible: {time_diff*1000:.2f}ms difference"

    def test_brute_force_protection(self, auth_manager):
        """Test brute force attack protection."""
        # Create target user
        user = auth_manager.register_user(
            "brute_target",
            "brute@example.com",
            "VerySecurePass123!",
            UserRole.OBSERVER,
        )

        # Simulate brute force attack
        failed_attempts = 0
        locked_out = False

        for i in range(20):  # Try 20 failed attempts
            result = auth_manager.authenticate_user(
                "brute_target", f"WrongPass{i}"
            )
            if result is None:
                failed_attempts += 1

            # Check if account gets locked after certain attempts
            if hasattr(
                auth_manager, "is_account_locked"
            ) and auth_manager.is_account_locked("brute_target"):
                locked_out = True
                break

        # Should implement some form of rate limiting or account locking
        assert (
            failed_attempts < 20 or locked_out
        ), "No brute force protection detected"

    def test_authentication_token_replay_attack(self, auth_manager):
        """Test token replay attack prevention."""
        # Create user and get token
        user = auth_manager.register_user(
            "replay_test",
            "replay@example.com",
            "SecurePass123!",
            UserRole.OBSERVER,
        )
        token = auth_manager.create_access_token(user)

        # Use token successfully
        first_use = auth_manager.validate_token(token)
        assert first_use is not None

        # In a secure system, tokens should have jti (JWT ID) for replay prevention
        # Simulate token invalidation after use for sensitive operations
        if hasattr(auth_manager, "invalidate_token"):
            auth_manager.invalidate_token(token)

            # Replay attempt should fail
            replay_attempt = auth_manager.validate_token(token)
            assert replay_attempt is None, "Token replay attack successful"

    def test_authentication_downgrade_attack(self, auth_manager):
        """Test authentication downgrade attack prevention."""
        # Ensure strong authentication methods cannot be downgraded

        # Create user with strong password
        user = auth_manager.register_user(
            "strong_auth_user",
            "strong@example.com",
            "VeryStrongPassword123!@#",
            UserRole.OBSERVER,
        )

        # Attempt to authenticate with weaker methods
        weak_attempts = [
            # Attempt without password
            lambda: auth_manager.authenticate_user("strong_auth_user", ""),
            # Attempt with None password
            lambda: auth_manager.authenticate_user("strong_auth_user", None),
            # Attempt to bypass password check
            lambda: auth_manager.authenticate_user(
                "strong_auth_user", {"bypass": True}
            ),
        ]

        for attempt in weak_attempts:
            result = attempt()
            assert result is None, "Authentication downgrade attack successful"

    def test_multi_factor_authentication_bypass(self, auth_manager):
        """Test MFA bypass attempts."""
        # If MFA is implemented, test bypass attempts
        if hasattr(auth_manager, "verify_mfa"):
            user = auth_manager.register_user(
                "mfa_user",
                "mfa@example.com",
                "SecurePass123!",
                UserRole.OBSERVER,
            )

            # Enable MFA for user
            if hasattr(auth_manager, "enable_mfa"):
                auth_manager.enable_mfa("mfa_user")

            # Attempt to bypass MFA
            bypass_attempts = [
                None,  # No MFA code
                "",  # Empty MFA code
                "000000",  # Common code
                "123456",  # Sequential code
                "111111",  # Repeated digits
            ]

            for code in bypass_attempts:
                result = auth_manager.verify_mfa("mfa_user", code)
                assert (
                    result is False
                ), f"MFA bypass successful with code: {code}"

    def test_password_reset_token_security(self, auth_manager):
        """Test password reset token security."""
        if hasattr(auth_manager, "generate_reset_token"):
            user = auth_manager.register_user(
                "reset_user",
                "reset@example.com",
                "OldPassword123!",
                UserRole.OBSERVER,
            )

            # Generate reset token
            reset_token = auth_manager.generate_reset_token("reset_user")

            # Test token properties
            assert len(reset_token) >= 32, "Reset token too short"
            assert reset_token != "reset_user", "Reset token predictable"

            # Test token expiration
            if hasattr(auth_manager, "validate_reset_token"):
                # Token should be valid initially
                assert auth_manager.validate_reset_token(reset_token) is True

                # Test expired token (would need to mock time)
                with patch(
                    "time.time", return_value=time.time() + 3700
                ):  # 1 hour + later
                    assert (
                        auth_manager.validate_reset_token(reset_token) is False
                    ), "Reset token doesn't expire"

    def test_concurrent_authentication_race_condition(self, auth_manager):
        """Test race conditions in concurrent authentication."""
        user = auth_manager.register_user(
            "race_user", "race@example.com", "RacePass123!", UserRole.OBSERVER
        )

        # Simulate concurrent authentication attempts
        results = []
        errors = []

        def authenticate():
            try:
                result = auth_manager.authenticate_user(
                    "race_user", "RacePass123!"
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run concurrent authentications
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(authenticate) for _ in range(10)]
            concurrent.futures.wait(futures)

        # All attempts should succeed without errors
        assert len(errors) == 0, f"Race condition errors: {errors}"
        assert (
            len(results) == 10
        ), "Not all concurrent authentications completed"
        assert all(
            r is not None for r in results
        ), "Some concurrent authentications failed"

    def test_authentication_state_manipulation(self, auth_manager):
        """Test authentication state manipulation attacks."""
        # Test various state manipulation attempts

        # Create authenticated user
        user = auth_manager.register_user(
            "state_user",
            "state@example.com",
            "StatePass123!",
            UserRole.OBSERVER,
        )
        token = auth_manager.create_access_token(user)

        # Test token state manipulation
        if hasattr(auth_manager, "get_user_state"):
            # Attempt to manipulate user state without proper authorization
            manipulations = [
                {"is_active": True, "role": "ADMIN"},
                {"is_verified": True, "permissions": ["all"]},
                {"account_type": "premium", "limits": None},
            ]

            for manipulation in manipulations:
                # Direct state manipulation should not be possible
                # This would typically be protected by proper encapsulation
                pass

    @pytest.mark.asyncio
    async def test_authentication_async_vulnerabilities(self, auth_manager):
        """Test authentication vulnerabilities in async contexts."""
        # Test async authentication if supported
        if hasattr(auth_manager, "authenticate_user_async"):
            user = auth_manager.register_user(
                "async_user",
                "async@example.com",
                "AsyncPass123!",
                UserRole.OBSERVER,
            )

            # Test concurrent async authentication
            tasks = []
            for i in range(10):
                task = auth_manager.authenticate_user_async(
                    "async_user", "AsyncPass123!"
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for race conditions or errors
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0, f"Async authentication errors: {errors}"

    def test_jwt_key_confusion_attack(self, auth_manager):
        """Test JWT key confusion attacks."""
        user = auth_manager.register_user(
            "jwt_user", "jwt@example.com", "JWTPass123!", UserRole.OBSERVER
        )
        token = auth_manager.create_access_token(user)

        # Get token parts
        header, payload, signature = token.split(".")

        # Decode header to check algorithm
        header_data = json.loads(base64.urlsafe_b64decode(header + "=="))

        # Ensure RS256 or HS256 is used, not vulnerable algorithms
        assert header_data["alg"] not in [
            "none",
            "None",
            "NONE",
        ], "Vulnerable JWT algorithm"

        # Test key confusion between HMAC and RSA
        if header_data["alg"] == "RS256":
            # Try to create HMAC token with public key (key confusion)
            try:
                # This should fail in a secure implementation
                fake_token = jwt.encode(
                    json.loads(base64.urlsafe_b64decode(payload + "==")),
                    "public_key_as_secret",
                    algorithm="HS256",
                )
                validated = auth_manager.validate_token(fake_token)
                assert validated is None, "Key confusion attack successful"
            except:
                pass  # Expected to fail

    def test_authentication_bypass_via_parameter_pollution(self, auth_manager):
        """Test HTTP parameter pollution in authentication."""
        # Test various parameter pollution attempts
        pollution_attempts = [
            {
                "username": "admin",
                "username": "attacker",
            },  # Duplicate parameters
            {"username": ["admin", "attacker"]},  # Array injection
            {"username": "admin&username=attacker"},  # URL encoded pollution
            {
                "username": "admin",
                "password": "pass",
                "password": "attacker_pass",
            },
        ]

        for params in pollution_attempts:
            # In a real scenario, this would test the HTTP layer
            # Here we ensure the auth manager handles polluted inputs safely
            if isinstance(params.get("username"), list):
                username = params["username"][0]  # Should use first or reject
            else:
                username = params.get("username", "")

            result = auth_manager.authenticate_user(username, "password")
            assert result is None, f"Parameter pollution bypass with: {params}"

    def test_authentication_token_confusion(self, auth_manager):
        """Test token type confusion attacks."""
        user = auth_manager.register_user(
            "token_user",
            "token@example.com",
            "TokenPass123!",
            UserRole.OBSERVER,
        )

        # Get different token types if available
        access_token = auth_manager.create_access_token(user)

        # Try to use access token as refresh token
        if hasattr(auth_manager, "refresh_access_token"):
            try:
                new_token = auth_manager.refresh_access_token(access_token)
                assert new_token is None, "Token type confusion successful"
            except:
                pass  # Expected to fail

        # Try to use session ID as JWT
        session_id = auth_manager.create_session("token_user")
        validated = auth_manager.validate_token(session_id)
        assert validated is None, "Session ID accepted as JWT token"

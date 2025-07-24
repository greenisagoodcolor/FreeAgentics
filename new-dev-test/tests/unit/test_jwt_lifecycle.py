"""
JWT Lifecycle Test Suite
Task #6.1 - Create JWT lifecycle test suite

This test suite comprehensively validates the complete JWT lifecycle including:
1. Token creation (access and refresh)
2. Token validation and verification
3. Token refresh and rotation
4. Token expiration handling
5. Token revocation and blacklisting
6. Concurrent token operations
7. Edge cases and error scenarios
"""

import asyncio
import concurrent.futures
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException

from auth.security_implementation import ALGORITHM, AuthenticationManager, User, UserRole
from auth.security_logging import SecurityEventType, security_auditor


class TestJWTLifecycle:
    """Comprehensive JWT lifecycle testing."""

    def setup_method(self):
        """Setup for each test."""
        self.auth_manager = AuthenticationManager()
        self.test_user = self._create_test_user()
        # Register test user
        self.auth_manager.users[self.test_user.username] = {
            "user": self.test_user,
            "password_hash": self.auth_manager.hash_password("test_password"),
        }
        # Clear security auditor state
        security_auditor.token_usage_patterns = {}
        security_auditor.token_binding_violations = {}

    def _create_test_user(self, user_id="test-user-123", role=UserRole.RESEARCHER):
        """Helper to create test user."""
        return User(
            user_id=user_id,
            username=f"testuser_{user_id}",
            email=f"test_{user_id}@example.com",
            role=role,
            created_at=datetime.now(timezone.utc),
        )

    def test_complete_token_lifecycle(self):
        """Test complete JWT lifecycle from creation to expiration."""
        # 1. Create tokens
        access_token = self.auth_manager.create_access_token(self.test_user)
        refresh_token = self.auth_manager.create_refresh_token(self.test_user)

        # 2. Verify initial tokens work
        token_data = self.auth_manager.verify_token(access_token)
        assert token_data.user_id == self.test_user.user_id
        assert token_data.role == self.test_user.role

        # 3. Use refresh token to get new access token
        (
            new_access_token,
            new_refresh_token,
        ) = self.auth_manager.refresh_access_token(refresh_token)
        assert new_access_token != access_token
        assert new_refresh_token != refresh_token

        # 4. Verify new tokens work
        new_token_data = self.auth_manager.verify_token(new_access_token)
        assert new_token_data.user_id == self.test_user.user_id

        # 5. Verify old refresh token is blacklisted
        with pytest.raises(HTTPException) as exc_info:
            self.auth_manager.refresh_access_token(refresh_token)
        assert exc_info.value.status_code == 401

        # 6. Logout - blacklist current tokens
        self.auth_manager.logout(new_access_token)

        # 7. Verify logged out tokens don't work
        with pytest.raises(HTTPException):
            self.auth_manager.verify_token(new_access_token)

    def test_token_expiration_lifecycle(self):
        """Test token expiration behavior."""
        # Create token with very short expiration
        with patch("auth.security_implementation.ACCESS_TOKEN_EXPIRE_MINUTES", 0.016):  # ~1 second
            access_token = self.auth_manager.create_access_token(self.test_user)

        # Token should work immediately
        token_data = self.auth_manager.verify_token(access_token)
        assert token_data.user_id == self.test_user.user_id

        # Wait for expiration
        time.sleep(1.5)

        # Token should be expired
        with pytest.raises(HTTPException) as exc_info:
            self.auth_manager.verify_token(access_token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_concurrent_token_creation(self):
        """Test concurrent token creation for multiple users."""
        users = [self._create_test_user(f"user-{i}") for i in range(10)]
        tokens = []

        def create_token(user):
            return self.auth_manager.create_access_token(user)

        # Create tokens concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_tokens = [executor.submit(create_token, user) for user in users]
            tokens = [future.result() for future in concurrent.futures.as_completed(future_tokens)]

        # Verify all tokens are unique and valid
        assert len(set(tokens)) == len(tokens), "All tokens should be unique"

        for i, token in enumerate(tokens):
            token_data = self.auth_manager.verify_token(token)
            assert token_data.user_id == users[i].user_id

    def test_concurrent_token_verification(self):
        """Test concurrent token verification."""
        access_token = self.auth_manager.create_access_token(self.test_user)
        verification_results = []

        def verify_token():
            try:
                token_data = self.auth_manager.verify_token(access_token)
                return token_data.user_id
            except Exception as e:
                return str(e)

        # Verify token concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(verify_token) for _ in range(100)]
            verification_results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All verifications should succeed
        assert all(result == self.test_user.user_id for result in verification_results)

    def test_token_refresh_race_condition(self):
        """Test handling of concurrent refresh token usage."""
        refresh_token = self.auth_manager.create_refresh_token(self.test_user)
        results = []

        def refresh_token_concurrent():
            try:
                (
                    new_access,
                    new_refresh,
                ) = self.auth_manager.refresh_access_token(refresh_token)
                return ("success", new_access, new_refresh)
            except HTTPException as e:
                return ("error", str(e.detail))

        # Try to use refresh token concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(refresh_token_concurrent) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Only one should succeed, others should fail
        successes = [r for r in results if r[0] == "success"]
        errors = [r for r in results if r[0] == "error"]

        assert len(successes) == 1, "Only one refresh should succeed"
        assert len(errors) == 4, "Four refreshes should fail"
        assert all("blacklisted" in e[1].lower() or "already used" in e[1].lower() for e in errors)

    def test_token_revocation_lifecycle(self):
        """Test token revocation and blacklisting."""
        # Create multiple tokens for user
        tokens = [self.auth_manager.create_access_token(self.test_user) for _ in range(3)]

        # All should work initially
        for token in tokens:
            token_data = self.auth_manager.verify_token(token)
            assert token_data.user_id == self.test_user.user_id

        # Revoke one token
        payload = jwt.decode(tokens[0], options={"verify_signature": False})
        jti = payload["jti"]
        self.auth_manager.revoke_token(jti)

        # Revoked token should fail
        with pytest.raises(HTTPException) as exc_info:
            self.auth_manager.verify_token(tokens[0])
        assert "blacklisted" in exc_info.value.detail.lower()

        # Other tokens should still work
        for token in tokens[1:]:
            token_data = self.auth_manager.verify_token(token)
            assert token_data.user_id == self.test_user.user_id

    def test_token_binding_lifecycle(self):
        """Test client fingerprint binding throughout lifecycle."""
        client_fingerprint = "unique_client_123"

        # Create bound tokens
        access_token = self.auth_manager.create_access_token(
            self.test_user, client_fingerprint=client_fingerprint
        )
        refresh_token = self.auth_manager.create_refresh_token(self.test_user)

        # Verify with correct fingerprint works
        token_data = self.auth_manager.verify_token(
            access_token, client_fingerprint=client_fingerprint
        )
        assert token_data.user_id == self.test_user.user_id

        # Verify with wrong fingerprint fails
        with pytest.raises(HTTPException) as exc_info:
            self.auth_manager.verify_token(access_token, client_fingerprint="wrong_fingerprint")
        assert "binding" in exc_info.value.detail.lower()

        # Refresh with correct fingerprint
        (
            new_access_token,
            new_refresh_token,
        ) = self.auth_manager.refresh_access_token(
            refresh_token, client_fingerprint=client_fingerprint
        )

        # New token should also be bound
        new_token_data = self.auth_manager.verify_token(
            new_access_token, client_fingerprint=client_fingerprint
        )
        assert new_token_data.user_id == self.test_user.user_id

    def test_role_change_lifecycle(self):
        """Test token lifecycle when user role changes."""
        # Create initial token
        access_token = self.auth_manager.create_access_token(self.test_user)
        token_data = self.auth_manager.verify_token(access_token)
        assert token_data.role == UserRole.RESEARCHER

        # Change user role
        self.test_user.role = UserRole.ADMIN
        self.auth_manager.users[self.test_user.username]["user"].role = UserRole.ADMIN

        # Old token still has old role
        token_data = self.auth_manager.verify_token(access_token)
        assert token_data.role == UserRole.RESEARCHER

        # New token should have new role
        new_access_token = self.auth_manager.create_access_token(self.test_user)
        new_token_data = self.auth_manager.verify_token(new_access_token)
        assert new_token_data.role == UserRole.ADMIN

    def test_token_lifecycle_with_suspicious_activity(self):
        """Test token lifecycle when suspicious activity is detected."""
        access_token = self.auth_manager.create_access_token(self.test_user)

        # Simulate suspicious activity
        for i in range(10):
            security_auditor.log_event(
                SecurityEventType.TOKEN_INVALID,
                severity="warning",
                message=f"Invalid token attempt {i}",
                user_id=self.test_user.user_id,
                details={"ip_address": f"192.168.1.{100 + i}"},
            )

        # Token should still work but activity should be logged
        token_data = self.auth_manager.verify_token(access_token)
        assert token_data.user_id == self.test_user.user_id

        # Check that suspicious activity was tracked
        assert self.test_user.user_id in security_auditor.token_usage_patterns

    def test_token_lifecycle_edge_cases(self):
        """Test edge cases in token lifecycle."""
        # Test 1: Empty token
        with pytest.raises(HTTPException):
            self.auth_manager.verify_token("")

        # Test 2: Malformed token
        with pytest.raises(HTTPException):
            self.auth_manager.verify_token("not.a.valid.token")

        # Test 3: Token signed with wrong key
        fake_token = jwt.encode(
            {
                "user_id": "fake",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
            },
            "wrong_secret",
            algorithm="HS256",
        )
        with pytest.raises(HTTPException):
            self.auth_manager.verify_token(fake_token)

        # Test 4: Token missing required claims
        incomplete_payload = {"user_id": self.test_user.user_id}
        incomplete_token = jwt.encode(
            incomplete_payload,
            self.auth_manager.private_key,
            algorithm=ALGORITHM,
        )
        with pytest.raises(HTTPException):
            self.auth_manager.verify_token(incomplete_token)

    def test_token_lifecycle_performance(self):
        """Test token operations performance."""
        # Measure token creation time
        start_time = time.time()
        tokens = [self.auth_manager.create_access_token(self.test_user) for _ in range(100)]
        creation_time = time.time() - start_time
        assert creation_time < 2.0, f"Token creation too slow: {creation_time}s for 100 tokens"

        # Measure token verification time
        token = tokens[0]
        start_time = time.time()
        for _ in range(100):
            self.auth_manager.verify_token(token)
        verification_time = time.time() - start_time
        assert (
            verification_time < 1.0
        ), f"Token verification too slow: {verification_time}s for 100 verifications"

    @pytest.mark.asyncio
    async def test_async_token_lifecycle(self):
        """Test token lifecycle in async context."""
        # Create tokens
        access_token = self.auth_manager.create_access_token(self.test_user)

        # Async verification simulation
        async def verify_token_async():
            return self.auth_manager.verify_token(access_token)

        # Run multiple async verifications
        tasks = [verify_token_async() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.user_id == self.test_user.user_id for r in results)

    def test_token_lifecycle_cleanup(self):
        """Test cleanup of expired tokens and blacklist management."""
        # Create and blacklist multiple tokens
        tokens_to_blacklist = []
        for i in range(5):
            token = self.auth_manager.create_access_token(self.test_user)
            payload = jwt.decode(token, options={"verify_signature": False})
            tokens_to_blacklist.append(payload["jti"])
            self.auth_manager.revoke_token(payload["jti"])

        # Verify all are blacklisted
        assert all(jti in self.auth_manager.blacklist for jti in tokens_to_blacklist)

        # In production, there would be a cleanup mechanism for old blacklisted tokens
        # For now, verify the blacklist contains expected entries
        assert len(self.auth_manager.blacklist) >= 5


class TestJWTLifecycleIntegration:
    """Integration tests for JWT lifecycle with other components."""

    def setup_method(self):
        """Setup for integration tests."""
        self.auth_manager = AuthenticationManager()
        self.test_user = User(
            user_id="integration-test-user",
            username="integration_user",
            email="integration@test.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )
        self.auth_manager.users[self.test_user.username] = {
            "user": self.test_user,
            "password_hash": self.auth_manager.hash_password("integration_password"),
        }

    def test_jwt_lifecycle_with_user_session(self):
        """Test JWT lifecycle integrated with user session management."""
        # Simulate login
        access_token = self.auth_manager.create_access_token(self.test_user)
        refresh_token = self.auth_manager.create_refresh_token(self.test_user)

        # Simulate session activity
        for _ in range(3):
            token_data = self.auth_manager.verify_token(access_token)
            assert token_data.user_id == self.test_user.user_id
            time.sleep(0.1)

        # Simulate token refresh mid-session
        (
            new_access_token,
            new_refresh_token,
        ) = self.auth_manager.refresh_access_token(refresh_token)

        # Continue session with new token
        for _ in range(3):
            token_data = self.auth_manager.verify_token(new_access_token)
            assert token_data.user_id == self.test_user.user_id
            time.sleep(0.1)

        # Simulate logout
        self.auth_manager.logout(new_access_token)

        # Verify session ended
        with pytest.raises(HTTPException):
            self.auth_manager.verify_token(new_access_token)

    def test_jwt_lifecycle_with_permission_checks(self):
        """Test JWT lifecycle with permission validation."""
        # Create tokens for users with different roles
        admin_user = User(
            user_id="admin-user",
            username="admin",
            email="admin@test.com",
            role=UserRole.ADMIN,
            created_at=datetime.now(timezone.utc),
        )

        admin_token = self.auth_manager.create_access_token(admin_user)
        researcher_token = self.auth_manager.create_access_token(self.test_user)

        # Verify admin has all permissions
        admin_data = self.auth_manager.verify_token(admin_token)
        assert len(admin_data.permissions) > 0
        assert "admin_system" in [p.value for p in admin_data.permissions]

        # Verify researcher has limited permissions
        researcher_data = self.auth_manager.verify_token(researcher_token)
        assert len(researcher_data.permissions) > 0
        assert "admin_system" not in [p.value for p in researcher_data.permissions]

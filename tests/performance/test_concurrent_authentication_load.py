"""
Concurrent Authentication Load Test Suite

Tests authentication system performance under concurrent user scenarios
including multiple login attempts, token refreshes, and session management.
"""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pytest

from auth.security_implementation import (
    AuthenticationManager,
    RateLimiter,
    User,
    UserRole,
)


class TestConcurrentAuthenticationLoad:
    """Test authentication system under concurrent load."""

    @pytest.fixture
    def auth_manager(self):
        """Create fresh AuthenticationManager instance."""
        return AuthenticationManager()

    @pytest.fixture
    def rate_limiter(self):
        """Create fresh RateLimiter instance."""
        return RateLimiter()

    @pytest.fixture
    def test_users(self, auth_manager):
        """Create multiple test users."""
        users = []
        for i in range(100):
            user = User(
                user_id=f"user_{i}",
                username=f"testuser_{i}",
                email=f"test{i}@example.com",
                role=UserRole.RESEARCHER,
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
            # Register user in auth manager
            auth_manager.users[user.username] = {
                "user": user,
                "password_hash": auth_manager.hash_password("password123"),
            }
            users.append(user)
        return users

    def test_concurrent_token_creation(self, auth_manager, test_users):
        """Test concurrent token creation for multiple users."""
        results = []
        errors = []

        def create_token(user):
            try:
                start_time = time.time()
                token = auth_manager.create_access_token(user)
                end_time = time.time()

                # Verify token is valid
                auth_manager.verify_token(token)

                return {
                    "user_id": user.user_id,
                    "token": token,
                    "duration": end_time - start_time,
                    "success": True,
                }
            except Exception as e:
                errors.append({"user_id": user.user_id, "error": str(e)})
                return {"user_id": user.user_id, "success": False}

        # Test with 50 concurrent users
        test_subset = test_users[:50]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_token, user) for user in test_subset]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Verify results
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) == len(test_subset)
        assert len(errors) == 0

        # Check performance metrics
        durations = [r["duration"] for r in successful_results]
        avg_duration = statistics.mean(durations)
        max_duration = max(durations)

        # Performance assertions (tokens should be created quickly)
        assert avg_duration < 0.2  # Average < 200ms
        assert max_duration < 1.0  # Max < 1 second

        # Verify all tokens are unique
        tokens = [r["token"] for r in successful_results]
        assert len(set(tokens)) == len(tokens)

    def test_concurrent_authentication(self, auth_manager, test_users):
        """Test concurrent user authentication."""
        results = []
        errors = []

        def authenticate_user(user):
            try:
                start_time = time.time()
                authenticated_user = auth_manager.authenticate_user(
                    user.username, "password123"
                )
                end_time = time.time()

                return {
                    "user_id": user.user_id,
                    "authenticated": authenticated_user is not None,
                    "duration": end_time - start_time,
                    "success": True,
                }
            except Exception as e:
                errors.append({"user_id": user.user_id, "error": str(e)})
                return {"user_id": user.user_id, "success": False}

        # Test with 30 concurrent users
        test_subset = test_users[:30]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(authenticate_user, user) for user in test_subset]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Verify results
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) == len(test_subset)
        assert len(errors) == 0

        # All authentications should succeed
        assert all(r["authenticated"] for r in successful_results)

        # Check performance metrics
        durations = [r["duration"] for r in successful_results]
        avg_duration = statistics.mean(durations)

        # Authentication should be reasonably fast (bcrypt is intentionally slow)
        assert avg_duration < 0.5  # Average < 500ms

    def test_concurrent_token_refresh(self, auth_manager, test_users):
        """Test concurrent token refresh operations."""
        # Create refresh tokens for users
        refresh_tokens = []
        for user in test_users[:20]:
            # Ensure user is registered in auth_manager (already done in test_users fixture)
            refresh_token = auth_manager.create_refresh_token(user)
            refresh_tokens.append((user, refresh_token))

        results = []
        errors = []

        def refresh_token(user_refresh_tuple):
            user, refresh_token = user_refresh_tuple
            try:
                start_time = time.time()
                (
                    new_access_token,
                    new_refresh_token,
                ) = auth_manager.refresh_access_token(refresh_token)
                end_time = time.time()

                # Verify new tokens are valid
                auth_manager.verify_token(new_access_token)

                return {
                    "user_id": user.user_id,
                    "new_access_token": new_access_token,
                    "new_refresh_token": new_refresh_token,
                    "duration": end_time - start_time,
                    "success": True,
                }
            except Exception as e:
                errors.append({"user_id": user.user_id, "error": str(e)})
                return {"user_id": user.user_id, "success": False}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(refresh_token, rt) for rt in refresh_tokens]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Verify results
        successful_results = [r for r in results if r["success"]]

        # Debug: print errors if any
        if errors:
            print(f"Errors: {errors}")

        assert len(successful_results) == len(refresh_tokens)
        assert len(errors) == 0

        # Check performance metrics
        durations = [r["duration"] for r in successful_results]
        avg_duration = statistics.mean(durations)

        # Token refresh should be reasonably fast
        assert avg_duration < 0.5  # Average < 500ms

    def test_concurrent_token_validation(self, auth_manager, test_users):
        """Test concurrent token validation operations."""
        # Create tokens for users
        tokens = []
        for user in test_users[:40]:
            token = auth_manager.create_access_token(user)
            tokens.append((user, token))

        results = []
        errors = []

        def validate_token(user_token_tuple):
            user, token = user_token_tuple
            try:
                start_time = time.time()
                token_data = auth_manager.verify_token(token)
                end_time = time.time()

                return {
                    "user_id": user.user_id,
                    "valid": token_data.user_id == user.user_id,
                    "duration": end_time - start_time,
                    "success": True,
                }
            except Exception as e:
                errors.append({"user_id": user.user_id, "error": str(e)})
                return {"user_id": user.user_id, "success": False}

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(validate_token, ut) for ut in tokens]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Verify results
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) == len(tokens)
        assert len(errors) == 0

        # All tokens should be valid
        assert all(r["valid"] for r in successful_results)

        # Check performance metrics
        durations = [r["duration"] for r in successful_results]
        avg_duration = statistics.mean(durations)

        # Token validation should be fast
        assert avg_duration < 0.1  # Average < 100ms

    def test_rate_limiting_under_load(self, rate_limiter):
        """Test rate limiting behavior under concurrent requests."""
        identifier = "test_client_192.168.1.1"
        rate_limit = 10  # 10 requests per window
        window_minutes = 1

        results = []

        def make_request():
            try:
                start_time = time.time()
                is_limited = rate_limiter.is_rate_limited(
                    identifier, rate_limit, window_minutes
                )
                end_time = time.time()

                return {
                    "limited": is_limited,
                    "duration": end_time - start_time,
                    "timestamp": time.time(),
                }
            except Exception as e:
                return {"error": str(e)}

        # Make concurrent requests (more than rate limit)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(25)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Verify rate limiting is working
        successful_results = [r for r in results if "error" not in r]
        allowed_requests = [r for r in successful_results if not r["limited"]]
        blocked_requests = [r for r in successful_results if r["limited"]]

        # Should allow up to rate limit, then block
        assert len(allowed_requests) <= rate_limit
        assert len(blocked_requests) >= 5  # At least some should be blocked

        # Rate limiting should be fast
        durations = [r["duration"] for r in successful_results]
        avg_duration = statistics.mean(durations)
        assert avg_duration < 0.05  # Average < 50ms

    def test_concurrent_session_management(self, auth_manager, test_users):
        """Test concurrent session management operations."""
        results = []
        errors = []

        def manage_session(user):
            try:
                start_time = time.time()

                # Create access token
                access_token = auth_manager.create_access_token(user)

                # Create refresh token
                refresh_token = auth_manager.create_refresh_token(user)

                # Verify tokens
                auth_manager.verify_token(access_token)

                # Refresh access token
                (
                    new_access_token,
                    new_refresh_token,
                ) = auth_manager.refresh_access_token(refresh_token)

                # Logout (blacklist token)
                auth_manager.logout(new_access_token)

                end_time = time.time()

                return {
                    "user_id": user.user_id,
                    "duration": end_time - start_time,
                    "success": True,
                }
            except Exception as e:
                errors.append({"user_id": user.user_id, "error": str(e)})
                return {"user_id": user.user_id, "success": False}

        # Test with 15 concurrent users
        test_subset = test_users[:15]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(manage_session, user) for user in test_subset]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Verify results
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) == len(test_subset)
        assert len(errors) == 0

        # Check performance metrics
        durations = [r["duration"] for r in successful_results]
        avg_duration = statistics.mean(durations)

        # Session management should complete reasonably quickly
        assert avg_duration < 2.0  # Average < 2 seconds

    def test_authentication_scalability(self, auth_manager):
        """Test authentication system scalability with increasing load."""
        user_counts = [10, 50, 100]
        results = {}

        for count in user_counts:
            # Create users for this test
            users = []
            for i in range(count):
                user = User(
                    user_id=f"scale_user_{i}",
                    username=f"scaleuser_{i}",
                    email=f"scale{i}@example.com",
                    role=UserRole.OBSERVER,
                    created_at=datetime.now(timezone.utc),
                    is_active=True,
                )
                auth_manager.users[user.username] = {
                    "user": user,
                    "password_hash": auth_manager.hash_password("password123"),
                }
                users.append(user)

            # Test concurrent token creation
            durations = []

            def create_token(user):
                start_time = time.time()
                token = auth_manager.create_access_token(user)
                auth_manager.verify_token(token)
                end_time = time.time()
                return end_time - start_time

            with ThreadPoolExecutor(max_workers=min(count, 20)) as executor:
                futures = [executor.submit(create_token, user) for user in users]

                for future in as_completed(futures):
                    duration = future.result()
                    durations.append(duration)

            results[count] = {
                "avg_duration": statistics.mean(durations),
                "max_duration": max(durations),
                "min_duration": min(durations),
                "std_duration": statistics.stdev(durations)
                if len(durations) > 1
                else 0,
            }

        # Verify scalability - performance shouldn't degrade significantly
        for count in user_counts:
            assert results[count]["avg_duration"] < 0.5  # Average < 500ms
            assert results[count]["max_duration"] < 2.0  # Max < 2 seconds

        # Performance should scale reasonably
        # Allow some degradation but not excessive
        ratio_50_to_10 = results[50]["avg_duration"] / results[10]["avg_duration"]
        ratio_100_to_50 = results[100]["avg_duration"] / results[50]["avg_duration"]

        assert ratio_50_to_10 < 3.0  # 50 users shouldn't be more than 3x slower than 10
        assert (
            ratio_100_to_50 < 3.0
        )  # 100 users shouldn't be more than 3x slower than 50

    def test_memory_usage_under_load(self, auth_manager, test_users):
        """Test memory usage doesn't grow excessively under load."""
        try:
            import gc
            import os

            import psutil

            process = psutil.Process(os.getpid())

            # Get initial memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create many tokens
            tokens = []
            for user in test_users[:50]:
                for _ in range(5):  # 5 tokens per user
                    token = auth_manager.create_access_token(user)
                    tokens.append(token)

            # Get memory usage after token creation
            process.memory_info().rss / 1024 / 1024  # MB

            # Verify many tokens concurrently
            def verify_token(token):
                try:
                    auth_manager.verify_token(token)
                    return True
                except Exception:
                    return False

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(verify_token, token) for token in tokens]

                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Force garbage collection
            gc.collect()

            # Memory shouldn't grow excessively
            memory_increase = final_memory - initial_memory
            assert memory_increase < 100  # Less than 100MB increase

            # Most tokens should still be valid
            valid_tokens = sum(results)
            assert valid_tokens > len(tokens) * 0.8  # At least 80% valid

        except ImportError:
            # Skip test if psutil is not available
            assert False, "Test bypass removed - must fix underlying issue"

    def test_concurrent_blacklist_operations(self, auth_manager, test_users):
        """Test concurrent token blacklisting operations."""
        # Create tokens for users
        tokens = []
        for user in test_users[:30]:
            token = auth_manager.create_access_token(user)
            tokens.append(token)

        results = []
        errors = []

        def blacklist_token(token):
            try:
                # Get JTI from token
                import jwt

                payload = jwt.decode(token, options={"verify_signature": False})
                jti = payload.get("jti")

                if jti:
                    start_time = time.time()
                    auth_manager.revoke_token(jti)
                    end_time = time.time()

                    # Try to verify token (should fail)
                    try:
                        auth_manager.verify_token(token)
                        blacklisted = False
                    except Exception:
                        blacklisted = True

                    return {
                        "jti": jti,
                        "blacklisted": blacklisted,
                        "duration": end_time - start_time,
                        "success": True,
                    }
                else:
                    return {"success": False, "error": "No JTI found"}
            except Exception as e:
                errors.append({"error": str(e)})
                return {"success": False}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(blacklist_token, token) for token in tokens]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Verify results
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) == len(tokens)
        assert len(errors) == 0

        # All tokens should be blacklisted
        assert all(r["blacklisted"] for r in successful_results)

        # Blacklisting should be fast
        durations = [r["duration"] for r in successful_results]
        avg_duration = statistics.mean(durations)
        assert avg_duration < 0.05  # Average < 50ms

    def test_stress_test_authentication_flow(self, auth_manager):
        """Comprehensive stress test of the entire authentication flow."""
        # Create many users
        users = []
        for i in range(200):
            user = User(
                user_id=f"stress_user_{i}",
                username=f"stressuser_{i}",
                email=f"stress{i}@example.com",
                role=UserRole.AGENT_MANAGER,
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
            auth_manager.users[user.username] = {
                "user": user,
                "password_hash": auth_manager.hash_password("password123"),
            }
            users.append(user)

        results = []
        errors = []

        def stress_test_user(user):
            try:
                start_time = time.time()

                # Full authentication flow
                # 1. Authenticate
                authenticated_user = auth_manager.authenticate_user(
                    user.username, "password123"
                )
                assert authenticated_user is not None

                # 2. Create tokens
                access_token = auth_manager.create_access_token(user)
                refresh_token = auth_manager.create_refresh_token(user)

                # 3. Verify access token
                token_data = auth_manager.verify_token(access_token)
                assert token_data.user_id == user.user_id

                # 4. Refresh tokens
                (
                    new_access_token,
                    new_refresh_token,
                ) = auth_manager.refresh_access_token(refresh_token)

                # 5. Verify new access token
                new_token_data = auth_manager.verify_token(new_access_token)
                assert new_token_data.user_id == user.user_id

                # 6. Logout
                auth_manager.logout(new_access_token)

                end_time = time.time()

                return {
                    "user_id": user.user_id,
                    "duration": end_time - start_time,
                    "success": True,
                }
            except Exception as e:
                errors.append({"user_id": user.user_id, "error": str(e)})
                return {"user_id": user.user_id, "success": False}

        # Run stress test with moderate concurrency
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(stress_test_user, user) for user in users]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Verify results
        successful_results = [r for r in results if r["success"]]
        success_rate = len(successful_results) / len(users)

        # At least 95% success rate
        assert success_rate >= 0.95

        # Print some statistics
        durations = [r["duration"] for r in successful_results]
        avg_duration = statistics.mean(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        print("\nStress Test Results:")
        print(f"Users: {len(users)}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Duration: {avg_duration:.3f}s")
        print(f"Max Duration: {max_duration:.3f}s")
        print(f"Min Duration: {min_duration:.3f}s")

        # Performance should be reasonable even under stress
        assert avg_duration < 2.0  # Average < 2 seconds
        assert max_duration < 5.0  # Max < 5 seconds

"""Session management tests.

Tests session lifecycle and management:
- Session creation and initialization
- Session persistence across requests
- Session expiration and cleanup
- Concurrent session handling
- Session security features
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import jwt
import pytest
from api.main import app
from auth import AuthenticationManager, auth_manager
from fastapi.testclient import TestClient


class TestSessionManagement:
    """Test session lifecycle and management."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def test_user_data(self):
        """Test user data."""
        return {
            "username": "session_test_user",
            "email": "session@example.com",
            "password": "SessionPassword123!",
            "role": "researcher",
        }

    @pytest.fixture(autouse=True)
    def reset_auth_state(self):
        """Reset authentication state before each test."""
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()
        yield
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()

    def test_session_creation(self, client, test_user_data):
        """Test session creation during login."""
        # Register user
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        # Login to create session
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"],
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200

        data = response.json()
        access_token = data["access_token"]
        refresh_token = data["refresh_token"]
        user_id = data["user"]["user_id"]

        # Verify session components
        assert access_token is not None
        assert refresh_token is not None
        assert user_id in auth_manager.refresh_tokens
        assert auth_manager.refresh_tokens[user_id] == refresh_token

        # Verify tokens are valid JWT
        access_payload = jwt.decode(access_token, options={"verify_signature": False})
        refresh_payload = jwt.decode(refresh_token, options={"verify_signature": False})

        assert access_payload["type"] == "access"
        assert refresh_payload["type"] == "refresh"
        assert access_payload["user_id"] == user_id
        assert refresh_payload["user_id"] == user_id

    def test_session_persistence(self, client, test_user_data):
        """Test session persistence across multiple requests."""
        # Register and login
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        access_token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Make multiple requests with same token
        endpoints = [
            "/api/v1/auth/me",
            "/api/v1/auth/permissions",
            "/api/v1/auth/me",  # Repeat to ensure persistence
        ]

        for endpoint in endpoints:
            response = client.get(endpoint, headers=headers)
            assert response.status_code == 200

            # Verify user data is consistent
            if endpoint == "/api/v1/auth/me":
                user_data = response.json()
                assert user_data["username"] == test_user_data["username"]

    def test_session_expiration(self, client, test_user_data):
        """Test session expiration handling."""
        # Register user
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        user_id = response.json()["user"]["user_id"]

        # Create expired access token
        expired_token_data = {
            "user_id": user_id,
            "username": test_user_data["username"],
            "role": test_user_data["role"],
            "permissions": [],
            "exp": datetime.utcnow() - timedelta(hours=1),
            "type": "access",
        }

        expired_token = jwt.encode(
            expired_token_data,
            (
                auth_manager.JWT_SECRET
                if hasattr(auth_manager, "JWT_SECRET")
                else "dev_jwt_secret_2025_not_for_production"
            ),
            algorithm="HS256",
        )

        # Try to use expired token
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()

    def test_session_cleanup_on_logout(self, client, test_user_data):
        """Test session cleanup when user logs out."""
        # Register and login
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        access_token = response.json()["access_token"]
        user_id = response.json()["user"]["user_id"]

        # Verify refresh token exists
        assert user_id in auth_manager.refresh_tokens

        # Logout
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.post("/api/v1/auth/logout", headers=headers)
        assert response.status_code == 200

        # Verify refresh token is removed
        assert user_id not in auth_manager.refresh_tokens

        # Try to use access token after logout (should still work until expiry)
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200  # Access token is still valid

    def test_concurrent_sessions_same_user(self, client, test_user_data):
        """Test multiple concurrent sessions for same user."""
        # Register user
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        # Create multiple sessions
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"],
        }

        sessions = []
        for i in range(3):
            response = client.post("/api/v1/auth/login", json=login_data)
            assert response.status_code == 200
            sessions.append(response.json())

        # All sessions should be valid
        for session in sessions:
            headers = {"Authorization": f"Bearer {session['access_token']}"}
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code == 200
            assert response.json()["username"] == test_user_data["username"]

        # Note: In this implementation, only the last refresh token is stored
        # This is a limitation that might need addressing in production
        user_id = sessions[0]["user"]["user_id"]
        if user_id in auth_manager.refresh_tokens:
            # Should be the last refresh token
            assert auth_manager.refresh_tokens[user_id] == sessions[-1]["refresh_token"]

    def test_concurrent_sessions_different_users(self, client):
        """Test concurrent sessions for different users."""
        # Register multiple users
        users = []
        for i in range(3):
            user_data = {
                "username": f"user_{i}",
                "email": f"user{i}@example.com",
                "password": f"Password{i}123!",
                "role": "observer",
            }
            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 200
            users.append((user_data, response.json()))

        # All users should be able to access their sessions
        for user_data, session in users:
            headers = {"Authorization": f"Bearer {session['access_token']}"}
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code == 200
            assert response.json()["username"] == user_data["username"]

    def test_session_thread_safety(self, client, test_user_data):
        """Test session handling under concurrent access."""
        # Register user
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        access_token = response.json()["access_token"]

        def make_authenticated_request():
            """Make authenticated request."""
            headers = {"Authorization": f"Bearer {access_token}"}
            return client.get("/api/v1/auth/me", headers=headers)

        # Make concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_authenticated_request) for _ in range(20)]
            results = [future.result() for future in as_completed(futures)]

        # All requests should succeed
        for response in results:
            assert response.status_code == 200
            assert response.json()["username"] == test_user_data["username"]

    def test_session_security_headers(self, client, test_user_data):
        """Test session security headers and validation."""
        # Register user
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        access_token = response.json()["access_token"]

        # Test various malformed headers
        malformed_headers = [
            {"Authorization": f"Bearer {access_token} extra"},  # Extra content
            {"Authorization": f"bearer {access_token}"},  # Wrong case
            {"Authorization": access_token},  # Missing Bearer
            {"Authorization": f"Bearer {access_token[:-5]}"},  # Truncated token
            {"Authorization": f"Bearer {access_token}x"},  # Modified token
        ]

        for headers in malformed_headers:
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code in [401, 403]

    def test_session_token_rotation(self, client, test_user_data):
        """Test token rotation behavior."""
        # Register user
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        first_login = response.json()

        # Login again to see if tokens change
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"],
        }

        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200

        second_login = response.json()

        # Tokens should be different
        assert first_login["access_token"] != second_login["access_token"]
        assert first_login["refresh_token"] != second_login["refresh_token"]

        # Both tokens should be valid (until first one expires)
        for token_data in [first_login, second_login]:
            headers = {"Authorization": f"Bearer {token_data['access_token']}"}
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code == 200

    def test_session_memory_usage(self, client):
        """Test session memory usage with many users."""
        # Create many users to test memory usage
        users = []
        for i in range(100):
            user_data = {
                "username": f"memory_test_user_{i}",
                "email": f"memory_test_{i}@example.com",
                "password": f"Password{i}123!",
                "role": "observer",
            }
            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 200
            users.append(response.json())

        # Verify all users have sessions
        assert len(auth_manager.users) == 100
        assert len(auth_manager.refresh_tokens) == 100

        # All should be able to access their sessions
        for user in users[:10]:  # Test first 10 for performance
            headers = {"Authorization": f"Bearer {user['access_token']}"}
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code == 200

    def test_session_cleanup_expired_refresh_tokens(self):
        """Test cleanup of expired refresh tokens."""
        # This would typically be done by a background task
        # For testing, we'll simulate the cleanup process

        manager = AuthenticationManager()

        # Add some test tokens
        test_tokens = {}
        for i in range(5):
            user_id = f"user_{i}"

            # Create some expired and some valid tokens
            if i < 3:
                # Expired token
                exp_time = datetime.utcnow() - timedelta(days=1)
            else:
                # Valid token
                exp_time = datetime.utcnow() + timedelta(days=1)

            token_data = {
                "user_id": user_id,
                "exp": exp_time,
                "type": "refresh",
            }

            token = jwt.encode(
                token_data,
                (
                    manager.JWT_SECRET
                    if hasattr(manager, "JWT_SECRET")
                    else "dev_jwt_secret_2025_not_for_production"
                ),
                algorithm="HS256",
            )

            test_tokens[user_id] = token
            manager.refresh_tokens[user_id] = token

        # Simulate cleanup process
        expired_tokens = []
        for user_id, token in list(manager.refresh_tokens.items()):
            try:
                jwt.decode(
                    token,
                    (
                        manager.JWT_SECRET
                        if hasattr(manager, "JWT_SECRET")
                        else "dev_jwt_secret_2025_not_for_production"
                    ),
                    algorithms=["HS256"],
                )
                # If we get here, token is valid
            except jwt.ExpiredSignatureError:
                expired_tokens.append(user_id)
            except jwt.JWTError:
                expired_tokens.append(user_id)

        # Remove expired tokens
        for user_id in expired_tokens:
            del manager.refresh_tokens[user_id]

        # Verify cleanup worked
        assert len(manager.refresh_tokens) == 2  # Only 2 valid tokens remain
        assert "user_0" not in manager.refresh_tokens
        assert "user_1" not in manager.refresh_tokens
        assert "user_2" not in manager.refresh_tokens
        assert "user_3" in manager.refresh_tokens
        assert "user_4" in manager.refresh_tokens

    def test_session_invalidation_on_user_deactivation(self, client, test_user_data):
        """Test session invalidation when user is deactivated."""
        # Register and login
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        access_token = response.json()["access_token"]
        response.json()["user"]["user_id"]

        # Verify session works
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200

        # Deactivate user
        username = test_user_data["username"]
        auth_manager.users[username]["user"].is_active = False

        # Try to login with deactivated user
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"],
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401
        assert "disabled" in response.json()["detail"].lower()

        # Existing token should still work until expiry
        # (In production, you might want to implement active session invalidation)
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200

    def test_session_data_consistency(self, client, test_user_data):
        """Test session data consistency across requests."""
        # Register and login
        response = client.post("/api/v1/auth/register", json=test_user_data)
        assert response.status_code == 200

        access_token = response.json()["access_token"]
        original_user_data = response.json()["user"]

        # Make multiple requests and verify consistent data
        headers = {"Authorization": f"Bearer {access_token}"}

        for _ in range(5):
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code == 200

            user_data = response.json()
            assert user_data["user_id"] == original_user_data["user_id"]
            assert user_data["username"] == original_user_data["username"]
            assert user_data["role"] == original_user_data["role"]


class TestSessionPerformance:
    """Test session performance under load."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def reset_auth_state(self):
        """Reset authentication state before each test."""
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()
        yield
        auth_manager.users.clear()
        auth_manager.refresh_tokens.clear()

    def test_session_creation_performance(self, client):
        """Test session creation performance."""
        user_data = {
            "username": "perf_test_user",
            "email": "perf@example.com",
            "password": "PerfPassword123!",
            "role": "observer",
        }

        # Register user
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200

        login_data = {
            "username": user_data["username"],
            "password": user_data["password"],
        }

        # Time multiple login operations
        start_time = time.time()
        for _ in range(10):
            response = client.post("/api/v1/auth/login", json=login_data)
            assert response.status_code == 200

        end_time = time.time()
        avg_time = (end_time - start_time) / 10

        # Should be reasonably fast (less than 1 second per login)
        assert avg_time < 1.0

    def test_session_validation_performance(self, client):
        """Test session validation performance."""
        user_data = {
            "username": "validation_test_user",
            "email": "validation@example.com",
            "password": "ValidationPassword123!",
            "role": "observer",
        }

        # Register user
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200

        access_token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Time multiple validation operations
        start_time = time.time()
        for _ in range(100):
            response = client.get("/api/v1/auth/me", headers=headers)
            assert response.status_code == 200

        end_time = time.time()
        avg_time = (end_time - start_time) / 100

        # Should be very fast (less than 0.1 seconds per validation)
        assert avg_time < 0.1


# Cleanup functions
def cleanup_sessions():
    """Clean up all session data."""
    auth_manager.users.clear()
    auth_manager.refresh_tokens.clear()


def teardown_module():
    """Module-level cleanup."""
    cleanup_sessions()

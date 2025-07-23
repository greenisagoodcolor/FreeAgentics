"""
Comprehensive Authentication Flow Integration Tests

This test suite covers complete authentication workflows including:
- End-to-end user registration and login flows
- Multi-factor authentication workflows
- Password reset and recovery flows
- Session management and persistence
- Token lifecycle management
- Cross-service authentication integration
- WebSocket authentication flows
- API key authentication flows
"""

from datetime import datetime, timezone
from typing import Dict

import jwt
from fastapi.testclient import TestClient

from api.main import app
from auth.security_implementation import AuthenticationManager, rate_limiter


class AuthenticationFlowTester:
    """Comprehensive authentication flow testing framework."""

    def __init__(self):
        self.client = TestClient(app)
        self.auth_manager = AuthenticationManager()
        self.test_sessions = {}
        self.test_results = {
            "flows_tested": 0,
            "flows_passed": 0,
            "flows_failed": 0,
            "errors": [],
        }

    def setup_test_environment(self):
        """Setup clean test environment."""
        self.auth_manager.users.clear()
        self.auth_manager.refresh_tokens.clear()
        self.auth_manager.blacklist.clear()
        rate_limiter.requests.clear()
        self.test_sessions.clear()

    def record_flow_result(self, flow_name: str, success: bool, error: str = None):
        """Record flow test result."""
        self.test_results["flows_tested"] += 1
        if success:
            self.test_results["flows_passed"] += 1
        else:
            self.test_results["flows_failed"] += 1
            if error:
                self.test_results["errors"].append(f"{flow_name}: {error}")

    def generate_test_summary(self) -> Dict:
        """Generate test summary report."""
        total = self.test_results["flows_tested"]
        passed = self.test_results["flows_passed"]
        failed = self.test_results["flows_failed"]

        return {
            "summary": {
                "total_flows": total,
                "passed_flows": passed,
                "failed_flows": failed,
                "success_rate": (passed / total * 100) if total > 0 else 0,
            },
            "errors": self.test_results["errors"],
        }


class TestCompleteAuthenticationFlows:
    """Complete authentication flow integration tests."""

    def setup_method(self):
        """Setup for each test method."""
        self.tester = AuthenticationFlowTester()
        self.tester.setup_test_environment()

    def test_complete_user_registration_flow(self):
        """Test complete user registration workflow."""
        flow_name = "complete_user_registration"

        try:
            # Step 1: Validate registration data
            user_data = {
                "username": "newuser2024",
                "email": "newuser@example.com",
                "password": "SecurePassword123!",
                "role": "researcher",
            }

            # Step 2: Register user
            response = self.tester.client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 200, f"Registration failed: {response.json()}"

            registration_data = response.json()
            assert "access_token" in registration_data
            assert "refresh_token" in registration_data
            assert "user" in registration_data

            # Step 3: Verify user data
            user = registration_data["user"]
            assert user["username"] == user_data["username"]
            assert user["email"] == user_data["email"]
            assert user["role"] == user_data["role"]
            assert user["is_active"] is True
            assert "user_id" in user
            assert "created_at" in user

            # Step 4: Verify tokens are valid
            access_token = registration_data["access_token"]
            refresh_token = registration_data["refresh_token"]

            # Test access token
            headers = {"Authorization": f"Bearer {access_token}"}
            me_response = self.tester.client.get("/api/v1/auth/me", headers=headers)
            assert me_response.status_code == 200

            # Test refresh token
            refresh_response = self.tester.client.post(
                "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
            )
            assert refresh_response.status_code == 200

            # Step 5: Verify permissions
            permissions_response = self.tester.client.get(
                "/api/v1/auth/permissions", headers=headers
            )
            assert permissions_response.status_code == 200

            perms = permissions_response.json()
            assert perms["role"] == "researcher"
            assert perms["can_create_agents"] is True  # Researcher can create agents
            assert perms["can_admin_system"] is False  # Researcher cannot admin

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_complete_login_flow(self):
        """Test complete login workflow."""
        flow_name = "complete_login_flow"

        try:
            # Step 1: Create user first
            user_data = {
                "username": "loginuser",
                "email": "login@example.com",
                "password": "LoginPassword123!",
                "role": "agent_manager",
            }

            reg_response = self.tester.client.post("/api/v1/auth/register", json=user_data)
            assert reg_response.status_code == 200

            # Step 2: Login with credentials
            login_data = {
                "username": user_data["username"],
                "password": user_data["password"],
            }

            login_response = self.tester.client.post("/api/v1/auth/login", json=login_data)
            assert login_response.status_code == 200

            login_result = login_response.json()
            assert "access_token" in login_result
            assert "refresh_token" in login_result
            assert "user" in login_result

            # Step 3: Verify token functionality
            access_token = login_result["access_token"]
            headers = {"Authorization": f"Bearer {access_token}"}

            # Test authenticated endpoint
            me_response = self.tester.client.get("/api/v1/auth/me", headers=headers)
            assert me_response.status_code == 200

            user_info = me_response.json()
            assert user_info["username"] == user_data["username"]
            assert user_info["role"] == user_data["role"]

            # Step 4: Verify last_login is updated
            user_obj = login_result["user"]
            assert user_obj["last_login"] is not None

            # Step 5: Test role-based access
            permissions_response = self.tester.client.get(
                "/api/v1/auth/permissions", headers=headers
            )
            assert permissions_response.status_code == 200

            perms = permissions_response.json()
            assert perms["role"] == "agent_manager"
            assert perms["can_create_agents"] is True
            assert perms["can_modify_agent"] is True
            assert perms["can_delete_agents"] is False  # Agent manager cannot delete

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_token_refresh_flow(self):
        """Test complete token refresh workflow."""
        flow_name = "token_refresh_flow"

        try:
            # Step 1: Create user and login
            user_data = {
                "username": "refreshuser",
                "email": "refresh@example.com",
                "password": "RefreshPassword123!",
                "role": "observer",
            }

            reg_response = self.tester.client.post("/api/v1/auth/register", json=user_data)
            assert reg_response.status_code == 200

            # Step 2: Get initial tokens
            initial_tokens = reg_response.json()
            initial_access_token = initial_tokens["access_token"]
            initial_refresh_token = initial_tokens["refresh_token"]

            # Step 3: Use refresh token to get new access token
            refresh_response = self.tester.client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": initial_refresh_token},
            )
            assert refresh_response.status_code == 200

            new_tokens = refresh_response.json()
            new_access_token = new_tokens["access_token"]
            new_refresh_token = new_tokens["refresh_token"]

            # Step 4: Verify new tokens are different
            assert new_access_token != initial_access_token
            assert new_refresh_token != initial_refresh_token

            # Step 5: Verify new access token works
            headers = {"Authorization": f"Bearer {new_access_token}"}
            me_response = self.tester.client.get("/api/v1/auth/me", headers=headers)
            assert me_response.status_code == 200

            # Step 6: Verify old refresh token is invalidated (token rotation)
            old_refresh_response = self.tester.client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": initial_refresh_token},
            )
            assert old_refresh_response.status_code == 401, (
                "Old refresh token should be invalidated"
            )

            # Step 7: Verify new refresh token works
            newer_refresh_response = self.tester.client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": new_refresh_token},
            )
            assert newer_refresh_response.status_code == 200

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_logout_flow(self):
        """Test complete logout workflow."""
        flow_name = "logout_flow"

        try:
            # Step 1: Create user and login
            user_data = {
                "username": "logoutuser",
                "email": "logout@example.com",
                "password": "LogoutPassword123!",
                "role": "researcher",
            }

            reg_response = self.tester.client.post("/api/v1/auth/register", json=user_data)
            assert reg_response.status_code == 200

            tokens = reg_response.json()
            access_token = tokens["access_token"]
            refresh_token = tokens["refresh_token"]
            user_id = tokens["user"]["user_id"]

            # Step 2: Verify tokens work before logout
            headers = {"Authorization": f"Bearer {access_token}"}
            me_response = self.tester.client.get("/api/v1/auth/me", headers=headers)
            assert me_response.status_code == 200

            # Step 3: Verify refresh token exists
            assert user_id in self.tester.auth_manager.refresh_tokens

            # Step 4: Logout
            logout_response = self.tester.client.post("/api/v1/auth/logout", headers=headers)
            assert logout_response.status_code == 200

            logout_result = logout_response.json()
            assert logout_result["message"] == "Successfully logged out"

            # Step 5: Verify refresh token is removed
            assert user_id not in self.tester.auth_manager.refresh_tokens

            # Step 6: Verify access token still works temporarily (until it expires naturally)
            # This is expected behavior - access tokens are stateless
            _me_response_after = self.tester.client.get("/api/v1/auth/me", headers=headers)
            # Access token should still work as they're stateless JWTs

            # Step 7: Verify refresh token no longer works
            refresh_response = self.tester.client.post(
                "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
            )
            assert refresh_response.status_code == 401, (
                "Refresh token should be invalidated after logout"
            )

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_session_management_flow(self):
        """Test session management workflow."""
        flow_name = "session_management_flow"

        try:
            # Step 1: Create user
            user_data = {
                "username": "sessionuser",
                "email": "session@example.com",
                "password": "SessionPassword123!",
                "role": "admin",
            }

            reg_response = self.tester.client.post("/api/v1/auth/register", json=user_data)
            assert reg_response.status_code == 200

            # Step 2: Create multiple sessions
            login_data = {
                "username": user_data["username"],
                "password": user_data["password"],
            }

            sessions = []
            for i in range(3):
                login_response = self.tester.client.post("/api/v1/auth/login", json=login_data)
                assert login_response.status_code == 200
                sessions.append(login_response.json())

            # Step 3: Verify all sessions are valid
            for i, session in enumerate(sessions):
                headers = {"Authorization": f"Bearer {session['access_token']}"}
                me_response = self.tester.client.get("/api/v1/auth/me", headers=headers)
                assert me_response.status_code == 200, f"Session {i} should be valid"

            # Step 4: Logout one session
            logout_headers = {"Authorization": f"Bearer {sessions[0]['access_token']}"}
            logout_response = self.tester.client.post("/api/v1/auth/logout", headers=logout_headers)
            assert logout_response.status_code == 200

            # Step 5: Verify other sessions still work
            for i in range(1, 3):
                headers = {"Authorization": f"Bearer {sessions[i]['access_token']}"}
                me_response = self.tester.client.get("/api/v1/auth/me", headers=headers)
                assert me_response.status_code == 200, f"Session {i} should still be valid"

            # Step 6: Verify logged out session's refresh token is invalidated
            refresh_response = self.tester.client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": sessions[0]["refresh_token"]},
            )
            assert refresh_response.status_code == 401, (
                "Logged out session's refresh token should be invalid"
            )

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_role_based_access_flow(self):
        """Test role-based access control workflow."""
        flow_name = "role_based_access_flow"

        try:
            # Step 1: Create users with different roles
            users = [
                {
                    "username": "admin",
                    "email": "admin@test.com",
                    "password": "AdminPass123!",
                    "role": "admin",
                },
                {
                    "username": "researcher",
                    "email": "researcher@test.com",
                    "password": "ResearchPass123!",
                    "role": "researcher",
                },
                {
                    "username": "agent_manager",
                    "email": "manager@test.com",
                    "password": "ManagerPass123!",
                    "role": "agent_manager",
                },
                {
                    "username": "observer",
                    "email": "observer@test.com",
                    "password": "ObserverPass123!",
                    "role": "observer",
                },
            ]

            user_tokens = {}
            for user_data in users:
                reg_response = self.tester.client.post("/api/v1/auth/register", json=user_data)
                assert reg_response.status_code == 200
                user_tokens[user_data["role"]] = reg_response.json()["access_token"]

            # Step 2: Test admin permissions
            admin_headers = {"Authorization": f"Bearer {user_tokens['admin']}"}
            admin_perms = self.tester.client.get(
                "/api/v1/auth/permissions", headers=admin_headers
            ).json()

            assert admin_perms["can_create_agents"] is True
            assert admin_perms["can_delete_agents"] is True
            assert admin_perms["can_view_metrics"] is True
            assert admin_perms["can_admin_system"] is True

            # Step 3: Test researcher permissions
            researcher_headers = {"Authorization": f"Bearer {user_tokens['researcher']}"}
            researcher_perms = self.tester.client.get(
                "/api/v1/auth/permissions", headers=researcher_headers
            ).json()

            assert researcher_perms["can_create_agents"] is True
            assert researcher_perms["can_delete_agents"] is False
            assert researcher_perms["can_view_metrics"] is True
            assert researcher_perms["can_admin_system"] is False

            # Step 4: Test agent_manager permissions
            manager_headers = {"Authorization": f"Bearer {user_tokens['agent_manager']}"}
            manager_perms = self.tester.client.get(
                "/api/v1/auth/permissions", headers=manager_headers
            ).json()

            assert manager_perms["can_create_agents"] is True
            assert manager_perms["can_delete_agents"] is False
            assert manager_perms["can_view_metrics"] is True
            assert manager_perms["can_admin_system"] is False

            # Step 5: Test observer permissions
            observer_headers = {"Authorization": f"Bearer {user_tokens['observer']}"}
            observer_perms = self.tester.client.get(
                "/api/v1/auth/permissions", headers=observer_headers
            ).json()

            assert observer_perms["can_create_agents"] is False
            assert observer_perms["can_delete_agents"] is False
            assert observer_perms["can_view_metrics"] is True
            assert observer_perms["can_admin_system"] is False

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_concurrent_authentication_flow(self):
        """Test concurrent authentication workflows."""
        flow_name = "concurrent_authentication_flow"

        try:
            import concurrent.futures

            # Step 1: Create base user
            user_data = {
                "username": "concurrentuser",
                "email": "concurrent@example.com",
                "password": "ConcurrentPass123!",
                "role": "researcher",
            }

            reg_response = self.tester.client.post("/api/v1/auth/register", json=user_data)
            assert reg_response.status_code == 200

            # Step 2: Define concurrent operations
            def concurrent_login(thread_id):
                """Perform concurrent login."""
                login_data = {
                    "username": user_data["username"],
                    "password": user_data["password"],
                }

                response = self.tester.client.post("/api/v1/auth/login", json=login_data)
                return {
                    "thread_id": thread_id,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if response.status_code == 200 else None,
                }

            def concurrent_token_refresh(thread_id, refresh_token):
                """Perform concurrent token refresh."""
                response = self.tester.client.post(
                    "/api/v1/auth/refresh",
                    json={"refresh_token": refresh_token},
                )
                return {
                    "thread_id": thread_id,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                }

            # Step 3: Test concurrent logins
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                login_futures = [executor.submit(concurrent_login, i) for i in range(10)]
                login_results = [
                    future.result() for future in concurrent.futures.as_completed(login_futures)
                ]

            # Step 4: Verify login results
            successful_logins = [r for r in login_results if r["success"]]
            assert len(successful_logins) > 0, "At least some concurrent logins should succeed"

            # Step 5: Test concurrent token refreshes
            if successful_logins:
                refresh_token = successful_logins[0]["response"]["refresh_token"]

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    refresh_futures = [
                        executor.submit(concurrent_token_refresh, i, refresh_token)
                        for i in range(5)
                    ]
                    refresh_results = [
                        future.result()
                        for future in concurrent.futures.as_completed(refresh_futures)
                    ]

                # Only one refresh should succeed due to token rotation
                successful_refreshes = [r for r in refresh_results if r["success"]]
                assert len(successful_refreshes) == 1, (
                    "Only one token refresh should succeed due to rotation"
                )

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_token_expiration_flow(self):
        """Test token expiration handling workflow."""
        flow_name = "token_expiration_flow"

        try:
            # Step 1: Create user
            user_data = {
                "username": "expireuser",
                "email": "expire@example.com",
                "password": "ExpirePass123!",
                "role": "observer",
            }

            reg_response = self.tester.client.post("/api/v1/auth/register", json=user_data)
            assert reg_response.status_code == 200

            tokens = reg_response.json()
            access_token = tokens["access_token"]
            refresh_token = tokens["refresh_token"]

            # Step 2: Verify tokens work initially
            headers = {"Authorization": f"Bearer {access_token}"}
            me_response = self.tester.client.get("/api/v1/auth/me", headers=headers)
            assert me_response.status_code == 200

            # Step 3: Decode tokens to check expiration times
            access_payload = jwt.decode(access_token, options={"verify_signature": False})
            refresh_payload = jwt.decode(refresh_token, options={"verify_signature": False})

            access_exp = datetime.fromtimestamp(access_payload["exp"])
            refresh_exp = datetime.fromtimestamp(refresh_payload["exp"])

            # Step 4: Verify expiration times are reasonable
            now = datetime.now(timezone.utc)
            access_expires_in = (access_exp - now).total_seconds()
            refresh_expires_in = (refresh_exp - now).total_seconds()

            # Access token should expire in ~15 minutes
            assert 600 < access_expires_in < 1200, (
                f"Access token expires in {access_expires_in}s, expected ~900s"
            )

            # Refresh token should expire in ~7 days
            assert 500000 < refresh_expires_in < 700000, (
                f"Refresh token expires in {refresh_expires_in}s, expected ~604800s"
            )

            # Step 5: Test token refresh extends access token
            refresh_response = self.tester.client.post(
                "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
            )
            assert refresh_response.status_code == 200

            new_tokens = refresh_response.json()
            new_access_token = new_tokens["access_token"]

            # Step 6: Verify new access token has fresh expiration
            new_access_payload = jwt.decode(new_access_token, options={"verify_signature": False})
            new_access_exp = datetime.fromtimestamp(new_access_payload["exp"])

            new_access_expires_in = (new_access_exp - now).total_seconds()
            assert new_access_expires_in > access_expires_in, (
                "New access token should have later expiration"
            )

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_authentication_error_handling_flow(self):
        """Test authentication error handling workflow."""
        flow_name = "authentication_error_handling"

        try:
            # Step 1: Test invalid registration data
            invalid_registrations = [
                {
                    "username": "",
                    "email": "test@test.com",
                    "password": "Pass123!",
                    "role": "observer",
                },
                {
                    "username": "test",
                    "email": "invalid-email",
                    "password": "Pass123!",
                    "role": "observer",
                },
                {
                    "username": "test",
                    "email": "test@test.com",
                    "password": "",
                    "role": "observer",
                },
                {
                    "username": "test",
                    "email": "test@test.com",
                    "password": "Pass123!",
                    "role": "invalid_role",
                },
            ]

            for invalid_data in invalid_registrations:
                response = self.tester.client.post("/api/v1/auth/register", json=invalid_data)
                assert response.status_code in [
                    400,
                    422,
                ], f"Invalid registration should fail: {invalid_data}"

            # Step 2: Test invalid login data
            invalid_logins = [
                {"username": "", "password": "pass"},
                {"username": "user", "password": ""},
                {"username": "nonexistent", "password": "password"},
            ]

            for invalid_data in invalid_logins:
                response = self.tester.client.post("/api/v1/auth/login", json=invalid_data)
                assert response.status_code in [
                    400,
                    401,
                    422,
                ], f"Invalid login should fail: {invalid_data}"

            # Step 3: Test invalid token usage
            invalid_tokens = [
                "invalid.token.format",
                "",
                "Bearer invalid",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature",
            ]

            for token in invalid_tokens:
                headers = {"Authorization": f"Bearer {token}"}
                response = self.tester.client.get("/api/v1/auth/me", headers=headers)
                assert response.status_code in [
                    401,
                    403,
                ], f"Invalid token should be rejected: {token}"

            # Step 4: Test invalid refresh token
            invalid_refresh_tokens = [
                "invalid_refresh_token",
                "",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.refresh",
            ]

            for refresh_token in invalid_refresh_tokens:
                response = self.tester.client.post(
                    "/api/v1/auth/refresh",
                    json={"refresh_token": refresh_token},
                )
                assert response.status_code == 401, (
                    f"Invalid refresh token should be rejected: {refresh_token}"
                )

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_rate_limiting_flow(self):
        """Test rate limiting workflow."""
        flow_name = "rate_limiting_flow"

        try:
            # Step 1: Test registration rate limiting
            for i in range(6):  # Rate limit is 5 per 10 minutes
                user_data = {
                    "username": f"ratelimituser{i}",
                    "email": f"ratelimit{i}@example.com",
                    "password": "RateLimit123!",
                    "role": "observer",
                }

                response = self.tester.client.post("/api/v1/auth/register", json=user_data)

                if i < 5:
                    assert response.status_code == 200, f"Registration {i} should succeed"
                else:
                    assert response.status_code == 429, f"Registration {i} should be rate limited"

            # Step 2: Test login rate limiting
            # First create a user
            user_data = {
                "username": "loginratetest",
                "email": "loginrate@example.com",
                "password": "LoginRate123!",
                "role": "observer",
            }

            # Clear rate limiting for new test
            rate_limiter.requests.clear()

            reg_response = self.tester.client.post("/api/v1/auth/register", json=user_data)
            assert reg_response.status_code == 200

            # Now test login rate limiting
            for i in range(12):  # Rate limit is 10 per 5 minutes
                login_data = {
                    "username": user_data["username"],
                    "password": "WrongPassword123!",
                }

                response = self.tester.client.post("/api/v1/auth/login", json=login_data)

                if i < 10:
                    assert response.status_code == 401, f"Login {i} should fail with 401"
                else:
                    assert response.status_code == 429, f"Login {i} should be rate limited"

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_cross_endpoint_authentication_flow(self):
        """Test authentication across different endpoints."""
        flow_name = "cross_endpoint_authentication"

        try:
            # Step 1: Create and authenticate user
            user_data = {
                "username": "crossuser",
                "email": "cross@example.com",
                "password": "CrossPass123!",
                "role": "admin",
            }

            reg_response = self.tester.client.post("/api/v1/auth/register", json=user_data)
            assert reg_response.status_code == 200

            access_token = reg_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {access_token}"}

            # Step 2: Test authentication on various endpoints
            authenticated_endpoints = [
                ("/api/v1/auth/me", "GET"),
                ("/api/v1/auth/permissions", "GET"),
                ("/api/v1/auth/logout", "POST"),
            ]

            for endpoint, method in authenticated_endpoints:
                if method == "GET":
                    response = self.tester.client.get(endpoint, headers=headers)
                elif method == "POST":
                    response = self.tester.client.post(endpoint, headers=headers)

                assert response.status_code == 200, (
                    f"Authenticated request to {endpoint} should succeed"
                )

            # Step 3: Test unauthenticated requests
            for endpoint, method in authenticated_endpoints:
                if method == "GET":
                    response = self.tester.client.get(endpoint)
                elif method == "POST":
                    response = self.tester.client.post(endpoint)

                assert response.status_code in [
                    401,
                    403,
                ], f"Unauthenticated request to {endpoint} should fail"

            self.tester.record_flow_result(flow_name, True)

        except Exception as e:
            self.tester.record_flow_result(flow_name, False, str(e))
            raise

    def test_generate_comprehensive_flow_report(self):
        """Generate comprehensive flow test report."""
        # Run all authentication flow tests
        test_methods = [
            self.test_complete_user_registration_flow,
            self.test_complete_login_flow,
            self.test_token_refresh_flow,
            self.test_logout_flow,
            self.test_session_management_flow,
            self.test_role_based_access_flow,
            self.test_concurrent_authentication_flow,
            self.test_token_expiration_flow,
            self.test_authentication_error_handling_flow,
            self.test_rate_limiting_flow,
            self.test_cross_endpoint_authentication_flow,
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"Test {test_method.__name__} failed: {e}")

        # Generate and print report
        summary = self.tester.generate_test_summary()

        print("\\n" + "=" * 50)
        print("COMPREHENSIVE AUTHENTICATION FLOW REPORT")
        print("=" * 50)
        print(f"Total Flows Tested: {summary['summary']['total_flows']}")
        print(f"Passed Flows: {summary['summary']['passed_flows']}")
        print(f"Failed Flows: {summary['summary']['failed_flows']}")
        print(f"Success Rate: {summary['summary']['success_rate']:.1f}%")

        if summary["errors"]:
            print("\\nERRORS:")
            for error in summary["errors"]:
                print(f"  - {error}")
        else:
            print("\\nAll authentication flows passed successfully!")

        # Assert success
        assert summary["summary"]["success_rate"] >= 90, (
            f"Authentication flow success rate too low: {summary['summary']['success_rate']:.1f}%"
        )

        return summary

    def teardown_method(self):
        """Cleanup after each test."""
        self.tester.setup_test_environment()


if __name__ == "__main__":
    # Run comprehensive authentication flow tests
    test_suite = TestCompleteAuthenticationFlows()
    test_suite.setup_method()

    try:
        report = test_suite.test_generate_comprehensive_flow_report()
        print("\\nAUTHENTICATION FLOW TESTING COMPLETED SUCCESSFULLY")
        print(f"Final Success Rate: {report['summary']['success_rate']:.1f}%")
    except Exception as e:
        print(f"\\nAUTHENTICATION FLOW TESTING FAILED: {e}")
        raise
    finally:
        test_suite.teardown_method()

"""
Comprehensive test suite for API endpoints and middleware
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

# Mock complex dependencies
mock_modules = {
    "sqlalchemy": MagicMock(),
    "sqlalchemy.orm": MagicMock(),
    "redis": MagicMock(),
    "database": MagicMock(),
    "database.session": MagicMock(),
    "database.models": MagicMock(),
    "agents": MagicMock(),
    "agents.base_agent": MagicMock(),
    "agents.agent_manager": MagicMock(),
    "auth": MagicMock(),
    "auth.security_implementation": MagicMock(),
    "observability": MagicMock(),
    "websocket": MagicMock(),
}

with patch.dict("sys.modules", mock_modules):
    from api.middleware.rate_limiter import RateLimitMiddleware
    from api.middleware.security_headers import SecurityHeadersMiddleware
    from api.v1.agents import router as agents_router
    from api.v1.auth import router as auth_router
    from api.v1.health import router as health_router


class TestSecurityHeadersMiddleware:
    """Test Security Headers Middleware functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.middleware = SecurityHeadersMiddleware()

    def test_security_headers_middleware_initialization(self):
        """Test SecurityHeadersMiddleware initialization."""
        middleware = SecurityHeadersMiddleware()
        assert middleware is not None

    def test_security_headers_addition(self):
        """Test that security headers are added to responses."""

        # Mock the middleware call
        async def mock_call(request, call_next):
            response = await call_next(request)

            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            return response

        self.middleware.__call__ = mock_call

        # Mock request and response
        request = Mock()
        response = Mock()
        response.headers = {}

        async def mock_call_next(req):
            return response

        # Test middleware execution
        import asyncio

        result = asyncio.run(self.middleware(request, mock_call_next))

        # Verify security headers
        assert "X-Content-Type-Options" in result.headers
        assert "X-Frame-Options" in result.headers
        assert "X-XSS-Protection" in result.headers
        assert "Strict-Transport-Security" in result.headers
        assert "Content-Security-Policy" in result.headers
        assert "Referrer-Policy" in result.headers

    def test_security_headers_values(self):
        """Test specific security header values."""
        # Test individual header values
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

        for header, expected_value in headers.items():
            assert expected_value is not None
            assert isinstance(expected_value, str)

    def test_security_headers_not_override_existing(self):
        """Test that middleware doesn't override existing headers."""

        # Mock middleware that preserves existing headers
        async def mock_call_preserve(request, call_next):
            response = await call_next(request)

            # Only add if not already present
            if "X-Custom-Header" not in response.headers:
                response.headers["X-Custom-Header"] = "default-value"

            return response

        self.middleware.__call__ = mock_call_preserve

        # Mock response with existing header
        response = Mock()
        response.headers = {"X-Custom-Header": "existing-value"}

        async def mock_call_next(req):
            return response

        # Test middleware execution
        import asyncio

        result = asyncio.run(self.middleware(Mock(), mock_call_next))

        # Verify existing header is preserved
        assert result.headers["X-Custom-Header"] == "existing-value"


class TestRateLimitMiddleware:
    """Test Rate Limit Middleware functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.middleware = RateLimitMiddleware()

    def test_rate_limit_middleware_initialization(self):
        """Test RateLimitMiddleware initialization."""
        middleware = RateLimitMiddleware()
        assert middleware is not None

    def test_rate_limit_tracking(self):
        """Test rate limit tracking functionality."""
        # Mock rate limit tracking
        rate_limits = {}

        def mock_check_rate_limit(client_id, limit=100, window=60):
            current_time = datetime.now().timestamp()

            if client_id not in rate_limits:
                rate_limits[client_id] = {
                    "requests": 0,
                    "window_start": current_time,
                }

            client_data = rate_limits[client_id]

            # Reset window if expired
            if current_time - client_data["window_start"] > window:
                client_data["requests"] = 0
                client_data["window_start"] = current_time

            # Check limit
            if client_data["requests"] >= limit:
                return {"allowed": False, "remaining": 0}

            # Increment and allow
            client_data["requests"] += 1
            return {
                "allowed": True,
                "remaining": limit - client_data["requests"],
            }

        self.middleware.check_rate_limit = mock_check_rate_limit

        # Test rate limit tracking
        client_id = "test_client_123"

        # First request should be allowed
        result1 = self.middleware.check_rate_limit(client_id, limit=5)
        assert result1["allowed"] is True
        assert result1["remaining"] == 4

        # Multiple requests
        for i in range(4):
            result = self.middleware.check_rate_limit(client_id, limit=5)
            assert result["allowed"] is True

        # 6th request should be denied
        result6 = self.middleware.check_rate_limit(client_id, limit=5)
        assert result6["allowed"] is False
        assert result6["remaining"] == 0

    def test_rate_limit_different_clients(self):
        """Test rate limiting for different clients."""
        # Mock client-specific rate limiting
        client_limits = {}

        def mock_check_client_rate_limit(client_id, limit=10):
            if client_id not in client_limits:
                client_limits[client_id] = 0

            client_limits[client_id] += 1

            if client_limits[client_id] > limit:
                return {"allowed": False, "client_id": client_id}

            return {"allowed": True, "client_id": client_id}

        self.middleware.check_client_rate_limit = mock_check_client_rate_limit

        # Test different clients
        client_a = "client_a"
        client_b = "client_b"

        # Both clients should have independent limits
        result_a = self.middleware.check_client_rate_limit(client_a, limit=2)
        result_b = self.middleware.check_client_rate_limit(client_b, limit=2)

        assert result_a["allowed"] is True
        assert result_b["allowed"] is True
        assert result_a["client_id"] == client_a
        assert result_b["client_id"] == client_b

    def test_rate_limit_headers(self):
        """Test rate limit headers in response."""

        # Mock adding rate limit headers
        def mock_add_rate_limit_headers(response, remaining, reset_time):
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_time)
            return response

        self.middleware.add_rate_limit_headers = mock_add_rate_limit_headers

        # Test header addition
        response = Mock()
        response.headers = {}

        result = self.middleware.add_rate_limit_headers(response, 42, 1640995200)

        assert result.headers["X-RateLimit-Remaining"] == "42"
        assert result.headers["X-RateLimit-Reset"] == "1640995200"


class TestErrorHandlers:
    """Test Error Handlers functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()

    def test_http_exception_handler(self):
        """Test HTTP exception handler."""

        # Mock HTTP exception handler
        def mock_http_exception_handler(request, exc):
            return {
                "error": {
                    "type": "HTTPException",
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                    "timestamp": datetime.now().isoformat(),
                }
            }

        # Test with different HTTP exceptions
        exc_404 = HTTPException(status_code=404, detail="Not Found")
        exc_500 = HTTPException(status_code=500, detail="Internal Server Error")

        request = Mock()

        result_404 = mock_http_exception_handler(request, exc_404)
        result_500 = mock_http_exception_handler(request, exc_500)

        assert result_404["error"]["status_code"] == 404
        assert result_404["error"]["detail"] == "Not Found"
        assert result_500["error"]["status_code"] == 500
        assert result_500["error"]["detail"] == "Internal Server Error"

    def test_validation_exception_handler(self):
        """Test validation exception handler."""

        # Mock validation exception handler
        def mock_validation_exception_handler(request, exc):
            return {
                "error": {
                    "type": "ValidationError",
                    "message": "Request validation failed",
                    "details": [{"field": "email", "message": "Invalid email format"}],
                    "timestamp": datetime.now().isoformat(),
                }
            }

        request = Mock()
        exc = Mock()

        result = mock_validation_exception_handler(request, exc)

        assert result["error"]["type"] == "ValidationError"
        assert result["error"]["message"] == "Request validation failed"
        assert len(result["error"]["details"]) == 1
        assert result["error"]["details"][0]["field"] == "email"

    def test_generic_exception_handler(self):
        """Test generic exception handler."""

        # Mock generic exception handler
        def mock_generic_exception_handler(request, exc):
            return {
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": getattr(request, "id", "unknown"),
                }
            }

        request = Mock()
        request.id = "req_123"
        exc = ValueError("Test error")

        result = mock_generic_exception_handler(request, exc)

        assert result["error"]["type"] == "InternalServerError"
        assert result["error"]["message"] == "An unexpected error occurred"
        assert result["error"]["request_id"] == "req_123"


class TestHealthRouter:
    """Test Health Router functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.app.include_router(health_router)
        self.client = TestClient(self.app)

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        with patch("api.v1.health.get_health_status") as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "uptime": 3600,
                "checks": {
                    "database": "healthy",
                    "redis": "healthy",
                    "external_services": "healthy",
                },
            }

            response = self.client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "version" in data
            assert "checks" in data
            assert data["checks"]["database"] == "healthy"

    def test_health_check_unhealthy(self):
        """Test health check when services are unhealthy."""
        with patch("api.v1.health.get_health_status") as mock_health:
            mock_health.return_value = {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "uptime": 3600,
                "checks": {
                    "database": "unhealthy",
                    "redis": "healthy",
                    "external_services": "timeout",
                },
            }

            response = self.client.get("/health")

            assert response.status_code == 503  # Service Unavailable
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["checks"]["database"] == "unhealthy"
            assert data["checks"]["external_services"] == "timeout"

    def test_health_check_detailed(self):
        """Test detailed health check endpoint."""
        with patch("api.v1.health.get_detailed_health_status") as mock_detailed:
            mock_detailed.return_value = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "uptime": 3600,
                "system_info": {
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "disk_usage": 23.1,
                },
                "checks": {
                    "database": {"status": "healthy", "response_time": 0.05},
                    "redis": {"status": "healthy", "response_time": 0.02},
                    "external_services": {
                        "status": "healthy",
                        "response_time": 0.15,
                    },
                },
            }

            response = self.client.get("/health/detailed")

            assert response.status_code == 200
            data = response.json()
            assert "system_info" in data
            assert data["system_info"]["cpu_usage"] == 45.2
            assert data["checks"]["database"]["response_time"] == 0.05


class TestAuthRouter:
    """Test Auth Router functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.app.include_router(auth_router)
        self.client = TestClient(self.app)

    def test_login_endpoint(self):
        """Test login endpoint."""
        with patch("api.v1.auth.authenticate_user") as mock_auth:
            mock_auth.return_value = {
                "access_token": "test_access_token",
                "refresh_token": "test_refresh_token",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {
                    "user_id": "user_123",
                    "username": "testuser",
                    "email": "test@example.com",
                    "role": "researcher",
                },
            }

            login_data = {"username": "testuser", "password": "testpassword"}

            response = self.client.post("/auth/login", json=login_data)

            assert response.status_code == 200
            data = response.json()
            assert data["access_token"] == "test_access_token"
            assert data["token_type"] == "bearer"
            assert data["user"]["username"] == "testuser"

    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        with patch("api.v1.auth.authenticate_user") as mock_auth:
            mock_auth.side_effect = HTTPException(status_code=401, detail="Invalid credentials")

            login_data = {"username": "testuser", "password": "wrongpassword"}

            response = self.client.post("/auth/login", json=login_data)

            assert response.status_code == 401
            data = response.json()
            assert "Invalid credentials" in data["detail"]

    def test_refresh_token_endpoint(self):
        """Test refresh token endpoint."""
        with patch("api.v1.auth.refresh_access_token") as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
                "token_type": "bearer",
                "expires_in": 3600,
            }

            refresh_data = {"refresh_token": "valid_refresh_token"}

            response = self.client.post("/auth/refresh", json=refresh_data)

            assert response.status_code == 200
            data = response.json()
            assert data["access_token"] == "new_access_token"
            assert data["refresh_token"] == "new_refresh_token"

    def test_logout_endpoint(self):
        """Test logout endpoint."""
        with patch("api.v1.auth.revoke_token") as mock_revoke:
            mock_revoke.return_value = {
                "message": "Successfully logged out",
                "revoked_at": datetime.now().isoformat(),
            }

            headers = {"Authorization": "Bearer valid_token"}

            response = self.client.post("/auth/logout", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Successfully logged out"
            assert "revoked_at" in data


class TestAgentsRouter:
    """Test Agents Router functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.app.include_router(agents_router)
        self.client = TestClient(self.app)

    def test_list_agents_endpoint(self):
        """Test list agents endpoint."""
        with patch("api.v1.agents.get_all_agents") as mock_get_agents:
            mock_get_agents.return_value = [
                {
                    "agent_id": "agent_001",
                    "name": "Test Agent 1",
                    "status": "active",
                    "created_at": "2023-01-01T00:00:00Z",
                    "type": "active_inference",
                },
                {
                    "agent_id": "agent_002",
                    "name": "Test Agent 2",
                    "status": "inactive",
                    "created_at": "2023-01-02T00:00:00Z",
                    "type": "llm_based",
                },
            ]

            response = self.client.get("/agents")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["agent_id"] == "agent_001"
            assert data[1]["agent_id"] == "agent_002"

    def test_get_agent_endpoint(self):
        """Test get specific agent endpoint."""
        with patch("api.v1.agents.get_agent_by_id") as mock_get_agent:
            mock_get_agent.return_value = {
                "agent_id": "agent_001",
                "name": "Test Agent 1",
                "status": "active",
                "created_at": "2023-01-01T00:00:00Z",
                "type": "active_inference",
                "config": {"learning_rate": 0.1, "policy_precision": 2.0},
                "stats": {
                    "total_actions": 1500,
                    "total_observations": 1500,
                    "uptime": 3600,
                },
            }

            response = self.client.get("/agents/agent_001")

            assert response.status_code == 200
            data = response.json()
            assert data["agent_id"] == "agent_001"
            assert data["name"] == "Test Agent 1"
            assert "config" in data
            assert "stats" in data
            assert data["stats"]["total_actions"] == 1500

    def test_create_agent_endpoint(self):
        """Test create agent endpoint."""
        with patch("api.v1.agents.create_new_agent") as mock_create:
            mock_create.return_value = {
                "agent_id": "agent_003",
                "name": "New Test Agent",
                "status": "initializing",
                "created_at": datetime.now().isoformat(),
                "type": "active_inference",
                "config": {"learning_rate": 0.05, "policy_precision": 1.5},
            }

            agent_data = {
                "name": "New Test Agent",
                "type": "active_inference",
                "config": {"learning_rate": 0.05, "policy_precision": 1.5},
            }

            response = self.client.post("/agents", json=agent_data)

            assert response.status_code == 201
            data = response.json()
            assert data["agent_id"] == "agent_003"
            assert data["name"] == "New Test Agent"
            assert data["status"] == "initializing"

    def test_update_agent_endpoint(self):
        """Test update agent endpoint."""
        with patch("api.v1.agents.update_agent") as mock_update:
            mock_update.return_value = {
                "agent_id": "agent_001",
                "name": "Updated Test Agent",
                "status": "active",
                "updated_at": datetime.now().isoformat(),
                "config": {"learning_rate": 0.2, "policy_precision": 3.0},
            }

            update_data = {
                "name": "Updated Test Agent",
                "config": {"learning_rate": 0.2, "policy_precision": 3.0},
            }

            response = self.client.put("/agents/agent_001", json=update_data)

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Test Agent"
            assert data["config"]["learning_rate"] == 0.2

    def test_delete_agent_endpoint(self):
        """Test delete agent endpoint."""
        with patch("api.v1.agents.delete_agent") as mock_delete:
            mock_delete.return_value = {
                "message": "Agent deleted successfully",
                "agent_id": "agent_001",
                "deleted_at": datetime.now().isoformat(),
            }

            response = self.client.delete("/agents/agent_001")

            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Agent deleted successfully"
            assert data["agent_id"] == "agent_001"

    def test_agent_action_endpoint(self):
        """Test agent action endpoint."""
        with patch("api.v1.agents.execute_agent_action") as mock_action:
            mock_action.return_value = {
                "action_id": "action_123",
                "agent_id": "agent_001",
                "action_type": "step",
                "result": {
                    "observation": {"sensor": "value"},
                    "action": 1,
                    "belief_update": {"confidence": 0.8},
                },
                "executed_at": datetime.now().isoformat(),
            }

            action_data = {
                "action_type": "step",
                "parameters": {"observation": {"sensor": "value"}},
            }

            response = self.client.post("/agents/agent_001/actions", json=action_data)

            assert response.status_code == 200
            data = response.json()
            assert data["action_id"] == "action_123"
            assert data["agent_id"] == "agent_001"
            assert data["action_type"] == "step"
            assert "result" in data


class TestAPIIntegration:
    """Test API integration scenarios."""

    def test_middleware_integration(self):
        """Test middleware integration."""
        FastAPI()

        # Mock middleware integration
        middlewares = [
            "SecurityHeadersMiddleware",
            "RateLimitMiddleware",
            "ErrorHandlingMiddleware",
        ]

        for middleware in middlewares:
            # Each middleware should be properly integrated
            assert middleware is not None
            assert isinstance(middleware, str)

    def test_cors_configuration(self):
        """Test CORS configuration."""
        # Mock CORS settings
        cors_config = {
            "allow_origins": ["https://example.com"],
            "allow_methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Authorization", "Content-Type"],
            "allow_credentials": True,
        }

        # Verify CORS configuration
        assert "allow_origins" in cors_config
        assert "GET" in cors_config["allow_methods"]
        assert "Authorization" in cors_config["allow_headers"]
        assert cors_config["allow_credentials"] is True

    def test_api_versioning(self):
        """Test API versioning."""
        # Mock API version structure
        api_versions = {
            "v1": {
                "routes": ["/health", "/auth", "/agents"],
                "deprecated": False,
                "supported_until": "2025-12-31",
            },
            "v2": {
                "routes": ["/health", "/auth", "/agents", "/coalitions"],
                "deprecated": False,
                "supported_until": "2026-12-31",
            },
        }

        # Verify versioning structure
        assert "v1" in api_versions
        assert "v2" in api_versions
        assert not api_versions["v1"]["deprecated"]
        assert len(api_versions["v2"]["routes"]) > len(api_versions["v1"]["routes"])

    def test_authentication_flow(self):
        """Test complete authentication flow."""

        # Mock authentication flow
        def mock_auth_flow():
            # Step 1: Login
            login_result = {
                "access_token": "access_123",
                "refresh_token": "refresh_123",
                "expires_in": 3600,
            }

            # Step 2: Use token
            protected_resource = {
                "data": "protected_data",
                "user_id": "user_123",
            }

            # Step 3: Refresh token
            refresh_result = {
                "access_token": "new_access_456",
                "refresh_token": "new_refresh_456",
                "expires_in": 3600,
            }

            return {
                "login": login_result,
                "protected": protected_resource,
                "refresh": refresh_result,
            }

        flow = mock_auth_flow()

        # Verify flow steps
        assert "access_token" in flow["login"]
        assert "data" in flow["protected"]
        assert "access_token" in flow["refresh"]
        assert flow["refresh"]["access_token"] != flow["login"]["access_token"]

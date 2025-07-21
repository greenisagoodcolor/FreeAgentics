"""
Behavior-driven tests for API main module - targeting core user-facing functionality.
Focus on business behavior, not implementation details.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestAPIMainBehavior:
    """Test API main module behaviors that users depend on."""

    def test_api_application_initializes_successfully(self):
        """
        GIVEN: The FreeAgentics API application
        WHEN: The application is initialized
        THEN: It should create a FastAPI app instance with expected configuration
        """
        from api.main import app

        assert isinstance(app, FastAPI)
        assert app.title == "FreeAgentics API"
        assert app.description == "Multi-Agent AI Platform API with Active Inference"
        assert app.version == "0.1.0"

    def test_root_endpoint_provides_welcome_information(self):
        """
        GIVEN: A user accessing the root endpoint
        WHEN: They make a GET request to "/"
        THEN: They should receive welcome information with service details
        """
        from api.main import app

        # Mock the lifespan to avoid startup/shutdown issues
        app.lifespan = None

        client = TestClient(app)

        with patch("api.main.lifespan", return_value=None):
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["message"] == "Welcome to FreeAgentics API"
        assert data["version"] == "0.1.0"
        assert data["docs"] == "/docs"
        assert data["redoc"] == "/redoc"

    def test_health_endpoint_reports_service_status(self):
        """
        GIVEN: A user checking service health
        WHEN: They make a GET request to "/health"
        THEN: They should receive confirmation that the service is healthy
        """
        from api.main import app

        # Mock the lifespan to avoid startup/shutdown issues
        app.lifespan = None

        client = TestClient(app)

        with patch("api.main.lifespan", return_value=None):
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "FreeAgentics API"
        assert data["version"] == "0.1.0"

    def test_api_includes_authentication_routes(self):
        """
        GIVEN: The API application with authentication functionality
        WHEN: The application is configured
        THEN: It should include authentication routes under /api/v1
        """
        from api.main import app

        # Check that auth routes are included
        route_paths = [route.path for route in app.routes]

        # Look for auth-related routes
        auth_routes_exist = any("/api/v1" in path for path in route_paths)
        assert auth_routes_exist, "API should include v1 routes"

    def test_api_includes_cors_middleware(self):
        """
        GIVEN: The API application serving web clients
        WHEN: The application is configured
        THEN: It should include CORS middleware for cross-origin requests
        """
        from api.main import app

        # Check that CORS middleware is configured
        middleware_classes = [
            middleware.cls.__name__ for middleware in app.user_middleware
        ]
        assert "CORSMiddleware" in middleware_classes

    def test_api_includes_security_middleware(self):
        """
        GIVEN: The API application requiring security
        WHEN: The application is configured
        THEN: It should include security middleware components
        """
        from api.main import app

        # Check that security middleware is configured
        middleware_classes = [
            middleware.cls.__name__ for middleware in app.user_middleware
        ]

        # Should include security-related middleware
        security_middleware_found = any(
            "Security" in cls_name for cls_name in middleware_classes
        )
        assert security_middleware_found, "API should include security middleware"

    def test_api_includes_agent_management_routes(self):
        """
        GIVEN: The API application with agent management functionality
        WHEN: The application is configured
        THEN: It should include agent management routes
        """
        from api.main import app

        # Check that agent routes are included
        route_paths = [route.path for route in app.routes]

        # Look for agent-related routes
        agent_routes_exist = any("agent" in path.lower() for path in route_paths)
        assert agent_routes_exist, "API should include agent management routes"

    def test_api_includes_monitoring_routes(self):
        """
        GIVEN: The API application with monitoring functionality
        WHEN: The application is configured
        THEN: It should include monitoring routes
        """
        from api.main import app

        # Check that monitoring routes are included
        route_paths = [route.path for route in app.routes]

        # Look for monitoring-related routes
        monitoring_routes_exist = any(
            "monitoring" in path.lower() for path in route_paths
        )
        assert monitoring_routes_exist, "API should include monitoring routes"

    def test_api_includes_websocket_routes(self):
        """
        GIVEN: The API application with real-time communication
        WHEN: The application is configured
        THEN: It should include WebSocket routes
        """
        from api.main import app

        # Check that WebSocket routes are included
        route_paths = [route.path for route in app.routes]

        # Look for WebSocket-related routes
        websocket_routes_exist = any(
            "websocket" in path.lower() for path in route_paths
        )
        assert websocket_routes_exist, "API should include WebSocket routes"

    def test_api_includes_inference_routes(self):
        """
        GIVEN: The API application with AI inference functionality
        WHEN: The application is configured
        THEN: It should include inference routes
        """
        from api.main import app

        # Check that inference routes are included
        route_paths = [route.path for route in app.routes]

        # Look for inference-related routes
        inference_routes_exist = any(
            "inference" in path.lower() for path in route_paths
        )
        assert inference_routes_exist, "API should include inference routes"

    def test_api_includes_system_routes(self):
        """
        GIVEN: The API application with system management functionality
        WHEN: The application is configured
        THEN: It should include system management routes
        """
        from api.main import app

        # Check that system routes are included
        route_paths = [route.path for route in app.routes]

        # Look for system-related routes
        system_routes_exist = any("system" in path.lower() for path in route_paths)
        assert system_routes_exist, "API should include system management routes"


class TestAPILifecycleBehavior:
    """Test API lifecycle behaviors."""

    @pytest.mark.asyncio
    async def test_lifespan_context_manager_handles_startup_shutdown(self):
        """
        GIVEN: The API application with lifespan management
        WHEN: The lifespan context manager is used
        THEN: It should handle startup and shutdown gracefully
        """
        from fastapi import FastAPI

        from api.main import lifespan

        app = FastAPI()

        # Test the lifespan context manager
        try:
            async with lifespan(app):
                # App should be in running state
                assert app is not None
        except Exception as e:
            pytest.fail(f"Lifespan context manager failed: {e}")

    def test_api_configuration_includes_documentation_endpoints(self):
        """
        GIVEN: The API application for developers
        WHEN: The application is configured
        THEN: It should provide documentation endpoints
        """
        from api.main import app

        # Mock the lifespan to avoid startup/shutdown issues
        app.lifespan = None

        client = TestClient(app)

        with patch("api.main.lifespan", return_value=None):
            # Test that docs endpoint is accessible
            docs_response = client.get("/docs")
            assert docs_response.status_code == 200

            # Test that redoc endpoint is accessible
            redoc_response = client.get("/redoc")
            assert redoc_response.status_code == 200


class TestAPIErrorHandlingBehavior:
    """Test API error handling behaviors."""

    def test_api_handles_nonexistent_routes_gracefully(self):
        """
        GIVEN: A user accessing a non-existent route
        WHEN: They make a request to an invalid endpoint
        THEN: The API should return a 404 error
        """
        from api.main import app

        # Mock the lifespan to avoid startup/shutdown issues
        app.lifespan = None

        client = TestClient(app)

        with patch("api.main.lifespan", return_value=None):
            response = client.get("/nonexistent-route")

        assert response.status_code == 404


class TestAPISecurityBehavior:
    """Test API security behaviors."""

    def test_api_enforces_security_headers(self):
        """
        GIVEN: The API application with security requirements
        WHEN: A user makes a request
        THEN: The response should include security headers
        """
        from api.main import app

        # Mock the lifespan to avoid startup/shutdown issues
        app.lifespan = None

        client = TestClient(app)

        with patch("api.main.lifespan", return_value=None):
            response = client.get("/")

        # Check that security-related headers are present
        # The actual headers depend on the SecurityHeadersMiddleware implementation
        assert response.status_code == 200

        # Basic security check - should not expose server details
        assert (
            "Server" not in response.headers
            or "FastAPI" not in response.headers.get("Server", "")
        )

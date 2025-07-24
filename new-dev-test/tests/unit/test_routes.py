"""Test FastAPI route registration following TDD principles.

This test file MUST fail first (RED phase), then we implement the route
registration to make it pass (GREEN phase), then refactor if needed.
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Set required environment variables for testing
test_env = {
    "DATABASE_URL": "sqlite:///test_routes.db",
    "API_KEY": "test-api-key",
    "SECRET_KEY": "test-secret-key-32-characters-minimum-required-for-security",
    "JWT_SECRET": "test-jwt-secret-32-characters-minimum-required-for-security",
    "ENVIRONMENT": "testing",
}

# Mock environment before importing the app
with patch.dict(os.environ, test_env, clear=False):
    # These imports will be available after implementation
    try:
        from api.main import app

        from database.session import get_db_context
    except ImportError:
        # Temporary to make tests syntactically valid
        from fastapi import FastAPI

        app = FastAPI()

        def get_db_context():
            yield None


class TestRouteRegistration:
    """Test suite for FastAPI route registration.

    Following TDD RED-GREEN-REFACTOR cycle:
    1. These tests will fail initially (RED)
    2. We implement minimal code to pass (GREEN)
    3. We refactor if needed (REFACTOR)
    """

    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            # Try with standard arguments
            return TestClient(app)
        except TypeError:
            # Fallback for compatibility issues
            return TestClient(app=app)

    def test_app_exists(self, client):
        """Test that the FastAPI app instance exists."""
        assert app is not None
        assert hasattr(app, "routes")

    def test_root_endpoint_exists(self, client):
        """Test that root endpoint is registered and returns expected response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert data["docs"] == "/docs"

    def test_health_endpoint_exists(self, client):
        """Test that health endpoint is registered."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "message" in data

    def test_agents_routes_registered(self, client):
        """Test that agent routes are properly registered with /api/v1 prefix."""
        # List agents endpoint - should not return 404 (route not found)
        response = client.get("/api/v1/agents")
        assert response.status_code != 404, f"Agents list route not found: {response.status_code}"
        assert response.status_code in [
            200,
            401,
            503,
        ]  # OK, auth required, or service unavailable

        # Agent creation endpoint - should not return 404 (route not found)
        response = client.post("/api/v1/agents", json={"name": "test"})
        assert (
            response.status_code != 404
        ), f"Agent creation route not found: {response.status_code}"
        assert response.status_code in [
            201,
            400,
            401,
            422,
            503,
        ]  # Created, bad request, auth, validation, or service unavailable

        # Single agent endpoint (with non-existent ID) - should not return 404 for route registration
        response = client.get("/api/v1/agents/nonexistent-id")
        assert (
            response.status_code != 405
        ), f"Agent detail route has wrong method: {response.status_code}"
        # 404 is OK here because the agent doesn't exist, but not because route isn't registered
        assert response.status_code in [
            404,
            401,
            503,
        ]  # Not found, auth required, or service unavailable

    def test_agent_converse_endpoint_exists(self, client):
        """Test that the agent converse endpoint is registered."""
        # This is the critical endpoint mentioned in the validation target
        response = client.post("/api/v1/agents/test-id/converse", json={"prompt": "test"})
        # Should NOT return 404 (route not found)
        # Valid responses: 401 (auth), 422 (validation), 400 (bad request), 403 (forbidden), 503 (service unavailable)
        assert response.status_code != 404, f"Converse route not found! Got {response.status_code}"
        assert response.status_code in [400, 401, 403, 422, 503]

    def test_gmn_routes_registered(self, client):
        """Test that GMN routes are registered."""
        response = client.get("/api/v1/gmn/examples")
        assert response.status_code != 404, f"GMN examples route not found: {response.status_code}"
        assert response.status_code in [
            200,
            401,
            503,
        ]  # OK, auth required, or service unavailable

    def test_websocket_routes_registered(self, client):
        """Test that websocket routes are registered."""
        # WebSocket routes typically don't respond to regular HTTP requests
        # but the route should be registered. Test the websocket info endpoint instead
        response = client.get("/api/v1/ws/connections")
        assert (
            response.status_code != 404
        ), f"WebSocket connections route not found: {response.status_code}"
        assert response.status_code in [
            200,
            401,
            503,
        ]  # OK, auth required, or service unavailable

    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

        # Check that our key routes are in the schema
        paths = schema["paths"]
        assert "/health" in paths
        assert "/" in paths

    def test_swagger_docs_available(self, client):
        """Test that Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_redoc_available(self, client):
        """Test that ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()

    def test_database_dependency_injection(self, client):
        """Test that database dependency injection is configured."""
        # This tests that the get_db dependency is properly registered
        # We'll make a request that requires DB and check it doesn't fail with DI error
        response = client.get("/api/v1/agents")
        # Should not get 500 Internal Server Error from missing dependency injection
        # Service might be unavailable (503) but shouldn't be a server error from bad DI
        assert response.status_code != 500, f"Dependency injection failed: {response.status_code}"
        assert response.status_code in [
            200,
            401,
            503,
        ]  # OK, auth required, or service unavailable

    def test_no_catch_all_error_handlers(self, client):
        """Test that exceptions propagate (no catch-all handlers)."""
        # Make a request that would cause an error
        # If there's a catch-all handler, we'd get a generic 500 response
        # Without it, we should get specific error details
        response = client.post("/api/v1/agents/invalid-id/converse", json={})
        # Should get specific error codes, not generic 500s
        assert response.status_code in [
            400,
            401,
            403,
            422,
            503,
        ], f"Unexpected error code: {response.status_code}"

        if response.status_code in [
            400,
            422,
        ]:  # Bad request or validation error
            # Check that we get detailed error info, not a generic message
            data = response.json()
            assert "detail" in data
            assert not any(
                generic in str(data.get("detail", "")).lower()
                for generic in [
                    "internal server error",
                    "something went wrong",
                ]
            )

    def test_route_naming_convention(self, client):
        """Test that routes follow consistent naming conventions."""
        response = client.get("/openapi.json")
        schema = response.json()
        paths = schema["paths"].keys()

        # All API routes should start with /api/v1 (except docs, health, and metrics)
        api_paths = [
            p
            for p in paths
            if p
            not in [
                "/",
                "/health",
                "/health/detailed",
                "/docs",
                "/redoc",
                "/openapi.json",
                "/metrics",
            ]
        ]
        for path in api_paths:
            if not path.startswith("/graphql"):  # GraphQL is an exception
                assert path.startswith("/api/v1"), f"Route {path} doesn't follow /api/v1 convention"

        # Check pluralization consistency (agents, not agent)
        agent_paths = [p for p in paths if "agent" in p and "/api/v1" in p]
        for path in agent_paths:
            if "/api/v1/agent/" in path:  # Single resource with trailing slash
                pytest.fail(f"Route {path} uses singular 'agent' instead of 'agents'")

    def test_all_routers_imported(self):
        """Test that all expected routers are imported in main.py."""
        # This test verifies the router imports are working
        expected_routers = [
            "agents",
            "gmn",
            "inference",
            "system",
            "websocket",
            "monitoring",
            "security",
            "auth",
        ]

        # Check OpenAPI schema for evidence of these routers
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        paths = list(schema["paths"].keys())

        for router in expected_routers:
            # Each router should contribute at least one path
            router_paths = [p for p in paths if f"/api/v1/{router}" in p or f"/{router}" in p]
            assert len(router_paths) > 0, f"No routes found for {router} router"

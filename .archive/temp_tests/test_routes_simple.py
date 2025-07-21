"""Simple route registration test following TDD principles.

This test file MUST fail first (RED phase), then we implement the route
registration to make it pass (GREEN phase), then refactor if needed.
"""

import os


# Set required environment variables for testing
test_env = {
    "DATABASE_URL": "sqlite:///test_routes.db",
    "API_KEY": "test-api-key",
    "SECRET_KEY": "test-secret-key-32-characters-minimum-required-for-security",
    "JWT_SECRET": "test-jwt-secret-32-characters-minimum-required-for-security",
    "REDIS_URL": "redis://localhost:6379/0",
    "ENVIRONMENT": "testing",
}

# Mock environment before importing anything
os.environ.update(test_env)

# Import FastAPI and test client
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Try to import the app
try:
    from api.main_minimal import app
except ImportError:
    # If minimal doesn't exist, create a placeholder
    app = FastAPI()

# Create test client
client = TestClient(app)


def test_app_exists():
    """Test that the FastAPI app instance exists."""
    assert app is not None
    assert hasattr(app, "routes")
    print("✓ App exists")


def test_root_endpoint_exists():
    """Test that root endpoint is registered and returns expected response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    assert data["docs"] == "/docs"
    print("✓ Root endpoint works")


def test_health_endpoint_exists():
    """Test that health endpoint is registered."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    print("✓ Health endpoint works")


def test_agents_routes_registered():
    """Test that agent routes are properly registered with /api/v1 prefix."""
    # List agents endpoint
    response = client.get("/api/v1/agents")
    assert response.status_code != 404, f"Agents list route not found: {response.status_code}"
    print(f"  - GET /api/v1/agents: {response.status_code}")

    # Agent creation endpoint
    response = client.post("/api/v1/agents", json={"name": "test"})
    assert response.status_code != 404, f"Agent creation route not found: {response.status_code}"
    print(f"  - POST /api/v1/agents: {response.status_code}")

    # Single agent endpoint
    response = client.get("/api/v1/agents/test-id")
    assert (
        response.status_code != 405
    ), f"Agent detail route has wrong method: {response.status_code}"
    print(f"  - GET /api/v1/agents/test-id: {response.status_code}")

    print("✓ Agent routes registered")


def test_agent_converse_endpoint_exists():
    """Test that the agent converse endpoint is registered."""
    response = client.post("/api/v1/agents/test-id/converse", json={"prompt": "test"})
    assert response.status_code != 404, f"Converse route not found! Got {response.status_code}"
    print(f"✓ Converse endpoint registered: {response.status_code}")


def test_gmn_routes_registered():
    """Test that GMN routes are registered."""
    response = client.get("/api/v1/gmn/examples")
    assert response.status_code != 404, f"GMN examples route not found: {response.status_code}"
    print(f"✓ GMN routes registered: {response.status_code}")


def test_openapi_schema_available():
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
    print("✓ OpenAPI schema available")


def test_route_naming_convention():
    """Test that routes follow consistent naming conventions."""
    response = client.get("/openapi.json")
    schema = response.json()
    paths = schema["paths"].keys()

    # Check API routes follow /api/v1 convention
    api_paths = [p for p in paths if p not in ["/", "/health", "/docs", "/redoc", "/openapi.json"]]
    for path in api_paths:
        if not path.startswith("/graphql"):
            assert path.startswith("/api/v1"), f"Route {path} doesn't follow /api/v1 convention"

    print("✓ Routes follow naming conventions")


if __name__ == "__main__":
    print("\n=== Running TDD Route Registration Tests (RED Phase) ===\n")

    tests = [
        test_app_exists,
        test_root_endpoint_exists,
        test_health_endpoint_exists,
        test_agents_routes_registered,
        test_agent_converse_endpoint_exists,
        test_gmn_routes_registered,
        test_openapi_schema_available,
        test_route_naming_convention,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: Unexpected error: {e}")
            failed += 1

    print(f"\n=== Results: {len(tests) - failed}/{len(tests)} passed ===")
    if failed > 0:
        print("\nThis is expected in the RED phase! Now implement the routes to make tests pass.")
        exit(1)
    else:
        print("\nAll tests passed! Move to REFACTOR phase if needed.")

"""Final route registration test following TDD principles.

This verifies all routes are correctly registered in the minimal app.
"""

import os
import sys

# Set up environment
os.environ.update(
    {
        "DATABASE_URL": "sqlite:///test_routes.db",
        "API_KEY": "test-api-key",
        "SECRET_KEY": "test-secret-key-32-characters-minimum-required-for-security",
        "JWT_SECRET": "test-jwt-secret-32-characters-minimum-required-for-security",
        "REDIS_URL": "redis://localhost:6379/0",
        "ENVIRONMENT": "testing",
    }
)

import httpx

# Import starlette's test client with proper httpx backend
from starlette.testclient import TestClient as StarletteTestClient


# Create a properly configured TestClient
class TestClient(StarletteTestClient):
    def __init__(self, app, **kwargs):
        # Pass app as first positional argument to httpx.Client
        super().__init__(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver", **kwargs
        )


# Import the app
from api.main_minimal import app

# Create test client
client = TestClient(app)


def test_routes():
    """Test all routes are registered correctly."""
    print("\n=== Running TDD Route Registration Tests (GREEN Phase) ===\n")

    passed = 0
    total = 0

    # Test 1: Root endpoint
    total += 1
    try:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert data["docs"] == "/docs"
        print("✓ Root endpoint works")
        passed += 1
    except AssertionError as e:
        print(f"✗ Root endpoint: {e}")

    # Test 2: Health endpoint
    total += 1
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print("✓ Health endpoint works")
        passed += 1
    except AssertionError as e:
        print(f"✗ Health endpoint: {e}")

    # Test 3: Agent routes
    total += 1
    try:
        # List agents
        response = client.get("/api/v1/agents")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Create agent
        response = client.post("/api/v1/agents", json={"name": "test"})
        assert response.status_code == 201, f"Expected 201, got {response.status_code}"

        # Get agent
        response = client.get("/api/v1/agents/test-id")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        print("✓ Agent routes registered and working")
        passed += 1
    except AssertionError as e:
        print(f"✗ Agent routes: {e}")

    # Test 4: Converse endpoint
    total += 1
    try:
        response = client.post("/api/v1/agents/test-id/converse", json={"prompt": "test"})
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "response" in data
        assert "agent_id" in data
        print("✓ Converse endpoint registered and working")
        passed += 1
    except AssertionError as e:
        print(f"✗ Converse endpoint: {e}")

    # Test 5: GMN routes
    total += 1
    try:
        response = client.get("/api/v1/gmn/examples")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print("✓ GMN routes registered and working")
        passed += 1
    except AssertionError as e:
        print(f"✗ GMN routes: {e}")

    # Test 6: WebSocket info endpoint
    total += 1
    try:
        response = client.get("/api/v1/ws/connections")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print("✓ WebSocket info endpoint registered and working")
        passed += 1
    except AssertionError as e:
        print(f"✗ WebSocket endpoint: {e}")

    # Test 7: OpenAPI schema
    total += 1
    try:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

        # Verify key paths exist
        paths = schema["paths"]
        assert "/" in paths
        assert "/health" in paths
        assert "/api/v1/agents" in paths
        assert "/api/v1/agents/{agent_id}/converse" in paths

        print("✓ OpenAPI schema available with correct paths")
        passed += 1
    except AssertionError as e:
        print(f"✗ OpenAPI schema: {e}")

    # Test 8: Route naming conventions
    total += 1
    try:
        response = client.get("/openapi.json")
        schema = response.json()
        paths = schema["paths"].keys()

        # Check API routes follow /api/v1 convention
        api_paths = [
            p for p in paths if p not in ["/", "/health", "/docs", "/redoc", "/openapi.json"]
        ]
        for path in api_paths:
            if not path.startswith("/graphql"):
                assert path.startswith("/api/v1"), f"Route {path} doesn't follow /api/v1 convention"

        print("✓ Routes follow naming conventions")
        passed += 1
    except AssertionError as e:
        print(f"✗ Route naming: {e}")

    # Test 9: Dependency injection
    total += 1
    try:
        # The get_db dependency should work without errors
        response = client.get("/api/v1/agents")
        assert response.status_code != 500, f"Dependency injection failed: {response.status_code}"
        print("✓ Dependency injection working")
        passed += 1
    except AssertionError as e:
        print(f"✗ Dependency injection: {e}")

    print(f"\n=== Results: {passed}/{total} tests passed ===")

    if passed == total:
        print("\n✅ All tests passed! GREEN phase complete.")
        print("\nNext steps:")
        print("1. Apply these fixes to the main api/main.py")
        print("2. Document route conventions in AGENTLESSONS.md")
        print("3. Consider REFACTOR phase if needed")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(test_routes())

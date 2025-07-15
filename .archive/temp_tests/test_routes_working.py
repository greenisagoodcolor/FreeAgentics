"""Working route registration test following TDD principles."""

import json
import os

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

# Work around the TestClient initialization issue
import httpx
from fastapi.testclient import TestClient

# Import the app and create a direct test approach
from api.main_minimal import app


def make_test_client(app):
    """Create a test client that works with current versions."""
    transport = httpx.ASGITransport(app=app)
    return httpx.Client(transport=transport, base_url="http://testserver")


client = make_test_client(app)


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
    except Exception as e:
        print(f"✗ Root endpoint error: {e}")

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
    except Exception as e:
        print(f"✗ Health endpoint error: {e}")

    # Test 3: Agent routes
    total += 1
    try:
        # List agents
        response = client.get("/api/v1/agents")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Create agent
        response = client.post(
            "/api/v1/agents",
            content=json.dumps({"name": "test"}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 201, f"Expected 201, got {response.status_code}"

        # Get agent
        response = client.get("/api/v1/agents/test-id")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        print("✓ Agent routes registered and working")
        passed += 1
    except AssertionError as e:
        print(f"✗ Agent routes: {e}")
    except Exception as e:
        print(f"✗ Agent routes error: {e}")

    # Test 4: Converse endpoint (critical for VC demo)
    total += 1
    try:
        response = client.post(
            "/api/v1/agents/test-id/converse",
            content=json.dumps({"prompt": "test"}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "response" in data
        assert "agent_id" in data
        print("✓ Converse endpoint registered and working (CRITICAL FOR VC DEMO)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Converse endpoint: {e}")
    except Exception as e:
        print(f"✗ Converse endpoint error: {e}")

    # Test 5: GMN routes
    total += 1
    try:
        response = client.get("/api/v1/gmn/examples")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print("✓ GMN routes registered and working")
        passed += 1
    except AssertionError as e:
        print(f"✗ GMN routes: {e}")
    except Exception as e:
        print(f"✗ GMN routes error: {e}")

    # Test 6: WebSocket info endpoint
    total += 1
    try:
        response = client.get("/api/v1/ws/connections")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print("✓ WebSocket info endpoint registered and working")
        passed += 1
    except AssertionError as e:
        print(f"✗ WebSocket endpoint: {e}")
    except Exception as e:
        print(f"✗ WebSocket endpoint error: {e}")

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
    except Exception as e:
        print(f"✗ OpenAPI schema error: {e}")

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
        violations = []
        for path in api_paths:
            if not path.startswith("/graphql") and not path.startswith("/api/v1"):
                violations.append(path)

        assert len(violations) == 0, f"Routes not following convention: {violations}"

        print("✓ Routes follow naming conventions")
        passed += 1
    except AssertionError as e:
        print(f"✗ Route naming: {e}")
    except Exception as e:
        print(f"✗ Route naming error: {e}")

    # Test 9: No catch-all error handlers (exceptions should propagate)
    total += 1
    try:
        # Make a request with invalid data that should cause a validation error
        response = client.post(
            "/api/v1/agents",
            content=json.dumps({}),  # Missing required 'name' field
            headers={"Content-Type": "application/json"},
        )
        # Should get 422 Unprocessable Entity for validation error
        assert (
            response.status_code == 422
        ), f"Expected 422 validation error, got {response.status_code}"
        data = response.json()
        assert "detail" in data
        # Check it's a specific validation error, not generic
        assert isinstance(data["detail"], list) or "validation" in str(data["detail"]).lower()
        print("✓ No catch-all error handlers - exceptions propagate correctly")
        passed += 1
    except AssertionError as e:
        print(f"✗ Error handling: {e}")
    except Exception as e:
        print(f"✗ Error handling error: {e}")

    print(f"\n=== Results: {passed}/{total} tests passed ===")

    if passed == total:
        print("\n✅ All tests passed! GREEN phase complete.")
        print("\nNext steps:")
        print("1. Apply these route registration patterns to api/main.py")
        print("2. Document route conventions in AGENTLESSONS.md")
        print("3. Consider REFACTOR phase for cleaner organization")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed.")
        return False


if __name__ == "__main__":
    success = test_routes()

    # Clean up
    client.close()

    exit(0 if success else 1)

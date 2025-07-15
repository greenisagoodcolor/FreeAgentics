"""Direct route testing approach using requests."""

import os
import threading
import time
from contextlib import contextmanager

import requests
import uvicorn

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

# Import the app
try:
    from api.main_minimal import app
except ImportError:
    print("✗ Failed to import api.main_minimal")
    exit(1)


@contextmanager
def run_server():
    """Run the FastAPI server in a separate thread."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error"))
    thread = threading.Thread(target=server.run)
    thread.start()
    time.sleep(1)  # Give server time to start
    yield
    server.should_exit = True
    thread.join()


def test_routes():
    """Test all routes."""
    base_url = "http://127.0.0.1:8000"

    print("\n=== Running TDD Route Registration Tests ===\n")

    # Test root endpoint
    try:
        r = requests.get(f"{base_url}/")
        assert r.status_code == 200
        data = r.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        print("✓ Root endpoint works")
    except Exception as e:
        print(f"✗ Root endpoint: {e}")

    # Test health endpoint
    try:
        r = requests.get(f"{base_url}/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        print("✓ Health endpoint works")
    except Exception as e:
        print(f"✗ Health endpoint: {e}")

    # Test agent routes
    try:
        # List agents
        r = requests.get(f"{base_url}/api/v1/agents")
        assert r.status_code != 404, f"Agents list route not found: {r.status_code}"
        print(f"  - GET /api/v1/agents: {r.status_code}")

        # Create agent
        r = requests.post(f"{base_url}/api/v1/agents", json={"name": "test"})
        assert r.status_code != 404, f"Agent creation route not found: {r.status_code}"
        print(f"  - POST /api/v1/agents: {r.status_code}")

        # Get agent
        r = requests.get(f"{base_url}/api/v1/agents/test-id")
        assert r.status_code != 405, f"Agent detail route has wrong method: {r.status_code}"
        print(f"  - GET /api/v1/agents/test-id: {r.status_code}")

        print("✓ Agent routes registered")
    except Exception as e:
        print(f"✗ Agent routes: {e}")

    # Test converse endpoint
    try:
        r = requests.post(f"{base_url}/api/v1/agents/test-id/converse", json={"prompt": "test"})
        assert r.status_code != 404, f"Converse route not found! Got {r.status_code}"
        print(f"✓ Converse endpoint registered: {r.status_code}")
    except Exception as e:
        print(f"✗ Converse endpoint: {e}")

    # Test GMN routes
    try:
        r = requests.get(f"{base_url}/api/v1/gmn/examples")
        assert r.status_code != 404, f"GMN examples route not found: {r.status_code}"
        print(f"✓ GMN routes registered: {r.status_code}")
    except Exception as e:
        print(f"✗ GMN routes: {e}")

    # Test OpenAPI schema
    try:
        r = requests.get(f"{base_url}/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "openapi" in schema
        assert "paths" in schema
        print("✓ OpenAPI schema available")

        # Check route naming conventions
        paths = schema["paths"].keys()
        api_paths = [
            p for p in paths if p not in ["/", "/health", "/docs", "/redoc", "/openapi.json"]
        ]
        for path in api_paths:
            if not path.startswith("/graphql"):
                assert path.startswith("/api/v1"), f"Route {path} doesn't follow /api/v1 convention"
        print("✓ Routes follow naming conventions")
    except Exception as e:
        print(f"✗ OpenAPI/naming: {e}")


if __name__ == "__main__":
    with run_server():
        test_routes()

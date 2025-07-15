#!/usr/bin/env python3
"""
Test different TestClient initialization patterns to find the correct one.
"""

import os
import sys

# Set minimal environment variables
os.environ.update(
    {
        "DATABASE_URL": "sqlite:///test.db",
        "API_KEY": "test_api_key_for_testing_only_32_characters_minimum",
        "SECRET_KEY": "test_secret_key_for_testing_only_32_characters_minimum",
        "JWT_SECRET_KEY": "test_jwt_secret_key_for_testing_only_32_characters_minimum",
        "REDIS_URL": "redis://localhost:6379/0",
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_password",
        "POSTGRES_DB": "test_db",
        "TESTING": "true",
    }
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def test_client_patterns():
    """Test different TestClient initialization patterns."""

    from fastapi import FastAPI

    # Create minimal app
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    print("Testing different TestClient patterns...")

    # Pattern 1: Direct initialization (current failing pattern)
    try:
        from fastapi.testclient import TestClient

        client = TestClient(app)
        print("✅ Pattern 1: TestClient(app) - SUCCESS")
    except Exception as e:
        print(f"❌ Pattern 1: TestClient(app) failed: {e}")

    # Pattern 2: With context manager
    try:
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            response = client.get("/test")
            print(f"✅ Pattern 2: with TestClient(app) - SUCCESS: {response.status_code}")
    except Exception as e:
        print(f"❌ Pattern 2: with TestClient(app) failed: {e}")

    # Pattern 3: Using httpx directly
    try:
        import httpx
        from fastapi import testclient

        # Check if there's a different import path
        print("Available attributes in fastapi.testclient:", dir(testclient))
    except Exception as e:
        print(f"❌ Pattern 3: httpx approach failed: {e}")

    # Pattern 4: Check Starlette TestClient directly
    try:
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.get("/test")
        print(f"✅ Pattern 4: Starlette TestClient - SUCCESS: {response.status_code}")
    except Exception as e:
        print(f"❌ Pattern 4: Starlette TestClient failed: {e}")

    # Pattern 5: Check what's actually imported
    try:
        from fastapi.testclient import TestClient

        print(f"TestClient class: {TestClient}")
        print(f"TestClient MRO: {TestClient.__mro__}")
        print(f"TestClient __init__ signature:")
        import inspect

        sig = inspect.signature(TestClient.__init__)
        print(f"  {sig}")
    except Exception as e:
        print(f"❌ Pattern 5: Inspection failed: {e}")


if __name__ == "__main__":
    test_client_patterns()

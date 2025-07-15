#!/usr/bin/env python3
"""
Test workaround for TestClient httpx compatibility issue.
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


def test_workaround():
    """Test a workaround for the TestClient issue."""

    from fastapi import FastAPI

    # Create minimal app
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    try:
        # Method 1: Try using ASGITransport explicitly
        import httpx
        from starlette.testclient import TestClient

        print("Testing ASGITransport workaround...")

        # Create transport explicitly
        transport = httpx.ASGITransport(app=app)

        # Create client with transport
        http_client = httpx.Client(transport=transport, base_url="http://testserver")

        # Test the request
        response = http_client.get("/test")
        print(f"✅ ASGITransport workaround successful: {response.status_code}")
        print(f"Response: {response.json()}")

        http_client.close()
        return True

    except Exception as e:
        print(f"❌ ASGITransport workaround failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        # Method 2: Try using Starlette's TestClient directly (if it works differently)
        from starlette.testclient import TestClient as StarletteTestClient

        print("Testing direct Starlette TestClient...")

        # Override the problematic init
        class FixedTestClient(StarletteTestClient):
            def __init__(self, app, **kwargs):
                # Extract app and use transport instead
                transport = httpx.ASGITransport(app=app)
                # Remove app from kwargs and add transport
                kwargs.pop("app", None)
                kwargs["transport"] = transport
                kwargs.setdefault("base_url", "http://testserver")
                # Call httpx.Client init directly, skipping the broken parent init
                import httpx

                httpx.Client.__init__(self, **kwargs)

        client = FixedTestClient(app)
        response = client.get("/test")
        print(f"✅ Fixed TestClient successful: {response.status_code}")
        client.close()
        return True

    except Exception as e:
        print(f"❌ Fixed TestClient failed: {e}")
        import traceback

        traceback.print_exc()

    return False


if __name__ == "__main__":
    success = test_workaround()
    if success:
        print("\n✅ TestClient workaround successful!")
        sys.exit(0)
    else:
        print("\n❌ TestClient workaround failed!")
        sys.exit(1)

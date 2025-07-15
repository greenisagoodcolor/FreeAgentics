#!/usr/bin/env python3
"""
Test the compatibility TestClient wrapper.
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


def test_compat_client():
    """Test the compatibility TestClient."""

    from fastapi import FastAPI

    from tests.utils.test_client_compat import TestClient

    # Create minimal app
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    @app.post("/echo")
    def echo_endpoint(data: dict):
        return {"echo": data}

    try:
        print("Testing compatibility TestClient...")

        # Test 1: Basic initialization
        client = TestClient(app)
        print("✅ TestClient initialization successful")

        # Test 2: GET request
        response = client.get("/test")
        print(f"✅ GET request successful: {response.status_code}")
        print(f"Response: {response.json()}")

        # Test 3: POST request with JSON
        response = client.post("/echo", json={"test": "data"})
        print(f"✅ POST request successful: {response.status_code}")
        print(f"Response: {response.json()}")

        # Test 4: Context manager
        with TestClient(app) as test_client:
            response = test_client.get("/test")
            print(f"✅ Context manager successful: {response.status_code}")

        client.close()

        print("✅ All compatibility tests passed!")
        return True

    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_compat_client()
    if success:
        print("\n✅ TestClient compatibility wrapper working!")
        sys.exit(0)
    else:
        print("\n❌ TestClient compatibility wrapper failed!")
        sys.exit(1)

#!/usr/bin/env python3
"""
Debug script to identify Client.__init__() errors in test files.
"""

import os
import sys

# Set minimal environment variables to avoid validation errors
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

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def test_testclient_import():
    """Test that TestClient can be imported and initialized properly."""
    try:
        print("Testing TestClient import and initialization...")

        # Test 1: Basic FastAPI TestClient import
        from fastapi.testclient import TestClient

        print("✅ TestClient import successful")

        # Test 2: Simple FastAPI app creation
        from fastapi import FastAPI

        # Create minimal app
        simple_app = FastAPI()

        @simple_app.get("/test")
        def test_endpoint():
            return {"message": "test"}

        # Test 3: TestClient initialization with simple app
        client = TestClient(simple_app)
        print("✅ TestClient initialization with simple app successful")

        # Test 4: Basic request
        response = client.get("/test")
        print(f"✅ Basic request successful: {response.status_code}")

        # Test 5: Try importing main app (this might fail)
        try:
            from api.main import app

            print("✅ Main app import successful")

            # Test 6: TestClient with main app
            try:
                TestClient(app)
                print("✅ TestClient with main app successful")
            except Exception as e:
                print(f"❌ TestClient with main app failed: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback

                traceback.print_exc()

        except Exception as e:
            print(f"❌ Main app import failed: {e}")
            print(f"Error type: {type(e).__name__}")
            # Continue with other tests

        return True

    except Exception as e:
        print(f"❌ TestClient test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_testclient_import()
    if success:
        print("\n✅ TestClient functionality working correctly!")
        sys.exit(0)
    else:
        print("\n❌ TestClient issues detected!")
        sys.exit(1)

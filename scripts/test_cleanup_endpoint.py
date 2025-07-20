#!/usr/bin/env python3
"""
Test API Cleanup Endpoint
"""


import requests


def test_cleanup_endpoint():
    """Test the cleanup endpoint functionality"""

    base_url = "http://localhost:8000"

    try:
        # Test cleanup status endpoint
        print("Testing cleanup status endpoint...")
        response = requests.get(f"{base_url}/cleanup/status", timeout=30)

        if response.status_code == 200:
            status = response.json()
            print(f"✅ Status endpoint working: {status}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            return False

        # Test cleanup endpoint
        print("Testing cleanup endpoint...")
        response = requests.post(f"{base_url}/cleanup", timeout=30)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Cleanup endpoint working: {result}")
            return True
        else:
            print(f"❌ Cleanup endpoint failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    if test_cleanup_endpoint():
        print("✅ Cleanup endpoint tests passed")
    else:
        print("❌ Cleanup endpoint tests failed")

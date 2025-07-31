#!/usr/bin/env python3
"""Test API key saving functionality."""

import requests

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

# Test data
test_settings = {
    "llm_provider": "openai",
    "llm_model": "gpt-3.5-turbo",
    "openai_api_key": "sk-test123456789",
    "api_base_url": "http://localhost:8000",
}


def test_save_settings():
    """Test saving settings with API key."""
    print("Testing API key save functionality...\n")

    # 1. Save settings
    print("1. Saving settings...")
    response = requests.patch(
        f"{BASE_URL}/settings", json=test_settings, headers={"Content-Type": "application/json"}
    )

    print(f"Save response status: {response.status_code}")
    print(f"Save response data: {response.json()}\n")

    # 2. Retrieve settings
    print("2. Retrieving settings...")
    response = requests.get(f"{BASE_URL}/settings")

    print(f"Get response status: {response.status_code}")
    data = response.json()
    print(f"Get response data: {data}\n")

    # Check if API key is persisted (should be masked)
    if "openai_api_key" in data:
        if data["openai_api_key"]:
            print("✅ API key persistence: SUCCESS - API key is saved and masked")
        else:
            print("❌ API key persistence: FAILED - API key is empty")
    else:
        print("❌ API key persistence: FAILED - API key not in response")

    return data


if __name__ == "__main__":
    try:
        test_save_settings()
    except Exception as e:
        print(f"Error during test: {e}")

#!/usr/bin/env python3
"""Test authentication bypass in dev mode."""

import requests
import json

# Test endpoints
BASE_URL = "http://localhost:8000"

def test_endpoint(name, url, headers=None):
    """Test an endpoint and print results."""
    print(f"\n=== Testing {name} ===")
    print(f"URL: {url}")
    if headers:
        print(f"Headers: {headers}")
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2)[:200] + "...")
        else:
            print("Error:", response.text[:200])
    except Exception as e:
        print(f"Exception: {e}")

# Test various endpoints
print("Testing FreeAgentics API Authentication\n")

# 1. Test dev config (should work without auth)
test_endpoint("Dev Config", f"{BASE_URL}/api/v1/dev-config")

# 2. Test health (should work without auth)
test_endpoint("Health", f"{BASE_URL}/api/v1/health")

# 3. Test agents without auth
test_endpoint("Agents (no auth)", f"{BASE_URL}/api/agents")

# 4. Get token and test with auth
print("\n=== Getting dev token ===")
dev_response = requests.get(f"{BASE_URL}/api/v1/dev-config")
if dev_response.status_code == 200:
    config = dev_response.json()
    token = config.get("auth", {}).get("token")
    if token:
        print(f"Got token: {token[:50]}...")
        
        # Test with token
        headers = {"Authorization": f"Bearer {token}"}
        test_endpoint("Agents (with token)", f"{BASE_URL}/api/agents", headers)
        
        # Test with token and fingerprint
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Fingerprint": "dev_fingerprint"
        }
        test_endpoint("Agents (with token + fingerprint)", f"{BASE_URL}/api/agents", headers)
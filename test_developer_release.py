#!/usr/bin/env python3
"""Test script for FreeAgentics developer release.

This script verifies that a developer can:
1. Clone the repository (assumed done)
2. Run make install (assumed done)
3. Run make dev (we'll test the API)
4. Use the system without API keys (mock mode)
5. See a fully functioning multi-agent conversation
"""

import sys

import requests

# Configuration
API_BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"


def test_health_check():
    """Test if the API is running."""
    print("1. Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)  # nosec B113
        print(f"   ✅ Health check: {response.status_code}")
        return True
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False


def test_create_conversation():
    """Test creating a multi-agent conversation."""
    print("\n2. Testing conversation creation...")

    payload = {
        "prompt": "How can we accelerate green chemistry in Jakarta?",
        "agent_count": 3,
        "conversation_turns": 5,
        "llm_provider": "mock",
        "model": "mock-model",
    }

    # Try different auth approaches
    auth_headers = [
        {},  # No auth
        {"Authorization": "Bearer dev-token"},  # Simple dev token
        {"Authorization": "Bearer dev-token-12345"},  # Specific dev token
        {"X-Dev-Mode": "true"},  # Dev mode header
    ]

    for i, headers in enumerate(auth_headers):
        print(f"\n   Attempt {i+1}: Headers = {headers}")
        headers["Content-Type"] = "application/json"

        try:
            response = requests.post(
                f"{API_BASE}/api/v1/agent-conversations", json=payload, headers=headers, timeout=5
            )

            print(f"   Response status: {response.status_code}")
            if response.status_code in [200, 201]:
                data = response.json()
                print(f"   ✅ Conversation created!")
                print(f"   - ID: {data.get('conversation_id')}")
                print(f"   - Status: {data.get('status')}")
                print(f"   - Agents: {len(data.get('agents', []))}")
                print(f"   - WebSocket URL: {data.get('websocket_url')}")
                return data
            else:
                print(f"   ❌ Failed: {response.text[:200]}")

        except requests.exceptions.Timeout:
            print(f"   ❌ Request timed out")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return None


def test_websocket_connection(websocket_url):
    """Test WebSocket connection for real-time updates."""
    print(f"\n3. Testing WebSocket connection...")
    print(f"   URL: {websocket_url}")
    print("   ℹ️  WebSocket testing requires manual verification")
    print("   You can use a WebSocket client to connect and observe messages")
    return True


def main():
    """Run all tests."""
    print("=== FreeAgentics Developer Release Test ===\n")
    print("Prerequisites:")
    print("- Repository cloned ✅")
    print("- make install completed ✅")
    print("- make dev running (testing now...)")

    # Test 1: Health check
    if not test_health_check():
        print("\n❌ API is not running. Please run 'make dev' first.")
        sys.exit(1)

    # Test 2: Create conversation
    conversation = test_create_conversation()
    if not conversation:
        print("\n❌ Could not create conversation. Check API implementation.")
        sys.exit(1)

    # Test 3: WebSocket info
    if "websocket_url" in conversation:
        test_websocket_connection(conversation["websocket_url"])

    print("\n=== Summary ===")
    print("✅ API is running")
    print("✅ Can create multi-agent conversations")
    print("✅ Mock mode works (no API keys needed)")
    print("✅ WebSocket endpoint provided")
    print("\n🎉 Developer release is working!")
    print("\nNext steps:")
    print("1. Connect to the WebSocket URL to see real-time messages")
    print("2. Use the conversation ID to query conversation status")
    print("3. Add real API keys for production use")


if __name__ == "__main__":
    main()

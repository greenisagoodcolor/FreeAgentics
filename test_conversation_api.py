#!/usr/bin/env python3
"""Test script for agent conversation API endpoint."""

import json

import requests

# Configuration
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"


def get_dev_token():
    """Get a development token."""
    try:
        response = requests.get(f"{API_URL}/auth/token/dev", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["access_token"]
    except requests.ConnectionError:
        print("❌ Cannot connect to server. Make sure 'make dev' is running.")
        return None
    except Exception as e:
        print(f"❌ Failed to get dev token: {e}")
        return None


def test_agent_conversation():
    """Test the agent conversation endpoint."""
    token = get_dev_token()
    if not token:
        return

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Test request
    request_data = {
        "prompt": "Design a sustainable green chemistry accelerator for Jakarta",
        "agent_count": 3,
        "conversation_turns": 3,
        "llm_provider": "openai",
        "model": "gpt-3.5-turbo",
    }

    print(f"Testing agent conversation endpoint...")
    print(f"Request: {json.dumps(request_data, indent=2)}")

    try:
        response = requests.post(
            f"{API_URL}/agent-conversations", headers=headers, json=request_data
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success! Conversation ID: {data['conversation_id']}")
            print(f"Agents created: {len(data['agents'])}")
            for agent in data["agents"]:
                print(f"  - {agent['name']} ({agent['role']}): {agent['personality']}")

            print(f"\nConversation messages: {len(data['messages'])}")
            for msg in data["messages"]:
                print(
                    f"  Turn {msg['turn_number']} - {msg['agent_name']}: {msg['content'][:100]}..."
                )

        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"\n❌ Request failed: {e}")


def test_process_prompt():
    """Test the process-prompt endpoint that frontend uses."""
    token = get_dev_token()
    if not token:
        return

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    request_data = {"prompt": "Create a Jakarta accelerator for green chemistry"}

    print(f"\n\nTesting process-prompt endpoint...")
    print(f"Request: {json.dumps(request_data, indent=2)}")

    try:
        response = requests.post(
            f"{BASE_URL}/api/process-prompt", headers=headers, json=request_data
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success!")
            print(f"Conversation ID: {data.get('conversationId', 'N/A')}")
            print(f"Agents returned: {len(data.get('agents', []))}")
            if data.get("agents"):
                for agent in data["agents"]:
                    print(f"  - {agent.get('name', 'Unknown')} ({agent.get('type', 'Unknown')})")
            print(f"Knowledge graph nodes: {len(data.get('knowledgeGraph', {}).get('nodes', []))}")
            print(f"Suggestions: {len(data.get('suggestions', []))}")
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"\n❌ Request failed: {e}")


if __name__ == "__main__":
    print("Testing FreeAgentics Agent Conversation API")
    print("=" * 50)

    # First test the direct agent conversation endpoint
    test_agent_conversation()

    # Then test the UI-compatible process-prompt endpoint
    test_process_prompt()

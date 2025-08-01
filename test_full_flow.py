import asyncio
import json

import httpx
import websockets


async def test_full_flow():
    print("=== Testing Full FreeAgentics Flow ===\n")

    # Step 1: Create conversation
    print("1. Creating conversation...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/agent-conversations",
            json={
                "prompt": "How can we improve urban sustainability?",
                "agent_count": 3,
                "conversation_turns": 5,
            },
        )

        if response.status_code not in [200, 201]:
            print(f"❌ Failed to create conversation: {response.status_code}")
            print(f"Response: {response.text}")
            return

        data = response.json()
        conversation_id = data.get("conversation_id")
        ws_url = data.get("websocket_url")

        print(f"✅ Conversation created!")
        print(f"   ID: {conversation_id}")
        print(f"   WebSocket URL: {ws_url}")
        print(f"   Status: {data.get('status')}")
        print(f"   Agents: {len(data.get('agents', []))}")

        # Step 2: Connect to conversation WebSocket
        if ws_url:
            print(f"\n2. Connecting to WebSocket...")
            try:
                # Convert relative to absolute URL
                if ws_url.startswith("/"):
                    ws_url = f"ws://localhost:8000{ws_url}"

                async with websockets.connect(ws_url) as websocket:
                    print(f"✅ Connected to conversation WebSocket")

                    # Listen for a few messages
                    for i in range(3):
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            msg_data = json.loads(message)
                            print(f"\n📨 Message {i+1}:")
                            print(f"   Type: {msg_data.get('type')}")
                            if "data" in msg_data:
                                print(f"   Content: {msg_data['data'].get('content', '')[:100]}...")
                        except asyncio.TimeoutError:
                            print(f"⏱️  No message received in 5 seconds")
                            break

            except Exception as e:
                print(f"❌ WebSocket connection failed: {e}")

        # Step 3: Try to retrieve conversation
        print(f"\n3. Retrieving conversation...")
        get_response = await client.get(
            f"http://localhost:8000/api/v1/agent-conversations/{conversation_id}"
        )

        if get_response.status_code == 200:
            print(f"✅ Conversation retrieved successfully!")
            conv_data = get_response.json()
            print(f"   Messages: {len(conv_data.get('messages', []))}")
            print(f"   Status: {conv_data.get('status')}")
        else:
            print(f"❌ Failed to retrieve conversation: {get_response.status_code}")
            print(f"   Error: {get_response.text}")


asyncio.run(test_full_flow())

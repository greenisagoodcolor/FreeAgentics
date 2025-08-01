import asyncio

import httpx


async def test():
    # Create conversation
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/agent-conversations",
            json={"prompt": "Test conversation", "agent_count": 2, "conversation_turns": 3},
        )
        print(f"Create response: {response.status_code}")
        if response.status_code == 200 or response.status_code == 201:
            data = response.json()
            print(f'Conversation ID: {data.get("conversation_id")}')
            print(f'WebSocket URL: {data.get("websocket_url")}')

            # Try to get the conversation
            conv_id = data.get("conversation_id")
            get_response = await client.get(
                f"http://localhost:8000/api/v1/agent-conversations/{conv_id}"
            )
            print(f"\nGet response: {get_response.status_code}")
            if get_response.status_code != 200:
                print(f"Error: {get_response.text}")
            else:
                print("Conversation retrieved successfully!")
        else:
            print(f"Create failed: {response.text}")


asyncio.run(test())

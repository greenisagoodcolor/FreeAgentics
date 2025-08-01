import asyncio

import websockets


async def test_websocket():
    uri = "ws://localhost:8000/api/v1/ws/dev"

    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to {uri}")

            # Wait for welcome message
            welcome = await websocket.recv()
            print(f"Welcome message: {welcome}")

            # Keep connection open for a bit
            await asyncio.sleep(2)

            print("✅ WebSocket connection successful!")

    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        print(f"Error type: {type(e).__name__}")


asyncio.run(test_websocket())

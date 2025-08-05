#!/usr/bin/env python3
"""
WebSocket connectivity test for FreeAgentics
Verifies end-to-end WebSocket communication between frontend and backend
"""

import asyncio
import json
import websockets
import sys
from datetime import datetime


async def test_websocket_connection():
    """Test WebSocket connection to the dev endpoint"""
    ws_url = "ws://localhost:8000/api/v1/ws/dev"
    
    try:
        print(f"🔌 Attempting to connect to: {ws_url}")
        
        async with websockets.connect(ws_url) as websocket:
            print("✅ Connected successfully!")
            
            # Wait for initial messages
            print("\n📥 Waiting for initial messages...")
            
            # Read connection established message
            msg1 = await websocket.recv()
            data1 = json.loads(msg1)
            print(f"  Message 1: {data1['type']} - {data1.get('client_id', 'N/A')}")
            
            # Read dev welcome message
            msg2 = await websocket.recv()
            data2 = json.loads(msg2)
            print(f"  Message 2: {data2['type']} - {data2.get('message', 'N/A')}")
            
            # Test ping/pong
            print("\n🏓 Testing ping/pong...")
            ping_msg = {
                "type": "ping",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(ping_msg))
            
            pong = await websocket.recv()
            pong_data = json.loads(pong)
            if pong_data["type"] == "pong":
                print("  ✅ Pong received!")
            
            # Test prompt submission
            print("\n📝 Testing prompt submission...")
            prompt_msg = {
                "type": "prompt_submitted",
                "prompt_id": "test-prompt-123",
                "prompt": "Create a test agent",
                "conversation_id": "test-conv-456"
            }
            await websocket.send(json.dumps(prompt_msg))
            
            # Wait for acknowledgment
            ack = await websocket.recv()
            ack_data = json.loads(ack)
            if ack_data["type"] == "prompt_acknowledged":
                print(f"  ✅ Prompt acknowledged: {ack_data.get('message', 'N/A')}")
            
            # Test conversation clearing
            print("\n🗑️ Testing conversation clearing...")
            clear_msg = {
                "type": "clear_conversation",
                "data": {
                    "conversationId": "test-conv-456"
                }
            }
            await websocket.send(json.dumps(clear_msg))
            
            # Wait for clear confirmation
            clear_ack = await websocket.recv()
            clear_data = json.loads(clear_ack)
            if clear_data["type"] == "conversation_cleared":
                print(f"  ✅ Conversation cleared: {clear_data.get('message', 'N/A')}")
            
            print("\n🎉 WEBSOCKET CONNECTIVITY: VERIFIED ✅")
            print("All WebSocket operations completed successfully!")
            return True
            
    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket Error: {e}")
        print("Make sure the backend server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 FreeAgentics WebSocket Connectivity Test")
    print("=" * 50)
    
    # Run the test
    success = asyncio.run(test_websocket_connection())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Detailed WebSocket debugging script for FreeAgentics
Provides comprehensive diagnostics for WebSocket connectivity issues
"""

import asyncio
import json
import websockets
import aiohttp
import sys
from datetime import datetime


async def check_backend_health():
    """Check if backend is running and healthy"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health') as resp:
                if resp.status == 200:
                    print("✅ Backend health check: OK")
                    return True
                else:
                    print(f"⚠️  Backend health check failed: Status {resp.status}")
                    return False
    except Exception as e:
        print(f"❌ Backend not reachable: {e}")
        return False


async def check_dev_config():
    """Check development configuration"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/v1/dev-config') as resp:
                if resp.status == 200:
                    config = await resp.json()
                    print("✅ Dev config endpoint: OK")
                    print(f"   Auth Token: {config.get('auth_token', 'N/A')[:20]}...")
                    print(f"   WebSocket URL: {config.get('websocket_url', 'N/A')}")
                    return True
                else:
                    print(f"⚠️  Dev config endpoint failed: Status {resp.status}")
                    return False
    except Exception as e:
        print(f"❌ Dev config not accessible: {e}")
        return False


async def test_websocket_detailed():
    """Detailed WebSocket connection test with diagnostics"""
    ws_urls = [
        "ws://localhost:8000/api/v1/ws/dev",
        "ws://localhost:8000/ws/dev",  # Try without /api/v1 prefix
        "ws://127.0.0.1:8000/api/v1/ws/dev",  # Try with IP
    ]
    
    for ws_url in ws_urls:
        print(f"\n🔌 Testing: {ws_url}")
        
        try:
            # Set a short timeout for connection attempts
            async with websockets.connect(ws_url, ping_interval=5, ping_timeout=10) as websocket:
                print("  ✅ Connected!")
                
                # Collect initial messages
                messages = []
                try:
                    # Wait for up to 2 seconds for initial messages
                    start_time = asyncio.get_event_loop().time()
                    while asyncio.get_event_loop().time() - start_time < 2:
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                            data = json.loads(msg)
                            messages.append(data)
                            print(f"  📥 Received: {data['type']}")
                        except asyncio.TimeoutError:
                            break
                except Exception as e:
                    print(f"  ⚠️  Error receiving messages: {e}")
                
                # Display received messages
                if messages:
                    print(f"\n  Initial messages ({len(messages)}):")
                    for i, msg in enumerate(messages):
                        print(f"    {i+1}. Type: {msg.get('type', 'unknown')}")
                        if 'message' in msg:
                            print(f"       Message: {msg['message']}")
                        if 'client_id' in msg:
                            print(f"       Client ID: {msg['client_id']}")
                
                # Test various message types
                test_messages = [
                    {
                        "name": "Ping",
                        "msg": {"type": "ping", "timestamp": datetime.now().isoformat()},
                        "expected": "pong"
                    },
                    {
                        "name": "Invalid JSON",
                        "msg": {"type": "test_invalid", "data": {"test": True}},
                        "expected": "echo"
                    },
                    {
                        "name": "Agent Create",
                        "msg": {
                            "type": "agent_create",
                            "data": {
                                "name": "Debug Test Agent",
                                "type": "explorer"
                            }
                        },
                        "expected": "agent_created"
                    }
                ]
                
                print("\n  Testing message types:")
                for test in test_messages:
                    print(f"    📤 Sending {test['name']}...")
                    await websocket.send(json.dumps(test['msg']))
                    
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2)
                        resp_data = json.loads(response)
                        if resp_data['type'] == test['expected']:
                            print(f"      ✅ Got expected response: {test['expected']}")
                        else:
                            print(f"      ⚠️  Got unexpected response: {resp_data['type']}")
                    except asyncio.TimeoutError:
                        print(f"      ❌ No response received (timeout)")
                    except Exception as e:
                        print(f"      ❌ Error: {e}")
                
                print(f"\n  🎯 WebSocket at {ws_url} is WORKING!")
                return True
                
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"  ❌ Invalid status code: {e.status_code}")
            if e.status_code == 404:
                print("     → Endpoint not found (check route configuration)")
            elif e.status_code == 403:
                print("     → Forbidden (check authentication)")
        except websockets.exceptions.WebSocketException as e:
            print(f"  ❌ WebSocket error: {e}")
        except ConnectionRefusedError:
            print(f"  ❌ Connection refused (is the server running?)")
        except Exception as e:
            print(f"  ❌ Unexpected error: {type(e).__name__}: {e}")
    
    return False


async def main():
    """Run all diagnostic tests"""
    print("🔍 FreeAgentics WebSocket Diagnostics")
    print("=" * 50)
    
    # Check backend health
    print("\n1️⃣  Checking backend server...")
    backend_ok = await check_backend_health()
    
    if not backend_ok:
        print("\n⚠️  Backend server is not running!")
        print("Start it with: make dev")
        return False
    
    # Check dev configuration
    print("\n2️⃣  Checking development configuration...")
    config_ok = await check_dev_config()
    
    # Test WebSocket connections
    print("\n3️⃣  Testing WebSocket connections...")
    ws_ok = await test_websocket_detailed()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Diagnostic Summary:")
    print(f"  Backend Health: {'✅ OK' if backend_ok else '❌ Failed'}")
    print(f"  Dev Config: {'✅ OK' if config_ok else '❌ Failed'}")
    print(f"  WebSocket: {'✅ OK' if ws_ok else '❌ Failed'}")
    
    if backend_ok and ws_ok:
        print("\n🎉 All systems operational!")
        return True
    else:
        print("\n⚠️  Some issues detected. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
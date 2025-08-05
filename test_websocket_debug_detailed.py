#!/usr/bin/env python3
"""Detailed WebSocket endpoint debugging."""

import asyncio
import json
import logging
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_endpoint_detailed(url: str):
    """Test a specific WebSocket endpoint with detailed logging."""
    logger.info(f"🔍 Testing endpoint: {url}")
    
    try:
        async with websockets.connect(url) as websocket:
            logger.info("✅ Connected successfully")
            
            # Receive all initial messages for 3 seconds
            messages = []
            timeout_counter = 0
            while timeout_counter < 6:  # 6 * 0.5s = 3s total
                try:
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    message = json.loads(raw_message)
                    messages.append(message)
                    logger.info(f"📥 Message {len(messages)}: {message}")
                except asyncio.TimeoutError:
                    timeout_counter += 1
                except Exception as e:
                    logger.error(f"❌ Error receiving message: {e}")
                    break
            
            logger.info(f"📊 Total messages received: {len(messages)}")
            
            # Send a ping to test bidirectional communication
            ping_msg = {"type": "ping", "timestamp": "test"}
            await websocket.send(json.dumps(ping_msg))
            logger.info("📤 Sent ping message")
            
            # Wait for pong
            try:
                raw_response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response = json.loads(raw_response)
                logger.info(f"📥 Response to ping: {response}")
            except asyncio.TimeoutError:
                logger.warning("⏰ No response to ping within 2s")
            
            return messages
                
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return []


async def main():
    """Test the dev endpoint in detail."""
    logger.info("🔬 DETAILED WEBSOCKET DEBUG ANALYSIS")
    logger.info("=" * 60)
    
    # Test the dev endpoint
    messages = await test_endpoint_detailed("ws://localhost:8000/api/v1/ws/dev")
    
    logger.info("\n📋 ANALYSIS:")
    if messages:
        first_msg = messages[0]
        msg_type = first_msg.get("type")
        logger.info(f"First message type: {msg_type}")
        
        if msg_type == "dev_welcome":
            logger.info("✅ SUCCESS: Connecting to dev endpoint correctly")
        elif msg_type == "connection_established":
            logger.error("❌ PROBLEM: Connecting to authenticated endpoint instead of dev endpoint")
            logger.error("  This means the routing fix didn't work properly")
            
            # Check if the client_id indicates which endpoint we hit
            client_id = first_msg.get("client_id", "unknown")
            logger.info(f"  Client ID: {client_id}")
            
            if client_id == "dev":
                logger.error("  -> We're being routed to /ws/{client_id} with client_id='dev'")
            else:
                logger.error(f"  -> We're being routed to /ws/dev but getting auth endpoint response")
        else:
            logger.warning(f"🤔 UNEXPECTED: Got unexpected message type: {msg_type}")
    else:
        logger.error("❌ FAILURE: No messages received")


if __name__ == "__main__":
    asyncio.run(main())
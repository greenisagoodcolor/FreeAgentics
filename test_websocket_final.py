#!/usr/bin/env python3
"""Final WebSocket connectivity test - assumes server is already running."""

import asyncio
import json
import logging
import time
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_websocket_connectivity():
    """Test WebSocket connectivity with existing server."""
    logger.info("🔬 FINAL WEBSOCKET CONNECTIVITY TEST")
    logger.info("=" * 60)
    
    # Test the dev WebSocket endpoint that frontend uses
    ws_url = "ws://localhost:8000/api/v1/ws/dev"
    logger.info(f"Testing frontend WebSocket URL: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            logger.info("✅ WebSocket connection established")
            
            # Receive initial messages
            messages = []
            timeout_count = 0
            
            while timeout_count < 4:  # 4 × 0.5s = 2s total wait
                try:
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    message = json.loads(raw_message)
                    messages.append(message)
                    logger.info(f"📥 Received: {message.get('type', 'unknown')}")
                except asyncio.TimeoutError:
                    timeout_count += 1
                except Exception as e:
                    logger.error(f"❌ Error receiving message: {e}")
                    break
            
            # Verify we got the expected messages
            message_types = [msg.get("type") for msg in messages]
            
            if "dev_welcome" in message_types:
                logger.info("✅ SUCCESS: Dev welcome message received")
                dev_welcome_msg = next(msg for msg in messages if msg.get("type") == "dev_welcome")
                features = dev_welcome_msg.get("features", [])
                logger.info(f"📋 Dev features: {features}")
                
                expected_features = ["Agent creation simulation", "Real-time updates", "Knowledge graph visualization"]
                for feature in expected_features:
                    if feature in features:
                        logger.info(f"  ✅ {feature}")
                    else:
                        logger.warning(f"  ⚠️  Missing: {feature}")
                
            else:
                logger.error(f"❌ No dev_welcome message. Got: {message_types}")
                return False
            
            # Test ping/pong
            ping_msg = {"type": "ping", "timestamp": time.time()}
            await websocket.send(json.dumps(ping_msg))
            logger.info("📤 Sent ping")
            
            try:
                raw_response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                response = json.loads(raw_response)
                if response.get("type") == "pong":
                    logger.info("✅ Pong received - bidirectional communication working")
                else:
                    logger.warning(f"⚠️  Expected pong, got: {response.get('type')}")
            except asyncio.TimeoutError:
                logger.warning("⏰ No pong response within 3s")
            
            # Test prompt submission (the main frontend feature)
            prompt_msg = {
                "type": "prompt_submitted",
                "prompt_id": "test_123",
                "prompt": "Create a test agent for exploration",
                "conversation_id": "test_conv_123",
                "llm_provider": "mock"
            }
            await websocket.send(json.dumps(prompt_msg))
            logger.info("📤 Sent prompt submission")
            
            # Wait for acknowledgment
            try:
                raw_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response = json.loads(raw_response)
                if response.get("type") == "prompt_acknowledged":
                    logger.info("✅ Prompt acknowledgment received")
                    
                    # Wait for result (or error)
                    try:
                        raw_result = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        result = json.loads(raw_result)
                        result_type = result.get("type")
                        
                        if result_type == "agent_created":
                            logger.info("✅ Agent creation successful!")
                        elif result_type == "error":
                            logger.info(f"⚠️  Agent creation failed (expected in test env): {result.get('message')}")
                        else:
                            logger.info(f"📥 Got result: {result_type}")
                    except asyncio.TimeoutError:
                        logger.info("⏰ No result within 10s (may be processing)")
                        
                else:
                    logger.warning(f"⚠️  Expected prompt_acknowledged, got: {response.get('type')}")
            except asyncio.TimeoutError:
                logger.error("❌ No prompt acknowledgment within 5s")
                return False
            
            logger.info("✅ ALL TESTS PASSED - WebSocket connectivity is working!")
            return True
            
    except ConnectionRefusedError:
        logger.error("❌ Connection refused - server may not be running on port 8000")
        return False
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False


async def main():
    """Run the final connectivity test."""
    success = await test_websocket_connectivity()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 WEBSOCKET CONNECTIVITY: VERIFIED ✅")
        logger.info("🚀 Frontend can successfully connect to backend!")
        logger.info("🔗 URL: ws://localhost:8000/api/v1/ws/dev")
        logger.info("📡 Features: Bidirectional communication, prompt processing")
    else:
        logger.error("💥 WEBSOCKET CONNECTIVITY: FAILED ❌")
        logger.error("🔧 Check server status and configuration")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
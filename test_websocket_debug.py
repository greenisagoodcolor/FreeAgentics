#!/usr/bin/env python3
"""Quick WebSocket endpoint debugging test."""

import asyncio
import json
import logging
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_endpoint(url: str, expected_message_type: str = None):
    """Test a specific WebSocket endpoint."""
    logger.info(f"Testing endpoint: {url}")

    try:
        async with websockets.connect(url) as websocket:
            logger.info("✅ Connected successfully")

            # Receive first message
            try:
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                message = json.loads(raw_message)
                logger.info(f"📥 Received: {message}")

                if expected_message_type:
                    if message.get("type") == expected_message_type:
                        logger.info(f"✅ Got expected message type: {expected_message_type}")
                        return True
                    else:
                        logger.error(
                            f"❌ Expected {expected_message_type}, got {message.get('type')}"
                        )
                        return False
                return True

            except asyncio.TimeoutError:
                logger.warning("⏰ No message received within timeout")
                return False

    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False


async def main():
    """Test different WebSocket endpoints to find the correct one."""
    endpoints_to_test = [
        ("ws://localhost:8000/api/v1/ws/dev", "dev_welcome"),
        ("ws://localhost:8000/ws/dev", "dev_welcome"),
        ("ws://localhost:8000/api/v1/ws/test_123", "connection_established"),
        ("ws://localhost:8000/ws/test_123", "connection_established"),
    ]

    logger.info("🔍 Debugging WebSocket endpoints...")

    for url, expected_type in endpoints_to_test:
        logger.info(f"\n{'='*60}")
        result = await test_endpoint(url, expected_type)
        logger.info(f"Result: {'✅ PASS' if result else '❌ FAIL'}")

        # Small delay between tests
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())

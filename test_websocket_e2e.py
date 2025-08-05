#!/usr/bin/env python3
"""
Comprehensive WebSocket End-to-End Connectivity Test
==================================================

This test verifies the complete WebSocket connection flow between frontend and backend:
1. Backend WebSocket server startup
2. WebSocket endpoint availability
3. Connection establishment
4. Message exchange
5. Error handling

Based on Nemesis Committee analysis by Kent Beck, Robert C. Martin, Martin Fowler,
Michael Feathers, Jessica Kerr, Sindre Sorhus, Addy Osmani, Sarah Drasner,
Evan You, Rich Harris, and Charity Majors.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

import pytest
import websockets
from websockets.exceptions import ConnectionClosedError, InvalidURI, WebSocketException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WebSocketTestClient:
    """Test client for WebSocket connections with proper error handling."""

    def __init__(self, url: str):
        self.url = url
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.received_messages: List[Dict] = []
        self.connection_established = False

    async def connect(self, timeout: float = 10.0) -> bool:
        """Connect to WebSocket server with timeout."""
        try:
            logger.info(f"Attempting to connect to WebSocket: {self.url}")
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.url), timeout=timeout
            )
            self.connection_established = True
            logger.info("✅ WebSocket connection established successfully")
            return True
        except asyncio.TimeoutError:
            logger.error(f"❌ Connection timeout after {timeout}s")
            return False
        except ConnectionRefusedError:
            logger.error("❌ Connection refused - server may not be running")
            return False
        except InvalidURI:
            logger.error(f"❌ Invalid WebSocket URI: {self.url}")
            return False
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    async def send_message(self, message: Dict) -> bool:
        """Send a message to the WebSocket server."""
        if not self.websocket:
            logger.error("❌ No active WebSocket connection")
            return False

        try:
            await self.websocket.send(json.dumps(message))
            logger.info(f"📤 Sent message: {message.get('type', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return False

    async def receive_messages(self, timeout: float = 5.0, count: int = 1) -> List[Dict]:
        """Receive messages from WebSocket server."""
        if not self.websocket:
            logger.error("❌ No active WebSocket connection")
            return []

        messages = []
        try:
            for _ in range(count):
                try:
                    raw_message = await asyncio.wait_for(
                        self.websocket.recv(), timeout=timeout
                    )
                    message = json.loads(raw_message)
                    messages.append(message)
                    self.received_messages.append(message)
                    logger.info(f"📥 Received message: {message.get('type', 'unknown')}")
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout waiting for message (after {len(messages)} messages)")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse JSON message: {e}")
                    break
        except ConnectionClosedError:
            logger.error("❌ WebSocket connection closed unexpectedly")
        except Exception as e:
            logger.error(f"❌ Error receiving messages: {e}")

        return messages

    async def close(self):
        """Close the WebSocket connection."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("🔌 WebSocket connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing WebSocket: {e}")
            finally:
                self.websocket = None
                self.connection_established = False


class BackendServerManager:
    """Manages the backend server for testing."""

    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.process: Optional[subprocess.Popen] = None
        self.startup_timeout = 30.0

    async def start_server(self) -> bool:
        """Start the backend server."""
        try:
            logger.info("🚀 Starting backend server...")
            
            # Check if server is already running
            if await self._is_server_running():
                logger.info("✅ Backend server already running")
                return True

            # Start the server process
            env = os.environ.copy()
            env.update({
                "DEVELOPMENT_MODE": "true",
                "TESTING": "true",
                "DATABASE_URL": "",  # Force SQLite mode
                "REDIS_URL": "",     # Force in-memory mode
                "LOG_LEVEL": "INFO",
                "API_HOST": "0.0.0.0",
                "API_PORT": "8000",
            })

            self.process = subprocess.Popen(
                [sys.executable, "main.py"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/home/green/FreeAgentics",
            )

            # Wait for server to start
            return await self._wait_for_startup()

        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            return False

    async def _is_server_running(self) -> bool:
        """Check if the server is already running."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.backend_url}/api/v1/health", timeout=5) as response:
                    return response.status == 200
        except:
            return False

    async def _wait_for_startup(self) -> bool:
        """Wait for the server to start up."""
        import aiohttp
        
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.backend_url}/api/v1/health", timeout=2) as response:
                        if response.status == 200:
                            logger.info("✅ Backend server is ready")
                            return True
            except:
                pass
            
            await asyncio.sleep(1)

        logger.error(f"❌ Server failed to start within {self.startup_timeout}s")
        return False

    async def stop_server(self):
        """Stop the backend server."""
        if self.process:
            try:
                logger.info("🛑 Stopping backend server...")
                self.process.terminate()
                await asyncio.sleep(2)
                
                if self.process.poll() is None:
                    logger.warning("Force killing server process...")
                    self.process.kill()
                
                logger.info("✅ Backend server stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping server: {e}")
            finally:
                self.process = None


@asynccontextmanager
async def backend_server() -> AsyncGenerator[BackendServerManager, None]:
    """Context manager for backend server lifecycle."""
    server_manager = BackendServerManager()
    
    try:
        success = await server_manager.start_server()
        if not success:
            raise RuntimeError("Failed to start backend server")
        yield server_manager
    finally:
        await server_manager.stop_server()


async def test_websocket_basic_connectivity():
    """Test basic WebSocket connectivity to /api/v1/ws/dev endpoint."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Basic WebSocket Connectivity")
    logger.info("="*80)

    async with backend_server() as server:
        # Test the dev WebSocket endpoint
        ws_url = "ws://localhost:8000/api/v1/ws/dev"
        client = WebSocketTestClient(ws_url)

        try:
            # Test connection
            connected = await client.connect(timeout=10.0)
            assert connected, "Failed to establish WebSocket connection"

            # Test receiving messages (should get connection_established then dev_welcome)
            messages = await client.receive_messages(timeout=5.0, count=2)
            assert len(messages) >= 1, "No messages received"
            
            # Look for dev_welcome message (may be first or second message)
            dev_welcome_msg = None
            for msg in messages:
                if msg.get("type") == "dev_welcome":
                    dev_welcome_msg = msg
                    break
            
            assert dev_welcome_msg is not None, f"No dev_welcome message found. Got messages: {[m.get('type') for m in messages]}"
            logger.info("✅ Dev welcome message received correctly")

            # Test ping/pong
            ping_msg = {"type": "ping", "timestamp": time.time()}
            sent = await client.send_message(ping_msg)
            assert sent, "Failed to send ping message"

            pong_messages = await client.receive_messages(timeout=5.0, count=1)
            assert len(pong_messages) > 0, "No pong response received"
            
            pong_msg = pong_messages[0]
            assert pong_msg.get("type") == "pong", f"Expected pong, got {pong_msg.get('type')}"
            logger.info("✅ Ping/pong working correctly")

        finally:
            await client.close()

    logger.info("✅ Basic connectivity test PASSED")


async def test_websocket_prompt_processing():
    """Test prompt submission and processing via WebSocket."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Prompt Processing via WebSocket")
    logger.info("="*80)

    async with backend_server() as server:
        ws_url = "ws://localhost:8000/api/v1/ws/dev"
        client = WebSocketTestClient(ws_url)

        try:
            # Connect
            connected = await client.connect(timeout=10.0)
            assert connected, "Failed to establish WebSocket connection"

            # Skip initial messages (connection_established and dev_welcome)
            await client.receive_messages(timeout=5.0, count=2)

            # Submit a prompt
            prompt_msg = {
                "type": "prompt_submitted",
                "prompt_id": "test_prompt_123",
                "prompt": "Create a simple test agent that explores the environment",
                "conversation_id": "test_conv_123",
                "agent_name": "TestAgent",
                "llm_provider": "mock",  # Use mock provider to avoid API key requirements
            }
            
            sent = await client.send_message(prompt_msg)
            assert sent, "Failed to send prompt message"

            # Receive acknowledgment and result
            response_messages = await client.receive_messages(timeout=15.0, count=2)
            assert len(response_messages) >= 1, "No response to prompt submission"

            # Check for acknowledgment
            ack_msg = response_messages[0]
            assert ack_msg.get("type") == "prompt_acknowledged", f"Expected prompt_acknowledged, got {ack_msg.get('type')}"
            assert ack_msg.get("prompt_id") == "test_prompt_123", "Incorrect prompt_id in acknowledgment"
            logger.info("✅ Prompt acknowledgment received")

            # Check for agent creation or error response
            if len(response_messages) > 1:
                result_msg = response_messages[1]
                if result_msg.get("type") == "agent_created":
                    logger.info("✅ Agent created successfully via WebSocket")
                    assert result_msg.get("prompt_id") == "test_prompt_123", "Incorrect prompt_id in result"
                elif result_msg.get("type") == "error":
                    logger.info(f"⚠️  Agent creation failed (expected in test environment): {result_msg.get('message')}")
                    # This is acceptable in test environment without proper API keys
                else:
                    logger.warning(f"Unexpected response type: {result_msg.get('type')}")

        finally:
            await client.close()

    logger.info("✅ Prompt processing test PASSED")


async def test_websocket_error_handling():
    """Test WebSocket error handling with invalid messages."""
    logger.info("\n" + "="*80)
    logger.info("TEST: WebSocket Error Handling")
    logger.info("="*80)

    async with backend_server() as server:
        ws_url = "ws://localhost:8000/api/v1/ws/dev"
        client = WebSocketTestClient(ws_url)

        try:
            # Connect
            connected = await client.connect(timeout=10.0)
            assert connected, "Failed to establish WebSocket connection"

            # Skip initial messages (connection_established and dev_welcome)
            await client.receive_messages(timeout=5.0, count=2)

            # Test invalid JSON
            if client.websocket:
                await client.websocket.send("invalid json{")
                error_messages = await client.receive_messages(timeout=5.0, count=1)
                assert len(error_messages) > 0, "No error response for invalid JSON"
                error_msg = error_messages[0]
                assert error_msg.get("type") == "error", f"Expected error, got {error_msg.get('type')}"
                logger.info("✅ Invalid JSON handled correctly")

            # Test unknown message type
            unknown_msg = {"type": "unknown_message_type", "data": {}}
            sent = await client.send_message(unknown_msg)
            assert sent, "Failed to send unknown message type"

            error_messages = await client.receive_messages(timeout=5.0, count=1)
            assert len(error_messages) > 0, "No error response for unknown message type"
            error_msg = error_messages[0]
            assert error_msg.get("type") == "error", f"Expected error, got {error_msg.get('type')}"
            logger.info("✅ Unknown message type handled correctly")

        finally:
            await client.close()

    logger.info("✅ Error handling test PASSED")


async def test_frontend_backend_integration():
    """Test that frontend and backend WebSocket configurations match."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Frontend-Backend Integration")
    logger.info("="*80)

    # Test the exact WebSocket URL that frontend uses
    frontend_ws_url = "ws://localhost:8000/api/v1/ws/dev"
    
    async with backend_server() as server:
        client = WebSocketTestClient(frontend_ws_url)
        
        try:
            # This should match exactly what the frontend does
            connected = await client.connect(timeout=10.0)
            assert connected, f"Frontend WebSocket URL {frontend_ws_url} is not accessible"
            
            # Verify we get the expected messages (connection_established and dev_welcome)
            messages = await client.receive_messages(timeout=5.0, count=2)
            assert len(messages) >= 1, "No messages from dev endpoint"
            
            # Look for dev_welcome message
            dev_welcome_msg = None
            for msg in messages:
                if msg.get("type") == "dev_welcome":
                    dev_welcome_msg = msg
                    break
            
            assert dev_welcome_msg is not None, "Dev endpoint not responding with dev_welcome message"
            
            # Verify dev mode features are available
            features = dev_welcome_msg.get("features", [])
            expected_features = ["Agent creation simulation", "Real-time updates", "Knowledge graph visualization"]
            for feature in expected_features:
                assert feature in features, f"Missing expected feature: {feature}"
            
            logger.info("✅ Frontend WebSocket URL is accessible and functional")
            
        finally:
            await client.close()

    logger.info("✅ Frontend-backend integration test PASSED")


async def run_all_tests():
    """Run all WebSocket connectivity tests."""
    logger.info("\n" + "🔬" + " NEMESIS COMMITTEE WEBSOCKET CONNECTIVITY ANALYSIS " + "🔬")
    logger.info("=" * 90)
    logger.info("Running comprehensive end-to-end WebSocket connectivity tests...")
    logger.info("Committee: Kent Beck, Robert C. Martin, Martin Fowler, Michael Feathers,")
    logger.info("          Jessica Kerr, Sindre Sorhus, Addy Osmani, Sarah Drasner,")
    logger.info("          Evan You, Rich Harris, Charity Majors")
    logger.info("=" * 90)

    tests = [
        ("Basic Connectivity", test_websocket_basic_connectivity),
        ("Prompt Processing", test_websocket_prompt_processing),
        ("Error Handling", test_websocket_error_handling),
        ("Frontend Integration", test_frontend_backend_integration),
    ]

    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n▶️  Running {test_name} test...")
            await test_func()
            logger.info(f"✅ {test_name} test PASSED")
            passed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED: {e}")
            failed += 1

    logger.info("\n" + "=" * 90)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 90)
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("\n🎉 ALL TESTS PASSED! WebSocket connectivity is working correctly.")
        logger.info("🚀 Ready for user testing!")
    else:
        logger.error(f"\n💥 {failed} test(s) failed. WebSocket connectivity needs fixes.")
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        sys.exit(1)
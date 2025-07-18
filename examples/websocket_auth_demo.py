#!/usr/bin/env python3
"""
WebSocket Authentication Demo

Demonstrates how to connect to WebSocket endpoints with JWT authentication.
This example shows the complete flow from user authentication to WebSocket
communication with proper token handling.
"""

import asyncio
import json
import logging
import os

import websockets

from auth.security_implementation import AuthenticationManager, UserRole

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_websocket_authentication():
    """Demonstrate WebSocket authentication with JWT tokens."""
    print("🚀 WebSocket Authentication Demo")
    print("=" * 50)

    # 1. Set up authentication manager and create test user
    auth_manager = AuthenticationManager()

    # Register a test user
    test_user = auth_manager.register_user(
        username="demo_user",
        email="demo@example.com",
        password=os.getenv("PASSWORD"),
        role=UserRole.RESEARCHER,
    )

    # Authenticate and get JWT token
    authenticated_user = auth_manager.authenticate_user(
        "demo_user", os.getenv("PASSWORD", "demo_password")
    )
    if not authenticated_user:
        print("❌ Authentication failed")
        return

    # Create access token
    access_token = auth_manager.create_access_token(authenticated_user)
    print(f"✅ JWT token created for user: {authenticated_user.username}")
    print(f"🔑 Token role: {authenticated_user.role}")
    print(
        f"📝 Permissions: {[p.value for p in auth_manager.verify_token(access_token).permissions]}"
    )

    # 2. Connect to WebSocket with authentication
    websocket_url = f"ws://localhost:8000/ws/demo_client?token={access_token}"

    try:
        print(f"\n🔌 Connecting to WebSocket: {websocket_url}")

        async with websockets.connect(websocket_url) as websocket:
            print("✅ WebSocket connection established with authentication")

            # Receive connection acknowledgment
            ack_message = await websocket.recv()
            ack_data = json.loads(ack_message)
            print(f"📨 Connection acknowledgment: {ack_data['type']}")

            # 3. Test ping/pong
            print("\n🏓 Testing ping/pong...")
            ping_message = {"type": "ping"}
            await websocket.send(json.dumps(ping_message))

            pong_response = await websocket.recv()
            pong_data = json.loads(pong_response)
            print(
                f"📨 Pong response: {pong_data['type']} at {pong_data['timestamp']}"
            )

            # 4. Test subscription management
            print("\n📺 Testing event subscriptions...")
            subscribe_message = {
                "type": "subscribe",
                "event_types": ["agent:status_update", "world:state_change"],
            }
            await websocket.send(json.dumps(subscribe_message))

            sub_response = await websocket.recv()
            sub_data = json.loads(sub_response)
            print(f"✅ Subscription confirmed: {sub_data['event_types']}")

            # 5. Test authorized agent command
            print("\n🤖 Testing authorized agent command...")
            agent_command = {
                "type": "agent_command",
                "data": {"command": "create", "agent_id": "demo_agent_001"},
            }
            await websocket.send(json.dumps(agent_command))

            command_response = await websocket.recv()
            command_data = json.loads(command_response)
            print(f"📨 Command response: {command_data['type']}")
            print(f"✅ Command status: {command_data['status']}")
            print(f"👤 Executed by: {command_data['user']}")

            # 6. Test authorized query
            print("\n🔍 Testing authorized query...")
            query_message = {
                "type": "query",
                "data": {"query_type": "agent_status"},
            }
            await websocket.send(json.dumps(query_message))

            query_response = await websocket.recv()
            query_data = json.loads(query_response)
            print(f"📨 Query response: {query_data['type']}")
            print(f"🔍 Query type: {query_data['query_type']}")
            print(f"👤 Query by: {query_data['data']['user']}")

            print("\n✅ WebSocket authentication demo completed successfully!")

    except websockets.exceptions.ConnectionClosed as e:
        if e.code == 4001:
            print(
                "❌ WebSocket connection closed: Authentication failed (4001)"
            )
        else:
            print(f"❌ WebSocket connection closed: {e}")
    except ConnectionRefusedError:
        print(
            "❌ Connection refused. Make sure the server is running on localhost:8000"
        )
    except Exception as e:
        print(f"❌ Error during WebSocket communication: {e}")


async def demo_authentication_failure():
    """Demonstrate authentication failure scenarios."""
    print("\n🔒 Authentication Failure Demo")
    print("=" * 50)

    # Test scenarios that should fail
    failure_scenarios = [
        ("No token", "ws://localhost:8000/ws/test_client"),
        (
            "Invalid token",
            "ws://localhost:8000/ws/test_client?token=invalid.jwt.token",
        ),
        ("Empty token", "ws://localhost:8000/ws/test_client?token="),
    ]

    for scenario_name, websocket_url in failure_scenarios:
        try:
            print(f"\n🧪 Testing: {scenario_name}")
            async with websockets.connect(websocket_url) as websocket:
                # Should not reach this point
                print(f"❌ Unexpected success for: {scenario_name}")

        except websockets.exceptions.ConnectionClosed as e:
            if e.code == 4001:
                print(
                    f"✅ Expected authentication failure (4001): {scenario_name}"
                )
            else:
                print(f"❓ Unexpected close code {e.code}: {scenario_name}")
        except ConnectionRefusedError:
            print(f"🔌 Server not running - cannot test: {scenario_name}")
        except Exception as e:
            print(f"❓ Unexpected error for {scenario_name}: {e}")


async def demo_permission_testing():
    """Demonstrate permission-based access control."""
    print("\n🛡️ Permission Testing Demo")
    print("=" * 50)

    # Create observer user with limited permissions
    auth_manager = AuthenticationManager()

    observer_user = auth_manager.register_user(
        username="observer_demo",
        email="observer@example.com",
        password=os.getenv("PASSWORD"),
        role=UserRole.OBSERVER,
    )

    authenticated_observer = auth_manager.authenticate_user(
        "observer_demo", os.getenv("PASSWORD", "observer_password")
    )
    observer_token = auth_manager.create_access_token(authenticated_observer)

    print(f"👀 Created observer user with limited permissions")
    print(
        f"📝 Observer permissions: {[p.value for p in auth_manager.verify_token(observer_token).permissions]}"
    )

    websocket_url = (
        f"ws://localhost:8000/ws/observer_client?token={observer_token}"
    )

    try:
        async with websockets.connect(websocket_url) as websocket:
            print("✅ Observer connected successfully")

            # Receive acknowledgment
            await websocket.recv()

            # Try to perform privileged operation (should fail)
            print(
                "\n🚫 Testing privileged operation with observer permissions..."
            )
            privileged_command = {
                "type": "agent_command",
                "data": {
                    "command": "create",
                    "agent_id": "unauthorized_agent",
                },
            }
            await websocket.send(json.dumps(privileged_command))

            error_response = await websocket.recv()
            error_data = json.loads(error_response)

            if (
                error_data["type"] == "error"
                and error_data["code"] == "PERMISSION_DENIED"
            ):
                print("✅ Permission denied as expected")
                print(f"📝 Error message: {error_data['message']}")
            else:
                print(f"❌ Unexpected response: {error_data}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed unexpectedly: {e}")
    except ConnectionRefusedError:
        print("🔌 Server not running - cannot test permissions")
    except Exception as e:
        print(f"❌ Error during permission testing: {e}")


def print_usage_instructions():
    """Print instructions for running the demo."""
    print("\n📋 Usage Instructions")
    print("=" * 50)
    print("1. Start the FreeAgentics server:")
    print("   python main.py")
    print()
    print("2. Run this demo script:")
    print("   python examples/websocket_auth_demo.py")
    print()
    print("3. The demo will:")
    print("   - Create test users with different roles")
    print("   - Generate JWT tokens")
    print("   - Connect to WebSocket with authentication")
    print("   - Test various authenticated operations")
    print("   - Demonstrate permission-based access control")
    print("   - Show authentication failure scenarios")
    print()
    print("🔐 Security Features Demonstrated:")
    print("   - JWT token-based authentication")
    print("   - Role-based access control (RBAC)")
    print("   - Permission checking for commands")
    print("   - Proper error handling for auth failures")
    print("   - WebSocket close code 4001 for auth failures")


async def main():
    """Main demo function."""
    print_usage_instructions()

    print("\n🎬 Starting WebSocket Authentication Demo...")

    # Run demos
    await demo_websocket_authentication()
    await demo_authentication_failure()
    await demo_permission_testing()

    print("\n🎉 Demo completed!")
    print("\nFor more information, see:")
    print("- auth/security_implementation.py - JWT authentication")
    print("- api/v1/websocket.py - WebSocket authentication")
    print("- tests/unit/test_websocket_auth_enhanced.py - Unit tests")
    print(
        "- tests/integration/test_websocket_auth_integration.py - Integration tests"
    )


if __name__ == "__main__":
    asyncio.run(main())

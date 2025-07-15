"""
Integration tests for WebSocket authentication functionality.

Tests the complete WebSocket authentication flow including token validation,
connection establishment, and message handling with proper permissions.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, WebSocket
from fastapi.websockets import WebSocketDisconnect

from api.v1.websocket import handle_agent_command, handle_query, manager, websocket_auth
from auth.security_implementation import AuthenticationManager, Permission, TokenData, UserRole


@pytest.fixture
def auth_manager():
    """Create authentication manager with test users."""
    manager = AuthenticationManager()

    # Register test users
    admin_user = manager.register_user(
        username="admin_test",
        email="admin@test.com",
        password="admin_password",
        role=UserRole.ADMIN,
    )

    researcher_user = manager.register_user(
        username="researcher_test",
        email="researcher@test.com",
        password="researcher_password",
        role=UserRole.RESEARCHER,
    )

    observer_user = manager.register_user(
        username="observer_test",
        email="observer@test.com",
        password="observer_password",
        role=UserRole.OBSERVER,
    )

    return manager, admin_user, researcher_user, observer_user


@pytest.fixture
def valid_tokens(auth_manager):
    """Create valid JWT tokens for test users."""
    manager, admin_user, researcher_user, observer_user = auth_manager

    admin_token = manager.create_access_token(admin_user)
    researcher_token = manager.create_access_token(researcher_user)
    observer_token = manager.create_access_token(observer_user)

    return {"admin": admin_token, "researcher": researcher_token, "observer": observer_token}


class TestWebSocketAuthenticationIntegration:
    """Test complete WebSocket authentication integration."""

    @pytest.mark.asyncio
    async def test_websocket_auth_integration_without_token(self):
        """Test WebSocket authentication without token."""
        mock_websocket = AsyncMock(spec=WebSocket)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            await websocket_auth(mock_websocket, None)

        assert exc_info.value.code == 4001
        mock_websocket.close.assert_called_once_with(code=4001)

    @pytest.mark.asyncio
    async def test_websocket_auth_integration_with_invalid_token(self):
        """Test WebSocket authentication with invalid token."""
        mock_websocket = AsyncMock(spec=WebSocket)

        with patch("api.v1.websocket.auth_manager.verify_token") as mock_verify:
            mock_verify.side_effect = HTTPException(status_code=401, detail="Invalid token")

            with pytest.raises(WebSocketDisconnect) as exc_info:
                await websocket_auth(mock_websocket, "invalid.token")

            assert exc_info.value.code == 4001
            mock_websocket.close.assert_called_once_with(code=4001)

    @pytest.mark.asyncio
    async def test_websocket_auth_integration_with_valid_token(self, valid_tokens):
        """Test WebSocket authentication with valid token."""
        mock_websocket = AsyncMock(spec=WebSocket)

        # Create expected token data
        expected_token_data = TokenData(
            user_id="admin_user",
            username="admin_test",
            role=UserRole.ADMIN,
            permissions=[
                Permission.CREATE_AGENT,
                Permission.DELETE_AGENT,
                Permission.VIEW_AGENTS,
                Permission.MODIFY_AGENT,
                Permission.CREATE_COALITION,
                Permission.VIEW_METRICS,
                Permission.ADMIN_SYSTEM,
            ],
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        with patch("api.v1.websocket.auth_manager.verify_token") as mock_verify:
            mock_verify.return_value = expected_token_data

            result = await websocket_auth(mock_websocket, valid_tokens["admin"])

            assert result == expected_token_data
            assert result.username == "admin_test"
            assert result.role == UserRole.ADMIN
            mock_verify.assert_called_once_with(valid_tokens["admin"])
            mock_websocket.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_connection_manager_integration(self):
        """Test that authenticated connections are properly managed."""
        mock_websocket = AsyncMock(spec=WebSocket)
        client_id = "integration_test_client"

        # Prepare authenticated user metadata
        metadata = {
            "user_id": "test_user_123",
            "username": "integration_user",
            "role": "admin",
            "permissions": ["view_agents", "create_agent"],
            "authenticated": True,
        }

        # Test connection establishment
        await manager.connect(mock_websocket, client_id, metadata)

        # Verify connection was established with metadata
        assert client_id in manager.active_connections
        assert client_id in manager.connection_metadata
        assert manager.connection_metadata[client_id] == metadata
        assert manager.connection_metadata[client_id]["authenticated"] is True

        # Test subscription management
        manager.subscribe(client_id, "agent:status_update")
        assert "agent:status_update" in manager.subscriptions
        assert client_id in manager.subscriptions["agent:status_update"]

        # Test disconnection cleanup
        manager.disconnect(client_id)
        assert client_id not in manager.active_connections
        assert client_id not in manager.connection_metadata
        assert client_id not in manager.subscriptions.get("agent:status_update", set())


class TestWebSocketCommandAuthorizationIntegration:
    """Test WebSocket command handling with authentication integration."""

    @pytest.mark.asyncio
    async def test_agent_command_authorization_flow(self):
        """Test complete agent command authorization flow."""
        # Test admin user with full permissions
        admin_user = TokenData(
            user_id="admin_123",
            username="admin",
            role=UserRole.ADMIN,
            permissions=[
                Permission.CREATE_AGENT,
                Permission.DELETE_AGENT,
                Permission.VIEW_AGENTS,
                Permission.MODIFY_AGENT,
            ],
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        client_id = "admin_client"

        # Test create command (should succeed)
        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_agent_command(
                client_id, {"command": "create", "agent_id": "new_agent"}, admin_user
            )

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "agent_command_result"
            assert message["command"] == "create"
            assert message["status"] == "acknowledged"
            assert message["user"] == "admin"

        # Test delete command (should succeed)
        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_agent_command(
                client_id, {"command": "delete", "agent_id": "old_agent"}, admin_user
            )

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "agent_command_result"
            assert message["command"] == "delete"

    @pytest.mark.asyncio
    async def test_agent_command_permission_denied_flow(self):
        """Test agent command with insufficient permissions."""
        # Test observer user with limited permissions
        observer_user = TokenData(
            user_id="observer_123",
            username="observer",
            role=UserRole.OBSERVER,
            permissions=[Permission.VIEW_AGENTS],  # Only view permission
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        client_id = "observer_client"

        # Test create command (should fail)
        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_agent_command(
                client_id, {"command": "create", "agent_id": "new_agent"}, observer_user
            )

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "error"
            assert "Insufficient permissions" in message["message"]
            assert message["code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_query_authorization_flow(self):
        """Test complete query authorization flow."""
        # Test researcher user with appropriate permissions
        researcher_user = TokenData(
            user_id="researcher_123",
            username="researcher",
            role=UserRole.RESEARCHER,
            permissions=[Permission.VIEW_AGENTS, Permission.VIEW_METRICS],
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        client_id = "researcher_client"

        # Test agent status query (should succeed)
        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_query(client_id, {"query_type": "agent_status"}, researcher_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "query_result"
            assert message["query_type"] == "agent_status"
            assert message["data"]["user"] == "researcher"

        # Test world state query (should succeed)
        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_query(client_id, {"query_type": "world_state"}, researcher_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "query_result"
            assert message["query_type"] == "world_state"

    @pytest.mark.asyncio
    async def test_query_permission_denied_flow(self):
        """Test query with insufficient permissions."""
        # Test user with no permissions
        no_perm_user = TokenData(
            user_id="no_perm_123",
            username="no_permissions",
            role=UserRole.OBSERVER,
            permissions=[],  # No permissions
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        client_id = "no_perm_client"

        # Test agent status query (should fail)
        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_query(client_id, {"query_type": "agent_status"}, no_perm_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "error"
            assert "Insufficient permissions" in message["message"]
            assert message["code"] == "PERMISSION_DENIED"


class TestWebSocketSecurityIntegration:
    """Test security features integration."""

    @pytest.mark.asyncio
    async def test_token_validation_security_flow(self):
        """Test that token validation follows security best practices."""
        mock_websocket = AsyncMock(spec=WebSocket)

        # Test various invalid token scenarios
        invalid_tokens = [None, "", "malformed.token", "expired.jwt.token", "tampered.jwt.token"]

        for invalid_token in invalid_tokens:
            with patch("api.v1.websocket.auth_manager.verify_token") as mock_verify:
                if invalid_token:
                    mock_verify.side_effect = HTTPException(status_code=401, detail="Invalid token")

                with pytest.raises(WebSocketDisconnect) as exc_info:
                    await websocket_auth(mock_websocket, invalid_token)

                # All authentication failures should result in 4001 close code
                assert exc_info.value.code == 4001

    @pytest.mark.asyncio
    async def test_permission_escalation_prevention(self):
        """Test that permission escalation is prevented."""
        # Test user attempting to perform actions beyond their role
        limited_user = TokenData(
            user_id="limited_123",
            username="limited_user",
            role=UserRole.OBSERVER,
            permissions=[Permission.VIEW_AGENTS],  # Only basic view permission
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        client_id = "limited_client"

        # Attempt privileged operations that should fail
        privileged_commands = [
            {"command": "create", "agent_id": "test"},
            {"command": "delete", "agent_id": "test"},
            {"command": "modify", "agent_id": "test"},
        ]

        for command_data in privileged_commands:
            with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
                await handle_agent_command(client_id, command_data, limited_user)

                # Should receive permission denied
                mock_send.assert_called_once()
                args, _ = mock_send.call_args
                message = args[0]

                assert message["type"] == "error"
                assert message["code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_authenticated_metadata_security(self):
        """Test that authenticated metadata is properly secured."""
        # Test that connection metadata contains security information
        metadata = {
            "user_id": "secure_user_123",
            "username": "secure_user",
            "role": "researcher",
            "permissions": ["view_agents", "create_agent"],
            "authenticated": True,
        }

        mock_websocket = AsyncMock(spec=WebSocket)
        client_id = "secure_client"

        await manager.connect(mock_websocket, client_id, metadata)

        # Verify metadata is stored securely
        stored_metadata = manager.connection_metadata[client_id]
        assert stored_metadata["authenticated"] is True
        assert stored_metadata["user_id"] == "secure_user_123"
        assert stored_metadata["role"] == "researcher"
        assert isinstance(stored_metadata["permissions"], list)

        # Cleanup
        manager.disconnect(client_id)

    @pytest.mark.asyncio
    async def test_command_logging_integration(self):
        """Test that security-relevant commands are properly tracked."""
        admin_user = TokenData(
            user_id="admin_logging",
            username="admin_logger",
            role=UserRole.ADMIN,
            permissions=[Permission.CREATE_AGENT, Permission.DELETE_AGENT],
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        client_id = "logging_client"

        # Test that command responses include user information for audit trails
        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_agent_command(
                client_id, {"command": "create", "agent_id": "audit_agent"}, admin_user
            )

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            # Response should include user information for audit trail
            assert "user" in message
            assert message["user"] == "admin_logger"
            assert "timestamp" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

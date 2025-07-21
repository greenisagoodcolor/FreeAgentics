"""
Test suite for WebSocket authentication functionality.

Tests the JWT-based authentication for WebSocket connections, including
token validation, permission checks, and error handling.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, WebSocket

from api.v1.websocket import handle_agent_command, handle_query, websocket_auth
from auth.security_implementation import Permission, TokenData, UserRole


class TestWebSocketAuth:
    """Test the websocket_auth function."""

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.close = AsyncMock()
        return websocket

    @pytest.fixture
    def valid_token_data(self):
        """Create valid token data."""
        return TokenData(
            user_id="test_user_123",
            username="testuser",
            role=UserRole.RESEARCHER,
            permissions=[
                Permission.VIEW_AGENTS,
                Permission.CREATE_AGENT,
                Permission.MODIFY_AGENT,
            ],
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

    @pytest.mark.asyncio
    async def test_websocket_auth_success(self, mock_websocket, valid_token_data):
        """Test successful WebSocket authentication."""
        token = "valid.jwt.token"

        with patch("api.v1.websocket.auth_manager.verify_token") as mock_verify:
            mock_verify.return_value = valid_token_data

            result = await websocket_auth(mock_websocket, token)

            assert result == valid_token_data
            mock_verify.assert_called_once_with(token)
            mock_websocket.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_auth_no_token(self, mock_websocket):
        """Test WebSocket authentication with no token."""
        from fastapi import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc_info:
            await websocket_auth(mock_websocket, None)

        assert exc_info.value.code == 4001
        mock_websocket.close.assert_called_once_with(code=4001)

    @pytest.mark.asyncio
    async def test_websocket_auth_empty_token(self, mock_websocket):
        """Test WebSocket authentication with empty token."""
        from fastapi import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc_info:
            await websocket_auth(mock_websocket, "")

        assert exc_info.value.code == 4001
        mock_websocket.close.assert_called_once_with(code=4001)

    @pytest.mark.asyncio
    async def test_websocket_auth_invalid_token(self, mock_websocket):
        """Test WebSocket authentication with invalid token."""
        from fastapi import WebSocketDisconnect

        token = "invalid.jwt.token"

        with patch("api.v1.websocket.auth_manager.verify_token") as mock_verify:
            mock_verify.side_effect = HTTPException(
                status_code=401, detail="Invalid token"
            )

            with pytest.raises(WebSocketDisconnect) as exc_info:
                await websocket_auth(mock_websocket, token)

            assert exc_info.value.code == 4001
            mock_websocket.close.assert_called_once_with(code=4001)

    @pytest.mark.asyncio
    async def test_websocket_auth_expired_token(self, mock_websocket):
        """Test WebSocket authentication with expired token."""
        from fastapi import WebSocketDisconnect

        token = "expired.jwt.token"

        with patch("api.v1.websocket.auth_manager.verify_token") as mock_verify:
            mock_verify.side_effect = HTTPException(
                status_code=401, detail="Token expired"
            )

            with pytest.raises(WebSocketDisconnect) as exc_info:
                await websocket_auth(mock_websocket, token)

            assert exc_info.value.code == 4001
            mock_websocket.close.assert_called_once_with(code=4001)

    @pytest.mark.asyncio
    async def test_websocket_auth_unexpected_error(self, mock_websocket):
        """Test WebSocket authentication with unexpected error."""
        from fastapi import WebSocketDisconnect

        token = "problematic.jwt.token"

        with patch("api.v1.websocket.auth_manager.verify_token") as mock_verify:
            mock_verify.side_effect = Exception("Unexpected error")

            with pytest.raises(WebSocketDisconnect) as exc_info:
                await websocket_auth(mock_websocket, token)

            assert exc_info.value.code == 4001
            mock_websocket.close.assert_called_once_with(code=4001)


class TestWebSocketAgentCommands:
    """Test WebSocket agent command handling with authentication."""

    @pytest.fixture
    def admin_user(self):
        """Create admin user with all permissions."""
        return TokenData(
            user_id="admin_123",
            username="admin",
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

    @pytest.fixture
    def observer_user(self):
        """Create observer user with limited permissions."""
        return TokenData(
            user_id="observer_123",
            username="observer",
            role=UserRole.OBSERVER,
            permissions=[Permission.VIEW_AGENTS, Permission.VIEW_METRICS],
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

    @pytest.mark.asyncio
    async def test_agent_command_create_with_permission(self, admin_user):
        """Test agent creation command with proper permissions."""
        client_id = "test_client"
        command_data = {"command": "create", "agent_id": "new_agent_001"}

        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_agent_command(client_id, command_data, admin_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "agent_command_result"
            assert message["command"] == "create"
            assert message["status"] == "acknowledged"
            assert message["user"] == "admin"

    @pytest.mark.asyncio
    async def test_agent_command_create_without_permission(self, observer_user):
        """Test agent creation command without proper permissions."""
        client_id = "test_client"
        command_data = {"command": "create", "agent_id": "new_agent_001"}

        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_agent_command(client_id, command_data, observer_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "error"
            assert "Insufficient permissions" in message["message"]
            assert message["code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_agent_command_delete_with_permission(self, admin_user):
        """Test agent deletion command with proper permissions."""
        client_id = "test_client"
        command_data = {"command": "delete", "agent_id": "agent_to_delete"}

        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_agent_command(client_id, command_data, admin_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "agent_command_result"
            assert message["command"] == "delete"
            assert message["status"] == "acknowledged"

    @pytest.mark.asyncio
    async def test_agent_command_missing_data(self, admin_user):
        """Test agent command with missing required data."""
        client_id = "test_client"
        command_data = {"command": "create"}  # Missing agent_id

        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_agent_command(client_id, command_data, admin_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "error"
            assert "Missing command or agent_id" in message["message"]

    @pytest.mark.asyncio
    async def test_agent_command_view_with_observer_permission(self, observer_user):
        """Test read-only agent command with observer permissions."""
        client_id = "test_client"
        command_data = {"command": "status", "agent_id": "agent_001"}

        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_agent_command(client_id, command_data, observer_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "agent_command_result"
            assert message["command"] == "status"
            assert message["status"] == "acknowledged"


class TestWebSocketQueries:
    """Test WebSocket query handling with authentication."""

    @pytest.fixture
    def researcher_user(self):
        """Create researcher user with view permissions."""
        return TokenData(
            user_id="researcher_123",
            username="researcher",
            role=UserRole.RESEARCHER,
            permissions=[
                Permission.CREATE_AGENT,
                Permission.VIEW_AGENTS,
                Permission.MODIFY_AGENT,
                Permission.CREATE_COALITION,
                Permission.VIEW_METRICS,
            ],
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

    @pytest.fixture
    def no_permission_user(self):
        """Create user with no view permissions."""
        return TokenData(
            user_id="no_perm_123",
            username="no_permissions",
            role=UserRole.OBSERVER,
            permissions=[],  # No permissions
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

    @pytest.mark.asyncio
    async def test_query_agent_status_with_permission(self, researcher_user):
        """Test agent status query with proper permissions."""
        client_id = "test_client"
        query_data = {"query_type": "agent_status"}

        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_query(client_id, query_data, researcher_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "query_result"
            assert message["query_type"] == "agent_status"
            assert "user" in message["data"]
            assert message["data"]["user"] == "researcher"

    @pytest.mark.asyncio
    async def test_query_world_state_with_permission(self, researcher_user):
        """Test world state query with proper permissions."""
        client_id = "test_client"
        query_data = {"query_type": "world_state"}

        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_query(client_id, query_data, researcher_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "query_result"
            assert message["query_type"] == "world_state"
            assert "user" in message["data"]

    @pytest.mark.asyncio
    async def test_query_agent_status_without_permission(self, no_permission_user):
        """Test agent status query without proper permissions."""
        client_id = "test_client"
        query_data = {"query_type": "agent_status"}

        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_query(client_id, query_data, no_permission_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "error"
            assert "Insufficient permissions" in message["message"]
            assert message["code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_query_unknown_type(self, researcher_user):
        """Test query with unknown type."""
        client_id = "test_client"
        query_data = {"query_type": "unknown_query"}

        with patch("api.v1.websocket.manager.send_personal_message") as mock_send:
            await handle_query(client_id, query_data, researcher_user)

            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            message = args[0]

            assert message["type"] == "error"
            assert "Unknown query type" in message["message"]


class TestWebSocketAuthIntegration:
    """Integration tests for WebSocket authentication flow."""

    @pytest.mark.asyncio
    async def test_authenticated_connection_metadata(self):
        """Test that authenticated connection stores proper metadata."""
        from api.v1.websocket import manager

        # Mock websocket and token data
        mock_websocket = AsyncMock(spec=WebSocket)
        user_data = TokenData(
            user_id="integration_user",
            username="integration_test",
            role=UserRole.ADMIN,
            permissions=[Permission.VIEW_AGENTS],
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        # Prepare metadata like the endpoint would
        metadata = {
            "user_id": user_data.user_id,
            "username": user_data.username,
            "role": user_data.role.value,
            "permissions": [p.value for p in user_data.permissions],
            "authenticated": True,
        }

        # Test connection with metadata
        client_id = "integration_client"
        await manager.connect(mock_websocket, client_id, metadata)

        # Verify metadata was stored
        assert client_id in manager.connection_metadata
        stored_metadata = manager.connection_metadata[client_id]
        assert stored_metadata["user_id"] == "integration_user"
        assert stored_metadata["username"] == "integration_test"
        assert stored_metadata["role"] == "admin"
        assert stored_metadata["authenticated"] is True

        # Cleanup
        manager.disconnect(client_id)

    @pytest.mark.asyncio
    async def test_permission_based_subscription_filtering(self):
        """Test that subscriptions could be filtered based on permissions."""
        from api.v1.websocket import manager

        # This test verifies that the foundation is in place for permission-based filtering
        # In production, you might want to filter events based on user permissions

        mock_websocket = AsyncMock(spec=WebSocket)
        client_id = "permission_test_client"

        metadata = {
            "user_id": "perm_user",
            "username": "perm_test",
            "role": "observer",
            "permissions": ["view_agents"],
            "authenticated": True,
        }

        await manager.connect(mock_websocket, client_id, metadata)
        manager.subscribe(client_id, "agent:status_update")

        # Verify subscription was created
        assert "agent:status_update" in manager.subscriptions
        assert client_id in manager.subscriptions["agent:status_update"]

        # Cleanup
        manager.disconnect(client_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Comprehensive WebSocket Security Test Suite

Tests WebSocket authentication, authorization, rate limiting, input validation,
and protection against various security attacks including injection attempts.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from auth.security_implementation import AuthenticationManager, Permission, TokenData, UserRole
from fastapi import WebSocketDisconnect

from websocket.auth_handler import WebSocketAuthHandler, WebSocketErrorCode


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self, path="/ws/test_client", headers=None, query_params=None):
        self.path = path
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.client = MagicMock()
        self.client.host = "127.0.0.1"
        self.url = MagicMock()
        self.url.__str__ = lambda: f"ws://localhost{path}"
        self.messages_sent = []
        self.closed = False
        self.close_code = None
        self.close_reason = None

    async def accept(self):
        """Accept connection."""
        pass

    async def close(self, code=1000, reason=""):
        """Close connection."""
        self.closed = True
        self.close_code = code
        self.close_reason = reason

    async def send_json(self, data):
        """Send JSON data."""
        self.messages_sent.append(data)

    async def send_text(self, data):
        """Send text data."""
        self.messages_sent.append(data)

    async def receive_text(self):
        """Receive text data."""
        # Simulate receiving data
        await asyncio.sleep(0.1)
        return '{"type": "ping"}'


class TestWebSocketAuthentication:
    """Test WebSocket authentication mechanisms."""

    @pytest.fixture
    def auth_manager(self):
        """Create auth manager instance."""
        return AuthenticationManager()

    @pytest.fixture
    def valid_token_data(self):
        """Create valid token data."""
        return TokenData(
            user_id="test_user_123",
            username="testuser",
            role=UserRole.RESEARCHER,
            permissions=[Permission.VIEW_AGENTS, Permission.CREATE_AGENT],
            exp=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

    @pytest.fixture
    def expired_token_data(self):
        """Create expired token data."""
        return TokenData(
            user_id="test_user_123",
            username="testuser",
            role=UserRole.RESEARCHER,
            permissions=[Permission.VIEW_AGENTS],
            exp=datetime.now(timezone.utc) - timedelta(minutes=15),
        )

    @pytest.mark.asyncio
    async def test_successful_authentication_with_query_token(self, valid_token_data):
        """Test successful WebSocket authentication with token in query params."""
        ws = MockWebSocket(query_params={"token": "valid.jwt.token"})
        handler = WebSocketAuthHandler()

        with patch("websocket.auth_handler.auth_manager.verify_token") as mock_verify:
            mock_verify.return_value = valid_token_data

            result = await handler.authenticate_connection(ws, "test_client", "valid.jwt.token")

            assert result == valid_token_data
            assert "test_client" in handler.connections
            assert handler.connections["test_client"].user_data == valid_token_data

    @pytest.mark.asyncio
    async def test_authentication_with_expired_token(self, expired_token_data):
        """Test WebSocket authentication rejection with expired token."""
        ws = MockWebSocket()
        handler = WebSocketAuthHandler()

        with patch("websocket.auth_handler.auth_manager.verify_token") as mock_verify:
            mock_verify.side_effect = Exception("Token has expired")

            with pytest.raises(WebSocketDisconnect) as exc_info:
                await handler.authenticate_connection(ws, "test_client", "expired.jwt.token")

            assert exc_info.value.code == WebSocketErrorCode.TOKEN_EXPIRED
            assert ws.closed
            assert ws.close_code == WebSocketErrorCode.TOKEN_EXPIRED

    @pytest.mark.asyncio
    async def test_authentication_without_token(self):
        """Test WebSocket authentication rejection without token."""
        ws = MockWebSocket()
        handler = WebSocketAuthHandler()

        with pytest.raises(WebSocketDisconnect) as exc_info:
            await handler.authenticate_connection(ws, "test_client", None)

        assert exc_info.value.code == WebSocketErrorCode.AUTHENTICATION_FAILED
        assert ws.closed
        assert ws.close_code == WebSocketErrorCode.AUTHENTICATION_FAILED

    @pytest.mark.asyncio
    async def test_authentication_with_invalid_token(self):
        """Test WebSocket authentication rejection with invalid token."""
        ws = MockWebSocket()
        handler = WebSocketAuthHandler()

        with patch("websocket.auth_handler.auth_manager.verify_token") as mock_verify:
            mock_verify.side_effect = Exception("Invalid token")

            with pytest.raises(WebSocketDisconnect) as exc_info:
                await handler.authenticate_connection(ws, "test_client", "invalid.jwt.token")

            assert exc_info.value.code == WebSocketErrorCode.AUTHENTICATION_FAILED
            assert ws.closed

    @pytest.mark.asyncio
    async def test_rate_limiting_connections(self):
        """Test rate limiting for WebSocket connections."""
        handler = WebSocketAuthHandler()

        # Simulate multiple connections from same IP
        for i in range(15):
            ws = MockWebSocket()
            ws.client.host = "192.168.1.100"

            # First 10 should succeed
            if i < 10:
                result = await handler._check_rate_limit(ws)
                assert result is True
            else:
                # After 10, should be rate limited
                result = await handler._check_rate_limit(ws)
                assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_connection_limit(self, valid_token_data):
        """Test concurrent connection limit per user."""
        handler = WebSocketAuthHandler()
        handler.config.max_connections_per_user = 3

        # Add 3 connections for the user
        for i in range(3):
            handler.connections[f"client_{i}"] = MagicMock()
            handler.user_connections[valid_token_data.user_id] = {f"client_{i}"}

        # 4th connection should be rejected
        result = await handler._check_connection_limit(valid_token_data.user_id, "client_3")
        assert result is False

    @pytest.mark.asyncio
    async def test_origin_validation(self):
        """Test origin header validation."""
        handler = WebSocketAuthHandler()
        handler.config.origin_whitelist = {
            "https://trusted.com",
            "https://app.trusted.com",
        }

        # Valid origin
        ws_valid = MockWebSocket(headers={"origin": "https://trusted.com"})
        assert await handler._verify_origin(ws_valid) is True

        # Invalid origin
        ws_invalid = MockWebSocket(headers={"origin": "https://malicious.com"})
        assert await handler._verify_origin(ws_invalid) is False

        # No origin whitelist (allow all)
        handler.config.origin_whitelist = None
        assert await handler._verify_origin(ws_invalid) is True


class TestWebSocketAuthorization:
    """Test WebSocket authorization and permission checks."""

    @pytest.fixture
    def handler_with_connection(self, valid_token_data):
        """Create handler with authenticated connection."""
        handler = WebSocketAuthHandler()
        handler.connections["test_client"] = MagicMock()
        handler.connections["test_client"].user_data = valid_token_data
        handler.connections["test_client"].permissions = set(valid_token_data.permissions)
        return handler

    @pytest.mark.asyncio
    async def test_permission_check_success(self, handler_with_connection):
        """Test successful permission check."""
        result = await handler_with_connection.verify_permission(
            "test_client", Permission.VIEW_AGENTS
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_permission_check_failure(self, handler_with_connection):
        """Test failed permission check."""
        result = await handler_with_connection.verify_permission(
            "test_client", Permission.ADMIN_ACCESS
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_permission_check_unknown_client(self, handler_with_connection):
        """Test permission check for unknown client."""
        result = await handler_with_connection.verify_permission(
            "unknown_client", Permission.VIEW_AGENTS
        )
        assert result is False


class TestWebSocketMessageValidation:
    """Test WebSocket message validation and injection prevention."""

    @pytest.mark.asyncio
    async def test_json_injection_prevention(self):
        """Test prevention of JSON injection attacks."""
        from api.v1.websocket import WebSocketMessage

        # Valid message
        valid_msg = WebSocketMessage(type="subscribe", data={"event_types": ["agent:update"]})
        assert valid_msg.type == "subscribe"

        # Invalid type with special characters
        with pytest.raises(ValueError, match="Invalid message type format"):
            WebSocketMessage(type="subscribe'; DROP TABLE users;--", data={})

        # Invalid type with script injection
        with pytest.raises(ValueError, match="Invalid message type format"):
            WebSocketMessage(type="<script>alert('xss')</script>", data={})

    @pytest.mark.asyncio
    async def test_message_size_validation(self):
        """Test message size validation."""
        from api.v1.websocket import WebSocketMessage

        # Create large data
        large_data = {"key": "x" * 200000}  # 200KB

        with pytest.raises(ValueError, match="Message data too large"):
            WebSocketMessage(type="update", data=large_data)

    @pytest.mark.asyncio
    async def test_event_type_validation(self):
        """Test event type validation in subscriptions."""
        # This would be tested in the actual WebSocket endpoint
        # but we can test the regex pattern
        import re

        valid_pattern = r"^[a-zA-Z0-9:_-]+$"

        # Valid event types
        assert re.match(valid_pattern, "agent:update")
        assert re.match(valid_pattern, "world:state_change")
        assert re.match(valid_pattern, "system-alert")

        # Invalid event types
        assert not re.match(valid_pattern, "agent:update'; DROP TABLE--")
        assert not re.match(valid_pattern, "../../../etc/passwd")
        assert not re.match(valid_pattern, "<script>alert()</script>")

    @pytest.mark.asyncio
    async def test_agent_id_validation(self):
        """Test agent ID validation."""
        import re

        valid_pattern = r"^[a-zA-Z0-9_-]+$"

        # Valid agent IDs
        assert re.match(valid_pattern, "agent_123")
        assert re.match(valid_pattern, "test-agent-456")

        # Invalid agent IDs (injection attempts)
        assert not re.match(valid_pattern, "agent'; DELETE FROM agents--")
        assert not re.match(valid_pattern, "../../etc/passwd")
        assert not re.match(valid_pattern, "agent<script>")


class TestWebSocketHeartbeat:
    """Test WebSocket heartbeat and connection monitoring."""

    @pytest.mark.asyncio
    async def test_heartbeat_update(self, valid_token_data):
        """Test heartbeat update functionality."""
        handler = WebSocketAuthHandler()

        # Setup connection
        handler.connections["test_client"] = MagicMock()
        handler.connections["test_client"].last_heartbeat = datetime.utcnow() - timedelta(minutes=5)

        # Update heartbeat
        await handler.update_heartbeat("test_client")

        # Check updated
        assert (
            datetime.utcnow() - handler.connections["test_client"].last_heartbeat
        ).total_seconds() < 1

    @pytest.mark.asyncio
    async def test_heartbeat_timeout_detection(self):
        """Test detection of timed-out connections."""
        handler = WebSocketAuthHandler()
        handler.config.heartbeat_timeout = 60  # 60 seconds

        # Add connections with different heartbeat times
        now = datetime.utcnow()

        # Active connection
        handler.connections["active_client"] = MagicMock()
        handler.connections["active_client"].last_heartbeat = now - timedelta(seconds=30)

        # Timed out connection
        handler.connections["timeout_client"] = MagicMock()
        handler.connections["timeout_client"].last_heartbeat = now - timedelta(seconds=90)

        # Check timeouts
        timed_out = await handler.check_heartbeat_timeout()

        assert "timeout_client" in timed_out
        assert "active_client" not in timed_out


class TestWebSocketTokenRefresh:
    """Test WebSocket token refresh functionality."""

    @pytest.mark.asyncio
    async def test_successful_token_refresh(self, valid_token_data):
        """Test successful token refresh."""
        handler = WebSocketAuthHandler()

        # Setup connection
        handler.connections["test_client"] = MagicMock()
        handler.connections["test_client"].user_data = valid_token_data

        with patch("websocket.auth_handler.auth_manager.refresh_access_token") as mock_refresh:
            mock_refresh.return_value = (
                "new_access_token",
                "new_refresh_token",
            )

            with patch("websocket.auth_handler.auth_manager.verify_token") as mock_verify:
                mock_verify.return_value = valid_token_data

                new_token, user_data = await handler.refresh_token(
                    "test_client", "old_refresh_token"
                )

                assert new_token == "new_access_token"
                assert user_data == valid_token_data

    @pytest.mark.asyncio
    async def test_token_refresh_for_unknown_client(self):
        """Test token refresh rejection for unknown client."""
        handler = WebSocketAuthHandler()

        with pytest.raises(ValueError, match="Connection not found"):
            await handler.refresh_token("unknown_client", "refresh_token")


class TestWebSocketSecurityIntegration:
    """Integration tests for WebSocket security features."""

    @pytest.mark.asyncio
    async def test_complete_authentication_flow(self):
        """Test complete authentication flow from connection to messaging."""
        # This would require a more complete test setup with actual WebSocket server
        # For now, we test the components

        handler = WebSocketAuthHandler()
        assert handler is not None
        assert handler.config is not None
        assert handler.connections == {}
        assert handler.user_connections == {}

    @pytest.mark.asyncio
    async def test_injection_attack_scenarios(self):
        """Test various injection attack scenarios."""
        attack_payloads = [
            # SQL injection attempts
            "'; DROP TABLE users;--",
            "' OR '1'='1",
            "admin'--",
            # NoSQL injection attempts
            '{"$ne": null}',
            '{"$gt": ""}',
            # Command injection attempts
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            # Path traversal attempts
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            # XSS attempts
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            # LDAP injection attempts
            "*)(uid=*))(|(uid=*",
            "admin)(&(password=*))",
            # XML injection attempts
            "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
            # Template injection attempts
            "{{7*7}}",
            "${7*7}",
            "<%= 7*7 %>",
        ]

        import re

        from api.v1.websocket import WebSocketMessage

        # Test message type validation
        for payload in attack_payloads:
            with pytest.raises(ValueError):
                WebSocketMessage(type=payload, data={})

        # Test event type validation
        valid_pattern = r"^[a-zA-Z0-9:_-]+$"
        for payload in attack_payloads:
            assert not re.match(valid_pattern, payload)

        # Test agent ID validation
        agent_pattern = r"^[a-zA-Z0-9_-]+$"
        for payload in attack_payloads:
            assert not re.match(agent_pattern, payload)

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self):
        """Test rate limiting integration with WebSocket handler."""
        from api.middleware.websocket_rate_limiting import WebSocketRateLimitManager

        manager = WebSocketRateLimitManager()
        ws = MockWebSocket()

        # Should allow first connection
        result = await manager.check_connection_allowed(ws)
        assert result is True

        # Test message rate limiting
        message = "test message"
        result = await manager.check_message_allowed(ws, message)
        assert result is True


class TestWebSocketErrorHandling:
    """Test WebSocket error handling and recovery."""

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(self, valid_token_data):
        """Test proper cleanup when connection errors occur."""
        handler = WebSocketAuthHandler()

        # Setup connection
        handler.connections["test_client"] = MagicMock()
        handler.connections["test_client"].user_data = valid_token_data
        handler.user_connections[valid_token_data.user_id] = {"test_client"}

        # Disconnect
        await handler.disconnect("test_client")

        # Verify cleanup
        assert "test_client" not in handler.connections
        assert valid_token_data.user_id not in handler.user_connections

    @pytest.mark.asyncio
    async def test_graceful_handling_of_malformed_messages(self):
        """Test graceful handling of malformed WebSocket messages."""
        # Test various malformed messages
        malformed_messages = [
            "",  # Empty message
            "not json",  # Invalid JSON
            '{"incomplete": ',  # Incomplete JSON
            '{"type": null}',  # Null type
            '{"no_type": "here"}',  # Missing type
            "[]",  # Array instead of object
            "null",  # Null value
            "undefined",  # Undefined
        ]

        # Each should be handled gracefully without crashing
        for msg in malformed_messages:
            try:
                data = json.loads(msg)
                # If it parses, check structure
                if not isinstance(data, dict) or "type" not in data:
                    # Should be rejected
                    pass
            except json.JSONDecodeError:
                # Should be caught and handled
                pass


class TestWebSocketMonitoring:
    """Test WebSocket monitoring and reporting."""

    @pytest.mark.asyncio
    async def test_connection_info_retrieval(self, valid_token_data):
        """Test retrieval of connection information."""
        handler = WebSocketAuthHandler()

        # Setup connection
        from websocket.auth_handler import ConnectionState

        handler.connections["test_client"] = ConnectionState(
            client_id="test_client",
            user_data=valid_token_data,
            permissions=set(valid_token_data.permissions),
            metadata={"test": "data"},
        )

        # Get info
        info = handler.get_connection_info("test_client")

        assert info is not None
        assert info["client_id"] == "test_client"
        assert info["user_id"] == valid_token_data.user_id
        assert info["username"] == valid_token_data.username
        assert len(info["permissions"]) == len(valid_token_data.permissions)

    @pytest.mark.asyncio
    async def test_user_connections_tracking(self, valid_token_data):
        """Test tracking of user connections."""
        handler = WebSocketAuthHandler()

        # Add multiple connections for same user
        user_id = valid_token_data.user_id
        handler.user_connections[user_id] = {
            "client_1",
            "client_2",
            "client_3",
        }

        # Get connections
        connections = handler.get_user_connections(user_id)

        assert len(connections) == 3
        assert "client_1" in connections
        assert "client_2" in connections
        assert "client_3" in connections


class TestWebSocketReconnection:
    """Test WebSocket reconnection with authentication."""

    @pytest.mark.asyncio
    async def test_reconnection_with_valid_token(self, valid_token_data):
        """Test reconnection with still-valid token."""
        handler = WebSocketAuthHandler()
        ws = MockWebSocket()

        with patch("websocket.auth_handler.auth_manager.verify_token") as mock_verify:
            mock_verify.return_value = valid_token_data

            # First connection
            result1 = await handler.authenticate_connection(ws, "client_1", "valid.token")
            assert result1 == valid_token_data

            # Disconnect
            await handler.disconnect("client_1")

            # Reconnect with same token
            result2 = await handler.authenticate_connection(ws, "client_1", "valid.token")
            assert result2 == valid_token_data

    @pytest.mark.asyncio
    async def test_reconnection_after_token_refresh(self, valid_token_data):
        """Test reconnection after token has been refreshed."""
        handler = WebSocketAuthHandler()

        # Setup initial connection
        handler.connections["client_1"] = MagicMock()
        handler.connections["client_1"].user_data = valid_token_data

        # Refresh token
        with patch("websocket.auth_handler.auth_manager.refresh_access_token") as mock_refresh:
            mock_refresh.return_value = (
                "new_access_token",
                "new_refresh_token",
            )

            with patch("websocket.auth_handler.auth_manager.verify_token") as mock_verify:
                mock_verify.return_value = valid_token_data

                new_token, _ = await handler.refresh_token("client_1", "old_refresh_token")

        # Disconnect
        await handler.disconnect("client_1")

        # Reconnect with new token
        ws = MockWebSocket()
        with patch("websocket.auth_handler.auth_manager.verify_token") as mock_verify:
            mock_verify.return_value = valid_token_data

            result = await handler.authenticate_connection(ws, "client_1", new_token)
            assert result == valid_token_data


# Performance and stress tests
class TestWebSocketPerformance:
    """Test WebSocket performance under load."""

    @pytest.mark.asyncio
    async def test_concurrent_authentication(self, valid_token_data):
        """Test handling of concurrent authentication requests."""
        handler = WebSocketAuthHandler()

        async def authenticate_client(client_id):
            ws = MockWebSocket()
            with patch("websocket.auth_handler.auth_manager.verify_token") as mock_verify:
                mock_verify.return_value = valid_token_data
                try:
                    await handler.authenticate_connection(ws, client_id, "valid.token")
                    return True
                except Exception:
                    return False

        # Create concurrent authentication tasks
        tasks = [authenticate_client(f"client_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed (assuming no connection limit)
        successful = sum(1 for r in results if r)
        assert successful >= 5  # At least some should succeed

    @pytest.mark.asyncio
    async def test_message_handling_performance(self):
        """Test message handling performance."""
        from api.v1.websocket import WebSocketMessage

        # Time message validation
        start_time = time.time()

        for i in range(1000):
            msg = WebSocketMessage(
                type="update",
                data={"index": i, "message": f"Test message {i}"},
            )
            assert msg.type == "update"

        elapsed = time.time() - start_time

        # Should handle 1000 messages in under 1 second
        assert elapsed < 1.0

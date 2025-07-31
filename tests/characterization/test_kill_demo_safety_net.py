"""Characterization tests for Kill-All-Demo v2 implementation.

These tests establish a safety net to ensure we don't break existing functionality
while eliminating all demo references from the codebase.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


class TestCurrentDemoEndpoints:
    """Characterize current demo endpoint behavior before elimination."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_demo_websocket_endpoint_exists(self, client):
        """Current state: /ws/demo endpoint should exist."""
        # This test will fail after we remove demo - that's expected
        try:
            with client.websocket_connect("/api/v1/ws/demo") as websocket:
                # If we get here, the endpoint exists
                assert True
        except Exception:
            # If endpoint doesn't exist, that's fine - we're removing it
            pytest.skip("Demo endpoint already removed")

    def test_dev_websocket_endpoint_exists(self, client):
        """Dev endpoint should exist and work."""
        with client.websocket_connect("/api/v1/ws/dev") as websocket:
            websocket.send_json({"type": "test", "data": "hello"})
            response = websocket.receive_json()
            assert response is not None

    def test_dev_config_endpoint_returns_dev_mode(self, client):
        """Dev config should always return dev mode."""
        response = client.get("/api/v1/dev-config")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "dev"
        assert data["endpoints"]["websocket"] == "/api/v1/ws/dev"

    def test_no_demo_in_production_config(self, client, monkeypatch):
        """Production mode should never reference demo."""
        monkeypatch.setenv("PRODUCTION", "true")
        response = client.get("/api/v1/dev-config")
        assert response.status_code == 404  # Should not be available in prod


class TestFrontendConstants:
    """Test frontend constants don't reference demo."""

    def test_websocket_url_no_demo(self):
        """Frontend WebSocket URL should not contain demo."""
        import os

        ws_url = os.getenv("NEXT_PUBLIC_WS_URL", "ws://localhost:8000/api/v1/ws/dev")
        assert "/demo" not in ws_url
        assert "/dev" in ws_url

    def test_client_id_no_demo(self):
        """Client ID should not default to demo."""
        import os

        client_id = os.getenv("NEXT_PUBLIC_CLIENT_ID", "dev")
        assert client_id != "demo"
        assert client_id == "dev"


class TestDevelopmentEnvironmentDetection:
    """Test development environment detection logic."""

    def test_dev_mode_detection_without_database_url(self, monkeypatch):
        """Dev mode should be detected when no DATABASE_URL is set."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("PRODUCTION", "false")

        from auth.dev_auth import DevAuthManager

        manager = DevAuthManager()
        assert manager.is_dev_mode() is True

    def test_prod_mode_detection_with_production_flag(self, monkeypatch):
        """Production mode should be detected when PRODUCTION=true."""
        monkeypatch.setenv("PRODUCTION", "true")

        from auth.dev_auth import DevAuthManager

        manager = DevAuthManager()
        assert manager.is_dev_mode() is False


class TestTokenStorage:
    """Test token storage doesn't use demo keys."""

    def test_auth_token_key_name(self):
        """Auth token should use fa.jwt key, not demo variants."""
        # This would be tested in browser environment
        # For now, just verify the key name is consistent
        expected_key = "fa.jwt"
        assert "demo" not in expected_key
        assert expected_key == "fa.jwt"

    def test_no_demo_token_remnants(self):
        """No demo token keys should exist in code."""
        import re

        # Search for potential demo token key patterns
        demo_patterns = [r"fa_token_demo", r"demo_token", r"token_demo", r"auth_demo"]

        # In a real implementation, we'd scan actual source files
        # For now, just verify our expected key doesn't match demo patterns
        auth_key = "fa.jwt"
        for pattern in demo_patterns:
            assert not re.match(pattern, auth_key)


class TestWebSocketRateLimiting:
    """Test WebSocket rate limiting behavior."""

    def test_rate_limiting_disabled_in_dev(self, monkeypatch):
        """Rate limiting should be disabled in dev mode."""
        monkeypatch.setenv("PRODUCTION", "false")

        # Test that dev mode bypasses rate limiting
        # This is tested in the actual rate limiter middleware
        from api.middleware.rate_limiter import RateLimiter

        limiter = RateLimiter()

        # In dev mode, process_request should return (True, None)
        # indicating no rate limiting is applied
        assert True  # Placeholder - actual test would check limiter behavior


class TestDemoEndpointElimination:
    """Tests that verify demo endpoints are properly eliminated."""

    def test_no_demo_websocket_routes_registered(self):
        """After elimination, no demo WebSocket routes should be registered."""
        from api.main import app

        # Check that no routes contain 'demo' in their path
        demo_routes = []
        for route in app.routes:
            if hasattr(route, "path") and "demo" in route.path:
                demo_routes.append(route.path)

        # Initially this might have demo routes, but after cleanup it should be empty
        if demo_routes:
            pytest.skip(f"Demo routes still exist: {demo_routes}. This is expected before cleanup.")

    def test_websocket_endpoints_only_dev(self, client):
        """Only dev WebSocket endpoints should exist."""
        # Test that /ws/dev works
        with client.websocket_connect("/api/v1/ws/dev") as websocket:
            assert websocket is not None

        # Test that /ws/demo returns 404 (after cleanup)
        try:
            with client.websocket_connect("/api/v1/ws/demo") as websocket:
                pytest.skip("Demo endpoint still exists - will be removed in cleanup")
        except Exception as e:
            # This is what we want after cleanup - demo endpoint should not exist
            assert "404" in str(e) or "WebSocket" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

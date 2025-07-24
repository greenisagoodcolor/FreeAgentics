"""Test health check endpoint following TDD principles.

This test file MUST fail first (RED phase), then we implement the health
endpoint to make it pass (GREEN phase), then refactor if needed.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

# Set required environment variables for testing
test_env = {
    "DATABASE_URL": "sqlite:///test.db",  # Use SQLite for testing to avoid PostgreSQL connection issues
    "API_KEY": "test-api-key",
    "SECRET_KEY": "test-secret-key-that-is-at-least-32-characters-long",
    "ENVIRONMENT": "testing",
    "TESTING": "true",  # Enable testing mode
}


class TestHealthCheckEndpoint:
    """Test suite for health check endpoint.

    Following TDD RED-GREEN-REFACTOR cycle:
    1. These tests will fail initially (RED)
    2. We implement minimal code to pass (GREEN)
    3. We refactor if needed (REFACTOR)
    """

    @pytest.fixture
    def client(self):
        """Create test client with mocked environment."""
        with patch.dict(os.environ, test_env, clear=False):
            from api.main import app

            from database.session import get_db

            # Override the get_db dependency with a mock
            def override_get_db():
                mock_session = MagicMock()
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db
            client = TestClient(app)

            # Cleanup after test
            yield client
            app.dependency_overrides.clear()

    def test_health_endpoint_exists(self, client):
        """Test that /health endpoint exists and responds."""
        response = client.get("/health")
        # Should not return 404
        assert response.status_code != 404

    def test_health_returns_200_when_db_connected(self, client):
        """Test that health returns 200 when database is connected."""
        # The client fixture already has a mocked database that simulates success
        # Override it to simulate a successful connection
        from api.main import app

        from database.session import get_db

        def mock_db_success():
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_session.execute.return_value = mock_result
            yield mock_session

        app.dependency_overrides[get_db] = mock_db_success

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "db" in data
        assert data["db"] == "connected"

    def test_health_returns_503_when_db_down(self, client):
        """Test that health returns 503 when database is down."""
        # Mock database to raise OperationalError
        from api.main import app

        from database.session import get_db

        def mock_db_failure():
            mock_session = MagicMock()
            mock_session.execute.side_effect = OperationalError(
                "Database connection failed", None, None
            )
            yield mock_session

        app.dependency_overrides[get_db] = mock_db_failure

        response = client.get("/health")
        assert response.status_code == 503

        data = response.json()
        assert "status" in data
        assert data["status"] == "unhealthy"
        assert "db" in data
        assert data["db"] == "disconnected"
        assert "error" in data

    def test_health_performs_actual_db_query(self, client):
        """Test that health check performs actual SELECT 1 query."""
        from api.main import app

        from database.session import get_db

        # Create a mock that we can inspect after the call
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        def mock_db_queryable():
            yield mock_session

        app.dependency_overrides[get_db] = mock_db_queryable

        client.get("/health")

        # Verify execute was called with SELECT 1
        mock_session.execute.assert_called_once()
        executed_query = mock_session.execute.call_args[0][0]
        # The query should be a sqlalchemy text object with "SELECT 1"
        assert hasattr(executed_query, "text") or str(executed_query) == "SELECT 1"

    def test_health_no_try_except_in_endpoint(self):
        """Test that the endpoint doesn't catch exceptions internally."""
        # This test verifies the implementation doesn't have try/except
        # We'll check by ensuring exceptions propagate to FastAPI handlers

        with patch.dict(os.environ, test_env, clear=False):
            from api.main import app

            # Get the health endpoint function
            health_endpoint = None
            for route in app.routes:
                if hasattr(route, "path") and route.path == "/health":
                    health_endpoint = route.endpoint
                    break

            assert health_endpoint is not None, "Health endpoint not found"

            # Check the function doesn't contain try/except by inspecting code
            # This is a bit hacky but ensures no try/except blocks
            import inspect

            source = inspect.getsource(health_endpoint)
            assert "try:" not in source, "Health endpoint should not have try/except blocks"

    def test_health_response_time_under_100ms(self, client):
        """Test that health check responds in under 100ms."""
        # The default mock should be fast enough
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000
        assert response.status_code == 200
        assert response_time_ms < 100, f"Response time {response_time_ms}ms exceeds 100ms limit"

    def test_health_returns_proper_json_structure(self, client):
        """Test that health endpoint returns proper JSON structure."""
        response = client.get("/health")
        data = response.json()

        # Required fields
        assert isinstance(data, dict)
        assert "status" in data
        assert isinstance(data["status"], str)
        assert data["status"] in ["healthy", "unhealthy"]

        assert "db" in data
        assert isinstance(data["db"], str)
        assert data["db"] in ["connected", "disconnected"]

        # Optional fields
        if "timestamp" in data:
            assert isinstance(data["timestamp"], str)

        if "response_time_ms" in data:
            assert isinstance(data["response_time_ms"], (int, float))

    def test_health_uses_fastapi_exception_handlers(self, client):
        """Test that health endpoint uses FastAPI exception handlers."""
        # Mock database to raise an unexpected exception
        from api.main import app

        from database.session import get_db

        def mock_db_runtime_error():
            mock_session = MagicMock()
            mock_session.execute.side_effect = RuntimeError("Unexpected error")
            yield mock_session

        app.dependency_overrides[get_db] = mock_db_runtime_error

        # FastAPI should handle this and return 500
        response = client.get("/health")
        assert response.status_code == 500

        # Should have error details (not a generic message)
        data = response.json()
        assert "detail" in data
        assert "Unexpected error" in str(data["detail"])

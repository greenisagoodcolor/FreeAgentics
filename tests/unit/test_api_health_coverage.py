"""Comprehensive tests for api.v1.health module to achieve high coverage."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from api.v1.health import database_exception_handler, health_check, router


class TestHealthEndpoint:
    """Test health check endpoint comprehensively."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with health router."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_health_check_success(self, client):
        """Test successful health check with database connection."""
        # Mock database session
        mock_db = Mock(spec=Session)
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)
        mock_db.execute.return_value = mock_result

        # Override dependency
        from database.session import get_db

        def override_get_db():
            yield mock_db

        client.app.dependency_overrides[get_db] = override_get_db

        # Make request
        response = client.get("/health")

        # Verify response
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "db": "connected"}

        # Verify database was queried
        mock_db.execute.assert_called_once()
        mock_result.fetchone.assert_called_once()

    def test_health_check_database_error(self, client):
        """Test health check when database is down."""
        # Mock database session that raises OperationalError
        mock_db = Mock(spec=Session)
        mock_db.execute.side_effect = OperationalError("Connection failed", None, None)

        # Override dependency
        from database.session import get_db

        def override_get_db():
            yield mock_db

        client.app.dependency_overrides[get_db] = override_get_db

        # Make request
        response = client.get("/health")

        # Verify response
        assert response.status_code == 503
        json_response = response.json()
        assert json_response["status"] == "unhealthy"
        assert json_response["db"] == "disconnected"
        assert "Connection failed" in json_response["error"]

    async def test_health_check_function_success(self):
        """Test health_check function directly with successful DB."""
        # Mock database session
        mock_db = Mock(spec=Session)
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)
        mock_db.execute.return_value = mock_result

        # Call function directly
        result = await health_check(db=mock_db)

        # Verify result
        assert result == {"status": "healthy", "db": "connected"}
        mock_db.execute.assert_called_once()
        mock_result.fetchone.assert_called_once()

    async def test_health_check_function_error(self):
        """Test health_check function directly with DB error."""
        # Mock database session that raises error
        mock_db = Mock(spec=Session)
        mock_db.execute.side_effect = OperationalError("DB down", None, None)

        # Call function directly
        result = await health_check(db=mock_db)

        # Verify result is JSONResponse
        assert hasattr(result, "status_code")
        assert result.status_code == 503
        assert hasattr(result, "body")

    def test_database_exception_handler(self):
        """Test database_exception_handler function."""
        # Mock request and exception
        mock_request = Mock()
        mock_exception = OperationalError("Database unreachable", None, None)

        # Call handler
        response = database_exception_handler(mock_request, mock_exception)

        # Verify response
        assert hasattr(response, "status_code")
        assert response.status_code == 503
        assert hasattr(response, "body")

        # Check content by parsing body
        import json

        content = json.loads(response.body.decode())
        assert content["status"] == "unhealthy"
        assert content["db"] == "disconnected"
        assert "Database unreachable" in content["error"]

    def test_router_included(self):
        """Test that router is properly configured."""
        # Check router has the health endpoint
        routes = [route.path for route in router.routes]
        assert "/health" in routes

    def test_health_endpoint_with_real_db_error(self, client):
        """Test health endpoint with different database error types."""
        from database.session import get_db

        # Test with connection error
        mock_db = Mock(spec=Session)
        mock_db.execute.side_effect = OperationalError(
            "could not connect to server", None, None
        )

        def override_get_db():
            yield mock_db

        client.app.dependency_overrides[get_db] = override_get_db

        response = client.get("/health")
        assert response.status_code == 503
        assert "could not connect to server" in response.json()["error"]

    def test_health_check_execute_query_format(self, client):
        """Test that health check executes correct SQL query."""

        from database.session import get_db

        mock_db = Mock(spec=Session)
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)
        mock_db.execute.return_value = mock_result

        def override_get_db():
            yield mock_db

        client.app.dependency_overrides[get_db] = override_get_db

        client.get("/health")

        # Verify the exact query
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args[0]
        assert len(call_args) == 1
        query = call_args[0]
        assert hasattr(query, "text")
        assert str(query) == "SELECT 1"

    def test_exception_handler_with_different_errors(self):
        """Test exception handler with various error types."""
        mock_request = Mock()

        # Test with simple error message
        exc = OperationalError("Simple error", None, None)
        response = database_exception_handler(mock_request, exc)
        content = response.body.decode()
        assert "Simple error" in content

        # Test with complex error
        exc = OperationalError(
            "FATAL: password authentication failed for user", None, None
        )
        response = database_exception_handler(mock_request, exc)
        content = response.body.decode()
        assert "password authentication failed" in content

    async def test_health_check_async_nature(self):
        """Test that health_check is properly async."""
        import asyncio

        mock_db = Mock(spec=Session)
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)
        mock_db.execute.return_value = mock_result

        # Should be awaitable
        coro = health_check(db=mock_db)
        assert asyncio.iscoroutine(coro)

        result = await coro
        assert result == {"status": "healthy", "db": "connected"}

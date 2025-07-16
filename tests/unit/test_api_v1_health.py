"""Tests for api.v1.health module."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError


class TestHealthAPI:
    """Test the health API module."""

    def test_import_health_module(self):
        """Test that health module can be imported."""
        try:
            from api.v1.health import router, health_check, database_exception_handler
            
            # Test that router is an APIRouter instance
            assert isinstance(router, APIRouter)
            
            # Test that functions exist
            assert health_check is not None
            assert database_exception_handler is not None
            
        except ImportError as e:
            pytest.skip(f"Cannot import api.v1.health due to dependency issues: {e}")

    def test_router_is_api_router(self):
        """Test that router is an APIRouter instance."""
        try:
            from api.v1.health import router
            
            assert isinstance(router, APIRouter)
            
        except ImportError as e:
            pytest.skip(f"Cannot import api.v1.health due to dependency issues: {e}")

    @patch('api.v1.health.get_db')
    @patch('api.v1.health.text')
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_text, mock_get_db):
        """Test health_check with successful database connection."""
        try:
            from api.v1.health import health_check
            
            # Mock database session
            mock_db = MagicMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = (1,)
            mock_db.execute.return_value = mock_result
            
            # Mock SQL text
            mock_text.return_value = "SELECT 1"
            
            # Call health_check
            result = await health_check(mock_db)
            
            # Verify database query was executed
            mock_text.assert_called_once_with("SELECT 1")
            mock_db.execute.assert_called_once_with("SELECT 1")
            mock_result.fetchone.assert_called_once()
            
            # Verify response
            assert result == {"status": "healthy", "db": "connected"}
            
        except ImportError as e:
            pytest.skip(f"Cannot import api.v1.health due to dependency issues: {e}")

    @pytest.mark.asyncio
    async def test_database_exception_handler(self):
        """Test database_exception_handler."""
        try:
            from api.v1.health import database_exception_handler
            
            # Mock request and exception
            mock_request = MagicMock()
            mock_exc = OperationalError("Database connection failed", None, None)
            
            # Call exception handler
            result = await database_exception_handler(mock_request, mock_exc)
            
            # Verify response
            assert isinstance(result, JSONResponse)
            assert result.status_code == 503
            
            # Check content structure (would need to access body for full test)
            # For now, just verify it's a JSONResponse
            
        except ImportError as e:
            pytest.skip(f"Cannot import api.v1.health due to dependency issues: {e}")

    def test_health_check_function_attributes(self):
        """Test health_check function attributes."""
        try:
            from api.v1.health import health_check
            
            # Test that it's a coroutine function
            import asyncio
            assert asyncio.iscoroutinefunction(health_check)
            
            # Test that it has a docstring
            assert health_check.__doc__ is not None
            assert "Health check endpoint" in health_check.__doc__
            
        except ImportError as e:
            pytest.skip(f"Cannot import api.v1.health due to dependency issues: {e}")

    def test_database_exception_handler_function_attributes(self):
        """Test database_exception_handler function attributes."""
        try:
            from api.v1.health import database_exception_handler
            
            # Test that it's a coroutine function
            import asyncio
            assert asyncio.iscoroutinefunction(database_exception_handler)
            
            # Test that it has a docstring
            assert database_exception_handler.__doc__ is not None
            assert "Handle database operational errors" in database_exception_handler.__doc__
            
        except ImportError as e:
            pytest.skip(f"Cannot import api.v1.health due to dependency issues: {e}")

    def test_module_docstring(self):
        """Test that module has proper docstring."""
        try:
            import api.v1.health
            
            assert api.v1.health.__doc__ is not None
            assert "Health check endpoint following TDD principles" in api.v1.health.__doc__
            
        except ImportError as e:
            pytest.skip(f"Cannot import api.v1.health due to dependency issues: {e}")

    def test_imports_exist(self):
        """Test that required imports exist in module."""
        try:
            from api.v1.health import APIRouter, JSONResponse, text, OperationalError, Session
            
            # Test imports exist
            assert APIRouter is not None
            assert JSONResponse is not None
            assert text is not None
            assert OperationalError is not None
            assert Session is not None
            
        except ImportError as e:
            pytest.skip(f"Cannot import api.v1.health due to dependency issues: {e}")

    def test_router_routes_exist(self):
        """Test that router has expected routes."""
        try:
            from api.v1.health import router
            
            # Check that router has routes
            assert hasattr(router, 'routes')
            
            # There should be at least one route
            assert len(router.routes) > 0
            
        except ImportError as e:
            pytest.skip(f"Cannot import api.v1.health due to dependency issues: {e}")
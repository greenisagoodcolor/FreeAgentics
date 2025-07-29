# tests/test_db_init_guard.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import asyncio


@pytest.mark.asyncio
async def test_db_initialization_on_startup():
    """Test that database is properly initialized on application startup"""
    # Arrange
    with patch('database.session.init_db') as mock_init_db:
        mock_init_db.return_value = AsyncMock()
        
        # Import after patching to ensure the mock is in place
        from api.main import app
        
        # Act - Create test client which triggers startup
        with TestClient(app) as client:
            # Assert - init_db should be called during startup
            mock_init_db.assert_called_once()


@pytest.mark.asyncio 
async def test_health_endpoint_requires_db_ready():
    """Test that health check endpoint verifies database is ready"""
    # Arrange
    with patch('database.session.get_db_health') as mock_health:
        mock_health.return_value = {"status": "healthy", "db_ready": True}
        
        from api.main import app
        
        # Act
        with TestClient(app) as client:
            response = client.get("/health")
        
        # Assert
        assert response.status_code == 200
        assert response.json()["db_ready"] is True
        mock_health.assert_called_once()


@pytest.mark.asyncio
async def test_db_init_race_condition_handling():
    """Test that concurrent database initialization requests are handled safely"""
    # Arrange
    init_count = 0
    
    async def mock_init_with_delay():
        nonlocal init_count
        init_count += 1
        await asyncio.sleep(0.01)  # Simulate init time
        return {"status": "initialized"}
    
    with patch('database.session.init_db', side_effect=mock_init_with_delay):
        from database.session import init_db_safe
        
        # Act - Simulate concurrent initialization attempts
        tasks = [init_db_safe() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Assert - Init should only happen once despite multiple calls
        assert init_count == 1
        assert all(result["status"] == "initialized" for result in results)


def test_background_db_errors_handled_gracefully():
    """Test that background database errors don't crash the app"""
    # Arrange
    with patch('database.session.get_db') as mock_get_db:
        # Mock database connection error
        mock_get_db.side_effect = Exception("Connection lost")
        
        from api.main import app
        
        # Act & Assert - App should still respond to health checks
        with TestClient(app) as client:
            response = client.get("/health")
            # Should return degraded status, not crash
            assert response.status_code in [200, 503]  # OK or Service Unavailable
            assert "error" in response.json() or "db_ready" in response.json()
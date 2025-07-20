"""Simple API test without authentication requirements."""

import pytest
from tests.test_client_compat import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock

# Create a simple test app
app = FastAPI()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Test API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Test API"}


@app.get("/test")
async def test_endpoint():
    """Test endpoint."""
    return {"data": "test response"}


class TestSimpleAPI:
    """Test simple API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Test API"
        assert data["version"] == "1.0.0"

    def test_health_endpoint(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Test API"

    def test_test_endpoint(self):
        """Test test endpoint."""
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == "test response"

    def test_not_found(self):
        """Test 404 for non-existent endpoint."""
        client = TestClient(app)
        response = client.get("/nonexistent")
        assert response.status_code == 404


# Test the main app endpoints without authentication
class TestMainAppEndpoints:
    """Test main app endpoints that don't require auth."""

    @patch("database.session.init_db")
    def test_main_app_root(self, mock_init_db):
        """Test main app root endpoint."""
        # Import here to avoid import-time Redis issues
        from api.main import app as main_app

        client = TestClient(main_app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    @patch("database.session.init_db")
    def test_main_app_health(self, mock_init_db):
        """Test main app health endpoint."""
        from api.main import app as main_app

        client = TestClient(main_app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
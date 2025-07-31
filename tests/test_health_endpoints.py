"""Test health check endpoints."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_basic_health_check(client):
    """Test basic health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "db" in data or "database" in data


def test_llm_health_check_unauthenticated(client):
    """Test LLM health check without authentication."""
    response = client.get("/api/v1/health/llm")
    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    assert "status" in data
    assert "has_api_key" in data

    # Without auth, should use environment or mock provider
    assert data["provider"] in ["mock", "openai", "anthropic"]


@patch("auth.security_implementation.get_current_user_optional")
@patch("core.providers.get_llm")
def test_llm_health_check_authenticated(mock_get_llm, mock_auth, client):
    """Test LLM health check with authenticated user."""
    from datetime import datetime, timedelta

    from auth.security_implementation import TokenData, UserRole

    # Mock authenticated user
    mock_user = TokenData(
        user_id="test_user",
        username="test",
        role=UserRole.RESEARCHER,
        exp=datetime.now() + timedelta(hours=1),
        permissions=[],
    )
    mock_auth.return_value = mock_user

    # Mock LLM provider
    mock_provider = Mock()
    mock_provider.__class__.__name__ = "OpenAIProvider"
    mock_provider.complete = Mock(return_value="Test response")
    mock_get_llm.return_value = mock_provider

    response = client.get("/api/v1/health/llm", headers={"Authorization": "Bearer fake-token"})
    assert response.status_code == 200
    data = response.json()

    assert data["provider"] == "openai"
    assert data["status"] in ["healthy", "unhealthy", "unknown"]
    assert "has_api_key" in data

    # Verify user-specific provider was requested
    mock_get_llm.assert_called_once_with(user_id="test_user")


def test_llm_health_check_mock_provider(client):
    """Test LLM health check when using mock provider."""
    with patch("core.providers.get_llm") as mock_get_llm:
        # Mock the mock provider
        mock_provider = Mock()
        mock_provider.__class__.__name__ = "MockLLMProvider"
        mock_get_llm.return_value = mock_provider

        response = client.get("/api/v1/health/llm")
        assert response.status_code == 200
        data = response.json()

        assert data["provider"] == "mock"
        assert data["status"] == "healthy"
        assert data["has_api_key"] == False


def test_detailed_health_check(client):
    """Test detailed health check endpoint."""
    response = client.get("/api/v1/health/detailed")
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "timestamp" in data
    assert "checks" in data
    assert "warnings" in data
    assert "user_authenticated" in data

    # Check component statuses
    checks = data["checks"]
    assert "database" in checks
    assert "llm" in checks

    # LLM check should include provider info
    llm_check = checks["llm"]
    assert "provider" in llm_check
    assert "status" in llm_check


@patch("auth.security_implementation.get_current_user_optional")
def test_detailed_health_check_authenticated(mock_auth, client):
    """Test detailed health check with authenticated user."""
    from datetime import datetime, timedelta

    from auth.security_implementation import TokenData, UserRole

    mock_user = TokenData(
        user_id="test_user",
        username="test",
        role=UserRole.RESEARCHER,
        exp=datetime.now() + timedelta(hours=1),
        permissions=[],
    )
    mock_auth.return_value = mock_user

    response = client.get("/api/v1/health/detailed", headers={"Authorization": "Bearer fake-token"})
    assert response.status_code == 200
    data = response.json()

    assert data["user_authenticated"] == True

    # Should have user-specific LLM info
    if "llm" in data["checks"]:
        llm_check = data["checks"]["llm"]
        # Verify it attempted to get user-specific provider
        assert "provider" in llm_check


def test_health_check_database_failure(client):
    """Test health check when database is down."""
    with patch("sqlalchemy.orm.Session.execute") as mock_execute:
        # Simulate database failure
        from sqlalchemy.exc import OperationalError

        mock_execute.side_effect = OperationalError("Connection failed", None, None)

        response = client.get("/api/v1/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["db"] == "disconnected"
        assert "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

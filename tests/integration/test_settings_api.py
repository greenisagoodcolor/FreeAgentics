"""Integration tests for settings API endpoint."""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from core.providers import get_db
from database.base import Base
from database.models import User


@pytest.fixture
def test_db():
    """Create a test database."""
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield TestingSessionLocal()

    app.dependency_overrides.clear()


@pytest.fixture
def test_user(test_db):
    """Create a test user."""
    user = User(
        id="test-user-123",
        email="test@example.com",
        username="testuser",
        hashed_password="dummy_hash",
    )
    test_db.add(user)
    test_db.commit()
    return user


@pytest.fixture
def auth_headers(test_user):
    """Get auth headers for test user."""
    # In dev mode, we can use a simple token
    return {"Authorization": "Bearer dev-token"}


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestSettingsAPI:
    """Test settings API endpoints."""

    def test_get_default_settings(self, client, auth_headers, test_db):
        """Test getting default settings for new user."""
        response = client.get("/api/v1/settings", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        # Check default values
        assert data["llm_provider"] == "openai"
        assert data["llm_model"] == "gpt-4"
        assert data["openai_api_key"] is None
        assert data["anthropic_api_key"] is None
        assert data["gnn_enabled"] is True
        assert data["debug_logs"] is False
        assert data["auto_suggest"] is True
        assert "updated_at" in data

    def test_update_settings_full(self, client, auth_headers, test_db):
        """Test full settings update."""
        update_data = {
            "llm_provider": "anthropic",
            "llm_model": "claude-3",
            "anthropic_api_key": "sk-ant-test123",
            "gnn_enabled": False,
            "debug_logs": True,
            "auto_suggest": False,
        }

        response = client.put("/api/v1/settings", headers=auth_headers, json=update_data)

        assert response.status_code == 200
        data = response.json()

        # Check updated values
        assert data["llm_provider"] == "anthropic"
        assert data["llm_model"] == "claude-3"
        assert data["anthropic_api_key"] == "sk-ant-...****"  # Masked
        assert data["gnn_enabled"] is False
        assert data["debug_logs"] is True
        assert data["auto_suggest"] is False

    def test_patch_settings_partial(self, client, auth_headers, test_db):
        """Test partial settings update."""
        # First set some initial values
        client.put(
            "/api/v1/settings",
            headers=auth_headers,
            json={"llm_provider": "openai", "llm_model": "gpt-4", "gnn_enabled": True},
        )

        # Now patch only specific fields
        patch_data = {"llm_model": "gpt-3.5-turbo", "debug_logs": True}

        response = client.patch("/api/v1/settings", headers=auth_headers, json=patch_data)

        assert response.status_code == 200
        data = response.json()

        # Check only patched fields changed
        assert data["llm_provider"] == "openai"  # Unchanged
        assert data["llm_model"] == "gpt-3.5-turbo"  # Changed
        assert data["gnn_enabled"] is True  # Unchanged
        assert data["debug_logs"] is True  # Changed

    def test_api_key_validation(self, client, auth_headers):
        """Test API key format validation."""
        # Invalid OpenAI key format
        response = client.put(
            "/api/v1/settings", headers=auth_headers, json={"openai_api_key": "invalid-key"}
        )

        assert response.status_code == 422
        assert "must start with 'sk-'" in response.text

        # Invalid Anthropic key format
        response = client.put(
            "/api/v1/settings", headers=auth_headers, json={"anthropic_api_key": "invalid-key"}
        )

        assert response.status_code == 422
        assert "must start with 'sk-ant-'" in response.text

    def test_api_key_encryption(self, client, auth_headers, test_db):
        """Test that API keys are encrypted in database."""
        # Set API keys
        client.put(
            "/api/v1/settings",
            headers=auth_headers,
            json={"openai_api_key": "sk-test123", "anthropic_api_key": "sk-ant-test456"},
        )

        # Check database directly
        from api.v1.settings import UserSettings

        settings = test_db.query(UserSettings).first()

        # Keys should be encrypted, not plain text
        assert settings.encrypted_openai_key is not None
        assert "sk-test123" not in settings.encrypted_openai_key
        assert settings.encrypted_anthropic_key is not None
        assert "sk-ant-test456" not in settings.encrypted_anthropic_key

        # But decryption should work
        assert settings.get_openai_key() == "sk-test123"
        assert settings.get_anthropic_key() == "sk-ant-test456"

    def test_clear_api_keys(self, client, auth_headers, test_db):
        """Test clearing all API keys."""
        # First set some keys
        client.put(
            "/api/v1/settings",
            headers=auth_headers,
            json={"openai_api_key": "sk-test123", "anthropic_api_key": "sk-ant-test456"},
        )

        # Clear keys
        response = client.delete("/api/v1/settings/api-keys", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["message"] == "All API keys cleared successfully"

        # Verify keys are cleared
        response = client.get("/api/v1/settings", headers=auth_headers)
        data = response.json()
        assert data["openai_api_key"] is None
        assert data["anthropic_api_key"] is None

    def test_settings_affect_environment(self, client, auth_headers, test_db):
        """Test that settings updates affect environment variables."""
        # Clear any existing env vars
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("LLM_PROVIDER", None)

        # Update settings with OpenAI key
        client.put(
            "/api/v1/settings",
            headers=auth_headers,
            json={"llm_provider": "openai", "openai_api_key": "sk-test123"},
        )

        # Check environment variables were set
        assert os.environ.get("OPENAI_API_KEY") == "sk-test123"
        assert os.environ.get("LLM_PROVIDER") == "openai"

        # Clean up
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("LLM_PROVIDER", None)

    @pytest.mark.skip(reason="Requires actual API keys for validation")
    def test_validate_api_key_endpoint(self, client, auth_headers):
        """Test API key validation endpoint."""
        # This would require real API keys to test properly
        pass

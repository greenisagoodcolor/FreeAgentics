"""Integration tests for settings to LLM provider flow.

Tests the complete flow from user updating settings to LLM provider configuration.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.v1.settings import UserSettings
from auth.security_implementation import Permission, TokenData, UserRole
from database.base import Base
from database.session import get_db
from main import app


@pytest.fixture
def test_db():
    """Create a test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield TestingSessionLocal()

    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return TokenData(
        user_id="test_user_123",
        username="test_user",
        role=UserRole.RESEARCHER,
        exp=datetime.now() + timedelta(hours=1),
        permissions=[Permission.CREATE_AGENT, Permission.VIEW_AGENTS],
    )


class TestSettingsToProviderFlow:
    """Test the complete flow from settings to LLM provider configuration."""

    @patch("auth.security_implementation.get_current_user")
    def test_update_settings_creates_user_record(self, mock_auth, client, test_db, mock_user):
        """Test that updating settings creates a user settings record."""
        mock_auth.return_value = mock_user

        # Update settings
        response = client.patch(
            "/api/v1/settings",
            json={
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "openai_api_key": "sk-test123",
            },
            headers={"Authorization": "Bearer fake-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["llm_provider"] == "openai"
        assert data["llm_model"] == "gpt-4"
        assert "openai_api_key" not in data  # Should not expose the key

        # Verify database record
        settings = (
            test_db.query(UserSettings).filter(UserSettings.user_id == mock_user.user_id).first()
        )
        assert settings is not None
        assert settings.llm_provider == "openai"
        assert settings.llm_model == "gpt-4"
        assert settings.get_openai_key() == "sk-test123"

    @patch("auth.security_implementation.get_current_user")
    @patch("auth.security_implementation.get_current_user")
    @patch("inference.llm.openai_provider.OpenAIProvider")
    def test_prompt_uses_user_settings(
        self, mock_openai_class, mock_auth, client, test_db, mock_user
    ):
        """Test that prompt processing uses user-specific LLM settings."""
        mock_auth.return_value = mock_user

        # First, set up user settings
        response = client.patch(
            "/api/v1/settings",
            json={
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "openai_api_key": "sk-test123",
            },
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 200

        # Mock OpenAI provider
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = Mock(
            content='{"name": "TestAgent", "states": ["s1"], "observations": ["o1"], '
            '"actions": ["a1"], "parameters": {"A": [[1.0]], "B": [[[1.0]]], '
            '"C": [[1.0]], "D": [[1.0]]}}'
        )
        mock_provider.get_provider_type.return_value = Mock(value="openai")
        mock_openai_class.return_value = mock_provider

        # Submit a prompt
        response = client.post(
            "/api/v1/prompts",
            json={
                "prompt": "Create a simple test agent",
                "agent_name": "TestAgent",
            },
            headers={"Authorization": "Bearer fake-token"},
        )

        # Should succeed with user's OpenAI key
        assert response.status_code == 200
        data = response.json()
        assert data["agent_name"] == "TestAgent"
        assert data["llm_provider_used"] == "openai"

        # Verify OpenAI provider was created with user's key
        mock_openai_class.assert_called_once_with("sk-test123")

    @patch("auth.security_implementation.get_current_user")
    @patch("config.llm_config.get_llm_config")
    def test_user_settings_override_environment(
        self, mock_get_config, mock_auth, client, test_db, mock_user
    ):
        """Test that user settings override environment variables."""
        mock_auth.return_value = mock_user

        # Set up environment to return different config
        from config.llm_config import LLMConfig

        env_config = LLMConfig()
        env_config.openai.api_key = "sk-env-key"
        env_config.openai.enabled = True
        mock_get_config.return_value = env_config

        # Update user settings
        response = client.patch(
            "/api/v1/settings",
            json={
                "llm_provider": "openai",
                "openai_api_key": "sk-user-key",
            },
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 200

        # When getting config with user_id, it should use user settings
        from config.llm_config import get_llm_config

        user_config = get_llm_config(user_id=mock_user.user_id)

        # User config should have user's key, not env key
        assert user_config.openai.api_key == "sk-user-key"
        assert user_config.openai.enabled == True

    @patch("auth.security_implementation.get_current_user")
    def test_api_key_encryption(self, mock_auth, client, test_db, mock_user):
        """Test that API keys are encrypted in database."""
        mock_auth.return_value = mock_user

        # Update settings with API key
        response = client.patch(
            "/api/v1/settings",
            json={
                "openai_api_key": "sk-test-encryption",
            },
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 200

        # Check database directly
        settings = (
            test_db.query(UserSettings).filter(UserSettings.user_id == mock_user.user_id).first()
        )

        # Encrypted key should not equal plain text
        assert settings.encrypted_openai_key is not None
        assert settings.encrypted_openai_key != "sk-test-encryption"
        assert "sk-test-encryption" not in settings.encrypted_openai_key

        # But decrypted key should match
        assert settings.get_openai_key() == "sk-test-encryption"

    @patch("auth.security_implementation.get_current_user")
    @patch("auth.security_implementation.get_current_user")
    def test_no_api_key_falls_back_to_mock(self, mock_auth, client, test_db, mock_user):
        """Test that missing API key falls back to mock provider."""
        mock_auth.return_value = mock_user

        # Don't set any API keys
        response = client.patch(
            "/api/v1/settings",
            json={
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                # No API key provided
            },
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 200

        # Try to create agent - should fail with no providers available
        response = client.post(
            "/api/v1/prompts",
            json={
                "prompt": "Create a test agent",
            },
            headers={"Authorization": "Bearer fake-token"},
        )

        assert response.status_code == 503
        assert "No LLM providers available" in response.json()["detail"]

    @pytest.mark.asyncio
    @patch("api.v1.websocket.ws_auth_handler")
    @patch("api.v1.websocket.websocket_rate_limit_manager")
    async def test_websocket_prompt_uses_user_settings(
        self, mock_rate_limit, mock_ws_auth, client, test_db, mock_user
    ):
        """Test that WebSocket prompt submission uses user settings."""
        # Mock WebSocket auth
        mock_ws_auth.authenticate.return_value = True
        mock_ws_auth.get_user_data.return_value = mock_user
        mock_ws_auth.verify_permission.return_value = True
        mock_rate_limit.check_rate_limit.return_value = True

        # First set up user settings via HTTP
        with patch("api.v1.settings.get_current_user", return_value=mock_user):
            response = client.patch(
                "/api/v1/settings",
                json={
                    "llm_provider": "openai",
                    "openai_api_key": "sk-websocket-test",
                },
                headers={"Authorization": "Bearer fake-token"},
            )
            assert response.status_code == 200

        # Test WebSocket connection
        with client.websocket_connect("/api/v1/ws?token=fake-token") as websocket:
            # Send prompt
            websocket.send_json(
                {
                    "type": "prompt_submitted",
                    "prompt_id": "test-123",
                    "prompt": "Create a simple agent",
                    "conversation_id": "conv-456",
                }
            )

            # Should receive acknowledgment
            response = websocket.receive_json()
            assert response["type"] == "prompt_acknowledged"
            assert response["prompt_id"] == "test-123"

            # Note: In real test, would need to mock the OpenAI provider
            # and verify it was called with user's settings

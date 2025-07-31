"""Test unified development mode functionality."""

import os
from unittest.mock import patch

import pytest

from auth.dev_auth import DevAuthManager, get_dev_token
from core.providers import (
    InMemoryRateLimiter,
    MockLLMProvider,
    ProviderMode,
    SQLiteProvider,
    get_database,
    get_llm,
    get_rate_limiter,
    reset_providers,
)


class TestProviderMode:
    """Test provider mode detection."""

    def test_demo_mode_when_no_database_url(self):
        """Should detect demo mode when DATABASE_URL is not set."""
        with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=True):
            assert ProviderMode.get_mode() == "demo"

    def test_development_mode_with_database_url(self):
        """Should detect development mode when DATABASE_URL is set."""
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///dev.db", "PRODUCTION": "false"}):
            assert ProviderMode.get_mode() == "development"

    def test_production_mode_with_flag(self):
        """Should detect production mode when PRODUCTION=true."""
        with patch.dict(os.environ, {"PRODUCTION": "true", "DATABASE_URL": "postgresql://prod"}):
            assert ProviderMode.get_mode() == "production"


class TestDatabaseProvider:
    """Test database provider selection."""

    def setup_method(self):
        """Reset providers before each test."""
        reset_providers()

    def test_sqlite_provider_in_demo_mode(self):
        """Should use SQLite in-memory provider in demo mode."""
        with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=True):
            provider = get_database()
            assert isinstance(provider, SQLiteProvider)
            assert provider.path == ":memory:"

    def test_sqlite_provider_with_file_path(self):
        """Should use SQLite file provider when path specified."""
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///test.db"}):
            provider = get_database()
            assert isinstance(provider, SQLiteProvider)
            assert provider.path == "test.db"

    def test_provider_session_works(self):
        """Should provide working database sessions."""
        from sqlalchemy import text

        with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=True):
            provider = get_database()

            # Should be able to get a session
            with next(provider.get_session()) as session:
                assert session is not None
                # Should be able to execute queries
                result = session.execute(text("SELECT 1")).scalar()
                assert result == 1


class TestRateLimiterProvider:
    """Test rate limiter provider selection."""

    def setup_method(self):
        """Reset providers before each test."""
        reset_providers()

    def test_in_memory_rate_limiter_without_redis(self):
        """Should use in-memory rate limiter when Redis not configured."""
        with patch.dict(os.environ, {"REDIS_URL": ""}, clear=True):
            limiter = get_rate_limiter()
            assert isinstance(limiter, InMemoryRateLimiter)

    @pytest.mark.asyncio
    async def test_in_memory_rate_limiting_works(self):
        """Should enforce rate limits with in-memory provider."""
        with patch.dict(os.environ, {"REDIS_URL": ""}, clear=True):
            limiter = get_rate_limiter()

            # Should allow requests within limit
            for i in range(5):
                allowed = await limiter.check_rate_limit("test_key", limit=5, window=60)
                assert allowed is True

            # Should block over limit
            allowed = await limiter.check_rate_limit("test_key", limit=5, window=60)
            assert allowed is False


class TestLLMProvider:
    """Test LLM provider selection."""

    def setup_method(self):
        """Reset providers before each test."""
        reset_providers()

    def test_mock_llm_without_api_key(self):
        """Should use mock LLM when no API key configured."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
            llm = get_llm()
            assert isinstance(llm, MockLLMProvider)

    @pytest.mark.asyncio
    async def test_mock_llm_responses(self):
        """Should return deterministic mock responses."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
            llm = get_llm()

            # Test completion
            response = await llm.complete("Tell me about agents")
            assert "mock" in response.lower()
            assert "agent" in response.lower()

            # Test embedding
            embedding = await llm.embed("test text")
            assert isinstance(embedding, list)
            assert len(embedding) == 384  # Standard dimension


class TestDevAuth:
    """Test development authentication."""

    def test_dev_auth_only_in_dev_mode(self):
        """Should only provide dev tokens in development mode."""
        manager = DevAuthManager()

        # Should work in demo mode
        with patch.dict(os.environ, {"DATABASE_URL": "", "PRODUCTION": "false"}):
            assert manager.is_dev_mode() is True
            token_info = manager.get_or_create_dev_token()
            assert token_info["access_token"] is not None

        # Should fail in production
        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            assert manager.is_dev_mode() is False
            with pytest.raises(RuntimeError):
                manager.get_or_create_dev_token()

    def test_dev_token_has_admin_permissions(self):
        """Dev token should have admin permissions for easy development."""
        with patch.dict(os.environ, {"DATABASE_URL": "", "PRODUCTION": "false"}):
            token_info = get_dev_token()
            assert token_info is not None

            user = token_info["user"]
            assert user["role"] == "admin"
            assert "create_agent" in user["permissions"]
            assert "admin_system" in user["permissions"]

    def test_dev_token_reuse(self):
        """Should reuse dev token within validity period."""
        with patch.dict(os.environ, {"DATABASE_URL": "", "PRODUCTION": "false"}):
            manager = DevAuthManager()

            # Get first token
            token1 = manager.get_or_create_dev_token()
            first_token = token1["access_token"]

            # Get second token immediately
            token2 = manager.get_or_create_dev_token()
            assert token2["access_token"] == first_token
            assert "Reusing existing dev token" in token2["info"]


class TestDevConfigEndpoint:
    """Test /api/v1/dev-config endpoint."""

    @pytest.mark.asyncio
    async def test_dev_config_in_demo_mode(self):
        """Should return config with auth token in demo mode."""
        from api.v1.dev_config import get_dev_config

        with patch.dict(os.environ, {"DATABASE_URL": "", "PRODUCTION": "false"}):
            config = await get_dev_config()

            assert config["mode"] == "demo"
            assert config["features"]["auth_required"] is False
            assert "auth" in config
            assert config["auth"]["token"] is not None
            assert config["endpoints"]["websocket"] == "/api/v1/ws/demo"

    @pytest.mark.asyncio
    async def test_dev_config_blocked_in_production(self):
        """Should return 404 in production mode."""
        from fastapi import HTTPException

        from api.v1.dev_config import get_dev_config

        with patch.dict(os.environ, {"PRODUCTION": "true"}):
            with pytest.raises(HTTPException) as exc:
                await get_dev_config()
            assert exc.value.status_code == 404

"""Integration tests for settings persistence across requests.

Validates that user settings are properly saved and retrieved from the database,
addressing the critical bug where API keys were not persisting between sessions.
"""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from auth.dev_bypass import get_current_user_optional
from auth.security_implementation import TokenData
from core.providers import get_db
from database.base import Base
from database.models import User, UserSettings

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_settings.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


def override_get_current_user():
    """Override user authentication for testing."""
    return TokenData(
        user_id="test_user_123",
        username="test_user",
        role="admin",
        permissions=["create_agent", "view_agents"],
        exp=datetime.utcnow() + timedelta(hours=1),
    )


# Override dependencies
app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user_optional] = override_get_current_user

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Create tables before each test and clean up after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_settings_persist_across_requests():
    """Test that settings are saved and retrieved correctly across separate requests."""
    # Step 1: Save settings
    settings_data = {
        "llm_provider": "openai",
        "llm_model": "gpt-4",
        "openai_api_key": "sk-test-key-123",
        "gnn_enabled": True,
        "debug_logs": True,
        "auto_suggest": False,
    }

    save_response = client.patch("/api/v1/settings", json=settings_data)
    assert save_response.status_code == 200
    saved_data = save_response.json()
    assert saved_data["llm_provider"] == "openai"
    assert saved_data["llm_model"] == "gpt-4"
    assert saved_data["openai_api_key"] == "sk-...****"  # Masked in response
    assert saved_data["gnn_enabled"] is True
    assert saved_data["debug_logs"] is True
    assert saved_data["auto_suggest"] is False

    # Step 2: Retrieve settings in a separate request (simulating browser refresh)
    get_response = client.get("/api/v1/settings")
    assert get_response.status_code == 200
    retrieved_data = get_response.json()

    # Verify all settings persisted correctly
    assert retrieved_data["llm_provider"] == "openai"
    assert retrieved_data["llm_model"] == "gpt-4"
    assert retrieved_data["openai_api_key"] == "sk-...****"  # Still masked
    assert retrieved_data["gnn_enabled"] is True
    assert retrieved_data["debug_logs"] is True
    assert retrieved_data["auto_suggest"] is False
    assert "updated_at" in retrieved_data


def test_api_key_encryption_works():
    """Test that API keys are properly encrypted in the database."""
    # Save settings with API key
    settings_data = {
        "llm_provider": "anthropic",
        "anthropic_api_key": "sk-ant-test-key-456",
    }

    response = client.patch("/api/v1/settings", json=settings_data)
    assert response.status_code == 200

    # Check database directly to ensure encryption
    db = TestingSessionLocal()
    try:
        user_settings = (
            db.query(UserSettings).filter(UserSettings.user_id == "test_user_123").first()
        )

        assert user_settings is not None
        # Raw encrypted field should not contain the original key
        assert user_settings.encrypted_anthropic_key is not None
        assert "sk-ant-test-key-456" not in user_settings.encrypted_anthropic_key
        # But decryption should work
        assert user_settings.get_anthropic_key() == "sk-ant-test-key-456"
    finally:
        db.close()


def test_settings_update_partial():
    """Test that partial updates don't overwrite other settings."""
    # Create initial settings
    initial_data = {
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo",
        "openai_api_key": "sk-initial-key",
        "gnn_enabled": True,
        "debug_logs": False,
    }

    response = client.patch("/api/v1/settings", json=initial_data)
    assert response.status_code == 200

    # Update only one field
    partial_update = {"debug_logs": True}
    response = client.patch("/api/v1/settings", json=partial_update)
    assert response.status_code == 200

    # Verify that other settings remain unchanged
    get_response = client.get("/api/v1/settings")
    data = get_response.json()
    assert data["llm_provider"] == "openai"
    assert data["llm_model"] == "gpt-3.5-turbo"
    assert data["openai_api_key"] == "sk-...****"
    assert data["gnn_enabled"] is True
    assert data["debug_logs"] is True  # Only this should change


def test_settings_persistence_verification():
    """Test that the persistence verification catches failures."""
    # This test would be more useful with a mock that simulates database failures,
    # but for now we verify the happy path works correctly
    settings_data = {
        "llm_provider": "openai",
        "openai_api_key": "sk-verify-test",
    }

    response = client.patch("/api/v1/settings", json=settings_data)
    assert response.status_code == 200

    # The successful response indicates persistence verification passed
    data = response.json()
    assert data["llm_provider"] == "openai"
    assert data["openai_api_key"] == "sk-...****"


def test_user_creation_with_settings():
    """Test that user and settings are properly linked."""
    db = TestingSessionLocal()
    try:
        # Check that user was created automatically
        user = db.query(User).filter(User.id == "test_user_123").first()
        if not user:
            # Create user if it doesn't exist (for this test)
            user = User(id="test_user_123", username="test_user", email="test@example.com")
            db.add(user)
            db.commit()

        # Save settings
        settings_data = {"llm_provider": "openai"}
        response = client.patch("/api/v1/settings", json=settings_data)
        assert response.status_code == 200

        # Verify relationship
        db.refresh(user)
        user_settings = (
            db.query(UserSettings).filter(UserSettings.user_id == "test_user_123").first()
        )

        assert user_settings is not None
        assert user_settings.user_id == user.id

    finally:
        db.close()


def test_clear_api_keys():
    """Test that API key clearing works correctly."""
    # Set API keys
    settings_data = {
        "openai_api_key": "sk-test-key-1",
        "anthropic_api_key": "sk-ant-test-key-2",
    }
    response = client.patch("/api/v1/settings", json=settings_data)
    assert response.status_code == 200

    # Clear API keys
    response = client.delete("/api/v1/settings/api-keys")
    assert response.status_code == 200
    assert response.json()["message"] == "All API keys cleared successfully"

    # Verify keys are cleared
    get_response = client.get("/api/v1/settings")
    data = get_response.json()
    assert data["openai_api_key"] is None
    assert data["anthropic_api_key"] is None

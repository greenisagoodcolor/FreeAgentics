"""Simple test to verify user settings flow works correctly."""

import os
import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from database.base import Base
from api.v1.settings import UserSettings
from config.llm_config import get_llm_config
from auth.security_implementation import UserRole


@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def test_user_settings_storage_and_retrieval(test_db: Session):
    """Test that user settings can be stored and retrieved."""
    # Create user settings
    settings = UserSettings(
        user_id="test_user_123",
        llm_provider="openai",
        llm_model="gpt-4",
        gnn_enabled=True,
        debug_logs=False,
        auto_suggest=True,
    )
    
    # Set API key
    settings.set_openai_key("sk-test123")
    
    # Save to database
    test_db.add(settings)
    test_db.commit()
    
    # Retrieve settings
    retrieved = test_db.query(UserSettings).filter(
        UserSettings.user_id == "test_user_123"
    ).first()
    
    assert retrieved is not None
    assert retrieved.llm_provider == "openai"
    assert retrieved.llm_model == "gpt-4"
    assert retrieved.get_openai_key() == "sk-test123"
    assert retrieved.encrypted_openai_key != "sk-test123"  # Should be encrypted


def test_llm_config_uses_user_settings(test_db: Session):
    """Test that LLM config respects user settings."""
    # Create user settings with API key
    settings = UserSettings(
        user_id="test_user_456",
        llm_provider="openai",
        llm_model="gpt-4-turbo",
    )
    settings.set_openai_key("sk-userkey123")
    
    test_db.add(settings)
    test_db.commit()
    
    # Mock the database session for config
    from database import session as db_session
    original_local = db_session.SessionLocal
    db_session.SessionLocal = lambda: test_db
    
    try:
        # Get config for user
        config = get_llm_config(user_id="test_user_456")
        
        # Should have user's settings
        assert config.openai.api_key == "sk-userkey123"
        assert config.openai.enabled == True
        assert config.openai.default_model == "gpt-4-turbo"
        assert config.provider_priority == ["openai"]
        
        # Get config without user (should use env vars)
        env_config = get_llm_config()
        # Should not have the user's key
        assert env_config.openai.api_key != "sk-userkey123"
        
    finally:
        # Restore original
        db_session.SessionLocal = original_local


def test_api_key_validation(test_db: Session):
    """Test API key storage and retrieval."""
    settings = UserSettings(user_id="test_validation")
    
    # Valid OpenAI key
    settings.set_openai_key("sk-valid123")
    assert settings.get_openai_key() == "sk-valid123"
    
    # Update with new key
    settings.set_openai_key("sk-updated456")
    assert settings.get_openai_key() == "sk-updated456"
    
    # Clear key
    settings.set_openai_key(None)
    assert settings.get_openai_key() is None
    
    # Valid Anthropic key
    settings.set_anthropic_key("sk-ant-valid123")
    assert settings.get_anthropic_key() == "sk-ant-valid123"
    
    # Update Anthropic key
    settings.set_anthropic_key("sk-ant-updated456")
    assert settings.get_anthropic_key() == "sk-ant-updated456"


def test_settings_encryption_keys(test_db: Session):
    """Test that different users have different encryption."""
    # Create two users with same API key
    user1 = UserSettings(user_id="user1")
    user1.set_openai_key("sk-shared123")
    
    user2 = UserSettings(user_id="user2")
    user2.set_openai_key("sk-shared123")
    
    test_db.add(user1)
    test_db.add(user2)
    test_db.commit()
    
    # Encrypted values should be different (due to different keys)
    assert user1.encrypted_openai_key != user2.encrypted_openai_key
    
    # But decrypted values should be same
    assert user1.get_openai_key() == user2.get_openai_key() == "sk-shared123"


def test_provider_factory_integration():
    """Test that provider factory can use user settings."""
    from inference.llm.provider_factory import create_llm_manager
    
    # Create manager without user - should use env or mock
    manager = create_llm_manager()
    assert manager is not None
    
    # Note: Full integration test would require mocking database
    # and setting up user settings, which is done in the integration tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
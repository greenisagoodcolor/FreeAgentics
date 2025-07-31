"""Settings API endpoint for managing user configuration.

This module provides endpoints for synchronizing frontend settings with the backend,
including LLM provider configuration, API keys, and feature flags.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from auth.dev_bypass import get_current_user_optional
from auth.security_implementation import TokenData
from database.models import User
from core.providers import get_db, reset_providers
from database.base import Base
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Boolean
import json
import os

logger = logging.getLogger(__name__)

# Generate or load encryption key
ENCRYPTION_KEY = os.getenv("SETTINGS_ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    # Generate a new key for development
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    logger.warning("Generated new encryption key for settings. Set SETTINGS_ENCRYPTION_KEY in production!")

cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

router = APIRouter()


# Database Models
class UserSettings(Base):
    """Store user-specific settings securely."""
    
    __tablename__ = "user_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), unique=True, index=True)
    
    # LLM Configuration (encrypted)
    llm_provider = Column(String, default="openai")
    llm_model = Column(String, default="gpt-3.5-turbo")
    encrypted_openai_key = Column(Text, nullable=True)
    encrypted_anthropic_key = Column(Text, nullable=True)
    
    # Feature flags
    gnn_enabled = Column(Boolean, default=True)
    debug_logs = Column(Boolean, default=False)
    auto_suggest = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def set_openai_key(self, key: Optional[str]):
        """Encrypt and store OpenAI API key."""
        if key:
            self.encrypted_openai_key = cipher_suite.encrypt(key.encode()).decode()
        else:
            self.encrypted_openai_key = None
    
    def get_openai_key(self) -> Optional[str]:
        """Decrypt and return OpenAI API key."""
        if self.encrypted_openai_key:
            try:
                return cipher_suite.decrypt(self.encrypted_openai_key.encode()).decode()
            except Exception as e:
                logger.error(f"Failed to decrypt OpenAI key: {e}")
                return None
        return None
    
    def set_anthropic_key(self, key: Optional[str]):
        """Encrypt and store Anthropic API key."""
        if key:
            self.encrypted_anthropic_key = cipher_suite.encrypt(key.encode()).decode()
        else:
            self.encrypted_anthropic_key = None
    
    def get_anthropic_key(self) -> Optional[str]:
        """Decrypt and return Anthropic API key."""
        if self.encrypted_anthropic_key:
            try:
                return cipher_suite.decrypt(self.encrypted_anthropic_key.encode()).decode()
            except Exception as e:
                logger.error(f"Failed to decrypt Anthropic key: {e}")
                return None
        return None


# Pydantic Models
class SettingsUpdate(BaseModel):
    """Settings update request."""
    
    llm_provider: Optional[str] = Field(None, pattern="^(openai|anthropic|ollama)$")
    llm_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gnn_enabled: Optional[bool] = None
    debug_logs: Optional[bool] = None
    auto_suggest: Optional[bool] = None
    
    @field_validator("openai_api_key")
    def validate_openai_key(cls, v):
        """Validate OpenAI API key format."""
        if v and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
    
    @field_validator("anthropic_api_key")
    def validate_anthropic_key(cls, v):
        """Validate Anthropic API key format."""
        if v and not v.startswith("sk-ant-"):
            raise ValueError("Anthropic API key must start with 'sk-ant-'")
        return v


class SettingsResponse(BaseModel):
    """Settings response with masked sensitive data."""
    
    llm_provider: str
    llm_model: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gnn_enabled: bool
    debug_logs: bool
    auto_suggest: bool
    updated_at: datetime
    
    @classmethod
    def from_db_model(cls, settings: UserSettings) -> "SettingsResponse":
        """Create response from database model with masked keys."""
        return cls(
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            openai_api_key="sk-...****" if settings.encrypted_openai_key else None,
            anthropic_api_key="sk-ant-...****" if settings.encrypted_anthropic_key else None,
            gnn_enabled=settings.gnn_enabled,
            debug_logs=settings.debug_logs,
            auto_suggest=settings.auto_suggest,
            updated_at=settings.updated_at
        )


# Helper functions
def get_or_create_settings(db: Session, user_id: str) -> UserSettings:
    """Get existing settings or create default ones."""
    settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
    if not settings:
        settings = UserSettings(user_id=user_id)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


def apply_settings_to_environment(settings: UserSettings):
    """Apply user settings to environment for provider configuration."""
    # Set environment variables for LLM providers
    if settings.llm_provider == "openai" and settings.get_openai_key():
        os.environ["OPENAI_API_KEY"] = settings.get_openai_key()
        os.environ["LLM_PROVIDER"] = "openai"
        logger.info(f"Applied OpenAI settings for user {settings.user_id}")
    elif settings.llm_provider == "anthropic" and settings.get_anthropic_key():
        os.environ["ANTHROPIC_API_KEY"] = settings.get_anthropic_key()
        os.environ["LLM_PROVIDER"] = "anthropic"
        logger.info(f"Applied Anthropic settings for user {settings.user_id}")
    else:
        os.environ["LLM_PROVIDER"] = settings.llm_provider
        logger.info(f"Applied {settings.llm_provider} settings for user {settings.user_id}")
    
    # In dev mode, also save to temporary file for persistence across requests
    try:
        from core.environment import environment
        if environment.is_development:
            import tempfile
            settings_file = os.path.join(tempfile.gettempdir(), f"fa_settings_{settings.user_id}.json")
            settings_data = {
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "openai_api_key": settings.get_openai_key(),
                "anthropic_api_key": settings.get_anthropic_key(),
                "gnn_enabled": settings.gnn_enabled,
                "debug_logs": settings.debug_logs,
                "auto_suggest": settings.auto_suggest,
                "updated_at": settings.updated_at.isoformat() if settings.updated_at else None
            }
            with open(settings_file, 'w') as f:
                json.dump(settings_data, f)
            logger.info(f"Saved dev mode settings to {settings_file}")
    except Exception as e:
        logger.warning(f"Failed to save dev mode settings: {e}")
    
    # Reset providers to pick up new configuration
    reset_providers()


# API Endpoints
@router.get("/settings", response_model=SettingsResponse)
async def get_settings(
    current_user: TokenData = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Get current user settings."""
    settings = get_or_create_settings(db, current_user.user_id)
    
    logger.info(
        f"Retrieved settings for user {current_user.user_id} - "
        f"provider: {settings.llm_provider}, model: {settings.llm_model}"
    )
    
    return SettingsResponse.from_db_model(settings)


@router.put("/settings", response_model=SettingsResponse)
async def update_settings(
    settings_update: SettingsUpdate,
    current_user: TokenData = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Update user settings (full replacement)."""
    settings = get_or_create_settings(db, current_user.user_id)
    
    # Update all fields
    if settings_update.llm_provider is not None:
        settings.llm_provider = settings_update.llm_provider
    if settings_update.llm_model is not None:
        settings.llm_model = settings_update.llm_model
    if settings_update.openai_api_key is not None:
        settings.set_openai_key(settings_update.openai_api_key)
    if settings_update.anthropic_api_key is not None:
        settings.set_anthropic_key(settings_update.anthropic_api_key)
    if settings_update.gnn_enabled is not None:
        settings.gnn_enabled = settings_update.gnn_enabled
    if settings_update.debug_logs is not None:
        settings.debug_logs = settings_update.debug_logs
    if settings_update.auto_suggest is not None:
        settings.auto_suggest = settings_update.auto_suggest
    
    settings.updated_at = datetime.now()
    db.commit()
    db.refresh(settings)
    
    # Apply settings to environment
    apply_settings_to_environment(settings)
    
    logger.info(
        f"Updated settings for user {current_user.user_id} - "
        f"provider: {settings.llm_provider}, model: {settings.llm_model}, "
        f"keys_configured: {bool(settings.encrypted_openai_key or settings.encrypted_anthropic_key)}"
    )
    
    return SettingsResponse.from_db_model(settings)


@router.patch("/settings", response_model=SettingsResponse)
async def patch_settings(
    settings_update: SettingsUpdate,
    current_user: TokenData = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Partially update user settings."""
    settings = get_or_create_settings(db, current_user.user_id)
    
    # Update only provided fields
    update_data = settings_update.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        if field == "openai_api_key":
            settings.set_openai_key(value)
        elif field == "anthropic_api_key":
            settings.set_anthropic_key(value)
        else:
            setattr(settings, field, value)
    
    settings.updated_at = datetime.now()
    db.commit()
    db.refresh(settings)
    
    # Apply settings to environment
    apply_settings_to_environment(settings)
    
    logger.info(
        f"Patched settings for user {current_user.user_id} - "
        f"updated fields: {list(update_data.keys())}"
    )
    
    return SettingsResponse.from_db_model(settings)


@router.post("/settings/validate-key")
async def validate_api_key(
    provider: str,
    api_key: str,
    current_user: TokenData = Depends(get_current_user_optional)
):
    """Validate an API key before saving."""
    try:
        if provider == "openai":
            # Try a simple API call to validate
            import openai
            client = openai.OpenAI(api_key=api_key)
            # List models as a validation check
            models = client.models.list()
            return {
                "valid": True,
                "message": "OpenAI API key is valid",
                "models_available": len(list(models))
            }
        
        elif provider == "anthropic":
            # Try a simple API call to validate
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Try a minimal completion
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return {
                "valid": True,
                "message": "Anthropic API key is valid"
            }
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown provider: {provider}"
            )
            
    except Exception as e:
        logger.warning(f"API key validation failed for {provider}: {str(e)}")
        return {
            "valid": False,
            "message": f"API key validation failed: {str(e)}"
        }


@router.delete("/settings/api-keys")
async def clear_api_keys(
    current_user: TokenData = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Clear all stored API keys."""
    settings = get_or_create_settings(db, current_user.user_id)
    
    settings.set_openai_key(None)
    settings.set_anthropic_key(None)
    
    db.commit()
    
    # Clear from environment
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    
    # Reset providers
    reset_providers()
    
    logger.info(f"Cleared all API keys for user {current_user.user_id}")
    
    return {"message": "All API keys cleared successfully"}
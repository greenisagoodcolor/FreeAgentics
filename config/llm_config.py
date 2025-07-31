"""
LLM Configuration Management.

Handles configuration and secrets for LLM providers following Clean Architecture.
Provides optional environment variable loading with sensible defaults.
"""

import logging
import os
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    api_key: Optional[str] = None
    organization_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 60  # Increased for GPT-4 requests
    rate_limit_requests_per_minute: Optional[int] = None
    enabled: bool = True

    model_config = ConfigDict(extra="allow")


class OpenAIConfig(LLMProviderConfig):
    """OpenAI-specific configuration."""

    default_model: str = "gpt-3.5-turbo"
    endpoint_url: Optional[str] = "https://api.openai.com/v1"

    # OpenAI pricing per 1K tokens (as of 2024)
    pricing: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }
    )


class AnthropicConfig(LLMProviderConfig):
    """Anthropic-specific configuration."""

    default_model: str = "claude-3-sonnet-20240229"
    endpoint_url: Optional[str] = "https://api.anthropic.com"

    # Anthropic pricing per 1K tokens (as of 2024)
    pricing: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "claude-3-sonnet-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-opus-20240229": {"input": 0.075, "output": 0.225},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        }
    )


class LLMConfig(BaseModel):
    """Complete LLM configuration for all providers."""

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)

    # Global settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    default_temperature: float = 0.7
    default_max_tokens: int = 1000

    # Provider priority order for fallback
    provider_priority: list[str] = Field(default_factory=lambda: ["openai", "anthropic"])

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_environment(cls, user_id: Optional[str] = None) -> "LLMConfig":
        """Load configuration from user settings or environment variables.

        Args:
            user_id: Optional user ID to load user-specific settings
        """
        config = cls()

        # First check user settings if user_id is provided
        if user_id:
            try:
                from api.v1.settings import UserSettings
                from core.environment import environment
                from database.session import SessionLocal

                # In dev mode, check the in-memory settings store first
                if environment.is_development:
                    try:
                        import tempfile

                        settings_file = os.path.join(
                            tempfile.gettempdir(), f"fa_settings_{user_id}.json"
                        )
                        if os.path.exists(settings_file):
                            import json

                            with open(settings_file, "r") as f:
                                saved_settings = json.load(f)

                            if saved_settings.get("openai_api_key"):
                                config.openai.api_key = saved_settings["openai_api_key"]
                                config.openai.enabled = True
                                config.openai.default_model = saved_settings.get(
                                    "llm_model", "gpt-4"
                                )
                                config.provider_priority = ["openai"]
                                logger.info(
                                    f"Loaded OpenAI config from dev settings for user {user_id}"
                                )
                                return config
                            elif saved_settings.get("anthropic_api_key"):
                                config.anthropic.api_key = saved_settings["anthropic_api_key"]
                                config.anthropic.enabled = True
                                config.anthropic.default_model = saved_settings.get(
                                    "llm_model", "claude-3-sonnet-20240229"
                                )
                                config.provider_priority = ["anthropic"]
                                return config
                    except Exception as dev_e:
                        pass  # Fall through to database check

                db = SessionLocal()
                try:
                    user_settings = (
                        db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
                    )
                    if user_settings:
                        # Use user's provider preference
                        provider = user_settings.llm_provider

                        if provider == "openai" and user_settings.get_openai_key():
                            config.openai.api_key = user_settings.get_openai_key()
                            config.openai.enabled = True
                            config.openai.default_model = user_settings.llm_model
                            config.provider_priority = ["openai"]
                            return config
                        elif provider == "anthropic" and user_settings.get_anthropic_key():
                            config.anthropic.api_key = user_settings.get_anthropic_key()
                            config.anthropic.enabled = True
                            config.anthropic.default_model = user_settings.llm_model
                            config.provider_priority = ["anthropic"]
                            return config
                finally:
                    db.close()
            except Exception as e:
                # Fall through to environment variables
                pass

        # Fall back to environment variables
        # Load OpenAI configuration
        if openai_key := os.getenv("OPENAI_API_KEY"):
            config.openai.api_key = openai_key
            config.openai.enabled = True
        else:
            config.openai.enabled = False

        if openai_org := os.getenv("OPENAI_ORGANIZATION"):
            config.openai.organization_id = openai_org

        # Load Anthropic configuration
        if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
            config.anthropic.api_key = anthropic_key
            config.anthropic.enabled = True
        else:
            config.anthropic.enabled = False

        # Global settings from environment
        if temp := os.getenv("LLM_DEFAULT_TEMPERATURE"):
            try:
                config.default_temperature = float(temp)
            except ValueError:
                pass  # Use default

        if max_tokens := os.getenv("LLM_DEFAULT_MAX_TOKENS"):
            try:
                config.default_max_tokens = int(max_tokens)
            except ValueError:
                pass  # Use default

        if enable_cache := os.getenv("LLM_ENABLE_CACHING"):
            config.enable_caching = enable_cache.lower() in ("true", "1", "yes", "on")

        # Update provider priority based on which providers are enabled
        enabled_providers = []
        if config.openai.enabled:
            enabled_providers.append("openai")
        if config.anthropic.enabled:
            enabled_providers.append("anthropic")

        if enabled_providers:
            config.provider_priority = enabled_providers

        return config

    def get_enabled_providers(self) -> Dict[str, LLMProviderConfig]:
        """Get all enabled provider configurations."""
        enabled = {}

        if self.openai.enabled:
            enabled["openai"] = self.openai
        if self.anthropic.enabled:
            enabled["anthropic"] = self.anthropic

        return enabled

    def validate_configuration(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        enabled_providers = self.get_enabled_providers()
        if not enabled_providers:
            issues.append("No LLM providers are enabled")

        for name, provider_config in enabled_providers.items():
            if not provider_config.api_key:
                issues.append(f"{name} provider is enabled but has no API key")

        return issues


# Singleton instance for global access
_global_config: Optional[LLMConfig] = None


def get_llm_config(user_id: Optional[str] = None) -> LLMConfig:
    """Get LLM configuration instance.

    Args:
        user_id: Optional user ID to load user-specific settings
    """
    # For user-specific config, always create fresh instance
    if user_id:
        return LLMConfig.from_environment(user_id)

    # For global config, use singleton
    global _global_config
    if _global_config is None:
        _global_config = LLMConfig.from_environment()

    return _global_config


def reload_llm_config() -> LLMConfig:
    """Reload LLM configuration from environment."""
    global _global_config
    _global_config = LLMConfig.from_environment()
    return _global_config


def set_llm_config(config: LLMConfig) -> None:
    """Set global LLM configuration (for testing)."""
    global _global_config
    _global_config = config

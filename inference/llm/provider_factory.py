"""
LLM Provider Factory and Error Handling.

Provides a clean factory interface for creating and managing LLM providers
with comprehensive error handling for production use.
"""

import logging
from typing import Dict, List, Optional, Type

from config.llm_config import get_llm_config

from .provider_interface import (
    ILLMProvider,
    ProviderCredentials,
    ProviderManager,
    ProviderType,
)

try:
    from .openai_provider import OpenAIProvider

    OPENAI_PROVIDER_AVAILABLE = True
except ImportError:
    OPENAI_PROVIDER_AVAILABLE = False

try:
    from .anthropic_provider import AnthropicProvider

    ANTHROPIC_PROVIDER_AVAILABLE = True
except ImportError:
    ANTHROPIC_PROVIDER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    pass


class ProviderConfigurationError(ProviderError):
    """Exception raised when provider configuration fails."""

    pass


class ProviderNotAvailableError(ProviderError):
    """Exception raised when requested provider is not available."""

    pass


class LLMProviderFactory:
    """Factory for creating and configuring LLM providers."""

    def __init__(self):
        """Initialize the factory."""
        self._provider_classes: Dict[ProviderType, Type[ILLMProvider]] = {}
        self._register_builtin_providers()

    def _register_builtin_providers(self) -> None:
        """Register built-in provider implementations."""
        if OPENAI_PROVIDER_AVAILABLE:
            self._provider_classes[ProviderType.OPENAI] = OpenAIProvider
            logger.debug("Registered OpenAI provider")
        else:
            logger.warning("OpenAI provider not available - install 'openai' package")

        if ANTHROPIC_PROVIDER_AVAILABLE:
            self._provider_classes[ProviderType.ANTHROPIC] = AnthropicProvider
            logger.debug("Registered Anthropic provider")
        else:
            logger.warning("Anthropic provider not available - install 'anthropic' package")

    def get_available_providers(self) -> List[ProviderType]:
        """Get list of available provider types."""
        return list(self._provider_classes.keys())

    def is_provider_available(self, provider_type: ProviderType) -> bool:
        """Check if a provider type is available."""
        return provider_type in self._provider_classes

    def create_provider(self, provider_type: ProviderType) -> ILLMProvider:
        """Create a new provider instance."""
        if not self.is_provider_available(provider_type):
            raise ProviderNotAvailableError(
                f"Provider {provider_type.value} is not available. "
                f"Available providers: {[p.value for p in self.get_available_providers()]}"
            )

        provider_class = self._provider_classes[provider_type]
        return provider_class()

    def create_configured_provider(
        self,
        provider_type: ProviderType,
        credentials: ProviderCredentials,
        **config_kwargs,
    ) -> ILLMProvider:
        """Create and configure a provider instance."""
        provider = self.create_provider(provider_type)

        success = provider.configure(credentials, **config_kwargs)
        if not success:
            raise ProviderConfigurationError(f"Failed to configure {provider_type.value} provider")

        return provider

    def create_from_config(self) -> ProviderManager:
        """Create a ProviderManager with providers from configuration."""
        config = get_llm_config()
        manager = ProviderManager()

        # Validate configuration
        issues = config.validate_configuration()
        if issues:
            logger.warning(f"LLM configuration issues: {', '.join(issues)}")

        # Create and register enabled providers
        enabled_providers = config.get_enabled_providers()

        for provider_name, provider_config in enabled_providers.items():
            provider_type = ProviderType(provider_name)

            if not self.is_provider_available(provider_type):
                logger.error(f"Provider {provider_name} is configured but not available")
                continue

            try:
                # Create credentials
                credentials = ProviderCredentials(
                    api_key=provider_config.api_key,
                    organization_id=provider_config.organization_id,
                    endpoint_url=provider_config.endpoint_url,
                )

                # Create configured provider
                provider = self.create_configured_provider(
                    provider_type,
                    credentials,
                    timeout=provider_config.timeout_seconds,
                    max_retries=provider_config.max_retries,
                )

                # Register with manager using priority from config
                priority = (
                    config.provider_priority.index(provider_name)
                    if provider_name in config.provider_priority
                    else 50
                )
                manager.registry.register_provider(provider, priority)

                logger.info(f"Successfully configured {provider_name} provider")

            except Exception as e:
                logger.error(f"Failed to configure {provider_name} provider: {e}")
                continue

        return manager


class ErrorHandler:
    """Comprehensive error handling for LLM operations."""

    @staticmethod
    def is_retryable_error(error: Exception) -> bool:
        """Determine if an error is retryable."""
        error_str = str(error).lower()

        # Rate limit errors are retryable with backoff
        if any(phrase in error_str for phrase in ["rate limit", "too many requests", "429"]):
            return True

        # Timeout errors are retryable
        if any(phrase in error_str for phrase in ["timeout", "timed out", "connection"]):
            return True

        # Server errors (5xx) are retryable
        if any(phrase in error_str for phrase in ["500", "502", "503", "504", "server error"]):
            return True

        # Authentication and client errors are not retryable
        if any(
            phrase in error_str for phrase in ["401", "403", "invalid api key", "authentication"]
        ):
            return False

        # Bad request errors are not retryable
        if any(phrase in error_str for phrase in ["400", "bad request", "invalid"]):
            return False

        # Default to non-retryable for unknown errors
        return False

    @staticmethod
    def get_retry_delay(attempt: int, error: Exception) -> float:
        """Get retry delay based on attempt number and error type."""
        base_delay = 1.0

        error_str = str(error).lower()

        # Use exponential backoff for rate limits
        if any(phrase in error_str for phrase in ["rate limit", "429"]):
            return min(base_delay * (2**attempt), 60.0)  # Max 60 seconds

        # Use shorter delays for connection issues
        if any(phrase in error_str for phrase in ["timeout", "connection"]):
            return min(base_delay * (1.5**attempt), 30.0)  # Max 30 seconds

        # Default exponential backoff
        return min(base_delay * (2**attempt), 45.0)  # Max 45 seconds

    @staticmethod
    def should_fallback(error: Exception) -> bool:
        """Determine if we should fallback to another provider."""
        error_str = str(error).lower()

        # Always fallback for authentication errors
        if any(
            phrase in error_str for phrase in ["401", "403", "invalid api key", "authentication"]
        ):
            return True

        # Fallback for persistent rate limiting
        if any(phrase in error_str for phrase in ["rate limit", "429"]):
            return True

        # Fallback for service unavailable
        if any(phrase in error_str for phrase in ["503", "service unavailable", "offline"]):
            return True

        # Don't fallback for client errors (user's fault)
        if any(phrase in error_str for phrase in ["400", "bad request"]):
            return False

        # Default to fallback for other errors
        return True


# Global factory instance
_factory_instance: Optional[LLMProviderFactory] = None


def get_provider_factory() -> LLMProviderFactory:
    """Get the global provider factory instance."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = LLMProviderFactory()
    return _factory_instance


def create_llm_manager() -> ProviderManager:
    """Create a fully configured LLM provider manager."""
    factory = get_provider_factory()
    return factory.create_from_config()

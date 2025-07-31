"""LLM Provider Factory for automatic provider selection and management.

This module provides a factory pattern implementation for creating and managing
LLM providers based on configuration, with fallback mechanisms and health checks.
"""

import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from llm.base import LLMError, LLMMessage, LLMProvider, LLMResponse
from llm.providers.anthropic import AnthropicProvider
from llm.providers.mock import MockLLMProvider
from llm.providers.ollama import OllamaProvider
from llm.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Available LLM provider types."""

    MOCK = "mock"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AUTO = "auto"


class ProviderHealth:
    """Health status tracking for a provider."""

    def __init__(self, provider_type: ProviderType):
        self.provider_type = provider_type
        self.is_healthy = True
        self.last_success: Optional[datetime] = None
        self.last_failure: Optional[datetime] = None
        self.consecutive_failures = 0
        self.total_requests = 0
        self.total_failures = 0
        self.average_latency: Optional[float] = None
        self.latencies: List[float] = []

    def record_success(self, latency: float):
        """Record a successful request."""
        self.last_success = datetime.now()
        self.consecutive_failures = 0
        self.total_requests += 1
        self.is_healthy = True

        # Update latency tracking
        self.latencies.append(latency)
        if len(self.latencies) > 100:  # Keep last 100 latencies
            self.latencies.pop(0)
        self.average_latency = sum(self.latencies) / len(self.latencies)

    def record_failure(self):
        """Record a failed request."""
        self.last_failure = datetime.now()
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1

        # Mark unhealthy after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_healthy = False

    def should_retry(self) -> bool:
        """Check if provider should be retried after being marked unhealthy."""
        if self.is_healthy:
            return True

        # Retry unhealthy providers after 5 minutes
        if self.last_failure:
            time_since_failure = datetime.now() - self.last_failure
            if time_since_failure > timedelta(minutes=5):
                self.is_healthy = True  # Give it another chance
                return True

        return False

    def get_success_rate(self) -> float:
        """Get the success rate as a percentage."""
        if self.total_requests == 0:
            return 100.0
        return ((self.total_requests - self.total_failures) / self.total_requests) * 100


class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""

    # Provider classes mapping
    PROVIDER_CLASSES: Dict[ProviderType, Type[LLMProvider]] = {
        ProviderType.MOCK: MockLLMProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.OLLAMA: OllamaProvider,
    }

    @classmethod
    def create_provider(cls, provider_name: str = "auto") -> LLMProvider:
        """Create a provider directly (convenience method).

        Args:
            provider_name: Provider name or "auto" for automatic selection

        Returns:
            LLMProvider instance
        """
        factory = create_llm_factory()
        if provider_name == "auto":
            # Get the primary provider from factory
            provider = factory._create_provider(factory._primary_provider)
        else:
            # Create specific provider
            provider_type = ProviderType(provider_name.lower())
            provider = factory._create_provider(provider_type)
        return provider

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the provider factory.

        Args:
            config: Configuration dictionary with provider settings
        """
        self.config = config or {}
        self._providers: Dict[ProviderType, LLMProvider] = {}
        self._health: Dict[ProviderType, ProviderHealth] = {}
        self._primary_provider: Optional[ProviderType] = None
        self._fallback_chain: List[ProviderType] = []

        # Initialize based on config
        self._setup_providers()

    def _setup_providers(self):
        """Setup providers based on configuration."""
        # Get provider preference from config or environment
        provider_pref = self.config.get("provider", os.getenv("LLM_PROVIDER", "auto"))

        if provider_pref == "auto":
            self._setup_auto_providers()
        else:
            # Try to use specified provider
            try:
                provider_type = ProviderType(provider_pref.lower())
                if provider_type != ProviderType.AUTO:
                    self._primary_provider = provider_type
                    self._fallback_chain = self._get_fallback_chain(provider_type)
            except ValueError:
                logger.warning(f"Unknown provider type: {provider_pref}, using auto mode")
                self._setup_auto_providers()

    def _setup_auto_providers(self):
        """Setup providers in auto mode based on available credentials."""
        available_providers = []

        # Check for API keys and services
        if os.getenv("OPENAI_API_KEY"):
            available_providers.append(ProviderType.OPENAI)

        if os.getenv("ANTHROPIC_API_KEY"):
            available_providers.append(ProviderType.ANTHROPIC)

        # For demo mode - prioritize mock provider
        if not available_providers:
            # No API keys - use mock as primary
            self._primary_provider = ProviderType.MOCK
            self._fallback_chain = [ProviderType.OLLAMA]  # Ollama as backup
        else:
            # API keys available - use them
            self._primary_provider = available_providers[0]
            # Add Ollama and Mock as fallbacks
            self._fallback_chain = available_providers[1:] + [
                ProviderType.OLLAMA,
                ProviderType.MOCK,
            ]

    def _get_fallback_chain(self, primary: ProviderType) -> List[ProviderType]:
        """Get fallback chain for a primary provider."""
        all_providers = [
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC,
            ProviderType.OLLAMA,
            ProviderType.MOCK,
        ]

        # Remove primary from chain and return others
        return [p for p in all_providers if p != primary]

    def _create_provider(self, provider_type: ProviderType) -> LLMProvider:
        """Create a provider instance."""
        if provider_type in self._providers:
            return self._providers[provider_type]

        provider_class = self.PROVIDER_CLASSES.get(provider_type)
        if not provider_class:
            raise LLMError(f"Unknown provider type: {provider_type}")

        # Get provider-specific config
        provider_config = self.config.get(provider_type.value, {})

        try:
            # Create provider with config
            if provider_type == ProviderType.MOCK:
                provider = provider_class(**provider_config)
            elif provider_type == ProviderType.OPENAI:
                provider = provider_class(
                    api_key=provider_config.get("api_key", os.getenv("OPENAI_API_KEY")),
                    model=provider_config.get("model", "gpt-4o"),
                    **{k: v for k, v in provider_config.items() if k not in ["api_key", "model"]},
                )
            elif provider_type == ProviderType.ANTHROPIC:
                provider = provider_class(
                    api_key=provider_config.get("api_key", os.getenv("ANTHROPIC_API_KEY")),
                    model=provider_config.get("model", "claude-3-5-sonnet-20241022"),
                    **{k: v for k, v in provider_config.items() if k not in ["api_key", "model"]},
                )
            elif provider_type == ProviderType.OLLAMA:
                provider = provider_class(
                    model=provider_config.get("model", "llama3.2"),
                    base_url=provider_config.get("base_url", "http://localhost:11434"),
                    **{k: v for k, v in provider_config.items() if k not in ["model", "base_url"]},
                )
            else:
                raise LLMError(f"Provider {provider_type} not implemented")

            # Cache the provider
            self._providers[provider_type] = provider
            self._health[provider_type] = ProviderHealth(provider_type)

            return provider

        except Exception as e:
            logger.error(f"Failed to create provider {provider_type}: {str(e)}")
            raise LLMError(f"Failed to create provider {provider_type}: {str(e)}")

    async def get_provider(self, provider_type: Optional[ProviderType] = None) -> LLMProvider:
        """Get a provider instance, creating if necessary.

        Args:
            provider_type: Specific provider type to get, or None for primary

        Returns:
            LLMProvider instance
        """
        if provider_type is None:
            provider_type = self._primary_provider

        if provider_type is None:
            raise LLMError("No provider configured")

        return self._create_provider(provider_type)

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        provider_type: Optional[ProviderType] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using the configured providers with fallback.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop_sequences: Stop sequences
            provider_type: Specific provider to use, or None for auto
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse from the first successful provider
        """
        # Determine providers to try
        if provider_type:
            providers_to_try = [provider_type]
        else:
            providers_to_try = [self._primary_provider] + self._fallback_chain

        last_error = None

        for provider_type in providers_to_try:
            # Skip unhealthy providers unless they should be retried
            health = self._health.get(provider_type)
            if health and not health.should_retry():
                continue

            try:
                # Get or create provider
                provider = await self.get_provider(provider_type)

                # Time the request
                start_time = datetime.now()

                # Make the request
                response = await provider.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_sequences=stop_sequences,
                    **kwargs,
                )

                # Record success
                latency = (datetime.now() - start_time).total_seconds()
                if provider_type in self._health:
                    self._health[provider_type].record_success(latency)

                # Add provider info to metadata
                if response.metadata is None:
                    response.metadata = {}
                response.metadata["provider"] = provider_type.value
                response.metadata["latency"] = latency

                logger.info(f"Successfully generated response using {provider_type.value}")

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_type.value} failed: {str(e)}")

                # Record failure
                if provider_type in self._health:
                    self._health[provider_type].record_failure()

                # Continue to next provider
                continue

        # All providers failed
        raise LLMError(f"All providers failed. Last error: {str(last_error)}")

    async def generate_gmn(
        self,
        prompt: str,
        agent_type: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        examples: Optional[List[str]] = None,
        provider_type: Optional[ProviderType] = None,
    ) -> str:
        """Generate GMN using the configured providers.

        Args:
            prompt: Natural language prompt
            agent_type: Type of agent
            constraints: GMN constraints
            examples: Example GMN specs
            provider_type: Specific provider to use

        Returns:
            Generated GMN specification
        """
        # Get provider
        provider = await self.get_provider(provider_type)

        # Use provider's GMN generation method
        return await provider.generate_gmn(
            prompt=prompt,
            agent_type=agent_type,
            constraints=constraints,
            examples=examples,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all configured providers."""
        results = {}

        for provider_type in [self._primary_provider] + self._fallback_chain:
            if provider_type is None:
                continue

            health = self._health.get(provider_type, ProviderHealth(provider_type))

            try:
                provider = await self.get_provider(provider_type)

                # Simple health check - validate a common model
                if provider_type == ProviderType.OPENAI:
                    is_available = await provider.validate_model("gpt-4")
                elif provider_type == ProviderType.ANTHROPIC:
                    is_available = await provider.validate_model("claude-3-sonnet")
                elif provider_type == ProviderType.OLLAMA:
                    is_available = await provider.validate_model("llama3.2")
                else:
                    is_available = True  # Mock is always available

                results[provider_type.value] = {
                    "available": is_available,
                    "healthy": health.is_healthy,
                    "success_rate": health.get_success_rate(),
                    "average_latency": health.average_latency,
                    "consecutive_failures": health.consecutive_failures,
                    "last_success": (
                        health.last_success.isoformat() if health.last_success else None
                    ),
                    "last_failure": (
                        health.last_failure.isoformat() if health.last_failure else None
                    ),
                }

            except Exception as e:
                results[provider_type.value] = {
                    "available": False,
                    "healthy": False,
                    "error": str(e),
                }

        return {
            "primary_provider": self._primary_provider.value if self._primary_provider else None,
            "fallback_chain": [p.value for p in self._fallback_chain],
            "providers": results,
        }

    async def close(self):
        """Close all provider connections."""
        for provider in self._providers.values():
            if hasattr(provider, "close"):
                await provider.close()

    def get_config_template(self) -> Dict[str, Any]:
        """Get a configuration template for all providers."""
        return {
            "provider": "auto",  # auto, openai, anthropic, ollama, mock
            "openai": {
                "api_key": "sk-...",  # Or use OPENAI_API_KEY env var
                "model": "gpt-4o",
                "organization": None,
                "base_url": "https://api.openai.com/v1",
                "timeout": 60.0,
                "max_retries": 3,
            },
            "anthropic": {
                "api_key": "sk-ant-...",  # Or use ANTHROPIC_API_KEY env var
                "model": "claude-3-5-sonnet-20241022",
                "base_url": "https://api.anthropic.com",
                "timeout": 60.0,
                "max_retries": 3,
            },
            "ollama": {
                "model": "llama3.2",
                "base_url": "http://localhost:11434",
                "timeout": 120.0,
                "keep_alive": "5m",
                "num_ctx": None,  # Context window override
                "num_predict": None,  # Max tokens override
                "num_gpu": None,  # GPU layers
            },
            "mock": {"delay": 0.1, "error_rate": 0.0},
        }


# Convenience function for creating a factory
def create_llm_factory(config: Optional[Dict[str, Any]] = None) -> LLMProviderFactory:
    """Create an LLM provider factory with the given configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured LLMProviderFactory instance
    """
    return LLMProviderFactory(config)

."""

LLM Inference Module
Multi-provider LLM management with usage tracking, health monitoring, and
    secure credential handling.
"""

# Fallback mechanisms
from .fallback_mechanisms import (
    FallbackLevel,
    FallbackManager,
    FallbackResponse,
    ResourceConstraints,
    ResponseCache,
)

# Local LLM support
from .local_llm_manager import (
    LlamaCppProvider,
    LLMResponse,
    LocalLLMConfig,
    LocalLLMManager,
    OllamaProvider,
)

# Provider interface exports
from .provider_interface import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    HealthCheckResult,
    ILLMProvider,
    ModelCapability,
    ModelInfo,
    ProviderCredentials,
    ProviderManager,
    ProviderRegistry,
    ProviderStatus,
    ProviderType,
    UsageMetrics,
)

__all__ = [
    # Provider interface
    "ILLMProvider",
    "BaseProvider",
    "ProviderType",
    "ProviderStatus",
    "ModelCapability",
    "ProviderCredentials",
    "ModelInfo",
    "UsageMetrics",
    "HealthCheckResult",
    "GenerationRequest",
    "GenerationResponse",
    "ProviderRegistry",
    "ProviderManager",
    # Local LLM
    "LocalLLMManager",
    "LocalLLMConfig",
    "LLMResponse",
    "OllamaProvider",
    "LlamaCppProvider",
    # Fallback
    "FallbackManager",
    "FallbackLevel",
    "FallbackResponse",
    "ResourceConstraints",
    "ResponseCache",
]

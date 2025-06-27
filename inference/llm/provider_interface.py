."""

LLMProvider Interface for Multi-Provider Management
Provides unified interface for managing multiple LLM providers with advanced features
including usage tracking, health monitoring, and secure credential handling.

ADR-002 Compliant: Core domain logic in /inference/llm
Integrates with existing fallback mechanisms and local LLM patterns.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetimedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    AZURE_OPENAI = "azure_openai"
    VERTEX_AI = "vertex_ai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LOCAL = "local"


class ProviderStatus(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    RATE_LIMITED = "rate_limited"


class ModelCapability(Enum):
    """Model capabilities."""

    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    CODE_GENERATION = "code_generation"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    STREAMING = "streaming"


@dataclass
class ProviderCredentials:
    """Secure credentials for LLM providers."""

    api_key: Optional[str] = None
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    region: Optional[str] = None
    # Encrypted storage reference
    encrypted_credential_id: Optional[str] = None

    def is_complete(self) -> bool:
        """Check if credentials are complete for the provider."""
        return bool(self.api_key or self.encrypted_credential_id)


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    provider: ProviderType
    capabilities: List[ModelCapability]
    context_window: int
    max_output_tokens: int
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    supports_streaming: bool = False
    supports_function_calling: bool = False
    description: Optional[str] = None
    version: Optional[str] = None


@dataclass
class UsageMetrics:
    """Usage tracking metrics for a provider."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    average_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    daily_usage: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)

    def update_request(
        self,
        success: bool,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost: float,
        error_type: Optional[str] = None,
    ) -> None:
        """Update metrics with new request data."""
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.last_request_time = datetime.now()

        # Update success/failure counts
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Update average latency
        if self.successful_requests > 0:
            self.average_latency_ms = (
                self.average_latency_ms * (self.successful_requests - 1) + latency_ms
            ) / self.successful_requests

        # Update daily usage
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_usage[today] = self.daily_usage.get(today, 0) + 1


@dataclass
class HealthCheckResult:
    """Result of provider health check."""

    status: ProviderStatus
    latency_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    model_availability: Dict[str, bool] = field(default_factory=dict)
    rate_limit_info: Optional[Dict[str, Any]] = None


@dataclass
class GenerationRequest:
    """Request for text generation."""

    model: str
    messages: List[Dict[str, str]]  # Chat format: [{"role": "user", "content": "..."}]
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    model: str
    provider: ProviderType
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    finish_reason: str
    function_call: Optional[Dict[str, Any]] = None
    usage_metadata: Optional[Dict[str, Any]] = None


class ILLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        pass

    @abstractmethod
    def configure(self, credentials: ProviderCredentials, **kwargs: Any) -> bool:
        """Configure the provider with credentials and settings."""
        pass

    @abstractmethod
    def test_connection(self) -> HealthCheckResult:
        """Test connection to the provider."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the provider."""
        pass

    @abstractmethod
    def estimate_cost(self, request: GenerationRequest) -> float:
        """Estimate cost for a generation request."""
        pass

    @abstractmethod
    def get_usage_metrics(self) -> UsageMetrics:
        """Get current usage metrics."""
        pass

    @abstractmethod
    def reset_usage_metrics(self) -> None:
        """Reset usage metrics."""
        pass

    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get current rate limit information."""
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        pass

    @abstractmethod
    def supports_function_calling(self) -> bool:
        """Check if provider supports function calling."""
        pass


class BaseProvider(ILLMProvider):
    """Base implementation with common functionality."""

    def __init__(self, provider_type: ProviderType) -> None:
        """Initialize."""
        self.provider_type = provider_type
        self.credentials: Optional[ProviderCredentials] = None
        self.usage_metrics = UsageMetrics()
        self._last_health_check: Optional[HealthCheckResult] = None
        self._health_check_interval = 300  # 5 minutes
        self._configuration: Dict[str, Any] = {}

    def get_provider_type(self) -> ProviderType:
        return self.provider_type

    def configure(self, credentials: ProviderCredentials, **kwargs: Any) -> bool:
        """Configure the provider with credentials and settings."""
        if not credentials.is_complete():
            logger.error(f"Incomplete credentials for {self.provider_type}")
            return False

        self.credentials = credentials
        self._configuration.update(kwargs)

        # Test connection after configuration
        health_check = self.test_connection()
        return health_check.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]

    def get_usage_metrics(self) -> UsageMetrics:
        return self.usage_metrics

    def reset_usage_metrics(self) -> None:
        self.usage_metrics = UsageMetrics()

    def _should_perform_health_check(self) -> bool:
        """Check if health check is needed."""
        if not self._last_health_check:
            return True

        time_since_check = datetime.now() - self._last_health_check.timestamp
        return time_since_check.total_seconds() > self._health_check_interval

    def _update_usage_metrics(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        """Update usage metrics with request/response data."""
        self.usage_metrics.update_request(
            success=success,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            cost=response.cost,
            error_type=error_type,
        )


class ProviderRegistry:
    """Registry for managing multiple LLM providers."""

    def __init__(self) -> None:
        """Initialize."""
        self._providers: Dict[ProviderType, ILLMProvider] = {}
        self._provider_priorities: List[ProviderType] = []
        self._health_check_cache: Dict[ProviderType, HealthCheckResult] = {}

    def register_provider(self, provider: ILLMProvider, priority: int = 100) -> None:
        """Register a provider with given priority (lower = higher priority)."""
        provider_type = provider.get_provider_type()
        self._providers[provider_type] = provider

        # Insert at appropriate priority position
        if provider_type in self._provider_priorities:
            self._provider_priorities.remove(provider_type)

        # Find insertion point based on priority
        insertion_point = len(self._provider_priorities)
        for i, existing_type in enumerate(self._provider_priorities):
            # This is a simplified priority system - in practice, you'd store priorities separately
            if priority < 100:  # Assume high priority if less than 100
                insertion_point = i
                break

        self._provider_priorities.insert(insertion_point, provider_type)
        logger.info(f"Registered provider {provider_type} with priority {priority}")

    def get_provider(self, provider_type: ProviderType) -> Optional[ILLMProvider]:
        """Get provider by type."""
        return self._providers.get(provider_type)

    def get_providers_by_priority(self) -> List[ILLMProvider]:
        """Get providers ordered by priority."""
        return [self._providers[pt] for pt in self._provider_priorities if pt in self._providers]

    def get_healthy_providers(self) -> List[ILLMProvider]:
        """Get only healthy providers."""
        healthy = []
        for provider in self.get_providers_by_priority():
            health = provider.test_connection()
            if health.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
                healthy.append(provider)
        return healthy

    def reorder_providers(self, new_order: List[ProviderType]) -> None:
        """Reorder provider priority."""
        # Validate all providers exist
        for provider_type in new_order:
            if provider_type not in self._providers:
                raise ValueError(f"Unknown provider type: {provider_type}")

        self._provider_priorities = new_order
        logger.info(f"Reordered providers: {[pt.value for pt in new_order]}")

    def remove_provider(self, provider_type: ProviderType) -> None:
        """Remove a provider from the registry."""
        if provider_type in self._providers:
            del self._providers[provider_type]
        if provider_type in self._provider_priorities:
            self._provider_priorities.remove(provider_type)
        logger.info(f"Removed provider {provider_type}")


class ProviderManager:
    """High-level manager for LLM provider operations."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize."""
        self.registry = ProviderRegistry()
        self._config_path = config_path
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load provider configuration from file."""
        if not self._config_path or not self._config_path.exists():
            return

        try:
            with open(self._config_path) as f:
                config = json.load(f)

            # Load providers from configuration
            for provider_config in config.get("providers", []):
                provider_type = ProviderType(provider_config["type"])
                # Implementation would create specific provider instances
                # This is a placeholder for the full implementation
                logger.info(f"Loaded provider configuration for {provider_type}")

        except Exception as e:
            logger.error(f"Failed to load provider configuration: {e}")

    def generate_with_fallback(self, request: GenerationRequest) -> GenerationResponse:
        """Generate with automatic provider fallback."""
        last_error = None

        for provider in self.registry.get_healthy_providers():
            try:
                logger.debug(f"Attempting generation with {provider.get_provider_type()}")
                response = provider.generate(request)
                logger.info(f"Successfully generated with {provider.get_provider_type()}")
                return response
            except Exception as e:
                logger.warning(f"Provider {provider.get_provider_type()} failed: {e}")
                last_error = e
                continue

        if last_error:
            raise last_error
        else:
            raise Exception("No healthy providers available")

    def get_all_usage_metrics(self) -> Dict[ProviderType, UsageMetrics]:
        """Get usage metrics for all providers."""
        return {
            provider_type: provider.get_usage_metrics()
            for provider_type, provider in self.registry._providers.items()
        }

    def perform_health_checks(self) -> Dict[ProviderType, HealthCheckResult]:
        """Perform health checks on all providers."""
        results = {}
        for provider_type, provider in self.registry._providers.items():
            try:
                health_result = provider.test_connection()
                results[provider_type] = health_result
                logger.debug(f"Health check for {provider_type}: {health_result.status}")
            except Exception as e:
                results[provider_type] = HealthCheckResult(
                    status=ProviderStatus.OFFLINE, latency_ms=0, error_message=str(e)
                )
        return results

    def get_provider_recommendations(
        self, request: GenerationRequest
    ) -> List[Tuple[ILLMProvider, float]]:
        """Get provider recommendations with confidence scores."""
        recommendations = []

        for provider in self.registry.get_healthy_providers():
            # Calculate recommendation score based on various factors
            score = 0.0

            # Base score from health
            health = provider.test_connection()
            if health.status == ProviderStatus.HEALTHY:
                score += 0.4
            elif health.status == ProviderStatus.DEGRADED:
                score += 0.2

            # Score from usage metrics
            metrics = provider.get_usage_metrics()
            if metrics.total_requests > 0:
                success_rate = metrics.successful_requests / metrics.total_requests
                score += success_rate * 0.3

                # Prefer faster providers
                if metrics.average_latency_ms > 0:
                    latency_score = max(
                        0, 1 - (metrics.average_latency_ms / 10000)
                    )  # Normalize to 10s
                    score += latency_score * 0.2

            # Cost consideration
            estimated_cost = provider.estimate_cost(request)
            if estimated_cost > 0:
                # Prefer lower cost (this is simplified scoring)
                cost_score = max(0, 1 - (estimated_cost / 0.1))  # Normalize to $0.10
                score += cost_score * 0.1

            recommendations.append((provider, score))

        # Sort by score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

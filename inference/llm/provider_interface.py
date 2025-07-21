"""Provider interface for LLM integrations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class ProviderStatus(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class ModelCapability(Enum):
    """Model capability types."""

    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"


@dataclass
class ProviderCredentials:
    """Credentials for accessing LLM provider."""

    api_key: Optional[str] = None
    organization_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    encrypted_credential_id: Optional[str] = None

    def is_complete(self) -> bool:
        """Check if credentials are complete for authentication."""
        return bool(self.api_key or self.encrypted_credential_id)


@dataclass
class ModelInfo:
    """Information about a specific model."""

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


@dataclass
class UsageMetrics:
    """Tracks usage metrics for a provider."""

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
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        error_type: Optional[str] = None,
    ) -> None:
        """Update metrics with new request data."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        # Update average latency
        if latency_ms > 0:
            if self.average_latency_ms == 0:
                self.average_latency_ms = latency_ms
            else:
                # Running average
                self.average_latency_ms = (
                    self.average_latency_ms * (self.successful_requests - 1) +
                        latency_ms
                ) / self.successful_requests

        self.last_request_time = datetime.now()

        # Update daily usage
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_usage[today] = self.daily_usage.get(today, 0) + 1


@dataclass
class HealthCheckResult:
    """Result of a provider health check."""

    status: ProviderStatus
    latency_ms: float
    error_message: Optional[str] = None
    model_availability: Dict[str, bool] = field(default_factory=dict)
    rate_limit_info: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GenerationRequest:
    """Request for text generation."""

    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    model: str
    provider: ProviderType
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None


class ILLMProvider(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    def get_provider_type(self) -> ProviderType:
        """Return the provider type."""

    @abstractmethod
    def configure(self, credentials: ProviderCredentials, **kwargs: Any) -> bool:
        """Configure the provider with credentials."""

    @abstractmethod
    def test_connection(self) -> HealthCheckResult:
        """Test connection to the provider."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text based on the request."""

    @abstractmethod
    def get_usage_metrics(self) -> UsageMetrics:
        """Get current usage metrics."""

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for given token counts."""


class BaseProvider(ILLMProvider):
    """Base implementation of LLM provider."""

    def __init__(self, provider_type: ProviderType):
        """Initialize base provider."""
        self.provider_type = provider_type
        self.credentials: Optional[ProviderCredentials] = None
        self.usage_metrics = UsageMetrics()
        self._last_health_check: Optional[datetime] = None
        self._health_check_interval = 300  # 5 minutes
        self._configuration: Dict[str, Any] = {}

    def get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return self.provider_type

    def configure(self, credentials: ProviderCredentials, **kwargs: Any) -> bool:
        """Configure the provider with credentials.

        Returns:
            True if configuration successful, False otherwise
        """
        if not credentials.is_complete():
            logger.error(f"Invalid credentials for {self.provider_type.value}")
            return False

        self.credentials = credentials
        self._configuration.update(kwargs)

        # Test connection if credentials are valid
        try:
            health_result = self.test_connection()
            if health_result.status in [
                ProviderStatus.HEALTHY,
                ProviderStatus.DEGRADED,
            ]:
                logger.info(f"Configured {self.provider_type.value} provider successfully")
                return True
            else:
                logger.error(
                    f"Provider {self.provider_type.value} unhealthy: {health_result.error_message}"
                )
                return False
        except Exception as e:
            logger.error(
                f"Failed to test connection for {self.provider_type.value}:"
                f" {e}"
            )
            return False

    def get_usage_metrics(self) -> UsageMetrics:
        """Get current usage metrics."""
        return self.usage_metrics

    def reset_usage_metrics(self) -> None:
        """Reset usage metrics."""
        self.usage_metrics = UsageMetrics()

    def _should_perform_health_check(self) -> bool:
        """Determine if health check should be performed."""
        if self._last_health_check is None:
            return True

        elapsed = (datetime.now() - self._last_health_check).total_seconds()
        return elapsed >= self._health_check_interval

    def _update_usage_metrics(
        self,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        error_type: Optional[str] = None,
    ) -> None:
        """Update usage metrics after a request."""
        self.usage_metrics.update_request(
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            error_type=error_type,
        )

    @abstractmethod
    def test_connection(self) -> HealthCheckResult:
        """Test connection to the provider."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text based on the request."""

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for given token counts."""


class ProviderRegistry:
    """Registry for managing multiple providers."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._providers: Dict[ProviderType, ILLMProvider] = {}
        self._provider_priorities: List[ProviderType] = []
        self._provider_priority_values: Dict[ProviderType, int] = {}
        self._health_check_cache: Dict[ProviderType, HealthCheckResult] = {}

    def register_provider(self, provider: ILLMProvider, priority: int = 50) -> None:
        """Register a provider with optional priority."""
        provider_type = provider.get_provider_type()
        self._providers[provider_type] = provider
        self._provider_priority_values[provider_type] = priority

        # Insert into priority list
        if provider_type not in self._provider_priorities:
            self._provider_priorities.append(provider_type)

        # Sort by priority (lower number = higher priority)
        self._provider_priorities.sort(key=lambda pt: self._provider_priority_values[pt])

        logger.info(
            f"Registered provider: {provider_type.value} with priority"
            f" {priority}"
        )

    def get_provider(self, provider_type: ProviderType) -> Optional[ILLMProvider]:
        """Get a specific provider by type."""
        return self._providers.get(provider_type)

    def get_providers_by_priority(self) -> List[ILLMProvider]:
        """Get all providers sorted by priority (highest first)."""
        return [self._providers[pt] for pt in self._provider_priorities if
            pt in self._providers]

    def get_healthy_providers(self) -> List[ILLMProvider]:
        """Get all healthy providers."""
        healthy_providers = []
        for provider_type, provider in self._providers.items():
            # Check cache or perform health check
            if provider_type in self._health_check_cache:
                result = self._health_check_cache[provider_type]
            else:
                result = provider.test_connection()
                self._health_check_cache[provider_type] = result

            if result.status in [
                ProviderStatus.HEALTHY,
                ProviderStatus.DEGRADED,
            ]:
                healthy_providers.append(provider)
        return healthy_providers

    def reorder_providers(self, priorities: List[ProviderType]) -> None:
        """Reorder providers based on new priorities."""
        for provider_type in priorities:
            if provider_type not in self._providers:
                raise ValueError(f"Unknown provider type: {provider_type.value}")
        self._provider_priorities = priorities

    def remove_provider(self, provider_type: ProviderType) -> bool:
        """Remove a provider from the registry."""
        if provider_type in self._providers:
            del self._providers[provider_type]
            if provider_type in self._provider_priorities:
                self._provider_priorities.remove(provider_type)
            if provider_type in self._health_check_cache:
                del self._health_check_cache[provider_type]
            logger.info(f"Removed provider: {provider_type.value}")
            return True
        return False


class ProviderManager:
    """Manages multiple LLM providers with fallback support."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the provider manager."""
        self.registry = ProviderRegistry()
        self._config_path = config_path
        if config_path:
            self._load_configuration(config_path)

    def generate_with_fallback(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text with automatic fallback to healthy providers."""
        providers = self.registry.get_healthy_providers()

        if not providers:
            raise Exception("No healthy providers available")

        for provider in providers:
            try:
                # Try to generate
                response = provider.generate(request)
                return response

            except Exception as e:
                logger.error(
                    f"Provider {provider.get_provider_type().value} failed:"
                    f" {str(e)}"
                )
                continue

        raise Exception("All LLM providers failed to generate response")

    def get_all_usage_metrics(self) -> Dict[ProviderType, UsageMetrics]:
        """Get usage metrics for all providers."""
        metrics = {}
        for provider_type, provider in self.registry._providers.items():
            metrics[provider_type] = provider.get_usage_metrics()
        return metrics

    def perform_health_checks(self) -> Dict[ProviderType, HealthCheckResult]:
        """Perform health checks on all providers."""
        results = {}
        for provider_type, provider in self.registry._providers.items():
            try:
                result = provider.test_connection()
                results[provider_type] = result
                self.registry._health_check_cache[provider_type] = result
            except Exception as e:
                results[provider_type] = HealthCheckResult(
                    status=ProviderStatus.OFFLINE,
                    latency_ms=0,
                    error_message=str(e),
                )
        return results

    def get_provider_recommendations(
        self, request: GenerationRequest
    ) -> List[Tuple[ILLMProvider, float]]:
        """Get provider recommendations based on request and metrics."""
        recommendations = []

        for provider in self.registry.get_healthy_providers():
            # Calculate score based on various factors
            score = 1.0

            # Check usage metrics
            metrics = provider.get_usage_metrics()
            if metrics.total_requests > 0:
                success_rate = metrics.successful_requests / metrics.total_requests
                score *= success_rate

                # Penalize high latency
                if metrics.average_latency_ms > 0:
                    latency_penalty = min(1.0, 1000.0 / metrics.average_latency_ms)
                    score *= latency_penalty

            # Check cost if available
            try:
                cost = provider.estimate_cost(100, 100, request.model)
                if cost > 0:
                    cost_factor = min(1.0, 0.1 / cost)  # Lower cost is better
                    score *= cost_factor
            except Exception:
                pass

            recommendations.append((provider, score))

        # Sort by score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def _load_configuration(self, config_path: str) -> None:
        """Load provider configuration from file."""
        # Implementation would load from JSON/YAML file
        logger.info(f"Loading configuration from {config_path}")

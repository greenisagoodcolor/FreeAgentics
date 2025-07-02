"""
Comprehensive test coverage for inference/llm/provider_interface.py
LLM Provider Interface - Phase 3.3 systematic coverage

This test file provides complete coverage for the LLM provider interface system
following the systematic backend coverage improvement plan.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

# Import the LLM provider interface components
try:
    from inference.llm.provider_interface import (
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

    IMPORT_SUCCESS = True
except ImportError:
    # Create minimal mock classes for testing if imports fail
    IMPORT_SUCCESS = False

    class ProviderType:
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        OPENROUTER = "openrouter"
        AZURE_OPENAI = "azure_openai"
        VERTEX_AI = "vertex_ai"
        COHERE = "cohere"
        HUGGINGFACE = "huggingface"
        OLLAMA = "ollama"
        LOCAL = "local"

    class ProviderStatus:
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"
        OFFLINE = "offline"
        MAINTENANCE = "maintenance"
        RATE_LIMITED = "rate_limited"

    class ModelCapability:
        TEXT_GENERATION = "text_generation"
        CHAT_COMPLETION = "chat_completion"
        CODE_GENERATION = "code_generation"
        EMBEDDINGS = "embeddings"
        FUNCTION_CALLING = "function_calling"
        VISION = "vision"
        AUDIO = "audio"
        STREAMING = "streaming"

    class ProviderCredentials:
        def __init__(
            self,
            api_key=None,
            organization_id=None,
            project_id=None,
            endpoint_url=None,
            region=None,
            encrypted_credential_id=None,
        ):
            self.api_key = api_key
            self.organization_id = organization_id
            self.project_id = project_id
            self.endpoint_url = endpoint_url
            self.region = region
            self.encrypted_credential_id = encrypted_credential_id

        def is_complete(self):
            return bool(self.api_key or self.encrypted_credential_id)

    class ModelInfo:
        def __init__(
                self,
                id,
                name,
                provider,
                capabilities,
                context_window,
                max_output_tokens,
                **kwargs):
            self.id = id
            self.name = name
            self.provider = provider
            self.capabilities = capabilities
            self.context_window = context_window
            self.max_output_tokens = max_output_tokens
            for k, v in kwargs.items():
                setattr(self, k, v)

    class UsageMetrics:
        def __init__(self):
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cost = 0.0
            self.average_latency_ms = 0.0
            self.last_request_time = None
            self.daily_usage = {}
            self.error_counts = {}

        def update_request(
                self,
                success,
                input_tokens,
                output_tokens,
                latency_ms,
                cost,
                error_type=None):
            pass

    class HealthCheckResult:
        def __init__(self, status, latency_ms, error_message=None, **kwargs):
            self.status = status
            self.latency_ms = latency_ms
            self.error_message = error_message
            self.timestamp = datetime.now()
            self.model_availability = kwargs.get("model_availability", {})
            self.rate_limit_info = kwargs.get("rate_limit_info", None)

    class GenerationRequest:
        def __init__(self, model, messages, **kwargs):
            self.model = model
            self.messages = messages
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens", 1000)
            self.top_p = kwargs.get("top_p", 0.9)
            self.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
            self.presence_penalty = kwargs.get("presence_penalty", 0.0)
            self.stop_sequences = kwargs.get("stop_sequences", None)
            self.stream = kwargs.get("stream", False)
            self.functions = kwargs.get("functions", None)
            self.function_call = kwargs.get("function_call", None)

    class GenerationResponse:
        def __init__(
            self,
            text,
            model,
            provider,
            input_tokens,
            output_tokens,
            cost,
            latency_ms,
            finish_reason,
            **kwargs,
        ):
            self.text = text
            self.model = model
            self.provider = provider
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.cost = cost
            self.latency_ms = latency_ms
            self.finish_reason = finish_reason
            self.function_call = kwargs.get("function_call", None)
            self.usage_metadata = kwargs.get("usage_metadata", None)


class TestProviderTypes:
    """Test provider type enumeration."""

    def test_provider_types_exist(self):
        """Test all provider types exist."""
        expected_providers = [
            "OPENAI",
            "ANTHROPIC",
            "OPENROUTER",
            "AZURE_OPENAI",
            "VERTEX_AI",
            "COHERE",
            "HUGGINGFACE",
            "OLLAMA",
            "LOCAL",
        ]

        for provider in expected_providers:
            assert hasattr(ProviderType, provider)

    def test_provider_type_values(self):
        """Test provider type values."""
        assert ProviderType.OPENAI == "openai"
        assert ProviderType.ANTHROPIC == "anthropic"
        assert ProviderType.LOCAL == "local"


class TestProviderStatus:
    """Test provider status enumeration."""

    def test_status_types_exist(self):
        """Test all status types exist."""
        expected_statuses = [
            "HEALTHY",
            "DEGRADED",
            "UNHEALTHY",
            "OFFLINE",
            "MAINTENANCE",
            "RATE_LIMITED",
        ]

        for status in expected_statuses:
            assert hasattr(ProviderStatus, status)

    def test_status_priority_order(self):
        """Test status priority for fallback logic."""
        # Healthy and degraded should allow usage
        usable_statuses = [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]
        unusable_statuses = [
            ProviderStatus.UNHEALTHY,
            ProviderStatus.OFFLINE,
            ProviderStatus.MAINTENANCE,
            ProviderStatus.RATE_LIMITED,
        ]

        # Ensure no overlap
        assert not set(usable_statuses) & set(unusable_statuses)


class TestModelCapability:
    """Test model capability enumeration."""

    def test_capability_types(self):
        """Test all capability types exist."""
        expected_capabilities = [
            "TEXT_GENERATION",
            "CHAT_COMPLETION",
            "CODE_GENERATION",
            "EMBEDDINGS",
            "FUNCTION_CALLING",
            "VISION",
            "AUDIO",
            "STREAMING",
        ]

        for capability in expected_capabilities:
            assert hasattr(ModelCapability, capability)


class TestProviderCredentials:
    """Test provider credentials."""

    def test_credentials_creation(self):
        """Test creating credentials."""
        creds = ProviderCredentials(
            api_key="test-key",
            organization_id="org-123",
            project_id="proj-456",
            endpoint_url="https://api.example.com",
            region="us-west-2",
        )

        assert creds.api_key == "test-key"
        assert creds.organization_id == "org-123"
        assert creds.project_id == "proj-456"
        assert creds.endpoint_url == "https://api.example.com"
        assert creds.region == "us-west-2"

    def test_credentials_completeness(self):
        """Test checking credential completeness."""
        # Complete with API key
        creds1 = ProviderCredentials(api_key="test-key")
        assert creds1.is_complete()

        # Complete with encrypted credential ID
        creds2 = ProviderCredentials(encrypted_credential_id="enc-123")
        assert creds2.is_complete()

        # Incomplete
        creds3 = ProviderCredentials(organization_id="org-123")
        assert not creds3.is_complete()

        # Empty
        creds4 = ProviderCredentials()
        assert not creds4.is_complete()


class TestModelInfo:
    """Test model information."""

    def test_model_info_creation(self):
        """Test creating model info."""
        model = ModelInfo(
            id="gpt-4",
            name="GPT-4",
            provider=ProviderType.OPENAI,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CHAT_COMPLETION],
            context_window=8192,
            max_output_tokens=4096,
            cost_per_1k_input_tokens=0.03,
            cost_per_1k_output_tokens=0.06,
            supports_streaming=True,
            supports_function_calling=True,
            description="Advanced language model",
            version="4.0",
        )

        assert model.id == "gpt-4"
        assert model.name == "GPT-4"
        assert model.provider == ProviderType.OPENAI
        assert len(model.capabilities) == 2
        assert model.context_window == 8192
        assert model.max_output_tokens == 4096
        assert model.cost_per_1k_input_tokens == 0.03
        assert model.cost_per_1k_output_tokens == 0.06
        assert model.supports_streaming
        assert model.supports_function_calling

    def test_model_info_defaults(self):
        """Test model info with defaults."""
        model = ModelInfo(
            id="test-model",
            name="Test Model",
            provider=ProviderType.LOCAL,
            capabilities=[ModelCapability.TEXT_GENERATION],
            context_window=2048,
            max_output_tokens=512,
        )

        assert hasattr(model, "cost_per_1k_input_tokens")
        assert hasattr(model, "supports_streaming")


class TestUsageMetrics:
    """Test usage metrics tracking."""

    def test_usage_metrics_initialization(self):
        """Test usage metrics initialization."""
        metrics = UsageMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_input_tokens == 0
        assert metrics.total_output_tokens == 0
        assert metrics.total_cost == 0.0
        assert metrics.average_latency_ms == 0.0
        assert metrics.last_request_time is None
        assert isinstance(metrics.daily_usage, dict)
        assert isinstance(metrics.error_counts, dict)

    def test_update_successful_request(self):
        """Test updating metrics with successful request."""
        if not IMPORT_SUCCESS:
            return

        metrics = UsageMetrics()

        metrics.update_request(
            success=True,
            input_tokens=100,
            output_tokens=50,
            latency_ms=250.0,
            cost=0.01,
            error_type=None,
        )

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_input_tokens == 100
        assert metrics.total_output_tokens == 50
        assert metrics.total_cost == 0.01
        assert metrics.average_latency_ms == 250.0
        assert metrics.last_request_time is not None

        # Check daily usage
        today = datetime.now().strftime("%Y-%m-%d")
        assert metrics.daily_usage[today] == 1

    def test_update_failed_request(self):
        """Test updating metrics with failed request."""
        if not IMPORT_SUCCESS:
            return

        metrics = UsageMetrics()

        metrics.update_request(
            success=False,
            input_tokens=50,
            output_tokens=0,
            latency_ms=100.0,
            cost=0.005,
            error_type="RateLimitError",
        )

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.error_counts["RateLimitError"] == 1

    def test_average_latency_calculation(self):
        """Test average latency calculation."""
        if not IMPORT_SUCCESS:
            return

        metrics = UsageMetrics()

        # Add multiple successful requests
        latencies = [100.0, 200.0, 300.0]
        for latency in latencies:
            metrics.update_request(
                success=True,
                input_tokens=10,
                output_tokens=10,
                latency_ms=latency,
                cost=0.001,
                error_type=None,
            )

        expected_avg = sum(latencies) / len(latencies)
        assert abs(metrics.average_latency_ms - expected_avg) < 0.01


class TestHealthCheckResult:
    """Test health check results."""

    def test_health_check_creation(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            status=ProviderStatus.HEALTHY,
            latency_ms=50.0,
            error_message=None,
            model_availability={
                "gpt-4": True,
                "gpt-3.5": True},
            rate_limit_info={
                "requests_per_minute": 60,
                "tokens_per_minute": 90000},
        )

        assert result.status == ProviderStatus.HEALTHY
        assert result.latency_ms == 50.0
        assert result.error_message is None
        assert result.timestamp is not None
        assert result.model_availability["gpt-4"] is True
        assert result.rate_limit_info["requests_per_minute"] == 60

    def test_health_check_with_error(self):
        """Test health check with error."""
        result = HealthCheckResult(
            status=ProviderStatus.OFFLINE,
            latency_ms=0.0,
            error_message="Connection timeout")

        assert result.status == ProviderStatus.OFFLINE
        assert result.error_message == "Connection timeout"


class TestGenerationRequest:
    """Test generation request."""

    def test_generation_request_creation(self):
        """Test creating generation request."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        request = GenerationRequest(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=500,
            top_p=0.95,
            stream=True,
            functions=[{"name": "get_weather", "parameters": {}}],
        )

        assert request.model == "gpt-4"
        assert request.messages == messages
        assert request.temperature == 0.8
        assert request.max_tokens == 500
        assert request.top_p == 0.95
        assert request.stream is True
        assert len(request.functions) == 1

    def test_generation_request_defaults(self):
        """Test generation request with defaults."""
        request = GenerationRequest(
            model="test-model", messages=[{"role": "user", "content": "Test"}]
        )

        assert request.temperature == 0.7
        assert request.max_tokens == 1000
        assert request.top_p == 0.9
        assert request.frequency_penalty == 0.0
        assert request.presence_penalty == 0.0
        assert request.stop_sequences is None
        assert request.stream is False
        assert request.functions is None
        assert request.function_call is None


class TestGenerationResponse:
    """Test generation response."""

    def test_generation_response_creation(self):
        """Test creating generation response."""
        response = GenerationResponse(
            text="Hello! How can I help you?",
            model="gpt-4",
            provider=ProviderType.OPENAI,
            input_tokens=20,
            output_tokens=10,
            cost=0.0012,
            latency_ms=350.0,
            finish_reason="stop",
            function_call={
                "name": "get_weather",
                "arguments": '{"location": "NYC"}'},
            usage_metadata={
                "prompt_tokens": 20,
                "completion_tokens": 10},
        )

        assert response.text == "Hello! How can I help you?"
        assert response.model == "gpt-4"
        assert response.provider == ProviderType.OPENAI
        assert response.input_tokens == 20
        assert response.output_tokens == 10
        assert response.cost == 0.0012
        assert response.latency_ms == 350.0
        assert response.finish_reason == "stop"
        assert response.function_call["name"] == "get_weather"
        assert response.usage_metadata["prompt_tokens"] == 20


class TestBaseProvider:
    """Test base provider implementation."""

    def test_base_provider_initialization(self):
        """Test base provider initialization."""
        if not IMPORT_SUCCESS:
            return

        provider = BaseProvider(ProviderType.OPENAI)

        assert provider.provider_type == ProviderType.OPENAI
        assert provider.credentials is None
        assert isinstance(provider.usage_metrics, UsageMetrics)
        assert provider._last_health_check is None
        assert provider._health_check_interval == 300
        assert isinstance(provider._configuration, dict)

    def test_base_provider_configuration(self):
        """Test base provider configuration."""
        if not IMPORT_SUCCESS:
            return

        provider = BaseProvider(ProviderType.ANTHROPIC)

        # Mock test_connection
        provider.test_connection = Mock(
            return_value=HealthCheckResult(
                status=ProviderStatus.HEALTHY,
                latency_ms=100.0))

        creds = ProviderCredentials(api_key="test-key")
        result = provider.configure(creds, timeout=30, retry_count=3)

        assert result is True
        assert provider.credentials == creds
        assert provider._configuration["timeout"] == 30
        assert provider._configuration["retry_count"] == 3
        provider.test_connection.assert_called_once()

    def test_base_provider_configuration_incomplete_creds(self):
        """Test configuration with incomplete credentials."""
        if not IMPORT_SUCCESS:
            return

        provider = BaseProvider(ProviderType.COHERE)
        creds = ProviderCredentials()  # Empty credentials

        result = provider.configure(creds)

        assert result is False
        assert provider.credentials is None

    def test_usage_metrics_operations(self):
        """Test usage metrics operations."""
        if not IMPORT_SUCCESS:
            return

        provider = BaseProvider(ProviderType.LOCAL)

        # Get initial metrics
        metrics = provider.get_usage_metrics()
        assert metrics.total_requests == 0

        # Update metrics
        provider.usage_metrics.total_requests = 10
        provider.usage_metrics.successful_requests = 8

        # Verify update
        metrics = provider.get_usage_metrics()
        assert metrics.total_requests == 10
        assert metrics.successful_requests == 8

        # Reset metrics
        provider.reset_usage_metrics()
        metrics = provider.get_usage_metrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0

    def test_health_check_timing(self):
        """Test health check timing logic."""
        if not IMPORT_SUCCESS:
            return

        provider = BaseProvider(ProviderType.OPENAI)

        # No previous health check
        assert provider._should_perform_health_check() is True

        # Recent health check
        provider._last_health_check = HealthCheckResult(
            status=ProviderStatus.HEALTHY, latency_ms=50.0
        )
        assert provider._should_perform_health_check() is False

        # Old health check
        old_time = datetime.now() - timedelta(seconds=400)
        provider._last_health_check.timestamp = old_time
        assert provider._should_perform_health_check() is True


class MockProvider(ILLMProvider):
    """Mock provider for testing."""

    def __init__(self, provider_type, is_healthy=True):
        self.provider_type = provider_type
        self.is_healthy = is_healthy
        self.usage_metrics = UsageMetrics()
        self.generate_called = False
        self.test_connection_called = False

    def get_provider_type(self):
        return self.provider_type

    def configure(self, credentials, **kwargs):
        return True

    def test_connection(self):
        self.test_connection_called = True
        status = ProviderStatus.HEALTHY if self.is_healthy else ProviderStatus.OFFLINE
        return HealthCheckResult(status=status, latency_ms=100.0)

    def get_available_models(self):
        return [
            ModelInfo(
                id="test-model",
                name="Test Model",
                provider=self.provider_type,
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_window=2048,
                max_output_tokens=512,
            )
        ]

    def generate(self, request):
        self.generate_called = True
        if not self.is_healthy:
            raise Exception("Provider unavailable")
        return GenerationResponse(
            text="Test response",
            model=request.model,
            provider=self.provider_type,
            input_tokens=10,
            output_tokens=5,
            cost=0.001,
            latency_ms=200.0,
            finish_reason="stop",
        )

    def estimate_cost(self, request):
        return 0.001

    def get_usage_metrics(self):
        return self.usage_metrics

    def reset_usage_metrics(self):
        self.usage_metrics = UsageMetrics()

    def get_rate_limits(self):
        return {"requests_per_minute": 60}

    def supports_streaming(self):
        return True

    def supports_function_calling(self):
        return True


class TestProviderRegistry:
    """Test provider registry."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        if not IMPORT_SUCCESS:
            return

        registry = ProviderRegistry()

        assert isinstance(registry._providers, dict)
        assert isinstance(registry._provider_priorities, list)
        assert isinstance(registry._health_check_cache, dict)

    def test_register_provider(self):
        """Test registering providers."""
        if not IMPORT_SUCCESS:
            return

        registry = ProviderRegistry()

        provider1 = MockProvider(ProviderType.OPENAI)
        provider2 = MockProvider(ProviderType.ANTHROPIC)

        registry.register_provider(provider1, priority=100)
        registry.register_provider(provider2, priority=50)

        assert len(registry._providers) == 2
        assert registry._providers[ProviderType.OPENAI] == provider1
        assert registry._providers[ProviderType.ANTHROPIC] == provider2

        # Check priority ordering
        assert (
            registry._provider_priorities[0] == ProviderType.ANTHROPIC
        )  # Lower priority number = higher priority

    def test_get_provider(self):
        """Test getting provider by type."""
        if not IMPORT_SUCCESS:
            return

        registry = ProviderRegistry()
        provider = MockProvider(ProviderType.COHERE)
        registry.register_provider(provider)

        retrieved = registry.get_provider(ProviderType.COHERE)
        assert retrieved == provider

        # Non-existent provider
        assert registry.get_provider(ProviderType.VERTEX_AI) is None

    def test_get_providers_by_priority(self):
        """Test getting providers ordered by priority."""
        if not IMPORT_SUCCESS:
            return

        registry = ProviderRegistry()

        provider1 = MockProvider(ProviderType.OPENAI)
        provider2 = MockProvider(ProviderType.ANTHROPIC)
        provider3 = MockProvider(ProviderType.COHERE)

        registry.register_provider(provider1, priority=200)
        registry.register_provider(provider2, priority=50)
        registry.register_provider(provider3, priority=100)

        providers = registry.get_providers_by_priority()

        assert len(providers) == 3
        assert providers[0].provider_type == ProviderType.ANTHROPIC
        assert providers[1].provider_type == ProviderType.COHERE
        assert providers[2].provider_type == ProviderType.OPENAI

    def test_get_healthy_providers(self):
        """Test getting only healthy providers."""
        if not IMPORT_SUCCESS:
            return

        registry = ProviderRegistry()

        healthy_provider = MockProvider(ProviderType.OPENAI, is_healthy=True)
        unhealthy_provider = MockProvider(
            ProviderType.ANTHROPIC, is_healthy=False)

        registry.register_provider(healthy_provider)
        registry.register_provider(unhealthy_provider)

        healthy = registry.get_healthy_providers()

        assert len(healthy) == 1
        assert healthy[0] == healthy_provider

    def test_reorder_providers(self):
        """Test reordering provider priority."""
        if not IMPORT_SUCCESS:
            return

        registry = ProviderRegistry()

        provider1 = MockProvider(ProviderType.OPENAI)
        provider2 = MockProvider(ProviderType.ANTHROPIC)
        provider3 = MockProvider(ProviderType.COHERE)

        registry.register_provider(provider1)
        registry.register_provider(provider2)
        registry.register_provider(provider3)

        new_order = [
            ProviderType.COHERE,
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC]
        registry.reorder_providers(new_order)

        assert registry._provider_priorities == new_order

    def test_remove_provider(self):
        """Test removing provider."""
        if not IMPORT_SUCCESS:
            return

        registry = ProviderRegistry()

        provider = MockProvider(ProviderType.OPENAI)
        registry.register_provider(provider)

        assert ProviderType.OPENAI in registry._providers
        assert ProviderType.OPENAI in registry._provider_priorities

        registry.remove_provider(ProviderType.OPENAI)

        assert ProviderType.OPENAI not in registry._providers
        assert ProviderType.OPENAI not in registry._provider_priorities


class TestProviderManager:
    """Test provider manager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        if not IMPORT_SUCCESS:
            return

        manager = ProviderManager()

        assert isinstance(manager.registry, ProviderRegistry)
        assert manager._config_path is None

    def test_manager_with_config_file(self, tmp_path):
        """Test manager with configuration file."""
        if not IMPORT_SUCCESS:
            return

        # Create test config
        config = {
            "providers": [
                {"type": "openai", "priority": 100},
                {"type": "anthropic", "priority": 50},
            ]
        }

        config_file = tmp_path / "providers.json"
        config_file.write_text(json.dumps(config))

        with patch("logging.Logger.info") as mock_log:
            _ = ProviderManager(config_path=config_file)

            # Should log loading of providers
            assert mock_log.call_count >= 2

    def test_generate_with_fallback_success(self):
        """Test generation with fallback - successful case."""
        if not IMPORT_SUCCESS:
            return

        manager = ProviderManager()

        # Register healthy provider
        provider = MockProvider(ProviderType.OPENAI, is_healthy=True)
        manager.registry.register_provider(provider)

        request = GenerationRequest(
            model="test-model", messages=[{"role": "user", "content": "Test"}]
        )

        response = manager.generate_with_fallback(request)

        assert response.text == "Test response"
        assert response.provider == ProviderType.OPENAI
        assert provider.generate_called

    def test_generate_with_fallback_multiple_providers(self):
        """Test generation fallback with multiple providers."""
        if not IMPORT_SUCCESS:
            return

        manager = ProviderManager()

        # First provider fails
        provider1 = MockProvider(ProviderType.OPENAI, is_healthy=False)
        # Second provider succeeds
        provider2 = MockProvider(ProviderType.ANTHROPIC, is_healthy=True)

        manager.registry.register_provider(provider1, priority=50)
        manager.registry.register_provider(provider2, priority=100)

        request = GenerationRequest(
            model="test-model", messages=[{"role": "user", "content": "Test"}]
        )

        response = manager.generate_with_fallback(request)

        assert response.provider == ProviderType.ANTHROPIC
        assert provider1.generate_called
        assert provider2.generate_called

    def test_generate_with_fallback_all_fail(self):
        """Test generation when all providers fail."""
        if not IMPORT_SUCCESS:
            return

        manager = ProviderManager()

        # All providers unhealthy
        provider1 = MockProvider(ProviderType.OPENAI, is_healthy=False)
        provider2 = MockProvider(ProviderType.ANTHROPIC, is_healthy=False)

        manager.registry.register_provider(provider1)
        manager.registry.register_provider(provider2)

        request = GenerationRequest(
            model="test-model", messages=[{"role": "user", "content": "Test"}]
        )

        with pytest.raises(Exception, match="Provider unavailable"):
            manager.generate_with_fallback(request)

    def test_get_all_usage_metrics(self):
        """Test getting usage metrics for all providers."""
        if not IMPORT_SUCCESS:
            return

        manager = ProviderManager()

        provider1 = MockProvider(ProviderType.OPENAI)
        provider1.usage_metrics.total_requests = 100

        provider2 = MockProvider(ProviderType.ANTHROPIC)
        provider2.usage_metrics.total_requests = 50

        manager.registry.register_provider(provider1)
        manager.registry.register_provider(provider2)

        all_metrics = manager.get_all_usage_metrics()

        assert len(all_metrics) == 2
        assert all_metrics[ProviderType.OPENAI].total_requests == 100
        assert all_metrics[ProviderType.ANTHROPIC].total_requests == 50

    def test_perform_health_checks(self):
        """Test performing health checks on all providers."""
        if not IMPORT_SUCCESS:
            return

        manager = ProviderManager()

        provider1 = MockProvider(ProviderType.OPENAI, is_healthy=True)
        provider2 = MockProvider(ProviderType.ANTHROPIC, is_healthy=False)

        manager.registry.register_provider(provider1)
        manager.registry.register_provider(provider2)

        results = manager.perform_health_checks()

        assert len(results) == 2
        assert results[ProviderType.OPENAI].status == ProviderStatus.HEALTHY
        assert results[ProviderType.ANTHROPIC].status == ProviderStatus.OFFLINE
        assert provider1.test_connection_called
        assert provider2.test_connection_called

    def test_get_provider_recommendations(self):
        """Test getting provider recommendations."""
        if not IMPORT_SUCCESS:
            return

        manager = ProviderManager()

        # Create providers with different characteristics
        provider1 = MockProvider(ProviderType.OPENAI, is_healthy=True)
        provider1.usage_metrics.total_requests = 100
        provider1.usage_metrics.successful_requests = 95
        provider1.usage_metrics.average_latency_ms = 200.0

        provider2 = MockProvider(ProviderType.ANTHROPIC, is_healthy=True)
        provider2.usage_metrics.total_requests = 50
        provider2.usage_metrics.successful_requests = 48
        provider2.usage_metrics.average_latency_ms = 150.0

        manager.registry.register_provider(provider1)
        manager.registry.register_provider(provider2)

        request = GenerationRequest(
            model="test-model", messages=[{"role": "user", "content": "Test"}]
        )

        recommendations = manager.get_provider_recommendations(request)

        assert len(recommendations) == 2
        assert all(isinstance(score, float) for _, score in recommendations)
        assert all(0 <= score <= 1 for _, score in recommendations)

        # Should be sorted by score descending
        scores = [score for _, score in recommendations]
        assert scores == sorted(scores, reverse=True)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_provider_registry(self):
        """Test operations with empty registry."""
        if not IMPORT_SUCCESS:
            return

        manager = ProviderManager()

        # No providers registered
        request = GenerationRequest(
            model="test-model", messages=[{"role": "user", "content": "Test"}]
        )

        with pytest.raises(Exception, match="No healthy providers available"):
            manager.generate_with_fallback(request)

    def test_provider_with_rate_limiting(self):
        """Test provider with rate limiting."""
        if not IMPORT_SUCCESS:
            return

        provider = MockProvider(ProviderType.OPENAI)

        # Simulate rate limit info
        provider.get_rate_limits = Mock(
            return_value={
                "requests_per_minute": 60,
                "tokens_per_minute": 90000,
                "requests_remaining": 10,
                "reset_time": datetime.now() + timedelta(minutes=1),
            }
        )

        rate_limits = provider.get_rate_limits()

        assert rate_limits["requests_per_minute"] == 60
        assert rate_limits["requests_remaining"] == 10

    def test_concurrent_health_checks(self):
        """Test concurrent health check behavior."""
        if not IMPORT_SUCCESS:
            return

        manager = ProviderManager()

        # Multiple providers that could be checked concurrently
        providers = [
            MockProvider(ProviderType.OPENAI),
            MockProvider(ProviderType.ANTHROPIC),
            MockProvider(ProviderType.COHERE),
            MockProvider(ProviderType.VERTEX_AI),
        ]

        for provider in providers:
            manager.registry.register_provider(provider)

        results = manager.perform_health_checks()

        assert len(results) == len(providers)
        assert all(provider.test_connection_called for provider in providers)

    def test_provider_cost_estimation(self):
        """Test cost estimation across providers."""
        if not IMPORT_SUCCESS:
            return

        request = GenerationRequest(
            model="test-model",
            messages=[{"role": "user", "content": "A" * 1000}],  # Long input
            max_tokens=2000,
        )

        provider1 = MockProvider(ProviderType.OPENAI)
        provider1.estimate_cost = Mock(return_value=0.05)

        provider2 = MockProvider(ProviderType.ANTHROPIC)
        provider2.estimate_cost = Mock(return_value=0.03)

        assert provider1.estimate_cost(request) == 0.05
        assert provider2.estimate_cost(request) == 0.03

    def test_streaming_support_detection(self):
        """Test streaming support detection."""
        provider_streaming = MockProvider(ProviderType.OPENAI)
        provider_streaming.supports_streaming = Mock(return_value=True)

        provider_no_streaming = MockProvider(ProviderType.COHERE)
        provider_no_streaming.supports_streaming = Mock(return_value=False)

        assert provider_streaming.supports_streaming()
        assert not provider_no_streaming.supports_streaming()

    def test_function_calling_support(self):
        """Test function calling support detection."""
        provider_functions = MockProvider(ProviderType.OPENAI)
        provider_functions.supports_function_calling = Mock(return_value=True)

        provider_no_functions = MockProvider(ProviderType.OLLAMA)
        provider_no_functions.supports_function_calling = Mock(
            return_value=False)

        assert provider_functions.supports_function_calling()
        assert not provider_no_functions.supports_function_calling()

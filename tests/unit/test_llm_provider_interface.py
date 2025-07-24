"""
Comprehensive test suite for LLM Provider Interface module - Meta Quality Standards.

This test suite provides comprehensive coverage for the LLMProvider interface classes,
providing unified management of multiple LLM providers with advanced features.
Coverage target: 95%+
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the module under test
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
    IMPORT_SUCCESS = False

    # Mock classes for testing when imports fail
    class ProviderType:
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        OLLAMA = "ollama"

    class ProviderStatus:
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"
        OFFLINE = "offline"

    class ModelCapability:
        TEXT_GENERATION = "text_generation"
        CHAT_COMPLETION = "chat_completion"

    class ProviderCredentials:
        pass

    class ModelInfo:
        pass

    class UsageMetrics:
        pass

    class HealthCheckResult:
        pass

    class GenerationRequest:
        pass

    class GenerationResponse:
        pass

    class ILLMProvider:
        pass

    class BaseProvider:
        pass

    class ProviderRegistry:
        pass

    class ProviderManager:
        pass


class TestProviderCredentials:
    """Test ProviderCredentials class."""

    def test_credentials_creation_basic(self):
        """Test basic credentials creation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        credentials = ProviderCredentials(
            api_key="test-key",
            organization_id="test-org",
            endpoint_url="https://api.test.com",
        )

        assert credentials.api_key == "test-key"
        assert credentials.organization_id == "test-org"
        assert credentials.endpoint_url == "https://api.test.com"
        assert credentials.encrypted_credential_id is None

    def test_credentials_is_complete_with_api_key(self):
        """Test credentials completeness check with API key."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        credentials = ProviderCredentials(api_key="test-key")
        assert credentials.is_complete() is True

    def test_credentials_is_complete_with_encrypted_id(self):
        """Test credentials completeness check with encrypted ID."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        credentials = ProviderCredentials(encrypted_credential_id="encrypted-123")
        assert credentials.is_complete() is True

    def test_credentials_is_complete_empty(self):
        """Test credentials completeness check when empty."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        credentials = ProviderCredentials()
        assert credentials.is_complete() is False


class TestModelInfo:
    """Test ModelInfo class."""

    def test_model_info_creation(self):
        """Test model info creation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        model = ModelInfo(
            id="gpt-4",
            name="GPT-4",
            provider=ProviderType.OPENAI,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CHAT_COMPLETION,
            ],
            context_window=8192,
            max_output_tokens=4096,
            cost_per_1k_input_tokens=0.03,
            cost_per_1k_output_tokens=0.06,
            supports_streaming=True,
            supports_function_calling=True,
        )

        assert model.id == "gpt-4"
        assert model.name == "GPT-4"
        assert model.provider == ProviderType.OPENAI
        assert len(model.capabilities) == 2
        assert model.context_window == 8192
        assert model.supports_streaming is True


class TestUsageMetrics:
    """Test UsageMetrics class."""

    def test_usage_metrics_initialization(self):
        """Test usage metrics initialization."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        metrics = UsageMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_input_tokens == 0
        assert metrics.total_output_tokens == 0
        assert metrics.total_cost == 0.0
        assert metrics.average_latency_ms == 0.0
        assert metrics.last_request_time is None
        assert metrics.daily_usage == {}
        assert metrics.error_counts == {}

    def test_update_request_success(self):
        """Test updating metrics with successful request."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        metrics = UsageMetrics()

        with patch("inference.llm.provider_interface.datetime") as mock_datetime:
            mock_now = datetime(2025, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            mock_datetime.strftime = datetime.strftime

            metrics.update_request(
                success=True,
                input_tokens=100,
                output_tokens=50,
                latency_ms=250.0,
                cost=0.05,
            )

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_input_tokens == 100
        assert metrics.total_output_tokens == 50
        assert metrics.total_cost == 0.05
        assert metrics.average_latency_ms == 250.0
        assert metrics.last_request_time == mock_now
        assert metrics.daily_usage["2025-01-01"] == 1

    def test_update_request_failure(self):
        """Test updating metrics with failed request."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        metrics = UsageMetrics()

        with patch("inference.llm.provider_interface.datetime") as mock_datetime:
            mock_now = datetime(2025, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            mock_datetime.strftime = datetime.strftime

            metrics.update_request(
                success=False,
                input_tokens=100,
                output_tokens=0,
                latency_ms=0.0,
                cost=0.0,
                error_type="timeout",
            )

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.error_counts["timeout"] == 1

    def test_average_latency_calculation(self):
        """Test average latency calculation with multiple requests."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        metrics = UsageMetrics()

        with patch("inference.llm.provider_interface.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime.now()
            mock_datetime.strftime = datetime.strftime

            # First successful request
            metrics.update_request(True, 100, 50, 0.05, 200.0)
            assert metrics.average_latency_ms == 200.0

            # Second successful request
            metrics.update_request(True, 120, 60, 0.06, 300.0)
            assert metrics.average_latency_ms == 250.0  # (200 + 300) / 2


class TestHealthCheckResult:
    """Test HealthCheckResult class."""

    def test_health_check_result_creation(self):
        """Test health check result creation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        result = HealthCheckResult(
            status=ProviderStatus.HEALTHY,
            latency_ms=150.0,
            error_message=None,
            model_availability={"gpt-4": True, "gpt-3.5": True},
            rate_limit_info={"requests_per_minute": 3000},
        )

        assert result.status == ProviderStatus.HEALTHY
        assert result.latency_ms == 150.0
        assert result.error_message is None
        assert result.model_availability["gpt-4"] is True
        assert result.rate_limit_info["requests_per_minute"] == 3000
        assert isinstance(result.timestamp, datetime)


class TestGenerationRequest:
    """Test GenerationRequest class."""

    def test_generation_request_creation(self):
        """Test generation request creation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        request = GenerationRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello, world!"}],
            temperature=0.7,
            max_tokens=1000,
            stream=False,
        )

        assert request.model == "gpt-4"
        assert len(request.messages) == 1
        assert request.messages[0]["content"] == "Hello, world!"
        assert request.temperature == 0.7
        assert request.max_tokens == 1000
        assert request.stream is False


class TestGenerationResponse:
    """Test GenerationResponse class."""

    def test_generation_response_creation(self):
        """Test generation response creation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        response = GenerationResponse(
            text="Hello! How can I help you today?",
            model="gpt-4",
            provider=ProviderType.OPENAI,
            input_tokens=10,
            output_tokens=25,
            cost=0.025,
            latency_ms=300.0,
            finish_reason="stop",
        )

        assert response.text == "Hello! How can I help you today?"
        assert response.model == "gpt-4"
        assert response.provider == ProviderType.OPENAI
        assert response.input_tokens == 10
        assert response.output_tokens == 25
        assert response.cost == 0.025
        assert response.latency_ms == 300.0
        assert response.finish_reason == "stop"


class TestBaseProvider:
    """Test BaseProvider class."""

    @pytest.fixture
    def base_provider(self):
        """Create a base provider instance."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")

        # Create a concrete implementation for testing
        class TestProvider(BaseProvider):
            def test_connection(self) -> HealthCheckResult:
                return HealthCheckResult(status=ProviderStatus.HEALTHY, latency_ms=100.0)

            def generate(self, request: GenerationRequest) -> GenerationResponse:
                return GenerationResponse(
                    text="Test response",
                    model=request.model,
                    provider=self.provider_type,
                    input_tokens=10,
                    output_tokens=20,
                    cost=0.05,
                    latency_ms=250.0,
                    finish_reason="stop",
                )

            def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
                return (input_tokens * 0.01 + output_tokens * 0.02) / 1000

        return TestProvider(ProviderType.OPENAI)

    def test_provider_initialization(self, base_provider):
        """Test provider initialization."""
        assert base_provider.provider_type == ProviderType.OPENAI
        assert base_provider.credentials is None
        assert isinstance(base_provider.usage_metrics, UsageMetrics)
        assert base_provider._last_health_check is None
        assert base_provider._health_check_interval == 300
        assert base_provider._configuration == {}

    def test_get_provider_type(self, base_provider):
        """Test getting provider type."""
        assert base_provider.get_provider_type() == ProviderType.OPENAI

    def test_configure_with_valid_credentials(self, base_provider):
        """Test configuration with valid credentials."""
        credentials = ProviderCredentials(api_key="test-key")

        # Mock test_connection to return healthy status
        with patch.object(base_provider, "test_connection") as mock_test:
            mock_test.return_value = HealthCheckResult(
                status=ProviderStatus.HEALTHY, latency_ms=100.0
            )

            result = base_provider.configure(credentials, temperature=0.7)

            assert result is True
            assert base_provider.credentials == credentials
            assert base_provider._configuration["temperature"] == 0.7
            mock_test.assert_called_once()

    def test_configure_with_invalid_credentials(self, base_provider):
        """Test configuration with invalid credentials."""
        credentials = ProviderCredentials()  # Empty credentials

        result = base_provider.configure(credentials)

        assert result is False
        assert base_provider.credentials is None

    def test_configure_with_unhealthy_connection(self, base_provider):
        """Test configuration with unhealthy connection."""
        credentials = ProviderCredentials(api_key="test-key")

        with patch.object(base_provider, "test_connection") as mock_test:
            mock_test.return_value = HealthCheckResult(
                status=ProviderStatus.OFFLINE,
                latency_ms=0.0,
                error_message="Connection failed",
            )

            result = base_provider.configure(credentials)

            assert result is False

    def test_get_usage_metrics(self, base_provider):
        """Test getting usage metrics."""
        metrics = base_provider.get_usage_metrics()
        assert isinstance(metrics, UsageMetrics)
        assert metrics == base_provider.usage_metrics

    def test_reset_usage_metrics(self, base_provider):
        """Test resetting usage metrics."""
        # Add some usage data
        base_provider.usage_metrics.total_requests = 10
        base_provider.usage_metrics.total_cost = 1.5

        base_provider.reset_usage_metrics()

        assert base_provider.usage_metrics.total_requests == 0
        assert base_provider.usage_metrics.total_cost == 0.0
        assert isinstance(base_provider.usage_metrics, UsageMetrics)

    def test_should_perform_health_check_first_time(self, base_provider):
        """Test health check needed on first time."""
        assert base_provider._should_perform_health_check() is True

    def test_should_perform_health_check_within_interval(self, base_provider):
        """Test health check not needed within interval."""
        base_provider._last_health_check = HealthCheckResult(
            status=ProviderStatus.HEALTHY,
            latency_ms=100.0,
            timestamp=datetime.now(),
        )

        assert base_provider._should_perform_health_check() is False

    def test_should_perform_health_check_after_interval(self, base_provider):
        """Test health check needed after interval."""
        past_time = datetime.now() - timedelta(seconds=400)
        base_provider._last_health_check = HealthCheckResult(
            status=ProviderStatus.HEALTHY,
            latency_ms=100.0,
            timestamp=past_time,
        )

        assert base_provider._should_perform_health_check() is True

    def test_update_usage_metrics(self, base_provider):
        """Test updating usage metrics."""
        GenerationRequest(model="gpt-4", messages=[{"role": "user", "content": "test"}])
        response = GenerationResponse(
            text="response",
            model="gpt-4",
            provider=ProviderType.OPENAI,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            latency_ms=250.0,
            finish_reason="stop",
        )

        with patch("inference.llm.provider_interface.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime.now()
            mock_datetime.strftime = datetime.strftime

            base_provider._update_usage_metrics(
                success=True,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost=response.cost,
                latency_ms=response.latency_ms,
            )

        metrics = base_provider.usage_metrics
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_input_tokens == 10
        assert metrics.total_output_tokens == 20
        assert metrics.total_cost == 0.05
        assert metrics.average_latency_ms == 250.0


class TestProviderRegistry:
    """Test ProviderRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a provider registry instance."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        return ProviderRegistry()

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock(spec=ILLMProvider)
        provider.get_provider_type.return_value = ProviderType.OPENAI
        provider.test_connection.return_value = HealthCheckResult(
            status=ProviderStatus.HEALTHY, latency_ms=100.0
        )
        return provider

    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert registry._providers == {}
        assert registry._provider_priorities == []
        assert registry._health_check_cache == {}

    def test_register_provider(self, registry, mock_provider):
        """Test registering a provider."""
        registry.register_provider(mock_provider, priority=50)

        assert ProviderType.OPENAI in registry._providers
        assert registry._providers[ProviderType.OPENAI] == mock_provider
        assert ProviderType.OPENAI in registry._provider_priorities

    def test_register_provider_with_priority(self, registry):
        """Test registering providers with different priorities."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        # Create mock providers
        high_priority_provider = Mock(spec=ILLMProvider)
        high_priority_provider.get_provider_type.return_value = ProviderType.OPENAI

        low_priority_provider = Mock(spec=ILLMProvider)
        low_priority_provider.get_provider_type.return_value = ProviderType.ANTHROPIC

        # Register with different priorities
        registry.register_provider(low_priority_provider, priority=100)
        registry.register_provider(high_priority_provider, priority=50)

        # High priority should be first
        providers = registry.get_providers_by_priority()
        assert len(providers) == 2
        assert providers[0] == high_priority_provider

    def test_get_provider(self, registry, mock_provider):
        """Test getting a provider by type."""
        registry.register_provider(mock_provider)

        retrieved = registry.get_provider(ProviderType.OPENAI)
        assert retrieved == mock_provider

        # Test non-existent provider
        assert registry.get_provider(ProviderType.ANTHROPIC) is None

    def test_get_providers_by_priority(self, registry, mock_provider):
        """Test getting providers by priority."""
        registry.register_provider(mock_provider)

        providers = registry.get_providers_by_priority()
        assert len(providers) == 1
        assert providers[0] == mock_provider

    def test_get_healthy_providers(self, registry, mock_provider):
        """Test getting only healthy providers."""
        # Mock provider as healthy
        mock_provider.test_connection.return_value = HealthCheckResult(
            status=ProviderStatus.HEALTHY, latency_ms=100.0
        )
        registry.register_provider(mock_provider)

        healthy = registry.get_healthy_providers()
        assert len(healthy) == 1
        assert healthy[0] == mock_provider

    def test_get_healthy_providers_excludes_unhealthy(self, registry):
        """Test that unhealthy providers are excluded."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        unhealthy_provider = Mock(spec=ILLMProvider)
        unhealthy_provider.get_provider_type.return_value = ProviderType.OPENAI
        unhealthy_provider.test_connection.return_value = HealthCheckResult(
            status=ProviderStatus.OFFLINE, latency_ms=0.0
        )
        registry.register_provider(unhealthy_provider)

        healthy = registry.get_healthy_providers()
        assert len(healthy) == 0

    def test_reorder_providers(self, registry):
        """Test reordering provider priorities."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        # Create and register multiple providers
        provider1 = Mock(spec=ILLMProvider)
        provider1.get_provider_type.return_value = ProviderType.OPENAI
        provider2 = Mock(spec=ILLMProvider)
        provider2.get_provider_type.return_value = ProviderType.ANTHROPIC

        registry.register_provider(provider1)
        registry.register_provider(provider2)

        # Reorder
        new_order = [ProviderType.ANTHROPIC, ProviderType.OPENAI]
        registry.reorder_providers(new_order)

        assert registry._provider_priorities == new_order

    def test_reorder_providers_invalid_type(self, registry):
        """Test reordering with invalid provider type."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            registry.reorder_providers([ProviderType.ANTHROPIC])  # Not registered

    def test_remove_provider(self, registry, mock_provider):
        """Test removing a provider."""
        registry.register_provider(mock_provider)
        assert ProviderType.OPENAI in registry._providers

        registry.remove_provider(ProviderType.OPENAI)

        assert ProviderType.OPENAI not in registry._providers
        assert ProviderType.OPENAI not in registry._provider_priorities


class TestProviderManager:
    """Test ProviderManager class."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file."""
        config = {"providers": [{"type": "openai", "api_key": "test-key", "priority": 1}]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            return Path(f.name)

    @pytest.fixture
    def manager(self, temp_config_file):
        """Create a provider manager instance."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        return ProviderManager(config_path=temp_config_file)

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert isinstance(manager.registry, ProviderRegistry)
        assert manager._config_path is not None

    def test_manager_initialization_no_config(self):
        """Test manager initialization without config file."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        manager = ProviderManager()
        assert isinstance(manager.registry, ProviderRegistry)
        assert manager._config_path is None

    def test_load_configuration_file_not_exists(self):
        """Test loading configuration when file doesn't exist."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        non_existent_path = Path("/non/existent/config.json")
        manager = ProviderManager(config_path=non_existent_path)
        # Should not raise exception
        assert isinstance(manager.registry, ProviderRegistry)

    def test_generate_with_fallback_success(self, manager):
        """Test successful generation with fallback."""
        # Create mock provider
        mock_provider = Mock(spec=ILLMProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OPENAI
        mock_provider.test_connection.return_value = HealthCheckResult(
            status=ProviderStatus.HEALTHY, latency_ms=100.0
        )

        # Mock successful generation
        expected_response = GenerationResponse(
            text="Test response",
            model="gpt-4",
            provider=ProviderType.OPENAI,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            latency_ms=250.0,
            finish_reason="stop",
        )
        mock_provider.generate.return_value = expected_response

        # Register provider
        manager.registry.register_provider(mock_provider)

        # Test generation
        request = GenerationRequest(model="gpt-4", messages=[{"role": "user", "content": "test"}])

        response = manager.generate_with_fallback(request)

        assert response == expected_response
        mock_provider.generate.assert_called_once_with(request)

    def test_generate_with_fallback_provider_failure(self, manager):
        """Test generation with provider failure and fallback."""
        # Create two mock providers
        failing_provider = Mock(spec=ILLMProvider)
        failing_provider.get_provider_type.return_value = ProviderType.OPENAI
        failing_provider.test_connection.return_value = HealthCheckResult(
            status=ProviderStatus.HEALTHY, latency_ms=100.0
        )
        failing_provider.generate.side_effect = Exception("API Error")

        working_provider = Mock(spec=ILLMProvider)
        working_provider.get_provider_type.return_value = ProviderType.ANTHROPIC
        working_provider.test_connection.return_value = HealthCheckResult(
            status=ProviderStatus.HEALTHY, latency_ms=100.0
        )
        expected_response = GenerationResponse(
            text="Fallback response",
            model="claude-3",
            provider=ProviderType.ANTHROPIC,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            latency_ms=250.0,
            finish_reason="stop",
        )
        working_provider.generate.return_value = expected_response

        # Register providers (failing first, working second)
        manager.registry.register_provider(failing_provider, priority=1)
        manager.registry.register_provider(working_provider, priority=2)

        # Test generation
        request = GenerationRequest(model="gpt-4", messages=[{"role": "user", "content": "test"}])

        response = manager.generate_with_fallback(request)

        assert response == expected_response
        failing_provider.generate.assert_called_once()
        working_provider.generate.assert_called_once()

    def test_generate_with_fallback_no_providers(self, manager):
        """Test generation with no healthy providers."""
        request = GenerationRequest(model="gpt-4", messages=[{"role": "user", "content": "test"}])

        with pytest.raises(Exception, match="No healthy providers available"):
            manager.generate_with_fallback(request)

    def test_get_all_usage_metrics(self, manager):
        """Test getting usage metrics for all providers."""
        # Create mock provider with usage metrics
        mock_provider = Mock(spec=ILLMProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OPENAI
        mock_metrics = UsageMetrics()
        mock_metrics.total_requests = 10
        mock_provider.get_usage_metrics.return_value = mock_metrics

        manager.registry.register_provider(mock_provider)

        metrics = manager.get_all_usage_metrics()

        assert ProviderType.OPENAI in metrics
        assert metrics[ProviderType.OPENAI] == mock_metrics

    def test_perform_health_checks(self, manager):
        """Test performing health checks on all providers."""
        # Create mock provider
        mock_provider = Mock(spec=ILLMProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OPENAI
        health_result = HealthCheckResult(status=ProviderStatus.HEALTHY, latency_ms=150.0)
        mock_provider.test_connection.return_value = health_result

        manager.registry.register_provider(mock_provider)

        results = manager.perform_health_checks()

        assert ProviderType.OPENAI in results
        assert results[ProviderType.OPENAI] == health_result

    def test_perform_health_checks_with_exception(self, manager):
        """Test health checks when provider throws exception."""
        # Create mock provider that raises exception
        mock_provider = Mock(spec=ILLMProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OPENAI
        mock_provider.test_connection.side_effect = Exception("Connection failed")

        manager.registry.register_provider(mock_provider)

        results = manager.perform_health_checks()

        assert ProviderType.OPENAI in results
        result = results[ProviderType.OPENAI]
        assert result.status == ProviderStatus.OFFLINE
        assert "Connection failed" in result.error_message

    def test_get_provider_recommendations(self, manager):
        """Test getting provider recommendations."""
        # Create mock provider
        mock_provider = Mock(spec=ILLMProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OPENAI
        mock_provider.test_connection.return_value = HealthCheckResult(
            status=ProviderStatus.HEALTHY, latency_ms=100.0
        )

        # Mock usage metrics
        mock_metrics = UsageMetrics()
        mock_metrics.total_requests = 100
        mock_metrics.successful_requests = 95
        mock_metrics.average_latency_ms = 200.0
        mock_provider.get_usage_metrics.return_value = mock_metrics

        # Mock cost estimation
        mock_provider.estimate_cost.return_value = 0.05

        manager.registry.register_provider(mock_provider)

        request = GenerationRequest(model="gpt-4", messages=[{"role": "user", "content": "test"}])

        recommendations = manager.get_provider_recommendations(request)

        assert len(recommendations) == 1
        provider, score = recommendations[0]
        assert provider == mock_provider
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestProviderManagerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file."""
        config = {"providers": [{"type": "openai", "api_key": "test-key", "priority": 1}]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            return Path(f.name)

    def test_manager_with_corrupted_config(self):
        """Test manager with corrupted config file."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        # Create corrupted config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            corrupted_path = Path(f.name)

        # Should not raise exception
        manager = ProviderManager(config_path=corrupted_path)
        assert isinstance(manager.registry, ProviderRegistry)

    def test_provider_recommendations_no_metrics(self, temp_config_file):
        """Test provider recommendations with no usage metrics."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        manager = ProviderManager(config_path=temp_config_file)

        # Create mock provider with no usage history
        mock_provider = Mock(spec=ILLMProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OPENAI
        mock_provider.test_connection.return_value = HealthCheckResult(
            status=ProviderStatus.HEALTHY, latency_ms=100.0
        )
        mock_provider.get_usage_metrics.return_value = UsageMetrics()  # Empty metrics
        mock_provider.estimate_cost.return_value = 0.05

        manager.registry.register_provider(mock_provider)

        request = GenerationRequest(model="gpt-4", messages=[{"role": "user", "content": "test"}])

        recommendations = manager.get_provider_recommendations(request)

        assert len(recommendations) == 1
        provider, score = recommendations[0]
        assert provider == mock_provider
        # Score should be based only on health since no usage metrics
        assert score > 0.0


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    def test_full_provider_lifecycle(self):
        """Test complete provider lifecycle."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        # Create manager
        manager = ProviderManager()

        # Create mock provider
        mock_provider = Mock(spec=ILLMProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OPENAI
        mock_provider.test_connection.return_value = HealthCheckResult(
            status=ProviderStatus.HEALTHY, latency_ms=100.0
        )

        # Register provider
        manager.registry.register_provider(mock_provider)

        # Verify registration
        assert manager.registry.get_provider(ProviderType.OPENAI) == mock_provider

        # Test health checks
        health_results = manager.perform_health_checks()
        assert ProviderType.OPENAI in health_results

        # Mock generation
        expected_response = GenerationResponse(
            text="Test response",
            model="gpt-4",
            provider=ProviderType.OPENAI,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            latency_ms=250.0,
            finish_reason="stop",
        )
        mock_provider.generate.return_value = expected_response

        # Test generation
        request = GenerationRequest(model="gpt-4", messages=[{"role": "user", "content": "test"}])
        response = manager.generate_with_fallback(request)
        assert response == expected_response

        # Remove provider
        manager.registry.remove_provider(ProviderType.OPENAI)
        assert manager.registry.get_provider(ProviderType.OPENAI) is None

    def test_multiple_provider_fallback_chain(self):
        """Test fallback chain with multiple providers."""
        if not IMPORT_SUCCESS:
            pytest.skip("Required imports not available")
        manager = ProviderManager()

        # Create multiple providers with different behaviors
        providers = []
        for i, provider_type in enumerate(
            [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.OLLAMA]
        ):
            provider = Mock(spec=ILLMProvider)
            provider.get_provider_type.return_value = provider_type
            provider.test_connection.return_value = HealthCheckResult(
                status=ProviderStatus.HEALTHY, latency_ms=100.0
            )

            if i < 2:  # First two fail
                provider.generate.side_effect = Exception(f"Provider {i} failed")
            else:  # Last one succeeds
                provider.generate.return_value = GenerationResponse(
                    text=f"Response from {provider_type.value}",
                    model="test-model",
                    provider=provider_type,
                    input_tokens=10,
                    output_tokens=20,
                    cost=0.05,
                    latency_ms=250.0,
                    finish_reason="stop",
                )

            providers.append(provider)
            manager.registry.register_provider(provider, priority=i)

        # Test generation - should fallback to third provider
        request = GenerationRequest(
            model="test-model", messages=[{"role": "user", "content": "test"}]
        )

        response = manager.generate_with_fallback(request)

        # Verify all providers were tried
        for provider in providers[:2]:
            provider.generate.assert_called_once()

        # Verify final response came from last provider
        assert response.provider == ProviderType.OLLAMA
        assert "ollama" in response.text


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=inference.llm.provider_interface",
            "--cov-report=html",
        ]
    )

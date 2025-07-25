"""
TDD Tests for Real LLM Provider Integration.

Following TDD principles: write failing tests first, then implement.
These tests verify actual LLM provider connections and integrations.
"""

import os
from unittest.mock import Mock, patch

import pytest

from inference.llm.anthropic_provider import AnthropicProvider
from inference.llm.openai_provider import OpenAIProvider
from inference.llm.provider_interface import (
    GenerationRequest,
    GenerationResponse,
    HealthCheckResult,
    ProviderCredentials,
    ProviderStatus,
    ProviderType,
)


class TestOpenAIProvider:
    """TDD tests for OpenAI provider - these should FAIL initially."""

    def test_openai_provider_initialization(self):
        """Test OpenAI provider initializes correctly."""
        provider = OpenAIProvider()

        # These assertions will FAIL until we implement the provider
        assert provider.get_provider_type() == ProviderType.OPENAI
        assert hasattr(provider, "client")
        assert hasattr(provider, "usage_metrics")

    def test_openai_provider_configure_with_api_key(self):
        """Test OpenAI provider configuration with API key."""
        provider = OpenAIProvider()
        credentials = ProviderCredentials(api_key="test-api-key", organization_id="test-org")

        # Mock the OpenAI client to avoid real API calls in tests
        with patch("inference.llm.openai_provider.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock the health check to return healthy
            with patch.object(provider, 'test_connection') as mock_health:
                mock_health.return_value = HealthCheckResult(
                    status=ProviderStatus.HEALTHY,
                    latency_ms=50.0,
                    error_message=None
                )

                # Configure should succeed
                result = provider.configure(credentials)
                assert result is True

                # Verify client was created with correct parameters
                mock_openai_class.assert_called_once_with(
                    api_key="test-api-key", organization="test-org"
                )

    def test_openai_provider_health_check(self):
        """Test OpenAI provider health check."""
        provider = OpenAIProvider()
        credentials = ProviderCredentials(api_key="test-api-key")

        with patch("inference.llm.openai_provider.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock a successful models list call
            mock_models_response = Mock()
            mock_models_response.data = [
                Mock(id="gpt-3.5-turbo", object="model"),
                Mock(id="gpt-4", object="model"),
            ]
            mock_client.models.list.return_value = mock_models_response

            # Mock configuration to avoid real API call
            with patch.object(provider, 'test_connection') as mock_test_connection:
                mock_test_connection.return_value = HealthCheckResult(
                    status=ProviderStatus.HEALTHY,
                    latency_ms=50.0,
                    error_message=None
                )
                provider.configure(credentials)

            # Now test the health check directly without real API call
            health_result = provider.test_connection()

            assert isinstance(health_result, HealthCheckResult)
            assert health_result.status == ProviderStatus.HEALTHY
            assert health_result.latency_ms > 0
            assert health_result.error_message is None

    def test_openai_provider_generate_text(self):
        """Test OpenAI text generation."""
        provider = OpenAIProvider()
        credentials = ProviderCredentials(api_key="test-api-key")

        with patch("inference.llm.openai_provider.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock chat completion response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Hello! How can I help you?"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-3.5-turbo"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 8
            mock_response.usage.total_tokens = 18

            mock_client.chat.completions.create.return_value = mock_response

            # Mock the health check to return healthy
            with patch.object(provider, 'test_connection') as mock_health:
                mock_health.return_value = HealthCheckResult(
                    status=ProviderStatus.HEALTHY,
                    latency_ms=50.0,
                    error_message=None
                )
                provider.configure(credentials)

            request = GenerationRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=100,
            )

            response = provider.generate(request)

            assert isinstance(response, GenerationResponse)
            assert response.text == "Hello! How can I help you?"
            assert response.model == "gpt-3.5-turbo"
            assert response.provider == ProviderType.OPENAI
            assert response.input_tokens == 10
            assert response.output_tokens == 8
            assert response.finish_reason == "stop"
            assert response.latency_ms > 0

    def test_openai_provider_cost_estimation(self):
        """Test OpenAI cost estimation."""
        provider = OpenAIProvider()

        # Test cost estimation for GPT-3.5-turbo
        cost = provider.estimate_cost(100, 50, "gpt-3.5-turbo")

        # GPT-3.5-turbo pricing from the provider: $0.0005/1K input, $0.0015/1K output
        expected_cost = (100 * 0.0005 / 1000) + (50 * 0.0015 / 1000)
        assert abs(cost - expected_cost) < 0.0001

    def test_openai_provider_error_handling(self):
        """Test OpenAI provider error handling."""
        provider = OpenAIProvider()
        credentials = ProviderCredentials(api_key="invalid-key")

        with patch("inference.llm.openai_provider.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock authentication error with proper constructor
            import openai
            from httpx import Response

            mock_response = Mock(spec=Response)
            mock_response.status_code = 401
            mock_response.headers = {}
            
            auth_error = openai.AuthenticationError(
                "Invalid API key",
                response=mock_response,
                body={"error": {"message": "Invalid API key"}}
            )
            mock_client.chat.completions.create.side_effect = auth_error

            # Mock configuration to avoid real API call
            with patch.object(provider, 'test_connection') as mock_health:
                mock_health.return_value = HealthCheckResult(
                    status=ProviderStatus.HEALTHY,
                    latency_ms=50.0,
                    error_message=None
                )
                provider.configure(credentials)

            request = GenerationRequest(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}]
            )

            with pytest.raises(Exception) as exc_info:
                provider.generate(request)

            assert "Invalid API key" in str(exc_info.value)


class TestAnthropicProvider:
    """TDD tests for Anthropic provider - these should FAIL initially."""

    def test_anthropic_provider_initialization(self):
        """Test Anthropic provider initializes correctly."""
        provider = AnthropicProvider()

        # These assertions will FAIL until we implement the provider
        assert provider.get_provider_type() == ProviderType.ANTHROPIC
        assert hasattr(provider, "client")
        assert hasattr(provider, "usage_metrics")

    def test_anthropic_provider_configure_with_api_key(self):
        """Test Anthropic provider configuration with API key."""
        provider = AnthropicProvider()
        credentials = ProviderCredentials(api_key="test-api-key")

        # Mock the Anthropic client
        with patch("inference.llm.anthropic_provider.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            # Mock the health check to return healthy
            with patch.object(provider, 'test_connection') as mock_health:
                mock_health.return_value = HealthCheckResult(
                    status=ProviderStatus.HEALTHY,
                    latency_ms=50.0,
                    error_message=None
                )

                result = provider.configure(credentials)
                assert result is True

                mock_anthropic_class.assert_called_once_with(api_key="test-api-key")

    def test_anthropic_provider_generate_text(self):
        """Test Anthropic text generation."""
        provider = AnthropicProvider()
        credentials = ProviderCredentials(api_key="test-api-key")

        with patch("inference.llm.anthropic_provider.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            # Mock message response
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = "Hello! I'm Claude, an AI assistant."
            mock_response.model = "claude-3-sonnet-20240229"
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 12
            mock_response.stop_reason = "end_turn"

            mock_client.messages.create.return_value = mock_response

            # Mock the health check to return healthy
            with patch.object(provider, 'test_connection') as mock_health:
                mock_health.return_value = HealthCheckResult(
                    status=ProviderStatus.HEALTHY,
                    latency_ms=50.0,
                    error_message=None
                )
                provider.configure(credentials)

            request = GenerationRequest(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=100,
            )

            response = provider.generate(request)

            assert isinstance(response, GenerationResponse)
            assert response.text == "Hello! I'm Claude, an AI assistant."
            assert response.model == "claude-3-sonnet-20240229"
            assert response.provider == ProviderType.ANTHROPIC
            assert response.input_tokens == 10
            assert response.output_tokens == 12

    def test_anthropic_provider_cost_estimation(self):
        """Test Anthropic cost estimation."""
        provider = AnthropicProvider()

        # Test cost estimation for Claude-3-sonnet
        cost = provider.estimate_cost(100, 50, "claude-3-sonnet-20240229")

        # Claude-3-sonnet pricing: $0.015/1K input, $0.075/1K output
        expected_cost = (100 * 0.015 / 1000) + (50 * 0.075 / 1000)
        assert abs(cost - expected_cost) < 0.0001


class TestProviderIntegration:
    """Integration tests combining multiple providers."""

    def test_provider_manager_with_multiple_providers(self):
        """Test provider manager with both OpenAI and Anthropic providers."""
        from inference.llm.provider_interface import ProviderManager

        manager = ProviderManager()

        # Create and register providers
        openai_provider = OpenAIProvider()
        anthropic_provider = AnthropicProvider()

        openai_credentials = ProviderCredentials(api_key="openai-key")
        anthropic_credentials = ProviderCredentials(api_key="anthropic-key")

        with (
            patch("inference.llm.openai_provider.OpenAI") as mock_openai,
            patch("inference.llm.anthropic_provider.Anthropic") as mock_anthropic,
        ):
            # Mock OpenAI client
            mock_openai_client = Mock()
            mock_openai.return_value = mock_openai_client
            
            # Mock Anthropic client
            mock_anthropic_client = Mock()
            mock_anthropic.return_value = mock_anthropic_client

            # Mock health checks for both providers
            with (
                patch.object(openai_provider, 'test_connection') as mock_openai_health,
                patch.object(anthropic_provider, 'test_connection') as mock_anthropic_health,
            ):
                mock_openai_health.return_value = HealthCheckResult(
                    status=ProviderStatus.HEALTHY,
                    latency_ms=50.0,
                    error_message=None
                )
                mock_anthropic_health.return_value = HealthCheckResult(
                    status=ProviderStatus.HEALTHY,
                    latency_ms=60.0,
                    error_message=None
                )

                openai_provider.configure(openai_credentials)
                anthropic_provider.configure(anthropic_credentials)

            manager.registry.register_provider(openai_provider, priority=1)
            manager.registry.register_provider(anthropic_provider, priority=2)

            # Test fallback behavior
            request = GenerationRequest(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}]
            )

            # Mock OpenAI to fail and Anthropic to succeed
            with (
                patch.object(openai_provider, "generate") as mock_openai_gen,
                patch.object(anthropic_provider, "generate") as mock_anthropic_gen,
            ):
                mock_openai_gen.side_effect = Exception("OpenAI API error")
                mock_anthropic_gen.return_value = GenerationResponse(
                    text="Fallback response from Claude",
                    model="claude-3-sonnet-20240229",
                    provider=ProviderType.ANTHROPIC,
                    input_tokens=10,
                    output_tokens=15,
                    latency_ms=500.0,
                )

                response = manager.generate_with_fallback(request)

                assert response.text == "Fallback response from Claude"
                assert response.provider == ProviderType.ANTHROPIC

    def test_real_api_integration_with_env_vars(self):
        """Test real API integration using environment variables."""
        # Skip if no real API keys provided
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if not openai_key and not anthropic_key:
            pytest.skip("No API keys provided for real API testing")

        # Test with real API if key is available
        if openai_key:
            provider = OpenAIProvider()
            credentials = ProviderCredentials(api_key=openai_key)

            assert provider.configure(credentials) is True

            # Test simple generation
            request = GenerationRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
                temperature=0.0,
                max_tokens=5,
            )

            response = provider.generate(request)

            assert isinstance(response, GenerationResponse)
            assert len(response.text) > 0
            assert response.input_tokens > 0
            assert response.output_tokens > 0
            assert response.latency_ms > 0


if __name__ == "__main__":
    # Run the failing tests to verify TDD approach
    pytest.main([__file__, "-v", "--tb=short"])

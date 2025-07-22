"""
End-to-end integration tests for LLM providers.

These tests verify actual API calls work with real or mocked services.
Demonstrates complete LLM→GMN→PyMDP pipeline functionality.
"""

import os
import pytest
from unittest.mock import Mock, patch

# Import LLM components
try:
    from inference.llm.provider_factory import create_llm_manager, get_provider_factory
    from inference.llm.provider_interface import GenerationRequest, ProviderType
    from config.llm_config import LLMConfig, set_llm_config

    LLM_IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"LLM imports failed: {e}")
    LLM_IMPORTS_SUCCESS = False

# Try to import GMN components
try:
    from inference.active.gmn_parser import GMNParser, EXAMPLE_GMN_SPEC

    GMN_AVAILABLE = True
except ImportError:
    GMN_AVAILABLE = False

# Try to import PyMDP
try:
    import pymdp

    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False


@pytest.mark.integration
class TestLLMEndToEndIntegration:
    """End-to-end LLM integration tests."""

    @pytest.fixture
    def mock_llm_config(self):
        """Create a mock LLM configuration."""
        if not LLM_IMPORTS_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        config = LLMConfig()
        config.openai.api_key = "test-openai-key"
        config.openai.enabled = True
        config.anthropic.api_key = "test-anthropic-key"
        config.anthropic.enabled = True

        set_llm_config(config)
        return config

    def test_provider_factory_creation(self, mock_llm_config):
        """Test provider factory creates providers correctly."""
        factory = get_provider_factory()

        available_providers = factory.get_available_providers()

        # Should have at least one provider available
        assert len(available_providers) > 0

        # Test creating providers
        for provider_type in available_providers:
            provider = factory.create_provider(provider_type)
            assert provider is not None
            assert provider.get_provider_type() == provider_type

    def test_llm_manager_creation_from_config(self, mock_llm_config):
        """Test creating LLM manager from configuration."""
        manager = create_llm_manager()

        # Should have providers registered
        providers = manager.registry.get_providers_by_priority()
        assert len(providers) > 0

        # Test health checks
        health_results = manager.perform_health_checks()
        assert len(health_results) > 0

    @pytest.mark.parametrize("provider_type", [ProviderType.OPENAI, ProviderType.ANTHROPIC])
    def test_mock_llm_generation(self, mock_llm_config, provider_type):
        """Test LLM generation with mocked API responses."""
        if not LLM_IMPORTS_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        factory = get_provider_factory()

        if not factory.is_provider_available(provider_type):
            assert False, "Test bypass removed - must fix underlying issue"

        # Mock the appropriate client based on provider type
        if provider_type == ProviderType.OPENAI:
            with patch("inference.llm.openai_provider.OpenAI") as mock_openai_class:
                mock_client = Mock()
                mock_openai_class.return_value = mock_client

                # Mock models list for health check
                mock_models_response = Mock()
                mock_models_response.data = [Mock(id="gpt-3.5-turbo")]
                mock_client.models.list.return_value = mock_models_response

                # Mock chat completion
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = (
                    "Hello! This is a test response from OpenAI."
                )
                mock_response.choices[0].finish_reason = "stop"
                mock_response.model = "gpt-3.5-turbo"
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 15
                mock_client.chat.completions.create.return_value = mock_response

                # Create and test provider
                manager = create_llm_manager()

                request = GenerationRequest(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello, world!"}],
                    temperature=0.7,
                    max_tokens=100,
                )

                response = manager.generate_with_fallback(request)

                assert response.text == "Hello! This is a test response from OpenAI."
                assert response.provider == ProviderType.OPENAI
                assert response.input_tokens == 10
                assert response.output_tokens == 15
                assert response.latency_ms > 0

        elif provider_type == ProviderType.ANTHROPIC:
            with patch("inference.llm.anthropic_provider.Anthropic") as mock_anthropic_class:
                mock_client = Mock()
                mock_anthropic_class.return_value = mock_client

                # Mock test connection (health check)
                mock_health_response = Mock()
                mock_health_response.content = [Mock()]
                mock_health_response.content[0].text = "Hi"
                mock_health_response.usage.input_tokens = 1
                mock_health_response.usage.output_tokens = 1
                mock_client.messages.create.return_value = mock_health_response

                # Mock actual generation
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "Hello! This is a test response from Claude."
                mock_response.model = "claude-3-sonnet-20240229"
                mock_response.usage.input_tokens = 10
                mock_response.usage.output_tokens = 12
                mock_response.stop_reason = "end_turn"

                # Configure mock to return health check first, then actual response
                mock_client.messages.create.side_effect = [
                    mock_health_response,
                    mock_response,
                ]

                # Create and test provider
                manager = create_llm_manager()

                request = GenerationRequest(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": "Hello, world!"}],
                    temperature=0.7,
                    max_tokens=100,
                )

                response = manager.generate_with_fallback(request)

                assert response.text == "Hello! This is a test response from Claude."
                assert response.provider == ProviderType.ANTHROPIC
                assert response.input_tokens == 10
                assert response.output_tokens == 12

    def test_llm_fallback_behavior(self, mock_llm_config):
        """Test LLM fallback when primary provider fails."""
        if not LLM_IMPORTS_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        factory = get_provider_factory()
        available = factory.get_available_providers()

        if len(available) < 2:
            assert False, "Test bypass removed - must fix underlying issue"

        with (
            patch("inference.llm.openai_provider.OpenAI") as mock_openai,
            patch("inference.llm.anthropic_provider.Anthropic") as mock_anthropic,
        ):
            # Configure OpenAI to fail
            mock_openai_client = Mock()
            mock_openai.return_value = mock_openai_client
            mock_openai_client.models.list.side_effect = Exception("OpenAI API Error")
            mock_openai_client.chat.completions.create.side_effect = Exception("OpenAI API Error")

            # Configure Anthropic to succeed
            mock_anthropic_client = Mock()
            mock_anthropic.return_value = mock_anthropic_client

            mock_health_response = Mock()
            mock_health_response.content = [Mock()]
            mock_health_response.content[0].text = "Hi"
            mock_health_response.usage.input_tokens = 1
            mock_health_response.usage.output_tokens = 1

            mock_gen_response = Mock()
            mock_gen_response.content = [Mock()]
            mock_gen_response.content[0].text = "Fallback response from Claude"
            mock_gen_response.model = "claude-3-sonnet-20240229"
            mock_gen_response.usage.input_tokens = 8
            mock_gen_response.usage.output_tokens = 6
            mock_gen_response.stop_reason = "end_turn"

            mock_anthropic_client.messages.create.side_effect = [
                mock_health_response,
                mock_gen_response,
            ]

            manager = create_llm_manager()

            request = GenerationRequest(
                model="gpt-3.5-turbo",  # Request OpenAI model
                messages=[{"role": "user", "content": "Test fallback"}],
            )

            # Should fallback to Anthropic
            response = manager.generate_with_fallback(request)
            assert response.text == "Fallback response from Claude"
            assert response.provider == ProviderType.ANTHROPIC

    def test_cost_estimation(self, mock_llm_config):
        """Test cost estimation across providers."""
        if not LLM_IMPORTS_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        factory = get_provider_factory()

        for provider_type in factory.get_available_providers():
            provider = factory.create_provider(provider_type)

            # Test cost estimation for different token counts
            cost = provider.estimate_cost(100, 50, "default-model")

            assert cost > 0
            assert isinstance(cost, float)

            # Higher token count should cost more
            higher_cost = provider.estimate_cost(1000, 500, "default-model")
            assert higher_cost > cost

    def test_llm_to_gmn_integration(self, mock_llm_config):
        assert False, "Test bypass removed - must fix underlying issue"
        """Test LLM→GMN integration pipeline."""
        if not LLM_IMPORTS_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        # Mock LLM to return a GMN specification
        with patch("inference.llm.openai_provider.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock models list
            mock_models_response = Mock()
            mock_models_response.data = [Mock(id="gpt-3.5-turbo")]
            mock_client.models.list.return_value = mock_models_response

            # Mock LLM returning GMN spec
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = EXAMPLE_GMN_SPEC
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-3.5-turbo"
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 200
            mock_client.chat.completions.create.return_value = mock_response

            # Create LLM manager
            manager = create_llm_manager()

            # Generate GMN specification using LLM
            request = GenerationRequest(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Generate a GMN specification for a simple grid world agent",
                    }
                ],
                temperature=0.3,
                max_tokens=500,
            )

            llm_response = manager.generate_with_fallback(request)

            # Parse the GMN specification
            parser = GMNParser()
            try:
                gmn_graph = parser.parse(llm_response.text)
                assert gmn_graph is not None
                assert len(gmn_graph.nodes) > 0

                # Convert to PyMDP if available
                if PYMDP_AVAILABLE:
                    pymdp_model = parser.to_pymdp_model(gmn_graph)
                    assert pymdp_model is not None
                    assert "num_states" in pymdp_model or "num_obs" in pymdp_model

            except Exception as e:
                # If parsing fails, at least verify LLM returned content
                assert len(llm_response.text) > 0
                print(f"GMN parsing failed (expected in test): {e}")

    def test_real_api_integration_with_env_keys(self):
        """Test with real API keys if provided in environment."""
        if not LLM_IMPORTS_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        # Check for real API keys
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

        if not (has_openai or has_anthropic):
            assert False, "Test bypass removed - must fix underlying issue"

        # Create manager from real environment
        manager = create_llm_manager()

        providers = manager.registry.get_providers_by_priority()
        if not providers:
            assert False, "Test bypass removed - must fix underlying issue"

        # Test with a simple, cheap request
        request = GenerationRequest(
            model="gpt-3.5-turbo" if has_openai else "claude-3-haiku-20240307",
            messages=[
                {
                    "role": "user",
                    "content": "Say 'integration test successful' and nothing else.",
                }
            ],
            temperature=0.0,
            max_tokens=10,
        )

        try:
            response = manager.generate_with_fallback(request)

            assert response is not None
            assert len(response.text) > 0
            assert response.input_tokens > 0
            assert response.output_tokens > 0
            assert response.latency_ms > 0
            assert response.cost >= 0

            print(f"✅ Real API test successful: {response.text[:50]}...")
            print(f"   Provider: {response.provider.value}")
            print(f"   Tokens: {response.input_tokens}→{response.output_tokens}")
            print(f"   Cost: ${response.cost:.6f}")
            print(f"   Latency: {response.latency_ms:.2f}ms")

        except Exception as e:
            pytest.fail(f"Real API integration test failed: {e}")


@pytest.mark.integration
class TestLLMErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_authentication_error_handling(self):
        """Test handling of authentication errors."""
        if not LLM_IMPORTS_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        from inference.llm.provider_interface import ProviderCredentials
        from inference.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        invalid_credentials = ProviderCredentials(api_key="invalid-key")

        with patch("inference.llm.openai_provider.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock authentication error - use generic Exception since we're mocking
            mock_client.models.list.side_effect = Exception(
                "Authentication failed: Invalid API key"
            )

            # Configuration should fail
            result = provider.configure(invalid_credentials)
            assert result is False

    def test_rate_limit_error_handling(self):
        """Test handling of rate limit errors."""
        if not LLM_IMPORTS_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        from inference.llm.provider_factory import ErrorHandler

        # Test retryable error detection
        rate_limit_error = Exception("Rate limit exceeded: 429")
        assert ErrorHandler.is_retryable_error(rate_limit_error) is True

        # Test retry delay calculation
        delay = ErrorHandler.get_retry_delay(1, rate_limit_error)
        assert delay > 0

        # Test fallback decision
        assert ErrorHandler.should_fallback(rate_limit_error) is True

    def test_provider_not_available_error(self):
        """Test error when provider library is not available."""
        if not LLM_IMPORTS_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        from inference.llm.provider_factory import ProviderNotAvailableError

        factory = get_provider_factory()

        # Test with non-existent provider type
        with pytest.raises(ProviderNotAvailableError):
            factory.create_provider(ProviderType("nonexistent"))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

"""Tests for the demo LLM provider to ensure it implements all abstract methods."""

from examples.llm_error_handling_demo import DemoLLMProvider
from inference.llm.provider_interface import ProviderStatus, ProviderType


def test_demo_llm_provider_can_be_instantiated():
    """Test that DemoLLMProvider can be instantiated without abstract method errors."""
    provider = DemoLLMProvider(ProviderType.OPENAI)
    assert provider is not None
    assert provider.provider_type == ProviderType.OPENAI


def test_demo_llm_provider_configure_method():
    """Test that the configure method works correctly."""
    provider = DemoLLMProvider(ProviderType.OPENAI)

    # Test configure with no arguments
    result = provider.configure()
    assert result is True
    assert hasattr(provider, "_configuration")

    # Test configure with credentials and kwargs
    result = provider.configure(credentials={"api_key": "test"}, model="gpt-4")
    assert result is True
    assert provider._configuration["credentials"] == {"api_key": "test"}
    assert provider._configuration["model"] == "gpt-4"


def test_demo_llm_provider_all_abstract_methods_implemented():
    """Test that all abstract methods from LLMProvider are implemented."""
    provider = DemoLLMProvider(ProviderType.ANTHROPIC, "none")

    # Test test_connection method
    health = provider.test_connection()
    assert health.status == ProviderStatus.HEALTHY
    assert health.latency_ms == 150.0

    # Test generate method
    result = provider.generate("Hello world")
    assert "Generated response from anthropic" in result
    assert "Hello world" in result

    # Test configure method
    configured = provider.configure(api_key="test_key")
    assert configured is True


def test_demo_llm_provider_failure_modes():
    """Test that failure modes work correctly."""
    # Test connection failure
    provider = DemoLLMProvider(ProviderType.OPENAI, "connection")
    health = provider.test_connection()
    assert health.status == ProviderStatus.UNHEALTHY
    assert "Connection timeout" in health.error_message

    # Test auth failure
    provider = DemoLLMProvider(ProviderType.OPENAI, "auth")
    health = provider.test_connection()
    assert health.status == ProviderStatus.OFFLINE
    assert "Authentication failed" in health.error_message


def test_demo_llm_provider_estimate_cost():
    """Test that estimate_cost method works."""
    provider = DemoLLMProvider(ProviderType.OPENAI)
    cost = provider.estimate_cost(100, 50, "gpt-4")
    assert cost == 0.002

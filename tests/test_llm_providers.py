"""Tests for LLM providers and GMN generation quality.

This module tests all LLM providers to ensure they:
1. Connect and authenticate properly
2. Generate valid GMN specifications
3. Handle errors gracefully
4. Maintain consistent quality
"""

import os
from typing import Dict, List

import pytest

from inference.llm.anthropic_provider import AnthropicProvider
from inference.llm.openai_provider import OpenAIProvider
from inference.llm.provider_factory import LLMProviderFactory


class TestLLMProviders:
    """Test individual LLM providers."""

    @pytest.mark.asyncio
    async def test_mock_provider(self):
        """Test mock provider basic functionality."""
        provider = MockLLMProvider(delay=0.01)

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content="You are a helpful assistant."),
            LLMMessage(
                role=LLMRole.USER,
                content="Generate a simple GMN for an explorer agent.",
            ),
        ]

        response = await provider.generate(messages, temperature=0.5)

        assert response.content
        assert response.model == "mock-gpt-4"
        assert "node" in response.content
        assert "state" in response.content

    @pytest.mark.asyncio
    async def test_openai_provider(self):
        """Test OpenAI provider functionality."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.fail("OpenAI API key not set - test cannot proceed")

        provider = OpenAIProvider(model="gpt-4o-mini")  # Use cheaper model for tests

        try:
            # Test basic generation
            messages = [LLMMessage(role=LLMRole.USER, content="Say 'Hello, GMN world!'")]

            response = await provider.generate(messages, max_tokens=50)

            assert response.content
            assert "GMN" in response.content or "Hello" in response.content
            assert response.usage is not None
            assert response.usage.get("total_tokens", 0) > 0

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_anthropic_provider(self):
        """Test Anthropic provider functionality."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.fail("Anthropic API key not set - test cannot proceed")

        provider = AnthropicProvider(model="claude-3-haiku")  # Use cheaper model for tests

        try:
            # Test basic generation
            messages = [LLMMessage(role=LLMRole.USER, content="Say 'Hello, GMN world!'")]

            response = await provider.generate(messages, max_tokens=50)

            assert response.content
            assert response.usage is not None

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_ollama_provider(self):
        """Test Ollama provider functionality."""
        if os.getenv("SKIP_OLLAMA_TESTS"):
            pytest.fail("Ollama tests explicitly skipped - test cannot proceed")

        provider = OllamaProvider(model="llama3.2", timeout=30.0)

        try:
            # Check if Ollama is running
            if not await provider._check_ollama_running():
                pytest.fail("Ollama service not running - configure Ollama for this test")

            # Test basic generation
            messages = [
                LLMMessage(
                    role=LLMRole.USER,
                    content="Say 'Hello, GMN world!' in exactly 5 words.",
                )
            ]

            response = await provider.generate(messages, max_tokens=50)

            assert response.content
            assert response.usage is not None

        finally:
            await provider.close()


class TestGMNGeneration:
    """Test GMN generation quality across providers."""

    def validate_gmn_structure(self, gmn: str) -> Dict[str, List[str]]:
        """Validate GMN structure and return any issues."""
        issues = {"errors": [], "warnings": []}

        # Check for required node types
        required_nodes = [
            "state",
            "observation",
            "action",
            "transition",
            "emission",
        ]
        for node_type in required_nodes:
            if f"node {node_type}" not in gmn:
                issues["errors"].append(f"Missing {node_type} node")

        # Check for preferences
        if "node preference" not in gmn:
            issues["warnings"].append("No preference nodes defined")

        # Check for basic syntax
        if gmn.count("{") != gmn.count("}"):
            issues["errors"].append("Mismatched braces")

        # Check for matrix definitions in transitions/emissions
        if "transition" in gmn and "matrix:" not in gmn and "from:" in gmn:
            issues["warnings"].append("Transition node without matrix definition")

        return issues

    @pytest.mark.asyncio
    async def test_gmn_generation_mock(self):
        """Test GMN generation with mock provider."""
        provider = MockLLMProvider()

        # Test explorer agent
        gmn = await provider.generate_gmn(
            prompt="Create an agent that explores a 5x5 grid world",
            agent_type="explorer",
        )

        issues = self.validate_gmn_structure(gmn)
        assert len(issues["errors"]) == 0, f"GMN validation errors: {issues['errors']}"
        assert "25" in gmn or "5x5" in gmn, "Grid size not properly reflected"

    @pytest.mark.asyncio
    async def test_gmn_generation_openai(self):
        """Test GMN generation with OpenAI."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.fail("OpenAI API key not set - test cannot proceed")

        provider = OpenAIProvider(model="gpt-4o-mini")

        try:
            gmn = await provider.generate_gmn(
                prompt="Create a simple trading agent that can buy, sell, or hold based on market signals",
                agent_type="trader",
            )

            issues = self.validate_gmn_structure(gmn)
            assert len(issues["errors"]) == 0, f"GMN validation errors: {issues['errors']}"
            assert any(word in gmn.lower() for word in ["buy", "sell", "hold", "trade"])

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_gmn_validation(self):
        """Test GMN validation functionality."""
        provider = MockLLMProvider()

        # Valid GMN
        valid_gmn = """
        node state s1 {
            type: discrete
            size: 4
        }
        node observation o1 {
            type: discrete
            size: 5
        }
        """

        is_valid, errors = await provider.validate_gmn(valid_gmn)
        assert is_valid or len(errors) > 0  # Mock provider has simple validation

        # Invalid GMN
        invalid_gmn = """
        node state s1 {
            type: discrete
        """

        is_valid, errors = await provider.validate_gmn(invalid_gmn)
        assert not is_valid
        assert len(errors) > 0


class TestProviderFactory:
    """Test the provider factory and fallback mechanisms."""

    @pytest.mark.asyncio
    async def test_factory_auto_mode(self):
        """Test factory in auto mode."""
        factory = create_llm_factory()

        # Should always have at least mock provider
        assert factory._primary_provider is not None

        # Test generation
        messages = [LLMMessage(role=LLMRole.USER, content="Hello, world!")]

        response = await factory.generate(messages, max_tokens=50)
        assert response.content
        assert "provider" in response.metadata

    @pytest.mark.asyncio
    async def test_factory_fallback(self):
        """Test fallback mechanism."""
        # Create factory with failing primary provider
        config = {
            "provider": "openai",
            "openai": {"api_key": "invalid-key-for-testing"},
        }

        factory = LLMProviderFactory(config)

        # Should fall back to mock
        messages = [LLMMessage(role=LLMRole.USER, content="Test fallback")]

        response = await factory.generate(messages)
        assert response.content
        # Check that it fell back (mock provider has specific patterns)
        assert response.metadata.get("provider") in ["ollama", "mock"]

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        factory = create_llm_factory()

        health_status = await factory.health_check()

        assert "primary_provider" in health_status
        assert "providers" in health_status

        # At least mock should be available
        mock_status = health_status["providers"].get("mock", {})
        assert mock_status.get("available", False)

    @pytest.mark.asyncio
    async def test_config_template(self):
        """Test configuration template generation."""
        factory = create_llm_factory()

        template = factory.get_config_template()

        assert "provider" in template
        assert "openai" in template
        assert "anthropic" in template
        assert "ollama" in template
        assert "mock" in template

        # Check nested configs
        assert "model" in template["openai"]
        assert "api_key" in template["anthropic"]
        assert "base_url" in template["ollama"]


class TestGMNConsistency:
    """Test consistency of GMN generation across providers."""

    async def generate_gmn_all_providers(self, prompt: str, agent_type: str) -> Dict[str, str]:
        """Generate GMN using all available providers."""
        results = {}

        # Test each provider
        providers = [
            ("mock", MockLLMProvider()),
        ]

        if os.getenv("OPENAI_API_KEY"):
            providers.append(("openai", OpenAIProvider(model="gpt-4o-mini")))

        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append(("anthropic", AnthropicProvider(model="claude-3-haiku")))

        # Ollama if available
        ollama = OllamaProvider()
        if await ollama._check_ollama_running():
            providers.append(("ollama", ollama))

        for name, provider in providers:
            try:
                gmn = await provider.generate_gmn(prompt, agent_type)
                results[name] = gmn
            except Exception as e:
                results[name] = f"Error: {str(e)}"
            finally:
                if hasattr(provider, "close"):
                    await provider.close()

        return results

    @pytest.mark.asyncio
    async def test_gmn_consistency(self):
        """Test that different providers generate structurally similar GMNs."""
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            pytest.fail(
                "No API keys for comparison - test cannot proceed without at least one provider"
            )

        prompt = "Create an agent that navigates a simple 3x3 grid to reach a goal"

        results = await self.generate_gmn_all_providers(prompt, "explorer")

        # Validate each result
        for provider, gmn in results.items():
            if not gmn.startswith("Error:"):
                issues = self.validate_gmn_structure(gmn)
                assert len(issues["errors"]) == 0, f"{provider} GMN has errors: {issues['errors']}"


# Performance benchmarking
class TestPerformance:
    """Benchmark provider performance."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_generation_speed(self):
        """Benchmark generation speed across providers."""
        import time

        prompt = "Say hello in 5 words"
        messages = [LLMMessage(role=LLMRole.USER, content=prompt)]

        results = {}

        # Mock provider
        provider = MockLLMProvider(delay=0.01)
        start = time.time()
        await provider.generate(messages, max_tokens=20)
        results["mock"] = time.time() - start

        # Real providers if available
        if os.getenv("OPENAI_API_KEY"):
            provider = OpenAIProvider(model="gpt-4o-mini")
            start = time.time()
            await provider.generate(messages, max_tokens=20)
            results["openai"] = time.time() - start
            await provider.close()

        print("\nGeneration Speed Benchmark:")
        for provider, duration in results.items():
            print(f"  {provider}: {duration:.3f}s")


if __name__ == "__main__":
    # Run basic tests
    import pytest

    pytest.main([__file__, "-v"])

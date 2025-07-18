"""Test suite for Mock LLM Provider."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.base import LLMError, LLMMessage, LLMRole
from llm.providers.mock import MockLLMProvider


class TestMockLLMProvider:
    """Test the mock LLM provider implementation."""

    @pytest.fixture
    def provider(self):
        """Create a mock provider instance."""
        return MockLLMProvider(delay=0.01)  # Short delay for testing

    @pytest.mark.asyncio
    async def test_generate_basic(self, provider):
        """Test basic generation functionality."""
        messages = [
            LLMMessage(role=LLMRole.USER, content="Create an explorer agent")
        ]

        response = await provider.generate(messages)

        assert response.content
        assert response.model == "mock-gpt-4"
        assert response.usage is not None
        assert response.usage["total_tokens"] > 0
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_gmn(self, provider):
        """Test GMN generation."""
        gmn = await provider.generate_gmn(
            prompt="Create an explorer agent for a 5x5 grid world",
            agent_type="explorer",
        )

        # Check GMN structure
        assert "node state" in gmn
        assert "node observation" in gmn
        assert "node action" in gmn
        assert "node transition" in gmn
        assert "size: 25" in gmn  # 5x5 grid

    @pytest.mark.asyncio
    async def test_generate_different_agent_types(self, provider):
        """Test generation for different agent types."""
        agent_types = ["explorer", "trader", "coordinator", "general"]

        for agent_type in agent_types:
            gmn = await provider.generate_gmn(
                prompt=f"Create a {agent_type} agent", agent_type=agent_type
            )

            assert "node state" in gmn
            assert "node action" in gmn

            # Type-specific checks
            if agent_type == "trader":
                assert "market" in gmn.lower() or "portfolio" in gmn.lower()
            elif agent_type == "coordinator":
                assert "team" in gmn.lower() or "task" in gmn.lower()

    @pytest.mark.asyncio
    async def test_validate_gmn_valid(self, provider):
        """Test GMN validation for valid specifications."""
        valid_gmn = """
        node state s1 {
            type: discrete
            size: 4
        }
        node action a1 {
            type: discrete
            size: 3
        }
        node transition T1 {
            from: [s1, a1]
            to: s1
        }
        """

        messages = [
            LLMMessage(
                role=LLMRole.SYSTEM,
                content="You are a GMN validation expert. Respond only with valid JSON.",
            ),
            LLMMessage(
                role=LLMRole.USER,
                content=f"Validate the following GMN specification:\n```\n{valid_gmn}\n```",
            ),
        ]

        response = await provider.generate(messages)
        result = json.loads(response.content)

        assert result["valid"] is True
        assert len(result.get("errors", [])) == 0

    @pytest.mark.asyncio
    async def test_validate_gmn_invalid(self, provider):
        """Test GMN validation for invalid specifications."""
        invalid_gmn = """
        node s1 {
            size: 4
        }
        """

        is_valid, errors = await provider.validate_gmn(invalid_gmn)

        assert is_valid is False
        assert len(errors) > 0
        assert any("state" in error for error in errors)

    @pytest.mark.asyncio
    async def test_refine_gmn(self, provider):
        """Test GMN refinement based on feedback."""
        original_gmn = """
        node state s1 {
            type: discrete
            size: 4
        }
        """

        refined_gmn = await provider.refine_gmn(
            original_gmn, "Add observation and preference nodes"
        )

        assert "node observation" in refined_gmn
        assert "node preference" in refined_gmn

    @pytest.mark.asyncio
    async def test_error_simulation(self):
        """Test error simulation functionality."""
        provider = MockLLMProvider(delay=0.01, error_rate=1.0)  # Always error

        with pytest.raises(LLMError):
            await provider.generate(
                [LLMMessage(role=LLMRole.USER, content="Test")]
            )

    @pytest.mark.asyncio
    async def test_cautious_preference_customization(self, provider):
        """Test that cautious prompts generate appropriate preferences."""
        gmn = await provider.generate_gmn(
            prompt="Create a cautious explorer agent", agent_type="explorer"
        )

        assert "preference" in gmn
        assert "0.9" in gmn  # High preference for safe states

    @pytest.mark.asyncio
    async def test_exploration_preference_customization(self, provider):
        """Test that exploration prompts generate appropriate preferences."""
        gmn = await provider.generate_gmn(
            prompt="Create a curious explorer agent", agent_type="explorer"
        )

        assert "preference" in gmn
        assert "exploration" in gmn.lower() or "information_gain" in gmn

    def test_validate_model(self, provider):
        """Test model validation."""
        assert provider.validate_model("mock-gpt-4") is True
        assert provider.validate_model("gpt-4") is False

    def test_get_token_limit(self, provider):
        """Test token limit retrieval."""
        assert provider.get_token_limit("mock-gpt-4") == 4096
        assert provider.get_token_limit("unknown-model") == 0

    @pytest.mark.asyncio
    async def test_grid_size_extraction(self, provider):
        """Test that grid sizes are extracted from prompts."""
        gmn = await provider.generate_gmn(
            prompt="Create an explorer for a 10x10 grid", agent_type="explorer"
        )

        assert "size: 100" in gmn  # 10x10 = 100

    @pytest.mark.asyncio
    async def test_no_user_message(self, provider):
        """Test handling of messages without user content."""
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content="System prompt only")
        ]

        with pytest.raises(LLMError, match="No user message found"):
            await provider.generate(messages)

    @pytest.mark.asyncio
    async def test_temperature_metadata(self, provider):
        """Test that temperature is included in response metadata."""
        response = await provider.generate(
            messages=[LLMMessage(role=LLMRole.USER, content="Test")],
            temperature=0.5,
        )

        assert response.metadata["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_concurrent_generations(self, provider):
        """Test concurrent generation requests."""
        tasks = []
        for i in range(5):
            task = provider.generate(
                [LLMMessage(role=LLMRole.USER, content=f"Test {i}")]
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        assert all(r.content for r in responses)

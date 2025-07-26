"""Test suite for GMN Generator service."""

from unittest.mock import AsyncMock

import pytest

from llm.base import LLMError
from services.gmn_generator import GMNGenerator


class TestGMNGenerator:
    """Test the GMN generator service."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.generate_gmn = AsyncMock()
        provider.validate_gmn = AsyncMock()
        provider.refine_gmn = AsyncMock()
        return provider

    @pytest.fixture
    def generator(self, mock_llm_provider):
        """Create GMN generator with mock provider."""
        return GMNGenerator(llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_prompt_to_gmn_success(self, generator, mock_llm_provider):
        """Test successful GMN generation from prompt."""
        expected_gmn = """
        node state s1 {
            type: discrete
            size: 4
        }
        node action a1 {
            type: discrete
            size: 3
        }
        """
        mock_llm_provider.generate_gmn.return_value = expected_gmn

        result = await generator.prompt_to_gmn(
            prompt="Create an explorer agent", agent_type="explorer"
        )

        assert result == expected_gmn
        mock_llm_provider.generate_gmn.assert_called_once_with(
            prompt="Create an explorer agent",
            agent_type="explorer",
            constraints=None,
        )

    @pytest.mark.asyncio
    async def test_prompt_to_gmn_empty_prompt(self, generator):
        """Test that empty prompts are rejected."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await generator.prompt_to_gmn("")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await generator.prompt_to_gmn("   ")

    @pytest.mark.asyncio
    async def test_prompt_to_gmn_empty_result(self, generator, mock_llm_provider):
        """Test handling of empty GMN generation."""
        mock_llm_provider.generate_gmn.return_value = ""

        with pytest.raises(LLMError, match="Generated empty GMN"):
            await generator.prompt_to_gmn("Test prompt")

    @pytest.mark.asyncio
    async def test_prompt_to_gmn_no_nodes(self, generator, mock_llm_provider):
        """Test handling of GMN without node definitions."""
        mock_llm_provider.generate_gmn.return_value = "invalid gmn specification"

        with pytest.raises(LLMError, match="no node definitions"):
            await generator.prompt_to_gmn("Test prompt")

    @pytest.mark.asyncio
    async def test_prompt_to_gmn_llm_error(self, generator, mock_llm_provider):
        """Test handling of LLM errors."""
        mock_llm_provider.generate_gmn.side_effect = LLMError("Provider error")

        with pytest.raises(LLMError):
            await generator.prompt_to_gmn("Test prompt")

    @pytest.mark.asyncio
    async def test_validate_gmn_success(self, generator, mock_llm_provider):
        """Test successful GMN validation with properly formed spec."""
        gmn_spec = """
        node state s1 {
            type: discrete
            size: 4
        }
        node action a1 {
            type: discrete
            size: 3
        }
        node observation o1 {
            type: discrete
            size: 5
        }
        node transition T1 {
            from: [s1, a1]
            to: s1
        }
        node emission E1 {
            from: s1
            to: o1
        }
        """
        # Mock returns True for valid GMN
        mock_llm_provider.validate_gmn.return_value = (True, [])

        is_valid, errors = await generator.validate_gmn(gmn_spec)

        assert is_valid is True
        assert errors == []

    @pytest.mark.asyncio
    async def test_validate_gmn_with_errors(self, generator, mock_llm_provider):
        """Test GMN validation with errors."""
        gmn_spec = "invalid gmn"
        mock_llm_provider.validate_gmn.return_value = (
            False,
            ["Missing node definitions"],
        )

        is_valid, errors = await generator.validate_gmn(gmn_spec)

        assert is_valid is False
        assert "Missing node definitions" in errors

    @pytest.mark.asyncio
    async def test_validate_gmn_structural_errors(self, generator, mock_llm_provider):
        """Test structural validation catches errors."""
        gmn_spec = """
        node state s1 {
            type: discrete
            size: 4
        }
        node transition T1 {
            from: [s1, undefined_node]
            to: s1
        }
        """
        mock_llm_provider.validate_gmn.return_value = (True, [])

        is_valid, errors = await generator.validate_gmn(gmn_spec)

        assert is_valid is False
        assert any("Undefined node references" in error for error in errors)
        assert "undefined_node" in str(errors)

    @pytest.mark.asyncio
    async def test_validate_gmn_unbalanced_braces(self, generator, mock_llm_provider):
        """Test detection of unbalanced braces."""
        gmn_spec = """
        node state s1 {
            type: discrete
            size: 4
        """  # Missing closing brace
        mock_llm_provider.validate_gmn.return_value = (True, [])

        is_valid, errors = await generator.validate_gmn(gmn_spec)

        assert is_valid is False
        assert any("Unbalanced braces" in error for error in errors)

    @pytest.mark.asyncio
    async def test_validate_gmn_missing_required_types(self, generator, mock_llm_provider):
        """Test detection of missing required node types."""
        gmn_spec = """
        node observation o1 {
            type: discrete
            size: 5
        }
        """  # Missing state and action nodes
        mock_llm_provider.validate_gmn.return_value = (True, [])

        is_valid, errors = await generator.validate_gmn(gmn_spec)

        assert is_valid is False
        assert any("Missing required node types" in error for error in errors)

    @pytest.mark.asyncio
    async def test_refine_gmn_success(self, generator, mock_llm_provider):
        """Test successful GMN refinement."""
        original_gmn = "node state s1 { size: 4 }"
        refined_gmn = "node state s1 { type: discrete\n size: 4 }"
        mock_llm_provider.refine_gmn.return_value = refined_gmn
        mock_llm_provider.validate_gmn.return_value = (True, [])

        result = await generator.refine_gmn(original_gmn, "Add type specification")

        assert result == refined_gmn
        mock_llm_provider.refine_gmn.assert_called_once_with(original_gmn, "Add type specification")

    @pytest.mark.asyncio
    async def test_refine_gmn_empty_inputs(self, generator):
        """Test that empty inputs are rejected."""
        with pytest.raises(ValueError, match="GMN specification cannot be empty"):
            await generator.refine_gmn("", "feedback")

        with pytest.raises(ValueError, match="Feedback cannot be empty"):
            await generator.refine_gmn("gmn spec", "")

    @pytest.mark.asyncio
    async def test_suggest_improvements(self, generator):
        """Test improvement suggestions."""
        gmn_without_preferences = """
        node state s1 { type: discrete; size: 4 }
        node action a1 { type: discrete; size: 3 }
        node transition T1 { from: [s1, a1]; to: s1 }
        """

        suggestions = await generator.suggest_improvements(gmn_without_preferences)

        assert any("preference" in s for s in suggestions)
        assert any("observation" in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_suggest_improvements_complete_gmn(self, generator):
        """Test suggestions for a more complete GMN."""
        complete_gmn = """
        node state s1 {
            type: discrete
            size: 4
            description: "Agent position"
            initial: [0.25, 0.25, 0.25, 0.25]
        }
        node observation o1 {
            type: discrete
            size: 5
            description: "Sensor readings"
        }
        node action a1 {
            type: discrete
            size: 3
            description: "Movement actions"
        }
        node transition T1 {
            from: [s1, a1]
            to: s1
            stochastic: true
            description: "State transitions"
        }
        node emission E1 {
            from: s1
            to: o1
            description: "Observation model"
        }
        node preference C1 {
            state: s1
            values: [0, 0, 1, 0]
            description: "Goal preference"
        }
        """

        suggestions = await generator.suggest_improvements(complete_gmn)

        # Should have fewer suggestions for complete GMN
        assert len(suggestions) < 3

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, generator, mock_llm_provider):
        """Test concurrent GMN operations."""
        import asyncio

        mock_llm_provider.generate_gmn.return_value = "node state s1 { size: 4 }"
        mock_llm_provider.validate_gmn.return_value = (True, [])

        # Run multiple operations concurrently
        tasks = [generator.prompt_to_gmn(f"Prompt {i}") for i in range(3)]
        tasks.extend([generator.validate_gmn(f"GMN {i}") for i in range(3)])

        results = await asyncio.gather(*tasks)

        assert len(results) == 6
        assert mock_llm_provider.generate_gmn.call_count == 3
        assert mock_llm_provider.validate_gmn.call_count == 3

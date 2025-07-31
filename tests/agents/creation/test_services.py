"""Tests for agent creation services.

Comprehensive tests for all service implementations with mocked LLM interactions
and database operations. Following Kent Beck's TDD approach.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents.creation.models import (
    AgentBuildError,
    AgentSpecification,
    AnalysisConfidence,
    PersonalityProfile,
    PromptAnalysisError,
)
from agents.creation.services import (
    AgentBuilder,
    LLMPromptAnalyzer,
    PersonalityGenerator,
    SystemPromptBuilder,
)
from database.models import AgentType


class TestLLMPromptAnalyzer:
    """Test LLM-powered prompt analyzer."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        service = Mock()
        service.get_provider_manager = AsyncMock()
        return service

    @pytest.fixture
    def analyzer(self, mock_llm_service):
        """Create analyzer with mocked LLM service."""
        return LLMPromptAnalyzer(mock_llm_service)

    @pytest.mark.asyncio
    async def test_validates_good_prompt(self, analyzer):
        """Should validate a good prompt as suitable."""
        result = await analyzer.validate_prompt("I need help analyzing market data trends")
        assert result is True

    @pytest.mark.asyncio
    async def test_rejects_empty_prompt(self, analyzer):
        """Should reject empty or too short prompts."""
        assert await analyzer.validate_prompt("") is False
        assert await analyzer.validate_prompt("   ") is False
        assert await analyzer.validate_prompt("test") is False

    @pytest.mark.asyncio
    async def test_rejects_invalid_patterns(self, analyzer):
        """Should reject prompts with invalid patterns."""
        assert await analyzer.validate_prompt("123456789") is False
        assert await analyzer.validate_prompt("!@#$%^&*()") is False
        assert await analyzer.validate_prompt("test test test") is False

    @pytest.mark.asyncio
    async def test_analyzes_prompt_with_llm_success(self, analyzer, mock_llm_service):
        """Should successfully analyze prompt using LLM."""
        # Mock the LLM response
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "agent_type": "analyst",
                "confidence": "high",
                "domain": "finance",
                "capabilities": ["data_analysis", "trend_identification"],
                "context": "market analysis",
                "reasoning": "User needs data analysis capabilities",
                "alternative_types": ["critic"],
            }
        )
        mock_provider.generate_with_fallback.return_value = mock_response
        mock_llm_service.get_provider_manager.return_value = mock_provider

        result = await analyzer.analyze_prompt("Help me analyze market trends")

        assert result.agent_type == AgentType.ANALYST
        assert result.confidence == AnalysisConfidence.HIGH
        assert result.domain == "finance"
        assert "data_analysis" in result.capabilities
        assert result.reasoning == "User needs data analysis capabilities"
        assert AgentType.CRITIC in result.alternative_types

    @pytest.mark.asyncio
    async def test_analyzes_prompt_with_fallback_on_llm_failure(self, analyzer, mock_llm_service):
        """Should fall back to rule-based analysis when LLM fails."""
        # Mock LLM failure
        mock_llm_service.get_provider_manager.side_effect = Exception("LLM service down")

        result = await analyzer.analyze_prompt("Help me analyze data trends")

        assert result.agent_type == AgentType.ANALYST  # Rule-based should detect "analyze"
        assert result.confidence == AnalysisConfidence.MEDIUM
        assert result.reasoning == "Fallback rule-based analysis"

    @pytest.mark.asyncio
    async def test_rule_based_classification_advocate(self, analyzer, mock_llm_service):
        """Should classify advocate prompts using rules."""
        mock_llm_service.get_provider_manager.side_effect = Exception("LLM down")

        result = await analyzer.analyze_prompt("Help me argue for climate change action")

        assert result.agent_type == AgentType.ADVOCATE

    @pytest.mark.asyncio
    async def test_rule_based_classification_critic(self, analyzer, mock_llm_service):
        """Should classify critic prompts using rules."""
        mock_llm_service.get_provider_manager.side_effect = Exception("LLM down")

        result = await analyzer.analyze_prompt("Find problems in this business plan")

        assert result.agent_type == AgentType.CRITIC

    @pytest.mark.asyncio
    async def test_rule_based_classification_creative(self, analyzer, mock_llm_service):
        """Should classify creative prompts using rules."""
        mock_llm_service.get_provider_manager.side_effect = Exception("LLM down")

        result = await analyzer.analyze_prompt("Help me brainstorm innovative solutions")

        assert result.agent_type == AgentType.CREATIVE

    @pytest.mark.asyncio
    async def test_handles_invalid_prompt_error(self, analyzer):
        """Should raise error for invalid prompts."""
        with pytest.raises(PromptAnalysisError):
            await analyzer.analyze_prompt("")

    @pytest.mark.asyncio
    async def test_parses_malformed_llm_json(self, analyzer, mock_llm_service):
        """Should handle malformed JSON from LLM gracefully."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "```json\n{invalid json}\n```"
        mock_provider.generate_with_fallback.return_value = mock_response
        mock_llm_service.get_provider_manager.return_value = mock_provider

        # Should fall back to rule-based analysis
        result = await analyzer.analyze_prompt("analyze market data")
        assert result.agent_type == AgentType.ANALYST
        assert result.reasoning == "Fallback rule-based analysis"


class TestPersonalityGenerator:
    """Test personality profile generator."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        service = Mock()
        service.get_provider_manager = AsyncMock()
        return service

    @pytest.fixture
    def generator(self, mock_llm_service):
        """Create generator with mocked LLM service."""
        return PersonalityGenerator(mock_llm_service)

    def test_provides_default_personalities(self, generator):
        """Should provide default personalities for all agent types."""
        for agent_type in AgentType:
            personality = generator.get_default_personality(agent_type)
            assert isinstance(personality, PersonalityProfile)
            assert 0.0 <= personality.assertiveness <= 1.0

    def test_default_advocate_personality(self, generator):
        """Should provide appropriate default for advocate agents."""
        personality = generator.get_default_personality(AgentType.ADVOCATE)

        assert personality.assertiveness > 0.7  # Advocates should be assertive
        assert personality.empathy > 0.5  # But also empathetic
        assert "persuasiveness" in personality.custom_traits

    def test_default_analyst_personality(self, generator):
        """Should provide appropriate default for analyst agents."""
        personality = generator.get_default_personality(AgentType.ANALYST)

        assert personality.analytical_depth > 0.8  # Highly analytical
        assert personality.skepticism > 0.5  # Appropriately skeptical
        assert "methodicalness" in personality.custom_traits

    def test_default_critic_personality(self, generator):
        """Should provide appropriate default for critic agents."""
        personality = generator.get_default_personality(AgentType.CRITIC)

        assert personality.skepticism > 0.8  # Very skeptical
        assert personality.empathy < 0.5  # Less empathetic, more objective
        assert "scrutiny" in personality.custom_traits

    @pytest.mark.asyncio
    async def test_generates_personality_with_llm_success(self, generator, mock_llm_service):
        """Should generate personality using LLM successfully."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "assertiveness": 0.8,
                "analytical_depth": 0.9,
                "creativity": 0.3,
                "empathy": 0.6,
                "skepticism": 0.7,
                "formality": 0.8,
                "verbosity": 0.7,
                "collaboration": 0.5,
                "speed": 0.4,
                "custom_traits": {"precision": 0.9},
            }
        )
        mock_provider.generate_with_fallback.return_value = mock_response
        mock_llm_service.get_provider_manager.return_value = mock_provider

        personality = await generator.generate_personality(
            AgentType.ANALYST, "financial analysis context"
        )

        assert personality.assertiveness == 0.8
        assert personality.analytical_depth == 0.9
        assert personality.custom_traits["precision"] == 0.9

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_failure(self, generator, mock_llm_service):
        """Should fall back to default with enhancements on LLM failure."""
        mock_llm_service.get_provider_manager.side_effect = Exception("LLM down")

        personality = await generator.generate_personality(
            AgentType.CREATIVE, "urgent creative project"
        )

        # Should be based on default creative personality
        assert personality.creativity > 0.8
        # Should have speed enhancement due to "urgent" context
        assert personality.speed > 0.8

    @pytest.mark.asyncio
    async def test_enhances_personality_based_on_context(self, generator, mock_llm_service):
        """Should enhance default personality based on context keywords."""
        mock_llm_service.get_provider_manager.side_effect = Exception("LLM down")

        # Test urgency enhancement
        personality = await generator.generate_personality(AgentType.ANALYST, "need quick analysis")
        assert personality.speed > 0.4  # Enhanced from default

        # Test detail enhancement
        personality = await generator.generate_personality(
            AgentType.ANALYST, "need detailed comprehensive report"
        )
        assert personality.verbosity > 0.8  # Enhanced from default
        assert personality.analytical_depth > 0.9  # Enhanced from default

    @pytest.mark.asyncio
    async def test_validates_llm_personality_values(self, generator, mock_llm_service):
        """Should validate and clamp LLM-generated personality values."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "assertiveness": 1.5,  # Too high
                "analytical_depth": -0.2,  # Too low
                "creativity": 0.7,
                "empathy": 0.5,
                "skepticism": 0.6,
                "formality": 0.5,
                "verbosity": 0.5,
                "collaboration": 0.5,
                "speed": 0.5,
            }
        )
        mock_provider.generate_with_fallback.return_value = mock_response
        mock_llm_service.get_provider_manager.return_value = mock_provider

        personality = await generator.generate_personality(AgentType.ADVOCATE)

        assert personality.assertiveness == 1.0  # Clamped to max
        assert personality.analytical_depth == 0.0  # Clamped to min


class TestSystemPromptBuilder:
    """Test system prompt builder."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        service = Mock()
        service.get_provider_manager = AsyncMock()
        return service

    @pytest.fixture
    def builder(self, mock_llm_service):
        """Create builder with mocked LLM service."""
        return SystemPromptBuilder(mock_llm_service)

    def test_provides_template_prompts(self, builder):
        """Should provide template prompts for all agent types."""
        for agent_type in AgentType:
            prompt = builder.get_template_prompt(agent_type)
            assert isinstance(prompt, str)
            assert len(prompt) > 50  # Should be substantial
            assert agent_type.value.title() in prompt or agent_type.value.upper() in prompt

    def test_advocate_template_mentions_arguments(self, builder):
        """Should include argument-building in advocate template."""
        prompt = builder.get_template_prompt(AgentType.ADVOCATE)
        assert "argument" in prompt.lower() or "position" in prompt.lower()

    def test_analyst_template_mentions_analysis(self, builder):
        """Should include analysis focus in analyst template."""
        prompt = builder.get_template_prompt(AgentType.ANALYST)
        assert "analy" in prompt.lower() and "data" in prompt.lower()

    @pytest.mark.asyncio
    async def test_builds_prompt_with_llm_success(self, builder, mock_llm_service):
        """Should build custom prompt using LLM successfully."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "You are a highly assertive Financial Analyst agent..."
        mock_provider.generate_with_fallback.return_value = mock_response
        mock_llm_service.get_provider_manager.return_value = mock_provider

        personality = PersonalityProfile(assertiveness=0.9, formality=0.8)

        prompt = await builder.build_system_prompt(
            AgentType.ANALYST, personality, "financial analysis", ["data_analysis", "reporting"]
        )

        assert prompt == "You are a highly assertive Financial Analyst agent..."

    @pytest.mark.asyncio
    async def test_falls_back_to_template_on_llm_failure(self, builder, mock_llm_service):
        """Should fall back to template-based approach on LLM failure."""
        mock_llm_service.get_provider_manager.side_effect = Exception("LLM down")

        personality = PersonalityProfile(assertiveness=0.9, formality=0.2)

        prompt = await builder.build_system_prompt(
            AgentType.CREATIVE, personality, "marketing campaign", ["brainstorming"]
        )

        # Should be based on template
        assert "Creative" in prompt
        # Should include personality guidelines
        assert "direct and confident" in prompt  # High assertiveness
        assert "conversational and approachable" in prompt  # Low formality
        # Should include context and capabilities
        assert "marketing campaign" in prompt
        assert "brainstorming" in prompt


class TestAgentBuilder:
    """Test agent builder and database persistence."""

    @pytest.fixture
    def builder(self):
        """Create agent builder."""
        return AgentBuilder()

    @pytest.fixture
    def sample_specification(self):
        """Create sample agent specification."""
        personality = PersonalityProfile(assertiveness=0.8, creativity=0.6)
        return AgentSpecification(
            name="Test Analyst",
            agent_type=AgentType.ANALYST,
            system_prompt="You are a test analyst agent.",
            personality=personality,
            source_prompt="Help me analyze data",
            capabilities=["analysis", "reporting"],
        )

    @pytest.mark.asyncio
    async def test_builds_agent_successfully(self, builder, sample_specification):
        """Should build and persist agent successfully."""
        with patch("agents.creation.services.get_db") as mock_session_ctx:
            mock_session = Mock()
            mock_session_ctx.return_value.__iter__ = Mock(return_value=iter([mock_session]))

            # Mock the agent creation
            from database.models import Agent

            created_agent = Agent(
                id="test-id",
                name=sample_specification.name,
                agent_type=sample_specification.agent_type,
                system_prompt=sample_specification.system_prompt,
            )
            mock_session.refresh = Mock()
            mock_session.refresh.side_effect = lambda agent: setattr(agent, "id", "test-id")

            result = await builder.build_agent(sample_specification)

            # Verify database operations
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_database_error(self, builder, sample_specification):
        """Should handle database errors gracefully."""
        with patch("agents.creation.services.get_db") as mock_session_ctx:
            mock_session = Mock()
            mock_session_ctx.return_value.__iter__ = Mock(return_value=iter([mock_session]))
            mock_session.commit.side_effect = Exception("Database error")

            with pytest.raises(AgentBuildError):
                await builder.build_agent(sample_specification)

    @pytest.mark.asyncio
    async def test_updates_existing_agent(self, builder, sample_specification):
        """Should update existing agent successfully."""
        with patch("agents.creation.services.get_db") as mock_session_ctx:
            mock_session = Mock()
            mock_session_ctx.return_value.__iter__ = Mock(return_value=iter([mock_session]))

            # Mock existing agent
            from database.models import Agent

            existing_agent = Agent(id="existing-id", name="Old Name")
            mock_session.get.return_value = existing_agent

            result = await builder.update_agent("existing-id", sample_specification)

            # Verify update operations
            mock_session.get.assert_called_once_with(Agent, "existing-id")
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_agent_not_found_error(self, builder, sample_specification):
        """Should handle agent not found during update."""
        with patch("agents.creation.services.get_db") as mock_session_ctx:
            mock_session = Mock()
            mock_session_ctx.return_value.__iter__ = Mock(return_value=iter([mock_session]))
            mock_session.get.return_value = None  # Agent not found

            with pytest.raises(AgentBuildError, match="Agent .* not found"):
                await builder.update_agent("nonexistent-id", sample_specification)

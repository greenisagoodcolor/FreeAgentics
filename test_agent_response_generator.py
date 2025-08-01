"""
Comprehensive tests for Agent Response Generator

Following TDD principles as advocated by Kent Beck, with characterization tests
for existing behavior and comprehensive coverage of the new implementation.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from api.v1.models.agent_conversation import (
    AgentRole,
    ConversationContext,
    ConversationTurnDomain,
    TurnStatus,
)
from api.v1.services.agent_response_generator import (
    AgentPersonaType,
    AgentResponseGenerator,
    ResponseAnalyzer,
    ResponseQuality,
    TemplatePromptBuilder,
    create_agent_response_generator,
)
from api.v1.services.llm_service import LLMService


class TestResponseAnalyzer:
    """Test suite for ResponseAnalyzer quality assessment."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes with correct default metrics."""
        analyzer = ResponseAnalyzer()

        assert analyzer.metrics["total_responses"] == 0
        assert analyzer.metrics["quality_scores"] == []
        assert analyzer.metrics["avg_response_length"] == 0.0
        assert analyzer.metrics["role_coherence_rate"] == 0.0

    def test_analyze_response_basic_metrics(self):
        """Test basic response analysis metrics calculation."""
        analyzer = ResponseAnalyzer()

        # Create test agent and context
        agent = AgentRole(
            agent_id="test-explorer",
            name="Explorer",
            role="explorer",
            system_prompt="You explore and discover new things.",
        )

        context = ConversationContext(
            conversation_id="test-conv",
            topic="Test exploration topic",
            participants=[agent],
        )

        response = "Let's explore this fascinating topic! What hidden aspects might we discover?"
        analysis = analyzer.analyze_response(response, agent, context)

        # Verify analysis structure
        assert "quality_score" in analysis
        assert "quality_level" in analysis
        assert "response_length" in analysis
        assert "word_count" in analysis
        assert "role_coherence" in analysis
        assert "analysis_timestamp" in analysis

        # Verify metrics updated
        assert analyzer.metrics["total_responses"] == 1
        assert len(analyzer.metrics["quality_scores"]) == 1
        assert analysis["response_length"] == len(response)
        assert analysis["word_count"] == len(response.split())

    def test_role_coherence_calculation(self):
        """Test role coherence calculation for different agent types."""
        analyzer = ResponseAnalyzer()

        # Explorer agent with exploration response
        explorer_agent = AgentRole(
            agent_id="explorer-1",
            name="Explorer",
            role="explorer",
            system_prompt="You explore and investigate.",
        )

        context = ConversationContext(
            conversation_id="test-conv",
            topic="Research topic",
            participants=[explorer_agent],
        )

        # Response with exploration keywords
        exploration_response = "Let's explore this further and discover what we can find!"
        analysis = analyzer.analyze_response(exploration_response, explorer_agent, context)

        # Should have good role coherence due to exploration keywords
        assert analysis["role_coherence"] > 0.5

        # Generic response with no role-specific keywords
        generic_response = "This is interesting and worth discussing."
        analysis_generic = analyzer.analyze_response(generic_response, explorer_agent, context)

        # Should have lower role coherence
        assert analysis_generic["role_coherence"] < analysis["role_coherence"]

    def test_quality_score_calculation(self):
        """Test quality score calculation logic."""
        analyzer = ResponseAnalyzer()

        agent = AgentRole(
            agent_id="test-agent",
            name="TestAgent",
            role="analyst",
            system_prompt="You analyze things.",
        )

        context = ConversationContext(
            conversation_id="test-conv",
            topic="Analysis topic",
            participants=[agent],
        )

        # High quality response
        high_quality = (
            "Let me analyze this carefully. The evidence suggests several important patterns. "
            "First, we can see clear trends in the data. Second, the implications are significant. "
            "What questions should we explore next?"
        )

        analysis_high = analyzer.analyze_response(high_quality, agent, context)

        # Low quality response (too short)
        low_quality = "Yes."
        analysis_low = analyzer.analyze_response(low_quality, agent, context)

        # High quality should score better
        assert analysis_high["quality_score"] > analysis_low["quality_score"]
        assert analysis_high["quality_level"] in [ResponseQuality.GOOD, ResponseQuality.EXCELLENT]
        assert analysis_low["quality_level"] in [ResponseQuality.POOR, ResponseQuality.FAILED]

    def test_get_metrics_aggregation(self):
        """Test metrics aggregation and computed values."""
        analyzer = ResponseAnalyzer()

        agent = AgentRole(
            agent_id="test-agent",
            name="TestAgent",
            role="default",
            system_prompt="Test agent.",
        )

        context = ConversationContext(
            conversation_id="test-conv",
            topic="Test topic",
            participants=[agent],
        )

        # Analyze multiple responses
        responses = [
            "This is an excellent response with great detail and insight!",
            "Good response with decent content.",
            "Okay response.",
            "Bad.",
        ]

        for response in responses:
            analyzer.analyze_response(response, agent, context)

        metrics = analyzer.get_metrics()

        # Verify aggregated metrics
        assert metrics["total_responses"] == 4
        assert "avg_quality_score" in metrics
        assert metrics["avg_quality_score"] > 0
        assert "quality_distribution" in metrics
        assert isinstance(metrics["quality_distribution"], dict)


class TestTemplatePromptBuilder:
    """Test suite for TemplatePromptBuilder."""

    def test_prompt_builder_initialization(self):
        """Test prompt builder initializes with role templates."""
        builder = TemplatePromptBuilder()

        assert len(builder.role_templates) > 0
        assert AgentPersonaType.EXPLORER in builder.role_templates
        assert AgentPersonaType.ANALYST in builder.role_templates
        assert AgentPersonaType.CRITIC in builder.role_templates
        assert AgentPersonaType.DEFAULT in builder.role_templates

    def test_persona_type_detection(self):
        """Test persona type detection from agent role descriptions."""
        builder = TemplatePromptBuilder()

        # Test explorer detection
        explorer_agent = AgentRole(
            agent_id="explorer-1",
            name="Explorer",
            role="explorer of new ideas",
            system_prompt="You love to explore and discover things.",
        )
        persona_type = builder._determine_persona_type(explorer_agent)
        assert persona_type == AgentPersonaType.EXPLORER

        # Test analyst detection
        analyst_agent = AgentRole(
            agent_id="analyst-1",
            name="DataAnalyst",
            role="data analyst",
            system_prompt="You analyze data and provide insights.",
        )
        persona_type = builder._determine_persona_type(analyst_agent)
        assert persona_type == AgentPersonaType.ANALYST

        # Test default fallback
        generic_agent = AgentRole(
            agent_id="generic-1",
            name="GenericAgent",
            role="conversation participant",
            system_prompt="You participate in conversations.",
        )
        persona_type = builder._determine_persona_type(generic_agent)
        assert persona_type == AgentPersonaType.DEFAULT

    def test_build_prompt_formatting(self):
        """Test prompt building with proper template formatting."""
        builder = TemplatePromptBuilder()

        agent = AgentRole(
            agent_id="test-explorer",
            name="TestExplorer",
            role="explorer of mysteries",
            system_prompt="You explore mysterious phenomena.",
        )

        # Create conversation context with some history
        turn1 = ConversationTurnDomain(
            turn_number=1,
            agent_id="other-agent",
            agent_name="OtherAgent",
            prompt="test prompt",
        )
        turn1.response = "This is an interesting mystery to investigate."
        turn1.status = TurnStatus.COMPLETED

        context = ConversationContext(
            conversation_id="test-conv",
            topic="Mysterious phenomena investigation",
            participants=[agent],
            turn_history=[turn1],
        )

        prompt = builder.build_prompt(agent, context)

        # Verify template formatting
        assert "TestExplorer" in prompt
        assert "explorer of mysteries" in prompt
        assert "You explore mysterious phenomena." in prompt
        assert "Mysterious phenomena investigation" in prompt
        assert "curiosity and exploration" in prompt  # Explorer-specific template content

    def test_different_persona_templates(self):
        """Test that different persona types generate different prompts."""
        builder = TemplatePromptBuilder()

        context = ConversationContext(
            conversation_id="test-conv",
            topic="Test discussion",
            participants=[],
        )

        # Create agents with different personas
        explorer = AgentRole(
            agent_id="explorer-1",
            name="Explorer",
            role="explorer",
            system_prompt="Explore things.",
        )

        critic = AgentRole(
            agent_id="critic-1",
            name="Critic",
            role="critic",
            system_prompt="Challenge assumptions.",
        )

        # Generate prompts
        explorer_prompt = builder.build_prompt(explorer, context)
        critic_prompt = builder.build_prompt(critic, context)

        # Verify different templates used
        assert "curiosity and exploration" in explorer_prompt
        assert "challenge assumptions" in critic_prompt
        assert explorer_prompt != critic_prompt


class TestAgentResponseGenerator:
    """Test suite for AgentResponseGenerator main functionality."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service for testing."""
        mock_service = Mock(spec=LLMService)
        mock_service.generate_conversation_response = AsyncMock()
        return mock_service

    @pytest.fixture
    def test_agent(self):
        """Create test agent for testing."""
        return AgentRole(
            agent_id="test-agent",
            name="TestAgent",
            role="test participant",
            system_prompt="You participate in test conversations.",
        )

    @pytest.fixture
    def test_context(self, test_agent):
        """Create test conversation context."""
        return ConversationContext(
            conversation_id="test-conversation",
            topic="Test conversation topic",
            participants=[test_agent],
        )

    def test_generator_initialization(self, mock_llm_service):
        """Test generator initializes with proper dependencies."""
        generator = AgentResponseGenerator(mock_llm_service)

        assert generator.llm_service == mock_llm_service
        assert generator.prompt_builder is not None
        assert generator.response_analyzer is not None
        assert generator.metrics["requests_total"] == 0
        assert len(generator._response_cache) == 0

    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_llm_service, test_agent, test_context):
        """Test successful response generation flow."""
        # Setup mock response
        mock_llm_service.generate_conversation_response.return_value = (
            "This is a test response from the agent."
        )

        generator = AgentResponseGenerator(mock_llm_service)

        # Generate response
        response = await generator.generate_response(test_agent, test_context)

        # Verify response
        assert response == "This is a test response from the agent."
        assert generator.metrics["requests_total"] == 1
        assert generator.metrics["requests_successful"] == 1

        # Verify LLM service called
        mock_llm_service.generate_conversation_response.assert_called_once()

        # Verify call arguments
        call_args = mock_llm_service.generate_conversation_response.call_args
        assert "system_prompt" in call_args.kwargs
        assert "TestAgent" in call_args.kwargs["system_prompt"]

    @pytest.mark.asyncio
    async def test_generate_response_with_caching(self, mock_llm_service, test_agent, test_context):
        """Test response caching functionality."""
        # Use a longer, higher quality response to avoid retry logic
        mock_llm_service.generate_conversation_response.return_value = "This is a high quality cached response with good detail and proper length for testing purposes."

        generator = AgentResponseGenerator(mock_llm_service)

        # Generate response twice
        response1 = await generator.generate_response(test_agent, test_context)
        response2 = await generator.generate_response(test_agent, test_context)

        # Both responses should be identical
        assert response1 == response2
        assert (
            response1
            == "This is a high quality cached response with good detail and proper length for testing purposes."
        )

        # LLM service should only be called once (second is cached)
        assert mock_llm_service.generate_conversation_response.call_count == 1
        assert generator.metrics["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_generate_response_timeout_fallback(
        self, mock_llm_service, test_agent, test_context
    ):
        """Test timeout handling with fallback response."""
        # Make LLM service timeout
        mock_llm_service.generate_conversation_response.side_effect = asyncio.TimeoutError()

        generator = AgentResponseGenerator(mock_llm_service)

        # Generate response with short timeout
        response = await generator.generate_response(test_agent, test_context, timeout_seconds=1)

        # Should get fallback response
        assert response is not None
        assert len(response) > 0
        assert "TestAgent" in response  # Fallback includes agent name
        assert generator.metrics["requests_failed"] == 1
        assert generator.metrics["fallback_used"] == 1

    @pytest.mark.asyncio
    async def test_generate_response_llm_error_fallback(
        self, mock_llm_service, test_agent, test_context
    ):
        """Test LLM error handling with fallback response."""
        # Make LLM service raise error
        mock_llm_service.generate_conversation_response.side_effect = Exception("LLM service error")

        generator = AgentResponseGenerator(mock_llm_service)

        # Generate response
        response = await generator.generate_response(test_agent, test_context)

        # Should get fallback response
        assert response is not None
        assert len(response) > 0
        assert generator.metrics["requests_failed"] == 1
        assert generator.metrics["fallback_used"] == 1

    @pytest.mark.asyncio
    async def test_generate_response_poor_quality_retry(
        self, mock_llm_service, test_agent, test_context
    ):
        """Test poor quality response triggers retry logic."""
        # First call returns poor quality response, second call returns better response
        mock_llm_service.generate_conversation_response.side_effect = [
            "Bad.",  # Poor quality (too short)
            "This is a much better quality response with proper content and detail.",
        ]

        generator = AgentResponseGenerator(mock_llm_service)

        # Generate response
        response = await generator.generate_response(test_agent, test_context)

        # Should get the retry response (better quality)
        assert response == "This is a much better quality response with proper content and detail."
        assert mock_llm_service.generate_conversation_response.call_count == 2
        assert generator.metrics["quality_failures"] == 1

    def test_input_validation(self, mock_llm_service):
        """Test input validation for generate_response."""
        generator = AgentResponseGenerator(mock_llm_service)

        # Test agent with valid fields but use manual validation check
        # Since Pydantic prevents creating empty name agents, we test the validator directly
        valid_agent = AgentRole(
            agent_id="valid",
            name="ValidAgent",
            role="test role",
            system_prompt="This is a valid system prompt for testing.",
        )

        # Manually test empty name validation by modifying after creation
        valid_agent.name = ""

        context = ConversationContext(
            conversation_id="test-conv",
            topic="test topic",
            participants=[valid_agent],
        )

        # Should raise ValueError during our validation
        with pytest.raises(ValueError, match="Agent name cannot be empty"):
            asyncio.run(generator.generate_response(valid_agent, context))

        # Test empty conversation topic
        valid_agent = AgentRole(
            agent_id="valid",
            name="ValidAgent",
            role="test role",
            system_prompt="test prompt",
        )

        invalid_context = ConversationContext(
            conversation_id="test-conv",
            topic="",  # Empty topic
            participants=[valid_agent],
        )

        with pytest.raises(ValueError, match="Conversation context must have valid topic"):
            asyncio.run(generator.generate_response(valid_agent, invalid_context))

    def test_get_metrics(self, mock_llm_service):
        """Test metrics collection and computation."""
        generator = AgentResponseGenerator(mock_llm_service)

        # Initial metrics
        metrics = generator.get_metrics()
        assert metrics["requests_total"] == 0
        assert metrics["success_rate"] == 0.0
        assert metrics["cache_hit_rate"] == 0.0

        # Update some metrics manually for testing
        generator.metrics["requests_total"] = 10
        generator.metrics["requests_successful"] = 8
        generator.metrics["requests_failed"] = 2
        generator.metrics["cache_hits"] = 3
        generator.metrics["fallback_used"] = 1

        metrics = generator.get_metrics()
        assert metrics["success_rate"] == 0.8
        assert metrics["failure_rate"] == 0.2
        assert metrics["cache_hit_rate"] == 0.3
        assert metrics["fallback_rate"] == 0.1
        assert "response_analysis" in metrics

    def test_cache_management(self, mock_llm_service):
        """Test response cache size management."""
        generator = AgentResponseGenerator(mock_llm_service)
        generator._cache_max_size = 3  # Set small cache for testing

        # Fill cache beyond limit
        for i in range(5):
            cache_key = f"test-key-{i}"
            response = f"test-response-{i}"
            generator._cache_response(cache_key, response)

        # Cache should not exceed max size
        assert len(generator._response_cache) == 3

        # Should contain the last 3 entries
        assert "test-key-2" in generator._response_cache
        assert "test-key-3" in generator._response_cache
        assert "test-key-4" in generator._response_cache
        assert "test-key-0" not in generator._response_cache
        assert "test-key-1" not in generator._response_cache

    def test_clear_cache(self, mock_llm_service):
        """Test cache clearing functionality."""
        generator = AgentResponseGenerator(mock_llm_service)

        # Add some cache entries
        generator._cache_response("key1", "response1")
        generator._cache_response("key2", "response2")
        assert len(generator._response_cache) == 2

        # Clear cache
        generator.clear_cache()
        assert len(generator._response_cache) == 0


class TestFactoryFunction:
    """Test suite for factory function."""

    def test_create_agent_response_generator(self):
        """Test factory function creates properly configured generator."""
        mock_llm_service = Mock(spec=LLMService)

        generator = create_agent_response_generator(mock_llm_service)

        assert isinstance(generator, AgentResponseGenerator)
        assert generator.llm_service == mock_llm_service
        assert isinstance(generator.prompt_builder, TemplatePromptBuilder)
        assert isinstance(generator.response_analyzer, ResponseAnalyzer)


class TestIntegrationScenarios:
    """Integration tests for realistic conversation scenarios."""

    @pytest.mark.asyncio
    async def test_multi_agent_conversation_scenario(self):
        """Test realistic multi-agent conversation scenario."""
        # Create mock LLM service with different responses per agent
        mock_llm_service = Mock(spec=LLMService)

        def mock_generate_response(system_prompt, context_messages, user_id, **kwargs):
            if "Explorer" in system_prompt:
                return "I'm curious about this topic! What hidden aspects should we investigate?"
            elif "Analyst" in system_prompt:
                return "Let me analyze the data here. I see several important patterns emerging."
            elif "Critic" in system_prompt:
                return (
                    "I have concerns about this approach. Have we considered the potential risks?"
                )
            else:
                return "This is an interesting discussion point worth exploring further."

        mock_llm_service.generate_conversation_response = AsyncMock(
            side_effect=mock_generate_response
        )

        generator = AgentResponseGenerator(mock_llm_service)

        # Create different agent types
        explorer = AgentRole(
            agent_id="explorer-1",
            name="Explorer",
            role="explorer of new ideas",
            system_prompt="You explore and investigate new concepts.",
        )

        analyst = AgentRole(
            agent_id="analyst-1",
            name="DataAnalyst",
            role="data analyst",
            system_prompt="You analyze information and identify patterns.",
        )

        critic = AgentRole(
            agent_id="critic-1",
            name="CriticalThinker",
            role="critical thinker",
            system_prompt="You challenge assumptions and identify potential issues.",
        )

        # Create conversation context
        context = ConversationContext(
            conversation_id="multi-agent-test",
            topic="Artificial Intelligence Ethics",
            participants=[explorer, analyst, critic],
        )

        # Generate responses from each agent
        explorer_response = await generator.generate_response(explorer, context)
        analyst_response = await generator.generate_response(analyst, context)
        critic_response = await generator.generate_response(critic, context)

        # Verify responses reflect agent personalities
        assert "curious" in explorer_response.lower() or "investigate" in explorer_response.lower()
        assert "analyze" in analyst_response.lower() or "patterns" in analyst_response.lower()
        assert "concerns" in critic_response.lower() or "risks" in critic_response.lower()

        # Verify metrics
        metrics = generator.get_metrics()
        assert metrics["requests_total"] == 3
        assert metrics["requests_successful"] == 3

    @pytest.mark.asyncio
    async def test_conversation_with_history(self):
        """Test response generation with conversation history context."""
        mock_llm_service = Mock(spec=LLMService)
        mock_llm_service.generate_conversation_response = AsyncMock(
            return_value="Building on the previous discussion, I think we should consider..."
        )

        generator = AgentResponseGenerator(mock_llm_service)

        agent = AgentRole(
            agent_id="test-agent",
            name="TestAgent",
            role="conversation participant",
            system_prompt="You participate thoughtfully in conversations.",
        )

        # Create conversation context with history
        turn1 = ConversationTurnDomain(
            turn_number=1,
            agent_id="other-agent",
            agent_name="OtherAgent",
            prompt="initial prompt",
        )
        turn1.response = "This is the first response in our conversation."
        turn1.status = TurnStatus.COMPLETED

        turn2 = ConversationTurnDomain(
            turn_number=2,
            agent_id="another-agent",
            agent_name="AnotherAgent",
            prompt="follow-up prompt",
        )
        turn2.response = "Here's my follow-up thought on that initial point."
        turn2.status = TurnStatus.COMPLETED

        context = ConversationContext(
            conversation_id="history-test",
            topic="Discussion with history",
            participants=[agent],
            turn_history=[turn1, turn2],
        )

        # Generate response
        response = await generator.generate_response(agent, context)

        # Verify response generated
        assert response == "Building on the previous discussion, I think we should consider..."

        # Verify LLM service called with context
        call_args = mock_llm_service.generate_conversation_response.call_args
        assert "context_messages" in call_args.kwargs
        context_messages = call_args.kwargs["context_messages"]

        # Should include previous turns as context
        assert len(context_messages) == 2
        assert "OtherAgent" in context_messages[0]["content"]
        assert "AnotherAgent" in context_messages[1]["content"]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

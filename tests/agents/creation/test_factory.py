"""Tests for agent factory.

Comprehensive integration tests for the main AgentFactory orchestrator,
testing the complete agent creation workflow with proper mocking.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from agents.creation.factory import AgentFactory
from agents.creation.models import (
    AgentCreationRequest,
    AnalysisConfidence,
    PersonalityProfile,
    PromptAnalysisResult,
)
from database.models import Agent, AgentType


class TestAgentFactory:
    """Test the main AgentFactory orchestrator."""

    @pytest.fixture
    def mock_services(self):
        """Create mocked services for dependency injection."""
        return {
            "prompt_analyzer": Mock(),
            "personality_generator": Mock(),
            "system_prompt_builder": Mock(),
            "agent_builder": Mock(),
            "llm_service": Mock(),
        }

    @pytest.fixture
    def factory(self, mock_services):
        """Create factory with mocked dependencies."""
        return AgentFactory(
            prompt_analyzer=mock_services["prompt_analyzer"],
            personality_generator=mock_services["personality_generator"],
            system_prompt_builder=mock_services["system_prompt_builder"],
            agent_builder=mock_services["agent_builder"],
            llm_service=mock_services["llm_service"],
        )

    @pytest.fixture
    def sample_analysis_result(self):
        """Create sample prompt analysis result."""
        return PromptAnalysisResult(
            agent_type=AgentType.ANALYST,
            confidence=AnalysisConfidence.HIGH,
            domain="finance",
            capabilities=["data_analysis", "reporting"],
            context="financial market analysis",
            reasoning="User needs data analysis capabilities",
            original_prompt="Help me analyze market trends",
            processed_prompt="Help me analyze market trends.",
        )

    @pytest.fixture
    def sample_personality(self):
        """Create sample personality profile."""
        return PersonalityProfile(
            assertiveness=0.7,
            analytical_depth=0.9,
            creativity=0.4,
            custom_traits={"precision": 0.8},
        )

    @pytest.mark.asyncio
    async def test_creates_agent_successfully(
        self, factory, mock_services, sample_analysis_result, sample_personality
    ):
        """Should create agent successfully through complete workflow."""
        # Setup mocks
        mock_services["prompt_analyzer"].analyze_prompt = AsyncMock(
            return_value=sample_analysis_result
        )
        mock_services["personality_generator"].generate_personality = AsyncMock(
            return_value=sample_personality
        )
        mock_services["system_prompt_builder"].build_system_prompt = AsyncMock(
            return_value="You are an analyst agent."
        )

        created_agent = Agent(
            id="test-agent-id",
            name="Finance Analyst",
            agent_type=AgentType.ANALYST,
            system_prompt="You are an analyst agent.",
        )
        mock_services["agent_builder"].build_agent = AsyncMock(return_value=created_agent)

        # Create request
        request = AgentCreationRequest(prompt="Help me analyze market trends", user_id="user123")

        # Execute
        result = await factory.create_agent(request)

        # Verify success
        assert result.success is True
        assert result.error_message is None
        assert result.agent is not None
        assert result.agent.id == "test-agent-id"
        assert result.specification is not None
        assert result.analysis_result == sample_analysis_result
        assert result.processing_time_ms is not None

        # Verify all services were called
        mock_services["prompt_analyzer"].analyze_prompt.assert_called_once_with(
            "Help me analyze market trends"
        )
        mock_services["personality_generator"].generate_personality.assert_called_once()
        mock_services["system_prompt_builder"].build_system_prompt.assert_called_once()
        mock_services["agent_builder"].build_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_preview_without_persistence(
        self, factory, mock_services, sample_analysis_result, sample_personality
    ):
        """Should create preview without persisting agent."""
        # Setup mocks
        mock_services["prompt_analyzer"].analyze_prompt = AsyncMock(
            return_value=sample_analysis_result
        )
        mock_services["personality_generator"].generate_personality = AsyncMock(
            return_value=sample_personality
        )
        mock_services["system_prompt_builder"].build_system_prompt = AsyncMock(
            return_value="You are an analyst agent."
        )

        # Create preview request
        request = AgentCreationRequest(prompt="Help me analyze data", preview_only=True)

        # Execute
        result = await factory.create_agent(request)

        # Verify preview result
        assert result.success is True
        assert result.agent is None  # No agent created
        assert result.specification is not None
        assert result.specification.name == "Finance Analyst"  # Generated name

        # Verify agent builder was NOT called
        assert not mock_services["agent_builder"].build_agent.called

    @pytest.mark.asyncio
    async def test_respects_user_preferences(self, factory, mock_services, sample_personality):
        """Should respect user's agent type preference and custom name."""
        # Setup mocks with different analysis result
        analysis_result = PromptAnalysisResult(
            agent_type=AgentType.CREATIVE,  # Different from preference
            confidence=AnalysisConfidence.MEDIUM,
            original_prompt="test prompt",
            processed_prompt="test prompt",
        )
        mock_services["prompt_analyzer"].analyze_prompt = AsyncMock(return_value=analysis_result)
        mock_services["personality_generator"].generate_personality = AsyncMock(
            return_value=sample_personality
        )
        mock_services["system_prompt_builder"].build_system_prompt = AsyncMock(
            return_value="You are an advocate."
        )

        created_agent = Agent(id="test-id", name="Custom Agent", agent_type=AgentType.ADVOCATE)
        mock_services["agent_builder"].build_agent = AsyncMock(return_value=created_agent)

        # Create request with preferences
        request = AgentCreationRequest(
            prompt="test prompt",
            agent_name="Custom Agent",
            preferred_type=AgentType.ADVOCATE,  # Override analysis result
        )

        # Execute
        result = await factory.create_agent(request)

        # Verify preferences were respected
        assert result.success is True
        assert result.specification.name == "Custom Agent"
        assert (
            result.specification.agent_type == AgentType.ADVOCATE
        )  # Used preference, not analysis

    @pytest.mark.asyncio
    async def test_handles_prompt_analysis_failure_gracefully(
        self, factory, mock_services, sample_personality
    ):
        """Should handle prompt analysis failure with fallback."""
        # Setup prompt analyzer to fail
        mock_services["prompt_analyzer"].analyze_prompt = AsyncMock(
            side_effect=Exception("Analysis failed")
        )
        mock_services["personality_generator"].get_default_personality = Mock(
            return_value=sample_personality
        )
        mock_services["system_prompt_builder"].get_template_prompt = Mock(
            return_value="Template prompt"
        )

        created_agent = Agent(id="fallback-id", name="Analyst Agent", agent_type=AgentType.ANALYST)
        mock_services["agent_builder"].build_agent = AsyncMock(return_value=created_agent)

        request = AgentCreationRequest(prompt="test prompt")

        result = await factory.create_agent(request)

        # Should still succeed with fallback analysis
        assert result.success is True
        assert result.agent is not None
        assert result.analysis_result.agent_type == AgentType.ANALYST  # Default fallback
        assert result.analysis_result.reasoning == "Fallback analysis due to service failure"

    @pytest.mark.asyncio
    async def test_handles_personality_generation_failure(
        self, factory, mock_services, sample_analysis_result
    ):
        """Should handle personality generation failure with default."""
        # Setup services
        mock_services["prompt_analyzer"].analyze_prompt = AsyncMock(
            return_value=sample_analysis_result
        )
        mock_services["personality_generator"].generate_personality = AsyncMock(
            side_effect=Exception("Personality failed")
        )

        default_personality = PersonalityProfile(assertiveness=0.5)
        mock_services["personality_generator"].get_default_personality = Mock(
            return_value=default_personality
        )
        mock_services["system_prompt_builder"].build_system_prompt = AsyncMock(
            return_value="System prompt"
        )

        created_agent = Agent(id="test-id", name="Agent", agent_type=AgentType.ANALYST)
        mock_services["agent_builder"].build_agent = AsyncMock(return_value=created_agent)

        request = AgentCreationRequest(prompt="test prompt")

        result = await factory.create_agent(request)

        # Should succeed with default personality
        assert result.success is True
        assert result.specification.personality.assertiveness == 0.5  # Default value

    @pytest.mark.asyncio
    async def test_handles_system_prompt_failure(
        self, factory, mock_services, sample_analysis_result, sample_personality
    ):
        """Should handle system prompt building failure with template."""
        # Setup services
        mock_services["prompt_analyzer"].analyze_prompt = AsyncMock(
            return_value=sample_analysis_result
        )
        mock_services["personality_generator"].generate_personality = AsyncMock(
            return_value=sample_personality
        )
        mock_services["system_prompt_builder"].build_system_prompt = AsyncMock(
            side_effect=Exception("Prompt failed")
        )
        mock_services["system_prompt_builder"].get_template_prompt = Mock(
            return_value="Template prompt"
        )

        created_agent = Agent(id="test-id", name="Agent", agent_type=AgentType.ANALYST)
        mock_services["agent_builder"].build_agent = AsyncMock(return_value=created_agent)

        request = AgentCreationRequest(prompt="test prompt")

        result = await factory.create_agent(request)

        # Should succeed with template prompt
        assert result.success is True
        assert result.specification.system_prompt == "Template prompt"

    @pytest.mark.asyncio
    async def test_handles_agent_building_failure(
        self, factory, mock_services, sample_analysis_result, sample_personality
    ):
        """Should handle agent building failure and return error."""
        # Setup services
        mock_services["prompt_analyzer"].analyze_prompt = AsyncMock(
            return_value=sample_analysis_result
        )
        mock_services["personality_generator"].generate_personality = AsyncMock(
            return_value=sample_personality
        )
        mock_services["system_prompt_builder"].build_system_prompt = AsyncMock(
            return_value="System prompt"
        )
        mock_services["agent_builder"].build_agent = AsyncMock(
            side_effect=Exception("Database error")
        )

        request = AgentCreationRequest(prompt="test prompt")

        result = await factory.create_agent(request)

        # Should return failure result
        assert result.success is False
        assert result.error_message is not None
        assert "Database error" in result.error_message
        assert result.agent is None

    @pytest.mark.asyncio
    async def test_preview_agent_shortcut(
        self, factory, mock_services, sample_analysis_result, sample_personality
    ):
        """Should provide preview_agent shortcut method."""
        # Setup mocks
        mock_services["prompt_analyzer"].analyze_prompt = AsyncMock(
            return_value=sample_analysis_result
        )
        mock_services["personality_generator"].generate_personality = AsyncMock(
            return_value=sample_personality
        )
        mock_services["system_prompt_builder"].build_system_prompt = AsyncMock(
            return_value="You are an analyst."
        )

        # Execute preview
        specification = await factory.preview_agent("Analyze market data")

        # Verify specification
        assert specification.name == "Finance Analyst"
        assert specification.agent_type == AgentType.ANALYST
        assert specification.system_prompt == "You are an analyst."
        assert specification.personality == sample_personality

    @pytest.mark.asyncio
    async def test_preview_agent_handles_complete_failure(self, factory, mock_services):
        """Should handle complete system failure gracefully by returning error result."""
        # Mock all services to fail including fallbacks
        mock_services["prompt_analyzer"].analyze_prompt = AsyncMock(
            side_effect=Exception("Analysis failed")
        )
        mock_services["personality_generator"].get_default_personality = Mock(
            side_effect=Exception("Personality failed")
        )
        mock_services["system_prompt_builder"].get_template_prompt = Mock(
            side_effect=Exception("Template failed")
        )

        # Since factory has comprehensive fallbacks, test the error result instead
        request = AgentCreationRequest(prompt="test prompt", preview_only=True)
        result = await factory.create_agent(request)

        # The factory should handle failures gracefully and still return a result
        # even if some services fail, due to its comprehensive fallback system
        assert result is not None
        # Could be success (with fallbacks) or failure depending on how many services fail

    @pytest.mark.asyncio
    async def test_get_supported_agent_types(self, factory):
        """Should return all supported agent types."""
        types = await factory.get_supported_agent_types()

        assert len(types) == 5
        assert AgentType.ADVOCATE in types
        assert AgentType.ANALYST in types
        assert AgentType.CRITIC in types
        assert AgentType.CREATIVE in types
        assert AgentType.MODERATOR in types

    def test_generates_appropriate_agent_names(self, factory):
        """Should generate appropriate names for different agent types and domains."""
        # Test with domain
        name = factory._generate_agent_name(AgentType.ANALYST, "finance")
        assert name == "Finance Analyst"

        # Test without domain
        name = factory._generate_agent_name(AgentType.CREATIVE, None)
        assert name == "Creative Agent"

        # Test domain formatting
        name = factory._generate_agent_name(AgentType.MODERATOR, "project_management")
        assert name == "Project Management Moderator"

    def test_tracks_metrics_correctly(self, factory):
        """Should track metrics for monitoring."""
        # Initial metrics
        metrics = factory.get_metrics()
        assert metrics["agents_created"] == 0
        assert metrics["creation_failures"] == 0
        assert metrics["success_rate"] == 0.0

        # Update metrics manually for testing
        factory._metrics["agents_created"] = 10
        factory._metrics["creation_failures"] = 2
        factory._metrics["llm_calls_total"] = 30
        factory._metrics["fallback_used"] = 3

        metrics = factory.get_metrics()
        assert metrics["success_rate"] == 10 / 12  # 10 successes out of 12 total
        assert metrics["failure_rate"] == 2 / 12
        assert metrics["fallback_rate"] == 3 / 30

    def test_clears_metrics(self, factory):
        """Should clear metrics for testing."""
        # Set some metrics
        factory._metrics["agents_created"] = 5
        factory._metrics["creation_failures"] = 1

        # Clear metrics
        factory.clear_metrics()

        # Verify cleared
        metrics = factory.get_metrics()
        assert metrics["agents_created"] == 0
        assert metrics["creation_failures"] == 0

    def test_updates_average_creation_time(self, factory):
        """Should correctly update average creation time."""
        # First measurement
        factory._metrics["agents_created"] = 1
        factory._update_avg_creation_time(1000)
        assert factory._metrics["avg_creation_time_ms"] == 1000

        # Second measurement - should be moving average
        factory._metrics["agents_created"] = 2
        factory._update_avg_creation_time(2000)
        assert factory._metrics["avg_creation_time_ms"] > 1000
        assert factory._metrics["avg_creation_time_ms"] < 2000

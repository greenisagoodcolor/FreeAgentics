"""Tests for agent creation models.

Tests all data models, value objects, and DTOs used in the agent creation system.
Following Kent Beck's TDD principles with comprehensive behavior testing.
"""

from datetime import datetime

from agents.creation.models import (
    AgentCreationError,
    AgentCreationRequest,
    AgentCreationResult,
    AgentSpecification,
    AnalysisConfidence,
    PersonalityProfile,
    PromptAnalysisError,
    PromptAnalysisResult,
)
from database.models import AgentType


class TestPersonalityProfile:
    """Test PersonalityProfile value object."""

    def test_creates_default_personality_profile(self):
        """Should create personality profile with default values."""
        personality = PersonalityProfile()

        assert personality.assertiveness == 0.5
        assert personality.analytical_depth == 0.5
        assert personality.creativity == 0.5
        assert personality.empathy == 0.5
        assert personality.skepticism == 0.5
        assert personality.formality == 0.5
        assert personality.verbosity == 0.5
        assert personality.collaboration == 0.5
        assert personality.speed == 0.5
        assert personality.custom_traits == {}

    def test_creates_personality_with_custom_traits(self):
        """Should create personality profile with custom values."""
        personality = PersonalityProfile(
            assertiveness=0.8, creativity=0.9, custom_traits={"enthusiasm": 0.7}
        )

        assert personality.assertiveness == 0.8
        assert personality.creativity == 0.9
        assert personality.custom_traits["enthusiasm"] == 0.7

    def test_converts_to_dict(self):
        """Should convert personality profile to dictionary."""
        personality = PersonalityProfile(
            assertiveness=0.8, creativity=0.6, custom_traits={"test": 0.5}
        )

        result = personality.to_dict()

        assert result["assertiveness"] == 0.8
        assert result["creativity"] == 0.6
        assert result["custom_traits"]["test"] == 0.5
        assert "empathy" in result

    def test_creates_from_dict(self):
        """Should create personality profile from dictionary."""
        data = {"assertiveness": 0.7, "creativity": 0.9, "custom_traits": {"test": 0.8}}

        personality = PersonalityProfile.from_dict(data)

        assert personality.assertiveness == 0.7
        assert personality.creativity == 0.9
        assert personality.empathy == 0.5  # Default value
        assert personality.custom_traits["test"] == 0.8

    def test_handles_missing_fields_in_dict(self):
        """Should handle missing fields when creating from dict."""
        data = {"assertiveness": 0.8}

        personality = PersonalityProfile.from_dict(data)

        assert personality.assertiveness == 0.8
        assert personality.empathy == 0.5  # Default
        assert personality.custom_traits == {}


class TestPromptAnalysisResult:
    """Test PromptAnalysisResult data structure."""

    def test_creates_basic_analysis_result(self):
        """Should create analysis result with required fields."""
        result = PromptAnalysisResult(
            agent_type=AgentType.ANALYST, confidence=AnalysisConfidence.HIGH
        )

        assert result.agent_type == AgentType.ANALYST
        assert result.confidence == AnalysisConfidence.HIGH
        assert result.capabilities == []
        assert result.alternative_types == []

    def test_creates_detailed_analysis_result(self):
        """Should create analysis result with all optional fields."""
        result = PromptAnalysisResult(
            agent_type=AgentType.CREATIVE,
            confidence=AnalysisConfidence.MEDIUM,
            domain="marketing",
            capabilities=["brainstorming", "ideation"],
            context="creative campaign development",
            reasoning="User needs innovative marketing ideas",
            alternative_types=[AgentType.ANALYST, AgentType.ADVOCATE],
            original_prompt="Help me create innovative marketing campaigns",
            processed_prompt="Help me create innovative marketing campaigns.",
        )

        assert result.agent_type == AgentType.CREATIVE
        assert result.domain == "marketing"
        assert "brainstorming" in result.capabilities
        assert AgentType.ANALYST in result.alternative_types

    def test_converts_to_dict(self):
        """Should convert analysis result to dictionary."""
        result = PromptAnalysisResult(
            agent_type=AgentType.ADVOCATE,
            confidence=AnalysisConfidence.HIGH,
            capabilities=["persuasion"],
            alternative_types=[AgentType.ANALYST],
        )

        data = result.to_dict()

        assert data["agent_type"] == "advocate"
        assert data["confidence"] == "high"
        assert data["capabilities"] == ["persuasion"]
        assert data["alternative_types"] == ["analyst"]


class TestAgentSpecification:
    """Test AgentSpecification aggregate."""

    def test_creates_basic_specification(self):
        """Should create agent specification with required fields."""
        personality = PersonalityProfile(assertiveness=0.8)

        spec = AgentSpecification(
            name="Test Agent",
            agent_type=AgentType.CRITIC,
            system_prompt="You are a critic agent.",
            personality=personality,
            source_prompt="I need help finding flaws in proposals",
        )

        assert spec.name == "Test Agent"
        assert spec.agent_type == AgentType.CRITIC
        assert spec.system_prompt == "You are a critic agent."
        assert spec.personality.assertiveness == 0.8
        assert spec.creation_source == "ai_generated"

    def test_converts_to_dict(self):
        """Should convert specification to dictionary."""
        personality = PersonalityProfile(creativity=0.9)

        spec = AgentSpecification(
            name="Creative Agent",
            agent_type=AgentType.CREATIVE,
            system_prompt="Be creative",
            personality=personality,
            source_prompt="Help brainstorm",
            capabilities=["ideation", "brainstorming"],
        )

        data = spec.to_dict()

        assert data["name"] == "Creative Agent"
        assert data["agent_type"] == "creative"
        assert data["personality"]["creativity"] == 0.9
        assert data["capabilities"] == ["ideation", "brainstorming"]


class TestAgentCreationRequest:
    """Test AgentCreationRequest DTO."""

    def test_creates_basic_request(self):
        """Should create request with minimal required fields."""
        request = AgentCreationRequest(prompt="Help me analyze data")

        assert request.prompt == "Help me analyze data"
        assert request.user_id is None
        assert request.preview_only is False
        assert request.enable_advanced_personality is True

    def test_creates_detailed_request(self):
        """Should create request with all optional fields."""
        request = AgentCreationRequest(
            prompt="Create marketing campaigns",
            user_id="user123",
            agent_name="Campaign Creator",
            preferred_type=AgentType.CREATIVE,
            enable_advanced_personality=False,
            preview_only=True,
        )

        assert request.user_id == "user123"
        assert request.agent_name == "Campaign Creator"
        assert request.preferred_type == AgentType.CREATIVE
        assert request.preview_only is True

    def test_converts_to_dict(self):
        """Should convert request to dictionary."""
        request = AgentCreationRequest(
            prompt="Test prompt", preferred_type=AgentType.MODERATOR, preview_only=True
        )

        data = request.to_dict()

        assert data["prompt"] == "Test prompt"
        assert data["preferred_type"] == "moderator"
        assert data["preview_only"] is True


class TestAgentCreationResult:
    """Test AgentCreationResult DTO."""

    def test_creates_successful_result(self):
        """Should create successful result with default values."""
        result = AgentCreationResult()

        assert result.success is True
        assert result.error_message is None
        assert result.llm_calls_made == 0
        assert result.tokens_used == 0
        assert isinstance(result.created_at, datetime)

    def test_creates_failed_result(self):
        """Should create failed result with error information."""
        result = AgentCreationResult(
            success=False, error_message="LLM service unavailable", processing_time_ms=1500
        )

        assert result.success is False
        assert result.error_message == "LLM service unavailable"
        assert result.processing_time_ms == 1500

    def test_converts_to_dict(self):
        """Should convert result to dictionary."""
        result = AgentCreationResult(success=True, processing_time_ms=2000, llm_calls_made=3)

        data = result.to_dict()

        assert data["success"] is True
        assert data["processing_time_ms"] == 2000
        assert data["llm_calls_made"] == 3
        assert "created_at" in data


class TestAgentCreationExceptions:
    """Test custom exceptions for agent creation."""

    def test_agent_creation_error(self):
        """Should create base agent creation error."""
        error = AgentCreationError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert isinstance(error, Exception)

    def test_prompt_analysis_error(self):
        """Should create prompt analysis specific error."""
        error = PromptAnalysisError("Invalid prompt format")

        assert str(error) == "Invalid prompt format"
        assert isinstance(error, AgentCreationError)

    def test_exception_inheritance(self):
        """Should maintain proper exception inheritance."""
        from agents.creation.models import (
            AgentBuildError,
            PersonalityGenerationError,
            SystemPromptBuildError,
        )

        assert issubclass(PersonalityGenerationError, AgentCreationError)
        assert issubclass(SystemPromptBuildError, AgentCreationError)
        assert issubclass(AgentBuildError, AgentCreationError)

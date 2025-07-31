"""Data models for agent creation system.

Defines value objects and data transfer objects for agent creation workflow.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from database.models import Agent, AgentType


class AnalysisConfidence(Enum):
    """Confidence level for prompt analysis results."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PersonalityProfile:
    """Value object representing an agent's personality traits."""

    # Core personality dimensions (0.0 to 1.0)
    assertiveness: float = 0.5  # How direct and forceful
    analytical_depth: float = 0.5  # How deep the analysis goes
    creativity: float = 0.5  # How creative and novel the thinking
    empathy: float = 0.5  # How much they consider others' perspectives
    skepticism: float = 0.5  # How much they question and doubt

    # Communication style
    formality: float = 0.5  # How formal vs casual
    verbosity: float = 0.5  # How much detail they provide

    # Working preferences
    collaboration: float = 0.5  # How much they like working with others
    speed: float = 0.5  # How quickly they work

    # Custom traits for specific agent types
    custom_traits: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "assertiveness": self.assertiveness,
            "analytical_depth": self.analytical_depth,
            "creativity": self.creativity,
            "empathy": self.empathy,
            "skepticism": self.skepticism,
            "formality": self.formality,
            "verbosity": self.verbosity,
            "collaboration": self.collaboration,
            "speed": self.speed,
            "custom_traits": self.custom_traits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityProfile":
        """Create from dictionary."""
        return cls(
            assertiveness=data.get("assertiveness", 0.5),
            analytical_depth=data.get("analytical_depth", 0.5),
            creativity=data.get("creativity", 0.5),
            empathy=data.get("empathy", 0.5),
            skepticism=data.get("skepticism", 0.5),
            formality=data.get("formality", 0.5),
            verbosity=data.get("verbosity", 0.5),
            collaboration=data.get("collaboration", 0.5),
            speed=data.get("speed", 0.5),
            custom_traits=data.get("custom_traits", {}),
        )


@dataclass
class PromptAnalysisResult:
    """Result of analyzing a natural language prompt."""

    # Primary analysis results
    agent_type: AgentType
    confidence: AnalysisConfidence

    # Extracted information
    domain: Optional[str] = None  # e.g., "finance", "healthcare", "education"
    capabilities: List[str] = field(default_factory=list)  # Required capabilities
    context: Optional[str] = None  # Additional context for agent creation

    # Analysis metadata
    reasoning: Optional[str] = None  # Why this agent type was chosen
    alternative_types: List[AgentType] = field(default_factory=list)  # Other possibilities

    # Raw prompt data
    original_prompt: str = ""
    processed_prompt: str = ""  # Cleaned/normalized version

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_type": self.agent_type.value,
            "confidence": self.confidence.value,
            "domain": self.domain,
            "capabilities": self.capabilities,
            "context": self.context,
            "reasoning": self.reasoning,
            "alternative_types": [t.value for t in self.alternative_types],
            "original_prompt": self.original_prompt,
            "processed_prompt": self.processed_prompt,
        }


@dataclass
class AgentSpecification:
    """Complete specification for creating an agent."""

    # Basic properties
    name: str
    agent_type: AgentType

    # Generated content
    system_prompt: str
    personality: PersonalityProfile

    # Creation metadata
    source_prompt: str
    creation_source: str = "ai_generated"

    # Optional configuration
    template: str = "ai_generated"
    parameters: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "agent_type": self.agent_type.value,
            "system_prompt": self.system_prompt,
            "personality": self.personality.to_dict(),
            "source_prompt": self.source_prompt,
            "creation_source": self.creation_source,
            "template": self.template,
            "parameters": self.parameters,
            "capabilities": self.capabilities,
        }


@dataclass
class AgentCreationRequest:
    """Request to create an agent from natural language."""

    # Required
    prompt: str

    # Optional parameters
    user_id: Optional[str] = None
    agent_name: Optional[str] = None  # Override generated name
    preferred_type: Optional[AgentType] = None  # Hint for agent type

    # Configuration options
    enable_advanced_personality: bool = True
    enable_custom_capabilities: bool = True

    # Preview mode - don't actually create, just return specification
    preview_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "preferred_type": self.preferred_type.value if self.preferred_type else None,
            "enable_advanced_personality": self.enable_advanced_personality,
            "enable_custom_capabilities": self.enable_custom_capabilities,
            "preview_only": self.preview_only,
        }


@dataclass
class AgentCreationResult:
    """Result of agent creation process."""

    # Created agent (None if preview_only)
    agent: Optional[Agent] = None

    # Creation process data
    specification: Optional[AgentSpecification] = None
    analysis_result: Optional[PromptAnalysisResult] = None

    # Success/failure status
    success: bool = True
    error_message: Optional[str] = None

    # Processing metadata
    processing_time_ms: Optional[int] = None
    llm_calls_made: int = 0
    tokens_used: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent": self.agent.to_dict() if self.agent else None,
            "specification": self.specification.to_dict() if self.specification else None,
            "analysis_result": self.analysis_result.to_dict() if self.analysis_result else None,
            "success": self.success,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
            "llm_calls_made": self.llm_calls_made,
            "tokens_used": self.tokens_used,
            "created_at": self.created_at.isoformat(),
        }


class AgentCreationError(Exception):
    """Base exception for agent creation errors."""

    pass


class PromptAnalysisError(AgentCreationError):
    """Error analyzing natural language prompt."""

    pass


class PersonalityGenerationError(AgentCreationError):
    """Error generating personality profile."""

    pass


class SystemPromptBuildError(AgentCreationError):
    """Error building system prompt."""

    pass


class AgentBuildError(AgentCreationError):
    """Error building or persisting agent."""

    pass

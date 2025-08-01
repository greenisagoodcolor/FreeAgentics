"""
Multi-Turn Agent Conversation Engine - Domain Models and API Models

This module provides both the domain models for conversation orchestration and
the API request/response models for the unified agent conversation service.

Architecture follows Domain-Driven Design principles with rich aggregates,
clean interfaces for dependency injection, and immutable context objects
as recommended by the Nemesis Committee.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# Domain Enums and Value Objects


class ConversationStatus(str, Enum):
    """Conversation lifecycle status."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TurnStatus(str, Enum):
    """Individual turn status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class CompletionReason(str, Enum):
    """Reasons why a conversation completed."""

    TURN_LIMIT_REACHED = "turn_limit_reached"
    CONSENSUS_REACHED = "consensus_reached"
    TASK_COMPLETED = "task_completed"
    MANUAL_STOP = "manual_stop"
    AGENT_FAILURE = "agent_failure"
    TIMEOUT = "timeout"


class AgentStatus(str, Enum):
    """Status enumeration for agent lifecycle states."""

    INITIALIZING = "initializing"
    GENERATING_GMN = "generating_gmn"
    INITIALIZING_PYMDP = "initializing_pymdp"
    READY = "ready"
    ERROR = "error"


class AgentConversationRequest(BaseModel):
    """Request model for creating agent conversations."""

    prompt: str = Field(
        ...,
        description="User prompt describing the desired agent behavior or conversation topic",
        min_length=1,
        max_length=2000,
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional configuration parameters for agent creation"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional metadata for the conversation session"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Create agents to discuss sustainable energy solutions",
                "config": {
                    "agent_count": 2,
                    "conversation_turns": 5,
                    "llm_provider": "openai",
                    "model": "gpt-3.5-turbo",
                },
                "metadata": {"session_type": "exploration", "user_intent": "research"},
            }
        }


class AgentConversationResponse(BaseModel):
    """Response model for agent conversation creation."""

    conversation_id: UUID = Field(..., description="Unique identifier for the conversation session")
    agent_id: UUID = Field(..., description="Primary agent identifier for this conversation")
    status: AgentStatus = Field(..., description="Current status of the agent/conversation")
    gmn_structure: Dict[str, Any] = Field(
        ..., description="Generated GMN (Generalized Notation) structure for the agent"
    )
    websocket_url: str = Field(..., description="WebSocket URL for real-time conversation updates")
    created_at: datetime = Field(..., description="Timestamp when the conversation was created")

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "ready",
                "gmn_structure": {
                    "name": "discussion_agent",
                    "states": ["listening", "thinking", "responding"],
                    "actions": ["listen", "respond", "question"],
                },
                "websocket_url": "ws://localhost:8000/api/v1/ws/conv_550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2024-07-31T23:45:00Z",
            }
        }


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details and context"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "AGENT_CREATION_FAILED",
                "message": "Failed to create agent from prompt",
                "details": {
                    "error_type": "llm_generation_failed",
                    "llm_provider": "openai",
                    "retry_possible": True,
                },
                "timestamp": "2024-07-31T23:45:00Z",
            }
        }


# Additional models for multi-agent conversations


class ConversationTurn(BaseModel):
    """A single turn in a multi-agent conversation."""

    turn_id: UUID = Field(..., description="Unique identifier for this turn")
    agent_id: UUID = Field(..., description="Agent that generated this turn")
    agent_name: str = Field(..., description="Name of the agent")
    content: str = Field(..., description="The agent's response content")
    timestamp: datetime = Field(..., description="When this turn was generated")
    turn_number: int = Field(..., description="Sequential turn number in conversation")

    class Config:
        json_schema_extra = {
            "example": {
                "turn_id": "550e8400-e29b-41d4-a716-446655440002",
                "agent_id": "550e8400-e29b-41d4-a716-446655440001",
                "agent_name": "Advocate",
                "content": "I believe sustainable energy is crucial for our future.",
                "timestamp": "2024-07-31T23:45:30Z",
                "turn_number": 1,
            }
        }


class MultiAgentConversationResponse(BaseModel):
    """Response model for multi-agent conversation sessions."""

    conversation_id: UUID = Field(..., description="Unique identifier for the conversation session")
    agents: List[Dict[str, Any]] = Field(
        ..., description="List of agents participating in the conversation"
    )
    turns: List[ConversationTurn] = Field(
        ..., description="All conversation turns generated so far"
    )
    status: str = Field(..., description="Current conversation status (active, completed, error)")
    total_turns: int = Field(..., description="Total number of turns in the conversation")
    websocket_url: str = Field(..., description="WebSocket URL for real-time updates")
    started_at: datetime = Field(..., description="When the conversation started")
    completed_at: Optional[datetime] = Field(
        default=None, description="When the conversation completed (if finished)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "agents": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440001",
                        "name": "Advocate",
                        "role": "advocate",
                        "status": "ready",
                    },
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440002",
                        "name": "Analyst",
                        "role": "analyst",
                        "status": "ready",
                    },
                ],
                "turns": [],
                "status": "active",
                "total_turns": 0,
                "websocket_url": "ws://localhost:8000/api/v1/ws/conv_550e8400-e29b-41d4-a716-446655440000",
                "started_at": "2024-07-31T23:45:00Z",
                "completed_at": None,
            }
        }


# Configuration models


class LLMConfig(BaseModel):
    """Configuration for LLM provider settings."""

    provider: str = Field(default="openai", description="LLM provider name")
    model: str = Field(default="gpt-3.5-turbo", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=150, ge=1, le=4000, description="Maximum tokens per response")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.8,
                "max_tokens": 200,
            }
        }


class ConversationConfig(BaseModel):
    """Configuration for conversation behavior."""

    agent_count: int = Field(default=2, ge=1, le=5, description="Number of agents to create")
    max_turns: int = Field(default=5, ge=1, le=20, description="Maximum conversation turns")
    turn_timeout_seconds: int = Field(default=30, ge=5, le=120, description="Timeout per turn")
    enable_websocket_streaming: bool = Field(default=True, description="Enable real-time streaming")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_count": 3,
                "max_turns": 8,
                "turn_timeout_seconds": 45,
                "enable_websocket_streaming": True,
            }
        }


class AdvancedAgentConversationRequest(BaseModel):
    """Advanced request model with detailed configuration options."""

    prompt: str = Field(
        ...,
        description="User prompt describing the conversation topic",
        min_length=1,
        max_length=2000,
    )
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig, description="LLM provider configuration"
    )
    conversation_config: ConversationConfig = Field(
        default_factory=ConversationConfig, description="Conversation behavior configuration"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata for the session"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Discuss the ethical implications of AI in healthcare",
                "llm_config": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.6,
                    "max_tokens": 300,
                },
                "conversation_config": {
                    "agent_count": 3,
                    "max_turns": 10,
                    "turn_timeout_seconds": 60,
                    "enable_websocket_streaming": True,
                },
                "metadata": {"topic_category": "ethics", "complexity_level": "advanced"},
            }
        }


# Domain Models for Conversation Orchestration


class ConversationConfig(BaseModel):
    """Configuration for conversation behavior (Domain Model)."""

    max_turns: int = Field(default=10, ge=1, le=50, description="Maximum number of turns")
    turn_timeout_seconds: int = Field(default=60, ge=5, le=300, description="Timeout per turn")
    conversation_timeout_minutes: int = Field(
        default=30, ge=1, le=120, description="Total conversation timeout"
    )
    allow_interruption: bool = Field(default=True, description="Allow manual pause/resume")
    enable_consensus_detection: bool = Field(
        default=False, description="Detect when agents reach consensus"
    )
    min_response_length: int = Field(default=10, ge=1, description="Minimum response length")
    max_response_length: int = Field(default=1000, ge=100, description="Maximum response length")

    class Config:
        frozen = True  # Immutable configuration


class AgentRole(BaseModel):
    """Agent role definition for conversations."""

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent display name")
    role: str = Field(..., description="Agent role (e.g., 'advocate', 'critic')")
    system_prompt: str = Field(..., description="System prompt for this agent")
    personality_traits: List[str] = Field(
        default_factory=list, description="Personality characteristics"
    )
    turn_order: int = Field(default=0, description="Order in turn sequence")

    @validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Agent name cannot be empty")
        return v.strip()

    @validator("system_prompt")
    def validate_system_prompt(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("System prompt must be at least 10 characters")
        return v.strip()


class ConversationTurnDomain(BaseModel):
    """Represents a single turn in the conversation (Domain Model)."""

    turn_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique turn identifier")
    turn_number: int = Field(..., ge=1, description="Sequential turn number")
    agent_id: str = Field(..., description="ID of agent taking this turn")
    agent_name: str = Field(..., description="Name of agent taking this turn")
    prompt: str = Field(..., description="Prompt given to the agent")
    response: Optional[str] = Field(None, description="Agent's response")
    status: TurnStatus = Field(default=TurnStatus.PENDING, description="Turn status")
    created_at: datetime = Field(default_factory=datetime.now, description="Turn creation time")
    started_at: Optional[datetime] = Field(None, description="Turn start time")
    completed_at: Optional[datetime] = Field(None, description="Turn completion time")
    error_message: Optional[str] = Field(None, description="Error message if turn failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional turn metadata")

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate turn duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_completed(self) -> bool:
        """Check if turn is completed (successfully or not)."""
        return self.status in [TurnStatus.COMPLETED, TurnStatus.FAILED, TurnStatus.TIMEOUT]

    def mark_started(self) -> None:
        """Mark turn as started."""
        self.started_at = datetime.now()
        self.status = TurnStatus.IN_PROGRESS

    def mark_completed(self, response: str) -> None:
        """Mark turn as completed with response."""
        self.response = response
        self.completed_at = datetime.now()
        self.status = TurnStatus.COMPLETED

    def mark_failed(self, error_message: str) -> None:
        """Mark turn as failed with error."""
        self.error_message = error_message
        self.completed_at = datetime.now()
        self.status = TurnStatus.FAILED

    def mark_timeout(self) -> None:
        """Mark turn as timed out."""
        self.error_message = "Turn timed out"
        self.completed_at = datetime.now()
        self.status = TurnStatus.TIMEOUT


class ConversationContext(BaseModel):
    """Immutable conversation context for agent prompts."""

    conversation_id: str = Field(..., description="Conversation identifier")
    topic: str = Field(..., description="Conversation topic")
    participants: List[AgentRole] = Field(..., description="All conversation participants")
    turn_history: List[ConversationTurnDomain] = Field(
        default_factory=list, description="Complete turn history"
    )
    current_turn_number: int = Field(default=0, description="Current turn number")
    context_window_size: int = Field(
        default=6, description="Number of recent turns to include in context"
    )

    @property
    def recent_turns(self) -> List[ConversationTurnDomain]:
        """Get recent turns within context window."""
        completed_turns = [t for t in self.turn_history if t.is_completed]
        return completed_turns[-self.context_window_size :] if completed_turns else []

    @property
    def conversation_summary(self) -> str:
        """Generate a summary of the conversation so far."""
        if not self.recent_turns:
            return f"Starting conversation about: {self.topic}"

        summary_parts = [f"Conversation topic: {self.topic}"]
        summary_parts.append(f"Participants: {', '.join(p.name for p in self.participants)}")

        if self.recent_turns:
            summary_parts.append("Recent discussion:")
            for turn in self.recent_turns:
                if turn.response:
                    summary_parts.append(f"- {turn.agent_name}: {turn.response}")

        return "\n".join(summary_parts)

    def create_agent_prompt(self, agent: AgentRole) -> str:
        """Create contextual prompt for a specific agent."""
        prompt_parts = [
            f"You are {agent.name}, playing the role of {agent.role}.",
            f"Your system prompt: {agent.system_prompt}",
            "",
            self.conversation_summary,
            "",
            f"Please respond as {agent.name} to continue this conversation. Keep your response focused and engaging.",
        ]

        return "\n".join(prompt_parts)


class ConversationEvent(BaseModel):
    """Events that occur during conversation lifecycle."""

    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Event identifier")
    conversation_id: str = Field(..., description="Associated conversation ID")
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")


# Domain Interfaces for Dependency Injection


class IConversationRepository(ABC):
    """Repository interface for conversation persistence."""

    @abstractmethod
    async def save_conversation(self, conversation: "ConversationAggregate") -> None:
        """Save conversation state."""

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional["ConversationAggregate"]:
        """Retrieve conversation by ID."""

    @abstractmethod
    async def list_active_conversations(self, user_id: str) -> List["ConversationAggregate"]:
        """List active conversations for a user."""


class IAgentResponseGenerator(ABC):
    """Interface for generating agent responses."""

    @abstractmethod
    async def generate_response(
        self, agent: AgentRole, context: ConversationContext, timeout_seconds: int = 30
    ) -> str:
        """Generate agent response given context."""


class ITurnController(ABC):
    """Interface for managing conversation turns."""

    @abstractmethod
    async def execute_turn(
        self, turn: ConversationTurnDomain, agent: AgentRole, context: ConversationContext
    ) -> ConversationTurnDomain:
        """Execute a single conversation turn."""


class IConversationEventPublisher(ABC):
    """Interface for publishing conversation events."""

    @abstractmethod
    async def publish_event(self, event: ConversationEvent) -> None:
        """Publish a conversation event."""


class ITurnLimitPolicy(ABC):
    """Policy interface for turn limit management."""

    @abstractmethod
    def should_continue(self, conversation: "ConversationAggregate") -> bool:
        """Determine if conversation should continue."""

    @abstractmethod
    def get_completion_reason(
        self, conversation: "ConversationAggregate"
    ) -> Optional[CompletionReason]:
        """Get reason for conversation completion."""


class ICompletionDetector(ABC):
    """Interface for detecting conversation completion conditions."""

    @abstractmethod
    async def detect_completion(self, context: ConversationContext) -> Optional[CompletionReason]:
        """Detect if conversation should be completed."""


# Core Domain Aggregate


class ConversationAggregate(BaseModel):
    """Rich domain aggregate representing a complete conversation."""

    conversation_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique conversation ID"
    )
    user_id: str = Field(..., description="User who owns this conversation")
    topic: str = Field(..., description="Conversation topic")
    participants: List[AgentRole] = Field(..., description="Conversation participants")
    turns: List[ConversationTurnDomain] = Field(
        default_factory=list, description="All conversation turns"
    )
    status: ConversationStatus = Field(
        default=ConversationStatus.CREATED, description="Current status"
    )
    config: ConversationConfig = Field(
        default_factory=ConversationConfig, description="Conversation configuration"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    completion_reason: Optional[CompletionReason] = Field(None, description="Reason for completion")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    events: List[ConversationEvent] = Field(default_factory=list, description="Conversation events")

    class Config:
        # Allow mutation for aggregate operations
        allow_mutation = True

    @validator("participants")
    def validate_participants(cls, v):
        if len(v) < 1:
            raise ValueError("Conversation must have at least 1 participant")
        if len(v) > 10:
            raise ValueError("Conversation cannot have more than 10 participants")
        return v

    @property
    def current_context(self) -> ConversationContext:
        """Get current conversation context."""
        return ConversationContext(
            conversation_id=self.conversation_id,
            topic=self.topic,
            participants=self.participants,
            turn_history=self.turns,
            current_turn_number=len(self.turns) + 1,
        )

    @property
    def is_active(self) -> bool:
        """Check if conversation is actively running."""
        return self.status == ConversationStatus.ACTIVE

    @property
    def is_completed(self) -> bool:
        """Check if conversation is completed."""
        return self.status in [
            ConversationStatus.COMPLETED,
            ConversationStatus.FAILED,
            ConversationStatus.CANCELLED,
        ]

    @property
    def current_turn_number(self) -> int:
        """Get current turn number."""
        return len(self.turns) + 1

    @property
    def next_agent(self) -> Optional[AgentRole]:
        """Get the next agent to take a turn."""
        if not self.participants:
            return None

        # Simple round-robin based on turn number
        agent_index = len(self.turns) % len(self.participants)
        return sorted(self.participants, key=lambda a: a.turn_order)[agent_index]

    def start_conversation(self) -> None:
        """Start the conversation (business rule: can only start if CREATED)."""
        if self.status != ConversationStatus.CREATED:
            raise ValueError(f"Cannot start conversation in status: {self.status}")

        self.status = ConversationStatus.ACTIVE
        self.started_at = datetime.now()

        # Publish start event
        self._add_event(
            "conversation_started",
            {
                "participant_count": len(self.participants),
                "topic": self.topic,
                "max_turns": self.config.max_turns,
            },
        )

    def pause_conversation(self) -> None:
        """Pause the conversation (business rule: can only pause if ACTIVE)."""
        if self.status != ConversationStatus.ACTIVE:
            raise ValueError(f"Cannot pause conversation in status: {self.status}")

        if not self.config.allow_interruption:
            raise ValueError("Conversation does not allow interruption")

        self.status = ConversationStatus.PAUSED
        self._add_event("conversation_paused", {"turn_number": self.current_turn_number})

    def resume_conversation(self) -> None:
        """Resume the conversation (business rule: can only resume if PAUSED)."""
        if self.status != ConversationStatus.PAUSED:
            raise ValueError(f"Cannot resume conversation in status: {self.status}")

        self.status = ConversationStatus.ACTIVE
        self._add_event("conversation_resumed", {"turn_number": self.current_turn_number})

    def complete_conversation(
        self, reason: CompletionReason, error_message: Optional[str] = None
    ) -> None:
        """Complete the conversation with a specific reason."""
        if self.is_completed:
            return  # Already completed

        self.status = (
            ConversationStatus.COMPLETED if error_message is None else ConversationStatus.FAILED
        )
        self.completed_at = datetime.now()
        self.completion_reason = reason
        self.error_message = error_message

        self._add_event(
            "conversation_completed",
            {
                "reason": reason.value,
                "total_turns": len(self.turns),
                "duration_minutes": self.duration_minutes,
                "error": error_message,
            },
        )

    def add_turn(self, turn: ConversationTurnDomain) -> None:
        """Add a completed turn to the conversation."""
        if not self.is_active:
            raise ValueError(f"Cannot add turn to conversation in status: {self.status}")

        # Validate turn number sequence
        expected_turn_number = len(self.turns) + 1
        if turn.turn_number != expected_turn_number:
            raise ValueError(
                f"Invalid turn number: expected {expected_turn_number}, got {turn.turn_number}"
            )

        # Add the turn
        self.turns.append(turn)

        # Publish turn event
        self._add_event(
            "turn_completed",
            {
                "turn_number": turn.turn_number,
                "agent_id": turn.agent_id,
                "agent_name": turn.agent_name,
                "status": turn.status.value,
                "duration_seconds": turn.duration_seconds,
            },
        )

    def should_continue(self, turn_limit_policy: ITurnLimitPolicy) -> bool:
        """Check if conversation should continue based on business rules."""
        if not self.is_active:
            return False

        return turn_limit_policy.should_continue(self)

    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate conversation duration in minutes."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() / 60
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds() / 60
        return None

    def _add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to the conversation."""
        event = ConversationEvent(
            conversation_id=self.conversation_id, event_type=event_type, data=data
        )
        self.events.append(event)

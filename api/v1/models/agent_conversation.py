"""
Pydantic models for Agent Conversation API

This module defines the request/response models for the unified agent conversation service
as specified in TaskMaster task 28.1.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


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

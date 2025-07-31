"""
Pydantic schemas for Agent Conversation Database Operations

This module provides comprehensive Pydantic schemas for the agent conversation
database system as specified in Task 39.2. These schemas validate all
request/response data for agent conversations, messages, and session management.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class ConversationStatusEnum(str, Enum):
    """Status enumeration for conversation lifecycle."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageTypeEnum(str, Enum):
    """Message type enumeration."""

    TEXT = "text"
    SYSTEM = "system"
    ERROR = "error"
    STATUS = "status"


class MessageRoleEnum(str, Enum):
    """Message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# Request schemas


class CreateAgentConversationSessionRequest(BaseModel):
    """Request schema for creating a new agent conversation session."""

    prompt: str = Field(
        ..., min_length=1, max_length=2000, description="The conversation prompt or topic"
    )
    title: Optional[str] = Field(
        None, max_length=255, description="Optional title for the conversation"
    )
    description: Optional[str] = Field(
        None, max_length=1000, description="Optional description of the conversation"
    )
    max_turns: int = Field(5, ge=1, le=50, description="Maximum number of conversation turns")
    llm_provider: Optional[str] = Field("openai", max_length=50, description="LLM provider to use")
    llm_model: Optional[str] = Field(
        "gpt-3.5-turbo", max_length=100, description="Specific model to use"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional configuration parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Discuss the benefits of renewable energy",
                "title": "Renewable Energy Discussion",
                "description": "A multi-agent conversation about renewable energy benefits",
                "max_turns": 10,
                "llm_provider": "openai",
                "llm_model": "gpt-3.5-turbo",
                "config": {"temperature": 0.7, "agent_count": 3},
            }
        }


class AddAgentToConversationRequest(BaseModel):
    """Request schema for adding an agent to a conversation."""

    agent_id: UUID = Field(..., description="ID of the agent to add")
    role: str = Field(
        "participant", max_length=50, description="Role of the agent in the conversation"
    )

    class Config:
        json_schema_extra = {
            "example": {"agent_id": "550e8400-e29b-41d4-a716-446655440000", "role": "moderator"}
        }


class CreateConversationMessageRequest(BaseModel):
    """Request schema for creating a conversation message."""

    agent_id: UUID = Field(..., description="ID of the agent sending the message")
    content: str = Field(..., min_length=1, max_length=5000, description="Message content")
    role: MessageRoleEnum = Field(MessageRoleEnum.ASSISTANT, description="Message role")
    message_type: MessageTypeEnum = Field(MessageTypeEnum.TEXT, description="Type of message")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "I think renewable energy is crucial for our future sustainability.",
                "role": "assistant",
                "message_type": "text",
                "metadata": {"confidence": 0.9, "processing_time": 150},
            }
        }


class UpdateConversationStatusRequest(BaseModel):
    """Request schema for updating conversation status."""

    status: ConversationStatusEnum = Field(..., description="New conversation status")

    class Config:
        json_schema_extra = {"example": {"status": "completed"}}


# Response schemas


class AgentConversationMessageResponse(BaseModel):
    """Response schema for conversation messages."""

    id: UUID = Field(..., description="Message ID")
    conversation_id: UUID = Field(..., description="ID of the conversation")
    agent_id: UUID = Field(..., description="ID of the agent who sent the message")
    content: str = Field(..., description="Message content")
    message_order: int = Field(..., description="Order of message in conversation")
    turn_number: int = Field(..., description="Turn number in conversation")
    role: MessageRoleEnum = Field(..., description="Message role")
    message_type: MessageTypeEnum = Field(..., description="Type of message")
    metadata: Dict[str, Any] = Field(..., description="Message metadata")
    is_processed: bool = Field(..., description="Whether message has been processed")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    created_at: datetime = Field(..., description="When message was created")
    updated_at: Optional[datetime] = Field(None, description="When message was last updated")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "550e8400-e29b-41d4-a716-446655440002",
                "content": "I believe we should prioritize solar and wind power.",
                "message_order": 1,
                "turn_number": 1,
                "role": "assistant",
                "message_type": "text",
                "metadata": {"confidence": 0.95},
                "is_processed": True,
                "processing_time_ms": 250,
                "created_at": "2024-07-31T23:45:00Z",
                "updated_at": "2024-07-31T23:45:01Z",
            }
        }


class AgentParticipantResponse(BaseModel):
    """Response schema for agent participation in conversation."""

    agent_id: UUID = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent's role in conversation")
    joined_at: datetime = Field(..., description="When agent joined the conversation")
    left_at: Optional[datetime] = Field(None, description="When agent left the conversation")
    message_count: int = Field(..., description="Number of messages sent by this agent")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "550e8400-e29b-41d4-a716-446655440002",
                "agent_name": "Energy Expert",
                "role": "specialist",
                "joined_at": "2024-07-31T23:45:00Z",
                "left_at": None,
                "message_count": 5,
            }
        }


class AgentConversationSessionResponse(BaseModel):
    """Response schema for conversation sessions."""

    id: UUID = Field(..., description="Conversation ID")
    prompt: str = Field(..., description="Conversation prompt")
    title: Optional[str] = Field(None, description="Conversation title")
    description: Optional[str] = Field(None, description="Conversation description")
    status: ConversationStatusEnum = Field(..., description="Current conversation status")
    message_count: int = Field(..., description="Total number of messages")
    agent_count: int = Field(..., description="Number of participating agents")
    max_turns: int = Field(..., description="Maximum allowed turns")
    current_turn: int = Field(..., description="Current turn number")
    llm_provider: Optional[str] = Field(None, description="LLM provider used")
    llm_model: Optional[str] = Field(None, description="LLM model used")
    config: Dict[str, Any] = Field(..., description="Conversation configuration")
    created_at: datetime = Field(..., description="When conversation was created")
    updated_at: Optional[datetime] = Field(None, description="When conversation was last updated")
    started_at: Optional[datetime] = Field(None, description="When conversation started")
    completed_at: Optional[datetime] = Field(None, description="When conversation completed")
    user_id: Optional[str] = Field(None, description="User who created the conversation")

    # Related data
    agents: List[AgentParticipantResponse] = Field(
        default_factory=list, description="Participating agents"
    )
    messages: List[AgentConversationMessageResponse] = Field(
        default_factory=list, description="Conversation messages"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "prompt": "Discuss renewable energy benefits",
                "title": "Renewable Energy Discussion",
                "description": "Multi-agent discussion on renewable energy advantages",
                "status": "active",
                "message_count": 8,
                "agent_count": 3,
                "max_turns": 10,
                "current_turn": 4,
                "llm_provider": "openai",
                "llm_model": "gpt-3.5-turbo",
                "config": {"temperature": 0.7},
                "created_at": "2024-07-31T23:45:00Z",
                "updated_at": "2024-07-31T23:46:00Z",
                "started_at": "2024-07-31T23:45:05Z",
                "completed_at": None,
                "user_id": "user123",
                "agents": [],
                "messages": [],
            }
        }


class ConversationListResponse(BaseModel):
    """Response schema for listing conversations."""

    conversations: List[AgentConversationSessionResponse] = Field(
        ..., description="List of conversations"
    )
    total_count: int = Field(..., description="Total number of conversations")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Number of items per page")

    class Config:
        json_schema_extra = {
            "example": {"conversations": [], "total_count": 25, "page": 1, "page_size": 20}
        }


class MessageListResponse(BaseModel):
    """Response schema for listing messages."""

    messages: List[AgentConversationMessageResponse] = Field(..., description="List of messages")
    total_count: int = Field(..., description="Total number of messages")
    conversation_id: UUID = Field(..., description="Conversation ID")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(50, description="Number of items per page")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [],
                "total_count": 15,
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "page": 1,
                "page_size": 50,
            }
        }


# Query parameter schemas


class ConversationQueryParams(BaseModel):
    """Query parameters for conversation filtering."""

    status: Optional[ConversationStatusEnum] = Field(
        None, description="Filter by conversation status"
    )
    user_id: Optional[str] = Field(None, max_length=255, description="Filter by user ID")
    created_after: Optional[datetime] = Field(
        None, description="Filter conversations created after this date"
    )
    created_before: Optional[datetime] = Field(
        None, description="Filter conversations created before this date"
    )
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")

    @validator("created_before")
    def validate_date_range(cls, v, values):
        if v and "created_after" in values and values["created_after"]:
            if v <= values["created_after"]:
                raise ValueError("created_before must be after created_after")
        return v


class MessageQueryParams(BaseModel):
    """Query parameters for message filtering."""

    agent_id: Optional[UUID] = Field(None, description="Filter by agent ID")
    message_type: Optional[MessageTypeEnum] = Field(None, description="Filter by message type")
    role: Optional[MessageRoleEnum] = Field(None, description="Filter by message role")
    turn_number: Optional[int] = Field(None, ge=1, description="Filter by turn number")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(50, ge=1, le=200, description="Items per page")


# Error response schemas


class ConversationErrorResponse(BaseModel):
    """Error response schema for conversation operations."""

    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    conversation_id: Optional[UUID] = Field(None, description="Related conversation ID")

    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "CONVERSATION_NOT_FOUND",
                "message": "The specified conversation does not exist",
                "details": {
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "user_id": "user123",
                },
                "timestamp": "2024-07-31T23:45:00Z",
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }

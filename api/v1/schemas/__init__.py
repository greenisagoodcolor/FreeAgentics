"""
API v1 Pydantic Schemas

This module provides all Pydantic schemas for API v1 endpoints,
ensuring proper request/response validation and documentation.
"""

from .agent_conversation_schemas import (  # Request schemas; Response schemas; Query parameter schemas; Error schemas; Enums
    AddAgentToConversationRequest,
    AgentConversationMessageResponse,
    AgentConversationSessionResponse,
    AgentParticipantResponse,
    ConversationErrorResponse,
    ConversationListResponse,
    ConversationQueryParams,
    ConversationStatusEnum,
    CreateAgentConversationSessionRequest,
    CreateConversationMessageRequest,
    MessageListResponse,
    MessageQueryParams,
    MessageRoleEnum,
    MessageTypeEnum,
    UpdateConversationStatusRequest,
)

__all__ = [
    # Request schemas
    "CreateAgentConversationSessionRequest",
    "AddAgentToConversationRequest",
    "CreateConversationMessageRequest",
    "UpdateConversationStatusRequest",
    # Response schemas
    "AgentConversationSessionResponse",
    "AgentConversationMessageResponse",
    "AgentParticipantResponse",
    "ConversationListResponse",
    "MessageListResponse",
    # Query parameter schemas
    "ConversationQueryParams",
    "MessageQueryParams",
    # Error schemas
    "ConversationErrorResponse",
    # Enums
    "ConversationStatusEnum",
    "MessageTypeEnum",
    "MessageRoleEnum",
]

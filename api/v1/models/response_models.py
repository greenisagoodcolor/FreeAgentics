"""
Comprehensive response models for API endpoints with proper type annotations.

This module provides standardized response models that ensure:
1. Type safety across all API endpoints
2. Consistent response format
3. Proper FastAPI integration with automatic OpenAPI schema generation
4. Clear documentation for API consumers
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model with common fields."""

    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )
    status: str = Field("success", description="Response status")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Error timestamp"
    )


class SuccessResponse(BaseResponse):
    """Standard success response model."""

    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


class PaginatedResponse(BaseResponse):
    """Paginated response model."""

    items: List[Any] = Field(..., description="List of items")
    total_count: int = Field(..., description="Total number of items")
    page: int = Field(1, description="Current page number")
    per_page: int = Field(20, description="Items per page")
    has_next: bool = Field(False, description="Whether there are more pages")


class AgentResponse(BaseModel):
    """Standardized agent response model."""

    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    template: str = Field(..., description="Agent template")
    status: str = Field(..., description="Agent status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")
    inference_count: int = Field(0, description="Number of inferences performed")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Agent parameters"
    )


class ConversationResponse(BaseModel):
    """Standardized conversation response model."""

    conversation_id: str = Field(..., description="Conversation ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    prompt: str = Field(..., description="Original prompt")
    response: Optional[str] = Field(None, description="LLM response")
    provider: str = Field(..., description="LLM provider used")
    token_count: int = Field(0, description="Total tokens used")
    processing_time_ms: int = Field(0, description="Processing time in milliseconds")
    created_at: datetime = Field(..., description="Conversation timestamp")


class SystemMetricsResponse(BaseModel):
    """System metrics response model."""

    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    uptime: int = Field(..., description="System uptime in seconds")
    active_agents: int = Field(..., description="Number of active agents")
    total_inferences: int = Field(..., description="Total inferences performed")
    avg_response_time: float = Field(..., description="Average response time in ms")
    success_rate: float = Field(..., description="Success rate percentage")
    error_rate: float = Field(..., description="Error rate percentage")


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Check timestamp"
    )
    services: List[Dict[str, Any]] = Field(
        ..., description="Individual service statuses"
    )
    version: str = Field(..., description="Application version")


class TokenResponse(BaseModel):
    """Authentication token response model."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: Dict[str, Union[str, bool, List[str]]] = Field(
        ..., description="User information"
    )


class UserPermissionsResponse(BaseModel):
    """User permissions response model."""

    permissions: List[str] = Field(..., description="List of user permissions")
    role: str = Field(..., description="User role")
    can_create_agents: bool = Field(..., description="Can create agents")
    can_delete_agents: bool = Field(..., description="Can delete agents")
    can_view_metrics: bool = Field(..., description="Can view metrics")
    can_admin_system: bool = Field(..., description="Can administer system")


class InferenceResponse(BaseModel):
    """Inference operation response model."""

    action: Dict[str, Any] = Field(..., description="Agent action")
    beliefs: Dict[str, Any] = Field(..., description="Agent beliefs")
    free_energy: float = Field(..., description="Free energy value")
    timestamp: str = Field(..., description="Inference timestamp")


class BeliefUpdateResponse(BaseModel):
    """Belief update response model."""

    beliefs: Dict[str, Any] = Field(..., description="Updated beliefs")
    timestamp: str = Field(..., description="Update timestamp")


class KnowledgeGraphResponse(BaseModel):
    """Knowledge graph response model."""

    graph_id: str = Field(..., description="Graph ID")
    node_count: int = Field(..., description="Number of nodes")
    edge_count: int = Field(..., description="Number of edges")


class NodeResponse(BaseModel):
    """Knowledge graph node response model."""

    node_id: str = Field(..., description="Node ID")
    data: Dict[str, Union[str, int, float, bool]] = Field(..., description="Node data")


class EdgeResponse(BaseModel):
    """Knowledge graph edge response model."""

    edge_id: str = Field(..., description="Edge ID")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    data: Dict[str, Union[str, int, float, bool]] = Field(..., description="Edge data")


class WebSocketConnectionResponse(BaseModel):
    """WebSocket connection status response model."""

    connections: List[Dict[str, Any]] = Field(..., description="Active connections")
    total_connections: int = Field(..., description="Total number of connections")


class QueueStatusResponse(BaseModel):
    """Message queue status response model."""

    client_id: str = Field(..., description="Client ID")
    messages_queued: int = Field(..., description="Number of queued messages")
    messages_delivered: int = Field(..., description="Number of delivered messages")
    queue_size: int = Field(..., description="Current queue size")


# Type aliases for common response types
AgentListResponse = List[AgentResponse]
ConversationListResponse = List[ConversationResponse]
NodeListResponse = List[NodeResponse]
EdgeListResponse = List[EdgeResponse]

# Union types for flexible responses
FlexibleResponse = Union[
    SuccessResponse, ErrorResponse, Dict[str, Any], List[Dict[str, Any]]
]

StringResponse = Dict[str, str]
IntResponse = Dict[str, int]
BoolResponse = Dict[str, bool]
MixedResponse = Dict[str, Union[str, int, bool, List[Any], Dict[str, Any]]]

"""Pydantic schemas for test data validation.

These schemas ensure test data matches production schemas exactly,
providing type safety and validation for all test objects.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import UUID4, BaseModel, Field, validator


class AgentStatus(str, Enum):
    """Agent status enum matching database model."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class CoalitionStatus(str, Enum):
    """Coalition status enum matching database model."""

    FORMING = "forming"
    ACTIVE = "active"
    DISBANDING = "disbanding"
    DISSOLVED = "dissolved"


class AgentRole(str, Enum):
    """Agent role in coalition enum matching database model."""

    LEADER = "leader"
    COORDINATOR = "coordinator"
    MEMBER = "member"
    OBSERVER = "observer"


class BeliefSchema(BaseModel):
    """Schema for agent beliefs following Active Inference patterns."""

    state_beliefs: Dict[str, float] = Field(default_factory=dict)
    observation_beliefs: Dict[str, float] = Field(default_factory=dict)
    policy_beliefs: Dict[str, float] = Field(default_factory=dict)
    precision: float = Field(default=1.0, ge=0.0)
    free_energy: Optional[float] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @validator("state_beliefs", "observation_beliefs", "policy_beliefs")
    def validate_probability_distribution(cls, v):
        """Ensure probability distributions sum to ~1.0."""
        if v and len(v) > 0:
            total = sum(v.values())
            if abs(total - 1.0) > 0.01:  # Allow small numerical errors
                # Normalize the distribution
                return {k: val / total for k, val in v.items()}
        return v


class PreferenceSchema(BaseModel):
    """Schema for agent preferences in Active Inference."""

    observation_preferences: Dict[str, float] = Field(default_factory=dict)
    state_preferences: Dict[str, float] = Field(default_factory=dict)
    precision: float = Field(default=1.0, ge=0.0)

    @validator("observation_preferences", "state_preferences")
    def validate_preferences(cls, v):
        """Preferences can be negative (aversion) or positive (attraction)."""
        return v


class PyMDPConfigSchema(BaseModel):
    """Schema for PyMDP configuration."""

    num_states: Optional[List[int]] = None
    num_observations: Optional[List[int]] = None
    num_controls: Optional[List[int]] = None
    planning_horizon: int = Field(default=3, ge=1)
    inference_algo: str = Field(default="mmp")
    control_fac_idx: Optional[List[int]] = None
    policies: Optional[List[List[int]]] = None
    gamma: float = Field(default=16.0, ge=0.0)
    use_states_info_gain: bool = Field(default=True)
    use_param_info_gain: bool = Field(default=False)

    class Config:
        extra = "allow"  # Allow additional PyMDP-specific parameters


class AgentMetricsSchema(BaseModel):
    """Schema for agent performance metrics."""

    total_steps: int = Field(default=0, ge=0)
    successful_actions: int = Field(default=0, ge=0)
    failed_actions: int = Field(default=0, ge=0)
    average_free_energy: Optional[float] = None
    average_expected_utility: Optional[float] = None
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None

    @validator("average_free_energy", "average_expected_utility")
    def validate_averages(cls, v):
        """Ensure averages are reasonable values."""
        if v is not None and (v < -1000 or v > 1000):
            raise ValueError(f"Average value {v} seems unreasonable")
        return v


class AgentParametersSchema(BaseModel):
    """Schema for agent-specific parameters."""

    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    discount_factor: float = Field(default=0.95, ge=0.0, le=1.0)
    memory_capacity: int = Field(default=1000, ge=1)
    update_frequency: int = Field(default=1, ge=1)

    class Config:
        extra = "allow"  # Allow template-specific parameters


class AgentSchema(BaseModel):
    """Complete schema for Agent test data."""

    id: UUID4 = Field(default_factory=uuid.uuid4)
    name: str = Field(..., min_length=1, max_length=100)
    template: str = Field(..., min_length=1, max_length=50)
    status: AgentStatus = Field(default=AgentStatus.PENDING)

    # Active Inference specific
    gmn_spec: Optional[str] = None
    pymdp_config: PyMDPConfigSchema = Field(default_factory=PyMDPConfigSchema)
    beliefs: BeliefSchema = Field(default_factory=BeliefSchema)
    preferences: PreferenceSchema = Field(default_factory=PreferenceSchema)

    # Metrics and state
    position: Optional[List[float]] = None
    metrics: AgentMetricsSchema = Field(default_factory=AgentMetricsSchema)
    parameters: AgentParametersSchema = Field(default_factory=AgentParametersSchema)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Statistics
    inference_count: int = Field(default=0, ge=0)
    total_steps: int = Field(default=0, ge=0)

    @validator("position")
    def validate_position(cls, v):
        """Validate position is 2D or 3D coordinates."""
        if v is not None:
            if len(v) not in [2, 3]:
                raise ValueError("Position must be 2D or 3D coordinates")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Position coordinates must be numeric")
        return v

    @validator("template")
    def validate_template(cls, v):
        """Validate template is a known type."""
        valid_templates = [
            "grid_world",
            "resource_collector",
            "explorer",
            "coordinator",
            "custom",
        ]
        if v not in valid_templates:
            raise ValueError(f"Template must be one of {valid_templates}")
        return v


class CoalitionObjectiveSchema(BaseModel):
    """Schema for coalition objectives."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    priority: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    status: str = Field(default="pending", pattern="^(pending|in_progress|completed|failed)$")
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class CoalitionSchema(BaseModel):
    """Complete schema for Coalition test data."""

    id: UUID4 = Field(default_factory=uuid.uuid4)
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    status: CoalitionStatus = Field(default=CoalitionStatus.FORMING)

    # Coalition objectives and capabilities
    objectives: Dict[str, CoalitionObjectiveSchema] = Field(default_factory=dict)
    required_capabilities: List[str] = Field(default_factory=list)
    achieved_objectives: List[str] = Field(default_factory=list)

    # Coalition metrics
    performance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cohesion_score: float = Field(default=1.0, ge=0.0, le=1.0)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    dissolved_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("required_capabilities")
    def validate_capabilities(cls, v):
        """Ensure capabilities are from known set."""
        valid_capabilities = [
            "resource_collection",
            "exploration",
            "defense",
            "coordination",
            "communication",
            "analysis",
            "planning",
            "execution",
            "monitoring",
        ]
        for cap in v:
            if cap not in valid_capabilities:
                raise ValueError(f"Unknown capability: {cap}")
        return v


class KnowledgeNodeSchema(BaseModel):
    """Complete schema for KnowledgeNode test data."""

    id: UUID4 = Field(default_factory=uuid.uuid4)
    type: str = Field(..., min_length=1, max_length=50)
    label: str = Field(..., min_length=1, max_length=200)
    properties: Dict[str, Any] = Field(default_factory=dict)

    # Versioning
    version: int = Field(default=1, ge=1)
    is_current: bool = Field(default=True)

    # Confidence and source
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Optional[str] = Field(default=None, max_length=100)

    # Creator agent relationship
    creator_agent_id: Optional[UUID4] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Embedding for semantic search
    embedding: Optional[List[float]] = None

    @validator("type")
    def validate_type(cls, v):
        """Validate node type."""
        valid_types = [
            "concept",
            "entity",
            "fact",
            "belief",
            "observation",
            "inference",
        ]
        if v not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
        return v

    @validator("embedding")
    def validate_embedding(cls, v):
        """Validate embedding dimensions."""
        if v is not None:
            if len(v) not in [
                128,
                256,
                384,
                512,
                768,
                1024,
            ]:  # Common embedding sizes
                raise ValueError(f"Embedding dimension {len(v)} not standard")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Embedding must contain numeric values")
        return v


class KnowledgeEdgeSchema(BaseModel):
    """Complete schema for KnowledgeEdge test data."""

    id: UUID4 = Field(default_factory=uuid.uuid4)
    source_id: UUID4
    target_id: UUID4
    type: str = Field(..., min_length=1, max_length=50)
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("type")
    def validate_type(cls, v):
        """Validate edge type."""
        valid_types = [
            "relates_to",
            "causes",
            "contradicts",
            "supports",
            "derived_from",
            "depends_on",
            "influences",
            "precedes",
            "follows",
            "equivalent_to",
        ]
        if v not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
        return v

    @validator("source_id", "target_id")
    def validate_not_self_loop(cls, v, values):
        """Ensure no self-loops unless explicitly allowed."""
        if "source_id" in values and v == values["source_id"]:
            # Check if self-loops are allowed in properties
            if not values.get("properties", {}).get("allow_self_loop", False):
                raise ValueError("Self-loops not allowed")
        return v


class AgentCoalitionMembershipSchema(BaseModel):
    """Schema for agent membership in coalition."""

    agent_id: UUID4
    coalition_id: UUID4
    role: AgentRole = Field(default=AgentRole.MEMBER)
    joined_at: datetime = Field(default_factory=datetime.utcnow)
    contribution_score: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_score: float = Field(default=1.0, ge=0.0, le=1.0)


# Batch schemas for performance testing
class BatchAgentSchema(BaseModel):
    """Schema for batch agent creation."""

    count: int = Field(..., ge=1, le=10000)
    template: str = Field(default="grid_world")
    name_prefix: str = Field(default="TestAgent")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE)
    distribute_positions: bool = Field(default=True)
    position_bounds: Optional[Dict[str, List[float]]] = None

    @validator("position_bounds")
    def validate_bounds(cls, v):
        """Validate position bounds structure."""
        if v is not None:
            if "min" not in v or "max" not in v:
                raise ValueError("Position bounds must have 'min' and 'max'")
            if len(v["min"]) != len(v["max"]):
                raise ValueError("Min and max bounds must have same dimensions")
        return v


class PerformanceTestConfigSchema(BaseModel):
    """Schema for performance test configuration."""

    num_agents: int = Field(default=100, ge=1)
    num_coalitions: int = Field(default=10, ge=0)
    num_knowledge_nodes: int = Field(default=1000, ge=0)
    knowledge_graph_connectivity: float = Field(default=0.1, ge=0.0, le=1.0)
    simulation_steps: int = Field(default=100, ge=1)
    seed: Optional[int] = None
    enable_metrics: bool = Field(default=True)
    batch_size: int = Field(default=100, ge=1)

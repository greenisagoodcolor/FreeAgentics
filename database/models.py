"""SQLAlchemy models for FreeAgentics database.

This module defines all database models using modern SQLAlchemy 2.0 patterns.
NO IN-MEMORY STORAGE - everything must persist to PostgreSQL.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database.base import Base
from database.types import GUID
from database.json_encoder import dumps, loads


# Enums for database columns
class AgentStatus(PyEnum):
    """Status of an agent."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class CoalitionStatus(PyEnum):
    """Status of a coalition."""

    FORMING = "forming"
    ACTIVE = "active"
    DISBANDING = "disbanding"
    DISSOLVED = "dissolved"


class AgentRole(PyEnum):
    """Role of an agent in a coalition."""

    LEADER = "leader"
    COORDINATOR = "coordinator"
    MEMBER = "member"
    OBSERVER = "observer"


# Association table for many-to-many relationship between agents and coalitions
agent_coalition_association = Table(
    "agent_coalition",
    Base.metadata,
    Column("agent_id", GUID(), ForeignKey("agents.id"), primary_key=True),
    Column(
        "coalition_id", GUID(), ForeignKey("coalitions.id"), primary_key=True
    ),
    Column("role", Enum(AgentRole, values_callable=lambda x: [e.value for e in x]), default=AgentRole.MEMBER.value),
    Column("joined_at", DateTime, server_default=func.now()),
    Column("contribution_score", Float, default=0.0),
    Column("trust_score", Float, default=1.0),
)


class Agent(Base):
    """Agent model representing an Active Inference agent.

    This replaces the in-memory agents_db dictionary with proper persistence.
    """

    __tablename__ = "agents"

    # Primary key
    id: Column[str] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Basic properties
    name = Column(String(100), nullable=False)
    template = Column(String(50), nullable=False)
    status: Column[AgentStatus] = Column(
        Enum(AgentStatus, values_callable=lambda x: [e.value for e in x]), 
        default=AgentStatus.PENDING.value
    )

    # Active Inference specific
    gmn_spec = Column(Text, nullable=True)
    pymdp_config = Column(JSON, default=dict)
    beliefs = Column(JSON, default=dict)
    preferences = Column(JSON, default=dict)

    # Metrics and state
    position = Column(JSON, nullable=True)  # For grid world agents
    metrics = Column(JSON, default=dict)
    parameters = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    last_active = Column(DateTime, nullable=True)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Statistics
    inference_count = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)

    # Relationships
    coalitions = relationship(
        "Coalition",
        secondary=agent_coalition_association,
        back_populates="agents",
        lazy="select",
    )

    # Knowledge graph nodes created by this agent
    knowledge_nodes = relationship(
        "KnowledgeNode", back_populates="creator_agent", lazy="select"
    )

    def to_dict(self) -> dict:
        """Convert agent to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "template": self.template,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat()
            if self.last_active
            else None,
            "inference_count": self.inference_count,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "position": self.position,
        }


class Coalition(Base):
    """Coalition model for multi-agent coordination.

    Implements real coalition formation with objectives and trust scoring.
    """

    __tablename__ = "coalitions"

    # Primary key
    id: Column[str] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Basic properties
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    status: Column[CoalitionStatus] = Column(
        Enum(CoalitionStatus, values_callable=lambda x: [e.value for e in x]), 
        default=CoalitionStatus.FORMING.value
    )

    # Coalition objectives and capabilities
    objectives = Column(JSON, default=dict)
    required_capabilities = Column(JSON, default=list)
    achieved_objectives = Column(JSON, default=list)

    # Coalition metrics
    performance_score = Column(Float, default=0.0)
    cohesion_score = Column(Float, default=1.0)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    dissolved_at = Column(DateTime, nullable=True)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    agents = relationship(
        "Agent",
        secondary=agent_coalition_association,
        back_populates="coalitions",
        lazy="select",
    )

    def to_dict(self) -> dict:
        """Convert coalition to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "objectives": self.objectives,
            "required_capabilities": self.required_capabilities,
            "achieved_objectives": self.achieved_objectives,
            "performance_score": self.performance_score,
            "cohesion_score": self.cohesion_score,
            "created_at": self.created_at.isoformat(),
            "dissolved_at": self.dissolved_at.isoformat()
            if self.dissolved_at
            else None,
            "agent_count": len(self.agents),
        }


class KnowledgeNode(Base):
    """Knowledge graph node stored in database.

    Persists knowledge graph nodes with full versioning.
    """

    __tablename__ = "db_knowledge_nodes"

    # Primary key
    id: Column[str] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Node properties
    type = Column(String(50), nullable=False)
    label = Column(String(200), nullable=False)
    properties = Column(JSON, default=dict)

    # Versioning
    version = Column(Integer, default=1)
    is_current = Column(Boolean, default=True)

    # Confidence and source
    confidence = Column(Float, default=1.0)
    source = Column(String(100), nullable=True)

    # Creator agent relationship
    creator_agent_id: Column[Optional[str]] = Column(
        GUID(), ForeignKey("agents.id"), nullable=True
    )
    creator_agent = relationship("Agent", back_populates="knowledge_nodes")

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships to edges
    outgoing_edges = relationship(
        "KnowledgeEdge",
        foreign_keys="KnowledgeEdge.source_id",
        back_populates="source_node",
        lazy="select",
    )
    incoming_edges = relationship(
        "KnowledgeEdge",
        foreign_keys="KnowledgeEdge.target_id",
        back_populates="target_node",
        lazy="select",
    )


class KnowledgeEdge(Base):
    """Knowledge graph edge stored in database.

    Persists relationships between knowledge nodes.
    """

    __tablename__ = "db_knowledge_edges"

    # Primary key
    id: Column[str] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Edge endpoints
    source_id: Column[str] = Column(
        GUID(), ForeignKey("db_knowledge_nodes.id")
    )
    target_id: Column[str] = Column(
        GUID(), ForeignKey("db_knowledge_nodes.id")
    )

    # Edge properties
    type = Column(String(50), nullable=False)
    properties = Column(JSON, default=dict)
    confidence = Column(Float, default=1.0)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    source_node = relationship(
        "KnowledgeNode",
        foreign_keys=[source_id],
        back_populates="outgoing_edges",
    )
    target_node = relationship(
        "KnowledgeNode",
        foreign_keys=[target_id],
        back_populates="incoming_edges",
    )

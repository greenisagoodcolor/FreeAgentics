"""SQLAlchemy models for FreeAgentics database.

This module defines all database models using modern SQLAlchemy 2.0 patterns.
NO IN-MEMORY STORAGE - everything must persist to PostgreSQL.
"""

import os
import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from cryptography.fernet import Fernet
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

# Encryption setup for UserSettings
ENCRYPTION_KEY = os.getenv("SETTINGS_ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    # Generate a new key for development
    ENCRYPTION_KEY = Fernet.generate_key().decode()

cipher_suite = Fernet(
    ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY
)


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


class AgentType(PyEnum):
    """Type/personality of an agent for natural language creation."""

    ADVOCATE = "advocate"  # Argues for specific positions, builds cases
    ANALYST = "analyst"  # Breaks down problems, provides data-driven insights
    CRITIC = "critic"  # Identifies flaws, challenges assumptions
    CREATIVE = "creative"  # Generates novel ideas, thinks outside the box
    MODERATOR = "moderator"  # Facilitates discussions, maintains balance


# Association table for many-to-many relationship between agents and coalitions
agent_coalition_association = Table(
    "agent_coalition",
    Base.metadata,
    Column("agent_id", GUID(), ForeignKey("agents.id"), primary_key=True),
    Column("coalition_id", GUID(), ForeignKey("coalitions.id"), primary_key=True),
    Column(
        "role",
        Enum(AgentRole, values_callable=lambda x: [e.value for e in x]),
        default=AgentRole.MEMBER.value,
    ),
    Column("joined_at", DateTime, server_default=func.now()),
    Column("contribution_score", Float, default=0.0),
    Column("trust_score", Float, default=1.0),
)

# Association table for many-to-many relationship between agents and conversations
agent_conversation_association = Table(
    "agent_conversations",
    Base.metadata,
    Column("agent_id", GUID(), ForeignKey("agents.id"), primary_key=True),
    Column(
        "conversation_id", GUID(), ForeignKey("agent_conversation_sessions.id"), primary_key=True
    ),
    Column("role", String(50), default="participant"),
    Column("joined_at", DateTime, server_default=func.now()),
    Column("left_at", DateTime, nullable=True),
    Column("message_count", Integer, default=0),
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
        default=AgentStatus.PENDING.value,
    )

    # Agent type for natural language created agents
    agent_type: Column[Optional[AgentType]] = Column(
        Enum(AgentType, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )

    # AI-generated properties
    system_prompt = Column(Text, nullable=True)  # Generated system prompt
    personality_traits = Column(JSON, default=dict)  # Generated personality profile
    creation_source = Column(String(50), default="manual")  # "manual", "ai_generated", "template"
    source_prompt = Column(Text, nullable=True)  # Original user prompt for AI-generated agents

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
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

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
    knowledge_nodes = relationship("KnowledgeNode", back_populates="creator_agent", lazy="select")

    # Agent conversation sessions this agent participates in
    conversation_sessions = relationship(
        "AgentConversationSession",
        secondary=agent_conversation_association,
        back_populates="agents",
        lazy="select",
    )

    def to_dict(self) -> dict:
        """Convert agent to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "template": self.template,
            "status": self.status.value,
            "agent_type": self.agent_type.value if self.agent_type else None,
            "system_prompt": self.system_prompt,
            "personality_traits": self.personality_traits,
            "creation_source": self.creation_source,
            "source_prompt": self.source_prompt,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None,
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
        default=CoalitionStatus.FORMING.value,
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
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

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
            "dissolved_at": (self.dissolved_at.isoformat() if self.dissolved_at else None),
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
    creator_agent_id: Column[Optional[str]] = Column(GUID(), ForeignKey("agents.id"), nullable=True)
    creator_agent = relationship("Agent", back_populates="knowledge_nodes")

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

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


class User(Base):
    """User model for authentication and authorization.

    Simple user model for testing database security features.
    """

    __tablename__ = "users"

    # Primary key
    id: Column[str] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Basic user properties
    username = Column(String(100), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    settings = relationship("UserSettings", back_populates="user", uselist=False)


class UserSettings(Base):
    """Store user-specific settings securely."""

    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), unique=True, index=True)

    # LLM Configuration (encrypted)
    llm_provider = Column(String, default="openai")
    llm_model = Column(String, default="gpt-3.5-turbo")
    encrypted_openai_key = Column(Text, nullable=True)
    encrypted_anthropic_key = Column(Text, nullable=True)

    # Feature flags
    gnn_enabled = Column(Boolean, default=True)
    debug_logs = Column(Boolean, default=False)
    auto_suggest = Column(Boolean, default=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    user = relationship("User", back_populates="settings")

    def set_openai_key(self, key: Optional[str]):
        """Encrypt and store OpenAI API key."""
        if key:
            self.encrypted_openai_key = cipher_suite.encrypt(key.encode()).decode()
        else:
            self.encrypted_openai_key = None

    def get_openai_key(self) -> Optional[str]:
        """Decrypt and return OpenAI API key."""
        if self.encrypted_openai_key:
            try:
                return cipher_suite.decrypt(self.encrypted_openai_key.encode()).decode()
            except Exception:
                return None
        return None

    def set_anthropic_key(self, key: Optional[str]):
        """Encrypt and store Anthropic API key."""
        if key:
            self.encrypted_anthropic_key = cipher_suite.encrypt(key.encode()).decode()
        else:
            self.encrypted_anthropic_key = None

    def get_anthropic_key(self) -> Optional[str]:
        """Decrypt and return Anthropic API key."""
        if self.encrypted_anthropic_key:
            try:
                return cipher_suite.decrypt(self.encrypted_anthropic_key.encode()).decode()
            except Exception:
                return None
        return None


class KnowledgeEdge(Base):
    """Knowledge graph edge stored in database.

    Persists relationships between knowledge nodes.
    """

    __tablename__ = "db_knowledge_edges"

    # Primary key
    id: Column[str] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Edge endpoints
    source_id: Column[str] = Column(GUID(), ForeignKey("db_knowledge_nodes.id"))
    target_id: Column[str] = Column(GUID(), ForeignKey("db_knowledge_nodes.id"))

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


# Enums for agent conversation system
class ConversationStatus(PyEnum):
    """Status of an agent conversation."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentConversationSession(Base):
    """Agent conversation session model.

    Represents a multi-agent conversation with persistent storage.
    """

    __tablename__ = "agent_conversation_sessions"

    # Primary key
    id: Column[str] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Basic properties
    prompt = Column(Text, nullable=False)
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)

    # Status and lifecycle
    status: Column[ConversationStatus] = Column(
        Enum(ConversationStatus, values_callable=lambda x: [e.value for e in x]),
        default=ConversationStatus.PENDING.value,
    )

    # Conversation metrics
    message_count = Column(Integer, default=0)
    agent_count = Column(Integer, default=0)
    max_turns = Column(Integer, default=5)
    current_turn = Column(Integer, default=0)

    # Configuration
    llm_provider = Column(String(50), nullable=True)
    llm_model = Column(String(100), nullable=True)
    config = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # User who created the conversation
    user_id = Column(String(255), nullable=True, index=True)

    # Relationships
    agents = relationship(
        "Agent",
        secondary=agent_conversation_association,
        back_populates="conversation_sessions",
        lazy="select",
    )

    messages = relationship(
        "AgentConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AgentConversationMessage.message_order",
    )

    def to_dict(self) -> dict:
        """Convert conversation to dictionary for API responses."""
        return {
            "id": str(self.id),
            "prompt": self.prompt,
            "title": self.title,
            "description": self.description,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "message_count": self.message_count,
            "agent_count": self.agent_count,
            "max_turns": self.max_turns,
            "current_turn": self.current_turn,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "user_id": self.user_id,
        }


class AgentConversationMessage(Base):
    """Message in an agent conversation.

    Represents individual messages exchanged during agent conversations.
    """

    __tablename__ = "agent_conversation_messages"

    # Primary key
    id: Column[str] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Foreign keys
    conversation_id: Column[str] = Column(
        GUID(), ForeignKey("agent_conversation_sessions.id"), nullable=False, index=True
    )
    agent_id: Column[str] = Column(GUID(), ForeignKey("agents.id"), nullable=False, index=True)

    # Message content
    content = Column(Text, nullable=False)
    message_order = Column(Integer, nullable=False)
    turn_number = Column(Integer, nullable=False, default=1)

    # Message metadata
    role = Column(String(50), default="assistant")  # 'system', 'user', 'assistant'
    message_type = Column(String(50), default="text")  # 'text', 'system', 'error'
    message_metadata = Column(JSON, default=dict)

    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_time_ms = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    conversation = relationship("AgentConversationSession", back_populates="messages")
    agent = relationship("Agent", lazy="select")

    def to_dict(self) -> dict:
        """Convert message to dictionary for API responses."""
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "agent_id": str(self.agent_id),
            "content": self.content,
            "message_order": self.message_order,
            "turn_number": self.turn_number,
            "role": self.role,
            "message_type": self.message_type,
            "metadata": self.message_metadata,
            "is_processed": self.is_processed,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

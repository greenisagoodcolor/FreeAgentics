"""
SQLAlchemy models for FreeAgentics database.

Defines the core database schema for agents, conversations, knowledge graphs,
coalitions, and system logs.
"""

import enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class AgentStatus(enum.Enum):
    """Enumeration for agent status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class ConversationType(enum.Enum):
    """Enumeration for conversation types"""

    DIRECT = "direct"
    GROUP = "group"
    BROADCAST = "broadcast"
    SYSTEM = "system"


class Agent(Base):
    """
    Agent model representing an AI agent in the system.
    """

    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    # e.g., "explorer", "merchant", "guardian"
    type = Column(String(100), nullable=False)
    status = Column(Enum(AgentStatus), default=AgentStatus.ACTIVE, nullable=False)

    # Agent configuration and state
    config = Column(JSON, default={})  # Agent-specific configuration
    state = Column(JSON, default={})  # Current agent state
    beliefs = Column(JSON, default={})  # Agent beliefs (for Active Inference)

    # Spatial location (H3 hex)
    location = Column(String(15))  # H3 hex string

    # Performance metrics
    energy_level = Column(Float, default=1.0)
    experience_points = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_active_at = Column(DateTime)

    # Relationships
    conversations = relationship("ConversationParticipant", back_populates="agent")
    coalitions = relationship("CoalitionMember", back_populates="agent")
    knowledge_graphs = relationship("KnowledgeGraph", back_populates="owner")

    __table_args__ = (
        Index("idx_agent_type_status", "type", "status"),
        Index("idx_agent_location", "location"),
    )


class Conversation(Base):
    """
    Conversation model for tracking agent interactions.
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, nullable=False, index=True)
    title = Column(String(255))
    type = Column(Enum(ConversationType), default=ConversationType.DIRECT)

    # Conversation metadata
    meta_data = Column(JSON, default={})
    context = Column(JSON, default={})  # Shared context for the conversation

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_message_at = Column(DateTime)

    # Relationships
    participants = relationship("ConversationParticipant", back_populates="conversation")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class ConversationParticipant(Base):
    """
    Many-to-many relationship between conversations and agents.
    """

    __tablename__ = "conversation_participants"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"))
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"))

    # Participant metadata
    role = Column(String(50))  # e.g., "initiator", "participant", "observer"
    joined_at = Column(DateTime, server_default=func.now())
    left_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="participants")
    agent = relationship("Agent", back_populates="conversations")

    __table_args__ = (
        UniqueConstraint("conversation_id", "agent_id", name="uq_conversation_agent"),
        Index("idx_participant_active", "conversation_id", "is_active"),
    )


class Message(Base):
    """
    Message model for conversation history.
    """

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"))
    sender_id = Column(Integer, ForeignKey("agents.id", ondelete="SET NULL"))

    # Message content
    content = Column(Text, nullable=False)
    type = Column(String(50), default="text")  # text, action, system, etc.
    meta_data = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    edited_at = Column(DateTime)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    sender = relationship("Agent")

    __table_args__ = (Index("idx_message_conversation_created", "conversation_id", "created_at"),)


class KnowledgeGraph(Base):
    """
    Knowledge graph model for storing agent knowledge.
    """

    __tablename__ = "knowledge_graphs"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"))

    # Graph metadata
    name = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(String(50))  # e.g., "personal", "shared", "global"

    # Graph data (stored as JSON for flexibility)
    nodes = Column(JSON, default=[])
    edges = Column(JSON, default=[])
    meta_data = Column(JSON, default={})

    # Access control
    is_public = Column(Boolean, default=False)
    access_list = Column(JSON, default=[])  # List of agent IDs with access

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    owner = relationship("Agent", back_populates="knowledge_graphs")

    __table_args__ = (Index("idx_knowledge_owner_type", "owner_id", "type"),)


class Coalition(Base):
    """
    Coalition model for multi-agent collaborations.
    """

    __tablename__ = "coalitions"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Coalition configuration
    type = Column(String(50))  # e.g., "business", "exploration", "defense"
    goal = Column(JSON, default={})
    rules = Column(JSON, default={})

    # Coalition state
    status = Column(String(50), default="forming")  # forming, active, disbanded
    value_pool = Column(Float, default=0.0)  # Shared value/resources

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    activated_at = Column(DateTime)
    disbanded_at = Column(DateTime)

    # Relationships
    members = relationship("CoalitionMember", back_populates="coalition")

    __table_args__ = (Index("idx_coalition_status_type", "status", "type"),)


class CoalitionMember(Base):
    """
    Many-to-many relationship between coalitions and agents.
    """

    __tablename__ = "coalition_members"

    id = Column(Integer, primary_key=True)
    coalition_id = Column(Integer, ForeignKey("coalitions.id", ondelete="CASCADE"))
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"))

    # Member metadata
    role = Column(String(50))  # e.g., "leader", "member", "observer"
    contribution = Column(Float, default=0.0)  # Value contributed
    share = Column(Float, default=0.0)  # Share of coalition value

    # Timestamps
    joined_at = Column(DateTime, server_default=func.now())
    left_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

    # Relationships
    coalition = relationship("Coalition", back_populates="members")
    agent = relationship("Agent", back_populates="coalitions")

    __table_args__ = (
        UniqueConstraint("coalition_id", "agent_id", name="uq_coalition_agent"),
        Index("idx_coalition_member_active", "coalition_id", "is_active"),
    )


class SystemLog(Base):
    """
    System log model for monitoring and debugging.
    """

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False, index=True)

    # Log information
    level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    component = Column(String(100), nullable=False)  # Component that generated the log
    message = Column(Text, nullable=False)

    # Optional context
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="SET NULL"))
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="SET NULL"))
    coalition_id = Column(Integer, ForeignKey("coalitions.id", ondelete="SET NULL"))

    # Additional data
    data = Column(JSON, default={})
    error_trace = Column(Text)  # For error logs

    __table_args__ = (
        Index("idx_log_timestamp_level", "timestamp", "level"),
        Index("idx_log_component_timestamp", "component", "timestamp"),
    )

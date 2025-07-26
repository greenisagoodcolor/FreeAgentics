"""Database models for prompt processing and conversation tracking.

This module defines models for tracking conversations, prompts, and their
relationships to agents and knowledge graph updates.
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
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database.base import Base
from database.types import GUID


class ConversationStatus(PyEnum):
    """Status of a conversation."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    ERROR = "error"


class PromptStatus(PyEnum):
    """Status of a prompt processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class Conversation(Base):
    """Conversation model for tracking multi-turn interactions."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conversations_user_id", "user_id"),
        Index("idx_conversations_status", "status"),
        Index("idx_conversations_created_at", "created_at"),
        {
            "extend_existing": True
        },  # TODO: ARCHITECTURAL DEBT - CRITICAL: Duplicate conversations table with conversation_models.py (see NEMESIS Committee findings)
    )

    # Primary key
    id: Column[uuid.UUID] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # User association
    user_id: Column[str] = Column(String(100), nullable=False)

    # Conversation metadata
    title: Column[Optional[str]] = Column(String(200), nullable=True)
    status: Column[ConversationStatus] = Column(
        Enum(ConversationStatus),
        default=ConversationStatus.ACTIVE,
        nullable=False,
    )

    # Context tracking
    context: Column[dict] = Column(JSON, default=dict)
    agent_ids: Column[list] = Column(JSON, default=list)  # List of agent IDs in conversation

    # Timestamps
    created_at: Column[datetime] = Column(DateTime, server_default=func.now())
    updated_at: Column[datetime] = Column(DateTime, server_default=func.now(), onupdate=func.now())
    completed_at: Column[Optional[datetime]] = Column(DateTime, nullable=True)

    # Relationships
    prompts = relationship("Prompt", back_populates="conversation", cascade="all, delete-orphan")


class Prompt(Base):
    """Prompt model for tracking individual prompts within conversations."""

    __tablename__ = "prompts"
    __table_args__ = (
        Index("idx_prompts_conversation_id", "conversation_id"),
        Index("idx_prompts_agent_id", "agent_id"),
        Index("idx_prompts_status", "status"),
        Index("idx_prompts_created_at", "created_at"),
        {
            "extend_existing": True
        },  # TODO: ARCHITECTURAL DEBT - Prevent table redefinition conflicts (see NEMESIS Committee findings)
    )

    # Primary key
    id: Column[uuid.UUID] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Conversation association
    conversation_id: Column[uuid.UUID] = Column(
        GUID(), ForeignKey("conversations.id"), nullable=False
    )

    # Prompt content
    prompt_text: Column[str] = Column(Text, nullable=False)
    iteration_count: Column[int] = Column(Integer, default=1)

    # Processing results
    agent_id: Column[Optional[uuid.UUID]] = Column(GUID(), ForeignKey("agents.id"), nullable=True)
    gmn_specification: Column[Optional[str]] = Column(Text, nullable=True)
    status: Column[PromptStatus] = Column(
        Enum(PromptStatus), default=PromptStatus.PENDING, nullable=False
    )

    # Response data
    response_data: Column[dict] = Column(JSON, default=dict)
    next_suggestions: Column[list] = Column(JSON, default=list)
    warnings: Column[list] = Column(JSON, default=list)
    errors: Column[list] = Column(JSON, default=list)

    # Performance metrics
    processing_time_ms: Column[Optional[float]] = Column(Float, nullable=True)
    tokens_used: Column[Optional[int]] = Column(Integer, nullable=True)

    # Timestamps
    created_at: Column[datetime] = Column(DateTime, server_default=func.now())
    processed_at: Column[Optional[datetime]] = Column(DateTime, nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="prompts")
    knowledge_updates = relationship(
        "KnowledgeGraphUpdate",
        back_populates="prompt",
        cascade="all, delete-orphan",
    )


class KnowledgeGraphUpdate(Base):
    """Track knowledge graph updates from prompt processing."""

    __tablename__ = "knowledge_graph_updates"
    __table_args__ = (
        Index("idx_kg_updates_prompt_id", "prompt_id"),
        Index("idx_kg_updates_node_type", "node_type"),
        Index("idx_kg_updates_created_at", "created_at"),
        {
            "extend_existing": True
        },  # TODO: ARCHITECTURAL DEBT - Prevent table redefinition conflicts (see NEMESIS Committee findings)
    )

    # Primary key
    id: Column[uuid.UUID] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Prompt association
    prompt_id: Column[uuid.UUID] = Column(GUID(), ForeignKey("prompts.id"), nullable=False)

    # Update details
    node_id: Column[str] = Column(String(100), nullable=False)
    node_type: Column[str] = Column(String(50), nullable=False)
    operation: Column[str] = Column(String(20), nullable=False)  # add, update, delete
    properties: Column[dict] = Column(JSON, default=dict)

    # Success tracking
    applied: Column[bool] = Column(Boolean, default=False)
    error_message: Column[Optional[str]] = Column(Text, nullable=True)

    # Timestamp
    created_at: Column[datetime] = Column(DateTime, server_default=func.now())

    # Relationships
    prompt = relationship("Prompt", back_populates="knowledge_updates")


class PromptTemplate(Base):
    """Reusable prompt templates for common agent creation patterns."""

    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint("name", name="uq_prompt_templates_name"),
        Index("idx_prompt_templates_category", "category"),
        Index("idx_prompt_templates_is_active", "is_active"),
        {
            "extend_existing": True
        },  # TODO: ARCHITECTURAL DEBT - Prevent table redefinition conflicts (see NEMESIS Committee findings)
    )

    # Primary key
    id: Column[uuid.UUID] = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Template metadata
    name: Column[str] = Column(String(100), nullable=False)
    description: Column[Optional[str]] = Column(Text, nullable=True)
    category: Column[str] = Column(String(50), nullable=False)

    # Template content
    template_text: Column[str] = Column(Text, nullable=False)
    default_parameters: Column[dict] = Column(JSON, default=dict)
    example_prompts: Column[list] = Column(JSON, default=list)

    # GMN hints
    suggested_gmn_structure: Column[Optional[str]] = Column(Text, nullable=True)
    constraints: Column[dict] = Column(JSON, default=dict)

    # Usage tracking
    usage_count: Column[int] = Column(Integer, default=0)
    success_rate: Column[Optional[float]] = Column(Float, nullable=True)
    is_active: Column[bool] = Column(Boolean, default=True)

    # Timestamps
    created_at: Column[datetime] = Column(DateTime, server_default=func.now())
    updated_at: Column[datetime] = Column(DateTime, server_default=func.now(), onupdate=func.now())

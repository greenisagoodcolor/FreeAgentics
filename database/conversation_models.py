"""
Database models for conversation management.

This module provides models for storing and managing conversation data,
including conversations, messages, and related metadata.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from database.base import Base


class ValidationStatus(Enum):
    """Validation status for messages."""

    PENDING = "pending"
    VALIDATED = "validated"
    FAILED = "failed"


class Conversation(Base):
    """Model representing a conversation."""

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    user_id = Column(UUID(as_uuid=True), nullable=True)

    # Relationships
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Conversation(id={self.id}, title='{self.title}')>"


class Message(Base):
    """Model representing a message in a conversation."""

    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False
    )
    content = Column(Text, nullable=False)
    role = Column(String(50), nullable=False)  # user, assistant, system
    created_at = Column(DateTime, default=datetime.utcnow)
    order = Column(Integer, nullable=False, default=0)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role}', content='{self.content[:50]}...')>"


class ConversationMetadata(Base):
    """Model for storing conversation metadata."""

    __tablename__ = "conversation_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False
    )
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation")

    def __repr__(self):
        return f"<ConversationMetadata(conversation_id={self.conversation_id}, key='{self.key}')>"

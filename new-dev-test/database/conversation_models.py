"""
Database models for conversation management.

This module provides models for storing and managing conversation data,
including conversations, messages, and related metadata.
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database.base import Base


class ValidationStatus(Enum):
    """GMN validation status enum."""

    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    ERROR = "error"
    VALIDATED = "validated"  # Legacy compatibility
    FAILED = "failed"  # Legacy compatibility


class Conversation(Base):
    """Model representing a conversation."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Conversation(id={self.id}, title='{self.title}')>"


class Message(Base):
    """Message model for storing individual messages in conversations."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    sender_type = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Optional fields for structured data
    gmn_spec = Column(Text, nullable=True)  # JSON string for GMN specifications
    validation_status = Column(String, nullable=True)  # ValidationStatus enum value
    validation_message = Column(Text, nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, sender_type='{self.sender_type}', content='{self.content[:50]}...')>"

"""Conversation and message database models."""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, Boolean
from sqlalchemy.orm import relationship

from database.base import Base


class ValidationStatus(Enum):
    """GMN validation status enum."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    ERROR = "error"


class Conversation(Base):
    """Conversation model for storing chat sessions."""
    
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


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
"""
Database module for FreeAgentics.

This module handles all database operations including:
- SQLAlchemy models and ORM
- Database migrations via Alembic
- Connection management
- Seed data scripts
"""

from .connection import SessionLocal, engine, get_db
from .models import (
    Agent,
    Base,
    Coalition,
    Conversation,
    KnowledgeGraph,
    SystemLog)

__all__ = [
    "Base",
    "Agent",
    "Conversation",
    "KnowledgeGraph",
    "Coalition",
    "SystemLog",
    "get_db",
    "engine",
    "SessionLocal",
]

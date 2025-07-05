"""Database package for FreeAgentics.

This module provides SQLAlchemy models and database utilities
for persistent storage of agents, coalitions, and knowledge graphs.
"""

from database.base import Base
from database.models import Agent, Coalition, KnowledgeEdge, KnowledgeNode
from database.session import SessionLocal, engine, get_db

__all__ = [
    "Base",
    "Agent",
    "Coalition",
    "KnowledgeNode",
    "KnowledgeEdge",
    "engine",
    "SessionLocal",
    "get_db",
]

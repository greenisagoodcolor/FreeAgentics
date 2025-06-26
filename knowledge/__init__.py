"""
Knowledge management and representation module.

This module provides knowledge graph capabilities for agents to store,
retrieve, and reason about information and beliefs.
"""

from .knowledge_graph import KnowledgeGraph, PatternExtractor

__all__ = ["KnowledgeGraph", "PatternExtractor"]

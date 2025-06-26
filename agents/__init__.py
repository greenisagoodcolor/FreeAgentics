"""Agents Package for FreeAgentics.

This package provides the agent system for FreeAgentics, including:
- Base agent framework and interfaces
- Specialized agent types (explorer, merchant, scholar, guardian)
- Agent behaviors and personality systems
- Communication and interaction protocols
"""

# Import specialized agent types
# Import base agent system
from . import base, explorer, merchant

__all__ = ["base", "explorer", "merchant"]

__version__ = "0.1.0"

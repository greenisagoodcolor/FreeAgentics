"""
WebSocket handlers for FreeAgentics API

This module contains WebSocket endpoints and connection managers for real-time communication.
"""

from .agent_conversation import router as agent_conversation_ws_router

__all__ = [
    "agent_conversation_ws_router",
]

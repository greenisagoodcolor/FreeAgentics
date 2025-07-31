"""
Conversation Service for Agent Conversation API

Provides dependency injection service for managing conversation lifecycle and database operations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.orm import Session

from database.models import Agent as AgentModel
from database.models import AgentStatus as DBAgentStatus

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversation lifecycle and database operations."""

    def __init__(self):
        """Initialize conversation service."""
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}

    async def create_conversation(
        self,
        prompt: str,
        agent_count: int,
        user_id: str,
        db: Session,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation with multiple agents."""

        conversation_id = str(uuid4())

        try:
            logger.info(f"Creating conversation {conversation_id} with {agent_count} agents")

            # Create agent roles based on conversation topic
            agent_roles = self._generate_agent_roles(prompt, agent_count)

            # Create agents in database
            agents = []
            for role in agent_roles:
                agent = await self._create_conversation_agent(role, db, user_id)
                agents.append(agent)

            # Initialize conversation state
            conversation_state = {
                "conversation_id": conversation_id,
                "prompt": prompt,
                "agents": [
                    {"id": str(a.id), "name": a.name, "role": a.parameters.get("role")}
                    for a in agents
                ],
                "status": "initialized",
                "created_at": datetime.now(),
                "user_id": user_id,
                "config": config or {},
                "turn_count": 0,
                "max_turns": config.get("max_turns", 5) if config else 5,
            }

            self.active_conversations[conversation_id] = conversation_state
            self.conversation_history[conversation_id] = []

            logger.info(f"Successfully created conversation {conversation_id}")
            return conversation_state

        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise HTTPException(status_code=500, detail=f"Conversation creation failed: {str(e)}")

    def _generate_agent_roles(self, prompt: str, agent_count: int) -> List[Dict[str, Any]]:
        """Generate agent roles based on conversation prompt."""

        # Define role templates for different conversation types
        role_templates = [
            {
                "name": "Advocate",
                "role": "advocate",
                "personality": "enthusiastic, supportive, optimistic",
                "system_prompt": "You are an advocate who supports ideas and looks for positive aspects. "
                "Be enthusiastic and constructive in your responses.",
            },
            {
                "name": "Analyst",
                "role": "analyst",
                "personality": "logical, methodical, detail-oriented",
                "system_prompt": "You are an analyst who examines ideas critically and methodically. "
                "Focus on facts, data, and logical reasoning.",
            },
            {
                "name": "Critic",
                "role": "critic",
                "personality": "cautious, questioning, thorough",
                "system_prompt": "You are a critic who identifies potential problems and asks tough questions. "
                "Be constructive but challenging.",
            },
            {
                "name": "Creative",
                "role": "creative",
                "personality": "imaginative, innovative, unconventional",
                "system_prompt": "You are a creative thinker who comes up with innovative ideas and alternative approaches. "
                "Think outside the box.",
            },
            {
                "name": "Moderator",
                "role": "moderator",
                "personality": "balanced, diplomatic, organized",
                "system_prompt": "You are a moderator who helps guide the conversation, summarizes points, "
                "and keeps discussions focused.",
            },
        ]

        # Select appropriate roles based on agent count
        selected_roles = role_templates[:agent_count]

        # Customize system prompts based on conversation topic
        for role in selected_roles:
            role["system_prompt"] += (
                f"\n\nThe conversation topic is: {prompt}\n\n"
                "Keep your responses concise (1-2 sentences) and engaging."
            )

        return selected_roles

    async def _create_conversation_agent(
        self, role: Dict[str, Any], db: Session, user_id: str
    ) -> AgentModel:
        """Create a database record for a conversation agent."""

        # Create GMN specification for the agent
        gmn_spec = {
            "name": role["name"],
            "role": role["role"],
            "personality": role["personality"],
            "system_prompt": role["system_prompt"],
            "states": ["listening", "thinking", "responding"],
            "actions": ["listen", "respond", "question", "agree", "disagree"],
            "conversation_mode": True,
        }

        agent_parameters = {
            "use_pymdp": False,  # Simplified for conversations
            "role": role["role"],
            "personality": role["personality"],
            "conversation_mode": True,
            "system_prompt": role["system_prompt"],
            "user_id": user_id,
        }

        # Create database record
        db_agent = AgentModel(
            name=role["name"],
            template="conversation_agent",
            status=DBAgentStatus.ACTIVE,
            parameters=agent_parameters,
            gmn_spec=str(gmn_spec),  # Store as string for now
        )

        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)

        logger.info(f"Created conversation agent: {db_agent.id} ({role['name']})")
        return db_agent

    async def add_message_to_conversation(
        self, conversation_id: str, agent_id: str, content: str, turn_number: int
    ) -> Dict[str, Any]:
        """Add a message to an existing conversation."""

        if conversation_id not in self.active_conversations:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

        message = {
            "id": str(uuid4()),
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "content": content,
            "timestamp": datetime.now(),
            "turn_number": turn_number,
        }

        # Add to conversation history
        self.conversation_history[conversation_id].append(message)

        # Update conversation state
        conversation_state = self.active_conversations[conversation_id]
        conversation_state["turn_count"] = max(conversation_state["turn_count"], turn_number)
        conversation_state["last_activity"] = datetime.now()

        logger.info(f"Added message to conversation {conversation_id} from agent {agent_id}")
        return message

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation state and history."""

        if conversation_id not in self.active_conversations:
            return None

        conversation_state = self.active_conversations[conversation_id]
        messages = self.conversation_history.get(conversation_id, [])

        return {**conversation_state, "messages": messages, "message_count": len(messages)}

    def list_conversations(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active conversations, optionally filtered by user."""

        conversations = []
        for conv_id, conv_state in self.active_conversations.items():
            if user_id is None or conv_state.get("user_id") == user_id:
                message_count = len(self.conversation_history.get(conv_id, []))
                conversations.append(
                    {
                        "conversation_id": conv_id,
                        "prompt": conv_state["prompt"][:100] + "..."
                        if len(conv_state["prompt"]) > 100
                        else conv_state["prompt"],
                        "agent_count": len(conv_state["agents"]),
                        "message_count": message_count,
                        "status": conv_state["status"],
                        "created_at": conv_state["created_at"],
                        "last_activity": conv_state.get("last_activity", conv_state["created_at"]),
                    }
                )

        # Sort by last activity
        conversations.sort(key=lambda x: x["last_activity"], reverse=True)
        return conversations

    async def complete_conversation(self, conversation_id: str) -> bool:
        """Mark a conversation as completed."""

        if conversation_id not in self.active_conversations:
            return False

        self.active_conversations[conversation_id]["status"] = "completed"
        self.active_conversations[conversation_id]["completed_at"] = datetime.now()

        logger.info(f"Marked conversation {conversation_id} as completed")
        return True

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its history."""

        if conversation_id not in self.active_conversations:
            return False

        del self.active_conversations[conversation_id]
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]

        logger.info(f"Deleted conversation {conversation_id}")
        return True

    def get_conversation_context(self, conversation_id: str, max_messages: int = 6) -> List[str]:
        """Get recent conversation context for agent prompting."""

        messages = self.conversation_history.get(conversation_id, [])

        # Get the most recent messages
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

        # Format as conversation context
        context = []
        for msg in recent_messages:
            # Find agent name from the conversation state
            conv_state = self.active_conversations.get(conversation_id, {})
            agent_name = "Unknown"
            for agent in conv_state.get("agents", []):
                if agent["id"] == msg["agent_id"]:
                    agent_name = agent["name"]
                    break

            context.append(f"{agent_name}: {msg['content']}")

        return context

    async def broadcast_conversation_update(
        self, conversation_id: str, event_type: str, data: Dict[str, Any]
    ):
        """Broadcast conversation updates via WebSocket."""

        try:
            # Import WebSocket notification functions
            from api.v1.websockets.agent_conversation import (
                notify_agent_message,
                notify_conversation_complete,
                notify_conversation_error,
                notify_conversation_status,
            )

            # Route to appropriate notification function based on event type
            if event_type == "conversation_message":
                await notify_agent_message(
                    conversation_id=conversation_id,
                    agent_id=data.get("agent_id", ""),
                    agent_name=data.get("agent_name", "Unknown"),
                    content=data.get("content", ""),
                    turn_number=data.get("turn_number", 0),
                )
            elif event_type == "conversation_completed":
                await notify_conversation_complete(
                    conversation_id=conversation_id,
                    total_turns=data.get("total_turns", 0),
                    total_messages=data.get("total_messages", 0),
                )
            elif event_type == "conversation_error":
                await notify_conversation_error(
                    conversation_id=conversation_id,
                    error_code=data.get("error_code", "UNKNOWN_ERROR"),
                    error_message=data.get("error_message", "An error occurred"),
                    details=data.get("details", {}),
                )
            else:
                # Generic status update
                await notify_conversation_status(
                    conversation_id=conversation_id, status=event_type, details=data
                )
        except Exception as e:
            logger.warning(f"Failed to broadcast conversation update: {e}")


# Dependency injection factory function
def get_conversation_service() -> ConversationService:
    """Factory function for FastAPI dependency injection."""
    return ConversationService()

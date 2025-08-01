"""
Unified Agent Conversation Service for FreeAgentics

This endpoint handles the complete flow:
1. Accept user prompt
2. Create agents with roles
3. Generate GMN specs (simplified templates for now)
4. Initialize PyMDP beliefs
5. Start conversation
6. Return conversation ID and initial state
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth.security_implementation import Permission, TokenData, get_current_user, require_permission
from database.models import Agent as AgentModel
from database.models import AgentStatus as DBAgentStatus
from database.session import get_db
from inference.llm.provider_factory import LLMProviderFactory
from inference.llm.provider_interface import GenerationRequest

logger = logging.getLogger(__name__)

router = APIRouter()


class ConversationRequest(BaseModel):
    """Request to start a multi-agent conversation."""

    prompt: str = Field(..., description="User's prompt describing the conversation topic")
    agent_count: int = Field(2, ge=1, le=5, description="Number of agents to create (1-5)")
    conversation_turns: int = Field(5, ge=1, le=20, description="Number of conversation turns")
    llm_provider: Optional[str] = Field("openai", description="LLM provider to use")
    model: Optional[str] = Field("gpt-3.5-turbo", description="Specific model to use")


class AgentRole(BaseModel):
    """Agent role definition."""

    name: str
    role: str
    personality: str
    system_prompt: str


class ConversationMessage(BaseModel):
    """A message in the conversation."""

    id: str
    agent_id: str
    agent_name: str
    content: str
    timestamp: datetime
    turn_number: int


class ConversationResponse(BaseModel):
    """Response containing the full conversation."""

    conversation_id: str
    agents: List[Dict[str, Any]]
    messages: List[ConversationMessage]
    status: str
    total_turns: int
    started_at: datetime
    completed_at: Optional[datetime] = None


class AgentConversationService:
    """Service for managing agent conversations."""

    def __init__(self):
        self.llm_factory = LLMProviderFactory()
        self.active_conversations: Dict[str, Dict] = {}

    def create_agent_roles(self, prompt: str, agent_count: int) -> List[AgentRole]:
        """Create agent roles based on the conversation prompt."""

        # Simple role templates based on common conversation patterns
        role_templates = [
            {
                "name": "Advocate",
                "role": "advocate",
                "personality": "enthusiastic, supportive, optimistic",
                "system_prompt": "You are an advocate who supports ideas and looks for positive aspects. Be enthusiastic and constructive in your responses.",
            },
            {
                "name": "Analyst",
                "role": "analyst",
                "personality": "logical, methodical, detail-oriented",
                "system_prompt": "You are an analyst who examines ideas critically and methodically. Focus on facts, data, and logical reasoning.",
            },
            {
                "name": "Critic",
                "role": "critic",
                "personality": "cautious, questioning, thorough",
                "system_prompt": "You are a critic who identifies potential problems and asks tough questions. Be constructive but challenging.",
            },
            {
                "name": "Creative",
                "role": "creative",
                "personality": "imaginative, innovative, unconventional",
                "system_prompt": "You are a creative thinker who comes up with innovative ideas and alternative approaches. Think outside the box.",
            },
            {
                "name": "Moderator",
                "role": "moderator",
                "personality": "balanced, diplomatic, organized",
                "system_prompt": "You are a moderator who helps guide the conversation, summarizes points, and keeps discussions focused.",
            },
        ]

        # Select roles based on agent count
        selected_roles = role_templates[:agent_count]

        # Customize system prompts based on the conversation topic
        for role in selected_roles:
            role[
                "system_prompt"
            ] += f"\n\nThe conversation topic is: {prompt}\n\nKeep your responses concise (1-2 sentences) and engaging."

        return [AgentRole(**role) for role in selected_roles]

    async def create_agents(
        self, roles: List[AgentRole], db: Session, user_id: str
    ) -> List[AgentModel]:
        """Create database records for agents."""

        agents = []
        for role in roles:
            # Simple GMN template for conversation agents
            gmn_spec = {
                "name": role.name,
                "role": role.role,
                "personality": role.personality,
                "system_prompt": role.system_prompt,
                "states": ["listening", "thinking", "responding"],
                "actions": ["listen", "respond", "question", "agree", "disagree"],
                "conversation_mode": True,
            }

            agent_parameters = {
                "use_pymdp": False,  # Simplified for conversations
                "role": role.role,
                "personality": role.personality,
                "conversation_mode": True,
                "system_prompt": role.system_prompt,
            }

            # Create database record
            db_agent = AgentModel(
                name=role.name,
                template="conversation_agent",
                status=DBAgentStatus.ACTIVE,
                parameters=agent_parameters,
                gmn_spec=json.dumps(gmn_spec),
            )

            db.add(db_agent)
            db.commit()
            db.refresh(db_agent)

            agents.append(db_agent)
            logger.info(f"Created conversation agent: {db_agent.id} ({role.name})")

        return agents

    async def run_conversation(
        self,
        agents: List[AgentModel],
        prompt: str,
        turns: int,
        user_id: str,
        llm_provider: str = "openai",
        model: str = "gpt-3.5-turbo",
    ) -> List[ConversationMessage]:
        """Run the actual conversation between agents."""

        messages = []
        conversation_history = []

        try:
            # Get LLM provider
            provider_manager = self.llm_factory.create_from_config(user_id=user_id)
            healthy_providers = provider_manager.registry.get_healthy_providers()

            if not healthy_providers:
                raise HTTPException(
                    status_code=503,
                    detail="No LLM providers available. Please configure API keys in settings.",
                )

            # Start conversation with the initial prompt
            conversation_history.append(f"Topic: {prompt}")

            for turn in range(turns):
                for agent in agents:
                    try:
                        # Get agent's system prompt and personality
                        system_prompt = agent.parameters.get(
                            "system_prompt", "You are a helpful assistant."
                        )

                        # Build conversation context
                        context_messages = [{"role": "system", "content": system_prompt}]

                        # Add recent conversation history (last 6 messages to keep context manageable)
                        recent_history = conversation_history[-6:] if conversation_history else []
                        if recent_history:
                            context_messages.append(
                                {
                                    "role": "user",
                                    "content": f"Here's the conversation so far:\n"
                                    + "\n".join(recent_history)
                                    + f"\n\nPlease respond as {agent.name} ({agent.parameters.get('role', 'participant')}).",
                                }
                            )
                        else:
                            context_messages.append(
                                {
                                    "role": "user",
                                    "content": f"Please respond to this topic as {agent.name} ({agent.parameters.get('role', 'participant')}): {prompt}",
                                }
                            )

                        # Generate response
                        generation_request = GenerationRequest(
                            messages=context_messages,
                            model=model,
                            temperature=0.8,  # Higher temperature for more varied responses
                            max_tokens=150,  # Keep responses concise
                        )

                        response = provider_manager.generate_with_fallback(generation_request)

                        # Extract content from response
                        if hasattr(response, "content"):
                            content = response.content
                        elif hasattr(response, "text"):
                            content = response.text
                        else:
                            content = str(response)

                        # Clean up the response
                        content = content.strip()
                        if content.startswith(f"{agent.name}:"):
                            content = content[len(f"{agent.name}:") :].strip()

                        # Create conversation message
                        message = ConversationMessage(
                            id=str(uuid4()),
                            agent_id=str(agent.id),
                            agent_name=agent.name,
                            content=content,
                            timestamp=datetime.now(),
                            turn_number=turn + 1,
                        )

                        messages.append(message)
                        conversation_history.append(f"{agent.name}: {content}")

                        logger.info(f"Turn {turn + 1} - {agent.name}: {content[:100]}...")

                        # Short delay between responses to avoid rate limits
                        await asyncio.sleep(0.5)

                    except Exception as e:
                        logger.error(f"Error generating response for agent {agent.name}: {e}")
                        # Continue with other agents even if one fails
                        continue

                # Delay between turns
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error during conversation: {e}")
            raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")

        return messages

    async def broadcast_conversation_update(
        self, conversation_id: str, message: ConversationMessage
    ):
        """Broadcast conversation updates via WebSocket."""

        try:
            from api.v1.websocket import manager

            # Send message in format expected by frontend
            await manager.broadcast(
                {
                    "type": "message",
                    "data": {
                        "id": f"agent-{message.id}",
                        "role": "assistant",  # Agent messages are assistant role
                        "content": f"{message.agent_name}: {message.content}",
                        "timestamp": message.timestamp.isoformat(),
                        "conversationId": conversation_id,
                        "isStreaming": False,
                        "metadata": {
                            "agent_id": message.agent_id,
                            "agent_name": message.agent_name,
                            "turn_number": message.turn_number,
                        },
                    },
                }
            )
        except Exception as e:
            logger.warning(f"Failed to broadcast conversation update: {e}")


# Global service instance
conversation_service = AgentConversationService()


@router.post("/agent-conversations", response_model=ConversationResponse)
@require_permission(Permission.CREATE_AGENT)
async def start_agent_conversation(
    request: ConversationRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationResponse:
    """Start a multi-agent conversation.

    This is the main endpoint that:
    1. Creates agents with different roles
    2. Runs a conversation between them
    3. Returns the full conversation
    4. Broadcasts updates via WebSocket
    """

    conversation_id = str(uuid4())
    started_at = datetime.now()

    logger.info(f"Starting agent conversation {conversation_id} for user {current_user.user_id}")
    logger.info(f"Prompt: {request.prompt}")
    logger.info(f"Agent count: {request.agent_count}, Turns: {request.conversation_turns}")

    try:
        # Step 1: Create agent roles
        roles = conversation_service.create_agent_roles(request.prompt, request.agent_count)
        logger.info(f"Created {len(roles)} agent roles")

        # Step 2: Create agents in database
        agents = await conversation_service.create_agents(roles, db, current_user.user_id)
        logger.info(f"Created {len(agents)} agents in database")

        # Step 3: Run the conversation
        messages = await conversation_service.run_conversation(
            agents=agents,
            prompt=request.prompt,
            turns=request.conversation_turns,
            user_id=current_user.user_id,
            llm_provider=request.llm_provider,
            model=request.model,
        )

        logger.info(f"Generated {len(messages)} conversation messages")

        # Step 4: Broadcast each message via WebSocket
        for message in messages:
            await conversation_service.broadcast_conversation_update(conversation_id, message)
            # Small delay to ensure ordered delivery
            await asyncio.sleep(0.1)

        # Step 5: Store conversation state
        conversation_service.active_conversations[conversation_id] = {
            "agents": [str(agent.id) for agent in agents],
            "messages": [msg.dict() for msg in messages],
            "status": "completed",
            "started_at": started_at,
            "completed_at": datetime.now(),
        }

        # Step 6: Prepare response
        agent_data = []
        for agent in agents:
            agent_data.append(
                {
                    "id": str(agent.id),
                    "name": agent.name,
                    "role": agent.parameters.get("role", "participant"),
                    "personality": agent.parameters.get("personality", ""),
                    "status": agent.status.value
                    if hasattr(agent.status, "value")
                    else str(agent.status),
                }
            )

        response = ConversationResponse(
            conversation_id=conversation_id,
            agents=agent_data,
            messages=messages,
            status="completed",
            total_turns=request.conversation_turns,
            started_at=started_at,
            completed_at=datetime.now(),
        )

        # Step 7: Broadcast completion
        try:
            from api.v1.websocket import manager

            # Send completion message in format expected by frontend
            await manager.broadcast(
                {
                    "type": "message",
                    "data": {
                        "id": f"system-completion-{conversation_id}",
                        "role": "system",
                        "content": f"✅ Conversation completed with {len(agents)} agents exchanging {len(messages)} messages.",
                        "timestamp": datetime.now().isoformat(),
                        "conversationId": conversation_id,
                        "isStreaming": False,
                        "metadata": {
                            "event": "conversation_completed",
                            "agent_count": len(agents),
                            "message_count": len(messages),
                            "status": "completed",
                        },
                    },
                }
            )
        except Exception as e:
            logger.warning(f"Failed to broadcast conversation completion: {e}")

        logger.info(f"Successfully completed conversation {conversation_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start agent conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")


@router.get("/agent-conversations/{conversation_id}", response_model=ConversationResponse)
@require_permission(Permission.VIEW_AGENTS)
async def get_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user),
) -> ConversationResponse:
    """Get a conversation by ID."""

    # Get conversation from service
    conversation_data = conversation_service.get_conversation(conversation_id)

    if not conversation_data:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

    # Convert stored message dicts back to ConversationMessage objects
    messages = [ConversationMessage(**msg_data) for msg_data in conversation_data["messages"]]

    return ConversationResponse(
        conversation_id=conversation_id,
        agents=conversation_data.get("agents", []),
        messages=messages,
        status=conversation_data["status"],
        total_turns=len(messages),
        started_at=conversation_data["started_at"],
        completed_at=conversation_data.get("completed_at"),
    )


@router.get("/agent-conversations")
@require_permission(Permission.VIEW_AGENTS)
async def list_conversations(
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """List active conversations."""

    conversations = []
    for conv_id, conv_data in conversation_service.active_conversations.items():
        conversations.append(
            {
                "conversation_id": conv_id,
                "status": conv_data["status"],
                "agent_count": len(conv_data.get("agents", [])),
                "message_count": len(conv_data.get("messages", [])),
                "started_at": conv_data["started_at"].isoformat(),
                "completed_at": conv_data.get("completed_at").isoformat()
                if conv_data.get("completed_at")
                else None,
            }
        )

    return {"conversations": conversations, "total_count": len(conversations)}


@router.delete("/agent-conversations/{conversation_id}")
@require_permission(Permission.DELETE_AGENT)
async def delete_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, str]:
    """Delete a conversation."""

    if conversation_id not in conversation_service.active_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    del conversation_service.active_conversations[conversation_id]
    logger.info(f"Deleted conversation {conversation_id}")

    return {"message": f"Conversation {conversation_id} deleted successfully"}

"""
Agent Conversation API Router

Main FastAPI endpoint that orchestrates the agent creation and conversation flow.
Implements task 28.3 requirements.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from api.v1.models.agent_conversation import (
    AdvancedAgentConversationRequest,
    AgentConversationRequest,
    MultiAgentConversationResponse,
)
from api.v1.services import (
    ConversationService,
    GMNParserService,
    LLMService,
    PyMDPService,
    get_conversation_service,
    get_gmn_parser_service,
    get_llm_service,
    get_pymdp_service,
)
from auth.security_implementation import Permission, TokenData, get_current_user, require_permission
from database.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-conversations", tags=["agent-conversations"])


@router.post("", response_model=MultiAgentConversationResponse, status_code=201)
@require_permission(Permission.CREATE_AGENT)
async def create_agent_conversation(
    request: AgentConversationRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service),
    gmn_parser: GMNParserService = Depends(get_gmn_parser_service),
    pymdp_service: PyMDPService = Depends(get_pymdp_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> MultiAgentConversationResponse:
    """
    Create a multi-agent conversation from a user prompt.

    This is the main orchestration endpoint that:
    1. Validates the request
    2. Generates a conversation ID
    3. Creates multiple agents with different roles
    4. Starts the conversation
    5. Returns the conversation details
    """

    conversation_id = str(uuid4())
    started_at = datetime.now()

    logger.info(f"Creating agent conversation {conversation_id} for user {current_user.user_id}")
    logger.info(f"Prompt: {request.prompt}")

    try:
        # Step 1: Validate request
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Extract configuration parameters
        config = request.config or {}
        agent_count = config.get("agent_count", 2)
        conversation_turns = config.get("conversation_turns", 5)
        llm_provider = config.get("llm_provider", "openai")
        model = config.get("model", "gpt-3.5-turbo")

        # Validate agent count
        if not (1 <= agent_count <= 5):
            raise HTTPException(status_code=400, detail="Agent count must be between 1 and 5")

        # Step 2: Create conversation with agents
        conversation_data = await conversation_service.create_conversation(
            prompt=request.prompt,
            agent_count=agent_count,
            user_id=current_user.user_id,
            db=db,
            config=config,
        )

        # Use the actual conversation ID from the service
        conversation_id = conversation_data.get("conversation_id", conversation_id)

        # Step 3: Start background task for conversation execution
        background_tasks.add_task(
            run_conversation_background,
            conversation_id,
            conversation_data,
            request.prompt,
            conversation_turns,
            current_user.user_id,
            llm_service,
            conversation_service,
            llm_provider,
            model,
        )

        # Step 4: Prepare response
        response = MultiAgentConversationResponse(
            conversation_id=conversation_id,
            agents=conversation_data["agents"],
            turns=[],  # Will be populated by background task
            status="active",
            total_turns=0,
            websocket_url=f"ws://localhost:8000/api/v1/ws/conv_{conversation_id}",
            started_at=started_at,
            completed_at=None,
        )

        logger.info(f"Successfully created conversation {conversation_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create agent conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversation creation failed: {str(e)}")


@router.post("/advanced", response_model=MultiAgentConversationResponse, status_code=201)
@require_permission(Permission.CREATE_AGENT)
async def create_advanced_agent_conversation(
    request: AdvancedAgentConversationRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service),
    gmn_parser: GMNParserService = Depends(get_gmn_parser_service),
    pymdp_service: PyMDPService = Depends(get_pymdp_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> MultiAgentConversationResponse:
    """
    Create an advanced multi-agent conversation with detailed configuration.

    This endpoint supports advanced configuration options including:
    - Custom LLM settings (provider, model, temperature)
    - Conversation behavior settings (turn count, timeouts)
    - Metadata and custom parameters
    """

    conversation_id = str(uuid4())
    started_at = datetime.now()

    logger.info(f"Creating advanced conversation {conversation_id} for user {current_user.user_id}")
    logger.info(f"Configuration: {request.llm_config.dict()}, {request.conversation_config.dict()}")

    try:
        # Step 1: Extract configuration
        llm_config = request.llm_config
        conv_config = request.conversation_config

        # Step 2: Create conversation with agents
        conversation_data = await conversation_service.create_conversation(
            prompt=request.prompt,
            agent_count=conv_config.agent_count,
            user_id=current_user.user_id,
            db=db,
            config={
                "max_turns": conv_config.max_turns,
                "turn_timeout": conv_config.turn_timeout_seconds,
                "llm_provider": llm_config.provider,
                "model": llm_config.model,
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens,
                **request.metadata,
            },
        )

        # Step 3: Start background conversation with advanced config
        background_tasks.add_task(
            run_conversation_background,
            conversation_id,
            conversation_data,
            request.prompt,
            conv_config.max_turns,
            current_user.user_id,
            llm_service,
            conversation_service,
            llm_config.provider,
            llm_config.model,
            llm_config.temperature,
            llm_config.max_tokens,
        )

        # Step 4: Prepare response
        response = MultiAgentConversationResponse(
            conversation_id=conversation_id,
            agents=conversation_data["agents"],
            turns=[],
            status="active",
            total_turns=0,
            websocket_url=f"ws://localhost:8000/api/v1/ws/conv_{conversation_id}",
            started_at=started_at,
            completed_at=None,
        )

        logger.info(f"Successfully created advanced conversation {conversation_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create advanced conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Advanced conversation creation failed: {str(e)}"
        )


@router.get("/{conversation_id}", response_model=MultiAgentConversationResponse)
@require_permission(Permission.VIEW_AGENTS)
async def get_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> MultiAgentConversationResponse:
    """Get details of a specific conversation."""

    conversation_data = conversation_service.get_conversation(conversation_id)

    if not conversation_data:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

    # Convert messages to turns format
    turns = []
    for msg in conversation_data.get("messages", []):
        # Find agent name
        agent_name = "Unknown"
        for agent in conversation_data.get("agents", []):
            if agent["id"] == msg["agent_id"]:
                agent_name = agent["name"]
                break

        from api.v1.models.agent_conversation import ConversationTurn

        turn = ConversationTurn(
            turn_id=msg["id"],
            agent_id=msg["agent_id"],
            agent_name=agent_name,
            content=msg["content"],
            timestamp=msg["timestamp"],
            turn_number=msg["turn_number"],
        )
        turns.append(turn)

    return MultiAgentConversationResponse(
        conversation_id=conversation_id,
        agents=conversation_data["agents"],
        turns=turns,
        status=conversation_data["status"],
        total_turns=len(turns),
        websocket_url=f"ws://localhost:8000/api/v1/ws/conv_{conversation_id}",
        started_at=conversation_data["created_at"],
        completed_at=conversation_data.get("completed_at"),
    )


@router.get("", response_model=Dict[str, Any])
@require_permission(Permission.VIEW_AGENTS)
async def list_conversations(
    current_user: TokenData = Depends(get_current_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> Dict[str, Any]:
    """List conversations for the current user."""

    conversations = conversation_service.list_conversations(user_id=current_user.user_id)

    return {"conversations": conversations, "total_count": len(conversations)}


@router.delete("/{conversation_id}")
@require_permission(Permission.DELETE_AGENT)
async def delete_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> Dict[str, str]:
    """Delete a conversation."""

    success = await conversation_service.delete_conversation(conversation_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

    return {"message": f"Conversation {conversation_id} deleted successfully"}


@router.get("/health")
async def agent_conversation_health(
    llm_service: LLMService = Depends(get_llm_service),
    gmn_parser: GMNParserService = Depends(get_gmn_parser_service),
    pymdp_service: PyMDPService = Depends(get_pymdp_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> Dict[str, Any]:
    """Health check endpoint that validates all service dependencies."""

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "metrics": {},
    }

    # Check LLM Service
    try:
        # Try to get a provider manager (this validates API keys)
        test_user_id = "health_check"
        llm_service.clear_cache()  # Clear cache to ensure fresh check

        # This will fail if no API keys are configured
        provider_manager = await llm_service.get_provider_manager(test_user_id)
        healthy_providers = provider_manager.registry.get_healthy_providers()

        health_status["services"]["llm_service"] = {
            "status": "healthy" if healthy_providers else "degraded",
            "healthy_providers": len(healthy_providers),
            "provider_types": [p.provider_type.value for p in healthy_providers],
        }

        llm_service.clear_cache()  # Clean up test cache

    except Exception as e:
        health_status["services"]["llm_service"] = {
            "status": "unhealthy",
            "error": str(e),
            "healthy_providers": 0,
        }
        health_status["status"] = "degraded"

    # Check GMN Parser Service
    try:
        # Test GMN parsing with a simple structure
        test_gmn = {
            "name": "test_agent",
            "states": ["active"],
            "observations": ["test"],
            "actions": ["wait"],
        }
        gmn_parser._validate_gmn_structure(test_gmn)

        health_status["services"]["gmn_parser"] = {"status": "healthy"}
    except Exception as e:
        health_status["services"]["gmn_parser"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"

    # Check PyMDP Service
    try:
        active_agents = pymdp_service.list_active_agents()
        health_status["services"]["pymdp_service"] = {
            "status": "healthy",
            "active_agents": len(active_agents),
        }
    except Exception as e:
        health_status["services"]["pymdp_service"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"

    # Check Conversation Service
    try:
        conversations = conversation_service.list_conversations()
        health_status["services"]["conversation_service"] = {
            "status": "healthy",
            "active_conversations": len(conversations),
        }
    except Exception as e:
        health_status["services"]["conversation_service"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"

    # Check WebSocket connections
    try:
        from api.v1.websockets.agent_conversation import conversation_manager

        active_conversations = conversation_manager.list_active_conversations()
        total_connections = sum(active_conversations.values())

        health_status["services"]["websocket_service"] = {
            "status": "healthy",
            "active_conversations": len(active_conversations),
            "total_connections": total_connections,
        }
    except Exception as e:
        health_status["services"]["websocket_service"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"

    # Add basic metrics
    health_status["metrics"] = {
        "total_services": len(health_status["services"]),
        "healthy_services": sum(
            1 for s in health_status["services"].values() if s["status"] == "healthy"
        ),
        "degraded_services": sum(
            1 for s in health_status["services"].values() if s["status"] == "degraded"
        ),
        "unhealthy_services": sum(
            1 for s in health_status["services"].values() if s["status"] == "unhealthy"
        ),
    }

    return health_status


@router.get("/metrics")
async def get_agent_conversation_metrics(
    llm_service: LLMService = Depends(get_llm_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> Dict[str, Any]:
    """Get metrics for agent conversation services."""

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "llm_service": llm_service.get_metrics(),
        "conversation_service": {
            "active_conversations": len(conversation_service.active_conversations),
            "total_messages": sum(
                len(messages) for messages in conversation_service.conversation_history.values()
            ),
        },
    }

    # Add WebSocket metrics
    try:
        from api.v1.websockets.agent_conversation import conversation_manager

        ws_conversations = conversation_manager.list_active_conversations()
        metrics["websocket_service"] = {
            "active_conversations": len(ws_conversations),
            "total_connections": sum(ws_conversations.values()),
            "conversations": ws_conversations,
        }
    except Exception as e:
        metrics["websocket_service"] = {
            "error": str(e),
            "active_conversations": 0,
            "total_connections": 0,
        }

    return metrics


# Background task functions


async def run_conversation_background(
    conversation_id: str,
    conversation_data: Dict[str, Any],
    prompt: str,
    max_turns: int,
    user_id: str,
    llm_service: LLMService,
    conversation_service: ConversationService,
    llm_provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.8,
    max_tokens: int = 150,
):
    """Background task to run the actual conversation between agents."""

    logger.info(f"Starting background conversation {conversation_id}")

    try:
        agents = conversation_data["agents"]
        conversation_history = []

        # Initialize conversation context
        conversation_history.append(f"Topic: {prompt}")

        for turn in range(max_turns):
            for agent in agents:
                try:
                    # Get agent's role and personality from database
                    # For now, use simple role-based system prompts
                    role = agent.get("role", "participant")
                    system_prompt = _get_system_prompt_for_role(role, prompt)

                    # Build context messages
                    context_messages = []

                    # Add recent conversation history
                    context = conversation_service.get_conversation_context(
                        conversation_id, max_messages=6
                    )

                    if context:
                        context_messages.append(
                            {
                                "role": "user",
                                "content": f"Here's the conversation so far:\n"
                                + "\n".join(context)
                                + f"\n\nPlease respond as {agent['name']} ({role}).",
                            }
                        )
                    else:
                        context_messages.append(
                            {
                                "role": "user",
                                "content": f"Please respond to this topic as {agent['name']} ({role}): {prompt}",
                            }
                        )

                    # Generate response
                    response_content = await llm_service.generate_conversation_response(
                        system_prompt=system_prompt,
                        context_messages=context_messages,
                        user_id=user_id,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    # Clean up response
                    response_content = response_content.strip()
                    if response_content.startswith(f"{agent['name']}:"):
                        response_content = response_content[len(f"{agent['name']}:") :].strip()

                    # Add message to conversation
                    await conversation_service.add_message_to_conversation(
                        conversation_id=conversation_id,
                        agent_id=agent["id"],
                        content=response_content,
                        turn_number=turn + 1,
                    )

                    # Broadcast update
                    await conversation_service.broadcast_conversation_update(
                        conversation_id=conversation_id,
                        event_type="conversation_message",
                        data={
                            "agent_id": agent["id"],
                            "agent_name": agent["name"],
                            "content": response_content,
                            "turn_number": turn + 1,
                        },
                    )

                    logger.info(f"Turn {turn + 1} - {agent['name']}: {response_content[:100]}...")

                    # Short delay between responses
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error generating response for agent {agent['name']}: {e}")
                    continue

            # Delay between turns
            await asyncio.sleep(1)

        # Mark conversation as completed
        await conversation_service.complete_conversation(conversation_id)

        # Broadcast completion
        await conversation_service.broadcast_conversation_update(
            conversation_id=conversation_id,
            event_type="conversation_completed",
            data={"total_turns": max_turns, "agent_count": len(agents)},
        )

        logger.info(f"Completed background conversation {conversation_id}")

    except Exception as e:
        logger.error(f"Background conversation {conversation_id} failed: {e}", exc_info=True)

        # Mark conversation as error
        if conversation_id in conversation_service.active_conversations:
            conversation_service.active_conversations[conversation_id]["status"] = "error"


def _get_system_prompt_for_role(role: str, topic: str) -> str:
    """Get system prompt based on agent role."""

    role_prompts = {
        "advocate": f"You are an advocate who supports ideas and looks for positive aspects. "
        f"Be enthusiastic and constructive in your responses about: {topic}. "
        f"Keep responses concise (1-2 sentences).",
        "analyst": f"You are an analyst who examines ideas critically and methodically. "
        f"Focus on facts, data, and logical reasoning about: {topic}. "
        f"Keep responses concise (1-2 sentences).",
        "critic": f"You are a critic who identifies potential problems and asks tough questions. "
        f"Be constructive but challenging regarding: {topic}. "
        f"Keep responses concise (1-2 sentences).",
        "creative": f"You are a creative thinker who comes up with innovative ideas and alternative approaches. "
        f"Think outside the box about: {topic}. "
        f"Keep responses concise (1-2 sentences).",
        "moderator": f"You are a moderator who helps guide the conversation, summarizes points, "
        f"and keeps discussions focused on: {topic}. "
        f"Keep responses concise (1-2 sentences).",
    }

    return role_prompts.get(
        role,
        f"You are a helpful participant in a discussion about: {topic}. "
        f"Keep responses concise (1-2 sentences).",
    )

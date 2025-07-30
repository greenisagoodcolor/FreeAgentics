"""
UI Compatibility Layer - Bridge between UI and backend API.

This module provides API endpoints that match what the UI expects,
while internally calling the existing v1 API endpoints.

Following TDD Green phase: minimal implementation to make tests pass.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.v1.agents import Agent as V1Agent
from api.v1.agents import AgentConfig as V1AgentConfig
from api.v1.agents import create_agent as v1_create_agent
from api.v1.agents import delete_agent as v1_delete_agent
from api.v1.agents import get_agent as v1_get_agent
from api.v1.agents import list_agents as v1_list_agents
from api.v1.agents import update_agent_status as v1_update_agent_status
from api.v1.knowledge_graph import get_knowledge_graph
from auth.security_implementation import TokenData, get_current_user
from database.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


# UI-Expected Models
class UIAgentCreateRequest(BaseModel):
    """Simple agent creation request from UI."""

    description: str = Field(..., min_length=1, description="Description of the agent")


class UIAgent(BaseModel):
    """Simple agent model expected by UI."""

    id: str
    name: str
    type: str
    status: str
    description: Optional[str] = None
    createdAt: Optional[str] = None
    lastActiveAt: Optional[str] = None


class UIAgentListResponse(BaseModel):
    """Agent list response expected by UI."""

    agents: List[UIAgent]


# Helper functions
def extract_agent_type_from_description(description: str) -> str:
    """Extract agent type from description using simple keyword matching."""
    description_lower = description.lower()

    if any(word in description_lower for word in ["explore", "search", "find", "discover"]):
        return "explorer"
    elif any(word in description_lower for word in ["collect", "gather", "resource"]):
        return "collector"
    elif any(word in description_lower for word in ["analyze", "study", "examine"]):
        return "analyzer"
    else:
        return "explorer"  # Default to explorer


def extract_agent_name_from_description(description: str) -> str:
    """Extract a reasonable name from description."""
    # Simple heuristic: take first few words and make it a name
    words = description.split()[:3]
    return " ".join(words).title()


def v1_agent_to_ui_agent(v1_agent: V1Agent) -> UIAgent:
    """Convert V1 agent format to UI format."""
    return UIAgent(
        id=v1_agent.id,
        name=v1_agent.name,
        type=extract_agent_type_from_description(v1_agent.parameters.get("description", "")),
        status=v1_agent.status,
        description=v1_agent.parameters.get("description"),
        createdAt=v1_agent.created_at.isoformat() if v1_agent.created_at else None,
        lastActiveAt=v1_agent.last_active.isoformat() if v1_agent.last_active else None,
    )


# UI-Compatible Endpoints
@router.post("/agents", response_model=UIAgent, status_code=201)
async def create_agent_ui(
    request: UIAgentCreateRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UIAgent:
    """Create agent using UI-compatible simple format."""
    # Convert UI request to V1 format
    agent_name = extract_agent_name_from_description(request.description)
    agent_type = extract_agent_type_from_description(request.description)

    # Map agent type to template
    template_map = {
        "explorer": "basic-explorer",
        "collector": "basic-explorer",  # Use explorer template for now
        "analyzer": "basic-explorer",  # Use explorer template for now
    }

    v1_config = V1AgentConfig(
        name=agent_name,
        template=template_map.get(agent_type, "basic-explorer"),
        parameters={"description": request.description},
        gmn_spec=None,  # Optional field
        use_pymdp=True,
        planning_horizon=3,
    )

    # Call existing V1 API
    v1_agent = await v1_create_agent(v1_config, current_user, db)

    # Convert to UI format
    ui_agent = v1_agent_to_ui_agent(v1_agent)

    # Create real agent instance in agent manager
    from api.v1.agents import AGENT_MANAGER_AVAILABLE, agent_manager

    if AGENT_MANAGER_AVAILABLE and agent_manager:
        try:
            # Create real agent in the manager
            real_agent_id = agent_manager.create_agent(
                agent_type=agent_type,
                name=agent_name,
                description=request.description,
            )

            # Start the agent
            agent_manager.start_agent(real_agent_id)
            ui_agent.status = "active"

            logger.info(f"Created and started real agent in manager: {real_agent_id}")

        except Exception as e:
            logger.warning(f"Failed to create real agent in manager: {e}")
            ui_agent.status = "pending"
    else:
        ui_agent.status = "pending"

    # Broadcast WebSocket event
    await broadcast_agent_created_event(ui_agent)

    logger.info(f"Created agent via UI compatibility layer: {ui_agent.id}")

    return ui_agent


@router.get("/agents", response_model=UIAgentListResponse)
async def list_agents_ui(
    status: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UIAgentListResponse:
    """List agents in UI-compatible format."""
    # Call existing V1 API
    v1_agents = await v1_list_agents(status, 100, current_user, db)

    # Convert to UI format
    ui_agents = [v1_agent_to_ui_agent(agent) for agent in v1_agents]

    return UIAgentListResponse(agents=ui_agents)


@router.get("/agents/{agent_id}", response_model=UIAgent)
async def get_agent_ui(
    agent_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UIAgent:
    """Get agent in UI-compatible format."""
    # Call existing V1 API
    v1_agent = await v1_get_agent(agent_id, current_user, db)

    # Convert to UI format
    return v1_agent_to_ui_agent(v1_agent)


@router.patch("/agents/{agent_id}/status")
async def update_agent_status_ui(
    agent_id: str,
    status_update: Dict[str, str],
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Update agent status in UI-compatible format."""
    status = status_update.get("status")
    if not status:
        raise HTTPException(status_code=400, detail="Status is required")

    # Map UI status to V1 status
    status_map = {
        "active": "active",
        "idle": "pending",
        "stopped": "stopped",
    }

    v1_status = status_map.get(status, status)

    # Call existing V1 API
    await v1_update_agent_status(agent_id, v1_status, current_user, db)

    # Update agent in agent manager if available
    from api.v1.agents import AGENT_MANAGER_AVAILABLE, agent_manager

    if AGENT_MANAGER_AVAILABLE and agent_manager:
        try:
            if status == "active":
                agent_manager.start_agent(agent_id)
            elif status == "stopped":
                agent_manager.stop_agent(agent_id)
            logger.info(f"Updated agent {agent_id} status in manager to: {status}")
        except Exception as e:
            logger.warning(f"Failed to update agent status in manager: {e}")

    # Get updated agent for broadcasting
    try:
        updated_agent = await get_agent_ui(agent_id, current_user, db)
        await broadcast_agent_updated_event(updated_agent)
    except Exception as e:
        logger.error(f"Failed to broadcast agent update event: {e}")

    # Return in UI format
    return {"status": status, "agent_id": agent_id}


@router.delete("/agents/{agent_id}")
async def delete_agent_ui(
    agent_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Delete agent in UI-compatible format."""
    # Remove from agent manager if available
    from api.v1.agents import AGENT_MANAGER_AVAILABLE, agent_manager

    if AGENT_MANAGER_AVAILABLE and agent_manager:
        try:
            agent_manager.delete_agent(agent_id)
            logger.info(f"Deleted agent {agent_id} from manager")
        except Exception as e:
            logger.warning(f"Failed to delete agent from manager: {e}")

    # Call existing V1 API
    await v1_delete_agent(agent_id, current_user, db)

    # Broadcast deletion event
    await broadcast_agent_deleted_event(agent_id)

    return {"message": f"Agent {agent_id} deleted successfully"}


# WebSocket event broadcasting functions
async def broadcast_agent_created_event(agent: UIAgent):
    """Broadcast agent creation event to WebSocket clients."""
    try:
        from api.v1.websocket import broadcast_agent_event

        await broadcast_agent_event(
            agent_id=agent.id,
            event_type="created",
            data={
                "agent": agent.dict(),
                "timestamp": datetime.now().isoformat(),
            },
        )
        logger.info(f"Broadcasted agent creation event for {agent.id}")
    except Exception as e:
        logger.error(f"Failed to broadcast agent creation event: {e}")


async def broadcast_agent_updated_event(agent: UIAgent):
    """Broadcast agent update event to WebSocket clients."""
    try:
        from api.v1.websocket import broadcast_agent_event

        await broadcast_agent_event(
            agent_id=agent.id,
            event_type="updated",
            data={
                "agent": agent.dict(),
                "timestamp": datetime.now().isoformat(),
            },
        )
        logger.info(f"Broadcasted agent update event for {agent.id}")
    except Exception as e:
        logger.error(f"Failed to broadcast agent update event: {e}")


async def broadcast_agent_deleted_event(agent_id: str):
    """Broadcast agent deletion event to WebSocket clients."""
    try:
        from api.v1.websocket import broadcast_agent_event

        await broadcast_agent_event(
            agent_id=agent_id,
            event_type="deleted",
            data={
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
            },
        )
        logger.info(f"Broadcasted agent deletion event for {agent_id}")
    except Exception as e:
        logger.error(f"Failed to broadcast agent deletion event: {e}")


# Knowledge graph endpoint
@router.get("/knowledge-graph")
async def get_knowledge_graph_ui(
    current_user: TokenData = Depends(get_current_user),
):
    """Get knowledge graph - UI compatibility endpoint."""
    return await get_knowledge_graph(current_user)


# Prompt processing endpoint
class ProcessPromptRequest(BaseModel):
    """Request model for processing prompts."""
    prompt: str = Field(..., description="The prompt to process")
    conversationId: Optional[str] = Field(None, description="Optional conversation ID")


class ProcessPromptResponse(BaseModel):
    """Response model for prompt processing."""
    response: str
    knowledgeGraph: Optional[Dict] = None


@router.post("/process-prompt")
async def process_prompt_ui(
    request: ProcessPromptRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProcessPromptResponse:
    """Process a prompt and send response via WebSocket."""
    # Generate a unique message ID
    import uuid
    message_id = str(uuid.uuid4())
    
    # Send the response via WebSocket to all connected clients
    try:
        from api.v1.websocket import manager
        
        # Send user message echo via WebSocket
        await manager.broadcast({
            "type": "message",
            "data": {
                "id": f"user-{message_id}",
                "role": "user",
                "content": request.prompt,
                "timestamp": datetime.now().isoformat(),
                "conversationId": request.conversationId
            }
        })
        
        # Simulate processing delay
        import asyncio
        await asyncio.sleep(0.5)
        
        # Send assistant response via WebSocket
        response_text = f"I received your message: '{request.prompt}'. This is a demo response from FreeAgentics."
        
        await manager.broadcast({
            "type": "message",
            "data": {
                "id": f"assistant-{message_id}",
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "conversationId": request.conversationId,
                "isStreaming": False
            }
        })
        
        # Also check if we should create an agent based on the prompt
        prompt_lower = request.prompt.lower()
        if any(word in prompt_lower for word in ["create", "make", "build", "add", "new"]):
            if any(word in prompt_lower for word in ["agent", "assistant", "helper"]):
                # Create an agent
                agent_request = UIAgentCreateRequest(description=request.prompt)
                agent = await create_agent_ui(agent_request, current_user, db)
                
                # Send agent creation notification
                await manager.broadcast({
                    "type": "agent_created",
                    "data": {
                        "agent": agent.dict(),
                        "message": f"Created new agent: {agent.name}",
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                response_text = f"I've created a new agent for you: {agent.name} ({agent.type})"
                
                # Send updated response
                await manager.broadcast({
                    "type": "message",
                    "data": {
                        "id": f"assistant-{message_id}-update",
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.now().isoformat(),
                        "conversationId": request.conversationId,
                        "isStreaming": False
                    }
                })
        
    except Exception as e:
        logger.error(f"Failed to send WebSocket message: {e}")
        # Continue with HTTP response even if WebSocket fails
    
    # Get current knowledge graph (optional)
    try:
        kg_response = await get_knowledge_graph_ui(current_user)
        # Convert Pydantic model to dict for v2 compatibility
        kg_data = kg_response.model_dump() if hasattr(kg_response, 'model_dump') else kg_response.dict()
    except Exception as e:
        logger.warning(f"Failed to get knowledge graph: {e}")
        kg_data = None
    
    return ProcessPromptResponse(
        response=response_text,
        knowledgeGraph=kg_data
    )
